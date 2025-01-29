"""
GPU-accelerated time gating operations using CuPy.
This module handles single-channel operations and assumes all arrays are CuPy arrays.
"""
import dask.array as da
import numpy as np
import cupy as cp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
# from cupyx.scipy import interpolate as cupy_interpolate # TODO: ASK NCI TO UPDATE RAPIDS
from scipy.interpolate import UnivariateSpline

from accelerated.gpu_funcs.flowmop_utils_accelerated_gpu import process_histogram, Peak

@dataclass
class TimeGateAnalysisResult:
    """Results from analyzing time gating for a channel."""
    thresholds: cp.ndarray
    is_valid: cp.bool_
    mad_results: Optional[Dict]

def find_events_per_bin(data: cp.ndarray, remove_zeros: bool, min_cells: cp.int32, 
                       max_bins: cp.int32, step: cp.int32) -> cp.int32:
    """Calculate optimal number of events per bin."""
    if remove_zeros:
        nonzero_counts = cp.sum(data != 0, axis=0)
        # Keep everything as CuPy arrays
        min_nonzero = cp.min(nonzero_counts).astype(cp.float32)
        max_bins_mass = min_nonzero / min_cells
        if max_bins_mass < max_bins:
            max_bins = max_bins_mass
    
    nr_events = data.shape[0]
    max_cells = cp.ceil((nr_events / max_bins) * 2).astype(cp.int32)
    max_cells = ((max_cells // step) * step) + step
    
    return cp.maximum(min_cells, max_cells) 

def make_time_breaks(events_per_bin: cp.int32, nr_events: cp.int32) -> cp.ndarray:
    """Create time bins with overlap using pure GPU operations."""
    # Ensure inputs are CuPy arrays
    events_per_bin = cp.asarray(events_per_bin, dtype=cp.int32)
    nr_events = cp.asarray(nr_events, dtype=cp.int32)
    
    overlap = cp.ceil(events_per_bin / 2).astype(cp.int32)
    step = events_per_bin - overlap
    
    # Create starts and ends
    starts = cp.arange(0, nr_events, step.item(), dtype=cp.int32)
    ends = cp.minimum(starts + events_per_bin, nr_events)
    n_slices = len(starts)
    
    # Create meshgrid of all indices
    row_idx = cp.arange(n_slices).reshape(-1, 1)
    col_idx = cp.arange(events_per_bin).reshape(1, -1)
    indices = starts.reshape(-1, 1) + col_idx
    
    # Create mask for valid indices
    mask = indices < ends.reshape(-1, 1)
    mask = mask & (indices < nr_events)
    
    # Fill result array with masked indices
    result = cp.where(mask, indices, -1)
    
    return result

def preprocess_channel_data(channel_data):
    """
    Preprocess channel data to handle limit-of-detection and data quality issues.
    Applies arcsinh transform to handle large dynamic range.
    
    Args:
        channel_data: 1D array of channel values
        
    Returns:
        Preprocessed channel data with arcsinh transform applied
    """
    # Apply arcsinh transform
    processed_data = cp.arcsinh(channel_data/150)
    
    # Handle limit of detection (if >0.5% at max value)
    max_val = cp.max(processed_data)
    pct_at_max = cp.mean(processed_data == max_val) * 100
    if pct_at_max > 0.5:
        # Drop everything above 99th percentile and below 1st percentile
        p99 = cp.percentile(processed_data, 99)
        processed_data = processed_data[(processed_data <= p99)]
    
    # # Clip to 99th quantile
    p99 = cp.quantile(processed_data, 0.99)
    processed_data = cp.clip(processed_data, None, p99)
    
    # Set negative values to zero
    processed_data[processed_data < 0] = 0
    
    return processed_data

def _timegate_threshold_detection(bin_data: cp.ndarray, smoothing= 2, max_peak=None, peak_removal=1/3, remove_zeros=True) -> cp.int32:
    """
    Detect thresholds in time-gated data.
    
    Ensures peaks are:
    1. Separated by at least the smoothing window
    2. Above the peak_removal threshold relative to max fluoresent peak
    3. Have at least 10% prominence relative to surrounding minima if >2 peaks
    """
    if remove_zeros:
        bin_data = cp.compress(bin_data != 0, bin_data)
    
    # Calculate the actual range of values
    min_val = cp.min(bin_data)
    max_val = cp.max(bin_data)
    value_range = max_val - min_val
    
    # Calculate cutoff points at 1% and 99% of the total range
    bottom_cutoff = min_val + (value_range * 0.01)  # Bottom 1%
    top_cutoff = max_val - (value_range * 0.01)     # Top 1%
    
    # Filter out values in the top and bottom 1% of the range
    mask = (bin_data >= bottom_cutoff) & (bin_data <= top_cutoff)
    bin_data = cp.compress(mask, bin_data)
    # Print data types of mask and bin_data

    # Apply quantile filtering on GPU
    q05, q95 = cp.quantile(bin_data, [0.001, 0.999])
    mask = (bin_data >= q05) & (bin_data <= q95)
    bin_data = cp.compress(mask, bin_data)
    
    if len(bin_data) < 3:
        return cp.array([])
    
    num_bins = cp.array(100, dtype=cp.int32)
    result = process_histogram(bin_data, smoothing_window=smoothing, num_bins=num_bins, filter_extremes=cp.asarray(True, dtype=cp.bool_))
    
    if result is None:
        return cp.array([])
        
    smoothed_hist, bin_edges, peak_indices, peak_densities = result

    peak_indices = cp.array(peak_indices, dtype=cp.int32)
    if max_peak is None:
        max_peak = cp.max(smoothed_hist[peak_indices])
    
    # Filter peaks based on threshold
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak_values = bin_centers[peak_indices]
    
    # If only 1 peak detected, use threshold filtering
    if len(peak_indices) == 1:
        peak_mask = peak_values > (peak_removal * max_peak)
        threshold_positive_index = peak_indices[peak_mask]
    # If multiple peaks, find lowest point between first and last peak
    else:
        first_peak_idx = peak_indices[0]
        last_peak_idx = peak_indices[-1]
        valley_region = smoothed_hist[first_peak_idx:last_peak_idx]
        # Find all points that match the minimum value
        min_value = cp.min(valley_region)
        min_indices = cp.where(valley_region == min_value)[0]
        # Take the middle minimum point
        middle_idx = int(len(min_indices) // 2)
        lowest_point_idx = first_peak_idx + min_indices[middle_idx]
        threshold = bin_centers[lowest_point_idx]
        # Find peaks above the threshold
        peaks_above_threshold = peak_indices[bin_centers[peak_indices] > threshold]
        if len(peaks_above_threshold) > 0:
            # Take average of peaks above threshold
            threshold_positive_index = cp.array([int(cp.mean(peaks_above_threshold))])
        else:
            # Fallback to lowest point if no peaks above threshold
            threshold_positive_index = cp.array([lowest_point_idx])

    # The prominence check is now handled in flowmop_utils.find_peaks
    return bin_centers[threshold_positive_index]

def find_most_occurring_thresholds(thresholds: cp.ndarray, min_nr_bins_peakdetection: cp.int32,
                            tolerance: cp.float32 = 0.05) -> Optional[cp.float32]:
    """Find most frequently occurring peaks."""
    if len(thresholds) == 0:
        return None

    min_val, max_val = cp.min(thresholds), cp.max(thresholds)
    bin_edges = cp.arange(min_val, max_val + tolerance, tolerance)
    hist, _ = cp.histogram(thresholds, bins=bin_edges)
    
    min_occurrences = (min_nr_bins_peakdetection / 100) * len(thresholds)
    candidate_bins = cp.where(hist >= min_occurrences)[0]
    
    if len(candidate_bins) > 0:
        candidate_thresholds = (bin_edges[candidate_bins] + bin_edges[candidate_bins + 1]) / 2
        return candidate_thresholds[cp.argmax(hist[candidate_bins])]
    
    return None

def mad_excluder(thresholds: cp.ndarray, mad_threshold: cp.float32, breaks: cp.ndarray,
                nr_cells: cp.int32, mad_smoothing_factor: cp.float32 = 0.1) -> Dict:
    """
    Apply MAD-based outlier detection.
    
    Args:
        thresholds: Threshold values array
        mad_threshold: MAD threshold for outlier detection
        breaks: Time breaks array
        nr_cells: Total number of cells
        
    Returns:
        Dictionary containing results of MAD-based outlier detection
    """
    threshold_values = thresholds
    
    # First normalize to mean=1
    threshold_values = threshold_values / cp.mean(threshold_values)
    
    # Then scale to target standard deviation of 0.1
    current_std = cp.std(threshold_values)
    target_std = 0.1
    threshold_values = threshold_values * (target_std / current_std)
    
    # Shift values so median is 1
    current_median = cp.median(threshold_values)
    threshold_values = threshold_values + (1 - current_median)
    
    # Scale smoothing factor based on number of bins
    n_bins = len(threshold_values)
    smoothing_factor = mad_smoothing_factor * n_bins / 100  # Base factor scaled by number of bins
    smoothing_factor = max(0.1, min(2.0, smoothing_factor))  # Clamp between 0.1 and 2.0
    
    # TODO: Replace with cupyx.scipy.interpolate once NCI updates RAPIDS
    # For now, move to CPU, run spline, then back to GPU
    x = cp.arange(len(threshold_values)).get()
    y = cp.asnumpy(threshold_values)
    spline = UnivariateSpline(x, y, s=smoothing_factor)
    smoothed_thresholds = cp.asarray(spline(x))
    
    # Calculate MAD thresholds on normalized data
    median_threshold = cp.median(smoothed_thresholds)
    mad_threshold = cp.median(cp.abs(smoothed_thresholds - median_threshold))
    
    upper_interval = median_threshold + mad_threshold * mad_threshold
    lower_interval = median_threshold - mad_threshold * mad_threshold
    to_remove_bins = (smoothed_thresholds > upper_interval) | (smoothed_thresholds < lower_interval)
    
    removed = _removed_bins(breaks, to_remove_bins, nr_cells)
    contribution_mad = cp.round((len(removed['cell_ids']) / nr_cells) * cp.asarray(100, dtype=cp.float32), 2)
    
    return {
        "cells": removed['cells'],
        "cell_ids": removed['cell_ids'],
        "MAD_bins": to_remove_bins,
        "Contribution_MAD": contribution_mad
    }

def _removed_bins(breaks, outlier_bins, nr_cells):
    """Calculate removed bins based on outliers."""
    if cp.any(outlier_bins):
        removed_cells = cp.concatenate([breaks[i] for i in cp.where(outlier_bins)[0]])
        removed_cells = cp.unique(removed_cells)
    else:
        removed_cells = cp.array([], dtype=int)

    bad_cells = cp.ones(nr_cells, dtype=bool)
    bad_cells[removed_cells] = False
    return {'cells': bad_cells, 'cell_ids': removed_cells}

class ChannelAnalyzer:
    """Handles analysis of individual channels for time gating."""
    
    def __init__(self, remove_zeros: bool = True, smoothing: cp.int32 = 2,
                 mad_smoothing: list[cp.float32] = [0.1, 1.0],
                 mad_threshold: cp.float32 = 6, peak_removal: cp.float32 = 1/3,
                 min_nr_bins_peakdetection: cp.int32 = 5):
        self.remove_zeros = remove_zeros
        self.smoothing = smoothing
        self.mad_smoothing = mad_smoothing
        self.mad_threshold = mad_threshold
        self.peak_removal = peak_removal
        self.min_nr_bins_peakdetection = min_nr_bins_peakdetection

    def analyze_channel(self, channel_data: cp.ndarray, breaks: cp.ndarray) -> TimeGateAnalysisResult:
        """Main entry point for channel analysis."""
        # Check full channel thresholds
        if not self._validate_full_channel(channel_data):
            return TimeGateAnalysisResult(thresholds=cp.array([]), is_valid=False, mad_results=None)
        
        # Process time breaks
        threshold_frame = self._process_time_breaks(channel_data, breaks)
        if threshold_frame is None:
            return TimeGateAnalysisResult(thresholds=cp.array([]), is_valid=cp.bool_(False), mad_results=None)
        # Update thresholds
        threshold_frame = self._update_thresholds(threshold_frame)
        
        # Apply MAD filtering and get both masks
        short_results, long_results = self._apply_mad_filtering(threshold_frame, breaks, len(channel_data))
        
        # Combine results
        rejected_cells = ~(short_results['cells'] & long_results['cells'])
        rejection_count = cp.zeros(len(channel_data), dtype=cp.int32)
        rejection_count[rejected_cells] += 1
        
        # Create final gate vector
        time_gate_vector = rejection_count < 2
        
        return TimeGateAnalysisResult(
            thresholds=threshold_frame,
            is_valid=True,
            mad_results={'short_results': short_results, 'long_results': long_results, 'final_gate': time_gate_vector}
        )

    def _validate_full_channel(self, channel_data: cp.ndarray) -> bool:
        """Validate thresholds for the full channel."""
        full_channel_thresholds = _timegate_threshold_detection(
            channel_data,
            remove_zeros=self.remove_zeros,
            smoothing=self.smoothing,
            peak_removal=self.peak_removal
        )
        return not cp.all(cp.isnan(full_channel_thresholds))

    def _process_time_breaks(self, channel_data: cp.ndarray, breaks: cp.ndarray) -> Optional[cp.ndarray]:
        """Process each time break and collect thresholds."""
        thresholds = cp.empty((len(breaks), 2), dtype=cp.float32)
        valid_thresholds = 0

        for i, break_indices in enumerate(breaks):
            break_data = channel_data[break_indices]
            max_peak = cp.max(break_data)

            threshold = _timegate_threshold_detection(
                break_data,
                remove_zeros=self.remove_zeros,
                smoothing=self.smoothing,
                max_peak=max_peak,
                peak_removal=self.peak_removal
            )
            
            if len(threshold) > 0:
                thresholds[i] = threshold
                valid_thresholds += 1

        if valid_thresholds == 0:
            return None

        valid_mask = ~cp.isnan(thresholds)
        return thresholds[valid_mask]

    def _update_thresholds(self, threshold_frame: cp.ndarray) -> cp.ndarray:
        """Update thresholds based on most occurring values."""
        most_occurring = find_most_occurring_thresholds(
            threshold_frame,
            min_nr_bins_peakdetection=self.min_nr_bins_peakdetection
        )
        if most_occurring is None:
            most_occurring = cp.median(threshold_frame)
        
        valid_mask = (threshold_frame[0] < len(threshold_frame)) & ~cp.isnan(threshold_frame)
        updated_frame = cp.full_like(threshold_frame, cp.array(cp.nan))
        
        if cp.any(valid_mask):
            closest_indices = cp.argmin(cp.abs(threshold_frame[valid_mask] - most_occurring))
            updated_frame[valid_mask] = threshold_frame[closest_indices]
            
        return updated_frame

    def _apply_mad_filtering(self, threshold_frame: cp.ndarray, breaks: cp.ndarray, 
                           data_length: int) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Apply MAD-based filtering to the thresholds.
        
        Returns:
            Tuple[cp.ndarray, cp.ndarray]: Boolean masks for (short_term_results, long_term_results)
        """
        # Short-term filtering
        short_results = mad_excluder(threshold_frame, self.mad_threshold, breaks, 
                                   data_length, mad_smoothing_factor=self.mad_smoothing[0])
        # Long-term filtering
        long_results = mad_excluder(threshold_frame, self.mad_threshold, breaks, 
                                  data_length, mad_smoothing_factor=self.mad_smoothing[1])
        
        return short_results, long_results

def analyze_channel(channel_data: cp.ndarray, breaks: cp.ndarray,
                   remove_zeros: bool = True, smoothing: cp.int32 = 2,
                   mad_smoothing: list[cp.float32] = [0.1, 1.0],
                   mad_threshold: cp.float32 = 6, peak_removal: cp.float32 = 1/3,
                   min_nr_bins_peakdetection: cp.int32 = 5) -> TimeGateAnalysisResult:
    """
    Analyze a single channel for time gating.
    This is a wrapper around the ChannelAnalyzer class for backward compatibility.
    """

    channel_data = cp.asarray(channel_data.get())
    analyzer = ChannelAnalyzer(
        remove_zeros=remove_zeros,
        smoothing=smoothing,
        mad_smoothing=mad_smoothing,
        mad_threshold=mad_threshold,
        peak_removal=peak_removal,
        min_nr_bins_peakdetection=min_nr_bins_peakdetection,
    )
    return analyzer.analyze_channel(channel_data, breaks)


