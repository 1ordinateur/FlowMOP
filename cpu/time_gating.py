"""
Time gating component for FlowMOP.
Handles detection and removal of time-based anomalies in flow cytometry data.
"""

import warnings
from typing import Union, Tuple, List, Dict
import numpy as np
import numpy.ma as ma
import dask as da
import dask.array as da
from scipy.ndimage import maximum_filter1d
from scipy.interpolate import UnivariateSpline
from abc import ABC, abstractmethod
from .flowmop_utils import process_histogram, Peak
import cProfile
import pstats
from pstats import SortKey
import io
import time
import os

ArrayType = Union[np.ndarray, da.Array]

def profile_function(func):
    """Decorator to profile a function using cProfile."""
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        try:
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            
            # Create stats object and sort by cumulative time
            stats = pstats.Stats(profiler)
            stats.sort_stats(SortKey.CUMULATIVE)
            
            # Print to both console and file
            print("\nProfiling results for {}:".format(func.__name__))
            stats.print_stats(20)  # Print top 20 functions
            
            # Save detailed stats to file
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"profile_{func.__name__}_{timestamp}.stats"
            stats_path = os.path.join(os.getcwd(), "profiling", filename)
            os.makedirs(os.path.dirname(stats_path), exist_ok=True)
            
            # Save full stats to file
            stats.dump_stats(stats_path)
            print(f"\nFull profiling stats saved to: {stats_path}")
            
            return result
        finally:
            profiler.disable()
    return wrapper

class TimeGateStrategy(ABC):
    def __init__(self):
        self._debug_info = {}

    @abstractmethod
    def gate(self, data: np.ndarray, time_channel_index: int, marker_names: list) -> tuple[np.ndarray, np.ndarray]:
        """Apply time gating to the data."""
        pass

class MADTimeGate(TimeGateStrategy):
    def __init__(self, remove_zeros=True, min_cells=150, max_bins=500, step=200, mad_threshold=6,
                 peak_removal=1/3, min_nr_bins_peakdetection=5, histogram_smoothing=5, mad_method='all', mad_smoothing=[0.1, 1.0],
                 enable_dask=True, chunk_size=None):
        super().__init__()
        self.remove_zeros = remove_zeros
        self.min_cells = min_cells
        self.max_bins = max_bins
        self.step = step
        self.mad_threshold = mad_threshold
        self.peak_removal = peak_removal
        self.min_nr_bins_peakdetection = min_nr_bins_peakdetection
        self.histogram_smoothing = histogram_smoothing
        self.mad_method = mad_method
        self.mad_smoothing = mad_smoothing
        self.plot_counter = 0
        self.enable_dask = enable_dask
        self.chunk_size = chunk_size

    @profile_function
    def gate(self, data: ArrayType, time_channel_index: int, marker_names: list[str]) -> Tuple[ArrayType, ArrayType]:
        """Apply MAD-based time gating with Dask parallelization."""
        import dask.array as da
        
        # Convert to Dask array if enabled
        if self.enable_dask and not isinstance(data, da.Array):
            data = da.from_array(data, chunks=self.chunk_size or 'auto')

        # Calculate optimal number of events per bin
        events_per_bin = self._find_events_per_bin(data)
        breaks = self._make_breaks(events_per_bin, data.shape[0])

        # Identify fluorescent channels to process
        fluoro_channels = [i for i, name in enumerate(marker_names)
                         if i != time_channel_index and
                         not any(x.lower() in name.lower() for x in ['fsc', 'ssc', 'time'])]

        # Precompute global channel statistics
        global_stats = {}
        for channel in fluoro_channels:
            channel_data = data[:, channel]
            if isinstance(channel_data, da.Array):
                global_stats[channel] = {
                    'p99': da.percentile(channel_data, 99).compute(),
                    'max': da.max(channel_data).compute()
                }
            else:
                global_stats[channel] = {
                    'p99': np.percentile(channel_data, 99),
                    'max': np.max(channel_data)
                }

        def process_chunk(chunk, channel_idx, global_p99, global_max):
            """Process a chunk of data for a specific channel."""
            processed_chunk = self._preprocess_channel_data(chunk)
            
            # Generate histogram for this chunk
            hist, bin_edges = self._generate_histogram(processed_chunk)
            
            # Find peaks and calculate MAD
            peaks = self.find_peaks(hist)
            mad = self.calculate_mad(peaks)
            
            # Identify valid time bins and create gate vector
            valid_bins = self._identify_valid_bins(peaks, mad)
            return self._create_gate_vector(processed_chunk, bin_edges, valid_bins)

        if isinstance(data, da.Array):
            # Dask processing using map_blocks
            gate_vectors = []
            for channel in fluoro_channels:
                channel_data = data[:, channel]
                gate_vector = da.map_blocks(
                    process_chunk,  # Function to apply
                    channel_data,   # Input array
                    channel,        # Channel index
                    global_stats[channel]['p99'],  # Precomputed 99th percentile
                    global_stats[channel]['max'],  # Precomputed max value
                    dtype=bool,     # Output dtype
                    meta=np.array((), dtype=bool)  # Metadata template
                ).persist()  # Keep in memory for reuse
                gate_vectors.append(gate_vector)
            
            # Combine results across channels
            combined_vector = da.all(da.stack(gate_vectors), axis=0)
            return data[combined_vector], combined_vector
            
        else:
            # Standard numpy processing
            rejection_count = np.zeros(data.shape[0], dtype=int)
            for channel in fluoro_channels:
                chunk_results = [process_chunk(data[:, channel][break_idx], channel)
                                for break_idx in breaks['breaks']]
                rejection_count += np.concatenate(chunk_results)
            
            time_gate_vector = rejection_count < 2
            return data[time_gate_vector], time_gate_vector

    def _find_events_per_bin(self, arr):
        """Calculate optimal number of events per bin."""
        nr_events = arr.shape[0]
        
        # Try maximum number of bins first
        events_per_bin = int(np.ceil((nr_events / self.max_bins) * 2))
        events_per_bin = ((events_per_bin // self.step) * self.step) + self.step
        
        # Create test bins
        test_breaks = self._split_with_overlap(np.arange(nr_events), events_per_bin, 
                                             int(np.ceil(events_per_bin / 2)))
        
        # Count events in each bin
        bin_sizes = np.array([len(bin_) for bin_ in test_breaks])
        # Check if we have 2 or more bins below minimum size
        if np.sum(bin_sizes < self.min_cells) >= 2:
            # Fall back to minimum cells per bin
            events_per_bin = self.min_cells
            events_per_bin = ((events_per_bin // self.step) * self.step) + self.step
        return events_per_bin

    def _make_breaks(self, events_per_bin, nr_events):
        """Create time bins with overlap."""
        breaks = self._split_with_overlap(np.arange(nr_events), events_per_bin, 
                                        int(np.ceil(events_per_bin / 2)))
        return {'breaks': breaks, 'events_per_bin': events_per_bin}

    def _determine_thresholds_all_channels(self, data, channels, breaks, marker_names):
        """
        Detect thresholds in all channels, analyzing each time bin separately.
        
        Args:
            data: Flow cytometry data array
            channels: List of channel indices to analyze
            breaks: List of time bin indices
            marker_names: List of marker names for labeling plots
            
        Returns:
            Dictionary mapping channel indices to threshold information
        """
        threshold_frames = {}
        for channel in channels:
            channel_data = data[:, channel]
            marker_name = marker_names[channel]
            result = self._determine_all_thresholds(channel_data, breaks, marker_name)
            if result is not None:
                threshold_frames[channel] = result
        
        return threshold_frames

    def _preprocess_channel_data(self, channel_data, global_p99=None, global_max=None):
        """
        Preprocess channel data with Dask-compatible statistics handling.
        Applies arcsinh transform and uses global statistics when available.
        
        Args:
            channel_data: 1D array of channel values
            global_p99: Precomputed 99th percentile (for Dask)
            global_max: Precomputed max value (for Dask)
            
        Returns:
            Preprocessed channel data with arcsinh transform applied
        """
        # Make a copy to avoid modifying original data
        processed_data = channel_data.copy()
        
        # Apply arcsinh transform
        processed_data = np.arcsinh(processed_data/150)
        
        # Use provided global statistics or compute locally
        max_val = global_max if global_max is not None else np.max(processed_data)
        p99 = global_p99 if global_p99 is not None else np.quantile(processed_data, 0.99)
        
        # Handle limit of detection using global/local stats
        if global_max is None:  # Only check if not using precomputed values
            pct_at_max = np.mean(processed_data == max_val) * 100
            if pct_at_max > 0.5:
                processed_data = processed_data[(processed_data <= p99)]
        
        # Clip to 99th quantile using appropriate value
        processed_data = np.clip(processed_data, None, p99)
        
        # Set negative values to zero
        processed_data[processed_data < 0] = 0
        
        return processed_data

    def _determine_all_thresholds(self, channel_data, breaks, marker_name):
        """Determine peaks for a single channel."""
        # Preprocess the channel data
        processed_data = self._preprocess_channel_data(channel_data)
        
        # Use processed data for peak detection
        full_channel_thresholds = self._timegate_threshold_detection(processed_data, smoothing=self.histogram_smoothing)
        
        if np.all(np.isnan(full_channel_thresholds)):
            return None
        
        # Find threshold for positive peak across all data
        positive_threshold = np.max(processed_data) * self.peak_removal
        
        # Process each time break - ensure indices are within bounds
        channel_breaks = {}
        for i, break_indices in enumerate(breaks):
            # Ensure indices are within bounds
            valid_indices = break_indices[break_indices < len(processed_data)]
            if len(valid_indices) > 0:  # Only process if we have valid indices
                channel_breaks[i] = processed_data[valid_indices]
        
        thresholds = {}
        for break_, break_data in channel_breaks.items():
            try:
                thresh = self._timegate_threshold_detection(break_data, 
                                                     smoothing=self.histogram_smoothing,
                                                     max_peak=positive_threshold)
                thresholds[break_] = thresh
            except Exception as e:
                print(f"Warning: Error processing break {break_}: {str(e)}")
                thresholds[break_] = np.array([])
        
        # Create array of thresholds and their time bins
        threshold_frame = np.array([(break_, thresh) for break_, thresh_list in thresholds.items() 
                                  for thresh in thresh_list], dtype=[('Bin', int), ('Threshold', float)])
        
        # Remove any NaN thresholds
        if np.any(np.isnan(threshold_frame['Threshold'])):
            threshold_frame = threshold_frame[~np.isnan(threshold_frame['Threshold'])]
            thresholds = {break_: thresh_list[~np.isnan(thresh_list)] 
                         for break_, thresh_list in thresholds.items()}

        # Find representative thresholds across time bins
        most_occurring_thresholds = self._find_most_occurring_thresholds(thresholds)  
        if most_occurring_thresholds is None:
            most_occurring_thresholds = np.median(threshold_frame['Threshold'])

        # After finding representative thresholds across time bins, we need to update each bin's threshold
        # assignments to match these representative values. This ensures consistent thresholding across
        # the entire time series and reduces the impact of local variations or noise.
        updated_threshold_frame = self._update_threshold_frame(thresholds, most_occurring_thresholds)  # Method name unchanged as it's outside selection
        
        return updated_threshold_frame[updated_threshold_frame['Bin'] != -1]

    def _timegate_threshold_detection(self, bin_data, smoothing=None, max_peak=None, window_size=2):
        """
        Detect peaks in time-gated data.
        
        Ensures thresholds are:
        1. Separated by at least the smoothing window
        2. Above the peak_removal threshold relative to max fluoresent peak
        3. Have at least 10% prominence relative to surrounding minima if >2 peaks
        """
        
        if self.remove_zeros:
            bin_data = bin_data[bin_data != 0]
        
        # Use shared histogram processing function with class-level smoothing parameter if not overridden
        smoothing = smoothing if smoothing is not None else self.histogram_smoothing
        
        # Calculate the actual range of values
        min_val = np.min(bin_data)
        max_val = np.max(bin_data)
        value_range = max_val - min_val
        
        # Calculate cutoff points at 1% and 99% of the total range
        bottom_cutoff = min_val + (value_range * 0.01)  # Bottom 1%
        top_cutoff = max_val - (value_range * 0.01)     # Top 1%
        
        # Filter out values in the top and bottom 1% of the range
        bin_data = bin_data[(bin_data >= bottom_cutoff) & (bin_data <= top_cutoff)]
        
        # Apply quantile filtering
        q05, q95 = np.quantile(bin_data, [0.001, 0.999])
        bin_data = bin_data[(bin_data >= q05) & (bin_data <= q95)]
        
        if len(bin_data) < 3:
            return np.array([])
            
        result = process_histogram(bin_data, smoothing_window=smoothing, num_bins=100)
        
        if result is None:
            return np.array([])
            
        smoothed_hist, bin_edges, peak_indices, peak_densities = result

        peak_indices = np.array(peak_indices, dtype=int)
        if max_peak is None:
            max_peak = np.max(smoothed_hist[peak_indices])
        
        # Filter peaks based on threshold
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        peak_values = bin_centers[peak_indices]
        
        # If only 1 peak detected, use threshold filtering
        if len(peak_indices) == 1:
            peak_mask = peak_values > (self.peak_removal * max_peak)
            threshold_positive_index = peak_indices[peak_mask]

        # If multiple peaks, find lowest point between first and last peak
        else:
            first_peak_idx = peak_indices[0]
            last_peak_idx = peak_indices[-1]
            valley_region = smoothed_hist[first_peak_idx:last_peak_idx]
            # Find all points that match the minimum value
            min_value = np.min(valley_region)
            min_indices = np.where(valley_region == min_value)[0]
            # Take the middle minimum point
            middle_idx = int(len(min_indices) // 2)
            lowest_point_idx = first_peak_idx + min_indices[middle_idx]
            threshold = bin_centers[lowest_point_idx]
            # Find peaks above the threshold
            peaks_above_threshold = peak_indices[bin_centers[peak_indices] > threshold]
            if len(peaks_above_threshold) > 0:
                # Take average of peaks above threshold
                threshold_positive_index = np.array([int(np.mean(peaks_above_threshold))])
            else:
                # Fallback to lowest point if no peaks above threshold
                threshold_positive_index = np.array([lowest_point_idx])

        return bin_centers[threshold_positive_index]

    def _find_most_occurring_thresholds(self, thresholds, tolerance=0.05):
        """Find the most frequently occurring peaks."""
        flattened_peaks = [(bin_num, peak) for bin_num, peak_list in thresholds.items() 
                          for peak in peak_list]
        if not flattened_peaks:
            return None
            
        bin_numbers, peak_values = zip(*flattened_peaks)
        bin_edges = np.arange(min(peak_values), max(peak_values) + tolerance, tolerance)
        hist, _ = np.histogram(peak_values, bins=bin_edges)
        
        min_occurrences = (self.min_nr_bins_peakdetection / 100) * len(thresholds)
        candidate_bins = np.where(hist >= min_occurrences)[0]
        
        if len(candidate_bins) > 0:
            candidate_thresholds = (bin_edges[candidate_bins] + bin_edges[candidate_bins + 1]) / 2
            return candidate_thresholds[np.argmax(hist[candidate_bins])]
        return None

    def _update_threshold_frame(self, thresholds, most_occurring_thresholds):
        """Update threshold frame with closest thresholds to the most occurring thresholds."""
        updated_threshold_frame = np.empty(len(thresholds), dtype=[('Bin', int), ('Threshold', float)])
            
        for i, (break_, threshold_list) in enumerate(thresholds.items()):
            if isinstance(break_, int) and break_ < len(updated_threshold_frame):
                if len(threshold_list) > 0:
                    closest_threshold_index = np.argmin(np.abs(threshold_list - most_occurring_thresholds))
                    updated_threshold_frame[i] = (break_, threshold_list[closest_threshold_index])
                else:
                    updated_threshold_frame[i] = (break_, np.nan)
            else:
                updated_threshold_frame[i] = (-1, np.nan)
        
        return updated_threshold_frame

    def _mad_excluder(self, peaks, mad_threshold, breaks, nr_cells, smoothing_factor=0.1):
        """Apply MAD-based outlier detection with configurable smoothing."""
        # Extract just the threshold values from the structured array
        peak_values = peaks['Threshold']  # Changed from peaks to peaks['Threshold']
        
        # First normalize to mean=1
        peak_values = peak_values / np.mean(peak_values)
        
        # Then scale to target standard deviation of 0.1
        current_std = np.std(peak_values)
        target_std = 0.1
        peak_values = peak_values * (target_std / current_std)
        
        # Shift values so median is 1
        current_median = np.median(peak_values)
        peak_values = peak_values + (1 - current_median)
        
        # Scale smoothing factor based on number of bins
        n_bins = len(peak_values)
        smoothing_factor = smoothing_factor * n_bins / 100  # Base factor scaled by number of bins
        smoothing_factor = max(0.1, min(2.0, smoothing_factor))  # Clamp between 0.1 and 2.0
        
        # Apply spline smoothing with fixed parameter
        spline = UnivariateSpline(np.arange(len(peak_values)), peak_values, s=smoothing_factor)
        smoothed_peaks = spline(np.arange(len(peak_values)))
        
        # Calculate MAD thresholds on normalized data
        median_peak = np.median(smoothed_peaks)
        mad_peak = np.median(np.abs(smoothed_peaks - median_peak))
        
        upper_interval = median_peak + mad_threshold * mad_peak
        lower_interval = median_peak - mad_threshold * mad_peak
        to_remove_bins = (smoothed_peaks > upper_interval) | (smoothed_peaks < lower_interval)
        
        removed = self._removed_bins(breaks, to_remove_bins, nr_cells)
        contribution_mad = round((len(removed['cell_ids']) / nr_cells) * 100, 2)
        
        return {
            "cells": removed['cells'],
            "cell_ids": removed['cell_ids'],
            "MAD_bins": to_remove_bins,
            "Contribution_MAD": contribution_mad
        }

    def _removed_bins(self, breaks, outlier_bins, nr_cells):
        """Calculate removed bins based on outliers."""
        if np.any(outlier_bins):
            removed_cells = np.concatenate([breaks[i] for i in np.where(outlier_bins)[0]])
            removed_cells = np.unique(removed_cells)
        else:
            removed_cells = np.array([], dtype=int)
        
        bad_cells = np.ones(nr_cells, dtype=bool)
        bad_cells[removed_cells] = False
        return {'cells': bad_cells, 'cell_ids': removed_cells}

    @staticmethod
    def _split_with_overlap(vec, seg_length, overlap):
        """Split vector into overlapping segments."""
        starts = np.arange(0, len(vec), seg_length - overlap)
        ends = starts + seg_length
        ends[ends > len(vec)] = len(vec)
        return [vec[start:end] for start, end in zip(starts, ends)]
