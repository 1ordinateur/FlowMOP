"""
GPU-accelerated debris gating operations using CuPy.
This module handles single-channel operations and assumes all arrays are CuPy arrays.
"""

import cupy as cp
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from accelerated.gpu_funcs.flowmop_utils_accelerated_gpu import Peak, process_histogram, is_excluded_marker, find_left_minimum

@dataclass
class ChannelAnalysisResult:
    """Results from analyzing a single channel."""
    peaks: List[Peak]
    positive_mask: Optional[cp.ndarray]
    is_valid: cp.bool_
    threshold: Optional[cp.float32]

def _has_low_peak(bin_edges: cp.ndarray, peak_indices: cp.ndarray, 
                lowest_reference_pos: cp.float32, base_tolerance: cp.float32 = 1.3) -> cp.bool_:
    """Check if any peaks are near or below the reference lowest peak."""
    total_range = bin_edges[-1] - bin_edges[0]
    relative_pos = (lowest_reference_pos - bin_edges[0]) / total_range
    dynamic_tolerance = base_tolerance * (cp.float32(1) + (cp.float32(1) - relative_pos))
    
    peak_positions = bin_edges[peak_indices]
    return cp.any(peak_positions <= lowest_reference_pos * dynamic_tolerance)

def _get_left_boundary_threshold(hist: cp.ndarray, bin_edges: cp.ndarray, 
                              peak_indices: cp.ndarray, smoothing_window: cp.int32) -> cp.float32:
    """Get threshold using left boundary of lowest peak."""
    lowest_peak_idx = peak_indices[0]
    valley_idx = find_left_minimum(hist, lowest_peak_idx, smoothing_window)
    return bin_edges[valley_idx]

def _get_max_peak_threshold(hist: cp.ndarray, bin_edges: cp.ndarray,
                         peak_indices: cp.ndarray, peak_densities: cp.ndarray,
                         smoothing_window: cp.int32) -> Optional[cp.float32]:
    """Get threshold using max peak logic."""
    max_peak_idx = cp.argmax(peak_densities)
    
    if (max_peak_idx == 0 and len(peak_indices) > 1) or max_peak_idx == 1:
        start_idx = peak_indices[0]
        end_idx = peak_indices[1]
        
        min_value = cp.min(hist[start_idx:end_idx])
        min_indices = cp.where(hist[start_idx:end_idx] == min_value)[0]
        min_idx = start_idx + (min_indices[len(min_indices)//2])
        return bin_edges[min_idx]

    # Get peak positions for comparison
    peak_positions = bin_edges[peak_indices]
    
    # Check each peak from max_peak_idx-1 down to 0
    for i in range(max_peak_idx - 1, -1, -1):
        valley_idx = find_left_minimum(hist, cp.int32(peak_indices[i]), smoothing_window)
        valley_position = bin_edges[valley_idx]
        # If we get to a valley before the first peak, use midpoint between first and second peaks
        if valley_idx < peak_indices[0]:
            return cp.float32((bin_edges[peak_indices[0]] + bin_edges[peak_indices[1]]) / cp.float32(2))
        # Use cp.less to compare valley position against all peaks
        if cp.all(cp.less(valley_position, peak_positions)):
            return valley_position

    return None

def _find_fsc_threshold(process_histogram_result: Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray], 
                      ref_peaks: cp.ndarray, smoothing_window: cp.int32) -> Optional[cp.float32]:
    """Find the FSC threshold by comparing peaks to reference peaks from all cells."""
    # Get the lowest FSC peak position from reference
    lowest_reference_peak_pos = cp.min(ref_peaks)
    smoothed_hist, pos_bin_edges, pos_peak_indices, peak_densities = process_histogram_result
    
    # Check if we have any peaks near or below the reference lowest peak
    if not _has_low_peak(pos_bin_edges, pos_peak_indices, lowest_reference_peak_pos):
        # If we don't have any low peaks, use left boundary of lowest available peak
        return _get_left_boundary_threshold(smoothed_hist, pos_bin_edges, pos_peak_indices, smoothing_window)
        
    # Use traditional max peak logic
    return _get_max_peak_threshold(smoothed_hist, pos_bin_edges, pos_peak_indices, peak_densities, smoothing_window)

def get_fsc_thresholds(fsc_data: cp.ndarray, positive_mask: cp.ndarray,
                      smoothing_window: cp.int32) -> Optional[cp.float32]:
    """
    Calculate FSC threshold for a positive population.
    
    Args:
        fsc_data: FSC channel data
        positive_mask: Mask indicating positive events
        smoothing_window: Window size for smoothing
        
    Returns:
        Optional float threshold value
    """
    # First get reference peaks from all FSC data
    ref_result = process_histogram(fsc_data, smoothing_window)
    if ref_result is None:
        return None
    
    ref_hist, ref_bin_edges, ref_peak_indices, _ = ref_result
    if len(ref_peak_indices) == 0:
        return None
        
    # Get reference peaks with positions
    ref_peaks_pos = ref_bin_edges[ref_peak_indices]
    sort_idx = cp.argsort(ref_peaks_pos)
    ref_peaks = cp.stack([ref_peak_indices[sort_idx], ref_peaks_pos[sort_idx]], axis=1)
    
    # Now process the positive population
    positive_fsc = fsc_data[positive_mask]
    pos_result = process_histogram(positive_fsc, smoothing_window)
    if pos_result is None:
        return None

    return _find_fsc_threshold(pos_result, ref_peaks, smoothing_window)

def peak_width_debris(hist: cp.ndarray, peak_indices: List[int], 
                     bin_edges: cp.ndarray, percentage_cells_present: cp.float32,
                     smoothing_window: cp.int32) -> List[Tuple[cp.float32, cp.float32]]:
    """Calculate peak widths using GPU operations."""
    peak_widths = []
    num_bins = len(hist)

    for i, peak_idx in enumerate(peak_indices):
        # Find left boundary
        if i == 0:
            left_idx = find_left_minimum(hist, cp.int32(peak_idx), smoothing_window)
        else:
            prev_peak = peak_indices[i-1]
            segment = hist[prev_peak:peak_idx+1]
            left_idx = prev_peak + cp.int32(cp.argmin(segment))

        # Find right boundary
        if i == len(peak_indices) - 1:
            right_idx = peak_idx
            while right_idx < num_bins - 1:
                if hist[right_idx] <= hist[right_idx + 1]:
                    right_idx += 1
                else:
                    break
        else:
            next_peak = peak_indices[i+1]
            segment = hist[peak_idx:next_peak+1]
            right_idx = peak_idx + cp.int32(cp.argmin(segment))

        # Calculate peak percentage
        peak_percentage = cp.float32(cp.sum(hist[left_idx:right_idx + 1]) / cp.sum(hist) * 100)
        if peak_percentage >= percentage_cells_present:
            peak_widths.append((cp.float32(bin_edges[left_idx]), cp.float32(bin_edges[right_idx])))

    return peak_widths

def analyze_channel(channel_data: cp.ndarray, marker_name: str, 
                   min_peaks: cp.int32 = 2, max_peaks: cp.int32 = 5,
                   smoothing_window: cp.int32 = 3, 
                   percentage_cells_present: cp.float32 = 5,
                   num_bins: cp.int32 = 100) -> ChannelAnalysisResult:
    """
    Analyze a single channel's data.
    
    Args:
        channel_data: Data for a single channel (CuPy array)
        marker_name: Name of the marker
        min_peaks: Minimum number of peaks required
        max_peaks: Maximum number of peaks to consider
        smoothing_window: Window size for smoothing
        percentage_cells_present: Minimum percentage of cells required in a peak
        num_bins: Number of bins for histogram analysis
        
    Returns:
        ChannelAnalysisResult containing peaks and analysis results
    """
    if is_excluded_marker(marker_name):
        return ChannelAnalysisResult(peaks=[], positive_mask=None, is_valid=cp.bool_(False), threshold=None)

    # Transform data
    data_transformed = cp.arcsinh(channel_data / cp.int32(150))
    
    # Process histogram
    result = process_histogram(
        data_transformed,
        smoothing_window=smoothing_window,
        num_bins=num_bins
    )

    if result is None:
        return ChannelAnalysisResult(peaks=[], positive_mask=None, is_valid=cp.bool_(False), threshold=None)

    smoothed_hist, bin_edges, peak_indices, peak_densities = result

    # Check if we have enough peaks
    if len(peak_indices) < min_peaks:
        print(f"Insufficient peaks ({len(peak_indices)}) for marker {marker_name}")
        return ChannelAnalysisResult(peaks=[], positive_mask=None, is_valid=cp.bool_(False), threshold=None)

    # Sort peaks by position
    peak_positions = bin_edges[peak_indices]
    sort_idx = cp.argsort(peak_positions)
    sorted_peak_indices = peak_indices[sort_idx]

    # Filter peaks by density if needed
    if len(sorted_peak_indices) > max_peaks:
        densities = smoothed_hist[sorted_peak_indices]
        # Find the density threshold (max_peaks-th highest value)
        density_threshold = cp.partition(densities, -max_peaks)[-max_peaks]
        # Keep all peaks with density >= threshold
        keep_mask = densities >= density_threshold
        sorted_peak_indices = sorted_peak_indices[keep_mask]

    # Calculate peak widths
    peak_widths = peak_width_debris(
        smoothed_hist, sorted_peak_indices, bin_edges, 
        percentage_cells_present, smoothing_window
    )

    # Convert to Peak objects
    peaks = [Peak(start=start, end=end) for start, end in peak_widths]

    # Calculate positive mask if we have enough peaks
    positive_mask = None
    threshold = None
    if len(peaks) >= 2:
        second_peak = peaks[1]
        positive_mask = data_transformed >= second_peak.start
        threshold = cp.float32(second_peak.start)

    return ChannelAnalysisResult(
        peaks=peaks,
        positive_mask=positive_mask,
        is_valid=cp.bool_(len(peaks) >= min_peaks),
        threshold=threshold
    )