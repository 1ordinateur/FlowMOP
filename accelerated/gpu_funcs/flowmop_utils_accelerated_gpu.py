"""
GPU-accelerated shared utility functions for FlowMOP modules.
All functions assume input arrays are CuPy arrays.
"""

import cupy as cp
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass

@dataclass
class Peak:
    """Represents a peak in the data with its boundaries."""
    start: cp.float32
    end: cp.float32

@dataclass
class HistogramAnalysis:
    """Results of histogram analysis."""
    hist: cp.ndarray
    bin_edges: cp.ndarray
    smoothed_hist: cp.ndarray

def analyze_histogram(data: cp.ndarray, num_bins: cp.int32, smoothing_window: cp.int32, 
                     filter_extremes: bool = True) -> HistogramAnalysis:
    """
    Analyze histogram of data using GPU operations.
    Assumes data is already a CuPy array.
    """
    # Calculate quantiles directly on GPU
    min_val, max_val = cp.percentile(data, cp.array([0.1, 99.9]))

    #For some reason, the linspace function doesn't work with arrays for the third value, so need to get CPU info.
    bin_edges = cp.linspace(min_val, max_val, num_bins.get()+1)
    
    # Compute histogram
    hist, _ = cp.histogram(data, bins=bin_edges, density=cp.asarray(True, dtype=cp.bool_))
    if filter_extremes:
        # Filter low-density regions
        density_threshold = cp.max(hist) * 0.01
        hist = cp.where(hist < density_threshold, cp.asarray(0, dtype=cp.int32), hist)
        # Zero out extremes
        bottom_bins = num_bins * 0.02
        top_bins = num_bins * 0.98
        hist[:bottom_bins] = hist[top_bins:] = cp.asarray(0, dtype=cp.int32)
        # Filter by average density
        avg_density_threshold = cp.sum(hist) / len(hist) * 0.25
        hist = cp.where(hist < avg_density_threshold, cp.asarray(0, dtype=cp.int32), hist)
    
    # Smooth histogram
    # For some reason, the smoothing window needs to be an int and not an array, so need to get the value.
    kernel = cp.ones(int(smoothing_window.item()), dtype=cp.int32) / smoothing_window.item()
    smoothed_hist = cp.convolve(hist, kernel, mode='same')
    return HistogramAnalysis(hist=hist, bin_edges=bin_edges, smoothed_hist=smoothed_hist)

def find_peaks(hist: cp.ndarray, smoothing_window: cp.int32) -> cp.ndarray:
    """
    Find peaks in histogram.
    Assumes hist is a CuPy array.
    """
    import time
    timings = {}
    
    # Time initial peak detection
    t0 = time.perf_counter()
    maxima = (hist >= cp.roll(hist, 1)) & (hist >= cp.roll(hist, -1))
    peak_candidates = cp.where(maxima)[0]
    t1 = time.perf_counter()
    timings['initial_peak_detection'] = t1 - t0
    
    if len(peak_candidates) == 0:
        return peak_candidates
        
    # Time candidate filtering
    t0 = time.perf_counter()
    valid_peaks = cp.array([], dtype=cp.int32)
    for peak_idx in peak_candidates:
        window_start = cp.maximum(0, peak_idx - smoothing_window)
        window_end = cp.minimum(len(hist), peak_idx + smoothing_window + 1)
        window_max_idx = window_start + cp.argmax(hist[window_start:window_end])
        
        if window_max_idx == peak_idx:
            valid_peaks = cp.append(valid_peaks, peak_idx)
    t1 = time.perf_counter()
    timings['candidate_filtering'] = t1 - t0
    
    # Time prominence filtering
    t0 = time.perf_counter()
    if len(valid_peaks) > 2:
        prominent_peaks = cp.array([], dtype=cp.int32)
        for i, peak_idx in enumerate(valid_peaks):
            peak_height = hist[peak_idx]
            min_heights = cp.array([], dtype=cp.float32)
            
            if i > 0:
                left_min = cp.min(hist[valid_peaks[i-1]:peak_idx])
                min_heights = cp.append(min_heights, left_min)
            
            if i < len(valid_peaks) - 1:
                right_min = cp.min(hist[peak_idx:valid_peaks[i+1]])
                min_heights = cp.append(min_heights, right_min)
            
            if len(min_heights) > 0:
                highest_min = cp.max(min_heights)
                prominence = (peak_height - highest_min) / peak_height
                if prominence >= 0.1:
                    prominent_peaks = cp.append(prominent_peaks, peak_idx)
            else:
                prominent_peaks = cp.append(prominent_peaks, peak_idx)
        
        t1 = time.perf_counter()
        timings['prominence_filtering'] = t1 - t0
        print(f"Peak finding timings: {timings}")
        return prominent_peaks
    
    t1 = time.perf_counter()
    timings['prominence_filtering'] = t1 - t0
    print(f"Peak finding timings: {timings}")
    return valid_peaks

def calculate_peak_densities(hist: cp.ndarray, peak_indices: cp.ndarray, 
                           smoothing_window: cp.int32) -> cp.ndarray:
    """
    Calculate density around each peak.
    Assumes inputs are CuPy arrays.
    """
    # Create windows for all peaks at once using advanced indexing
    window_starts = cp.maximum(cp.int32(0), peak_indices - smoothing_window)
    window_ends = cp.minimum(cp.int32(len(hist)), peak_indices + smoothing_window + 1)
    
    # Use cp.add.reduceat to sum each window efficiently
    densities = cp.array([cp.sum(hist[start:end]) for start, end in zip(window_starts, window_ends)], dtype=cp.float32)
    
    return densities

def find_left_minimum(hist: cp.ndarray, start_idx: cp.int32, smoothing_window: cp.int32) -> cp.int32:
    """
    Find left minimum before peak using array operations.
    Assumes hist is a CuPy array.
    """
    # Handle edge case - if start_idx <= 1, return 0
    if start_idx == 0:
        return cp.int32(0)
        
    # Subset histogram up to start_idx
    left_hist = hist[:start_idx]
    
    # Find indices where value is 0 or is less than both neighbors
    # Skip first and last elements since they need both neighbors
    is_zero = left_hist[1:-1] == 0
    is_minimum = (left_hist[1:-1] <= left_hist[:-2]) & (left_hist[1:-1] <= left_hist[2:])
    valid_idx = cp.where(is_zero | is_minimum)[0]
    
    # Return rightmost valid index + 1 (to account for the offset from slicing),
    # or 0 if no valid minimum found
    return cp.int32(valid_idx[-1] + 1) if len(valid_idx) > 0 else cp.int32(0)

def process_histogram(feature: cp.ndarray, smoothing_window: cp.int32, 
                     num_bins: cp.int32 = 100, 
                     filter_extremes: cp.bool_ = True) -> Optional[Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]]:
    """
    Process histogram and find peaks.
    Assumes feature is a CuPy array.
    
    Returns:
        Tuple of (smoothed_hist, bin_edges, peak_indices, peak_densities)
        All returned arrays are CuPy arrays
    """

    try:
        hist_analysis = analyze_histogram(
            feature, 
            num_bins=num_bins,
            smoothing_window=smoothing_window,
            filter_extremes=filter_extremes
        )
        
        peak_indices = find_peaks(hist_analysis.smoothed_hist, smoothing_window)
        peak_densities = calculate_peak_densities(
            hist_analysis.smoothed_hist, 
            peak_indices, 
            smoothing_window
        )
        
        return (hist_analysis.smoothed_hist, hist_analysis.bin_edges, 
                peak_indices, peak_densities)
    except Exception as e:
        print(f"Error in process_histogram: {str(e)}")
        return None

def standardize_marker_name(name: str) -> str:
    """Standardize marker names by removing symbols and converting to lowercase."""
    import re
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def is_excluded_marker(marker: str) -> bool:
    """Check if marker should be excluded from analysis."""
    excluded = ['time', 'fsc-a', 'fsc-h', 'fsc-w', 'ssc-a', 'ssc-h', 'ssc-w']
    return marker.lower() in excluded