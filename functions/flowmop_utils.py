"""
Shared utility functions for FlowMOP modules.
"""

import numpy as np
from typing import List, Tuple, NamedTuple
from scipy.ndimage import maximum_filter1d

class Peak(NamedTuple):
    """Represents a peak in the data with its boundaries."""
    start: float
    end: float

class HistogramAnalysis(NamedTuple):
    """Results of histogram analysis."""
    hist: np.ndarray
    bin_edges: np.ndarray
    smoothed_hist: np.ndarray

def analyze_histogram(data: np.ndarray, num_bins: int, smoothing_window: int, filter_extremes: bool = True) -> HistogramAnalysis:
    """
    Analyze histogram of data.
    
    Args:
        data: Input data
        num_bins: Number of bins for histogram
        smoothing_window: Window size for smoothing
        
    Returns:
        HistogramAnalysis object
    """

    min_val, max_val = np.percentile(data, [0.1, 99.9])
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    hist, _ = np.histogram(data, bins=bin_edges, density=True)

    # Zero out low-density bins (less than 1% of max density)
    if filter_extremes:
        density_threshold = np.max(hist) * 0.01
        hist[hist < density_threshold] = 0
        
    # Zero out extremes
    bottom_bins = int(num_bins * 0.02)
    top_bins = int(num_bins * 0.98)
    hist[:bottom_bins] = hist[top_bins:] = 0
        
    # Zero out bins below average density threshold
    if filter_extremes:
        avg_density_threshold = np.sum(hist) / len(hist) * 0.25
        hist[hist < avg_density_threshold] = 0
    
    # Smooth histogram
    smoothed_hist = np.convolve(hist, np.ones(smoothing_window) / smoothing_window, mode='same')

    return HistogramAnalysis(hist=hist, bin_edges=bin_edges, smoothed_hist=smoothed_hist)

def find_peaks(hist: np.ndarray, smoothing_window: int) -> np.ndarray:
    """
    Find peaks in histogram, ensuring they are separated by at least the smoothing window
    and have sufficient prominence relative to surrounding minima.
    
    Args:
        hist: Input histogram
        smoothing_window: Window size for peak detection
        
    Returns:
        Array of peak indices
    """
    import time
    timings = {}
    
    # Time initial peak detection
    t0 = time.perf_counter()
    maxima = (hist >= np.roll(hist, 1)) & (hist >= np.roll(hist, -1))
    peak_candidates = np.where(maxima)[0]
    t1 = time.perf_counter()
    timings['initial_peak_detection'] = t1 - t0
    
    if len(peak_candidates) == 0:
        return peak_candidates
        
    # Time candidate filtering
    t0 = time.perf_counter()
    valid_peaks = []
    for peak_idx in peak_candidates:
        window_start = max(0, peak_idx - smoothing_window)
        window_end = min(len(hist), peak_idx + smoothing_window + 1)
        
        window_max_idx = window_start + np.argmax(hist[window_start:window_end])
        if window_max_idx == peak_idx:
            valid_peaks.append(peak_idx)
    t1 = time.perf_counter()
    timings['candidate_filtering'] = t1 - t0
    
    # Time prominence filtering
    t0 = time.perf_counter()
    if len(valid_peaks) > 2:
        prominent_peaks = []
        for i, peak_idx in enumerate(valid_peaks):
            peak_height = hist[peak_idx]
            min_heights = []
            
            if i > 0:
                left_peak_idx = valid_peaks[i-1]
                left_min = np.min(hist[left_peak_idx:peak_idx])
                min_heights.append(left_min)
            
            if i < len(valid_peaks) - 1:
                right_peak_idx = valid_peaks[i+1]
                right_min = np.min(hist[peak_idx:right_peak_idx])
                min_heights.append(right_min)
            
            if min_heights:
                highest_min = max(min_heights)
                # Avoid division by zero or negative values that could cause warnings
                if peak_height > 0 and peak_height > highest_min:
                    prominence = (peak_height - highest_min) / peak_height
                else:
                    prominence = 0.0  # Default to zero prominence if calculation is invalid
                if prominence >= 0.1:
                    prominent_peaks.append(peak_idx)
            else:
                prominent_peaks.append(peak_idx)
        
        t1 = time.perf_counter()
        timings['prominence_filtering'] = t1 - t0
        return np.array(prominent_peaks)
    
    t1 = time.perf_counter()
    timings['prominence_filtering'] = t1 - t0
    return np.array(valid_peaks)

def calculate_peak_densities(hist: np.ndarray, peak_indices: np.ndarray, 
                           smoothing_window: int) -> List[float]:
    """
    Calculate density around each peak.
    
    Args:
        hist: Input histogram
        peak_indices: Array of peak indices
        smoothing_window: Window size for density calculation
        
    Returns:
        List of peak densities
    """
    return [
        np.sum(hist[max(0, idx - smoothing_window):
                   min(idx + smoothing_window + 1, len(hist))])
            for idx in peak_indices
        ]

def find_left_minimum(hist: np.ndarray, start_idx: int, smoothing_window: int) -> int:
    """
    Find left minimum before peak.
    
    Args:
        hist: Input histogram
        start_idx: Starting index
        smoothing_window: Window size for minimum detection
        
    Returns:
        Index of left minimum
    """
    # Start from the peak and move left until we find a minimum
    idx = start_idx
    while idx > 0:
        # If we hit a zero value or find a minimum, stop
        if hist[idx] == 0 or (hist[idx] <= hist[idx-1] and hist[idx] <= hist[idx+1]):
            break
        idx -= 1
    
    return idx

def process_histogram(feature: np.ndarray, smoothing_window: int, num_bins: int = 100, filter_extremes: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process histogram and find peaks.
    
    Args:
        feature: Array of values
        smoothing_window: Window size for smoothing
        num_bins: Number of bins for histogram
            
    Returns:
        Tuple of (thresholded_hist, bin_edges, peak_indices, peak_densities)
        Returns None only if histogram analysis fails
    """
    hist_analysis = analyze_histogram(feature, num_bins=num_bins, smoothing_window=smoothing_window, filter_extremes=filter_extremes)
    
    if hist_analysis.smoothed_hist is None:
        return None

    peak_indices = find_peaks(hist_analysis.smoothed_hist, smoothing_window)    
    # Always calculate peak densities, even with one peak
    peak_densities = calculate_peak_densities(hist_analysis.smoothed_hist, peak_indices, smoothing_window)
    return hist_analysis.smoothed_hist, hist_analysis.bin_edges, peak_indices, peak_densities

def standardize_marker_name(name: str) -> str:
    """Standardize marker names by removing symbols and converting to lowercase."""
    import re
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def is_excluded_marker(marker: str) -> bool:
    """Check if marker should be excluded from analysis."""
    return marker.lower() in ['time', 'fsc-a', 'fsc-h', 'fsc-w', 'ssc-a', 'ssc-h', 'ssc-w']

def normalize_timeseries_values(values: np.ndarray) -> np.ndarray:
    """
    Normalize time series values for consistent MAD-based outlier detection.
    
    Performs these steps:
    1. Normalize to mean=1
    2. Scale to target standard deviation of 0.1
    3. Shift values so median is 1
    
    Args:
        values: Time series values to normalize
        
    Returns:
        Normalized values
    """
    # Ensure we have a copy to avoid modifying the original
    normalized = values.copy()
    
    # Normalize to mean=1
    normalized = normalized / np.mean(normalized)
    
    # Scale to target standard deviation of 0.1
    current_std = np.std(normalized)
    if current_std > 0:  # Avoid division by zero
        target_std = 0.1
        normalized = normalized * (target_std / current_std)
    
    # Shift values so median is 1
    current_median = np.median(normalized)
    normalized = normalized + (1 - current_median)
    
    return normalized

def apply_spline_smoothing(values: np.ndarray, smoothing_factor: float, n_bins: int = None) -> np.ndarray:
    """
    Apply spline smoothing to time series values with adaptive smoothing factor.
    
    Args:
        values: Time series values to smooth
        smoothing_factor: Base smoothing factor
        n_bins: Number of bins for scaling the smoothing factor
        
    Returns:
        Smoothed values
    """
    from scipy.interpolate import UnivariateSpline
    
    # Scale smoothing factor based on number of bins
    if n_bins is None:
        n_bins = len(values)

    if len(values) <= 3 or smoothing_factor <= 0:
        return np.asarray(values)
    
    # Scale smoothing factor based on number of bins
    scaled_smoothing = smoothing_factor * n_bins / 100
    # Clamp between 0.1 and 2.0
    scaled_smoothing = max(0.1, min(2.0, scaled_smoothing))
    
    # Apply spline smoothing
    x_indices = np.arange(len(values))
    spline = UnivariateSpline(x_indices, values, s=scaled_smoothing)
    return spline(x_indices)

def calculate_mad_thresholds(values: np.ndarray, mad_threshold: float) -> tuple[float, float, np.ndarray]:
    """
    Calculate MAD-based outlier detection thresholds.
    
    Args:
        values: Time series values (typically smoothed)
        mad_threshold: Number of MADs to use as threshold
        
    Returns:
        Tuple of (lower_threshold, upper_threshold, outlier_mask)
    """
    # Calculate MAD thresholds
    median_val = np.median(values)
    mad_val = np.median(np.abs(values - median_val))
    
    upper_threshold = median_val + mad_threshold * mad_val
    lower_threshold = median_val - mad_threshold * mad_val
    
    # Identify outliers
    outliers = (values > upper_threshold) | (values < lower_threshold)
    
    return lower_threshold, upper_threshold, outliers 
