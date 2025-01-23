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

def analyze_histogram(data: np.ndarray, num_bins: int, smoothing_window: int) -> HistogramAnalysis:
    """
    Analyze histogram of data.
    
    Args:
        data: Input data
        num_bins: Number of bins for histogram
        smoothing_window: Window size for smoothing
        
    Returns:
        HistogramAnalysis object
    """
    min_val, max_val = np.min(data), np.max(data)
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    hist, _ = np.histogram(data, bins=bin_edges, density=True)
    
    # Zero out extremes
    bottom_bins = int(num_bins * 0.02)
    top_bins = int(num_bins * 0.98)
    hist[:bottom_bins] = hist[top_bins:] = 0

    # Zero out low-density bins (less than 1% of max density)
    density_threshold = np.max(hist) * 0.02
    hist[hist < density_threshold] = 0
    
    # Zero out bins below average density threshold
    avg_density_threshold = np.sum(hist) / len(hist) * 0.5
    hist[hist < avg_density_threshold] = 0
    
    # Smooth histogram
    smoothed_hist = np.convolve(hist, np.ones(smoothing_window) / smoothing_window, mode='same')
    
    return HistogramAnalysis(hist=hist, bin_edges=bin_edges, smoothed_hist=smoothed_hist)

def find_peaks(hist: np.ndarray, smoothing_window: int) -> np.ndarray:
    """
    Find peaks in histogram, ensuring they are separated by at least the smoothing window.
    
    Args:
        hist: Input histogram
        smoothing_window: Window size for peak detection
        
    Returns:
        Array of peak indices
    """
    # Find points that are higher than their immediate neighbors
    maxima = (hist > np.roll(hist, 1)) & (hist > np.roll(hist, -1))
    maxima[:smoothing_window] = maxima[-smoothing_window:] = False
    
    # Get initial peak candidates
    peak_candidates = np.where(maxima)[0]
    
    if len(peak_candidates) == 0:
        return peak_candidates
        
    # Filter peaks to ensure they are the highest point within smoothing_window
    valid_peaks = []
    for peak_idx in peak_candidates:
        # Define window boundaries
        window_start = max(0, peak_idx - smoothing_window)
        window_end = min(len(hist), peak_idx + smoothing_window + 1)
        
        # Check if this point is the highest in its window
        window_max_idx = window_start + np.argmax(hist[window_start:window_end])
        if window_max_idx == peak_idx:
            valid_peaks.append(peak_idx)
    
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

def process_histogram(feature: np.ndarray, smoothing_window: int, num_bins: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process histogram and find peaks.
    
    Args:
        feature: Array of values
        smoothing_window: Window size for smoothing
        num_bins: Number of bins for histogram
            
    Returns:
        Tuple of (thresholded_hist, bin_edges, peak_indices, peak_densities)
    """
    hist_analysis = analyze_histogram(feature, num_bins=num_bins, smoothing_window=smoothing_window)
    
    if hist_analysis.smoothed_hist is None:
        return None
            
    peak_indices = find_peaks(hist_analysis.smoothed_hist, smoothing_window)
    if len(peak_indices) <= 1:
        return None
        
    peak_densities = calculate_peak_densities(hist_analysis.smoothed_hist, peak_indices, smoothing_window)
    return hist_analysis.smoothed_hist, hist_analysis.bin_edges, peak_indices, peak_densities

def standardize_marker_name(name: str) -> str:
    """Standardize marker names by removing symbols and converting to lowercase."""
    import re
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def is_excluded_marker(marker: str) -> bool:
    """Check if marker should be excluded from analysis."""
    return marker.lower() in ['time', 'fsc-a', 'fsc-h', 'fsc-w', 'ssc-a', 'ssc-h', 'ssc-w'] 