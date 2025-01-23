"""
Debris gating component for FlowMOP.
Handles detection and removal of debris in flow cytometry data.
"""

import numpy as np
from abc import ABC, abstractmethod
import warnings
from typing import List, Tuple, Optional, NamedTuple
import matplotlib.pyplot as plt

class Peak(NamedTuple):
    """Represents a peak in the data with its boundaries."""
    start: float
    end: float

class HistogramAnalysis(NamedTuple):
    """Results of histogram analysis."""
    hist: np.ndarray
    bin_edges: np.ndarray
    smoothed_hist: np.ndarray

class DebrisGateStrategy(ABC):
    @abstractmethod
    def gate(self, data: np.ndarray, marker_names: list[str]) -> tuple[np.ndarray, float, np.ndarray]:
        """Apply debris gating to the data."""
        pass

class FSCDebrisGate(DebrisGateStrategy):
    def __init__(self, min_peaks=2, max_peaks=5, smoothing_window=3, percentage_cells_present=5, num_bins=100):
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks
        self.smoothing_window = smoothing_window
        self.percentage_cells_present = percentage_cells_present
        self.num_bins = num_bins

    def gate(self, data: np.ndarray, marker_names: list[str]) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Apply FSC-based debris gating to the data.
        
        Args:
            data: Flow cytometry data array
            marker_names: List of marker names
            
        Returns:
            tuple: (filtered_data, fsc_threshold, debris_vector)
        """
        if not self._check_events_in_bottom_bin(data, marker_names):
            return data, None, np.ones(data.shape[0], dtype=int)

        valid_peaks_mask, peaks_list, positive_masks = detect_fluoropeaks(
            data.T, 
            marker_names,
            min_peaks=self.min_peaks,
            max_peaks=self.max_peaks,
            smoothing_window=self.smoothing_window,
            percentage_cells_present=self.percentage_cells_present,
            num_bins=self.num_bins
        )
        print(f"Valid peaks mask: {valid_peaks_mask}")
        print("len peaks is ", len(peaks_list))

        fsc_column = self._get_fsc_column(marker_names)
        # Get FSC thresholds from valid peaks
        fsc_thresholds = self._get_fsc_thresholds(data, valid_peaks_mask, peaks_list, positive_masks, fsc_column)

        if not fsc_thresholds:
            return data, None, np.ones(data.shape[0], dtype=int)

        # Calculate final threshold and apply gating
        fsc_gate_threshold = np.nanmedian(fsc_thresholds)
        print("FSC gate threshold: ", fsc_gate_threshold)
        debris_vector = (data[:, fsc_column] >= fsc_gate_threshold).astype(int)
        filtered_data = data[debris_vector == 1]

        return filtered_data, fsc_gate_threshold, debris_vector

    def _get_fsc_thresholds(self, data: np.ndarray, valid_peaks_mask: np.ndarray, 
                           peaks_list: List[List[Peak]], positive_masks: List[np.ndarray],
                           fsc_column: int) -> List[float]:
        """Get FSC thresholds for features with valid peaks."""
        # First get reference FSC peaks from all cells using _process_histogram
        result = _process_histogram(data[:, fsc_column], self.smoothing_window)
        if result is None:
            return []
        
        _, bin_edges, ref_peak_indices, _ = result
        all_fsc_peaks = [(idx, bin_edges[idx]) for idx in ref_peak_indices]
        all_fsc_peaks = sorted(all_fsc_peaks, key=lambda x: x[1])
        
        fsc_thresholds = []

        # For every feature, run through and check for peaks
        for i, (is_valid, peaks, positive_mask) in enumerate(zip(valid_peaks_mask, peaks_list, positive_masks)):
            if is_valid and len(peaks) >= 2 and positive_mask is not None:
                # Use the positive mask from detect_fluoropeaks
                positive_fsc = data[positive_mask, fsc_column]
                
                # Plot FSC histogram for positive cells
                plt.figure(figsize=(10, 4))
                plt.hist(positive_fsc, bins=self.num_bins, density=True, alpha=0.7)
                plt.title(f'FSC Distribution for Feature {i} Positive Cells')
                plt.xlabel('FSC-A')
                plt.ylabel('Density')
                
                # Process histogram for this positive population
                pos_result = _process_histogram(positive_fsc, self.smoothing_window)
                if pos_result is not None:
                    pos_hist, pos_bin_edges, pos_peak_indices, _ = pos_result
                    
                    # Plot detected peaks in this population
                    peak_x = pos_bin_edges[pos_peak_indices]
                    peak_y = pos_hist[pos_peak_indices]
                    plt.plot(peak_x, peak_y, 'mo', label='Detected Peaks')
                    # Add vertical lines for detected peaks
                    for x in peak_x:
                        plt.axvline(x=x, color='black', linestyle='--', alpha=0.3)
                
                # Add vertical lines for reference peaks from all cells
                for _, peak_pos in all_fsc_peaks:
                    plt.axvline(x=peak_pos, color='r', linestyle='--', alpha=0.5, label='Reference Peaks')
                
                # Find FSC threshold using positive cells and reference peaks
                threshold = self._find_fsc_threshold(pos_result, all_fsc_peaks, self.smoothing_window)
                if threshold is not None:
                    plt.axvline(x=threshold, color='g', linestyle='-', label='Threshold')
                    fsc_thresholds.append(threshold)
                else:
                    fsc_thresholds.append(np.nan)
                
                # Plot smoothed histogram if available
                if pos_result is not None:
                    plt.plot(pos_bin_edges[:-1], pos_result[0], 'b-', label='Smoothed', alpha=0.7)
                
                plt.legend()
                plt.show()
                plt.close()
                
        return fsc_thresholds

    def _is_excluded_marker(self, marker: str) -> bool:
        """Check if marker should be excluded from analysis."""
        return marker.lower() in ['time', 'fsc-a', 'fsc-h', 'fsc-w', 'ssc-a', 'ssc-h', 'ssc-w']

    def _check_events_in_bottom_bin(self, fcs_array: np.ndarray, marker_names: list[str]) -> bool:
        """Check if events exist in the bottom 10th bin of FSC-A and SSC-A."""
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        
        try:
            fsc_a_column = standardized_names.index('fsca')
            ssc_a_column = standardized_names.index('ssca')
        except ValueError:
            warnings.warn("Required FSC-A or SSC-A parameters not found.")
            return False

        fsc_a_max = np.max(fcs_array[:, fsc_a_column])
        ssc_a_max = np.max(fcs_array[:, ssc_a_column])
        
        fsc_a_bottom_10th = fsc_a_max * 0.1
        ssc_a_bottom_10th = ssc_a_max * 0.1
        
        events_in_bottom_bin = np.any(
            (fcs_array[:, fsc_a_column] <= fsc_a_bottom_10th) & 
            (fcs_array[:, ssc_a_column] <= ssc_a_bottom_10th)
        )
        
        if not events_in_bottom_bin:
            warnings.warn("No events found in the bottom 10th bin of FSC-A and SSC-A. Debris removal will be skipped.")
        
        return events_in_bottom_bin

    def _get_fsc_column(self, marker_names: list[str]) -> int:
        """Get the index of the FSC-A column."""
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        try:
            return standardized_names.index('fsca')
        except ValueError:
            raise ValueError("FSC-A parameter not found in marker names.")

    @staticmethod
    def _standardize_marker_name(name: str) -> str:
        """Standardize marker names by removing symbols and converting to lowercase."""
        import re
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

    def _has_low_peak(self, bin_edges: np.ndarray, peak_indices: np.ndarray, 
                     lowest_reference_pos: float, tolerance: float = 1.05) -> bool:
        """
        Check if any peaks are near or below the reference lowest peak.
        
        Args:
            bin_edges: Bin edges from histogram
            peak_indices: Indices of peaks
            lowest_reference_pos: Position of lowest reference peak
            tolerance: Multiplier for reference position to define "near"
            
        Returns:
            bool: True if low peak exists
        """
        print("Checking for low peaks near or below reference peak")
        print("Lowest reference peak position: ", lowest_reference_pos)
        print("Peak indices: ", peak_indices)
        return any(bin_edges[idx] <= lowest_reference_pos * tolerance for idx in peak_indices)

    def _get_left_boundary_threshold(self, hist: np.ndarray, bin_edges: np.ndarray, 
                                   peak_indices: np.ndarray, smoothing_window: int) -> float:
        """
        Get threshold using left boundary of lowest peak.
        
        Args:
            hist: Histogram
            bin_edges: Bin edges
            peak_indices: Peak indices
            smoothing_window: Smoothing window size
            
        Returns:
            float: Threshold value
        """
        lowest_peak_idx = peak_indices[0]  # Already sorted by position
        valley_idx = _find_left_minimum(hist, lowest_peak_idx, smoothing_window)
        return bin_edges[valley_idx]

    def _get_max_peak_threshold(self, thresholded_hist: np.ndarray, bin_edges: np.ndarray,
                              peak_indices: np.ndarray, peak_densities: np.ndarray,
                              smoothing_window: int) -> Optional[float]:
        """
        Get threshold using max peak logic.
        
        Args:
            thresholded_hist: Thresholded histogram
            bin_edges: Bin edges
            peak_indices: Peak indices
            peak_densities: Peak densities
            smoothing_window: Window size for smoothing
            
        Returns:
            float or None: Threshold value
        """
        max_peak_idx = np.argmax(peak_densities)
        
        # If biggest peak is first peak (debris peak), find valley between first and second peaks
        if max_peak_idx == 0 and len(peak_indices) > 1:
            left_min_idx = _find_left_minimum(thresholded_hist, peak_indices[1], smoothing_window)
            print("First peak is debris peak, using valley between first and second peaks")
            return bin_edges[left_min_idx]
            
        # If biggest peak is second peak
        if max_peak_idx == 1:
            left_min_idx = _find_left_minimum(thresholded_hist, peak_indices[max_peak_idx], smoothing_window)
            print("Second peak is biggest peak, using valley between second and first peaks")
            return bin_edges[left_min_idx]
        
        # Get all peak positions for comparison
        all_peak_positions = bin_edges[peak_indices]
        
        # Search for appropriate minimum before max peak
        # For each potential valley, check if it's smaller than ALL peaks
        for i in range(max_peak_idx - 1, -1, -1):
            left_min_idx = _find_left_minimum(thresholded_hist, peak_indices[i], smoothing_window)
            valley_position = bin_edges[left_min_idx]
            print("Valley position: ", valley_position)
            # Check if this valley is smaller than ALL peaks
            if all(valley_position < peak_pos for peak_pos in all_peak_positions):
                return valley_position
        
        return None

    def _find_fsc_threshold(self, process_histogram_result: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], reference_peaks: List[Tuple[int, float]], 
                          smoothing_window: int) -> Optional[float]:
        """
        Find the FSC threshold by comparing peaks to reference peaks from all cells.
        
        Args:
            process_histogram_result: Result from _process_histogram containing (smoothed_hist, bin_edges, peak_indices, peak_densities)
            reference_peaks: List of (peak_index, peak_position) from all cells
            smoothing_window: Window size for smoothing
            pos_peak_indices: Peak indices from the positive population
            pos_bin_edges: Bin edges from the positive population histogram
            
        Returns:
            float or None: The FSC threshold value, or None if no valid peaks found
        """
        # Get the lowest FSC peak position from reference
        lowest_reference_peak_pos = reference_peaks[0][1]
        smoothed_hist, pos_bin_edges, pos_peak_indices, peak_densities = process_histogram_result
        
        # Check if we have any peaks near or below the reference lowest peak
        if not self._has_low_peak(pos_bin_edges, pos_peak_indices, lowest_reference_peak_pos):
            # If we don't have any low peaks, use left boundary of lowest available peak
            return self._get_left_boundary_threshold(smoothed_hist, pos_bin_edges, pos_peak_indices, smoothing_window)
            
        # Use traditional max peak logic
        return self._get_max_peak_threshold(smoothed_hist, pos_bin_edges, pos_peak_indices, peak_densities, smoothing_window)

# Utility functions
def _analyze_histogram(data: np.ndarray, num_bins: int, smoothing_window: int) -> HistogramAnalysis:
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

def _is_excluded_marker(marker: str) -> bool:
    """Check if marker should be excluded from analysis."""
    return marker.lower() in ['time', 'fsc-a', 'fsc-h', 'fsc-w', 'ssc-a', 'ssc-h', 'ssc-w']

def _process_histogram(feature: np.ndarray, smoothing_window: int, num_bins: int = 100) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Process histogram and find peaks.
    
    Args:
        feature: Array of values
        smoothing_window: Window size for smoothing
        num_bins: Number of bins for histogram
            
    Returns:
        Optional tuple of (thresholded_hist, bin_edges, peak_indices, peak_densities)
    """
    hist_analysis = _analyze_histogram(feature, num_bins=num_bins, smoothing_window=smoothing_window)
    
    if hist_analysis.smoothed_hist is None:
        return None
            
    peak_indices = _find_peaks(hist_analysis.smoothed_hist, smoothing_window)
    if len(peak_indices) <= 1:
        return None
        
    peak_densities = _calculate_peak_densities(hist_analysis.smoothed_hist, peak_indices, smoothing_window)
    return hist_analysis.smoothed_hist, hist_analysis.bin_edges, peak_indices, peak_densities

def detect_fluoropeaks(data: np.ndarray, marker_names: List[str], min_peaks: int = 2, max_peaks: int = 5, 
                         smoothing_window: int = 2, percentage_cells_present: float = 5, num_bins: int = 100) -> Tuple[np.ndarray, List[List[Peak]], List[np.ndarray]]:
    """
    Detect debris peaks in the data.
    
    Args:
        data: Input data array
        marker_names: List of marker names
        min_peaks: Minimum number of peaks required
        max_peaks: Maximum number of peaks to consider
        smoothing_window: Window size for smoothing
        percentage_cells_present: Minimum percentage of cells in a peak
        num_bins: Number of bins for histogram analysis
        
    Returns:
        Tuple of (valid_peaks_mask, peaks_list, positive_masks)
    """
    num_features, num_samples = data.shape
    data_transformed = np.arcsinh(data / 150)
    peaks_list = []
    valid_peaks_mask = np.zeros(num_features, dtype=bool)
    positive_masks = []  # Store positive masks for each feature
    
    for i, marker in enumerate(marker_names):
        # Skip excluded channels
        if _is_excluded_marker(marker):
            peaks_list.append([])
            positive_masks.append(None)
            continue
            
        # Process histogram and get peaks
        result = _process_histogram(data_transformed[i], smoothing_window)
        if result is None:
            peaks_list.append([])
            positive_masks.append(None)
            continue
            
        thresholded_hist, bin_edges, peak_indices, peak_densities = result
        
        # First sort peaks by position
        peak_positions = [(idx, bin_edges[idx]) for idx in peak_indices]
        sorted_peak_positions = sorted(peak_positions, key=lambda x: x[1])
        sorted_peak_indices = [pos[0] for pos in sorted_peak_positions]
        
        # Then get the top peaks by density, but maintain position order
        if len(sorted_peak_indices) > max_peaks:
            peak_densities = [thresholded_hist[idx] for idx in sorted_peak_indices]
            density_threshold = sorted(peak_densities, reverse=True)[max_peaks-1]
            sorted_peak_indices = [idx for idx, density in zip(sorted_peak_indices, peak_densities) 
                                 if density >= density_threshold]
        
        # Get peak widths
        peak_widths = peak_width_debris(thresholded_hist, sorted_peak_indices, 
                                      bin_edges, percentage_cells_present)
        
        # Convert to Peak objects
        peaks = [Peak(start=start, end=end) for start, end in peak_widths]
        peaks_list.append(peaks)
        valid_peaks_mask[i] = len(peaks) >= min_peaks
        
        # If we have at least 2 peaks, calculate positive cells
        if len(peaks) >= 2:
            second_peak = peaks[1]
            positive_mask = data_transformed[i] >= second_peak.start
            positive_masks.append(positive_mask)
        else:
            positive_masks.append(None)
            
    return valid_peaks_mask, peaks_list, positive_masks

def peak_width_debris(smoothed_hist, peak_indices, bin_edges, percentage_cells_present=5):
    """Calculate the width of debris peaks."""
    num_bins = len(smoothed_hist)
    
    # Sort peak indices by position
    peak_indices = sorted(peak_indices)
    peak_widths = []
    
    for i, peak_index in enumerate(peak_indices):
        # For first peak, search left until we find a minimum
        if i == 0:
            # Search backwards from peak to find minimum
            left_idx = peak_index
            while left_idx > 0:
                if smoothed_hist[left_idx] <= smoothed_hist[left_idx - 1]:
                    left_idx -= 1
                else:
                    break
        else:
            # For all other peaks, find minimum between this peak and previous peak
            prev_peak = peak_indices[i-1]
            segment = smoothed_hist[prev_peak:peak_index+1]
            left_idx = prev_peak + np.argmin(segment)
        
        # For last peak, search right until we find a minimum
        if i == len(peak_indices) - 1:
            # Search forwards from peak to find minimum
            right_idx = peak_index
            while right_idx < num_bins - 1:
                if smoothed_hist[right_idx] <= smoothed_hist[right_idx + 1]:
                    right_idx += 1
                else:
                    break
        else:
            # For all other peaks, find minimum between this peak and next peak
            next_peak = peak_indices[i+1]
            segment = smoothed_hist[peak_index:next_peak+1]
            right_idx = peak_index + np.argmin(segment)
        
        # Calculate peak percentage
        peak_percentage = np.sum(smoothed_hist[left_idx:right_idx + 1]) / np.sum(smoothed_hist) * 100
        if peak_percentage >= percentage_cells_present:
            peak_widths.append((bin_edges[left_idx], bin_edges[right_idx]))
    
    return peak_widths

def _find_peaks(hist: np.ndarray, smoothing_window: int) -> np.ndarray:
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

def _calculate_peak_densities(hist: np.ndarray, peak_indices: np.ndarray, 
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

def _find_left_minimum(hist: np.ndarray, start_idx: int, smoothing_window: int) -> int:
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
