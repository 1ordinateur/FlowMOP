"""
Debris gating component for FlowMOP.
Handles detection and removal of debris in flow cytometry data.
"""

import numpy as np
from abc import ABC, abstractmethod
import warnings
from typing import List, Tuple, Optional, NamedTuple
import matplotlib.pyplot as plt
from flowmop_utils import Peak, process_histogram, standardize_marker_name, is_excluded_marker, find_left_minimum

def plot_debris_gating_results(data: np.ndarray, fsc_column: int, threshold: float, 
                             positive_mask: Optional[np.ndarray] = None,
                             smoothing_window: int = 3, num_bins: int = 100):
    """
    Plot debris gating results showing original and filtered FSC distributions.
    
    Args:
        data: Flow cytometry data array
        fsc_column: Index of FSC-A column
        threshold: FSC threshold value
        positive_mask: Optional mask for positive population
        smoothing_window: Window size for smoothing
        num_bins: Number of bins for histogram
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: All cells FSC distribution
    result = process_histogram(data[:, fsc_column], smoothing_window, num_bins)
    if result is not None:
        hist, bin_edges, peak_indices, _ = result
        
        # Plot raw histogram
        ax1.hist(data[:, fsc_column], bins=num_bins, density=True, alpha=0.5, label='Raw Data')
        
        # Plot smoothed histogram and peaks
        ax1.plot(bin_edges[:-1], hist, 'b-', label='Smoothed', alpha=0.7)
        peak_x = bin_edges[peak_indices]
        peak_y = hist[peak_indices]
        ax1.plot(peak_x, peak_y, 'ro', label='Peaks')
        
        # Add vertical lines for peaks
        for x in peak_x:
            ax1.axvline(x=x, color='gray', linestyle='--', alpha=0.3)
            
        ax1.set_title('FSC Distribution - All Cells')
        ax1.set_xlabel('FSC-A')
        ax1.set_ylabel('Density')
        ax1.legend()
    
    # Plot 2: Positive population FSC distribution
    if positive_mask is not None:
        positive_fsc = data[positive_mask, fsc_column]
        result = process_histogram(positive_fsc, smoothing_window, num_bins)
        if result is not None:
            hist, bin_edges, peak_indices, _ = result
            
            # Plot raw histogram
            ax2.hist(positive_fsc, bins=num_bins, density=True, alpha=0.5, label='Raw Data')
            
            # Plot smoothed histogram and peaks
            ax2.plot(bin_edges[:-1], hist, 'b-', label='Smoothed', alpha=0.7)
            peak_x = bin_edges[peak_indices]
            peak_y = hist[peak_indices]
            ax2.plot(peak_x, peak_y, 'ro', label='Peaks')
            
            # Add vertical lines for peaks
            for x in peak_x:
                ax2.axvline(x=x, color='gray', linestyle='--', alpha=0.3)
            
            # Add threshold line
            ax2.axvline(x=threshold, color='g', linestyle='-', label='Threshold')
            
            ax2.set_title('FSC Distribution - Positive Population')
            ax2.set_xlabel('FSC-A')
            ax2.set_ylabel('Density')
            ax2.legend()
    
    plt.tight_layout()
    plt.show()
    plt.close()

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

        # Set negative fluorescence values to 0
        data = np.where(data < 0, 0, data)

        valid_peaks_mask, peaks_list, positive_masks = detect_fluoropeaks(
            data.T, 
            marker_names,
            min_peaks=self.min_peaks,
            max_peaks=self.max_peaks,
            smoothing_window=self.smoothing_window,
            percentage_cells_present=self.percentage_cells_present,
            num_bins=self.num_bins
        )

        fsc_column = self._get_fsc_column(marker_names)
        # Get FSC thresholds from valid peaks
        fsc_thresholds = self._get_fsc_thresholds(data, valid_peaks_mask, peaks_list, positive_masks, fsc_column)
        print("fsc_thresholds: ", fsc_thresholds)

        if not fsc_thresholds:
            return data, None, np.ones(data.shape[0], dtype=int)

        # Calculate final threshold and apply gating
        fsc_gate_threshold = np.nanmedian(fsc_thresholds)
        print("FSC gate threshold: ", fsc_gate_threshold)
        debris_vector = (data[:, fsc_column] >= fsc_gate_threshold).astype(int)
        filtered_data = data[debris_vector == 1]

        # # Plot results for visualization
        # for i, (is_valid, peaks, positive_mask) in enumerate(zip(valid_peaks_mask, peaks_list, positive_masks)):
        #     if is_valid and len(peaks) >= 2 and positive_mask is not None:
        #         plot_debris_gating_results(data, fsc_column, fsc_gate_threshold, positive_mask, 
        #                                  self.smoothing_window, self.num_bins)

        return filtered_data, fsc_gate_threshold, debris_vector

    def _check_events_in_bottom_bin(self, fcs_array: np.ndarray, marker_names: list[str]) -> bool:
        """Check if events exist in the bottom 10th bin of FSC-A and SSC-A."""
        standardized_names = [standardize_marker_name(name) for name in marker_names]
        
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
        standardized_names = [standardize_marker_name(name) for name in marker_names]
        try:
            return standardized_names.index('fsca')
        except ValueError:
            raise ValueError("FSC-A parameter not found in marker names.")

    def _has_low_peak(self, bin_edges: np.ndarray, peak_indices: np.ndarray, 
                     lowest_reference_pos: float, tolerance: float = 1.15) -> bool:
        """Check if any peaks are near or below the reference lowest peak."""
        return any(bin_edges[idx] <= lowest_reference_pos * tolerance for idx in peak_indices)

    def _get_left_boundary_threshold(self, hist: np.ndarray, bin_edges: np.ndarray, 
                                   peak_indices: np.ndarray, smoothing_window: int) -> float:
        """Get threshold using left boundary of lowest peak."""
        lowest_peak_idx = peak_indices[0]  # Already sorted by position
        valley_idx = find_left_minimum(hist, lowest_peak_idx, smoothing_window)
        return bin_edges[valley_idx]

    def _get_max_peak_threshold(self, thresholded_hist: np.ndarray, bin_edges: np.ndarray,
                              peak_indices: np.ndarray, peak_densities: np.ndarray,
                              smoothing_window: int) -> Optional[float]:
        """Get threshold using max peak logic."""
        max_peak_idx = np.argmax(peak_densities)
        
        # If biggest peak is first peak (debris peak), find valley between first and second peaks
        if max_peak_idx == 0 and len(peak_indices) > 1:
            left_min_idx = find_left_minimum(thresholded_hist, peak_indices[1], smoothing_window)
            return bin_edges[left_min_idx]
            
        # If biggest peak is second peak
        if max_peak_idx == 1:
            left_min_idx = find_left_minimum(thresholded_hist, peak_indices[max_peak_idx], smoothing_window)
            return bin_edges[left_min_idx]
        
        # Get all peak positions for comparison
        all_peak_positions = bin_edges[peak_indices]
        
        # Search for appropriate minimum before max peak
        # For each potential valley, check if it's smaller than ALL peaks
        for i in range(max_peak_idx - 1, -1, -1):
            left_min_idx = find_left_minimum(thresholded_hist, peak_indices[i], smoothing_window)
            valley_position = bin_edges[left_min_idx]
            # Check if this valley is smaller than ALL peaks
            if all(valley_position < peak_pos for peak_pos in all_peak_positions):
                return valley_position
        
        return None

    def _find_fsc_threshold(self, process_histogram_result: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], reference_peaks: List[Tuple[int, float]], 
                          smoothing_window: int) -> Optional[float]:
        """Find the FSC threshold by comparing peaks to reference peaks from all cells."""
        # Get the lowest FSC peak position from reference
        lowest_reference_peak_pos = reference_peaks[0][1]
        smoothed_hist, pos_bin_edges, pos_peak_indices, peak_densities = process_histogram_result

        print("pos_peak_indices: ", pos_peak_indices)
        
        # Check if we have any peaks near or below the reference lowest peak
        if not self._has_low_peak(pos_bin_edges, pos_peak_indices, lowest_reference_peak_pos):
            # If we don't have any low peaks, use left boundary of lowest available peak
            return self._get_left_boundary_threshold(smoothed_hist, pos_bin_edges, pos_peak_indices, smoothing_window)
            
        # Use traditional max peak logic
        return self._get_max_peak_threshold(smoothed_hist, pos_bin_edges, pos_peak_indices, peak_densities, smoothing_window)

    def _get_fsc_thresholds(self, data: np.ndarray, valid_peaks_mask: np.ndarray, 
                           peaks_list: List[List[Peak]], positive_masks: List[np.ndarray],
                           fsc_column: int) -> List[float]:
        """Get FSC thresholds for features with valid peaks."""
        # First get reference FSC peaks from all cells using process_histogram
        result = process_histogram(data[:, fsc_column], self.smoothing_window)
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
                
                # Process histogram for this positive population
                pos_result = process_histogram(positive_fsc, self.smoothing_window)
                if pos_result is not None:
                    # Find FSC threshold using positive cells and reference peaks
                    threshold = self._find_fsc_threshold(pos_result, all_fsc_peaks, self.smoothing_window)
                    if threshold is not None:
                        fsc_thresholds.append(threshold)
                    else:
                        fsc_thresholds.append(np.nan)
                
        return fsc_thresholds

def detect_fluoropeaks(data: np.ndarray, marker_names: List[str], min_peaks: int = 2, max_peaks: int = 5, 
                         smoothing_window: int = 2, percentage_cells_present: float = 5, num_bins: int = 100) -> Tuple[np.ndarray, List[List[Peak]], List[np.ndarray]]:
    """Detect debris peaks in the data."""
    num_features, num_samples = data.shape
    data_transformed = np.arcsinh(data / 150)
    peaks_list = []
    valid_peaks_mask = np.zeros(num_features, dtype=bool)
    positive_masks = []  # Store positive masks for each feature
    
    for i, marker in enumerate(marker_names):
        # Skip excluded channels
        if is_excluded_marker(marker):
            peaks_list.append([])
            positive_masks.append(None)
            continue
            
        # Process histogram and get peaks
        result = process_histogram(data_transformed[i], smoothing_window)
        if result is None:
            print(f"No valid histogram for marker {marker}")
            peaks_list.append([])
            positive_masks.append(None)
            continue
            
        thresholded_hist, bin_edges, peak_indices, peak_densities = result
        
        # Check if we have enough peaks
        if len(peak_indices) < min_peaks:
            print(f"Insufficient peaks ({len(peak_indices)}) for marker {marker}")
            peaks_list.append([])
            positive_masks.append(None)
            continue

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
            print(f"Not enough valid peaks after width analysis for marker {marker}")
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

