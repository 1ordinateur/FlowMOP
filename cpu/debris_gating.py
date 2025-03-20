"""
Debris gating component for FlowMOP.
Handles detection and removal of debris in flow cytometry data.
"""

import numpy as np
import dask.array as da
import dask
from typing import Union
from abc import ABC, abstractmethod
import warnings
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from cpu.flowmop_utils import Peak, process_histogram, standardize_marker_name, is_excluded_marker, find_left_minimum

@dataclass
class DebrisGateResult:
    """Results from debris gating operation."""
    filtered_data: np.ndarray
    fsc_threshold: Optional[float]
    debris_vector: np.ndarray

class DebrisGateStrategy(ABC):
    @abstractmethod
    def gate(self, data: np.ndarray, marker_names: list[str]) -> DebrisGateResult:
        """Apply debris gating to the data."""
        pass

class FSCDebrisGate(DebrisGateStrategy):
    def __init__(self, min_peaks=2, max_peaks=5, smoothing_window=3, percentage_cells_present=5, num_bins=100, enable_dask=True, chunk_size=None):
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks
        self.smoothing_window = smoothing_window
        self.percentage_cells_present = percentage_cells_present
        self.num_bins = num_bins
        self.enable_dask = enable_dask
        self.chunk_size = chunk_size

    def gate(self, data: np.ndarray, marker_names: list[str]) -> DebrisGateResult:
        """
        Apply FSC-based debris gating to the data.
        
        Args:
            data: Flow cytometry data array
            marker_names: List of marker names
            
        Returns:
            DebrisGateResult containing filtered data, threshold and debris vector
        """
        if not self._check_events_in_bottom_bin(data, marker_names):
            return DebrisGateResult(filtered_data=data, fsc_threshold=None, debris_vector=np.ones(data.shape[0], dtype=int))

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

        if not fsc_thresholds:
            return DebrisGateResult(filtered_data=data, fsc_threshold=None, debris_vector=np.ones(data.shape[0], dtype=int))

        # Calculate final threshold and apply gating
        fsc_gate_threshold = np.nanmedian(fsc_thresholds)
        debris_vector = (data[:, fsc_column] >= fsc_gate_threshold).astype(int)
        filtered_data = data[debris_vector == 1]

        return DebrisGateResult(filtered_data=filtered_data, fsc_threshold=fsc_gate_threshold, debris_vector=debris_vector)

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
                     lowest_reference_pos: float, base_tolerance: float = 1.3) -> bool:
        """
        Check if any peaks are near or below the reference lowest peak.
        The tolerance is dynamically adjusted based on where the lowest reference peak sits in the overall range.
        If the reference peak is very low in the range, we use a higher tolerance.
        
        Args:
            bin_edges: Bin edges from histogram
            peak_indices: Indices of detected peaks
            lowest_reference_pos: Position of lowest peak in reference distribution
            base_tolerance: Base tolerance multiplier (default 1.3)
            
        Returns:
            bool: Whether any peaks are below the adjusted threshold
        """
        # Calculate the relative position of the lowest reference peak in the overall range
        total_range = bin_edges[-1] - bin_edges[0]
        relative_pos = (lowest_reference_pos - bin_edges[0]) / total_range
        
        # Adjust tolerance - higher tolerance when reference peak is lower in the range
        # Using an inverse relationship: tolerance increases as relative_pos decreases
        # Adding 1 to ensure tolerance is always at least base_tolerance
        dynamic_tolerance = base_tolerance * (1 + (1 - relative_pos))
        
        return any(bin_edges[idx] <= lowest_reference_pos * dynamic_tolerance for idx in peak_indices)

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
        
        # If biggest peak is first or second peak, find middle minimum between first and second peaks
        if (max_peak_idx == 0 and len(peak_indices) > 1) or max_peak_idx == 1:
            # Get indices between first and second peak
            start_idx = peak_indices[0]
            end_idx = peak_indices[1]
            # Find all minimum points between the peaks
            min_value = np.min(thresholded_hist[start_idx:end_idx])
            min_indices = np.where(thresholded_hist[start_idx:end_idx] == min_value)[0]
            # Take the middle minimum point
            middle_min_idx = start_idx + min_indices[len(min_indices)//2]
            return bin_edges[middle_min_idx]
        # Get all peak positions for comparison
        all_peak_positions = bin_edges[peak_indices]
        
        # Search for appropriate minimum before max peak if neither of above conditions are met
        for i in range(max_peak_idx - 1, -1, -1):
            left_min_idx = find_left_minimum(thresholded_hist, peak_indices[i], smoothing_window)
            # If we get to a valley before the first peak, use midpoint between first and second peaks
            if left_min_idx < peak_indices[0]:
                return (bin_edges[peak_indices[0]] + bin_edges[peak_indices[1]]) / 2
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
        
        smoothed_hist, bin_edges, ref_peak_indices, _ = result
        all_fsc_peaks = [(idx, bin_edges[idx]) for idx in ref_peak_indices]
        all_fsc_peaks = sorted(all_fsc_peaks, key=lambda x: x[1])
        
        fsc_thresholds = []

        # For every feature, run through and check for peaks
        delayed_thresholds = []
        
        for i, (is_valid, peaks, positive_mask) in enumerate(zip(valid_peaks_mask, peaks_list, positive_masks)):
            if is_valid and len(peaks) >= 2 and positive_mask is not None:
                # Use the positive mask from detect_fluoropeaks
                positive_fsc = data[positive_mask, fsc_column]
                
                # Define a function to process a single feature
                def process_feature(feature_idx, pos_fsc, ref_peaks, window):
                    # Process histogram for this positive population
                    pos_result = process_histogram(pos_fsc, window)
                    if pos_result is not None:
                        # Find FSC threshold using positive cells and reference peaks
                        threshold = self._find_fsc_threshold(pos_result, ref_peaks, window)
                        
                        if threshold is not None:
                            # Optional plotting
                            if False:
                                # Create a figure with two subplots side by side
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                                
                                # Plot 1: Full FSC Distribution
                                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                                ax1.plot(bin_centers, smoothed_hist, 'b-', label='All Cells')
                                ax1.plot(bin_centers[ref_peak_indices], smoothed_hist[ref_peak_indices], 'ro', label='Peaks')
                                ax1.set_xlabel('FSC-A')
                                ax1.set_ylabel('Density')
                                ax1.set_title('Full FSC Distribution')
                                ax1.axvline(x=threshold, color='g', linestyle='--', label='Threshold')
                                ax1.legend()
                                
                                # Plot 2: Positive Population
                                pos_hist, pos_bin_edges, pos_peak_indices, _ = pos_result
                                pos_bin_centers = (pos_bin_edges[:-1] + pos_bin_edges[1:]) / 2
                                ax2.plot(pos_bin_centers, pos_hist, 'b-', label='Positive Cells')
                                ax2.plot(pos_bin_centers[pos_peak_indices], pos_hist[pos_peak_indices], 'ro', label='Peaks')
                                ax2.axvline(x=threshold, color='g', linestyle='--', label='Threshold')
                                ax2.set_xlabel('FSC-A')
                                ax2.set_ylabel('Density')
                                ax2.set_title('Positive Population FSC Distribution')
                                ax2.legend()
                                
                                plt.suptitle(f'FSC Distribution Analysis - Feature {feature_idx}')
                                plt.tight_layout()
                                plt.show()
                            
                            return threshold
                        
                    return np.nan
                
                # Add the computation for this feature - with or without Dask
                if self.enable_dask:
                    delayed_thresholds.append(dask.delayed(process_feature)(i, positive_fsc, all_fsc_peaks, self.smoothing_window))
                else:
                    threshold = process_feature(i, positive_fsc, all_fsc_peaks, self.smoothing_window)
                    if not (isinstance(threshold, float) and np.isnan(threshold)):
                        fsc_thresholds.append(threshold)
            else:
                # For invalid features, add a placeholder if using Dask
                if self.enable_dask:
                    delayed_thresholds.append(dask.delayed(lambda: np.nan)())
        
        # Compute all thresholds in parallel if using Dask
        if self.enable_dask:
            computed_thresholds = dask.compute(*delayed_thresholds)
            # Filter out the valid thresholds (not NaN)
            fsc_thresholds = [t for t in computed_thresholds if not (isinstance(t, float) and np.isnan(t))]
        
        return fsc_thresholds
    
def detect_fluoropeaks(data: np.ndarray, marker_names: List[str], min_peaks: int = 2, max_peaks: int = 5, 
                         smoothing_window: int = 2, percentage_cells_present: float = 5, num_bins: int = 100) -> Tuple[np.ndarray, List[List[Peak]], List[np.ndarray]]:
    """Detect debris peaks in the data."""
    num_features, num_samples = data.shape
    data_transformed = np.arcsinh(data / 150)
    peaks_list = []
    valid_peaks_mask = np.zeros(num_features, dtype=bool)
    positive_masks = []  # Store positive masks for each feature
    
    # Define a function to process a single marker
    @dask.delayed
    def process_marker(i, marker, data_transformed, smoothing_window, min_peaks, max_peaks, percentage_cells_present):
        # Skip excluded channels
        if is_excluded_marker(marker):
            return i, [], None
            
        # Process histogram and get peaks
        result = process_histogram(data_transformed[i], smoothing_window)
        if result is None:
            print(f"No valid histogram for marker {marker}")
            return i, [], None
            
        thresholded_hist, bin_edges, peak_indices, peak_densities = result
        
        # Check if we have enough peaks
        if len(peak_indices) < min_peaks:
            print(f"Insufficient peaks ({len(peak_indices)}) for marker {marker}")
            return i, [], None

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
        
        # If we have at least 2 peaks, calculate positive cells
        if len(peaks) >= 2:
            second_peak = peaks[1]
            positive_mask = data_transformed[i] >= second_peak.start
            return i, peaks, positive_mask
        else:
            print(f"Not enough valid peaks after width analysis for marker {marker}")
            return i, peaks, None
    
    # Create delayed tasks for all markers
    delayed_results = [process_marker(i, marker, data_transformed, smoothing_window, 
                                     min_peaks, max_peaks, percentage_cells_present) 
                      for i, marker in enumerate(marker_names)]
    
    # Compute all results in parallel
    computed_results = dask.compute(*delayed_results)
    
    # Process the results
    for idx, peaks, positive_mask in computed_results:
        peaks_list.append(peaks)
        valid_peaks_mask[idx] = len(peaks) >= min_peaks
        positive_masks.append(positive_mask)
    
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
            left_idx = find_left_minimum(smoothed_hist, peak_index, None)
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
