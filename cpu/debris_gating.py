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
    ssc_threshold: Optional[float] = None
    beads_threshold: Optional[float] = None

class DebrisGateStrategy(ABC):
    @abstractmethod
    def gate(self, data: np.ndarray, marker_names: list[str]) -> DebrisGateResult:
        """Apply debris gating to the data."""
        pass

class FSCDebrisGate(DebrisGateStrategy):
    def __init__(self, min_peaks=2, max_peaks=5, smoothing_window=3, percentage_cells_present=5, num_bins=100, enable_dask=True, chunk_size=None, enable_ssc=False, remove_beads=False):
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks
        self.smoothing_window = smoothing_window
        self.percentage_cells_present = percentage_cells_present
        self.num_bins = num_bins
        self.enable_dask = enable_dask
        self.chunk_size = chunk_size
        self.enable_ssc = enable_ssc
        self.remove_beads = remove_beads

    def gate(self, data: np.ndarray, marker_names: list[str]) -> DebrisGateResult:
        """
        Apply debris gating to the data based on FSC-A and optionally SSC-A.
        
        Args:
            data: Flow cytometry data array
            marker_names: List of marker names
            
        Returns:
            DebrisGateResult containing filtered data, threshold and debris vector
        """
        if not self._check_events_in_bottom_bin(data, marker_names):
            return DebrisGateResult(filtered_data=data, fsc_threshold=None, debris_vector=np.ones(data.shape[0], dtype=int), ssc_threshold=None, beads_threshold=None)

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

        # FSC gating (always performed)
        fsc_column = self._get_fsc_column(marker_names)
        fsc_thresholds = self._get_fsc_thresholds(data, valid_peaks_mask, peaks_list, positive_masks, fsc_column)

        if not fsc_thresholds:
            return DebrisGateResult(filtered_data=data, fsc_threshold=None, debris_vector=np.ones(data.shape[0], dtype=int), ssc_threshold=None, beads_threshold=None)

        # Calculate FSC threshold and apply gating
        fsc_gate_threshold = np.nanmedian(fsc_thresholds)
        debris_vector = (data[:, fsc_column] >= fsc_gate_threshold).astype(int)
        
        # Default SSC threshold is None
        ssc_gate_threshold = None
        
        # SSC gating (only if enabled)
        if self.enable_ssc:
            try:
                ssc_column = self._get_ssc_column(marker_names)
                # Use the generic method with SSC parameters
                ssc_thresholds = self._get_ssc_thresholds(data, valid_peaks_mask, peaks_list, positive_masks, ssc_column)
                
                if ssc_thresholds:
                    ssc_gate_threshold = np.nanmedian(ssc_thresholds)
                    ssc_debris_vector = (data[:, ssc_column] >= ssc_gate_threshold).astype(int)
                    
                    # Combine FSC and SSC vectors with logical AND
                    debris_vector = debris_vector & ssc_debris_vector
            except ValueError as e:
                warnings.warn(f"SSC gating failed: {str(e)}. Using only FSC gating.")
        
        # Bead removal (only if enabled and SSC is available)
        beads_threshold = None
        if self.remove_beads and self.enable_ssc:
            try:
                # Get FSC and SSC columns
                fsc_column = self._get_fsc_column(marker_names)
                ssc_column = self._get_ssc_column(marker_names)
                
                # Identify and remove beads
                beads_vector, beads_threshold = self._identify_and_remove_beads(
                    data, fsc_column, ssc_column
                )
                
                # Update the debris vector to exclude beads
                if beads_vector is not None:
                    debris_vector = debris_vector & beads_vector
            except ValueError as e:
                warnings.warn(f"Bead removal failed: {str(e)}. Continuing without bead removal.")
        
        # Apply the debris vector to filter the data
        filtered_data = data[debris_vector == 1]

        return DebrisGateResult(
            filtered_data=filtered_data, 
            fsc_threshold=fsc_gate_threshold, 
            debris_vector=debris_vector,
            ssc_threshold=ssc_gate_threshold,
            beads_threshold=beads_threshold
        )

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
            
    def _get_ssc_column(self, marker_names: list[str]) -> int:
        """Get the index of the SSC-A column."""
        standardized_names = [standardize_marker_name(name) for name in marker_names]
        try:
            return standardized_names.index('ssca')
        except ValueError:
            raise ValueError("SSC-A parameter not found in marker names.")

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

    def _find_scatter_threshold(self, process_histogram_result: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], reference_peaks: List[Tuple[int, float]], 
                             smoothing_window: int) -> Optional[float]:
        """
        Find a scatter threshold by comparing peaks to reference peaks from all cells.
        
        This is a generic method that works for both FSC and SSC thresholding.
        
        Args:
            process_histogram_result: Results from processing the histogram
            reference_peaks: List of reference peaks from the full dataset
            smoothing_window: Window size for smoothing
            
        Returns:
            Threshold value or None if no valid threshold could be determined
        """
        smoothed_hist, pos_bin_edges, pos_peak_indices, peak_densities = process_histogram_result
        
        # If there are no peaks in the positive population's histogram, 
        # or no reference peaks from the global population for comparison,
        # a threshold cannot be reliably determined by this method.
        if len(pos_peak_indices) == 0 or len(reference_peaks) == 0:
            return None
            
        # Get the lowest peak position from reference
        lowest_reference_peak_pos = reference_peaks[0][1]
        
        # Check if we have any peaks near or below the reference lowest peak
        if not self._has_low_peak(pos_bin_edges, pos_peak_indices, lowest_reference_peak_pos):
            # If we don't have any low peaks, use left boundary of lowest available peak
            return self._get_left_boundary_threshold(smoothed_hist, pos_bin_edges, pos_peak_indices, smoothing_window)
            
        # Use traditional max peak logic
        return self._get_max_peak_threshold(smoothed_hist, pos_bin_edges, pos_peak_indices, peak_densities, smoothing_window)
    
    def _find_fsc_threshold(self, process_histogram_result: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], reference_peaks: List[Tuple[int, float]], 
                          smoothing_window: int) -> Optional[float]:
        """Find the FSC threshold by comparing peaks to reference peaks from all cells."""
        return self._find_scatter_threshold(process_histogram_result, reference_peaks, smoothing_window)
        
    def _find_ssc_threshold(self, process_histogram_result: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], reference_peaks: List[Tuple[int, float]], 
                          smoothing_window: int) -> Optional[float]:
        """Find the SSC threshold by comparing peaks to reference peaks from all cells."""
        return self._find_scatter_threshold(process_histogram_result, reference_peaks, smoothing_window)

    def _get_scatter_thresholds(self, data: np.ndarray, valid_peaks_mask: np.ndarray, 
                               peaks_list: List[List[Peak]], positive_masks: List[np.ndarray],
                               scatter_column: int, scatter_type: str) -> List[float]:
        """
        Get scatter thresholds for features with valid peaks.
        
        Args:
            data: Flow cytometry data array
            valid_peaks_mask: Boolean mask indicating which features have valid peaks
            peaks_list: List of peak objects for each feature
            positive_masks: List of boolean masks indicating positive cells for each feature
            scatter_column: Index of the scatter column (FSC-A or SSC-A)
            scatter_type: Type of scatter parameter ('FSC-A' or 'SSC-A')
            
        Returns:
            List of threshold values
        """
        # First get reference scatter peaks from all cells using process_histogram
        result = process_histogram(data[:, scatter_column], self.smoothing_window)
        if result is None:
            return []
        
        smoothed_hist, bin_edges, ref_peak_indices, _ = result
        all_scatter_peaks = [(idx, bin_edges[idx]) for idx in ref_peak_indices]
        all_scatter_peaks = sorted(all_scatter_peaks, key=lambda x: x[1])
        
        scatter_thresholds = []

        # For every feature, run through and check for peaks
        delayed_thresholds = []
        
        for i, (is_valid, peaks, positive_mask) in enumerate(zip(valid_peaks_mask, peaks_list, positive_masks)):
            if is_valid and len(peaks) >= 2 and positive_mask is not None:
                # Use the positive mask from detect_fluoropeaks
                positive_scatter = data[positive_mask, scatter_column]
                
                # Define a function to process a single feature
                def process_feature(feature_idx, pos_scatter, ref_peaks, window):
                    # Process histogram for this positive population
                    pos_result = process_histogram(pos_scatter, window)
                    if pos_result is not None:
                        # Find threshold using the generic threshold finder
                        threshold = self._find_scatter_threshold(pos_result, ref_peaks, window)
                        
                        if threshold is not None:
                            # Optional plotting
                            if False:
                                # Create a figure with two subplots side by side
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                                
                                # Plot 1: Full Distribution
                                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                                ax1.plot(bin_centers, smoothed_hist, 'b-', label='All Cells')
                                ax1.plot(bin_centers[ref_peak_indices], smoothed_hist[ref_peak_indices], 'ro', label='Peaks')
                                ax1.set_xlabel(scatter_type)
                                ax1.set_ylabel('Density')
                                ax1.set_title(f'Full {scatter_type} Distribution')
                                ax1.axvline(x=threshold, color='g', linestyle='--', label='Threshold')
                                ax1.legend()
                                
                                # Plot 2: Positive Population
                                pos_hist, pos_bin_edges, pos_peak_indices, _ = pos_result
                                pos_bin_centers = (pos_bin_edges[:-1] + pos_bin_edges[1:]) / 2
                                ax2.plot(pos_bin_centers, pos_hist, 'b-', label='Positive Cells')
                                ax2.plot(pos_bin_centers[pos_peak_indices], pos_hist[pos_peak_indices], 'ro', label='Peaks')
                                ax2.axvline(x=threshold, color='g', linestyle='--', label='Threshold')
                                ax2.set_xlabel(scatter_type)
                                ax2.set_ylabel('Density')
                                ax2.set_title(f'Positive Population {scatter_type} Distribution')
                                ax2.legend()
                                
                                plt.suptitle(f'{scatter_type} Distribution Analysis - Feature {feature_idx}')
                                plt.tight_layout()
                                plt.show()
                            
                            return threshold
                        
                    return np.nan
                
                # Add the computation for this feature - with or without Dask
                if self.enable_dask:
                    delayed_thresholds.append(dask.delayed(process_feature)(i, positive_scatter, all_scatter_peaks, self.smoothing_window))
                else:
                    threshold = process_feature(i, positive_scatter, all_scatter_peaks, self.smoothing_window)
                    if not (isinstance(threshold, float) and np.isnan(threshold)):
                        scatter_thresholds.append(threshold)
            else:
                # For invalid features, add a placeholder if using Dask
                if self.enable_dask:
                    delayed_thresholds.append(dask.delayed(lambda: np.nan)())
        
        # Compute all thresholds in parallel if using Dask
        if self.enable_dask:
            computed_thresholds = dask.compute(*delayed_thresholds)
            # Filter out the valid thresholds (not NaN)
            scatter_thresholds = [t for t in computed_thresholds if not (isinstance(t, float) and np.isnan(t))]
        
        return scatter_thresholds
        
    def _get_fsc_thresholds(self, data: np.ndarray, valid_peaks_mask: np.ndarray, 
                           peaks_list: List[List[Peak]], positive_masks: List[np.ndarray],
                           fsc_column: int) -> List[float]:
        """Get FSC thresholds for features with valid peaks."""
        return self._get_scatter_thresholds(data, valid_peaks_mask, peaks_list, positive_masks, fsc_column, 'FSC-A')
        
    def _get_ssc_thresholds(self, data: np.ndarray, valid_peaks_mask: np.ndarray, 
                           peaks_list: List[List[Peak]], positive_masks: List[np.ndarray],
                           ssc_column: int) -> List[float]:
        """Get SSC thresholds for features with valid peaks."""
        return self._get_scatter_thresholds(data, valid_peaks_mask, peaks_list, positive_masks, ssc_column, 'SSC-A')
        
    def _identify_and_remove_beads(self, data: np.ndarray, fsc_column: int, ssc_column: int) -> Tuple[np.ndarray, float]:
        """
        Identify and remove beads based on SSC and FSC parameters.
        
        Beads typically have high SSC values but fall in the lower half of the FSC range.
        This method:
        1. Finds the largest peak in the SSC histogram
        2. Creates a mask to remove events in this peak that are also in the bottom half of FSC
        
        Args:
            data: Flow cytometry data array
            fsc_column: Index of the FSC-A column
            ssc_column: Index of the SSC-A column
            
        Returns:
            Tuple of (beads_vector, beads_threshold) where:
                - beads_vector is a boolean mask (True = keep, False = remove)
                - beads_threshold is the SSC threshold used to identify beads
        """
        # Extract SSC data
        ssc_data = data[:, ssc_column]
        
        # Process SSC histogram to find peaks
        result = process_histogram(ssc_data, self.smoothing_window)
        if result is None:
            warnings.warn("Could not process SSC histogram for bead removal.")
            return None, None
            
        smoothed_hist, bin_edges, peak_indices, peak_densities = result
        
        # If no peaks are found, return None
        if len(peak_indices) == 0:
            warnings.warn("No peaks found in SSC histogram for bead removal.")
            return None, None
            
        # Find the largest peak by height
        largest_peak_idx = peak_indices[np.argmax(peak_densities)]
        largest_peak_pos = bin_edges[largest_peak_idx]
        
        # Calculate the threshold for beads (use the position of the largest peak)
        beads_threshold = largest_peak_pos
        
        # Get FSC range and calculate the midpoint
        fsc_data = data[:, fsc_column]
        fsc_min = np.min(fsc_data)
        fsc_max = np.max(fsc_data)
        fsc_midpoint = fsc_min + (fsc_max - fsc_min) / 2
        
        # Create a mask to exclude:
        # 1. Events with SSC values greater than or equal to the largest peak position
        # 2. But only if they also have FSC values below the midpoint
        beads_vector = ~((ssc_data >= beads_threshold) & (fsc_data < fsc_midpoint))
        
        # Optional visualization (disabled by default)
        if False:
            plt.figure(figsize=(12, 8))
            
            # Plot 1: SSC Histogram
            plt.subplot(2, 2, 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.plot(bin_centers, smoothed_hist, 'b-', label='SSC Distribution')
            plt.plot(bin_centers[peak_indices], smoothed_hist[peak_indices], 'ro', label='Peaks')
            plt.axvline(x=beads_threshold, color='g', linestyle='--', label='Beads Threshold')
            plt.xlabel('SSC-A')
            plt.ylabel('Density')
            plt.title('SSC Distribution with Beads Threshold')
            plt.legend()
            
            # Plot 2: FSC Histogram
            plt.subplot(2, 2, 2)
            plt.hist(fsc_data, bins=100, alpha=0.7, color='blue')
            plt.axvline(x=fsc_midpoint, color='r', linestyle='--', label='FSC Midpoint')
            plt.xlabel('FSC-A')
            plt.ylabel('Count')
            plt.title('FSC Distribution')
            plt.legend()
            
            # Plot 3: Scatter plot of FSC vs SSC with beads highlighted
            plt.subplot(2, 2, 3)
            plt.scatter(fsc_data[beads_vector], ssc_data[beads_vector], alpha=0.1, color='blue', label='Kept Events')
            plt.scatter(fsc_data[~beads_vector], ssc_data[~beads_vector], alpha=0.3, color='red', label='Removed Beads')
            plt.axhline(y=beads_threshold, color='g', linestyle='--', label='Beads SSC Threshold')
            plt.axvline(x=fsc_midpoint, color='r', linestyle='--', label='FSC Midpoint')
            plt.xlabel('FSC-A')
            plt.ylabel('SSC-A')
            plt.title('FSC vs SSC with Beads Highlighted')
            plt.legend()
            
            # Plot 4: Statistics
            plt.subplot(2, 2, 4)
            plt.text(0.1, 0.8, f"Total Events: {len(data)}", fontsize=12)
            plt.text(0.1, 0.7, f"Beads Removed: {np.sum(~beads_vector)}", fontsize=12)
            plt.text(0.1, 0.6, f"Percent Removed: {np.sum(~beads_vector)/len(data)*100:.2f}%", fontsize=12)
            plt.text(0.1, 0.5, f"SSC Threshold: {beads_threshold:.2f}", fontsize=12)
            plt.text(0.1, 0.4, f"FSC Midpoint: {fsc_midpoint:.2f}", fontsize=12)
            plt.axis('off')
            plt.title('Bead Removal Statistics')
            
            plt.tight_layout()
            plt.savefig("beads_removal_analysis.png", dpi=300)
            plt.close()
        
        return beads_vector, beads_threshold
    
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
