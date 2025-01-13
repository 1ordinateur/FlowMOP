"""
Debris gating component for FlowMOP.
Handles detection and removal of debris in flow cytometry data.
"""

import numpy as np
from abc import ABC, abstractmethod
import warnings

class DebrisGateStrategy(ABC):
    @abstractmethod
    def gate(self, data: np.ndarray, marker_names: list[str]) -> tuple[np.ndarray, float, np.ndarray]:
        """Apply debris gating to the data."""
        pass

class FSCDebrisGate(DebrisGateStrategy):
    def __init__(self, min_peaks=2, max_peaks=3, smoothing_window=3, percentage_cells_present=3):
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks
        self.smoothing_window = smoothing_window
        self.percentage_cells_present = percentage_cells_present

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

        valid_peaks_mask, valid_peaks = peak_detection_debris(
            data.T, 
            min_peaks=self.min_peaks,
            max_peaks=self.max_peaks,
            smoothing_window=self.smoothing_window,
            percentage_cells_present=self.percentage_cells_present
        )

        fsc_column = self._get_fsc_column(marker_names)
        fsc_thresholds = [
            find_fsc_threshold(data[:, fsc_column], peaks, self.smoothing_window) 
            for peaks in valid_peaks if peaks
        ]

        fsc_gate_threshold = np.nanmedian(fsc_thresholds)
        debris_vector = (data[:, fsc_column] >= fsc_gate_threshold).astype(int)
        filtered_data = data[debris_vector == 1]

        return filtered_data, fsc_gate_threshold, debris_vector

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

# Utility functions moved from original implementation
def peak_detection_debris(data, min_peaks=2, max_peaks=3, smoothing_window=2, percentage_cells_present=5):
    """Detect debris peaks in the data."""
    num_features, num_samples = data.shape
    num_bins = 100
    data_transformed = np.arcsinh(data / 150)
    
    peaks = np.zeros((num_features, max_peaks))
    valid_peaks = []
    
    for i in range(num_features):
        min_val, max_val = np.min(data_transformed[i]), np.max(data_transformed[i])
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist, _ = np.histogram(data_transformed[i], bins=bin_edges, density=True)
        
        # Set the bottom 2% and top 2% of bins to 0
        bottom_bins = int(num_bins * 0.02)
        top_bins = int(num_bins * 0.98)
        hist[:bottom_bins] = hist[top_bins:] = 0
        
        smoothed_hist = np.convolve(hist, np.ones(smoothing_window) / smoothing_window, mode='same')
        maxima = (smoothed_hist > np.roll(smoothed_hist, 1)) & (smoothed_hist > np.roll(smoothed_hist, -1))
        maxima[:smoothing_window] = maxima[-smoothing_window:] = False
    
        if np.sum(maxima) > 1:
            peak_indices = np.where(maxima)[0]
            peak_densities = [smoothed_hist[idx] for idx in peak_indices]
            peak_info = sorted(zip(peak_indices, peak_densities), key=lambda x: x[1], reverse=True)[:max_peaks]

            largest_peak_indices = [peak[0] for peak in peak_info]
            peaks[i, :len(largest_peak_indices)] = bin_centers[largest_peak_indices]

            peak_widths = peak_width_debris(smoothed_hist, largest_peak_indices, bin_edges, percentage_cells_present)
            valid_peaks.append(peak_widths)
        else:
            valid_peaks.append([])
            
    valid_peaks_mask = np.array([len(peaks) >= min_peaks for peaks in valid_peaks])
    return valid_peaks_mask, valid_peaks

def peak_width_debris(smoothed_hist, peak_indices, bin_edges, percentage_cells_present=5, smoothing_window=2):
    """Calculate the width of debris peaks."""
    num_bins = len(smoothed_hist)
    
    # Apply threshold to the smoothed histogram
    non_zero_bins = smoothed_hist[smoothed_hist != 0]
    if len(non_zero_bins) == 0:
        return []
    threshold = np.percentile(non_zero_bins, 0.05 * 100)
    smoothed_hist[smoothed_hist < threshold] = 0
    
    peak_widths = []
    
    for peak_index in peak_indices:
        left_minima_index = peak_index
        right_minima_index = peak_index
        
        # Find left minima
        while left_minima_index > 0:
            if (smoothed_hist[left_minima_index] < smoothed_hist[left_minima_index - smoothing_window:left_minima_index]).all():
                break
            left_minima_index -= 1
        
        # Find right minima
        while right_minima_index < num_bins - 1:
            if (smoothed_hist[right_minima_index] < smoothed_hist[right_minima_index + 1:right_minima_index + smoothing_window + 1]).all():
                break
            right_minima_index += 1        
        
        peak_percentage = np.sum(smoothed_hist[left_minima_index:right_minima_index + 1]) / np.sum(smoothed_hist) * 100
        if peak_percentage >= percentage_cells_present:
            peak_widths.append((bin_edges[left_minima_index], bin_edges[right_minima_index]))
            
    return sorted(peak_widths, key=lambda x: x[0])

def find_fsc_threshold(feature, peaks, smoothing_window):
    """Find the FSC threshold for debris gating."""
    num_bins = 300
    
    min_val, max_val = np.min(feature), np.max(feature)
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    hist, _ = np.histogram(feature, bins=bin_edges, density=True)
    
    # Set the bottom 2% and top 2% of bins to 0
    bottom_bins = int(num_bins * 0.02)
    top_bins = int(num_bins * 0.98)
    hist[:bottom_bins] = hist[top_bins:] = 0
    
    smoothed_hist = np.convolve(hist, np.ones(smoothing_window), mode='same')
    
    # Apply threshold to the smoothed histogram
    non_zero_bins = smoothed_hist[smoothed_hist != 0]
    if len(non_zero_bins) == 0:
        return None
    threshold = np.percentile(non_zero_bins, 1)
    smoothed_hist[smoothed_hist < threshold] = 0

    maxima = (smoothed_hist > np.roll(smoothed_hist, 1)) & (smoothed_hist > np.roll(smoothed_hist, -1))
    maxima[:smoothing_window] = maxima[-smoothing_window:] = False
    peak_indices = np.where(maxima)[0]
    
    if len(peak_indices) > 1:
        peak_densities = [
            np.sum(smoothed_hist[max(0, idx - smoothing_window):min(idx + smoothing_window + 1, len(smoothed_hist))])
            for idx in peak_indices
        ]
        max_peak_index = np.argmax(peak_densities)
        
        if max_peak_index == 1:  # If the biggest peak is the second peak
            left_minima_index = peak_indices[max_peak_index]
            while left_minima_index > 0:
                if (smoothed_hist[left_minima_index] < smoothed_hist[left_minima_index - smoothing_window:left_minima_index]).all():
                    break
                left_minima_index -= 1
            return bin_edges[left_minima_index]
        else:
            for i in range(max_peak_index - 1, -1, -1):
                left_minima_index = peak_indices[i]
                while left_minima_index > 0:
                    if (smoothed_hist[left_minima_index] < smoothed_hist[left_minima_index - smoothing_window:left_minima_index]).all():
                        break
                    left_minima_index -= 1
                if bin_edges[left_minima_index] < bin_edges[peak_indices[max_peak_index]]:
                    return bin_edges[left_minima_index]
    
    return None
