"""
GPU-accelerated debris gating implementation using DASK arrays.
"""

import warnings
from typing import Union, Tuple, List, Optional
import dask.array as da
import cupy as cp
from cupyx.scipy import signal as cusignal

from .debris_gating import DebrisGateStrategy

# Type aliases
ArrayType = Union[da.Array, cp.ndarray]

class DaskGPUFSCDebrisGate(DebrisGateStrategy):
    """FSC-based debris gating implementation using GPU-accelerated DASK arrays."""
    
    def __init__(self, min_peaks: int = 2, max_peaks: int = 3, smoothing_window: int = 3, 
                 percentage_cells_present: float = 3):
        """
        Initialize the gating strategy.
        
        Args:
            min_peaks: Minimum number of peaks required
            max_peaks: Maximum number of peaks to consider
            smoothing_window: Window size for smoothing operations
            percentage_cells_present: Minimum percentage of cells required in a peak
        """
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks
        self.smoothing_window = smoothing_window
        self.percentage_cells_present = percentage_cells_present
        self._debug_info = {}

    def gate(self, data: da.Array, marker_names: List[str]) -> Tuple[da.Array, float, da.Array]:
        """
        Apply FSC-based debris gating using GPU-accelerated DASK arrays.
        
        Args:
            data: Input flow cytometry data
            marker_names: List of marker names corresponding to columns
            
        Returns:
            Tuple of (filtered_data, fsc_threshold, debris_vector)
        """
        # Check for events in bottom bin
        if not self._check_events_in_bottom_bin(data, marker_names):
            return data, None, da.ones(data.shape[0], chunks=data.chunks[0], dtype=cp.int32)

        # Get FSC column index
        fsc_column = self._get_fsc_column(marker_names)
        
        # Detect peaks and get valid peaks mask
        valid_peaks_mask, valid_peaks = self._peak_detection_gpu(
            data.T,
            min_peaks=self.min_peaks,
            max_peaks=self.max_peaks,
            smoothing_window=self.smoothing_window,
            percentage_cells_present=self.percentage_cells_present
        )
        print("valid_peaks_mask", valid_peaks_mask)
        print("valid_peaks", valid_peaks)
        self._debug_info['valid_peaks'] = valid_peaks

        # Calculate FSC thresholds for each set of valid peaks
        fsc_thresholds = [
            self._find_fsc_threshold_gpu(data[:, fsc_column], peaks, self.smoothing_window)
            for peaks in valid_peaks if peaks
        ]
        self._debug_info['fsc_thresholds'] = fsc_thresholds

        # Calculate final threshold and apply gating
        valid_thresholds = [t for t in fsc_thresholds if t is not None]
        if not valid_thresholds:
            return data, None, da.ones(data.shape[0], chunks=data.chunks[0], dtype=cp.int32)
            
        fsc_gate_threshold = da.nanmedian(da.from_array(valid_thresholds))
        debris_vector = self._apply_threshold_gpu(data[:, fsc_column], fsc_gate_threshold)
        filtered_data = data[debris_vector]

        return filtered_data, fsc_gate_threshold, debris_vector

    def _check_events_in_bottom_bin(self, data: da.Array, marker_names: List[str]) -> bool:
        """Check if events exist in the bottom 10th bin of FSC-A and SSC-A."""
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        
        try:
            fsc_a_column = standardized_names.index('fsca')
            ssc_a_column = standardized_names.index('ssca')
        except ValueError:
            warnings.warn("Required FSC-A or SSC-A parameters not found.")
            return False

        def check_bottom_bin(block):
            gpu_block = cp.asarray(block)
            fsc_max = cp.max(gpu_block[:, fsc_a_column])
            ssc_max = cp.max(gpu_block[:, ssc_a_column])
            
            fsc_bottom = fsc_max * 0.1
            ssc_bottom = ssc_max * 0.1
            
            return cp.any(
                (gpu_block[:, fsc_a_column] <= fsc_bottom) & 
                (gpu_block[:, ssc_a_column] <= ssc_bottom)
            )

        events_in_bottom = data.map_blocks(check_bottom_bin, dtype=bool)
        return da.any(events_in_bottom)

    def _peak_detection_gpu(self, data: da.Array, min_peaks: int = 2, max_peaks: int = 3,
                          smoothing_window: int = 2, percentage_cells_present: float = 5) -> Tuple[da.Array, List]:
        """Detect debris peaks using GPU operations."""
        num_features = data.shape[0]
        num_bins = 100

        def transform_and_detect(block):
            gpu_block = cp.asarray(block)
            data_transformed = cp.arcsinh(gpu_block / 150)
            
            peaks = cp.zeros((gpu_block.shape[0], max_peaks))
            valid_peaks_list = []
            
            for i in range(gpu_block.shape[0]):
                min_val, max_val = cp.min(data_transformed[i]), cp.max(data_transformed[i])
                bin_edges = cp.linspace(min_val, max_val, num_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Compute histogram
                hist, _ = cp.histogram(data_transformed[i], bins=bin_edges, density=True)
                
                # Zero out bottom and top bins
                bottom_bins = int(num_bins * 0.02)
                top_bins = int(num_bins * 0.98)
                hist[:bottom_bins] = hist[top_bins:] = 0
                
                # Smooth histogram
                smoothed_hist = cusignal.convolve(hist, cp.ones(smoothing_window) / smoothing_window, mode='same')
                
                # Find maxima
                maxima = (smoothed_hist > cp.roll(smoothed_hist, 1)) & (smoothed_hist > cp.roll(smoothed_hist, -1))
                maxima[:smoothing_window] = maxima[-smoothing_window:] = False
                
                if cp.sum(maxima) > 1:
                    peak_indices = cp.where(maxima)[0]
                    peak_densities = [float(smoothed_hist[idx]) for idx in peak_indices]
                    peak_info = sorted(zip(peak_indices, peak_densities), key=lambda x: x[1], reverse=True)[:max_peaks]
                    
                    largest_peak_indices = [peak[0] for peak in peak_info]
                    peaks[i, :len(largest_peak_indices)] = bin_centers[largest_peak_indices]
                    
                    # Calculate peak widths
                    peak_widths = self._peak_width_gpu(smoothed_hist, largest_peak_indices, bin_edges, 
                                                     percentage_cells_present, smoothing_window)
                    valid_peaks_list.append(peak_widths)
                else:
                    valid_peaks_list.append([])
            
            valid_peaks_mask = cp.array([len(peaks) >= min_peaks for peaks in valid_peaks_list])
            return cp.stack([valid_peaks_mask, peaks])

        result = data.map_blocks(transform_and_detect, dtype=data.dtype)
        valid_peaks_mask = result[0].astype(bool)
        peaks = result[1:]
        
        return valid_peaks_mask, peaks.tolist()

    def _peak_width_gpu(self, smoothed_hist: cp.ndarray, peak_indices: List[int],
                       bin_edges: cp.ndarray, percentage_cells_present: float,
                       smoothing_window: int) -> List[Tuple[float, float]]:
        """Calculate peak widths using GPU operations."""
        num_bins = len(smoothed_hist)
        
        # Apply threshold
        non_zero_bins = smoothed_hist[smoothed_hist != 0]
        if len(non_zero_bins) == 0:
            return []
            
        threshold = cp.percentile(non_zero_bins, 0.05 * 100)
        smoothed_hist[smoothed_hist < threshold] = 0
        
        peak_widths = []
        
        for peak_index in peak_indices:
            left_minima_index = peak_index
            right_minima_index = peak_index
            
            # Find left minima
            while left_minima_index > 0:
                if cp.all(smoothed_hist[left_minima_index] < smoothed_hist[left_minima_index - smoothing_window:left_minima_index]):
                    break
                left_minima_index -= 1
            
            # Find right minima
            while right_minima_index < num_bins - 1:
                if cp.all(smoothed_hist[right_minima_index] < smoothed_hist[right_minima_index + 1:right_minima_index + smoothing_window + 1]):
                    break
                right_minima_index += 1
            
            # Calculate peak percentage
            peak_sum = cp.sum(smoothed_hist[left_minima_index:right_minima_index + 1])
            total_sum = cp.sum(smoothed_hist)
            peak_percentage = (peak_sum / total_sum) * 100
            
            if peak_percentage >= percentage_cells_present:
                peak_widths.append((bin_edges[left_minima_index], bin_edges[right_minima_index]))
        
        return sorted(peak_widths, key=lambda x: x[0])

    def _find_fsc_threshold_gpu(self, feature: da.Array, peaks: List, 
                              smoothing_window: int) -> Optional[float]:
        """Find the FSC threshold using GPU operations."""
        num_bins = 300

        def threshold_compute(block):
            gpu_block = cp.asarray(block)
            min_val, max_val = cp.min(gpu_block), cp.max(gpu_block)
            bin_edges = cp.linspace(min_val, max_val, num_bins + 1)
            hist, _ = cp.histogram(gpu_block, bins=bin_edges, density=True)
            
            # Zero out bottom and top bins
            bottom_bins = int(num_bins * 0.02)
            top_bins = int(num_bins * 0.98)
            hist[:bottom_bins] = hist[top_bins:] = 0
            
            # Smooth histogram
            smoothed_hist = cusignal.convolve(hist, cp.ones(smoothing_window), mode='same')
            
            # Apply threshold
            non_zero_bins = smoothed_hist[smoothed_hist != 0]
            if len(non_zero_bins) == 0:
                return None
                
            threshold = cp.percentile(non_zero_bins, 1)
            smoothed_hist[smoothed_hist < threshold] = 0
            
            # Find maxima
            maxima = (smoothed_hist > cp.roll(smoothed_hist, 1)) & (smoothed_hist > cp.roll(smoothed_hist, -1))
            maxima[:smoothing_window] = maxima[-smoothing_window:] = False
            peak_indices = cp.where(maxima)[0]
            
            if len(peak_indices) > 1:
                # Calculate peak densities
                peak_densities = []
                for idx in peak_indices:
                    start_idx = max(0, idx - smoothing_window)
                    end_idx = min(idx + smoothing_window + 1, len(smoothed_hist))
                    density = cp.sum(smoothed_hist[start_idx:end_idx])
                    peak_densities.append(density)
                
                max_peak_index = int(cp.argmax(cp.array(peak_densities)))
                
                if max_peak_index == 1:  # If biggest peak is second peak
                    left_minima_index = int(peak_indices[max_peak_index])
                    while left_minima_index > 0:
                        if cp.all(smoothed_hist[left_minima_index] < smoothed_hist[left_minima_index - smoothing_window:left_minima_index]):
                            break
                        left_minima_index -= 1
                    return bin_edges[left_minima_index]
                else:
                    for i in range(max_peak_index - 1, -1, -1):
                        left_minima_index = int(peak_indices[i])
                        while left_minima_index > 0:
                            if cp.all(smoothed_hist[left_minima_index] < smoothed_hist[left_minima_index - smoothing_window:left_minima_index]):
                                break
                            left_minima_index -= 1
                        if bin_edges[left_minima_index] < bin_edges[peak_indices[max_peak_index]]:
                            return bin_edges[left_minima_index]
            
            return None

        results = feature.map_blocks(threshold_compute, dtype=object)
        return results

    def _apply_threshold_gpu(self, feature: da.Array, threshold: float) -> da.Array:
        """Apply threshold using GPU operations."""
        def threshold_compute(block):
            gpu_block = cp.asarray(block)
            return (gpu_block >= threshold).astype(cp.int32)
            
        return feature.map_blocks(threshold_compute, dtype=cp.int32)

    def _get_fsc_column(self, marker_names: List[str]) -> int:
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

    def get_debug_info(self) -> dict:
        """Get debugging information from the last gate operation."""
        return self._debug_info
