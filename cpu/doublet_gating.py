"""
Doublet gating component for FlowMOP.
Handles detection and removal of doublets in flow cytometry data.
"""

import numpy as np
import dask.array as da
import dask
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from typing import Union, Tuple, List, Optional, Dict, Any
import matplotlib.pyplot as plt
from cpu.flowmop_utils import process_histogram, find_peaks, find_left_minimum

# Type definitions
ArrayType = Union[np.ndarray, da.Array]

class DoubletGateStrategy(ABC):
    def __init__(self, enable_dask=True, chunk_size=None):
        self.enable_dask = enable_dask
        self.chunk_size = chunk_size

    @abstractmethod
    def gate(self, data: ArrayType, marker_names: List[str]) -> Tuple[ArrayType, ArrayType]:
        """Apply doublet gating to the data."""
        pass

    def _check_required_parameters(self, marker_names: List[str]) -> bool:
        """Check if all required parameters are present."""
        required_params = ['fsca', 'fsch', 'ssca', 'ssch']
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        all_params_present = all(param in standardized_names for param in required_params)
        
        if not all_params_present:
            warnings.warn("Not all required parameters (FSC-A, FSC-H, SSC-A, SSC-H) are present. "
                        "Doublet removal will be skipped.")
        return all_params_present

    def _calculate_ratios(self, data: ArrayType, marker_names: List[str]) -> Tuple[ArrayType, ArrayType]:
        """
        Calculate aspect ratios for FSC and SSC measurements.
        
        Args:
            data: Flow cytometry data array
            marker_names: List of marker names
            
        Returns:
            tuple[ArrayType, ArrayType]: (fsc_ratio, ssc_ratio)
        """
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        
        try:
            fsc_a_idx = standardized_names.index('fsca')
            fsc_h_idx = standardized_names.index('fsch')
            ssc_a_idx = standardized_names.index('ssca')
            ssc_h_idx = standardized_names.index('ssch')
        except ValueError:
            raise ValueError("Required scatter parameters not found.")
        
        # Set negative values to 0 and clip top 0.1 percentile outliers
        if self.enable_dask and isinstance(data, da.Array):
            # For Dask arrays, use da.clip
            scatter_data = da.clip(
                data[:, [fsc_a_idx, fsc_h_idx, ssc_a_idx, ssc_h_idx]], 
                0, 
                None
            )
            
            # Calculate 99.9th percentile for each channel
            p999 = []
            for i, idx in enumerate([fsc_a_idx, fsc_h_idx, ssc_a_idx, ssc_h_idx]):
                # Create a delayed computation for percentile
                p999.append(dask.delayed(np.percentile)(data[:, idx].compute(), 99.9))
                
            # Compute all percentiles at once
            p999 = dask.compute(*p999)
            
            # Clip values above 99.9th percentile
            for i, idx in enumerate([fsc_a_idx, fsc_h_idx, ssc_a_idx, ssc_h_idx]):
                scatter_data[:, i] = da.clip(scatter_data[:, i], None, p999[i])
                
            # Calculate ratios with handling for division by zero
            fsc_ratio = da.divide(
                scatter_data[:, 0], 
                scatter_data[:, 1], 
                out=da.full_like(scatter_data[:, 0], np.nan), 
                where=scatter_data[:, 1] > 0
            )
            
            ssc_ratio = da.divide(
                scatter_data[:, 2], 
                scatter_data[:, 3], 
                out=da.full_like(scatter_data[:, 2], np.nan), 
                where=scatter_data[:, 3] > 0
            )
            
        else:
            # For NumPy arrays, use np.clip
            scatter_data = np.clip(
                data[:, [fsc_a_idx, fsc_h_idx, ssc_a_idx, ssc_h_idx]], 
                0, 
                None
            )
            
            # Calculate 99.9th percentile for each channel and clip values above it
            for i, idx in enumerate([fsc_a_idx, fsc_h_idx, ssc_a_idx, ssc_h_idx]):
                p999 = np.percentile(data[:, idx], 99.9)
                scatter_data[:, i] = np.clip(scatter_data[:, i], None, p999)
                
            # Calculate ratios with handling for division by zero
            fsc_ratio = np.divide(
                scatter_data[:, 0], 
                scatter_data[:, 1], 
                out=np.full_like(scatter_data[:, 0], np.nan), 
                where=scatter_data[:, 1] > 0
            )
            
            ssc_ratio = np.divide(
                scatter_data[:, 2], 
                scatter_data[:, 3], 
                out=np.full_like(scatter_data[:, 2], np.nan), 
                where=scatter_data[:, 3] > 0
            )
            
        return fsc_ratio, ssc_ratio

    @staticmethod
    def _standardize_marker_name(name: str) -> str:
        """Standardize marker names by removing symbols and converting to lowercase."""
        import re
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

    def plot_ratio_histograms(self, fsc_ratio: np.ndarray, ssc_ratio: np.ndarray, 
                            fsc_threshold: float, ssc_threshold: float,
                            title: str = "Doublet Gating Thresholds") -> None:
        """
        Plot histogram of FSC ratio with threshold, peaks, and derivatives.
        
        Args:
            fsc_ratio: Forward scatter ratio values
            ssc_ratio: Not used (kept for backward compatibility)
            fsc_threshold: FSC threshold value
            ssc_threshold: Not used (kept for backward compatibility)
            title: Plot title
        """
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, figure=fig)
        ax_hist = fig.add_subplot(gs[0, :])  # Top row: histogram
        ax_deriv1 = fig.add_subplot(gs[1, 0])  # Bottom left: first derivative
        ax_deriv2 = fig.add_subplot(gs[1, 1])  # Bottom right: second derivative
        
        # Process histogram using flowmop_utils function
        fsc_data = fsc_ratio[~np.isnan(fsc_ratio)]
        fsc_data = np.clip(fsc_data, 0, None)
        
        # Process histogram with smoothing
        print("plotting")
        fsc_hist = process_histogram(fsc_data, smoothing_window=2)
        
        if fsc_hist is not None:
            smoothed_hist, bin_edges, peak_indices, _ = fsc_hist
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
            
            # Plot main histogram
            ax_hist.plot(bin_edges[:-1], smoothed_hist, 'b-', alpha=0.6, label='FSC Ratio Distribution')
            ax_hist.axvline(fsc_threshold, color='r', linestyle='--', linewidth=2, 
                          label=f'Threshold: {fsc_threshold:.2f}')
            
            # Plot detected peaks
            for peak_idx in peak_indices:
                ax_hist.plot(bin_edges[peak_idx], smoothed_hist[peak_idx], 'go', 
                           markersize=10, label='Peak' if peak_idx == peak_indices[0] else None)
            
            ax_hist.set_title('FSC-A/FSC-H Ratio Distribution')
            ax_hist.set_xlabel('FSC-A/FSC-H Ratio')
            ax_hist.set_ylabel('Count')
            ax_hist.set_xlim(0, min(5, fsc_threshold * 2))
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)
            
            # Calculate derivatives
            first_derivative = np.gradient(smoothed_hist, bin_centers)
            second_derivative = np.gradient(first_derivative, bin_centers)
            
            # Plot first derivative
            ax_deriv1.plot(bin_centers, first_derivative, 'r-', label='First Derivative')
            ax_deriv1.axvline(fsc_threshold, color='r', linestyle='--', linewidth=2)
            ax_deriv1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax_deriv1.set_title('First Derivative')
            ax_deriv1.set_xlabel('FSC-A/FSC-H Ratio')
            ax_deriv1.set_ylabel('First Derivative')
            ax_deriv1.set_xlim(0, min(5, fsc_threshold * 2))
            ax_deriv1.grid(True, alpha=0.3)
            ax_deriv1.legend()
            
            # Plot second derivative
            ax_deriv2.plot(bin_centers, second_derivative, 'g-', label='Second Derivative')
            ax_deriv2.axvline(fsc_threshold, color='r', linestyle='--', linewidth=2)
            ax_deriv2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax_deriv2.set_title('Second Derivative')
            ax_deriv2.set_xlabel('FSC-A/FSC-H Ratio')
            ax_deriv2.set_ylabel('Second Derivative')
            ax_deriv2.set_xlim(0, min(5, fsc_threshold * 2))
            ax_deriv2.grid(True, alpha=0.3)
            ax_deriv2.legend()
        
        plt.tight_layout()
        plt.show()

class MADDoubletGate(DoubletGateStrategy):
    def __init__(self, mad_threshold=5, enable_dask=True, chunk_size=None):
        super().__init__(enable_dask=enable_dask, chunk_size=chunk_size)
        self.mad_threshold = mad_threshold

    def gate(self, data: ArrayType, marker_names: List[str]) -> Tuple[ArrayType, ArrayType]:
        """
        Apply MAD-based doublet gating to the data.
        
        Args:
            data: Flow cytometry data array
            marker_names: List of marker names
            
        Returns:
            Tuple of (filtered_array, doublet_gate_vector)
        """
        # Ensure data is in Dask array format if Dask is enabled
        if self.enable_dask and not isinstance(data, da.Array):
            data = da.from_array(data, chunks=(self.chunk_size or 'auto', -1))
            
        # Check if required parameters are present
        if not self._check_required_parameters(marker_names):
            if self.enable_dask and isinstance(data, da.Array):
                return data, da.ones(data.shape[0], chunks=data.chunks[0], dtype=np.int32)
            else:
                return data, np.ones(data.shape[0], dtype=np.int32)
                
        # Calculate aspect ratios
        fsc_ratio, ssc_ratio = self._calculate_ratios(data, marker_names)
        
        if self.enable_dask:
            # Create delayed functions for MAD calculation
            @dask.delayed
            def calculate_mad_threshold(ratio):
                # Remove NaN values
                ratio = ratio[~np.isnan(ratio)]
                # Calculate median
                median = np.median(ratio)
                # Calculate MAD
                mad = np.median(np.abs(ratio - median))
                # Calculate threshold
                return median + (self.mad_threshold * mad)
                
            # Calculate thresholds using delayed functions
            fsc_threshold_delayed = calculate_mad_threshold(fsc_ratio.compute())
            ssc_threshold_delayed = calculate_mad_threshold(ssc_ratio.compute())
            
            # Compute thresholds
            fsc_threshold, ssc_threshold = dask.compute(fsc_threshold_delayed, ssc_threshold_delayed)
            
            # Create gate vectors
            fsc_gate = (fsc_ratio <= fsc_threshold).astype(np.int32)
            ssc_gate = (ssc_ratio <= ssc_threshold).astype(np.int32)
            
            # Combine gates
            doublet_gate = fsc_gate & ssc_gate
            
            # Filter data
            filtered_data = data[doublet_gate > 0]
            
            return filtered_data, doublet_gate
        else:
            # For NumPy arrays, compute directly
            # Remove NaN values
            fsc_ratio_clean = fsc_ratio[~np.isnan(fsc_ratio)]
            ssc_ratio_clean = ssc_ratio[~np.isnan(ssc_ratio)]
            
            # Calculate medians
            fsc_median = np.median(fsc_ratio_clean)
            ssc_median = np.median(ssc_ratio_clean)
            
            # Calculate MADs
            fsc_mad = np.median(np.abs(fsc_ratio_clean - fsc_median))
            ssc_mad = np.median(np.abs(ssc_ratio_clean - ssc_median))
            
            # Calculate thresholds
            fsc_threshold = fsc_median + (self.mad_threshold * fsc_mad)
            ssc_threshold = ssc_median + (self.mad_threshold * ssc_mad)
            
            # Create gate vectors
            fsc_gate = (fsc_ratio <= fsc_threshold).astype(np.int32)
            ssc_gate = (ssc_ratio <= ssc_threshold).astype(np.int32)
            
            # Combine gates
            doublet_gate = fsc_gate & ssc_gate
            
            # Filter data
            filtered_data = data[doublet_gate > 0]
            
            return filtered_data, doublet_gate

class InflectionDoubletGate(DoubletGateStrategy):
    """
    Alternative doublet gating strategy using inflection points in ratio histograms.
    This method finds natural breakpoints in the FSC and SSC ratio distributions
    to determine optimal thresholds for doublet discrimination.
    """
    def __init__(self, bins='auto', smoothing_factor=0.5, fallback_mad_threshold=5, enable_dask=True, chunk_size=None):
        """
        Initialize the inflection point-based doublet gating strategy.
        
        Args:
            bins: Number of bins or method for histogram ('auto', 'fd', 'scott')
            smoothing_factor: Bandwidth factor for KDE smoothing (0.1 to 1.0)
            fallback_mad_threshold: MAD threshold to use if inflection point method fails
        """
        super().__init__(enable_dask=enable_dask, chunk_size=chunk_size)
        self.bins = bins
        self.smoothing_factor = smoothing_factor
        self.fallback_mad_threshold = fallback_mad_threshold
        self._debug_info = {}  # Store intermediate results for debugging

    def gate(self, data: ArrayType, marker_names: List[str]) -> Tuple[ArrayType, ArrayType]:
        """
        Apply inflection point-based doublet gating to the data.
        Falls back to MAD-based thresholding if inflection method fails.
        
        Args:
            data: Flow cytometry data array
            marker_names: List of marker names
            
        Returns:
            tuple: (filtered_data, doublet_vector)
        """
        # Ensure data is in Dask array format if Dask is enabled
        if self.enable_dask and not isinstance(data, da.Array):
            data = da.from_array(data, chunks=(self.chunk_size or 'auto', -1))
            
        # Check if required parameters are present
        if not self._check_required_parameters(marker_names):
            if self.enable_dask and isinstance(data, da.Array):
                return data, da.ones(data.shape[0], chunks=data.chunks[0], dtype=np.int32)
            else:
                return data, np.ones(data.shape[0], dtype=np.int32)
                
        # Calculate aspect ratios
        fsc_ratio, ssc_ratio = self._calculate_ratios(data, marker_names)
        self._debug_info['ratios'] = {'fsc': fsc_ratio}

        # Try inflection point method first
        inflection_success = False
        fsc_threshold = None

        # Step 2: Generate histogram starting from ratio = 1
        fsc_hist, fsc_bin_edges, fsc_peaks = self._generate_smooth_histogram(fsc_ratio)
        self._debug_info['histograms'] = {'fsc': fsc_hist}

        # Step 3: Analyze histogram and find threshold
        fsc_threshold, inflection_success = self._analyze_histogram_for_threshold(
            fsc_hist, fsc_bin_edges, fsc_peaks
        )

        # Fall back to MAD-based thresholding if inflection method failed
        if not inflection_success:
            warnings.warn("Inflection point method failed to find threshold. Falling back to MAD-based thresholding.")
            # Use MAD threshold calculation
            mad_gate = MADDoubletGate(mad_threshold=self.fallback_mad_threshold)
            filtered_data, doublet_vector = mad_gate.gate(data, marker_names)
            
            # Get the threshold that was used for plotting
            if self.enable_dask and isinstance(fsc_ratio, da.Array):
                valid_ratios = fsc_ratio[~da.isnan(fsc_ratio)].compute()
            else:
                valid_ratios = fsc_ratio[~np.isnan(fsc_ratio)]
                
            fsc_threshold = np.median(valid_ratios) + self.fallback_mad_threshold * stats.median_abs_deviation(valid_ratios)
            self._debug_info['thresholds'] = {'fsc': fsc_threshold}
            return filtered_data, doublet_vector
        else:
            # Apply inflection-based threshold
            filtered_data, doublet_vector = self._apply_inflection_threshold(data, fsc_ratio, fsc_threshold)
            self._debug_info['thresholds'] = {'fsc': fsc_threshold}
            return filtered_data, doublet_vector

    def _generate_smooth_histogram(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate smoothed histogram using process_histogram from flowmop_utils.
        Only considers ratios >= 1 since values below 1 are not biologically meaningful
        (would mean height > area which is physically impossible).
        
        Args:
            data: Input ratio data
            
        Returns:
            tuple: (smoothed_hist, bin_edges)
        """
        # Remove NaN values and restrict to biologically meaningful ratios (>= 1)
        data = data[~np.isnan(data)]
        data = data[data >= 1]  # Only consider ratios >= 1
        
        # Clip outliers at 99th percentile
        q99 = np.percentile(data, 99)
        data = np.clip(data, None, q99)
        
        # Use process_histogram with a reasonable smoothing window
        hist_result = process_histogram(data, smoothing_window=5, num_bins=100)
        
        if hist_result is None:
            return np.array([]), np.array([])
            
        smoothed_hist, bin_edges, peak_indices, peak_densities = hist_result
        
        return smoothed_hist, bin_edges, peak_indices

    def _analyze_histogram_for_threshold(self, hist: np.ndarray, bin_edges: np.ndarray, 
                                        peaks: np.ndarray) -> Tuple[Optional[float], bool]:
        """
        Analyze histogram to find threshold based on peaks and valleys.
        
        Args:
            hist: Histogram counts
            bin_edges: Bin edges
            peaks: Peak indices
            
        Returns:
            tuple: (threshold, success_flag)
        """

        if len(hist) == 0 or len(peaks) < 2:
            return None, False
            
        # Get peak densities and find the maximum peak
        peak_densities = hist[peaks]
        max_peak_idx = np.argmax(peak_densities)
        max_peak_bin_value = bin_edges[peaks[max_peak_idx]]
        
        # Only consider peaks after the maximum peak
        if max_peak_idx >= len(peaks) - 1:
            return None, False
            
        potential_valleys = []
        
        # First, find the valley between max peak and next peak
        start_idx = peaks[max_peak_idx]
        end_idx = peaks[max_peak_idx + 1]
        valley_idx = start_idx + np.argmin(hist[start_idx:end_idx])
        # Plot the histogram segment between peaks
        valley_position = bin_edges[valley_idx]
        potential_valleys.append(valley_position)
        
        # Then check for additional valleys between subsequent peaks
        # but only if they're not beyond the max peak's bin value
        for i in range(max_peak_idx + 1, len(peaks) - 1):
            # Stop if we've reached a peak beyond the max peak's bin value
            if bin_edges[peaks[i]] > max_peak_bin_value:
                break
                
            start_idx = peaks[i]
            end_idx = peaks[i + 1]
            valley_idx = start_idx + np.argmin(hist[start_idx:end_idx])
            valley_position = bin_edges[valley_idx]
            
            # Only consider valleys not beyond max peak bin value
            if valley_position <= max_peak_bin_value:
                potential_valleys.append(valley_position)
        
        # Take the smallest valley
        if potential_valleys:
            return min(potential_valleys), True
            
        return None, False

    def _apply_inflection_threshold(self, data: ArrayType, fsc_ratio: ArrayType, 
                                  fsc_threshold: float) -> Tuple[ArrayType, ArrayType]:
        """
        Apply inflection-based threshold to the data.
        
        Args:
            data: Input data array
            fsc_ratio: FSC ratio array
            fsc_threshold: Threshold value
            
        Returns:
            tuple: (filtered_data, doublet_vector)
        """
        # Handle Dask arrays appropriately
        if self.enable_dask and isinstance(fsc_ratio, da.Array):
            doublet_vector = (fsc_ratio <= fsc_threshold).astype(np.int32)
            filtered_data = data[doublet_vector == 1]
        else:
            doublet_vector = (fsc_ratio <= fsc_threshold).astype(np.int32)
            filtered_data = data[doublet_vector == 1]
            
        return filtered_data, doublet_vector
        
    def _get_threshold_from_inflection(self, ratio: np.ndarray, hist: tuple, inflections: np.ndarray) -> float:
        """
        Get threshold value from inflection points.
        
        Args:
            ratio: Original ratio data
            hist: Histogram tuple (counts, bin_edges, smoothed_counts)
            inflections: Array of inflection point indices
            
        Returns:
            float: Threshold value
        """
        if len(inflections) == 0:
            print("no inflections")
            return self._calculate_mad_threshold(ratio)

        # Find the main peak
        peak_idx = np.argmax(hist[2])  # Using smoothed counts
        
        # Get inflection points after the peak
        valid_points = inflections[inflections > peak_idx]
        
        if len(valid_points) > 0:
            # Use the first inflection point after the peak
            threshold_idx = valid_points[0]
            return hist[1][threshold_idx]  # Use bin edge at inflection point
        
        return self._calculate_mad_threshold(ratio)

    def _fallback_to_mad(self, data: np.ndarray, fsc_ratio: np.ndarray, ssc_ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fallback to MAD-based thresholding when inflection point method fails."""
        warnings.warn("Falling back to MAD-based thresholding")
        fsc_threshold = self._calculate_mad_threshold(fsc_ratio)
        ssc_threshold = self._calculate_mad_threshold(ssc_ratio)
        
        doublet_vector = ((fsc_ratio <= fsc_threshold) & 
                         (ssc_ratio <= ssc_threshold)).astype(int)
        filtered_data = data[doublet_vector == 1]
        
        return filtered_data, doublet_vector

    def _calculate_mad_threshold(self, ratio: np.ndarray) -> float:
        """Fallback MAD-based threshold calculation."""
        median_ratio = np.nanmedian(ratio)
        mad = np.nanmedian(np.abs(ratio - median_ratio))
        return median_ratio + self.fallback_mad_threshold * mad

    def get_debug_info(self) -> dict:
        """Get debugging information from the last gate operation."""
        return self._debug_info
