"""
Doublet gating component for FlowMOP.
Handles detection and removal of doublets in flow cytometry data.
"""

import numpy as np
from abc import ABC, abstractmethod
import warnings
from scipy import stats
import matplotlib.pyplot as plt
from .flowmop_utils import process_histogram, find_peaks, find_left_minimum

class DoubletGateStrategy(ABC):
    @abstractmethod
    def gate(self, data: np.ndarray, marker_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Apply doublet gating to the data."""
        pass

    def _check_required_parameters(self, marker_names: list[str]) -> bool:
        """Check if all required parameters are present."""
        required_params = ['fsca', 'fsch', 'ssca', 'ssch']
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        all_params_present = all(param in standardized_names for param in required_params)
        
        if not all_params_present:
            warnings.warn("Not all required parameters (FSC-A, FSC-H, SSC-A, SSC-H) are present. "
                        "Doublet removal will be skipped.")
        return all_params_present

    def _calculate_ratios(self, data: np.ndarray, marker_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate aspect ratios for FSC and SSC measurements.
        
        Args:
            data: Flow cytometry data array
            marker_names: List of marker names
            
        Returns:
            tuple[np.ndarray, np.ndarray]: (fsc_ratio, ssc_ratio)
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
        data[:, [fsc_a_idx, fsc_h_idx, ssc_a_idx, ssc_h_idx]] = np.clip(
            data[:, [fsc_a_idx, fsc_h_idx, ssc_a_idx, ssc_h_idx]], 
            0, 
            None
        )

        # Calculate 99.9th percentile for each channel and clip values above it
        for idx in [fsc_a_idx, fsc_h_idx, ssc_a_idx, ssc_h_idx]:
            p999 = np.percentile(data[:, idx], 99.9)
            data[:, idx] = np.clip(data[:, idx], None, p999)

        # Calculate ratios with handling for division by zero and negative values
        fsc_ratio = np.divide(data[:, fsc_a_idx], data[:, fsc_h_idx], 
                            out=np.full_like(data[:, fsc_a_idx], np.nan), 
                            where=data[:, fsc_h_idx] > 0)
        ssc_ratio = np.divide(data[:, ssc_a_idx], data[:, ssc_h_idx],
                            out=np.full_like(data[:, ssc_a_idx], np.nan),
                            where=data[:, ssc_h_idx] > 0)
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
    def __init__(self, mad_threshold=5):
        self.mad_threshold = mad_threshold

    def gate(self, data: np.ndarray, marker_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not self._check_required_parameters(marker_names):
            return data, np.ones(data.shape[0], dtype=int)

        try:
            fsc_ratio, ssc_ratio = self._calculate_ratios(data, marker_names)
        except ValueError as e:
            warnings.warn(str(e))
            return data, np.ones(data.shape[0], dtype=int)

        # Calculate MAD thresholds
        fsc_threshold = self._calculate_mad_threshold(fsc_ratio)
        ssc_threshold = self._calculate_mad_threshold(ssc_ratio)
        
        # Apply thresholds
        doublet_vector = ((fsc_ratio <= fsc_threshold) & 
                         (ssc_ratio <= ssc_threshold)).astype(int)
        filtered_data = data[doublet_vector == 1]
        
        return filtered_data, doublet_vector

    def _calculate_mad_threshold(self, ratio: np.ndarray) -> float:
        """Calculate MAD-based threshold for ratio values."""
        median_ratio = np.nanmedian(ratio)
        mad = np.nanmedian(np.abs(ratio - median_ratio))
        return median_ratio + self.mad_threshold * mad

class InflectionDoubletGate(DoubletGateStrategy):
    """
    Alternative doublet gating strategy using inflection points in ratio histograms.
    This method finds natural breakpoints in the FSC and SSC ratio distributions
    to determine optimal thresholds for doublet discrimination.
    """
    def __init__(self, bins='auto', smoothing_factor=0.5, fallback_mad_threshold=5):
        """
        Initialize the inflection point-based doublet gating strategy.
        
        Args:
            bins: Number of bins or method for histogram ('auto', 'fd', 'scott')
            smoothing_factor: Bandwidth factor for KDE smoothing (0.1 to 1.0)
            fallback_mad_threshold: MAD threshold to use if inflection point method fails
        """
        self.bins = bins
        self.smoothing_factor = smoothing_factor
        self.fallback_mad_threshold = fallback_mad_threshold
        self._debug_info = {}  # Store intermediate results for debugging

    def gate(self, data: np.ndarray, marker_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply inflection point-based doublet gating to the data.
        Falls back to MAD-based thresholding if inflection method fails.
        
        Args:
            data: Flow cytometry data array
            marker_names: List of marker names
            
        Returns:
            tuple: (filtered_data, doublet_vector)
        """
        if not self._check_required_parameters(marker_names):
            return data, np.ones(data.shape[0], dtype=int)

        # Step 1: Calculate ratios
        fsc_ratio, ssc_ratio = self._calculate_ratios(data, marker_names)
        self._debug_info['ratios'] = {'fsc': fsc_ratio}

        # Try inflection point method first
        inflection_success = False
        fsc_threshold = None

        # Step 2: Generate histogram starting from ratio = 1
        fsc_hist, fsc_bin_edges, fsc_peaks = self._generate_smooth_histogram(fsc_ratio)
        self._debug_info['histograms'] = {'fsc': fsc_hist}

        if len(fsc_hist) > 0:  # Check if histogram generation was successful
            # Step 3: Find peaks
            
            if len(fsc_peaks) >= 2:
                # Get peak densities and find the maximum peak
                peak_densities = fsc_hist[fsc_peaks]
                max_peak_idx = np.argmax(peak_densities)
                max_peak_bin_value = fsc_bin_edges[fsc_peaks[max_peak_idx]]
                
                # Only consider peaks after the maximum peak
                if max_peak_idx < len(fsc_peaks) - 1:
                    potential_valleys = []
                    
                    # First, find the valley between max peak and next peak
                    start_idx = fsc_peaks[max_peak_idx]
                    end_idx = fsc_peaks[max_peak_idx + 1]
                    print("Start and end indices: ", start_idx, end_idx)
                    valley_idx = start_idx + np.argmin(fsc_hist[start_idx:end_idx])
                    # Plot the histogram segment between peaks
                    valley_position = fsc_bin_edges[valley_idx]

                    potential_valleys.append(valley_position)
                    # Then check for additional valleys between subsequent peaks
                    # but only if they're not beyond the max peak's bin value
                    for i in range(max_peak_idx + 1, len(fsc_peaks) - 1):
                        # Stop if we've reached a peak beyond the max peak's bin value
                        if fsc_bin_edges[fsc_peaks[i]] > max_peak_bin_value:
                            break
                            
                        start_idx = fsc_peaks[i]
                        end_idx = fsc_peaks[i + 1]
                        valley_idx = start_idx + np.argmin(fsc_hist[start_idx:end_idx])
                        valley_position = fsc_bin_edges[valley_idx]
                        
                        # Only consider valleys not beyond max peak bin value
                        if valley_position <= max_peak_bin_value:
                            potential_valleys.append(valley_position)
                    
                    # Take the smallest valley
                    if potential_valleys:
                        fsc_threshold = min(potential_valleys)
                        inflection_success = True

        # Fall back to MAD-based thresholding if inflection method failed
        if not inflection_success:
            warnings.warn("Inflection point method failed to find threshold. Falling back to MAD-based thresholding.")
            # Use MAD threshold calculation
            mad_gate = MADDoubletGate(mad_threshold=self.fallback_mad_threshold)
            filtered_data, doublet_vector = mad_gate.gate(data, marker_names)
            
            # Get the threshold that was used for plotting
            valid_ratios = fsc_ratio[~np.isnan(fsc_ratio)]
            fsc_threshold = np.median(valid_ratios) + self.fallback_mad_threshold * stats.median_abs_deviation(valid_ratios)
        else:
            # Apply inflection-based threshold
            doublet_vector = (fsc_ratio <= fsc_threshold).astype(int)
            filtered_data = data[doublet_vector == 1]
            
        self._debug_info['thresholds'] = {'fsc': fsc_threshold}
        
        # Plot the ratio histograms with derivatives
        # self.plot_ratio_histograms(fsc_ratio, ssc_ratio, fsc_threshold, fsc_threshold)
        
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
        q99 = np.quantile(data, 0.99)
        data = np.clip(data, None, q99)
        
        # Use process_histogram with a reasonable smoothing window
        hist_result = process_histogram(data, smoothing_window=5, num_bins=100)
        
        if hist_result is None:
            return np.array([]), np.array([])
            
        smoothed_hist, bin_edges, peak_indices, peak_densities = hist_result
        
        # # Plot the histogram
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 6))
        # plt.hist(data, bins=bin_edges, alpha=0.5, density=True, label='Raw Data')
        # plt.plot((bin_edges[:-1] + bin_edges[1:]) / 2, smoothed_hist, 'r-', label='Smoothed')
        # if len(peak_indices) > 0:
        #     print("Peaks: ", peak_indices)
        #     plt.plot(bin_edges[peak_indices], smoothed_hist[peak_indices], 'go', label='Peaks')
        # plt.xlabel('Ratio')
        # plt.ylabel('Density')
        # plt.title('Ratio Distribution')
        # plt.legend()
        # plt.show()
        
        return smoothed_hist, bin_edges, peak_indices

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
