"""
Doublet gating component for FlowMOP.
Handles detection and removal of doublets in flow cytometry data.
"""

import numpy as np
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from scipy.signal import savgol_filter

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

class MADDoubletGate(DoubletGateStrategy):
    def __init__(self, mad_threshold=5):
        self.mad_threshold = mad_threshold

    def gate(self, data: np.ndarray, marker_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply MAD-based doublet gating to the data.
        
        Args:
            data: Flow cytometry data array
            marker_names: List of marker names
            
        Returns:
            tuple: (filtered_data, doublet_vector)
        """
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
        
        Args:
            data: Flow cytometry data array
            marker_names: List of marker names
            
        Returns:
            tuple: (filtered_data, doublet_vector)
        """
        if not self._check_required_parameters(marker_names):
            return data, np.ones(data.shape[0], dtype=int)

        try:
            # Step 1: Calculate ratios
            fsc_ratio, ssc_ratio = self._calculate_ratios(data, marker_names)
            self._debug_info['ratios'] = {'fsc': fsc_ratio, 'ssc': ssc_ratio}

            # Step 2: Generate histograms
            fsc_hist = self._generate_smooth_histogram(fsc_ratio)
            ssc_hist = self._generate_smooth_histogram(ssc_ratio)
            self._debug_info['histograms'] = {'fsc': fsc_hist, 'ssc': ssc_hist}

            # Step 3: Find inflection points
            fsc_inflections = self._find_inflection_points(fsc_hist[2])  # Using smoothed counts
            ssc_inflections = self._find_inflection_points(ssc_hist[2])
            self._debug_info['inflections'] = {'fsc': fsc_inflections, 'ssc': ssc_inflections}

            # Step 4: Get thresholds
            fsc_threshold = self._get_threshold_from_inflection(fsc_ratio, fsc_hist, fsc_inflections)
            ssc_threshold = self._get_threshold_from_inflection(ssc_ratio, ssc_hist, ssc_inflections)
            self._debug_info['thresholds'] = {'fsc': fsc_threshold, 'ssc': ssc_threshold}

        except ValueError as e:
            warnings.warn(f"Error in inflection point analysis: {str(e)}. Using MAD-based thresholds.")
            return self._fallback_to_mad(data, fsc_ratio, ssc_ratio)

        # Apply thresholds
        doublet_vector = ((fsc_ratio <= fsc_threshold) & 
                         (ssc_ratio <= ssc_threshold)).astype(int)
        filtered_data = data[doublet_vector == 1]
        
        return filtered_data, doublet_vector

    def _generate_smooth_histogram(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate smoothed histogram using Gaussian KDE.
        
        Args:
            data: Input ratio data
            
        Returns:
            tuple: (counts, bin_edges, smoothed_counts)
        """
        # Remove NaN values for histogram generation
        data = data[~np.isnan(data)]
        
        # Generate histogram
        counts, bin_edges = np.histogram(data, bins=self.bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Apply KDE smoothing
        kde = stats.gaussian_kde(data, bw_method=self.smoothing_factor)
        smoothed_counts = kde(bin_centers) * len(data) * (bin_edges[1] - bin_edges[0])
        
        return counts, bin_edges, smoothed_counts

    def _find_inflection_points(self, curve: np.ndarray) -> np.ndarray:
        """
        Detect inflection points using second derivative.
        
        Args:
            curve: Smoothed histogram curve
            
        Returns:
            np.ndarray: Indices of inflection points
        """
        # Apply Savitzky-Golay filter for noise reduction
        smoothed = savgol_filter(curve, window_length=7, polyorder=3)
        
        # Calculate second derivative
        second_derivative = np.gradient(np.gradient(smoothed))
        
        # Find zero crossings of second derivative
        zero_crossings = np.where(np.diff(np.signbit(second_derivative)))[0]
        
        # Filter weak inflection points
        strength_threshold = np.max(np.abs(second_derivative)) * 0.1
        strong_points = [i for i in zero_crossings 
                        if abs(second_derivative[i]) > strength_threshold]
        
        return np.array(strong_points)

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
