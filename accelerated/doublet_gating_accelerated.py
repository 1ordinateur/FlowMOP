"""
GPU-accelerated doublet gating implementation using CuPy.
All functions assume input arrays are CuPy arrays.
"""

import warnings
from typing import Tuple, List, Optional
import cupy as cp
from dataclasses import dataclass

from cpu.doublet_gating import DoubletGateStrategy
from .gpu_funcs.flowmop_utils_accelerated_gpu import process_histogram

@dataclass
class DebrisGateResult:
    """Results from doublet gating operation."""
    filtered_data: cp.ndarray
    doublet_vector: cp.ndarray

class CuPyMADDoubletGate(DoubletGateStrategy):
    """MAD-based doublet gating implementation using GPU acceleration."""
    
    def __init__(self, mad_threshold: float = 5):
        """
        Initialize the gating strategy.
        
        Args:
            mad_threshold: Threshold multiplier for MAD-based gating
        """
        self.mad_threshold = cp.float32(mad_threshold)

    def gate(self, data: cp.ndarray, marker_names: List[str]) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Apply doublet gating using GPU acceleration.
        
        Args:
            data: Input flow cytometry data (CuPy array)
            marker_names: List of marker names corresponding to columns
            
        Returns:
            Tuple of (filtered_data, doublet_vector)
        """
        if not self._check_required_parameters(marker_names):
            return data, cp.ones(data.shape[0], dtype=cp.int32)

        try:
            # Calculate ratios using GPU operations
            fsc_ratio, ssc_ratio = self._calculate_ratios(data, marker_names)
            
            # Calculate MAD thresholds
            fsc_threshold = self._calculate_mad_threshold(fsc_ratio)
            ssc_threshold = self._calculate_mad_threshold(ssc_ratio)
            
            # Apply thresholds
            doublet_vector = ((fsc_ratio <= fsc_threshold) & 
                            (ssc_ratio <= ssc_threshold)).astype(cp.int32)
            filtered_data = data[doublet_vector == 1]
            
            return filtered_data, doublet_vector
            
        except Exception as e:
            warnings.warn(f"GPU acceleration failed: {str(e)}.")
            return data, cp.ones(data.shape[0], dtype=cp.int32)

    def _calculate_ratios(self, data: cp.ndarray, marker_names: List[str]) -> Tuple[cp.ndarray, cp.ndarray]:
        """Calculate FSC and SSC ratios using GPU operations."""
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        
        try:
            fsc_a_idx = standardized_names.index('fsca')
            fsc_h_idx = standardized_names.index('fsch')
            ssc_a_idx = standardized_names.index('ssca')
            ssc_h_idx = standardized_names.index('ssch')
        except ValueError:
            raise ValueError("Required scatter parameters not found.")
        
        # Set negative values to 0 and clip top 0.1 percentile outliers
        data_subset = data[:, [fsc_a_idx, fsc_h_idx, ssc_a_idx, ssc_h_idx]]
        data_subset = cp.clip(data_subset, cp.float32(0), None)

        # Calculate 99.9th percentile for each channel and clip values above it
        for i, idx in enumerate([fsc_a_idx, fsc_h_idx, ssc_a_idx, ssc_h_idx]):
            p999 = cp.float32(cp.percentile(data[:, idx], 99.9))
            data[:, idx] = cp.clip(data[:, idx], None, p999)

        # Calculate ratios with handling for division by zero
        fsc_ratio = cp.divide(data[:, fsc_a_idx], data[:, fsc_h_idx],
                            out=cp.full_like(data[:, fsc_a_idx], cp.nan),
                            where=data[:, fsc_h_idx] > 0)
        ssc_ratio = cp.divide(data[:, ssc_a_idx], data[:, ssc_h_idx],
                            out=cp.full_like(data[:, ssc_a_idx], cp.nan),
                            where=data[:, ssc_h_idx] > 0)
        
        return fsc_ratio, ssc_ratio

    def _calculate_mad_threshold(self, ratio: cp.ndarray) -> cp.float32:
        """Calculate MAD-based threshold for ratio values."""
        median_ratio = cp.nanmedian(ratio)
        mad = cp.nanmedian(cp.abs(ratio - median_ratio))
        return cp.float32(median_ratio + self.mad_threshold * mad)

class CuPyInflectionDoubletGate(DoubletGateStrategy):
    """
    GPU-accelerated inflection point-based doublet gating.
    """
    
    def __init__(self, bins='auto', smoothing_factor: float = 0.5, fallback_mad_threshold: float = 5):
        self.bins = bins
        self.smoothing_factor = cp.float32(smoothing_factor)
        self.fallback_mad_threshold = cp.float32(fallback_mad_threshold)
        self._debug_info = {}

    def gate(self, data: cp.ndarray, marker_names: List[str]) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Apply inflection point-based doublet gating using GPU acceleration.
        
        Args:
            data: Input flow cytometry data (CuPy array)
            marker_names: List of marker names
            
        Returns:
            Tuple of (filtered_data, doublet_vector)
        """
        if not self._check_required_parameters(marker_names):
            return data, cp.ones(data.shape[0], dtype=cp.int32)

        try:
            # Calculate ratios using GPU operations
            fsc_ratio, ssc_ratio = self._calculate_ratios(data, marker_names)
            self._debug_info['ratios'] = {'fsc': fsc_ratio, 'ssc': ssc_ratio}

            # Generate histograms and find inflection points
            fsc_hist = self._generate_smooth_histogram(fsc_ratio)
            ssc_hist = self._generate_smooth_histogram(ssc_ratio)
            self._debug_info['histograms'] = {'fsc': fsc_hist, 'ssc': ssc_hist}

            # Get thresholds
            fsc_threshold = self._get_threshold_from_histogram(fsc_ratio, fsc_hist)
            ssc_threshold = self._get_threshold_from_histogram(ssc_ratio, ssc_hist)
            self._debug_info['thresholds'] = {'fsc': fsc_threshold, 'ssc': ssc_threshold}

            # Apply thresholds
            doublet_vector = ((fsc_ratio <= fsc_threshold) & 
                            (ssc_ratio <= ssc_threshold)).astype(cp.int32)
            filtered_data = data[doublet_vector == 1]
            
            return filtered_data, doublet_vector

        except Exception as e:
            warnings.warn(f"GPU acceleration failed: {str(e)}. Using fallback thresholding.")
            return self._fallback_to_mad(data, fsc_ratio, ssc_ratio)

    def _calculate_ratios(self, data: cp.ndarray, marker_names: List[str]) -> Tuple[cp.ndarray, cp.ndarray]:
        """Calculate FSC and SSC ratios using GPU operations."""
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        
        try:
            fsc_a_idx = standardized_names.index('fsca')
            fsc_h_idx = standardized_names.index('fsch')
            ssc_a_idx = standardized_names.index('ssca')
            ssc_h_idx = standardized_names.index('ssch')
        except ValueError:
            raise ValueError("Required scatter parameters not found.")
        
        # Set negative values to 0 and clip top 0.1 percentile outliers
        data_subset = data[:, [fsc_a_idx, fsc_h_idx, ssc_a_idx, ssc_h_idx]]
        data_subset = cp.clip(data_subset, cp.float32(0), None)

        # Calculate 99.9th percentile for each channel and clip values above it
        for i, idx in enumerate([fsc_a_idx, fsc_h_idx, ssc_a_idx, ssc_h_idx]):
            p999 = cp.float32(cp.percentile(data[:, idx], 99.9))
            data[:, idx] = cp.clip(data[:, idx], None, p999)

        # Calculate ratios with handling for division by zero
        fsc_ratio = cp.divide(data[:, fsc_a_idx], data[:, fsc_h_idx],
                            out=cp.full_like(data[:, fsc_a_idx], cp.nan),
                            where=data[:, fsc_h_idx] > 0)
        ssc_ratio = cp.divide(data[:, ssc_a_idx], data[:, ssc_h_idx],
                            out=cp.full_like(data[:, ssc_a_idx], cp.nan),
                            where=data[:, ssc_h_idx] > 0)
        
        return fsc_ratio, ssc_ratio

    def _generate_smooth_histogram(self, data: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """Generate smoothed histogram using GPU operations."""
        # Remove NaN values and restrict to biologically meaningful ratios (>= 1)
        valid_data = data[~cp.isnan(data)]
        valid_data = valid_data[valid_data >= cp.float32(1)]
        
        # Clip outliers at 99th percentile
        q99 = cp.float32(cp.quantile(valid_data, 0.99))
        valid_data = cp.clip(valid_data, None, q99)
        
        # Use process_histogram from flowmop_utils_accelerated
        hist_result = process_histogram(valid_data, cp.int32(5), cp.int32(100))
        
        if hist_result is None:
            return cp.array([]), cp.array([]), cp.array([])
            
        return hist_result

    def _get_threshold_from_histogram(self, ratio: cp.ndarray, hist_result: Tuple) -> cp.float32:
        """Get threshold from histogram analysis."""
        if len(hist_result) == 0 or hist_result[0] is None:
            return self._calculate_mad_threshold(ratio)

        smoothed_hist, bin_edges, peak_indices, peak_densities = hist_result
        
        if len(peak_indices) >= 2:
            # Find the main peak
            peak_idx = cp.int32(cp.argmax(peak_densities))
            
            # Get peak positions for comparison
            peak_positions = bin_edges[peak_indices]
            
            # Find valley between peaks
            from .gpu_funcs.flowmop_utils_accelerated_gpu import find_left_minimum
            valley_idx = find_left_minimum(smoothed_hist, peak_indices[peak_idx], cp.int32(5))
            threshold = cp.float32(bin_edges[valley_idx])
            
            return threshold
        
        return self._calculate_mad_threshold(ratio)

    def _calculate_mad_threshold(self, ratio: cp.ndarray) -> cp.float32:
        """Calculate MAD-based threshold for ratio values."""
        median_ratio = cp.nanmedian(ratio)
        mad = cp.nanmedian(cp.abs(ratio - median_ratio))
        return cp.float32(median_ratio + self.fallback_mad_threshold * mad)

    def _fallback_to_mad(self, data: cp.ndarray, fsc_ratio: cp.ndarray, ssc_ratio: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        """Fallback to MAD-based thresholding when inflection point method fails."""
        warnings.warn("Falling back to MAD-based thresholding")
        fsc_threshold = self._calculate_mad_threshold(fsc_ratio)
        ssc_threshold = self._calculate_mad_threshold(ssc_ratio)
        
        doublet_vector = ((fsc_ratio <= fsc_threshold) & 
                         (ssc_ratio <= ssc_threshold)).astype(cp.int32)
        filtered_data = data[doublet_vector == 1]
        
        return filtered_data, doublet_vector

    def get_debug_info(self) -> dict:
        """Get debugging information from the last gate operation."""
        return self._debug_info

