"""
GPU-accelerated doublet gating implementation using DASK arrays.
"""

import numpy as np
import warnings
from typing import Union, Tuple, List
import dask.array as da

# Try importing required acceleration libraries
try:
    import cupy as cp
    from cupyx.scipy import signal as cusignal
    HAS_ACCELERATION = True
except ImportError:
    HAS_ACCELERATION = False
    warnings.warn("GPU acceleration libraries not available. Falling back to CPU implementation.")

from .doublet_gating import DoubletGateStrategy

# Type aliases
ArrayType = Union[np.ndarray, da.Array]

class DaskGPUMADDoubletGate(DoubletGateStrategy):
    """MAD-based doublet gating implementation using GPU-accelerated DASK arrays."""
    
    def __init__(self, mad_threshold: float = 5):
        """
        Initialize the gating strategy.
        
        Args:
            mad_threshold: Threshold multiplier for MAD-based gating
        """
        self.mad_threshold = mad_threshold
        # Import standard implementation for fallback
        from .doublet_gating import MADDoubletGate
        self._fallback_gate = MADDoubletGate(mad_threshold=mad_threshold)

    def gate(self, data: ArrayType, marker_names: List[str]) -> Tuple[ArrayType, ArrayType]:
        """
        Apply doublet gating using GPU-accelerated DASK arrays.
        
        Args:
            data: Input flow cytometry data
            marker_names: List of marker names corresponding to columns
            
        Returns:
            Tuple of (filtered_data, doublet_vector)
        """
        # Fall back to standard implementation if acceleration not available
        if not HAS_ACCELERATION or not isinstance(data, da.Array):
            warnings.warn("GPU acceleration unavailable or input not DASK array. Using CPU implementation.")
            return self._fallback_gate.gate(data, marker_names)

        if not self._check_required_parameters(marker_names):
            return data, da.ones(data.shape[0], dtype=int)

        try:
            # Calculate ratios using GPU operations
            fsc_ratio, ssc_ratio = self._calculate_ratios_gpu(data, marker_names)
            
            # Calculate thresholds
            fsc_threshold = self._calculate_mad_threshold_gpu(fsc_ratio)
            ssc_threshold = self._calculate_mad_threshold_gpu(ssc_ratio)
            
            # Apply thresholds
            doublet_vector = self._apply_thresholds_gpu(fsc_ratio, ssc_ratio,
                                                      fsc_threshold, ssc_threshold)
            
            # Filter data
            filtered_data = data[doublet_vector]
            
            return filtered_data, doublet_vector
            
        except Exception as e:
            warnings.warn(f"GPU acceleration failed: {str(e)}. Falling back to CPU implementation.")
            return self._fallback_gate.gate(data, marker_names)

    def _calculate_ratios_gpu(self, data: da.Array, marker_names: List[str]) -> Tuple[da.Array, da.Array]:
        """Calculate FSC and SSC ratios using GPU operations."""
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        
        try:
            fsc_a_idx = standardized_names.index('fsca')
            fsc_h_idx = standardized_names.index('fsch')
            ssc_a_idx = standardized_names.index('ssca')
            ssc_h_idx = standardized_names.index('ssch')
        except ValueError:
            raise ValueError("Required scatter parameters not found.")

        def compute_ratios(block):
            gpu_block = cp.asarray(block)
            # Calculate ratios with handling for division by zero
            fsc_ratio = cp.divide(gpu_block[:, fsc_a_idx], gpu_block[:, fsc_h_idx],
                                out=cp.full_like(gpu_block[:, fsc_a_idx], cp.nan),
                                where=gpu_block[:, fsc_h_idx] > 0)
            ssc_ratio = cp.divide(gpu_block[:, ssc_a_idx], gpu_block[:, ssc_h_idx],
                                out=cp.full_like(gpu_block[:, ssc_a_idx], cp.nan),
                                where=gpu_block[:, ssc_h_idx] > 0)
            return cp.stack([fsc_ratio, ssc_ratio], axis=1)

        ratios = data.map_blocks(compute_ratios,
                               dtype=data.dtype,
                               chunks=(data.chunks[0], 2))
        
        return ratios[:, 0], ratios[:, 1]

    def _calculate_mad_threshold_gpu(self, ratio: da.Array) -> float:
        """Calculate MAD threshold using GPU operations."""
        def mad_compute(block):
            gpu_block = cp.asarray(block)
            median = cp.nanmedian(gpu_block)
            mad = cp.nanmedian(cp.abs(gpu_block - median))
            return cp.array([median + (self.mad_threshold * mad)])
            
        result = ratio.map_blocks(mad_compute,
                                dtype=ratio.dtype,
                                chunks=(1,)).compute()
        return float(result[0])

    def _apply_thresholds_gpu(self, fsc_ratio: da.Array, ssc_ratio: da.Array,
                            fsc_threshold: float, ssc_threshold: float) -> da.Array:
        """Apply thresholds using GPU operations."""
        def threshold_compute(fsc_block, ssc_block):
            gpu_fsc = cp.asarray(fsc_block)
            gpu_ssc = cp.asarray(ssc_block)
            return ((gpu_fsc <= fsc_threshold) &
                   (gpu_ssc <= ssc_threshold)).astype(cp.int32)
                   
        return da.map_blocks(threshold_compute,
                           fsc_ratio, ssc_ratio,
                           dtype=np.int32)


class DaskGPUInflectionDoubletGate(DoubletGateStrategy):
    """
    GPU-accelerated inflection point-based doublet gating using DASK arrays.
    Falls back to CPU implementation if requirements not met.
    """
    
    def __init__(self, bins='auto', smoothing_factor=0.5, fallback_mad_threshold=5):
        self.bins = bins
        self.smoothing_factor = smoothing_factor
        self.fallback_mad_threshold = fallback_mad_threshold
        self._debug_info = {}
        # Import standard implementation for fallback
        from .doublet_gating import InflectionDoubletGate
        self._fallback_gate = InflectionDoubletGate(
            bins=bins,
            smoothing_factor=smoothing_factor,
            fallback_mad_threshold=fallback_mad_threshold
        )

    def gate(self, data: ArrayType, marker_names: List[str]) -> Tuple[ArrayType, ArrayType]:
        """
        Apply inflection point-based doublet gating using GPU acceleration.
        """
        # Fall back to standard implementation if acceleration not available
        if not HAS_ACCELERATION or not isinstance(data, da.Array):
            warnings.warn("GPU acceleration unavailable or input not DASK array. Using CPU implementation.")
            return self._fallback_gate.gate(data, marker_names)

        if not self._check_required_parameters(marker_names):
            return data, da.ones(data.shape[0], dtype=int)

        try:
            # Calculate ratios using GPU operations
            fsc_ratio, ssc_ratio = self._calculate_ratios_gpu(data, marker_names)
            self._debug_info['ratios'] = {'fsc': fsc_ratio, 'ssc': ssc_ratio}

            # Generate histograms and find inflection points
            fsc_hist = self._generate_smooth_histogram_gpu(fsc_ratio)
            ssc_hist = self._generate_smooth_histogram_gpu(ssc_ratio)
            self._debug_info['histograms'] = {'fsc': fsc_hist, 'ssc': ssc_hist}

            # Find inflection points
            fsc_inflections = self._find_inflection_points_gpu(fsc_hist[2])
            ssc_inflections = self._find_inflection_points_gpu(ssc_hist[2])
            self._debug_info['inflections'] = {'fsc': fsc_inflections, 'ssc': ssc_inflections}

            # Get thresholds
            fsc_threshold = self._get_threshold_from_inflection_gpu(fsc_ratio, fsc_hist, fsc_inflections)
            ssc_threshold = self._get_threshold_from_inflection_gpu(ssc_ratio, ssc_hist, ssc_inflections)
            self._debug_info['thresholds'] = {'fsc': fsc_threshold, 'ssc': ssc_threshold}

            # Apply thresholds
            doublet_vector = self._apply_thresholds_gpu(fsc_ratio, ssc_ratio,
                                                      fsc_threshold, ssc_threshold)
            
            # Filter data
            filtered_data = data[doublet_vector]
            
            return filtered_data, doublet_vector

        except Exception as e:
            warnings.warn(f"GPU acceleration failed: {str(e)}. Falling back to CPU implementation.")
            return self._fallback_gate.gate(data, marker_names)

    def _calculate_ratios_gpu(self, data: da.Array, marker_names: List[str]) -> Tuple[da.Array, da.Array]:
        """Calculate FSC and SSC ratios using GPU operations."""
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        
        try:
            fsc_a_idx = standardized_names.index('fsca')
            fsc_h_idx = standardized_names.index('fsch')
            ssc_a_idx = standardized_names.index('ssca')
            ssc_h_idx = standardized_names.index('ssch')
        except ValueError:
            raise ValueError("Required scatter parameters not found.")

        def compute_ratios(block):
            gpu_block = cp.asarray(block)
            # Calculate ratios with handling for division by zero
            fsc_ratio = cp.divide(gpu_block[:, fsc_a_idx], gpu_block[:, fsc_h_idx],
                                out=cp.full_like(gpu_block[:, fsc_a_idx], cp.nan),
                                where=gpu_block[:, fsc_h_idx] > 0)
            ssc_ratio = cp.divide(gpu_block[:, ssc_a_idx], gpu_block[:, ssc_h_idx],
                                out=cp.full_like(gpu_block[:, ssc_a_idx], cp.nan),
                                where=gpu_block[:, ssc_h_idx] > 0)
            return cp.stack([fsc_ratio, ssc_ratio], axis=1)

        ratios = data.map_blocks(compute_ratios,
                               dtype=data.dtype,
                               chunks=(data.chunks[0], 2))
        
        return ratios[:, 0], ratios[:, 1]

    def _generate_smooth_histogram_gpu(self, data: da.Array) -> Tuple[da.Array, da.Array, da.Array]:
        """Generate smoothed histogram using GPU-accelerated operations."""
        def histogram_compute(block):
            gpu_block = cp.asarray(block)
            valid_data = gpu_block[~cp.isnan(gpu_block)]
            
            # Generate histogram
            counts, bin_edges = cp.histogram(valid_data, bins=self.bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Apply Gaussian smoothing
            kernel_size = max(3, int(len(counts) * self.smoothing_factor))
            if kernel_size % 2 == 0:
                kernel_size += 1
            gaussian_kernel = cp.exp(-cp.linspace(-2, 2, kernel_size)**2)
            gaussian_kernel /= gaussian_kernel.sum()
            
            # Use CuPy's convolution
            smoothed_counts = cusignal.convolve(counts, gaussian_kernel, mode='same')
            
            return cp.stack([counts, bin_edges[:-1], smoothed_counts])

        result = data.map_blocks(histogram_compute,
                               dtype=data.dtype,
                               chunks=(3, -1))
        return result[0], result[1], result[2]

    def _find_inflection_points_gpu(self, curve: da.Array) -> da.Array:
        """Find inflection points using GPU-accelerated operations."""
        def inflection_compute(block):
            gpu_block = cp.asarray(block)
            
            # Apply Savitzky-Golay filter using CuPy's implementation
            window = 7
            polyorder = 3
            smoothed = cusignal.savgol_filter(gpu_block, window, polyorder)
            
            # Calculate second derivative
            second_derivative = cp.gradient(cp.gradient(smoothed))
            
            # Find zero crossings
            zero_crossings = cp.where(cp.diff(cp.signbit(second_derivative)))[0]
            
            # Filter weak inflection points
            strength_threshold = cp.max(cp.abs(second_derivative)) * 0.1
            strong_points = zero_crossings[cp.abs(second_derivative[zero_crossings]) > strength_threshold]
            
            return strong_points

        return data.map_blocks(inflection_compute,
                             dtype=np.int64,
                             chunks=(-1,))

    def _get_threshold_from_inflection_gpu(self, ratio: da.Array,
                                         hist: Tuple[da.Array, da.Array, da.Array],
                                         inflections: da.Array) -> float:
        """Get threshold value from inflection points using GPU operations."""
        def threshold_compute(ratio_block, hist_blocks, inflection_block):
            if len(inflection_block) == 0:
                return self._calculate_mad_threshold_gpu(ratio_block)

            counts, bin_edges, smoothed_counts = hist_blocks
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
            
            # Find the main peak
            peak_idx = cp.argmax(smoothed_counts)
            
            # Get inflection points after the peak
            valid_points = inflection_block[inflection_block > peak_idx]
            
            if len(valid_points) > 0:
                threshold_idx = valid_points[0]
                return bin_centers[threshold_idx]
            
            return self._calculate_mad_threshold_gpu(ratio_block)

        result = da.map_blocks(threshold_compute,
                             ratio, hist, inflections,
                             dtype=ratio.dtype,
                             chunks=(1,))
        return float(result.compute()[0])

    def _calculate_mad_threshold_gpu(self, ratio: da.Array) -> float:
        """Calculate MAD threshold using GPU operations."""
        def mad_compute(block):
            gpu_block = cp.asarray(block)
            median = cp.nanmedian(gpu_block)
            mad = cp.nanmedian(cp.abs(gpu_block - median))
            return cp.array([median + (self.fallback_mad_threshold * mad)])

        result = ratio.map_blocks(mad_compute,
                                dtype=ratio.dtype,
                                chunks=(1,)).compute()
        return float(result[0])

    def _apply_thresholds_gpu(self, fsc_ratio: da.Array, ssc_ratio: da.Array,
                            fsc_threshold: float, ssc_threshold: float) -> da.Array:
        """Apply thresholds using GPU operations."""
        def threshold_compute(fsc_block, ssc_block):
            gpu_fsc = cp.asarray(fsc_block)
            gpu_ssc = cp.asarray(ssc_block)
            return ((gpu_fsc <= fsc_threshold) &
                   (gpu_ssc <= ssc_threshold)).astype(cp.int32)
                   
        return da.map_blocks(threshold_compute,
                           fsc_ratio, ssc_ratio,
                           dtype=np.int32)

    def get_debug_info(self) -> dict:
        """Get debugging information from the last gate operation."""
        return self._debug_info

