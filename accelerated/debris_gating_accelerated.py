"""
GPU-accelerated debris gating implementation using Dask and CuPy.
Handles Dask orchestration and delegates channel operations to debris_gating_accelerated_gpu.
"""

import warnings
from typing import Tuple, List, Optional, Dict
import dask.array as da
import cupy as cp
from dataclasses import dataclass

from cpu.debris_gating import DebrisGateStrategy
from .gpu_funcs.debris_gating_accelerated_gpu import (
    ChannelAnalysisResult, analyze_channel, get_fsc_thresholds
)
from .gpu_funcs.flowmop_utils_accelerated_gpu import is_excluded_marker

@dataclass
class DebrisGateResult:
    """Results from debris gating operation."""
    filtered_data: da.Array
    fsc_threshold: Optional[cp.float32]
    debris_vector: da.Array
    channel_results: Dict[int, ChannelAnalysisResult]

class DaskGPUFSCDebrisGate(DebrisGateStrategy):
    """FSC-based debris gating implementation using GPU acceleration."""
    
    def __init__(self, min_peaks: int = 2, max_peaks: int = 5, smoothing_window: int = 3, 
                 percentage_cells_present: float = 5, num_bins: int = 100):
        """Initialize the gating strategy."""
        self.min_peaks = cp.int32(min_peaks)
        self.max_peaks = cp.int32(max_peaks)
        self.smoothing_window = cp.int32(smoothing_window)
        self.percentage_cells_present = cp.float32(percentage_cells_present)
        self.num_bins = cp.int32(num_bins)
        self._debug_info = {}

    def gate(self, data: da.Array, marker_names: List[str]) -> Tuple[da.Array, da.Array]:
        """
        Apply debris gating using GPU acceleration.
        
        Args:
            data: Input flow cytometry data as a Dask array
            marker_names: List of marker names
            
        Returns:
            Tuple of (filtered_data, debris_vector)
        """
        # Check for events in bottom bin
        if not self._check_events_in_bottom_bin(data, marker_names):
            return data, da.ones(data.shape[0], dtype=cp.int32)

        # Process each channel in parallel
        channel_results = {}
        for i, marker in enumerate(marker_names):
            # Extract channel data
            channel_data = data[:, i]
            
            # Process channel using Dask's map_overlap for better chunking
            result = da.map_overlap(
                lambda x: analyze_channel(
                    x, marker,
                    min_peaks=self.min_peaks,
                    max_peaks=self.max_peaks,
                    smoothing_window=self.smoothing_window,
                    percentage_cells_present=self.percentage_cells_present,
                    num_bins=self.num_bins
                ),
                channel_data,
                depth=0,
                boundary='none',
                dtype=object
            ).compute()
            
            if result.is_valid:
                channel_results[i] = result

        # Get FSC thresholds from valid channels
        fsc_thresholds = self._get_fsc_thresholds(data, channel_results, marker_names)

        if not fsc_thresholds:
            return data, da.ones(data.shape[0], dtype=cp.int32)

        # Calculate final threshold and apply gating
        fsc_gate_threshold = cp.float32(cp.nanmedian(cp.asarray(fsc_thresholds)))
        fsc_column = self._get_fsc_column(marker_names)

        # Compute the gating vector and filtered data
        debris_vector = (data[:, fsc_column] >= fsc_gate_threshold).astype(cp.int32)
        filtered_data = data[debris_vector]

        return filtered_data, debris_vector

    def _check_events_in_bottom_bin(self, data: da.Array, marker_names: List[str]) -> bool:
        """Check if events exist in the bottom 10th bin of FSC-A and SSC-A."""
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        
        try:
            fsc_a_column = standardized_names.index('fsca')
            ssc_a_column = standardized_names.index('ssca')
        except ValueError:
            warnings.warn("Required FSC-A or SSC-A parameters not found.")
            return False

        def check_bottom_bin(chunk):
            chunk_gpu = cp.asarray(chunk)
            fsc_max = cp.float32(cp.max(chunk_gpu[:, fsc_a_column]))
            ssc_max = cp.float32(cp.max(chunk_gpu[:, ssc_a_column]))
            
            fsc_bottom = fsc_max * cp.float32(0.1)
            ssc_bottom = ssc_max * cp.float32(0.1)
            
            return cp.any(
                (chunk_gpu[:, fsc_a_column] <= fsc_bottom) & 
                (chunk_gpu[:, ssc_a_column] <= ssc_bottom)
            )

        # Apply check to each chunk
        result = da.map_blocks(check_bottom_bin, data, dtype=cp.bool_).any().compute()
        
        if not result:
            warnings.warn("No events found in the bottom 10th bin of FSC-A and SSC-A. Debris removal will be skipped.")
        
        return bool(result)

    def _get_fsc_column(self, marker_names: List[str]) -> int:
        """Get the index of the FSC-A column."""
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        try:
            return standardized_names.index('fsca')
        except ValueError:
            raise ValueError("FSC-A parameter not found in marker names.")

    def _get_fsc_thresholds(self, data: da.Array, channel_results: Dict[int, ChannelAnalysisResult],
                           marker_names: List[str]) -> List[cp.float32]:
        """Get FSC thresholds for features with valid peaks."""
        fsc_column = self._get_fsc_column(marker_names)
        fsc_data = data[:, fsc_column]
        
        thresholds = []
        for marker, result in channel_results.items():
            if result.is_valid and result.positive_mask is not None:
                threshold = da.map_overlap(
                    lambda x, mask: get_fsc_thresholds(
                        x, mask, self.smoothing_window
                    ),
                    fsc_data, result.positive_mask,
                    depth=0,
                    boundary='none',
                    dtype=cp.float32
                ).compute()
                
                if threshold is not None:
                    thresholds.append(threshold)

        return thresholds

    def _standardize_marker_name(self, name: str) -> str:
        """Standardize marker names by removing symbols and converting to lowercase."""
        import re
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

    def get_debug_info(self) -> dict:
        """Get debugging information from the last gate operation."""
        return self._debug_info


