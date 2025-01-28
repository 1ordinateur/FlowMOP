"""
GPU-accelerated time gating implementation using DASK arrays.
"""

import warnings
from typing import Union, Tuple, List, Dict
import dask.array as da
import cupy as cp
from cupyx.scipy import signal as cusignal
from cupyx.scipy import ndimage as cundimage

from .time_gating import TimeGateStrategy

# Type aliases
ArrayType = Union[da.Array, cp.ndarray]

class DaskGPUMADTimeGate(TimeGateStrategy):
    """MAD-based time gating implementation using GPU-accelerated DASK arrays."""
    
    def __init__(self, remove_zeros=True, min_cells=150, max_bins=500, step=200, mad_threshold=6,
                 peak_removal=1/3, min_nr_bins_peakdetection=5):
        """Initialize the gating strategy."""
        self.remove_zeros = remove_zeros
        self.min_cells = min_cells
        self.max_bins = max_bins
        self.step = step
        self.mad_threshold = mad_threshold
        self.peak_removal = peak_removal
        self.min_nr_bins_peakdetection = min_nr_bins_peakdetection
        self._debug_info = {}

    def gate(self, data: da.Array, time_channel_index: int, marker_names: list) -> Tuple[da.Array, da.Array]:
        """
        Apply MAD-based time gating using GPU-accelerated DASK arrays.
        
        Args:
            data: Flow cytometry data array (DASK array)
            time_channel_index: Index of the time channel
            marker_names: List of marker names corresponding to each channel
            
        Returns:
            tuple: (filtered_data, time_gate_vector)
        """
        # Calculate optimal number of events per bin
        events_per_bin = self._find_events_per_bin(data)
        
        # Create time bins with overlap
        breaks = self._make_breaks(events_per_bin, data.shape[0])
        
        # Get all fluorescence channels (exclude time, FSC, SSC channels)
        fluoro_channels = [i for i, name in enumerate(marker_names) 
                         if i != time_channel_index and 
                         not any(x.lower() in name.lower() for x in ['fsc', 'ssc', 'time'])]
        
        # Detect peaks in all fluorescence channels
        peaks = self._determine_peaks_all_channels(data, fluoro_channels, breaks['breaks'])
        
        # Apply MAD-based outlier detection
        time_gate_vector = da.ones(data.shape[0], dtype=bool)
        for channel_peaks in peaks.values():
            mad_results = self._mad_excluder_gpu(channel_peaks, self.mad_threshold, 
                                               breaks['breaks'], data.shape[0])
            time_gate_vector &= mad_results['cells']
        
        # Create time gate vector and filter data
        filtered_data = data[time_gate_vector]
        
        return filtered_data, time_gate_vector

    def _find_events_per_bin(self, data: da.Array) -> int:
        """Calculate optimal number of events per bin using GPU operations."""
        def count_nonzeros(block):
            gpu_block = cp.asarray(block)
            return cp.sum(gpu_block != 0, axis=0)
            
        if self.remove_zeros:
            nonzero_counts = data.map_blocks(count_nonzeros, dtype=cp.int32)
            min_nonzero = da.min(nonzero_counts).compute()
            max_bins_mass = min_nonzero / self.min_cells
            if max_bins_mass < self.max_bins:
                self.max_bins = max_bins_mass
        
        nr_events = data.shape[0]
        max_cells = int(cp.ceil((nr_events / self.max_bins) * 2))
        max_cells = ((max_cells // self.step) * self.step) + self.step
        
        return max(self.min_cells, max_cells)

    def _make_breaks(self, events_per_bin: int, nr_events: int) -> dict:
        """Create time bins with overlap."""
        def split_with_overlap(block):
            gpu_block = cp.asarray(block)
            overlap = int(cp.ceil(events_per_bin / 2))
            step = events_per_bin - overlap
            starts = cp.arange(0, len(gpu_block) - events_per_bin + 1, step)
            return cp.array([cp.arange(start, start + events_per_bin) for start in starts])
            
        indices = cp.arange(nr_events)
        indices = da.from_array(indices, chunks=min(events_per_bin * 10, nr_events))
        breaks = indices.map_blocks(split_with_overlap, dtype=cp.int32)
        
        return {'breaks': breaks, 'events_per_bin': events_per_bin}

    def _determine_peaks_all_channels(self, data: da.Array, channels: List[int], 
                                    breaks: da.Array) -> Dict[int, da.Array]:
        """Detect peaks in all channels using GPU operations."""
        data_channels = data[:, channels].T
                
        peaks = {}
        for channel, channel_data in zip(channels, data_channels):
            channel_peaks = self.determine_peaks_for_channel(channel_data, breaks)
            if channel_peaks is not None:
                peaks[channel] = channel_peaks
        
        return peaks

    def determine_peaks_for_channel(self, channel_data, breaks):
        full_channel_peaks = self._timegate_peak_detection_gpu(channel_data)
        if da.all(da.isnan(full_channel_peaks)):
            return None
        
        def process_break(break_data, break_idx):
            gpu_data = cp.asarray(break_data)
            max_peak = cp.max(gpu_data)
            peaks = self._timegate_peak_detection_gpu(break_data, smoothing=2, max_peak=max_peak)
            return cp.stack([cp.full(len(peaks), break_idx), peaks])
        
        peaks_list = []
        for i, break_indices in enumerate(breaks):
            break_data = channel_data[break_indices]
            break_peaks = break_data.map_blocks(
                lambda x: process_break(x, i),
                dtype=cp.float32
            )
            peaks_list.append(break_peaks)
        
        peak_frame = da.concatenate(peaks_list)
        
        # Remove NaN peaks
        valid_mask = ~da.isnan(peak_frame[:, 1])
        peak_frame = peak_frame[valid_mask]
        
        # Find most occurring peaks
        most_occurring = self._find_most_occurring_peaks_gpu(peak_frame[:, 1])
        if most_occurring is None:
            most_occurring = da.median(peak_frame[:, 1])
        
        # Update peak frame with closest peaks
        return self._update_peak_frame_gpu(peak_frame, most_occurring)

    def _timegate_peak_detection_gpu(self, data: da.Array, smoothing: int = 2, 
                                   max_peak: float = None, window_size: int = 10) -> da.Array:
        """Detect peaks in time-gated data using GPU operations."""
        def detect_peaks(block):
            gpu_block = cp.asarray(block)
            
            if self.remove_zeros:
                gpu_block = gpu_block[gpu_block != 0]
            
            if len(gpu_block) < 3:
                return cp.array([])
            
            # Calculate histogram
            hist, bin_edges = cp.histogram(gpu_block, bins=100, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Smooth histogram
            smoothed_hist = cusignal.convolve(hist, cp.ones(smoothing) / smoothing, mode='same')
            
            # Find maxima using maximum filter
            max_filter = cundimage.maximum_filter1d(smoothed_hist, size=2 * window_size + 1)
            maxima = ((smoothed_hist == max_filter) & 
                     (smoothed_hist > cp.roll(smoothed_hist, 1)) & 
                     (smoothed_hist > cp.roll(smoothed_hist, -1)))
            maxima[:window_size] = maxima[-window_size:] = False
            
            if max_peak is None:
                max_peak = cp.max(hist)
            
            # Get peak indices and filter by height
            peak_indices = cp.where(maxima)[0]
            filtered_peak_indices = peak_indices[bin_centers[peak_indices] > self.peak_removal * max_peak]
            
            return bin_centers[filtered_peak_indices]
        
        return data.map_blocks(detect_peaks, dtype=cp.float32)

    def _find_most_occurring_peaks_gpu(self, peaks: da.Array, tolerance: float = 0.05) -> float:
        """Find most frequently occurring peaks using GPU operations."""
        def find_peaks(block):
            gpu_block = cp.asarray(block)
            if len(gpu_block) == 0:
                return None
                
            min_val, max_val = cp.min(gpu_block), cp.max(gpu_block)
            bin_edges = cp.arange(min_val, max_val + tolerance, tolerance)
            hist, _ = cp.histogram(gpu_block, bins=bin_edges)
            
            min_occurrences = (self.min_nr_bins_peakdetection / 100) * len(block)
            candidate_bins = cp.where(hist >= min_occurrences)[0]
            
            if len(candidate_bins) > 0:
                candidate_peaks = (bin_edges[candidate_bins] + bin_edges[candidate_bins + 1]) / 2
                return candidate_peaks[cp.argmax(hist[candidate_bins])]
            return None
            
        result = peaks.map_blocks(find_peaks, dtype=cp.float32)
        return float(result[0]) if result[0] is not None else None

    def _update_peak_frame_gpu(self, peak_frame: da.Array, most_occurring_peaks: float) -> da.Array:
        """Update peak frame with closest peaks using GPU operations."""
        def update_peaks(block):
            gpu_block = cp.asarray(block)
            if len(gpu_block) == 0:
                return cp.array([])
            
            closest_peak_indices = cp.argmin(cp.abs(gpu_block[:, 1] - most_occurring_peaks))
            return gpu_block[closest_peak_indices]
            
        return peak_frame.map_blocks(update_peaks, dtype=cp.float32)

    def _mad_excluder_gpu(self, peaks: da.Array, mad_threshold: float, breaks: da.Array, 
                         nr_cells: int) -> Dict:
        """Apply MAD-based outlier detection using GPU operations."""
        def process_peaks(block):
            gpu_block = cp.asarray(block)
            
            # Create spline interpolation using CuPy
            x = cp.arange(len(gpu_block))
            coeffs = cp.polyfit(x, gpu_block, deg=3)
            kernel_y = cp.polyval(coeffs, x)
            
            # Calculate median and MAD
            median_peak = cp.median(kernel_y)
            mad_peak = cp.median(cp.abs(kernel_y - median_peak))
            
            # Calculate intervals
            upper_interval = median_peak + mad_threshold * mad_peak
            lower_interval = median_peak - mad_threshold * mad_peak
            
            # Find bins to remove
            to_remove = (kernel_y > upper_interval) | (kernel_y < lower_interval)
            return to_remove
            
        peak_values = peaks[:, 1]
        to_remove_bins = peak_values.map_blocks(process_peaks, dtype=bool)
        
        def create_mask(block, remove_bins):
            gpu_block = cp.asarray(block)
            mask = cp.ones(len(gpu_block), dtype=cp.int32)
            cell_ids = cp.array([], dtype=cp.int32)
            
            for bin_idx in cp.where(remove_bins)[0]:
                bin_start = breaks[bin_idx][0]
                bin_end = breaks[bin_idx][-1]
                mask[bin_start:bin_end + 1] = 0
                cell_ids = cp.concatenate([cell_ids, cp.arange(bin_start, bin_end + 1)])
            
            return cp.stack([mask, cell_ids])
        
        # Create mask for cells
        cells_data = da.ones(nr_cells, chunks=peaks.chunks[0], dtype=cp.int32)
        result = cells_data.map_blocks(
            lambda x: create_mask(x, to_remove_bins),
            dtype=cp.int32
        )
        
        cells_mask = result[0]
        cell_ids = result[1]
        
        # Calculate contribution
        removed_count = nr_cells - da.sum(cells_mask)
        contribution_mad = float((removed_count / nr_cells) * 100)
        
        return {
            "cells": cells_mask,
            "cell_ids": cell_ids,
            "MAD_bins": to_remove_bins,
            "Contribution_MAD": contribution_mad
        }

    def get_debug_info(self) -> dict:
        """Get debugging information from the last gate operation."""
        return self._debug_info
