"""
GPU-accelerated time gating implementation using DASK arrays.
"""

import warnings
from typing import Union, Tuple, List, Dict
import dask.array as da
import cupy as cp
import dask
import cProfile
import pstats
import io
from cpu.time_gating import TimeGateStrategy
from .gpu_funcs.time_gating_accelerated_gpu import (
    TimeGateAnalysisResult, find_events_per_bin, make_time_breaks,
    analyze_channel
)
from .gpu_funcs.flowmop_utils_accelerated_gpu import is_excluded_marker

class DaskGPUMADTimeGate(TimeGateStrategy):
    """MAD-based time gating implementation using GPU-accelerated DASK arrays."""
    
    def __init__(self, remove_zeros=True, min_cells=150, max_bins=500, step=200, mad_threshold=6,
                 peak_removal=1/3, min_nr_bins_peakdetection=5, histogram_smoothing=0.1, mad_smoothing=[0.1, 1.0], mad_method='all'):
        """Initialize the gating strategy."""
        self.remove_zeros = cp.array(remove_zeros, dtype=bool)
        self.min_cells = cp.array(min_cells, dtype=cp.int32)
        self.max_bins = cp.array(max_bins, dtype=cp.int32)
        self.step = cp.array(step, dtype=cp.int32)
        self.mad_threshold = cp.array(mad_threshold, dtype=cp.float32)
        self.peak_removal = cp.array(peak_removal, dtype=cp.float32)
        self.min_nr_bins_peakdetection = cp.array(min_nr_bins_peakdetection, dtype=cp.int32)
        self.histogram_smoothing = cp.array(histogram_smoothing, dtype=cp.float32)
        self.mad_smoothing = cp.array(mad_smoothing, dtype=cp.float32)
        self.mad_method = mad_method
        self._debug_info = {}
        self.profiler = cProfile.Profile()

    def gate(self, data: cp.ndarray, time_channel_index: int, marker_names: list) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Apply MAD-based time gating using pure CuPy arrays.
        
        Args:
            data: Flow cytometry data array (CuPy array)
            time_channel_index: Index of the time channel
            marker_names: List of marker names corresponding to each channel
            
        Returns:
            tuple: (filtered_data, time_gate_vector)
        """
        # Start profiling
        self.profiler.enable()

        # Time the data transfer to GPU
        cp.cuda.get_current_stream().synchronize()
        start_transfer = cp.cuda.Event()
        end_transfer = cp.cuda.Event()
        start_transfer.record()
        data = cp.asarray(data.compute())
        end_transfer.record()
        end_transfer.synchronize()
        transfer_time = cp.cuda.get_elapsed_time(start_transfer, end_transfer)
        print(f"Data transfer to GPU took: {transfer_time:.2f} ms")

        # Time events_per_bin calculation
        start_events = cp.cuda.Event()
        end_events = cp.cuda.Event()
        start_events.record()
        events_per_bin = find_events_per_bin(
            data,
            remove_zeros=self.remove_zeros,
            min_cells=self.min_cells,
            max_bins=self.max_bins,
            step=self.step
        )
        end_events.record()
        end_events.synchronize()
        events_time = cp.cuda.get_elapsed_time(start_events, end_events)
        print(f"Events per bin calculation took: {events_time:.2f} ms")

        # Time breaks creation
        start_breaks = cp.cuda.Event()
        end_breaks = cp.cuda.Event()
        start_breaks.record()
        breaks = make_time_breaks(events_per_bin, len(data))
        end_breaks.record()
        end_breaks.synchronize()
        breaks_time = cp.cuda.get_elapsed_time(start_breaks, end_breaks)
        print(f"Time breaks creation took: {breaks_time:.2f} ms")

        # Get fluorescence channels
        fluoro_channels = [i for i, name in enumerate(marker_names) 
                         if i != time_channel_index and 
                         not is_excluded_marker(name)]

        # Time channel processing
        start_channels = cp.cuda.Event()
        end_channels = cp.cuda.Event()
        start_channels.record()
        
        channel_results = {}
        for channel in fluoro_channels:
            channel_data = data[:, channel]
            result = analyze_channel(
                channel_data,
                breaks=breaks,
                remove_zeros=self.remove_zeros,
                smoothing=self.histogram_smoothing,
                mad_smoothing=self.mad_smoothing,
                mad_threshold=self.mad_threshold,
                peak_removal=self.peak_removal,
                min_nr_bins_peakdetection=self.min_nr_bins_peakdetection
            )
            
            if result.is_valid:
                channel_results[channel] = result

        end_channels.record()
        end_channels.synchronize()
        channels_time = cp.cuda.get_elapsed_time(start_channels, end_channels)
        print(f"Channel processing took: {channels_time:.2f} ms")

        # Time final processing
        start_final = cp.cuda.Event()
        end_final = cp.cuda.Event()
        start_final.record()

        rejection_count = cp.zeros(data.shape[0], dtype=cp.int32)

        for result in channel_results.values():
            if result.mad_results is not None:
                if self.mad_method == 'short':
                    rejected = ~result.mad_results['short_results']['cells']
                elif self.mad_method == 'long':
                    rejected = ~result.mad_results['long_results']['cells'] 
                else:  # 'all' - default
                    rejected = ~(result.mad_results['short_results']['cells'] & 
                               result.mad_results['long_results']['cells'])
                
                rejection_count = cp.where(rejected, rejection_count + 1, rejection_count)

        time_gate_vector = rejection_count < 2
        filtered_data = data[time_gate_vector]

        end_final.record()
        end_final.synchronize()
        final_time = cp.cuda.get_elapsed_time(start_final, end_final)
        print(f"Final processing took: {final_time:.2f} ms")

        # Stop profiling and print results
        self.profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

        print(f"Total GPU time: {transfer_time + events_time + breaks_time + channels_time + final_time:.2f} ms")
        
        return filtered_data, time_gate_vector

    def get_debug_info(self) -> dict:
        """Get debugging information from the last gate operation."""
        return self._debug_info
