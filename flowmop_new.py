"""
FlowMOP: Flow Cytometry Multi-Operation Pipeline
Main integration module that coordinates the gating operations.
"""

import warnings
from typing import Optional, Tuple, Dict, Literal, Union
import dask.array as da
import numpy as np

# Try importing GPU acceleration libraries
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    warnings.warn("GPU acceleration not available. Using CPU implementation.")

from .time_gating import TimeGateStrategy, MADTimeGate
from .debris_gating import DebrisGateStrategy, FSCDebrisGate
from .doublet_gating import DoubletGateStrategy, MADDoubletGate, InflectionDoubletGate
from .debris_gating_accelerated import DaskGPUFSCDebrisGate
from .doublet_gating_accelerated import DaskGPUMADDoubletGate, DaskGPUInflectionDoubletGate
from .time_gating_accelerated import DaskGPUMADTimeGate

# Type aliases
ArrayType = Union[np.ndarray, da.Array]

class FlowMOP:
    """Main class for the Flow Cytometry Multi-Operation Pipeline."""
    
    def __init__(self,
                 remove_zeros=True,
                 min_cells=150,
                 max_bins=500,
                 step=200,
                 MAD=6,
                 IT_limit=0.6,
                 peak_detection_smoothing=2,
                 spectral=False,
                 mad=5,
                 min_peaks=2,
                 max_peaks=3,
                 smoothing_window=3,
                 percentage_cells_present=3,
                 time_channel_index=None,
                 doublet_method: Literal['mad', 'inflection'] = 'mad',
                 doublet_config: Optional[dict] = None,
                 use_gpu: bool = True,
                 chunk_size: Optional[int] = None):
        """
        Initialize FlowMOP with configuration parameters.
        
        Args:
            remove_zeros: Remove zero-value events
            min_cells: Minimum number of cells required for gating
            max_bins: Maximum number of bins for histogram-based gating
            step: Step size for peak detection
            MAD: Median Absolute Deviation threshold for MAD-based gating
            IT_limit: Limit for IT-based gating
            peak_detection_smoothing: Smoothing factor for peak detection
            spectral: Whether to use spectral gating
            mad: MAD threshold for MAD-based doublet gating
            min_peaks: Minimum number of peaks for peak detection
            max_peaks: Maximum number of peaks for peak detection
            smoothing_window: Smoothing window for peak detection
            percentage_cells_present: Percentage of cells required for gating
            time_channel_index: Index of time channel
            doublet_method: Method for doublet gating ('mad' or 'inflection')
            doublet_config: Configuration for doublet gating method
                For inflection method:
                    - bins: Number of bins or method ('auto', 'fd', 'scott')
                    - smoothing_factor: KDE smoothing factor (0.1 to 1.0)
                    - fallback_mad_threshold: Fallback MAD threshold
            use_gpu: Whether to use GPU acceleration if available
            chunk_size: Size of chunks for DASK array operations
        """
        self.use_gpu = use_gpu and HAS_GPU
        self.chunk_size = chunk_size
        
        # Configure time gating
        if self.use_gpu:
            self.time_gate = DaskGPUMADTimeGate(
                remove_zeros=remove_zeros,
                min_cells=min_cells,
                max_bins=max_bins,
                step=step,
                mad_threshold=MAD,
                peak_removal=1/3,
                min_nr_bins_peakdetection=5
            )
        else:
            self.time_gate = MADTimeGate(
                remove_zeros=remove_zeros,
                min_cells=min_cells,
                max_bins=max_bins,
                step=step,
                mad_threshold=MAD,
                peak_removal=1/3,
                min_nr_bins_peakdetection=5
            )
        
        # Configure debris gating
        if self.use_gpu:
            self.debris_gate = DaskGPUFSCDebrisGate(
                min_peaks=min_peaks,
                max_peaks=max_peaks,
                smoothing_window=smoothing_window,
                percentage_cells_present=percentage_cells_present
            )
        else:
            self.debris_gate = FSCDebrisGate(
                min_peaks=min_peaks,
                max_peaks=max_peaks,
                smoothing_window=smoothing_window,
                percentage_cells_present=percentage_cells_present
            )
        
        # Configure doublet gating based on method
        if doublet_method == 'inflection':
            config = doublet_config or {}
            if self.use_gpu:
                self.doublet_gate = DaskGPUInflectionDoubletGate(
                    bins=config.get('bins', 'auto'),
                    smoothing_factor=config.get('smoothing_factor', 0.5),
                    fallback_mad_threshold=config.get('fallback_mad_threshold', 5)
                )
            else:
                self.doublet_gate = InflectionDoubletGate(
                    bins=config.get('bins', 'auto'),
                    smoothing_factor=config.get('smoothing_factor', 0.5),
                    fallback_mad_threshold=config.get('fallback_mad_threshold', 5)
                )
        else:  # 'mad'
            if self.use_gpu:
                self.doublet_gate = DaskGPUMADDoubletGate(mad_threshold=mad)
            else:
                self.doublet_gate = MADDoubletGate(mad_threshold=mad)
            
        self.time_channel_index = time_channel_index
        self.skip_doublet_removal = False
        self.skip_debris_removal = False
        self.standardized_names = None
        self._debug_info = {}

    def _prepare_gpu_array(self, data: np.ndarray) -> da.Array:
        """Convert numpy array to GPU-backed DASK array."""
        if not isinstance(data, da.Array):
            # Create DASK array with appropriate chunks
            dask_array = da.from_array(data, chunks=(self.chunk_size, -1))
            
            if self.use_gpu:
                # Move data to GPU using map_blocks
                def to_gpu(block):
                    return cp.asarray(block)
                return dask_array.map_blocks(to_gpu)
            return dask_array
        return data

    def _finalize_array(self, data: da.Array) -> np.ndarray:
        """Convert GPU-backed DASK array back to numpy array."""
        if isinstance(data, da.Array):
            if self.use_gpu:
                # Move data back to CPU if needed
                def to_cpu(block):
                    if isinstance(block, cp.ndarray):
                        return cp.asnumpy(block)
                    return block
                return data.map_blocks(to_cpu).compute()
            return data.compute()
        return data

    def process_fcs_data(self, marker_names: list[str], fcs_array: ArrayType) -> Tuple[ArrayType, Dict[str, ArrayType]]:
        """
        Process FCS data through the gating pipeline.
        
        Args:
            marker_names: List of marker names
            fcs_array: Array containing FCS data (numpy array or dask array)
            
        Returns:
            tuple: (filtered_data, filter_vectors)
        """
        if len(marker_names) != fcs_array.shape[1]:
            raise ValueError("The number of marker names does not match the number of dimensions in the FCS array.")

        # Convert input to GPU-backed DASK array if using GPU
        if self.use_gpu:
            fcs_array = self._prepare_gpu_array(fcs_array)

        # Remove events at the limit of detection
        filtered_data, lod_vector = self.remove_limit_of_detection_events(fcs_array, marker_names)
        
        # Initialize vectors with appropriate array type
        if isinstance(fcs_array, da.Array):
            ones = da.ones(fcs_array.shape[0], chunks=fcs_array.chunks[0], dtype=cp.int32 if self.use_gpu else np.int32)
        else:
            ones = np.ones(fcs_array.shape[0], dtype=np.int32)
            
        vectors = {
            'lod': lod_vector,
            'debris': ones,
            'time': ones,
            'doublet': ones
        }

        # Apply debris gating
        if filtered_data is not None and not self.skip_debris_removal:
            filtered_data, _, vectors['debris'] = self.debris_gate.gate(filtered_data, marker_names)
            if isinstance(vectors['debris'], da.Array):
                vectors['debris'] = vectors['debris'].persist()

        # Apply time gating
        if filtered_data is not None and self.time_channel_index is not None:
            # Ensure data is in GPU memory for time gating
            if self.use_gpu and not isinstance(filtered_data, da.Array):
                filtered_data = self._prepare_gpu_array(filtered_data)
            
            # Apply time gating
            filtered_data, vectors['time'] = self.time_gate.gate(filtered_data, self.time_channel_index)
            
            # Persist results if using DASK
            if isinstance(vectors['time'], da.Array):
                vectors['time'] = vectors['time'].persist()
                filtered_data = filtered_data.persist()

        # Apply doublet gating
        if filtered_data is not None and not self.skip_doublet_removal:
            filtered_data, vectors['doublet'] = self.doublet_gate.gate(filtered_data, marker_names)
            if isinstance(vectors['doublet'], da.Array):
                vectors['doublet'] = vectors['doublet'].persist()

        # Calculate final vector
        if isinstance(fcs_array, da.Array):
            vectors['final'] = da.logical_and.reduce([
                vectors['lod'],
                vectors['debris'],
                vectors['time'],
                vectors['doublet']
            ]).persist()
        else:
            vectors['final'] = (vectors['lod'] & vectors['debris'] & 
                              vectors['time'] & vectors['doublet'])

        # Convert results back to numpy if not using GPU
        if not self.use_gpu:
            filtered_data = self._finalize_array(filtered_data)
            vectors = {k: self._finalize_array(v) for k, v in vectors.items()}
            
        return filtered_data, vectors

    def remove_limit_of_detection_events(self, fcs_array: ArrayType, marker_names: list[str]) -> Tuple[ArrayType, ArrayType]:
        """Remove events at the limit of detection."""
        try:
            fsca_column = self._get_marker_index(marker_names, 'fsca')
        except ValueError:
            warnings.warn("No FSC-A parameter found. Skipping limit of detection removal.")
            if isinstance(fcs_array, da.Array):
                return fcs_array, da.ones(fcs_array.shape[0], chunks=fcs_array.chunks[0],
                                        dtype=cp.int32 if self.use_gpu else np.int32)
            return fcs_array, np.ones(fcs_array.shape[0], dtype=np.int32)

        if isinstance(fcs_array, da.Array):
            fsca_max = da.max(fcs_array[:, fsca_column]).compute()
            max_events = da.sum(fcs_array[:, fsca_column] == fsca_max).compute()
            total_events = fcs_array.shape[0]
        else:
            fsca_max = np.max(fcs_array[:, fsca_column])
            max_events = np.sum(fcs_array[:, fsca_column] == fsca_max)
            total_events = fcs_array.shape[0]

        if max_events / total_events > 0.01:
            if isinstance(fcs_array, da.Array):
                lod_vector = (fcs_array[:, fsca_column] < fsca_max).astype(cp.int32 if self.use_gpu else np.int32).persist()
                filtered_array = fcs_array[lod_vector].persist()
            else:
                lod_vector = (fcs_array[:, fsca_column] < fsca_max).astype(np.int32)
                filtered_array = fcs_array[lod_vector == 1]
            print(f"Removed {max_events} events ({max_events/total_events:.2%}) at the limit of detection.")
        else:
            if isinstance(fcs_array, da.Array):
                lod_vector = da.ones(total_events, chunks=fcs_array.chunks[0],
                                   dtype=cp.int32 if self.use_gpu else np.int32)
            else:
                lod_vector = np.ones(total_events, dtype=np.int32)
            filtered_array = fcs_array
            print("Threshold events below limit of 1%. Retaining events.")

        return filtered_array, lod_vector

    @staticmethod
    def _get_marker_index(marker_names: list[str], target: str) -> int:
        """Get the index of a marker by its standardized name."""
        standardized_names = [FlowMOP._standardize_marker_name(name) for name in marker_names]
        try:
            return standardized_names.index(target)
        except ValueError:
            raise ValueError(f"Marker {target} not found in marker names.")

    @staticmethod
    def _standardize_marker_name(name: str) -> str:
        """Standardize marker names by removing symbols and converting to lowercase."""
        import re
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

    def export_to_csv(self, output_file: str, fcs_array: ArrayType, 
                     marker_names: list[str], vectors: Dict[str, ArrayType]) -> None:
        """Export the processed data and filter vectors to CSV."""
        # Convert DASK/GPU arrays to numpy before creating DataFrame
        if isinstance(fcs_array, da.Array):
            fcs_array = self._finalize_array(fcs_array)
            vectors = {k: self._finalize_array(v) for k, v in vectors.items()}
            
        df = pd.DataFrame(fcs_array, columns=marker_names)
        
        for name, vector in vectors.items():
            df[f'passed_{name}'] = vector.astype(bool)

        df.to_csv(output_file, index=False)
        print(f"Data exported to {output_file}")

    def process_and_export(self, marker_names: list[str], fcs_array: ArrayType, 
                          output_file: str) -> Tuple[ArrayType, Dict[str, ArrayType]]:
        """Process the data and export results to CSV."""
        results = self.process_fcs_data(marker_names, fcs_array)
        self.export_to_csv(output_file, fcs_array, marker_names, results[1])
        return results

    def get_debug_info(self) -> dict:
        """Get debugging information from the last pipeline run."""
        debug_info = {
            'time_gate': getattr(self.time_gate, 'get_debug_info', lambda: {})(),
            'debris_gate': getattr(self.debris_gate, 'get_debug_info', lambda: {})(),
            'doublet_gate': getattr(self.doublet_gate, 'get_debug_info', lambda: {})()
        }
        debug_info.update(self._debug_info)
        return debug_info
