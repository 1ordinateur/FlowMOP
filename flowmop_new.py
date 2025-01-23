"""
FlowMOP: Flow Cytometry Multi-Operation Pipeline
Main integration module that coordinates the gating operations.
"""

import warnings
from typing import Optional, Tuple, Dict, Literal, Union, Any
import numpy as np

# Runtime array type
ArrayType = np.ndarray
DEFAULT_ARRAY_MODULE = np

# Try importing GPU acceleration libraries
try:
    import cupy as cp
    import dask.array as da
    HAS_GPU = True
    ArrayType = da.Array
    DEFAULT_ARRAY_MODULE = da
except ImportError:
    HAS_GPU = False
    warnings.warn("GPU acceleration not available. Using CPU implementation.")

# Type hint that works with type checkers
StaticArrayType = Union[np.ndarray, 'da.Array']

from time_gating import TimeGateStrategy, MADTimeGate
from debris_gating import DebrisGateStrategy, FSCDebrisGate
from doublet_gating import DoubletGateStrategy, MADDoubletGate, InflectionDoubletGate

# Only import GPU-accelerated classes if GPU is available
if HAS_GPU:
    from debris_gating_accelerated import DaskGPUFSCDebrisGate
    from doublet_gating_accelerated import DaskGPUMADDoubletGate, DaskGPUInflectionDoubletGate
    from time_gating_accelerated import DaskGPUMADTimeGate

class FlowMOP:
    """Main class for the Flow Cytometry Multi-Operation Pipeline."""
    
    def __init__(self,
                 remove_zeros=True,
                 min_cells=150,
                 max_bins=500,
                 step=200,
                 MAD=6,
                 mad=5,
                 min_peaks=2,
                 max_peaks=5,
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
        
        # Common parameters for all gating strategies
        time_params = {
            'remove_zeros': remove_zeros,
            'min_cells': min_cells,
            'max_bins': max_bins,
            'step': step,
            'mad_threshold': MAD,
            'peak_removal': 1/3,
            'min_nr_bins_peakdetection': 5
        }
        
        debris_params = {
            'min_peaks': min_peaks,
            'max_peaks': max_peaks,
            'smoothing_window': smoothing_window,
            'percentage_cells_present': percentage_cells_present,
            'num_bins': 100
        }
        
        # Configure gating strategies based on GPU availability
        self.time_gate = (DaskGPUMADTimeGate(**time_params) if self.use_gpu 
                         else MADTimeGate(**time_params))
        
        self.debris_gate = (DaskGPUFSCDebrisGate(**debris_params) if self.use_gpu 
                           else FSCDebrisGate(**debris_params))
        
        # Configure doublet gating
        doublet_config = doublet_config or {}
        if doublet_method == 'inflection':
            config = {
                'bins': doublet_config.get('bins', 'auto'),
                'smoothing_factor': doublet_config.get('smoothing_factor', 0.5),
                'fallback_mad_threshold': doublet_config.get('fallback_mad_threshold', 5)
            }
            self.doublet_gate = (DaskGPUInflectionDoubletGate(**config) if self.use_gpu 
                                else InflectionDoubletGate(**config))
        else:
            self.doublet_gate = (DaskGPUMADDoubletGate(mad_threshold=mad) if self.use_gpu 
                                else MADDoubletGate(mad_threshold=mad))
            
        self.time_channel_index = time_channel_index
        self.skip_doublet_removal = False
        self.skip_debris_removal = False
        self.standardized_names = None
        self._debug_info = {}

    def _create_array(self, shape: tuple, dtype: Any = np.int32, chunks: Optional[tuple] = None) -> StaticArrayType:
        """Create an array with the appropriate type based on GPU availability."""
        if self.use_gpu and HAS_GPU:
            return DEFAULT_ARRAY_MODULE.ones(shape, chunks=chunks, dtype=cp.int32 if self.use_gpu else dtype)
        return np.ones(shape, dtype=dtype)

    def _is_gpu_array(self, arr: StaticArrayType) -> bool:
        """Check if array is a GPU-backed array."""
        return HAS_GPU and isinstance(arr, ArrayType)

    def _persist_if_gpu(self, arr: StaticArrayType) -> StaticArrayType:
        """Persist array if it's a GPU array."""
        return arr.persist() if self._is_gpu_array(arr) else arr

    def _prepare_gpu_array(self, data: np.ndarray) -> StaticArrayType:
        """Convert numpy array to GPU-backed DASK array if GPU is available."""
        if not self.use_gpu or not HAS_GPU:
            return data
        if not self._is_gpu_array(data):
            return da.from_array(data, chunks=(self.chunk_size, -1)).map_blocks(cp.asarray)
        return data

    def _finalize_array(self, data: StaticArrayType) -> np.ndarray:
        """Convert GPU-backed DASK array back to numpy array if needed."""
        if not self.use_gpu or not HAS_GPU:
            return data
        if self._is_gpu_array(data):
            return data.map_blocks(lambda x: cp.asnumpy(x) if isinstance(x, cp.ndarray) else x).compute()
        return data

    def process_fcs_data(self, marker_names: list[str], fcs_array: StaticArrayType) -> Dict[str, StaticArrayType]:
        """
        Process FCS data through the gating pipeline.
        
        Args:
            marker_names: List of marker names
            fcs_array: Array containing FCS data (numpy array or dask array)
            
        Returns:
            Dict[str, StaticArrayType]: Dictionary of filter vectors for each gating step
        """
        if len(marker_names) != fcs_array.shape[1]:
            raise ValueError("The number of marker names does not match the number of dimensions in the FCS array.")

        # Initialize processing
        if self.use_gpu:
            fcs_array = self._prepare_gpu_array(fcs_array)
        _, lod_vector = self.remove_limit_of_detection_events(fcs_array, marker_names)

        # Initialize vectors
        ones = self._create_array(fcs_array.shape[0], chunks=fcs_array.chunks[0] if self._is_gpu_array(fcs_array) else None)
        vectors = {'lod': lod_vector, 'debris': ones, 'time': ones, 'doublet': ones}
        print("vectors['lod']", vectors['lod'])
        # Apply gating strategies
        if not self.skip_debris_removal:
            fcs_array_device = self._prepare_gpu_array(fcs_array) if self.use_gpu else fcs_array
            _, _, vectors['debris'] = self.debris_gate.gate(fcs_array_device, marker_names)
            vectors['debris'] = self._persist_if_gpu(vectors['debris'])
        print("vectors['debris']", vectors['debris'])
        if self.time_channel_index is not None:
            fcs_array_device = self._prepare_gpu_array(fcs_array) if self.use_gpu else fcs_array
            _, vectors['time'] = self.time_gate.gate(fcs_array_device, self.time_channel_index)
            if self._is_gpu_array(vectors['time']):
                vectors['time'] = self._persist_if_gpu(vectors['time'])
        print("vectors['time']", vectors['time'])
        if not self.skip_doublet_removal:
            fcs_array_device = self._prepare_gpu_array(fcs_array) if self.use_gpu else fcs_array
            _, vectors['doublet'] = self.doublet_gate.gate(fcs_array_device, marker_names)
            vectors['doublet'] = self._persist_if_gpu(vectors['doublet'])
        print("vectors['doublet']", vectors['doublet'])

        # Calculate final vector
        if self._is_gpu_array(fcs_array):
            # Perform logical AND reduction explicitly
            final_vector = vectors['lod']
            for key in ['debris', 'time', 'doublet']:
                final_vector = final_vector & vectors[key]
            vectors['final'] = final_vector
            vectors['final'] = self._persist_if_gpu(vectors['final'])
        else:
            # Perform logical AND reduction explicitly
            final_vector = vectors['lod']
            for key in ['debris', 'time', 'doublet']:
                final_vector = final_vector & vectors[key]
            vectors['final'] = final_vector

        # Finalize results
        if not self.use_gpu:
            vectors = {k: self._finalize_array(v) for k, v in vectors.items()}
            
        return vectors

    def remove_limit_of_detection_events(self, fcs_array: StaticArrayType, marker_names: list[str]) -> Tuple[StaticArrayType, StaticArrayType]:
        """Remove events at the limit of detection."""
        try:
            fsca_column = self._get_marker_index(marker_names, 'fsca')
        except ValueError:
            warnings.warn("No FSC-A parameter found. Skipping limit of detection removal.")
            if HAS_GPU and isinstance(fcs_array, ArrayType):
                return fcs_array, DEFAULT_ARRAY_MODULE.ones(fcs_array.shape[0], chunks=fcs_array.chunks[0],
                                        dtype=cp.int32 if self.use_gpu else np.int32)
            return fcs_array, np.ones(fcs_array.shape[0], dtype=np.int32)

        if HAS_GPU and isinstance(fcs_array, ArrayType):
            fsca_max = DEFAULT_ARRAY_MODULE.max(fcs_array[:, fsca_column]).compute()
            max_events = DEFAULT_ARRAY_MODULE.sum(fcs_array[:, fsca_column] == fsca_max).compute()
            total_events = fcs_array.shape[0]
        else:
            fsca_max = np.max(fcs_array[:, fsca_column])
            max_events = np.sum(fcs_array[:, fsca_column] == fsca_max)
            total_events = fcs_array.shape[0]

        if max_events / total_events > 0.01:
            if HAS_GPU and isinstance(fcs_array, ArrayType):
                lod_vector = (fcs_array[:, fsca_column] < fsca_max).astype(cp.int32 if self.use_gpu else np.int32).persist()
                filtered_array = fcs_array[lod_vector].persist()
            else:
                lod_vector = (fcs_array[:, fsca_column] < fsca_max).astype(np.int32)
                filtered_array = fcs_array[lod_vector == 1]
            print(f"Removed {max_events} events ({max_events/total_events:.2%}) at the limit of detection.")
        else:
            if HAS_GPU and isinstance(fcs_array, ArrayType):
                lod_vector = DEFAULT_ARRAY_MODULE.ones(total_events, chunks=fcs_array.chunks[0],
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

    def export_to_csv(self, output_file: str, fcs_array: StaticArrayType, 
                     marker_names: list[str], vectors: Dict[str, StaticArrayType]) -> None:
        import pandas as pd
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

    def process_and_export(self, marker_names: list[str], fcs_array: StaticArrayType, 
                          output_file: str) -> Tuple[StaticArrayType, Dict[str, StaticArrayType]]:
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
