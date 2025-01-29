"""
FlowMOP: Flow Cytometry Multi-Operation Pipeline
Main integration module that coordinates the gating operations.
"""

import warnings
from typing import Optional, Tuple, Dict, Literal, Union, Any
import numpy as np
import cupy as cp

# Type definitions
import dask.array as da
import dask
ArrayType = Union[np.ndarray, da.Array]
DEFAULT_ARRAY_MODULE = np
HAS_GPU = cp.cuda.is_available()

# Import CPU implementations
from cpu.time_gating import TimeGateStrategy, MADTimeGate
from cpu.debris_gating import DebrisGateStrategy, FSCDebrisGate
from cpu.doublet_gating import DoubletGateStrategy, MADDoubletGate, InflectionDoubletGate

# Import GPU implementations if available
if HAS_GPU:
    from accelerated.debris_gating_accelerated import DaskGPUFSCDebrisGate
    from accelerated.doublet_gating_accelerated import CuPyMADDoubletGate, CuPyInflectionDoubletGate
    from accelerated.time_gating_accelerated import DaskGPUMADTimeGate
    DEFAULT_ARRAY_MODULE = da

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
                 smoothing_window=2,
                 mad_smoothings=[0.1, 1.0],
                 mad_method='all',
                 percentage_cells_present=3,
                 time_channel_index=None,
                 doublet_method: Literal['mad', 'inflection'] = 'inflection',
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
        self.chunk_size = chunk_size or 10000  # Default chunk size if none provided
        
        # Convert parameters to GPU types if using GPU
        if self.use_gpu:
            self.dtype = cp.float32
            self.int_dtype = cp.int32
            dask.config.set({"array.backend": "cupy"})

        else:
            self.dtype = np.float32
            self.int_dtype = np.int32

        # Common parameters for all gating strategies
        time_params = {
            'remove_zeros': remove_zeros,
            'min_cells': self.int_dtype(min_cells),
            'max_bins': self.int_dtype(max_bins),
            'step': self.int_dtype(step),
            'mad_threshold': self.dtype(MAD),
            'peak_removal': self.dtype(1/3),
            'min_nr_bins_peakdetection': self.int_dtype(5),
            'histogram_smoothing': self.int_dtype(smoothing_window*2),
            'mad_smoothing': self.dtype(mad_smoothings),
            'mad_method': mad_method
        }
        
        debris_params = {
            'min_peaks': self.int_dtype(min_peaks),
            'max_peaks': self.int_dtype(max_peaks),
            'smoothing_window': self.int_dtype(smoothing_window),
            'percentage_cells_present': self.dtype(percentage_cells_present),
            'num_bins': self.int_dtype(100)
        }
        # Configure gating strategies
        self.time_gate = (DaskGPUMADTimeGate(**time_params) if self.use_gpu 
                         else MADTimeGate(**time_params))
        
        self.debris_gate = (DaskGPUFSCDebrisGate(**debris_params) if self.use_gpu 
                           else FSCDebrisGate(**debris_params))
        
        # Configure doublet gating
        doublet_config = doublet_config or {}
        if doublet_method == 'inflection':
            config = {
                'bins': doublet_config.get('bins', 'auto'),
                'smoothing_factor': self.dtype(doublet_config.get('smoothing_factor', 0.5)),
                'fallback_mad_threshold': self.dtype(doublet_config.get('fallback_mad_threshold', 5))
            }
            self.doublet_gate = (CuPyInflectionDoubletGate(**config) if self.use_gpu 
                                else InflectionDoubletGate(**config))
        else:
            self.doublet_gate = (CuPyMADDoubletGate(mad_threshold=self.dtype(mad)) if self.use_gpu 
                                else MADDoubletGate(mad_threshold=mad))
            
        self.time_channel_index = time_channel_index
        self.skip_doublet_removal = False
        self.skip_debris_removal = False
        self._debug_info = {}

    def _to_gpu_array(self, data: ArrayType) -> da.Array:
        """Convert input array to GPU-backed dask array if using GPU."""
        if not self.use_gpu:
            return data
            
        if isinstance(data, da.Array):
            if data.chunks is None:
                data = data.rechunk(chunks=(self.chunk_size, -1))
            return data.map_blocks(cp.asarray)
        
        return da.from_array(data, chunks=(self.chunk_size, -1)).map_blocks(cp.asarray)

    def _process_array(self, data: ArrayType) -> da.Array:
        """Process array to ensure it's in the correct format for computations."""
        if self.use_gpu:
            return self._to_gpu_array(data)
        return data

    def process_fcs_data(self, marker_names: list[str], fcs_array: ArrayType) -> Dict[str, ArrayType]:
        """Process FCS data through the gating pipeline."""
        if len(marker_names) != fcs_array.shape[1]:
            raise ValueError("The number of marker names does not match the number of dimensions in the FCS array.")

        # Convert data to GPU array if using GPU
        fcs_array = self._process_array(fcs_array)
        
        # Initialize vectors with correct array type
        if isinstance(fcs_array, da.Array):
            ones = da.ones(fcs_array.shape[0],
                         chunks=fcs_array.chunks[0],
                         dtype=self.int_dtype)
        else:
            ones = np.ones(fcs_array.shape[0], dtype=self.int_dtype)
        
        vectors = {
            'lod': ones,
            'debris': ones.copy(),
            'time': ones.copy(),
            'doublet': ones.copy()
        }

        # Remove limit of detection events
        _, vectors['lod'] = self.remove_limit_of_detection_events(fcs_array, marker_names)
        vectors['lod'] = self._process_array(vectors['lod'])

        # Apply time gating if time channel is specified
        if self.time_channel_index is not None:
            _, vectors['time'] = self.time_gate.gate(fcs_array, self.time_channel_index, marker_names)
            vectors['time'] = self._process_array(vectors['time'])

        # Apply debris gating
        if not self.skip_debris_removal:
            result = self.debris_gate.gate(fcs_array, marker_names)
            vectors['debris'] = self._process_array(result.debris_vector)

        # Apply doublet gating
        if not self.skip_doublet_removal:
            _, vectors['doublet'] = self.doublet_gate.gate(fcs_array, marker_names)
            vectors['doublet'] = self._process_array(vectors['doublet'])

        # Calculate final vector using GPU operations if available
        final_vector = vectors['lod']
        for key in ['debris', 'time', 'doublet']:
            final_vector = final_vector & vectors[key]
        vectors['final'] = final_vector

        return vectors

    def remove_limit_of_detection_events(self, fcs_array: ArrayType, marker_names: list[str]) -> Tuple[ArrayType, ArrayType]:
        """Remove events at the limit of detection."""
        try:
            fsca_column = self._get_marker_index(marker_names, 'fsca')
        except ValueError:
            warnings.warn("No FSC-A parameter found. Skipping limit of detection removal.")
            return fcs_array, da.ones(fcs_array.shape[0], 
                                    chunks=fcs_array.chunks[0] if isinstance(fcs_array, da.Array) else None,
                                    dtype=self.int_dtype)

        # Process on GPU if available
        fsca_data = fcs_array[:, fsca_column]
        fsca_max = da.max(fsca_data).compute()
        max_events = da.sum(fsca_data == fsca_max).compute()
        total_events = fcs_array.shape[0]

        if max_events / total_events > 0.01:
            lod_vector = (fsca_data < fsca_max).astype(self.int_dtype)
            filtered_array = fcs_array[lod_vector]
            print(f"Removed {max_events} events ({max_events/total_events:.2%}) at the limit of detection.")
        else:
            lod_vector = da.ones(total_events, 
                               chunks=fcs_array.chunks[0] if isinstance(fcs_array, da.Array) else None,
                               dtype=self.int_dtype)
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
        import pandas as pd
        
        # Convert to numpy for final export
        if self.use_gpu:
            fcs_array = fcs_array.map_blocks(cp.asnumpy).compute()
            vectors = {k: v.map_blocks(cp.asnumpy).compute() for k, v in vectors.items()}
            
        df = pd.DataFrame(fcs_array, columns=marker_names)
        for name, vector in vectors.items():
            df[f'passed_{name}'] = vector.astype(bool)

        df.to_csv(output_file, index=False)
        print(f"Data exported to {output_file}")

    def get_debug_info(self) -> dict:
        """Get debugging information from the last pipeline run."""
        debug_info = {
            'time_gate': getattr(self.time_gate, 'get_debug_info', lambda: {})(),
            'debris_gate': getattr(self.debris_gate, 'get_debug_info', lambda: {})(),
            'doublet_gate': getattr(self.doublet_gate, 'get_debug_info', lambda: {})()
        }
        debug_info.update(self._debug_info)
        return debug_info
