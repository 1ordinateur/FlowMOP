"""
FlowMOP: Flow Cytometry Multi-Operation Pipeline
Main integration module that coordinates the gating operations.
"""

import warnings
from typing import Optional, Tuple, Dict, Literal, Union, Any
import numpy as np

# Type definitions
import dask.array as da
ArrayType = Union[np.ndarray, da.Array]
DEFAULT_ARRAY_MODULE = np

# Import CPU implementations
from cpu.time_gating import TimeGateStrategy, MADTimeGate
from cpu.debris_gating import DebrisGateStrategy, FSCDebrisGate
from cpu.doublet_gating import DoubletGateStrategy, MADDoubletGate, InflectionDoubletGate

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
                 enable_dask: bool = True,
                 chunk_size: Optional[int] = None,
                 fluor_mode: str = 'positives'):
        """
        Initialize FlowMOP with configuration parameters.
        
        Args:
            remove_zeros: Remove zero-value events
            min_cells: Minimum number of cells required for gating
            max_bins: Maximum number of bins for histogram-based gating
            step: Step size for peak detection
            MAD: Median Absolute Deviation threshold for MAD-based gating
            mad: MAD threshold for MAD-based doublet gating
            min_peaks: Minimum number of peaks for peak detection
            max_peaks: Maximum number of peaks for peak detection
            smoothing_window: Smoothing window for peak detection
            mad_smoothings: List of smoothing factors for MAD-based time gating
            mad_method: Method for MAD-based time gating ('short', 'long', or 'all')
            percentage_cells_present: Percentage of cells required for gating
            time_channel_index: Index of time channel
            doublet_method: Method for doublet gating ('mad' or 'inflection')
            doublet_config: Configuration for doublet gating method
                For inflection method:
                    - bins: Number of bins or method ('auto', 'fd', 'scott')
                    - smoothing_factor: KDE smoothing factor (0.1 to 1.0)
                    - fallback_mad_threshold: Fallback MAD threshold
            enable_dask: Whether to use Dask for parallel computation
            chunk_size: Size of chunks for DASK array operations
            fluor_mode: Mode for fluorescence analysis ('positives', 'geomean', or 'both')
        """
        if enable_dask:
            self.chunk_size = chunk_size or 10000  # Default chunk size if none provided
        else:
            self.chunk_size = None
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
            'mad_method': mad_method,
            'enable_dask': enable_dask,
            'fluor_mode': fluor_mode
        }
        
        debris_params = {
            'min_peaks': self.int_dtype(min_peaks),
            'max_peaks': self.int_dtype(max_peaks),
            'smoothing_window': self.int_dtype(smoothing_window),
            'percentage_cells_present': self.dtype(percentage_cells_present),
            'num_bins': self.int_dtype(100),
            'enable_dask': enable_dask
        }
        
        # Configure gating strategies
        self.time_gate = MADTimeGate(**time_params)
        self.debris_gate = FSCDebrisGate(**debris_params)
        
        # Configure doublet gating
        doublet_config = doublet_config or {}
        if doublet_method == 'inflection':
            config = {
                'bins': doublet_config.get('bins', 'auto'),
                'smoothing_factor': self.dtype(doublet_config.get('smoothing_factor', 0.5)),
                'fallback_mad_threshold': self.dtype(doublet_config.get('fallback_mad_threshold', 5))
            }
            self.doublet_gate = InflectionDoubletGate(**config)
        else:
            self.doublet_gate = MADDoubletGate(mad_threshold=mad)
            
        self.time_channel_index = time_channel_index
        self.skip_doublet_removal = False
        self.skip_debris_removal = False
        self.enable_dask = enable_dask
        self.fluor_mode = fluor_mode
        self._debug_info = {}

    def _process_array(self, data: ArrayType) -> da.Array:
        """Process array to ensure it's in the correct format for computations."""
        return data

    def process_fcs_data(self, marker_names: list[str], fcs_array: ArrayType) -> Dict[str, ArrayType]:
        """Process FCS data through the gating pipeline."""
        if len(marker_names) != fcs_array.shape[1]:
            raise ValueError("The number of marker names does not match the number of dimensions in the FCS array.")

        # Initialize vectors with proper Dask chunking
        if self.enable_dask:
            # Ensure consistent chunking along rows, preserve column chunks
            if isinstance(fcs_array, np.ndarray):
                fcs_array = da.from_array(fcs_array, chunks=(self.chunk_size or 'auto', -1))
            elif not fcs_array.chunks[0][0] == fcs_array.chunks[0][-1]:  # Check chunk uniformity
                fcs_array = fcs_array.rechunk((self.chunk_size or 'auto', -1))
            
            # Create vectors matching input chunk structure
            ones = da.ones_like(fcs_array[:,0], chunks=fcs_array.chunks[0], dtype=self.int_dtype)
        else:
            ones = np.ones(fcs_array.shape[0], dtype=self.int_dtype)
        
        vectors = {
            'lod': ones,
            'debris': ones.copy(),
            'time': ones.copy(),
            'doublet': ones.copy()
        }

        # Process through gating pipeline
        # Note: Individual gating operations may still compute internally
        # as they need immediate results for threshold calculations
        vectors = self._do_gating(fcs_array, vectors, marker_names)

        # Calculate final vector using DASK optimized operations
        if self.enable_dask:
            # Combine vectors using parallel logical AND
            final_vector = da.all(da.stack([
                vectors['lod'],
                vectors['debris'], 
                vectors['time'],
                vectors['doublet']
            ]), axis=0)
            
            # Persist intermediate results in memory
            vectors['final'] = final_vector.compute()
        else:
            final_vector = vectors['lod']
            for key in ['debris', 'time', 'doublet']:
                final_vector = final_vector & vectors[key]
            vectors['final'] = final_vector

        return vectors
    
    def _do_gating(self, data: ArrayType, vectors: Dict[str, ArrayType], marker_names: list[str]) -> Dict[str, ArrayType]:
        """Perform all gating operations in parallel using DASK if enabled, otherwise sequentially."""
        if self.enable_dask:
            import dask
            
            # Create delayed objects for parallel execution
            gates = {
                'lod': dask.delayed(self.remove_limit_of_detection_events)(data, marker_names),
                'time': dask.delayed(self.time_gate.gate)(data, self.time_channel_index, marker_names) 
                    if self.time_channel_index else None,
                'debris': dask.delayed(self.debris_gate.gate)(data, marker_names) 
                    if not self.skip_debris_removal else None,
                'doublet': dask.delayed(self.doublet_gate.gate)(data, marker_names) 
                    if not self.skip_doublet_removal else None
            }

            # Compute independent gates in parallel
            computed = dask.compute(gates)[0]

        else:
            # Execute gates sequentially without dask
            computed = {}
            
            # Run each gate operation
            computed['lod'] = self.remove_limit_of_detection_events(data, marker_names)
            
            if self.time_channel_index:
                computed['time'] = self.time_gate.gate(data, self.time_channel_index, marker_names)
            else:
                computed['time'] = None
                
            if not self.skip_debris_removal:
                computed['debris'] = self.debris_gate.gate(data, marker_names)
            else:
                computed['debris'] = None
                
            if not self.skip_doublet_removal:
                computed['doublet'] = self.doublet_gate.gate(data, marker_names)
            else:
                computed['doublet'] = None

        # Process results
        vectors['lod'] = self._process_array(computed['lod'][1])
        
        if computed['time']:
            vectors['time'] = self._process_array(computed['time'][1])
            
        if computed['debris']:
            vectors['debris'] = self._process_array(computed['debris'].debris_vector)
            
        if computed['doublet']:
            vectors['doublet'] = self._process_array(computed['doublet'][1])

        return vectors

    def remove_limit_of_detection_events(self, fcs_array: ArrayType, marker_names: list[str]) -> Tuple[ArrayType, ArrayType]:
        """Remove events at the limit of detection."""
        try:
            fsca_column = self._get_marker_index(marker_names, 'fsca')
        except ValueError:
            warnings.warn("No FSC-A parameter found. Skipping limit of detection removal.")
            # Conditionally use dask based on enable_dask flag
            if hasattr(self, 'enable_dask') and self.enable_dask:
                return fcs_array, da.ones(fcs_array.shape[0], 
                                        chunks=fcs_array.chunks[0] if isinstance(fcs_array, da.Array) else None,
                                        dtype=self.int_dtype)
            else:
                # Use numpy when dask is disabled
                return fcs_array, np.ones(fcs_array.shape[0], dtype=self.int_dtype)

        fsca_data = fcs_array[:, fsca_column]
        
        # Conditionally compute max and sum based on enable_dask flag
        if hasattr(self, 'enable_dask') and self.enable_dask:
            fsca_max = da.max(fsca_data).compute()
            max_events = da.sum(fsca_data == fsca_max).compute()
        else:
            # Use numpy when dask is disabled
            fsca_max = np.max(fsca_data)
            max_events = np.sum(fsca_data == fsca_max)
        
        total_events = fcs_array.shape[0]

        if max_events / total_events > 0.01:
            # Conditionally create vector based on enable_dask flag
            if hasattr(self, 'enable_dask') and self.enable_dask:
                lod_vector = (fsca_data < fsca_max).astype(self.int_dtype)
            else:
                # Use numpy when dask is disabled
                lod_vector = (fsca_data < fsca_max).astype(self.int_dtype)
            
            filtered_array = fcs_array[lod_vector]
            print(f"Removed {max_events} events ({max_events/total_events:.2%}) at the limit of detection.")
        else:
            # Conditionally create ones array based on enable_dask flag
            if hasattr(self, 'enable_dask') and self.enable_dask:
                lod_vector = da.ones(total_events, 
                                   chunks=fcs_array.chunks[0] if isinstance(fcs_array, da.Array) else self.chunk_size,
                                   dtype=self.int_dtype)
            else:
                # Use numpy when dask is disabled
                lod_vector = np.ones(total_events, dtype=self.int_dtype)
            
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
