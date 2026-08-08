"""
FlowMOP: Flow Cytometry Multi-Operation Pipeline
Main integration module that coordinates the gating operations.
"""

import warnings
import inspect
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict, Literal, Union, Any, Callable, TYPE_CHECKING
import numpy as np

# Type definitions
if TYPE_CHECKING:
    import dask.array as da
    from dask.distributed import Client
    ArrayType = Union[np.ndarray, da.Array]
else:
    Client = Any
    ArrayType = Any
DEFAULT_ARRAY_MODULE = np

# Import CPU implementations
from functions.time_gating import TimeGateStrategy, MADTimeGate
from functions.debris_gating import DebrisGateStrategy, FSCDebrisGate
from functions.doublet_gating import DoubletGateStrategy, MADDoubletGate, InflectionDoubletGate


@dataclass(frozen=True)
class FlowMOPContext:
    """Per-file state shared by gates."""
    data: np.ndarray
    marker_names: list[str]
    standardized_names: list[str]
    channel_index: Dict[str, int]
    fluorescence_indices: np.ndarray
    scatter_indices: Dict[str, int]
    time_channel_index: Optional[int]


@dataclass
class GateVectors:
    """Gate pass/fail vectors for one file."""
    lod: np.ndarray
    debris: np.ndarray
    time: np.ndarray
    doublet: np.ndarray


class GateExecutor:
    """Small executor abstraction for independent gate tasks."""

    def __init__(self, enabled: bool = False, max_workers: Optional[int] = None):
        self.enabled = enabled
        self.max_workers = max_workers

    def run(self, tasks: Dict[str, Callable[[], Any]]) -> Dict[str, Any]:
        if not self.enabled or len(tasks) <= 1:
            return {name: task() for name, task in tasks.items()}

        workers = self.max_workers or min(len(tasks), 4)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {name: pool.submit(task) for name, task in tasks.items()}
            return {name: future.result() for name, future in futures.items()}

class FlowMOP:
    """Main class for the Flow Cytometry Multi-Operation Pipeline."""
    
    def __init__(self,
                 remove_zeros=True,
                 min_cells=150,
                 max_bins=500,
                 step=200,
                 MAD=5,
                 mad=5,
                 min_peaks=2,
                 max_peaks=5,
                 smoothing_window=2,
                 mad_smoothings=None,
                 mad_method='all',
                 percentage_cells_present=3,
                 time_channel_index=None,
                 doublet_method: Literal['mad', 'inflection'] = 'inflection',
                 doublet_config: Optional[dict] = None,
                 enable_dask: bool = True,
                 chunk_size: Optional[int] = None,
                 fluor_mode: str = 'positives',
                 enable_plots: bool = False,
                 plots_dir: str = 'time_gate_plots',
                 enable_ssc: bool = False,
                 remove_beads: bool = False,
                 skip_debris: bool = False,
                 skip_time: bool = False,
                 skip_doublets: bool = False,
                 existing_client: Optional[Client] = None,
                 executor: Optional[GateExecutor] = None,
                 parallel_workers: Optional[int] = None):
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
            enable_dask: Legacy flag controlling within-file gate parallelism
            chunk_size: Kept for compatibility with older Dask-array callers
            fluor_mode: Mode for fluorescence analysis ('positives', 'geomean', or 'both')
            enable_plots: Whether to generate diagnostic plots during time gating
            plots_dir: Directory to save time gate diagnostic plots
            enable_ssc: Whether to use SSC-A for debris gating in addition to FSC-A
            remove_beads: Whether to detect and remove beads based on SSC/FSC characteristics
        """
        if mad_smoothings is None:
            mad_smoothings = [0.01, 0.05]

        if enable_dask:
            self.chunk_size = chunk_size or 10000  # Default chunk size if none provided
        else:
            self.chunk_size = None
        self.dtype = np.float32
        self.int_dtype = np.int32

        # When Dask is enabled, time gating owns within-file parallelism with
        # coarse channel blocks. Keep the gate executor off to avoid nested
        # scheduling and memory contention.
        time_params = {
            'remove_zeros': remove_zeros,
            'min_cells': self.int_dtype(min_cells),
            'max_bins': self.int_dtype(max_bins),
            'step': self.int_dtype(step),
            'mad_threshold': self.dtype(MAD),
            'peak_removal': self.dtype(1/3),
            'min_nr_bins_peakdetection': self.int_dtype(5),
            'histogram_smoothing': self.int_dtype(smoothing_window*2),
            'mad_smoothing': [float(value) for value in mad_smoothings],
            'mad_method': mad_method,
            'enable_dask': enable_dask,
            'fluor_mode': fluor_mode,
            'enable_plots': enable_plots,
            'plots_dir': plots_dir
        }
        
        debris_params = {
            'min_peaks': self.int_dtype(min_peaks),
            'max_peaks': self.int_dtype(max_peaks),
            'smoothing_window': self.int_dtype(smoothing_window),
            'percentage_cells_present': self.dtype(percentage_cells_present),
            'num_bins': self.int_dtype(100),
            'enable_dask': False,
            'enable_ssc': enable_ssc,
            'remove_beads': remove_beads
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
                'fallback_mad_threshold': self.dtype(doublet_config.get('fallback_mad_threshold', 5)),
                'enable_dask': False
            }
            self.doublet_gate = InflectionDoubletGate(**config)
        else:
            self.doublet_gate = MADDoubletGate(mad_threshold=mad, enable_dask=False)
            
        self.time_channel_index = time_channel_index
        self.skip_doublet_removal = skip_doublets
        self.skip_debris_removal = skip_debris
        self.skip_time_removal = skip_time
        self.enable_dask = enable_dask
        self.fluor_mode = fluor_mode
        self._debug_info = {}
        self.enable_plots = enable_plots
        self.plots_dir = plots_dir
        self.existing_client = existing_client
        self.executor = executor or GateExecutor(enabled=False, max_workers=parallel_workers)

    def _process_array(self, data: ArrayType) -> np.ndarray:
        """Process array to ensure it's in the correct format for computations."""
        if hasattr(data, "compute"):
            data = data.compute()
        return np.asarray(data)

    def _build_context(self, marker_names: list[str], fcs_array: ArrayType) -> FlowMOPContext:
        """Create reusable per-file channel lookup state."""
        if hasattr(fcs_array, "compute"):
            fcs_array = fcs_array.compute()

        data = np.asarray(fcs_array)
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        channel_index = {}
        for index, name in enumerate(standardized_names):
            channel_index.setdefault(name, index)

        excluded_terms = ('fsc', 'ssc', 'time', 'sample')
        fluorescence_indices = np.array(
            [
                i for i, name in enumerate(standardized_names)
                if i != self.time_channel_index and not any(term in name for term in excluded_terms)
            ],
            dtype=int,
        )
        scatter_indices = {
            name: channel_index[name]
            for name in ('fsca', 'fsch', 'ssca', 'ssch')
            if name in channel_index
        }

        return FlowMOPContext(
            data=data,
            marker_names=marker_names,
            standardized_names=standardized_names,
            channel_index=channel_index,
            fluorescence_indices=fluorescence_indices,
            scatter_indices=scatter_indices,
            time_channel_index=self.time_channel_index,
        )

    def process_fcs_data(self, marker_names: list[str], fcs_array: ArrayType) -> Dict[str, ArrayType]:
        """Process FCS data through the gating pipeline."""
        if len(marker_names) != fcs_array.shape[1]:
            raise ValueError("The number of marker names does not match the number of dimensions in the FCS array.")

        context = self._build_context(marker_names, fcs_array)
        ones = np.ones(context.data.shape[0], dtype=self.int_dtype)
        vectors = GateVectors(
            lod=ones,
            debris=ones.copy(),
            time=ones.copy(),
            doublet=ones.copy(),
        )

        # Process through gating pipeline
        # Note: Individual gating operations may still compute internally
        # as they need immediate results for threshold calculations
        vectors = self._do_gating(context, vectors)
        
        # Apply skipping logic for each filter type
        if self.skip_debris_removal:
            vectors.debris = ones.copy()  # Reset to all ones
            print("Skipping debris filtering.")

        if self.skip_time_removal and self.time_channel_index is not None:
            vectors.time = ones.copy()  # Reset to all ones
            print("Skipping time filtering.")

        if self.skip_doublet_removal:
            vectors.doublet = ones.copy()  # Reset to all ones
            print("Skipping doublet filtering.")

        final_vector = vectors.lod
        for vector in (vectors.debris, vectors.time, vectors.doublet):
            final_vector = final_vector & vector

        return {
            'lod': vectors.lod,
            'debris': vectors.debris,
            'time': vectors.time,
            'doublet': vectors.doublet,
            'final': final_vector,
        }
    
    def _do_gating(self, context: FlowMOPContext, vectors: GateVectors) -> GateVectors:
        """Perform independent gate operations through the configured executor."""
        tasks: Dict[str, Callable[[], Any]] = {
            'lod': lambda: self.remove_limit_of_detection_events(context.data, context.marker_names),
        }

        if context.time_channel_index is not None and not self.skip_time_removal:
            tasks['time'] = lambda: self._run_time_gate_vector_only(context)

        if not self.skip_debris_removal:
            tasks['debris'] = lambda: self.debris_gate.gate(context.data, context.marker_names)

        if not self.skip_doublet_removal:
            tasks['doublet'] = lambda: self.doublet_gate.gate(context.data, context.marker_names)

        computed = self.executor.run(tasks)

        # Process results
        vectors.lod = self._process_array(computed['lod'][1]).astype(self.int_dtype, copy=False)

        if computed.get('time') is not None:
            vectors.time = self._process_array(computed['time'][1]).astype(self.int_dtype, copy=False)

        if computed.get('debris') is not None:
            vectors.debris = self._process_array(computed['debris'].debris_vector).astype(self.int_dtype, copy=False)

        if computed.get('doublet') is not None:
            vectors.doublet = self._process_array(computed['doublet'][1]).astype(self.int_dtype, copy=False)

        return vectors

    def _run_time_gate_vector_only(self, context: FlowMOPContext) -> Tuple[Any, ArrayType]:
        """Run time gating without materializing filtered data when supported."""
        gate = self.time_gate.gate
        if "return_filtered_data" in inspect.signature(gate).parameters:
            return gate(
                context.data,
                context.time_channel_index,
                context.marker_names,
                return_filtered_data=False,
            )
        return gate(context.data, context.time_channel_index, context.marker_names)

    def remove_limit_of_detection_events(self, fcs_array: ArrayType, marker_names: list[str]) -> Tuple[ArrayType, ArrayType]:
        """Remove events at the limit of detection."""
        fcs_array = self._process_array(fcs_array)
        try:
            fsca_column = self._get_marker_index(marker_names, 'fsca')
        except ValueError:
            warnings.warn("No FSC-A parameter found. Skipping limit of detection removal.")
            # When FSC-A is missing, just return a pass-through vector.
            # Use NumPy here to avoid Dask backend issues when chunks are not explicit.
            return fcs_array, np.ones(fcs_array.shape[0], dtype=self.int_dtype)

        fsca_data = fcs_array[:, fsca_column]
        
        fsca_max = np.max(fsca_data)
        max_events = np.sum(fsca_data == fsca_max)
        
        total_events = fcs_array.shape[0]

        if max_events / total_events > 0.01:
            lod_vector = (fsca_data < fsca_max).astype(self.int_dtype)
            filtered_array = fcs_array[lod_vector.astype(bool)]
            print(f"Removed {max_events} events ({max_events/total_events:.2%}) at the limit of detection.")
        else:
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
        
    def set_skip_options(self, skip_debris: bool = None, skip_time: bool = None, skip_doublets: bool = None) -> None:
        """Update skipping options for different gating steps."""
        if skip_debris is not None:
            self.skip_debris_removal = skip_debris
        if skip_time is not None:
            self.skip_time_removal = skip_time
        if skip_doublets is not None:
            self.skip_doublet_removal = skip_doublets
            
    def set_time_channel_index(self, time_channel_index: int) -> None:
        """Update the time channel index."""
        self.time_channel_index = time_channel_index
        
    def set_client(self, client: Client) -> None:
        """Store a Dask client for backward compatibility."""
        self.existing_client = client

    def get_debug_info(self) -> dict:
        """Get debugging information from the last pipeline run."""
        debug_info = {
            'time_gate': getattr(self.time_gate, 'get_debug_info', lambda: {})(),
            'debris_gate': getattr(self.debris_gate, 'get_debug_info', lambda: {})(),
            'doublet_gate': getattr(self.doublet_gate, 'get_debug_info', lambda: {})()
        }
        debug_info.update(self._debug_info)
        return debug_info
