"""
FlowMOP: Flow Cytometry Multi-Operation Pipeline
Main integration module that coordinates the gating operations.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Tuple, Dict, Literal

from .time_gating import TimeGateStrategy, MADTimeGate
from .debris_gating import DebrisGateStrategy, FSCDebrisGate
from .doublet_gating import DoubletGateStrategy, MADDoubletGate, InflectionDoubletGate

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
                 doublet_config: Optional[dict] = None):
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
        """
        self.time_gate = MADTimeGate(
            remove_zeros=remove_zeros,
            min_cells=min_cells,
            max_bins=max_bins,
            step=step,
            mad_threshold=MAD,
            peak_removal=1/3,
            min_nr_bins_peakdetection=5
        )
        self.debris_gate = FSCDebrisGate()
        
        # Configure doublet gating based on method
        if doublet_method == 'inflection':
            config = doublet_config or {}
            self.doublet_gate = InflectionDoubletGate(
                bins=config.get('bins', 'auto'),
                smoothing_factor=config.get('smoothing_factor', 0.5),
                fallback_mad_threshold=config.get('fallback_mad_threshold', 5)
            )
        else:  # 'mad'
            self.doublet_gate = MADDoubletGate(mad_threshold=mad)
            
        self.time_channel_index = time_channel_index
        self.skip_doublet_removal = False
        self.skip_debris_removal = False
        self.standardized_names = None
        self._debug_info = {}

    def process_fcs_data(self, marker_names: list[str], fcs_array: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Process FCS data through the gating pipeline.
        
        Args:
            marker_names: List of marker names
            fcs_array: Array containing FCS data
            
        Returns:
            tuple: (filtered_data, filter_vectors)
        """
        if len(marker_names) != fcs_array.shape[1]:
            raise ValueError("The number of marker names does not match the number of dimensions in the FCS array.")

        # Remove events at the limit of detection
        filtered_data, lod_vector = self.remove_limit_of_detection_events(fcs_array, marker_names)
        
        # Initialize vectors
        vectors = {
            'lod': lod_vector,
            'debris': np.ones(fcs_array.shape[0], dtype=int),
            'time': np.ones(fcs_array.shape[0], dtype=int),
            'doublet': np.ones(fcs_array.shape[0], dtype=int)
        }

        # Apply debris gating
        if filtered_data is not None and not self.skip_debris_removal:
            filtered_data, _, vectors['debris'] = self.debris_gate.gate(filtered_data, marker_names)

        # Apply time gating
        if filtered_data is not None and self.time_channel_index is not None:
            filtered_data, vectors['time'] = self.time_gate.gate(filtered_data, self.time_channel_index)

        # Apply doublet gating
        if filtered_data is not None and not self.skip_doublet_removal:
            filtered_data, vectors['doublet'] = self.doublet_gate.gate(filtered_data, marker_names)

        # Calculate final vector
        vectors['final'] = (vectors['lod'] & vectors['debris'] & 
                          vectors['time'] & vectors['doublet'])

        return filtered_data, vectors

    def remove_limit_of_detection_events(self, fcs_array: np.ndarray, marker_names: list[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Remove events at the limit of detection."""
        try:
            fsca_column = self._get_marker_index(marker_names, 'fsca')
        except ValueError:
            warnings.warn("No FSC-A parameter found. Skipping limit of detection removal.")
            return fcs_array, np.ones(fcs_array.shape[0], dtype=int)

        fsca_max = np.max(fcs_array[:, fsca_column])
        max_events = np.sum(fcs_array[:, fsca_column] == fsca_max)
        total_events = fcs_array.shape[0]

        if max_events / total_events > 0.01:
            lod_vector = (fcs_array[:, fsca_column] < fsca_max).astype(int)
            filtered_array = fcs_array[lod_vector == 1]
            print(f"Removed {max_events} events ({max_events/total_events:.2%}) at the limit of detection.")
        else:
            lod_vector = np.ones(total_events, dtype=int)
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

    def export_to_csv(self, output_file: str, fcs_array: np.ndarray, 
                     marker_names: list[str], vectors: Dict[str, np.ndarray]) -> None:
        """Export the processed data and filter vectors to CSV."""
        df = pd.DataFrame(fcs_array, columns=marker_names)
        
        for name, vector in vectors.items():
            df[f'passed_{name}'] = vector.astype(bool)

        df.to_csv(output_file, index=False)
        print(f"Data exported to {output_file}")

    def process_and_export(self, marker_names: list[str], fcs_array: np.ndarray, 
                          output_file: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Process the data and export results to CSV."""
        results = self.process_fcs_data(marker_names, fcs_array)
        self.export_to_csv(output_file, fcs_array, marker_names, results[1])
        return results

    def get_debug_info(self) -> dict:
        """
        Get debugging information from the last pipeline run.
        Includes information from each gating step if available.
        """
        debug_info = {
            'time_gate': getattr(self.time_gate, 'get_debug_info', lambda: {})(),
            'debris_gate': getattr(self.debris_gate, 'get_debug_info', lambda: {})(),
            'doublet_gate': getattr(self.doublet_gate, 'get_debug_info', lambda: {})()
        }
        debug_info.update(self._debug_info)
        return debug_info
