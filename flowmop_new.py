"""
FlowMOP: Flow Cytometry Multi-Operation Pipeline
Main integration module that coordinates the gating operations.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Tuple, Dict

from .time_gating import TimeGateStrategy, MADTimeGate
from .debris_gating import DebrisGateStrategy, FSCDebrisGate
from .doublet_gating import DoubletGateStrategy, MADDoubletGate

class FlowMOP:
    """Main class for the Flow Cytometry Multi-Operation Pipeline."""
    
    def __init__(self,
                 time_gate: Optional[TimeGateStrategy] = None,
                 debris_gate: Optional[DebrisGateStrategy] = None,
                 doublet_gate: Optional[DoubletGateStrategy] = None,
                 time_channel_index: Optional[int] = None):
        """
        Initialize FlowMOP with gating strategies.
        
        Args:
            time_gate: Strategy for time gating
            debris_gate: Strategy for debris removal
            doublet_gate: Strategy for doublet discrimination
            time_channel_index: Index of the time channel
        """
        self.time_gate = time_gate or MADTimeGate()
        self.debris_gate = debris_gate or FSCDebrisGate()
        self.doublet_gate = doublet_gate or MADDoubletGate()
        self.time_channel_index = time_channel_index

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
        filtered_data, lod_vector = self._remove_limit_of_detection_events(fcs_array, marker_names)
        
        # Initialize vectors
        vectors = {
            'lod': lod_vector,
            'debris': np.ones(fcs_array.shape[0], dtype=int),
            'time': np.ones(fcs_array.shape[0], dtype=int),
            'doublet': np.ones(fcs_array.shape[0], dtype=int)
        }

        # Apply debris gating
        if filtered_data is not None:
            filtered_data, _, vectors['debris'] = self.debris_gate.gate(filtered_data, marker_names)

        # Apply time gating
        if filtered_data is not None and self.time_channel_index is not None:
            filtered_data, vectors['time'] = self.time_gate.gate(filtered_data, self.time_channel_index)

        # Apply doublet gating
        if filtered_data is not None:
            filtered_data, vectors['doublet'] = self.doublet_gate.gate(filtered_data, marker_names)

        # Calculate final vector
        vectors['final'] = (vectors['lod'] & vectors['debris'] & 
                          vectors['time'] & vectors['doublet'])

        return filtered_data, vectors

    def _remove_limit_of_detection_events(self, fcs_array: np.ndarray, marker_names: list[str]) -> Tuple[np.ndarray, np.ndarray]:
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
