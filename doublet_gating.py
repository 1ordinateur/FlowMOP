"""
Doublet gating component for FlowMOP.
Handles detection and removal of doublets in flow cytometry data.
"""

import numpy as np
from abc import ABC, abstractmethod
import warnings

class DoubletGateStrategy(ABC):
    @abstractmethod
    def gate(self, data: np.ndarray, marker_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Apply doublet gating to the data."""
        pass

class MADDoubletGate(DoubletGateStrategy):
    def __init__(self, mad_threshold=5):
        self.mad_threshold = mad_threshold

    def gate(self, data: np.ndarray, marker_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply MAD-based doublet gating to the data.
        
        Args:
            data: Flow cytometry data array
            marker_names: List of marker names
            
        Returns:
            tuple: (filtered_data, doublet_vector)
        """
        if not self._check_required_parameters(marker_names):
            return data, np.ones(data.shape[0], dtype=int)

        try:
            fsc_ratio = self._calculate_fsc_ratio(data, marker_names)
            ssc_ratio = self._calculate_ssc_ratio(data, marker_names)
        except ValueError as e:
            warnings.warn(str(e))
            return data, np.ones(data.shape[0], dtype=int)

        fsc_threshold = self._calculate_mad_threshold(fsc_ratio)
        ssc_threshold = self._calculate_mad_threshold(ssc_ratio)
        
        doublet_vector = ((fsc_ratio <= fsc_threshold) & 
                         (ssc_ratio <= ssc_threshold)).astype(int)
        filtered_data = data[doublet_vector == 1]
        
        return filtered_data, doublet_vector

    def _check_required_parameters(self, marker_names: list[str]) -> bool:
        """Check if all required parameters are present."""
        required_params = ['fsca', 'fsch', 'ssca', 'ssch']
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        all_params_present = all(param in standardized_names for param in required_params)
        
        if not all_params_present:
            warnings.warn("Not all required parameters (FSC-A, FSC-H, SSC-A, SSC-H) are present. "
                        "Doublet removal will be skipped.")
        return all_params_present

    def _calculate_fsc_ratio(self, data: np.ndarray, marker_names: list[str]) -> np.ndarray:
        """Calculate FSC-A/FSC-H ratio."""
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        try:
            fsc_a_column = standardized_names.index('fsca')
            fsc_h_column = standardized_names.index('fsch')
        except ValueError:
            raise ValueError("FSC-A or FSC-H parameters not found.")
        
        return data[:, fsc_a_column] / data[:, fsc_h_column]

    def _calculate_ssc_ratio(self, data: np.ndarray, marker_names: list[str]) -> np.ndarray:
        """Calculate SSC-A/SSC-H ratio."""
        standardized_names = [self._standardize_marker_name(name) for name in marker_names]
        try:
            ssc_a_column = standardized_names.index('ssca')
            ssc_h_column = standardized_names.index('ssch')
        except ValueError:
            raise ValueError("SSC-A or SSC-H parameters not found.")
        
        return data[:, ssc_a_column] / data[:, ssc_h_column]

    def _calculate_mad_threshold(self, ratio: np.ndarray) -> float:
        """Calculate MAD-based threshold for ratio values."""
        median_ratio = np.median(ratio)
        mad = np.median(np.abs(ratio - median_ratio))
        return median_ratio + self.mad_threshold * mad

    @staticmethod
    def _standardize_marker_name(name: str) -> str:
        """Standardize marker names by removing symbols and converting to lowercase."""
        import re
        return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

class InflectionDoubletGate(DoubletGateStrategy):
    """
    Alternative doublet gating strategy using inflection points in ratio histograms.
    This could be implemented as an alternative to the MAD-based approach.
    """
    def gate(self, data: np.ndarray, marker_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Implement inflection point-based doublet gating
        raise NotImplementedError("Inflection point-based doublet gating not yet implemented")
