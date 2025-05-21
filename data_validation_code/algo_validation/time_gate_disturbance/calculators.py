"""Module containing flow rate calculation logic."""

import numpy as np
import pandas as pd


class FlowRateCalculator:
    """Class responsible for calculating flow rates from time channel data"""
    
    @staticmethod
    def calculate(df: pd.DataFrame, time_channel: str) -> float:
        """Calculate the flow rate of a sample based on the time channel"""
        if len(df) < 2:
            return 1.0  # Default flow rate if not enough events
            
        # Calculate average time between events
        time_diffs = np.diff(df[time_channel].values)
        return 1.0 / np.median(time_diffs) if np.median(time_diffs) > 0 else 1.0 