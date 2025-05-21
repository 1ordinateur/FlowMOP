"""
Time gating component for FlowMOP.
Handles detection and removal of time-based anomalies in flow cytometry data.
"""

import warnings
from typing import Union, Tuple, List, Dict
import numpy as np
import numpy.ma as ma
from scipy.ndimage import maximum_filter1d
from scipy.interpolate import UnivariateSpline
from abc import ABC, abstractmethod
from cpu.flowmop_utils import process_histogram, Peak  # Changed to relative import
from cpu.flowmop_utils import normalize_timeseries_values, apply_spline_smoothing, calculate_mad_thresholds
import matplotlib.pyplot as plt
import logging
from pathlib import Path

class TimeGateStrategy(ABC):
    @abstractmethod
    def gate(self, data: np.ndarray, time_channel_index: int, marker_names: list) -> tuple[np.ndarray, np.ndarray]:
        """Apply time gating to the data."""
        pass

class MADTimeGate(TimeGateStrategy):
    """
    Implements time gating using Median Absolute Deviation (MAD) to detect anomalies.
    
    This strategy analyzes time-dependent patterns in flow cytometry data using overlapping time bins
    and MAD-based outlier detection with configurable smoothing factors.
    
    Supports multiple operational modes:
    - 'positives': Detects positive peaks in each time bin and monitors their stability
    - 'geomean': Calculates geometric mean of all fluorescence channels in each time bin
    - 'positive_geomeans': Uses global thresholds to identify positive events, then tracks their geometric mean
    - 'both': Combines 'positives' and 'geomean' approaches for maximum sensitivity
    
    Attributes:
        remove_zeros (bool): Whether to remove zero values during processing
        min_cells (int): Minimum number of cells required per bin
        max_bins (int): Maximum number of bins to create
        step (int): Step size for bin creation
        mad_threshold (int): Number of MADs to use as threshold for outlier detection
        peak_removal (float): Threshold for peak removal (fraction of maximum peak)
        min_nr_bins_peakdetection (int): Minimum number of bins required for peak detection
        histogram_smoothing (int): Smoothing window size for histograms
        mad_method (str): Method for MAD calculation ('short', 'long', or 'all')
        mad_smoothing (list): Smoothing factors for MAD calculation
        enable_dask (bool): Whether to use Dask for parallel computation
        fluor_mode (str): Mode for fluorescence analysis
        enable_plots (bool): Whether to generate diagnostic plots
        plots_dir (str): Directory to save diagnostic plots
    """
    def __init__(self, remove_zeros=True, min_cells=150, max_bins=500, step=200, mad_threshold=6, 
                 peak_removal=1/3, min_nr_bins_peakdetection=5, histogram_smoothing=5, mad_method='all', 
                 mad_smoothing=[0.1, 1.0], enable_dask=True, fluor_mode='positives', enable_plots=False, 
                 plots_dir="time_gate_plots"):
        self.remove_zeros = remove_zeros
        self.min_cells = min_cells
        self.max_bins = max_bins
        self.step = step
        self.mad_threshold = mad_threshold
        self.peak_removal = peak_removal
        self.min_nr_bins_peakdetection = min_nr_bins_peakdetection
        self.histogram_smoothing = histogram_smoothing
        self.mad_method = mad_method
        self.mad_smoothing = mad_smoothing
        self.plot_counter = 0
        self.enable_dask = enable_dask
        self.fluor_mode = fluor_mode  # 'positives', 'geomean', 'positive_geomeans', or 'both'
        self.enable_plots = enable_plots
        self.plots_dir = plots_dir
        self.current_marker = "Unknown"
        
        # Create plots directory if it doesn't exist
        if self.enable_plots:
            import os
            os.makedirs(self.plots_dir, exist_ok=True)

        # Configure logging
        self.logger = logging.getLogger('MADTimeGate')
        self.logger.setLevel(logging.DEBUG)
        log_file = Path('log_file.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

    def gate(self, data: np.ndarray, time_channel_index: int, marker_names: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply MAD-based time gating to the data.
        
        Args:
            data: Flow cytometry data array
            time_channel_index: Index of the time channel
            marker_names: List of marker names corresponding to each channel
            
        Returns:
            tuple: (filtered_data, time_gate_vector)
        """
        # Calculate optimal number of events per bin
        events_per_bin = self._find_events_per_bin(data)
        breaks = self._make_breaks(events_per_bin, data.shape[0])

        # Find fluorescence channels
        fluoro_channels = self._get_fluorescence_channels(marker_names, time_channel_index)
        # Initialize array to count how many channels reject each cell
        rejection_count = np.zeros(data.shape[0], dtype=int)
        
        # Process based on selected mode(s)
        if self.fluor_mode in ['positives', 'both']:
            rejection_count = self._process_positive_peaks(data, fluoro_channels, breaks['breaks'], 
                                                        marker_names, rejection_count)
        if self.fluor_mode in ['geomean', 'both']:
            rejection_count = self._process_geometric_mean(data, fluoro_channels, breaks['breaks'], 
                                                        rejection_count)
        if self.fluor_mode == 'positive_geomeans':
            rejection_count = self._process_positive_geomeans(data, fluoro_channels, breaks['breaks'], 
                                                         marker_names, rejection_count)
            
        # Create final gate vector based on mode
        time_gate_vector = self._create_final_gate(rejection_count)
        filtered_data = data[time_gate_vector]
        
        return filtered_data, time_gate_vector

    def _get_fluorescence_channels(self, marker_names: list, time_channel_index: int) -> List[int]:
        """Extract indices of fluorescence channels."""
        excluded_terms = ['fsc', 'ssc', 'time', 'sample']
        return [i for i, name in enumerate(marker_names) 
                if i != time_channel_index and 
                not any(term in name.lower().replace('-','') for term in excluded_terms)]

    def _create_final_gate(self, rejection_count: np.ndarray) -> np.ndarray:
        """Create the final gate vector based on the fluor_mode."""
        if self.fluor_mode == 'both':
            # When using both methods, cells must be rejected by 2+ channels
            return rejection_count < 2
        else:
            # When using only one method, cells rejected by any channel are excluded
            return rejection_count < 1

    def _process_positive_peaks(self, data: np.ndarray, fluoro_channels: List[int], 
                          breaks: List[np.ndarray], marker_names: list,
                          rejection_count: np.ndarray) -> np.ndarray:
        """
        Process data using the 'positive_peaks' method:
        1. Find peaks for each fluorescence channel independently
        2. Apply MAD thresholding to each channel's peak time series
        
        Args:
            data: Flow cytometry data array
            fluoro_channels: List of fluorescence channel indices
            breaks: List of time bin indices
            marker_names: List of marker names
            rejection_count: Array counting rejections for each cell
            
        Returns:
            Updated rejection_count array
        """
        self.logger.info("Processing using positive_peaks mode")
        
        # Dictionary to map channel indices to marker names
        channel_to_marker = {channel: marker_names[channel] for channel in fluoro_channels}
        
        # Determine thresholds for all channels
        thresholds = self._determine_thresholds_all_channels(data, fluoro_channels, breaks, marker_names)
        
        if len(thresholds) == 0:
            self.logger.warning("No valid peaks detected for any channel")
            return rejection_count
        
        # Process peaks for each channel
        if self.enable_dask:
            # Use Dask for parallel processing
            rejection_count = self._process_peaks_with_dask(
                thresholds, breaks, data.shape[0], rejection_count, channel_to_marker
            )
        else:
            # Process sequentially
            rejection_count = self._process_peaks_without_dask(
                thresholds, breaks, data.shape[0], rejection_count, channel_to_marker
            )
        
        return rejection_count

    def _process_peaks_with_dask(self, thresholds: Dict, breaks: List[np.ndarray], 
                               nr_cells: int, rejection_count: np.ndarray, channel_to_marker: Dict = None) -> np.ndarray:
        """Process peaks using Dask for parallel computation."""
        for channel, channel_thresholds in thresholds.items():
            if channel_to_marker:
                self.current_marker = channel_to_marker.get(channel, f"Channel_{channel}")
            
            # Apply MAD analysis with Dask
            rejected_cells = self._apply_mad_analysis(channel_thresholds, breaks, nr_cells, use_dask=True)
            rejection_count[rejected_cells] += 1
        
        return rejection_count

    def _process_peaks_without_dask(self, thresholds: Dict, breaks: List[np.ndarray], 
                                  nr_cells: int, rejection_count: np.ndarray,
                                  channel_to_marker: Dict = None) -> np.ndarray:
        """Process peaks sequentially without Dask."""
        for channel, channel_thresholds in thresholds.items():
            if channel_to_marker:
                self.current_marker = channel_to_marker.get(channel, f"Channel_{channel}")
            
            # Apply MAD analysis without Dask
            rejected_cells = self._apply_mad_analysis(channel_thresholds, breaks, nr_cells, use_dask=False)
            rejection_count[rejected_cells] += 1
        
        return rejection_count

    def _process_geometric_mean(self, data: np.ndarray, fluoro_channels: List[int], 
                              breaks: List[np.ndarray], rejection_count: np.ndarray) -> np.ndarray:
        """Process data using geometric mean method."""
        # Calculate geometric mean for each bin
        geomean_results = self._geomean_mad_check(data, fluoro_channels, breaks)
        
        # Apply MAD analysis based on processing method
        if self.enable_dask:
            rejected_cells = self._process_geomean_with_dask(geomean_results, breaks, data.shape[0])
        else:
            rejected_cells = self._process_geomean_without_dask(geomean_results, breaks, data.shape[0])
        
        rejection_count[rejected_cells] += 1
        return rejection_count

    def _process_geomean_with_dask(self, geomean_results: np.ndarray, 
                                 breaks: List[np.ndarray], nr_cells: int) -> np.ndarray:
        """Process geometric mean using Dask for parallel computation."""
        return self._apply_mad_analysis(geomean_results, breaks, nr_cells, use_dask=True)

    def _process_geomean_without_dask(self, geomean_results: np.ndarray, 
                                    breaks: List[np.ndarray], nr_cells: int) -> np.ndarray:
        """Process geometric mean sequentially without Dask."""
        return self._apply_mad_analysis(geomean_results, breaks, nr_cells, use_dask=False)

    def _get_smoothing_factor(self, factor_type: str) -> float:
        """Get appropriate smoothing factor based on type ('short' or 'long')."""
        smooth_factors = self.mad_smoothing if isinstance(self.mad_smoothing, list) else [0.1, 1.0]
        
        if factor_type == 'short':
            return smooth_factors[0] if len(smooth_factors) > 0 else 0.1
        elif factor_type == 'long':
            return smooth_factors[-1] if len(smooth_factors) > 0 else 1.0
        else:
            return 0.1  # Default fallback

    def _mad_excluder(self, peaks, mad_threshold, breaks, nr_cells, smoothing_factor=0.1):
        """Apply MAD-based outlier detection with configurable smoothing."""
        # Extract just the threshold values from the structured array
        peak_values = peaks['Threshold']
        
        # Normalize values for consistent MAD calculation
        peak_values = normalize_timeseries_values(peak_values)
        
        # Apply spline smoothing
        smoothed_peaks = apply_spline_smoothing(peak_values, smoothing_factor, len(peak_values))
        
        # Calculate MAD thresholds and identify outliers
        _, _, to_remove_bins = calculate_mad_thresholds(smoothed_peaks, mad_threshold)
        
        # Calculate which cells to remove
        removed = self._removed_bins(breaks, to_remove_bins, nr_cells)
        contribution_mad = round((len(removed['cell_ids']) / nr_cells) * 100, 2)
        
        # Generate plot if enabled
        if self.enable_plots:
            self._generate_mad_plot(peak_values, smoothed_peaks, to_remove_bins, smoothing_factor)
        
        return {
            "cells": removed['cells'],
            "cell_ids": removed['cell_ids'],
            "MAD_bins": to_remove_bins,
            "Contribution_MAD": contribution_mad
        }

    def _geomean_mad_check(self, data: np.ndarray, channels: List[int], breaks: List[np.ndarray]) -> np.ndarray:
        """
        Calculate geometric mean of fluorescence for each time bin and prepare for MAD check.
        
        Args:
            data: Flow cytometry data
            channels: List of fluorescence channel indices
            breaks: Time bins
            
        Returns:
            Structured array with bin indices and their geometric means
        """
        # Create array to store geometric means for each bin
        geomean_values = np.zeros(len(breaks), dtype=[('Bin', int), ('Threshold', float)])
        
        # Define a function to calculate geometric mean for a single bin
        def calculate_bin_geomean(break_idx, break_indices, data, channels):
            # Get fluorescence values for all channels in this time bin
            bin_data = data[break_indices][:, channels]
            
            # Handle zeros and negative values for geometric mean calculation
            # Add small epsilon to avoid zeros, and take absolute value to handle negatives
            epsilon = 1e-10
            bin_data_processed = np.abs(bin_data) + epsilon
            
            # Calculate geometric mean across all fluorescence channels for each cell
            # Then calculate the mean of those geometric means for the bin
            geomean = np.exp(np.mean(np.log(bin_data_processed), axis=1)).mean()
            
            return break_idx, geomean
        
        # Process bins sequentially or with dask based on configuration
        if self.enable_dask:
            import dask
            # Create delayed tasks for each bin
            delayed_results = [
                dask.delayed(calculate_bin_geomean)(i, break_indices, data, channels)
                for i, break_indices in enumerate(breaks)
            ]
            # Compute all tasks in parallel
            results = dask.compute(*delayed_results)
            # Store results in the output array
            for break_idx, geomean in results:
                geomean_values[break_idx] = (break_idx, geomean)
        else:
            # Sequential processing
            for i, break_indices in enumerate(breaks):
                break_idx, geomean = calculate_bin_geomean(i, break_indices, data, channels)
                geomean_values[i] = (break_idx, geomean)
        return geomean_values
    
    def _apply_geomean_mad(self, geomean_values: np.ndarray, mad_threshold: float, 
                           breaks: List[np.ndarray], nr_cells: int, smoothing_factor: float = 0.1):
        """
        Apply MAD-based outlier detection to geometric mean values.
        
        Args:
            geomean_values: Structured array with bin indices and geometric mean values
            mad_threshold: Number of MADs to use as threshold
            breaks: Time bins
            nr_cells: Total number of cells
            smoothing_factor: Smoothing factor for spline
            
        Returns:
            Dictionary with cell filter results
        """
        # Extract values
        values = geomean_values['Threshold']
        
        # Normalize values for consistent MAD calculation
        values = normalize_timeseries_values(values)
        
        # Apply spline smoothing
        smoothed_values = apply_spline_smoothing(values, smoothing_factor, len(values))
        
        # Calculate MAD thresholds and identify outliers
        _, _, to_remove_bins = calculate_mad_thresholds(smoothed_values, mad_threshold)
        
        # Calculate which cells to remove
        removed = self._removed_bins(breaks, to_remove_bins, nr_cells)
        contribution_mad = round((len(removed['cell_ids']) / nr_cells) * 100, 2)
        
        # Generate plot if enabled
        if self.enable_plots:
            self._generate_geomean_plot(values, smoothed_values, to_remove_bins, smoothing_factor)
        
        return {
            "cells": removed['cells'],
            "cell_ids": removed['cell_ids'],
            "MAD_bins": to_remove_bins,
            "Contribution_MAD": contribution_mad
        }

    def _generate_mad_plot(self, peak_values, smoothed_peaks, to_remove_bins, smoothing_factor):
        """Generate diagnostic plot for MAD outlier detection."""
        plt.figure(figsize=(10, 6))
        time_points = np.arange(len(peak_values))
        
        # Plot original peaks and removed peaks (normalized)
        plt.scatter(time_points[~to_remove_bins], peak_values[~to_remove_bins], alpha=0.5, label='Valid Peaks')
        plt.scatter(time_points[to_remove_bins], peak_values[to_remove_bins], alpha=0.5, color='red', label='Removed Peaks')
        
        # Plot the smoothing spline
        plt.plot(time_points, smoothed_peaks, 'b-', label=f'Smoothing Spline (s={smoothing_factor})', alpha=0.7)
        
        # Calculate thresholds for plotting
        median_peak = np.median(smoothed_peaks)
        mad_peak = np.median(np.abs(smoothed_peaks - median_peak))
        upper_interval = median_peak + self.mad_threshold * mad_peak
        lower_interval = median_peak - self.mad_threshold * mad_peak
        
        # Plot MAD thresholds
        plt.axhline(y=upper_interval, color='g', linestyle='--', label='Upper MAD Threshold')
        plt.axhline(y=lower_interval, color='g', linestyle='--', label='Lower MAD Threshold')
        plt.axhline(y=1.0, color='k', linestyle=':', label='Mean (1.0)', alpha=0.5)
        
        plt.xlabel('Time Bin')
        plt.ylabel('Normalized Peak Value')
        plt.title(f'Normalized Peak Values with MAD Thresholds (smoothing={smoothing_factor})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the figure instead of displaying it
        self.plot_counter += 1
        plt.savefig(f"{self.plots_dir}/mad_threshold_plot_{self.plot_counter}_{self.current_marker}.png", dpi=300)
        plt.close()

    def _generate_geomean_plot(self, values, smoothed_values, to_remove_bins, smoothing_factor):
        """Generate diagnostic plot for geometric mean MAD outlier detection."""
        plt.figure(figsize=(10, 6))
        x_indices = np.arange(len(values))
        
        plt.scatter(x_indices[~to_remove_bins], values[~to_remove_bins], alpha=0.5, label='Valid Bins')
        plt.scatter(x_indices[to_remove_bins], values[to_remove_bins], alpha=0.5, color='red', label='Removed Bins')
        plt.plot(x_indices, smoothed_values, 'b-', label=f'Smoothing Spline (s={smoothing_factor})', alpha=0.7)
        
        # Calculate thresholds for plotting
        median_val = np.median(smoothed_values)
        mad_val = np.median(np.abs(smoothed_values - median_val))
        upper_interval = median_val + self.mad_threshold * mad_val
        lower_interval = median_val - self.mad_threshold * mad_val
        
        plt.axhline(y=upper_interval, color='g', linestyle='--', label='Upper MAD Threshold')
        plt.axhline(y=lower_interval, color='g', linestyle='--', label='Lower MAD Threshold')
        plt.axhline(y=1.0, color='k', linestyle=':', label='Mean (1.0)', alpha=0.5)
        
        plt.xlabel('Time Bin')
        plt.ylabel('Normalized Geometric Mean')
        plt.title(f'Normalized Geometric Mean with MAD Thresholds (smoothing={smoothing_factor})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        self.plot_counter += 1
        plt.savefig(f"{self.plots_dir}/geomean_mad_plot_{self.plot_counter}_{self.current_marker}.png", dpi=300)
        plt.close()

    def _find_events_per_bin(self, arr):
        """Calculate optimal number of events per bin."""
        nr_events = arr.shape[0]
        
        # Try maximum number of bins first
        events_per_bin = int(np.ceil((nr_events / self.max_bins) * 2))
        events_per_bin = ((events_per_bin // self.step) * self.step) + self.step
        # Create test bins
        test_breaks = self._split_with_overlap(np.arange(nr_events), events_per_bin, 
                                             int(np.ceil(events_per_bin / 2)))
        
        # Count events in each bin
        bin_sizes = np.array([len(bin_) for bin_ in test_breaks])
        # Check if we have 2 or more bins below minimum size
        if np.sum(bin_sizes < self.min_cells) >= 2:
            # Fall back to minimum cells per bin
            events_per_bin = self.min_cells
            events_per_bin = ((events_per_bin // self.step) * self.step) + self.step
        return events_per_bin

    def _make_breaks(self, events_per_bin, nr_events):
        """Create time bins with overlap."""
        breaks = self._split_with_overlap(np.arange(nr_events), events_per_bin, 
                                        int(np.ceil(events_per_bin / 2)))
        return {'breaks': breaks, 'events_per_bin': events_per_bin}

    def _determine_thresholds_all_channels(self, data, channels, breaks, marker_names):
        """
        Detect thresholds in all channels, analyzing each time bin separately.
        
        Args:
            data: Flow cytometry data array
            channels: List of channel indices to analyze
            breaks: List of time bin indices
            marker_names: List of marker names for labeling plots
            
        Returns:
            Dictionary mapping channel indices to threshold information
        """
        threshold_frames = {}
        import dask
        # Process channels based on whether Dask is enabled
        if self.enable_dask:
            from dask import delayed
            
            # Create delayed tasks for each channel
            delayed_results = {}
            for channel in channels:
                channel_data = data[:, channel]
                marker_name = marker_names[channel]
                # Create a delayed task for threshold determination
                delayed_result = delayed(self._determine_all_thresholds)(channel_data, breaks, marker_name)
                delayed_results[channel] = delayed_result
            
            # Compute all delayed tasks in parallel
            computed_results = dask.compute(delayed_results)[0]
            
            # Filter out None results
            for channel, result in computed_results.items():
                if result is not None:
                    threshold_frames[channel] = result
        else:
            # Sequential processing when Dask is disabled
            for channel in channels:
                channel_data = data[:, channel]
                marker_name = marker_names[channel]
                result = self._determine_all_thresholds(channel_data, breaks, marker_name)
                if result is not None:
                    threshold_frames[channel] = result
        
        return threshold_frames

    def _preprocess_channel_data(self, channel_data):
        """
        Preprocess channel data to handle limit-of-detection and data quality issues.
        Applies arcsinh transform to handle large dynamic range.
        
        Args:
            channel_data: 1D array of channel values
            
        Returns:
            Preprocessed channel data with arcsinh transform applied
        """
        # Make a copy to avoid modifying original data
        processed_data = channel_data.copy()
        
        # Apply arcsinh transform
        processed_data = np.arcsinh(processed_data/150)
        
        # Handle limit of detection (if >0.5% at max value)
        max_val = np.max(processed_data)
        pct_at_max = np.mean(processed_data == max_val) * 100
        if pct_at_max > 0.5:
            # Drop everything above 99th percentile and below 1st percentile
            p99 = np.percentile(processed_data, 99)
            processed_data = processed_data[(processed_data <= p99)]
        
        # # Clip to 99th quantile
        p99 = np.quantile(processed_data, 0.99)
        processed_data = np.clip(processed_data, None, p99)
        
        # Set negative values to zero
        processed_data[processed_data < 0] = 0
        
        if len(processed_data) < 10:
            self.logger.warning(f"Insufficient data after preprocessing ({len(processed_data)} events)")
            return np.array([])
        
        return processed_data

    def _determine_all_thresholds(self, channel_data, breaks, marker_name):
        """Determine peaks for a single channel."""
        self.logger.info(f"Processing marker: {marker_name}")
        
        # Preprocess the channel data
        processed_data = self._preprocess_channel_data(channel_data)
        if processed_data.size == 0: # Check if preprocessing resulted in empty data
            self.logger.warning(f"Preprocessing resulted in empty data for marker: {marker_name}. Cannot determine thresholds.")
            return None
        
        # Step 1a: Collect initial thresholds for all time bins
        initial_thresholds_dict = {}
        for i, break_indices in enumerate(breaks):
            # Ensure indices are within bounds of processed_data, not original channel_data
            valid_indices = break_indices[break_indices < len(processed_data)]
            if len(valid_indices) > 0:
                bin_data_segment = processed_data[valid_indices]
                # The max_peak argument for _timegate_threshold_detection can be set using a fraction of the overall processed_data's max value.
                # This provides a consistent reference for peak filtering across bins.
                overall_max_val = np.max(processed_data) if processed_data.size > 0 else None
                adaptive_max_peak = overall_max_val * self.peak_removal if overall_max_val is not None and self.peak_removal is not None else None
                
                thresh_array = self._timegate_threshold_detection(bin_data_segment,
                                                              smoothing=self.histogram_smoothing,
                                                              max_peak=adaptive_max_peak)
                initial_thresholds_dict[i] = thresh_array
            else:
                initial_thresholds_dict[i] = np.array([]) # No data for this bin

        # Step 1b: Perform backward fill imputation
        imputed_thresholds_dict = initial_thresholds_dict.copy()
        # Ensure breaks are processed in order for correct backward fill
        sorted_break_keys = sorted(imputed_thresholds_dict.keys(), reverse=True) # Iterate from last bin backwards
        
        last_valid_threshold = np.array([])
        for break_key in sorted_break_keys:
            if imputed_thresholds_dict[break_key].size == 0:
                if last_valid_threshold.size > 0:
                    imputed_thresholds_dict[break_key] = last_valid_threshold
                    self.logger.debug(f"Imputed threshold for bin {break_key} with value from subsequent bin: {last_valid_threshold[0] if last_valid_threshold.size > 0 else 'N/A'}")
            else:
                last_valid_threshold = imputed_thresholds_dict[break_key]

        # Step 1c: Collect all valid (non-empty) threshold values after imputation
        all_valid_threshold_values = []
        for break_key in sorted(imputed_thresholds_dict.keys()): # Process in original order for collection
            thresh_arr = imputed_thresholds_dict[break_key]
            if thresh_arr.size > 0:
                all_valid_threshold_values.append(thresh_arr[0]) # Should be a single value
        
        if not all_valid_threshold_values:
            self.logger.warning(f"No valid thresholds could be determined or imputed for any bin in marker: {marker_name}. Skipping this marker.")
            return None

        # Step 1d: Determine a single representative_threshold for the entire channel
        representative_threshold = self._find_most_occurring_thresholds(imputed_thresholds_dict, tolerance=0.05)
        if representative_threshold is None:
            representative_threshold = np.median(all_valid_threshold_values)
            self.logger.info(f"Used median of {len(all_valid_threshold_values)} values as representative threshold ({representative_threshold}) for {marker_name}.")
        else:
            self.logger.info(f"Used most occurring as representative threshold ({representative_threshold}) for {marker_name}.")

        # Step 1e: Call _update_threshold_frame with the imputed dictionary and the representative threshold
        # _update_threshold_frame will assign np.nan to bins that are still empty in imputed_thresholds_dict
        threshold_frame = self._update_threshold_frame(imputed_thresholds_dict, representative_threshold)

        # Step 1f: Filter the frame to remove any rows where the 'Threshold' is np.nan
        final_frame = threshold_frame[~np.isnan(threshold_frame['Threshold'])]

        # Step 1g: If this final_frame is empty, return None. Otherwise, apply existing filter.
        if final_frame.size == 0:
            self.logger.warning(f"No valid thresholds remaining after imputation and final filtering for marker: {marker_name}. Skipping this marker.")
            return None
        
        # Original filter: return updated_threshold_frame[updated_threshold_frame['Bin'] != -1]
        # The Bin != -1 filter is less critical now if NaNs are handled, but let's keep it for consistency if _update_threshold_frame can still produce -1 bins.
        # However, since we initialize with (-1, np.nan) and then populate, bins that are NOT updated would have bin=-1 and threshold=nan.
        # The ~np.isnan filter already handles the threshold part. If a bin truly had index -1, it would be problematic.
        # Assuming bin indices are always >=0 from `breaks`.
        return final_frame

    def _timegate_threshold_detection(self, bin_data, smoothing=None, max_peak=None, window_size=2):
        """
        Detect peaks in time-gated data.
        
        Ensures thresholds are:
        1. Separated by at least the smoothing window
        2. Above the peak_removal threshold relative to max fluoresent peak
        3. Have at least 10% prominence relative to surrounding minima if >2 peaks
        """
        # Log the initial length of bin_data
        initial_length = len(bin_data)
        self.logger.debug(f"Initial bin_data length: {initial_length}")
        
        if self.remove_zeros:
            bin_data = bin_data[bin_data != 0]
            self.logger.debug(f"After zero removal: {len(bin_data)} events (removed {initial_length - len(bin_data)})")
        
        # Use shared histogram processing function with class-level smoothing parameter if not overridden
        smoothing = smoothing if smoothing is not None else self.histogram_smoothing
        
        # Calculate the actual range of values
        min_val = np.min(bin_data)
        max_val = np.max(bin_data)
        value_range = max_val - min_val
        
        # Calculate cutoff points at 1% and 99% of the total range
        bottom_cutoff = min_val + (value_range * 0.01)  # Bottom 1%
        top_cutoff = max_val - (value_range * 0.01)     # Top 1%
        
        # Filter out values in the top and bottom 1% of the range
        pre_filter_length = len(bin_data)
        bin_data = bin_data[(bin_data >= bottom_cutoff) & (bin_data <= top_cutoff)]
        self.logger.debug(f"After range filtering: {len(bin_data)} events (removed {pre_filter_length - len(bin_data)})")
        
        if len(bin_data) == 0:
            self.logger.warning("Empty bin_data after filtering in threshold detection")
            return np.array([])

        try:
            q05, q95 = np.quantile(bin_data, [0.001, 0.999])
        except Exception as e:
            self.logger.error(f"Quantile calculation failed: {str(e)}")
            return np.array([])

        bin_data = bin_data[(bin_data >= q05) & (bin_data <= q95)]
        
        if len(bin_data) < 3:
            return np.array([])
            
        result = process_histogram(bin_data, smoothing_window=smoothing, num_bins=100)
        
        if result is None:
            self.logger.warning(f"Histogram processing failed for {self.current_marker}. Cannot determine timegate threshold.")
            return np.array([])
            
        smoothed_hist, bin_edges, peak_indices, peak_densities = result
        peak_indices = np.array(peak_indices, dtype=int)

        if peak_indices.size == 0:
            self.logger.warning(f"No peaks identified in histogram for {self.current_marker} by process_histogram. Cannot determine threshold in _timegate_threshold_detection.")
            return np.array([])

        # At this point, peak_indices.size > 0 is guaranteed.
        # effective_peak_comparison_value is used for the single-peak filtering.
        # It's either the max_peak (arg, an X-threshold) or a calculated Y-height if max_peak (arg) was None.
        effective_peak_comparison_value = max_peak # max_peak is the function argument
        if max_peak is None:
            effective_peak_comparison_value = np.max(smoothed_hist[peak_indices])
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        peak_x_values = bin_centers[peak_indices]
        
        threshold_determining_indices = np.array([], dtype=int)

        if len(peak_indices) == 1:
            if effective_peak_comparison_value is not None and self.peak_removal is not None:
                # This comparison can be X > fraction*Y or X > fraction*X, depending on how effective_peak_comparison_value was set.
                # This is a known complexity from the original logic.
                mask = peak_x_values > (self.peak_removal * effective_peak_comparison_value)
                threshold_determining_indices = peak_indices[mask]
            else: # Not enough info for filtering, keep the single peak
                threshold_determining_indices = peak_indices
        else: # len(peak_indices) > 1
            first_pidx = peak_indices[0]
            last_pidx = peak_indices[-1]
            
            if first_pidx >= last_pidx: 
                self.logger.warning(f"Peak indices misordered or duplicated ({first_pidx} >= {last_pidx}) for multi-peak logic in {self.current_marker}. Attempting fallback.")
                # Fallback to single-peak style logic using effective_peak_comparison_value if available
                if effective_peak_comparison_value is not None and self.peak_removal is not None:
                    mask = peak_x_values > (self.peak_removal * effective_peak_comparison_value)
                    threshold_determining_indices = peak_indices[mask]
                else: # If no fallback possible, use the first peak
                    threshold_determining_indices = np.array([first_pidx], dtype=int) if peak_indices.size > 0 else np.array([], dtype=int)
            else:
                valley_hist_segment = smoothed_hist[first_pidx:last_pidx] 
                if valley_hist_segment.size == 0:
                    # This occurs if first_pidx and last_pidx are adjacent (e.g., hist[5:5] for peaks at 5 and 6, means last_pidx should be exclusive for segment *between*)
                    # If segment is truly between, like first_pidx+1 : last_pidx.
                    # Original code implies hist[first_pidx] is part of valley search if first_pidx and last_pidx are adjacent.
                    # Let's assume if size is 0 here, it's an issue (e.g. first_pidx == last_pidx-1 implies size 1 if slice is [first:last]).
                    # This specific condition (size 0 for first_pidx:last_pidx) means first_pidx == last_pidx, which shouldn't happen here.
                    self.logger.warning(f"Valley segment calculation error or unexpected empty segment ({first_pidx}:{last_pidx}) for multi-peak logic in {self.current_marker}. Using first peak as fallback.")
                    threshold_determining_indices = np.array([first_pidx], dtype=int)
                else:
                    min_hist_val_in_valley = np.min(valley_hist_segment)
                    relative_indices_of_min = np.where(valley_hist_segment == min_hist_val_in_valley)[0]
                    mid_relative_idx = relative_indices_of_min[len(relative_indices_of_min) // 2]
                    valley_bottom_hist_idx = first_pidx + mid_relative_idx
                    
                    valley_x_pos = bin_centers[valley_bottom_hist_idx]
                    
                    peaks_to_average = peak_indices[peak_x_values > valley_x_pos]
                    
                    if peaks_to_average.size > 0:
                        avg_idx = int(np.mean(peaks_to_average))
                        threshold_determining_indices = np.array([avg_idx], dtype=int)
                    else:
                        threshold_determining_indices = np.array([valley_bottom_hist_idx], dtype=int)

        if threshold_determining_indices.size > 0:
            final_selected_hist_idx = threshold_determining_indices[0] 
            if 0 <= final_selected_hist_idx < len(bin_centers):
                return np.array([bin_centers[final_selected_hist_idx]]) 
            else:
                self.logger.warning(f"Final selected histogram index {final_selected_hist_idx} out of bounds for bin_centers (len {len(bin_centers)}) in {self.current_marker}.")
                return np.array([])
        else:
            self.logger.debug(f"No threshold-determining peak index found for {self.current_marker} after applying logic.")
            return np.array([])

    def _find_most_occurring_thresholds(self, thresholds, tolerance=0.05):
        """Find the most frequently occurring peaks."""
        flattened_peaks = [(bin_num, peak) for bin_num, peak_list in thresholds.items() 
                          for peak in peak_list]
        if not flattened_peaks:
            return None
            
        bin_numbers, peak_values = zip(*flattened_peaks)
        bin_edges = np.arange(min(peak_values), max(peak_values) + tolerance, tolerance)
        hist, _ = np.histogram(peak_values, bins=bin_edges)
        
        min_occurrences = (self.min_nr_bins_peakdetection / 100) * len(thresholds)
        candidate_bins = np.where(hist >= min_occurrences)[0]
        
        if len(candidate_bins) > 0:
            candidate_thresholds = (bin_edges[candidate_bins] + bin_edges[candidate_bins + 1]) / 2
            return candidate_thresholds[np.argmax(hist[candidate_bins])]
        return None

    def _update_threshold_frame(self, thresholds, most_occurring_thresholds):
        """Update threshold frame with closest thresholds to the most occurring thresholds."""
        # Initialize with Bin = -1 and Threshold = np.nan to handle cases where bins might not be updated.
        num_elements = len(thresholds)
        updated_threshold_frame = np.empty(num_elements, dtype=[('Bin', int), ('Threshold', float)])
        if num_elements > 0: # Avoid assignment to empty array fields if num_elements is 0
            updated_threshold_frame['Bin'] = -1
            updated_threshold_frame['Threshold'] = np.nan
        
        # Ensure thresholds keys are sorted for consistent processing, assuming keys are bin indices 0, 1, ... N-1
        sorted_break_indices = sorted(thresholds.keys())

        for i, break_ in enumerate(sorted_break_indices):
            threshold_list = thresholds[break_]

            if not isinstance(break_, int):
                self.logger.warning(f"Invalid break index type: {break_} (value: {threshold_list}). Skipping this entry in _update_threshold_frame.")
                # The entry in updated_threshold_frame for this 'i' will remain (-1, np.nan)
                continue

            if threshold_list.size == 0:
                self.logger.debug(f"Empty threshold list for break {break_} after imputation. Marking as NaN.")
                updated_threshold_frame[i] = (break_, np.nan)
            elif most_occurring_thresholds is None:
                self.logger.warning(f"most_occurring_thresholds is None for break {break_}. Cannot update frame. Marking as NaN.")
                updated_threshold_frame[i] = (break_, np.nan)
            else:
                # threshold_list should contain a single value after imputation and prior processing.
                if threshold_list.size == 1:
                    # Find the single threshold value that is closest to the most_occurring_thresholds value.
                    # Since threshold_list has one item, it is trivially the closest to any single reference value.
                    # We effectively snap this bin's threshold to be *one of* the representative values if it's a list,
                    # or just use the single imputed value if we want to preserve it before a global snap.
                    # The original logic was to pick from threshold_list based on most_occurring_thresholds.
                    # If most_occurring_thresholds is a single float, and threshold_list is np.array([val]), then closest is val.
                    closest_threshold_value = threshold_list[0] # Directly use the (imputed) value
                    updated_threshold_frame[i] = (break_, closest_threshold_value)
                else:
                    # This case should ideally not happen if _timegate_threshold_detection returns single value or empty
                    self.logger.warning(f"Threshold list for break {break_} has unexpected size {threshold_list.size} in _update_threshold_frame. Using first value if available, else NaN.")
                    if threshold_list.size > 0:
                        updated_threshold_frame[i] = (break_, threshold_list[0])
                    else: # Should be caught by initial threshold_list.size == 0
                        updated_threshold_frame[i] = (break_, np.nan)
        
        return updated_threshold_frame

    def _removed_bins(self, breaks, outlier_bins, nr_cells):
        """Calculate removed bins based on outliers."""
        if np.any(outlier_bins):
            removed_cells = np.concatenate([breaks[i] for i in np.where(outlier_bins)[0]])
            removed_cells = np.unique(removed_cells)
        else:
            removed_cells = np.array([], dtype=int)
        
        bad_cells = np.ones(nr_cells, dtype=bool)
        bad_cells[removed_cells] = False
        return {'cells': bad_cells, 'cell_ids': removed_cells}

    @staticmethod
    def _split_with_overlap(vec, seg_length, overlap):
        """Split vector into overlapping segments."""
        print(f"Splitting vector into overlapping segments of length {seg_length} with overlap {overlap}")
        starts = np.arange(0, len(vec), seg_length - overlap)
        ends = starts + seg_length
        ends[ends > len(vec)] = len(vec)
        return [vec[start:end] for start, end in zip(starts, ends)]

    def _calculate_bin_geomean(self, bin_idx: int, bin_indices: np.ndarray, 
                           data: np.ndarray, valid_channels: List[int], 
                           global_thresholds: Dict[int, float]) -> Dict[int, float]:
        """
        Calculate the geometric mean of positive cells for each channel in a given time bin.
        
        For each channel:
        1. Identify positive cells (above channel-specific threshold)
        2. Calculate geometric mean of those positive values
        
        Args:
            bin_idx: Index of the current bin
            bin_indices: Indices of cells in this bin
            data: Flow cytometry data array
            valid_channels: List of channel indices to use
            global_thresholds: Dictionary mapping channel indices to their thresholds
            
        Returns:
            Dictionary mapping channel indices to their geometric means for this bin
            (channels with no valid data will be omitted)
        """
        if len(bin_indices) < self.min_cells:
            self.logger.debug(f"Bin {bin_idx} has fewer than {self.min_cells} cells, skipping")
            return {}
        
        # Get cells in this time bin
        bin_data = data[bin_indices]
        
        # For each channel, extract positive cell values and calculate geometric mean
        channel_geomeans = {}
        
        for channel in valid_channels:
            channel_data = bin_data[:, channel]
            threshold = global_thresholds[channel]
            
            # Get cells above threshold
            positives = channel_data[channel_data > threshold]
            
            if len(positives) > 0:
                # Calculate geometric mean for this channel
                channel_geomean = np.exp(np.mean(np.log(positives + 1e-10)))
                channel_geomeans[channel] = channel_geomean
        
        return channel_geomeans

    def _process_positive_geomeans(self, data: np.ndarray, fluoro_channels: List[int], 
                                 breaks: List[np.ndarray], marker_names: list,
                                 rejection_count: np.ndarray) -> np.ndarray:
        """
        Process data using the 'positive_geomeans' method:
        1. Determine global thresholds for each channel using existing infrastructure
        2. For each time bin and channel, calculate geometric mean of cells above threshold
        3. Apply MAD thresholding to each channel's time series independently
        
        Args:
            data: Flow cytometry data array
            fluoro_channels: List of fluorescence channel indices
            breaks: List of time bin indices
            marker_names: List of marker names
            rejection_count: Array counting rejections for each cell
            
        Returns:
            Updated rejection_count array
        """
        self.logger.info("Processing using positive_geomeans mode")
        
        # Get thresholds using existing method that already leverages Dask
        threshold_frames = self._determine_thresholds_all_channels(data, fluoro_channels, breaks, marker_names)
        
        # Extract global thresholds from threshold frames
        global_thresholds = {}
        for channel, thresholds in threshold_frames.items():
            if len(thresholds) > 0:
                # Find the smallest threshold for this channel
                global_threshold = np.min(thresholds['Threshold'])
                
                global_thresholds[channel] = global_threshold
                self.logger.info(f"Global threshold for channel {channel} ({marker_names[channel]}): {global_threshold}")
        
        # Skip channels with no valid thresholds
        valid_channels = [ch for ch in fluoro_channels if ch in global_thresholds]
        
        if len(valid_channels) == 0:
            self.logger.warning("No valid thresholds detected for any channel")
            return rejection_count
        
        # Dictionary to store geometric means for each channel across all bins
        channel_geomean_values = {channel: [] for channel in valid_channels}
        
        # Calculate geometric means for each bin and channel
        if self.enable_dask:
            import dask
            from dask import delayed
            
            # Create delayed tasks for each bin
            delayed_tasks = [
                delayed(self._calculate_bin_geomean)(bin_idx, bin_indices, data, valid_channels, global_thresholds)
                for bin_idx, bin_indices in enumerate(breaks)
            ]
            
            # Compute all tasks in parallel
            bin_results = dask.compute(*delayed_tasks)
            
            # Organize results by channel
            for bin_idx, bin_geomeans in enumerate(bin_results):
                for channel, geomean in bin_geomeans.items():
                    channel_geomean_values[channel].append((bin_idx, geomean))
        else:
            # Sequential processing for non-Dask mode
            for bin_idx, bin_indices in enumerate(breaks):
                bin_geomeans = self._calculate_bin_geomean(bin_idx, bin_indices, data, valid_channels, global_thresholds)
                for channel, geomean in bin_geomeans.items():
                    channel_geomean_values[channel].append((bin_idx, geomean))
        
        # Convert lists to structured arrays for MAD analysis
        channel_thresholds = {}
        for channel, values in channel_geomean_values.items():
            if len(values) > 0:
                # Convert to structured array
                geomean_array = np.array(values, dtype=[('Bin', int), ('Threshold', float)])
                channel_thresholds[channel] = geomean_array
        
        # Apply MAD analysis to each channel separately
        for channel, geomean_values in channel_thresholds.items():
            self.current_marker = f"{marker_names[channel]}_PositiveGeomean"  # For plot titles
            rejected_cells = self._apply_mad_analysis(geomean_values, breaks, data.shape[0])
            rejection_count[rejected_cells] += 1
        
        return rejection_count

    def _apply_mad_analysis(self, geomean_values: np.ndarray, breaks: List[np.ndarray], 
                           nr_cells: int, use_dask: bool = None) -> np.ndarray:
        """
        Apply MAD-based analysis with the configured method and smoothing factors.
        
        This is a common utility method that centralizes the MAD analysis logic
        used across different processing functions.
        
        Args:
            geomean_values: Structured array with bin indices and values
            breaks: List of time bin indices
            nr_cells: Total number of cells
            use_dask: Whether to use Dask (if None, uses the class setting)
            
        Returns:
            Boolean array indicating which cells were rejected (True = rejected)
        """
        # Default to class setting if not specified
        if use_dask is None:
            use_dask = self.enable_dask
            
        if self.mad_method == 'short':
            # Short-term filtering only
            smoothing_factor = self._get_smoothing_factor('short')
            
            if use_dask:
                import dask
                task = dask.delayed(self._apply_geomean_mad)(
                    geomean_values, self.mad_threshold, breaks, 
                    nr_cells, smoothing_factor=smoothing_factor
                )
                result = dask.compute(task)[0]
            else:
                result = self._apply_geomean_mad(
                    geomean_values, self.mad_threshold, breaks, 
                    nr_cells, smoothing_factor=smoothing_factor
                )
                
            rejected_cells = ~result['cells']
            
        elif self.mad_method == 'long':
            # Long-term filtering only
            smoothing_factor = self._get_smoothing_factor('long')
            
            if use_dask:
                import dask
                task = dask.delayed(self._apply_geomean_mad)(
                    geomean_values, self.mad_threshold, breaks, 
                    nr_cells, smoothing_factor=smoothing_factor
                )
                result = dask.compute(task)[0]
            else:
                result = self._apply_geomean_mad(
                    geomean_values, self.mad_threshold, breaks, 
                    nr_cells, smoothing_factor=smoothing_factor
                )
                
            rejected_cells = ~result['cells']
            
        else:  # 'all' - default
            # Both short and long-term filtering
            short_smoothing = self._get_smoothing_factor('short')
            long_smoothing = self._get_smoothing_factor('long')
            
            if use_dask:
                import dask
                short_task = dask.delayed(self._apply_geomean_mad)(
                    geomean_values, self.mad_threshold, breaks, 
                    nr_cells, smoothing_factor=short_smoothing
                )
                long_task = dask.delayed(self._apply_geomean_mad)(
                    geomean_values, self.mad_threshold, breaks, 
                    nr_cells, smoothing_factor=long_smoothing
                )
                short_result, long_result = dask.compute(short_task, long_task)
            else:
                short_result = self._apply_geomean_mad(
                    geomean_values, self.mad_threshold, breaks, 
                    nr_cells, smoothing_factor=short_smoothing
                )
                long_result = self._apply_geomean_mad(
                    geomean_values, self.mad_threshold, breaks, 
                    nr_cells, smoothing_factor=long_smoothing
                )
            
            rejected_cells = ~(short_result['cells'] & long_result['cells'])
            
        return rejected_cells
