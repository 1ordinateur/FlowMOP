"""
Time gating component for FlowMOP.
Handles detection and removal of time-based anomalies in flow cytometry data.
"""

import numpy as np
import numpy.ma as ma
from scipy.ndimage import maximum_filter1d
from scipy.interpolate import UnivariateSpline
from abc import ABC, abstractmethod

class TimeGateStrategy(ABC):
    @abstractmethod
    def gate(self, data: np.ndarray, time_channel_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Apply time gating to the data."""
        pass

class MADTimeGate(TimeGateStrategy):
    def __init__(self, remove_zeros=True, min_cells=150, max_bins=500, step=200, mad_threshold=6,
                 peak_removal=1/3, min_nr_bins_peakdetection=5):
        self.remove_zeros = remove_zeros
        self.min_cells = min_cells
        self.max_bins = max_bins
        self.step = step
        self.mad_threshold = mad_threshold
        self.peak_removal = peak_removal
        self.min_nr_bins_peakdetection = min_nr_bins_peakdetection

    def gate(self, data: np.ndarray, time_channel_index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply MAD-based time gating to the data.
        
        Args:
            data: Flow cytometry data array
            time_channel_index: Index of the time channel
            
        Returns:
            tuple: (filtered_data, time_gate_vector)
        """
        # Calculate optimal number of events per bin
        events_per_bin = self._find_events_per_bin(data)
        
        # Create time bins with overlap
        breaks = self._make_breaks(events_per_bin, data.shape[0])
        
        # Detect peaks in all channels
        peaks = self._determine_peaks_all_channels(data, [time_channel_index], breaks['breaks'])
        
        # Apply MAD-based outlier detection
        mad_results = self._mad_excluder(peaks[time_channel_index], self.mad_threshold, breaks['breaks'], data.shape[0])
        
        # Create time gate vector and filter data
        time_gate_vector = mad_results['cells']
        filtered_data = data[time_gate_vector]
        
        return filtered_data, time_gate_vector

    def _find_events_per_bin(self, arr):
        """Calculate optimal number of events per bin."""
        nr_events = arr.shape[0]
        
        if self.remove_zeros:
            max_bins_mass = min(np.sum(arr != 0, axis=0)) / self.min_cells
            if max_bins_mass < self.max_bins:
                self.max_bins = max_bins_mass
        
        max_cells = int(np.ceil((nr_events / self.max_bins) * 2))
        max_cells = ((max_cells // self.step) * self.step) + self.step
        
        return max(self.min_cells, max_cells)

    def _make_breaks(self, events_per_bin, nr_events):
        """Create time bins with overlap."""
        breaks = self._split_with_overlap(np.arange(nr_events), events_per_bin, 
                                        int(np.ceil(events_per_bin / 2)))
        return {'breaks': breaks, 'events_per_bin': events_per_bin}

    def _determine_peaks_all_channels(self, data, channels, breaks):
        """Detect peaks in all channels."""
        data_reshaped = data[:, channels].T

        def determine_all_peaks_wrapped(x):
            result = self._determine_all_peaks(x, breaks)
            return result if result is not None else ma.masked

        determine_all_peaks_vec = np.vectorize(determine_all_peaks_wrapped, signature='(n)->(m)')
        peak_frames = determine_all_peaks_vec(data_reshaped)

        return {channel: peak_frame for channel, peak_frame in zip(channels, peak_frames) 
                if not ma.is_masked(peak_frame)}

    def _determine_all_peaks(self, channel_data, breaks):
        """Determine peaks for a single channel."""
        full_channel_peaks = self._timegate_peak_detection(channel_data)
        if np.all(np.isnan(full_channel_peaks)):
            return None
        
        max_peak = np.max(channel_data)
        channel_breaks = {i: channel_data[break_indices] for i, break_indices in enumerate(breaks)}
        peaks = {break_: self._timegate_peak_detection(break_data, smoothing=2, max_peak=max_peak) 
                for break_, break_data in channel_breaks.items()}
        
        peak_frame = np.array([(break_, peak) for break_, peak_list in peaks.items() 
                              for peak in peak_list], dtype=[('Bin', int), ('Peak', float)])
        
        if np.any(np.isnan(peak_frame['Peak'])):
            peak_frame = peak_frame[~np.isnan(peak_frame['Peak'])]
            peaks = {break_: peak_list[~np.isnan(peak_list)] 
                    for break_, peak_list in peaks.items()}

        most_occurring_peaks = self._find_most_occurring_peaks(peaks)
        if most_occurring_peaks is None:
            most_occurring_peaks = np.median(peak_frame['Peak'])

        updated_peak_frame = self._update_peak_frame(peaks, most_occurring_peaks)
        return updated_peak_frame[updated_peak_frame['Bin'] != -1]

    def _timegate_peak_detection(self, bin_data, smoothing=2, max_peak=None, window_size=10):
        """Detect peaks in time-gated data."""
        if self.remove_zeros:
            bin_data = bin_data[bin_data != 0]
        
        if len(bin_data) < 3:
            return np.array([])
        
        hist, bin_edges = np.histogram(bin_data, bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed_hist = np.convolve(hist, np.ones(smoothing), mode='same') / smoothing
        
        max_filter = maximum_filter1d(smoothed_hist, size=2 * window_size + 1, mode='constant')
        maxima = ((smoothed_hist == max_filter) & 
                 (smoothed_hist > np.roll(smoothed_hist, 1)) & 
                 (smoothed_hist > np.roll(smoothed_hist, -1)))
        maxima[:window_size] = maxima[-window_size:] = False
        
        if max_peak is None:
            max_peak = np.max(hist)
        
        peak_indices = np.where(maxima)[0]
        filtered_peak_indices = peak_indices[bin_centers[peak_indices] > self.peak_removal * max_peak]
        
        return bin_centers[filtered_peak_indices]

    def _find_most_occurring_peaks(self, peaks, tolerance=0.05):
        """Find the most frequently occurring peaks."""
        flattened_peaks = [(bin_num, peak) for bin_num, peak_list in peaks.items() 
                          for peak in peak_list]
        if not flattened_peaks:
            return None
            
        bin_numbers, peak_values = zip(*flattened_peaks)
        bin_edges = np.arange(min(peak_values), max(peak_values) + tolerance, tolerance)
        hist, _ = np.histogram(peak_values, bins=bin_edges)
        
        min_occurrences = (self.min_nr_bins_peakdetection / 100) * len(peaks)
        candidate_bins = np.where(hist >= min_occurrences)[0]
        
        if len(candidate_bins) > 0:
            candidate_peaks = (bin_edges[candidate_bins] + bin_edges[candidate_bins + 1]) / 2
            return candidate_peaks[np.argmax(hist[candidate_bins])]
        return None

    def _update_peak_frame(self, peaks, most_occurring_peaks):
        """Update peak frame with closest peaks to the most occurring peaks."""
        updated_peak_frame = np.empty(len(peaks), dtype=[('Bin', int), ('Peak', float)])
        
        for i, (break_, peak_list) in enumerate(peaks.items()):
            if isinstance(break_, int) and break_ < len(updated_peak_frame):
                if len(peak_list) > 0:
                    closest_peak_index = np.argmin(np.abs(peak_list - most_occurring_peaks))
                    updated_peak_frame[i] = (break_, peak_list[closest_peak_index])
                else:
                    updated_peak_frame[i] = (break_, np.nan)
            else:
                updated_peak_frame[i] = (-1, np.nan)
        
        return updated_peak_frame

    def _mad_excluder(self, peaks, mad_threshold, breaks, nr_cells):
        """Apply MAD-based outlier detection."""
        peak_values = peaks['Peak']
        
        spline = UnivariateSpline(np.arange(len(peak_values)), peak_values, s=0.2 * len(peak_values))
        kernel_y = spline(np.arange(len(peak_values)))
        
        median_peak = np.median(kernel_y)
        mad_peak = np.median(np.abs(kernel_y - median_peak))
        
        upper_interval = median_peak + mad_threshold * mad_peak
        lower_interval = median_peak - mad_threshold * mad_peak
        to_remove_bins = (kernel_y > upper_interval) | (kernel_y < lower_interval)
        
        removed = self._removed_bins(breaks, to_remove_bins, nr_cells)
        contribution_mad = round((len(removed['cell_ids']) / nr_cells) * 100, 2)
        
        return {
            "cells": removed['cells'],
            "cell_ids": removed['cell_ids'],
            "MAD_bins": to_remove_bins,
            "Contribution_MAD": contribution_mad
        }

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
        starts = np.arange(0, len(vec), seg_length - overlap)
        ends = starts + seg_length
        ends[ends > len(vec)] = len(vec)
        return [vec[start:end] for start, end in zip(starts, ends)]
