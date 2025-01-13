"""
Time gating component for FlowMOP.
Handles detection and removal of time-based anomalies in flow cytometry data.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import maximum_filter1d
from abc import ABC, abstractmethod

class TimeGateStrategy(ABC):
    @abstractmethod
    def gate(self, data: np.ndarray, time_channel_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Apply time gating to the data."""
        pass

class MADTimeGate(TimeGateStrategy):
    def __init__(self, remove_zeros=True, min_cells=150, max_bins=500, step=200, mad_threshold=6):
        self.remove_zeros = remove_zeros
        self.min_cells = min_cells
        self.max_bins = max_bins
        self.step = step
        self.mad_threshold = mad_threshold

    def gate(self, data: np.ndarray, time_channel_index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply MAD-based time gating to the data.
        
        Args:
            data: Flow cytometry data array
            time_channel_index: Index of the time channel
            
        Returns:
            tuple: (filtered_data, time_gate_vector)
        """
        time_channel = data[:, time_channel_index]
        events_per_bin = self._find_events_per_bin(data)
        breaks = self._make_breaks(events_per_bin, len(time_channel))
        
        peaks = self._determine_peaks_all_channels(data, [time_channel_index], breaks['breaks'])
        outlier_bins = self._identify_outliers(peaks, breaks['breaks'], len(time_channel))
        
        time_gate_vector = outlier_bins['cells']
        filtered_data = data[time_gate_vector]
        
        return filtered_data, time_gate_vector

    def _find_events_per_bin(self, arr):
        """Calculate optimal number of events per bin."""
        return FindEventsPerBin(arr, [], self.remove_zeros, self.min_cells, self.max_bins, self.step)

    def _make_breaks(self, events_per_bin, nr_events):
        """Create time bins with overlap."""
        breaks = SplitWithOverlap(np.arange(nr_events), events_per_bin, 
                                int(np.ceil(events_per_bin / 2)))
        return {'breaks': breaks, 'events_per_bin': events_per_bin}

    def _determine_peaks_all_channels(self, data, channels, breaks):
        """Detect peaks in all channels."""
        return determine_peaks_all_channels(data, channels, breaks, 
                                         self.remove_zeros, 1/3, 5)

    def _identify_outliers(self, peaks, breaks, nr_cells):
        """Identify outlier bins using MAD."""
        return RemovedBins(breaks, peaks['MAD_bins'], nr_cells)

# Utility functions moved from original implementation
def SplitWithOverlap(vec, seg_length, overlap):
    starts = np.arange(0, len(vec), seg_length - overlap)
    ends = starts + seg_length
    ends[ends > len(vec)] = len(vec)
    return [vec[start:end] for start, end in zip(starts, ends)]

def FindEventsPerBin(arr, channels, remove_zeros, min_cells, max_bins, step):
    nr_events = arr.shape[0]
    if remove_zeros and len(channels) > 0:
        max_bins_mass = min(np.sum(arr[:, channels] != 0, axis=0)) / min_cells
        if max_bins_mass < max_bins:
            max_bins = max_bins_mass
    max_cells = int(np.ceil((nr_events / max_bins) * 2))
    max_cells = ((max_cells // step) * step) + step
    return max(min_cells, max_cells)

def determine_peaks_all_channels(data, channels, breaks, remove_zeros, peak_removal, min_nr_bins_peakdetection):
    data_reshaped = data[:, channels].T
    peaks = {channel: timegate_peak_detection(data_reshaped[i], remove_zeros, peak_removal) 
            for i, channel in enumerate(channels)}
    return peaks

def timegate_peak_detection(bin_data, remove_zeros, peak_removal=1/3, smoothing=2, max_peak=None, window_size=10):
    if remove_zeros:
        bin_data = bin_data[bin_data != 0]
    
    if len(bin_data) < 3:
        return np.array([])
    
    hist, bin_edges = np.histogram(bin_data, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    smoothed_hist = np.convolve(hist, np.ones(smoothing), mode='same') / smoothing
    
    maxima = (smoothed_hist > np.roll(smoothed_hist, 1)) & (smoothed_hist > np.roll(smoothed_hist, -1))
    maxima[:window_size] = maxima[-window_size:] = False
    
    if max_peak is None:
        max_peak = np.max(hist)
    
    peak_indices = np.where(maxima)[0]
    filtered_peak_indices = peak_indices[bin_centers[peak_indices] > peak_removal * max_peak]
    
    return bin_centers[filtered_peak_indices]

def RemovedBins(breaks, outlier_bins, nr_cells):
    if np.any(outlier_bins):
        removed_cells = np.concatenate([breaks[i] for i in np.where(outlier_bins)[0]])
        removed_cells = np.unique(removed_cells)
    else:
        removed_cells = np.array([], dtype=int)

    bad_cells = np.ones(nr_cells, dtype=bool)
    bad_cells[removed_cells] = False

    return {'cells': bad_cells, 'cell_ids': removed_cells}
