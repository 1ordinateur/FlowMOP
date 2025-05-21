import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.interpolate
from scipy.interpolate import UnivariateSpline
import scipy.stats
from scipy.ndimage import maximum_filter1d
import warnings
import re

def standardize_marker_name(name):
    """
    Standardize marker names by removing symbols and converting to lowercase.
    
    Args:
    name (str): Original marker name.
    
    Returns:
    str: Standardized marker name.
    """
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def check_required_parameters(marker_names):
    """
    Check if the required parameters are present in the marker names.
    
    Args:
    marker_names (list): List of marker names.
    
    Returns:
    tuple: (bool, bool) - (all_params_present, only_area_params)
    """
    required_params = ['fsca', 'fsch', 'fscw', 'ssca', 'ssch', 'sscw']
    area_params = ['fsca', 'ssca']
    
    standardized_names = [standardize_marker_name(name) for name in marker_names]
    
    all_params_present = all(param in standardized_names for param in required_params)
    only_area_params = all(param in standardized_names for param in area_params) and not all_params_present
    
    if only_area_params:
        warnings.warn("Only FSC-A and SSC-A parameters are present. Doublet removal will be skipped.")
    
    return all_params_present, only_area_params

def check_events_in_bottom_bin(fcs_array, marker_names):
    """
    Check if events exist in the bottom 10th bin of FSC-A and SSC-A.
    
    Args:
    fcs_array (np.array): Array containing FCS data.
    marker_names (list): List of marker names.
    
    Returns:
    bool: True if events are present in the bottom 10th bin, False otherwise.
    """
    standardized_names = [standardize_marker_name(name) for name in marker_names]
    
    fsc_a_column = standardized_names.index('fsca')
    ssc_a_column = standardized_names.index('ssca')
    
    fsc_a_max = np.max(fcs_array[:, fsc_a_column])
    ssc_a_max = np.max(fcs_array[:, ssc_a_column])
    
    fsc_a_bottom_10th = fsc_a_max * 0.1
    ssc_a_bottom_10th = ssc_a_max * 0.1
    
    events_in_bottom_bin = np.any((fcs_array[:, fsc_a_column] <= fsc_a_bottom_10th) & 
                                  (fcs_array[:, ssc_a_column] <= ssc_a_bottom_10th))
    
    if not events_in_bottom_bin:
        warnings.warn("No events found in the bottom 10th bin of FSC-A and SSC-A. Debris removal will be skipped.")
    
    return events_in_bottom_bin

def determine_peaks_all_channels(data, channels, breaks, remove_zeros, peak_removal, min_nr_bins_peakdetection):
    # Reshape the data to have channels as the first dimension
    data_reshaped = data[:, channels].T

    # Create a vectorized version of determine_all_peaks
    def determine_all_peaks_wrapped(x):
        result = determine_all_peaks(x, breaks, remove_zeros, peak_removal, min_nr_bins_peakdetection)
        return result if result is not None else ma.masked

    determine_all_peaks_vec = np.vectorize(determine_all_peaks_wrapped, signature='(n)->(m)')

    # Apply the vectorized function to the reshaped data
    peak_frames = determine_all_peaks_vec(data_reshaped)

    # Create a dictionary to store the results for each channel
    results = {channel: peak_frame for channel, peak_frame in zip(channels, peak_frames) if not ma.is_masked(peak_frame)}

    return results

def determine_all_peaks(channel_data, breaks, remove_zeros, peak_removal, min_nr_bins_peakdetection):
    full_channel_peaks = timegate_peak_detection(channel_data, remove_zeros, peak_removal)
    if np.all(np.isnan(full_channel_peaks)):
        return None
    
    max_peak = np.max(channel_data)  # Calculate the global channel maximum
    
    minimum = np.min(channel_data)
    maximum = np.max(channel_data)
    range_ = abs(minimum) + abs(maximum)
    
    channel_breaks = {i: channel_data[break_indices] for i, break_indices in enumerate(breaks)}
    peaks = {break_: timegate_peak_detection(break_data, remove_zeros, peak_removal, smoothing=2, max_peak=max_peak) for break_, break_data in channel_breaks.items()}
    
    peak_frame = np.array([(break_, peak) for break_, peak_list in peaks.items() for peak in peak_list], dtype=[('Bin', int), ('Peak', float)])
    
    if np.any(np.isnan(peak_frame['Peak'])):
        peak_frame = peak_frame[~np.isnan(peak_frame['Peak'])]
        peaks = {break_: peak_list[~np.isnan(peak_list)] for break_, peak_list in peaks.items()}

    def is_within_tolerance(peak1, peak2, tolerance=0.1):
        return abs(peak1 - peak2) <= tolerance * peak1

    def find_most_occurring_peaks(peaks, min_nr_bins_peakdetection, tolerance=0.05):
        # Flatten the peaks and keep track of the bin numbers
        flattened_peaks = [(bin_num, peak) for bin_num, peak_list in peaks.items() for peak in peak_list]
        bin_numbers, peak_values = zip(*flattened_peaks)
    
        # Create a histogram of the peak values with bins determined by the tolerance
        bin_edges = np.arange(min(peak_values), max(peak_values) + tolerance, tolerance)
        hist, _ = np.histogram(peak_values, bins=bin_edges)
    
        # Find the histogram bins that meet the minimum number of occurrences
        min_occurrences = (min_nr_bins_peakdetection / 100) * len(peaks)
        candidate_bins = np.where(hist >= min_occurrences)[0]
    
        if len(candidate_bins) > 0:
            # Get the peak values corresponding to the candidate bins
            candidate_peaks = (bin_edges[candidate_bins] + bin_edges[candidate_bins + 1]) / 2
    
            # Return the candidate peak with the highest histogram count
            return candidate_peaks[np.argmax(hist[candidate_bins])]
        else:
            return None

    most_occurring_peaks = find_most_occurring_peaks(peaks, min_nr_bins_peakdetection)

    if most_occurring_peaks is None:
        # Handle the case when no peaks satisfy the condition
        # You can set a default value, skip the subsequent steps, or raise an exception
        # depending on your requirements
        # For example, you can set a default value:
        most_occurring_peaks = np.median(peak_frame['Peak'])

    # Create a new peak_frame with the closest peaks to most_occurring_peaks for each sample
    updated_peak_frame = np.empty(len(peaks), dtype=[('Bin', int), ('Peak', float)])
    for i, (break_, peak_list) in enumerate(peaks.items()):
        if isinstance(break_, int) and break_ < len(updated_peak_frame):
            if len(peak_list) > 0:
                closest_peak_index = np.argmin(np.abs(peak_list - most_occurring_peaks))
                updated_peak_frame[i] = (break_, peak_list[closest_peak_index])
            else:
                # Handle empty peak_list
                updated_peak_frame[i] = (break_, np.nan)  # Assign a default value or handle it as needed
        else:
            # Handle invalid break_ values
            updated_peak_frame[i] = (-1, np.nan)  # Assign a default value or handle it as needed
    
    # Filter out any rows with invalid bin values
    valid_rows = updated_peak_frame['Bin'] != -1
    updated_peak_frame = updated_peak_frame[valid_rows]
    return updated_peak_frame

def timegate_peak_detection(bin_data, remove_zeros, peak_removal=1/3, smoothing=2, max_peak=None, window_size=10):
    if remove_zeros:
        bin_data = bin_data[bin_data != 0]
    
    if len(bin_data) < 3:
        return np.array([])
    
    # Calculate histogram
    hist, bin_edges = np.histogram(bin_data, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Smooth the histogram
    smoothed_hist = np.convolve(hist, np.ones(smoothing), mode='same') / smoothing
    
    # Find local maxima with a window size of 10
    max_filter = maximum_filter1d(smoothed_hist, size=2 * window_size + 1, mode='constant')
    maxima = (smoothed_hist == max_filter) & (smoothed_hist > np.roll(smoothed_hist, 1)) & (smoothed_hist > np.roll(smoothed_hist, -1))
    maxima[:window_size] = maxima[-window_size:] = False  # Ensure boundaries are not considered as maxima
    
    # Determine the maximum peak value from the original histogram if not provided
    if max_peak is None: 
        max_peak = np.max(hist)
    
    # Filter peaks based on the original histogram values
    peak_indices = np.where(maxima)[0]
    filtered_peak_indices = peak_indices[bin_centers[peak_indices] > peak_removal * max_peak]
    
    # Get the peak positions from the bin centers
    peak_values = bin_centers[filtered_peak_indices]
    return peak_values
    
def RemovedBins(breaks, outlier_bins, nr_cells):
    if np.any(outlier_bins):
        removed_cells = np.concatenate([breaks[i] for i in np.where(outlier_bins)[0]])
        removed_cells = np.unique(removed_cells)
    else:
        removed_cells = np.array([], dtype=int)

    bad_cells = np.ones(nr_cells, dtype=bool)
    bad_cells[removed_cells] = False

    return {
        'cells': bad_cells,
        'cell_ids': removed_cells
    }

def SplitWithOverlap(vec, seg_length, overlap):
    starts = np.arange(0, len(vec), seg_length - overlap)
    ends = starts + seg_length
    ends[ends > len(vec)] = len(vec)
    
    return [vec[start:end] for start, end in zip(starts, ends)]

def SplitWithOverlapMids(vec, seg_length, overlap):
    starts = np.arange(0, len(vec), seg_length - overlap)
    ends = starts + seg_length
    ends[ends > len(vec)] = len(vec)
    
    mids = np.array([start + int(np.ceil(overlap / 2)) for start in starts])
    mids = mids[mids < len(vec)]
    
    return mids

def MakeBreaks(events_per_bin, nr_events):
    breaks = SplitWithOverlap(np.arange(nr_events), events_per_bin, int(np.ceil(events_per_bin / 2)))
    return {
        'breaks': breaks,
        'events_per_bin': events_per_bin
    }

def FindEventsPerBin(arr, channels, remove_zeros, min_cells, max_bins, step):
    nr_events = arr.shape[0]
    
    if remove_zeros:
        max_bins_mass = min(np.sum(arr[:, channels] != 0, axis=0)) / min_cells
        if max_bins_mass < max_bins:
            max_bins = max_bins_mass
    
    max_cells = int(np.ceil((nr_events / max_bins) * 2))
    max_cells = ((max_cells // step) * step) + step
    
    events_per_bin = max(min_cells, max_cells)
    
    return events_per_bin

def MAD_excluder(peaks, outlier_bins, MAD, breaks, nr_cells):
    names_bins = np.where(outlier_bins)[0]

    x = np.arange(len(peaks))
    
    # Using a lower smoothing factor to speed up the computation
    spline = UnivariateSpline(x, peaks, s=0.2 * len(peaks) )  # Adjust 's' as needed for balance
    kernel_y = spline(x)
    median_peak = np.median(kernel_y)
    mad_peak = np.median(np.abs(kernel_y - median_peak))

    upper_interval = median_peak + MAD * mad_peak
    lower_interval = median_peak - MAD * mad_peak
    to_remove_bins = (kernel_y > upper_interval) | (kernel_y < lower_interval)
    contribution_MAD = []
    removed = RemovedBins([breaks[i] for i in names_bins], to_remove_bins, nr_cells)
    contribution_MAD.append(round((len(removed['cell_ids']) / nr_cells) * 100, 2))

    contribution_MAD = np.array(contribution_MAD)

    return {
        "MAD_bins": to_remove_bins,
        "Contribution_MAD": contribution_MAD
    }
    
def safe_concatenate(arrays):
    non_empty_arrays = [arr for arr in arrays if len(arr) > 0]
    if len(non_empty_arrays) > 0:
        return np.concatenate(non_empty_arrays).astype(int)
    else:
        return None

def peak_width_debris(smoothed_hist, peak_indices, bin_edges, percentage_cells_present=5, smoothing_window=2):
    num_bins = len(smoothed_hist)
    num_peaks = len(peak_indices)
    
    # Apply threshold to the smoothed histogram
    non_zero_bins = smoothed_hist[smoothed_hist != 0]
    if len(non_zero_bins) == 0:
        return []
    threshold = np.percentile(non_zero_bins, 0.05 * 100)
    smoothed_hist[smoothed_hist < threshold] = 0
    
    peak_widths = []
    
    for peak_index in peak_indices:
        left_minima_index = peak_index
        right_minima_index = peak_index
        
        # Find left minima
        while left_minima_index > 0:
            if (smoothed_hist[left_minima_index] < smoothed_hist[left_minima_index - smoothing_window:left_minima_index]).all():
                break
            left_minima_index -= 1
        
        # Find right minima
        while right_minima_index < num_bins - 1:
            if (smoothed_hist[right_minima_index] < smoothed_hist[right_minima_index + 1:right_minima_index + smoothing_window + 1]).all():
                break
            right_minima_index += 1        
        
        peak_percentage = np.sum(smoothed_hist[left_minima_index:right_minima_index + 1]) / np.sum(smoothed_hist) * 100
        left_boundary = bin_edges[left_minima_index]
        right_boundary = bin_edges[right_minima_index]
        
        if peak_percentage >= percentage_cells_present:
            peak_widths.append((left_boundary, right_boundary))
            
    peak_widths = sorted(peak_widths, key=lambda x: x[0])  # Sort based on left boundary
    return peak_widths
    
def peak_detection_debris(data, min_peaks=2, max_peaks=3, smoothing_window=2, percentage_cells_present=5):
    num_features, num_samples = data.shape
    num_bins = 100
    data_transformed = np.arcsinh(data / 150)
    
    peaks = np.zeros((num_features, max_peaks))
    fsc_thresholds = np.zeros(num_features)
    valid_peaks = []
    
    for i in range(num_features):
        min_val, max_val = np.min(data_transformed[i]), np.max(data_transformed[i])
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hist, _ = np.histogram(data_transformed[i], bins=bin_edges, density=True)
        
        # Set the bottom 2% and top 2% of bins to 0
        bottom_bins = int(num_bins * 0.02)
        top_bins = int(num_bins * 0.98)
        hist[:bottom_bins] = 0
        hist[top_bins:] = 0
        
        smoothed_hist = np.convolve(hist, np.ones(smoothing_window) / smoothing_window, mode='same')
        maxima = (smoothed_hist > np.roll(smoothed_hist, 1)) & (smoothed_hist > np.roll(smoothed_hist, -1))
        maxima[:smoothing_window] = maxima[-smoothing_window:] = False
    
        if np.sum(maxima) > 1:
            peak_indices = np.where(maxima)[0]
            peak_densities = [smoothed_hist[idx] for idx in peak_indices]
            peak_info = sorted(zip(peak_indices, peak_densities), key=lambda x: x[1], reverse=True)[:max_peaks]

            largest_peak_indices = [peak[0] for peak in peak_info]
            peaks[i, :len(largest_peak_indices)] = bin_centers[largest_peak_indices]

            peak_widths = peak_width_debris(smoothed_hist, largest_peak_indices, bin_edges, percentage_cells_present)
            valid_peaks.append(peak_widths)
        else:
            valid_peaks.append([])
            
    valid_peaks_mask = np.array([len(peaks) >= min_peaks for peaks in valid_peaks])
    
    return valid_peaks_mask, valid_peaks

def find_fsc_threshold(feature, peaks, smoothing_window):
    num_bins = 300
    smoothing_window = smoothing_window
    
    min_val, max_val = np.min(feature), np.max(feature)
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    hist, _ = np.histogram(feature, bins=bin_edges, density=True)
    
    # Set the bottom 2% and top 2% of bins to 0
    bottom_bins = int(num_bins * 0.02)
    top_bins = int(num_bins * 0.98)
    hist[:bottom_bins] = 0
    hist[top_bins:] = 0
    
    smoothed_hist = np.convolve(hist, np.ones(smoothing_window), mode='same')
    
    # Apply threshold to the smoothed histogram
    non_zero_bins = smoothed_hist[smoothed_hist != 0]
    if len(non_zero_bins) == 0:
        return None
    threshold = np.percentile(non_zero_bins, 1)
    smoothed_hist[smoothed_hist < threshold] = 0

    maxima = (smoothed_hist > np.roll(smoothed_hist, 1)) & (smoothed_hist > np.roll(smoothed_hist, -1))
    maxima[:smoothing_window] = maxima[-smoothing_window:] = False
    peak_indices = np.where(maxima)[0]
    
    if len(peak_indices) > 1:
        peak_densities = [np.sum(smoothed_hist[max(0, idx - smoothing_window):min(idx + smoothing_window + 1, len(smoothed_hist))]) for idx in peak_indices]
        max_peak_index = np.argmax(peak_densities)
        
        if max_peak_index == 1:  # If the biggest peak is the second peak
            left_minima_index = peak_indices[max_peak_index]
            while left_minima_index > 0:
                if (smoothed_hist[left_minima_index] < smoothed_hist[left_minima_index - smoothing_window:left_minima_index]).all():
                    break
                left_minima_index -= 1
            return bin_edges[left_minima_index]
        else:
            # If the max_peak is not the second last peak, we need to be sure that the peaks that are selected are the ones that 
            # have a minima below all the preceding maxima 
            for i in range(max_peak_index - 1, -1, -1):
                left_minima_index = peak_indices[i]
                while left_minima_index > 0:
                    if (smoothed_hist[left_minima_index] < smoothed_hist[left_minima_index - smoothing_window:left_minima_index]).all():
                        break
                    left_minima_index -= 1
                if bin_edges[left_minima_index] < bin_edges[peak_indices[max_peak_index]]:
                    return bin_edges[left_minima_index]
    
    return None

class autocleanup:
    def __init__(self, remove_zeros=True, min_cells=150, max_bins=500, step=200, MAD=6, IT_limit=0.6,
                 peak_detection_smoothing=2, spectral=False, mad=5, min_peaks=2, max_peaks=3, smoothing_window=3,
                 percentage_cells_present=3, time_channel_index=None):
        self.remove_zeros = remove_zeros
        self.min_cells = min_cells
        self.max_bins = max_bins
        self.step = step
        self.MAD = MAD
        self.IT_limit = IT_limit
        self.peak_detection_smoothing = peak_detection_smoothing
        self.spectral = spectral
        self.mad = mad
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks
        self.smoothing_window = smoothing_window
        self.percentage_cells_present = percentage_cells_present
        self.time_channel_index = time_channel_index
        self.skip_doublet_removal = False
        self.skip_debris_removal = False
        self.standardized_names = None

    def process_fcs_data(self, marker_names, fcs_array):
        if len(marker_names) != fcs_array.shape[1]:
            raise ValueError("The number of marker names does not match the number of dimensions in the FCS array.")

        self.standardized_names = [standardize_marker_name(name) for name in marker_names]
        
        # Remove events at the limit of detection
        fcs_array, lod_vector = self.remove_limit_of_detection_events(fcs_array, marker_names)
        
        all_params_present, only_area_params = check_required_parameters(marker_names)
        self.skip_doublet_removal = only_area_params

        events_in_bottom_bin = check_events_in_bottom_bin(fcs_array, marker_names)
        self.skip_debris_removal = not events_in_bottom_bin

        # Initialize vectors
        debris_vector = np.ones(fcs_array.shape[0], dtype=int)
        timegate_vector = np.ones(fcs_array.shape[0], dtype=int)
        doublet_vector = np.ones(fcs_array.shape[0], dtype=int)

        if not self.skip_debris_removal:
            filtered_cells, fsc_gate_threshold, debris_vector = self.automatic_debris_removal(fcs_array)
        else:
            filtered_cells = fcs_array

        if self.time_channel_index is not None:
            filtered_cells, timegate_vector = self.apply_timegate(filtered_cells)

        if not self.skip_doublet_removal:
            doublet_filtered_cells, doublet_vector = self.doublet_removal(filtered_cells)
        else:
            doublet_filtered_cells = filtered_cells

        # Combine all vectors
        final_vector = lod_vector & debris_vector & timegate_vector & doublet_vector

        return fcs_array, final_vector, lod_vector, debris_vector, timegate_vector, doublet_vector

    def remove_limit_of_detection_events(self, fcs_array, marker_names):
        try:
            fsca_column = self.standardized_names.index('fsca')
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

    def automatic_debris_removal(self, fcs_array):
        if self.skip_debris_removal:
            return fcs_array, None, np.ones(fcs_array.shape[0], dtype=int)

        valid_peaks_mask, valid_peaks = peak_detection_debris(fcs_array.T, min_peaks=self.min_peaks, max_peaks=self.max_peaks,
                                                               smoothing_window=self.smoothing_window,
                                                               percentage_cells_present=self.percentage_cells_present)
        
        fsc_column = self.standardized_names.index('fsca')
        fsc_thresholds = [find_fsc_threshold(fcs_array[:, fsc_column], peaks, self.smoothing_window) for peaks in valid_peaks if peaks]
        
        fsc_gate_threshold = np.nanmedian(fsc_thresholds)
        debris_vector = (fcs_array[:, fsc_column] >= fsc_gate_threshold).astype(int)
        filtered_cells = fcs_array[debris_vector == 1]
        
        return filtered_cells, fsc_gate_threshold, debris_vector

    def apply_timegate(self, filtered_cells):
        if self.time_channel_index is None:
            return filtered_cells, np.ones(filtered_cells.shape[0], dtype=int)

        time_channel = filtered_cells[:, self.time_channel_index]
        
        # Apply your timegate logic here
        # For example, let's assume we want to keep events within 2 standard deviations of the mean
        mean_time = np.mean(time_channel)
        std_time = np.std(time_channel)
        lower_bound = mean_time - 2 * std_time
        upper_bound = mean_time + 2 * std_time
        
        timegate_vector = ((time_channel >= lower_bound) & (time_channel <= upper_bound)).astype(int)
        filtered_cells = filtered_cells[timegate_vector == 1]
        
        return filtered_cells, timegate_vector

    def doublet_removal(self, filtered_cells):
        if self.skip_doublet_removal:
            return filtered_cells, np.ones(filtered_cells.shape[0], dtype=int)

        fsc_a_column = self.standardized_names.index('fsca')
        fsc_h_column = self.standardized_names.index('fsch')
        ssc_a_column = self.standardized_names.index('ssca')
        ssc_h_column = self.standardized_names.index('ssch')

        fsc_ratio = filtered_cells[:, fsc_a_column] / filtered_cells[:, fsc_h_column]
        fsc_median_ratio = np.median(fsc_ratio)
        fsc_mad = np.median(np.abs(fsc_ratio - fsc_median_ratio))
        fsc_threshold = fsc_median_ratio + self.mad * fsc_mad
        
        ssc_ratio = filtered_cells[:, ssc_a_column] / filtered_cells[:, ssc_h_column]
        ssc_median_ratio = np.median(ssc_ratio)
        ssc_mad = np.median(np.abs(ssc_ratio - ssc_median_ratio))
        ssc_threshold = ssc_median_ratio + self.mad * ssc_mad
        
        doublet_vector = ((fsc_ratio <= fsc_threshold) & (ssc_ratio <= ssc_threshold)).astype(int)
        doublet_filtered_cells = filtered_cells[doublet_vector == 1]
        
        return doublet_filtered_cells, doublet_vector

    def export_to_csv(self, output_file, fcs_array, marker_names, final_vector, lod_vector, debris_vector, timegate_vector, doublet_vector):
        # Create a DataFrame with the original data and the filter vectors
        df = pd.DataFrame(fcs_array, columns=marker_names)
        
        # Add individual filter columns
        df['passed_lod'] = lod_vector.astype(bool)
        df['passed_debris'] = debris_vector.astype(bool)
        df['passed_timegate'] = timegate_vector.astype(bool)
        df['passed_doublet'] = doublet_vector.astype(bool)
        df['passed_all_filters'] = final_vector.astype(bool)

        # Export to CSV
        df.to_csv(output_file, index=False)
        print(f"Data exported to {output_file}")

    def process_and_export(self, marker_names, fcs_array, output_file):
        results = self.process_fcs_data(marker_names, fcs_array)
        self.export_to_csv(output_file, *results)
        return results