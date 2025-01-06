# Flow Cytometry Data Cleanup and Analysis

This repository provides a set of tools for the automated cleanup and analysis of flow cytometry data, with a focus on peak detection, debris removal, and doublet removal.

## Features

- **Time Gating**: Segment data based on time and identify peaks in each segment.
- **Debris Removal**: Automatically detect and remove debris based on peak analysis and thresholds.
- **Doublet Removal**: Identify and filter out doublet events using FSC and SSC ratios.
- **Flexible Parameters**: Customize the cleanup process with various parameters for peak detection and outlier removal.

## Requirements

- Python 3.x
- numpy
- scipy

## Installation

Install the required packages using pip:

```bash
pip install numpy scipy
```

## Methods Overview

### `timegate` Method

| Argument | Type         | Description                          |
|----------|--------------|--------------------------------------|
| arr      | numpy.ndarray| Input flow cytometry data array.     |

### `automatic_debris_removal` Method

| Argument | Type         | Description                                  |
|----------|--------------|----------------------------------------------|
| fcs_array| numpy.ndarray| Time-gated flow cytometry data array.        |

### `doublet_removal` Method

| Argument      | Type         | Description                                      |
|---------------|--------------|--------------------------------------------------|
| filtered_cells| numpy.ndarray| Debris-removed flow cytometry data array.        |


## `autocleanup` Class Arguments

| Argument                  | Type  | Description                                                         |
|---------------------------|-------|---------------------------------------------------------------------|
| remove_zeros              | bool  | Whether to remove zero values from the data.                        |
| min_cells                 | int   | Minimum number of cells per bin.                                    |
| max_bins                  | int   | Maximum number of bins.                                             |
| step                      | int   | Step size for bin range.                                            |
| MAD                       | float | Median Absolute Deviation (MAD) threshold for outlier detection.    |
| IT_limit                  | float | Threshold for the IT (Intensity-Time) limit.                        |
| peak_detection_smoothing  | int   | Smoothing window size for peak detection.                           |
| spectral                  | bool  | Whether to use spectral channels for analysis.                      |
| mad                       | float | Median Absolute Deviation (MAD) threshold for doublet removal.      |
| min_peaks                 | int   | Minimum number of peaks for feature selection.                      |
| max_peaks                 | int   | Maximum number of peaks for feature selection.                      |
| smoothing_window          | int   | Smoothing window size for feature selection.                        |
| percentage_cells_present  | float | Percentage of cells present for peak width calculation.             |
| time_channel_index        | int   | Index of the time channel in the data array.                        |

### Usage

Time Gating: 
```python
time_gated_data = cleaner.timegate(data_array)
```

Doublet Gating: 
```python
final_data = cleaner.doublet_removal(debris_removed_data)
```

Debris Removal (FSC-A) Gating: 
```python
debris_removed_data, fsc_threshold = cleaner.automatic_debris_removal(data)
```