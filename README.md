# FlowMOP - Flow Cytometry Multi-Operation Pipeline

A command-line tool for automated quality control of flow cytometry data. FlowMOP processes FCS and Parquet files through a gating pipeline that removes debris, doublets, and time-based anomalies.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Batch Processing](#batch-processing)
- [Usage](#usage)
- [CLI Arguments](#cli-arguments)
- [Output Files](#output-files)
- [Architecture](#architecture)
- [Notes](#notes)
- [Related Scripts](#related-scripts)

## Installation

### Dependencies

FlowMOP requires the following Python packages:

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `readfcs` | FCS file reading |
| `fcswrite` | FCS file writing |
| `scipy` | Histogram smoothing, interpolation, and statistics |
| `matplotlib` | Diagnostic plot generation |
| `dask` | Parallel array operations |
| `distributed` | Dask distributed scheduler |
| `fcsparser` | Optional fallback FCS reader |
| `pyarrow` | Parquet file support through pandas |

**Note:** `distributed` is installed separately from `dask`.

### Install Command

```bash
pip install -r requirements.txt
```

## Quick Start

Process a single FCS file:

```bash
python flowmop_exec.py sample.fcs --output-dir ./output
```

For batch processing of multiple files, see [Batch Processing](#batch-processing) below.

## Batch Processing

For processing multiple files in parallel, use `run_flowmop_directory.sh`:

```bash
./run_flowmop_directory.sh --input-dir ./fcs_files --output-dir ./output
```

### Batch Processing Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | required | Directory containing input FCS/Parquet files |
| `--output-dir` | required | Directory for output files |
| `--file-pattern` | `*.fcs` | Glob pattern to match files |
| `--max-parallel` | `4` | Number of parallel workers |
| `--no-skip-processed` | false | Reprocess files that already have output |
| `--dry-run` | false | List files that would be processed without running |
| `--verbose` | false | Show detailed output |

All FlowMOP options (`--fluor-mode`, `--mad-smoothing`, etc.) are passed through to `flowmop_exec.py`.

### Example

```bash
./run_flowmop_directory.sh \
    --input-dir ./data \
    --output-dir ./output \
    --max-parallel 8 \
    --fluor-mode positive_geomeans \
    --mad-factor 5
```

### Requirements

- **GNU Parallel** must be installed (`apt install parallel` or `brew install parallel`)

## Usage

```bash
python flowmop_exec.py [files] [options]
```

## CLI Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `files` | One or more paths to data files (FCS or Parquet format) |

### Output Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | None | Directory to save output files |
| `--export-filtered-fcs` | False | Also emit filtered FCS files (subsetted events) |
| `--filtered-output-dir` | output-dir | Directory for filtered FCS files |

### Filtering Controls

| Argument | Description |
|----------|-------------|
| `--skip-debris` | Skip debris filtering |
| `--skip-time` | Skip time filtering |
| `--skip-doublets` | Skip doublet filtering |

### Fluorescence Mode

| Argument | Default | Options |
|----------|---------|---------|
| `--fluor-mode` | `positive_geomeans` | `positives`, `geomean`, `positive_geomeans`, `both` |

- **positives**: Detect positive peaks in each time bin
- **geomean**: Calculate geometric mean across all channels
- **positive_geomeans**: Use global thresholds and track geometric mean of positive cells
- **both**: Combine positives and geomean approaches

### Gating Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--mad-smoothing` | `0.1 0.9` | Smoothing factors for MAD-based time gating (space-separated) |
| `--min-cells` | `1000` | Minimum cells required for processing a bin |
| `--max-bins` | `600` | Maximum number of bins to divide data into |
| `--step-val` | `200` | Step size for binning |
| `--mad-factor` | `4` | Factor for MAD calculation for gating |

### Advanced Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-ssc` | False | Use SSC-A for debris gating in addition to FSC-A |
| `--remove-beads` | False | Detect and remove beads based on SSC/FSC characteristics |
| `--disable-remove-zeros` | False | Disable removal of zero values (zeros removed by default) |
| `--disable-dask` | False | Disable within-file gate parallelism |

### Plotting Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-plots` | False | Generate time gate diagnostic plots |
| `--plots-dir` | `time_gate_plots` | Directory to save time gate plots |

## Output Files

### Naming Convention

Output files follow the pattern: `flowmop_<original_name>.fcs`

Example: `sample001.fcs` produces `flowmop_sample001.fcs`

### Added Columns

The output FCS file contains all original channels plus gating result columns:

| Column | Description |
|--------|-------------|
| `passed_lod` | Limit of detection gate (1 = passed, 0 = failed) |
| `passed_debris` | Debris gate result |
| `passed_time` | Time gate result |
| `passed_doublet` | Doublet gate result |
| `passed_final` | Combined result of all gates |

### Metadata Preservation

FlowMOP preserves all original FCS metadata and adds processing metadata:

- `flowmop_processed`: Processing flag
- `flowmop_processing_date`: ISO timestamp
- `flowmop_original_file`: Source file path
- `flowmop_fluor_mode`: Mode used for fluorescence analysis
- `flowmop_events_original`: Original event count
- `flowmop_events_final`: Events passing all filters
- `flowmop_retention_percent`: Percentage of events retained
- Per-filter statistics (`flowmop_<filter>_passed`, `flowmop_<filter>_percent`)

### Filtered Outputs (Optional)

When `--export-filtered-fcs` is enabled, additional FCS files containing only events that passed specific filters are created:

| Output Directory | Filter Criteria |
|------------------|-----------------|
| `passfiltered/` | Events passing all filters (`passed_final`) |
| `timepass/` | Events passing time + LOD filters |
| `debrispass/` | Events passing debris + LOD filters |
| `doubletpass/` | Events passing doublet + LOD filters |

## Architecture

### Module Structure

```
src/
├── flowmop_exec.py              # CLI entry point
├── run_flowmop_directory.sh     # Batch processing script
├── base/
│   ├── flowmop_new.py           # FlowMOP class (main pipeline)
│   └── output_complete_flowmop.py
└── functions/
    ├── time_gating.py           # Time-based anomaly detection
    ├── debris_gating.py         # Debris removal (FSC-based)
    ├── doublet_gating.py        # Doublet detection
    └── flowmop_utils.py         # Utility functions
```

### Key Functions

#### `load_data(file_path) -> (meta_raw, data_frame, channel_names)`

Loads data from FCS or Parquet files. For FCS files, uses `readfcs` to extract data and metadata. Handles channel naming by preferring marker names over channel short names (except for scatter parameters FSC/SSC).

#### `process_file(file_path, output_dir, ...)`

Main processing function that:
1. Loads data via `load_data()`
2. Filters to numerical columns only
3. Initializes `FlowMOP` with configuration
4. Runs the gating pipeline
5. Exports results with metadata preservation

#### `filter_numerical_columns(data) -> DataFrame`

Filters DataFrame to keep only numerical columns. Raises `ValueError` if no numerical columns exist.

#### `write_filtered_fcs_files(base_output_dir, base_name, output_df, meta_raw)`

Creates filtered FCS files containing only events that passed specific gate combinations.

#### `_clean_metadata(meta_raw, set_total_events=None) -> dict`

Cleans and standardizes metadata for FCS output:
- Strips structural byte offsets (recalculated by fcswrite)
- Handles both fcsparser-style (`$KEY`) and readfcs-style (lowercase) keys
- Preserves parameter metadata (`$PnG`, `$PnV`, etc.)

### FlowMOP Class Integration

`flowmop_exec.py` uses the `FlowMOP` class from `base/flowmop_new.py`:

```python
from base import flowmop_new

flowmop = flowmop_new.FlowMOP(
    time_channel_index=time_channel_index,
    remove_zeros=remove_zeros,
    min_cells=min_cells,
    # ... other parameters
)
vectors = flowmop.process_fcs_data(marker_names, fcs_array)
```

The `process_fcs_data()` method returns a dictionary of gate vectors:
- `lod`: Limit of detection gate
- `debris`: Debris gate
- `time`: Time gate
- `doublet`: Doublet gate
- `final`: Combined result

## Notes

### Parquet Support

FlowMOP accepts Parquet files as input. When processing Parquet files, minimal synthetic metadata is created since Parquet files don't contain FCS metadata.

### Within-File Parallelism

By default, FlowMOP can run independent gates in parallel within a file. Disable with `--disable-dask` for debugging, batch jobs that already parallelize by file, or small files where parallel overhead exceeds benefits.

### Channel Naming

For FCS files, FlowMOP uses marker names when available, falling back to channel short names. Scatter parameters (FSC/SSC) always use the channel name since they typically don't have marker assignments.

## Related Scripts

### base/output_complete_flowmop.py

This script is largely redundant with `flowmop_exec.py --export-filtered-fcs`:

| Feature | flowmop_exec.py | output_complete_flowmop.py |
|---------|-----------------|---------------------------|
| Filtered FCS export | `--export-filtered-fcs` flag | Always runs |
| Integration | Inline during processing | Standalone post-processor |

**Use case**: Post-process already-processed FCS files to generate filtered subsets without re-running the full pipeline.

**Recommendation**: Use `flowmop_exec.py --export-filtered-fcs` for new processing.
