# FlowMOP - Flow Cytometry Multi-Operation Pipeline

FlowMOP does automated quality control of flow cytometry data. It applies three gates to FCS and Parquet files:

- a **time gate**, which removes time bins with unusual fluorescence (for example, from air bubbles or blockages);
- a **debris gate**, which removes small, low-FSC events;
- a **doublet gate**, which removes events that are two or more cells.

FlowMOP does not delete events. Each output FCS file keeps all events in their original order and adds `passed_*` columns (1 = passed, 0 = failed), so that you can gate on them in your usual software.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Output](#output)
- [Options](#options)
- [Use FlowMOP from Python](#use-flowmop-from-python)
- [Old batch script](#old-batch-script)
- [Problems and solutions](#problems-and-solutions)
- [For developers](#for-developers)
- [License](#license)

## Install

You need Python 3.9 or later.

1. Optional but recommended: make a separate Python environment for FlowMOP.

   ```bash
   # with conda
   conda create -n flowmop python=3.12
   conda activate flowmop

   # or with venv (Linux and macOS)
   python3 -m venv flowmop-env
   source flowmop-env/bin/activate

   # or with venv (Windows PowerShell)
   py -m venv flowmop-env
   flowmop-env\Scripts\Activate.ps1
   ```

2. Install FlowMOP and its dependencies with one command:

   ```bash
   pip install flowmop
   ```

   To get the newest code from GitHub, use this command (you must have git):

   ```bash
   pip install git+https://github.com/1ordinateur/FlowMOP.git
   ```

   If you have a copy of the FlowMOP folder, run `pip install .` in that folder.

3. Make sure that the install is correct:

   ```bash
   flowmop --version
   ```

## Quick start

Process one file:

```bash
flowmop sample.fcs --output-dir results
```

Process all FCS and Parquet files in a folder and its subfolders, with 4 files at the same time:

```bash
flowmop ./fcs_files --output-dir results --jobs 4
```

The screen shows one line for each file:

```text
FlowMOP 0.3.0: 3 file(s), 3 job(s), output in results
[1/3] 12A.fcs  253,728 events  83.0% kept  10.6 s
[2/3] 11B.fcs  258,952 events  34.0% kept  11.1 s
[3/3] 10A.fcs  463,512 events  66.9% kept  14.7 s
Done: 3 of 3 file(s) processed. Summary: results/flowmop_summary.csv  Logs: results/logs
```

More examples:

```bash
flowmop ./fcs_files -o results --dry-run                 # list the files, process nothing
flowmop ./fcs_files -o results --skip-existing           # continue a run that stopped
flowmop ./fcs_files -o results --file-pattern '*_stained.fcs'
flowmop sample.fcs -o results --skip-debris --skip-doublets   # time gate only
flowmop sample.fcs -o results --enable-plots             # also write time gate plots
```

If you do not give `--output-dir`, FlowMOP writes to `./flowmop_output`. The output folder can also be the input folder. FlowMOP never changes or renames the input files, and it never uses its own output files as inputs.

To stop a run, push Ctrl+C. FlowMOP stops the files that are in progress, writes the summary for the finished files, and exits with code 130. To continue later, run the same command with `--skip-existing`.

## Output

```text
results/
├── flowmop_sample1.fcs      # all events + passed_* columns
├── flowmop_sample2.fcs
├── flowmop_summary.csv      # one row for each input file
├── logs/
│   ├── sample1.log          # full processing detail and warnings
│   └── sample2.log
└── plots/sample1/           # only with --enable-plots
```

### Annotated FCS files

FlowMOP writes one annotated FCS file for each input, with the name `flowmop_<input name>.fcs`. For example, `sample001.fcs` gives `flowmop_sample001.fcs`. The file contains all original events in their original order, and these added channels:

| Column | Description |
|--------|-------------|
| `passed_lod` | Limit of detection gate (1 = passed, 0 = failed) |
| `passed_debris` | Debris gate result |
| `passed_time` | Time gate result |
| `passed_doublet` | Doublet gate result |
| `passed_final` | 1 only when the event passed all gates |

To get the clean events, keep the events with `passed_final = 1`.

### Summary table

`flowmop_summary.csv` has one row for each input file. You can open it in Excel. Each run adds its rows to the summary in the output folder. When a run processes a file again, its new row replaces the old row. A file that `--skip-existing` skips keeps the counts of the run that made its output.

| Column | Description |
|--------|-------------|
| `file` | Input file name |
| `status` | `ok`, `failed` or `skipped` |
| `events` | Number of events in the input |
| `passed_lod`, `passed_debris`, `passed_time`, `passed_doublet` | Number of events that passed each gate |
| `passed_final` | Number of events that passed all gates |
| `retained_percent` | `passed_final` as a percentage of `events` |
| `seconds` | Processing time |
| `output`, `log` | Paths of the output FCS file and the log file |
| `error` | The error message for a failed file |
| `input` | Full path of the input file |

### Log files

The screen shows only the result for each file. The full detail (channels, peaks, warnings, errors) goes to `logs/<input name>.log`. Use `--verbose` to also show the detail on the screen.

### Metadata

FlowMOP keeps the non-structural FCS metadata of the input, including the original detector names (`$PnN`), marker labels (`$PnS`), parameter ranges, and the serialized spillover matrix. FlowMOP uses the marker labels internally, but it does not replace the detector names in the output files. FlowMOP also adds processing metadata:

- `flowmop_processed`: Processing flag
- `flowmop_processing_date`: ISO timestamp
- `flowmop_original_file`: Source file path
- `flowmop_fluor_mode`: Mode used for fluorescence analysis
- `flowmop_events_original`: Original event count
- `flowmop_events_final`: Events that passed all gates
- `flowmop_retention_percent`: Percentage of events kept
- Statistics for each gate (`flowmop_<gate>_passed`, `flowmop_<gate>_percent`)

FlowMOP makes the structural fields, for example the byte offsets, again as necessary. The `passed_*` channels come after the original parameter table, and the metadata of the original parameters does not change.

### Filtered FCS files (optional)

With `--export-filtered-fcs`, FlowMOP also writes FCS files that contain only the events that passed specified gates. These files contain only the original input parameters, so that you can compare them directly with the clean output of other quality-control tools:

| Output folder | Events in the file |
|---------------|--------------------|
| `passfiltered/` | Passed all gates (`passed_final`) |
| `timepass/` | Passed the time and LOD gates |
| `debrispass/` | Passed the debris and LOD gates |
| `doubletpass/` | Passed the doublet and LOD gates |

To make these files later from annotated FCS files, run `python -m flowmop.output_complete --help`.

## Options

Run `flowmop --help` for the full list. The gate defaults are the values that the FlowMOP paper used.

### Run options

| Option | Default | Description |
|--------|---------|-------------|
| `INPUT ...` | required | FCS or Parquet files, or folders (searched recursively) |
| `-o`, `--output-dir` | `./flowmop_output` | Folder for the output files, logs and summary |
| `-j`, `--jobs` | `1` | Number of files to process at the same time |
| `--file-pattern` | all `.fcs` and `.parquet` | In folders, use only file names that match this pattern |
| `--skip-existing` | off | Do not process an input when its output FCS exists |
| `--dry-run` | off | List the files to process and stop |
| `-v`, `--verbose` | off | Also show the full processing detail on the screen |
| `--version` | | Show the FlowMOP version |

### Output options

| Option | Default | Description |
|--------|---------|-------------|
| `--no-output-fcs` | off | Process the files, but do not write the annotated FCS files |
| `--export-filtered-fcs` | off | Also write filtered FCS files (see above) |
| `--filtered-output-dir` | output folder | Folder for the filtered FCS files |
| `--enable-plots` | off | Write time gate diagnostic plots |
| `--plots-dir` | `<output-dir>/plots` | Folder for the plots; each input gets a subfolder |

### Gate options

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-debris` | off | Do not apply the debris gate |
| `--skip-time` | off | Do not apply the time gate |
| `--skip-doublets` | off | Do not apply the doublet gate |
| `--fluor-mode` | `positives` | Fluorescence mode of the time gate: `positives`, `peaks`, `geomean` or `both` |
| `--mad-smoothing` | `0.01 0.05` | Spline smoothing factors for the time gate |
| `--mad-factor` | `5` | MAD threshold for the time gate |
| `--min-cells` | `1000` | Minimum number of events in a time bin |
| `--max-bins` | `600` | Maximum number of time bins |
| `--step-val` | `200` | Step size for the time bins |
| `--enable-ssc` | off | Also use SSC-A in the debris gate |
| `--remove-beads` | off | Find and remove beads from SSC/FSC |
| `--disable-remove-zeros` | off | Keep zero values (FlowMOP removes them by default) |
| `--disable-dask` | off | Do not run the gates of one file in parallel |

Fluorescence modes:

- `positives` (default): the mode of the FlowMOP paper. Global thresholds find the positive events, and the time gate uses the geometric mean of the positive events in each time bin.
- `peaks`: find the positive peaks again in each time bin.
- `geomean`: use the geometric mean of all events in each time bin.
- `both`: use `peaks` and `geomean` together. FlowMOP removes an event only when two or more checks reject it.

Names before version 0.3.0: the `positives` mode had the name `positive_geomeans`, and the `peaks` mode had the name `positives`. FlowMOP still accepts `positive_geomeans`. If an old command or script uses `--fluor-mode positives` and you want the old peak behaviour, change it to `--fluor-mode peaks`.

## Use FlowMOP from Python

```python
from flowmop import process_file

summary = process_file("sample.fcs", output_dir="results")
print(summary["passed_final"], summary["retained_percent"])
```

`process_file()` has the same defaults as the `flowmop` command. It returns a dict with the counts of the summary table and the path of the output FCS file.

The lower-level `flowmop.FlowMOP` class works on a NumPy array. Its defaults are different from the defaults of `process_file()` (`fluor_mode="peaks"`, `min_cells=150`, `max_bins=500`). To get the same result as the `flowmop` command, use `process_file()`, or give the class the values from the [gate options](#gate-options).

## Old batch script

The old commands continue to work:

```bash
python /path/to/FlowMOP/flowmop_exec.py sample.fcs --output-dir results
./run_flowmop_directory.sh --input-dir ./fcs_files --output-dir results --max-parallel 4
```

`run_flowmop_directory.sh` now sends its options to the `flowmop` command line. It does not need GNU Parallel, and it does not rename input files. As before, it skips the files that have an output; use `--no-skip-processed` to process them again.

## Problems and solutions

| Message | Cause and solution |
|---------|--------------------|
| `flowmop: command not found` | The Python environment with FlowMOP is not active. Activate it, or install FlowMOP again (see [Install](#install)). |
| `flowmop: error: not found: <path>` | The path is not correct. Look for typing errors, and put quotes around a path with spaces. |
| `these inputs give the same output name` | Two inputs in different folders have the same file name. Rename one of them, or process the folders one at a time with different output folders. |
| `FAILED: ...` on one line | FlowMOP could not process that file. The other files continue. The log file named on the line contains the full error. |
| `FAILED: the worker process stopped ...` | The system stopped the process of that file, usually because there was not sufficient memory. Use a smaller `--jobs` value. |
| `flowmop: error: the output path is not a folder` | The `--output-dir` path is a file. Give a folder path. |
| Processing is slow | Use `--jobs` with a number up to the number of CPU cores. Each job holds one file in memory (approximately 0.7 GB for 460,000 events with 35 channels). |

FlowMOP finds the time channel from its name: the first channel with `time` in its name. It finds the scatter channels from the names `FSC-A`, `FSC-H`, `SSC-A` and `SSC-H`. If a file has no FSC-A channel, FlowMOP does not apply the limit of detection gate to it.

## For developers

```bash
pip install -e ".[test]"
pytest
```

Module structure:

```text
flowmop_exec.py              # old entry point (calls flowmop.cli)
run_flowmop_directory.sh     # old batch entry point (calls flowmop_exec.py)
flowmop/
├── cli.py                   # command line: inputs, parallel jobs, logs, summary
├── process.py               # process_file(): read, gate and write one file
├── pipeline.py              # FlowMOP class: runs the gates on an array
├── time_gating.py           # time gate
├── debris_gating.py         # debris gate (FSC-based)
├── doublet_gating.py        # doublet gate
├── utils.py                 # histogram and smoothing utilities
└── output_complete.py       # make filtered FCS files from annotated files
```

Notes:

- FlowMOP also reads Parquet files. A Parquet file has no FCS metadata, so FlowMOP makes minimal metadata for its output.
- For FCS files, FlowMOP uses the marker names when they exist, and the channel names when they do not. The scatter channels (FSC/SSC) always use the channel name.
- By default, FlowMOP runs the gates of one file in parallel. Use `--disable-dask` for debugging, or when you use many `--jobs`.
- The tests in `tests/test_benchmark_*.py` need the benchmark scripts of the FlowMOP paper repository. In a separate copy of FlowMOP, pytest skips them.

## License

FlowMOP has the MIT License. The full text is in the `LICENSE` file.
