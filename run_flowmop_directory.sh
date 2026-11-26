#!/bin/bash
# run_flowmop_directory.sh - process all FCS/Parquet files in a folder with FlowMOP.
#
# This wrapper keeps the old batch command. It forwards to the FlowMOP command
# line (flowmop_exec.py), which runs files in parallel with no GNU Parallel and
# does not change the input files. New use: flowmop <folder> --output-dir <dir>.
#
# Environment variables:
#   FLOWMOP_SCRIPT  Path to flowmop_exec.py (default: next to this script)
#   PYTHON          Python interpreter (default: python3)

set -euo pipefail

# Builtins only, so that --help works with no programs on PATH.
SOURCE="${BASH_SOURCE[0]}"
if [[ "$SOURCE" == */* ]]; then SCRIPT_DIR="${SOURCE%/*}"; else SCRIPT_DIR="."; fi
SCRIPT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
FLOWMOP_SCRIPT="${FLOWMOP_SCRIPT:-$SCRIPT_DIR/flowmop_exec.py}"
PYTHON="${PYTHON:-python3}"

usage() {
    local text
    IFS= read -r -d '' text <<EOF || true
Usage: $0 --input-dir <dir> --output-dir <dir> [OPTIONS]

Process multiple FCS/parquet files in parallel using FlowMOP.

Required arguments:
  --input-dir <dir>           Folder with the input files (searched recursively)
  --output-dir <dir>          Folder for the output files, logs and summary

Batch options:
  --max-parallel <num>        Number of files to process at the same time (default: 4)
  --file-pattern <pattern>    Use only file names that match the pattern
                              (default: all .fcs and .parquet files)
  --no-skip-processed         Process all files, also files with an output
  --dry-run                   List the files to process and stop
  --verbose                   Also show the full processing detail

All other options go to FlowMOP without change, for example:
  --fluor-mode <mode>  --mad-smoothing '<a> <b>'  --mad-factor <num>
  --skip-debris  --skip-time  --skip-doublets  --enable-plots  --export-filtered-fcs
For the full list, run: $PYTHON $FLOWMOP_SCRIPT --help

Examples:
  $0 --input-dir ./data --output-dir ./output
  $0 --input-dir ./data --output-dir ./output --max-parallel 8 --export-filtered-fcs
EOF
    printf '%s' "$text"
}

INPUT_DIR=""
OUTPUT_DIR=""
JOBS=4
SKIP_EXISTING=1
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir) INPUT_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max-parallel) JOBS="$2"; shift 2 ;;
        --no-skip-processed) SKIP_EXISTING=0; shift ;;
        --file-pattern)
            # The old default "*.fcs" also included Parquet files.
            if [[ "$2" != "*.fcs" ]]; then ARGS+=(--file-pattern "$2"); fi
            shift 2 ;;
        --mad-smoothing)
            # The old script took the values as one quoted string, for example '0.01 0.05'.
            read -r -a VALUES <<< "$2"
            ARGS+=(--mad-smoothing "${VALUES[@]}")
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: --input-dir is required (see --help)" >&2
    exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output-dir is required (see --help)" >&2
    exit 1
fi
if [[ $SKIP_EXISTING -eq 1 ]]; then
    ARGS+=(--skip-existing)
fi

exec "$PYTHON" "$FLOWMOP_SCRIPT" "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --jobs "$JOBS" ${ARGS[@]+"${ARGS[@]}"}
