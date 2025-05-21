#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 <input_dir> <output_dir> [batch_process_flowmop.py_options...]"
    echo ""
    echo "Processes all .fcs and .parquet files in <input_dir> using batch_process_flowmop.py"
    echo "and saves the output to <output_dir>."
    echo ""
    echo "Arguments:"
    echo "  <input_dir>         Directory containing input .fcs or .parquet files."
    echo "  <output_dir>        Directory where processed files will be saved."
    echo "  [options...]        Optional arguments to pass to batch_process_flowmop.py."
    echo "                      For example: --skip-debris --fluor-mode geomean --min-cells 500"
    echo ""
    echo "Example:"
    echo "  $0 ./my_fcs_files/ ./processed_output/ --skip-debris --enable-plots --n-workers 8 --max-parallel-files 4"
    exit 1
}

# Check if input_dir and output_dir are provided
if [ "$#" -lt 2 ]; then
    echo "Error: Input directory and output directory are required."
    usage
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
shift 2 # Remove the first two arguments (input_dir and output_dir)
BATCH_OPTIONS=("$@") # Remaining arguments are options for batch_process_flowmop.py

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' not found."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
echo "Output directory set to: $OUTPUT_DIR"

# Construct the command
CMD="python src/batch_process_flowmop.py \"$INPUT_DIR\" \"$OUTPUT_DIR\""
if [ ${#BATCH_OPTIONS[@]} -gt 0 ]; then
    CMD+=" ${BATCH_OPTIONS[*]}"
fi

echo "-----------------------------------------------------"
echo "Starting batch processing"
echo "Executing: $CMD"
eval "$CMD"
echo "-----------------------------------------------------"

echo "All processing complete."
echo "Output files are in: $OUTPUT_DIR"