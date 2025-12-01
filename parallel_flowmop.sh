#!/bin/bash
# parallel_flowmop.sh - Parallel execution script for FlowMOP
#
# This script processes multiple FCS/parquet files in parallel using GNU Parallel
#
# Usage:
#   ./parallel_flowmop.sh --input-dir /path/to/input --output-dir /path/to/output [OPTIONS]
#
# Environment variables:
#   FLOWMOP_SCRIPT  Path to flowmop_exec.py (default: auto-detected relative to this script)

set -e

# Determine script directory and flowmop_exec.py location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOWMOP_SCRIPT="${FLOWMOP_SCRIPT:-$SCRIPT_DIR/flowmop_exec.py}"

# Function to remove spaces from filenames
remove_spaces() {
    local dir="$1"
    find "$dir" -type f -name "* *" | while read -r file; do
        newname=$(echo "$file" | tr ' ' '_')
        if [ "$file" != "$newname" ]; then
            echo "Renaming: $file -> $newname"
            mv "$file" "$newname"
        fi
    done
}

# Default values
INPUT_DIR=""
OUTPUT_DIR=""
FILE_PATTERN="*.fcs"
MAX_PARALLEL=4
FLUOR_MODE="positive_geomeans"
MAD_SMOOTHING="0.1 0.9"
ENABLE_PLOTS=0
PLOTS_DIR="time_gate_plots"
ENABLE_SSC=0
REMOVE_BEADS=0
SKIP_DEBRIS=0
SKIP_TIME=0
SKIP_DOUBLETS=0
REMOVE_ZEROS=1
MIN_CELLS=1000
MAX_BINS=600
STEP_VAL=200
MAD_FACTOR=4
SKIP_PROCESSED=1
DISABLE_DASK=0
EXPORT_FILTERED=0
FILTERED_OUTPUT_DIR=""
DRY_RUN=0
VERBOSE=0

# Check if parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "Error: GNU Parallel is not installed. Please install it to use this script."
    exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --file-pattern)
            FILE_PATTERN="$2"
            shift 2
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --fluor-mode)
            FLUOR_MODE="$2"
            shift 2
            ;;
        --mad-smoothing)
            MAD_SMOOTHING="$2"
            shift 2
            ;;
        --enable-plots)
            ENABLE_PLOTS=1
            shift
            ;;
        --plots-dir)
            PLOTS_DIR="$2"
            shift 2
            ;;
        --enable-ssc)
            ENABLE_SSC=1
            shift
            ;;
        --remove-beads)
            REMOVE_BEADS=1
            shift
            ;;
        --skip-debris)
            SKIP_DEBRIS=1
            shift
            ;;
        --skip-time)
            SKIP_TIME=1
            shift
            ;;
        --skip-doublets)
            SKIP_DOUBLETS=1
            shift
            ;;
        --disable-remove-zeros)
            REMOVE_ZEROS=0
            shift
            ;;
        --min-cells)
            MIN_CELLS="$2"
            shift 2
            ;;
        --max-bins)
            MAX_BINS="$2"
            shift 2
            ;;
        --step-val)
            STEP_VAL="$2"
            shift 2
            ;;
        --mad-factor)
            MAD_FACTOR="$2"
            shift 2
            ;;
        --no-skip-processed)
            SKIP_PROCESSED=0
            shift
            ;;
        --disable-dask)
            DISABLE_DASK=1
            shift
            ;;
        --export-filtered-fcs)
            EXPORT_FILTERED=1
            shift
            ;;
        --filtered-output-dir)
            FILTERED_OUTPUT_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --help)
            echo "Usage: $0 --input-dir <dir> --output-dir <dir> [OPTIONS]"
            echo ""
            echo "Process multiple FCS/parquet files in parallel using FlowMOP."
            echo ""
            echo "Required arguments:"
            echo "  --input-dir <dir>           Directory containing input files"
            echo "  --output-dir <dir>          Directory to save output files"
            echo ""
            echo "Parallel processing:"
            echo "  --max-parallel <num>        Maximum parallel processes (default: 4)"
            echo "  --file-pattern <pattern>    Glob pattern to match files (default: *.fcs)"
            echo "  --no-skip-processed         Process all files, even if already processed"
            echo "  --dry-run                   List files that would be processed without running"
            echo "  --verbose                   Show detailed output"
            echo ""
            echo "Gating options:"
            echo "  --fluor-mode <mode>         Mode for fluorescence analysis (default: positive_geomeans)"
            echo "                              Options: positives, geomean, positive_geomeans, both"
            echo "  --mad-smoothing <values>    Smoothing factors for MAD-based time gating (default: '0.1 0.9')"
            echo "  --min-cells <num>           Minimum cells required per bin (default: 1000)"
            echo "  --max-bins <num>            Maximum number of bins (default: 600)"
            echo "  --step-val <num>            Step size for binning (default: 200)"
            echo "  --mad-factor <num>          Factor for MAD calculation (default: 4)"
            echo ""
            echo "Gate skipping:"
            echo "  --skip-debris               Skip debris filtering"
            echo "  --skip-time                 Skip time filtering"
            echo "  --skip-doublets             Skip doublet filtering"
            echo "  --disable-remove-zeros      Disable removal of zero values"
            echo ""
            echo "Additional features:"
            echo "  --enable-plots              Generate time gate plots"
            echo "  --plots-dir <dir>           Directory for time gate plots (default: time_gate_plots)"
            echo "  --enable-ssc                Use SSC-A for debris gating in addition to FSC-A"
            echo "  --remove-beads              Detect and remove beads based on SSC/FSC"
            echo "  --disable-dask              Disable Dask parallel processing"
            echo "  --export-filtered-fcs       Export filtered FCS subsets (passfiltered, timepass, etc.)"
            echo "  --filtered-output-dir <dir> Directory for filtered FCS output (default: output-dir)"
            echo ""
            echo "Environment variables:"
            echo "  FLOWMOP_SCRIPT              Path to flowmop_exec.py (default: auto-detected)"
            echo ""
            echo "Examples:"
            echo "  # Basic usage"
            echo "  $0 --input-dir ./data --output-dir ./output"
            echo ""
            echo "  # Process with 8 parallel jobs and export filtered files"
            echo "  $0 --input-dir ./data --output-dir ./output --max-parallel 8 --export-filtered-fcs"
            echo ""
            echo "  # Dry run to see what files would be processed"
            echo "  $0 --input-dir ./data --output-dir ./output --dry-run"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: --input-dir is required"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output-dir is required"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Remove spaces from filenames in input directory
echo "Checking for and removing spaces from filenames..."
remove_spaces "$INPUT_DIR"

# Find files to process
if [[ "$FILE_PATTERN" == "*.fcs" ]]; then
    # Also include parquet files
    FILES=( $(find "$INPUT_DIR" -type f \( -iname "*.fcs" -o -iname "*.parquet" \)) )
else
    FILES=( $(find "$INPUT_DIR" -type f -iname "$FILE_PATTERN") )
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No matching files found in $INPUT_DIR"
    exit 0
fi

# Filter already processed files if requested
if [[ $SKIP_PROCESSED -eq 1 ]]; then
    PROCESSED_FILES=()
    while IFS= read -r file; do
        base_name=$(basename "$file" | sed 's/_processed\.fcs$//')
        PROCESSED_FILES+=("$base_name")
    done < <(find "$OUTPUT_DIR" -name "*_processed.fcs" -type f)
    
    FILTERED_FILES=()
    for file in "${FILES[@]}"; do
        base_name=$(basename "$file" | sed 's/\.[^.]*$//')
        skip=0
        for processed in "${PROCESSED_FILES[@]}"; do
            if [[ "$base_name" == "$processed" ]]; then
                skip=1
                break
            fi
        done
        if [[ $skip -eq 0 ]]; then
            FILTERED_FILES+=("$file")
        fi
    done
    
    skipped=$((${#FILES[@]} - ${#FILTERED_FILES[@]}))
    if [[ $skipped -gt 0 ]]; then
        echo "Skipping $skipped already processed files"
    fi
    
    FILES=("${FILTERED_FILES[@]}")
    
    if [[ ${#FILES[@]} -eq 0 ]]; then
        echo "All files already processed"
        exit 0
    fi
fi

# Build command options
CMD_OPTIONS="--output-dir $OUTPUT_DIR --fluor-mode $FLUOR_MODE --mad-smoothing $MAD_SMOOTHING --plots-dir $PLOTS_DIR --min-cells $MIN_CELLS --max-bins $MAX_BINS --step-val $STEP_VAL --mad-factor $MAD_FACTOR"

if [[ $ENABLE_PLOTS -eq 1 ]]; then
    CMD_OPTIONS="$CMD_OPTIONS --enable-plots"
fi
if [[ $ENABLE_SSC -eq 1 ]]; then
    CMD_OPTIONS="$CMD_OPTIONS --enable-ssc"
fi
if [[ $REMOVE_BEADS -eq 1 ]]; then
    CMD_OPTIONS="$CMD_OPTIONS --remove-beads"
fi
if [[ $SKIP_DEBRIS -eq 1 ]]; then
    CMD_OPTIONS="$CMD_OPTIONS --skip-debris"
fi
if [[ $SKIP_TIME -eq 1 ]]; then
    CMD_OPTIONS="$CMD_OPTIONS --skip-time"
fi
if [[ $SKIP_DOUBLETS -eq 1 ]]; then
    CMD_OPTIONS="$CMD_OPTIONS --skip-doublets"
fi
if [[ $REMOVE_ZEROS -eq 0 ]]; then
    CMD_OPTIONS="$CMD_OPTIONS --disable-remove-zeros"
fi
if [[ $DISABLE_DASK -eq 1 ]]; then
    CMD_OPTIONS="$CMD_OPTIONS --disable-dask"
fi
if [[ $EXPORT_FILTERED -eq 1 ]]; then
    CMD_OPTIONS="$CMD_OPTIONS --export-filtered-fcs"
fi
if [[ -n "$FILTERED_OUTPUT_DIR" ]]; then
    CMD_OPTIONS="$CMD_OPTIONS --filtered-output-dir $FILTERED_OUTPUT_DIR"
fi

# Verify flowmop_exec.py exists
if [[ ! -f "$FLOWMOP_SCRIPT" ]]; then
    echo "Error: Cannot find flowmop_exec.py at: $FLOWMOP_SCRIPT"
    echo "Set FLOWMOP_SCRIPT environment variable to override."
    exit 1
fi

# Verbose output
if [[ $VERBOSE -eq 1 ]]; then
    echo "Configuration:"
    echo "  Input directory:  $INPUT_DIR"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  FlowMOP script:   $FLOWMOP_SCRIPT"
    echo "  Max parallel:     $MAX_PARALLEL"
    echo "  Fluor mode:       $FLUOR_MODE"
    echo "  MAD smoothing:    $MAD_SMOOTHING"
    echo "  MAD factor:       $MAD_FACTOR"
    echo "  Min cells:        $MIN_CELLS"
    echo "  Max bins:         $MAX_BINS"
    echo "  Step value:       $STEP_VAL"
    echo ""
fi

# Print processing information
echo "Found ${#FILES[@]} files to process with max $MAX_PARALLEL parallel jobs"
if [[ $VERBOSE -eq 1 ]]; then
    echo "Files to process:"
    for f in "${FILES[@]}"; do
        echo "  $f"
    done
    echo ""
fi

# Dry run mode - just list files and exit
if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY RUN] Would process the following files:"
    for f in "${FILES[@]}"; do
        echo "  $f"
    done
    echo ""
    echo "[DRY RUN] Command that would be executed:"
    echo "  python3 -u $FLOWMOP_SCRIPT <file> $CMD_OPTIONS"
    exit 0
fi

echo "Output will be saved to $OUTPUT_DIR"

# Run processes in parallel using GNU Parallel
parallel --bar --jobs $MAX_PARALLEL --halt never python3 -u "$FLOWMOP_SCRIPT" {} $CMD_OPTIONS ::: "${FILES[@]}"

echo ""
echo "Processing complete!"
echo "Processed ${#FILES[@]} files"
echo "Output saved to $OUTPUT_DIR"

exit 0