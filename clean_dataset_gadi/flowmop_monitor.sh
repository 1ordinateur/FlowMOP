#!/bin/bash
# flowmop_monitor.sh - Monitor FlowMOP autosubmission system
#
# Usage:
#   ./flowmop_monitor.sh <work_directory>

set -e

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <work_directory>"
    echo "Example: $0 /scratch/eu59/flowmop_work"
    exit 1
fi

WORK_DIR="$1"

if [[ ! -d "$WORK_DIR" ]]; then
    echo "Error: Work directory '$WORK_DIR' does not exist"
    exit 1
fi

echo "FlowMOP Autosubmission System Monitor"
echo "====================================="
echo "Work directory: $WORK_DIR"
echo ""

# Check if system is initialized
if [[ ! -f "$WORK_DIR/job_control/job_params.env" ]]; then
    echo "Error: Autosubmission system not initialized in this directory"
    echo "Run flowmop_autosubmit_master.sh first"
    exit 1
fi

# Load job parameters for context
source "$WORK_DIR/job_control/job_params.env"

# File counts
UNPROCESSED=$(find "$WORK_DIR/unprocessed" -name "*.fcs" -o -name "*.parquet" 2>/dev/null | wc -l)
PROCESSING=$(find "$WORK_DIR/processing" -name "*.fcs" -o -name "*.parquet" 2>/dev/null | wc -l)
FINISHED=$(find "$WORK_DIR/finished" -name "*.fcs" -o -name "*.parquet" 2>/dev/null | wc -l)
FAILED=$(find "$WORK_DIR/failed" -name "*.fcs" -o -name "*.parquet" 2>/dev/null | wc -l)
OUTPUT=$(find "$WORK_DIR/flowmop_output" -name "flowmop_*.fcs" 2>/dev/null | wc -l)

TOTAL=$((UNPROCESSED + PROCESSING + FINISHED + FAILED))

echo "File Processing Status:"
echo "  Unprocessed: $UNPROCESSED"
echo "  Processing:  $PROCESSING"
echo "  Finished:    $FINISHED"
echo "  Failed:      $FAILED"
echo "  Output:      $OUTPUT"
echo "  Total:       $TOTAL"

if [[ $TOTAL -gt 0 ]]; then
    PERCENT_COMPLETE=$(echo "scale=1; ($FINISHED + $FAILED) * 100 / $TOTAL" | bc -l 2>/dev/null || echo "0")
    echo "  Progress:    ${PERCENT_COMPLETE}% complete"
fi

echo ""

# Job status
if [[ -f "$WORK_DIR/job_control/job_counter.txt" ]]; then
    JOB_COUNT=$(cat "$WORK_DIR/job_control/job_counter.txt")
    echo "Job Status:"
    echo "  Current iteration: $JOB_COUNT"
    echo "  Maximum iterations: $MAX_JOB_ITERATIONS"
fi

echo ""

# Active PBS jobs
echo "Active PBS Jobs:"
ACTIVE_JOBS=$(qstat -u $USER 2>/dev/null | grep flowmop_worker || echo "")
if [[ -n "$ACTIVE_JOBS" ]]; then
    echo "$ACTIVE_JOBS"
else
    echo "  No active FlowMOP jobs found"
fi

echo ""

# Recent log entries
echo "Recent Activity (last 10 entries):"
if [[ -f "$WORK_DIR/job_control/walltime_log.txt" ]]; then
    tail -n 10 "$WORK_DIR/job_control/walltime_log.txt" | sed 's/^/  /'
else
    echo "  No log file found"
fi

echo ""

# System recommendations
echo "Recommendations:"

if [[ $PROCESSING -gt 5 ]]; then
    echo "  WARNING: Many files stuck in processing state ($PROCESSING files)"
    echo "  Check for failed jobs or filesystem issues"
fi

if [[ $FAILED -gt 0 ]]; then
    echo "  NOTICE: $FAILED files failed processing"
    echo "  Check failed files in: $WORK_DIR/failed/"
fi

if [[ $UNPROCESSED -eq 0 ]] && [[ $PROCESSING -eq 0 ]]; then
    if [[ $FINISHED -gt 0 ]]; then
        echo "  SUCCESS: All files processed successfully!"
        echo "  Output files available in: $WORK_DIR/flowmop_output/"
    else
        echo "  WARNING: No files to process found"
    fi
fi

if [[ $JOB_COUNT -gt $((MAX_JOB_ITERATIONS - 5)) ]]; then
    echo "  WARNING: Approaching maximum job iterations"
    echo "  Consider increasing --max-job-iterations if needed"
fi

echo ""
echo "Monitor commands:"
echo "  Watch queue: watch -n 30 qstat -u $USER"
echo "  Follow log:  tail -f $WORK_DIR/job_control/walltime_log.txt"
echo "  Re-run monitor: $0 $WORK_DIR"

exit 0