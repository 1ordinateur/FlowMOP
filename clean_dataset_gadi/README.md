# FlowMOP Autosubmission System for NCI Gadi

This directory contains the complete autosubmission system for processing large datasets of FCS files using FlowMOP on NCI's Gadi supercomputer. The system automatically handles job chaining, walltime management, and file state tracking.

## System Overview

The autosubmission system consists of three main components:

1. **Master Initialization Script** (`flowmop_autosubmit_master.sh`) - Sets up the system and submits the first job
2. **PBS Worker Script** (`flowmop_worker.pbs`) - Processes files with automatic job chaining
3. **Monitor Script** (`flowmop_monitor.sh`) - Tracks progress and system status

## Features

- ✅ **Automatic Job Chaining**: Jobs automatically submit continuation jobs before walltime expires
- ✅ **Walltime Management**: 30-minute safety buffer with emergency exit at 5 minutes
- ✅ **Complete Metadata Preservation**: All original FCS metadata preserved in output files
- ✅ **Atomic File Operations**: Prevents race conditions and double-processing
- ✅ **Robust Error Handling**: Failed files tracked separately, processing continues
- ✅ **Progress Monitoring**: Real-time status tracking and logging
- ✅ **Recovery Capability**: System can resume from any interruption point

## Quick Start

### 1. Initialize the System

```bash
./flowmop_autosubmit_master.sh \
    --input-dir /path/to/fcs/files \
    --work-dir /scratch/eu59/flowmop_work \
    --pbs-project eu59 \
    --pbs-storage gdata/eu59
```

### 2. Monitor Progress

```bash
# Check overall status
./flowmop_monitor.sh /scratch/eu59/flowmop_work

# Watch job queue
watch -n 30 qstat -u $USER

# Follow processing log
tail -f /scratch/eu59/flowmop_work/job_control/walltime_log.txt
```

### 3. Collect Results

Processed files with complete metadata will be available in:
```
/scratch/eu59/flowmop_work/flowmop_output/flowmop_*.fcs
```

## Directory Structure

The system creates the following directory structure:

```
work_directory/
├── job_control/           # Job management files
│   ├── job_params.env     # Job parameters for continuity
│   ├── job_counter.txt    # Current job iteration counter
│   └── walltime_log.txt   # Processing and timing log
├── unprocessed/           # FCS files waiting to be processed
├── processing/            # Files currently being processed
├── finished/              # Successfully processed original files
├── failed/                # Files that failed processing
└── flowmop_output/        # Final output files (flowmop_*.fcs)
```

## Command Line Options

### Master Script Options

**Required:**
- `--input-dir <dir>` - Directory containing input FCS files
- `--work-dir <dir>` - Working directory for the autosubmission system
- `--pbs-project <project>` - PBS project code for job submission

**FlowMOP Processing Options:**
- `--fluor-mode <mode>` - Fluorescence analysis mode (default: positive_geomeans)
- `--mad-smoothing <values>` - MAD smoothing factors (default: "0.1 0.9")
- `--min-cells <num>` - Minimum cells required (default: 1000)
- `--max-bins <num>` - Maximum bins (default: 600)
- `--step-val <num>` - Step size (default: 200)
- `--mad-factor <num>` - MAD factor (default: 5)
- `--enable-plots` - Generate time gate plots
- `--enable-ssc` - Use SSC-A for debris gating
- `--remove-beads` - Detect and remove beads
- `--skip-debris` - Skip debris filtering
- `--skip-time` - Skip time filtering
- `--skip-doublets` - Skip doublet filtering

**PBS Job Options:**
- `--pbs-queue <queue>` - PBS queue (default: normal)
- `--pbs-walltime <time>` - Walltime per job (default: 02:00:00)
- `--pbs-ncpus <num>` - CPUs per job (default: 4)
- `--pbs-mem <size>` - Memory per job (default: 16GB)
- `--pbs-storage <list>` - Storage requirements
- `--max-job-iterations <n>` - Maximum job chain length (default: 50)

## Example Usage

### Basic Usage
```bash
./flowmop_autosubmit_master.sh \
    --input-dir /g/data/eu59/raw_fcs_data \
    --work-dir /scratch/eu59/flowmop_batch1 \
    --pbs-project eu59 \
    --pbs-storage gdata/eu59
```

### Advanced Usage with Custom Parameters
```bash
./flowmop_autosubmit_master.sh \
    --input-dir /g/data/eu59/experiment_data \
    --work-dir /scratch/eu59/flowmop_experiment \
    --pbs-project eu59 \
    --pbs-storage gdata/eu59+scratch/eu59 \
    --pbs-walltime 04:00:00 \
    --pbs-ncpus 8 \
    --pbs-mem 32GB \
    --fluor-mode both \
    --mad-smoothing "0.05 0.95" \
    --enable-plots \
    --enable-ssc \
    --max-job-iterations 100
```

## Monitoring and Troubleshooting

### Check System Status
```bash
./flowmop_monitor.sh /path/to/work/directory
```

### View Active Jobs
```bash
qstat -u $USER
```

### Check Processing Log
```bash
tail -f /path/to/work/directory/job_control/walltime_log.txt
```

### Examine Failed Files
```bash
ls /path/to/work/directory/failed/
```

### Manual Recovery
If the system stops unexpectedly, you can restart it by running the master script again with the same work directory. The system will automatically resume from where it left off.

## File Naming Convention

- **Input files**: Original FCS filenames
- **Output files**: `flowmop_<original_name>.fcs`
- **Example**: `sample001.fcs` → `flowmop_sample001.fcs`

## Metadata Preservation

The enhanced FlowMOP processor preserves:

- ✅ **All original FCS metadata** (instrument settings, acquisition parameters, etc.)
- ✅ **FlowMOP processing parameters** (fluor_mode, mad_smoothing, etc.)
- ✅ **Processing statistics** (event counts, retention percentages, filter results)
- ✅ **Provenance information** (processing date, original file path, etc.)

## Safety Features

### Walltime Management
- Monitors elapsed time continuously
- Submits continuation jobs 30 minutes before walltime expires
- Emergency exit 5 minutes before walltime to prevent job termination

### File State Protection
- Atomic file operations prevent double-processing
- Failed files moved to separate directory
- Original files preserved in finished directory

### Job Chain Limits
- Maximum iteration limit prevents infinite loops
- Uses NCI's recommended `beforeok` dependency pattern
- Comprehensive logging for audit trail

## Best Practices

1. **Use appropriate walltime**: 2-4 hours typically optimal for most datasets
2. **Monitor initial jobs**: Check first few jobs to ensure proper operation
3. **Check storage quotas**: Ensure sufficient scratch space for work directory
4. **Verify PBS project**: Ensure you have allocation in the specified project
5. **Test small datasets first**: Validate parameters before large-scale processing

## Troubleshooting

### Common Issues

**Jobs not starting:**
- Check PBS project allocation: `nci-account-info`
- Verify storage access: ensure gdata/project is accessible
- Check queue limits: `qstat -Q`

**Files stuck in processing:**
- Check for crashed jobs: `qstat -u $USER`
- Look for filesystem errors in job logs
- Manually move stuck files back to unprocessed if needed

**High failure rate:**
- Check failed files for common issues
- Verify FlowMOP parameters are appropriate for dataset
- Check available memory per job

**Job chain stops:**
- Check walltime_log.txt for error messages
- Verify maximum iteration limit not exceeded
- Check PBS project allocation remaining

### Getting Help

For issues specific to:
- **FlowMOP processing**: Check FlowMOP documentation and parameters
- **PBS/Gadi issues**: Contact NCI Help Desk
- **System bugs**: Check logs and file permissions

## Technical Details

### Job Dependency Strategy
Following NCI best practices, the system uses `beforeok` dependencies rather than `afterok` to avoid issues with rapidly completing jobs.

### Walltime Calculation
```bash
# 30-minute resubmission buffer
RESUBMIT_THRESHOLD = WALLTIME - 1800 seconds

# 5-minute emergency exit
EMERGENCY_THRESHOLD = WALLTIME - 300 seconds
```

### Atomic File Operations
```bash
# Prevents race conditions
if mv "unprocessed/file.fcs" "processing/file.fcs" 2>/dev/null; then
    # Process file
    # Move to finished/ or failed/ based on result
fi
```

This ensures only one job can process a given file, preventing double-processing even with multiple concurrent jobs.