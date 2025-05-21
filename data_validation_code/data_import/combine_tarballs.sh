#!/bin/bash

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    echo "Example: $0 /path/to/fcs_files"
    exit 1
fi

SEARCH_DIR="$1"
if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory $SEARCH_DIR does not exist"
    exit 1
fi

# Set up logging with a more portable mutex
LOCK_FILE="/tmp/combine_tarballs.lock"
LOG_FILE="/tmp/combine_tarballs.log"

log() {
    # Portable locking mechanism
    while ! mkdir "$LOCK_FILE" 2>/dev/null; do
        sleep 0.1
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
    rmdir "$LOCK_FILE"
}

# Create output directory if it doesn't exist
mkdir -p "$SEARCH_DIR/combined_tarballs"

# Change to search directory
cd "$SEARCH_DIR"

# Find all unique base names
log "Finding unique base names in $SEARCH_DIR..."
bases=$(ls *.tar.gz.* 2>/dev/null | sed 's/\.tar\.gz\.[a-z]*$//' | sort -u)

# Process tarballs asynchronously
for base in $bases; do
    (
        log "Processing $base..."
        
        # Combine split files
        cat "$base".tar.gz.* > "combined_tarballs/${base}_combined.tar.gz"
        
        # Verify the combined file
        if tar -tzf "combined_tarballs/${base}_combined.tar.gz" &> /dev/null; then
            log "✓ Successfully combined $base"
        else
            log "✗ Error: Invalid tar.gz for $base"
        fi
    ) &  # Run in background
done

# Wait for all background processes to complete
wait

log "Done! Combined files are in $SEARCH_DIR/combined_tarballs/"