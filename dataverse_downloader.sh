#!/bin/bash

# Download data from Harvard Dataverse using curl
# Usage: ./download_dataverse.sh <doi> <output_dir> [api_token]

# Check if required arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Please provide both DOI and output directory"
    echo "Usage: ./download_dataverse.sh <doi> <output_dir> [api_token]"
    exit 1
fi

# Set variables
SERVER_URL="https://dataverse.harvard.edu"
PERSISTENT_ID="$1"
OUTPUT_DIR="$2"
API_TOKEN="$3"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create download command
if [ -z "$API_TOKEN" ]; then
    # Download without API token
    cd "$OUTPUT_DIR" && curl -L -O -J "$SERVER_URL/api/access/dataset/:persistentId/?persistentId=$PERSISTENT_ID"
else
    # Download with API token
    cd "$OUTPUT_DIR" && curl -L -O -J -H "X-Dataverse-key:$API_TOKEN" "$SERVER_URL/api/access/dataset/:persistentId/?persistentId=$PERSISTENT_ID"
fi

# Print success message
echo "Download completed in directory: $OUTPUT_DIR"