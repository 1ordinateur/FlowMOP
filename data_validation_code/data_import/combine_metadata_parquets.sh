#!/bin/bash

cd /g/data/eu59/FlowMOP/data

# Process metadata for each dataset in parallel
python3 ./code/json_to_dataframe.py \
    --json-dir ./hu_dataset/fcs_files/extracted/Bonn/ \
    --output-dir ./hu_dataset/fcs_files/process_metadata \
    --output-name BONN &

python3 ./code/json_to_dataframe.py \
    --json-dir ./hu_dataset/fcs_files/extracted/MLL5F/CLL5F/ \
    --output-dir ./hu_dataset/fcs_files/process_metadata \
    --output-name MLL5F &

python3 ./code/json_to_dataframe.py \
    --json-dir ./hu_dataset/fcs_files/extracted/Erlangen \
    --output-dir ./hu_dataset/fcs_files/process_metadata \
    --output-name ERLANGEN &

python3 ./code/json_to_dataframe.py \
    --json-dir ./hu_dataset/fcs_files/extracted/MLL9F/decCLL-9F \
    --output-dir ./hu_dataset/fcs_files/process_metadata \
    --output-name MLL9F &

# Wait for all background processes to complete
wait