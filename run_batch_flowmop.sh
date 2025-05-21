python src/batch_process_flowmop.py \
    --input-dirs path/to/your/input_fcs_files_dir another/path/to/input_files \
    --output-dir path/to/your/output_dir \
    --workers 4 \
    --file-extension .fcs \
    --output-suffix _cleaned \
    --skip-doublet-gating \
    --mad-threshold 5.0 \
    --verbose