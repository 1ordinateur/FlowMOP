## Trimix
python execute_time_gate_disturbance.py \
    --specific-files \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_a/A05_rep1.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_a/A3_rep1.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_a/A1_rep1.fcs" \
    --specific-proportions 0.2 0.2 0.6 \
    --output-file "/z/Bruestle-EndersLab/Tony/staining_trials/A0531_202060_trimix.fcs" \
    --enable-mixing &

python execute_time_gate_disturbance.py \
    --specific-files \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_b/B3_rep1.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_b/B05_rep1.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_b/B1_rep1.fcs" \
    --specific-proportions 0.40 0.40 0.20 \
    --output-file "/z/Bruestle-EndersLab/Tony/staining_trials/B3051_404020_trimix.fcs" \
    --enable-mixing &

python execute_time_gate_disturbance.py \
    --specific-files \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_c/C05_rep1.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_c/C3_rep1.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_c/C1_rep1.fcs" \
    --specific-proportions 0.33 0.33 0.33 \
    --output-file "/z/Bruestle-EndersLab/Tony/staining_trials/C0531_333333_trimix.fcs" \
    --enable-mixing &

# Trimix - 3

python execute_time_gate_disturbance.py \
    --specific-files \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_c/C05_rep2.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_c/C3_rep2.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_c/C1_rep2.fcs" \
    --specific-proportions 0.33 0.33 0.33 \
    --output-file "/z/Bruestle-EndersLab/Tony/staining_trials/C0531_333333_trimix.fcs" \
    --enable-mixing &

python execute_time_gate_disturbance.py \
    --specific-files \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_b/B05_rep2.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_b/B1_rep2.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_b/B3_rep2.fcs" \
    --specific-proportions 0.2 0.2 0.6 \
    --output-file "/z/Bruestle-EndersLab/Tony/staining_trials/B0513_202060_trimix.fcs" \
    --enable-mixing &

python execute_time_gate_disturbance.py \
    --specific-files \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_a/A05_rep2.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_a/A1_rep2.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_a/A3_rep2.fcs" \
    --specific-proportions 0.4 0.4 0.2 \
    --output-file "/z/Bruestle-EndersLab/Tony/staining_trials/A0513_404020_trimix.fcs" \
    --enable-mixing &

wait 
