cd /c/Users/Tony/Documents/github_remotes/FlowMOP/data_validation_code/algo_validation

## Segmented 0.5 - 1 
python execute_time_gate_disturbance.py \
    --specific-files \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_a/A05_rep1.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_a/A1_rep2.fcs" \
    --specific-proportions 0.2 0.8 \
    --output-file "/z/Bruestle-EndersLab/Tony/staining_trials/A051_2080_bimix.fcs" \
    --enable-mixing & 

python execute_time_gate_disturbance.py \
    --specific-files \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_b/B05_rep1.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_b/B1_rep2.fcs" \
    --specific-proportions 0.65 0.35 \
    --output-file "/z/Bruestle-EndersLab/Tony/staining_trials/B051_6535_bimix.fcs" \
    --enable-mixing & 

python execute_time_gate_disturbance.py \
    --specific-files \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_c/C05_rep1.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_c/C1_rep2.fcs" \
    --specific-proportions 0.50 0.50 \
    --output-file "/z/Bruestle-EndersLab/Tony/staining_trials/C051_5050_bimix.fcs" \
    --enable-mixing & 

# Segmented 0.5 - 3

python execute_time_gate_disturbance.py \
    --specific-files \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_c/C05_rep1.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_c/C3_rep2.fcs" \
    --specific-proportions 0.50 0.50 \
    --output-file "/z/Bruestle-EndersLab/Tony/staining_trials/C053_5050_bimix.fcs" \
    --enable-mixing & 

python execute_time_gate_disturbance.py \
    --specific-files \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_b/B05_rep1.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_b/B3_rep2.fcs" \
    --specific-proportions 0.2 0.8 \
    --output-file "/z/Bruestle-EndersLab/Tony/staining_trials/B053_2080_bimix.fcs" \
    --enable-mixing & 

python execute_time_gate_disturbance.py \
    --specific-files \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_a/A05_rep1.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_a/A3_rep2.fcs" \
    --specific-proportions 0.65 0.35 \
    --output-file "/z/Bruestle-EndersLab/Tony/staining_trials/A053_6535_bimix.fcs" \
    --enable-mixing & 

# Segmented 1 - 3

python execute_time_gate_disturbance.py \
    --specific-files \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_a/A3_rep1.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_a/A1_rep1.fcs" \
    --specific-proportions 0.65 0.35 \
    --output-file "/z/Bruestle-EndersLab/Tony/staining_trials/A31_6535_bimix.fcs" \
    --enable-mixing & 

python execute_time_gate_disturbance.py \
    --specific-files \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_c/C3_rep1.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_c/C1_rep1.fcs" \
    --specific-proportions 0.50 0.50 \
    --output-file "/z/Bruestle-EndersLab/Tony/staining_trials/C31_5050_bimix.fcs" \
    --enable-mixing &  

python execute_time_gate_disturbance.py \
    --specific-files \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_b/B3_rep1.fcs" \
        "/z/Bruestle-EndersLab/Tony/staining_trials/stain_b/B1_rep1.fcs" \
    --specific-proportions 0.80 0.20 \
    --output-file "/z/Bruestle-EndersLab/Tony/staining_trials/B31_8020_bimix.fcs" \
    --enable-mixing & 

wait 
