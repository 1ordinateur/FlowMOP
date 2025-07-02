
#!/bin/bash

#PBS -N autogen_sample_combos
#PBS -P eu59
#PBS -q expresssr
#PBS -l walltime=5:00:00  
#PBS -l ncpus=24
#PBS -l mem=100GB           
#PBS -l jobfs=40GB        
#PBS -l storage=scratch/eu59+gdata/eu59+gdata/dk92
#PBS -M tony.xu@anu.edu.au
#PBS -m abe

module use /g/data/dk92/apps/Modules/modulefiles/; module load NCI-data-analysis; module load parallel; cd /g/data/eu59/FlowMOP

# Segments

python3 /g/data/eu59/FlowMOP/src/data_validation_code/algo_validation/auto_generate_combinations.py \
--input-dir /g/data/eu59/data_flowmop/staining_trials/stain_a \
--output-dir /g/data/eu59/data_flowmop/fig_2_timegate_combos \
--num-combinations 15 \ & 

python3 /g/data/eu59/FlowMOP/src/data_validation_code/algo_validation/auto_generate_combinations.py \
--input-dir /g/data/eu59/data_flowmop/staining_trials/stain_b \
--output-dir /g/data/eu59/data_flowmop/fig_2_timegate_combos \
--num-combinations 15 \ & 

python3 /g/data/eu59/FlowMOP/src/data_validation_code/algo_validation/auto_generate_combinations.py \
--input-dir /g/data/eu59/data_flowmop/staining_trials/stain_c \
--output-dir /g/data/eu59/data_flowmop/fig_2_timegate_combos \
--num-combinations 15 \ & 

# Bimix
python3 /g/data/eu59/FlowMOP/src/data_validation_code/algo_validation/auto_generate_combinations.py \
--input-dir /g/data/eu59/data_flowmop/staining_trials/stain_a \
--output-dir /g/data/eu59/data_flowmop/fig_2_timegate_combos \
--suffix bimix \
--num-combinations 15 \ 
--enable-mixing \
--mixing-chunk-size 2000 &

python3 /g/data/eu59/FlowMOP/src/data_validation_code/algo_validation/auto_generate_combinations.py \
--input-dir /g/data/eu59/data_flowmop/staining_trials/stain_b \
--output-dir /g/data/eu59/data_flowmop/fig_2_timegate_combos \
--suffix bimix \
--num-combinations 15 \ 
--enable-mixing \
--mixing-chunk-size 2000 &

python3 /g/data/eu59/FlowMOP/src/data_validation_code/algo_validation/auto_generate_combinations.py \
--input-dir /g/data/eu59/data_flowmop/staining_trials/stain_c \
--output-dir /g/data/eu59/data_flowmop/fig_2_timegate_combos \
--suffix bimix \
--num-combinations 15 \ 
--enable-mixing \
--mixing-chunk-size 2000 &

# Trimix
python3 /g/data/eu59/FlowMOP/src/data_validation_code/algo_validation/auto_generate_combinations.py \
--input-dir /g/data/eu59/data_flowmop/staining_trials/stain_a \
--output-dir /g/data/eu59/data_flowmop/fig_2_timegate_combos \
--suffix trimix \
--num-combinations 15 \ 
--files-per-combo 3 \
--enable-mixing \
--mixing-chunk-size 2000 &

python3 /g/data/eu59/FlowMOP/src/data_validation_code/algo_validation/auto_generate_combinations.py \
--input-dir /g/data/eu59/data_flowmop/staining_trials/stain_b \
--output-dir /g/data/eu59/data_flowmop/fig_2_timegate_combos \
--suffix trimix \
--num-combinations 15 \ 
--files-per-combo 3 \
--enable-mixing \
--mixing-chunk-size 2000 &

python3 /g/data/eu59/FlowMOP/src/data_validation_code/algo_validation/auto_generate_combinations.py \
--input-dir /g/data/eu59/data_flowmop/staining_trials/stain_c \
--output-dir /g/data/eu59/data_flowmop/fig_2_timegate_combos \
--suffix trimix \
--num-combinations 15 \ 
--files-per-combo 3 \
--enable-mixing \
--mixing-chunk-size 2000 & 

wait
