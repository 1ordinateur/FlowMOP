#!/bin/bash

#PBS -N flowmop_multiple
#PBS -P eu59
#PBS -q expresssr
#PBS -l walltime=06:00:00  
#PBS -l ncpus=24
#PBS -l mem=100GB           
#PBS -l jobfs=40GB        
#PBS -l storage=scratch/eu59+gdata/eu59+gdata/dk92
#PBS -M tony.xu@anu.edu.au
#PBS -m abe

###load module set
module use /g/data/dk92/apps/Modules/modulefiles/; module load NCI-data-analysis; module load parallel; cd /g/data/eu59/

# FlowMOP/src/parallel_flowmop.sh \
#     --input-dir /g/data/eu59/data_flowmop/fig_4_data/heuvue/train/ \
#     --output-dir /g/data/eu59/data_flowmop/fig_4_data/heuvue/train_cleaned/ 

# FlowMOP/src/parallel_flowmop.sh \
#     --input-dir /g/data/eu59/data_flowmop/fig_4_data/heuvue/test/ \
#     --output-dir /g/data/eu59/data_flowmop/fig_4_data/heuvue/test_cleaned/ 

# FlowMOP/src/parallel_flowmop.sh \
#     --input-dir /g/data/eu59/data_flowmop/ANUDC_16/train_test_split_cellspanel/train/ \
#     --output-dir /g/data/eu59/data_flowmop/ANUDC_16/train_test_split_cellspanel/train_flowmopped/ 

# FlowMOP/src/parallel_flowmop.sh \
#     --input-dir /g/data/eu59/data_flowmop/ANUDC_16/train_test_split_cellspanel/test/ \
#     --output-dir /g/data/eu59/data_flowmop/ANUDC_16/train_test_split_cellspanel/test_flowmopped/ 

# FlowMOP/src/parallel_flowmop.sh \
#     --input-dir /g/data/eu59/data_flowmop/cleaned_compiled_fig_2_dataset/ \
#     --output-dir /g/data/eu59/data_flowmop/cleaned_compiled_fig_2_dataset_flowmopped/ 

FlowMOP/src/parallel_flowmop.sh \
    --input-dir /g/data/eu59/data_flowmop/cleaned_compiled_fig_3_dataset/ \
    --output-dir /g/data/eu59/data_flowmop/cleaned_compiled_fig_3_dataset_flowmopped/ 