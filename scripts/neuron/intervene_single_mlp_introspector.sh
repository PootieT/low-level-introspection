#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_intervention
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=12:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1

## SCC env setup
## Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

module load cuda/11.8   # 11.2
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

#EXP_NAME=single_mlp_l10_n24_debug
EXP_NAME=single_mlp_l10_n24_train3K

python intervene_introspector.py \
    --config experiment_configs/$EXP_NAME.json
