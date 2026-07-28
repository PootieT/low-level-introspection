#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_trace
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

#module load cuda/11.8   # 11.2
#module load miniconda/23.11.0
#conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

#EXP_NAME=intervention_train_reg_extreme10_300K
#EXP_GROUP=intervention_train_regressor_extreme

# ===== local setup =====
conda activate introspection
EXP_NAME=intervention_train_reg_N10.24_3K_fixedPaddingBug
EXP_GROUP=internvetion_train_regressor


python trace_introspection_circuit.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
    --n-examples 200 \
    --start-layer 0 \
    --output-dir outputs/circuit/"${EXP_GROUP}/${EXP_NAME}"
