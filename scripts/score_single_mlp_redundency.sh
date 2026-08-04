#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_redundancy
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

EXP_NAME=intervention_train_reg_extreme10_300K
EXP_GROUP=intervention_train_regressor_extreme

## test predicatbility of each neuron from all other neurons in the same layer
#python score_neuron_redundancy.py \
#    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --layer 10 \
#    --positions introspected \
#    --fire-thresh 0.2 \
#    --per-bin 4 \
#    --n-texts 30000 \
#    --layer 10 \
#    --ridge 0.2 \
#    --output-dir outputs/circuit/"${EXP_GROUP}/${EXP_NAME}"

## test raw signal before lora from read-out position
python score_neuron_bypass_ceiling.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
    --cache-glob 'data/intervention_train_regressor_redundancy/intervention_train_reg_*Bin*_300K/test' \
    --n-examples 2000 \
    --output-dir outputs/circuit/"${EXP_GROUP}/${EXP_NAME}"