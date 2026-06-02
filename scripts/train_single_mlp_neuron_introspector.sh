#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_l10_n24
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=47G
#$ -l h_rt=6:00:00
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

## environment setup
export WANDB_PROJECT=low-level-introspection
export WANDB_WATCH=all
export CUDA_LAUNCH_BLOCKING=1  # for debugging cuda errors

### generate experiment config first as JSON
python generate_experiment_config.py \
    --experiment-name single_mlp_l10_n24 \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 \
    --train-dataset-split "train" \
    --train-max-samples 3000 \
    --dev-max-samples 100 \
    --test-dataset allenai/c4 \
    --test-dataset-split "validation" \
    --test-max-samples 1000 \
    --max-seq-length 32 \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals N10.24 \
    --test-signals N10.24 \
    --precision 3 \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --learning-rate 1e-5 \
    --num-epochs 1 \
    --disable-adapter-in-input \
    --output experiment_configs/single_mlp_l10_n24.json

### cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/single_mlp_l10_n24.json

## train introspector
python train_introspector.py \
    --config experiment_configs/single_mlp_l10_n24.json

## some hyperparameter notes here
#