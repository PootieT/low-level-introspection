#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_intervention_train
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=24:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-4

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
#conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env
conda activate introspection

## environment setup
export WANDB_PROJECT=low-level-introspection
#export WANDB_WATCH=all
#export CUDA_LAUNCH_BLOCKING=1  # for debugging cuda errors

export WANDB_RUN_GROUP=single_mlp_hyper_lora_dropout_improve_intervention

### generate experiment config first as JSON
EXP_NAME="intervention_train_debug"
python generate_experiment_config.py \
    --experiment-name $EXP_NAME \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 \
    --train-dataset-split "train" \
    --train-max-samples 3000 \
    --dev-max-samples 200 \
    --dev-offset-from-train 1000000 \
    --test-dataset allenai/c4 \
    --test-dataset-split "validation" \
    --test-max-samples 3000 \
    --max-seq-length 32 \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals N10.24 \
    --test-signals N10.24 \
    --precision 3 \
    --train-batch-size 32 \
    --eval-batch-size 128 \
    --num-epochs 1 \
    --disable-adapter-in-input \
    --bf16 \
    --cache-dir data/internvetion_train \
    --output-dir outputs/runs/internvetion_train \
    --output experiment_configs/internvetion_train/$EXP_NAME.json

### cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/internvetion_train/$EXP_NAME.json

## train introspector
python train_introspector_with_intervention.py \
    --config experiment_configs/internvetion_train/$EXP_NAME.json \
#    --overwrite-cache
