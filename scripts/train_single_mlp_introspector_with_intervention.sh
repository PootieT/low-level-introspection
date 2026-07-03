#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_intervention_train_hyper
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=6:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-9

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
#conda activate introspection

## environment setup
export WANDB_PROJECT=low-level-introspection
#export WANDB_WATCH=all
#export CUDA_LAUNCH_BLOCKING=1  # for debugging cuda errors
export WANDB_RUN_GROUP=single_mlp_intervention_train_hyper

## Batch job variables
index=$(($SGE_TASK_ID-1))
## vary lora r
#LORA_RS=(2 1)
#LORA_R=${LORA_RS[$index]}
#LORA_ALPHAS=(4 2)
#LORA_ALPHA=${LORA_ALPHAS[$index]}
LORA_R=1
LORA_ALPHA=2

## vary lr
LRS=(1e-6 5e-7 2e-7 1e-7 5e-8)
LR=${LRS[$index]}

## dropout
LORA_DOS=(0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
LORA_DO=${LORA_DOS[$index]}
#LORA_DO=0.05


### generate experiment config first as JSON
EXP_NAME="intervention_train_drop${LORA_DO}_300K"
python generate_experiment_config.py \
    --experiment-name $EXP_NAME \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 \
    --train-dataset-split "train" \
    --train-max-samples 300000 \
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
    --lora-dropout $LORA_DO \
    --num-epochs 1 \
    --disable-adapter-in-input \
    --bf16 \
    --lora-dropout $LORA_DO \
    --cache-dir data/internvetion_train \
    --output-dir outputs/runs/internvetion_train \
    --output experiment_configs/internvetion_train/$EXP_NAME.json
#    --learning-rate $LR \
#    --lora-r $LORA_R \
#    --lora-alpha $LORA_ALPHA \

### cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/internvetion_train/$EXP_NAME.json

## train introspector
python train_introspector_with_intervention.py \
    --config experiment_configs/internvetion_train/$EXP_NAME.json \
#    --overwrite-cache

# fixing lora_r=16, lr={1e-3 -> 1e-6}
# lr=1e-6 so far had best result, which consistently gets 0.2 ish correlation (not prediction eval though)
# fixing lr=5e-5, lora_r={128 -> 2}
# lora_r=2 had I think best (although by mistake never submitted lora_r=1)
