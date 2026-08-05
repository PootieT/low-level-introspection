#!/bin/bash -l
#$ -P llamagrp
#$ -N single_sae_detection
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=2:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-10

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
export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_RUN_GROUP=single_sae_detection

# batch job variables
index=$(($SGE_TASK_ID-1))
## vary lora r
#LORA_RS=(8 4 2 1)
#LORA_R=${LORA_RS[$index]}
#LORA_ALPHAS=(16 8 4 2)
#LORA_ALPHA=${LORA_ALPHAS[$index]}
LORA_R=16
LORA_ALPHA=32
LR=5e-5

SIG="S9.12408"
TOKEN_POS=-1

### generate experiment config first as JSON
EXP_NAME="sae_${SIG}_lr${LR}_train300K"
EXP_GROUP="single_sae_detection"
DATASET_NAME="c4_gemma2b_${SIG}_pos${TOKEN_POS}_seq32_1M"
python generate_experiment_config.py \
    --experiment-name $EXP_NAME \
    --dataset-name $EXP_GROUP/$DATASET_NAME \
    --model google/gemma-2-2b \
    --objective detection \
    --train-dataset allenai/c4 \
    --train-dataset-split "train" \
    --train-max-samples 300000 \
    --dev-max-samples 500 \
    --dev-offset-from-train 1000000 \
    --test-dataset allenai/c4 \
    --test-dataset-split "validation" \
    --test-max-samples 3000 \
    --max-seq-length 32 \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals S9.12408 \
    --test-signals S9.12408 \
    --precision 3 \
    --train-batch-size 32 \
    --eval-batch-size 128 \
    --learning-rate $LR \
    --num-epochs 1 \
    --disable-adapter-in-input \
    --bf16 \
    --cache-dir data/$EXP_GROUP \
    --output-dir outputs/runs/$EXP_GROUP \
    --output experiment_configs/$EXP_GROUP/$EXP_NAME.json \

### cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache

## train introspector
python train_sae_detector.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache
