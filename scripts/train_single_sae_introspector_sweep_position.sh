#!/bin/bash -l
#$ -P llamagrp
#$ -N single_sae_detection
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=14:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 2-10

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


# batch job variables
index=$(($SGE_TASK_ID-1))

LRS=(1e-5 2e-5 5e-5 1e-4 2e-4)
LR=${LRS[$index]}
LR=2e-5

TOKEN_POSS=(-1 -4 -7 -10 -13 -16 -19 -22 -25 -28)
TOKEN_POS=${TOKEN_POSS[$index]}
TOKEN_POS=-1

SIG="S9.12408"

### generate experiment config first as JSON
EXP_NAME="sae_${SIG}_POS${TOKEN_POS}_lr${LR}_train300K"
EXP_GROUP="single_sae_detection"
#EXP_GROUP="single_sae_detection_sweep_pos"
DATASET_NAME="c4_gemma2b_${SIG}_pos${TOKEN_POS}_seq32_300K"
export WANDB_RUN_GROUP=$EXP_GROUP

python generate_experiment_config.py \
    --experiment-name $EXP_NAME \
    --dataset-name $EXP_GROUP/$DATASET_NAME \
    --model google/gemma-2-2b \
    --objective detection \
    --detection-balance-pos-frac 0.5 \
    --train-dataset allenai/c4 \
    --train-dataset-split "train" \
    --train-max-samples 300000 \
    --dev-max-samples 2000 \
    --dev-offset-from-train 1000000 \
    --test-dataset allenai/c4 \
    --test-dataset-split "validation" \
    --test-max-samples 30000 \
    --max-seq-length 32 \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals S9.12408 \
    --test-signals S9.12408 \
    --precision 3 \
    --train-token-selector $TOKEN_POS \
    --test-token-selector $TOKEN_POS \
    --train-batch-size 32 \
    --eval-batch-size 128 \
    --learning-rate $LR \
    --num-epochs 1 \
    --disable-adapter-in-input \
    --bf16 \
    --output-dir outputs/runs/$EXP_GROUP \
    --output experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache

### cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache

## train introspector
python train_sae_detector.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache

# sweep across lr between 1e-5 to 5e-4, best is 2e-5 (for f1@0)
# lr=1e-5, ap=0.544, f1=0.588
# lr=2e-5, ap=0.577, f1=0.593
# lr=5e-5, ap=0.596, f1=0.562
# lr=1e-4, ap=0.632, f1=0.570
# lr=2e-5, ap=0.611, f1=0.583

# okay now we need to sweep across tokens and layers for random sae features
