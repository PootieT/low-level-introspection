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
#$ -t 1-6

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
conda activate introspection

## environment setup
export WANDB_PROJECT=low-level-introspection
#export WANDB_WATCH=all
#export CUDA_LAUNCH_BLOCKING=1  # for debugging cuda errors
export WANDB_RUN_GROUP=single_mlp_intervention_train_reg_hyper

## Batch job variables
index=$(($SGE_TASK_ID-1))
## vary lora r
#LORA_RS=(2 1)
#LORA_R=${LORA_RS[$index]}
#LORA_ALPHAS=(4 2)
#LORA_ALPHA=${LORA_ALPHAS[$index]}
LORA_R=16
LORA_ALPHA=32

## vary lr
LRS=(5e-5 1e-6 5e-7 2e-7 1e-7 5e-8)
LR=${LRS[$index]}
LR=5e-5

## dropout
LORA_DOS=(0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
LORA_DO=${LORA_DOS[$index]}
LORA_DO=0 # previously 0.05 bug because of bug it's never passed in


### generate experiment config first as JSON
EXP_NAME="no_intervention_train_reg_N1.24_r${LORA_R}_alpha${LORA_ALPHA}_lr${LR}_3K"
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
    --train-signals N1.24 \
    --test-signals N1.24 \
    --precision 3 \
    --train-batch-size 32 \
    --eval-batch-size 128 \
    --num-epochs 1 \
    --disable-adapter-in-input \
    --bf16 \
    --learning-rate $LR \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --use-regressor \
    --cache-dir data/internvetion_train \
    --output-dir outputs/runs/internvetion_train \
    --output experiment_configs/internvetion_train_regressor/$EXP_NAME.json
#    --lora-dropout $LORA_DO \

### cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/internvetion_train_regressor/$EXP_NAME.json

## train introspector (without intervention)
python train_introspector_with_intervention.py \
    --config experiment_configs/internvetion_train_regressor/$EXP_NAME.json \
    --no-intervention
#    --overwrite-cache

#python intervene_introspector.py \
#    --config "experiment_configs/internvetion_train/${EXP_NAME}.json"

# This works, which means regressor heads is implemented correctly without intervention training
# so the culprit is really intervention training. Let's figure out how to get this to work
