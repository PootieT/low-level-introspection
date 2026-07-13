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
LORA_DO=0.05


### generate experiment config first as JSON
EXP_NAME="intervention_train_reg_N1.24_r${LORA_R}_alpha${LORA_ALPHA}_lr${LR}_3K_debug"
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

## train introspector
python train_introspector_with_intervention.py \
    --config experiment_configs/internvetion_train_regressor/$EXP_NAME.json \
#    --overwrite-cache

#python intervene_introspector.py \
#    --config "experiment_configs/internvetion_train/${EXP_NAME}.json"

# fixing lora_r=16, lr={1e-3 -> 1e-6}
# lr=1e-6 so far had best result, which consistently gets 0.2 ish correlation (not prediction eval though)
# fixing lr=5e-5, lora_r={128 -> 2}
# lora_r=2 had I think best (although by mistake never submitted lora_r=1)
# fixing lr=5e-5, lora_r=16, dropout={0.1-0.8}
# I don't think any of it won particularly, maybe 0.4? but similar to 0.05 default honestly
# and best so far is lr=1e-7, r=2, alpha=4, corr peaked at 0.28, but drops to 0.2 at epoch 1
# fixed a bug: needed to add a space before the target string to keep 1st label token consistent
# also changed neuron from 10.24 -> 1.24, hopefully better? if not it's not an layer issue, but learnability issue, then we need to try regression head
# best for N1.24 is weirdly around -0.19 (R=16,lr=1e-7)/-0.18 (R=2,lr=5e-7), but nothing more than that. then on test (NTP), it's -0.9, then almost zero for pred and regression
# I think these may have to do with the random value used as label, maybe some runs are more natural than others