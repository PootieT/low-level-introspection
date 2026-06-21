#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_l10_n24
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=24:00:00
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
#export CUDA_LAUNCH_BLOCKING=1  # for debugging cuda errors

# batch job variables
index=$(($SGE_TASK_ID-1))
## vary lora r
LORA_RS=(16 8 4 2 1)
LORA_R=${LORA_RS[$index]}
LORA_ALPHAS=(32 16 8 4 2)
LORA_ALPHA=${LORA_ALPHAS[$index]}
export WANDB_RUN_GROUP=single_mlp_hyper_lora_r_improve_intervention

# vary lora dropout
LORA_DOS=(0 0.05 0.1 0.15 0.2 0.3 0.4)
LORA_DO=${LORA_DOS[$index]}
export WANDB_RUN_GROUP=single_mlp_hyper_lora_dropout_improve_intervention

# vary lora target module
LORA_MODS=("all-linear" \
 "q_proj" "v_proj" "k_proj" "o_proj" "q_proj,v_proj" "q_proj,k_proj,v_proj,o_proj" \
"gate_proj" "up_proj" "down_proj" "gate_proj,up_proj,down_proj" \
)
LORA_MOD=${LORA_MODS[$index]}
export WANDB_RUN_GROUP=single_mlp_hyper_lora_target_module_improve_intervention

### generate experiment config first as JSON
EXP_NAME=single_mlp_l10_n24_train300K
python generate_experiment_config.py \
    --experiment-name $EXP_NAME \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 \
    --train-dataset-split "train" \
    --train-max-samples 1000000 \
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
    --learning-rate 5e-5 \
    --num-epochs 1 \
    --disable-adapter-in-input \
    --lora-r \
    --output experiment_configs/$EXP_NAME.json

### cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/$EXP_NAME.json

## train introspector
python train_introspector.py \
    --config experiment_configs/$EXP_NAME.json

python intervene_introspector.py \
    --config experiment_configs/$EXP_NAME.json

## some hyperparameter notes here
# change lora modules, change lora rank, dropout, to see if we can improve intervention correlation
