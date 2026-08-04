#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_intervention_train_regressor_redundancy
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=4:00:00
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
export WANDB_RUN_GROUP=single_mlp_intervention_train_reg_redundancy

## Batch job variables
index=$(($SGE_TASK_ID-1))

NEURONS=(3298 125 3096 2697 7684 6232 2648 4554 6534 8052 7407 7758 6111 2485 4985 4235 4472 308 5006 7715 6215 7111 6072 3992 2023 2962 1310 5901 693 6167 5205 5571 670 5282 6452 7592 7073 4100 6338 7497)
NEURON=${NEURONS[$index]}

BINS=(0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7 8 8 8 8 9 9 9 9)
BIN=${BINS[$index]}

### generate experiment config first as JSON
EXP_NAME="intervention_train_reg_N10.${NEURON}_Bin${BIN}_300K"
EXP_GROUP="intervention_train_regressor_redundancy"
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
    --train-signals "N10.${NEURON}" \
    --test-signals "N10.${NEURON}" \
    --precision 3 \
    --train-batch-size 32 \
    --eval-batch-size 128 \
    --num-epochs 1 \
    --disable-adapter-in-input \
    --bf16 \
    --use-regressor \
    --intervention-signal-distribution extreme \
    --intervention-extreme-scale 10 \
    --cache-dir data/$EXP_GROUP \
    --output-dir outputs/runs/$EXP_GROUP \
    --output experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache
#    --lora-dropout $LORA_DO \

### cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache

## train introspector
python train_introspector_with_intervention.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
    --refit-head \
    --refit-estimator "ridge" \
    --refit-lambda-grid 1,10,30,100,300,1000,3000,10000,30000,100000 \
#    --overwrite-cache \
