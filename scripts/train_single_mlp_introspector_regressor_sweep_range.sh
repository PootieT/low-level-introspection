#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_intervention_train_regressor_extreme
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=6:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 3

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
export WANDB_RUN_GROUP=single_mlp_intervention_train_reg_extreme

## Batch job variables
index=$(($SGE_TASK_ID-1))

SCALES=(1 3 10 30 100 300 1000)
SCALE=${SCALES[$index]}
SCALE=10
### generate experiment config first as JSON
EXP_NAME="intervention_train_reg_extreme${SCALE}_300K"
EXP_GROUP="internvetion_train_regressor_extreme"
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
    --num-epochs 1 \
    --disable-adapter-in-input \
    --bf16 \
    --use-regressor \
    --intervention-signal-distribution extreme \
    --intervention-extreme-scale $SCALE \
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

# with 300K data, extreme-range of 10 is enough (1 or 3 doesn't work) to get good correlation, that regular eval corr = 0.18 (significant)
