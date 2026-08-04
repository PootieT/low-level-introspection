#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_intervention_train_joint_sweep
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=3:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-5

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
export WANDB_RUN_GROUP=single_mlp_intervention_train_reg_joint_sweep

## Batch job variables
index=$(($SGE_TASK_ID-1))

#WS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
WS=(0.01 0.03 0.05 0.07 0.09)
W=${WS[$index]}
#ANNEAL_END=1.0

SCALE=10
SIG=10.24

### generate experiment config first as JSON
EXP_NAME="intervention_train_reg_joint_scale${SCALE}_w${W}"
EXP_GROUP=intervention_train_regressor_joint_sweep
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
    --train-signals "N${SIG}" \
    --test-signals "N${SIG}" \
    --precision 3 \
    --train-batch-size 32 \
    --eval-batch-size 128 \
    --num-epochs 1 \
    --disable-adapter-in-input \
    --bf16 \
    --learning-rate 5e-5 \
    --use-regressor \
    --intervention-signal-distribution extreme \
    --intervention-extreme-scale $SCALE \
    --cache-dir data/$EXP_GROUP \
    --output-dir outputs/runs/$EXP_GROUP \
    --output experiment_configs/$EXP_GROUP/$EXP_NAME.json \
    --overwrite-cache

### cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache

## train introspector
python train_introspector_joint.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
    --joint-loss-weight $W
    #    --overwrite-cache
