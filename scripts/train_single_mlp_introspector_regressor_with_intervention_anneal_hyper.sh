#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_intervention_train_anneal_hyper
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=6:00:00
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
#conda activate introspection

## environment setup
export WANDB_PROJECT=low-level-introspection
#export WANDB_WATCH=all
#export CUDA_LAUNCH_BLOCKING=1  # for debugging cuda errors
export WANDB_RUN_GROUP=single_mlp_intervention_train_reg_anneal_hyper

## Batch job variables
index=$(($SGE_TASK_ID-1))
## vary lora r
#LORA_RS=(2 1)
#LORA_R=${LORA_RS[$index]}
#LORA_ALPHAS=(4 2)
#LORA_ALPHA=${LORA_ALPHAS[$index]}
#LORA_R=16
#LORA_ALPHA=32

DISTRIBUTION="extreme" # extreme gaussian
SCALE_START=100  # 30 100 300 1000 3000 10000 30000
SCALE_END=1  # 1 3 10

ANNEAL_ENDS=(1.0 0.8 0.6)
ANNEAL_END=${ANNEAL_ENDS[$index]}
ANNEAL_END=1.0

SIG=14.32
TOKEN_POS=-28

### generate experiment config first as JSON
EXP_NAME="intervention_train_reg_POS${TOKEN_POS}_N${SIG}_${DISTRIBUTION}_scale${SCALE_START}-${SCALE_END}_epoch0-${ANNEAL_END}"
EXP_GROUP=intervention_train_regressor_anneal_hyper
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
    --train-token-selector $TOKEN_POS \
    --test-token-selector $TOKEN_POS \
    --train-batch-size 32 \
    --eval-batch-size 128 \
    --num-epochs 1 \
    --disable-adapter-in-input \
    --bf16 \
    --learning-rate 5e-5 \
    --use-regressor \
    --intervention-signal-distribution $DISTRIBUTION \
    --intervention-anneal \
    --intervention-scale-start $SCALE_START \
    --intervention-scale-end $SCALE_END \
    --intervention-anneal-schedule cosine \
    --intervention-anneal-start-frac 0.0 \
    --intervention-anneal-end-frac $ANNEAL_END \
    --cache-dir data/$EXP_GROUP \
    --output-dir outputs/runs/$EXP_GROUP \
    --output experiment_configs/$EXP_GROUP/$EXP_NAME.json \
    --overwrite-cache

### cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache

## train introspector
python train_introspector_with_intervention.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
    --refit-head \
    --refit-estimator "ridge" \
    --refit-lambda-grid 1,10,30,100,300,1000,3000,10000,30000,100000
#    --overwrite-cache

# best in terms of test_intervention_corr where
# 1. gaussian 30->10, 0->0.2
# 2. gaussian 100->10, 0->0.8.
# but really everything was learning, even if we set start-scale = 10, so I wonder if it was just learning rate bug
# here we set to 5e-5, where default for sweep was 1e-5 --> tried it on one sweep example, still fails
# okay N14.3 still worked.. okay checked the results again, it's mainly earlier positions, with -1, most of neurons worked
# so now testing N14.32, POS=-28, so far seems like I need to scale up a lot
# extreme > gaussian most of the cases, more stable, gaussian often collaps
# best seem to be 100-10, end at 1.0