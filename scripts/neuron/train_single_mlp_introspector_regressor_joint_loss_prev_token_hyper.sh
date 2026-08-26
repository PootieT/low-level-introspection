#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_intervention_train_joint_prev_token_hyper
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=6:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-3

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
export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection
#export WANDB_WATCH=all
#export CUDA_LAUNCH_BLOCKING=1  # for debugging cuda errors
export WANDB_RUN_GROUP=single_mlp_intervention_train_reg_joint_prev_token_hyper

## Batch job variables
index=$(($SGE_TASK_ID-1))

#WS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
#WS=(0.01 0.03 0.05 0.07 0.09)
#W=${WS[$index]}
W=0.01

LORA_R=128
LORA_ALPHA=$((LORA_R * 2))

SCALE=30 # 10 30 100
SIG=10.24
TOKEN_POS='-1:-31' #  -28  "0:31"
LRS=(1e-5 2e-5 3e-5 5e-5 1e-4 2e-4)
LR=${LRS[$index]}
#LR=5e-5
BS=128


### generate experiment config first as JSON
EXP_NAME="intervention_train_reg_joint_POS${TOKEN_POS}_scale${SCALE}_r${LORA_R}_lr${LR}_w${W}_1M"
DATASET_NAME="c4_N${SIG}_pos${TOKEN_POS}_seq32_1M"
EXP_GROUP=intervention_train_regressor_joint_prev_token_hyper
echo $EXP_NAME
python generate_experiment_config.py \
    --experiment-name $EXP_NAME \
    --dataset-name $DATASET_NAME \
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
    --train-signals "N${SIG}" \
    --test-signals "N${SIG}" \
    --train-token-selector="${TOKEN_POS}" \
    --test-token-selector="${TOKEN_POS}" \
    --precision 3 \
    --train-batch-size $BS \
    --eval-batch-size 128 \
    --num-epochs 1 \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --disable-adapter-in-input \
    --bf16 \
    --learning-rate $LR \
    --warmup-ratio 0.03 \
    --lr-scheduler-type cosine \
    --save-steps 0.2 \
    --use-regressor \
    --intervention-signal-distribution extreme \
    --intervention-extreme-scale $SCALE \
    --cache-dir data/$EXP_GROUP \
    --output-dir outputs/runs/$EXP_GROUP \
    --output experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache

### cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache

## train introspector
python train_introspector_joint.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
    --joint-loss-weight $W \
    #    --overwrite-cache

# post training analysis of each checkpoints
for CK in outputs/runs/$EXP_GROUP/$EXP_NAME/checkpoint-*; do
  python train_introspector_joint.py \
   --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
   --joint-loss-weight $W \
   --diag-only \
   --checkpoint "$CK"
done