#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_lengen_sft
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=12:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-20

# ============================================================================
# LENGTH GENERALIZATION, SFT-ONLY ARM: does the joint (interventional) loss improve
# ENDOGENOUS length generalization over training on the endogenous loss alone?
#
# The comparison partner is train_single_mlp_introspector_length_generalization.sh
# (group mlp_lengen, W=0.01). This script changes ONE thing: W=1.0, i.e. the endogenous
# (SFT) term only -- train_introspector_joint.py then skips the interventional forward
# entirely. Everything else is held identical so the difference is the objective:
# same 5 L4 neurons, same data (min length 128, 300K train, 30K test), same LoRA/LR/BS,
# same unconstrained KL, same prompt, same eval lengths. The interventional eval still
# runs after training, as a diagnostic (an SFT readout is not expected to read an
# injected value, so its interventional numbers are the bypass reference, not a score).
#
# DIAGONAL CELLS ONLY (window == train_len): relative_generalization.png normalises each
# run against its own in-distribution score and uses only the diagonal, because off the
# diagonal the in-distribution reference is itself a generalization result. 4 cells x 5
# neurons = 20 tasks. For the full off-diagonal grid, copy the CELLS list from the joint
# script and set #$ -t 1-50.
#
# SEPARATE GROUP (mlp_lengen_sft): plot_length_generalization.py reads a group by name
# pattern and does not parse W, so mixing both objectives into mlp_lengen would average
# them together in every existing figure. Compare the two with
#   python analysis/plot_length_generalization.py --compare mlp_lengen:Joint mlp_lengen_sft:SFT
#
# DATA CACHES are shared with the joint arm: the cache key is derived from the data
# fields, not the experiment group or W, so these runs reuse the joint arm's caches and
# never rebuild them (and so never race on a build).
# ============================================================================

## SCC env setup
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection
export WANDB_RUN_GROUP=mlp_lengen_sft

set -e

index=$(($SGE_TASK_ID-1))
# ---- grid: the diagonal of the joint arm's grid -----------------------------
CELLS=("16:16" "32:32" "64:64" "128:128")
NEURONS=(4.5676 4.520 4.1216 4.6896 4.6386)   # identical to the joint arm

N_CELLS=${#CELLS[@]}
CELL_IDX=$(( index % N_CELLS ))
NEURON_IDX=$(( index / N_CELLS ))

CELL=${CELLS[$CELL_IDX]}
TRAIN_LEN=${CELL%%:*}
WINDOW=${CELL##*:}
SIG=${NEURONS[$NEURON_IDX]}
TOKEN_SEL="-${WINDOW}:-1"

MIN_LEN=128
EVAL_LENGTHS=16,32,64,128
GATE=off
W=1.0                    # ENDOGENOUS ONLY -- the one change from the joint arm (W=0.01)
SCALE=10                 # only affects the post-training interventional diagnostics
KL_TAG=klinf             # unconstrained, as in the joint arm; no --kl-anchor-target
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
N_TRAIN=300000
N_TEST=30000
PROMPT_TYPE=prompt_with_token_index

OVERWRITE=""

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

if [ "$PROMPT_TYPE" = "prompt_with_token_index" ]; then
  EXP_GROUP="mlp_lengen_sft"
else
  EXP_GROUP="mlp_lengen_sft_prompt_no_token_index"
fi
EXP_NAME="lengen_N${SIG}_len${TRAIN_LEN}_win${WINDOW}_gate${GATE}_${KL_TAG}_w${W}"
CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
echo "task=$SGE_TASK_ID cell=$CELL neuron=$SIG -> $EXP_NAME"

python generate_experiment_config.py \
    --experiment-name "$EXP_NAME" \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 \
    --train-dataset-split "train" \
    --train-max-samples $N_TRAIN \
    --dev-max-samples 1000 \
    --dev-dataset-split "validation" \
    --dev-offset-from-train 1000000 \
    --test-dataset allenai/c4 \
    --test-dataset-split "validation" \
    --test-max-samples $N_TEST \
    --max-seq-length $TRAIN_LEN \
    --min-seq-length $MIN_LEN \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "N${SIG}" \
    --test-signals "N${SIG}" \
    --train-token-selector="${TOKEN_SEL}" \
    --test-token-selector="${TOKEN_SEL}" \
    --prompt-type $PROMPT_TYPE \
    --precision 3 \
    --train-batch-size $BS \
    --eval-batch-size 128 \
    --num-epochs 1 \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DO \
    $GATE_FLAG \
    --bf16 \
    --learning-rate $LR \
    --use-regressor \
    --intervention-signal-distribution extreme \
    --intervention-extreme-scale $SCALE \
    --cache-dir data \
    --output-dir outputs/runs/$EXP_GROUP \
    --output "$CONFIG" \
    $OVERWRITE

# no-op: the joint arm already built this cell's cache
python cache_signals.py --config "$CONFIG" $OVERWRITE

python train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    --force-drift-eval \
    --length-eval \
    --eval-lengths $EVAL_LENGTHS \
    $OVERWRITE

# the long-range evals the joint arm also has (eval lengths up to 64x the shortest
# training length), so every ratio on the joint arm's relative_generalization plot
# has an SFT counterpart
python train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    --length-only \
    --eval-lengths 256,512,1024 \
    --eval-n-test 3000
