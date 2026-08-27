#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_joint_prev_token_hyper
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=6:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1

# ============================================================================
# LR sweep for joint (endogenous + interventional) training on ONE mlp neuron,
# introspecting a FIXED token position in [-31, -1].
#
# Migrated from scripts_old_with_off_by_one_bug/. What changed, and why:
#
#  BOS / OFF-BY-ONE  preprocess_example used to tokenize without BOS while
#     cache_signals.py tokenized with it, so the intervention hook injected into
#     the token AFTER the one the prompt names, and the live neuron matched the
#     cached target at only ~0.81. Fixed in preprocess_example; every run here
#     passes --force-drift-eval so the alignment check runs as a regression test
#     (want: best offset +0, corr ~0.9999, "alignment OK"). Results from the old
#     scripts are NOT comparable to these.
#
#  SHARED CACHE  no --dataset-name any more. generate_experiment_config derives
#     the cache key from the fields that determine the data (+ an identity hash),
#     so all six LR arms share ONE dataset under data/<model>/..., and any other
#     script with the same data lands on the same directory. --cache-dir must stay
#     at the shared root `data` for that to work across experiment groups.
#
#  INPUT GATE  now driven by the config (training_config.disable_adapter_in_input)
#     rather than being hardcoded on. GATE=off leaves the adapter active over the
#     input text, which is what made earlier token positions work at all: under
#     GATE=on no trainable parameter touches the computation at the introspected
#     token, so the injected signal is routed downstream at fixed base-model gain.
#     Flip GATE to reproduce the gated control arm.
#
#  DROPOUT  GatedLinear silently dropped lora_dropout before, so every gated run
#     trained at 0 regardless of config. Now applied; LORA_DO is passed explicitly
#     and defaults to 0.0 to match the old effective behavior.
#
#  SCALE  10, not 30: scale 10 was consistently better at the unit-gaussian eval.
#
#  NO WARMUP / NO COSINE (config defaults: warmup_ratio 0.0, linear decay). Pre-fix,
#     cosine+warmup looked NECESSARY at earlier positions; post-fix it is harmful --
#     at pos=-28 gate-off it takes interventional corr 0.957 -> 0.356 and drives the
#     neuron's std ratio to 0.29. That earlier finding was made while the injection
#     was landing on the wrong token, i.e. on a different task.
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

## environment setup
export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection
export WANDB_RUN_GROUP=mlp_joint_prev_token_hyper

# stop on the first failure: without this a failed config generation silently
# falls through to training on a stale config from a previous run
set -e

## Batch job variables
index=$(($SGE_TASK_ID-1))

GATE=on                 # off = adapter active over input text (working setup) | on = gated control (adaptor is disabled in input text so behavior is exactly the same as frozen)
W=0.01                   # weight on the ENDOGENOUS term; interventional gets (1-W)
SCALE=10                 # injected-value spread (extreme distribution)
SIG=10.24                # layer.neuron
TOKEN_POS=-1     #'-1:-31'
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
#LRS=(1e-5 2e-5 3e-5 5e-5 1e-4 2e-4)
#LR=${LRS[$index]}
LR=5e-5

# set to --overwrite-cache to force config/dataset/run regeneration
OVERWRITE=""

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

EXP_GROUP=mlp_joint_prev_token_hyper
EXP_NAME="joint_N${SIG}_POS${TOKEN_POS}_gate${GATE}_scale${SCALE}_w${W}_lrcos"
CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
echo "$EXP_NAME"

### generate experiment config first as JSON
python generate_experiment_config.py \
    --experiment-name "$EXP_NAME" \
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
    --train-token-selector="${TOKEN_POS}" \
    --test-token-selector="${TOKEN_POS}" \
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


### cache activations and generate dataset
# no-op once any arm of the sweep has built it (the six LRs resolve to one
# canonical dataset name); grep the config for the derived dataset_name to see where
python cache_signals.py --config "$CONFIG" $OVERWRITE

## train introspector
# --force-drift-eval: runs anyway when GATE=off, but keep it so the gated arm also
# reports the cache-alignment check
python train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    --force-drift-eval \
    $OVERWRITE

# post training analysis of each checkpoint (same config => same gating as training)
#for CK in outputs/runs/$EXP_GROUP/$EXP_NAME/checkpoint-*; do
#  python train_introspector_joint.py \
#     --config "$CONFIG" \
#     --joint-loss-weight $W \
#     --force-drift-eval \
#     --diag-only \
#     --checkpoint "$CK"
#done
