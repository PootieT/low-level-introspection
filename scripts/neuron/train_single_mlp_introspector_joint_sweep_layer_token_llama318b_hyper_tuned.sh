#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_joint_grid_layer_token_llama318b_hyper_tuned
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=36:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-16

# ============================================================================
# MAIN GRID AT THE TUNED LR, Llama-3.1-8B: joint introspection training, KL 0.1 only,
# 4 layers x 4 token positions, 5 neurons per cell = 16 array tasks, 80 runs.
#
# >>> SUBMIT -- paste on SCC (from the repo root):
# cd /projectnb/llamagrp/peter/low-level-introspection
# qsub scripts/neuron/train_single_mlp_introspector_joint_sweep_layer_token_llama318b_hyper_tuned.sh
# # once runs finish (the SFT arm at the tuned LR comes later and joins the same table):
# python plotters/compare_sft_vs_joint.py --model-suffix _llama318b_hyper_tuned
# <<< SUBMIT
#
# WHY. The LR sweep (train_single_mlp_introspector_joint_lr_sweep.sh, plotters/
# plot_lr_sweep.py) found the 1B learning rate does not transfer: under the KL 0.1
# anchor, 1e-4 is at or near the best interventional corr in every swept cell of both
# 3B and 8B, and 2e-4 collapses training at 8B. So this re-runs the grid at 1e-4.
#
# IDENTICAL to train_single_mlp_introspector_joint_sweep_layer_token_llama318b.sh except:
#   - same layers 4/12/20/28 and positions: every cache already exists, and each
#     cell trains the SAME 5 neurons as the 5e-5 grid, so the two grids pair exactly
#   - KL 0.1 only (the unanchored arm is dropped), so 16 tasks rather than 32
#   - LR 1e-4 (was 5e-5)
#   - a new run group, mlp_joint_grid_layer_token_llama318b_hyper_tuned. The run names do
#     not carry the LR, so in the old group every run would be skipped as already
#     done. plotters/compare_sft_vs_joint.py knows 'llama318b_hyper_tuned' as its own grid
#   - OVERWRITE is empty, so resubmitting skips finished runs instead of retraining
#
# CAVEAT from the sweep: at 8B L4 pos-1 one of the two tuning neurons collapsed at 1e-4
# (interventional 0.32 while the other reached 0.91). Training here is close to
# all-or-nothing near the top of the usable LR range, so expect the occasional failed
# neuron in early-layer cells; they show as low interventional with high endogenous.
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

EXP_GROUP=mlp_joint_grid_layer_token_llama318b_hyper_tuned

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection
export WANDB_RUN_GROUP=$EXP_GROUP

set -e

## ---- decode (layer, position) from the flat array index ----
# One task = one cell, so no two concurrent tasks share a cache directory.
index=$(($SGE_TASK_ID-1))
LAYERS=(4 12 20 28)
POSS=(-1 -8 -20 -28)
LAYER=${LAYERS[$((index / 4))]}
TOKEN_POS=${POSS[$((index % 4))]}

KL_FLAG="--kl-anchor-target 0.1"
KL_TAG="kl0.1"
GATE=off
W=0.01
SEED=21
# The DATASET carries N_NEURONS_CACHE signals; we TRAIN on the first N_NEURONS_TRAIN of
# them -- the same 5 neurons per layer as the 5e-5 grid (same sampler and seed), so the
# two grids pair neuron for neuron. The LR sweep tuned on samples 5-6, never used here.
# (--signal-filter is not part of the cache identity, so all neurons share one dataset.)
N_NEURONS_CACHE=10
N_NEURONS_TRAIN=5
SCALE=10
LR=1e-4
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=16  # effective batch 32, as everywhere
ACCUM=2
LM_SCORE_N=1
# OOD generalization eval on EVERY run, as in the 5e-5 grid
OOD_FLAG="--ood-eval"
# Empty: this is a fresh group, and an empty OVERWRITE makes resubmission skip finished
# runs instead of retraining them. Never pass --overwrite-cache to cache_signals: the
# caches are shared with the 5e-5 grid and the LR sweep.
OVERWRITE=""

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

CELL_TAG="l${LAYER}_pos${TOKEN_POS}"
SAMPLER="{\"type\":\"uniform_layer_sampler\",\"signal_type\":\"mlp\",\"layers\":[${LAYER}],\"signals_per_layer\":${N_NEURONS_CACHE},\"seed\":${SEED}}"
echo "cell=$CELL_TAG regime=$KL_TAG lr=$LR"

## ---- resolve this layer's neuron set ONCE (identical at every position) ----
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_${CELL_TAG}.json
python data_prep/generate_experiment_config.py \
    --experiment-name "_cell_${CELL_TAG}" \
    --model meta-llama/Llama-3.1-8B \
    --train-dataset allenai/c4 --train-dataset-split "train" --train-max-samples 300000 \
    --dev-max-samples 200 --dev-offset-from-train 1000000 \
    --test-dataset allenai/c4 --test-dataset-split "validation" --test-max-samples 3000 \
    --max-seq-length 32 \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "$SAMPLER" \
    --test-signals "$SAMPLER" \
    --same-train-test-signals \
    --train-token-selector $TOKEN_POS \
    --test-token-selector $TOKEN_POS \
    --precision 3 \
    --cache-dir data --output-dir outputs/runs/$EXP_GROUP \
    --output "$CELL_CONFIG" \
    --overwrite-cache

NEURONS=($(python -c "
import json
cfg = json.load(open('$CELL_CONFIG'))
print(' '.join(f\"N{s['layer']}.{s['index']}\" for s in cfg['train_signals']))
"))
NEURON_LIST=$(IFS=,; echo "${NEURONS[*]}")
echo "cell $CELL_TAG neurons: $NEURON_LIST"

for SAMPLE in $(seq 0 $((N_NEURONS_TRAIN - 1))); do
  SIG=${NEURONS[$SAMPLE]}
  EXP_NAME="joint_${SIG}_POS${TOKEN_POS}_gate${GATE}_scale${SCALE}_${KL_TAG}_w${W}_300K"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo "=== [$CELL_TAG sample $SAMPLE] $EXP_NAME ==="

  # every sample passes the SAME neuron list, so all 5 share one cached dataset;
  # --signal-filter is not part of the cache identity
  python data_prep/generate_experiment_config.py \
      --experiment-name "$EXP_NAME" \
      --model meta-llama/Llama-3.1-8B \
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
      --train-signals "$NEURON_LIST" \
      --test-signals "$NEURON_LIST" \
      --signal-filter "$SIG" \
      --train-token-selector $TOKEN_POS \
      --test-token-selector $TOKEN_POS \
      --precision 3 \
      --train-batch-size $BS \
      --gradient-accumulation-steps $ACCUM \
      --eval-batch-size 128 \
      --num-epochs 1 \
      --lora-r $LORA_R \
      --lora-alpha $LORA_ALPHA \
      --lora-dropout $LORA_DO \
      $GATE_FLAG \
      --bf16 \
      --learning-rate $LR \
      --save-steps 0 \
      --use-regressor \
      --intervention-signal-distribution extreme \
      --intervention-extreme-scale $SCALE \
      --cache-dir data \
      --output-dir outputs/runs/$EXP_GROUP \
      --output "$CONFIG" \
      $OVERWRITE

  # builds the cell's dataset on first use (or finds the existing one), no-ops afterwards
  python data_prep/cache_signals.py --config "$CONFIG"

  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      $KL_FLAG \
      $OOD_FLAG \
      $OVERWRITE

  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      $KL_FLAG \
      $OOD_FLAG \
      --ood-only \
      $OVERWRITE

  # LM cost. Both scorers skip when their output exists, so resubmitting a partially
  # finished task is cheap.
  if [ "$SAMPLE" -lt "$LM_SCORE_N" ]; then
    python evals/score_lm_degradation.py --config "$CONFIG"
  fi
  # benchmarks are far more expensive and vary with the training config, not the
  # neuron -> one per cell
  if [ "$SAMPLE" -eq 0 ]; then
    python evals/score_lighteval_benchmarks.py --config "$CONFIG" --max-samples 200
  fi
done
