#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_sft_kl_grid_layer_token
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=12:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-64

# ============================================================================
# SFT + KL BASELINE (rebuttal): endogenous-only training (W=1.0) under the SAME KL
# budget as the main joint arm (--kl-anchor-target 0.1), on the main 1B grid:
# 8 layers x 8 token positions x 5 neurons = 320 runs, one array task per cell.
#
# THE ONE CHANGE from scripts/neuron/train_single_mlp_introspector_joint_sweep_layer_token.sh
# (anchored arm) is W=0.01 -> W=1.0, so SFT+KL vs Joint+KL isolates the interventional
# loss at equal behavioural cost. Everything else is copied from that script on purpose,
# INCLUDING its LR schedule (linear, no warm-up). The plain SFT grid
# (train_single_mlp_introspector_endogenous_sweep_layer_token.sh) used cosine with 3%
# warm-up, so SFT -> SFT+KL changes the schedule as well as adding the anchor.
#
# PRIORITY. Tasks 1-4 are the four rebuttal locales (L2/L8 x POS-1/POS-28, i.e. t=31/4);
# tasks 5-64 are the rest of the grid in layer-major order. SGE dispatches low task ids
# first, so one submission runs the rebuttal cells first. To run ONLY those, change the
# header to "#$ -t 1-4" (qsub rejects -t on the command line), then "#$ -t 5-64" later.
#
# OUTPUT GROUP is its own (mlp_sft_kl_grid_layer_token), NOT mlp_endo_sweep_layer_token:
# plotters/compare_sft_vs_joint.py's sft_gateoff arm filters on gate and w only, so
# KL-tagged runs placed in the SFT group would be silently averaged into plain SFT.
#
# CACHING. The neuron list is the joint/SFT grids' (same sampler, same 10-neuron cache
# list), but a config generated today hashes to a NEW dataset name (the split identity
# gained min_seq_length on 2026-09-06). data_prep/pin_dataset_cache.py therefore points
# each run at the plain SFT run's existing 300K cache after checking every content field
# matches, so cache_signals no-ops; if that cache is missing it falls back to building.
# Cells are one per task, so concurrent tasks never share a cache.
#
# Afterwards:
#   python plotters/table_sft_kl_baseline.py                 # 4-locale rebuttal table
#   python plotters/compare_sft_vs_joint.py --include-sft-kl # OOD ladder, once ALL 320 exist
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

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection
export WANDB_RUN_GROUP=mlp_sft_kl_grid_layer_token

set -e

## ---- cell order: rebuttal locales first, then the rest of the grid ----
LAYERS=(0 2 4 6 8 10 12 14)
POSS=(-1 -4 -8 -12 -16 -20 -24 -28)
PRIORITY=("2:-1" "2:-28" "8:-1" "8:-28")
CELLS=("${PRIORITY[@]}")
for L in "${LAYERS[@]}"; do
  for P in "${POSS[@]}"; do
    case " ${PRIORITY[*]} " in *" $L:$P "*) continue ;; esac
    CELLS+=("$L:$P")
  done
done
if [ ${#CELLS[@]} -ne 64 ]; then echo "expected 64 cells, got ${#CELLS[@]}"; exit 1; fi
CELL=${CELLS[$((SGE_TASK_ID - 1))]}
LAYER=${CELL%%:*}
TOKEN_POS=${CELL##*:}

KL_BUDGET=0.1
KL_TAG="kl${KL_BUDGET}"
GATE=off                 # the KL anchor only applies with the adapter live over the text
W=1.0                    # ENDOGENOUS ONLY -- the one change from the joint arm (W=0.01)
SEED=21
N_NEURONS_CACHE=10
N_NEURONS_TRAIN=5
SCALE=10                 # only shapes the post-training interventional diagnostics at W=1
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
LM_SCORE_N=1             # achieved KL on one neuron per cell, as in the joint anchored arm
OOD_FLAG="--ood-eval"
OVERWRITE=""

EXP_GROUP=mlp_sft_kl_grid_layer_token
CELL_TAG="l${LAYER}_pos${TOKEN_POS}"
SAMPLER="{\"type\":\"uniform_layer_sampler\",\"signal_type\":\"mlp\",\"layers\":[${LAYER}],\"signals_per_layer\":${N_NEURONS_CACHE},\"seed\":${SEED}}"
echo "task $SGE_TASK_ID -> cell=$CELL_TAG (layer $LAYER, position $TOKEN_POS), W=$W, KL budget $KL_BUDGET"

## ---- resolve this layer's neuron set ONCE (identical to the joint grid's) ----
mkdir -p experiment_configs/$EXP_GROUP
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_${CELL_TAG}.json
python data_prep/generate_experiment_config.py \
    --experiment-name "_cell_${CELL_TAG}" \
    --model meta-llama/Llama-3.2-1B \
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
  EXP_NAME="endo_${SIG}_POS${TOKEN_POS}_gate${GATE}_${KL_TAG}_w${W}_300K"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo "=== [$CELL_TAG sample $SAMPLE] $EXP_NAME ==="

  python data_prep/generate_experiment_config.py \
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
      --train-signals "$NEURON_LIST" \
      --test-signals "$NEURON_LIST" \
      --signal-filter "$SIG" \
      --train-token-selector $TOKEN_POS \
      --test-token-selector $TOKEN_POS \
      --precision 3 \
      --train-batch-size $BS \
      --eval-batch-size 128 \
      --num-epochs 1 \
      --lora-r $LORA_R \
      --lora-alpha $LORA_ALPHA \
      --lora-dropout $LORA_DO \
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

  # reuse the plain SFT run's 300K cache (same neurons, same texts): a config generated
  # today hashes to a new name since 2026-09-06 and would rebuild it from scratch
  python data_prep/pin_dataset_cache.py --config "$CONFIG" \
      --reference experiment_configs/mlp_endo_sweep_layer_token/endo_${SIG}_POS${TOKEN_POS}_gate${GATE}_w${W}_300K.json

  python data_prep/cache_signals.py --config "$CONFIG" $OVERWRITE

  # --require-kl-anchor: refuse rather than train unanchored under a _kl0.1_ name
  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      --kl-anchor-target $KL_BUDGET \
      --require-kl-anchor \
      $OOD_FLAG \
      $OVERWRITE

  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      --kl-anchor-target $KL_BUDGET \
      --require-kl-anchor \
      $OOD_FLAG \
      --ood-only \
      $OVERWRITE

  # achieved KL, to confirm the budget binds the same way it does for the joint arm
  if [ "$SAMPLE" -lt "$LM_SCORE_N" ]; then
    python evals/score_lm_degradation.py --config "$CONFIG"
  fi
done
