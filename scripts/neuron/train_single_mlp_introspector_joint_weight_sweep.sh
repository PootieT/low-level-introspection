#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_joint_weight_sweep
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=12:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 14,15

# ============================================================================
# Joint-loss WEIGHT sweep: fixes W for the main layer x token grid, and produces the
# endogenous-vs-interventional pareto with error bars.
#
# 3 cells x 6 weights x 5 neurons = 90 runs. One array task = one (cell, weight);
# the 5 neurons run inside it and share the cell's cached dataset.
#
# WHY 5 NEURONS. At N10.24 POS-1 interventional corr swung 0.68 -> 0.90
# non-monotonically across KL budgets on a SINGLE neuron, so run-to-run spread is
# ~+-0.1. A weight difference smaller than that cannot be read off one neuron; the
# spread across 5 is also the error bar for the pareto.
#
# WHY THESE CELLS. From the KL pre-flight (measured at L=32):
#   N2.24  POS-1   easy    interventional 0.98 @ budget 0.1
#   N10.24 POS-1   medium  interventional 0.84 @ budget 0.1
#   N10.24 POS-28  hard    interventional 0.18 @ budget 0.1  <-- FLOOR
# The hard cell is floor-limited at 0.1: only budget 3.0 lifts it off the floor
# (0.73), so it runs at 3.0. That buys a readable W trend at the hard site, at the
# cost of a heavily degraded LM there (ppl x5.7) -- so compare W-shapes WITHIN a
# cell, never absolute corr ACROSS cells with different budgets.
#
# W=0 is pure interventional and W=1 pure endogenous; trainers/train_introspector_joint.py
# skips the zero-weighted forward, so the endpoints cost what a single-objective
# trainer costs.
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
export WANDB_RUN_GROUP=mlp_joint_weight_sweep

set -e

## ---- decode (cell, weight) from the flat array index ----
index=$(($SGE_TASK_ID-1))  # missing job id: 7, 14, 15
# cell := layer:position:kl_budget
CELLS=("2:-1:0.1" "10:-1:0.1" "10:-28:3.0")
WS=(0.0 0.01 0.03 0.1 0.3 1.0)
CELL=${CELLS[$((index / 6))]}
W=${WS[$((index % 6))]}
IFS=':' read -r LAYER TOKEN_POS KL <<< "$CELL"

GATE=off                 # required: gated runs read nothing (interventional ~0)
SEED=21                  # neuron-selection seed (resolution.py seeds per (seed, layer))
N_NEURONS=5
SCALE=10
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
OVERWRITE=""

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

EXP_GROUP=mlp_joint_weight_sweep
CELL_TAG="l${LAYER}_pos${TOKEN_POS}_kl${KL}"
SAMPLER="{\"type\":\"uniform_layer_sampler\",\"signal_type\":\"mlp\",\"layers\":[${LAYER}],\"signals_per_layer\":${N_NEURONS},\"seed\":${SEED}}"
echo "cell=$CELL_TAG  W=$W"

## ---- resolve this layer's neuron set ONCE (identical at every weight) ----
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

for SAMPLE in $(seq 0 $((N_NEURONS - 1))); do
  SIG=${NEURONS[$SAMPLE]}
  EXP_NAME="joint_${SIG}_POS${TOKEN_POS}_gate${GATE}_scale${SCALE}_kl${KL}_w${W}_300K"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo "=== [$CELL_TAG w=$W sample $SAMPLE] $EXP_NAME ==="

  # all samples pass the SAME neuron list, so the 5 runs share one cached dataset;
  # --signal-filter is what differs and is not part of the cache identity
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

  # builds the cell's 5-neuron dataset on the first sample, no-ops afterwards
  python data_prep/cache_signals.py --config "$CONFIG" $OVERWRITE

  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      --kl-anchor-target $KL \
      $OVERWRITE

  # LM cost. Both scorers skip when their output already exists, so re-submitting a
  # partially-finished task is cheap. Benchmarks only on SAMPLE=0: they are far more
  # expensive than the KL scorer and vary with the training config, not the neuron.
  python evals/score_lm_degradation.py --config "$CONFIG"
  if [ "$SAMPLE" -eq 0 ]; then
    python evals/score_lighteval_benchmarks.py --config "$CONFIG" --max-samples 200
  fi
done
