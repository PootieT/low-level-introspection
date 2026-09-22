#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_joint_grid_layer_token_llama318b
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=12:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-32

# ============================================================================
# MAIN GRID: joint introspection training across 8 layers x 8 token positions,
# 5 neurons per cell, in TWO cost regimes.
#
#   ANCHORED   --kl-anchor-target 0.1   "at equal, small behavioural cost, where can
#                                        the model read itself?"  (ppl ~x1.08)
#   UNANCHORED no anchor                 the performance CEILING, and -- via
#                                        score_lm_degradation -- the KL the recipe
#                                        pays when nothing stops it. That achieved KL
#                                        is a MEASUREMENT per cell, which is the
#                                        cost-of-access map.
#
# 4 x 4 x 2 = 32 array tasks, 5 neurons inside each = 160 runs. One task = one
# (layer, position, regime) cell; its 5 neurons share one cached dataset.
#
# WHY NOT A SECOND FIXED BUDGET. A "loose" cap of 3.0 binds unevenly across cells
# (achieved KL 0.93 / 1.17 / 1.77 at the three pre-flight cells), so cross-cell
# comparisons would mix site difficulty with how much the cap happened to bite.
# Unanchored has no such hyperparameter, and its achieved KL is the quantity of
# interest rather than a setting.
#
# READING THE UNANCHORED ARM. Those models are behaviourally wrecked (ppl x15.8 at
# L10 POS-1), so its INTERVENTIONAL score is the ceiling while its ENDOGENOUS score
# carries the drift caveat -- at high budgets the neuron's own value is compressed
# (gate_proj std ratio down to 0.34), i.e. the model reports a signal it rewrote.
# And the achieved KL is "what this recipe pays when unconstrained" (LR, rank, one
# epoch), not an intrinsic property of the site.
#
# W=0.01: within 1 SE of the best in both non-floor cells of the weight sweep.
# GATE OFF is mandatory -- gated runs read nothing (interventional ~0 everywhere).
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
export WANDB_RUN_GROUP=mlp_joint_grid_layer_token_llama318b

set -e

## ---- decode (layer, position, regime) from the flat array index ----
# REGIME is the OUTER dimension on purpose. The two regimes of a cell share its cached
# dataset, and whichever runs first builds it; putting them adjacent (regime innermost)
# means concurrent array slots would race to build the same directory. With regime
# outer they are 64 tasks apart, and every set of concurrently-running tasks touches a
# DIFFERENT cell. (For a hard guarantee, run a cache-only pass first and submit the
# training array with -hold_jid.)
index=$(($SGE_TASK_ID-1))
#index=11
LAYERS=(4 12 20 28)
# N2.2566,N2.1886,N2.5949,N2.6998,N2.1447,N2.868,N2.1134,N2.1027,N2.8085,N2.1583
POSS=(-1 -8 -20 -28)
REGIMES=("kl0.1" "unanchored")
REGIME=${REGIMES[$((index / 16))]}
cell_idx=$((index % 16))
LAYER=${LAYERS[$((cell_idx / 4))]}
TOKEN_POS=${POSS[$((cell_idx % 4))]}

if [ "$REGIME" = "unanchored" ]; then
  KL_FLAG=""
  KL_TAG="klnone"
else
  KL_BUDGET=${REGIME#kl}
  KL_FLAG="--kl-anchor-target $KL_BUDGET"
  KL_TAG="kl${KL_BUDGET}"
fi

GATE=off
W=0.01
SEED=21
# The DATASET carries N_NEURONS_CACHE signals; we TRAIN on the first
# N_NEURONS_TRAIN of them. Keeping the cached list at 10 means the canonical dataset
# key matches the caches already built for the 10-neuron sweep, so a 5-neuron run
# reuses them instead of triggering 64 fresh caching passes over 300K examples.
# (--signal-filter is not part of the cache identity, so all neurons share one dataset.)
N_NEURONS_CACHE=10
N_NEURONS_TRAIN=5
SCALE=10
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
# the unanchored arm's achieved KL IS the cost map, so score more than one neuron
# there to get an error bar on it; the anchored arm only needs a spot check
if [ "$REGIME" = "unanchored" ]; then LM_SCORE_N=2; else LM_SCORE_N=1; fi
# OOD generalization eval on EVERY run: 3K examples per corpus, a couple of
# minutes each, and the per-(cell, corpus) activation cache is built once and
# shared by the cell's 5 neurons (and by both regimes).
OOD_FLAG="--ood-eval"
OVERWRITE=""

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

EXP_GROUP=mlp_joint_grid_layer_token_llama318b
CELL_TAG="l${LAYER}_pos${TOKEN_POS}"
SAMPLER="{\"type\":\"uniform_layer_sampler\",\"signal_type\":\"mlp\",\"layers\":[${LAYER}],\"signals_per_layer\":${N_NEURONS_CACHE},\"seed\":${SEED}}"
echo "cell=$CELL_TAG regime=$REGIME"

## ---- resolve this layer's neuron set ONCE (identical at every position/regime) ----
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_${CELL_TAG}.json
python generate_experiment_config.py \
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
  echo "=== [$CELL_TAG $REGIME sample $SAMPLE] $EXP_NAME ==="

  # every sample passes the SAME neuron list, so all 5 share one cached dataset (and
  # both regimes share it too); --signal-filter is not part of the cache identity
  python generate_experiment_config.py \
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

  # builds the cell's 5-neuron dataset on first use, no-ops afterwards (shared by
  # both regimes, so the second array task for this cell never re-caches)
  python cache_signals.py --config "$CONFIG" $OVERWRITE

  python train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      $KL_FLAG \
      $OOD_FLAG \
      $OVERWRITE

  python train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      $KL_FLAG \
      $OOD_FLAG \
      --ood-only \
      $OVERWRITE

  # LM cost. Both scorers skip when their output exists, so resubmitting a partially
  # finished task is cheap.
  if [ "$SAMPLE" -lt "$LM_SCORE_N" ]; then
    python score_lm_degradation.py --config "$CONFIG"
  fi
  # benchmarks are far more expensive and vary with the training config, not the
  # neuron -> one per cell
  if [ "$SAMPLE" -eq 0 ]; then
    python score_lighteval_benchmarks.py --config "$CONFIG" --max-samples 200
  fi
done
