#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_endo_sweep_layer_token_qwen3_14b_hyper_tuned
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=80G
#$ -l h_rt=24:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-80

# ============================================================================
# SFT BASELINE GRID AT THE TUNED LR, Qwen3-14B-Base: endogenous loss only (w=1), gate off, no
# KL anchor. 4 layers x 4 token positions, 5 neurons per cell = 80 runs, ONE NEURON PER
# TASK (80 tasks) -- see the joint grid's header for why, and for the cache job both
# grids are held on.
# The baseline arm for train_single_mlp_introspector_joint_sweep_layer_token_qwen3_14b_hyper_tuned.sh:
# same cells, same 5 neurons per cell, same caches.
#
# >>> SUBMIT -- paste on SCC (from the repo root):
# cd /projectnb/llamagrp/peter/low-level-introspection
# qsub -N cache_qwen3_14b_ht scripts/neuron/cache_mlp_grid_qwen3_14b_hyper_tuned.sh   # skip if already submitted with the joint grid
# qsub -hold_jid cache_qwen3_14b_ht scripts/neuron/train_single_mlp_introspector_endogenous_sweep_layer_token_qwen3_14b_hyper_tuned.sh
# # once both arms finish:
# python plotters/compare_sft_vs_joint.py --model-suffix _qwen3_14b_hyper_tuned
# <<< SUBMIT
#
# LR 1e-4: one LR for both arms, and both peak there (SFT endogenous 0.97 vs 0.89 at
# 5e-5 with its sweep 19 of 36 runs in; joint interventional 0.48 vs 0.45 at 2e-4).
#
# IDENTICAL to train_single_mlp_introspector_endogenous_sweep_layer_token_llama318b.sh
# (the 5e-5 SFT grid) except: one neuron per task; model and LAYERS (5 15 25 35 -- the
# depths 1/8, 3/8, 5/8,
# 7/8 shared by every grid); LR; gate OFF only (the old grid also ran gate on); effective
# batch 32 written as 16 x 2; group mlp_endo_sweep_layer_token_qwen3_14b_hyper_tuned, a new directory because run names do
# not carry the LR; and cache_signals never gets --overwrite-cache.
#
# QWEN: base checkpoint, no BOS; the Qwen3 smoke test (cache, training with the
# alignment check, OOD eval) passes end to end.
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
export WANDB_RUN_GROUP=mlp_endo_sweep_layer_token_qwen3_14b_hyper_tuned

set -e

## ---- decode (layer, position) from the flat array index ----
# ONE NEURON PER TASK: task = gate x 80 + cell x 5 + sample. Gate OFF only here (80
# tasks); for the gated arm too, set GATES=(on off) and "#$ -t 1-160". Caches come from
# the cache job, so tasks of a cell never race.
index=$(($SGE_TASK_ID-1))
LAYERS=(5 15 25 35)
POSS=(-1 -8 -20 -28)
GATES=(off)
GATE=${GATES[$((index / 80))]}
cell_idx=$(((index % 80) / 5))
SAMPLE_ONLY=$((index % 5))
LAYER=${LAYERS[$((cell_idx / 4))]}
TOKEN_POS=${POSS[$((cell_idx % 4))]}

#   off = the adapter reaches the site, and the endogenous-only objective is the SFT
#         baseline the joint loss is compared against (same access, different objective).
W=1.0                    # weight on the ENDOGENOUS term; 1.0 => no interventional loss
SEED=21                  # neuron-selection seed (now actually used)
# The DATASET carries N_NEURONS_CACHE signals; we TRAIN on the first
# N_NEURONS_TRAIN of them. Keeping the cached list at 10 means the canonical dataset
# key matches the caches already built for the 10-neuron sweep, so a 5-neuron run
# reuses them instead of triggering 64 fresh caching passes over 300K examples.
# (--signal-filter is not part of the cache identity, so all neurons share one dataset.)
N_NEURONS_CACHE=10
N_NEURONS_TRAIN=5
LR=1e-4
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=8  # effective batch 32, as everywhere
ACCUM=4
SCALE=10                 # only affects the post-training interventional diagnostics
# OOD generalization eval on EVERY run: 3K examples per corpus, a couple of
# minutes each, and the per-(cell, corpus) activation cache is built once and
# shared by the cell's 5 neurons (and by both regimes).
OOD_FLAG="--ood-eval"
OVERWRITE=""             # empty: resubmission skips finished runs. Never pass
                         # --overwrite-cache: the caches are shared with the joint grid

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

EXP_GROUP=mlp_endo_sweep_layer_token_qwen3_14b_hyper_tuned
CELL="l${LAYER}_pos${TOKEN_POS}"
SAMPLER="{\"type\":\"uniform_layer_sampler\",\"signal_type\":\"mlp\",\"layers\":[${LAYER}],\"signals_per_layer\":${N_NEURONS_CACHE},\"seed\":${SEED}}"

## ---- resolve this layer's neuron set ONCE (identical at every position) ----
# generated, not run: it exists so resolution.py stays the single source of truth for
# which neurons a (seed, layer) maps to, instead of the selection being duplicated here
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_${CELL}.json
python data_prep/generate_experiment_config.py \
    --experiment-name "_cell_${CELL}" \
    --model Qwen/Qwen3-14B-Base \
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
echo "cell $CELL neurons: $NEURON_LIST"

for SAMPLE in $SAMPLE_ONLY; do   # this task's one neuron (see the header)
  SIG=${NEURONS[$SAMPLE]}
  EXP_NAME="endo_${SIG}_POS${TOKEN_POS}_gate${GATE}_w${W}_300K"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo "=== [$CELL sample $SAMPLE] $EXP_NAME ==="

  # every sample passes the SAME 10-signal list, so all 10 resolve to one cached
  # dataset; --signal-filter is what differs and is not part of the cache identity
  python data_prep/generate_experiment_config.py \
      --experiment-name "$EXP_NAME" \
      --model Qwen/Qwen3-14B-Base \
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

  # never builds: five tasks per cell run at once. The cache job builds it (see header)
  CACHE_DIR=$(python -c "
from pathlib import Path
from src.introspector.config import ExperimentConfig
print(ExperimentConfig.model_validate_json(Path('$CONFIG').read_text()).dataset_cache_dir)
")
  if [ ! -f "$CACHE_DIR/metadata.json" ]; then
    echo "ERROR: no complete cache at $CACHE_DIR -- run scripts/neuron/cache_mlp_grid_qwen3_14b_hyper_tuned.sh first" >&2
    exit 1
  fi

  # alignment regression check once per cell rather than on all 640 runs (it is a
  # property of the data pipeline, so one run per dataset is enough)
  if [ "$SAMPLE" -eq 0 ]; then DRIFT="--force-drift-eval"; else DRIFT=""; fi

  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      $DRIFT \
      $OOD_FLAG \
      $OVERWRITE

  python trainers/train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    $DRIFT \
    $OOD_FLAG \
    --ood-only \
    $OVERWRITE
done
