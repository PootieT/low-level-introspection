#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_endo_sweep_layer_token_llama318b_hyper_tuned
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=36:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-32

# ============================================================================
# SFT BASELINE GRID AT THE TUNED LR, Llama-3.1-8B: endogenous loss only (w=1), no KL anchor,
# BOTH gates. 4 layers x 4 token positions x 2 gates, 5 neurons per cell = 32 array tasks,
# 160 runs. Tasks 1-16 = gate ON, 17-32 = gate OFF.
# The baseline arm for train_single_mlp_introspector_joint_sweep_layer_token_llama318b_hyper_tuned.sh:
# same cells, same 5 neurons per cell, same caches.
#
# >>> SUBMIT -- paste on SCC (from the repo root):
# cd /projectnb/llamagrp/peter/low-level-introspection
# # both gates (finished runs are skipped, so this is safe to resubmit):
# qsub scripts/neuron/train_single_mlp_introspector_endogenous_sweep_layer_token_llama318b_hyper_tuned.sh
# # only the gate-ON half (tasks 1-16) -- e.g. when the gate-off half has already run:
# sed 's/^#\$ -t 1-32$/#$ -t 1-16/' scripts/neuron/train_single_mlp_introspector_endogenous_sweep_layer_token_llama318b_hyper_tuned.sh > log/train_single_mlp_introspector_endogenous_sweep_layer_token_llama318b_hyper_tuned_gateon.sh
# qsub log/train_single_mlp_introspector_endogenous_sweep_layer_token_llama318b_hyper_tuned_gateon.sh
# # once both arms finish:
# python plotters/compare_sft_vs_joint.py --model-suffix _llama318b_hyper_tuned
# <<< SUBMIT
#
# LR 1e-4: the joint grid's LR for this model. Wherever both were swept (Llama-3B,
# Llama-8B, Qwen3-0.6B) the SFT optimum and the joint optimum were the same LR, so the
# two arms share one tuned LR and the comparison is between objectives, not LRs. Here
# its own SFT sweep agrees: endogenous corr 5e-5 0.87 | 1e-4 0.92 | 2e-4 0.52, all three cells at 1e-4.
#
# IDENTICAL to train_single_mlp_introspector_endogenous_sweep_layer_token_llama318b.sh
# (the 5e-5 SFT grid) except: model and LAYERS (4 12 20 28 -- the depths 1/8, 3/8, 5/8,
# 7/8 shared by every grid); LR; effective
# batch 32 written as 16 x 2; group mlp_endo_sweep_layer_token_llama318b_hyper_tuned, a new directory because run names do
# not carry the LR; and cache_signals never gets --overwrite-cache.
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
export WANDB_RUN_GROUP=mlp_endo_sweep_layer_token_llama318b_hyper_tuned

set -e

## ---- decode (layer, position) from the flat array index ----
# GATE is the OUTER dimension (tasks 1-16 = gate on, 17-32 = gate off), so the arms of a
# cell are 16 tasks apart and never race to build its cache. Gate off was the first arm
# run here (as tasks 1-16 of a GATES=(off) version, under the same run names); gate on
# was added after, as the bypass baseline the old 5e-5 grids also had.
index=$(($SGE_TASK_ID-1))
LAYERS=(4 12 20 28)
POSS=(-1 -8 -20 -28)
GATES=(on off)
GATE=${GATES[$((index / 16))]}
cell_idx=$((index % 16))
LAYER=${LAYERS[$((cell_idx / 4))]}
TOKEN_POS=${POSS[$((cell_idx % 4))]}

#   on  = gated. No trainable parameter touches the introspected token, so the model
#         cannot read the neuron: interventional corr pins at 0, and whatever endogenous
#         score it gets is context bypass. Same LR as gate off -- not separately tuned.
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
BS=16  # effective batch 32, as everywhere
ACCUM=2
SCALE=10                 # only affects the post-training interventional diagnostics
# OOD generalization eval on EVERY run: 3K examples per corpus, a couple of
# minutes each, and the per-(cell, corpus) activation cache is built once and
# shared by the cell's 5 neurons (and by both regimes).
OOD_FLAG="--ood-eval"
OVERWRITE=""             # empty: resubmission skips finished runs. Never pass
                         # --overwrite-cache: the caches are shared with the joint grid

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

EXP_GROUP=mlp_endo_sweep_layer_token_llama318b_hyper_tuned
CELL="l${LAYER}_pos${TOKEN_POS}"
SAMPLER="{\"type\":\"uniform_layer_sampler\",\"signal_type\":\"mlp\",\"layers\":[${LAYER}],\"signals_per_layer\":${N_NEURONS_CACHE},\"seed\":${SEED}}"

## ---- resolve this layer's neuron set ONCE (identical at every position) ----
# generated, not run: it exists so resolution.py stays the single source of truth for
# which neurons a (seed, layer) maps to, instead of the selection being duplicated here
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_${CELL}.json
python data_prep/generate_experiment_config.py \
    --experiment-name "_cell_${CELL}" \
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
echo "cell $CELL neurons: $NEURON_LIST"

for SAMPLE in $(seq 0 $((N_NEURONS_TRAIN - 1))); do
  SIG=${NEURONS[$SAMPLE]}
  EXP_NAME="endo_${SIG}_POS${TOKEN_POS}_gate${GATE}_w${W}_300K"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo "=== [$CELL sample $SAMPLE] $EXP_NAME ==="

  # every sample passes the SAME 10-signal list, so all 10 resolve to one cached
  # dataset; --signal-filter is what differs and is not part of the cache identity
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

  # builds the cell's 10-neuron dataset on SAMPLE=0, no-ops for the other 9
  python data_prep/cache_signals.py --config "$CONFIG"

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
