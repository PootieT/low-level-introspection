#!/bin/bash -l
#$ -P llamagrp
#$ -N sae_detect_sweep_layer_token
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=18:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-128

# ============================================================================
# SAE-FEATURE DETECTION sweep over 8 layers x 8 token positions x N features,
# post-BOS-fix re-run of
# scripts_old_with_off_by_one_bug/sae/train_single_sae_introspector_sweep_layer_position.sh
#
# ONE array task = one (layer, position, gate) cell; the features run inside it, so a
# cell's dataset is built once and reused by all of them.
#
# WHY DETECTION. SAE features are sparse (~1% density), so at a fixed position a
# magnitude target is ~all zeros and a regressor wins by predicting 0. The readout is
# binary ("active"/"inactive") in the model's own tokens, scored with AP (the
# imbalance-honest headline) and AUROC; see trainers/train_sae_detector.py.
#
# FEATURE SELECTION. Drawn DENSITY-STRATIFIED from the per-layer parquet built by
# precompute_sae_feature_stats.sh: the valid pool is sorted by density and split into
# N bins, one feature per bin, so the set spans rare -> common instead of piling onto
# the frequent (punctuation-like) end. Seeded by LAYER only, so a layer's features are
# FIXED ACROSS ALL POSITIONS and the position curve is read per feature.
#
# CACHING. All N features of a cell live in ONE dataset -- encoding N features costs
# one forward pass per text -- and --signal-filter splits it per run. --cache-dir is
# the shared root, and dataset_name is derived by generate_experiment_config, so cells
# other scripts also need are not rebuilt.
#
# PREREQUISITE: qsub scripts/sae/precompute_sae_feature_stats.sh first (the picker
# falls back to uniform-random, which hits dead features, without those parquets).
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
export WANDB_RUN_GROUP=sae_detect_sweep_layer_token

set -e

## ---- decode (gate, layer, position) from the flat array index ----
# GATE is the OUTER dimension (tasks 1-64 = gate on, 65-128 = gate off): both arms of a
# cell share its cached dataset, so keeping them 64 tasks apart stops concurrent slots
# racing to build the same directory.
index=$(($SGE_TASK_ID-1))
LAYERS=(0 2 4 6 8 10 12 14)
POSS=(-1 -4 -8 -12 -16 -20 -24 -28)
GATES=(on off)
GATE=${GATES[$((index / 64))]}
cell_idx=$((index % 64))
LAYER=${LAYERS[$((cell_idx / 8))]}
TOKEN_POS=${POSS[$((cell_idx % 8))]}
#   on  = gated. No trainable parameter touches the introspected token, so the model
#         cannot read the feature: whatever detection score it gets is context bypass.
#   off = the adapter reaches the site, and the same objective can produce causal reading.

SEED=21
# cache N_FEAT_CACHE features per cell, train on the first N_FEAT_TRAIN of them
# (--signal-filter is not part of the cache identity, so all of them share one dataset)
N_FEAT_CACHE=10
N_FEAT_TRAIN=5
LR=2e-5                  # best lr for f1@0 in the single-feature lr sweep
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
OOD_FLAG="--ood-eval"
OVERWRITE=""             # set to --overwrite-cache to force regeneration

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

EXP_GROUP=sae_detect_sweep_layer_token
CELL="l${LAYER}_pos${TOKEN_POS}"

## ---- resolve this layer's feature set ONCE (identical at every position) ----
# density-stratified over the valid pool; seeded by LAYER so positions are paired
FEATURES=($(LAYER=$LAYER N=$N_FEAT_CACHE python -c '
import os, sys
import numpy as np, pandas as pd
layer=int(os.environ["LAYER"]); n=int(os.environ["N"])
p=f"data/sae_stats/gemma2-2b_res{layer}_16k.parquet"
rng=np.random.RandomState(layer)
if os.path.exists(p):
    df=pd.read_parquet(p)
    feats=df[df["selected"]].sort_values("density")["feature"].to_numpy()
    bins=[b for b in np.array_split(feats, n) if len(b)]      # rare -> common strata
    chosen=[int(rng.choice(b)) for b in bins]
    if len(chosen) < n:
        extra=[int(f) for f in feats if int(f) not in set(chosen)]
        rng.shuffle(extra); chosen += extra[:n-len(chosen)]
else:
    print(f"[warn] no stats parquet {p}; uniform-random features (may be dead/frequent)",
          file=sys.stderr)
    chosen=[int(x) for x in rng.choice(np.arange(16384), size=n, replace=False)]
print(" ".join(f"S{layer}.{i}" for i in chosen))
'))
FEATURE_LIST=$(IFS=,; echo "${FEATURES[*]}")
echo "cell $CELL features: $FEATURE_LIST"

for SAMPLE in $(seq 0 $((N_FEAT_TRAIN - 1))); do
  SIG=${FEATURES[$SAMPLE]}
  EXP_NAME="saedet_sample${SAMPLE}_${SIG}_POS${TOKEN_POS}_gate${GATE}_300K"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo "=== [$CELL sample $SAMPLE] $EXP_NAME ==="

  # every sample passes the SAME feature list, so all of them resolve to one cached
  # dataset; --signal-filter is what differs and is not part of the cache identity
  python data_prep/generate_experiment_config.py \
      --experiment-name "$EXP_NAME" \
      --model google/gemma-2-2b \
      --objective detection \
      --detection-balance-pos-frac 0.5 \
      --train-dataset allenai/c4 \
      --train-dataset-split "train" \
      --train-max-samples 300000 \
      --dev-max-samples 2000 \
      --dev-offset-from-train 1000000 \
      --test-dataset allenai/c4 \
      --test-dataset-split "validation" \
      --test-max-samples 30000 \
      --max-seq-length 32 \
      --hf-data-kwargs '{"config": "en", "streaming": true}' \
      --train-signals "$FEATURE_LIST" \
      --test-signals "$FEATURE_LIST" \
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
      --warmup-ratio 0.03 \
      --lr-scheduler-type cosine \
      --save-steps 0 \
      --cache-dir data \
      --output-dir outputs/runs/$EXP_GROUP \
      --output "$CONFIG" \
      $OVERWRITE

  # builds the cell's N-feature dataset on SAMPLE=0, no-ops for the other samples
  python data_prep/cache_signals.py --config "$CONFIG" $OVERWRITE

  # alignment regression check once per cell rather than on every run (it is a
  # property of the data pipeline, so one run per dataset is enough).
  # WANT: best offset +0, corr ~1.0, "alignment OK".
  if [ "$SAMPLE" -eq 0 ]; then DRIFT="--force-drift-eval"; else DRIFT=""; fi

  python trainers/train_sae_detector.py \
      --config "$CONFIG" \
      $DRIFT \
      $OOD_FLAG \
      $OVERWRITE

  # LM-damage checks are only meaningful for gate-OFF checkpoints (with the gate on
  # the adapter never ran over ordinary text, so a plain PEFT forward describes a
  # model that does not exist). Run them once per cell.
  if [ "$GATE" = "off" ] && [ "$SAMPLE" -eq 0 ]; then
    python evals/score_lm_degradation.py --config "$CONFIG"
    python evals/score_lighteval_benchmarks.py --config "$CONFIG" --max-samples 200
  fi
done
