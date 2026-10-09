#!/bin/bash
#SBATCH --job-name=sae_feat_stats
#SBATCH --partition=rtx-batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=log/%x-%A_%a.out
#SBATCH --array=1-5    # one task per (layer, position); for the 5-position check use --array=1-25

# ============================================================================
# Step 1 -- per-layer SAE feature density stats, consumed by Steps 1.5/1.6 and the
# training sweep's feature picker.
#
# The sweep draws features DENSITY-STRATIFIED from `selected` (density in
# [min_density, max_density]); max_density=0.01 mirrors Neuronpedia's default
# "hide >1% density" filter, which keeps punctuation/position features out.
# Without these parquets the picker falls back to uniform-random over 16k features,
# most of which are dead or near-dead at a fixed position.
#
# Grid = LAYERS x POSITIONS (task id -> layer-major). Selection runs at POSITIONS=-1;
# the later per-position density check re-runs it at the training positions:
#     sbatch scripts/sae/precompute_sae_feature_stats.sh                       # selection
#     POSITIONS="1 8 16 24 31" sbatch --array=1-25 scripts/sae/precompute_sae_feature_stats.sh
# Positions are absolute token indices (0 = BOS is invalid for SAEs) or negative from the end.
#
# SLURM (AICR cluster). Submit from the repo root.
# Uses the dedicated introspection_env (torch cu128 + sae_lens + nnsight +
# transformers); override with INTRO_ENV. Each task SKIPS if its parquet exists.
# ============================================================================
echo "Start: $(date)  array task $SLURM_ARRAY_TASK_ID on $(hostname)"
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${INTRO_ENV:-/work/neu/p2026_0061_neu/low-level-introspection/introspection_env}
export HF_HOME=${HF_HOME:-/work/neu/p2026_0061_neu/low-level-introspection/.hf_cache}

set -e -o pipefail

LAYERS=(${LAYERS_OVERRIDE:-0 5 10 15 20})
POSITIONS=(${POSITIONS:--1})
IDX=$(($SLURM_ARRAY_TASK_ID-1))
LAYER=${LAYERS[$((IDX / ${#POSITIONS[@]}))]}
POS=${POSITIONS[$((IDX % ${#POSITIONS[@]}))]}
if [ -z "$LAYER" ]; then echo "task $SLURM_ARRAY_TASK_ID out of grid range; nothing to do."; exit 0; fi
OUT=data/sae_stats/gemma2-2b_res${LAYER}_pos${POS}_16k.parquet

# Gate 0 once per layer (only on the selection pass): is the SAE on the right residual?
if [ "$POS" = "-1" ]; then
    python validate_sae_hook.py --model google/gemma-2-2b \
        --sae-release gemma-scope-2b-pt-res-canonical --layer $LAYER --width 16k --site res \
        | tee log/validate_sae_hook_res${LAYER}.txt
    if ! grep -qE "^[[:space:]]*PASS \(smell test\)" log/validate_sae_hook_res${LAYER}.txt; then   # verdict line only, not the printed contexts
        echo "hook validation FAILED for layer $LAYER; not computing stats."; exit 1
    fi
fi

if [ -f "$OUT" ]; then
    echo "stats already exist for layer $LAYER pos $POS ($OUT); skipping."
else
    echo "computing density stats for layer $LAYER pos $POS -> $OUT"
    python select_sae_features.py \
        --model google/gemma-2-2b \
        --sae-release gemma-scope-2b-pt-res-canonical \
        --layer $LAYER --width 16k --site res \
        --n-tokens 2000000 --max-seq-length 128 \
        --fixed-position $POS \
        --min-density 1e-3 --max-density 1e-2 \
        --out $OUT
fi
echo "Done: $(date)"
