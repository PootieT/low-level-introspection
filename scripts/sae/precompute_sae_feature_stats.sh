#!/bin/bash -l
#$ -P llamagrp
#$ -N sae_feat_stats
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=2:00:00
#$ -o log
#$ -j y
#$ -t 1-8    # one task per swept layer

# ============================================================================
# Per-layer SAE feature density stats, consumed by the sweep's feature picker.
#
# The sweep draws features DENSITY-STRATIFIED from `selected` (density in
# [min_density, max_density]); max_density=0.01 mirrors Neuronpedia's default
# "hide >1% density" filter, which is what keeps punctuation/position features out.
# Without these parquets the picker falls back to uniform-random over 16k features,
# most of which are dead or near-dead at a fixed position.
#
# Run BEFORE the sweep; one task per layer, so no two tasks write the same file:
#     JID=$(qsub scripts/sae/precompute_sae_feature_stats.sh | awk '{print $3}' | cut -d. -f1)
#     qsub -hold_jid $JID scripts/sae/train_single_sae_introspector_sweep_layer_position.sh
# Each task SKIPS if its parquet already exists.
# ============================================================================

echo "Start: $(date)  task $SGE_TASK_ID"
module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env
export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

set -e

LAYERS=(0 2 4 6 8 10 12 14)
LAYER=${LAYERS[$(($SGE_TASK_ID-1))]}
OUT=data/sae_stats/gemma2-2b_res${LAYER}_16k.parquet

if [ -f "$OUT" ]; then
    echo "stats already exist for layer $LAYER ($OUT); skipping."
else
    echo "computing density stats for layer $LAYER -> $OUT"
    python select_sae_features.py \
        --model google/gemma-2-2b \
        --sae-release gemma-scope-2b-pt-res-canonical \
        --layer $LAYER --width 16k --site res \
        --n-tokens 2000000 --max-seq-length 128 \
        --fixed-position -1 \
        --min-density 1e-3 --max-density 1e-2 \
        --out $OUT
fi
echo "Done: $(date)"
