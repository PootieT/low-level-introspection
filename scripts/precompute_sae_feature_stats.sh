#!/bin/bash -l
#$ -P llamagrp
#$ -N sae_feat_stats
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=1:00:00
#$ -o log
#$ -j y
#$ -t 1-7    # one task per swept layer

# Precompute per-layer SAE feature density stats used by the sweep to pick
# density-valid, non-punctuation features. Run this BEFORE the sweep, e.g.:
#     JID=$(qsub scripts/precompute_sae_feature_stats.sh | awk '{print $3}' | cut -d. -f1)
#     qsub -hold_jid $JID scripts/train_single_sae_introspector_sweep_layer_position.sh
# Each task handles one layer and SKIPS if its parquet already exists.

echo "Start: $(date)  task $SGE_TASK_ID"
module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env
export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

LAYERS=(0 2 4 6 8 10 12 14)
LAYER=${LAYERS[$(($SGE_TASK_ID-1))]}
OUT=data/sae_stats/gemma2-2b_res${LAYER}_16k.parquet

if [ -f "$OUT" ]; then
    echo "stats already exist for layer $LAYER ($OUT); skipping."
else
    echo "computing density stats for layer $LAYER -> $OUT"
    # min/max density band = Neuronpedia's alive-but-not->1% window. The sweep
    # then samples ACROSS this band (stratified) so we don't only pick the
    # high-density (punctuation-like) end.
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
