#!/bin/bash
echo "Start: $(date)"
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${INTRO_ENV:-/work/neu/p2026_0061_neu/low-level-introspection/introspection_env}

set -e

LAYERS=(${LAYERS_OVERRIDE:-0 5 10 15 20})
for LAYER in "${LAYERS[@]}"; do
    STATS=data/sae_stats/gemma2-2b_res${LAYER}_pos-1_16k.parquet
    OUT=data/sae_stats/gemma2-2b_res${LAYER}_pos-1_16k.annotated.parquet
    if [ ! -f "$STATS" ]; then
        echo "no stats for layer $LAYER ($STATS); run Step 1 first. skipping."
        continue
    fi
    if [ -f "$OUT" ]; then
        echo "annotated parquet already exists for layer $LAYER ($OUT); skipping."
        continue
    fi
    echo "annotating layer $LAYER -> $OUT"
    python annotate_sae_features_neuronpedia.py \
        --stats "$STATS" \
        --model gemma-2-2b --layer $LAYER --width 16k --site res \
        --max-fetch ${MAX_FETCH:-200} \
        --out "$OUT"
done
echo "Done: $(date)"
