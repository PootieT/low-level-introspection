#!/bin/bash -l
#$ -P llamagrp
#$ -N sae_featgen_pool
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=04:00:00
#$ -o log
#$ -j y

# ============================================================================
# Stage 0 of the SAE feature-generalization experiment: the 200-feature layer-9 POOL
# (data/sae_featgen/pool_layer9.json) that featgen_common.sh trains and evaluates on.
# Re-runs the collaborator's layer-9 selection on SCC (scripts/sae/README.md section 3),
# with two additions for this experiment:
#
#   1. Gate 0 (tests/validate_sae_hook.py), then density stats over 10M C4 tokens at 128-token
#      length -- the SAME band as the layer-9 selection (density in [1e-3, 1e-2]) -- but
#      pos_density measured at the last token of a 32-token WINDOW (--pos-window 32),
#      i.e. where the training caches read out. 10M rather than 2M tokens: the
#      positives floor below sits at pos_density ~3e-4, which is only ~5 hits in 2M
#      tokens' worth of sequences.
#   2. candidates: band AND >= 50 projected positives over the 150k training texts,
#      then a SEEDED RANDOM sample of 400 (plus the 5 steering-judged features).
#   3. Neuronpedia explanations for exactly those (needs internet; SCC compute nodes have it).
#   4. build: the judged features first, then the rest in seeded random order, minus
#      features without an explanation and near-duplicate labels (MiniLM cosine >= 0.7),
#      up to 200.
#
#   qsub scripts/sae/build_featgen_pool_layer9.sh
#   STAGE=label bash scripts/sae/build_featgen_pool_layer9.sh   # redo 3-4 only (no GPU)
# Each step skips when its output exists; delete the file to redo it.
# ============================================================================

echo "Start: $(date)  on $(hostname)"
nvidia-smi || true

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

set -e -o pipefail
STAGE=${STAGE:-all}
LAYER=9
JUDGED=S9.1568,S9.4854,S9.4120,S9.978,S9.12408   # passed the blind steering judge (train_sae_joint_layer9.sh)
STATS=data/sae_stats/gemma2-2b_res${LAYER}_pos-1_w32_16k.parquet
OUT_DIR=data/sae_featgen
CANDS=$OUT_DIR/layer${LAYER}_candidates.parquet
ANNOT=$OUT_DIR/layer${LAYER}_candidates.annotated.parquet
POOL=$OUT_DIR/pool_layer${LAYER}.json
mkdir -p log $OUT_DIR

if [ "$STAGE" = "all" ] || [ "$STAGE" = "select" ]; then
    python tests/validate_sae_hook.py --model google/gemma-2-2b \
        --sae-release gemma-scope-2b-pt-res-canonical --layer $LAYER --width 16k --site res \
        | tee log/validate_sae_hook_res${LAYER}.txt
    if ! grep -qE "^[[:space:]]*PASS \(smell test\)" log/validate_sae_hook_res${LAYER}.txt; then
        echo "hook validation FAILED for layer $LAYER; stopping."; exit 1
    fi
    if [ -f "$STATS" ]; then
        echo "stats exist: $STATS"
    else
        python data_prep/select_sae_features.py \
            --model google/gemma-2-2b \
            --sae-release gemma-scope-2b-pt-res-canonical --layer $LAYER --width 16k --site res \
            --n-tokens 10000000 --max-seq-length 128 \
            --fixed-position -1 --pos-window 32 \
            --min-density 1e-3 --max-density 1e-2 \
            --cache-size 150000 \
            --out "$STATS"
    fi
    if [ -f "$CANDS" ]; then
        echo "candidates exist: $CANDS"
    else
        python data_prep/build_sae_featgen_pool.py candidates \
            --stats "$STATS" --layer $LAYER --include $JUDGED \
            --n-train-texts 150000 --min-positives 50 --n-candidates 400 --seed 0 \
            --out "$CANDS"
    fi
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "label" ]; then
    if [ -f "$ANNOT" ]; then
        echo "annotations exist: $ANNOT"
    else
        python data_prep/annotate_sae_features_neuronpedia.py \
            --stats "$CANDS" --model gemma-2-2b --layer $LAYER --width 16k --site res \
            --out "$ANNOT"
    fi
    python data_prep/build_sae_featgen_pool.py build \
        --annotated "$ANNOT" --layer $LAYER --include $JUDGED \
        --pool-size 200 --sim-threshold 0.7 --seed 0 \
        --out "$POOL"
fi
echo "Done: $(date)"
