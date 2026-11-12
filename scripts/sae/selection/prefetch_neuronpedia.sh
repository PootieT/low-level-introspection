#!/bin/bash
# ============================================================================
# Fetch and cache the Neuronpedia records of every candidate up to config max_rank, for every
# layer in config.json (data/neuronpedia_cache/). Needs network, so run it on a login or
# interactive node, not in the offline selection job. It skips features already cached, so
# rerunning it after raising max_rank fetches only the new ranks.
#     bash scripts/sae/selection/prefetch_neuronpedia.sh
# ============================================================================
set -e
ROOT=/work/neu/p2026_0061_neu/low-level-introspection; cd $ROOT
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ${INTRO_ENV:-$ROOT/introspection_env}
MAXR=$(python -c "import json; print(json.load(open('scripts/sae/selection/config.json'))['candidates']['max_rank'])")
python data_prep/sae_select_candidates.py --depth-all $MAXR \
    --out data/sae_selection/prefetch/candidates_$MAXR.csv --annotated-out data/sae_selection/prefetch/annotated
