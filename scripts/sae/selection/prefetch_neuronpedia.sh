#!/bin/bash
set -e
ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}}; cd $ROOT
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ${INTRO_ENV:-$ROOT/introspection_env}
MAXR=$(python -c "import json; print(json.load(open('scripts/sae/selection/config.json'))['candidates']['max_rank'])")
python data_prep/sae_select_candidates.py --depth-all $MAXR \
    --out data/sae_selection/prefetch/candidates_$MAXR.csv --annotated-out data/sae_selection/prefetch/annotated
