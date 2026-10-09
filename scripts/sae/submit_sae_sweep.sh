#!/bin/bash
# ============================================================================
# Submit the layer x position SAE sweep: cache array, then the training array
# chained with --dependency=afterok (training starts only once every cache exists).
# Array sizes follow the grid in sweep_common.sh, so overrides propagate:
#
#     bash scripts/sae/submit_sae_sweep.sh                     # full grid, W = 1.0 0.9
#     WEIGHTS="0.9" bash scripts/sae/submit_sae_sweep.sh       # one weight only
#     MAX_PARALLEL=20 bash scripts/sae/submit_sae_sweep.sh     # cap concurrent training tasks
#     DRY_RUN=1 bash scripts/sae/submit_sae_sweep.sh           # print the plan, submit nothing
#
# Run from a node with sbatch (login node or an interactive job).
# ============================================================================
set -e
cd /work/neu/p2026_0061_neu/low-level-introspection
source scripts/sae/sweep_common.sh

for L in "${LAYERS[@]}"; do features_for_layer "$L" > /dev/null; done   # fail fast on TODOs
N_CACHE=$((N_L * N_P))
N_TRAIN=$((N_FEATS * N_L * N_P * N_W))
THROTTLE=${MAX_PARALLEL:+%$MAX_PARALLEL}
PER_FEAT=$((N_L * N_P * N_W))

echo "layers=(${LAYERS[*]})  positions=(${POSITIONS[*]})  weights=(${WEIGHTS[*]})  group=$EXP_GROUP"
for L in "${LAYERS[@]}"; do echo "  layer $L: $(features_for_layer "$L")"; done
echo "cache tasks: $N_CACHE   training tasks: $N_TRAIN"
for k in $(seq 1 $N_FEATS); do
    echo "  F$k = tasks $(( (k-1)*PER_FEAT + 1 ))-$(( k*PER_FEAT ))"
done
[ -n "$DRY_RUN" ] && { echo "DRY_RUN set; nothing submitted."; exit 0; }

mkdir -p log
CACHE_JOB=$(sbatch --parsable --array=1-$N_CACHE scripts/sae/cache_sae_sweep.sh)
TRAIN_JOB=$(sbatch --parsable --array=1-$N_TRAIN$THROTTLE --dependency=afterok:$CACHE_JOB \
            scripts/sae/train_sae_sweep.sh)
echo "submitted cache job $CACHE_JOB and training job $TRAIN_JOB (after cache succeeds)"
