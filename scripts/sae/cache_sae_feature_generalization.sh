#!/bin/bash -l
#$ -P llamagrp
#$ -N sae_featgen_cache
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=08:00:00
#$ -o log
#$ -j y

# ============================================================================
# Stage 1 of the SAE feature-generalization experiment: build EVERY cache it needs,
# in one job, so the training tasks (stage 2) never race to build the same directory.
#
#   the POOL cache   all 200 pool features: 150k train texts (active rows only, for the
#                    eval's 'on' magnitudes) + 30k C4-validation test texts (the eval set)
#   one RUN cache per distinct (cache_feats, per-feature quota) in the table: the
#                    first cache_feats pool features over the 150k train texts, inactive
#                    rows subsampled to the quota. The label and index rows at one N share
#                    one; the three data100k rows share one 100-feature cache.
#
# Each forward pass over 150k 32-token texts is ~15 min on one GPU (the layer-9
# single-feature cache, 180k texts, took ~13 min). Resubmitting skips built caches.
#
#   qsub scripts/sae/cache_sae_feature_generalization.sh
#   qsub -hold_jid sae_featgen_cache -t 1-5 scripts/sae/train_sae_feature_generalization.sh
# Only the data100k cache (pool cache already built):
#   qsub -v STAGE_GROUPS=data100k scripts/sae/cache_sae_feature_generalization.sh
#
# PREREQUISITE: data/sae_featgen/pool_layer9.json (scripts/sae/build_featgen_pool_layer9.sh).
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME   Job ID : $JOB_ID"
echo "=========================================================="
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

set -e
source scripts/sae/featgen_common.sh
[ -f "$POOL_JSON" ] || { echo "ERROR: $POOL_JSON missing; run build_featgen_pool_layer9.sh first" >&2; exit 1; }

# A partial (STAGE_GROUPS) build leaves an existing pool config alone: training tasks
# that are still running read it at eval time, and make_pool_config rewrites it.
if [ -z "$STAGE_GROUPS" ] || [ ! -f "$POOL_CONFIG" ]; then make_pool_config; fi
echo "=== pool cache: $POOL_CONFIG -> $(cache_dir_of "$POOL_CONFIG") ==="
python cache_signals.py --config "$POOL_CONFIG"

# one run cache per distinct (features cached, inactive keep rate) -- the cache's
# identity: both arms at one N share one, and so do the data100k rows at every N.
# The first row with a given identity generates the config that builds it.
# STAGE_GROUPS limits which table groups get caches (e.g. "data100k" to add those
# without re-walking the rest; built caches are skipped either way).
DONE=" "
for row in "${RUNS[@]}"; do
    read -r name group arm n total cache_feats <<< "$row"
    if [ -n "$STAGE_GROUPS" ] && [[ " $STAGE_GROUPS " != *" $group "* ]]; then continue; fi
    key="${cache_feats}:$(negative_keep "$n" "$total")"
    case "$DONE" in *" $key "*) continue ;; esac
    DONE="$DONE$key "
    CONFIG=$(make_run_config "$name" "$arm" "$n" "$total" "$cache_feats")
    echo "=== run cache $key (row $name): $CONFIG -> $(cache_dir_of "$CONFIG") ==="
    python cache_signals.py --config "$CONFIG"
done
echo "Done: $(date)"
