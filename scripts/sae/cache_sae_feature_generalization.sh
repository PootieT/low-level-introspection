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
#   one RUN cache per distinct N in the table: its first N features over the 150k
#                    train texts, inactive rows subsampled to that N's quota. The label
#                    and index rows at one N share it.
#
# Each forward pass over 150k 32-token texts is ~15 min on one GPU (the layer-9
# single-feature cache, 180k texts, took ~13 min). Resubmitting skips built caches.
#
#   qsub scripts/sae/cache_sae_feature_generalization.sh
#   qsub -hold_jid sae_featgen_cache -t 1-5 scripts/sae/train_sae_feature_generalization.sh
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

make_pool_config
echo "=== pool cache: $POOL_CONFIG -> $(cache_dir_of "$POOL_CONFIG") ==="
python cache_signals.py --config "$POOL_CONFIG"

# one run cache per distinct N (the first row at each N generates the shared config)
DONE_N=" "
for row in "${RUNS[@]}"; do
    read -r name _ arm n <<< "$row"
    case "$DONE_N" in *" $n "*) continue ;; esac
    DONE_N="$DONE_N$n "
    CONFIG=$(make_run_config "$name" "$arm" "$n")
    echo "=== run cache N=$n: $CONFIG -> $(cache_dir_of "$CONFIG") ==="
    python cache_signals.py --config "$CONFIG"
done
echo "Done: $(date)"
