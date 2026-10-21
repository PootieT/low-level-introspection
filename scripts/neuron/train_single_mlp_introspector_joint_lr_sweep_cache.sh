#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_joint_lr_sweep_cache
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=12:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-3

# ============================================================================
# CACHE PASS for train_single_mlp_introspector_joint_lr_sweep.sh: one task per cell.
#
# qsub here does not take -t, so the array size has to live in a header -- and the
# sweep needs 72 slots where the cache pass needs 3. This wrapper only supplies the
# 3-slot header and CACHE_ONLY=1; everything else (model table, cells, cache identity)
# is the sweep script's, so the caches it builds are exactly the ones the sweep reads.
#
#   qsub -N lrcache_qwen3_8b -v SWEEP_MODEL=qwen3_8b scripts/neuron/train_single_mlp_introspector_joint_lr_sweep_cache.sh
#   qsub -N lrsweep_qwen3_8b -hold_jid lrcache_qwen3_8b -v SWEEP_MODEL=qwen3_8b scripts/neuron/train_single_mlp_introspector_joint_lr_sweep.sh
#
# Submit from the repo root (as with every script here): SGE runs a spooled copy of
# this file, so the sweep script is found relative to the working directory, not $0.
# ============================================================================

SWEEP=scripts/neuron/train_single_mlp_introspector_joint_lr_sweep.sh
if [ ! -f "$SWEEP" ]; then
  echo "ERROR: $SWEEP not found from $(pwd) -- submit from the repo root" >&2; exit 1
fi
export CACHE_ONLY=1
exec bash -l "$SWEEP"
