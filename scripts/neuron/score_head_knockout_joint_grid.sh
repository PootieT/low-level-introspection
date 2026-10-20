#!/bin/bash -l
#$ -P llamagrp
#$ -N head_knockout_joint_grid
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=6:00:00
#$ -o log
#$ -j y
#$ -t 1-64

# ============================================================================
# Single-head knockout over the MAIN GRID's anchored arm (KL budget 0.1):
# 8 layers x 8 positions x 5 neurons = 320 runs, sharded over 64 array tasks
# (~5 runs each). Sharding is done by score_head_knockout.py itself from
# $SGE_TASK_ID / $SGE_TASK_LAST (src/introspector/run_discovery.shard), over the
# runs sorted by name -- so the task count above is the only knob, and changing it
# re-partitions cleanly.
#
# Per run: every head in layers L+1..15 mean-ablated alone over tokens t..last,
# evaluated endogenously and at x1 on 200 successful test examples, then
# cumulative knockouts down each arm's ranking and 3 random orderings (k <= 30).
# Cost scales with heads above the signal: L=0 runs ablate 480 heads, L=14 runs 32.
#
# --skip-existing makes resubmission cheap: a run whose head_knockout_meta.json
# exists is skipped without loading the model, so re-running this array after
# timeouts only does the unfinished runs.
#
# Aggregate afterwards (CPU, seconds):
#   python analysis/plot_head_knockout.py --group mlp_joint_grid_layer_token --kl kl0.1
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID / $SGE_TASK_LAST"
echo "=========================================================="
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

set -e

python score_head_knockout.py \
    --group mlp_joint_grid_layer_token \
    --run-glob 'joint_*_kl0.1_*' \
    --skip-existing
