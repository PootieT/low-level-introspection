#!/bin/bash -l
#$ -P llamagrp
#$ -N head_set_ablation_joint_grid
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=4:00:00
#$ -o log
#$ -j y
#$ -t 1-10

# ============================================================================
# Head-SET ablation (rebuttal: do endogenous reports read the site?) over the MAIN
# GRID's anchored arm (KL budget 0.1), 4 cells x 5 neurons = 20 runs, 2 per task:
#   layers {2, 8} x positions {-1 (t=31), -28 (t=4)}
#
# Needs each run's head_knockout/ results (scripts/neuron/score_head_knockout_joint_grid.sh).
# Per run: E and I head sets (80%-mass and top-30) from the single-head knockout,
# then ablate E\I, I\E, E&I, all-but-E, all-but-I (+5 layer-matched random controls
# each), evaluated endogenously and at x1 on the ~2.8K test rows the knockout did NOT
# rank heads on. Reports corr(pred, a), corr(pred, v) and corr(pred, a) under x1.
#
# Optional args: $1 = cells, $2 = head set definitions (see the defaults below).
# Sharding over the 20 runs is done by the script from $SGE_TASK_ID / $SGE_TASK_LAST.
#
# Aggregate afterwards (CPU, seconds):
#   python plotters/plot_head_set_ablation.py --group mlp_joint_grid_layer_token
# ============================================================================

CELLS=${1:-2:-1,2:-28,8:-1,8:-28}
HEAD_DEFS=${2:-mass0.8,top30}

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID / $SGE_TASK_LAST"
echo "Cells : $CELLS   Head defs : $HEAD_DEFS"
echo "=========================================================="
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

set -e

python mechanistic_analysis/score_head_set_ablation.py \
    --group mlp_joint_grid_layer_token \
    --run-glob 'joint_*_kl0.1_*' \
    --cells "$CELLS" \
    --head-defs "$HEAD_DEFS" \
    --skip-existing
