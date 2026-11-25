#!/bin/bash -l
#$ -P llamagrp
#$ -N write_channel_ablation_joint_grid
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=2:00:00
#$ -o log
#$ -j y
#$ -t 1-4

# ============================================================================
# Write-channel ablation for the rebuttal (W7: the "private channel" of Sec. 6.1).
# MAIN GRID's anchored arm (KL budget 0.1), 4 cells x 5 neurons = 20 runs, 5 per task:
#   layers {2, 8} x positions {-1 (t=31), -28 (t=4)}
#
# Per run, at the signal layer and token only: remove the target neuron's write along
# its base W_down column vs along the LoRA-added part, undo its W_up edit, write exactly
# what the base model would, or remove the whole write. Whole test split, endogenous and
# x1 gaussian; reports corr(pred, a), corr(pred, v) and corr(pred, a) under injection.
#
# Optional arg: $1 = cells (default below).
#
# Tables afterwards (CPU, seconds):
#   python plotters/table_write_channel_ablation.py --group mlp_joint_grid_layer_token
# ============================================================================

CELLS=${1:-2:-1,2:-28,8:-1,8:-28}

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID / $SGE_TASK_LAST"
echo "Cells : $CELLS"
echo "=========================================================="
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

set -e

python mechanistic_analysis/score_write_channel_ablation.py \
    --group mlp_joint_grid_layer_token \
    --run-glob 'joint_*_kl0.1_*' \
    --cells "$CELLS" \
    --skip-existing
