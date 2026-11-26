#!/bin/bash -l
#$ -P llamagrp
#$ -N attn_lora_positions_joint_grid
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=3:00:00
#$ -o log
#$ -j y
#$ -t 1-4

# ============================================================================
# Token-resolved signal-layer attention LoRA ablation for the rebuttal (W7).
# MAIN GRID's anchored arm (KL budget 0.1), 4 cells x 5 neurons = 20 runs, 5 per task:
#   layers {2, 8} x positions {-1 (t=31), -28 (t=4)}
#
# Zeroing layer L's attention LoRA is the largest single-block drop in
# ablate_lora_modules.py, though that attention runs before the signal MLP. Per run, the
# LoRA-off attention output of layer L is spliced in at one token group at a time
# (t / text before t / text after t / prompt / readout / all / all but t), and the
# target neuron's gain u at t is logged (does the LoRA at t keep the write h = a*u
# tracking a?). Whole test split, endogenous and x1 gaussian.
#
# Optional arg: $1 = cells (default below).
#
# Tables afterwards (CPU, seconds):
#   python plotters/table_attn_lora_positions.py --group mlp_joint_grid_layer_token
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

python mechanistic_analysis/score_attention_lora_positions.py \
    --group mlp_joint_grid_layer_token \
    --run-glob 'joint_*_kl0.1_*' \
    --cells "$CELLS" \
    --skip-existing
