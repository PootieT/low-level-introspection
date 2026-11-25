#!/bin/bash -l
#$ -P llamagrp
#$ -N lora_ablation_rebuttal
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=6:00:00
#$ -o log
#$ -j y
#$ -t 1-10

# ============================================================================
# LoRA ablation for the rebuttal (W7: "LoRA update norms support only a hypothesis").
# MAIN GRID's anchored arm (KL budget 0.1), 4 cells x 5 neurons = 20 runs, 2 per task:
#   layers {2, 8} x positions {-1 (t=31), -28 (t=4)}
#
# Per run: zero one LoRA module at a time (16 layers x 7 modules) AND each layer's
# whole attention / MLP LoRA (16 x 2), scored on the WHOLE test split (no success
# filter) at the paper's interventional regime (x1 gaussian) and endogenously.
# Writes <run>/lora_ablation_x1/ so older lora_ablation/ results are never reused.
#
# Optional arg: $1 = cells (default below).
#
# Tables afterwards (CPU, seconds; needs outputs/plots/<group>/lora_weights/lora_profiles.csv):
#   python plotters/table_lora_ablation.py --group mlp_joint_grid_layer_token
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

python mechanistic_analysis/ablate_lora_modules.py \
    --group mlp_joint_grid_layer_token \
    --run-glob 'joint_*_kl0.1_*' \
    --cells "$CELLS" \
    --no-success-filter --n-examples 100000 \
    --interv-distribution gaussian --interv-scale 1 \
    --granularity both \
    --analysis-name lora_ablation_x1 \
    --skip-existing
