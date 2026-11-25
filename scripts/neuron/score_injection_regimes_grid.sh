#!/bin/bash -l
#$ -P llamagrp
#$ -N injection_regimes_grid
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=3:00:00
#$ -o log
#$ -j y
#$ -t 1-4

# ============================================================================
# Injection-regime sweep (rebuttal: is the joint model's interventional advantage
# specific to the training intervention?). SFT and joint KL=0.1 runs scored the same
# way, on the same test rows and prompts, at 4 cells x the joint grid's 5 neurons:
#   layers {2, 8} x positions {-1 (t=31), -28 (t=4)}  -> 20 joint + 20 SFT runs
#
# Per run (evals/score_injection_regimes.py, full 3K test split):
#   endogenous, gaussian replacement at x1/x3/x10/x30/x100 the natural std, and
#   local@1 / local@0.25: each example's LIVE value + delta, scored within example
#   as corr(pred - pred_clean, delta). Training used uniform over 10x the natural
#   range, so x1-x10 stay inside it and x30/x100 partly leave it.
#
# Optional args: $1 = cells, $2 = arms (see the defaults below).
# Each python call shards its own group's runs over $SGE_TASK_ID / $SGE_TASK_LAST, so
# every task does ~5 joint and ~5 SFT runs. --skip-existing makes resubmission cheap.
#
# Aggregate afterwards (CPU, seconds):
#   python plotters/plot_injection_regimes.py
# ============================================================================

CELLS=${1:-2:-1,2:-28,8:-1,8:-28}
ARMS=${2:-x1,x3,x10,x30,x100,local@1,local@0.25}
# the joint grid's neurons in these cells; the SFT sweep has 10 per cell
NEURONS=2.1447,2.1886,2.2566,2.5949,2.6998,8.3852,8.4144,8.4956,8.5777,8.6370

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID / $SGE_TASK_LAST"
echo "Cells : $CELLS   Arms : $ARMS"
echo "Neurons : $NEURONS"
echo "=========================================================="
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

set -e

# joint loss, KL budget 0.1 (the paper's main arm)
python evals/score_injection_regimes.py \
    --group mlp_joint_grid_layer_token \
    --run-glob 'joint_*_gateoff_scale10_kl0.1_w0.01_300K' \
    --cells "$CELLS" \
    --neurons "$NEURONS" \
    --arms "$ARMS" \
    --skip-existing

# SFT, gate off (the paper's SFT comparison arm), same neurons
python evals/score_injection_regimes.py \
    --group mlp_endo_sweep_layer_token \
    --run-glob 'endo_*_gateoff_w1.0_300K' \
    --cells "$CELLS" \
    --neurons "$NEURONS" \
    --arms "$ARMS" \
    --skip-existing
