#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_scan
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=2:00:00
#$ -o logs
#$ -j y

# ============================================================================
# STAGE 1: measure the post-softmax attention weight over the FULL
# 16 layers x 32 heads x 10 source offsets grid, so the training cells can be picked
# from measurements rather than sampled.
#
# Run this FIRST. Everything downstream is defined by the cells it picks, and the
# attention-logit sweep is the argument for doing it: 1280 randomly-chosen cells produced
# a flat result (mean corr 0.894, no variation by layer or offset).
#
# One forward per batch covers the whole grid (AttentionRowTap reads a destination row
# rather than a cell), so 4000 documents is minutes, not hours. Nothing is written to
# data/ -- this only reads.
#
# The printed picks are a starting point; the cells actually trained are hand-picked
# from the CSVs as a {lower, mid, upper layer} x {near, far source} design and pasted
# into CELLS= in the Stage 2/3/4 scripts (they must match across all three).
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID"
echo "=========================================================="
nvidia-smi

module load cuda/12.8
module load miniconda/23.11.0
conda activate /projectnb/multilm/jongin/anaconda3/envs/introspection

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

set -e

N_DOCS=${N_DOCS:-4000}
BATCH_SIZE=${BATCH_SIZE:-64}
OUT=${OUT:-plots/attn_weight_scan}

python mechanistic_analysis/scan_attention_weight_spread.py \
    --n-docs $N_DOCS \
    --batch-size $BATCH_SIZE \
    --out "$OUT"

echo
echo "=========================================================="
echo "SCAN DONE -> $OUT"
echo "Paste the recommended cells into CELLS= in:"
echo "  scripts/attention_weight/2_train_attn_weight_endogenous_picked_cells.sh"
echo "  scripts/attention_weight/3_train_attn_weight_joint_weight_sweep.sh"
echo "  scripts/attention_weight/4_train_attn_weight_interventional_picked_cells.sh"
echo "Then: qsub scripts/attention_weight/pilot/pilot_attn_weight_one_cell.sh"
echo "End date : $(date)"
echo "=========================================================="
