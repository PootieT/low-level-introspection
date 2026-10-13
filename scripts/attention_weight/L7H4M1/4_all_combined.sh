#!/bin/bash -l
#$ -P multilm
#$ -N L7H4M1_4_all
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=08:00:00
#$ -o logs
#$ -j y
#$ -t 1-5

# ============================================================================
# L7 H4 @-1, VARIANT 4: variants 1 + 2 + 3 together.
#
#   head        sigmoid (variant 1)
#   training    prob_uniform injection (variant 2)
#   headline    empirical_logit eval (variant 3); gaussian@1 also written for Stage 3
#
# Usage
#   qsub scripts/attention_weight/L7H4M1/4_all_combined.sh
#   SMOKE=1 SGE_TASK_ID=1 bash scripts/attention_weight/L7H4M1/4_all_combined.sh
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

VARIANT_TAG=probunif_sigmoid
HEAD_ACT=sigmoid
TRAIN_DIST=prob_uniform
EVAL_DISTS=empirical_logit,gaussian
source scripts/attention_weight/L7H4M1/_common.sh
