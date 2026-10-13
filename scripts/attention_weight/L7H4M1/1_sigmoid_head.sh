#!/bin/bash -l
#$ -P multilm
#$ -N L7H4M1_1_sigmoid
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=08:00:00
#$ -o logs
#$ -j y
#$ -t 1-5

# ============================================================================
# L7 H4 @-1, VARIANT 1: sigmoid regression head (logistic probe).
#
# The target is a probability, and the head so far was an unbounded linear map. This
# swaps it for sigmoid(W h + b) (SigmoidRegressionHead) and keeps everything else as
# Stage 3: extreme x4 injection in log-odds for training, var-normalized MSE on the
# probability, gaussian@1 as the headline interventional eval.
#
# What it can and cannot fix: Stage 3's W=0 run had the fresh ridge CEILING at ~0 on
# every scale, i.e. the injected value was not linearly present in the last hidden state
# at all. A different head cannot read what is not there, so a flat result here points
# at routing, not the readout. interventional_ceiling_empirical_logit.json records the
# ceiling in logit(t) as well, which is the space this head is linear in.
#
# empirical_logit is also evaluated (an extra file, nothing replaced) so variant 4 can be
# compared against 1 and 2 with the eval distribution held fixed.
#
# 5 array tasks = W in {0, 0.01, 0.03, 0.1, 0.3}; ~1 h each.
#
# Usage
#   qsub scripts/attention_weight/L7H4M1/1_sigmoid_head.sh
#   SMOKE=1 SGE_TASK_ID=1 bash scripts/attention_weight/L7H4M1/1_sigmoid_head.sh
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

VARIANT_TAG=sigmoid_scale4
HEAD_ACT=sigmoid
TRAIN_DIST=extreme
EVAL_DISTS=gaussian,empirical_logit
source scripts/attention_weight/L7H4M1/_common.sh
