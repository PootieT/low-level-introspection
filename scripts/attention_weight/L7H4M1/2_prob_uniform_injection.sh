#!/bin/bash -l
#$ -P multilm
#$ -N L7H4M1_2_probunif
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=08:00:00
#$ -o logs
#$ -j y
#$ -t 1-5

# ============================================================================
# L7 H4 @-1, VARIANT 2: training injection drawn UNIFORMLY IN PROBABILITY SPACE.
#
# Stage 3 trained on extreme x4 in log-odds: uniform over 4x the cell's logit range, then
# clamped to [1e-4, 1-1e-4] and pushed through a sigmoid. Most of that range lies past
# the clamp, so the targets piled up on ~0 and ~1 and the head could score by telling
# the two piles apart. Here t ~ U(0, 1), clamped only at the same 1e-4 guard band
# (prob_uniform in make_bounded_prob_intervention_fn; ~2e-4 of draws are clamped).
#
# CAVEAT to read the results with: U(0,1) puts 90% of draws above 0.1, while only ~35%
# of this cell's natural weights are. Training and eval distributions therefore differ
# here (both evals stay near the natural distribution).
#
# Linear head, headline eval gaussian@1 as in Stage 3; empirical_logit recorded too.
#
# Usage
#   qsub scripts/attention_weight/L7H4M1/2_prob_uniform_injection.sh
#   SMOKE=1 SGE_TASK_ID=1 bash scripts/attention_weight/L7H4M1/2_prob_uniform_injection.sh
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

VARIANT_TAG=probunif
HEAD_ACT=identity
TRAIN_DIST=prob_uniform
EVAL_DISTS=gaussian,empirical_logit
source scripts/attention_weight/L7H4M1/_common.sh
