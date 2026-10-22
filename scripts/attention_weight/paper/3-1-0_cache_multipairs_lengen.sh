#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_paper31_cache
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=12:00:00
#$ -o logs
#$ -j y
#$ -t 1-5

#! SGE느 array job의 각 task에 있는 `SGE_TASK_ID`라는 환경변수로 1,2,..., 5를 넣어줍니다. 이 환경 변수는 3-1-0이 실행한 자식 프로세스인 3-1 스크립트에도 그대로 전달됩니다.
# ============================================================================
# PAPER EXPERIMENT 3-1, step 0: every cache the 12 runs of 3-1 read.
#
#   tasks 1-4  one training length n each (16 32 64 128): the base cache (train 300K /
#              dev / test = the n grid), then every length cache (m > n) and OOD cache
#              those runs evaluate on (trainers/train_introspector_joint.py --prepare-only)
#   task  5    the injection pool (data_prep/cache_signals.py --pair-pool), shared by every n
#
# The three heads of an n share all of these, so each is built by exactly one task here
# and the training tasks only read them (see 3-1's header for the race this avoids).
#
# WHY THIS CALLS 3-1 INSTEAD OF REPEATING ITS ARGUMENTS: the cache directories are
# hashes of the generated config, so the config that builds a cache must come from the
# exact generate_experiment_config call that 3-1 trains from (7-0's reasoning).
#
# SMOKE=1 (pairs with 3-1's SMOKE=1; n in {16, 64}, so tasks 1-2 are the lengths and
# task 3 the pool):
#   sed '/^#\$ -t /d' scripts/attention_weight/paper/3-1-0_cache_multipairs_lengen.sh \
#     | qsub -v SMOKE=1 -t 1-3
#
# Usage
#   qsub scripts/attention_weight/paper/3-1-0_cache_multipairs_lengen.sh
#   qsub scripts/attention_weight/paper/3-1_multipairs_lengen_endogenous_logistic_norm_dev.sh   # holds on this job
# ============================================================================

set -e

# `bash -l`: 3-1 calls `module` and `conda activate`, which need the login-shell setup.
CACHE_ONLY=1 bash -l scripts/attention_weight/paper/3-1_multipairs_lengen_endogenous_logistic_norm_dev.sh
