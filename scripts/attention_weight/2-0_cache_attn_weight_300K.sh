#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_cache300K
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=08:00:00
#$ -o logs
#$ -j y
#$ -t 1-6

# ============================================================================
# STAGE 2-0: build the six 300K / 30K cell datasets that 2-1, 3-1 and 4-1 all reuse.
#
# One task per cell, so no two tasks ever write the same cache directory. The training
# scripts split each cell over several concurrent tasks (2-1: 3 arms, 3-1: 6 weights,
# 4-1: 2 regimes), and cache_signals.py skips on `train/` existing alone -- if they built
# the cache themselves, the losers of the race could read a half-written dataset with no
# error. So they refuse to build it and wait for this job via `#$ -hold_jid attn_w_cache300K`.
#
# WHY THIS CALLS 2-1 INSTEAD OF REPEATING ITS ARGUMENTS. The cache directory is a hash of
# every content field of the generated config (dataset_utils.dataset_identity). Running
# 2-1 itself with CACHE_ONLY=1 means the config that builds the cache is produced by the
# exact generate_experiment_config call 2-1 trains from; a copied argument list could
# drift and silently build a dataset nobody reads. 2-1's tasks 1-6 decode to arm 0 x the
# six cells, which is one task per cell.
#
# COST. 60K took ~20 min of caching per cell; 300K train + 30K test is ~5.5x that, plus
# the streaming skip of DEV_OFFSET=1M raw documents for dev. h_rt=8h leaves headroom.
# Disk: ~1.6 GB per cell at 60K, so ~9 GB per cell, ~55 GB in total.
#
# Usage
#   qsub scripts/attention_weight/2-0_cache_attn_weight_300K.sh
#   qsub scripts/attention_weight/2-1_train_attn_weight_endogenous_picked_cells_300K.sh  # holds on this job
#   qsub scripts/attention_weight/3-1_train_attn_weight_joint_weight_sweep_300K.sh       # holds on this job
# ============================================================================

set -e

# `bash -l`: 2-1 calls `module` and `conda activate`, which need the login-shell setup
# that qsub gives a `#!/bin/bash -l` script but a plain `bash` subprocess would not.
CACHE_ONLY=1 bash -l scripts/attention_weight/2-1_train_attn_weight_endogenous_picked_cells_300K.sh
