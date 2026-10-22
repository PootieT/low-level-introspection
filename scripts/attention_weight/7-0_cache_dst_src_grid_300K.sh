#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_dstsrc_cache300K
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=08:00:00
#$ -o logs
#$ -j y
#$ -t 1-9

# ============================================================================
# STAGE 7-0: build the nine per-DESTINATION 300K / 30K datasets that 7-1 trains from.
#
# One task per destination (absolute offsets 0 -1 -3 -7 -11 -15 -19 -23 -27; -31 has only
# the dropped (-31,-31) pair). Each dataset holds the three pinned heads at every source
# offset that destination can reach -- up to 30 signals, still one forward per text.
# 7-1 runs six to ten pairs per destination concurrently, and data_prep/cache_signals.py skips on
# `train/` existing alone, so if 7-1 built caches itself the losers of the race could read
# a half-written dataset with no error. It refuses to, and holds on this job instead.
#
# WHY THIS CALLS 7-1 INSTEAD OF REPEATING ITS ARGUMENTS. The cache directory is a hash of
# every content field of the generated config (dataset_utils.dataset_identity). Running
# 7-1 with CACHE_ONLY=1 means the config that builds the cache is produced by the exact
# generate_experiment_config call 7-1 trains from; a copied argument list could drift and
# silently build a dataset nobody reads. Under CACHE_ONLY=1, 7-1 reads SGE_TASK_ID as a
# destination index (1-9), not a pair index.
#
# COST. cache_signals writes one ROW per (text, signal), so both time and disk scale with
# the signal count even though the forward does not. 2-0's 8-signal cells took ~20 min
# and 3.5-7.7 GB each; destination 0 here carries 30 signals (~75 min, up to ~30 GB) and
# destination -27 only 6. 162 signals over the nine tasks is roughly 80-160 GB in total.
#
# Usage
#   qsub scripts/attention_weight/7-0_cache_dst_src_grid_300K.sh
#   qsub scripts/attention_weight/7-1_train_dst_src_grid_endogenous_300K.sh   # holds on this job
# ============================================================================

set -e

# `bash -l`: 7-1 calls `module` and `conda activate`, which need the login-shell setup
# that qsub gives a `#!/bin/bash -l` script but a plain `bash` subprocess would not.
CACHE_ONLY=1 bash -l scripts/attention_weight/7-1_train_dst_src_grid_endogenous_300K.sh
