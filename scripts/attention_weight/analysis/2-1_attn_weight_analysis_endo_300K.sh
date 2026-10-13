#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_anal300K
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=8:00:00
# SGE writes into this DIRECTORY. It must already exist -- otherwise SGE treats the
# value as a filename.
#$ -o logs
#$ -j y
# Waits for 2-1's training. If that job has already finished there is nothing to hold on
# and this starts immediately.
#$ -hold_jid attn_w_endo_picked300K
# ============================================================================
# Stage 2 analysis for 2-1_train_attn_weight_endogenous_picked_cells_300K.sh.
#
#   qsub scripts/attention_weight/analysis/2-1_attn_weight_analysis_endo_300K.sh
#
# A thin wrapper: the pipeline itself lives in 1_attn_weight_analysis.sh, and this only
# pins EXP_GROUP=attn_weight_endo_picked_300K. It exists because SGE's qsub does NOT forward the submitting
# shell's environment, so `EXP_GROUP=... qsub 1_attn_weight_analysis.sh` silently ran
# on the default group (job 7919934 analysed the 60K runs when 300K was asked for).
# ============================================================================

export EXP_GROUP=attn_weight_endo_picked_300K
# -l: a login shell, so `module` and `conda` are defined for 1_'s own setup, exactly as
# when it is submitted directly under its #!/bin/bash -l shebang.
exec bash -l scripts/attention_weight/analysis/1_attn_weight_analysis.sh
