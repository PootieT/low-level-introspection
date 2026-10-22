#!/bin/bash -l
#$ -P llamagrp
#$ -N prompt_baseline_layer_token
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=24G
#$ -l h_rt=6:00:00
#$ -o log
#$ -j y
#$ -t 1-64

# ============================================================================
# PROMPTED-BASELINE over the MAIN GRID: the UNTRAINED Llama-3.2-1B, asked with the
# grid's own introspection prompts, read out by closed-form ridge probes on its
# final-layer hidden state (evals/score_prompt_baseline.py). Nothing is trained, so the
# result depends only on the (neuron, position) -- both KL regimes of the grid map to
# the same output dir, and only the kl0.1 configs are enumerated.
#
# 8 layers x 8 positions x 5 neurons = 320 runs over 64 array tasks (~5 each), sharded
# by the script from $SGE_TASK_ID / $SGE_TASK_LAST over the runs sorted by name; the
# model is loaded once per task. --skip-existing makes resubmission after timeouts
# only redo unfinished runs.
#
# Per run: 30k probe-training rows (10% held out to pick the ridge penalty), the
# grid's 3k test rows, greedy continuation of k=16 tokens pooled at k in
# {0,1,2,4,8,16}, endo + joint probes, the verbal (written-number) readout, the
# no-prompt control, the alignment check, the k=0 scale sweep and the OOD ladder.
#
# MODE=lc is the learning-curve pilot: k=0 only, probes refit on up to 270k rows at
# the three cells whose trained runs are synced locally. It answers "is the 30k probe
# data-limited?" before the grid's numbers are read. Submit it as
#     qsub -v MODE=lc -t 1-3 scripts/neuron/score_prompt_baseline_layer_token.sh
#
# Aggregate afterwards (CPU, seconds):
#   python plotters/plot_prompt_baseline.py
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID / $SGE_TASK_LAST"
echo "=========================================================="
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

set -e

MODE=${MODE:-grid}
#MODE=lc
GROUP=mlp_joint_grid_layer_token

if [ "$MODE" = "lc" ]; then
  LC_CONFIGS=(
    joint_N2.1447_POS-1_gateoff_scale10_kl0.1_w0.01_300K
    joint_N0.1377_POS-12_gateoff_scale10_kl0.1_w0.01_300K
    joint_N12.2721_POS-28_gateoff_scale10_kl0.1_w0.01_300K
  )
  CFG=${LC_CONFIGS[$((SGE_TASK_ID - 1))]}
  # --task-index/--n-tasks 1: one explicit config, so no sharding of a 1-run list
  python evals/score_prompt_baseline.py \
      --config experiment_configs/$GROUP/$CFG.json \
      --baseline-root outputs/runs/mlp_prompt_baseline_learning_curve \
      --max-train-examples 300000 \
      --k-values 0 \
      --learning-curve 1000,3000,10000,30000,100000,270000 \
      --scale-sweep "" \
      --task-index 1 --n-tasks 1 \
      --skip-existing
else
  python evals/score_prompt_baseline.py \
      --group $GROUP \
      --run-glob 'joint_*_kl0.1_*' \
      --baseline-root outputs/runs/mlp_prompt_baseline_layer_token \
      --max-train-examples 30000 \
      --k-values 0,1,2,4,8,16 \
      --learning-curve 1000,3000,10000,27000 \
      --no-prompt-control \
      --ood-eval \
      --skip-existing
fi
