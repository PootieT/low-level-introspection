#!/bin/bash -l
#$ -P llamagrp
#$ -N signal_subspace_neurongen
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=6:00:00
#$ -o log
#$ -j y
#$ -t 1-16

# ============================================================================
# Signal-subspace analysis over the neuron-generalization runs: per run, the write-in
# and transport directions of every trained neuron (up to 30) plus 10 held-out pool
# neurons, cross-talk by measurement layer, the held-out write-in vs transformation
# split, and the subspace-patch ablation. See mechanistic_analysis/trace_signal_subspace.py.
#
# Runs are sharded over the array by mechanistic_analysis/trace_signal_subspace.py itself
# ($SGE_TASK_ID / $SGE_TASK_LAST); --skip-existing makes resubmission resume.
#
# The run glob is the FIRST SCRIPT ARGUMENT (default: every neurongen run, unconstrained
# and KL=0.1). SGE passes script arguments to the job but not the caller's environment,
# so `RUN_GLOB=x qsub <script>` would silently analyse every run. E.g.
#   qsub <script> 'neurongen_kl0.1_*'
#   qsub <script> 'neurongen_armA_*'           # unconstrained scaling cells
# Cost: quadratic in neurons analysed (<=40 here): ~minutes per run on one GPU, fp32.
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

RUN_GLOB=${1:-${RUN_GLOB:-neurongen_*}}
echo "run glob: $RUN_GLOB"

python mechanistic_analysis/trace_signal_subspace.py \
    --group mlp_neurongen \
    --run-glob "$RUN_GLOB" \
    --n-texts 64 --max-trained 30 --n-heldout 10 \
    --skip-existing
