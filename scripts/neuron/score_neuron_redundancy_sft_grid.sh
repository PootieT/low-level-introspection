#!/bin/bash -l
#$ -P llamagrp
#$ -N redundancy_sft_grid
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=2:00:00
#$ -o log
#$ -j y
#$ -t 1-8

# ============================================================================
# Same-layer redundancy (leave-one-out R^2 of each neuron from every other neuron of
# its layer) for the 8 layers of the SFT layer x token grid, one array task per layer.
# Base model only -- no checkpoint is read -- so this is cheap: one forward pass over
# N_TEXTS texts of 32 tokens, plus an 8192 x 8192 covariance inversion.
#
# Every neuron of the layer is scored; the plotter then picks out the grid's own.
# Texts come from the layer's POS-1 SFT config's cache (the cache's activations are
# not used, only its texts). All token positions except BOS go into the covariance.
#
# Afterwards (CPU, seconds):
#   python plotters/plot_sft_vs_redundancy.py
# ============================================================================

echo "Start date : $(date)  Job: $JOB_NAME $JOB_ID task $SGE_TASK_ID"
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env
export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

set -e

LAYERS=(0 2 4 6 8 10 12 14)
L=${LAYERS[$((SGE_TASK_ID - 1))]}
N_TEXTS=20000            # x ~31 non-BOS tokens = ~620k activations for 8192 neurons
OUT=outputs/plots/mlp_endo_sweep_layer_token/redundancy/L${L}

CONFIG=$(ls experiment_configs/mlp_endo_sweep_layer_token/endo_N${L}.*_POS-1_gateoff_w1.0_300K.json | head -1)
echo "layer $L <- texts from $CONFIG"

python mechanistic_analysis/score_neuron_redundancy.py \
    --config "$CONFIG" \
    --layer "$L" \
    --positions all \
    --n-texts $N_TEXTS \
    --output-dir "$OUT"
