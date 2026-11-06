#!/bin/bash -l
#$ -P llamagrp
#$ -N context_dep_sft_grid
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l h_rt=3:00:00
#$ -o log
#$ -j y

# ============================================================================
# Context dependence of the SFT grid's neurons: how well each neuron's activation is
# reproduced from BOS + only the last n tokens (n = 1..16), against the full 32-token
# window. One job covers all 8 layers -- one forward pass captures every layer.
# Base model only, no checkpoint. Also scores N_RANDOM random neurons per layer as the
# layer's reference distribution, and a token-identity lookup table.
#
# Texts: any cache of the grid's corpus (only text_input_ids are read); the layer-0
# POS-1 cell's cache is used here.
#
# Afterwards (CPU, seconds):
#   python plotters/plot_sft_vs_neuron_property.py --metric context              # n_eff
#   python plotters/plot_sft_vs_neuron_property.py --metric context --x r2_1
#   python plotters/plot_sft_vs_neuron_property.py --metric context --x r2_token_table
# ============================================================================

echo "Start date : $(date)  Job: $JOB_NAME $JOB_ID"
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env
export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

set -e

CONFIG=$(ls experiment_configs/mlp_endo_sweep_layer_token/endo_N0.*_POS-1_gateoff_w1.0_300K.json | head -1)
DATASET_DIR=$(python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from src.introspector.config import ExperimentConfig
print(ExperimentConfig.model_validate_json(Path('$CONFIG').read_text()).dataset_cache_dir)")
echo "texts from $DATASET_DIR"

python mechanistic_analysis/score_neuron_context_dependence.py \
    --dataset-dir "$DATASET_DIR" \
    --n-texts 3000 \
    --table-texts 30000 \
    --n-max 16 \
    --n-random 50
