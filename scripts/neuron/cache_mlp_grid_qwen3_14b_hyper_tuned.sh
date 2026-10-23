#!/bin/bash -l
#$ -P llamagrp
#$ -N cache_qwen3_14b_ht
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=80G
#$ -l h_rt=12:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-16

# ============================================================================
# CACHE PASS for the Qwen3-14B *_hyper_tuned grids: one task per cell (4 layers x 4
# positions), building the cell's activation dataset AND its OOD caches. Nothing trains.
#
# Both 14B grids (joint and SFT) run one neuron per task, so five tasks of a cell start
# together; they would race to build one cache directory (and the shuffled-OOD corpus,
# written to one shared path). They are held on this job instead and refuse to start
# without a complete cache. Submit commands: the joint grid's header.
#
# The configs here carry the grid's data arguments verbatim, so the caches land at the
# keys the grids read (dataset identity excludes experiment name, output dir and every
# training hyperparameter). They write to their own output dir, outside the analysed
# groups. L5 pos-1, L15 pos-8 and L35 pos-28 are already cached by the LR sweep, and
# every step skips work that is already done, so resubmitting is cheap.
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection

set -e

index=$(($SGE_TASK_ID-1))
LAYERS=(5 15 25 35)             # must match both 14B grids
POSS=(-1 -8 -20 -28)
LAYER=${LAYERS[$((index / 4))]}
TOKEN_POS=${POSS[$((index % 4))]}
MODEL=Qwen/Qwen3-14B-Base
SEED=21
N_NEURONS_CACHE=10
CFG_DIR=experiment_configs/_cache_qwen3_14b_hyper_tuned
OUT_DIR=outputs/runs/_cache_qwen3_14b_hyper_tuned
CELL_TAG="l${LAYER}_pos${TOKEN_POS}"
SAMPLER="{\"type\":\"uniform_layer_sampler\",\"signal_type\":\"mlp\",\"layers\":[${LAYER}],\"signals_per_layer\":${N_NEURONS_CACHE},\"seed\":${SEED}}"
echo "cache pass: cell=$CELL_TAG"

python data_prep/generate_experiment_config.py \
    --experiment-name "_cell_${CELL_TAG}" \
    --model $MODEL \
    --train-dataset allenai/c4 --train-dataset-split "train" --train-max-samples 300000 \
    --dev-max-samples 200 --dev-offset-from-train 1000000 \
    --test-dataset allenai/c4 --test-dataset-split "validation" --test-max-samples 3000 \
    --max-seq-length 32 \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "$SAMPLER" \
    --test-signals "$SAMPLER" \
    --same-train-test-signals \
    --train-token-selector $TOKEN_POS \
    --test-token-selector $TOKEN_POS \
    --precision 3 \
    --cache-dir data --output-dir $OUT_DIR \
    --output "$CFG_DIR/_cell_${CELL_TAG}.json" \
    --overwrite-cache

NEURONS=($(python -c "
import json
cfg = json.load(open('$CFG_DIR/_cell_${CELL_TAG}.json'))
print(' '.join(f\"N{s['layer']}.{s['index']}\" for s in cfg['train_signals']))
"))
NEURON_LIST=$(IFS=,; echo "${NEURONS[*]}")
CONFIG=$CFG_DIR/_cache_${CELL_TAG}.json

# the grids' run arguments, verbatim (the training ones do not enter the cache key)
python data_prep/generate_experiment_config.py \
    --experiment-name "_cache_${CELL_TAG}" \
    --model $MODEL \
    --train-dataset allenai/c4 \
    --train-dataset-split "train" \
    --train-max-samples 300000 \
    --dev-max-samples 200 \
    --dev-offset-from-train 1000000 \
    --test-dataset allenai/c4 \
    --test-dataset-split "validation" \
    --test-max-samples 3000 \
    --max-seq-length 32 \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "$NEURON_LIST" \
    --test-signals "$NEURON_LIST" \
    --signal-filter "${NEURONS[0]}" \
    --train-token-selector $TOKEN_POS \
    --test-token-selector $TOKEN_POS \
    --precision 3 \
    --train-batch-size 8 \
    --gradient-accumulation-steps 4 \
    --eval-batch-size 128 \
    --num-epochs 1 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.0 \
    --bf16 \
    --learning-rate 1e-4 \
    --save-steps 0 \
    --use-regressor \
    --intervention-signal-distribution extreme \
    --intervention-extreme-scale 10 \
    --cache-dir data \
    --output-dir $OUT_DIR \
    --output "$CONFIG"

# the cell's activation dataset (skips if complete)
python data_prep/cache_signals.py --config "$CONFIG"
# its OOD caches, the ones --ood-eval reads in both grids; --prepare-only trains nothing
python trainers/train_introspector_joint.py --config "$CONFIG" --ood-eval --prepare-only
echo "cache pass: cell $CELL_TAG ready"
