#!/bin/bash -l
#$ -P llamagrp
#$ -N single_sae_sweep_layer_pos
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=18:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 400-800

## SCC env setup
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

## environment setup
export WANDB_PROJECT=low-level-introspection
export WANDB_WATCH=all
export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

# SAE detection sweep over (layer x token_position x feature), ONE run per array
# task (8 layers x 10 positions x 10 features = 800). train_sae_detector.py runs
# BOTH the endogenous and the interventional (steer on/off) eval, so every run
# writes sae_detection_metrics.json with a "test_interventional" block.
#
# PREREQUISITE: run scripts/precompute_sae_feature_stats.sh first so the per-layer
# density parquets exist; otherwise the feature picker falls back to uniform-random
# (which may hit dead/frequent features).

LR=2e-5   # best lr for f1@0 in the single-feature lr sweep
EXP_GROUP="single_sae_detection_sweep_layer_pos"
export WANDB_RUN_GROUP=$EXP_GROUP

# ---- decode (layer, position, sample) from the flat array index ----
# layout: layer outer (100 each) -> position (10 each) -> sample (10)
index=$(($SGE_TASK_ID-1))
LAYERS=(0 2 4 6 8 10 12) # 14 was originally scheduled but takes too long so dropped
POSS=(-1 -4 -7 -10 -13 -16 -19 -22 -25 -28)
LAYER=${LAYERS[$((index / 100))]}
rem=$((index % 100))
TOKEN_POS=${POSS[$((rem / 10))]}
SAMPLE=$((rem % 10))

# Pick a density-STRATIFIED SAE feature. Seed by LAYER (not sample) so the SAME 10
# features are used across ALL positions => the position axis is controlled. We
# split the density-valid pool into 10 bins by density and take one feature per
# bin, so the 10 span rare->common instead of clustering on the frequent
# (punctuation-like) end. `sample` selects which stratum.
IDX=$(LAYER=$LAYER SAMPLE=$SAMPLE python -c '
import os, sys
import numpy as np, pandas as pd
layer=int(os.environ["LAYER"]); sample=int(os.environ["SAMPLE"]); n=10
p=f"data/sae_stats/gemma2-2b_res{layer}_16k.parquet"
rng=np.random.RandomState(layer)
if os.path.exists(p):
    df=pd.read_parquet(p)
    feats=df[df["selected"]].sort_values("density")["feature"].to_numpy()
    bins=[b for b in np.array_split(feats, n) if len(b)]      # rare -> common strata
    chosen=[int(rng.choice(b)) for b in bins]
    if len(chosen) < n:                                        # top up if few strata
        extra=[int(f) for f in feats if int(f) not in set(chosen)]
        rng.shuffle(extra); chosen += extra[:n-len(chosen)]
else:
    print(f"[warn] no stats parquet {p}; uniform-random feature (may be dead/frequent)", file=sys.stderr)
    chosen=[int(x) for x in rng.choice(np.arange(16384), size=n, replace=False)]
print(chosen[sample])
')

SIG="S${LAYER}.${IDX}"
EXP_NAME="sae_sample${SAMPLE}_${SIG}_POS${TOKEN_POS}_lr${LR}_train300K"
echo "Running Experiment: ${EXP_NAME}  (layer=$LAYER pos=$TOKEN_POS sample=$SAMPLE)"

### generate experiment config first as JSON
python generate_experiment_config.py \
    --experiment-name $EXP_NAME \
    --model google/gemma-2-2b \
    --objective detection \
    --detection-balance-pos-frac 0.5 \
    --train-dataset allenai/c4 \
    --train-dataset-split "train" \
    --train-max-samples 300000 \
    --dev-max-samples 2000 \
    --dev-offset-from-train 1000000 \
    --test-dataset allenai/c4 \
    --test-dataset-split "validation" \
    --test-max-samples 30000 \
    --max-seq-length 32 \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals $SIG \
    --test-signals $SIG \
    --precision 3 \
    --train-token-selector $TOKEN_POS \
    --test-token-selector $TOKEN_POS \
    --train-batch-size 32 \
    --eval-batch-size 128 \
    --learning-rate $LR \
    --num-epochs 1 \
    --disable-adapter-in-input \
    --bf16 \
    --cache-dir data/$EXP_GROUP \
    --output-dir outputs/runs/$EXP_GROUP \
    --output experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache

## cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache

## train detector (endogenous + interventional eval)
python train_sae_detector.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json

## delete model checkpoints to save space
python utils/delete_checkpoint_after_experiment_completion.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
    --expected-files "sae_interventional_scores.png"
