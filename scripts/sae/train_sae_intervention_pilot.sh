#!/bin/bash -l
#$ -P llamagrp
#$ -N sae_interv_pilot
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=18:00:00
#$ -o log
#$ -j y
#$ -t 1-9    # 3 features x 3 positions

# ============================================================================
# PILOT interventional SAE-detection training: does steering on/off DURING training
# move paired_directional_acc toward 1 (the feature is actually read), where
# endogenous-only training leaves it at ~0.5 (bypass)? Small grid, run before
# committing to the full sweep.
#
# The label is the STEERED state, drawn 50/50 and independent of the text, so context
# cannot predict it -- that is what removes the bypass. Distribution 'extreme' at
# scale 1 is, for the SAE sampler, uniform over the feature's EMPIRICAL active range
# [amin, amax]; raise --intervention-extreme-scale (10, 100) for larger-than-natural
# injections, or add --intervention-anneal (+ --intervention-scale-start/-end) to
# start large and anneal to natural, as the neuron sweeps do.
#
# GATE stays ON here: the question is whether the model can READ the steered feature
# through attention without any trainable parameter touching the input text.
# ============================================================================

echo "Start: $(date)  task $SGE_TASK_ID"
nvidia-smi
module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env
export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection

set -e

LR=2e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
SCALE=1                  # 1 => uniform over the empirical active range
EXP_GROUP=sae_intervention_pilot
export WANDB_RUN_GROUP=$EXP_GROUP

# a known-endogenously-detectable feature + two others; edit indices as you like
SIGS=(S9.12408 S9.5999 S9.10004)
# include early positions so we test the long-transport regime, not just pos -1
POSS=(-1 -13 -25)

index=$(($SGE_TASK_ID-1))
SIG=${SIGS[$((index / 3))]}
TOKEN_POS=${POSS[$((index % 3))]}

EXP_NAME="saeInterv_${SIG}_POS${TOKEN_POS}_scale${SCALE}_300K"
CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
echo "=== $EXP_NAME  (sig=$SIG pos=$TOKEN_POS) ==="

python data_prep/generate_experiment_config.py \
    --experiment-name "$EXP_NAME" \
    --model google/gemma-2-2b \
    --objective detection \
    --intervention-signal-distribution extreme \
    --intervention-extreme-scale $SCALE \
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
    --train-signals "$SIG" \
    --test-signals "$SIG" \
    --train-token-selector $TOKEN_POS \
    --test-token-selector $TOKEN_POS \
    --precision 3 \
    --train-batch-size $BS \
    --eval-batch-size 128 \
    --num-epochs 1 \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DO \
    --disable-adapter-in-input \
    --bf16 \
    --learning-rate $LR \
    --warmup-ratio 0.03 \
    --lr-scheduler-type cosine \
    --save-steps 0 \
    --cache-dir data \
    --output-dir outputs/runs/$EXP_GROUP \
    --output "$CONFIG"

python data_prep/cache_signals.py --config "$CONFIG"

# --force-drift-eval FAILS FAST here: steering is applied during training, so a
# misaligned position would invalidate the whole run rather than just its diagnostics
python trainers/train_sae_detector_intervention.py \
    --config "$CONFIG" \
    --force-drift-eval
