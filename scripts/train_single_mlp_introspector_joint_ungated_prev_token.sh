#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_intervention_train_joint_ungated_prev_token
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=6:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-6

# ============================================================================
# CONTROL for the earlier-token generalization failure: LoRA INPUT GATE OFF.
#
# In every prev-token run so far the adapter was gated to the introspection
# prompt (collator mask = 0 over the input text), so NO trainable parameter ever
# touched the computation at the introspected token. The injected value at
# layers.L.mlp.act_fn was therefore routed downstream at FIXED base-model gain:
# through down_proj[:, n] (scaled by up_proj(x)[n]), the layers above it at that
# position, and the base K/V that the prompt tokens attend to. At pos=-1 the
# prompt is adjacent so a small gain suffices; at earlier positions the signal
# has more layers + more attention dilution to survive, and nothing trainable
# could boost it. This run removes exactly that constraint at LoRA cost, before
# committing to full-weight finetuning.
#
# Caveat (shared with full FT, but milder): with the adapter active over the
# text, the model's own activation drifts away from the CACHED endogenous
# target. The INTERVENTIONAL metric is unaffected -- the injected value is
# ground truth by construction -- so read interventional corr (and the routing
# probe by position) as the headline, endogenous corr as secondary.
#
# Compare against: intervention_train_regressor_joint_prev_token_hyper (same
# data, same hyperparameters, gate ON). The cached dataset is REUSED from that
# group, so no re-caching of the 1M-example dataset happens here.
# ============================================================================

## SCC env setup
## Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

module load cuda/11.8   # 11.2
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env
#conda activate introspection

## environment setup
export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection
#export WANDB_WATCH=all
#export CUDA_LAUNCH_BLOCKING=1  # for debugging cuda errors
export WANDB_RUN_GROUP=single_mlp_intervention_train_reg_joint_ungated_prev_token

## Batch job variables
index=$(($SGE_TASK_ID-1))

W=0.01

LORA_R=128
LORA_ALPHA=$((LORA_R * 2))

SCALE=30
SIG=10.24
TOKEN_POS='-1:-31'
# same LR grid as the gated prev-token hyper sweep, so gated-vs-ungated is a
# best-vs-best comparison rather than a comparison at one arbitrary LR
LRS=(1e-5 2e-5 3e-5 5e-5 1e-4 2e-4)
LR=${LRS[$index]}
BS=128

### generate experiment config first as JSON
EXP_NAME="intervention_train_reg_joint_ungated_POS${TOKEN_POS}_scale${SCALE}_r${LORA_R}_lr${LR}_w${W}_1M"
DATASET_NAME="c4_N${SIG}_pos${TOKEN_POS}_seq32_1M"
EXP_GROUP=intervention_train_regressor_joint_ungated_prev_token
# NOTE: cache dir deliberately points at the GATED sweep's group so the already
# cached 1M-example dataset is reused as-is (cache_signals.py no-ops when
# <cache-dir>/<dataset-name>/train exists). Only the outputs are new.
CACHE_GROUP=intervention_train_regressor_joint_prev_token_hyper
echo $EXP_NAME
python generate_experiment_config.py \
    --experiment-name $EXP_NAME \
    --dataset-name $DATASET_NAME \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 \
    --train-dataset-split "train" \
    --train-max-samples 1000000 \
    --dev-max-samples 200 \
    --dev-offset-from-train 1000000 \
    --test-dataset allenai/c4 \
    --test-dataset-split "validation" \
    --test-max-samples 3000 \
    --max-seq-length 32 \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "N${SIG}" \
    --test-signals "N${SIG}" \
    --train-token-selector="${TOKEN_POS}" \
    --test-token-selector="${TOKEN_POS}" \
    --precision 3 \
    --train-batch-size $BS \
    --eval-batch-size 128 \
    --num-epochs 1 \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --bf16 \
    --learning-rate $LR \
    --warmup-ratio 0.03 \
    --lr-scheduler-type cosine \
    --save-steps 0.2 \
    --use-regressor \
    --intervention-signal-distribution extreme \
    --intervention-extreme-scale $SCALE \
    --cache-dir data/$CACHE_GROUP \
    --output-dir outputs/runs/$EXP_GROUP \
    --output experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache
# NB: --disable-adapter-in-input is intentionally NOT passed. THAT is the switch:
# train_introspector_joint.py now honors training_config.disable_adapter_in_input
# (as train_introspector.py always did), so omitting it here is what turns the
# input gate off. The gated prev-token sweep passes it and is otherwise identical.

### cache activations and generate dataset (no-op: reuses $CACHE_GROUP cache)
python cache_signals.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache

## train introspector -- INPUT GATE OFF
python train_introspector_joint.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
    --joint-loss-weight $W \
    #    --overwrite-cache

# post training analysis of each checkpoint (same config => same gating as
# training; the eval forwards must not silently switch gating behavior)
for CK in outputs/runs/$EXP_GROUP/$EXP_NAME/checkpoint-*; do
  python train_introspector_joint.py \
   --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
   --joint-loss-weight $W \
   --diag-only \
   --checkpoint "$CK"
done
