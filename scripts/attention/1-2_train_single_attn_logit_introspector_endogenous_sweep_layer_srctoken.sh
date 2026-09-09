#!/bin/bash -l
# NOTE: -P multilm, not llamagrp. Every other script under scripts/ charges llamagrp and
# activates peter's env; the attention experiments run out of this account, so both the
# SGE project and the conda env below differ on purpose. Do not align them "for
# consistency" with the neuron/SAE scripts.
#$ -P multilm
#$ -N attn_endo_sweep_layer_src
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=24:00:00
# SGE writes one file per task into this DIRECTORY. It must already exist -- if it
# does not, SGE treats the value as a filename and every task dumps into one file
# instead (which is how the stray `log` file in the repo root came to be).
#$ -o logs
#$ -j y
###$ -m ea
#$ -t 81-160

# ============================================================================
# ENDOGENOUS-ONLY (no interventional loss) regressor sweep over ATTENTION LOGITS:
# 8 layers x 10 source offsets x 8 heads x {gate on, off} = 1280 runs.
# The attention analogue of neuron/train_single_mlp_introspector_endogenous_sweep_layer_token.sh.
#
# THE SIGNAL. The pre-softmax attention logit
#     z[l,h,i,j] = (R_i W_Q^{l,h} x_i) . (R_j W_K^{l,g(h)} x_j) / sqrt(d_head)
# after RoPE and the 1/sqrt(d) scaling, BEFORE the causal mask and the softmax.
# It is not the output of any module, so it is read and written through a custom
# attention implementation (src/introspector/models/attention_logits.py) rather than a
# forward hook. That implementation MUST be used for caching and training alike; both
# paths select it from the config's signal types, so nothing here has to say so.
#
# ONE array task = one (layer, source offset) cell; the 8 heads run inside it, so a
# cell's dataset is built once and reused 8 times. 80 datasets instead of 1280.
#
# W=1.0 => pure endogenous, exactly as in the neuron sweep: the interventional forward is
# skipped during training, and the interventional TEST afterwards measures CONTEXT BYPASS
# (how much of the endogenous score survives when the reported value is replaced by an
# injected one) rather than training progress.
#
# WHY min_seq_length == max_seq_length == 32. The deepest offset (-31) is meant to be the
# BOS token, the attention sink. Without a length floor C4 supplies documents shorter than
# 32 tokens, and on those sample_token_position clamps the destination toward the front --
# so "-31" would silently resolve to position 0 on a short document and to an ordinary
# token on a long one, and the sink measurement would be right only by accident. Pinning
# min == max makes every cached document exactly 32 tokens: destination -1 is index 31 and
# offset -31 is index 0 = BOS, exactly. cache_signals refuses the config otherwise.
# NOTE this differs from the neuron sweep, which set no floor: the two corpora are
# therefore not the identical document population. Say so when the plots go side by side.
#
# HEAD SELECTION. HEAD_SELECT=random samples N_HEADS_CACHE heads per layer, seeded per
# (seed, layer) by the same hash resolution.py uses for neurons -- so a layer's heads are
# IDENTICAL at every source offset and the offset curve reads per head. Drawing fresh
# heads per offset would let head variance leak into the offset effect, which is the bug
# the neuron sweep's position axis already had once. HEAD_SELECT=fixed pins
# (0 4 8 12 16 20 24 28) instead: stride 4 over 32 query heads visits each of the model's
# 8 GQA key/value groups exactly once. That is EXPERIMENT 2; nothing else changes.
#
# READING THE RESULT.
#   * layer 0 is a CEILING CONTROL, not a data point. Its pre-softmax logit is a function
#     of embeddings and RoPE only, hence fully determined by (source token id, destination
#     token id, relative distance). With the offset fixed and the destination pinned to
#     the last token, a model can score perfectly from the prompt's token strings without
#     reading anything internal. It measures "how much is predictable from text alone".
#   * offset -31 (BOS) is a separate category, not the far end of the offset axis. Sink
#     logits have a very different mean and can have small variance, and both the loss
#     (_var_norm_mse) and the metric (Pearson corr) are scale invariant -- a near-constant
#     signal gives nan or noise, the same trap as a dead neuron. Check the per-signal std
#     in dataset_signal_stats.json before reading anything into such a cell.
#   * the prompt names the source token by string only, so a token repeated in the
#     32-token context is ambiguous. cache_signals records src_token_unique_rate in the
#     cache metadata; split the analysis on it rather than assuming the cost is zero.
#
# CAVEAT ON SCALE. --intervention-extreme-scale 10 is kept for parity with the neuron
# sweep and is INERT here: at W=1.0 no interventional forward runs during training, and
# evaluate_interventional_test forces gaussian at scale 1.0. It does reach
# interventional_scale_sweep, and there attention behaves unlike a neuron: "extreme" draws
# uniformly from scale x the empirical RANGE, so a logit range of ~30 at scale 10 injects
# +/-150, which saturates the softmax -- past that point different injected values produce
# the SAME forward pass and the model cannot report what it was given. Read the scale
# sweep before choosing a scale for any joint (W<1) follow-up.
# ============================================================================

## SCC env setup
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

# NOT the env the neuron scripts use. scripts/neuron/*.sh activate
# /projectnb/llamagrp/peter/low-level-introspection/introspection_env; the attention
# experiments are run out of this account instead, so they pin their own env (and the
# cuda version it was built against). Do not "fix" this back to match the neuron
# scripts.
module load cuda/12.8
module load miniconda/23.11.0
conda activate /projectnb/multilm/jongin/anaconda3/envs/introspection

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection

set -e

## ---- experiment arm ----
# EXPERIMENT 1 (random heads per layer):  HEAD_SELECT=random
# EXPERIMENT 2 (pinned heads):            HEAD_SELECT=fixed
HEAD_SELECT=${HEAD_SELECT:-random}
FIXED_HEADS=(0 4 8 12 16 20 24 28)

if [ "$HEAD_SELECT" = "fixed" ]; then
  EXP_GROUP=attn_endo_sweep_layer_src_fixedhead
else
  EXP_GROUP=attn_endo_sweep_layer_src
fi
export WANDB_RUN_GROUP=$EXP_GROUP

## ---- decode (gate, layer, source offset) from the flat array index ----
# GATE is the OUTER dimension (tasks 1-80 = gate on, 81-160 = gate off): both arms of a
# cell share its cached dataset, so keeping them 80 tasks apart stops concurrent slots
# racing to build the same directory.
index=$(($SGE_TASK_ID-1))
LAYERS=(0 2 4 6 8 10 12 14)                       # 8
SRC_OFFSETS=(0 -1 -3 -7 -11 -15 -19 -23 -27 -31)  # 10; 0 = self, -31 = BOS
GATES=(on off)
GATE=${GATES[$((index / 80))]}
cell_idx=$((index % 80))
LAYER=${LAYERS[$((cell_idx / 10))]}
SRC_OFFSET=${SRC_OFFSETS[$((cell_idx % 10))]}

DST_POS=-1               # destination token: last token of the input text
W=1.0                    # weight on the ENDOGENOUS term; 1.0 => no interventional loss
SEED=21                  # head-selection seed (HEAD_SELECT=random)
N_HEADS_CACHE=8          # heads cached per cell
N_HEADS_TRAIN=8          # heads trained per cell (<= N_HEADS_CACHE)
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
SCALE=10                 # only affects the post-training interventional diagnostics
SEQ=32                   # max AND min: every cached document is exactly this long
OOD_FLAG="--ood-eval"
OVERWRITE=""             # set to --overwrite-cache to force regeneration

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

CELL="l${LAYER}_src${SRC_OFFSET}"

if [ "$HEAD_SELECT" = "fixed" ]; then
  HEAD_JSON=$(IFS=,; echo "[${FIXED_HEADS[*]}]")
  SAMPLER="{\"type\":\"attention_layer_sampler\",\"signal_type\":\"attention_score\",\"layers\":[${LAYER}],\"heads\":${HEAD_JSON},\"src_offsets\":[${SRC_OFFSET}],\"seed\":${SEED}}"
else
  SAMPLER="{\"type\":\"attention_layer_sampler\",\"signal_type\":\"attention_score\",\"layers\":[${LAYER}],\"heads_per_layer\":${N_HEADS_CACHE},\"src_offsets\":[${SRC_OFFSET}],\"seed\":${SEED}}"
fi

## ---- resolve this cell's head set ONCE ----
# Generated, not run: it exists so resolution.py stays the single source of truth for
# which heads a (seed, layer) maps to, instead of the selection being duplicated here.
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_${CELL}.json
python generate_experiment_config.py \
    --experiment-name "_cell_${CELL}" \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 --train-dataset-split "train" --train-max-samples 300000 \
    --dev-max-samples 200 --dev-offset-from-train 1000000 \
    --test-dataset allenai/c4 --test-dataset-split "validation" --test-max-samples 3000 \
    --max-seq-length $SEQ --min-seq-length $SEQ \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "$SAMPLER" \
    --test-signals "$SAMPLER" \
    --same-train-test-signals \
    --train-token-selector $DST_POS \
    --test-token-selector $DST_POS \
    --precision 3 \
    --cache-dir data --output-dir outputs/runs/$EXP_GROUP \
    --output "$CELL_CONFIG" \
    --overwrite-cache

SIGNALS=($(python -c "
import json
cfg = json.load(open('$CELL_CONFIG'))
print(' '.join(f\"A{s['layer']}.{s['index']}@{s['src_offset']}\" for s in cfg['train_signals']))
")) # (A8.5@-3 A8.9@-3 ... )
SIGNAL_LIST=$(IFS=,; echo "${SIGNALS[*]}") # "A8.5@-3,A8.9@-3,..."
echo "cell $CELL signals: $SIGNAL_LIST"

for SAMPLE in $(seq 0 $((N_HEADS_TRAIN - 1))); do
  SIG=${SIGNALS[$SAMPLE]}
  EXP_NAME="endo_${SIG}_POS${DST_POS}_gate${GATE}_w${W}_300K"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo "=== [$CELL sample $SAMPLE] $EXP_NAME ==="

  # every sample passes the SAME signal list, so all 8 resolve to one cached dataset;
  # --signal-filter is what differs and is not part of the cache identity
  python generate_experiment_config.py \
      --experiment-name "$EXP_NAME" \
      --model meta-llama/Llama-3.2-1B \
      --train-dataset allenai/c4 \
      --train-dataset-split "train" \
      --train-max-samples 300000 \
      --dev-max-samples 200 \
      --dev-offset-from-train 1000000 \
      --test-dataset allenai/c4 \
      --test-dataset-split "validation" \
      --test-max-samples 3000 \
      --max-seq-length $SEQ --min-seq-length $SEQ \
      --hf-data-kwargs '{"config": "en", "streaming": true}' \
      --train-signals "$SIGNAL_LIST" \
      --test-signals "$SIGNAL_LIST" \
      --signal-filter "$SIG" \
      --train-token-selector $DST_POS \
      --test-token-selector $DST_POS \
      --precision 3 \
      --train-batch-size $BS \
      --eval-batch-size 128 \
      --num-epochs 1 \
      --lora-r $LORA_R \
      --lora-alpha $LORA_ALPHA \
      --lora-dropout $LORA_DO \
      $GATE_FLAG \
      --bf16 \
      --learning-rate $LR \
      --warmup-ratio 0.03 \
      --lr-scheduler-type cosine \
      --save-steps 0 \
      --use-regressor \
      --intervention-signal-distribution extreme \
      --intervention-extreme-scale $SCALE \
      --cache-dir data \
      --output-dir outputs/runs/$EXP_GROUP \
      --output "$CONFIG" \
      $OVERWRITE

  # builds the cell's 8-head dataset on SAMPLE=0, no-ops for the other 7
  python cache_signals.py --config "$CONFIG" $OVERWRITE

  # Cache-alignment regression check once per cell (it is a property of the data
  # pipeline, so one run per dataset is enough). For attention this is the ONLY
  # automatic check that the live logit equals the cached one -- if caching and training
  # ever ran under different attention implementations, or the (dst, src) pair were off
  # by one, this is where it shows. Want best offset +0, corr ~0.9999, "alignment OK".
  if [ "$SAMPLE" -eq 0 ]; then DRIFT="--force-drift-eval"; else DRIFT=""; fi

  python train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      $DRIFT \
      $OOD_FLAG \
      $OVERWRITE

  python train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    $DRIFT \
    $OOD_FLAG \
    --ood-only \
    $OVERWRITE
done
