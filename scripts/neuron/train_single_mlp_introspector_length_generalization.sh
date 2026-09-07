#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_lengen
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=6:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-50

# ============================================================================
# LENGTH GENERALIZATION: does the readout reach tokens further back than it was
# trained to reach?
#
# The introspection prompt sits AFTER the input text, so introspecting position p
# means reaching back (text_len - 1 - p) tokens from where the readout happens.
# Training on a narrow window at the end of the text leaves that reach untested --
# "attend to the last few tokens" solves training perfectly and generalizes to
# nothing.
#
# GRID (10 cells x 5 neurons = 50 tasks)
#   train_len   train window (tokens from the end)
#      16       16                                  <- window == length: the ceiling
#      32       16, 32
#      64       16, 32, 64
#     128       16, 32, 64, 128
#
#   Windows are NESTED and expressed as [-w,-1], so a wider window is a superset of a
#   narrower one. Two things fall out of that:
#     - the diagonal (window == train_len) has NO generalization gap. It measures raw
#       reach, and it is the control that says whether a failure off the diagonal is a
#       generalization failure or an inability to reach that far at all.
#     - "unseen distance" means something DIFFERENT per cell, so the eval labels every
#       distance bin in_train_range per run and never averages the two together.
#   Matched windows across lengths (16 appears at all four, 32 at three, 64 at two) ask
#   whether training on longer sequences changes the mechanism, holding reach fixed.
#
# EVAL is not this script's business: --length-eval runs the whole
# (eval length x distance bin) grid at the end of every run, from ONE cache per eval
# length, binned on distance afterwards. Every model is evaluated at 16/32/64/128
# regardless of what it trained on -- that is the point. The eval lengths mirror the
# TRAIN lengths so every run has its own length among them: without 32 in the list, a
# run trained at 32 would have no in-distribution reference point to fall away from.
#
# WHY MIN_LEN, AND WHY IT IS THE SAME FOR EVERY ARM
#   max_seq_length only TRUNCATES. Without a floor a "length 128" cache is really
#   "whatever C4 gave us, capped at 128", and sample_token_position clamps a [-128,-1]
#   range down into whatever each document actually has -- so both the length axis and
#   the distance axis quietly collapse toward the short end, with nothing in the config
#   or the logs to say so.
#   Measured on C4-en (n=10k, Llama-3.2 tokenizer): median 247 tokens, 72.2% reach 128,
#   48.8% reach 256, 25.1% reach 512. Hence 128: a mild ~1.4x oversample.
#   The floor is 128 for EVERY arm, including train_len=16. A per-arm floor would make
#   the 16-token arm and the 128-token arm draw from different document populations
#   (long C4 is prose; short C4 is boilerplate and listings) and the content shift would
#   ride along inside the length effect. So the 16-token arm is "the first 16 tokens of
#   documents that are at least 128 tokens long".
#   NOTE this makes these runs NOT directly comparable to the earlier unfiltered pos=-1
#   sweeps; min_seq_length is part of the cache identity, so they will not collide.
#
# KL: UNCONSTRAINED, and this is not a free choice -- a budget would silently decide
#   the result. From the layer-10 KL sweep already in outputs/, interventional corr:
#       distance  0 (pos -1) : KL 0.1 -> 0.845    unconstrained -> 0.836
#       distance 27 (pos -28): KL 0.1 -> 0.185    KL 3.0 -> 0.726    unconstrained -> 0.957
#   A budget that is ample at distance 0 is catastrophic at distance 27. Running this
#   grid at KL 0.1 would starve every wide-window cell and the curve would read as
#   "no length generalization" when it was really "no capacity". So the budget is
#   removed as a variable here and the grid measures the mechanism.
#   NOTE this does NOT measure capacity. Unconstrained training drifts as far as the
#   optimizer finds convenient, not as far as the task requires, and with the anchor on
#   the achieved KL just tracks the target by construction -- neither is a minimum.
#   Capacity is the KNEE of the budget-vs-performance frontier, so measure it with a KL
#   sweep on a few cells (e.g. len 128 x windows 16/64/128 x KL {0.1,0.3,1,3,inf}) and
#   read how the knee moves with window width. Also note the unconstrained arm wrecks
#   the LM (ppl x500+ at gate-off), so it answers "does the mechanism exist and
#   generalize", not "is this deployable".
#
# DATA: total examples held FIXED across all cells, deliberately. As the window widens,
#   examples per position fall -- and that is the measurement, not a confound. If
#   positions share one mechanism, examples are fungible across them and per-position
#   fidelity stays flat as the window widens; if each position needs private machinery,
#   fidelity falls like a data-starvation curve. Calibrate against a single-position
#   data-scaling run before reading the shape.
# ============================================================================

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

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection
export WANDB_RUN_GROUP=mlp_lengen

# stop on the first failure: without this a failed config generation silently falls
# through to training on a stale config from a previous run
set -e

index=$(($SGE_TASK_ID-1))

# ---- grid ----------------------------------------------------------------
# One entry per cell, "train_len:train_window". Add a cell here and bump #$ -t.
CELLS=(
  "16:16"
  "32:16"  "32:32"
  "64:16"  "64:32"  "64:64"
  "128:16" "128:32" "128:64" "128:128"
)
# LAYER 4, not 10. The injected signal has to be carried by ATTENTION from the
# introspected position to the readout position, and attention can only move it in
# layers ABOVE the injection site. Llama-3.2-1B has 16 layers, so injecting at L10
# leaves 6 layers of transport and L4 leaves 12 -- and transport across distance is
# exactly the thing this experiment asks to generalize. Not earlier than ~4: neurons in
# the first couple of layers are close to token-identity detectors, which makes the
# ENDOGENOUS half solvable by reading the token off the input (bypass) and inflates its
# correlation for the wrong reason. The interventional half stays valid either way.
# These are the same 5 neurons the layer/token grid draws at L4 (seed 21, first 5 of
# 10), so the two experiments are directly comparable at this layer.
NEURONS=(4.5676 4.520 4.1216 4.6896 4.6386)

# NEURON is the FAST axis and CELL the slow one, so the 5 tasks that share a cell's
# dataset cache are adjacent -- but they are then serialized by the cache-build race,
# so the ordering below is deliberately the other way round: consecutive task ids land
# in DIFFERENT cells, and the 5 neurons of one cell are 8 apart. Neurons within a cell
# share one cache; two tasks starting a build of the same cache at the same moment is
# the race this avoids. With 10 cells the 5 neurons of a cell are 10 task ids apart.
N_CELLS=${#CELLS[@]}
CELL_IDX=$(( index % N_CELLS ))
NEURON_IDX=$(( index / N_CELLS ))

CELL=${CELLS[$CELL_IDX]}
TRAIN_LEN=${CELL%%:*}
WINDOW=${CELL##*:}
SIG=${NEURONS[$NEURON_IDX]}

# Introspect uniformly over the last WINDOW tokens. A window equal to the length is
# the whole document, i.e. the no-gap ceiling arm.
TOKEN_SEL="-${WINDOW}:-1"

MIN_LEN=128              # same floor for every arm -- see header
EVAL_LENGTHS=16,32,64,128  # every model evaluated at all four
GATE=off                 # adapter active over the input text; gate-on cannot read
                         # earlier positions at all (no trainable param at the site)
W=0.01                   # weight on the ENDOGENOUS term; interventional gets (1-W)
SCALE=10
KL_TAG=klinf             # unconstrained -- see below; no --kl-anchor-target is passed
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
N_TRAIN=300000           # held FIXED across cells on purpose -- see header
N_TEST=30000

# PROMPT: the index arm. At 128 tokens duplicate tokens are near-certain, so naming a
# token by its string alone is genuinely ambiguous and the index is not optional.
# The two prompt variants now differ ONLY by the " (token #N)" clause -- they used to
# also differ in quoting, which would have confounded this comparison with a
# tokenization change. Flip to prompt_no_token_index for the content-addressing arm.
PROMPT_TYPE=prompt_with_token_index

# set to --overwrite-cache to force config/dataset/run regeneration
OVERWRITE=""

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

if [ "$PROMPT_TYPE" = "prompt_with_token_index" ]; then EXP_GROUP="mlp_lengen"; else GATE_FLAG="mlp_lengen_prompt_no_token_index"; fi
EXP_NAME="lengen_N${SIG}_len${TRAIN_LEN}_win${WINDOW}_gate${GATE}_${KL_TAG}_w${W}"
CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
echo "task=$SGE_TASK_ID cell=$CELL neuron=$SIG -> $EXP_NAME"

### generate experiment config first as JSON
# --max-seq-length is the TRAIN length; the eval lengths are handled by --length-eval,
# which derives its own configs (and its own max_seq_length, since this run's is sized
# for TRAIN_LEN and would truncate every longer eval document).
python generate_experiment_config.py \
    --experiment-name "$EXP_NAME" \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 \
    --train-dataset-split "train" \
    --train-max-samples $N_TRAIN \
    --dev-max-samples 200 \
    --dev-offset-from-train 1000000 \
    --test-dataset allenai/c4 \
    --test-dataset-split "validation" \
    --test-max-samples $N_TEST \
    --max-seq-length $TRAIN_LEN \
    --min-seq-length $MIN_LEN \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "N${SIG}" \
    --test-signals "N${SIG}" \
    --train-token-selector="${TOKEN_SEL}" \
    --test-token-selector="${TOKEN_SEL}" \
    --prompt-type $PROMPT_TYPE \
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
    --use-regressor \
    --intervention-signal-distribution extreme \
    --intervention-extreme-scale $SCALE \
    --cache-dir data \
    --output-dir outputs/runs/$EXP_GROUP \
    --output "$CONFIG" \
    $OVERWRITE

### cache activations and generate dataset
# no-op once another task in the same cell has built it (the 5 neurons of a cell
# resolve to one canonical dataset name). min_seq_length is part of the cache
# identity, so these will NOT collide with the earlier unfiltered caches.
python cache_signals.py --config "$CONFIG" $OVERWRITE

## train, then run the full (eval length x distance bin) grid
# --force-drift-eval keeps the cache-alignment regression check running
# (want: best offset +0, corr ~0.9999, "alignment OK")
python train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    --force-drift-eval \
    --length-eval \
    --eval-lengths $EVAL_LENGTHS \
    $OVERWRITE

# Re-run ONLY the length eval on an existing checkpoint (no retraining):
#   python train_introspector_joint.py --config "$CONFIG" --joint-loss-weight $W \
#       --length-only --eval-lengths $EVAL_LENGTHS
