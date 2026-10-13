#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_dstsrc_endo300K
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=24:00:00
# SGE writes one file per task into this DIRECTORY. It must already exist -- otherwise
# SGE treats the value as a filename and every task dumps into one file.
#$ -o logs
#$ -j y
#$ -t 1-54
# Waits for 7-0's cache build. If that job has already finished (or was never submitted)
# there is nothing to hold on and this starts immediately -- the cache guard below is what
# catches the never-submitted case.
#$ -hold_jid attn_w_dstsrc_cache300K

# ============================================================================
# STAGE 7: (destination, source) PAIR GRID, endogenous, train 300K / test 30K.
#
# Every earlier P run pinned the destination to the last token and moved only the source.
# This asks whether the readout survives moving the DESTINATION too. All 32 x 32 pairs is
# too many, so both ends are drawn from one set of ABSOLUTE offsets,
#     OFFSETS = {0 -1 -3 -7 -11 -15 -19 -23 -27 -31}     (0 = last token, -31 = BOS)
# and every causal pair (dst, src) with src <= dst is trained: 10+9+...+1 = 55, minus
# (-31,-31) = 54. Each pair is trained at the SAME three heads, one per layer band, and
# the analysis averages over them (analysis/plot_attention_dst_src_grid.py).
#
# The recipe is 2-1's gate-off unanchored arm (klnone:off) unchanged; read 2-1's header
# for why each hyperparameter is what it is. Only what differs is described here.
#
# OFFSET CONVENTIONS -- the user-facing offsets are NOT what the code takes.
#   destination: the token selector is end-relative with -1 = LAST token
#     (cache_signals.py sample_token_position: pos<0 -> seq_len+pos), so absolute offset
#     D is selector D-1. 0 -> -1, -27 -> -28, -31 -> -32 = position 0 = BOS.
#   source: src_offset is relative to the DESTINATION (cache_signals.py
#     attention_src_positions: src = dst + src_offset), so pair (D, S) is src_offset S-D.
#   Getting either wrong raises nothing -- it trains a different, plausible cell. The run
#   name keeps the code's convention (P<L>.<H>@<S-D>_POS<D-1>) so RUN_RE parses it
#   unchanged; the analysis converts back and asserts the result lies on OFFSETS.
#
# WHY (-31,-31) IS DROPPED. With the destination at BOS the causal row has length 1, so
# the weight is exactly 1.0 for every document: zero variance, undefined corr. Worse, the
# interventional eval and the scale sweep renormalize by (1 - p_j) and AttentionWeightPort
# asserts p_j < 1 - EPS, so the run would crash rather than score. The analysis draws it
# as a constant cell instead.
#
# KNOWN RISK: BOS WEIGHT ~ 1 AT AN EARLY DESTINATION. In pairs like (-27,-31) the row has
# only 5 entries and a sink head can put > 1 - 1e-6 on BOS for some documents. That trips
# the same p_j assert inside the interventional eval. It is loud (a crash in the task log),
# not silent, so it is left to surface rather than guarded around -- a clamp would inject a
# value different from the one recorded as the target.
#
# HEADS: one per layer band, drawn once with random.Random(7) and pinned here so every
# pair is trained at the SAME heads -- that is what makes the 3-head mean comparable
# across the grid. They are passed as explicit signal names, not through
# attention_layer_sampler, whose `heads` list applies to every layer alike.
#
# ONE CACHE PER DESTINATION, built by 7-0. The token selector is part of the cache
# identity, so each destination is its own dataset; that dataset holds the three heads at
# EVERY source offset the destination can reach (up to 30 signals, one forward per text).
# Every pair sharing a destination passes the same signal list and differs only in
# --signal-filter, which is not part of the identity, so they all read one cache. 54 tasks
# would otherwise race into 9 directories (cache_signals.py skips on `train/` alone), hence
# the 2-0/2-1 split again: 7-0 runs THIS script with CACHE_ONLY=1, once per destination.
#
# ONE ARRAY TASK PER PAIR, three trainings each (one per head). A 300K training took
# ~46 min in 2-1; three plus evals fit h_rt=24:00:00 with room.
#
# Usage
#   qsub scripts/attention_weight/7-0_cache_dst_src_grid_300K.sh
#   qsub scripts/attention_weight/7-1_train_dst_src_grid_endogenous_300K.sh   # holds on 7-0
#   then: python analysis/plot_attention_dst_src_grid.py outputs/runs/attn_weight_dst_src_grid_300K
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

# The attention family's environment, deliberately NOT the neuron/SAE one (they charge
# llamagrp and activate peter's env, under cuda/11.8).
module load cuda/12.8
module load miniconda/23.11.0
conda activate /projectnb/multilm/jongin/anaconda3/envs/introspection

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection

set -e

# 1 only when 7-0 calls this script: build one destination's dataset and stop.
CACHE_ONLY=${CACHE_ONLY:-0}

TRAIN_SAMPLES=300000
DEV_SAMPLES=200
# Dev skips this many RAW documents, so it must clear train's raw span. 300K train at a
# ~99% length-filter pass rate spans ~303K raw documents, so 1M still leaves a wide margin.
DEV_OFFSET=1000000
TEST_SAMPLES=30000
# Derived, never hardcoded: the run directory name is the analysis scripts' only record
# of how much data a run saw.
SAMPLE_TAG="$((TRAIN_SAMPLES / 1000))K"

EXP_GROUP=attn_weight_dst_src_grid_${SAMPLE_TAG}
export WANDB_RUN_GROUP=$EXP_GROUP
OUT_ROOT=outputs/runs/$EXP_GROUP

## ---- the grid ----
# ABSOLUTE offsets from the last token (0) back to BOS (-31). Must stay in sync with
# OFFSETS in analysis/plot_attention_dst_src_grid.py, which asserts every run lands on it.
OFFSETS=(0 -1 -3 -7 -11 -15 -19 -23 -27 -31)
BOS_OFFSET=-31
# layer:head, one per band (lower / mid / upper). random.Random(7).randrange(32) per layer.
HEADS=("2:20" "7:9" "11:25")

# Pairs, destination-major: all pairs of one destination get consecutive task ids.
PAIRS=()
DSTS=()
for ((i = 0; i < ${#OFFSETS[@]}; i++)); do
  D=${OFFSETS[$i]}
  if [ "$D" -eq "$BOS_OFFSET" ]; then continue; fi   # (-31,-31) only; see header
  DSTS+=($D)
  for ((j = i; j < ${#OFFSETS[@]}; j++)); do
    PAIRS+=("${D}:${OFFSETS[$j]}")
  done
done
if [ ${#PAIRS[@]} -ne 54 ] || [ ${#DSTS[@]} -ne 9 ]; then
  echo "FATAL: built ${#PAIRS[@]} pairs over ${#DSTS[@]} destinations, expected 54 over 9" >&2
  exit 1
fi

## ---- decode this task ----
# CACHE_ONLY tasks index DESTINATIONS (7-0 runs -t 1-9) and use the self pair (D, D): any
# pair of that destination yields the same signal list, hence the same cache.
index=$(($SGE_TASK_ID - 1))
if [ "$CACHE_ONLY" = "1" ]; then
  D=${DSTS[$index]}
  S=$D
else
  IFS=':' read -r D S <<< "${PAIRS[$index]}"
fi
if [ -z "$D" ] || [ -z "$S" ]; then
  echo "FATAL: could not decode a pair from SGE_TASK_ID=$SGE_TASK_ID (CACHE_ONLY=$CACHE_ONLY)" >&2
  exit 1
fi
if [ "$S" -gt "$D" ] || [ "$D" -eq "$BOS_OFFSET" ]; then
  echo "FATAL: decoded invalid pair (dst=$D, src=$S); need src <= dst and dst != BOS" >&2
  exit 1
fi

DST_SEL=$((D - 1))       # token selector: -1 = last token, so absolute 0 -> -1
REL=$((S - D))           # src_offset, relative to the destination, <= 0
W=1.0                    # weight on the ENDOGENOUS term; 1.0 => no interventional loss
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
SEQ=32                   # max AND min: a post-softmax weight is normalized over the
                         # whole row, and absolute offset -31 must BE BOS in every document
PRECISION=6              # see the note in the pilot: at 3 the `target` string collapses
SCALE=4                  # log-odds units now, not the logit experiment's 10
SCALE_SWEEP=1,2,4,8,16
GATE=off                 # 2-1's klnone:off arm, the only one this grid runs
KL_TAG=klnone
OOD_FLAG="--ood-eval"
OVERWRITE=""             # set to --overwrite-cache to force regeneration

## ---- this destination's signal list: 3 heads x every source it can reach ----
# Identical for every pair sharing D, so all of them resolve to one cached dataset.
SIGNALS=()
for HS in "${HEADS[@]}"; do
  IFS=':' read -r L H <<< "$HS"
  for SRC in "${OFFSETS[@]}"; do
    if [ "$SRC" -le "$D" ]; then SIGNALS+=("P${L}.${H}@$((SRC - D))"); fi
  done
done
SIGNAL_LIST=$(IFS=,; echo "${SIGNALS[*]}")

PAIR_TAG="dst${D}_src${S}"
echo
echo "=== STAGE 7 @ ${SAMPLE_TAG} pair (dst, src) = ($D, $S) ======================="
echo "  token selector=$DST_SEL  src_offset=$REL  cache_only=$CACHE_ONLY"
echo "  heads: ${HEADS[*]}   arm: gate=$GATE $KL_TAG"
echo "  cache signals (${#SIGNALS[@]}): $SIGNAL_LIST"
echo "  samples: train=$TRAIN_SAMPLES dev=$DEV_SAMPLES test=$TEST_SAMPLES seq=$SEQ"
echo "  output: $OUT_ROOT"
echo "=========================================================="

mkdir -p experiment_configs/$EXP_GROUP

for SAMPLE in $(seq 0 $((${#HEADS[@]} - 1))); do
  IFS=':' read -r L H <<< "${HEADS[$SAMPLE]}"
  SIG="P${L}.${H}@${REL}"
  EXP_NAME="endo_${SIG}_POS${DST_SEL}_gate${GATE}_${KL_TAG}_w${W}_${SAMPLE_TAG}"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo
  echo "=== [$PAIR_TAG head $SAMPLE] $EXP_NAME ==="

  # Every pair of this destination passes the SAME signal list so all of them resolve to
  # one cached dataset; --signal-filter is what differs and is not part of the identity.
  python generate_experiment_config.py \
      --experiment-name "$EXP_NAME" \
      --model meta-llama/Llama-3.2-1B \
      --train-dataset allenai/c4 \
      --train-dataset-split "train" \
      --train-max-samples $TRAIN_SAMPLES \
      --dev-max-samples $DEV_SAMPLES \
      --dev-offset-from-train $DEV_OFFSET \
      --test-dataset allenai/c4 \
      --test-dataset-split "validation" \
      --test-max-samples $TEST_SAMPLES \
      --max-seq-length $SEQ --min-seq-length $SEQ \
      --hf-data-kwargs '{"config": "en", "streaming": true}' \
      --train-signals "$SIGNAL_LIST" \
      --test-signals "$SIGNAL_LIST" \
      --signal-filter "$SIG" \
      --train-token-selector $DST_SEL \
      --test-token-selector $DST_SEL \
      --precision $PRECISION \
      --train-batch-size $BS \
      --eval-batch-size 128 \
      --num-epochs 1 \
      --lora-r $LORA_R \
      --lora-alpha $LORA_ALPHA \
      --lora-dropout $LORA_DO \
      --bf16 \
      --learning-rate $LR \
      --warmup-ratio 0.03 \
      --lr-scheduler-type cosine \
      --save-steps 0 \
      --use-regressor \
      --intervention-signal-distribution extreme \
      --intervention-extreme-scale $SCALE \
      --cache-dir data \
      --output-dir $OUT_ROOT \
      --output "$CONFIG" \
      $OVERWRITE

  if [ "$CACHE_ONLY" = "1" ]; then
    # 7-0's path: the only place a destination's 300K dataset is built.
    python cache_signals.py --config "$CONFIG" $OVERWRITE
    echo "CACHE_ONLY: destination $D cached, exiting before training."
    exit 0
  fi

  # Training path: the cache must already exist. Building it here would race the other
  # pairs of this destination into one directory (cache_signals.py skips on `train/`
  # alone, so a loser could read a half-written dataset without any error).
  CACHE_DIR=$(python -c "
from pathlib import Path
from src.introspector.config import ExperimentConfig
print(ExperimentConfig.model_validate_json(Path('$CONFIG').read_text()).dataset_cache_dir)
")
  if [ ! -d "$CACHE_DIR/train" ]; then
    echo "FATAL: no cache at $CACHE_DIR -- qsub scripts/attention_weight/7-0_cache_dst_src_grid_300K.sh first" >&2
    exit 1
  fi
  python cache_signals.py --config "$CONFIG" $OVERWRITE

  # Cache-alignment check once per pair: every pair is a different (dst, src) position,
  # and a moved destination is exactly where an off-by-one in the selector conversion
  # above would show up -- as best offset != +0 here and nowhere else.
  if [ "$SAMPLE" -eq 0 ]; then DRIFT="--force-drift-eval"; else DRIFT=""; fi

  python train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      --scale-sweep "$SCALE_SWEEP" \
      $DRIFT \
      $OOD_FLAG \
      $OVERWRITE

  python train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      $OOD_FLAG \
      --ood-only \
      $OVERWRITE

  # LM cost: the unanchored arm's achieved KL IS the cost-of-access result, so score two
  # heads for an error bar, as 2-1 does for its klnone gate-off arm.
  if [ "$SAMPLE" -lt 2 ]; then
    python score_lm_degradation.py --config "$CONFIG"
  fi
done

echo
echo "=========================================================="
echo "STAGE 7 @ ${SAMPLE_TAG} PAIR (dst, src) = ($D, $S) DONE. Once all 54 tasks finish:"
echo "  python analysis/plot_attention_dst_src_grid.py $OUT_ROOT"
echo "End date : $(date)"
echo "=========================================================="
