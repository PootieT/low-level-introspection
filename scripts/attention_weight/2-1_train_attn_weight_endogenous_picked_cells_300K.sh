#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_endo_picked300K
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=24:00:00
# SGE writes one file per task into this DIRECTORY. It must already exist -- otherwise
# SGE treats the value as a filename and every task dumps into one file.
#$ -o logs
#$ -j y
#$ -t 1-18
# Waits for 2-0's cache build. If that job has already finished (or was never submitted)
# there is nothing to hold on and this starts immediately -- the cache guard below is what
# catches the never-submitted case.
#$ -hold_jid attn_w_cache300K

# ============================================================================
# STAGE 2 @ 300K: endogenous training on the six hand-picked (layer, head, offset) cells,
# train 300K / test 30K instead of 2_train_attn_weight_endogenous_picked_cells.sh's 60K / 3K.
# The design is identical to that script; read its header for WHY each choice was made
# (the 3 x 2 cell design, the three arms, no "gate on + KL" arm, 8 heads cached / 3 trained).
# Only what differs is described here.
#
# ONE ARRAY TASK PER (ARM, CELL), not per cell. At 60K the slowest task (9 trainings) took
# 3h35m on an A6000; training grows 5x and the test eval 10x here, so 9 trainings per task
# would not fit h_rt=24:00:00. 3 arms x 6 cells = 18 tasks of 3 trainings each, roughly
# 4-6 h per task in the worst case.
#
# THE CACHE COMES FROM 2-0, NOT FROM THIS SCRIPT. Splitting the arms puts three tasks on the
# same cell dataset, and data_prep/cache_signals.py skips on `train/` existing, so concurrent tasks
# would race into one directory and could read a half-written cache. 2-0 builds each cell
# exactly once by running THIS script with CACHE_ONLY=1 (so the generate_experiment_config
# call, and hence the cache identity, cannot drift between the two), and the training path
# below refuses to run without a finished cache instead of building one.
#
# OUTPUT IS SEPARATED FROM 60K BY EXP_GROUP (attn_weight_endo_picked_300K), so the analysis
# scripts that glob a whole run directory never mix the two dataset sizes.
#
# Usage
#   qsub scripts/attention_weight/2-0_cache_attn_weight_300K.sh
#   qsub scripts/attention_weight/2-1_train_attn_weight_endogenous_picked_cells_300K.sh
#   then: qsub scripts/attention_weight/analysis/2-1_attn_weight_analysis_endo_300K.sh
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

# 1 only when 2-0 calls this script: build the cell's dataset and stop before training.
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

EXP_GROUP=attn_weight_endo_picked_${SAMPLE_TAG}
export WANDB_RUN_GROUP=$EXP_GROUP
OUT_ROOT=outputs/runs/$EXP_GROUP

## ---- decode (arm, cell) from the flat array index ----
# ARM is the outer dimension, so tasks 1-6 are arm 0 x the six cells: 2-0 runs exactly
# those six to build each cell's cache once.
# The SAME six cells, in the same order, as 2_train_attn_weight_endogenous_picked_cells.sh;
# see its comments for how they were picked.
index=$(($SGE_TASK_ID - 1))
CELLS=("low_near:2:31:-1"  "mid_near:7:4:-1"   "up_near:11:9:-1" \
       "low_far:2:26:-27"  "mid_far:7:14:-27"  "up_far:11:25:-27")
ARM_IDX=$((index / 6))
CELL=${CELLS[$((index % 6))]}
IFS=':' read -r BAND LAYER HEAD SRC_OFFSET <<< "$CELL"

if [ -z "$BAND" ] || [ -z "$LAYER" ] || [ -z "$HEAD" ] || [ -z "$SRC_OFFSET" ]; then
  echo "FATAL: could not decode cell '$CELL' from SGE_TASK_ID=$SGE_TASK_ID" >&2
  exit 1
fi

DST_POS=-1               # destination token: last token of the input text
W=1.0                    # weight on the ENDOGENOUS term; 1.0 => no interventional loss
SEED=21
N_HEADS_CACHE=8          # one forward pass regardless; Stage 3/4 reuse this cache
N_HEADS_TRAIN=3          # the spread across these 3 is the error bar
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
SEQ=32                   # max AND min: a post-softmax weight is normalized over the
                         # whole row, so documents of different lengths would not carry
                         # the same quantity at the same offset
PRECISION=6              # see the note in the pilot: at 3 the `target` string collapses
SCALE=4                  # log-odds units now, not the logit experiment's 10
SCALE_SWEEP=1,2,4,8,16
KL_BUDGET=0.1
OOD_FLAG="--ood-eval"
OVERWRITE=""             # set to --overwrite-cache to force regeneration

## ---- the three arms; this task runs ONE of them ----
# ARM format: name:gate:kl_flag_or_none
ARMS=("kl${KL_BUDGET}:off:--kl-anchor-target ${KL_BUDGET} --require-kl-anchor"
      "klnone:off:"
      "klnone:on:")
ARM=${ARMS[$ARM_IDX]}
if [ -z "$ARM" ]; then
  echo "FATAL: could not decode arm $ARM_IDX from SGE_TASK_ID=$SGE_TASK_ID" >&2
  exit 1
fi
IFS=':' read -r KL_TAG GATE KL_FLAG <<< "$ARM"
if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

echo
echo "=== STAGE 2 @ ${SAMPLE_TAG} (arm, cell) ======================="
echo "  arm=$KL_TAG gate=$GATE  cache_only=$CACHE_ONLY"
echo "  band=$BAND  layer=$LAYER  head=$HEAD  src_offset=$SRC_OFFSET  dst=$DST_POS"
echo "  cache $N_HEADS_CACHE heads / train $N_HEADS_TRAIN"
echo "  samples: train=$TRAIN_SAMPLES dev=$DEV_SAMPLES test=$TEST_SAMPLES seq=$SEQ"
echo "  output: $OUT_ROOT"
echo "=========================================================="

CELL_TAG="l${LAYER}_h${HEAD}_src${SRC_OFFSET}"

# The cell caches 8 heads at this (layer, offset). The hand-picked head is pinned first
# so it is always among the trained ones; the rest fill in around it for the error bar.
HEADS=($HEAD)
for h in $(seq 0 31); do
  if [ "$h" -ne "$HEAD" ] && [ ${#HEADS[@]} -lt $N_HEADS_CACHE ]; then HEADS+=($h); fi
done
HEAD_JSON=$(IFS=,; echo "[${HEADS[*]}]")
SAMPLER="{\"type\":\"attention_layer_sampler\",\"signal_type\":\"attention_weight\",\"layers\":[${LAYER}],\"heads\":${HEAD_JSON},\"src_offsets\":[${SRC_OFFSET}],\"seed\":${SEED}}"

## ---- resolve the cell's signal names ONCE ----
# Generated, not trained: it keeps resolution.py the single source of truth for the
# (seed, layer) -> head-set mapping instead of duplicating the selection in shell.
mkdir -p experiment_configs/$EXP_GROUP
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_${CELL_TAG}.json
python data_prep/generate_experiment_config.py \
    --experiment-name "_cell_${CELL_TAG}" \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 --train-dataset-split "train" --train-max-samples $TRAIN_SAMPLES \
    --dev-max-samples $DEV_SAMPLES --dev-offset-from-train $DEV_OFFSET \
    --test-dataset allenai/c4 --test-dataset-split "validation" --test-max-samples $TEST_SAMPLES \
    --max-seq-length $SEQ --min-seq-length $SEQ \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "$SAMPLER" \
    --test-signals "$SAMPLER" \
    --same-train-test-signals \
    --train-token-selector $DST_POS \
    --test-token-selector $DST_POS \
    --precision $PRECISION \
    --cache-dir data --output-dir $OUT_ROOT \
    --output "$CELL_CONFIG" \
    --overwrite-cache

SIGNALS=($(python -c "
import json
cfg = json.load(open('$CELL_CONFIG'))
print(' '.join(f\"P{s['layer']}.{s['index']}@{s['src_offset']}\" for s in cfg['train_signals']))
"))
SIGNAL_LIST=$(IFS=,; echo "${SIGNALS[*]}")
echo "cell $CELL_TAG signals: $SIGNAL_LIST"

for SAMPLE in $(seq 0 $((N_HEADS_TRAIN - 1))); do
  SIG=${SIGNALS[$SAMPLE]}
  EXP_NAME="endo_${SIG}_POS${DST_POS}_gate${GATE}_${KL_TAG}_w${W}_${SAMPLE_TAG}"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo
  echo "=== [$CELL_TAG arm=$KL_TAG gate=$GATE sample $SAMPLE] $EXP_NAME ==="

  # Every sample passes the SAME signal list so all heads resolve to one cached
  # dataset; --signal-filter is what differs and is not part of the cache identity.
  python data_prep/generate_experiment_config.py \
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
      --train-token-selector $DST_POS \
      --test-token-selector $DST_POS \
      --precision $PRECISION \
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
      --output-dir $OUT_ROOT \
      --output "$CONFIG" \
      $OVERWRITE

  if [ "$CACHE_ONLY" = "1" ]; then
    # 2-0's path: the only place a 300K cell dataset is built.
    python data_prep/cache_signals.py --config "$CONFIG" $OVERWRITE
    echo "CACHE_ONLY: cell $CELL_TAG cached, exiting before training."
    exit 0
  fi

  # Training path: the cache must already exist. Building it here would race the other
  # two arms of this cell into one directory (data_prep/cache_signals.py skips on `train/` alone,
  # so a loser could read a half-written dataset without any error).
  CACHE_DIR=$(python -c "
from pathlib import Path
from src.introspector.config import ExperimentConfig
print(ExperimentConfig.model_validate_json(Path('$CONFIG').read_text()).dataset_cache_dir)
")
  if [ ! -d "$CACHE_DIR/train" ]; then
    echo "FATAL: no cache at $CACHE_DIR -- qsub scripts/attention_weight/2-0_cache_attn_weight_300K.sh first" >&2
    exit 1
  fi
  python data_prep/cache_signals.py --config "$CONFIG" $OVERWRITE

  # Cache-alignment check once per (cell, arm): it is a property of the data pipeline,
  # so one run per dataset is enough. For a post-softmax weight this is the only
  # automatic check that the live value equals the cached one, and the failure it
  # catches is a NORMALIZATION mismatch, which otherwise just looks like a low score.
  if [ "$SAMPLE" -eq 0 ]; then DRIFT="--force-drift-eval"; else DRIFT=""; fi

  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      --scale-sweep "$SCALE_SWEEP" \
      $KL_FLAG \
      $DRIFT \
      $OOD_FLAG \
      $OVERWRITE

  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      $KL_FLAG \
      $OOD_FLAG \
      --ood-only \
      $OVERWRITE

  # LM cost. Only meaningful with the gate OFF: with it on the adapter never touches
  # the input text, so the measurement is 0 by construction. The unanchored arm's
  # achieved KL IS the cost-of-access result, so score two heads there for an error
  # bar; the anchored arm only needs a spot check.
  if [ "$GATE" = "off" ]; then
    if [ "$KL_TAG" = "klnone" ]; then LM_SCORE_N=2; else LM_SCORE_N=1; fi
    if [ "$SAMPLE" -lt "$LM_SCORE_N" ]; then
      python evals/score_lm_degradation.py --config "$CONFIG"
    fi
  fi
done

echo
echo "=========================================================="
echo "STAGE 2 @ ${SAMPLE_TAG} CELL $CELL_TAG ARM $KL_TAG gate=$GATE DONE. Next:"
echo "  qsub scripts/attention_weight/analysis/2-1_attn_weight_analysis_endo_300K.sh"
echo "End date : $(date)"
echo "=========================================================="
