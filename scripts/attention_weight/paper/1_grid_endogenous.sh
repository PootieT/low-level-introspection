#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_paper_grid_endo300K
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=24:00:00
# SGE writes one file per task into this DIRECTORY. It must already exist -- otherwise
# SGE treats the value as a filename and every task dumps into one file.
#$ -o logs
#$ -j y
#$ -t 19-20

# full run -t 1-20
# ============================================================================
# PAPER EXPERIMENT 1: (layer, source offset) GRID, endogenous, train 300K / test 30K.
#
#     layers  = {2 6 10 14}                 (x axis of the paper grid)
#     offsets = {-1 -3 -11 -19 -27}         (y axis; relative to the last token, BOS = -31)
#
# 4 x 5 = 20 cells, three heads per cell, 60 runs. The headline number for a cell is the
# mean over its three heads +- SE (notebook/attention_weight/paper/1_grid_endogenous_analysis.ipynb).
#
# THE HEADS ARE HAND-PICKED PER CELL, not shared across the grid as in 7-1. They come from
# notebook/attention_weight/1-2_handpic_target_analysis.ipynb on the 10K-document scan:
# drop BOS sink heads (per layer, keep the half with the lowest BOS sink score
# S = mean_t alpha_{t,0}), then take the three highest-std heads of the cell. The result
# is plots/attn_weight_scan_10K/grid_picks_keep0.5.csv, and CELLS below is checked
# against that file at start-up: a mistyped head number would otherwise train a
# different, plausible head without any error.
#
# The recipe is 2-1's gate-off unanchored arm (klnone:off), the same one 7-1 runs; read
# 2-1's header for why each hyperparameter is what it is.
#
# ONE ARRAY TASK PER CELL, AND THAT TASK BUILDS THE CELL'S OWN CACHE. The cache holds only
# the cell's three signals, so no other task reads or writes it. 2-0/7-0 exist because
# several tasks shared one cache directory and cache_signals.py skips on `train/` existing
# alone (a loser of the race reads a half-written dataset with no error); with one task
# per cache there is no race, so there is no separate cache job and no -hold_jid. The
# first head's cache_signals.py builds it, the next two skip.
#
# COST. A 300K cache of 8 signals took 13 GB on disk, so a 3-signal cell is ~5 GB and the
# grid ~100 GB. Per task: caching (~2 h, including the DEV_OFFSET=1M streaming skip) plus
# three ~46 min trainings and their evals, inside h_rt=24:00:00.
#
# Usage
#   qsub scripts/attention_weight/paper/1_grid_endogenous.sh
#   then run notebook/attention_weight/paper/1_grid_endogenous_analysis.ipynb
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

TRAIN_SAMPLES=300000
DEV_SAMPLES=200
# Dev skips this many RAW documents, so it must clear train's raw span. 300K train at a
# ~99% length-filter pass rate spans ~303K raw documents, so 1M still leaves a wide margin.
DEV_OFFSET=1000000
TEST_SAMPLES=30000
# Derived, never hardcoded: the run directory name is the analysis scripts' only record
# of how much data a run saw.
SAMPLE_TAG="$((TRAIN_SAMPLES / 1000))K"

EXP_GROUP=attn_weight_paper_grid_endo_${SAMPLE_TAG}
export WANDB_RUN_GROUP=$EXP_GROUP
OUT_ROOT=outputs/runs/$EXP_GROUP

## ---- the grid ----
# layer:src_offset:head,head,head -- heads in pick rank order (highest std first).
# Layer-major, so tasks 1-5 are layer 2. Must match PICKS below; checked at start-up.
PICKS=plots/attn_weight_scan_10K/grid_picks_keep0.5.csv
CELLS=("2:-1:31,30,0"   "2:-3:31,13,0"   "2:-11:13,24,10"  "2:-19:5,19,24"   "2:-27:5,19,24"
       "6:-1:3,0,31"    "6:-3:3,1,31"    "6:-11:31,29,19"  "6:-19:31,29,19"  "6:-27:31,29,19"
       "10:-1:28,8,16"  "10:-3:7,28,29"  "10:-11:10,9,7"   "10:-19:10,9,15"  "10:-27:10,9,15"
       "14:-1:6,10,5"   "14:-3:18,5,6"   "14:-11:17,18,27" "14:-19:17,18,26" "14:-27:26,17,27")
if [ ${#CELLS[@]} -ne 20 ]; then
  echo "FATAL: ${#CELLS[@]} cells, expected 4 layers x 5 offsets = 20" >&2
  exit 1
fi

## ---- decode this task ----
index=$(($SGE_TASK_ID - 1))
CELL=${CELLS[$index]}
IFS=':' read -r LAYER SRC_OFFSET HEAD_CSV <<< "$CELL"
IFS=',' read -r -a HEADS <<< "$HEAD_CSV"
if [ -z "$LAYER" ] || [ -z "$SRC_OFFSET" ] || [ ${#HEADS[@]} -ne 3 ]; then
  echo "FATAL: could not decode cell '$CELL' (3 heads) from SGE_TASK_ID=$SGE_TASK_ID" >&2
  exit 1
fi

# The hand-copied heads must be the picked ones, in rank order.
python -c "
import pandas as pd
p = pd.read_csv('$PICKS')
c = p[(p['layer'] == $LAYER) & (p['src_offset'] == $SRC_OFFSET)].sort_values('rank')
want, got = c['head'].tolist(), [$HEAD_CSV]
if want != got:
    raise SystemExit(f'FATAL: cell L$LAYER @$SRC_OFFSET heads {got} != picks {want} in $PICKS')
"

DST_POS=-1               # destination token: last token of the input text
W=1.0                    # weight on the ENDOGENOUS term; 1.0 => no interventional loss
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
GATE=off                 # 2-1's klnone:off arm
KL_TAG=klnone
OOD_FLAG="--ood-eval"
OVERWRITE=""             # set to --overwrite-cache to force regeneration

## ---- this cell's signal list: its 3 heads at its one offset ----
# Every head of the cell passes the SAME list so all three resolve to one cached dataset;
# --signal-filter is what differs and is not part of the cache identity.
SIGNALS=()
for H in "${HEADS[@]}"; do SIGNALS+=("P${LAYER}.${H}@${SRC_OFFSET}"); done
SIGNAL_LIST=$(IFS=,; echo "${SIGNALS[*]}")

CELL_TAG="l${LAYER}_src${SRC_OFFSET}"
echo
echo "=== PAPER GRID @ ${SAMPLE_TAG} cell (layer, offset) = ($LAYER, $SRC_OFFSET) ==========="
echo "  heads: ${HEADS[*]}   arm: gate=$GATE $KL_TAG   dst=$DST_POS"
echo "  signals: $SIGNAL_LIST"
echo "  samples: train=$TRAIN_SAMPLES dev=$DEV_SAMPLES test=$TEST_SAMPLES seq=$SEQ"
echo "  output: $OUT_ROOT"
echo "=========================================================="

mkdir -p experiment_configs/$EXP_GROUP

for SAMPLE in 0 1 2; do
  SIG=${SIGNALS[$SAMPLE]}
  EXP_NAME="endo_${SIG}_POS${DST_POS}_gate${GATE}_${KL_TAG}_w${W}_${SAMPLE_TAG}"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo
  echo "=== [$CELL_TAG head $SAMPLE] $EXP_NAME ==="

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
      --train-token-selector $DST_POS \
      --test-token-selector $DST_POS \
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

  # Builds the cell's cache on the first head, skips on the next two. Safe only because
  # this task is the cache's sole user (see header).
  python cache_signals.py --config "$CONFIG" $OVERWRITE

  # Cache-alignment check once per cell: it is a property of the cached dataset. For a
  # post-softmax weight it is the only automatic check that the live value equals the
  # cached one; a normalization mismatch otherwise just looks like a low score.
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
  # heads for an error bar, as 2-1 and 7-1 do for this arm.
  if [ "$SAMPLE" -lt 2 ]; then
    python score_lm_degradation.py --config "$CONFIG"
  fi
done

echo
echo "=========================================================="
echo "PAPER GRID @ ${SAMPLE_TAG} CELL ($LAYER, $SRC_OFFSET) DONE. Once all 20 tasks finish:"
echo "  notebook/attention_weight/paper/1_grid_endogenous_analysis.ipynb"
echo "End date : $(date)"
echo "=========================================================="
