#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_paper_grid_interv300K
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=24:00:00
# SGE writes one file per task into this DIRECTORY. It must already exist -- otherwise
# SGE treats the value as a filename and every task dumps into one file.
#$ -o logs
#$ -j y
#$ -t 1-60
# Waits for experiment 1, which builds the cell caches this script reads. If that job has
# already finished (or was never submitted) this starts immediately -- the cache guard
# below is what catches the never-submitted case.
#$ -hold_jid attn_w_paper_grid_endo300K

# ============================================================================
# PAPER EXPERIMENT 2: the experiment-1 grid with INTERVENTIONAL training, W sweep.
#
# Same grid (layers {2 6 10 14} x offsets {-1 -3 -11 -19 -27}), same three hand-picked
# heads per cell as scripts/attention_weight/paper/1_grid_endogenous.sh, trained at
#
#     L = W * L_endogenous + (1 - W) * L_interventional,   W in {0.01, 0.1, 0.3}
#
# (W = 1.0 is experiment 1 itself). 20 cells x 3 W x 3 heads = 180 runs; the analysis
# pairs each run with experiment 1's run at the same (layer, head, offset)
# (notebook/attention_weight/paper/2_grid_interventional_analysis.ipynb).
#
# The recipe is 3-2's unanchored W sweep (gate off, no KL anchor) on experiment 1's data
# settings; read 3-2's header for the W semantics and why there is no --kl-anchor-*.
#
# ONE ARRAY TASK PER (CELL, W), three trainings each. CELL is the outer dimension and W
# the inner one, so the three W arms of a cell are adjacent tasks and run concurrently on
# ONE cell dataset. That is why this script never builds a cache: cache_signals.py skips
# on `train/` existing alone, so concurrent builders could read a half-written dataset
# with no error. The caches are experiment 1's -- every config argument below that enters
# the cache identity is copied from 1_grid_endogenous.sh (W and --ood-eval are trainer
# CLI flags, not config), and a missing cache is FATAL rather than built.
#
# RUN NAMES HAVE NO `_scale4_` TOKEN, unlike 3-2's. RUN_RE in
# analysis/plot_attention_srctoken_sweep.py expects `_gate.._kl.._w..` adjacent; a token in
# between leaves kl and w unmatched (w = NaN) without an error. SCALE is constant across
# both paper experiments, so nothing is lost.
#
# Usage
#   qsub scripts/attention_weight/paper/1_grid_endogenous.sh       # first; builds caches
#   qsub scripts/attention_weight/paper/2_grid_interventional.sh   # holds on experiment 1
#   then run notebook/attention_weight/paper/2_grid_interventional_analysis.ipynb
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

# ---- everything from here to the training loop must match 1_grid_endogenous.sh ----
TRAIN_SAMPLES=300000
DEV_SAMPLES=200
DEV_OFFSET=1000000
TEST_SAMPLES=30000
SAMPLE_TAG="$((TRAIN_SAMPLES / 1000))K"

EXP_GROUP=attn_weight_paper_grid_interv_${SAMPLE_TAG}
export WANDB_RUN_GROUP=$EXP_GROUP
OUT_ROOT=outputs/runs/$EXP_GROUP

## ---- the grid: identical to experiment 1 ----
# layer:src_offset:head,head,head -- heads in pick rank order (highest std first).
PICKS=plots/attn_weight_scan_10K/grid_picks_keep0.5.csv
CELLS=("2:-1:31,30,0"   "2:-3:31,13,0"   "2:-11:13,24,10"  "2:-19:5,19,24"   "2:-27:5,19,24"
       "6:-1:3,0,31"    "6:-3:3,1,31"    "6:-11:31,29,19"  "6:-19:31,29,19"  "6:-27:31,29,19"
       "10:-1:28,8,16"  "10:-3:7,28,29"  "10:-11:10,9,7"   "10:-19:10,9,15"  "10:-27:10,9,15"
       "14:-1:6,10,5"   "14:-3:18,5,6"   "14:-11:17,18,27" "14:-19:17,18,26" "14:-27:26,17,27")
WS=(0.01 0.1 0.3)
if [ ${#CELLS[@]} -ne 20 ] || [ ${#WS[@]} -ne 3 ]; then
  echo "FATAL: ${#CELLS[@]} cells x ${#WS[@]} weights, expected 20 x 3" >&2
  exit 1
fi

## ---- decode (cell, W) from the flat array index ----
index=$(($SGE_TASK_ID - 1))
CELL=${CELLS[$((index / ${#WS[@]}))]}
W=${WS[$((index % ${#WS[@]}))]}
IFS=':' read -r LAYER SRC_OFFSET HEAD_CSV <<< "$CELL"
IFS=',' read -r -a HEADS <<< "$HEAD_CSV"
if [ -z "$LAYER" ] || [ -z "$SRC_OFFSET" ] || [ ${#HEADS[@]} -ne 3 ] || [ -z "$W" ]; then
  echo "FATAL: could not decode (cell '$CELL', W '$W') from SGE_TASK_ID=$SGE_TASK_ID" >&2
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
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
SEQ=32                   # max AND min, as in experiment 1
PRECISION=6
SCALE=4                  # log-odds units
SCALE_SWEEP=1,2,4,8,16
GATE=off                 # not a flag: no --disable-adapter-in-input => UngatedCollator
KL_TAG=klnone            # unanchored: no --kl-anchor-target / --require-kl-anchor
OOD_FLAG="--ood-eval"
OVERWRITE=""             # set to --overwrite-cache to force regeneration

SIGNALS=()
for H in "${HEADS[@]}"; do SIGNALS+=("P${LAYER}.${H}@${SRC_OFFSET}"); done
SIGNAL_LIST=$(IFS=,; echo "${SIGNALS[*]}")

CELL_TAG="l${LAYER}_src${SRC_OFFSET}"
echo
echo "=== PAPER GRID INTERV @ ${SAMPLE_TAG} cell (layer, offset) = ($LAYER, $SRC_OFFSET), W = $W ==="
echo "  heads: ${HEADS[*]}   arm: gate=$GATE $KL_TAG   dst=$DST_POS"
echo "  signals: $SIGNAL_LIST"
echo "  samples: train=$TRAIN_SAMPLES dev=$DEV_SAMPLES test=$TEST_SAMPLES seq=$SEQ"
echo "  output: $OUT_ROOT"
echo "=========================================================="

mkdir -p experiment_configs/$EXP_GROUP

for SAMPLE in 0 1 2; do
  SIG=${SIGNALS[$SAMPLE]}
  EXP_NAME="joint_${SIG}_POS${DST_POS}_gate${GATE}_${KL_TAG}_w${W}_${SAMPLE_TAG}"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo
  echo "=== [$CELL_TAG W=$W head $SAMPLE] $EXP_NAME ==="

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

  # The cache must already exist (experiment 1). Missing means either experiment 1 has not
  # cached this cell, or something that enters the cache identity drifted from
  # 1_grid_endogenous.sh -- either way fail here rather than let the three W tasks of this
  # cell each build 300K examples into one directory.
  CACHE_DIR=$(python -c "
from pathlib import Path
from src.introspector.config import ExperimentConfig
print(ExperimentConfig.model_validate_json(Path('$CONFIG').read_text()).dataset_cache_dir)
")
  if [ ! -d "$CACHE_DIR/train" ]; then
    echo "FATAL: no cache at $CACHE_DIR -- qsub scripts/attention_weight/paper/1_grid_endogenous.sh first" >&2
    exit 1
  fi
  python cache_signals.py --config "$CONFIG" $OVERWRITE

  # Cache-alignment check once per (cell, W): cheap, and it is what tells a normalization
  # mismatch apart from a low score.
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

  # LM cost as a function of W. Unanchored, nothing holds the KL down, so this measures
  # what each W actually costs the model -- score every head for the error bar (as 3-2).
  python score_lm_degradation.py --config "$CONFIG"
done

echo
echo "=========================================================="
echo "PAPER GRID INTERV @ ${SAMPLE_TAG} CELL ($LAYER, $SRC_OFFSET) W=$W DONE. Once all 60 tasks finish:"
echo "  notebook/attention_weight/paper/2_grid_interventional_analysis.ipynb"
echo "End date : $(date)"
echo "=========================================================="
