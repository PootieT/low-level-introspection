#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_weight_sweep300K
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=24:00:00
#$ -o logs
#$ -j y
#$ -t 12
# Waits for 2-0's cache build; see the CACHES paragraph below.; full run -t 1-36
#$ -hold_jid attn_w_cache300K

# ============================================================================
# STAGE 3 @ 300K: joint-loss WEIGHT (W) sweep, train 300K / test 30K instead of
# 3_train_attn_weight_joint_weight_sweep.sh's 60K / 3K. The design is identical to that
# script (6 cells x 6 weights x 3 heads, gate off is mandatory, three heads per cell for
# the error bar); read its header for why. Only what differs is described here.
#
# L = W * L_endogenous + (1 - W) * L_interventional. W = 1 is pure endogenous,
# W = 0 pure interventional.
#
# CACHES COME FROM 2-0, and this script refuses to build one. The six W arms of a cell are
# ADJACENT task indices (W innermost), so they run concurrently against ONE cell dataset;
# had they built it themselves they would race into the same directory, and
# cache_signals.py skips on `train/` existing alone, so a loser could read a half-written
# dataset with no error. (The 60K script's comment claims the opposite ordering protects
# it; it does not -- that script was safe only because Stage 2 had built the cache first.)
#
# COST. At 60K the longest task (mid W, which runs both forwards) took ~77 min for 3
# heads. Training grows 5x and the test eval 10x, so ~6-8 h per task against h_rt=24h.
#
# OUTPUT IS SEPARATED FROM 60K BY EXP_GROUP (attn_weight_joint_weight_sweep_300K).
#
# Usage
#   qsub scripts/attention_weight/2-0_cache_attn_weight_300K.sh   # once, if not done yet
#   qsub scripts/attention_weight/3-1_train_attn_weight_joint_weight_sweep_300K.sh
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

module load cuda/12.8
module load miniconda/23.11.0
conda activate /projectnb/multilm/jongin/anaconda3/envs/introspection

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection

set -e

TRAIN_SAMPLES=300000
DEV_SAMPLES=200
DEV_OFFSET=1000000
TEST_SAMPLES=30000
SAMPLE_TAG="$((TRAIN_SAMPLES / 1000))K"

EXP_GROUP=attn_weight_joint_weight_sweep_${SAMPLE_TAG}
export WANDB_RUN_GROUP=$EXP_GROUP
OUT_ROOT=outputs/runs/$EXP_GROUP

## ---- decode (cell, W) from the flat array index ----
# CELL is the outer dimension and W the inner one, so the six W arms of a cell are
# adjacent tasks and DO run concurrently on one cell dataset -- which is why the cache
# must come from 2-0.
index=$(($SGE_TASK_ID - 1))
# The SAME six cells as 2-1, in the same order. They must match exactly: these runs
# reuse 2-0's cached datasets, and any difference here hits the cache guard below.
CELLS=("low_near:2:31:-1"  "mid_near:7:4:-1"   "up_near:11:9:-1" \
       "low_far:2:26:-27"  "mid_far:7:14:-27"  "up_far:11:25:-27")
WS=(0.0 0.01 0.03 0.1 0.3 1.0) #! Weights
CELL=${CELLS[$((index / 6))]}
W=${WS[$((index % 6))]}
IFS=':' read -r BAND LAYER HEAD SRC_OFFSET <<< "$CELL"

if [ -z "$BAND" ] || [ -z "$W" ]; then
  echo "FATAL: could not decode (cell, W) from SGE_TASK_ID=$SGE_TASK_ID" >&2
  exit 1
fi

GATE=off                 # mandatory; see the 60K script's header
DST_POS=-1
SEED=21
N_HEADS_CACHE=8 
N_HEADS_TRAIN=3          # the spread across these 3 is the error bar
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
SEQ=32
PRECISION=6
SCALE=4                  # log-odds units
SCALE_SWEEP=1,2,4,8,16
KL_BUDGET=0.1
OVERWRITE=""

CELL_TAG="l${LAYER}_h${HEAD}_src${SRC_OFFSET}"
echo
echo "=== STAGE 3 @ ${SAMPLE_TAG} (cell=$CELL_TAG band=$BAND, W=$W) ============="
echo "  samples: train=$TRAIN_SAMPLES dev=$DEV_SAMPLES test=$TEST_SAMPLES seq=$SEQ"
echo "  output: $OUT_ROOT"

# Identical head set to 2-1, so the cache is the same directory.
HEADS=($HEAD)
for h in $(seq 0 31); do
  if [ "$h" -ne "$HEAD" ] && [ ${#HEADS[@]} -lt $N_HEADS_CACHE ]; then HEADS+=($h); fi
done
HEAD_JSON=$(IFS=,; echo "[${HEADS[*]}]")
SAMPLER="{\"type\":\"attention_layer_sampler\",\"signal_type\":\"attention_weight\",\"layers\":[${LAYER}],\"heads\":${HEAD_JSON},\"src_offsets\":[${SRC_OFFSET}],\"seed\":${SEED}}"

mkdir -p experiment_configs/$EXP_GROUP
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_${CELL_TAG}.json
python generate_experiment_config.py \
    --experiment-name "_cell_${CELL_TAG}" \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 --train-dataset-split "train" --train-max-samples $TRAIN_SAMPLES \
    --dev-max-samples $DEV_SAMPLES --dev-offset-from-train $DEV_OFFSET \
    --test-dataset allenai/c4 --test-dataset-split "validation" --test-max-samples $TEST_SAMPLES \
    --max-seq-length $SEQ --min-seq-length $SEQ \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "$SAMPLER" --test-signals "$SAMPLER" --same-train-test-signals \
    --train-token-selector $DST_POS --test-token-selector $DST_POS \
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
  EXP_NAME="joint_${SIG}_POS${DST_POS}_gate${GATE}_scale${SCALE}_kl${KL_BUDGET}_w${W}_${SAMPLE_TAG}"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo
  echo "=== [$CELL_TAG W=$W sample $SAMPLE] $EXP_NAME ==="

  python generate_experiment_config.py \
      --experiment-name "$EXP_NAME" \
      --model meta-llama/Llama-3.2-1B \
      --train-dataset allenai/c4 --train-dataset-split "train" \
      --train-max-samples $TRAIN_SAMPLES \
      --dev-max-samples $DEV_SAMPLES --dev-offset-from-train $DEV_OFFSET \
      --test-dataset allenai/c4 --test-dataset-split "validation" \
      --test-max-samples $TEST_SAMPLES \
      --max-seq-length $SEQ --min-seq-length $SEQ \
      --hf-data-kwargs '{"config": "en", "streaming": true}' \
      --train-signals "$SIGNAL_LIST" --test-signals "$SIGNAL_LIST" \
      --signal-filter "$SIG" \
      --train-token-selector $DST_POS --test-token-selector $DST_POS \
      --precision $PRECISION \
      --train-batch-size $BS --eval-batch-size 128 --num-epochs 1 \
      --lora-r $LORA_R --lora-alpha $LORA_ALPHA --lora-dropout $LORA_DO \
      --bf16 --learning-rate $LR \
      --warmup-ratio 0.03 --lr-scheduler-type cosine --save-steps 0 \
      --use-regressor \
      --intervention-signal-distribution extreme \
      --intervention-extreme-scale $SCALE \
      --cache-dir data --output-dir $OUT_ROOT \
      --output "$CONFIG" \
      $OVERWRITE

  # The cache must already exist (2-0). Missing means either 2-0 was not run, or
  # something entered the cache identity that differs from 2-1 -- either way, fail here
  # rather than let six concurrent tasks each build 300K examples into one directory.
  CACHE_DIR=$(python -c "
from pathlib import Path
from src.introspector.config import ExperimentConfig
print(ExperimentConfig.model_validate_json(Path('$CONFIG').read_text()).dataset_cache_dir)
")
  if [ ! -d "$CACHE_DIR/train" ]; then
    echo "FATAL: no cache at $CACHE_DIR -- qsub scripts/attention_weight/2-0_cache_attn_weight_300K.sh first" >&2
    exit 1
  fi
  python cache_signals.py --config "$CONFIG" $OVERWRITE

  if [ "$SAMPLE" -eq 0 ]; then DRIFT="--force-drift-eval"; else DRIFT=""; fi

  python train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      --kl-anchor-target $KL_BUDGET \
      --require-kl-anchor \
      --scale-sweep "$SCALE_SWEEP" \
      $DRIFT \
      $OVERWRITE

  # LM cost as a function of W: a weight that buys reading by damaging the model more is
  # not free, and that is the third panel of analysis/plot_weight_sweep.py.
  python score_lm_degradation.py --config "$CONFIG"
done

echo
echo "=========================================================="
echo "STAGE 3 @ ${SAMPLE_TAG} (cell=$CELL_TAG, W=$W) DONE."
echo "When all 36 tasks finish, pick W from the pareto, set W= in"
echo "  scripts/attention_weight/4-1_train_attn_weight_interventional_picked_cells_300K.sh"
echo "End date : $(date)"
echo "=========================================================="
