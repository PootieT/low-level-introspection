#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_interv300K
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=24:00:00
#$ -o logs
#$ -j y
#$ -t 1-12

# ============================================================================
# STAGE 4 @ 300K: interventional training on the six hand-picked cells at the W chosen by
# 3-1, train 300K / test 30K instead of 4_train_attn_weight_interventional_picked_cells.sh's
# 60K / 3K. The design is identical to that script (6 cells x {kl0.1, unanchored}, 3 heads
# each, renormalizing intervention, log-odds sampler, gate off); read its header for why,
# including the CAUTION that the saturated fraction is not reported. Only what differs is
# described here.
#
# W IS SET IN THIS FILE, not passed in. Pick it from 3-1's pareto and edit the `W=` line
# below before submitting.
#
# CACHES COME FROM 2-0. The two regimes of a cell are 6 tasks apart, but SGE starts every
# task it has slots for at once, so that spacing does not keep them off one directory; the
# cache guard below is what does. There is no hold_jid here: this stage is submitted after
# 3-1 has finished, by which time the cache exists.
#
# COST. 3 heads per task, each a joint training (both forwards) plus an OOD-only pass;
# roughly the Stage 3 per-task cost, ~6-8 h against h_rt=24h.
#
# OUTPUT IS SEPARATED FROM 60K BY EXP_GROUP (attn_weight_interv_picked_300K).
#
# Usage
#   (edit W= below)
#   qsub scripts/attention_weight/4-1_train_attn_weight_interventional_picked_cells_300K.sh
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

EXP_GROUP=attn_weight_interv_picked_${SAMPLE_TAG}
export WANDB_RUN_GROUP=$EXP_GROUP
OUT_ROOT=outputs/runs/$EXP_GROUP

## ---- decode (regime, cell) ----
index=$(($SGE_TASK_ID - 1))
# The SAME six cells as 2-1, in the same order -- these reuse 2-0's caches.
CELLS=("low_near:2:31:-1"  "mid_near:7:4:-1"   "up_near:11:9:-1" \
       "low_far:2:26:-27"  "mid_far:7:14:-27"  "up_far:11:25:-27")
REGIMES=("kl0.1" "unanchored")
REGIME=${REGIMES[$((index / 6))]}
CELL=${CELLS[$((index % 6))]}
IFS=':' read -r BAND LAYER HEAD SRC_OFFSET <<< "$CELL"

if [ -z "$REGIME" ] || [ -z "$BAND" ]; then
  echo "FATAL: could not decode (regime, cell) from SGE_TASK_ID=$SGE_TASK_ID" >&2
  exit 1
fi

if [ "$REGIME" = "unanchored" ]; then
  KL_FLAG=""
  KL_TAG="klnone"
else
  KL_BUDGET=${REGIME#kl}
  # --require-kl-anchor: this arm's whole claim is that it held a budget, so a silently
  # dropped anchor must fail the run rather than leave a directory named kl0.1 behind.
  KL_FLAG="--kl-anchor-target $KL_BUDGET --require-kl-anchor"
  KL_TAG="kl${KL_BUDGET}"
fi

# THE OUTPUT OF 3-1. Do not guess it: set it from the 300K pareto before submitting.
W=0.01
GATE=off
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
SCALE=4
SCALE_SWEEP=1,2,4,8,16
OOD_FLAG="--ood-eval"
OVERWRITE=""

CELL_TAG="l${LAYER}_h${HEAD}_src${SRC_OFFSET}"
echo
echo "=== STAGE 4 @ ${SAMPLE_TAG} (cell=$CELL_TAG band=$BAND, regime=$REGIME, W=$W) ==="
echo "  samples: train=$TRAIN_SAMPLES dev=$DEV_SAMPLES test=$TEST_SAMPLES seq=$SEQ"
echo "  output: $OUT_ROOT"

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
  EXP_NAME="interv_${SIG}_POS${DST_POS}_gate${GATE}_scale${SCALE}_${KL_TAG}_w${W}_${SAMPLE_TAG}"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo
  echo "=== [$CELL_TAG $REGIME W=$W sample $SAMPLE] $EXP_NAME ==="

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
      --intervention-signal-distribution gaussian \
      --intervention-extreme-scale $SCALE \
      --cache-dir data --output-dir $OUT_ROOT \
      --output "$CONFIG" \
      $OVERWRITE

  # The cache must already exist (2-0); building it here would race the cell's other
  # regime into one directory. See the header.
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
      --scale-sweep "$SCALE_SWEEP" \
      $KL_FLAG \
      $DRIFT \
      $OOD_FLAG \
      $OVERWRITE

  python train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      $KL_FLAG \
      $OOD_FLAG \
      --ood-only \
      $OVERWRITE

  # The unanchored arm's achieved KL IS the cost-of-access result, so score two heads
  # there for an error bar; the anchored arm only needs a spot check.
  if [ "$REGIME" = "unanchored" ]; then LM_SCORE_N=2; else LM_SCORE_N=1; fi
  if [ "$SAMPLE" -lt "$LM_SCORE_N" ]; then
    python score_lm_degradation.py --config "$CONFIG"
  fi
done

echo
echo "=========================================================="
echo "STAGE 4 @ ${SAMPLE_TAG} (cell=$CELL_TAG, $REGIME) DONE. Next:"
echo "  attribution in dose mode (an injected value now exists):"
echo "  python attribute_attention_weight_readout.py --config <config> --mode dose"
echo "End date : $(date)"
echo "=========================================================="
