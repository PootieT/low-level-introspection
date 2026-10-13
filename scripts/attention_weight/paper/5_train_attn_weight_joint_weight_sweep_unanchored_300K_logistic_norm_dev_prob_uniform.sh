#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_paper5_wsweep_logistic_punif300K
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=24:00:00
# SGE writes one file per task into this DIRECTORY. It must already exist -- otherwise
# SGE treats the value as a filename and every task dumps into one file.
#$ -o logs
#$ -j y
#$ -t 1-36
# Waits for 2-0's cache build; see the CACHES paragraph below.
#$ -hold_jid attn_w_cache300K

# ============================================================================
# PAPER EXPERIMENT 5: the unanchored W sweep of
# ../3-2_train_attn_weight_joint_weight_sweep_unanchored_300K.sh with 2-2's readout and
# injection.
#
#     L = W * L_endogenous + (1 - W) * L_interventional,   W in {0.0 0.01 0.03 0.1 0.3 1.0}
#
# Identical to 3-2 (same 6 cells x 6 W x 3 heads = 108 runs in 36 array tasks, gate off,
# no KL anchor, 300K / 30K, hyperparameters) except for what 2-2 changed:
#
#   readout    --regression-head-activation sigmoid --regression-loss norm_deviance
#              (1-1's logistic head + normalized deviance, on BOTH terms). 3-2 used the
#              default linear regressor.
#   injection  --intervention-signal-distribution prob_uniform: t ~ U(0, 1) in
#              PROBABILITY space, as in
#              2-2_grid_endogenous_logistic_norm_dev_prob_uniform.sh. 3-2 used extreme at
#              scale 4. prob_uniform has no scale (model.py asserts scale == 1), so there
#              is no --intervention-extreme-scale and no `_scale` token in the run name.
#
# RUN NAMES follow 2-2 (`joint_<sig>_POS-1_gateoff_klnone_w<W>_300K`), which RUN_RE in
# analysis/plot_attention_srctoken_sweep.py parses. 3-2's notebook RUN_RE requires a
# `_scale` token, so reusing it on this group needs that token made optional.
#
# EVALUATION, as 2-2: the headline interventional corr stays gaussian@1 in log-odds
# (interventional_test_metrics_gaussian.json) plus 3-2's gaussian scale sweep;
# empirical_prob (the natural weight distribution) is REQUIRED; prob_uniform, the training
# distribution, is scored too.
#
# CACHES COME FROM 2-0, and this script refuses to build one, for the same race reason as
# 3-2 (six W arms of a cell run concurrently against one cell dataset). Neither the
# readout nor the injection distribution enters the cache identity, so 2-0's caches are
# reused as is.
#
# SMOKE=1: 2000 training examples, `_smoke` EXP_GROUP, no LM-degradation scoring.
# This SGE rejects a command-line -t when the script embeds `#$ -t`, so strip it and pipe:
#   sed '/^#\$ -t /d' scripts/attention_weight/paper/5_train_attn_weight_joint_weight_sweep_unanchored_300K_logistic_norm_dev_prob_uniform.sh \
#     | qsub -v SMOKE=1 -t 1
#
# Usage
#   qsub scripts/attention_weight/2-0_cache_attn_weight_300K.sh   # once, if not done yet
#   qsub scripts/attention_weight/paper/5_train_attn_weight_joint_weight_sweep_unanchored_300K_logistic_norm_dev_prob_uniform.sh
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

# ---- everything that enters the cache identity must match 2-0 / 2-1 ----
TRAIN_SAMPLES=300000
DEV_SAMPLES=200
DEV_OFFSET=1000000
TEST_SAMPLES=30000
SAMPLE_TAG="$((TRAIN_SAMPLES / 1000))K"

EXP_GROUP=attn_weight_paper_joint_weight_sweep_unanchored_logistic_ndev_punif_${SAMPLE_TAG}
if [ "${SMOKE:-0}" = "1" ]; then EXP_GROUP=${EXP_GROUP}_smoke; fi
export WANDB_RUN_GROUP=$EXP_GROUP
OUT_ROOT=outputs/runs/$EXP_GROUP

## ---- decode (cell, W) from the flat array index ----
# CELL is the outer dimension and W the inner one, so the six W arms of a cell are
# adjacent tasks and DO run concurrently on one cell dataset -- which is why the cache
# must come from 2-0.
index=$(($SGE_TASK_ID - 1))
# The SAME six cells as 2-1 and 3-2, in the same order. They must match exactly: these
# runs reuse 2-0's cached datasets, and any difference here hits the cache guard below.
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
KL_TAG=klnone            # unanchored: no --kl-anchor-target, see 3-2's header
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
SCALE_SWEEP=1,2,4,8,16   # gaussian eval sweep only; training draws have no scale
OVERWRITE=""
HEAD_ACT=sigmoid         # 1-1 / 2-2's readout head
REG_LOSS=norm_deviance   # 1-1 / 2-2's loss, on both terms
INTERV_DIST=prob_uniform # t ~ U(0, 1), as 2-2
# gaussian = 3-2's headline; empirical_prob = natural weights (required); prob_uniform =
# training's
INTERV_EVALS=gaussian,empirical_prob,prob_uniform
# Fail here, not after the job is queued for hours: empirical_prob must be scored.
if [[ ",$INTERV_EVALS," != *",empirical_prob,"* ]]; then
  echo "FATAL: INTERV_EVALS=$INTERV_EVALS lacks empirical_prob" >&2
  exit 1
fi
# SMOKE caps the trainer, not the cache, so a smoke run reads the real 300K cache
if [ "${SMOKE:-0}" = "1" ]; then SMOKE_FLAG="--max-train-examples 2000"; else SMOKE_FLAG=""; fi

CELL_TAG="l${LAYER}_h${HEAD}_src${SRC_OFFSET}"
echo
echo "=== PAPER 5 @ ${SAMPLE_TAG} (cell=$CELL_TAG band=$BAND, W=$W) ============="
echo "  readout: head=$HEAD_ACT loss=$REG_LOSS   injection: $INTERV_DIST   smoke=${SMOKE:-0}"
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
  EXP_NAME="joint_${SIG}_POS${DST_POS}_gate${GATE}_${KL_TAG}_w${W}_${SAMPLE_TAG}"
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
      --regression-head-activation $HEAD_ACT \
      --regression-loss $REG_LOSS \
      --intervention-signal-distribution $INTERV_DIST \
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

  # No --kl-anchor-target / --require-kl-anchor: this is the unanchored arm.
  python train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      --interv-eval-distributions "$INTERV_EVALS" \
      --scale-sweep "$SCALE_SWEEP" \
      $DRIFT \
      $SMOKE_FLAG \
      $OVERWRITE

  # LM cost as a function of W. Unanchored, nothing holds the KL down, so this measures
  # what each W actually costs the model -- score every head for the error bar.
  if [ "${SMOKE:-0}" != "1" ]; then
    python score_lm_degradation.py --config "$CONFIG"
  fi
done

echo
echo "=========================================================="
echo "PAPER 5 @ ${SAMPLE_TAG} (cell=$CELL_TAG, W=$W) DONE."
echo "Outputs: $OUT_ROOT (compare against 3-2's attn_weight_joint_weight_sweep_unanchored_${SAMPLE_TAG})"
echo "End date : $(date)"
echo "=========================================================="
