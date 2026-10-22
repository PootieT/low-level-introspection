#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_paper_grid_interv_logistic_punif300K
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
# Waits for experiment 1-1, the endogenous baseline this experiment is paired with. If
# that job has already finished (or was never submitted) this starts immediately -- the
# cache guard below is what catches a missing cache.
#$ -hold_jid attn_w_paper_grid_endo_logistic300K

# ============================================================================
# PAPER EXPERIMENT 2-2: 2-1 with the injection drawn UNIFORMLY over [0, 1].
#
# Identical to 2-1_grid_endogenous_logistic_norm_dev_prob_true_dist.sh (same grid, heads,
# data, 1-1's readout, joint loss and W sweep),
#
#     L = W * L_endogenous + (1 - W) * L_interventional,   W in {0.01, 0.1, 0.3}
#
# except for the training injection:
#
#   injection  --intervention-signal-distribution prob_uniform: each injected value is
#              t ~ U(0, 1) in PROBABILITY space (prob_uniform in
#              make_bounded_prob_intervention_fn). 2-1 bootstrapped the cell's own train
#              weights, so its targets carried the cell's skew (mostly near 0 at most
#              offsets); here every cell gets the same flat target distribution, which
#              asks whether reading the weight needs the natural distribution at all.
#              Draws outside the 1e-4 guard band are clamped (<= 2e-4 of draws, counted
#              in the sampler's n_clamped) -- kept so prob_uniform stays the same
#              distribution L7H4M1/2_prob_uniform_injection.sh trained on.
#
# The baseline ("endogenous training") is still 1-1 (W = 1.0, same readout). 20 cells x
# 3 W x 3 heads = 180 runs, in 60 array tasks of three heads each.
#
# EVALUATION. The headline interventional corr stays gaussian@1 in log-odds
# (interventional_test_metrics_gaussian.json), exactly as 1-1 and 2-1 were scored.
# empirical_prob (interventional_test_metrics_empirical_prob.json) is REQUIRED: it is the
# natural weight distribution, the one the endogenous readout must handle, and the one
# that pairs with 2-1's training distribution; 1-1's runs get that file from
# 2-1-0_reeval_1-1_empirical_prob.sh. prob_uniform, the training distribution, is scored
# too (interventional_test_metrics_prob_uniform.json).
#
# ONE ARRAY TASK PER (CELL, W), CELL outer and W inner, and this script never builds a
# cache -- see 2's header. The caches are experiment 1's (1-1 reads the same ones: the
# cache identity depends on neither the readout nor the injection distribution), and a
# missing cache is FATAL rather than built.
#
# RUN NAMES HAVE NO EXTRA TOKEN: RUN_RE in plotters/plot_attention_srctoken_sweep.py
# expects `_gate.._kl.._w..` adjacent (see 2's header). The readout and the injection
# distribution are checked from the configs by the analysis notebook instead.
#
# SMOKE=1: 2000 training examples, `_smoke` EXP_GROUP, no LM-degradation scoring.
# This SGE rejects a command-line -t when the script embeds `#$ -t`, so strip it and pipe:
#   sed '/^#\$ -t /d' scripts/attention_weight/paper/2-2_grid_endogenous_logistic_norm_dev_prob_uniform.sh \
#     | qsub -v SMOKE=1 -t 1
#
# Usage
#   qsub scripts/attention_weight/paper/2-1-0_reeval_1-1_empirical_prob.sh   # baseline, any time
#   qsub scripts/attention_weight/paper/2-2_grid_endogenous_logistic_norm_dev_prob_uniform.sh
#   then run notebook/attention_weight/paper/2-2_grid_endogenous_logistic_norm_dev_prob_uniform_analysis.ipynb
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

# ---- everything that enters the cache identity must match 1_grid_endogenous.sh ----
TRAIN_SAMPLES=300000
DEV_SAMPLES=200
DEV_OFFSET=1000000
TEST_SAMPLES=30000
SAMPLE_TAG="$((TRAIN_SAMPLES / 1000))K"

EXP_GROUP=attn_weight_paper_grid_interv_logistic_ndev_punif_${SAMPLE_TAG}
if [ "${SMOKE:-0}" = "1" ]; then EXP_GROUP=${EXP_GROUP}_smoke; fi
export WANDB_RUN_GROUP=$EXP_GROUP
OUT_ROOT=outputs/runs/$EXP_GROUP

## ---- the grid: identical to experiments 1, 1-1 and 2 ----
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
SCALE_SWEEP=1,2,4,8,16   # gaussian eval sweep only; training draws have no scale
GATE=off                 # not a flag: no --disable-adapter-in-input => UngatedCollator
KL_TAG=klnone            # unanchored: no --kl-anchor-target / --require-kl-anchor
OOD_FLAG="--ood-eval"
OVERWRITE=""             # set to --overwrite-cache to force regeneration
HEAD_ACT=sigmoid         # 1-1's readout head
REG_LOSS=norm_deviance   # 1-1's loss, now on both terms
INTERV_DIST=prob_uniform # t ~ U(0, 1); the one change from 2-1
# gaussian = 1-1's headline; empirical_prob = natural weights (required, pairs with 1-1
# and 2-1); prob_uniform = training's
INTERV_EVALS=gaussian,empirical_prob,prob_uniform
# Fail here, not after the job is queued for hours: empirical_prob must be scored.
if [[ ",$INTERV_EVALS," != *",empirical_prob,"* ]]; then
  echo "FATAL: INTERV_EVALS=$INTERV_EVALS lacks empirical_prob" >&2
  exit 1
fi
# SMOKE caps the trainer, not the cache, so a smoke run reads the real 300K cache
if [ "${SMOKE:-0}" = "1" ]; then SMOKE_FLAG="--max-train-examples 2000"; else SMOKE_FLAG=""; fi

SIGNALS=()
for H in "${HEADS[@]}"; do SIGNALS+=("P${LAYER}.${H}@${SRC_OFFSET}"); done
SIGNAL_LIST=$(IFS=,; echo "${SIGNALS[*]}")

CELL_TAG="l${LAYER}_src${SRC_OFFSET}"
echo
echo "=== PAPER GRID 2-2 @ ${SAMPLE_TAG} cell (layer, offset) = ($LAYER, $SRC_OFFSET), W = $W ==="
echo "  heads: ${HEADS[*]}   arm: gate=$GATE $KL_TAG   dst=$DST_POS"
echo "  readout: head=$HEAD_ACT loss=$REG_LOSS   injection: $INTERV_DIST   smoke=${SMOKE:-0}"
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
      --bf16 \
      --learning-rate $LR \
      --warmup-ratio 0.03 \
      --lr-scheduler-type cosine \
      --save-steps 0 \
      --use-regressor \
      --regression-head-activation $HEAD_ACT \
      --regression-loss $REG_LOSS \
      --intervention-signal-distribution $INTERV_DIST \
      --cache-dir data \
      --output-dir $OUT_ROOT \
      --output "$CONFIG" \
      $OVERWRITE

  # The cache must already exist (experiment 1). Missing means either it has not cached
  # this cell, or something that enters the cache identity drifted from
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
  python data_prep/cache_signals.py --config "$CONFIG" $OVERWRITE

  # Cache-alignment check once per (cell, W): cheap, and it is what tells a normalization
  # mismatch apart from a low score.
  if [ "$SAMPLE" -eq 0 ]; then DRIFT="--force-drift-eval"; else DRIFT=""; fi

  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      --interv-eval-distributions "$INTERV_EVALS" \
      --scale-sweep "$SCALE_SWEEP" \
      $DRIFT \
      $SMOKE_FLAG \
      $OOD_FLAG \
      $OVERWRITE

  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      $OOD_FLAG \
      --ood-only \
      $SMOKE_FLAG \
      $OVERWRITE

  # LM cost as a function of W. Unanchored, so this is what each W actually costs the
  # model -- score every head for the error bar, as 2 does.
  if [ "${SMOKE:-0}" != "1" ]; then
    python evals/score_lm_degradation.py --config "$CONFIG"
  fi
done

echo
echo "=========================================================="
echo "PAPER GRID 2-2 @ ${SAMPLE_TAG} CELL ($LAYER, $SRC_OFFSET) W=$W DONE. Once all 60 tasks finish:"
echo "  notebook/attention_weight/paper/2-2_grid_endogenous_logistic_norm_dev_prob_uniform_analysis.ipynb"
echo "End date : $(date)"
echo "=========================================================="
