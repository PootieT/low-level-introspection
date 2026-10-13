# ============================================================================
# Shared body of the L7 H4 @-1 variant scripts (1_, 2_, 4_). NOT a job script: each
# variant sets the knobs below and then `source`s this file, so the three training runs
# cannot drift apart in anything except what they are meant to vary.
#
# Knobs the caller MUST set:
#   VARIANT_TAG     short tag in EXP_GROUP / run names
#   HEAD_ACT        identity | sigmoid                     (--regression-head-activation)
#   TRAIN_DIST      extreme | prob_uniform                 (--intervention-signal-distribution)
#   EVAL_DISTS      comma list for --interv-eval-distributions
#
# Everything else matches Stage 3 (3_train_attn_weight_joint_weight_sweep.sh) exactly, so
# a variant's W row is comparable to Stage 3's L7@-1 h4 row: same cache, same
# hyperparameters, same KL budget, gate off, seed 21. The only differences are the
# knobs above and ONE head (h4) instead of three -- so there is no across-head error bar.
#
# Paths are relative to the repo root, as in every other launch script here.
# SMOKE=1: 2000 train rows, `_smoke` EXP_GROUP, no LM-degradation scoring. Same cache.
# ============================================================================

: "${VARIANT_TAG:?}" "${HEAD_ACT:?}" "${TRAIN_DIST:?}" "${EVAL_DISTS:?}"

module load cuda/12.8
module load miniconda/23.11.0
conda activate /projectnb/multilm/jongin/anaconda3/envs/introspection

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection

set -e

EXP_GROUP=attn_weight_L7H4M1_${VARIANT_TAG}
if [ "${SMOKE:-0}" = "1" ]; then EXP_GROUP=${EXP_GROUP}_smoke; fi
export WANDB_RUN_GROUP=$EXP_GROUP
OUT_ROOT=outputs/runs/$EXP_GROUP

## ---- W from the array index ----
# W = 1 is left out on purpose: it has no interventional term, and the endogenous
# results already exist (Stage 2 / Stage 3).
WS=(0.0 0.01 0.03 0.1 0.3)
W=${WS[$((SGE_TASK_ID - 1))]}
if [ -z "$W" ]; then
  echo "FATAL: SGE_TASK_ID=$SGE_TASK_ID is outside 1-${#WS[@]}" >&2
  exit 1
fi

## ---- the cell: identical to Stage 3's mid_near, so the cache is the same directory ----
LAYER=7; HEAD=4; SRC_OFFSET=-1
DST_POS=-1
GATE=off
SEED=21
N_HEADS_CACHE=8
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
SEQ=32
PRECISION=6
SCALE=4                  # extreme's log-odds width; unused by prob_uniform
SCALE_SWEEP=${SCALE_SWEEP:-1,2,4,8,16}
KL_BUDGET=0.1
TRAIN_SAMPLES=60000      # the CACHE size; SMOKE caps the trainer, not the cache
DEV_SAMPLES=200
DEV_OFFSET=1000000
TEST_SAMPLES=3000
SAMPLE_TAG="$((TRAIN_SAMPLES / 1000))K"

if [ "${SMOKE:-0}" = "1" ]; then SMOKE_FLAG="--max-train-examples 2000"; else SMOKE_FLAG=""; fi

CELL_TAG="l${LAYER}_h${HEAD}_src${SRC_OFFSET}"
echo
echo "=== L7H4M1 variant=$VARIANT_TAG W=$W head=$HEAD_ACT train=$TRAIN_DIST eval=$EVAL_DISTS ==="

# Same head set and order as Stage 2/3 (hand-picked first), so the signal list -- and
# with it the cache identity -- is unchanged.
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
SIG=${SIGNALS[0]}
if [ "$SIG" != "P${LAYER}.${HEAD}@${SRC_OFFSET}" ]; then
  echo "FATAL: first resolved signal is $SIG, expected P${LAYER}.${HEAD}@${SRC_OFFSET}" >&2
  exit 1
fi

EXP_NAME="joint_${SIG}_POS${DST_POS}_gate${GATE}_${VARIANT_TAG}_kl${KL_BUDGET}_w${W}_${SAMPLE_TAG}"
CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
echo "=== [$CELL_TAG W=$W] $EXP_NAME ==="

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
    --intervention-signal-distribution $TRAIN_DIST \
    --intervention-extreme-scale $SCALE \
    --cache-dir data --output-dir $OUT_ROOT \
    --output "$CONFIG" \
    --overwrite-cache      # the CONFIG file only (not the data cache): never reuse a stale one

# Must be a no-op: the cache is Stage 2's. If this starts building 60K examples, a
# field that should not be part of the cache identity has become part of it.
python cache_signals.py --config "$CONFIG"

python train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    --kl-anchor-target $KL_BUDGET \
    --require-kl-anchor \
    --scale-sweep "$SCALE_SWEEP" \
    --interv-eval-distributions "$EVAL_DISTS" \
    --force-drift-eval \
    $SMOKE_FLAG

if [ "${SMOKE:-0}" != "1" ]; then
  python score_lm_degradation.py --config "$CONFIG"
fi

echo
echo "=========================================================="
echo "L7H4M1 variant=$VARIANT_TAG W=$W DONE -> $OUT_ROOT/$EXP_NAME"
echo "End date : $(date)"
echo "=========================================================="
