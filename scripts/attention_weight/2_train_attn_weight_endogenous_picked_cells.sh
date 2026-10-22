#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_endo_picked
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=24:00:00
# SGE writes one file per task into this DIRECTORY. It must already exist -- otherwise
# SGE treats the value as a filename and every task dumps into one file.
#$ -o logs
#$ -j y
#$ -t 1-6

# ============================================================================
# STAGE 2: endogenous training on the six hand-picked (layer, head, offset) cells.
#
# A 3 x 2 design, one array task per cell:
#     {lower, mid, upper layer} x {source NEAR the destination, source FAR from it}
# picked by hand from mechanistic_analysis/scan_attention_weight_spread.py's tables (Stage 1). This is NOT a
# sweep -- the attention-LOGIT sweep already ran 1280 cells and came out flat (mean corr
# 0.894, no variation across layer or offset), so this experiment measures every cell
# first and trains a few good ones instead of sampling at random.
#
# THREE ARMS PER CELL:
#   1  gate off, --kl-anchor-target 0.1   "at equal, small behavioural cost, how well
#                                          can the model read itself?"
#   2  gate off, unanchored               the performance ceiling, plus -- via
#                                          score_lm_degradation -- the KL this recipe
#                                          pays when nothing constrains it, which is the
#                                          cost-of-access measurement
#   3  gate on,  no KL                     the main arm of the logit experiment: the
#                                          adapter cannot touch the input text
#
# THERE IS NO "gate on + KL" ARM, on purpose. With the gate on, GatedLinear zeroes the
# LoRA delta over the input text, so the tuned and base next-token distributions are
# identical and KL is 0 BY CONSTRUCTION. Worse, the anchor would not even be applied:
# LoRAGatingCollator leaves the mask as `lora_token_mask`, compute_loss finds no
# `anchor_region_mask`, and the KL term is dropped silently -- leaving a directory named
# _kl0.1_ that never had a KL term. The arms below pass --require-kl-anchor so that
# combination fails loudly rather than producing that artifact.
#
# WHY 8 HEADS ARE CACHED AND 3 TRAINED. One cell gives no within-cell replicate, and the
# neuron weight sweep found run-to-run spread of about +-0.1 -- big enough to swamp any
# difference worth reading. Caching 8 heads is one forward pass either way, and
# --signal-filter is not part of the cache identity, so 3 runs share one dataset and
# their spread becomes the error bar. Stage 3 reuses these same caches.
#
# Three and not five, because this experiment went from 3 cells to 6: at 5 heads a task
# is 15 trainings (~15 h) and h_rt=24:00:00 stops having headroom. The price is honest and
# worth stating -- the standard error on a cell mean widens by sqrt(5/3) ~ 1.29x, so a
# "the cells agree within error" reading has to be weighed against n=3, not waved through.
# Below 3 the LM_SCORE_N=2 spot check further down breaks silently.
#
# COST. 3 arms x 3 heads = 9 trainings per task. At 60K samples that is roughly 1 h per
# training plus ~20 min of caching, so ~9.5 h against h_rt=24:00:00 -- enough headroom,
# but not so much that N_HEADS_TRAIN or TRAIN_SAMPLES can be raised without re-checking.
#
# Usage
#   qsub scripts/attention_weight/2_train_attn_weight_endogenous_picked_cells.sh
#   qsub -t 1-3 ...      # the NEAR arm only
#   qsub -t 4-6 ...      # the FAR arm only
#   SGE_TASK_ID=1 bash scripts/attention_weight/2_train_attn_weight_endogenous_picked_cells.sh
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

EXP_GROUP=attn_weight_endo_picked
export WANDB_RUN_GROUP=$EXP_GROUP
OUT_ROOT=outputs/runs/$EXP_GROUP

## ---- the six hand-picked cells: {lower, mid, upper} x {near, far} ----
# Hand-picked from mechanistic_analysis/scan_attention_weight_spread.py's tables (run 1_scan.sh first).
# Format: band:layer:head:src_offset, ordered NEAR first then FAR so that
# `-t 1-3` alone runs the near arm and `-t 4-6` the far one.
#
# Layer bands: lower 0-4, mid 5-10, upper 11-15.
# The lower and upper layers match the neuron experiment's choices (layer 2 and layer 10)
# so the two signal families can be read against each other; the neuron grid's positions
# -1 and -28 are what "near" and "far" mirror here.
#
# WHY THE SECOND AXIS. The attention-logit sweep varied the source offset across ten
# values and came out flat, so offset is not treated as a sweep axis any more -- it is
# two contrasting conditions. -1 is the immediately preceding token; -27 is far back but
# deliberately NOT the deepest offset, because at seq length 32 the deepest (-31) is BOS,
# a different regime whose mean weight is ~0.65 against ~0.005 elsewhere.
CELLS=(${CELLS:-"low_near:2:31:-1"  "mid_near:7:4:-1"   "up_near:11:9:-1" \
                "low_far:2:26:-27"  "mid_far:7:14:-27"  "up_far:11:25:-27"})
CELL=${CELLS[$((SGE_TASK_ID - 1))]}
IFS=':' read -r BAND LAYER HEAD SRC_OFFSET <<< "$CELL"

if [ -z "$BAND" ] || [ -z "$LAYER" ] || [ -z "$HEAD" ] || [ -z "$SRC_OFFSET" ]; then
  echo "FATAL: could not decode cell '$CELL' from SGE_TASK_ID=$SGE_TASK_ID" >&2
  exit 1
fi

DST_POS=-1               # destination token: last token of the input text
W=1.0                    # weight on the ENDOGENOUS term; 1.0 => no interventional loss
SEED=21
N_HEADS_CACHE=8          # one forward pass regardless; Stage 3 reuses this cache
N_HEADS_TRAIN=${N_HEADS_TRAIN:-3}   # the spread across these 3 is the error bar
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
SCALE_SWEEP=${SCALE_SWEEP:-1,2,4,8,16}
KL_BUDGET=0.1
OOD_FLAG="--ood-eval"
OVERWRITE=""             # set to --overwrite-cache to force regeneration

TRAIN_SAMPLES=${TRAIN_SAMPLES:-60000}
DEV_SAMPLES=200
# Dev skips this many RAW documents, so it must clear train's raw span. 60K train at a
# ~99% length-filter pass rate spans ~61K raw documents, so 1M leaves a wide margin and
# cache_signals' disjointness check passes regardless of TRAIN_SAMPLES.
DEV_OFFSET=1000000
TEST_SAMPLES=3000
# Derived, never hardcoded: the run directory name is the analysis scripts' only record
# of how much data a run saw, and a stale "300K" on a 60K run is not something any later
# reader can catch.
SAMPLE_TAG="$((TRAIN_SAMPLES / 1000))K"

echo
echo "=== STAGE 2 CELL =========================================="
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

## ---- the three arms ----
# ARM format: name:gate:kl_flag_or_none
ARMS=("kl${KL_BUDGET}:off:--kl-anchor-target ${KL_BUDGET} --require-kl-anchor"
      "klnone:off:"
      "klnone:on:")

for ARM in "${ARMS[@]}"; do
  IFS=':' read -r KL_TAG GATE KL_FLAG <<< "$ARM"
  if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

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

    # Builds the cell's 8-head dataset the first time and no-ops afterwards, including
    # across arms (the arms differ only in trainer CLI flags, which are not part of the
    # dataset identity).
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
done

echo
echo "=========================================================="
echo "STAGE 2 CELL $CELL_TAG DONE. Next:"
echo "  qsub scripts/attention_weight/analysis/2_attn_weight_analysis_endo_60K.sh"
echo "  qsub scripts/attention_weight/3_train_attn_weight_joint_weight_sweep.sh"
echo "End date : $(date)"
echo "=========================================================="
