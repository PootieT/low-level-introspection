#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_paper421_multipairs_lengen_interv
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=48:00:00
# SGE writes one file per task into this DIRECTORY. It must already exist -- otherwise
# SGE treats the value as a filename and every task dumps into one file.
#$ -o logs
#$ -j y
#$ -t 1-12
# Holds on 3-1's cache job by NAME: this experiment reads 3-1's caches (see CACHES
# below). If no such job is queued this starts at once, and the cache guard below then
# refuses to run if the caches are missing.
#$ -hold_jid attn_w_paper31_cache
# ============================================================================
# PAPER EXPERIMENT 4-2-1: 3-1 (multiple (dst, src) pairs + length generalization) with
# the JOINT loss at W = 0.3 and the injection drawn uniformly over [0, 1].
#
#     L = W * L_endogenous + (1 - W) * L_interventional,   W = 0.3
#
#     layer 6, heads 3 31 29          x   n in {16 32 64 128}     = 12 runs (one per task)
#
# Identical to 3-1_multipairs_lengen_endogenous_logistic_norm_dev.sh (same heads, n grid,
# documents, pair sampling, eval lengths m >= n, readout) except for:
#
#   W          0.3 instead of 1.0, so every training document is also injected at its
#              own random (dst, src) pair.
#   injection  --intervention-signal-distribution prob_uniform: t ~ U(0, 1) in
#              PROBABILITY space, as in 2-2_grid_endogenous_logistic_norm_dev_prob_uniform.sh.
#              No scale (prob_uniform asserts scale == 1).
#   eval       interventional gaussian, empirical_prob (3-1's) + prob_uniform (training's),
#              all per absolute pair from 3-1's injection pool.
#
# RECIPE: 2-2's gate-off unanchored arm (klnone:off), logistic head + normalized deviance
# on BOTH terms.
#
# OFF-GRID TRAINING PAIRS. The injection pool holds the eval grid only (all pairs up to
# m = 32, a log grid above), but at W < 1 each training row is injected at its own random
# pair, which at n = 64/128 is mostly off the grid. train_introspector_joint.py gives
# those cells ranges from the run's own train+dev rows (grid cells keep the pool's, and
# it asserts every test cell is a pool cell). prob_uniform reads no per-cell statistic,
# so the source of those ranges does not change a single training draw.
#
# CACHES are 3-1's, not rebuilt: dataset_identity / pair_pool_identity contain neither W,
# the injection distribution, nor the experiment name, so the config below points at the
# same base, length, OOD and pool caches AS LONG AS every cache-identity argument
# (samples, offsets, floor, selectors, signals, n and m grids) matches 3-1 verbatim. This
# script requires 3-1-0's done-markers AND checks that the generated config's caches
# exist, so a drifted argument fails at start-up instead of building a cache per task.
#
# SMOKE=1: 2000 training documents, 50 test/dev documents, n in {16, 64}, eval lengths up
# to 64, `_smoke` EXP_GROUP, no LM-degradation scoring. Needs 3-1's smoke caches:
#   sed '/^#\$ -t /d' scripts/attention_weight/paper/3-1-0_cache_multipairs_lengen.sh \
#     | qsub -v SMOKE=1 -t 1-3
#   sed '/^#\$ -t /d' scripts/attention_weight/paper/4-2-1_multipairs_lengen_interventional_logistic_norm_dev_prob_uniform.sh \
#     | qsub -v SMOKE=1 -t 4        # n = 64, head 3: off-grid pairs occur
#
# Usage
#   qsub scripts/attention_weight/paper/3-1-0_cache_multipairs_lengen.sh   # if not run yet
#   qsub scripts/attention_weight/paper/4-2-1_multipairs_lengen_interventional_logistic_norm_dev_prob_uniform.sh
#   python analysis/plot_multipairs_lengen.py --exp-group attn_weight_paper421_multipairs_lengen_interv_w0.3_logistic_ndev_punif_300K
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

# The attention family's environment, deliberately NOT the neuron/SAE one.
module load cuda/12.8
module load miniconda/23.11.0
conda activate /projectnb/multilm/jongin/anaconda3/envs/introspection

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection

set -e

# ---- everything that enters the cache identity must match 3-1 ----
TRAIN_SAMPLES=300000
DEV_SAMPLES=200
DEV_OFFSET=2000000
TEST_DOCS=2000
FLOOR=512                 # min_seq_length on EVERY split: one document population
if [ "${SMOKE:-0}" = "1" ]; then
  TRAIN_SAMPLES=2000; DEV_SAMPLES=50; TEST_DOCS=50; DEV_OFFSET=100000
fi
SAMPLE_TAG="$((TRAIN_SAMPLES / 1000))K"

EXP_GROUP=attn_weight_paper421_multipairs_lengen_interv_w0.3_logistic_ndev_punif_${SAMPLE_TAG}
CACHE_GROUP=attn_weight_paper31_multipairs_lengen_endo_logistic_ndev_${SAMPLE_TAG}
if [ "${SMOKE:-0}" = "1" ]; then EXP_GROUP=${EXP_GROUP}_smoke; CACHE_GROUP=${CACHE_GROUP}_smoke; fi
export WANDB_RUN_GROUP=$EXP_GROUP
OUT_ROOT=outputs/runs/$EXP_GROUP
# 3-1-0's done-markers: the caches this experiment reads were built there
MARKERS=outputs/runs/$CACHE_GROUP/_cache_markers

## ---- the sweep: identical to 3-1 ----
LAYER=6
HEADS=(3 31 29)
NS=(16 32 64 128)
ALL_M=(16 32 64 128 256 512)
if [ "${SMOKE:-0}" = "1" ]; then NS=(16 64); ALL_M=(16 32 64); fi
PICKS=plots/attn_weight_scan_10K/grid_picks_keep0.5.csv

# Every head must be one of paper 1's layer-6 picks: a mistyped number would otherwise
# train a different, plausible head without any error.
python -c "
import pandas as pd
p = pd.read_csv('$PICKS')
picked = set(p[p['layer'] == $LAYER]['head'])
bad = [h for h in [${HEADS[0]}, ${HEADS[1]}, ${HEADS[2]}] if h not in picked]
if bad:
    raise SystemExit(f'FATAL: heads {bad} are not layer-$LAYER picks {sorted(picked)} in $PICKS')
"

## ---- decode this task: n outer, head inner ----
index=$(($SGE_TASK_ID - 1))
N_IDX=$((index / 3)); HEAD_IDX=$((index % 3))
if [ $N_IDX -ge ${#NS[@]} ]; then
  echo "FATAL: task $SGE_TASK_ID out of range for ${#NS[@]} lengths x 3 heads" >&2; exit 1
fi
N=${NS[$N_IDX]}
H=${HEADS[$HEAD_IDX]}

# eval lengths strictly above n; m = n is the run's own test split
EVAL_LENGTHS=()
for M in "${ALL_M[@]}"; do if [ "$M" -gt "$N" ]; then EVAL_LENGTHS+=("$M"); fi; done
EVAL_LENGTHS_CSV=$(IFS=,; echo "${EVAL_LENGTHS[*]}")

W=0.3                    # weight on the ENDOGENOUS term; 1 - W on the interventional
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
PRECISION=6
GATE=off                 # not a flag: no --disable-adapter-in-input => UngatedCollator
KL_TAG=klnone            # unanchored: no --kl-anchor-target / --require-kl-anchor
HEAD_ACT=sigmoid
REG_LOSS=norm_deviance   # on both terms
INTERV_DIST=prob_uniform # t ~ U(0, 1), as in 2-2
# gaussian, empirical_prob = 3-1's (pair with its endogenous runs); prob_uniform = training's
INTERV_DISTS=gaussian,empirical_prob,prob_uniform
# eval batch: a 512-token eval document plus prompt is ~16x a training row
EVAL_BS=${EVAL_BS:-64}
if [ "${SMOKE:-0}" = "1" ]; then SMOKE_FLAG="--max-train-examples 2000"; else SMOKE_FLAG=""; fi

SIGNALS=()
for HH in "${HEADS[@]}"; do SIGNALS+=("P${LAYER}.${HH}@pairs"); done
SIGNAL_LIST=$(IFS=,; echo "${SIGNALS[*]}")
SIG="P${LAYER}.${H}@pairs"

EXP_NAME="joint_P${LAYER}.${H}_pairs_n${N}_gate${GATE}_${KL_TAG}_w${W}_${SAMPLE_TAG}"
CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
mkdir -p experiment_configs/$EXP_GROUP

echo
echo "=== PAPER 4-2-1 @ ${SAMPLE_TAG}: n=$N head=L${LAYER}.${H} W=$W smoke=${SMOKE:-0} ==="
echo "  arm: gate=$GATE $KL_TAG   readout: head=$HEAD_ACT loss=$REG_LOSS   injection: $INTERV_DIST"
echo "  signals: $SIGNAL_LIST   filter: $SIG"
echo "  eval lengths > n: ${EVAL_LENGTHS_CSV:-none}   test docs: $TEST_DOCS"
echo "  caches: 3-1's ($MARKERS)"
echo "  output: $OUT_ROOT"
echo "=========================================================="

for MARK in pool.done n${N}.done; do
  if [ ! -f "$MARKERS/$MARK" ]; then
    echo "FATAL: $MARKERS/$MARK missing -- run scripts/attention_weight/paper/3-1-0_cache_multipairs_lengen.sh first" >&2
    exit 1
  fi
done

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
    --test-max-samples $TEST_DOCS \
    --max-seq-length $N --min-seq-length $FLOOR \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "$SIGNAL_LIST" \
    --test-signals "$SIGNAL_LIST" \
    --signal-filter "$SIG" \
    --train-token-selector random_pair \
    --test-token-selector "pair_grid:$N" \
    --precision $PRECISION \
    --train-batch-size $BS \
    --eval-batch-size $EVAL_BS \
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
    --output "$CONFIG"

# The markers say 3-1-0 finished; this says THIS config resolves to what it built. A
# miss means a cache-identity argument above drifted from 3-1 -- fail here rather than
# let train_introspector_joint build a cache per task (three heads would race on one).
python -c "
from pathlib import Path
from src.introspector.config import ExperimentConfig
from src.introspector.dataset_utils import pair_pool_dir
c = ExperimentConfig.model_validate_json(Path('$CONFIG').read_text())
missing = [str(p) for p in (c.dataset_cache_dir / 'train', pair_pool_dir(c)) if not p.exists()]
if missing:
    raise SystemExit(f'FATAL: no 3-1 cache at {missing} -- a cache-identity argument differs from 3-1')
"

LENGTH_FLAGS=""
if [ -n "$EVAL_LENGTHS_CSV" ]; then LENGTH_FLAGS="--length-eval --eval-lengths $EVAL_LENGTHS_CSV"; fi

# Cache-alignment check once per n: it is a property of the cached dataset.
if [ "$HEAD_IDX" -eq 0 ]; then DRIFT="--force-drift-eval"; else DRIFT=""; fi

python train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    --scale-sweep "" \
    --interv-eval-distributions $INTERV_DISTS \
    $LENGTH_FLAGS \
    --ood-eval \
    $DRIFT \
    $SMOKE_FLAG

# LM cost of the unanchored arm, two heads for an error bar (as 3-1)
if [ "$HEAD_IDX" -lt 2 ] && [ "${SMOKE:-0}" != "1" ]; then
  python score_lm_degradation.py --config "$CONFIG"
fi

echo
echo "=========================================================="
echo "PAPER 4-2-1 n=$N head L${LAYER}.${H} W=$W DONE."
echo "End date : $(date)"
echo "=========================================================="
