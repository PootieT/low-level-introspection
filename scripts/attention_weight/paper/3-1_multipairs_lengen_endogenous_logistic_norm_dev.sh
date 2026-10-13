#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_paper31_multipairs_lengen
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
# Holds on the cache job by NAME (as 7-1 does on 7-0). If no such job is queued, there is
# nothing to hold on and this starts at once -- the cache guard below then refuses to run.
#$ -hold_jid attn_w_paper31_cache
# ============================================================================
# PAPER EXPERIMENT 3-1: MULTIPLE (dst, src) PAIRS + LENGTH GENERALIZATION, endogenous.
#
# Experiments 1/1-1 trained one cell per run: dst = the last token, src = a fixed offset.
# Here ONE run learns every cell of a head: each training document carries one (dst, src)
# pair drawn uniformly from all causal pairs of its n tokens, except (0, 0) (BOS -> BOS,
# always 1.0). One pair per document, so the step count equals the single-cell runs'.
#
#     layer 6, heads 3 31 29          x   n in {16 32 64 128}     = 12 runs (one per task)
#
# Heads: paper 1's layer-6 picks (grid_picks_keep0.5.csv), one per behaviour -- 3 is the
# top pick of the near offsets (-1, -3), 31 is picked in every layer-6 cell, 29 is the
# second pick of the far offsets (-11, -19, -27). Checked against the CSV at start-up.
#
# POSITIONS ARE ABSOLUTE (0 == BOS). Pairs (d, s), 0 <= s <= d.
#
# DOCUMENTS. Every split of every run streams C4 documents of >= 512 tokens and keeps the
# first L of them (right truncation): L = n for train/dev, L = m for test. So all runs see
# one document population, and the test documents are THE SAME at every eval length m --
# by the causal mask the weight at (d, s) depends only on tokens 0..d, so for d < m the
# target of (document, d, s) is the same number at every m. score_length_generalization
# asserts the shared documents (prefix check). Dev skips 2M raw documents: 300K train at
# C4's ~25% pass rate spans ~1.2M; cache_signals._assert_dev_disjoint re-checks it.
#
# EVALUATION. A fixed pair list per eval length (src/introspector/signals/pair_grid.py):
# every pair for m <= 32 (135 / 527), a log grid above (57 / 74 / 93 / 114 pairs at
# m = 64 / 128 / 256 / 512: dst in {2^k-1, 2^k, m-1}, dst-src in {0,1,2,4,...}, + BOS).
# Each pair is evaluated on the same TEST_DOCS documents, so every pair has exactly
# TEST_DOCS samples. The run's own test split is m = n; --length-eval adds every m > n.
# Headline numbers are WITHIN-PAIR (endogenous_test_by_pair*.csv, "pairs" in the metrics
# json): a pooled corr is dominated by between-pair variance (BOS vs the rest) and a
# pair-mean readout already scores ~0.97 on it (src/introspector/training/pair_metrics.py).
#
# INTERVENTIONAL EVAL: gaussian and empirical_prob, both PER ABSOLUTE PAIR, from the
# injection pool -- POOL_DOCS train documents read at 512 tokens at every pair any eval
# length asks for (cache_signals.py --pair-pool). It covers dst >= n, which no n-token
# training cache contains, and uses no test document. No scale sweep.
#
# RECIPE: 1-1's gate-off unanchored arm (klnone:off), logistic head + normalized deviance.
#
# CACHES are built by 3-1-0 (this script with CACHE_ONLY=1): per n, the base cache, every
# length cache and every OOD cache; plus the pool. The three heads of an n share them, so
# building them here would race (cache_signals skips on `train/` existing alone, and a
# loser of the race reads a half-written dataset with no error). This script refuses to
# run without 3-1-0's done-marker for its n.
#
# SMOKE=1: 2000 training documents, 50 test/dev documents, eval lengths up to 64,
# `_smoke` EXP_GROUP, no LM-degradation scoring. Run 3-1-0 with SMOKE=1 first:
#   sed '/^#\$ -t /d' scripts/attention_weight/paper/3-1-0_cache_multipairs_lengen.sh \
#     | qsub -v SMOKE=1 -t 1-3
#   sed -e '/^#\$ -t /d' scripts/attention_weight/paper/3-1_multipairs_lengen_endogenous_logistic_norm_dev.sh \
#     | qsub -v SMOKE=1 -t 1
#
# Usage
#   qsub scripts/attention_weight/paper/3-1-0_cache_multipairs_lengen.sh
#   qsub scripts/attention_weight/paper/3-1_multipairs_lengen_endogenous_logistic_norm_dev.sh
#   python analysis/plot_multipairs_lengen.py --exp-group attn_weight_paper31_multipairs_lengen_endo_logistic_ndev_300K
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

CACHE_ONLY=${CACHE_ONLY:-0}

TRAIN_SAMPLES=300000
DEV_SAMPLES=200
DEV_OFFSET=2000000
TEST_DOCS=2000
FLOOR=512                 # min_seq_length on EVERY split: one document population
if [ "${SMOKE:-0}" = "1" ]; then
  # 2000 train documents span ~8K raw ones; a 2M raw skip would cost ~12 min for nothing
  TRAIN_SAMPLES=2000; DEV_SAMPLES=50; TEST_DOCS=50; DEV_OFFSET=100000
fi
SAMPLE_TAG="$((TRAIN_SAMPLES / 1000))K"

EXP_GROUP=attn_weight_paper31_multipairs_lengen_endo_logistic_ndev_${SAMPLE_TAG}
if [ "${SMOKE:-0}" = "1" ]; then EXP_GROUP=${EXP_GROUP}_smoke; fi
export WANDB_RUN_GROUP=$EXP_GROUP
OUT_ROOT=outputs/runs/$EXP_GROUP
MARKERS=$OUT_ROOT/_cache_markers

## ---- the sweep ----
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

## ---- decode this task ----
index=$(($SGE_TASK_ID - 1))
if [ "$CACHE_ONLY" = "1" ]; then
  # 3-1-0: tasks 1..len(NS) build one n each; task len(NS)+1 builds the injection pool
  if [ $index -lt ${#NS[@]} ]; then
    N=${NS[$index]}; TASK=cache
  elif [ $index -eq ${#NS[@]} ]; then
    N=${NS[0]}; TASK=pool          # any n: the pool does not depend on it
  else
    echo "FATAL: CACHE_ONLY task $SGE_TASK_ID out of range 1..$(( ${#NS[@]} + 1 ))" >&2; exit 1
  fi
  HEAD_IDX=0
else
  N_IDX=$((index / 3)); HEAD_IDX=$((index % 3))
  if [ $N_IDX -ge ${#NS[@]} ]; then
    echo "FATAL: task $SGE_TASK_ID out of range for ${#NS[@]} lengths x 3 heads" >&2; exit 1
  fi
  N=${NS[$N_IDX]}; TASK=train
fi
H=${HEADS[$HEAD_IDX]}

# eval lengths strictly above n; m = n is the run's own test split
EVAL_LENGTHS=()
for M in "${ALL_M[@]}"; do if [ "$M" -gt "$N" ]; then EVAL_LENGTHS+=("$M"); fi; done
EVAL_LENGTHS_CSV=$(IFS=,; echo "${EVAL_LENGTHS[*]}")

W=1.0                    # weight on the ENDOGENOUS term; 1.0 => no interventional loss
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
PRECISION=6
SCALE=4
GATE=off                 # 2-1's klnone:off arm
KL_TAG=klnone
HEAD_ACT=sigmoid
REG_LOSS=norm_deviance
INTERV_DISTS=gaussian,empirical_prob
# eval batch: a 512-token eval document plus prompt is ~16x a training row
EVAL_BS=${EVAL_BS:-64}
if [ "${SMOKE:-0}" = "1" ]; then SMOKE_FLAG="--max-train-examples 2000"; else SMOKE_FLAG=""; fi

SIGNALS=()
for HH in "${HEADS[@]}"; do SIGNALS+=("P${LAYER}.${HH}@pairs"); done
SIGNAL_LIST=$(IFS=,; echo "${SIGNALS[*]}")
SIG="P${LAYER}.${H}@pairs"

EXP_NAME="endo_P${LAYER}.${H}_pairs_n${N}_gate${GATE}_${KL_TAG}_w${W}_${SAMPLE_TAG}"
CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
mkdir -p experiment_configs/$EXP_GROUP $MARKERS

echo
echo "=== PAPER 3-1 @ ${SAMPLE_TAG}: n=$N head=L${LAYER}.${H} task=$TASK smoke=${SMOKE:-0} ==="
echo "  signals: $SIGNAL_LIST   filter: $SIG"
echo "  eval lengths > n: ${EVAL_LENGTHS_CSV:-none}   test docs: $TEST_DOCS"
echo "  output: $OUT_ROOT"
echo "=========================================================="

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
    --intervention-signal-distribution extreme \
    --intervention-extreme-scale $SCALE \
    --cache-dir data \
    --output-dir $OUT_ROOT \
    --output "$CONFIG"

LENGTH_FLAGS=""
if [ -n "$EVAL_LENGTHS_CSV" ]; then LENGTH_FLAGS="--length-eval --eval-lengths $EVAL_LENGTHS_CSV"; fi

## ---- CACHE_ONLY (3-1-0) ----
if [ "$CACHE_ONLY" = "1" ]; then
  if [ "$TASK" = "pool" ]; then
    python cache_signals.py --config "$CONFIG" --pair-pool
    touch $MARKERS/pool.done
  else
    python cache_signals.py --config "$CONFIG"
    # every length and OOD cache this n's runs will ask for, built once here
    python train_introspector_joint.py --config "$CONFIG" --prepare-only \
        $LENGTH_FLAGS --ood-eval
    touch $MARKERS/n${N}.done
  fi
  echo "CACHE_ONLY: $TASK for n=$N done."
  exit 0
fi

## ---- training task ----
for MARK in pool.done n${N}.done; do
  if [ ! -f "$MARKERS/$MARK" ]; then
    echo "FATAL: $MARKERS/$MARK missing -- run scripts/attention_weight/paper/3-1-0_cache_multipairs_lengen.sh first" >&2
    exit 1
  fi
done

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

# LM cost of the unanchored arm, two heads for an error bar (as 1-1)
if [ "$HEAD_IDX" -lt 2 ] && [ "${SMOKE:-0}" != "1" ]; then
  python score_lm_degradation.py --config "$CONFIG"
fi

echo
echo "=========================================================="
echo "PAPER 3-1 n=$N head L${LAYER}.${H} DONE."
echo "End date : $(date)"
echo "=========================================================="
