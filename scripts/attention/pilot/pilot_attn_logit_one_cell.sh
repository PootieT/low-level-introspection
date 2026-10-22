#!/bin/bash -l
#$ -P multilm
#$ -N attn_pilot
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=4:00:00
# SGE writes one file per task into this DIRECTORY. It must already exist -- if it
# does not, SGE treats the value as a filename and every task dumps into one file
# instead (which is how the stray `log` file in the repo root came to be).
#$ -o logs
#$ -j y

# ============================================================================
# PILOT: one (layer, source offset) cell of the attention-logit sweep, end to end.
#
# Run this BEFORE committing 1280 runs. It answers three questions and prints a
# verdict for each, so nobody has to grep a log to find out whether it worked:
#
#   1. ALIGNMENT (the one that matters). Does the live model reproduce the cached
#      attention logit, at the cached (dst, src)? measure_attention_alignment sweeps
#      the destination offset and exactly +0 must win with corr ~1.0. This is the only
#      automatic check that the new extraction path reads what cache_signals wrote --
#      a mismatch means caching and training ran under different attention
#      implementations, or the (dst, src) pair is off by one.
#   2. INTERVENTION SATURATION. interventional_scale_sweep at several injection scales.
#      Attention is unlike a neuron here: "extreme" draws from scale x the empirical
#      RANGE, so a logit range of ~30 at scale 10 injects +/-150, the softmax saturates,
#      and different injected values give the SAME forward pass -- a value the model
#      cannot report even in principle. Read the printed table to choose a scale for any
#      joint (W<1) follow-up.
#   3. COST. Wall time per caching pass and per training run, extrapolated to a full
#      sweep cell, so h_rt in the sweep script can be sized from a measurement.
#
# NOT A WARM-UP FOR THE SWEEP. The pilot uses TRAIN_SAMPLES=3000 while the sweep uses
# 300000, and max_samples is part of the dataset identity -- so this builds its OWN
# cache directory and the sweep will still cache from scratch. That is deliberate
# (fast feedback). Set TRAIN_SAMPLES=300000 if you want this cell's real cache built
# here instead, and expect it to take as long as a sweep task.
#
# Usage
#   qsub scripts/attention/pilot/pilot_attn_logit_one_cell.sh
#   bash scripts/attention/pilot/pilot_attn_logit_one_cell.sh          # interactive GPU node
#   LAYER=0 SRC_OFFSET=-31 bash scripts/attention/pilot/pilot_attn_logit_one_cell.sh
#
# Every knob below is overridable from the environment (qsub -v LAYER=0,...).
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID"
echo "=========================================================="
nvidia-smi

# Same environment as the sweep script, and deliberately NOT the one the neuron/SAE
# scripts use (they charge llamagrp and activate peter's env). Keep these in sync with
# scripts/attention/train_single_attn_logit_introspector_endogenous_sweep_layer_srctoken.sh
# -- a pilot that passes under a different env has not tested the sweep.
module load cuda/12.8
module load miniconda/23.11.0
conda activate /projectnb/multilm/jongin/anaconda3/envs/introspection

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection
export WANDB_RUN_GROUP=attn_pilot

set -e
# Capture $? on the FIRST line of the trap: every command inside it (the bare `echo`
# included) overwrites $?, so reading it later reports the echo's status and the banner
# printed "exit 0" for a run that had just died.
trap 'rc=$?; echo; echo "############################################################"; \
      echo "### PILOT FAILED (exit $rc) -- see the traceback above."; \
      echo "############################################################"' ERR

## ---- the cell ----
# Defaults picked on purpose:
#   GATE=off       the alignment/drift check auto-runs when the adapter is active over
#                  the input text; --force-drift-eval is passed anyway.
#   SRC_OFFSET=-3  neither 0 (self-attention) nor the BOS sink, so the pipeline is
#                  tested on an ordinary cell without the low-variance confound that
#                  makes a BOS correlation hard to read.
#   LAYER=8        mid-stack. Layer 0 would be the text-only ceiling control.
LAYER=${LAYER:-8}
SRC_OFFSET=${SRC_OFFSET:--3}
GATE=${GATE:-off}
DST_POS=${DST_POS:--1}

HEAD_SELECT=${HEAD_SELECT:-random}          # random | fixed  (matches the sweep's arms)
FIXED_HEADS=(0 4 8 12 16 20 24 28)

## ---- size (small = fast feedback; see "NOT A WARM-UP" above) ----
TRAIN_SAMPLES=${TRAIN_SAMPLES:-3000}
DEV_SAMPLES=${DEV_SAMPLES:-200}
TEST_SAMPLES=${TEST_SAMPLES:-500}
# Dev skips this many RAW documents. Must clear train's raw span (TRAIN_SAMPLES divided
# by the ~99% length-filter pass rate), or cache_signals refuses the config outright.
DEV_OFFSET=${DEV_OFFSET:-20000}

# The cache always holds the full 8 heads (that is what a sweep cell caches, and it is
# one forward pass either way); only the number TRAINED is cut down. 2 is enough to
# check that --signal-filter really splits the shared cache AND that the second run
# reuses it instead of rebuilding.
N_HEADS_CACHE=${N_HEADS_CACHE:-8}
N_HEADS_TRAIN=${N_HEADS_TRAIN:-2}

## ---- everything else mirrors the sweep ----
SEQ=32                   # max AND min: every cached document is exactly this long
W=1.0                    # pure endogenous
SEED=21
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
SCALE=10
SCALE_SWEEP=${SCALE_SWEEP:-1,3,10,30,100}
# A reduced corpus set: enough to exercise score_ood_generalization's derived-config
# path (which had to be fixed to carry min_seq_length through, and would otherwise fail
# for attention signals), without paying for all six corpora in a pilot.
OOD_CORPORA=${OOD_CORPORA:-shuffled,wikipedia}
RUN_OOD=${RUN_OOD:-1}
# Re-running the pilot reuses whatever cache is already on disk. Set to
# --overwrite-cache to force a rebuild (e.g. after changing the extraction code).
CACHE_OVERWRITE=${CACHE_OVERWRITE:-}

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

EXP_GROUP=attn_pilot
CELL="l${LAYER}_src${SRC_OFFSET}"
OUT_ROOT=outputs/runs/$EXP_GROUP

if [ "$HEAD_SELECT" = "fixed" ]; then
  HEAD_JSON=$(IFS=,; echo "[${FIXED_HEADS[*]}]")
  SAMPLER="{\"type\":\"attention_layer_sampler\",\"signal_type\":\"attention_score\",\"layers\":[${LAYER}],\"heads\":${HEAD_JSON},\"src_offsets\":[${SRC_OFFSET}],\"seed\":${SEED}}"
else
  SAMPLER="{\"type\":\"attention_layer_sampler\",\"signal_type\":\"attention_score\",\"layers\":[${LAYER}],\"heads_per_layer\":${N_HEADS_CACHE},\"src_offsets\":[${SRC_OFFSET}],\"seed\":${SEED}}"
fi

echo
echo "=== PILOT CELL ==========================================="
echo "  layer=$LAYER  src_offset=$SRC_OFFSET  dst=$DST_POS  gate=$GATE"
echo "  heads: $HEAD_SELECT, cache $N_HEADS_CACHE / train $N_HEADS_TRAIN"
echo "  samples: train=$TRAIN_SAMPLES dev=$DEV_SAMPLES test=$TEST_SAMPLES  seq=$SEQ"
echo "  output: $OUT_ROOT"
echo "=========================================================="

## ---- resolve this cell's head set ONCE ----
# Generated, not run: it keeps resolution.py the single source of truth for which heads
# a (seed, layer) maps to, instead of duplicating the selection in shell.
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_${CELL}.json
python data_prep/generate_experiment_config.py \
    --experiment-name "_cell_${CELL}" \
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
    --precision 3 \
    --cache-dir data --output-dir $OUT_ROOT \
    --output "$CELL_CONFIG" \
    --overwrite-cache

SIGNALS=($(python -c "
import json
cfg = json.load(open('$CELL_CONFIG'))
print(' '.join(f\"A{s['layer']}.{s['index']}@{s['src_offset']}\" for s in cfg['train_signals']))
"))
SIGNAL_LIST=$(IFS=,; echo "${SIGNALS[*]}")
echo "cell $CELL signals: $SIGNAL_LIST"

CACHE_SECONDS=0
TRAIN_SECONDS=0
LAST_RUN_DIR=""

for SAMPLE in $(seq 0 $((N_HEADS_TRAIN - 1))); do
  SIG=${SIGNALS[$SAMPLE]}
  EXP_NAME="pilot_${SIG}_POS${DST_POS}_gate${GATE}_w${W}_${TRAIN_SAMPLES}"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  LAST_RUN_DIR=$OUT_ROOT/$EXP_NAME
  echo
  echo "=== [$CELL sample $SAMPLE] $EXP_NAME ==="

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
      --precision 3 \
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
      --overwrite-cache

  # Builds the cell's 8-head dataset on SAMPLE=0 and no-ops afterwards. If the second
  # sample re-caches, --signal-filter has leaked into the dataset identity.
  t0=$SECONDS
  python data_prep/cache_signals.py --config "$CONFIG" $CACHE_OVERWRITE
  dt=$((SECONDS - t0))
  if [ "$SAMPLE" -eq 0 ]; then
    CACHE_SECONDS=$dt
    echo "[pilot] caching took ${dt}s"
    # The most dangerous single misconfiguration: a cache built under sdpa would make
    # every attention logit a different quantity from the one training reads.
    python - <<PYEOF
import json, sys
from pathlib import Path
from src.introspector.config import ExperimentConfig
from src.introspector.dataset_utils import canonical_dataset_name
cfg = ExperimentConfig.model_validate_json(Path("$CONFIG").read_text())
meta = json.load(open(Path(cfg.cache_dir) / canonical_dataset_name(cfg) / "metadata.json"))
impl = meta["capture"].get("attn_implementation")
print(f"[pilot] cache capture: {meta['capture']['signal_site']}")
print(f"[pilot] cache attn_implementation: {impl}")
rate = meta["realized"]["train"].get("src_token_unique_rate")
if rate is not None:
    # The prompt names the source token by string only, so a token that occurs more than
    # once in the 32-token context does not identify a position. Not a failure -- an
    # accepted confound -- but the size of it belongs in the pilot log.
    print(f"[pilot] src_token_unique_rate: {rate:.4f} "
          f"({(1 - rate) * 100:.1f}% of rows name an ambiguous source token)")
if impl != "introspection_logit":
    sys.exit(f"FAIL: cache was built with attn_implementation={impl!r}, not "
             f"'introspection_logit'. Every cached logit is the wrong quantity.")
PYEOF
  elif [ "$dt" -gt 60 ]; then
    # Sample 0 built the cell; every later sample must find it and no-op. If it rebuilt,
    # --signal-filter has leaked into the dataset identity -- which would silently turn
    # the sweep's 80 shared caches into 1280 separate ones.
    echo "[pilot] *** WARNING: sample $SAMPLE re-cached (${dt}s), it should have reused"
    echo "[pilot] *** sample 0's dataset. Check that --signal-filter is still excluded"
    echo "[pilot] *** from dataset_identity() in src/introspector/dataset_utils.py."
  else
    echo "[pilot] cache step took ${dt}s (correctly reused sample 0's dataset)"
  fi

  t0=$SECONDS
  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      --force-drift-eval \
      --scale-sweep "$SCALE_SWEEP" \
      $( [ "$RUN_OOD" = "1" ] && echo "--ood-eval --ood-corpora $OOD_CORPORA" )
  dt=$((SECONDS - t0))
  TRAIN_SECONDS=$((TRAIN_SECONDS + dt))
  echo "[pilot] training + eval took ${dt}s"
done

## ---- verdict ----
# The point of a pilot is a decision, not a log. Everything above can succeed while the
# alignment check quietly reports the wrong offset, so read it back and say so plainly.
echo
echo "=========================================================="
echo "=== PILOT VERDICT"
echo "=========================================================="
python - <<PYEOF
import json, sys
from pathlib import Path

run = Path("$LAST_RUN_DIR")
ok = True

drift_path = run / "activation_drift.json"
if not drift_path.exists():
    sys.exit(f"FAIL: {drift_path} missing -- the alignment check did not run. "
             f"With gate={'on' if '$GATE' == 'on' else 'off'} it should have.")
d = json.loads(drift_path.read_text())
best, corr = d["best_alignment_offset"], d["best_alignment_corr"]
print(f"[1] ALIGNMENT  best offset {best:+d}, corr={corr:.6f}, alignment_ok={d['alignment_ok']}")
for row in d["alignment"]:
    print(f"        offset {row['offset']:+d}: corr={row['corr']:.6f}")
if not d["alignment_ok"]:
    ok = False
    print("    *** FAIL. Expected offset +0 with corr > 0.99. Either caching and training")
    print("        ran under different attention implementations (compare")
    print("        capture.attn_implementation in the cache metadata.json against this run),")
    print("        or the (dst, src) pair is off by one.")
else:
    print("    PASS -- the live logit is the quantity the cache holds.")

drift = d["drift_tuned_vs_base"]
print(f"\n[2] DRIFT      tuned vs base @offset 0: corr={drift['corr']:.4f} "
      f"std_ratio={drift['std_ratio']:.3f}")
print("    (gate off lets the adapter move the signal the target was frozen from;")
print("     a low corr here with a high endogenous score means recompute, not readout.)")

endo = run / "endogenous_test_metrics_endogenous.json"
if endo.exists():
    m = json.loads(endo.read_text())
    print(f"\n[3] ENDOGENOUS corr={m.get('corr')}  l2={m.get('l2_dist')}")

sweep = run / "interventional_scale_sweep.csv"
if sweep.exists():
    print(f"\n[4] SATURATION (interventional corr vs injection scale)")
    print("    Pick the largest scale before corr stops improving; past it the softmax")
    print("    saturates and the injected value is unreportable in principle.")
    print("    " + sweep.read_text().strip().replace("\n", "\n    "))
else:
    print(f"\n[4] SATURATION missing ({sweep})")

print()
sys.exit(0 if ok else "PILOT FAILED: do not launch the sweep.")
PYEOF

## ---- cost extrapolation ----
# A sweep task caches 300K examples and trains 8 heads. Scale what we just measured so
# h_rt in the sweep script is set from a measurement rather than from the neuron sweep's.
python - <<PYEOF
cache_s, train_s = $CACHE_SECONDS, $TRAIN_SECONDS
n_trained, train_n = $N_HEADS_TRAIN, $TRAIN_SAMPLES
sweep_n, sweep_heads = 300_000, 8
f = sweep_n / train_n
cache_est = cache_s * f
per_run = train_s / max(n_trained, 1)
# training cost scales with the data; the eval suite is a fixed cost per run, so this
# over-estimates a little -- fine for sizing h_rt, which wants headroom anyway.
train_est = per_run * f * sweep_heads
total = cache_est + train_est
print()
print("=========================================================="
      "\n=== COST (measured here, extrapolated to one sweep task)"
      "\n==========================================================")
print(f"  measured: caching {cache_s}s @ {train_n} samples, "
      f"{per_run:.0f}s per training run")
print(f"  a sweep task ({sweep_n} samples, {sweep_heads} heads) would be roughly:")
print(f"      caching  ~{cache_est/3600:6.2f} h")
print(f"      training ~{train_est/3600:6.2f} h")
print(f"      TOTAL    ~{total/3600:6.2f} h   (sweep script currently sets h_rt=24:00:00)")
if total > 24 * 3600:
    print("  *** over h_rt=24:00:00. Raise it, or split the heads across two array")
    print("      dimensions (320 tasks x 4 heads) in the sweep script.")
PYEOF

echo
echo "=========================================================="
echo "PILOT PASSED. Next:"
echo "  qsub scripts/attention/train_single_attn_logit_introspector_endogenous_sweep_layer_srctoken.sh"
echo "  qsub -v HEAD_SELECT=fixed scripts/attention/train_single_attn_logit_introspector_endogenous_sweep_layer_srctoken.sh"
echo "End date : $(date)"
echo "=========================================================="
