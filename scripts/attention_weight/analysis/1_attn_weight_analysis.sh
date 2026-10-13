#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_analysis
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=8:00:00
# SGE writes into this DIRECTORY. It must already exist -- otherwise SGE treats the
# value as a filename.
#$ -o logs
#$ -j y

# ============================================================================
# EVERYTHING Stage 2 produced, from one submission.
#
#   qsub scripts/attention_weight/analysis/1_attn_weight_analysis.sh
#
# WHY THIS NEEDS A GPU. Steps 3-6 are pure table work and would run on a login node, but
# step 2 (gradient attribution) is a forward+backward pass through the model and cannot.
# Splitting the script in two would mean running two things and remembering the order, so
# the whole pipeline is one GPU job and the CPU steps ride along; they cost minutes
# against the ~2 h of attribution.
#
# ATTRIBUTION IS THE ONLY EXPENSIVE STEP, and it is idempotent. A cell whose
# attribution_meta.json already exists is skipped, so re-running after a table tweak
# costs minutes, not hours. Delete that file to force a redo.
#
# WHICH CHECKPOINTS GET ATTRIBUTED. One per cell: the gate-off unanchored arm at the
# HAND-PICKED head. Gate off because the adapter has to be able to act at the introspected
# token for there to be a readout path worth tracing; unanchored because it is the
# performance ceiling, so a weak attribution there is about the signal rather than about
# the KL budget. The picked head comes from the `_cell_l<L>_h<H>_src<OFF>.json` filenames
# rather than a list repeated here -- one record of which head was picked, not two.
#
# WHAT THIS DOES *NOT* PRODUCE. The Layer x Head exact-decomposition grid (16x32) and the
# Layer x Position MLP grid from the plan's section 7-2. Those come from
# trace_introspection_circuit.py, which does not run on an attention_weight config: its
# signal_ranges key is a 2-tuple (:430) while attention runs build 3-tuples, and
# SIGNAL_PATH_RESOLVERS (:307-320) raises NotImplementedError for attention. Porting it is
# a separate piece of work.
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID"
echo "=========================================================="
nvidia-smi

# The attention family's environment, deliberately NOT the neuron/SAE one (they charge
# llamagrp and activate peter's env, under cuda/11.8).
module load cuda/12.8
module load miniconda/23.11.0
conda activate /projectnb/multilm/jongin/anaconda3/envs/introspection

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
# Python block-buffers stdout when it goes to a file, which the SGE log is. A step that is
# killed (h_rt, OOM killer) then loses every buffered progress line and prints no
# traceback, so the log shows the job stopping after the model load with no reason given.
# Unbuffered, the last line in the log is the last thing that actually happened.
export PYTHONUNBUFFERED=1

# `conda activate` failing does not stop the script (set -e is not on yet, and it must not
# be: the module/conda shell functions return non-zero on harmless warnings). A silent
# failure leaves the node's system python on PATH, and every step then dies on an import
# or on `from __future__ import annotations`. Check the one thing that matters instead.
ENV_PY=/projectnb/multilm/jongin/anaconda3/envs/introspection/bin/python
if [ "$(command -v python)" != "$ENV_PY" ]; then
  echo "FATAL: python is $(command -v python), expected $ENV_PY (conda activate failed?)" >&2
  exit 1
fi

set -e

# Every step runs through this. A failing step is recorded and the job moves on: the
# tables do not depend on attribution, and attribution of one cell does not depend on
# another. `set -e` stays on for everything else (setup, target resolution).
FAILED=()
run_step() {           # run_step <label> <command...>; returns the command's status
  local label=$1; shift
  local rc=0
  "$@" || rc=$?
  if [ $rc -ne 0 ]; then
    FAILED+=("$label (rc=$rc)")
    echo "*** step failed: $label (rc=$rc)" >&2
  fi
  return 0             # never trips set -e; callers read the outcome from FAILED
}

EXP_GROUP=${EXP_GROUP:-attn_weight_endo_picked}
RUNS=outputs/runs/$EXP_GROUP
CONFIGS=experiment_configs/$EXP_GROUP
PLOTS=outputs/plots/$EXP_GROUP
# One subdirectory per attributed run, written by attribute_attention_weight_readout.py.
ATTR_ROOT=plots/$EXP_GROUP/attribution

# The source offset that IS BOS: -(seq_length - 1), from the Stage 2 cache at seq 32.
# It is NOT the sweep's deepest offset -- that is -27, chosen deliberately to stop short
# of the attention sink -- and deriving it from the data would star an ordinary cell as
# the sink. Passed explicitly so nothing has to guess.
BOS_OFFSET=${BOS_OFFSET:--31}

ATTR_METHOD=${ATTR_METHOD:-ig}   # ig is the published figure; gradxact is ~20x cheaper

if [ ! -d "$RUNS" ]; then
  echo "FATAL: no run group at $RUNS. Run Stage 2 first." >&2
  exit 1
fi

echo
echo "=== STAGE 2 ANALYSIS ======================================"
echo "  runs   : $RUNS   ($(ls -1 $RUNS | wc -l) run directories)"
echo "  configs: $CONFIGS"
echo "  plots  : $PLOTS and plots/$EXP_GROUP"
echo "  attribution method: $ATTR_METHOD"
echo "=========================================================="

## ---- 1. resolve the six attribution targets from the cell configs ----
TARGETS=()
for cell_cfg in $CONFIGS/_cell_*.json; do
  [ -e "$cell_cfg" ] || continue
  base=$(basename "$cell_cfg" .json)              # _cell_l2_h31_src-1
  L=$(echo "$base" | sed -n 's/^_cell_l\([0-9]*\)_h.*/\1/p')
  H=$(echo "$base" | sed -n 's/^_cell_l[0-9]*_h\([0-9]*\)_src.*/\1/p')
  OFF=$(echo "$base" | sed -n 's/^.*_src\(-\?[0-9]*\)$/\1/p')
  # glob on the sample tag rather than assuming 60K: the tag is derived from
  # TRAIN_SAMPLES in the training script and must not be duplicated as a literal here
  match=$(ls $CONFIGS/endo_P${L}.${H}@${OFF}_POS*_gateoff_klnone_*.json 2>/dev/null | head -1)
  if [ -z "$match" ]; then
    echo "WARN: no gate-off unanchored config for cell l${L} h${H} src${OFF}" >&2
    continue
  fi
  TARGETS+=("$match")
done

if [ ${#TARGETS[@]} -eq 0 ]; then
  echo "FATAL: resolved no attribution targets from $CONFIGS/_cell_*.json" >&2
  exit 1
fi
echo
echo "[1] ${#TARGETS[@]} attribution targets:"
printf '      %s\n' "${TARGETS[@]}"

## ---- 2. gradient attribution (GPU) ----
# Each cell goes through run_step: one cell that OOMs or has a bad checkpoint must not
# cost the other five AND the tables, which do not depend on it at all.
#
# FAIL FAST WHEN NOTHING HAS WORKED YET. Job 7605926 died on the same TypeError in all six
# cells -- six model loads to print one traceback six times. If the first cell attempted
# fails before any cell has ever produced output, the code path is broken, not the cell,
# and the other five are skipped. Once one cell has succeeded, a later failure is
# cell-specific (OOM, a bad checkpoint) and the loop keeps going.
N_OK=0
for cfg in "${TARGETS[@]}"; do
  name=$(basename "$cfg" .json)
  attr_dir=$ATTR_ROOT/$name
  if [ -f "$attr_dir/attribution_meta.json" ]; then
    echo "[2] $name: attribution already present, skipping (delete $attr_dir to redo)"
    N_OK=$((N_OK + 1))
    continue
  fi
  echo
  echo "[2] === attribution: $name ==="
  n_failed_before=${#FAILED[@]}
  run_step "$name attribution" \
    python attribute_attention_weight_readout.py --config "$cfg" --attr-method "$ATTR_METHOD"
  if [ ${#FAILED[@]} -eq "$n_failed_before" ]; then
    N_OK=$((N_OK + 1))
  elif [ "$N_OK" -eq 0 ]; then
    echo "*** first attribution failed with nothing succeeded yet: treating the code path" \
         "as broken and skipping the remaining cells (tables still follow)" >&2
    FAILED+=("remaining attribution cells skipped after the first failure")
    break
  fi
done

## ---- 3. attribution figures (CPU) ----
for cfg in "${TARGETS[@]}"; do
  name=$(basename "$cfg" .json)
  attr_dir=$ATTR_ROOT/$name
  [ -f "$attr_dir/attribution_meta.json" ] || continue
  echo
  echo "[3] === attribution figures: $name ==="
  run_step "$name attribution plots" \
    python analysis/plot_attention_weight_attribution.py "$attr_dir"
done

## ---- 4. the headline readout figure: arm x eval ----
# plot_attention_srctoken_sweep.py owns its own run-name regex and parses the `P` code.
# It judges low variance per signal code: absolute std for A (whose std is ~1.6),
# coefficient of variation plus a dead-cell floor for P (whose std is ~1e-4, so the
# absolute rule would flag every cell). A directory holding both codes is split into
# code_A/ and code_P/ subdirectories rather than drawn on shared axes.
echo
echo "[4] === readout by arm (arm_grid_corr.png) ==="
run_step "readout by arm" \
  python analysis/plot_attention_srctoken_sweep.py "$RUNS" --bos-offset "$BOS_OFFSET"

## ---- 5. master tables + scale sweep / OOD / skew figures ----
echo
echo "[5] === master tables and diagnostics ==="
run_step "master tables" \
  python analysis/summarize_attention_weight_stage2.py "$RUNS" --configs-dir "$CONFIGS"

## ---- 6. cost of access: measured KL against readout quality ----
# gate off only: with the gate on the adapter never touches ordinary text, so the KL is
# 0 by construction and score_lm_degradation.py refuses to run at all.
echo
echo "[6] === KL / capability frontier ==="
run_step "KL frontier" \
  python analysis/plot_kl_tradeoff.py --exp-group "$EXP_GROUP" --gate off

## ---- what was produced ----
echo
echo "=========================================================="
echo "STAGE 2 ANALYSIS DONE"
echo "----------------------------------------------------------"
echo "Headline figure:"
echo "  plots/$EXP_GROUP/arm_grid_corr.png      arm x (endogenous, interventional)"
echo "Tables:"
echo "  $PLOTS/stage2_master.csv                one row per run"
echo "  $PLOTS/stage2_by_cell.csv               mean +- SE over heads  <- the paper table"
echo "  $PLOTS/stage2_validity.csv              alignment / low-variance checks"
echo "Diagnostics:"
echo "  $PLOTS/scale_sweep_grid.png             head vs fresh-probe ceiling"
echo "  $PLOTS/inj_std_saturation.png           sigmoid saturation"
echo "  $PLOTS/ood_ladder.png                   off-distribution corr"
echo "  $PLOTS/pearson_vs_spearman.png          how much the right tail carries"
echo "  $PLOTS/kl_frontier_L32.png              cost of access"
echo "Attribution (per cell):"
for cfg in "${TARGETS[@]}"; do
  name=$(basename "$cfg" .json)
  [ -f "$ATTR_ROOT/$name/attribution_meta.json" ] \
    && echo "  $ATTR_ROOT/$name/*.png"
done
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "----------------------------------------------------------"
  echo "*** ${#FAILED[@]} STEP(S) FAILED -- everything not listed here completed:"
  printf '  %s\n' "${FAILED[@]}"
fi
echo "End date : $(date)"
echo "=========================================================="

# Non-zero when anything failed. The steps are isolated so one bad cell cannot cost the
# tables, but the JOB must still report failure: job 7605926 printed "6 STEP(S) FAILED"
# and then exited 0, so qacct and SGE mail recorded a success with no attribution at all.
[ ${#FAILED[@]} -eq 0 ] || exit 1
