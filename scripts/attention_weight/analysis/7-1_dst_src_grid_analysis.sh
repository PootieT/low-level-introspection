#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_dstsrc_analysis
#$ -pe omp 4
#$ -l h_rt=01:00:00
# SGE writes into this DIRECTORY. It must already exist -- otherwise SGE treats the
# value as a filename.
#$ -o logs
#$ -j y

# ============================================================================
# STAGE 7 ANALYSIS: the (destination, source) pair grid of 7-1.
#
#   bash scripts/attention_weight/analysis/7-1_dst_src_grid_analysis.sh     # login node is fine
#   qsub scripts/attention_weight/analysis/7-1_dst_src_grid_analysis.sh     # or as a CPU job
#
# CPU ONLY. Everything here reads metrics files the training runs already wrote; there is
# no attribution step as in 1_attn_weight_analysis.sh, so no GPU is requested.
#
# SAFE ON A PARTIAL SWEEP. analysis/plot_attention_dst_src_grid.py prints every pair with
# no run or fewer than 3 heads and marks them in the figures ("--" / "n=2"), so this can be
# run while 7-1 is still filling in; rerun it once all 54 tasks are done.
#
# WHY NOT plot_attention_srctoken_sweep.py / summarize_attention_weight_stage2.py as well.
# Both assume the destination is pinned: the first groups by (layer, src_offset) and would
# pool nine different destinations into one point per relative offset, and the second
# reads the `_cell_*.json` picked-head configs that 7-1 does not write.
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID"
echo "=========================================================="

# The attention family's environment, deliberately NOT the neuron/SAE one.
module load miniconda/23.11.0
conda activate /projectnb/multilm/jongin/anaconda3/envs/introspection
export PYTHONUNBUFFERED=1

# `conda activate` failing does not stop the script (set -e is not on yet). A silent
# failure leaves the system python on PATH; check the one thing that matters instead.
ENV_PY=/projectnb/multilm/jongin/anaconda3/envs/introspection/bin/python
if [ "$(command -v python)" != "$ENV_PY" ]; then
  echo "FATAL: python is $(command -v python), expected $ENV_PY (conda activate failed?)" >&2
  exit 1
fi

set -e

# Must match EXP_GROUP in scripts/attention_weight/7-1_train_dst_src_grid_endogenous_300K.sh.
EXP_GROUP=${EXP_GROUP:-attn_weight_dst_src_grid_300K}
RUNS=outputs/runs/$EXP_GROUP
PLOTS=outputs/plots/$EXP_GROUP
EXPECTED_RUNS=162        # 54 pairs x 3 heads

if [ ! -d "$RUNS" ]; then
  echo "FATAL: no run group at $RUNS. Run 7-0 and 7-1 first." >&2
  exit 1
fi

# A run directory is created before training starts, so the directory count over-reports
# progress; count finished runs by their headline metrics file instead.
N_DONE=$(ls $RUNS/*/endogenous_test_metrics_endogenous.json 2>/dev/null | wc -l)

echo
echo "=== STAGE 7 ANALYSIS ======================================"
echo "  runs : $RUNS   ($N_DONE / $EXPECTED_RUNS finished runs)"
echo "  plots: $PLOTS"
echo "=========================================================="
if [ "$N_DONE" -lt "$EXPECTED_RUNS" ]; then
  echo "NOTE: partial sweep -- the figures mark the missing cells; rerun when 7-1 is done."
fi

python analysis/plot_attention_dst_src_grid.py "$RUNS" --out "$PLOTS"

echo
echo "=========================================================="
echo "STAGE 7 ANALYSIS DONE: $PLOTS"
ls -1 "$PLOTS" | sed 's/^/  /'
echo "End date : $(date)"
echo "=========================================================="
