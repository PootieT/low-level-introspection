#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_paper_reeval_2_emp
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=04:00:00
# SGE writes one file per task into this DIRECTORY. It must already exist -- otherwise
# SGE treats the value as a filename and every task dumps into one file.
#$ -o logs
#$ -j y
#$ -t 1-60

# ============================================================================
# PAPER EXPERIMENT 2, step 0 for the notebook: score experiment 2's runs under
# EMPIRICAL_PROB injection.
#
# Experiment 2 (2_grid_interventional.sh) was only ever scored with gaussian injection
# (log-odds N(mean_logit, std_logit) -> sigmoid). The analysis notebook
# (notebook/attention_weight/paper/2_grid_interventional_analysis.ipynb) also draws the
# grids with empirical_prob injection -- one of the cell's own train-split weights,
# as-is -- so this loads each finished experiment-2 checkpoint and runs ONLY that eval
# (trainers/train_introspector_joint.py --interv-eval-only), writing one new file,
# interventional_test_metrics_empirical_prob.json (+ its predictions csv and scatter),
# into the run directory. Nothing experiment 2 already wrote is touched.
#
# THE PAIRED DIFF STAYS EXACT. The eval reseeds the sampler with config.seed and the train
# values come from the same cell cache in the same order, so an experiment-2 run and the
# experiment-1 run of the same head (scored by 1-0_reeval_1_empirical_prob.sh) inject the
# same value into the same test example; only the adapter differs.
#
# One task per (cell, W), three heads each -- the same decoding as 2_grid_interventional.sh.
# Idempotent: a head whose file exists is skipped.
#
# Usage
#   qsub scripts/attention_weight/paper/2-0_reeval_2_empirical_prob.sh
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

BASE_GROUP=attn_weight_paper_grid_interv_300K   # experiment 2
RUN_ROOT=outputs/runs/$BASE_GROUP
CFG_ROOT=experiment_configs/$BASE_GROUP
DIST=empirical_prob

## ---- the grid: identical to experiments 1 and 2 ----
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

# (cell, W) from the flat index, CELL outer and W inner -- as in 2_grid_interventional.sh
index=$(($SGE_TASK_ID - 1))
CELL=${CELLS[$((index / ${#WS[@]}))]}
W=${WS[$((index % ${#WS[@]}))]}
IFS=':' read -r LAYER SRC_OFFSET HEAD_CSV <<< "$CELL"
IFS=',' read -r -a HEADS <<< "$HEAD_CSV"
if [ -z "$LAYER" ] || [ -z "$SRC_OFFSET" ] || [ ${#HEADS[@]} -ne 3 ] || [ -z "$W" ]; then
  echo "FATAL: could not decode (cell '$CELL', W '$W') from SGE_TASK_ID=$SGE_TASK_ID" >&2
  exit 1
fi

python -c "
import pandas as pd
p = pd.read_csv('$PICKS')
c = p[(p['layer'] == $LAYER) & (p['src_offset'] == $SRC_OFFSET)].sort_values('rank')
want, got = c['head'].tolist(), [$HEAD_CSV]
if want != got:
    raise SystemExit(f'FATAL: cell L$LAYER @$SRC_OFFSET heads {got} != picks {want} in $PICKS')
"

echo "=== REEVAL 2 cell (layer, offset) = ($LAYER, $SRC_OFFSET), W=$W, heads ${HEADS[*]}, dist=$DIST ==="

for H in "${HEADS[@]}"; do
  RUN="joint_P${LAYER}.${H}@${SRC_OFFSET}_POS-1_gateoff_klnone_w${W}_300K"
  CONFIG=$CFG_ROOT/$RUN.json
  RUN_DIR=$RUN_ROOT/$RUN
  echo
  echo "=== $RUN ==="
  if [ -f "$RUN_DIR/interventional_test_metrics_${DIST}.json" ]; then
    echo "  already scored, skipping"
    continue
  fi
  if [ ! -f "$CONFIG" ]; then
    echo "FATAL: no experiment-2 config $CONFIG" >&2
    exit 1
  fi
  # Exactly one checkpoint: experiment 2 saves once per epoch for one epoch. More than one means
  # the run was resumed or rerun, and picking one silently could score the wrong weights.
  CKPTS=("$RUN_DIR"/checkpoint-*)
  if [ ${#CKPTS[@]} -ne 1 ] || [ ! -d "${CKPTS[0]}" ]; then
    echo "FATAL: expected exactly one checkpoint in $RUN_DIR, found: ${CKPTS[*]}" >&2
    exit 1
  fi

  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --interv-eval-only \
      --checkpoint "${CKPTS[0]}" \
      --interv-eval-distributions $DIST
done

echo
echo "=========================================================="
echo "REEVAL 2 CELL ($LAYER, $SRC_OFFSET) W=$W DONE."
echo "End date : $(date)"
echo "=========================================================="
