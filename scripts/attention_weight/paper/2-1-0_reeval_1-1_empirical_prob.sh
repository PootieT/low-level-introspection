#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_paper_reeval_1-1_emp
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=04:00:00
# SGE writes one file per task into this DIRECTORY. It must already exist -- otherwise
# SGE treats the value as a filename and every task dumps into one file.
#$ -o logs
#$ -j y
#$ -t 1-20

# ============================================================================
# PAPER EXPERIMENT 2-1, step 0: score 1-1's runs under EMPIRICAL_PROB injection.
#
# 2-1 trains on empirical_prob draws and scores both gaussian (1-1's headline) and
# empirical_prob. 1-1 was only ever scored with gaussian, so its runs have no
# empirical_prob number to pair with. This loads each finished 1-1 checkpoint and runs
# ONLY that eval (train_introspector_joint.py --interv-eval-only), writing one new file,
# interventional_test_metrics_empirical_prob.json (+ its predictions csv and scatter),
# into the 1-1 run directory. Nothing 1-1 already wrote is touched.
#
# The pairing is exact: the eval reseeds the sampler with config.seed and the train
# values come from the same cache in the same order, so a 1-1 run and every 2-1 run of
# the same head inject the same value into the same test example.
#
# One task per cell, three heads each. Idempotent: a head whose file exists is skipped.
#
# Usage
#   qsub scripts/attention_weight/paper/2-1-0_reeval_1-1_empirical_prob.sh
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

BASE_GROUP=attn_weight_paper_grid_endo_logistic_ndev_300K   # experiment 1-1
RUN_ROOT=outputs/runs/$BASE_GROUP
CFG_ROOT=experiment_configs/$BASE_GROUP
DIST=empirical_prob

## ---- the grid: identical to 1-1 ----
PICKS=plots/attn_weight_scan_10K/grid_picks_keep0.5.csv
CELLS=("2:-1:31,30,0"   "2:-3:31,13,0"   "2:-11:13,24,10"  "2:-19:5,19,24"   "2:-27:5,19,24"
       "6:-1:3,0,31"    "6:-3:3,1,31"    "6:-11:31,29,19"  "6:-19:31,29,19"  "6:-27:31,29,19"
       "10:-1:28,8,16"  "10:-3:7,28,29"  "10:-11:10,9,7"   "10:-19:10,9,15"  "10:-27:10,9,15"
       "14:-1:6,10,5"   "14:-3:18,5,6"   "14:-11:17,18,27" "14:-19:17,18,26" "14:-27:26,17,27")
if [ ${#CELLS[@]} -ne 20 ]; then
  echo "FATAL: ${#CELLS[@]} cells, expected 20" >&2
  exit 1
fi

index=$(($SGE_TASK_ID - 1))
CELL=${CELLS[$index]}
IFS=':' read -r LAYER SRC_OFFSET HEAD_CSV <<< "$CELL"
IFS=',' read -r -a HEADS <<< "$HEAD_CSV"
if [ -z "$LAYER" ] || [ -z "$SRC_OFFSET" ] || [ ${#HEADS[@]} -ne 3 ]; then
  echo "FATAL: could not decode cell '$CELL' (3 heads) from SGE_TASK_ID=$SGE_TASK_ID" >&2
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

echo "=== REEVAL 1-1 cell (layer, offset) = ($LAYER, $SRC_OFFSET), heads ${HEADS[*]}, dist=$DIST ==="

for H in "${HEADS[@]}"; do
  RUN="endo_P${LAYER}.${H}@${SRC_OFFSET}_POS-1_gateoff_klnone_w1.0_300K"
  CONFIG=$CFG_ROOT/$RUN.json
  RUN_DIR=$RUN_ROOT/$RUN
  echo
  echo "=== $RUN ==="
  if [ -f "$RUN_DIR/interventional_test_metrics_${DIST}.json" ]; then
    echo "  already scored, skipping"
    continue
  fi
  if [ ! -f "$CONFIG" ]; then
    echo "FATAL: no 1-1 config $CONFIG" >&2
    exit 1
  fi
  # Exactly one checkpoint: 1-1 saves once per epoch for one epoch. More than one means
  # the run was resumed or rerun, and picking one silently could score the wrong weights.
  CKPTS=("$RUN_DIR"/checkpoint-*)
  if [ ${#CKPTS[@]} -ne 1 ] || [ ! -d "${CKPTS[0]}" ]; then
    echo "FATAL: expected exactly one checkpoint in $RUN_DIR, found: ${CKPTS[*]}" >&2
    exit 1
  fi

  python train_introspector_joint.py \
      --config "$CONFIG" \
      --interv-eval-only \
      --checkpoint "${CKPTS[0]}" \
      --interv-eval-distributions $DIST
done

echo
echo "=========================================================="
echo "REEVAL 1-1 CELL ($LAYER, $SRC_OFFSET) DONE."
echo "End date : $(date)"
echo "=========================================================="
