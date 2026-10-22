#!/bin/bash -l
#$ -P multilm
#$ -N L7H4M1_3_empeval
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=04:00:00
#$ -o logs
#$ -j y

# ============================================================================
# L7 H4 @-1, VARIANT 3: interventional EVAL drawn from the cell's real log-odds
# distribution. No training.
#
# Stage 3's headline eval assumed the log-odds are normal and drew gaussian@1 in log-odds,
# clamped at 1e-4. Here the injected value comes from the cell's own train-split log-odds
# histogram (100 bins; bin by frequency, uniform within the bin), clamped only at 1e-5
# (EMPIRICAL_EPS; the port itself rejects 1e-6).
#
# The models are Stage 3's own checkpoints, re-evaluated: the weights are identical, so
# any difference from Stage 3's numbers is the eval distribution and nothing else.
# Results go to THIS group's run dirs; Stage 3's dirs are only read. gaussian is
# re-run as well -- with the seed fixed it must reproduce Stage 3's
# interventional_test_metrics_gaussian.json, which is the check that the right
# checkpoint was loaded.
#
# Expectation worth stating before the numbers: on L7 H4 gaussian@1 is already close to
# the real distribution (KS 0.035 vs 0.003 for the histogram, 0 clamps either way), so
# this cell may move little. It matters more on cells whose log-odds hit the clamp.
#
# One job, the 5 W in sequence (eval only, a few minutes each).
#
# Usage
#   qsub scripts/attention_weight/L7H4M1/3_empirical_logit_eval.sh
#   WS_OVERRIDE="0.0" bash scripts/attention_weight/L7H4M1/3_empirical_logit_eval.sh
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID"
echo "=========================================================="
nvidia-smi

module load cuda/12.8
module load miniconda/23.11.0
conda activate /projectnb/multilm/jongin/anaconda3/envs/introspection

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection

set -e

EXP_GROUP=attn_weight_L7H4M1_evalemp_scale4
OUT_ROOT=outputs/runs/$EXP_GROUP
STAGE3_GROUP=attn_weight_joint_weight_sweep
EVAL_DISTS=empirical_logit,gaussian

# Stage 3's run config is copied as-is (same cell, cache, head, seed) with only the name
# and output dir changed, so nothing about the model can differ from the checkpoint's.
SIG="P7.4@-1"
WS=(${WS_OVERRIDE:-0.0 0.01 0.03 0.1 0.3})
mkdir -p experiment_configs/$EXP_GROUP

for W in "${WS[@]}"; do
  SRC_NAME="joint_${SIG}_POS-1_gateoff_scale4_kl0.1_w${W}_60K"
  SRC_CONFIG=experiment_configs/$STAGE3_GROUP/$SRC_NAME.json
  CKPT=outputs/runs/$STAGE3_GROUP/$SRC_NAME/checkpoint-1875
  for f in "$SRC_CONFIG" "$CKPT"; do
    if [ ! -e "$f" ]; then echo "FATAL: missing Stage 3 artifact $f" >&2; exit 1; fi
  done

  EXP_NAME="joint_${SIG}_POS-1_gateoff_evalemp_scale4_kl0.1_w${W}_60K"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  python - "$SRC_CONFIG" "$CONFIG" "$EXP_NAME" "$OUT_ROOT" <<'PY'
import json, sys
src, dst, name, out_root = sys.argv[1:]
cfg = json.load(open(src))
# the checkpoint is a linear-head, extreme x4 run; say so, and refuse anything else
tc = cfg["training_config"]
assert tc.get("regression_head_activation", "identity") == "identity", tc
assert tc["intervention_signal_distribution"] == "extreme", tc["intervention_signal_distribution"]
assert cfg["signal_filter"] == "P7.4@-1", cfg["signal_filter"]
cfg["experiment_name"], cfg["output_dir"] = name, out_root
json.dump(cfg, open(dst, "w"), indent=2)
PY
  echo
  echo "=== [W=$W] $EXP_NAME  <-  $CKPT ==="
  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --interv-eval-only \
      --checkpoint "$CKPT" \
      --interv-eval-distributions "$EVAL_DISTS"
done

echo
echo "=========================================================="
echo "L7H4M1 variant 3 DONE -> $OUT_ROOT"
echo "End date : $(date)"
echo "=========================================================="
