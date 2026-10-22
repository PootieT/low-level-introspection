#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_transmission
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=4:00:00
#$ -o logs
#$ -j y

# ============================================================================
# Does an injected attention weight reach the readout token? (Stage 3 follow-up)
#
#   qsub scripts/attention_weight/analysis/2_transmission_probe.sh
#
# Stage 3 gave interventional corr ~0 at every W. mechanistic_analysis/probe_attention_weight_transmission.py
# measures, without training, how far the injection travels: ||dh|| per layer at dst and
# at the readout token, and whether logit(t) is linearly decodable from dh / h_interv.
# Both the adapter-disabled base and the checkpoint are measured, so "the base model
# never carries it" and "the adapter destroyed a path that existed" are told apart.
#
# One cell by default: L2 h31 @-1, the best endogenous cell, at W=1 and W=0. Override
# with CONFIGS="a.json b.json" qsub ... to run others.
# ============================================================================

echo "Start date : $(date)   Job ID : $JOB_ID"
nvidia-smi

module load cuda/12.8
module load miniconda/23.11.0
conda activate /projectnb/multilm/jongin/anaconda3/envs/introspection
export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export PYTHONUNBUFFERED=1

ENV_PY=/projectnb/multilm/jongin/anaconda3/envs/introspection/bin/python
if [ "$(command -v python)" != "$ENV_PY" ]; then
  echo "FATAL: python is $(command -v python), expected $ENV_PY (conda activate failed?)" >&2
  exit 1
fi
set -e

CFG_DIR=experiment_configs/attn_weight_joint_weight_sweep
CONFIGS=(${CONFIGS:-"$CFG_DIR/joint_P2.31@-1_POS-1_gateoff_scale4_kl0.1_w1.0_60K.json" \
                    "$CFG_DIR/joint_P2.31@-1_POS-1_gateoff_scale4_kl0.1_w0.0_60K.json"})

for cfg in "${CONFIGS[@]}"; do
  echo
  echo "=== $cfg ==="
  python mechanistic_analysis/probe_attention_weight_transmission.py --config "$cfg"
done

echo "End date : $(date)"
