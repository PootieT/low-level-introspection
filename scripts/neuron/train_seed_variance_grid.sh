#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_seed_variance
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=4:00:00
#$ -o log
#$ -j y
#$ -t 1-160

# ============================================================================
# SEED VARIANCE (rebuttal: "means +/- std over 3-5 neurons ... no seeds"). Is the
# joint - SFT gap larger than what retraining the same run with another seed moves?
#
# 4 cells (layers {2, 8} x positions {-1 (t=31), -28 (t=4)}) x the joint grid's 5
# neurons x {joint KL=0.1, SFT} x train seeds {1, 2, 3, 4} = 160 runs, one per task.
# The original grid runs are the 5th replicate of every (neuron, cell, arm).
#
# The configs are COPIES of the original runs' configs (data_prep/make_seed_configs.py),
# differing only in name, output dir and training_config.train_seed -- same cache, same
# schedule, same ExperimentConfig.seed (so every replicate is evaluated on the same
# injected values). train_seed seeds init, training injections and data order
# (src/introspector/training/seeding.py); runs before it existed were unseeded.
#
# The loss weight and KL budget are CLI flags of the trainer, not config fields, so they
# are taken from the name prefix exactly as the original grid scripts passed them:
#   joint_*  --joint-loss-weight 0.01 --kl-anchor-target 0.1
#            (scripts/neuron/train_single_mlp_introspector_joint_sweep_layer_token.sh)
#   endo_*   --joint-loss-weight 1.0
#            (scripts/neuron/train_single_mlp_introspector_endogenous_sweep_layer_token.sh)
# In-domain endogenous + interventional evals only (no OOD, no LM degradation): that is
# what Fig. 3 shows.
#
# Configs were generated with (re-run only to add seeds / cells):
#   python data_prep/make_seed_configs.py \
#     --source mlp_joint_grid_layer_token:'joint_*_gateoff_scale10_kl0.1_w0.01_300K' \
#     --source mlp_endo_sweep_layer_token:'endo_*_gateoff_w1.0_300K' \
#     --cells 2:-1,2:-28,8:-1,8:-28 \
#     --neurons 2.1447,2.1886,2.2566,2.5949,2.6998,8.3852,8.4144,8.4956,8.5777,8.6370 \
#     --seeds 1,2,3,4
#
# Optional arg: $1 = config-name glob within the group (default: all 160). The task id
# indexes the SORTED matching configs, so a narrower glob needs a matching #$ -t.
#
# Table afterwards (CPU, seconds; reads the run dirs, so run it on SCC):
#   python plotters/table_seed_variance.py
# ============================================================================

EXP_GROUP=mlp_seed_variance_layer_token
GLOB=${1:-*_seed*}

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID / $SGE_TASK_LAST"
echo "Group : $EXP_GROUP   Glob : $GLOB"
echo "=========================================================="
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection
export WANDB_RUN_GROUP=$EXP_GROUP

set -e

CONFIGS=($(ls experiment_configs/$EXP_GROUP/$GLOB.json | sort))
N=${#CONFIGS[@]}
index=$((SGE_TASK_ID - 1))
if [ "$index" -ge "$N" ]; then
  echo "task $SGE_TASK_ID: only $N configs match $GLOB -- nothing to do"
  exit 0
fi
CONFIG=${CONFIGS[$index]}
NAME=$(basename "$CONFIG" .json)

case "$NAME" in
  joint_*) ARM_FLAGS="--joint-loss-weight 0.01 --kl-anchor-target 0.1" ;;
  endo_*)  ARM_FLAGS="--joint-loss-weight 1.0" ;;
  *) echo "unknown arm for $NAME"; exit 1 ;;
esac

# the copied config points at the ORIGINAL run's cache; it must already exist (a
# rebuild here would race 20 tasks onto one cache and is not what a replicate is)
CACHE=$(python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['cache_dir'] + '/' + d['dataset_name'])" "$CONFIG")
if [ ! -d "$CACHE/train" ]; then
  echo "missing cache $CACHE for $NAME (built by the original grid run)"; exit 1
fi

echo "task=$SGE_TASK_ID/$N -> $NAME | $ARM_FLAGS | cache=$CACHE"

python trainers/train_introspector_joint.py \
    --config "$CONFIG" \
    $ARM_FLAGS
