#!/bin/bash
#SBATCH --job-name=sae_sweep_train
#SBATCH --partition=rtx-batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=log/%x-%A_%a.out
#SBATCH --array=1-150   # 3 features x 5 layers x 5 positions x 2 weights

# ============================================================================
# Stage 2 of the layer x position sweep: one joint gate-off run per task.
#
# Task order is FEATURE-MAJOR: tasks 1-50 are feature rank 1 (F1) of every layer, over
# all layers x positions x weights; 51-100 are F2; 101-150 are F3. SLURM starts lower
# array indices first, so F1's full layer x position grid finishes first.
#
# Every run does the cache-alignment check (--force-drift-eval): it recomputes the
# feature at the cached position and stops the run if steering would hit the wrong
# token. Layer-9 reference: one run took ~1 h (W=0.9). The trainer skips a finished
# run and resumes from its latest checkpoint, so a killed task can simply be resubmitted.
# ============================================================================
echo "Start: $(date)  array task $SLURM_ARRAY_TASK_ID on $(hostname)"; nvidia-smi -L || true
module load miniforge3; source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${INTRO_ENV:-/work/neu/p2026_0061_neu/low-level-introspection/introspection_env}
export HF_HOME=${HF_HOME:-/work/neu/p2026_0061_neu/low-level-introspection/.hf_cache}
export WANDB_MODE=offline
set -e
source scripts/sae/sweep_common.sh

# task -> (feature rank K, layer, position, weight), feature-major then layer, position, weight
IDX=$(($SLURM_ARRAY_TASK_ID-1))
if [ $IDX -ge $((N_FEATS * N_L * N_P * N_W)) ]; then echo "task beyond grid; nothing to do."; exit 0; fi
I_W=$((IDX % N_W));                 IDX=$((IDX / N_W))
I_P=$((IDX % N_P));                 IDX=$((IDX / N_P))
I_L=$((IDX % N_L));                 K=$((IDX / N_L))
LAYER=${LAYERS[$I_L]}; POS=${POSITIONS[$I_P]}; W=${WEIGHTS[$I_W]}
FEATS=($(features_for_layer "$LAYER"))
SIG=${FEATS[$K]}

CONFIG=$(make_config "$SIG" "$POS" "$W")
echo "=== F$((K+1)) $SIG  layer=$LAYER pos=$POS W(endo)=$W -> $CONFIG ==="

python cache_signals.py --config "$CONFIG"          # no-op: built by cache_sae_sweep.sh
python train_sae_detector_joint.py --config "$CONFIG" --joint-loss-weight $W \
    --kl-anchor-target $KL_TARGET --force-drift-eval
echo "Done: $(date)"
