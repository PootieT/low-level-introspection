#!/bin/bash
#SBATCH --job-name=sae_sweep_cache
#SBATCH --partition=rtx-batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=log/%x-%A_%a.out
#SBATCH --array=1-16    # one task per (layer, position) = 4 x 4

# ============================================================================
# Stage 1 of the layer x position sweep: build ONE signal cache per (layer, position)
# cell, holding all 3 features of that layer. Run before train_sae_sweep.sh so
# parallel training tasks never race to build the same cache. Use
# submit_sae_sweep.sh, which chains the two stages with --dependency.
# (Layer-9 reference: one cache took ~13 min.)
# ============================================================================
echo "Start: $(date)  array task $SLURM_ARRAY_TASK_ID on $(hostname)"; nvidia-smi -L || true
module load miniforge3; source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${INTRO_ENV:-/work/neu/p2026_0061_neu/low-level-introspection/introspection_env}
export HF_HOME=${HF_HOME:-/work/neu/p2026_0061_neu/low-level-introspection/.hf_cache}
set -e
source scripts/sae/sweep_common.sh

IDX=$(($SLURM_ARRAY_TASK_ID-1))
if [ $IDX -ge $((N_L * N_P)) ]; then echo "task beyond grid; nothing to do."; exit 0; fi
LAYER=${LAYERS[$((IDX / N_P))]}
POS=${POSITIONS[$((IDX % N_P))]}
FEATS=($(features_for_layer "$LAYER"))

# Any feature/weight of the cell gives the same dataset_name; use F1 at the first weight.
CONFIG=$(make_config "${FEATS[0]}" "$POS" "${WEIGHTS[0]}")
echo "=== cache layer=$LAYER pos=$POS features=${FEATS[*]} -> $CONFIG ==="
python data_prep/cache_signals.py --config "$CONFIG"          # no-op if already built
echo "Cache built: $(date)"
