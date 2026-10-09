#!/bin/bash
#SBATCH --job-name=sae_steer_gen
#SBATCH --partition=rtx-batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=log/%x-%A_%a.out
#SBATCH --array=1-5    # one task per swept layer

echo "Start: $(date)  array task $SLURM_ARRAY_TASK_ID on $(hostname)"
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${INTRO_ENV:-/work/neu/p2026_0061_neu/low-level-introspection/introspection_env}
export HF_HOME=${HF_HOME:-/work/neu/p2026_0061_neu/low-level-introspection/.hf_cache}

set -e

# Settings reproduce the layer-9 selection run (recovered from its samples/key files):
# top 40 explained features, 50 prompts each from a 200-prompt pool, doses 3x and 5x of peak.

LAYERS=(${LAYERS_OVERRIDE:-0 5 10 15 20})
LAYER=${LAYERS[$(($SLURM_ARRAY_TASK_ID-1))]}
ANN=data/sae_stats/gemma2-2b_res${LAYER}_pos-1_16k.annotated.parquet

if [ ! -f "$ANN" ]; then
    echo "no annotated parquet for layer $LAYER ($ANN); run Step 1.5 first."
    exit 1
fi

python generate_sae_steering_samples.py \
    --annotated "$ANN" \
    --model google/gemma-2-2b \
    --sae-release gemma-scope-2b-pt-res-canonical \
    --layer $LAYER --width 16k --site res \
    --n-prompts 50 --prompt-pool 200 --prompt-words 8 --max-new-tokens 40 \
    --strengths 3 5 \
    --max-features 40 \
    --out-dir data/sae_steering_samples
echo "Done: $(date)"
