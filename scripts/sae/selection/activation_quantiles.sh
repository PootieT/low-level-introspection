#!/bin/bash
#SBATCH --job-name=sae_act_quantiles
#SBATCH --partition=rtx-batch,rtx-devel,b200-batch,b200-devel
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=log/%x-%A_%a.out
#SBATCH --array=1-4

echo "Start: $(date) task $SLURM_ARRAY_TASK_ID on $(hostname)"; nvidia-smi -L || true
set -e
ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}}; cd $ROOT
module load miniforge3; source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${INTRO_ENV:-$ROOT/introspection_env}
export HF_HOME=${HF_HOME:-$ROOT/.hf_cache}
LAYERS=(${LAYERS_OVERRIDE:-5 10 15 20})
L=${LAYERS[$((SLURM_ARRAY_TASK_ID-1))]}
OUT=data/sae_stats/gemma2-2b_res${L}_act_quantiles_16k.parquet
if [ -f $OUT ]; then echo "$OUT exists; skipping"; exit 0; fi
python data_prep/sae_activation_quantiles.py --layer $L \
    --stats data/sae_stats/gemma2-2b_res${L}_pos-1_16k.parquet --out $OUT
echo "Done: $(date)"
