#!/bin/bash
#SBATCH --job-name=sae_joint_l9
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=log/%x-%j.out

echo "Start: $(date) on $(hostname)"; nvidia-smi -L || true
module load miniforge3; source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${INTRO_ENV:-/work/neu/p2026_0061_neu/low-level-introspection/introspection_env}
export HF_HOME=${HF_HOME:-/work/neu/p2026_0061_neu/low-level-introspection/.hf_cache}
export WANDB_MODE=offline
set -e
cd /work/neu/p2026_0061_neu/low-level-introspection

LAYER=9; TOKEN_POS=-1
FEATURES=(${FEATS_OVERRIDE:-S9.1568 S9.4854 S9.4120 S9.978 S9.12408})
FEATURE_LIST=$(IFS=,; echo "${FEATURES[*]}")
LR=2e-5; LORA_R=16; LORA_ALPHA=$((LORA_R * 2)); BS=32
W=${W:-0.5}                               # joint weight on the endogenous term
KL_TARGET=${KL_TARGET:-0.1}               # adaptive KL-anchor budget (nats/token); matches the neuron track
TRAIN_SAMPLES=${TRAIN_SAMPLES:-150000}
EXP_GROUP=sae_l9_joint

echo "JOINT gate-OFF | features=$FEATURE_LIST  pos=$TOKEN_POS  W(endo)=$W"

first=1
for SIG in "${FEATURES[@]}"; do
  WTAG=$(echo "$W" | tr -d '.')
  EXP_NAME="saeJoint_${SIG}_POS${TOKEN_POS}_gateoff_w${WTAG}"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo "=== joint feature=$SIG W=$W -> $EXP_NAME ==="

  python data_prep/generate_experiment_config.py \
      --experiment-name "$EXP_NAME" \
      --model google/gemma-2-2b \
      --objective detection --detection-balance-pos-frac 0.5 \
      --intervention-signal-distribution extreme --intervention-extreme-scale 1 \
      --train-dataset allenai/c4 --train-dataset-split "train" --train-max-samples $TRAIN_SAMPLES \
      --dev-max-samples 2000 --dev-offset-from-train 1000000 \
      --test-dataset allenai/c4 --test-dataset-split "validation" --test-max-samples 30000 \
      --max-seq-length 32 --hf-data-kwargs '{"config": "en", "streaming": true}' \
      --train-signals "$FEATURE_LIST" --test-signals "$FEATURE_LIST" --signal-filter "$SIG" \
      --train-token-selector $TOKEN_POS --test-token-selector $TOKEN_POS --precision 3 \
      --train-batch-size $BS --eval-batch-size 128 --num-epochs 1 \
      --lora-r $LORA_R --lora-alpha $LORA_ALPHA --lora-dropout 0.0 --bf16 \
      --learning-rate $LR --warmup-ratio 0.03 --lr-scheduler-type cosine --save-steps 0.25 \
      --cache-dir data --output-dir outputs/runs/$EXP_GROUP --output "$CONFIG"

  python data_prep/cache_signals.py --config "$CONFIG"          # shared cache; no-op if built

  if [ "$first" = "1" ]; then DRIFT="--force-drift-eval"; else DRIFT=""; fi
  python trainers/train_sae_detector_joint.py --config "$CONFIG" --joint-loss-weight $W \
      --kl-anchor-target $KL_TARGET $DRIFT
  first=0
done
echo "Done: $(date)"
