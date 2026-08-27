#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_endo_sweep_layer_token
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=12:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 44-47

# ============================================================================
# ENDOGENOUS-ONLY (no interventional loss) regressor sweep over
# 8 layers x 8 token positions x 10 neurons = 640 runs, post-BOS-fix re-run of
# scripts_old_with_off_by_one_bug/neuron/train_single_mlp_introspector_regressor_with_intervention_sweep_layer_token.sh
#
# ONE array task = one (layer, position) cell; the 10 neurons run inside it, so a
# cell's dataset is built once and reused 10 times.
#
# W=1.0 => pure endogenous. train_introspector_joint.py now SKIPS the forward whose
# weight is zero, so this costs exactly what a single-objective trainer costs, and
# the interventional test still runs afterwards -- where, with no interventional
# training, it measures CONTEXT BYPASS: how much of the endogenous score survives
# when the reported value is replaced by an injected one. That number is the w=1
# endpoint of the pareto and is why this goes through the joint script rather than
# train_introspector_with_intervention.py --no-intervention.
#
# NEURON SELECTION. The old sweep's `"seed":21` was never read -- resolution.py drew
# from the global RNG -- so every config generation picked a DIFFERENT neuron, and
# the position axis was unpaired (a different neuron at every position, with neuron
# variance leaking into the position effect). resolution.py now seeds per
# (seed, layer), so a layer's 10 neurons are FIXED ACROSS ALL POSITIONS and the
# position curve can be read per neuron. The resolved list is pulled back out of the
# generated config so the run configs carry it explicitly.
#
# CACHING. All 10 neurons of a cell live in ONE dataset -- extracting 10 signals
# costs one forward pass per text -- and --signal-filter splits it per run. 64
# datasets instead of 640. --cache-dir is the shared root so cells that other scripts
# also need are not rebuilt.
#
# CAVEAT: neurons are drawn uniformly, so a cell can catch a dead / near-constant
# neuron whose correlation is nan. cache_signals prints per-signal describe(); check
# the activation std before reading too much into an outlier point.
# ============================================================================

## SCC env setup
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection
export WANDB_RUN_GROUP=mlp_endo_sweep_layer_token

set -e

## ---- decode (layer, position) from the flat array index ----
index=$(($SGE_TASK_ID-1))
LAYERS=(0 2 4 6 8 10 12 14)
POSS=(-1 -4 -8 -12 -16 -20 -24 -28)
LAYER=${LAYERS[$((index / 8))]}
TOKEN_POS=${POSS[$((index % 8))]}

GATE=on                  # on = gated (matches the old sweep; the model cannot move
                         # the neuron it reports, so endogenous corr means "read or
                         # predict the TRUE activation"). off = second arm, where the
                         # recompute shortcut is available and the drift eval fires.
W=1.0                    # weight on the ENDOGENOUS term; 1.0 => no interventional loss
SEED=21                  # neuron-selection seed (now actually used)
N_NEURONS=10
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
SCALE=10                 # only affects the post-training interventional diagnostics
OVERWRITE=""             # set to --overwrite-cache to force regeneration

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

EXP_GROUP=mlp_endo_sweep_layer_token
CELL="l${LAYER}_pos${TOKEN_POS}"
SAMPLER="{\"type\":\"uniform_layer_sampler\",\"signal_type\":\"mlp\",\"layers\":[${LAYER}],\"signals_per_layer\":${N_NEURONS},\"seed\":${SEED}}"

## ---- resolve this layer's neuron set ONCE (identical at every position) ----
# generated, not run: it exists so resolution.py stays the single source of truth for
# which neurons a (seed, layer) maps to, instead of the selection being duplicated here
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_${CELL}.json
python generate_experiment_config.py \
    --experiment-name "_cell_${CELL}" \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 --train-dataset-split "train" --train-max-samples 300000 \
    --dev-max-samples 200 --dev-offset-from-train 1000000 \
    --test-dataset allenai/c4 --test-dataset-split "validation" --test-max-samples 3000 \
    --max-seq-length 32 \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "$SAMPLER" \
    --test-signals "$SAMPLER" \
    --same-train-test-signals \
    --train-token-selector $TOKEN_POS \
    --test-token-selector $TOKEN_POS \
    --precision 3 \
    --cache-dir data --output-dir outputs/runs/$EXP_GROUP \
    --output "$CELL_CONFIG" \
    --overwrite-cache

NEURONS=($(python -c "
import json
cfg = json.load(open('$CELL_CONFIG'))
print(' '.join(f\"N{s['layer']}.{s['index']}\" for s in cfg['train_signals']))
"))
NEURON_LIST=$(IFS=,; echo "${NEURONS[*]}")
echo "cell $CELL neurons: $NEURON_LIST"

for SAMPLE in $(seq 0 $((N_NEURONS - 1))); do
  SIG=${NEURONS[$SAMPLE]}
  EXP_NAME="endo_${SIG}_POS${TOKEN_POS}_gate${GATE}_r${LORA_R}_lr${LR}_w${W}_300K"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo "=== [$CELL sample $SAMPLE] $EXP_NAME ==="

  # every sample passes the SAME 10-signal list, so all 10 resolve to one cached
  # dataset; --signal-filter is what differs and is not part of the cache identity
  python generate_experiment_config.py \
      --experiment-name "$EXP_NAME" \
      --model meta-llama/Llama-3.2-1B \
      --train-dataset allenai/c4 \
      --train-dataset-split "train" \
      --train-max-samples 300000 \
      --dev-max-samples 200 \
      --dev-offset-from-train 1000000 \
      --test-dataset allenai/c4 \
      --test-dataset-split "validation" \
      --test-max-samples 3000 \
      --max-seq-length 32 \
      --hf-data-kwargs '{"config": "en", "streaming": true}' \
      --train-signals "$NEURON_LIST" \
      --test-signals "$NEURON_LIST" \
      --signal-filter "$SIG" \
      --train-token-selector $TOKEN_POS \
      --test-token-selector $TOKEN_POS \
      --precision 3 \
      --train-batch-size $BS \
      --eval-batch-size 128 \
      --num-epochs 1 \
      --lora-r $LORA_R \
      --lora-alpha $LORA_ALPHA \
      --lora-dropout $LORA_DO \
      $GATE_FLAG \
      --bf16 \
      --learning-rate $LR \
      --warmup-ratio 0.03 \
      --lr-scheduler-type cosine \
      --save-steps 0 \
      --use-regressor \
      --intervention-signal-distribution extreme \
      --intervention-extreme-scale $SCALE \
      --cache-dir data \
      --output-dir outputs/runs/$EXP_GROUP \
      --output "$CONFIG" \
      $OVERWRITE

  # builds the cell's 10-neuron dataset on SAMPLE=0, no-ops for the other 9
  python cache_signals.py --config "$CONFIG" $OVERWRITE

  # alignment regression check once per cell rather than on all 640 runs (it is a
  # property of the data pipeline, so one run per dataset is enough)
  if [ "$SAMPLE" -eq 0 ]; then DRIFT="--force-drift-eval"; else DRIFT=""; fi

  python train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      $DRIFT \
      $OVERWRITE
done
