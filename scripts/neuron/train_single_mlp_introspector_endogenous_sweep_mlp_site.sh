#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_site_sweep_layer_token
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=12:00:00
#$ -o log
#$ -j y
#$ -t 1-36

# ============================================================================
# WHICH MLP QUANTITY to introspect: the endogenous SFT grid
# (train_single_mlp_introspector_endogenous_sweep_layer_token.sh) re-run at the two
# other per-neuron sites of the Llama MLP  down_proj(act_fn(gate_proj(x)) * up_proj(x)):
#   gate     g = gate_proj(x)                 linear in the MLP input -> exact input bypass
#   down_in  h = act_fn(g) * up_proj(x)       the neuron's write is h * W_down[:, i], so a
#                                             fixed readout of the write recovers h exactly
# The main grid's site is act_fn (a = silu(g)), whose write a * u(x) * W_down[:, i] flips
# sign with u(x). In the base model (mechanistic_analysis/score_neuron_write_stats.py,
# plotters/plot_target_site_readability.py):
#                  bypass R^2 from pre-MLP resid    causal ceiling of a fixed readout
#   gate           ~0.99                            ~0.04-0.14
#   act_fn         ~0.85-0.99                       ~0.04-0.20
#   down_in        ~0.35-0.55                       1
# so if SFT's non-causal bypass comes from the training target being easier to read from
# the input than from the write, interventional corr should rise act_fn -> down_in and
# fall (or stay at floor) for gate. The cached value, the regression target and the
# injected value are all the SAME tensor at each site (ExperimentConfig.mlp_site).
#
# GRID: a subset of the main grid's cells, in layers 0-6 where interventional corr
# varies (it is ~0 with no spread at 8-14): 3 layers x 3 positions x 5 neurons x 2 sites
# x 2 gates = 180 runs in 36 tasks. Same seed and sampler as the main grid, so the
# neurons are the SAME ones and the act_fn arm is the main grid's own runs at these
# cells (outputs/runs/mlp_endo_sweep_layer_token, both gates).
#
# Array layout: GATE outermost (1-18 on, 19-36 off), then SITE, then (layer, position).
# Both gate arms of a (site, cell) share one cached dataset; keeping them 18 tasks apart
# stops concurrent slots racing to build the same directory. A site's cache is its own
# (mlp_site is part of the cache identity), so the act_fn caches are never touched.
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
export WANDB_RUN_GROUP=mlp_site_sweep_layer_token

set -e

## ---- decode (gate, site, layer, position) from the flat array index ----
index=$(($SGE_TASK_ID-1))
LAYERS=(0 2 6)           # 3 of the main grid's layers, spanning 0-6
POSS=(-1 -12 -24)        # 3 of the main grid's positions, spanning its range
SITES=(gate down_in)
GATES=(on off)
N_CELLS=$(( ${#LAYERS[@]} * ${#POSS[@]} ))          # 9
N_PER_GATE=$(( ${#SITES[@]} * N_CELLS ))            # 18
GATE=${GATES[$((index / N_PER_GATE))]}
rem=$((index % N_PER_GATE))
SITE=${SITES[$((rem / N_CELLS))]}
cell_idx=$((rem % N_CELLS))
LAYER=${LAYERS[$((cell_idx / ${#POSS[@]}))]}
TOKEN_POS=${POSS[$((cell_idx % ${#POSS[@]}))]}
echo "task $SGE_TASK_ID -> gate $GATE, site $SITE, layer $LAYER, position $TOKEN_POS"

# everything below matches the main grid except --mlp-site
W=1.0                    # endogenous only; the interventional test still runs afterwards
SEED=21                  # neuron-selection seed: same neurons as the main grid
N_NEURONS_CACHE=10       # cached signals per cell (one forward pass serves all)
N_NEURONS_TRAIN=5        # the main grid trains the first 5
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
SCALE=10                 # only affects the post-training interventional diagnostics
OOD_FLAG="--ood-eval"
OVERWRITE=""             # set to --overwrite-cache to force regeneration

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

EXP_GROUP=mlp_site_sweep_layer_token
CELL="l${LAYER}_pos${TOKEN_POS}"
SAMPLER="{\"type\":\"uniform_layer_sampler\",\"signal_type\":\"mlp\",\"layers\":[${LAYER}],\"signals_per_layer\":${N_NEURONS_CACHE},\"seed\":${SEED}}"

## ---- resolve this layer's neuron set (identical to the main grid's) ----
mkdir -p experiment_configs/$EXP_GROUP
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_${SITE}_gate${GATE}_${CELL}.json
python data_prep/generate_experiment_config.py \
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

for SAMPLE in $(seq 0 $((N_NEURONS_TRAIN - 1))); do
  SIG=${NEURONS[$SAMPLE]}
  EXP_NAME="endo_${SIG}_POS${TOKEN_POS}_site${SITE}_gate${GATE}_w${W}_300K"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo "=== [$SITE $CELL sample $SAMPLE] $EXP_NAME ==="

  python data_prep/generate_experiment_config.py \
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
      --mlp-site $SITE \
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

  # builds the (site, cell) 10-neuron dataset on SAMPLE=0, no-ops for the other 4
  python data_prep/cache_signals.py --config "$CONFIG" $OVERWRITE

  # the drift/alignment check reads the live value at the same site as the cache, so
  # on SAMPLE=0 it also confirms the new site's cache and hook agree (corr ~1 at offset 0)
  if [ "$SAMPLE" -eq 0 ]; then DRIFT="--force-drift-eval"; else DRIFT=""; fi

  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      $DRIFT \
      $OOD_FLAG \
      $OVERWRITE

  python trainers/train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    $DRIFT \
    $OOD_FLAG \
    --ood-only \
    $OVERWRITE
done
