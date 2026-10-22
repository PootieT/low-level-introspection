#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_joint_lr_sweep_llama318b
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=12:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-60

# ============================================================================
# LR SWEEP, Llama-3.1-8B: did the 1B learning rate transfer?
#
# Every 3B/8B result so far reuses the hyperparameters tuned on Llama-3.2-1B (LR 5e-5,
# w=0.01, scale 10, r=16). The depth-matched 1B -> 8B decline is -0.03 to -0.09 in
# interventional corr -- the same size an LR one step off the optimum can produce. So
# "introspection degrades slightly with scale" and "5e-5 is slightly wrong for 8B"
# currently predict the same table. This sweep separates them.
#
# WHAT VARIES: LR only, under both KL regimes. Everything else is copied verbatim from
# train_single_mlp_introspector_joint_sweep_layer_token_llama318b.sh, including the
# EFFECTIVE batch: 16 x 2 accumulation = 32, the batch the 1B LR was tuned at. A
# plain bs=16 would change the step count and gradient noise and therefore move the
# optimal LR -- the sweep would tune for a different optimization than the grid ran.
#
# CELLS, chosen from the per-cell 1B-vs-8B gaps at matched relative depth:
#   L12 pos -8   depth 0.375  the largest 8B deficit in BOTH regimes (-0.38 kl0.1, -0.32 free)
#   L4  pos -1   depth 0.125  the CONTROL: ~0 gap in both regimes. If LR moves this cell
#                             too, the effect is general; if only L12, it is specific
#   L28 pos -28  depth 0.875  the gap flips sign by regime (-0.28 kl0.1, +0.21 free) --
#                             where an LR x regime interaction would show
#
# TUNING NEURONS ARE NOT THE GRID'S NEURONS. Each cell's cache holds 10 sampled neurons
# and the grid trained the first 5 (samples 0-4). This sweep trains samples 5 and 6:
# already cached, so no new caching and no cache race, and zero leakage -- whatever LR
# is picked here is then evaluated on neurons that played no part in picking it.
#
# REGIMES: "kl0.1" = adaptive KL anchor held near 0.1 nats/token; "unanchored" = no
# anchor. Swept separately because the anchor is where 8B looks most unlike 1B (it
# costs 0.12 interventional corr at 8B vs 0.06 at 1B).
#
# DECISION RULE, fixed before the results exist: choose LR by interventional corr --
# the criterion that chose 5e-5 at 1B. If 8B's best is 5e-5 or within noise of it, the
# transfer held and the grid stands. If another LR closes most of the L12 gap, re-run
# the 8B grid at that LR before drawing any conclusion about scale.
# plotters/plot_lr_sweep.py applies it.
#
# TASK LAYOUT  60 = 3 cells x 2 regimes x 2 neurons x 5 LRs, LR innermost, so any
# contiguous block of 5 tasks is one complete LR curve. The most informative cell is
# first: -t 1-20 alone answers the main question.
#   DRY_RUN=1 bash <script>      prints the task -> (cell, regime, neuron, LR) table
#   SGE_TASK_ID=7 bash <script>  runs one task interactively
# ============================================================================

# ---- what is swept -------------------------------------------------------------
LRS=(1e-5 2e-5 5e-5 1e-4 2e-4)
REGIMES=("kl0.1" "unanchored")
CELLS=("12:-8" "4:-1" "28:-28")     # layer:position, most informative first
TUNE_SAMPLES=(5 6)                  # indices into the cell's 10 cached neurons; grid used 0-4

N_LR=${#LRS[@]}; N_REG=${#REGIMES[@]}; N_TUNE=${#TUNE_SAMPLES[@]}; N_CELL=${#CELLS[@]}
N_TASKS=$((N_LR * N_REG * N_TUNE * N_CELL))

decode() {   # $1 = 0-based task index -> sets LR REGIME SAMPLE LAYER TOKEN_POS
  local i=$1
  LR=${LRS[$((i % N_LR))]};          i=$((i / N_LR))
  SAMPLE=${TUNE_SAMPLES[$((i % N_TUNE))]}; i=$((i / N_TUNE))
  REGIME=${REGIMES[$((i % N_REG))]}; i=$((i / N_REG))
  local cell=${CELLS[$i]}
  LAYER=${cell%%:*}; TOKEN_POS=${cell##*:}
}

if [ -n "$DRY_RUN" ]; then
  echo "$N_TASKS tasks (#\$ -t must be 1-$N_TASKS)"
  printf "%5s  %-6s %-5s %-11s %-7s %s\n" task layer pos regime sample lr
  for t in $(seq 0 $((N_TASKS - 1))); do
    decode "$t"
    printf "%5s  %-6s %-5s %-11s %-7s %s\n" "$((t + 1))" "$LAYER" "$TOKEN_POS" "$REGIME" "$SAMPLE" "$LR"
  done
  exit 0
fi

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
export WANDB_RUN_GROUP=mlp_joint_lr_sweep_llama318b

set -e

if [ -z "$SGE_TASK_ID" ] || [ "$SGE_TASK_ID" = "undefined" ]; then
  echo "ERROR: SGE_TASK_ID unset; submit with qsub, or SGE_TASK_ID=<n> bash $0" >&2; exit 1
fi
index=$(($SGE_TASK_ID - 1))
if [ "$index" -ge "$N_TASKS" ]; then
  echo "ERROR: task $SGE_TASK_ID > $N_TASKS tasks in this sweep" >&2; exit 1
fi
decode "$index"

# ---- fixed: copied from the 8B grid script -------------------------------------------
MODEL=meta-llama/Llama-3.1-8B
GATE=off
W=0.01
SEED=21
N_NEURONS_CACHE=10
SCALE=10
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=16
ACCUM=2                 # effective batch 32 -- the batch the 1B LR was tuned at

if [ "$REGIME" = "unanchored" ]; then
  KL_FLAG=""; KL_TAG="klnone"
else
  KL_FLAG="--kl-anchor-target ${REGIME#kl}"; KL_TAG="$REGIME"
fi
if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

# Leave empty. --overwrite-cache would retrain any finished task on resubmission, and
# must NEVER reach cache_signals here: the 8B caches are shared with the grid.
OVERWRITE=""

# A separate group so these runs never land in the grid's paired tables (the analyses
# glob the group directory). The CACHE is still the grid's: dataset identity excludes
# experiment name, output dir and every training hyperparameter.
EXP_GROUP=mlp_joint_lr_sweep_llama318b
CELL_TAG="l${LAYER}_pos${TOKEN_POS}"
echo "task=$SGE_TASK_ID cell=$CELL_TAG regime=$REGIME sample=$SAMPLE lr=$LR"

# ---- the cell's 10-neuron list (identical to the grid's: same sampler and seed) ------
SAMPLER="{\"type\":\"uniform_layer_sampler\",\"signal_type\":\"mlp\",\"layers\":[${LAYER}],\"signals_per_layer\":${N_NEURONS_CACHE},\"seed\":${SEED}}"
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_${CELL_TAG}.json
python data_prep/generate_experiment_config.py \
    --experiment-name "_cell_${CELL_TAG}" \
    --model $MODEL \
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
SIG=${NEURONS[$SAMPLE]}
if [ -z "$SIG" ]; then
  echo "ERROR: sample $SAMPLE out of range for ${#NEURONS[@]} cached neurons" >&2; exit 1
fi
echo "cell $CELL_TAG neurons: $NEURON_LIST  -> tuning on sample $SAMPLE = $SIG"

EXP_NAME="lrsweep_${SIG}_POS${TOKEN_POS}_gate${GATE}_scale${SCALE}_${KL_TAG}_w${W}_lr${LR}"
CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json

# The full neuron list is passed so the cache identity matches the grid's cell cache;
# --signal-filter is what restricts training to this one neuron.
python data_prep/generate_experiment_config.py \
    --experiment-name "$EXP_NAME" \
    --model $MODEL \
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
    --gradient-accumulation-steps $ACCUM \
    --eval-batch-size 128 \
    --num-epochs 1 \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DO \
    $GATE_FLAG \
    --bf16 \
    --learning-rate $LR \
    --save-steps 0 \
    --use-regressor \
    --intervention-signal-distribution extreme \
    --intervention-extreme-scale $SCALE \
    --cache-dir data \
    --output-dir outputs/runs/$EXP_GROUP \
    --output "$CONFIG" \
    $OVERWRITE

# No-op: the grid already built this cell's cache. No $OVERWRITE here, ever.
python data_prep/cache_signals.py --config "$CONFIG"

# In-domain evals only (no OOD, no benchmarks): the decision needs interventional corr,
# and the eval now also writes Spearman / Kendall / monotone R^2 alongside Pearson.
python trainers/train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    $KL_FLAG \
    $OVERWRITE

# LM cost on one neuron per (cell, regime, LR): in the unanchored regime KL is free to
# move with LR, and how far it moves is part of the answer. Skips if already scored.
if [ "$SAMPLE" -eq "${TUNE_SAMPLES[0]}" ]; then
  python evals/score_lm_degradation.py --config "$CONFIG"
fi
