#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_joint_lr_sweep
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=12:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-72

# ============================================================================
# LR SWEEP, ANY MODEL: does the Llama-3.2-1B learning rate transfer?
#
# The model-generic version of train_single_mlp_introspector_joint_lr_sweep_llama318b.sh
# (same design, same decision rule, same run names and groups -- llama318b here reproduces
# that script's runs exactly, so finished 8B tasks are skipped on resubmission). Pick the
# model with SWEEP_MODEL:
#
#   key          model                 layers  cells (layer:pos)        batch    cache
#   llama323b    meta-llama/Llama-3.2-3B   28  11:-8   4:-1  25:-28     32 x 1   NEW
#   llama318b    meta-llama/Llama-3.1-8B   32  12:-8   4:-1  28:-28     16 x 2   grid's
#   qwen25_05b   Qwen/Qwen2.5-0.5B         24   9:-8   3:-1  21:-28     32 x 1   NEW
#   qwen25_15b   Qwen/Qwen2.5-1.5B         28  11:-8   4:-1  25:-28     32 x 1   NEW
#   qwen25_3b    Qwen/Qwen2.5-3B           36  14:-8   5:-1  32:-28     32 x 1   NEW
#   qwen25_7b    Qwen/Qwen2.5-7B           28  11:-8   4:-1  25:-28     16 x 2   NEW
#   qwen25_14b   Qwen/Qwen2.5-14B          48  18:-8   6:-1  42:-28      8 x 4   NEW (80G)
#
# CELLS play the same three roles in every model, in the same order:
#   pos -8  at depth ~3/8   the cell where 8B fell furthest behind 1B (the main question)
#   pos -1  at depth ~1/8   the CONTROL (~0 gap at 8B)
#   pos -28 at depth ~7/8   where the 8B gap flipped sign by regime
# Layers are round(depth * n_layers), halves up, for EVERY model, so the three cells sit
# at the same relative depth across the comparison. That takes Llama-3B OFF its grid
# (layers 6/12/18/24 of 28), so it needs new caches like Qwen; a later 3B or Qwen grid
# should use the same rule (1/8, 3/8, 5/8, 7/8) so it reuses these caches. 8B's cells
# are exact depths and are its grid's own.
#
# EFFECTIVE BATCH IS 32 everywhere -- the batch the 1B LR was tuned at -- so only the
# micro-batch moves with model size (BS x ACCUM above).
#
# TUNING NEURONS are samples 5 and 6 of each cell's 10 cached neurons; a grid trains 0-4.
# No neuron that picks the LR is ever used to evaluate it.
#
# LRs: 1e-5 .. 5e-4. 5e-4 is new relative to the 8B sweep: there the unanchored optimum
# sat at the old upper edge (2e-4). To add only it to the finished 8B sweep:
#   qsub -t 1-12 -v SWEEP_MODEL=llama318b,LR_LIST=5e-4 <script>
# (LR_LIST is colon-separated, because qsub -v splits on commas.)
#
# ---- SUBMITTING -------------------------------------------------------------
# Llama 8B (caches exist):
#   qsub -N lrsweep_llama318b -v SWEEP_MODEL=llama318b <script>
# Llama 3B and Qwen (NEW caches): all 24 tasks of a cell would otherwise race to build
# one cache directory, so build the caches first -- one task per cell -- and hold the
# sweep on it:
#   qsub -N lrcache_qwen25_7b -t 1-3 -v SWEEP_MODEL=qwen25_7b,CACHE_ONLY=1 <script>
#   qsub -N lrsweep_qwen25_7b -hold_jid lrcache_qwen25_7b -v SWEEP_MODEL=qwen25_7b <script>
# Training tasks refuse to start without a complete cache rather than build one.
# qwen25_14b: add -l gpu_memory=80G (and probably -l h_rt=24:00:00) to both.
#
# TASK LAYOUT  72 = 3 cells x 2 regimes x 2 neurons x 6 LRs, LR innermost (any block
# of 6 contiguous tasks is one LR curve); the main cell is first, so -t 1-24 alone
# answers the main question.
#   SWEEP_MODEL=qwen25_7b DRY_RUN=1 bash <script>      prints the task table
#   SWEEP_MODEL=qwen25_7b SGE_TASK_ID=7 bash <script>  runs one task interactively
# Read with: python analysis/plot_lr_sweep.py --model <key>   (--model a,b,... compares)
# ============================================================================

# ---- the model --------------------------------------------------------------------
case "$SWEEP_MODEL" in
  llama323b)  MODEL=meta-llama/Llama-3.2-3B; CELLS=("11:-8" "4:-1" "25:-28"); BS=32; ACCUM=1 ;;
  llama318b)  MODEL=meta-llama/Llama-3.1-8B; CELLS=("12:-8" "4:-1" "28:-28"); BS=16; ACCUM=2 ;;
  qwen25_05b) MODEL=Qwen/Qwen2.5-0.5B;       CELLS=("9:-8"  "3:-1" "21:-28"); BS=32; ACCUM=1 ;;
  qwen25_15b) MODEL=Qwen/Qwen2.5-1.5B;       CELLS=("11:-8" "4:-1" "25:-28"); BS=32; ACCUM=1 ;;
  qwen25_3b)  MODEL=Qwen/Qwen2.5-3B;         CELLS=("14:-8" "5:-1" "32:-28"); BS=32; ACCUM=1 ;;
  qwen25_7b)  MODEL=Qwen/Qwen2.5-7B;         CELLS=("11:-8" "4:-1" "25:-28"); BS=16; ACCUM=2 ;;
  qwen25_14b) MODEL=Qwen/Qwen2.5-14B;        CELLS=("18:-8" "6:-1" "42:-28"); BS=8;  ACCUM=4 ;;
  *) echo "ERROR: set SWEEP_MODEL to one of llama323b llama318b qwen25_05b qwen25_15b" \
          "qwen25_3b qwen25_7b qwen25_14b (got '$SWEEP_MODEL')" >&2; exit 1 ;;
esac
if [ $((BS * ACCUM)) -ne 32 ]; then
  echo "ERROR: effective batch $BS x $ACCUM != 32" >&2; exit 1
fi

# ---- what is swept -------------------------------------------------------------
if [ -n "$LR_LIST" ]; then
  IFS=: read -r -a LRS <<< "$LR_LIST"
else
  LRS=(1e-5 2e-5 5e-5 1e-4 2e-4 5e-4)
fi
REGIMES=("kl0.1" "unanchored")
TUNE_SAMPLES=(5 6)                  # indices into the cell's 10 cached neurons; grids use 0-4

N_LR=${#LRS[@]}; N_REG=${#REGIMES[@]}; N_TUNE=${#TUNE_SAMPLES[@]}; N_CELL=${#CELLS[@]}
N_PER_CELL=$((N_LR * N_REG * N_TUNE))
N_TASKS=$((N_PER_CELL * N_CELL))

decode() {   # $1 = 0-based task index -> sets LR REGIME SAMPLE LAYER TOKEN_POS
  local i=$1
  LR=${LRS[$((i % N_LR))]};          i=$((i / N_LR))
  SAMPLE=${TUNE_SAMPLES[$((i % N_TUNE))]}; i=$((i / N_TUNE))
  REGIME=${REGIMES[$((i % N_REG))]}; i=$((i / N_REG))
  local cell=${CELLS[$i]}
  LAYER=${cell%%:*}; TOKEN_POS=${cell##*:}
}

if [ -n "$DRY_RUN" ]; then
  echo "$SWEEP_MODEL ($MODEL), batch $BS x $ACCUM"
  if [ -n "$CACHE_ONLY" ]; then
    echo "CACHE_ONLY: $N_CELL tasks (#\$ -t must be 1-$N_CELL)"
    for c in $(seq 0 $((N_CELL - 1))); do echo "  task $((c + 1)): cell ${CELLS[$c]}"; done
    exit 0
  fi
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

# A separate group per model so these runs never land in a grid's paired tables (the
# analyses glob the group directory). The CACHE is still the grid's where one exists:
# dataset identity excludes experiment name, output dir and every training hyperparameter.
EXP_GROUP=mlp_joint_lr_sweep_${SWEEP_MODEL}

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection
export WANDB_RUN_GROUP=$EXP_GROUP

set -e

if [ -z "$SGE_TASK_ID" ] || [ "$SGE_TASK_ID" = "undefined" ]; then
  echo "ERROR: SGE_TASK_ID unset; submit with qsub, or SGE_TASK_ID=<n> bash $0" >&2; exit 1
fi
index=$(($SGE_TASK_ID - 1))
if [ -n "$CACHE_ONLY" ]; then
  # one task per cell: decode the cell's first training task. The cache it builds is the
  # one every task of the cell reads (signal filter, LR and regime are not in its key)
  if [ "$index" -ge "$N_CELL" ]; then
    echo "ERROR: CACHE_ONLY task $SGE_TASK_ID > $N_CELL cells" >&2; exit 1
  fi
  index=$((index * N_PER_CELL))
elif [ "$index" -ge "$N_TASKS" ]; then
  echo "ERROR: task $SGE_TASK_ID > $N_TASKS tasks in this sweep" >&2; exit 1
fi
decode "$index"

# ---- fixed: the 1B-tuned settings every grid used -----------------------------------
GATE=off
W=0.01
SEED=21
N_NEURONS_CACHE=10
SCALE=10
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0

if [ "$REGIME" = "unanchored" ]; then
  KL_FLAG=""; KL_TAG="klnone"
else
  KL_FLAG="--kl-anchor-target ${REGIME#kl}"; KL_TAG="$REGIME"
fi
if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi

# Leave empty. --overwrite-cache would retrain any finished task on resubmission, and
# must NEVER reach cache_signals here: the 8B caches are shared with its grid.
OVERWRITE=""

CELL_TAG="l${LAYER}_pos${TOKEN_POS}"
echo "model=$SWEEP_MODEL task=$SGE_TASK_ID cell=$CELL_TAG regime=$REGIME sample=$SAMPLE lr=$LR" \
     "${CACHE_ONLY:+(CACHE_ONLY)}"

# ---- the cell's 10-neuron list (identical to a grid's: same sampler and seed) --------
SAMPLER="{\"type\":\"uniform_layer_sampler\",\"signal_type\":\"mlp\",\"layers\":[${LAYER}],\"signals_per_layer\":${N_NEURONS_CACHE},\"seed\":${SEED}}"
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_${CELL_TAG}.json
python generate_experiment_config.py \
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

# The full neuron list is passed so the cache identity matches a grid's cell cache;
# --signal-filter is what restricts training to this one neuron.
python generate_experiment_config.py \
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

if [ -n "$CACHE_ONLY" ]; then
  # skips if the cache is already complete. No $OVERWRITE here, ever
  python cache_signals.py --config "$CONFIG"
  echo "CACHE_ONLY: cell $CELL_TAG cache ready"
  exit 0
fi

# Training tasks never build a cache: 24 of them start together per cell and would race
# on one directory. A missing cache means the CACHE_ONLY pass has not run (or failed).
CACHE_DIR=$(python -c "
from pathlib import Path
from src.introspector.config import ExperimentConfig
print(ExperimentConfig.model_validate_json(Path('$CONFIG').read_text()).dataset_cache_dir)
")
if [ ! -f "$CACHE_DIR/metadata.json" ]; then
  echo "ERROR: no complete cache at $CACHE_DIR -- run the CACHE_ONLY pass first:" >&2
  echo "  qsub -t 1-$N_CELL -v SWEEP_MODEL=$SWEEP_MODEL,CACHE_ONLY=1 $0" >&2
  exit 1
fi

# In-domain evals only (no OOD, no benchmarks): the decision needs interventional corr,
# and the eval also writes Spearman / Kendall / monotone R^2 alongside Pearson.
python train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    $KL_FLAG \
    $OVERWRITE

# LM cost on one neuron per (cell, regime, LR): in the unanchored regime KL is free to
# move with LR, and how far it moves is part of the answer. Skips if already scored.
if [ "$SAMPLE" -eq "${TUNE_SAMPLES[0]}" ]; then
  python score_lm_degradation.py --config "$CONFIG"
fi
