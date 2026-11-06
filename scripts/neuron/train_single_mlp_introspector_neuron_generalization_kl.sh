#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_neurongen_kl
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=12:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-15

# ============================================================================
# NEURON GENERALIZATION UNDER A FIXED KL BUDGET (0.1): is there a capacity limit on
# how many neuron signals the adapter can carry, once it cannot buy capacity by
# changing the model's behaviour?
#
# Same design as train_single_mlp_introspector_neuron_generalization.sh (row table,
# GROUP/RUN selectors, pool cache, nested neuron sets from offset 0, run_spec.json
# sidecar). The two changes:
#   --kl-anchor-target 0.1   the anchored budget used by the main grid's kl0.1 arm
#   LoRA rank swept {1,2,4,8,16}, small enough that rank can bind
# Rank 16 is the unconstrained neurongen rank, so the r16 rows pair one-for-one with
# the existing klinf runs (scaling armA n1/n3/n10 and data_n30_30k).
#
# NOTE ON RANK: LoRA is all-linear, so rank r is a rank-r update in EVERY module of
# EVERY layer -- not an r-dimensional transport channel. The channel can also reuse
# directions the base model already writes. No knee even at r=1 is a real answer.
#
# CACHES: the cache key is the data fields (neuron set, text counts), not KL or rank,
# so these rows should reuse the klinf runs' caches (n1 @300k = armA_n1, n3 @100k =
# armA_n3, n10 @30k = armA_n10, n30 @30k = data_n30_30k). If a cache is missing, the
# five rank rows of that N would all try to build it at once: check
# `ls data/` for it first, or submit one row of that N alone and the rest after.
#
# SUBMITTING
# SCC's qsub rejects -t on the command line and does not export the caller's
# environment, so the array size lives in the header and the group is an ARGUMENT:
#   qsub <script> kl_rank          header #$ -t 1-15   N = 1, 3, 10 at 300k total rows
#   qsub <script> kl_rank_n30      header #$ -t 1-5    N = 30 @ 30k/neuron = 900k rows,
#                                  3x the longest 300k run: also raise h_rt (~36h)
#   RUN=kl0.1_n10_30k_r1 bash <script>                 one row by name (interactive)
# Runs land in outputs/runs/mlp_neurongen next to the klinf runs; run_spec.json carries
# kl_budget, which plotters/plot_neuron_generalization.py uses to keep them out of the
# unconstrained baselines and to draw kl_capacity.png.
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
export WANDB_RUN_GROUP=mlp_neurongen

# stop on the first failure: without this a failed config generation silently falls
# through to training on a stale config from a previous run
set -e

# ---- fixed settings ---------------------------------------------------------
LAYER=4                  # same layer as the length-gen experiment and the main grid
TOKEN_POS=-1             # the easiest cell: this experiment varies the NEURON axis
N_CACHE=200              # neurons in the POOL cache: trainable prefix + held-out rest
N_POOL_TEST=3000         # texts in the pool cache's test split (rows = this x 200)
SEED=21                  # same sampler seed as the layer/token grid

GATE=off
W=0.01
SCALE=10
LR=5e-5
LORA_DO=0.0
BS=32
N_TEST=3000              # texts in each run's OWN test split
# Dev is sized in ROWS. rows = texts x neurons, so a flat text count made dev 100x
# more expensive at N=100 than at N=1. The 50-text FLOOR keeps enough distinct contexts
# once N is large. The in-training dev metric is POOLED over rows, not per-neuron, so
# what stabilises it is total rows.
DEV_ROWS=2000
DEV_MIN_TEXTS=50
MIN_LEN=""               # no length floor at pos=-1; a floor would make this cache
                         # incomparable to the other pos=-1 runs
PROMPT_TYPE=prompt_no_token_index
KL_BUDGET=0.1            # the one objective change from the klinf neurongen runs
KL_TAG=kl${KL_BUDGET}

# set to --overwrite-cache to force config/dataset/run regeneration.
# LEAVE EMPTY for a table submission: it would retrain every completed row.
OVERWRITE=""

# ---- the table ---------------------------------------------------------------
#      name                         group        neurons offset rows/n  epochs lora_r
RUNS=(
  # N = 1, 3, 10 at 300k TOTAL rows, as in the klinf scaling group (1-15)
  "kl0.1_n1_300k_r1               kl_rank        1      0   300000   1      1"
  "kl0.1_n1_300k_r2               kl_rank        1      0   300000   1      2"
  "kl0.1_n1_300k_r4               kl_rank        1      0   300000   1      4"
  "kl0.1_n1_300k_r8               kl_rank        1      0   300000   1      8"
  "kl0.1_n1_300k_r16              kl_rank        1      0   300000   1     16"
  "kl0.1_n3_100k_r1               kl_rank        3      0   100000   1      1"
  "kl0.1_n3_100k_r2               kl_rank        3      0   100000   1      2"
  "kl0.1_n3_100k_r4               kl_rank        3      0   100000   1      4"
  "kl0.1_n3_100k_r8               kl_rank        3      0   100000   1      8"
  "kl0.1_n3_100k_r16              kl_rank        3      0   100000   1     16"
  "kl0.1_n10_30k_r1               kl_rank       10      0    30000   1      1"
  "kl0.1_n10_30k_r2               kl_rank       10      0    30000   1      2"
  "kl0.1_n10_30k_r4               kl_rank       10      0    30000   1      4"
  "kl0.1_n10_30k_r8               kl_rank       10      0    30000   1      8"
  "kl0.1_n10_30k_r16              kl_rank       10      0    30000   1     16"
  # N = 30 at 30k PER NEURON (900k rows) -- above the data floor; long jobs (16-20)
  "kl0.1_n30_30k_r1               kl_rank_n30   30      0    30000   1      1"
  "kl0.1_n30_30k_r2               kl_rank_n30   30      0    30000   1      2"
  "kl0.1_n30_30k_r4               kl_rank_n30   30      0    30000   1      4"
  "kl0.1_n30_30k_r8               kl_rank_n30   30      0    30000   1      8"
  "kl0.1_n30_30k_r16              kl_rank_n30   30      0    30000   1     16"
)

# ---- pick this task's row ------------------------------------------------------
# GROUP filters the table first; RUN selects by name; otherwise SGE_TASK_ID indexes.
# The group comes from the FIRST SCRIPT ARGUMENT (`qsub <script> <group>`): SGE passes
# script arguments to the job, but NOT the submitting shell's environment, so
# `GROUP=x qsub <script>` silently runs with GROUP unset and indexes the whole table.
# The env var still works for `qsub -v GROUP=x <script>` and for interactive `bash`.
GROUP=${1:-$GROUP}
echo "selection: GROUP='${GROUP:-<all rows>}' RUN='${RUN:-}' task=${SGE_TASK_ID:-}"
SELECTED=()
for row in "${RUNS[@]}"; do
  read -r r_name r_group _ <<< "$row"
  if [ -n "$GROUP" ] && [ "$r_group" != "$GROUP" ]; then continue; fi
  if [ -n "$RUN" ]   && [ "$r_name"  != "$RUN"   ]; then continue; fi
  SELECTED+=("$row")
done
if [ ${#SELECTED[@]} -eq 0 ]; then
  echo "ERROR: no rows match GROUP='${GROUP:-}' RUN='${RUN:-}'" >&2; exit 1
fi
if [ -n "$RUN" ]; then
  ROW=${SELECTED[0]}
else
  if [ -z "$SGE_TASK_ID" ] || [ "$SGE_TASK_ID" = "undefined" ]; then
    echo "ERROR: SGE_TASK_ID unset; set '#$ -t 1-${#SELECTED[@]}' in the header, or RUN=<name> for one row" >&2
    exit 1
  fi
  index=$(($SGE_TASK_ID-1))
  if [ "$index" -ge ${#SELECTED[@]} ]; then
    echo "ERROR: task $SGE_TASK_ID exceeds the ${#SELECTED[@]} rows selected" >&2; exit 1
  fi
  ROW=${SELECTED[$index]}
fi
read -r NAME R_GROUP N_NEURONS OFFSET RPN EPOCHS LORA_R <<< "$ROW"
LORA_ALPHA=$((LORA_R * 2))

# rows = texts x neurons, so rows-per-neuron IS the text count
N_TEXTS=$RPN
N_ROWS=$((RPN * N_NEURONS))
DEV_TEXTS=$((DEV_ROWS / N_NEURONS))
if [ "$DEV_TEXTS" -lt "$DEV_MIN_TEXTS" ]; then DEV_TEXTS=$DEV_MIN_TEXTS; fi

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi
if [ -n "$MIN_LEN" ]; then MIN_FLAG="--min-seq-length $MIN_LEN"; else MIN_FLAG=""; fi

EXP_GROUP=mlp_neurongen
EXP_NAME="neurongen_${NAME}"
CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
echo "row: name=$NAME group=$R_GROUP neurons=$N_NEURONS offset=$OFFSET rows/n=$RPN" \
     "epochs=$EPOCHS lora_r=$LORA_R -> texts=$N_TEXTS rows=$N_ROWS dev_texts=$DEV_TEXTS"

# ---- step 1: the shared POOL cache (test split only) -----------------------
# All 200 neurons over N_POOL_TEST texts. Every task generates the identical config, so
# it resolves to ONE canonical dataset name and is built once; the rest no-op.
# Train/dev are stubbed to 64 texts -- nothing trains here, but the neuron-gen eval
# reads them to fit the injection distribution for held-out neurons.
SAMPLER="{\"type\":\"uniform_layer_sampler\",\"signal_type\":\"mlp\",\"layers\":[${LAYER}],\"signals_per_layer\":${N_CACHE},\"seed\":${SEED}}"
POOL_CONFIG=experiment_configs/$EXP_GROUP/_pool_L${LAYER}_POS${TOKEN_POS}.json

python data_prep/generate_experiment_config.py \
    --experiment-name "_pool_L${LAYER}_POS${TOKEN_POS}" \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 --train-dataset-split "train" \
    --train-max-samples 64 \
    --dev-max-samples 64 \
    --dev-offset-from-train 2000 \
    --test-dataset allenai/c4 --test-dataset-split "validation" \
    --test-max-samples $N_POOL_TEST \
    --max-seq-length 32 $MIN_FLAG \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "$SAMPLER" \
    --test-signals "$SAMPLER" \
    --same-train-test-signals \
    --train-token-selector=$TOKEN_POS \
    --test-token-selector=$TOKEN_POS \
    --prompt-type $PROMPT_TYPE \
    --precision 3 \
    --cache-dir data --output-dir outputs/runs/$EXP_GROUP \
    --output "$POOL_CONFIG"

# the cached neuron list, in sampler order. The slice below depends on this order being
# stable, which the per-(seed,layer) sampler guarantees.
ALL_NEURONS=($(python -c "
import json
cfg = json.load(open('$POOL_CONFIG'))
print(' '.join(f\"N{s['layer']}.{s['index']}\" for s in cfg['train_signals']))
"))

python data_prep/cache_signals.py --config "$POOL_CONFIG" $OVERWRITE

# ---- step 2: this row's run ------------------------------------------------
# Slice [OFFSET, OFFSET+N_NEURONS) of the pool. offset=0 gives the nested prefixes the
# scaling/interference readouts rely on; offset=k with one neuron isolates pool neuron k.
if [ $((OFFSET + N_NEURONS)) -gt ${#ALL_NEURONS[@]} ]; then
  echo "ERROR: offset $OFFSET + neurons $N_NEURONS exceeds the pool (${#ALL_NEURONS[@]})" >&2
  exit 1
fi
TRAIN_NEURONS=$(IFS=,; echo "${ALL_NEURONS[*]:$OFFSET:$N_NEURONS}")

# This run's OWN cache holds exactly its neurons over N_TEXTS texts. The held-out
# neurons come from the pool cache at eval time (--neuron-gen-pool-config).
python data_prep/generate_experiment_config.py \
    --experiment-name "$EXP_NAME" \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 --train-dataset-split "train" \
    --train-max-samples $N_TEXTS \
    --dev-max-samples $DEV_TEXTS \
    --dev-dataset-split "validation" \
    --dev-offset-from-train 8000 \
    --test-dataset allenai/c4 --test-dataset-split "validation" \
    --test-max-samples $N_TEST \
    --max-seq-length 32 $MIN_FLAG \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "$TRAIN_NEURONS" \
    --test-signals "$TRAIN_NEURONS" \
    --same-train-test-signals \
    --train-token-selector=$TOKEN_POS \
    --test-token-selector=$TOKEN_POS \
    --prompt-type $PROMPT_TYPE \
    --precision 3 \
    --train-batch-size $BS \
    --eval-batch-size 128 \
    --num-epochs $EPOCHS \
    --lora-r $LORA_R --lora-alpha $LORA_ALPHA --lora-dropout $LORA_DO \
    $GATE_FLAG \
    --bf16 \
    --learning-rate $LR \
    --use-regressor \
    --intervention-signal-distribution extreme \
    --intervention-extreme-scale $SCALE \
    --cache-dir data \
    --output-dir outputs/runs/$EXP_GROUP \
    --output "$CONFIG" \
    $OVERWRITE

# The row, as data, next to the results. The plotter reads this rather than parsing
# directory names, so the naming can stay readable and legacy runs get the same fields.
RUN_DIR=outputs/runs/$EXP_GROUP/$EXP_NAME
mkdir -p "$RUN_DIR"
python - <<PY
import json
json.dump({"name": "$NAME", "group": "$R_GROUP", "neurons": $N_NEURONS, "offset": $OFFSET,
           "rows_per_neuron": $RPN, "rows": $N_ROWS, "epochs": $EPOCHS, "lora_r": $LORA_R, "kl_budget": $KL_BUDGET,
           "layer": $LAYER, "token_pos": $TOKEN_POS, "pool_size": $N_CACHE,
           "train_neurons": "$TRAIN_NEURONS".split(",")},
          open("$RUN_DIR/run_spec.json", "w"), indent=2)
PY

### cache this run's OWN dataset (no-op when a row with the same content built it)
python data_prep/cache_signals.py --config "$CONFIG" $OVERWRITE

python trainers/train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    --kl-anchor-target $KL_BUDGET \
    --force-drift-eval \
    --neuron-gen-eval \
    --neuron-gen-pool-config "$POOL_CONFIG" \
    $OVERWRITE

# Re-run ONLY the neuron-generalization eval on an existing checkpoint:
#   python trainers/train_introspector_joint.py --config "$CONFIG" --joint-loss-weight $W \
#       --kl-anchor-target $KL_BUDGET --neuron-gen-only --neuron-gen-pool-config "$POOL_CONFIG"
