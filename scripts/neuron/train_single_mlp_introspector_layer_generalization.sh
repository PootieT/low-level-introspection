#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_layergen
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=12:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 19

# ============================================================================
# LAYER GENERALIZATION: does the readout carry a layer it was never trained on?
#
# Third axis of the locale triple (layer, position, index). ONE index is held fixed at
# every layer, so the only coordinate that varies is the layer. Unlike positions, the
# test layers can sit BETWEEN train layers -- this is interpolation, and the layer is
# named in the prompt as a number that lies between numbers the model has seen.
#
# SPLIT.  Train candidates = EVEN layers 0..14 (the main grid's layers). Test = ODD
#   layers 1..13. Layer 15 is excluded everywhere: no attention layer sits above it, so
#   a signal injected there can never reach the readout position. Unreadable by
#   construction, not a hard case.
#
# TRAIN SETS: dyadic refinement of the evens. Nested (layer 8 is in every set, so it
#   can be followed as neighbours are added) AND evenly spaced (every test layer is
#   bracketed by two train layers once N >= 2; at N=1, six of seven are extrapolation).
#       N=1  {8}     N=2  {0,8}     N=4  {0,4,8,12}     N=8  {0,2,...,14}
#
# INDEX. Chosen by scanning, not picked: index i at layer 0 and at layer 12 are
#   unrelated neurons, and an arbitrary index is near-silent somewhere -- which would
#   read as an undefined correlation at that layer, indistinguishable from a
#   generalization failure. data_prep/select_layer_gen_index.py ranks candidates by their
#   WEAKEST layer's activation std. Every task runs the scan (its pool is cached once;
#   the choice is deterministic), so the whole array agrees on the index.
#
# DATA. Total rows fixed at 300k, as in the neuron sweep. That works HERE because N
#   tops out at 8, leaving 37.5k rows/layer -- above the ~30k floor the neuron sweep
#   located. It failed there because N=30/100 pushed rows/locale to 10k/3k. Same rule,
#   different regime. Rows = texts x layers, so N=8 sees 8x fewer DISTINCT contexts
#   than N=1; rows are matched, contexts are not.
#
# CEILINGS on the seven test layers, same index, single layer each:
#   ceil37k    37.5k rows -- matched to N=8's per-layer data. "Of what this layer can
#              do with this much data, how much did generalization recover?" This is
#              the reference for the generalization claim: it separates "did not
#              generalize to layer 13" from "layer 13 is intrinsically hard". Deeper
#              layers have fewer layers above them to carry the signal, so the two are
#              otherwise inseparable. ~0.9 of one full run for all seven.
#   ceil300k   full budget: "how far from the best achievable?" Seven full runs; queue
#              these last.
#   The grid's even-layer results are NOT ceilings for the odd test layers.
#
# EVAL. One pool cache (the chosen index x 15 layers, test split + stubbed train/dev
#   for the injection stats). Every run evaluates all 15 locales in one pass and labels
#   each seen/unseen by layer -- the neuron-generalization eval, unchanged.
#
# SELECTING RUNS
#   qsub <script>                                  every row (#$ -t above)
#   GROUP=ceil37k qsub <script>   (+ #$ -t 1-7)    only that group, indexed within it
#   RUN=lg_scale_n4 bash <script>                  one row by name
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
export WANDB_RUN_GROUP=mlp_layergen

set -e

# ---- fixed settings ---------------------------------------------------------
TOKEN_POS=-1
ALL_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"      # 15 excluded: no transport above it
CANDIDATE_INDICES="5676,520,1216,6896,6386"         # seed-21 layer-4 sample; scanned below
N_POOL_TEST=3000
N_SCAN_TEST=1000         # enough to estimate a std per (layer, index)

GATE=off
W=0.01
SCALE=10
LR=5e-5
LORA_DO=0.0
BS=32
N_TEST=3000
DEV_ROWS=2000
DEV_MIN_TEXTS=50
MIN_LEN=""
PROMPT_TYPE=prompt_no_token_index
OVERWRITE=""             # leave empty for a table submission

# ---- the table ----------------------------------------------------------------
# rows/locale is the text count (k=1, so rows = texts x layers). Total rows = that x layers.
#      name              group     layers                    rows/locale  epochs lora_r
RUNS=(
  # scaling: dyadic train sets, 300k rows total
  "lg_scale_n1        scaling   8                          300000       1      16"
  "lg_scale_n2        scaling   0,8                        150000       1      16"
  "lg_scale_n4        scaling   0,4,8,12                    75000       1      16"
  "lg_scale_n8        scaling   0,2,4,6,8,10,12,14          37500       1      16"
  # ceilings on the test layers at N=8's per-layer budget
  "lg_ceil37k_L1      ceil37k   1                           37500       1      16"
  "lg_ceil37k_L3      ceil37k   3                           37500       1      16"
  "lg_ceil37k_L5      ceil37k   5                           37500       1      16"
  "lg_ceil37k_L7      ceil37k   7                           37500       1      16"
  "lg_ceil37k_L9      ceil37k   9                           37500       1      16"
  "lg_ceil37k_L11     ceil37k   11                          37500       1      16"
  "lg_ceil37k_L13     ceil37k   13                          37500       1      16"
  # ceilings at the full budget (seven full runs -- queue last)
  "lg_ceil300k_L1     ceil300k  1                          300000       1      16"
  "lg_ceil300k_L3     ceil300k  3                          300000       1      16"
  "lg_ceil300k_L5     ceil300k  5                          300000       1      16"
  "lg_ceil300k_L7     ceil300k  7                          300000       1      16"
  "lg_ceil300k_L9     ceil300k  9                          300000       1      16"
  "lg_ceil300k_L11    ceil300k  11                         300000       1      16"
  "lg_ceil300k_L13    ceil300k  13                         300000       1      16"
  # raising the ceiling: the N=4 cell at 300k rows PER LAYER (1.2M total, 4x the scaling
  # budget). Same layers as lg_scale_n4, so the two differ only in data. Shows how much
  # of the seen-layer fidelity was budget-limited, and whether more data moves the
  # unseen layers at all.
  "lg_data300k_n4     data300k  0,4,8,12                   300000       1      16"
)

# ---- pick this task's row --------------------------------------------------------
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
    echo "ERROR: SGE_TASK_ID unset; use 'qsub -t 1-${#SELECTED[@]}', or RUN=<name> for one row" >&2
    exit 1
  fi
  index=$(($SGE_TASK_ID-1))
  if [ "$index" -ge ${#SELECTED[@]} ]; then
    echo "ERROR: task $SGE_TASK_ID exceeds the ${#SELECTED[@]} rows selected" >&2; exit 1
  fi
  ROW=${SELECTED[$index]}
fi

read -r NAME R_GROUP LAYERS RPL EPOCHS LORA_R <<< "$ROW"
LORA_ALPHA=$((LORA_R * 2))
N_LAYERS=$(echo "$LAYERS" | tr ',' '\n' | wc -l | tr -d ' ')
N_TEXTS=$RPL
N_ROWS=$((RPL * N_LAYERS))
DEV_TEXTS=$((DEV_ROWS / N_LAYERS))
if [ "$DEV_TEXTS" -lt "$DEV_MIN_TEXTS" ]; then DEV_TEXTS=$DEV_MIN_TEXTS; fi

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi
if [ -n "$MIN_LEN" ]; then MIN_FLAG="--min-seq-length $MIN_LEN"; else MIN_FLAG=""; fi

EXP_GROUP=mlp_layergen
EXP_NAME="layergen_${NAME}"
CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
echo "row: name=$NAME group=$R_GROUP layers=[$LAYERS] rows/locale=$RPL epochs=$EPOCHS" \
     "lora_r=$LORA_R -> texts=$N_TEXTS rows=$N_ROWS dev_texts=$DEV_TEXTS"

# helper: "N<L>.<i>" for every layer in a list, one index
signals_for() {   # $1 = comma-separated layers, $2 = index
  local out=""; local L
  for L in $(echo "$1" | tr ',' ' '); do out="${out:+$out,}N${L}.$2"; done
  echo "$out"
}

# ---- step 1: the SCAN pool -> the index --------------------------------------------
# All candidate indices at every layer, test split only. Built once (identical config
# for every task), then ranked. Deterministic, so every task in the array agrees.
SCAN_SIGNALS=""
for i in $(echo "$CANDIDATE_INDICES" | tr ',' ' '); do
  SCAN_SIGNALS="${SCAN_SIGNALS:+$SCAN_SIGNALS,}$(signals_for "$ALL_LAYERS" "$i")"
done
SCAN_CONFIG=experiment_configs/$EXP_GROUP/_scan_POS${TOKEN_POS}.json
python data_prep/generate_experiment_config.py \
    --experiment-name "_scan_POS${TOKEN_POS}" \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 --train-dataset-split "train" \
    --train-max-samples 64 --dev-max-samples 64 --dev-offset-from-train 2000 \
    --test-dataset allenai/c4 --test-dataset-split "validation" \
    --test-max-samples $N_SCAN_TEST \
    --max-seq-length 32 $MIN_FLAG \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "$SCAN_SIGNALS" --test-signals "$SCAN_SIGNALS" --same-train-test-signals \
    --train-token-selector=$TOKEN_POS --test-token-selector=$TOKEN_POS \
    --prompt-type $PROMPT_TYPE --precision 3 \
    --cache-dir data --output-dir outputs/runs/$EXP_GROUP \
    --output "$SCAN_CONFIG"
python data_prep/cache_signals.py --config "$SCAN_CONFIG" $OVERWRITE
INDEX=$(python data_prep/select_layer_gen_index.py --config "$SCAN_CONFIG" | tail -1)
echo "chosen index: $INDEX"

# ---- step 2: the POOL cache (chosen index x all 15 layers) --------------------------
POOL_SIGNALS=$(signals_for "$ALL_LAYERS" "$INDEX")
POOL_CONFIG=experiment_configs/$EXP_GROUP/_pool_i${INDEX}_POS${TOKEN_POS}.json
python data_prep/generate_experiment_config.py \
    --experiment-name "_pool_i${INDEX}_POS${TOKEN_POS}" \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 --train-dataset-split "train" \
    --train-max-samples 64 --dev-max-samples 64 --dev-offset-from-train 2000 \
    --test-dataset allenai/c4 --test-dataset-split "validation" \
    --test-max-samples $N_POOL_TEST \
    --max-seq-length 32 $MIN_FLAG \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "$POOL_SIGNALS" --test-signals "$POOL_SIGNALS" --same-train-test-signals \
    --train-token-selector=$TOKEN_POS --test-token-selector=$TOKEN_POS \
    --prompt-type $PROMPT_TYPE --precision 3 \
    --cache-dir data --output-dir outputs/runs/$EXP_GROUP \
    --output "$POOL_CONFIG"
python data_prep/cache_signals.py --config "$POOL_CONFIG" $OVERWRITE

# ---- step 3: this row's run ----------------------------------------------------------
TRAIN_SIGNALS=$(signals_for "$LAYERS" "$INDEX")
python data_prep/generate_experiment_config.py \
    --experiment-name "$EXP_NAME" \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 --train-dataset-split "train" \
    --train-max-samples $N_TEXTS \
    --dev-max-samples $DEV_TEXTS --dev-dataset-split "validation" --dev-offset-from-train 8000 \
    --test-dataset allenai/c4 --test-dataset-split "validation" \
    --test-max-samples $N_TEST \
    --max-seq-length 32 $MIN_FLAG \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "$TRAIN_SIGNALS" --test-signals "$TRAIN_SIGNALS" --same-train-test-signals \
    --train-token-selector=$TOKEN_POS --test-token-selector=$TOKEN_POS \
    --prompt-type $PROMPT_TYPE --precision 3 \
    --train-batch-size $BS --eval-batch-size 128 \
    --num-epochs $EPOCHS \
    --lora-r $LORA_R --lora-alpha $LORA_ALPHA --lora-dropout $LORA_DO \
    $GATE_FLAG --bf16 --learning-rate $LR --use-regressor \
    --intervention-signal-distribution extreme --intervention-extreme-scale $SCALE \
    --cache-dir data --output-dir outputs/runs/$EXP_GROUP \
    --output "$CONFIG" \
    $OVERWRITE

RUN_DIR=outputs/runs/$EXP_GROUP/$EXP_NAME
mkdir -p "$RUN_DIR"
python - <<PY
import json
json.dump({"name": "$NAME", "group": "$R_GROUP", "index": $INDEX,
           "train_layers": [int(x) for x in "$LAYERS".split(",")],
           "n_layers": $N_LAYERS, "rows_per_locale": $RPL, "rows": $N_ROWS,
           "epochs": $EPOCHS, "lora_r": $LORA_R, "token_pos": $TOKEN_POS,
           "all_layers": [int(x) for x in "$ALL_LAYERS".split(",")],
           "train_signals": "$TRAIN_SIGNALS".split(",")},
          open("$RUN_DIR/run_spec.json", "w"), indent=2)
PY

python data_prep/cache_signals.py --config "$CONFIG" $OVERWRITE

python trainers/train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    --force-drift-eval \
    --neuron-gen-eval \
    --neuron-gen-pool-config "$POOL_CONFIG" \
    $OVERWRITE

# Re-run ONLY the generalization eval on an existing checkpoint:
#   python trainers/train_introspector_joint.py --config "$CONFIG" --joint-loss-weight $W \
#       --neuron-gen-only --neuron-gen-pool-config "$POOL_CONFIG"
