#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_neurongen
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=12:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-28

# ============================================================================
# NEURON GENERALIZATION: does the readout carry neurons it was never trained on,
# and what does adding neurons cost the ones already there?
#
# ONE TABLE, ONE ROW PER RUN. Every quantity that varies is an explicit column; nothing
# is derived from a row's position. To add a run, add a row and bump #$ -t.
#
#   name      unique, human-readable; becomes the run directory (neurongen_<name>)
#   group     what the plotter selects on (scaling / single / single_n10 / epochs / ...)
#             NOTE: never name a group "null" -- pandas reads that literal string back
#             from any CSV as NaN, and the row silently vanishes from every plot.
#   neurons   how many neurons this run trains on
#   offset    which slice of the POOL: neurons [offset, offset+neurons). offset=k with
#             neurons=1 trains the k-th pool neuron ALONE -- how the paired null works
#   rows/n    training rows PER NEURON. This is the variable the sharing question is
#             about, so matched-data configs read the same number in the same column.
#             Since rows = texts x neurons, this is also the text count directly.
#   epochs    passes over those rows
#   lora_r    adapter rank (alpha = 2r, the convention everywhere else)
#
# SELECTING RUNS -- three ways, so a one-off never means editing the table:
#   qsub -t 1-26 <script>                     every row, by index
#   GROUP=rank qsub -t 1-4 <script>           only rows in that group, indexed within it
#   RUN=steps_n1_30k_ep10 bash <script>       one row by name (interactive)
#
# CACHES. The POOL cache (all 200 neurons, test split) is built once. Each run's own
# train cache is keyed on its content -- neuron set and text count -- NOT on epochs or
# rank, so the epoch and rank rows reuse caches that already exist and cost only the
# training. Rows with identical (neurons, offset, rows/n) share one cache.
#
# LEGACY ROWS. The first nine reproduce the exact directory names of runs already on
# disk from the earlier arm-A/arm-B version of this script, so resubmitting them is a
# no-op (the trainer skips a completed run). They stay in the table so it is the single
# source of truth, and so they gain the run_spec.json sidecar the plotter reads.
#
# WHAT THE GROUPS ANSWER
#   scaling    seen-neuron fidelity as N grows at a FIXED total budget (300k rows). Per-
#              neuron data falls as 1/N by design -- that is the sharing measurement.
#   single     1 neuron at each of scaling's per-neuron budgets. scaling ~= single means
#              data does not transfer between neurons.
#              FOUND: single is at CHANCE at 10k and 3k rows/neuron, so the N=30 and N=100
#              scaling cells were starved, not capacity-limited. Hence the groups below.
#   single_n10 the 10 neurons of the N=10 cell, each trained ALONE at the same 30k
#              rows/neuron. Paired with the N=10 run neuron-for-neuron, it gives the
#              null a real across-neuron spread at the one budget where a gap exists.
#              Offset 0 is the legacy single row at 30k; only offsets 1-9 are new.
#   epochs     the N=10 run gets 10x the optimization STEPS of the 30k single (300k row-
#              passes vs 30k). 10 epochs on one neuron at 30k equalises steps while
#              holding distinct data at 30k -- if that closes the gap, it was steps, not
#              sharing. The N=10 variant costs 10x and answers a weaker question.
#   data30k    N=30 and N=100 at 30k rows/neuron, i.e. above the data floor the single
#              located (between 10k and 30k). 900k and 3M rows: 3x and 10x the longest
#              run so far. Run N=30 first; N=100 only if N=30 recovers.
#   rank       LoRA rank on the N=30 @ 30k cell -- the capacity hypothesis, after data.
#              Shares data30k's cache, so each row is training only.
#
# WHICH NUMBER TO READ: interventional corr on the UNSEEN arm is the headline (the
# injected value is independent per row, so a readout that ignores the named neuron
# scores 0). Endogenous must be read MACRO-averaged over neurons, never pooled.


# HOW TO RUN
# either specify with task script in the file above, or
# do this if you want to run particular runs
# qsub -t 19-21 scripts/neuron/train_single_mlp_introspector_neuron_generalization.sh
# or this if you want to run particular groups/runs
# GROUP=epochs qsub -t 1-3 scripts/neuron/train_single_mlp_introspector_neuron_generalization.sh
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

# set to --overwrite-cache to force config/dataset/run regeneration.
# LEAVE EMPTY for a table submission: it would retrain every completed row.
OVERWRITE=""

# ---- the table ---------------------------------------------------------------
#      name                               group      neurons offset rows/n  epochs lora_r
RUNS=(
  # legacy: arm A, total rows fixed at 300k (names match the runs already on disk) (1-5)
  "armA_L4_n1_r300000_gateoff_w0.01     scaling      1      0    300000   1     16"
  "armA_L4_n3_r300000_gateoff_w0.01     scaling      3      0    100000   1     16"
  "armA_L4_n10_r300000_gateoff_w0.01    scaling     10      0     30000   1     16"
  "armA_L4_n30_r300000_gateoff_w0.01    scaling     30      0     10000   1     16"
  "armA_L4_n100_r300000_gateoff_w0.01   scaling    100      0      3000   1     16"
  # legacy: arm B, one neuron at each per-neuron budget (6-9)
  "armB_L4_n1_r100000_gateoff_w0.01     single       1      0    100000   1     16"
  "armB_L4_n1_r30000_gateoff_w0.01      single       1      0     30000   1     16"
  "armB_L4_n1_r10000_gateoff_w0.01      single       1      0     10000   1     16"
  "armB_L4_n1_r3000_gateoff_w0.01       single       1      0      3000   1     16"
  # exp 1: the other nine neurons of the N=10 cell, each alone at 30k (offset 0 = legacy) (10-18)
  "null10_o1_30k                        single_n10   1      1     30000   1     16"
  "null10_o2_30k                        single_n10   1      2     30000   1     16"
  "null10_o3_30k                        single_n10   1      3     30000   1     16"
  "null10_o4_30k                        single_n10   1      4     30000   1     16"
  "null10_o5_30k                        single_n10   1      5     30000   1     16"
  "null10_o6_30k                        single_n10   1      6     30000   1     16"
  "null10_o7_30k                        single_n10   1      7     30000   1     16"
  "null10_o8_30k                        single_n10   1      8     30000   1     16"
  "null10_o9_30k                        single_n10   1      9     30000   1     16"
  # exp 2: equalise optimization steps (19-21)
  "steps_n1_o0_30k_ep10                 epochs       1      0     30000  10     16"
  "steps_n1_o1_30k_ep10                 epochs       1      1     30000  10     16"
  "steps_n1_o2_30k_ep10                 epochs       1      2     30000  10     16"
  # exp 3: N=30 / N=100 above the data floor (22-24)
  "data_n30_30k                         data30k     30      0     30000   1     16"
  "data_n100_30k                        data30k    100      0     30000   1     16"
  # exp 4: rank, on the N=30 @ 30k cell (25-28)
  "rank_n30_30k_r32                     rank        30      0     30000   1     32"
  "rank_n30_30k_r64                     rank        30      0     30000   1     64"
  "rank_n30_30k_r128                    rank        30      0     30000   1    128"
  "rank_n30_30k_r256                    rank        30      0     30000   1    256"
)

# ---- pick this task's row ------------------------------------------------------
# GROUP filters the table first; RUN selects by name; otherwise SGE_TASK_ID indexes.
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

python generate_experiment_config.py \
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

python cache_signals.py --config "$POOL_CONFIG" $OVERWRITE

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
python generate_experiment_config.py \
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
           "rows_per_neuron": $RPN, "rows": $N_ROWS, "epochs": $EPOCHS, "lora_r": $LORA_R,
           "layer": $LAYER, "token_pos": $TOKEN_POS, "pool_size": $N_CACHE,
           "train_neurons": "$TRAIN_NEURONS".split(",")},
          open("$RUN_DIR/run_spec.json", "w"), indent=2)
PY

### cache this run's OWN dataset (no-op when a row with the same content built it)
python cache_signals.py --config "$CONFIG" $OVERWRITE

python train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    --force-drift-eval \
    --neuron-gen-eval \
    --neuron-gen-pool-config "$POOL_CONFIG" \
    $OVERWRITE

# Re-run ONLY the neuron-generalization eval on an existing checkpoint:
#   python train_introspector_joint.py --config "$CONFIG" --joint-loss-weight $W \
#       --neuron-gen-only --neuron-gen-pool-config "$POOL_CONFIG"
