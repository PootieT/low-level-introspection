#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_neurongen
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=8:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-9

# ============================================================================
# NEURON GENERALIZATION: does the readout carry neurons it was never trained on,
# and what does adding neurons cost the ones already there?
#
# TWO caches, and the split between them is the whole trick:
#
#   POOL cache (shared, built once)   N_POOL_TEST texts x N_CACHE neurons, TEST SPLIT ONLY. Its train
#       split is stubbed to 64 texts because nothing trains on it. This is what the
#       unseen arm reads: it holds every neuron, trained or not.
#   PER-CELL train cache              (TARGET_ROWS / N) texts x N neurons. Sized so every
#       cell costs the SAME number of rows, which is what "fixed total budget" means.
#
# Why not one shared cache for both. A row is one (text, neuron) pair, and max_samples
# counts TEXTS, so rows = texts x neurons. Sizing a single cache to serve the N=1 cell
# (300k texts, so that 1 neuron reaches the budget) while also holding 200 neurons gives
# 300k x 200 = 60 MILLION rows -- roughly 15 GB with text_input_ids duplicated per row --
# of which each run would use 300k. Splitting it costs 443k forward passes across the
# five cells against 300k for one, since a forward serves every neuron of a text at once,
# and drops the cache from 60M rows to about 2M.
#
# ARM A -- scaling (tasks 1-5): N = 1, 3, 10, 30, 100 neurons, TOTAL ROWS fixed.
#   Prefixes are NESTED, so neuron #1 is in every run and its fidelity can be tracked
#   as N grows. That is the interference measurement: does training neuron 100 damage
#   neuron 1? Non-nested sets would confound it with neuron heterogeneity. The prefix is
#   cut from the POOL's neuron list explicitly rather than by re-sampling at a smaller
#   signals_per_layer -- random.sample gives no guarantee that sample(pool, 10) is a
#   prefix of sample(pool, 200), so re-sampling could silently break the nesting.
#
# ARM B -- data-matched null (tasks 6-9; arm A's N=1 cell serves as its full-data point): N = 1 neuron, examples = TOTAL/{1,3,10,30,100}.
#   Arm A holds total data fixed, so per-neuron data falls as 1/N. That is deliberate --
#   it is the sharing measurement, not a confound -- but it means a falling curve in
#   arm A has two possible causes, and only this arm tells them apart:
#       arm A at N=k  ~=  arm B at TOTAL/k   -> each neuron trained privately; data is
#                                               NOT shared across neurons, so no reason
#                                               to expect unseen neurons to work
#       arm A at N=k  >>  arm B at TOTAL/k   -> examples for one neuron help the others,
#                                               i.e. a shared code
#   Without arm B a fall in arm A reads as "capacity limit" when it may be plain
#   starvation. Arm B is cheap: less data per run, and only 1 neuron.
#
# WHICH NUMBER TO READ
#   interventional corr on the UNSEEN arm is the headline. The injected value is drawn
#   independently per row, so a readout that ignores which neuron the prompt names
#   cannot report it -- it scores 0 by construction.
#   Endogenous corr must be read MACRO-AVERAGED over neurons (corr_macro), never
#   pooled: pooling neurons with different activation scales counts between-neuron
#   variance as signal and inflates the number badly at 100 neurons. And even macro,
#   a positive unseen endogenous score can just mean "unseen neurons behave like seen
#   ones, so predict a typical activation" -- which is why the interventional metric,
#   not this one, decides whether addressing generalized.
#
# EXPECTATION: unseen neurons are named by INDEX in the prompt ("neuron 5676"), so a
#   held-out neuron requires generalizing over a number never seen in training. A
#   failure here is therefore not clean evidence about introspection -- it may be plain
#   numeric generalization. The scaling curve and the interference readout are
#   interpretable regardless, which is why they are arm A rather than an afterthought.
#   If unseen fails, the follow-ups are head-only transfer (freeze LoRA, retrain only
#   the regression head on held-out neurons: separates "no channel" from "channel the
#   decoder cannot address") and the read-direction predictability probe.
#
# POSITION: pos=-1, the easiest cell, deliberately. This experiment varies the NEURON
#   axis; using a hard position would confound a neuron-generalization failure with a
#   position-generalization failure.
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

index=$(($SGE_TASK_ID-1))
#index=4
LAYER=4                  # same layer as the length-gen experiment, and in the main
                         # layer/token grid's list, so the three are comparable
TOKEN_POS=-1
N_CACHE=200              # neurons in the POOL cache: trainable prefix + held-out rest
TARGET_ROWS=300000       # (text, neuron) pairs per run -- the fixed budget for arm A
N_POOL_TEST=3000          # texts in the pool cache's test split (rows = this x 200)
SEED=21                  # same sampler seed as the layer/token grid

GATE=off
W=0.01
SCALE=10
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
N_TEST=3000              # texts in each run's OWN test split
# Dev is sized in ROWS, like the train budget. rows = texts x neurons, so a flat text
# count made dev 100x more expensive at N=100 than at N=1. The 50-text FLOOR keeps enough
# distinct contexts once N is large (the row target alone would give 20 texts at N=100).
# The in-training dev metric is POOLED over all rows -- not macro-averaged per neuron --
# so what stabilises it is TOTAL rows, not rows per neuron.
DEV_ROWS=2000
DEV_MIN_TEXTS=50
MIN_LEN=""               # no length floor: pos=-1 needs no length guarantee, and a
                         # floor would make this cache incomparable to the pos=-1 runs
PROMPT_TYPE=prompt_no_token_index   # neuron is named by index either way; the token
                                    # index clause is irrelevant at pos=-1

# ---- arms -----------------------------------------------------------------
# "n_neurons:n_train_examples"
# Arm A's N=1 cell IS arm B's full-data cell (1 neuron, the whole budget), so it is
# listed once and shared: with a fixed seed a second copy would be bit-identical, not a
# noise estimate. Arm B therefore has 4 tasks, not 5.
CELLS=(
  # arm A: scaling, total ROWS fixed
  "1:$TARGET_ROWS"  "3:$TARGET_ROWS"  "10:$TARGET_ROWS"  "30:$TARGET_ROWS"  "100:$TARGET_ROWS"
  # arm B: data-matched null, 1 neuron at arm A's per-neuron budget
  "1:$((TARGET_ROWS/3))"  "1:$((TARGET_ROWS/10))"  "1:$((TARGET_ROWS/30))"  "1:$((TARGET_ROWS/100))"
)
ARMS=(A A A A A B B B B)

CELL=${CELLS[$index]}
ARM=${ARMS[$index]}
N_NEURONS=${CELL%%:*}
N_ROWS=${CELL##*:}
# rows = texts x neurons, and --train-max-samples counts TEXTS
N_TEXTS=$((N_ROWS / N_NEURONS))
DEV_TEXTS=$((DEV_ROWS / N_NEURONS))
if [ "$DEV_TEXTS" -lt "$DEV_MIN_TEXTS" ]; then DEV_TEXTS=$DEV_MIN_TEXTS; fi

# set to --overwrite-cache to force config/dataset/run regeneration
OVERWRITE="--overwrite-cache"

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi
if [ -n "$MIN_LEN" ]; then MIN_FLAG="--min-seq-length $MIN_LEN"; else MIN_FLAG=""; fi

EXP_GROUP=mlp_neurongen

# ---- step 1: the shared POOL cache (test split only) -----------------------
# All 200 neurons over N_POOL_TEST texts. Every task generates the identical config, so
# it resolves to ONE canonical dataset name and is built once; the rest no-op.
# Train/dev are stubbed to 64 texts -- nothing trains here, and at 200 neurons every
# extra text costs 200 rows.
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

# the full cached neuron list, in sampler order -- the prefix structure below depends
# on this order being stable, which the per-(seed,layer) sampler guarantees
ALL_NEURONS=($(python -c "
import json
cfg = json.load(open('$POOL_CONFIG'))
print(' '.join(f\"N{s['layer']}.{s['index']}\" for s in cfg['train_signals']))
"))

python cache_signals.py --config "$POOL_CONFIG" $OVERWRITE

# ---- step 2: this task's run ----------------------------------------------
# NESTED prefix of the train pool: run with N neurons trains on the first N, so every
# run shares neuron #1 and the interference readout is meaningful.
TRAIN_NEURONS=$(IFS=,; echo "${ALL_NEURONS[*]:0:$N_NEURONS}")

EXP_NAME="neurongen_arm${ARM}_L${LAYER}_n${N_NEURONS}_r${N_ROWS}_gate${GATE}_w${W}"
CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
echo "task=$SGE_TASK_ID arm=$ARM neurons=$N_NEURONS rows=$N_ROWS texts=$N_TEXTS dev_texts=$DEV_TEXTS -> $EXP_NAME"

# This run's OWN cache holds exactly its N neurons over N_TEXTS texts, so it is already
# the right size and needs no --signal-filter and no row cap. The held-out neurons come
# from the pool cache at eval time (--neuron-gen-pool-config).
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
    --num-epochs 1 \
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

### cache this run's OWN dataset
# Required since each cell now has its own cache (its N neurons over N_TEXTS texts).
# In the previous design every run shared the pool cache via --signal-filter, so caching
# the pool alone was enough; it no longer is.
python cache_signals.py --config "$CONFIG" $OVERWRITE

python train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    --force-drift-eval \
    --neuron-gen-eval \
    --neuron-gen-pool-config "$POOL_CONFIG" \
    $OVERWRITE

# Re-run ONLY the neuron-generalization eval on an existing checkpoint:
 python train_introspector_joint.py --config "$CONFIG" --joint-loss-weight $W \
     --neuron-gen-only --neuron-gen-pool-config "$POOL_CONFIG"
