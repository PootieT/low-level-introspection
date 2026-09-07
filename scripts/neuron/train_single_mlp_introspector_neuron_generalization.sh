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
# One (layer, position) cell is cached ONCE with 200 neurons -- one forward pass per
# text yields every neuron's activation, so the marginal cost of the 200th is nothing.
# The first 100 are the TRAIN pool, the last 100 are held out and never trained. Every
# run trains on a prefix of the train pool and is evaluated on both arms off that same
# cached test split (no second dataset, no extra caching).
#
# ARM A -- scaling (tasks 1-5): N = 1, 3, 10, 30, 100 neurons, TOTAL examples fixed.
#   Prefixes are NESTED, so neuron #1 is in every run and its fidelity can be tracked
#   as N grows. That is the interference measurement: does training neuron 100 damage
#   neuron 1? Non-nested sets would confound it with neuron heterogeneity.
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

LAYER=4                  # same layer as the length-gen experiment, and in the main
                         # layer/token grid's list, so the three are comparable
TOKEN_POS=-1
N_CACHE=200              # 100 train pool + 100 held out, cached once
N_POOL=100               # first N_POOL of the cached set are trainable
SEED=21                  # same sampler seed as the layer/token grid

GATE=off
W=0.01
SCALE=10
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
N_TOTAL=300000           # total TRAIN examples, held fixed across arm A
N_TEST=3000
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
  # arm A: scaling, total data fixed
  "1:$N_TOTAL"  "3:$N_TOTAL"  "10:$N_TOTAL"  "30:$N_TOTAL"  "100:$N_TOTAL"
  # arm B: data-matched null, 1 neuron at arm A's per-neuron budget
  "1:$((N_TOTAL/3))"  "1:$((N_TOTAL/10))"  "1:$((N_TOTAL/30))"  "1:$((N_TOTAL/100))"
)
ARMS=(A A A A A B B B B)

CELL=${CELLS[$index]}
ARM=${ARMS[$index]}
N_NEURONS=${CELL%%:*}
N_TRAIN=${CELL##*:}

# set to --overwrite-cache to force config/dataset/run regeneration
OVERWRITE=""

if [ "$GATE" = "on" ]; then GATE_FLAG="--disable-adapter-in-input"; else GATE_FLAG=""; fi
if [ -n "$MIN_LEN" ]; then MIN_FLAG="--min-seq-length $MIN_LEN"; else MIN_FLAG=""; fi

EXP_GROUP=mlp_neurongen

# ---- step 1: one shared 200-neuron cache for the cell ----------------------
# Every run passes the SAME 200-signal sampler, so all 10 tasks resolve to ONE canonical
# dataset name and the cache is built once. --signal-filter selects each run's subset
# and is deliberately NOT part of the cache identity.
SAMPLER="{\"type\":\"uniform_layer_sampler\",\"signal_type\":\"mlp\",\"layers\":[${LAYER}],\"signals_per_layer\":${N_CACHE},\"seed\":${SEED}}"
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_L${LAYER}_POS${TOKEN_POS}.json

python generate_experiment_config.py \
    --experiment-name "_cell_L${LAYER}_POS${TOKEN_POS}" \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 --train-dataset-split "train" \
    --train-max-samples $N_TOTAL \
    --dev-max-samples 200 \
    --dev-dataset-split "validation" \
    --dev-offset-from-train 8000 \
    --test-dataset allenai/c4 --test-dataset-split "validation" \
    --test-max-samples $N_TEST \
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
    --output "$CELL_CONFIG"

# the full cached neuron list, in sampler order -- the prefix structure below depends
# on this order being stable, which the per-(seed,layer) sampler guarantees
ALL_NEURONS=($(python -c "
import json
cfg = json.load(open('$CELL_CONFIG'))
print(' '.join(f\"N{s['layer']}.{s['index']}\" for s in cfg['train_signals']))
"))
ALL_LIST=$(IFS=,; echo "${ALL_NEURONS[*]}")

python cache_signals.py --config "$CELL_CONFIG" $OVERWRITE

# ---- step 2: this task's run ----------------------------------------------
# NESTED prefix of the train pool: run with N neurons trains on the first N, so every
# run shares neuron #1 and the interference readout is meaningful.
TRAIN_NEURONS=$(IFS=,; echo "${ALL_NEURONS[*]:0:$N_NEURONS}")

EXP_NAME="neurongen_arm${ARM}_L${LAYER}_n${N_NEURONS}_d${N_TRAIN}_gate${GATE}_w${W}"
CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
echo "task=$SGE_TASK_ID arm=$ARM neurons=$N_NEURONS train_examples=$N_TRAIN -> $EXP_NAME"

# --train-signals is the FULL cached list so the cache identity matches the cell config;
# --signal-filter is what actually restricts this run to its N neurons.
python generate_experiment_config.py \
    --experiment-name "$EXP_NAME" \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 --train-dataset-split "train" \
    --train-max-samples $N_TOTAL \
    --dev-max-samples 200 \
    --dev-dataset-split "validation" \
    --dev-offset-from-train 8000 \
    --test-dataset allenai/c4 --test-dataset-split "validation" \
    --test-max-samples $N_TEST \
    --max-seq-length 32 $MIN_FLAG \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "$ALL_LIST" \
    --test-signals "$ALL_LIST" \
    --same-train-test-signals \
    --signal-filter "$TRAIN_NEURONS" \
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

# max-train-examples caps the TRAIN rows for arm B without rebuilding the cache
python train_introspector_joint.py \
    --config "$CONFIG" \
    --joint-loss-weight $W \
    --force-drift-eval \
    --max-train-examples $N_TRAIN \
    --neuron-gen-eval \
    $OVERWRITE

# Re-run ONLY the neuron-generalization eval on an existing checkpoint:
#   python train_introspector_joint.py --config "$CONFIG" --joint-loss-weight $W \
#       --neuron-gen-only
