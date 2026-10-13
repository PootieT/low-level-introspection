#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_weight_sweep
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=24:00:00
#$ -o logs
#$ -j y
#$ -t 1-36

# ============================================================================
# STAGE 3: joint-loss WEIGHT (W) sweep, to fix W for the interventional runs.
#
# L = W * L_endogenous + (1 - W) * L_interventional. W = 1 is pure endogenous,
# W = 0 pure interventional; train_introspector_joint.py skips the zero-weighted forward
# entirely, so the endpoints cost what a single-objective trainer costs.
#
# 6 cells x 6 weights x 3 heads = 108 runs, one array task per (cell, W). This mirrors
# scripts/neuron/train_single_mlp_introspector_joint_weight_sweep.sh, which is how the
# neuron experiment picked W = 0.01 ("within 1 SE of the best in both non-floor cells").
#
# GATE OFF IS MANDATORY, not a default. With the gate on, no trainable parameter touches
# the computation at the introspected token, so the injected value is routed downstream
# at fixed base-model gain and the interventional correlation is ~0 everywhere -- the
# sweep would compare six identical floors.
#
# THREE HEADS PER CELL because one cell gives no within-cell replicate. The neuron sweep
# measured run-to-run spread at about +-0.1, which is larger than the W differences being
# chosen between; the spread across the 3 heads is the error bar on every point. The neuron
# sweep used five; three is what six cells fit into, at the cost of a standard error
# sqrt(5/3) ~ 1.29x wider. Stage 2 and Stage 4 use the same 3 so the error bars are
# comparable across stages.
#
# CACHES COME FROM STAGE 2. --joint-loss-weight is a trainer CLI argument and is not part
# of the dataset identity, so every W reuses the same cell dataset. Run Stage 2 FIRST:
# submitting both together would race several tasks into building one cache directory.
#
# Usage
#   qsub scripts/attention_weight/3_train_attn_weight_joint_weight_sweep.sh
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

module load cuda/12.8
module load miniconda/23.11.0
conda activate /projectnb/multilm/jongin/anaconda3/envs/introspection

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection

set -e

EXP_GROUP=attn_weight_joint_weight_sweep
export WANDB_RUN_GROUP=$EXP_GROUP
OUT_ROOT=outputs/runs/$EXP_GROUP

## ---- decode (cell, W) from the flat array index ----
# CELL is the outer dimension: the six W arms of one cell share that cell's cached
# dataset, so keeping them adjacent (W innermost) means concurrently running tasks touch
# DIFFERENT cells and cannot race to build the same directory.
index=$(($SGE_TASK_ID - 1))
# The SAME six cells as Stage 2, in the same order. They must match exactly: these runs
# reuse Stage 2's cached datasets, and any difference here (including the order, which
# decides which cell a task index maps to) silently builds fresh 60K-example caches
# instead.
CELLS=(${CELLS:-"low_near:2:31:-1"  "mid_near:7:4:-1"   "up_near:11:9:-1" \
                "low_far:2:26:-27"  "mid_far:7:14:-27"  "up_far:11:25:-27"})
WS=(0.0 0.01 0.03 0.1 0.3 1.0) #! Weights
CELL=${CELLS[$((index / 6))]}
W=${WS[$((index % 6))]}
IFS=':' read -r BAND LAYER HEAD SRC_OFFSET <<< "$CELL"

if [ -z "$BAND" ] || [ -z "$W" ]; then
  echo "FATAL: could not decode (cell, W) from SGE_TASK_ID=$SGE_TASK_ID" >&2
  exit 1
fi

GATE=off                 # mandatory; see the header
DST_POS=-1
SEED=21
N_HEADS_CACHE=8
N_HEADS_TRAIN=${N_HEADS_TRAIN:-3}   # the spread across these 3 is the error bar; see the header
LR=5e-5
LORA_R=16
LORA_ALPHA=$((LORA_R * 2))
LORA_DO=0.0
BS=32
SEQ=32
PRECISION=6
SCALE=4                  # log-odds units
SCALE_SWEEP=${SCALE_SWEEP:-1,2,4,8,16}
KL_BUDGET=${KL_BUDGET:-0.1}
OVERWRITE=""

TRAIN_SAMPLES=${TRAIN_SAMPLES:-60000}
DEV_SAMPLES=200
DEV_OFFSET=1000000
TEST_SAMPLES=3000
SAMPLE_TAG="$((TRAIN_SAMPLES / 1000))K"

CELL_TAG="l${LAYER}_h${HEAD}_src${SRC_OFFSET}"
echo
echo "=== STAGE 3 (cell=$CELL_TAG band=$BAND, W=$W) ============="

# Identical head set to Stage 2, so the cache is the same directory.
HEADS=($HEAD)
for h in $(seq 0 31); do
  if [ "$h" -ne "$HEAD" ] && [ ${#HEADS[@]} -lt $N_HEADS_CACHE ]; then HEADS+=($h); fi
done
HEAD_JSON=$(IFS=,; echo "[${HEADS[*]}]")
SAMPLER="{\"type\":\"attention_layer_sampler\",\"signal_type\":\"attention_weight\",\"layers\":[${LAYER}],\"heads\":${HEAD_JSON},\"src_offsets\":[${SRC_OFFSET}],\"seed\":${SEED}}"

mkdir -p experiment_configs/$EXP_GROUP
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_${CELL_TAG}.json
python generate_experiment_config.py \
    --experiment-name "_cell_${CELL_TAG}" \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 --train-dataset-split "train" --train-max-samples $TRAIN_SAMPLES \
    --dev-max-samples $DEV_SAMPLES --dev-offset-from-train $DEV_OFFSET \
    --test-dataset allenai/c4 --test-dataset-split "validation" --test-max-samples $TEST_SAMPLES \
    --max-seq-length $SEQ --min-seq-length $SEQ \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "$SAMPLER" --test-signals "$SAMPLER" --same-train-test-signals \
    --train-token-selector $DST_POS --test-token-selector $DST_POS \
    --precision $PRECISION \
    --cache-dir data --output-dir $OUT_ROOT \
    --output "$CELL_CONFIG" \
    --overwrite-cache

SIGNALS=($(python -c "
import json
cfg = json.load(open('$CELL_CONFIG'))
print(' '.join(f\"P{s['layer']}.{s['index']}@{s['src_offset']}\" for s in cfg['train_signals']))
"))
SIGNAL_LIST=$(IFS=,; echo "${SIGNALS[*]}")
echo "cell $CELL_TAG signals: $SIGNAL_LIST"

for SAMPLE in $(seq 0 $((N_HEADS_TRAIN - 1))); do
  SIG=${SIGNALS[$SAMPLE]}
  EXP_NAME="joint_${SIG}_POS${DST_POS}_gate${GATE}_scale${SCALE}_kl${KL_BUDGET}_w${W}_${SAMPLE_TAG}"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo
  echo "=== [$CELL_TAG W=$W sample $SAMPLE] $EXP_NAME ==="

  python generate_experiment_config.py \
      --experiment-name "$EXP_NAME" \
      --model meta-llama/Llama-3.2-1B \
      --train-dataset allenai/c4 --train-dataset-split "train" \
      --train-max-samples $TRAIN_SAMPLES \
      --dev-max-samples $DEV_SAMPLES --dev-offset-from-train $DEV_OFFSET \
      --test-dataset allenai/c4 --test-dataset-split "validation" \
      --test-max-samples $TEST_SAMPLES \
      --max-seq-length $SEQ --min-seq-length $SEQ \
      --hf-data-kwargs '{"config": "en", "streaming": true}' \
      --train-signals "$SIGNAL_LIST" --test-signals "$SIGNAL_LIST" \
      --signal-filter "$SIG" \
      --train-token-selector $DST_POS --test-token-selector $DST_POS \
      --precision $PRECISION \
      --train-batch-size $BS --eval-batch-size 128 --num-epochs 1 \
      --lora-r $LORA_R --lora-alpha $LORA_ALPHA --lora-dropout $LORA_DO \
      --bf16 --learning-rate $LR \
      --warmup-ratio 0.03 --lr-scheduler-type cosine --save-steps 0 \
      --use-regressor \
      --intervention-signal-distribution extreme \
      --intervention-extreme-scale $SCALE \
      --cache-dir data --output-dir $OUT_ROOT \
      --output "$CONFIG" \
      $OVERWRITE

  # Reuses Stage 2's dataset. If this rebuilds, something entered the cache identity
  # that should not have -- check before letting 36 tasks each rebuild 60K examples.
  python cache_signals.py --config "$CONFIG" $OVERWRITE

  if [ "$SAMPLE" -eq 0 ]; then DRIFT="--force-drift-eval"; else DRIFT=""; fi

  python train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      --kl-anchor-target $KL_BUDGET \
      --require-kl-anchor \
      --scale-sweep "$SCALE_SWEEP" \
      $DRIFT \
      $OVERWRITE

  # LM cost as a function of W: a weight that buys reading by damaging the model more is
  # not free, and that is the third panel of analysis/plot_weight_sweep.py.
  python score_lm_degradation.py --config "$CONFIG"
done

echo
echo "=========================================================="
echo "STAGE 3 (cell=$CELL_TAG, W=$W) DONE."
echo "When all 18 tasks finish, pick W from the pareto:"
echo "  python analysis/plot_weight_sweep.py   # ported to this EXP_GROUP"
echo "End date : $(date)"
echo "=========================================================="
