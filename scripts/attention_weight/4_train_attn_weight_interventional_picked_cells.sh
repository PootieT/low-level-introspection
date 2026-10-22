#!/bin/bash -l
#$ -P multilm
#$ -N attn_w_interv
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=24:00:00
#$ -o logs
#$ -j y
#$ -t 1-12

# ============================================================================
# STAGE 4: interventional training on the six hand-picked cells, at the W chosen by
# Stage 3, in the same two cost regimes Stage 2 used.
#
# 6 cells x {kl0.1 anchored, unanchored} = 12 array tasks, 3 heads inside each = 36 runs.
# Three and not five for the reason given in Stage 2's header: six cells at five heads no
# longer fits h_rt, and the price is a standard error sqrt(5/3) ~ 1.29x wider. All three
# stages use the same 3 so their error bars mean the same thing.
#
# WHAT AN INTERVENTION IS FOR A POST-SOFTMAX WEIGHT. A cell of a probability row cannot
# simply be overwritten -- the row would stop summing to 1, and no attention-logit row
# softmaxes to that, so the model would be asked to report a state it cannot produce.
# AttentionWeightPort renormalizes instead:
#     p'_j = t,   p'_k = p_k (1-t)/(1-p_j)
# which is exactly equal to shifting the single pre-softmax logit z_j by
# logit(t) - logit(p_j). So this is the SAME KIND of intervention the attention-logit
# experiment performed, and the two sets of numbers can be read side by side.
# tests/validate_attention_weights.py (block EQUIVALENCE) checks that equality numerically.
#
# WHERE THE INJECTED VALUES COME FROM. Not from the built-in gaussian/uniform/extreme
# branches -- all three leave (0,1), and on a cell with mean ~1e-3 gaussian draws
# negatives about half the time. The trainer installs a log-odds sampler
# (make_bounded_prob_intervention_fn) which draws u and sets t = sigmoid(u), so every
# draw is a valid probability at every scale. `--scale-sweep` is therefore in LOG-ODDS
# units; 1,3,10,30,100 (the logit experiment's) would put nearly every draw on the guard
# band.
#
# CAUTION -- THE SATURATED FRACTION IS NOT ACTUALLY REPORTED. The sampler counts clamped
# draws (sample.n_drawn / n_clamped, model.py:336-347) and model.py's docstring says the
# scale sweep reports them, but interventional_scale_sweep (trainers/train_introspector_joint.py
# :1240) never reads those counters: the CSV is only
# scale,head_corr,ceiling_corr,inj_std,best_alpha. So at a large scale this file cannot
# distinguish "the model read the value" from "the model learned the clamp". The one
# indirect clue is inj_std, which stops growing linearly with scale once the sigmoid
# compresses. Wiring the counters into the CSV is the real fix and has not been done.
#
# BOTH DISTRIBUTIONS ARE RECORDED. The headline metric forces gaussian (in log-odds
# space), and the uniform arm is scored alongside it so the result does not rest on one
# distribution choice.
#
# GATE OFF, for the same reason as Stage 3: a gated run reads nothing.
#
# Usage
#   W=0.01 qsub scripts/attention_weight/4_train_attn_weight_interventional_picked_cells.sh
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

EXP_GROUP=attn_weight_interv_picked
export WANDB_RUN_GROUP=$EXP_GROUP
OUT_ROOT=outputs/runs/$EXP_GROUP

## ---- decode (regime, cell) ----
# REGIME outer, so the two regimes of a cell are 6 tasks apart and concurrently running
# tasks touch different cells rather than racing on one cache directory.
index=$(($SGE_TASK_ID - 1))
# The SAME six cells as Stage 2, in the same order -- these reuse its caches.
CELLS=(${CELLS:-"low_near:2:31:-1"  "mid_near:7:4:-1"   "up_near:11:9:-1" \
                "low_far:2:26:-27"  "mid_far:7:14:-27"  "up_far:11:25:-27"})
REGIMES=("kl0.1" "unanchored")
REGIME=${REGIMES[$((index / 6))]}
CELL=${CELLS[$((index % 6))]}
IFS=':' read -r BAND LAYER HEAD SRC_OFFSET <<< "$CELL"

if [ "$REGIME" = "unanchored" ]; then
  KL_FLAG=""
  KL_TAG="klnone"
else
  KL_BUDGET=${REGIME#kl}
  # --require-kl-anchor: this arm's whole claim is that it held a budget, so a silently
  # dropped anchor must fail the run rather than leave a directory named kl0.1 behind.
  KL_FLAG="--kl-anchor-target $KL_BUDGET --require-kl-anchor"
  KL_TAG="kl${KL_BUDGET}"
fi

# THE OUTPUT OF STAGE 3. Do not guess it: pass W=<value> from the pareto.
W=${W:-0.01}
GATE=off
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
SCALE=4
SCALE_SWEEP=${SCALE_SWEEP:-1,2,4,8,16}
OOD_FLAG="--ood-eval"
OVERWRITE=""

TRAIN_SAMPLES=${TRAIN_SAMPLES:-60000}
DEV_SAMPLES=200
DEV_OFFSET=1000000
TEST_SAMPLES=3000
SAMPLE_TAG="$((TRAIN_SAMPLES / 1000))K"

CELL_TAG="l${LAYER}_h${HEAD}_src${SRC_OFFSET}"
echo
echo "=== STAGE 4 (cell=$CELL_TAG band=$BAND, regime=$REGIME, W=$W) ==="

HEADS=($HEAD)
for h in $(seq 0 31); do
  if [ "$h" -ne "$HEAD" ] && [ ${#HEADS[@]} -lt $N_HEADS_CACHE ]; then HEADS+=($h); fi
done
HEAD_JSON=$(IFS=,; echo "[${HEADS[*]}]")
SAMPLER="{\"type\":\"attention_layer_sampler\",\"signal_type\":\"attention_weight\",\"layers\":[${LAYER}],\"heads\":${HEAD_JSON},\"src_offsets\":[${SRC_OFFSET}],\"seed\":${SEED}}"

mkdir -p experiment_configs/$EXP_GROUP
CELL_CONFIG=experiment_configs/$EXP_GROUP/_cell_${CELL_TAG}.json
python data_prep/generate_experiment_config.py \
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
  EXP_NAME="interv_${SIG}_POS${DST_POS}_gate${GATE}_scale${SCALE}_${KL_TAG}_w${W}_${SAMPLE_TAG}"
  CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
  echo
  echo "=== [$CELL_TAG $REGIME W=$W sample $SAMPLE] $EXP_NAME ==="

  python data_prep/generate_experiment_config.py \
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
      --intervention-signal-distribution gaussian \
      --intervention-extreme-scale $SCALE \
      --cache-dir data --output-dir $OUT_ROOT \
      --output "$CONFIG" \
      $OVERWRITE

  python data_prep/cache_signals.py --config "$CONFIG" $OVERWRITE

  if [ "$SAMPLE" -eq 0 ]; then DRIFT="--force-drift-eval"; else DRIFT=""; fi

  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      --scale-sweep "$SCALE_SWEEP" \
      $KL_FLAG \
      $DRIFT \
      $OOD_FLAG \
      $OVERWRITE

  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight $W \
      $KL_FLAG \
      $OOD_FLAG \
      --ood-only \
      $OVERWRITE

  # The unanchored arm's achieved KL IS the cost-of-access result, so score two heads
  # there for an error bar; the anchored arm only needs a spot check.
  if [ "$REGIME" = "unanchored" ]; then LM_SCORE_N=2; else LM_SCORE_N=1; fi
  if [ "$SAMPLE" -lt "$LM_SCORE_N" ]; then
    python evals/score_lm_degradation.py --config "$CONFIG"
  fi
done

echo
echo "=========================================================="
echo "STAGE 4 (cell=$CELL_TAG, $REGIME) DONE. Next:"
echo "  attribution in dose mode (an injected value now exists):"
echo "  python mechanistic_analysis/attribute_attention_weight_readout.py --config <config> --mode dose"
echo "End date : $(date)"
echo "=========================================================="
