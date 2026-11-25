#!/bin/bash -l
#$ -P llamagrp
#$ -N cross_model_sft
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=18:00:00
#$ -o log
#$ -j y
#$ -t 1-24

# ============================================================================
# Privileged-access baseline (rebuttal W4): each model trained, SFT, to predict ANOTHER
# model's neuron from the text alone, on exactly the source's own rows.
#
#   models   llama1b (Llama-3.2-1B), qwen06b (Qwen3-0.6B-Base), qwen17b (Qwen3-1.7B-Base)
#   pairs    the 6 off-diagonal (predictor, source) pairs of the 3x3; the diagonal is
#            each model's existing SFT grid run (w=1, no KL, gate off), reused as is
#   cells    per SOURCE: early/late layer x t=31/t=4 (POS -1 / -28)
#              llama1b L2 / L8, qwen3 L4 / L11 (1/8 depth; L8 = 1/2 has no qwen twin,
#              L11 = 0.39 is the nearest grid layer)
#   signals  5 neurons per cell: llama1b the joint grid's 5 (its SFT grid has 10),
#            qwen3 its SFT grid's 5
#   -> 6 pairs x 4 cells = 24 tasks, 5 runs each, 120 runs
#
# Each run: SFT (w=1, no KL anchor, gate off) at the PREDICTOR's own LR/batch/LoRA,
# endogenous eval only. The config carries signal_model_name, which makes the trainer
# refuse every injection or cache-building eval, and cache_signals refuse the config.
# The source caches must already exist (they do: the self runs trained on them).
#
# Aggregate afterwards (CPU, seconds):
#   python plotters/plot_cross_model_prediction.py
# ============================================================================

KEYS=(llama1b qwen06b qwen17b)
SELF_GROUPS=(mlp_endo_sweep_layer_token
             mlp_endo_sweep_layer_token_qwen3_06b_hyper_tuned
             mlp_endo_sweep_layer_token_qwen3_17b_hyper_tuned)
EARLY=(2 4 4)
LATE=(8 11 11)
# (predictor, source) as indices into KEYS
PRED_IDX=(1 2 0 2 0 1)
SRC_IDX=(0 0 1 1 2 2)
POSS=(-1 -28)
LLAMA_NEURONS=" 2.1447 2.1886 2.2566 2.5949 2.6998 8.3852 8.4144 8.4956 8.5777 8.6370 "
N_PER_CELL=5
GROUP=mlp_cross_model_sft

index=$((SGE_TASK_ID - 1))
pair=$((index / 4)); cell=$((index % 4))
P=${PRED_IDX[$pair]}; S=${SRC_IDX[$pair]}
PRED_KEY=${KEYS[$P]}; SRC_KEY=${KEYS[$S]}
if [ $((cell / 2)) -eq 0 ]; then LAYER=${EARLY[$S]}; else LAYER=${LATE[$S]}; fi
POS=${POSS[$((cell % 2))]}

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID / $SGE_TASK_LAST"
echo "Predictor : $PRED_KEY   Source : $SRC_KEY   Cell : L$LAYER POS$POS"
echo "=========================================================="
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

set -e

# ---- the source's own (self) configs in this cell ----
SOURCES=()
for f in experiment_configs/${SELF_GROUPS[$S]}/endo_N${LAYER}.*_POS${POS}_gateoff_w1.0_300K.json; do
  sig=$(basename "$f" | sed -E 's/^endo_N([0-9]+\.[0-9]+)_.*/\1/')
  if [ "$SRC_KEY" = "llama1b" ] && [[ "$LLAMA_NEURONS" != *" $sig "* ]]; then continue; fi
  SOURCES+=("$f")
done
if [ ${#SOURCES[@]} -ne $N_PER_CELL ]; then
  echo "expected $N_PER_CELL source configs for $SRC_KEY L$LAYER POS$POS, found ${#SOURCES[@]}:"
  printf '  %s\n' "${SOURCES[@]}"
  exit 1
fi
# any config of the predictor's own SFT grid supplies its model and training_config
PRED_TEMPLATE=$(ls experiment_configs/${SELF_GROUPS[$P]}/endo_*_gateoff_w1.0_300K.json | head -1)
echo "predictor template: $PRED_TEMPLATE"

python data_prep/make_cross_model_configs.py \
    --source-configs "${SOURCES[@]}" \
    --predictor-config "$PRED_TEMPLATE" \
    --source-key "$SRC_KEY" --predictor-key "$PRED_KEY" \
    --group "$GROUP"

for f in "${SOURCES[@]}"; do
  stem=$(basename "$f" .json); stem=${stem#endo_}
  CONFIG=experiment_configs/$GROUP/cross_${PRED_KEY}_from_${SRC_KEY}_${stem}.json
  echo "=== $CONFIG ==="
  # no cache_signals here, ever: it would run the PREDICTOR (and refuses this config)
  python trainers/train_introspector_joint.py \
      --config "$CONFIG" \
      --joint-loss-weight 1.0
done
