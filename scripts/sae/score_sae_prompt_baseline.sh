#!/bin/bash -l
#$ -P llamagrp
#$ -N sae_prompt_baseline
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=12:00:00
#$ -o log
#$ -j y
#$ -t 1-27

# ============================================================================
# PROMPTED BASELINE for SAE detection (evals/score_sae_prompt_baseline.py): the UNTRAINED
# gemma-2-2b asked with the trained runs' prompts, over every (feature, position) of
#   - the layer x position grid   (sweep_features.txt: 4 layers x 4 positions x 3 features = 48)
#   - the layer-9 reference runs  (5 judged features at position -1)
# = 53 runs per prompt arm. Nothing is trained, so the grid's two W arms share one result
# and only the w0.9 configs are enumerated. Results: outputs/runs/sae_prompt_baseline/<PROMPT>/.
#
#   PROMPT=index  the trained grid's own prompt (feature named by its index)
#   PROMPT=label  the feature named by its Neuronpedia explanation, as in the feature-
#                 generalization label arm (scripts/sae/sweep_feature_labels.json)
#
# Configs come from the grid's own make_config (sweep_common.sh) and from the layer-9
# script's flags, so every baseline reads exactly the cache and test split its trained
# runs were scored on.
#
#   qsub -N sae_pb_cache -v STAGE=cache -t 1-17 scripts/sae/score_sae_prompt_baseline.sh
#   qsub -hold_jid sae_pb_cache -v PROMPT=index -t 1-27 scripts/sae/score_sae_prompt_baseline.sh
#   qsub -hold_jid sae_pb_cache -v PROMPT=label -t 1-27 scripts/sae/score_sae_prompt_baseline.sh
#
# Stage `cache` (17 tasks): task 1-16 builds one grid (layer, position) cell -- one cache
# holds the cell's 3 features -- and task 17 the layer-9 cache (5 features). ~15 min each.
# Stage `score`: the 53 runs sharded over the array ($SGE_TASK_ID/$SGE_TASK_LAST), model
# loaded once per task; --skip-existing makes a resubmission finish only what is missing.
# Each run: alignment check, verbal@k + logit + probe@k for k in {0,1,2,4,8,16} on the full
# endogenous test split and the interventional (on/off) test, the no-prompt control,
# the probe learning curve over active rows, and the steering-scale sweep.
#
# Aggregate: python plotters/plot_sae_prompt_baseline.py
# ============================================================================

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID / $SGE_TASK_LAST"
echo "=========================================================="
nvidia-smi

module load cuda/11.8
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

set -e
STAGE=${STAGE:-score}
PROMPT=${PROMPT:-index}
LABELS=scripts/sae/sweep_feature_labels.json
source scripts/sae/sweep_common.sh     # LAYERS, POSITIONS, features_for_layer, make_config

L9_FEATURES=(S9.1568 S9.4854 S9.4120 S9.978 S9.12408)    # train_sae_joint_layer9.sh
L9_GROUP=sae_l9_joint

# make_l9_config SIG -> config path; the flags of train_sae_joint_layer9.sh at W=0.9
make_l9_config() {
    local SIG=$1
    local LIST; LIST=$(IFS=,; echo "${L9_FEATURES[*]}")
    local CONFIG=experiment_configs/$L9_GROUP/saeJoint_${SIG}_POS-1_gateoff_w09.json
    rm -f "$CONFIG"
    python data_prep/generate_experiment_config.py \
        --experiment-name "saeJoint_${SIG}_POS-1_gateoff_w09" \
        --model google/gemma-2-2b \
        --objective detection --detection-balance-pos-frac 0.5 \
        --intervention-signal-distribution extreme --intervention-extreme-scale 1 \
        --train-dataset allenai/c4 --train-dataset-split "train" --train-max-samples 150000 \
        --dev-max-samples 2000 --dev-offset-from-train 1000000 \
        --test-dataset allenai/c4 --test-dataset-split "validation" --test-max-samples 30000 \
        --max-seq-length 32 --hf-data-kwargs '{"config": "en", "streaming": true}' \
        --train-signals "$LIST" --test-signals "$LIST" --signal-filter "$SIG" \
        --train-token-selector -1 --test-token-selector -1 --precision 3 \
        --train-batch-size 32 --eval-batch-size 128 --num-epochs 1 \
        --lora-r 16 --lora-alpha 32 --lora-dropout 0.0 --bf16 \
        --learning-rate 2e-5 --warmup-ratio 0.03 --lr-scheduler-type cosine --save-steps 0.25 \
        --cache-dir data --output-dir outputs/runs/$L9_GROUP --output "$CONFIG" >&2 || return 1
    [ -f "$CONFIG" ] || return 1
    echo "$CONFIG"
}

all_configs() {
    local L P SIG
    for L in "${LAYERS[@]}"; do
        for P in "${POSITIONS[@]}"; do
            for SIG in $(features_for_layer "$L"); do
                echo "experiment_configs/$EXP_GROUP/saeJoint_${SIG}_POS${P}_gateoff_w09.json"
            done
        done
    done
    for SIG in "${L9_FEATURES[@]}"; do
        echo "experiment_configs/$L9_GROUP/saeJoint_${SIG}_POS-1_gateoff_w09.json"
    done
}

if [ "$STAGE" = "cache" ]; then
    i=$((SGE_TASK_ID - 1))
    n_cells=$(( ${#LAYERS[@]} * ${#POSITIONS[@]} ))
    if [ "$i" -lt "$n_cells" ]; then
        L=${LAYERS[$((i / ${#POSITIONS[@]}))]}
        P=${POSITIONS[$((i % ${#POSITIONS[@]}))]}
        for SIG in $(features_for_layer "$L"); do CONFIG=$(make_config "$SIG" "$P" 0.9); done
        echo "=== grid cell layer=$L pos=$P -> $CONFIG ==="
    elif [ "$i" -eq "$n_cells" ]; then
        for SIG in "${L9_FEATURES[@]}"; do CONFIG=$(make_l9_config "$SIG"); done
        echo "=== layer-9 reference -> $CONFIG ==="
    else
        echo "task beyond the $((n_cells + 1)) caches; nothing to do"; exit 0
    fi
    python data_prep/cache_signals.py --config "$CONFIG"
    echo "Done: $(date)"; exit 0
fi

# ---- score ----
CONFIGS=($(all_configs))
for c in "${CONFIGS[@]}"; do
    [ -f "$c" ] || { echo "ERROR: $c missing; run STAGE=cache first" >&2; exit 1; }
done
LABEL_FLAGS=""
if [ "$PROMPT" = "label" ]; then LABEL_FLAGS="--labels-file $LABELS"; fi
echo "prompt=$PROMPT  runs=${#CONFIGS[@]}"
python evals/score_sae_prompt_baseline.py --configs "${CONFIGS[@]}" \
    --prompt "$PROMPT" $LABEL_FLAGS \
    --baseline-root outputs/runs/sae_prompt_baseline \
    --k-values 0,1,2,4,8,16 --no-prompt-control --skip-existing
echo "Done: $(date)"
