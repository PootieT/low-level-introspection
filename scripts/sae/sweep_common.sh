#!/bin/bash
# ============================================================================
# Shared grid + config generation for the layer x position SAE sweep. Sourced by
# cache_sae_sweep.sh and train_sae_sweep.sh (not run directly).
#
# Grid: LAYERS x POSITIONS x FEATURE RANK (3 per layer, from sweep_features.txt) x WEIGHTS.
# Everything except the readout layer/position/feature matches train_sae_joint_layer9.sh,
# plus --min-seq-length 32 so a fixed absolute position is never clamped to an earlier
# token on a short text (the cacher falls back to the last token otherwise).
# ============================================================================
# the collaborator's cluster checkout; elsewhere (SCC) callers run from their repo root
if [ -d /work/neu/p2026_0061_neu/low-level-introspection ]; then
    cd /work/neu/p2026_0061_neu/low-level-introspection
fi

LAYERS=(${LAYERS_OVERRIDE:-5 10 15 20})
POSITIONS=(${POSITIONS_OVERRIDE:-8 16 24 31})
WEIGHTS=(${WEIGHTS:-1.0 0.9})
N_FEATS=3
FEATURES_FILE=${FEATURES_FILE:-scripts/sae/sweep_features.txt}
EXP_GROUP=${EXP_GROUP:-sae_layerpos_joint}
SEQ_LEN=32
TRAIN_SAMPLES=${TRAIN_SAMPLES:-150000}
LR=2e-5; LORA_R=16; LORA_ALPHA=$((LORA_R * 2)); BS=32
KL_TARGET=${KL_TARGET:-0.1}

N_L=${#LAYERS[@]}; N_P=${#POSITIONS[@]}; N_W=${#WEIGHTS[@]}

# features_for_layer L -> "S<L>.a S<L>.b S<L>.c" (ranked), or exit if missing / TODO.
features_for_layer() {
    local line
    line=$(grep -E "^[[:space:]]*$1[[:space:]]" "$FEATURES_FILE" | head -1)
    [ -n "$line" ] || { echo "no line for layer $1 in $FEATURES_FILE" >&2; exit 1; }
    if echo "$line" | grep -q TODO; then
        echo "layer $1 still has TODO features in $FEATURES_FILE" >&2; exit 1
    fi
    local feats; feats=$(echo "$line" | awk '{print $2, $3, $4}')
    for f in $feats; do
        [[ "$f" =~ ^S$1\.[0-9]+$ ]] || { echo "bad feature '$f' for layer $1 (want S$1.<index>)" >&2; exit 1; }
    done
    echo "$feats"
}

# make_config SIG POS W -> prints the config path. The signal list is ALL 3 features
# of the layer in file order, so every run of one (layer, position) cell resolves to the
# same dataset_name and shares one cache. The config is regenerated every time, so edits
# to sweep_features.txt or the settings above are never masked by a stale file. Callers
# use CONFIG=$(make_config ...), where set -e does not apply inside the function, so
# every step returns explicitly on failure.
make_config() {
    local SIG=$1 POS=$2 W=$3
    local LAYER=${SIG#S}; LAYER=${LAYER%%.*}
    local FEATURE_LIST; FEATURE_LIST=$(features_for_layer "$LAYER") || return 1
    FEATURE_LIST=$(echo "$FEATURE_LIST" | tr ' ' ',')
    local WTAG; WTAG=$(echo "$W" | tr -d '.')
    local EXP_NAME="saeJoint_${SIG}_POS${POS}_gateoff_w${WTAG}"
    local CONFIG=experiment_configs/$EXP_GROUP/$EXP_NAME.json
    rm -f "$CONFIG"
    python data_prep/generate_experiment_config.py \
        --experiment-name "$EXP_NAME" \
        --model google/gemma-2-2b \
        --objective detection --detection-balance-pos-frac 0.5 \
        --intervention-signal-distribution extreme --intervention-extreme-scale 1 \
        --train-dataset allenai/c4 --train-dataset-split "train" --train-max-samples $TRAIN_SAMPLES \
        --dev-max-samples 2000 --dev-offset-from-train 1000000 \
        --test-dataset allenai/c4 --test-dataset-split "validation" --test-max-samples 30000 \
        --max-seq-length $SEQ_LEN --min-seq-length $SEQ_LEN \
        --hf-data-kwargs '{"config": "en", "streaming": true}' \
        --train-signals "$FEATURE_LIST" --test-signals "$FEATURE_LIST" --signal-filter "$SIG" \
        --train-token-selector $POS --test-token-selector $POS --precision 3 \
        --train-batch-size $BS --eval-batch-size 128 --num-epochs 1 \
        --lora-r $LORA_R --lora-alpha $LORA_ALPHA --lora-dropout 0.0 --bf16 \
        --learning-rate $LR --warmup-ratio 0.03 --lr-scheduler-type cosine --save-steps 0.25 \
        --cache-dir data --output-dir outputs/runs/$EXP_GROUP --output "$CONFIG" >&2 || return 1
    [ -f "$CONFIG" ] || { echo "config not written: $CONFIG" >&2; return 1; }
    echo "$CONFIG"
}
