#!/bin/bash
# ============================================================================
# Shared table + config generation for the SAE FEATURE-GENERALIZATION experiment.
# Sourced by cache_sae_feature_generalization.sh and
# train_sae_feature_generalization.sh (not run directly). Run from the repo root.
#
# One introspector per row, trained on the first N features of the layer-9 pool
# (data/sae_featgen/pool_layer9.json, built by build_featgen_pool_layer9.sh) and
# evaluated on ALL 200 pool features -- trained ones are the seen arm, the rest the
# held-out arm (score_sae_feature_generalization.py).
#
#   arm=label   the prompt names each feature by its Neuronpedia explanation
#               (prompt_sae_label). The hypothesis: addressing by MEANING gives a
#               held-out feature something to go on, where an index gives nothing.
#   arm=index   the prompt names it by "S9.<index>" (prompt_no_token_index) -- the
#               control, expected to reproduce the neuron-gen null on held-out features.
#
# Everything else is the layer-9 joint recipe (train_sae_joint_layer9.sh): gemma-2-2b,
# position -1 of a 32-token text, gate off, w(endo)=0.9, KL budget 0.1, lr 2e-5, rank
# 16, 'extreme' scale 1 (natural-magnitude) steering, 150k C4 train texts.
#
# ROW BUDGET. TOTAL_ROWS rows for every N, split equally across features (TOTAL/N each),
# half active, half inactive (sae_featgen.balance_rows_per_feature): inactive rows are
# subsampled to the quota, active rows repeated to it. Distinct active examples per
# feature are therefore fixed by its density at every N, while distinct inactive ones
# shrink as 1/N -- state this wherever the N curve is reported. The cache keeps only
# what the trainer can use: every active row plus inactive rows at rate
# NEG_MARGIN x quota / TRAIN_TEXTS (DatasetConfig.negative_keep), which is 1.0 at N=1.
# ============================================================================

EXP_GROUP=${EXP_GROUP:-sae_featgen}
MODEL=google/gemma-2-2b
LAYER=9
TOKEN_POS=-1
SEQ_LEN=32
POOL_JSON=${POOL_JSON:-data/sae_featgen/pool_layer9.json}

TOTAL_ROWS=${TOTAL_ROWS:-300000}   # per run, at every N (fixed total, as in neuron-gen's scaling group)
TRAIN_TEXTS=150000                 # = the layer-9 single-feature runs' train split
NEG_MARGIN=1.15                    # cache this much more than the inactive quota (sampling noise)
OWN_TEST_TEXTS=500                 # a run's own test split; the eval reads the pool cache instead
POOL_TEST_TEXTS=30000              # = the single-feature runs' test split (C4 validation)
POOL_TEST_KEEP=0.1                 # inactive test rows kept: ~3k texts' worth per feature + all actives
POOL_TRAIN_TEXTS=150000            # active rows only; source of the eval's 'on' magnitudes
LABEL_PROMPT_TOKENS=96             # prompt budget for label prompts (index prompts fit in 30)

W=${W:-0.9}
KL_TARGET=${KL_TARGET:-0.1}
LR=2e-5; LORA_R=16; LORA_ALPHA=$((LORA_R * 2)); BS=32

# ---- the table: one row per run --------------------------------------------------
#      name          group   arm     n_features
RUNS=(
  # label arm (1-5): run first
  "label_n1        label   label     1"
  "label_n3        label   label     3"
  "label_n10       label   label    10"
  "label_n30       label   label    30"
  "label_n100      label   label   100"
  # index arm (6-10): the control, after the label arm
  "index_n1        index   index     1"
  "index_n3        index   index     3"
  "index_n10       index   index    10"
  "index_n30       index   index    30"
  "index_n100      index   index   100"
)

POOL_CONFIG=experiment_configs/$EXP_GROUP/_featgen_pool_L${LAYER}.json

pool_features() {   # first N pool features, comma-separated (all of them if N omitted)
    python - "$POOL_JSON" "${1:-0}" <<'PY'
import json, sys
pool = json.load(open(sys.argv[1]))
n = int(sys.argv[2]) or len(pool["features"])
if n > len(pool["features"]):
    sys.exit(f"pool has {len(pool['features'])} features, asked for {n}")
print(",".join(pool["features"][:n]))
PY
}

# select_row: GROUP filters the table, RUN picks one row by name, else SGE_TASK_ID
# indexes (within the GROUP if given). Sets NAME R_GROUP ARM N_FEAT.
select_row() {
    local selected=() row r_name r_group
    for row in "${RUNS[@]}"; do
        read -r r_name r_group _ <<< "$row"
        if [ -n "$GROUP" ] && [ "$r_group" != "$GROUP" ]; then continue; fi
        if [ -n "$RUN" ] && [ "$r_name" != "$RUN" ]; then continue; fi
        selected+=("$row")
    done
    if [ ${#selected[@]} -eq 0 ]; then
        echo "ERROR: no rows match GROUP='${GROUP:-}' RUN='${RUN:-}'" >&2; return 1
    fi
    local pick
    if [ -n "$RUN" ]; then
        pick=${selected[0]}
    else
        if [ -z "$SGE_TASK_ID" ] || [ "$SGE_TASK_ID" = "undefined" ]; then
            echo "ERROR: SGE_TASK_ID unset; use qsub -t, or RUN=<name> for one row" >&2; return 1
        fi
        local i=$((SGE_TASK_ID - 1))
        if [ "$i" -ge ${#selected[@]} ]; then
            echo "ERROR: task $SGE_TASK_ID exceeds the ${#selected[@]} rows selected" >&2; return 1
        fi
        pick=${selected[$i]}
    fi
    read -r NAME R_GROUP ARM N_FEAT <<< "$pick"
}

# make_pool_config: every pool feature. TRAIN keeps active rows only (the 'on'
# magnitudes); TEST is the eval set. Regenerated every call so edits are never masked.
make_pool_config() {
    local feats; feats=$(pool_features) || return 1
    rm -f "$POOL_CONFIG"
    python generate_experiment_config.py \
        --experiment-name "_featgen_pool_L${LAYER}" \
        --model $MODEL --objective detection \
        --train-dataset allenai/c4 --train-dataset-split "train" \
        --train-max-samples $POOL_TRAIN_TEXTS --train-negative-keep 0.0 \
        --dev-max-samples 200 --dev-offset-from-train 1000000 \
        --test-dataset allenai/c4 --test-dataset-split "validation" \
        --test-max-samples $POOL_TEST_TEXTS --test-negative-keep $POOL_TEST_KEEP \
        --max-seq-length $SEQ_LEN --hf-data-kwargs '{"config": "en", "streaming": true}' \
        --train-signals "$feats" --test-signals "$feats" \
        --train-token-selector $TOKEN_POS --test-token-selector $TOKEN_POS --precision 3 \
        --eval-batch-size 128 \
        --cache-dir data --output-dir outputs/runs/$EXP_GROUP --output "$POOL_CONFIG" >&2 || return 1
    [ -f "$POOL_CONFIG" ] || { echo "pool config not written" >&2; return 1; }
}

# make_run_config NAME ARM N -> prints the config path. Both arms at one N resolve to
# the SAME dataset_name (the prompt is not part of the cache identity).
make_run_config() {
    local name=$1 arm=$2 n=$3
    local feats; feats=$(pool_features "$n") || return 1
    local keep; keep=$(python -c "print(round(min(1.0, $NEG_MARGIN * ($TOTAL_ROWS / (2 * $n)) / $TRAIN_TEXTS), 4))")
    local prompt_flags
    case $arm in
        label) prompt_flags="--prompt-type prompt_sae_label --feature-labels-file $POOL_JSON --max-prompt-tokens $LABEL_PROMPT_TOKENS" ;;
        index) prompt_flags="--prompt-type prompt_no_token_index" ;;
        *) echo "unknown arm $arm" >&2; return 1 ;;
    esac
    local config=experiment_configs/$EXP_GROUP/saeFeatgen_${name}.json
    rm -f "$config"
    python generate_experiment_config.py \
        --experiment-name "saeFeatgen_${name}" \
        --model $MODEL \
        --objective detection --detection-balance-pos-frac 0.5 \
        --detection-total-train-rows $TOTAL_ROWS \
        --intervention-signal-distribution extreme --intervention-extreme-scale 1 \
        --train-dataset allenai/c4 --train-dataset-split "train" \
        --train-max-samples $TRAIN_TEXTS --train-negative-keep $keep \
        --dev-max-samples 200 --dev-offset-from-train 1000000 \
        --test-dataset allenai/c4 --test-dataset-split "validation" --test-max-samples $OWN_TEST_TEXTS \
        --max-seq-length $SEQ_LEN --hf-data-kwargs '{"config": "en", "streaming": true}' \
        --train-signals "$feats" --test-signals "$feats" \
        --train-token-selector $TOKEN_POS --test-token-selector $TOKEN_POS --precision 3 \
        --train-batch-size $BS --eval-batch-size 128 --num-epochs 1 \
        --lora-r $LORA_R --lora-alpha $LORA_ALPHA --lora-dropout 0.0 --bf16 \
        --learning-rate $LR --warmup-ratio 0.03 --lr-scheduler-type cosine --save-steps 0.25 \
        $prompt_flags \
        --cache-dir data --output-dir outputs/runs/$EXP_GROUP --output "$config" >&2 || return 1
    [ -f "$config" ] || { echo "config not written: $config" >&2; return 1; }
    echo "$config"
}

cache_dir_of() {   # the dataset directory a config resolves to
    python -c "
from pathlib import Path
from src.introspector.config import ExperimentConfig
print(ExperimentConfig.model_validate_json(Path('$1').read_text()).dataset_cache_dir)"
}
