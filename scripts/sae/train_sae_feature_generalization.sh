#!/bin/bash -l
#$ -P llamagrp
#$ -N sae_featgen
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=48:00:00
#$ -o log
#$ -j y
#$ -t 13

# ============================================================================
# SAE FEATURE GENERALIZATION: does an introspector trained on N SAE features read
# features it never trained on, when the prompt names a feature by its natural-language
# label instead of an index? The SAE analogue of
# scripts/neuron/train_single_mlp_introspector_neuron_generalization.sh, where indices
# gave no generalization at all (held-out interventional corr ~0 at every N).
#
# The table, settings and config generation live in featgen_common.sh -- read its
# header for the arms and the row budget. One task = one row:
#
#   qsub scripts/sae/cache_sae_feature_generalization.sh                    # stage 1, once
#   qsub -hold_jid sae_featgen_cache -t 1-5  <this script>                  # label arm
#   qsub -hold_jid sae_featgen_cache -t 6-10 <this script>                  # index arm, later
#   GROUP=index qsub -t 1-5 <this script>                                   # same, by group
#   RUN=label_n10 bash <this script>                                        # one row, interactive
# data100k (N=3's 100k rows PER feature at N=10/30/100; needs its own cache first):
#   qsub -v STAGE_GROUPS=data100k scripts/sae/cache_sae_feature_generalization.sh
#   qsub -hold_jid sae_featgen_cache -t 11 -l h_rt=24:00:00 <this script>   # N=10,  1M rows
#   qsub -hold_jid sae_featgen_cache -t 12 -l h_rt=48:00:00 <this script>   # N=30,  3M rows
#   qsub -hold_jid sae_featgen_cache -t 13 -l h_rt=48:00:00 <this script>   # N=100, 10M rows:
#       longer than one job; resubmit the same task and it resumes from its latest
#       checkpoint (every 10% of training on runs over 1M rows).
#
# Caches must exist (stage 1); a task never builds one, so array tasks cannot race.
#
# WHICH NUMBER TO READ: interventional balanced accuracy on the HELD-OUT arm, next to
# the distractor matrix (feature_generalization_distractor_summary.csv): a readout that
# only detects "something was injected" also scores well on held-out features, and
# only the off-diagonal (steer j, ask about i) separates that from reading the label.
# Endogenous held-out accuracy can rise from the label alone (it describes the text).
#
# Outputs per run (outputs/runs/sae_featgen/saeFeatgen_<name>/):
#   feature_generalization.csv               macro BA per (eval_type, seen/unseen)
#   feature_generalization_by_feature.csv    every pool feature x eval_type
#   feature_generalization_distractor*.csv   steer x ask matrix + summary
#   train_balance_by_feature.csv             realized rows / repeats per trained feature
#   run_spec.json                            the table row, read by the plotter
# Aggregate: python plotters/plot_sae_feature_generalization.py
# ============================================================================

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
export WANDB_RUN_GROUP=sae_featgen

set -e
source scripts/sae/featgen_common.sh
select_row
echo "row: name=$NAME group=$R_GROUP arm=$ARM n_features=$N_FEAT total_rows=$TOTAL cache_feats=$CACHE_FEATS w=$W kl=$KL_TARGET"

# the pool config is written by stage 1 only: regenerating it here would rewrite a file
# that concurrent tasks are reading
[ -f "$POOL_CONFIG" ] || { echo "ERROR: $POOL_CONFIG missing; run cache_sae_feature_generalization.sh" >&2; exit 1; }
CONFIG=$(make_run_config "$NAME" "$ARM" "$N_FEAT" "$TOTAL" "$CACHE_FEATS")
for c in "$POOL_CONFIG" "$CONFIG"; do
    d=$(cache_dir_of "$c")
    # metadata.json is written last by data_prep/cache_signals.py, so it marks a COMPLETE cache
    if [ ! -f "$d/metadata.json" ]; then
        echo "ERROR: cache $d missing for $c; run cache_sae_feature_generalization.sh first" >&2
        exit 1
    fi
done

# where the trainer writes (config output_dir / experiment_name), so the sidecar can't drift
RUN_DIR=$(python -c "
import json; c = json.load(open('$CONFIG')); print(c['output_dir'] + '/' + c['experiment_name'])")
mkdir -p "$RUN_DIR"
python - <<PY
import json
pool = json.load(open("$POOL_JSON"))
json.dump({"name": "$NAME", "group": "$R_GROUP", "arm": "$ARM", "n_features": $N_FEAT,
           "total_rows": $TOTAL, "rows_per_feature": $TOTAL // $N_FEAT, "cache_feats": $CACHE_FEATS,
           "joint_loss_weight": $W, "kl_anchor_target": $KL_TARGET,
           "layer": $LAYER, "token_pos": $TOKEN_POS, "pool_json": "$POOL_JSON",
           "pool_size": len(pool["features"]), "train_features": pool["features"][:$N_FEAT]},
          open("$RUN_DIR/run_spec.json", "w"), indent=2)
PY

python trainers/train_sae_detector_joint.py --config "$CONFIG" \
    --joint-loss-weight $W --kl-anchor-target $KL_TARGET --force-drift-eval \
    --feature-gen-pool-config "$POOL_CONFIG"

# Re-run ONLY the eval on an existing checkpoint:
#   python trainers/train_sae_detector_joint.py --config "$CONFIG" --joint-loss-weight $W \
#       --feature-gen-only --feature-gen-pool-config "$POOL_CONFIG"
echo "Done: $(date)"
