#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_intervention_train_joint_ungated_prev_token
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=6:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-8

# ============================================================================
# CONTROL for the earlier-token generalization failure: LoRA INPUT GATE OFF.
#
# In every prev-token run so far the adapter was gated to the introspection
# prompt (collator mask = 0 over the input text), so NO trainable parameter ever
# touched the computation at the introspected token. The injected value at
# layers.L.mlp.act_fn was therefore routed downstream at FIXED base-model gain:
# through down_proj[:, n] (scaled by up_proj(x)[n]), the layers above it at that
# position, and the base K/V that the prompt tokens attend to. At pos=-1 the
# prompt is adjacent so a small gain suffices; at earlier positions the signal
# has more layers + more attention dilution to survive, and nothing trainable
# could boost it. This run removes exactly that constraint at LoRA cost, before
# committing to full-weight finetuning.
#
# Caveat (shared with full FT, but milder): with the adapter active over the
# text, the model's own activation drifts away from the CACHED endogenous
# target. The INTERVENTIONAL metric is unaffected -- the injected value is
# ground truth by construction -- so read interventional corr (and the routing
# probe by position) as the headline, endogenous corr as secondary.
#
# Compare against: intervention_train_regressor_joint_prev_token_hyper (same
# data, same hyperparameters, gate ON). The cached dataset is REUSED from that
# group, so no re-caching of the 1M-example dataset happens here.
# ============================================================================

## SCC env setup
## Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

module load cuda/11.8   # 11.2
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env
#conda activate introspection

## environment setup
export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache
export WANDB_PROJECT=low-level-introspection
export WANDB_RUN_GROUP=single_mlp_intervention_train_reg_joint_ungated_prev_token

## Batch job variables
index=$(($SGE_TASK_ID-1))

W=0.01
SCALE=10
SIG=10.24

####### Setting 1: maxxing hyperparameter to see if signal is trainable
#LORA_R=128
#TOKEN_POS='-1:-31'
#BS=128
#TRAIN_SAMPLES=1000000
#EXP_NAME="intervention_train_reg_joint_ungated_POS${TOKEN_POS}_scale${SCALE}_r${LORA_R}_lr${LR}_w${W}_1M"
#DATASET_NAME="c4_N${SIG}_pos${TOKEN_POS}_seq32_1M"
## plus these args
#    --warmup-ratio 0.03 \
#    --lr-scheduler-type cosine \
#    --save-steps 0.2 \
######## Setting 2: keep default hyperparameter to see if we can still do it
LORA_R=16
TOKEN_POS='-28'
BS=32
TRAIN_SAMPLES=300000
LR=5e-5

LORA_ALPHA=$((LORA_R * 2))
# same LR grid as the gated prev-token hyper sweep, so gated-vs-ungated is a
# best-vs-best comparison rather than a comparison at one arbitrary LR
#LRS=(1e-5 2e-5 3e-5 5e-5 1e-4 2e-4)
#LR=${LRS[$index]}

#DOS=(0.01 0.05 0.1 0.15 0.2 0.3 0.4 0.5)
#DO=${DOS[$index]}

EXP_NAME="intervention_train_reg_joint_ungated_POS${TOKEN_POS}_scale${SCALE}_w${W}_lrcos"
DATASET_NAME="c4_N${SIG}_pos${TOKEN_POS}_seq32_300K"


### generate experiment config first as JSON
EXP_GROUP=intervention_train_regressor_joint_ungated_prev_token
# NOTE: cache dir deliberately points at the GATED sweep's group so the already
# cached 1M-example dataset is reused as-is (cache_signals.py no-ops when
# <cache-dir>/<dataset-name>/train exists). Only the outputs are new.
CACHE_GROUP=intervention_train_regressor_joint_prev_token_hyper
echo $EXP_NAME


#RUN_DIR=outputs/runs/$EXP_GROUP/$EXP_NAME/diag_checkpoint-9375; python - <<EOF
#import pandas as pd, numpy as np
#d = pd.read_csv("$RUN_DIR/activation_drift.csv")
#e = pd.read_csv("$RUN_DIR/endogenous_test_predictions_endogenous.csv")
#assert len(d) == len(e), (len(d), len(e))
#print("row alignment ok:", np.allclose(d["cached"], e["activation"], atol=1e-4))
#d["pred"] = e["predictions"].values
#for c in ["cached", "base_off-1", "base_off0", "tuned_off-1", "tuned_off0"]:
#    print(f"corr(pred, {c:12s}) = {np.corrcoef(d['pred'], d[c])[0,1]: .4f}")
#print(f"corr(tuned-1, base-1)  = {np.corrcoef(d['tuned_off-1'], d['base_off-1'])[0,1]: .4f}")
#print(f"std ratio @-1          = {d['tuned_off-1'].std() / d['base_off-1'].std(): .4f}")
#EOF
#exit

#python probe_cache_alignment.py --config experiment_configs/$EXP_GROUP/$EXP_NAME.json --n-examples 1024 --also-fp32
#exit

# temporary just for debugging
#EXP_GROUP=intervention_train_regressor_joint_sweep
#EXP_NAME=intervention_train_reg_joint_scale10_w0.01
#
#RUN_DIR=outputs/runs/intervention_train_regressor_joint_sweep/intervention_train_reg_joint_scale10_w0.01/diag_checkpoint-9375; python - <<EOF
#import pandas as pd, numpy as np
#d = pd.read_csv("$RUN_DIR/activation_drift.csv")
#print("corr(tuned,base) @-1 =", np.corrcoef(d["tuned_off-1"], d["base_off-1"])[0,1])
#print("corr(tuned,base) @ 0 =", np.corrcoef(d["tuned_off0"],  d["base_off0"])[0,1])
#print("std ratio @-1 =", d["tuned_off-1"].std()/d["base_off-1"].std())
#EOF
python train_introspector_joint.py --config experiment_configs/intervention_train_regressor_joint_sweep/intervention_train_reg_joint_scale10_w0.01.json --joint-loss-weight 0.01 --diag-only --checkpoint outputs/runs/intervention_train_regressor_joint_sweep/intervention_train_reg_joint_scale10_w0.01/checkpoint-9375 --force-drift-eval
exit

python generate_experiment_config.py \
    --experiment-name $EXP_NAME \
    --dataset-name $DATASET_NAME \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 \
    --train-dataset-split "train" \
    --train-max-samples $TRAIN_SAMPLES \
    --dev-max-samples 200 \
    --dev-offset-from-train 1000000 \
    --test-dataset allenai/c4 \
    --test-dataset-split "validation" \
    --test-max-samples 3000 \
    --max-seq-length 32 \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals "N${SIG}" \
    --test-signals "N${SIG}" \
    --train-token-selector="${TOKEN_POS}" \
    --test-token-selector="${TOKEN_POS}" \
    --precision 3 \
    --train-batch-size $BS \
    --eval-batch-size 128 \
    --num-epochs 1 \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --bf16 \
    --learning-rate $LR \
    --warmup-ratio 0.03 \
    --lr-scheduler-type cosine \
    --use-regressor \
    --intervention-signal-distribution extreme \
    --intervention-extreme-scale $SCALE \
    --cache-dir data/$CACHE_GROUP \
    --output-dir outputs/runs/$EXP_GROUP \
    --output experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache
# NB: --disable-adapter-in-input is intentionally NOT passed. THAT is the switch:
# train_introspector_joint.py now honors training_config.disable_adapter_in_input
# (as train_introspector.py always did), so omitting it here is what turns the
# input gate off. The gated prev-token sweep passes it and is otherwise identical.
#    --warmup-ratio 0.03 \
#    --lr-scheduler-type cosine \
#    --save-steps 0.2 \


### cache activations and generate dataset (no-op: reuses $CACHE_GROUP cache)
python cache_signals.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --overwrite-cache

## train introspector -- INPUT GATE OFF
python train_introspector_joint.py \
    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
    --joint-loss-weight $W \
    #    --overwrite-cache

# post training analysis of each checkpoint (same config => same gating as
# training; the eval forwards must not silently switch gating behavior)
for CK in outputs/runs/$EXP_GROUP/$EXP_NAME/checkpoint-*; do
  echo "Analyzing checkpoint ${CK} ..."
  python train_introspector_joint.py \
   --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
   --joint-loss-weight $W \
   --diag-only \
   --checkpoint "$CK" \
   --force-drift-eval
done



### minimum setup logs
# lr=2e-4,5e-5 doesn't work, maybe setting scale=30 is the only resolution?
# scale=30 is also not doing it
# but maybe I wasn't learning long enough.
# what worked was 300K, scale=10, cosine + warmup (after 0.4 epochs)-> endo:0.80, inter (unit gaus) 0.43
# same setup without cosine + warmup fails
# now let's sweep some hyperparameters:
#LRS=(1e-5 2e-5 3e-5 5e-5 1e-4 2e-4) => 2e-4 endo 0.66, inter 0.43 -> default 5e-5 best
#LR=5e-5, DOS=(0.01 0.05 0.1 0.15 0.2 0.3 0.4 0.5) => 0.01, but endo:0.82, inter 0.3, so no drop best
# then I think we need to sweep pareto front across tokens/layers/weights{0.0,0.01,0.03,0.1,0.3,1.0} to get a sense of where things are



### maximum setup logs
# scale 30 bad, scale 10 works, best lr=5e-5 endogenous 0.87, intervention (unit gaussian) 0.71
