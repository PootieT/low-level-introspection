#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_intervention_train_regressor_sweep
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=15:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-80

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

## environment setup
export WANDB_PROJECT=low-level-introspection
#export WANDB_WATCH=all
#export CUDA_LAUNCH_BLOCKING=1  # for debugging cuda errors
export WANDB_RUN_GROUP=single_mlp_intervention_train_reg_sweep_layer_token

## Batch job variables
index=$(($SGE_TASK_ID-1))

declare -a params
idx=0
IFS=' ' # space is set as delimiter

for layer in 0 2 4 6 8 10 12 14
do
    for pos in -1 -4 -7 -10 -13 -16 -19 -22 -25 -28
    do
        params[idx]=$layer$IFS$pos
        ((idx++))
    done
done

read -ra taskinput <<< "${params[$index]}" # str is read into an array as tokens separated by IFS
LAYER=${taskinput[0]}
TOKEN_POS=${taskinput[1]}

for SAMPLE in {0..9}
do
  ### generate experiment config first as JSON
  EXP_NAME="l${LAYER}_pos${TOKEN_POS}_sample${SAMPLE}_reg_extreme10_300K"
  EXP_GROUP="intervention_train_regressor_sweep_layer_token"
  python generate_experiment_config.py \
      --experiment-name $EXP_NAME \
      --model meta-llama/Llama-3.2-1B \
      --train-dataset allenai/c4 \
      --train-dataset-split "train" \
      --train-max-samples 300000 \
      --dev-max-samples 200 \
      --dev-offset-from-train 1000000 \
      --test-dataset allenai/c4 \
      --test-dataset-split "validation" \
      --test-max-samples 3000 \
      --max-seq-length 32 \
      --hf-data-kwargs '{"config": "en", "streaming": true}' \
      --train-signals '{"type":"uniform_layer_sampler", "signal_type":"mlp","layers":['$LAYER'], "signals_per_layer":1, "seed":21}' \
      --test-signals '{"type":"uniform_layer_sampler", "signal_type":"mlp","layers":['$LAYER'], "signals_per_layer":1, "seed":21}' \
      --same-train-test-signals \
      --train-token-selector $TOKEN_POS \
      --test-token-selector $TOKEN_POS \
      --precision 3 \
      --train-batch-size 32 \
      --eval-batch-size 128 \
      --num-epochs 1 \
      --learning-rate 5e-5 \
      --disable-adapter-in-input \
      --bf16 \
      --use-regressor \
      --intervention-signal-distribution extreme \
      --intervention-anneal \
      --intervention-scale-start 100 \
      --intervention-scale-end 10 \
      --intervention-anneal-schedule cosine \
      --intervention-anneal-start-frac 0.0 \
      --intervention-anneal-end-frac 1.0 \
      --cache-dir data/$EXP_GROUP \
      --output-dir outputs/runs/$EXP_GROUP \
      --output experiment_configs/$EXP_GROUP/$EXP_NAME.json \
      --overwrite-cache
#      --intervention-extreme-scale 10 \


  ### cache activations and generate dataset
  python cache_signals.py \
      --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
  #    --overwrite-cache

  ## train introspector
  python train_introspector_with_intervention.py \
      --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
      --refit-head \
      --refit-estimator "ridge" \
      --refit-lambda-grid 1,10,30,100,300,1000,3000,10000,30000,100000 \
  #    --overwrite-cache \
  exit
  ## delete model checkpoints to save space
#  python utils/delete_checkpoint_after_experiment_completion.py \
#    --config experiment_configs/$EXP_GROUP/$EXP_NAME.json \
#    --expected-files "test_results.json,interventional_test_metrics_gaussian.json,test_alt_intervention_results.json"
done