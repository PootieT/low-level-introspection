#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_l10_n24
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=24:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1

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
export WANDB_WATCH=all
#export CUDA_LAUNCH_BLOCKING=1  # for debugging cuda errors

### generate experiment config first as JSON
EXP_NAME=single_mlp_l10_n24_train1M
python generate_experiment_config.py \
    --experiment-name $EXP_NAME \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 \
    --train-dataset-split "train" \
    --train-max-samples 1000000 \
    --dev-max-samples 200 \
    --dev-offset-from-train 1000000 \
    --test-dataset allenai/c4 \
    --test-dataset-split "validation" \
    --test-max-samples 3000 \
    --max-seq-length 32 \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals N10.24 \
    --test-signals N10.24 \
    --precision 3 \
    --train-batch-size 32 \
    --eval-batch-size 128 \
    --learning-rate 5e-5 \
    --num-epochs 1 \
    --disable-adapter-in-input \
    --output experiment_configs/$EXP_NAME.json

### cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/$EXP_NAME.json

## train introspector
python train_introspector.py \
    --config experiment_configs/$EXP_NAME.json

## some hyperparameter notes here
## before introducing consistent dev offset (=0)
# lr=1e-5, bs=16 -> dev corr 0.636
# lr=1e-5, bs=32 -> dev corr -0.1229
# lr=2e-5, bs=32 -> dev corr 0.512
# lr=3e-5, bs=32 -> dev corr 0.7257
# lr=4e-5, bs=32 -> dev corr 0.743
# lr=5e-5, bs=32 -> dev corr 0.729 (but other metrics lowest so far)  <- picking this one
# lr=7e-5, bs=32 -> dev corr 0.7288 (but other metrics lowest so far)
# lr=1e-4, bs=64 -> dev corr 0.7308 (highest exact match, but slightly higher loss)
# lr=2e-4, bs=128 -> dev corr 0.719 (need higher lr)
# lr=5e-4, bs=128 -> dev corr -0.5... (very strange, but it is lowest loss)
# lr=1e-3, bs=128 -> dev corr -0.5... (also degenerate, loss slightly higher than above)
# lr=1e-4, bs=128 -> dev corr 0.719 (i think this need more epochs, loss high. I think character based metrics can be misleading, i.e. exact match, mean char match, loss)
# lr=2e-4, bs=64 -> dev corr 0.7326 (corr unstable, lr too high)

## dev offset 1M, train=3K, dev 200, test=3K, lr=5e-5, bs=32,
# train=3K -> dev corr 0.722
# train=10K -> dev corr 0.723
# train=30K -> dev corr 0.823
# train=100K -> dev corr 0.861 (avg_diff, std_diff, loss, l2 all best)
# train=300K -> dev corr 0.925
