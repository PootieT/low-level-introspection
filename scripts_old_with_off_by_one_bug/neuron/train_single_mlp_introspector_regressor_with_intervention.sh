#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_intervention_train_hyper
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=6:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-6

## SCC env setup
## Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
nvidia-smi

#module load cuda/11.8   # 11.2
#module load miniconda/23.11.0
#conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env
conda activate introspection

## environment setup
export WANDB_PROJECT=low-level-introspection
#export WANDB_WATCH=all
#export CUDA_LAUNCH_BLOCKING=1  # for debugging cuda errors
export WANDB_RUN_GROUP=single_mlp_intervention_train_reg_hyper

## Batch job variables
index=$(($SGE_TASK_ID-1))
## vary lora r
#LORA_RS=(2 1)
#LORA_R=${LORA_RS[$index]}
#LORA_ALPHAS=(4 2)
#LORA_ALPHA=${LORA_ALPHAS[$index]}
LORA_R=16
LORA_ALPHA=32

## vary lr
LRS=(5e-5 1e-6 5e-7 2e-7 1e-7 5e-8)
LR=${LRS[$index]}
LR=5e-5

## dropout
LORA_DOS=(0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
LORA_DO=${LORA_DOS[$index]}
LORA_DO=0.05


### generate experiment config first as JSON
EXP_NAME="intervention_train_reg_N10.24_3K_fixedPaddingBug"
python generate_experiment_config.py \
    --experiment-name $EXP_NAME \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 \
    --train-dataset-split "train" \
    --train-max-samples 3000 \
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
    --num-epochs 1 \
    --disable-adapter-in-input \
    --bf16 \
    --learning-rate $LR \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --use-regressor \
    --intervention-signal-distribution extreme \
    --cache-dir data/internvetion_train \
    --output-dir outputs/runs/internvetion_train \
    --output experiment_configs/internvetion_train_regressor/$EXP_NAME.json \
#    --intervention-extreme-scale 1000 \
#    --overwrite-cache
#    --lora-dropout $LORA_DO \

### cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/internvetion_train_regressor/$EXP_NAME.json \
#    --overwrite-cache

## train introspector
python train_introspector_with_intervention.py \
    --config experiment_configs/internvetion_train_regressor/$EXP_NAME.json \
    --refit-head \
    --refit-estimator "ridge" \
    --refit-lambda-grid 1,10,30,100,300,1000,3000,10000,30000,100000
#    --overwrite-cache

# fitting ridge (l2) on the regressor
#  lambda=100        dev_corr=0.2798  dev_l2=0.0655
#  lambda=150        dev_corr=0.2902  dev_l2=0.0638
#  lambda=200        dev_corr=0.2972  dev_l2=0.0627
#  lambda=250        dev_corr=0.3025  dev_l2=0.0621
#  lambda=300        dev_corr=0.3066  dev_l2=0.0617
#  lambda=350        dev_corr=0.3101  dev_l2=0.0613
#  lambda=400        dev_corr=0.3131  dev_l2=0.0611
#  lambda=450        dev_corr=0.3156  dev_l2=0.0609
#  lambda=500        dev_corr=0.3179  dev_l2=0.0607
#  lambda=550        dev_corr=0.3199  dev_l2=0.0606
#  lambda=600        dev_corr=0.3216  dev_l2=0.0605
#  lambda=650        dev_corr=0.3232  dev_l2=0.0604
#  lambda=700        dev_corr=0.3247  dev_l2=0.0603
#  lambda=750        dev_corr=0.3260  dev_l2=0.0602
#  lambda=800        dev_corr=0.3272  dev_l2=0.0601
#  lambda=850        dev_corr=0.3283  dev_l2=0.0601
#  lambda=900        dev_corr=0.3293  dev_l2=0.0600
#  lambda=1100       dev_corr=0.3327  dev_l2=0.0599
#  lambda=1300       dev_corr=0.3352  dev_l2=0.0597
#  lambda=1500       dev_corr=0.3372  dev_l2=0.0597
#  lambda=2000       dev_corr=0.3406  dev_l2=0.0596
#  lambda=2500       dev_corr=0.3427  dev_l2=0.0595
#  lambda=3000       dev_corr=0.3441  dev_l2=0.0595
#  lambda=3500       dev_corr=0.3450  dev_l2=0.0595
#  lambda=4000       dev_corr=0.3457  dev_l2=0.0595
#  lambda=5000       dev_corr=0.3465  dev_l2=0.0595
#  lambda=6000       dev_corr=0.3468  dev_l2=0.0596
#  lambda=7000       dev_corr=0.3470  dev_l2=0.0596
#  lambda=8000       dev_corr=0.3470  dev_l2=0.0596
#  lambda=9000       dev_corr=0.3469  dev_l2=0.0596
#  lambda=10000      dev_corr=0.3468  dev_l2=0.0597
#  lambda=15000      dev_corr=0.3457  dev_l2=0.0598
#  lambda=20000      dev_corr=0.3444  dev_l2=0.0600
#  lambda=30000      dev_corr=0.3422  dev_l2=0.0602
#  lambda=50000      dev_corr=0.3392  dev_l2=0.0606
#  lambda=100000     dev_corr=0.3338  dev_l2=0.0612

#[refit_regression_head] best lambda=8000 (dev_corr=0.3470)
#  test (best lambda) corr=0.2990  l2=0.0558
# tried fp32 because there maybe small signal not read. intervention eval=0.65, but normal eval almost 0. refit endogenous=0.47, but refit intervention~0
# fixed the padding bug, so all above results are kind of outdated
# still, after fixing, training intervention corr~0.92, normal~0, refit endogenous 0.52, refit intervention=0