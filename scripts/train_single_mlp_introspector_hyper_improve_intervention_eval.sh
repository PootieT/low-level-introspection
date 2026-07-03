#!/bin/bash -l
#$ -P llamagrp
#$ -N single_mlp_improve_intervention_eval_lora_module
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=2:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1-10

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

# batch job variables
index=$(($SGE_TASK_ID-1))
## vary lora r
#LORA_RS=(8 4 2 1)
#LORA_R=${LORA_RS[$index]}
#LORA_ALPHAS=(16 8 4 2)
#LORA_ALPHA=${LORA_ALPHAS[$index]}
LORA_R=16
LORA_ALPHA=32
LR=5e-5
#export WANDB_RUN_GROUP=single_mlp_hyper_lora_r_improve_intervention

# vary lora dropout
#LORA_DOS=(0 0.05 0.1 0.15 0.2 0.3 0.4)
#LORA_DO=${LORA_DOS[$index]}
LORA_DO=0.05
#export WANDB_RUN_GROUP=single_mlp_hyper_lora_dropout_improve_intervention

# vary lora target module
LORA_MODS=(#"all-linear" \
 "q_proj" "v_proj" "k_proj" "o_proj" "q_proj,v_proj" "q_proj,k_proj,v_proj,o_proj" \
"gate_proj" "up_proj" "down_proj" "gate_proj,up_proj,down_proj" \
)
LORA_MOD=${LORA_MODS[$index]}
export WANDB_RUN_GROUP=single_mlp_hyper_lora_target_module_improve_intervention

### generate experiment config first as JSON
EXP_NAME="r${LORA_R}_alpha${LORA_ALPHA}_drop${LORA_DO}_mod${LORA_MOD}_lr${LR}_train300K"
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
    --train-signals N10.24 \
    --test-signals N10.24 \
    --precision 3 \
    --train-batch-size 32 \
    --eval-batch-size 128 \
    --learning-rate $LR \
    --num-epochs 1 \
    --disable-adapter-in-input \
    --bf16 \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DO \
    --lora-target-modules $LORA_MOD \
    --cache-dir data/single_mlp_hyper_improve_intervention_eval \
    --output-dir outputs/runs/single_mlp_hyper_improve_intervention_eval \
    --output experiment_configs/single_mlp_hyper_improve_intervention_eval/$EXP_NAME.json \

### cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/single_mlp_hyper_improve_intervention_eval/$EXP_NAME.json \
#    --overwrite-cache

## train introspector
python train_introspector.py \
    --config experiment_configs/single_mlp_hyper_improve_intervention_eval/$EXP_NAME.json \
#    --overwrite-cache

python intervene_introspector.py \
    --config experiment_configs/single_mlp_hyper_improve_intervention_eval/$EXP_NAME.json

python utils/delete_checkpoint_after_experiment_completion.py \
  --config experiment_configs/single_mlp_hyper_improve_intervention_eval/$EXP_NAME.json \
  --expected-files "test_results.json,test_pred_results.json,intervention_results_metrics.json"

#done

## some hyperparameter notes here
# change lora rank (no good)
# r=8,lr=5e-5,test_cor=0.498,inter_cor=0.01
# r=4,lr=5e-5,test_cor=0.299,inter_cor=0.00
# r=2,lr=5e-5,test_cor=0.133,inter_cor=0.00
# r=1,lr=5e-5,test_cor=0.052,inter_cor=-0.01  # smaller rank, peak eval is earlier in epoch, need smaller lr
# r=1,lr=2e-5,test_cor=0.27,inter_cor=0.00
# r=1,lr=1e-5,test_cor=-0.02,inter_cor=-0.02

# change lora dropout
# r=16,lr=5e-5,drop=0.0,test_cor=0.56,inter_cor=-0.01
# r=16,lr=5e-5,drop=0.1,test_cor=0.57,inter_cor=-0.0
# r=16,lr=5e-5,drop=0.1,test_cor=0.54,inter_cor=-0.00
# r=16,lr=5e-5,drop=0.2,test_cor=0.56,inter_cor=-0.00
# change lora modules, dropout, to see if we can improve intervention correlation

# turns out, 3K is really not enough data to evaluate this, so we are gonna
# have to be patient and test result on 300K, where generative eval results
# are actually non-trivial
# lora r {16,8,4,2,1}, non significant
# dropout {0 0.05 0.1 0.15 0.2 0.3 0.4}, 0.3 has intervention res of 0.042, but neither end does so I don't think it's generalizable under bootstrap
# module {all-mlp, all-attn, q+v} non significant
