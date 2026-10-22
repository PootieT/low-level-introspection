#!/bin/bash -l
#$ -P llamagrp
#$ -N mlp_localization
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=8.5
#$ -l gpu_memory=47G
#$ -l h_rt=6:00:00
#$ -o log
#$ -j y
###$ -m ea
#$ -t 1

# ============================================================================
# Run MI related scripts that tries to localize introspection circuit components
# ============================================================================

## SCC env setup
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
export WANDB_RUN_GROUP=mlp_localization

set -e

####### for joint-loss sweep #######
# compare lora norm across layers
#python plotters/compare_lora_weight_patterns.py --group mlp_joint_grid_layer_token --run-glob 'joint_*kl0.1*' --from-cache

# look at direct attribution from each layer components to final regressor direction, look at token level signal
#python mechanistic_analysis/trace_introspection_circuit.py --group mlp_joint_grid_layer_token --run-glob 'joint_*kl0.1*'

# inject {native, foreign} signal in randomly chosen (by index/layer/position) neurons and look at affects in trained neuron
#python mechanistic_analysis/score_injection_specificity.py --group mlp_joint_grid_layer_token --run-glob 'joint_*kl0.1*' --from-cache --n-test 500

# run endogenous eval setting, compute gradient X activation for all neuron in (L, t, i)
#python mechanistic_analysis/score_gradient_attribution.py --group mlp_joint_grid_layer_token --run-glob 'joint_*kl0.1*' --from-cache --n-test 500

# trace the top contributors to mover head that moves info to final token regressor direction
#python mechanistic_analysis/trace_mover_head.py --group mlp_joint_grid_layer_token --run-glob 'joint_*kl0.1*'

# ablate heads and check evaluations to see which kind of heads ablate which eval, how are they related
#python mechanistic_analysis/score_head_ablation.py --group mlp_joint_grid_layer_token --run-glob 'joint_*kl0.1*' --skip-existing

# analyze head taxonomy distribution
#python plotters/plot_head_taxonomy.py --group mlp_joint_grid_layer_token --kl kl0.1

# replot trace mover head plots b ut with head knockout experimients
python mechanistic_analysis/trace_mover_head.py --group mlp_joint_grid_layer_token --run-glob 'joint_*_kl0.1_*' --heads-from-knockout --top-heads 4