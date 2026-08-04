module load cuda/11.8   # 11.2
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env


BASE=/projectnb/llamagrp/peter/low-level-introspection/outputs/runs/intervention_train_regressor_joint_prev_token_hyper

OUT_DIR=single_neuron_multi_token_hyper
# lr sweep at scale=30 (your best routing scale)
python analysis/aggregate_joint_sweep.py --runs-glob "$BASE/*scale30*_1M" --out-name $OUT_DIR/lr_sweep_s30

# rank sweep at scale=30 (does higher rank raise routing?)
python analysis/aggregate_joint_sweep.py --runs-glob "$BASE/*scale30_r*_w0.01_multi" --out-name $OUT_DIR/rank_sweep_s30

# scale comparison at fixed r128/lr5e-5, + the fixed-position baseline overlaid
python analysis/aggregate_joint_sweep.py \
    --runs-glob "$BASE"/*_r128_lr5e-5_w0.01_1M "$BASE"/*POS-28_scale*_w0.01 \
    --out-name scale_cmp

# analyze evals across 5 checkpoints
RUN=$BASE/intervention_train_reg_joint_POS-1:-31_scale30_r128_lr5e-5_w0.01_1M
python analysis/aggregate_joint_sweep.py --runs-glob "$RUN"/diag_checkpoint-* --out-name conv_s30_r128
