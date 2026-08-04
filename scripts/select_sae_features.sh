module load cuda/11.8   # 11.2
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

python select_sae_features.py \
--model google/gemma-2-2b \
--sae-release gemma-scope-2b-pt-res-canonical \
--layer 9 \
--width 16k \
--n-tokens 1000000 \
--max-seq-length 128 \
--fixed-position -1 \
--cache-size 20000 \
--out data/sae_stats/gemma2-2b_res9_16k.parquet