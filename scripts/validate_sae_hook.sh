module load cuda/11.8   # 11.2
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

python validate_sae_hook.py \
--model google/gemma-2-2b \
--sae-release gemma-scope-2b-pt-res-canonical \
--layer 9 \
--width 16k \
--site res \
--n-texts 64 \
--max-seq-length 64