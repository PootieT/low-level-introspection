# generate experiment config first as JSON
python generate_experiment_config.py \
    --experiment-name single_mlp_l10_n24 \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 \
    --train-dataset-split "train" \
    --train-max-samples 1000 \
    --test-dataset allenai/c4 \
    --test-dataset-split "validation" \
    --test-max-samples 200 \
    --max-seq-length 32 \
    --hf-data-kwargs '{"config": "en", "streaming": true}' \
    --train-signals N10.24 \
    --test-signals N10.24 \
    --precision 3 \
    --train-batch-size 2 \
    --eval-batch-size 4 \
    --learning-rate 1e-5 \
    --num-epochs 3 \
    --output experiment_configs/single_mlp_l10_n24.json

# cache activations and generate dataset
python cache_signals.py \
    --config experiment_configs/single_mlp_l10_n24.json

## train introspector
python train_introspector.py \
    --config experiment_configs/single_mlp_l10_n24.json