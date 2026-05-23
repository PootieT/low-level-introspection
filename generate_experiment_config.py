#!/usr/bin/env python3

"""
Generate a reproducible experiment config JSON.

Example:

python scripts/generate_experiment_config.py \
    --experiment-name mlp_l10_n200 \
    --model meta-llama/Llama-3.2-1B \
    --train-dataset allenai/c4 \
    --train-dataset-split train[:1000] \
    --test-dataset allenai/c4 \
    --test-dataset-split test[:1000] \
    --signal-types mlp \
    --train-signals 10.1 \
    --test-signals 10.1 \
    --train-token-position -1 \
    --test-token-positions \
    --precision 3 \
    --output experiment_configs/mlp_l10_n200.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from transformers import AutoConfig

from src.introspector.config import DatasetConfig, ExperimentConfig, TrainingConfig
from src.introspector.signals.schema import ExplicitSignalSpec, FixedTokenSelector, RandomRangeTokenSelector
from src.introspector.signals.resolution import resolve_signal_source





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # ---- experiment ----
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)

    # ---- model ----
    parser.add_argument("--model", required=True)
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model weights in 4bit")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model weights in 8bit")

    # ---- dataset ----
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--train-dataset-split", default="train")
    parser.add_argument("--train-max-samples", type=int)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--test-dataset-split", default="test")
    parser.add_argument("--test-max-samples", type=int)
    parser.add_argument("--hf-data-kwargs", type=str, default="", help="json string of kwargs to pass to HF Dataset constructor (e.g. config)")
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--text-field", default="text")
    # parser.add_argument("--sample-per-data", type=int, default=1)  # TODO maybe no need better to have diverse datapoints

    # ---- signal ----
    parser.add_argument(
        "--train-signals",
        type=str,
        required=True,
        help="Inline or path to json configuration of training signals."
             'For fixed signals it might look like {"type":"explicit", "signals":["N12.1","N12.2"]}'
             'For sampling random sets it might look like {"type":"uniform_layer_sampler", '
             '  "signal_type":"attention_score","layers":[1,2,3,4], "signals_per_layer"=2, "seed"=21}'
    )
    parser.add_argument("--test-signals", type=str, required=True, help="See train-signals")

    parser.add_argument(
        "--train-token-selector",
        type=Union[int,str],
        default=-1,
        help="Token index to introspect on. Integer (fixed position) or string range value (e.g. '10:20') for "
             "random selection",
    )
    parser.add_argument(
        "--test-token-selector",
        type=Union[int, str],
        default=-1,
        help="Token index to introspect on (for test split).",
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Number of significant figures for text labels.",
    )

    # ---- training ----
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=3)
    # TODO check if this is possible/this might be the same as masking input as -100
    parser.add_argument("--disable-adapter-in-input", action="store_true",
                        help="Whether to deactivate LoRA adaptor in input")
    parser.add_argument("--prompt-type", type=str, default="prompt_no_token_index")
    # ---- lora ----
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    # ---- paths ----
    parser.add_argument(
        "--cache-dir",
        default="data",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/runs",
    )

    return parser.parse_args()


def load_spec(value: str):
    if value.endswith(".json"):
        return json.loads(Path(value).read_text())
    else:
        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            print(f"{value} is not a valid JSON, assuming it is a single signal or comma separated list of signals.")
            return ExplicitSignalSpec(type="explicit", signals=value.split(","))

def load_token_selector(value: Union[int,str]):
    if isinstance(value, int):
        return FixedTokenSelector(type="fixed", position=value)
    else:
        start, stop = value.split(":")
        return RandomRangeTokenSelector(type="random_range", start=start, stop=stop)

def main() -> None:
    args = parse_args()

    hf_data_kwargs = json.loads(args.hf_data_kwargs) if args.hf_data_kwargs else {}

    model_config = AutoConfig.from_pretrained(args.model)
    train_spec = resolve_signal_source(load_spec(args.train_signals), model_config=model_config, precision=args.precision)
    test_spec = resolve_signal_source(load_spec(args.test_signals), model_config=model_config, no_sample_signals=train_spec, precision=args.precision)

    train_token_selector = load_token_selector(args.train_token_selector)
    test_token_selector = load_token_selector(args.test_token_selector)

    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        created_at=datetime.now(timezone.utc).isoformat(),

        model_name=args.model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        seed=args.seed,

        train_dataset=DatasetConfig(
            name=args.train_dataset,
            split=args.train_dataset_split,
            text_field=args.text_field,
            max_samples=args.train_max_samples,
            hf_data_kwargs=hf_data_kwargs,
            max_seq_length=args.max_seq_length,
        ),
        test_dataset=DatasetConfig(
            name=args.test_dataset,
            split=args.test_dataset_split,
            text_field=args.text_field,
            max_samples=args.test_max_samples,
            hf_data_kwargs=hf_data_kwargs,
            max_seq_length=args.max_seq_length,
        ),

        train_signals=train_spec,
        test_signals=test_spec,

        train_token_selector=train_token_selector,
        test_token_selector=test_token_selector,

        training_config =TrainingConfig(
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_epochs=args.num_epochs,
            max_seq_length=args.max_seq_length + 30, # input + introspection prefill
            prompt_type=args.prompt_type,
        ),

        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            config.model_dump(),
            f,
            indent=2,
            sort_keys=True,
        )

    print(f"Saved config to: {output_path}")


if __name__ == "__main__":
    main()