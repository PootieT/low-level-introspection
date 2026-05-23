#!/usr/bin/env python3

"""
Cache activation targets for introspection training.

This script:
1. Loads a frozen experiment config JSON
2. Loads the HF dataset + model
3. Samples token positions
4. Extracts signals using NNsight
5. Saves train/test HF datasets to disk

Example:

python scripts/cache_signals.py \
    --config configs/experiments/mlp_generalization.json
"""

from __future__ import annotations

import argparse
import json
import operator
import pdb
import random
from pathlib import Path
from typing import Any, List, Dict

from rich.diagnose import report
from tqdm import tqdm
import torch
from datasets import Dataset, load_dataset
from nnsight import LanguageModel
import transformers
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, BatchEncoding

from src.introspector.config import ExperimentConfig, DatasetConfig
from src.introspector.models.signal_paths import resolve_mlp_signal_path, resolve_resid_signal_path
from src.introspector.prompts import PROMPT_REGISTRY
from src.introspector.signals.schema import SignalSpec, TokenSelector
from src.introspector.utils import get_quantization_config


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


def sample_token_position(
    seq_len: int,
    token_selector: TokenSelector,
) -> int:
    """
    Sample token position according to selector config.
    """

    selector_type = token_selector.type

    if selector_type == "fixed":
        pos = token_selector.position
        if pos < 0:
            pos = seq_len + pos
        selected_pos =  max(0, min(pos, seq_len - 1))
    elif selector_type == "random_range":
        start = token_selector.start
        stop = token_selector.stop
        if start < 0:
            start = seq_len + start
        if stop < 0:
            stop = seq_len + stop
        start = max(0, start)
        stop = min(seq_len - 1, stop)
        selected_pos = random.randint(start, stop)

    else:
        raise ValueError(f"Unknown token selector: {selector_type}")

    return selected_pos


def extract_activations_batch(
    model: LanguageModel,
    batch_tokens: BatchEncoding,
    signal_specs: List[SignalSpec],
    batch_token_positions: List[int],
) -> Dict[SignalSpec, List[float]]:
    """
    Extract internal signals using NNsight.
    :param model: Language model.
    :param batch_tokens: Batch of text tokens encoded.
    :param signal_specs: Signal specifications for each piece of text in batch
    :param batch_token_positions: Batch token positions for signal.
    :return: Extracted signals.
    """
    signal_activations = {}
    mlp_signal_specs = [s for s in signal_specs if s.type == "mlp"]
    att_signal_specs = [s for s in signal_specs if s.type == "attention_score"]
    rs_signal_specs = [s for s in signal_specs if s.type == "residual_stream"]
    bs = len(batch_tokens["input_ids"])
    with torch.no_grad(), model.trace(batch_tokens):
        for layer in range(model.config.num_hidden_layers): # TODO llama3 path here, not sure for others
            # extract MLPs
            if mlp_signal_specs:
                mlp_signal_specs_layer = [s for s in mlp_signal_specs if s.layer == layer]
                if mlp_signal_specs_layer:
                    mlp_layer_path = resolve_mlp_signal_path(model, layer)
                    getter = operator.attrgetter(mlp_layer_path)
                    # assuming dimensions are [bs, token, h], output dim is [bs, h]
                    mlp_out_layer = getter(model)[range(bs), batch_token_positions].detach()
                    for signal_spec in mlp_signal_specs_layer:
                        signal_activations[signal_spec] = mlp_out_layer[:, signal_spec.index].save().tolist()

            if rs_signal_specs:
                rs_signal_specs_layer = [s for s in rs_signal_specs if s.layer == layer]
                if rs_signal_specs_layer:
                    rs_layer_path = resolve_resid_signal_path(model, layer)
                    getter = operator.attrgetter(rs_layer_path)
                    # assuming dimensions are [bs, token, h]
                    rs_out_layer = getter(model)[range(bs), batch_token_positions].detach()
                    for signal_spec in rs_signal_specs_layer:
                        signal_activations[signal_spec] = rs_out_layer[signal_spec.index].save().tolist().item()

            if att_signal_specs:
                raise NotImplementedError("avoiding the headache right now")
    return signal_activations


def create_batch_example(
    texts: List[str],
    signals: List[SignalSpec],
    token_selector: TokenSelector,
    model: LanguageModel,
    max_seq_length: int = 512,
) -> List[dict[str, Any]]:
    """
    Create a single example of (text, signal, token_position, value)
    :param texts: a batch of texts to be forwarded
    :param signals: the list of signals we want to extract per piece of text
    :param token_selector: how to pick which token(s) to use
    :param model: NNsight model wrapper
    :param max_seq_length: maximum sequence length
    :return:
    """
    encoded = model.tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=max_seq_length, padding_side="right")
    seq_lens = encoded["attention_mask"].sum(1)
    batch_token_positions = [sample_token_position(seq_len=l, token_selector=token_selector) for l in seq_lens]
    signals_activations = extract_activations_batch(
        model=model,
        batch_tokens=encoded,
        signal_specs=signals,
        batch_token_positions=batch_token_positions,
    )
    out = []

    truncated_texts = model.tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
    for data_idx, text in enumerate(texts):
        for signal in signals:
            token_position = batch_token_positions[data_idx]
            token_str = model.tokenizer.decode(encoded["input_ids"][data_idx][token_position], clean_up_tokenization_spaces=False)
            out.append({
                "text": truncated_texts[data_idx],
                "signal_type": signal.type,
                "signal_layer": signal.layer,
                "signal_index": signal.index,
                "token_position": batch_token_positions[data_idx],
                "token": token_str,
                "activation": signals_activations[signal][data_idx],
                "target": f"{signals_activations[signal][data_idx]:.{signal.precision}g}",
            })

    return out


# ============================================================
# Split processing
# ============================================================


def build_split(
    hf_dataset: Dataset,
    signals: list[SignalSpec],
    model: LanguageModel,
    token_selector: TokenSelector,
    dataset_config: DatasetConfig,
    config: ExperimentConfig,
):
    """
    Build training dataset of (text, signal, token_position, value) given specifications
    :param hf_dataset:
    :param signals: what signals to extract
    :param model: NNsight model wrapper
    :param token_selector: how to pick which token(s) to use
    :param dataset_config: DatasetConfig
    :param config: ExperimentConfig
    :return:
    """
    rows = []
    n_batches = dataset_config.max_samples // config.training_config.eval_batch_size
    for batch in tqdm(hf_dataset.batch(config.training_config.eval_batch_size), total=n_batches):
        text_batch = batch[dataset_config.text_field]
        batch_row = create_batch_example(
            texts=text_batch,
            signals=signals,
            token_selector=token_selector,
            model=model,
            max_seq_length=dataset_config.max_seq_length,
        )

        rows.extend(batch_row)

    return Dataset.from_list(rows)


def report_signal_statistics(dataset: Dataset):
    df = dataset.to_pandas()
    print(df.groupby(["signal_type", "signal_layer", "signal_index"])["activation"].describe())
    return

def report_token_overlap(train_dataset: Dataset, test_dataset: Dataset):
    train_tokens = set(train_dataset["token"])
    test_tokens = set(test_dataset["token"])
    overlap = len(train_tokens & test_tokens)
    print(f"Total train data= {len(train_dataset)}, test data= {len(test_dataset)}")
    print(f"Unique train tokens= {len(train_tokens)}, test tokens= {len(test_tokens)}, overlap= {overlap}")
    return




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(Path(args.config).read_text())

    set_seed(config.seed)

    # load datasets from hf
    if "config" in config.train_dataset.hf_data_kwargs:
        rest_kwargs = {k:v for k,v in config.train_dataset.hf_data_kwargs.items() if k != "config"}
        hf_dataset_train = load_dataset(config.train_dataset.name, config.train_dataset.hf_data_kwargs["config"],
                                        split=config.train_dataset.split, **rest_kwargs)
        hf_dataset_test = load_dataset(config.test_dataset.name, config.test_dataset.hf_data_kwargs["config"],
                                       split=config.test_dataset.split, **rest_kwargs)
    else:
        hf_dataset_train = load_dataset(config.train_dataset.name, split=config.train_dataset.split,
                                        **config.train_dataset.hf_data_kwargs)
        hf_dataset_test = load_dataset(config.test_dataset.name, split=config.test_dataset.split,
                                       **config.test_dataset.hf_data_kwargs)

    # assume dataset is large and is not truncated so using streaming mode
    if isinstance(hf_dataset_train, IterableDataset):
        hf_dataset_train = hf_dataset_train.take(config.train_dataset.max_samples)
        hf_dataset_test = hf_dataset_test.take(config.test_dataset.max_samples)


    # load model and tokenizer
    model = LanguageModel(
        config.model_name,
        device_map="auto",
        quantization_config=get_quantization_config(config)
    )
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    # now build the train/test datasets
    print("caching training dataset ..")
    train_dataset = build_split(
        hf_dataset=hf_dataset_train,
        signals=config.train_signals,
        model=model,
        token_selector=config.train_token_selector,
        dataset_config=config.train_dataset,
        config=config,
    )

    print("caching test dataset ..")
    test_dataset = build_split(
        hf_dataset=hf_dataset_test,
        signals=config.test_signals,
        model=model,
        token_selector=config.test_token_selector,
        dataset_config=config.test_dataset,
        config=config,
    )

    print("Training data signal statistics:")
    report_signal_statistics(train_dataset)
    print("Test data signal statistics:")
    report_signal_statistics(test_dataset)

    report_token_overlap(train_dataset, test_dataset)

    output_dir = Path(config.cache_dir) / config.experiment_name
    train_path = output_dir / "train"
    test_path = output_dir / "test"

    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset.save_to_disk(str(train_path))
    print(f"Saved train dataset to: {train_path}")
    test_dataset.save_to_disk(str(test_path))
    print(f"Saved test dataset to: {test_path}")


if __name__ == "__main__":
    main()