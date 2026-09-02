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
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Dict

from tqdm import tqdm
from datasets import Dataset, load_dataset
from nnsight import LanguageModel
import torch
import transformers
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, BatchEncoding

from src.introspector.config import ExperimentConfig, DatasetConfig
from src.introspector.dataset_utils import (
    canonical_dataset_name, dataset_identity, identity_hash,
)
from src.introspector.models.signal_paths import (
    resolve_mlp_signal_path, resolve_resid_signal_path, resolve_sae_hook_path,
)
from src.introspector.prompts import PROMPT_REGISTRY
from src.introspector.signals.schema import SignalSpec, TokenSelector
from src.introspector.signals.sae import load_sae
from src.introspector.utils import get_quantization_config, float_to_str_target


# One SAE per (release, site, width, layer), loaded lazily and reused across
# batches. Keyed off the SAEFeatureSpec so different SAEs never get mixed up.
_SAE_BUNDLES: dict = {}


def get_sae_bundle(spec, device):
    key = (spec.sae_release, spec.sae_site, spec.sae_width, spec.layer)
    if key not in _SAE_BUNDLES:
        _SAE_BUNDLES[key] = load_sae(
            release=spec.sae_release, site=spec.sae_site, width=spec.sae_width,
            layer=spec.layer, device=str(device),
        )
    return _SAE_BUNDLES[key]


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
        if start > stop:
            start, stop = stop, start
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
    sae_signal_specs = [s for s in signal_specs if s.type == "sae"]
    bs = len(batch_tokens["input_ids"])

    # SAE features are a non-native (encoder) readout, so we cannot index them
    # in-trace like a module output: we save the residual at the SAE's hook site
    # and encode AFTER the trace. Resolve the (static) hook paths up front, mirror-
    # ing how validate_sae_hook.py resolves before tracing.
    sae_layers = sorted({s.layer for s in sae_signal_specs})
    sae_hook_paths = {L: resolve_sae_hook_path(model, L, path_to_output=True) for L in sae_layers}
    sae_resid_proxies = {}

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

        # save full residual [bs, token, h] at each SAE hook site (encode below)
        for L in sae_layers:
            sae_resid_proxies[L] = operator.attrgetter(sae_hook_paths[L])(model).detach().save()

    # ---- SAE encode outside the trace: one bundle per (release,site,width,layer),
    #      encode the residual once, then read each requested feature at its token.
    for L in sae_layers:
        proxy = sae_resid_proxies[L]
        resid = (proxy.value if hasattr(proxy, "value") else proxy).float()  # [bs, token, h]
        specs_layer = [s for s in sae_signal_specs if s.layer == L]
        # group by SAE identity so we encode once per distinct SAE
        by_bundle: dict = {}
        for s in specs_layer:
            by_bundle.setdefault((s.sae_release, s.sae_site, s.sae_width), []).append(s)
        for _, specs in by_bundle.items():
            bundle = get_sae_bundle(specs[0], device=resid.device)
            feats = bundle.encode(resid)                     # [bs, token, d_sae]
            # [bs, d_sae] at each example's target token, then pick the feature
            feats_at_tok = feats[range(bs), batch_token_positions]   # [bs, d_sae]
            for signal_spec in specs:
                acts = feats_at_tok[:, signal_spec.index]
                signal_activations[signal_spec] = acts.float().cpu().tolist()

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
        # The EXACT token ids the activations were read from. `text` is a decoded
        # round trip, so re-tokenizing it downstream is an assumption; these ids make
        # it checkable (and make training-from-ids possible later, which would remove
        # the assumption entirely). Named text_input_ids, NOT input_ids, so it cannot
        # collide with the tokenized-prompt column preprocess_example emits.
        # Duplicated across a text's signal rows -- a few hundred bytes/row, worth it
        # against the cost of an undetected off-by-one.
        row_input_ids = encoded["input_ids"][data_idx][:int(seq_lens[data_idx])].tolist()
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
                "text_input_ids": row_input_ids,
                "activation": signals_activations[signal][data_idx],
                "target": float_to_str_target(signals_activations[signal][data_idx], signal.precision) #f"{'+' if signals_activations[signal][data_idx]>0 else ''}{signals_activations[signal][data_idx]:.{signal.precision}f}",
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
    assert len(df) > 0, "dataset is empty!"
    dscr = df.groupby(["signal_type", "signal_layer", "signal_index"])["activation"].describe()
    print(dscr)
    return dscr.reset_index().to_dict(orient="records")

def report_token_overlap(train_dataset: Dataset, test_dataset: Dataset):
    train_tokens = set(train_dataset["token"])
    test_tokens = set(test_dataset["token"])
    overlap = len(train_tokens & test_tokens)
    print(f"Total train data= {len(train_dataset)}, test data= {len(test_dataset)}")
    print(f"Unique train tokens= {len(train_tokens)}, test tokens= {len(test_tokens)}, overlap= {overlap}")
    return {
        "unique_train": len(train_tokens),
        "unique_eval": len(test_tokens),
        "unique_overlap": overlap,
        "total_train": len(train_dataset),
        "total_eval": len(test_dataset),
    }




def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def write_dataset_metadata(config: ExperimentConfig, output_dir: Path, model, row_counts: dict):
    """
    Make the cache self-describing, so "can these two runs share this dataset?" and
    "what tokenization produced these activations?" are answerable from disk instead
    of re-derived from a config (or, as happened once, from a week of debugging).

    TOKENIZATION is the field that matters most: activations were captured from
    tokenizer(..., add_special_tokens=True), i.e. WITH a leading BOS, right-padded.
    Anything consuming token_position must tokenize the same way -- see
    preprocess_example, which now does, and probe_cache_alignment.py, which checks it.
    """
    try:
        dtype = str(next(model._model.parameters()).dtype)
    except Exception:
        try:
            dtype = str(next(model.parameters()).dtype)
        except Exception:
            dtype = "unknown"
    meta = {
        "canonical_name": canonical_dataset_name(config),
        "dataset_name": config.dataset_name,
        "identity_hash": identity_hash(config),
        "identity": dataset_identity(config),
        "tokenization": {
            "add_special_tokens": True,
            "leading_bos": True,
            "padding_side": "right",
            "truncation": True,
            "max_length_per_split": {
                "train": config.train_dataset.max_seq_length,
                "dev": config.dev_dataset.max_seq_length,
                "test": config.test_dataset.max_seq_length,
            },
        },
        "capture": {"model_dtype": dtype, "signal_site": "mlp.act_fn output (post-activation neuron)"},
        "row_counts": row_counts,
        "columns_include_text_input_ids": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by_experiment": config.experiment_name,
        "git_sha": _git_sha(),
    }
    path = output_dir / "metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Wrote dataset metadata to: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="Whether to overwrite cached dataset files if there exist one")

    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(Path(args.config).read_text())

    output_dir = config.dataset_cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train"
    dev_path = output_dir / "dev"
    test_path = output_dir / "test"
    if train_path.exists() and not args.overwrite_cache:
        print("Signal dataset cached, skipping ...")
        return

    set_seed(config.seed)

    # load datasets from hf. Each split is loaded from ITS OWN DatasetConfig: the
    # previous version branched on whether the TRAIN kwargs carried "config" and then
    # indexed config.test_dataset.hf_data_kwargs["config"] unconditionally, so a test
    # corpus without a config key raised KeyError, and the test split silently inherited
    # the train split's remaining kwargs (dropping e.g. data_files). That only ever
    # showed up once train and test came from different corpora -- as in the OOD eval.
    # dev keeps using the TRAIN corpus with the dev split, which is the original behaviour.
    def _load_split(ds_cfg, split=None):
        kwargs = dict(ds_cfg.hf_data_kwargs)
        name_config = kwargs.pop("config", None)
        split = split or ds_cfg.split
        if name_config is not None:
            return load_dataset(ds_cfg.name, name_config, split=split, **kwargs)
        return load_dataset(ds_cfg.name, split=split, **kwargs)

    hf_dataset_train = _load_split(config.train_dataset)
    hf_dataset_dev = _load_split(config.train_dataset, split=config.dev_dataset.split)
    hf_dataset_test = _load_split(config.test_dataset)

    # assume dataset is large and is not truncated so using streaming mode
    if isinstance(hf_dataset_train, IterableDataset):
        hf_dataset_train = hf_dataset_train.take(config.train_dataset.max_samples)
        hf_dataset_dev = hf_dataset_dev.skip(max(config.train_dataset.max_samples, config.dev_dataset.offset)).take(config.dev_dataset.max_samples)
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

    print("caching dev dataset ..")
    dev_dataset = build_split(
        hf_dataset=hf_dataset_dev,
        signals=config.dev_signals,
        model=model,
        token_selector=config.dev_token_selector,
        dataset_config=config.dev_dataset,
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
    train_stats = report_signal_statistics(train_dataset)
    print("Dev data signal statistics:")
    dev_stats = report_signal_statistics(dev_dataset)
    print("Test data signal statistics:")
    test_stats = report_signal_statistics(test_dataset)

    print("Train dev overlap:")
    train_dev_overlap = report_token_overlap(train_dataset, dev_dataset)

    print("Train test overlap:")
    train_test_overlap = report_token_overlap(train_dataset, test_dataset)

    with open(output_dir / "dataset_signal_stats.json", "w") as f:
        json.dump({"train": train_stats, "dev": dev_stats, "test": test_stats}, f, indent=2)
    with open(output_dir / "token_overlaps.json", "w") as f:
        json.dump({"train_dev_overlaps": train_dev_overlap, "train_test_overlaps": train_test_overlap}, f, indent=2)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset.save_to_disk(str(train_path))
    print(f"Saved train dataset to: {train_path}")
    dev_dataset.save_to_disk(str(dev_path))
    print(f"Saved dev dataset to: {dev_path}")
    test_dataset.save_to_disk(str(test_path))
    print(f"Saved test dataset to: {test_path}")

    write_dataset_metadata(config, output_dir, model,
                           {"train": len(train_dataset), "dev": len(dev_dataset), "test": len(test_dataset)})


if __name__ == "__main__":
    main()