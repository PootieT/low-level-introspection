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
import contextlib
import json
import math
from collections import Counter
import operator
import pdb
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Dict

from tqdm import tqdm
from datasets import Dataset, concatenate_datasets, load_dataset
from nnsight import LanguageModel
import torch
import transformers
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, BatchEncoding

from src.introspector.config import ExperimentConfig, DatasetConfig
from src.introspector.dataset_utils import (
    canonical_dataset_name, dataset_identity, identity_hash, subsample_key,
)
from src.introspector.models.signal_paths import (
    resolve_mlp_signal_path, resolve_resid_signal_path, resolve_sae_hook_path,
)
from src.introspector.models.attention_logits import (
    AttentionCellGridTap, AttentionLogitPort, AttentionWeightPort, attention_model_kwargs,
    fan_out_ports, installed_ports, is_attention_type, port_site,
)
from src.introspector.prompts import PROMPT_REGISTRY
from src.introspector.signals.schema import SignalSpec, TokenSelector, is_pair_selector
from src.introspector.signals.pair_grid import all_pairs
from src.introspector.signals.sae import load_sae
from src.introspector.utils import get_quantization_config, float_to_str_target


# build_split flushes accumulated rows to an Arrow chunk at this size (see there).
ROWS_PER_CHUNK = 50_000

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


def attention_src_positions(spec, batch_token_positions, seq_lens) -> List[int]:
    """Absolute source positions for one attention spec, one per example.

    `batch_token_positions` are the DESTINATION positions (what the token selector
    chose); the source sits `spec.src_offset` tokens before each of them.

    Deliberately does NOT clamp. sample_token_position clamps because a token selector
    is allowed to ask for more context than a document has; here an out-of-range source
    means the cell being asked for does not exist in this document, and clamping would
    quietly substitute a DIFFERENT, plausible-looking position -- exactly the failure
    that produced scripts_old_with_off_by_one_bug/. `validate_attention_offsets` refuses
    such a config up front (raise), and this asserts the invariant per batch.
    """
    src = [dst + spec.src_offset for dst in batch_token_positions]
    for i, (s_pos, dst, seq_len) in enumerate(zip(src, batch_token_positions, seq_lens)):
        assert 0 <= s_pos <= dst < int(seq_len), (
            f"{spec.type} src position out of range for example {i}: "
            f"src={s_pos} (dst={dst} + offset={spec.src_offset}), seq_len={int(seq_len)}. "
            f"Set --min-seq-length so every document is long enough for this offset.")
    return src


def extract_activations_batch(
    model: LanguageModel,
    batch_tokens: BatchEncoding,
    signal_specs: List[SignalSpec],
    batch_token_positions: List[int],
    seq_lens=None,
    batch_src_positions: List[int] | None = None,
) -> Dict[SignalSpec, List[float]]:
    """
    Extract internal signals using NNsight.
    :param model: Language model.
    :param batch_tokens: Batch of text tokens encoded.
    :param signal_specs: Signal specifications for each piece of text in batch
    :param batch_token_positions: Batch token positions for signal (DESTINATION token
        for attention signals).
    :param seq_lens: unpadded length per example; required when attention signals are
        present, to bound-check the source positions.
    :param batch_src_positions: PAIR MODE only (specs with src_offset=None): the absolute
        SOURCE position per example, shared by every attention spec of the batch.
    :return: Extracted signals.
    """
    signal_activations = {}
    mlp_signal_specs = [s for s in signal_specs if s.type == "mlp"]
    att_signal_specs = [s for s in signal_specs if is_attention_type(s.type)]
    rs_signal_specs = [s for s in signal_specs if s.type == "residual_stream"]
    sae_signal_specs = [s for s in signal_specs if s.type == "sae"]
    bs = len(batch_tokens["input_ids"])

    # ---- attention: one port per layer, read for every (head, src_offset) at once ----
    # A port reads ONE cell per example, so a layer holding k attention signals needs k
    # ports. They all read the same [B, Hq, L, L] tensor that the single forward already
    # computes, so k costs k gathers, not k forwards -- which is what makes caching a
    # whole cell (8 heads x 10 offsets) as cheap as caching one signal.
    #
    # SPLIT BY SITE. attention_score lives before the mask and the softmax,
    # attention_weight after both, so they are two different tensors at two different
    # points of the same forward and need two port families on two attributes. Grouping
    # them by `is_attention_type` alone -- which is what this did before P was wired --
    # sends a `P` spec through AttentionLogitPort and caches the PRE-softmax value under
    # a `P` name with an "attention weight" prompt. Nothing raises; the dataset is just
    # the wrong quantity. `port_site` is the single source of truth for the mapping.
    att_ports = {"logit": {}, "weight": {}}   # site -> {spec: port}
    _PORT_CLS = {"logit": AttentionLogitPort, "weight": AttentionWeightPort}
    if att_signal_specs:
        assert seq_lens is not None, "attention signals need seq_lens for bound checks"
        for spec in att_signal_specs:
            if spec.src_offset is None:
                # PAIR MODE: the source is per row, not dst + a fixed offset.
                assert batch_src_positions is not None, (
                    f"{spec} is a pair-mode spec but no batch_src_positions were given")
                src = list(batch_src_positions)
                assert len(src) == bs, f"{len(src)} src positions for a batch of {bs}"
                for i, (s_pos, dst, seq_len) in enumerate(
                        zip(src, batch_token_positions, seq_lens)):
                    assert 0 <= s_pos <= dst < int(seq_len), (
                        f"pair out of range for example {i}: src={s_pos}, dst={dst}, "
                        f"seq_len={int(seq_len)}")
            else:
                assert batch_src_positions is None, (
                    f"{spec} has a fixed src_offset but per-row src positions were given")
                src = attention_src_positions(spec, batch_token_positions, seq_lens)
            site = port_site(spec.type)
            att_ports[site][spec] = _PORT_CLS[site](
                heads=torch.full((bs,), spec.index, dtype=torch.long),
                dst=torch.tensor(batch_token_positions, dtype=torch.long),
                src=torch.tensor(src, dtype=torch.long),
            )

    # SAE features are a non-native (encoder) readout, so we cannot index them
    # in-trace like a module output: we save the residual at the SAE's hook site
    # and encode AFTER the trace. Resolve the (static) hook paths up front, mirror-
    # ing how validate_sae_hook.py resolves before tracing.
    sae_layers = sorted({s.layer for s in sae_signal_specs})
    sae_hook_paths = {L: resolve_sae_hook_path(model, L, path_to_output=True) for L in sae_layers}
    sae_resid_proxies = {}

    # ports are keyed by layer, but several specs can share a layer, so wrap the
    # per-layer port in a fan-out that feeds each spec's own port.
    # Two managers, one per site. An empty dict attaches nothing, so a run with no
    # weight signals (every MLP, SAE and attention_score run that exists today) enters
    # and leaves the second manager without touching the model.
    logit_layer_ports = fan_out_ports(att_ports["logit"])
    weight_layer_ports = fan_out_ports(att_ports["weight"])

    with torch.no_grad(), \
         installed_ports(model, logit_layer_ports, site="logit"), \
         installed_ports(model, weight_layer_ports, site="weight"), \
         model.trace(batch_tokens):
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

            # attention logits are read by the ports installed around the trace, not
            # from a module output -- they are not the output of any module (see
            # src/introspector/models/attention_logits.py for why).

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

    # both sites, or a P run would report "never fired" for signals nobody collected
    for site, ports in att_ports.items():
        for spec, port in ports.items():
            assert port.read is not None, (
                f"{site} port for {spec} never fired -- the model is not running the "
                f"introspection attention implementation, so nothing was captured")
            assert len(port.read) == bs, f"port read {len(port.read)} values, expected {bs}"
            signal_activations[spec] = port.read.cpu().tolist()

    return signal_activations


def create_batch_example(
    texts: List[str],
    signals: List[SignalSpec],
    token_selector: TokenSelector,
    model: LanguageModel,
    max_seq_length: int = 512,
    negative_keep: float | None = None,
    seed: int = 0,
    doc_offset: int = 0,
) -> List[dict[str, Any]]:
    """
    Create a single example of (text, signal, token_position, value)
    :param texts: a batch of texts to be forwarded
    :param signals: the list of signals we want to extract per piece of text
    :param token_selector: how to pick which token(s) to use
    :param model: NNsight model wrapper
    :param max_seq_length: maximum sequence length
    :param negative_keep: keep an inactive (activation <= 0) row only if its
        subsample_key is below this; active rows are always kept, and every row gets a
        `subsample_u` column. None => keep everything, no column (DatasetConfig.negative_keep).
    :param seed: mixed into subsample_key
    :param doc_offset: index of texts[0] within its split, recorded as `doc_index`
    :return:
    """
    if token_selector.type == "pair_list":
        return create_pair_list_examples(texts, signals, token_selector, model,
                                         max_seq_length, doc_offset)
    encoded = model.tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=max_seq_length, padding_side="right")
    seq_lens = encoded["attention_mask"].sum(1)
    batch_src_positions = None
    if token_selector.type == "random_pair":
        # PAIR MODE training rows: one (dst, src) pair per document, uniform over all
        # causal pairs except (0, 0), SHARED by every signal of the document so the three
        # heads of a cell see the same pairs. Every document is exactly max_seq_length
        # tokens (validate_attention_offsets requires min_seq_length >= max_seq_length),
        # so the pair set is the same for all of them -- asserted, because a short
        # document would silently draw from a smaller set and skew the dst distribution.
        assert bool((seq_lens == max_seq_length).all()), (
            f"random_pair needs every document at exactly {max_seq_length} tokens, got "
            f"lengths {sorted(set(seq_lens.tolist()))}")
        pairs = [random.choice(all_pairs(int(l))) for l in seq_lens]
        batch_token_positions = [d for d, _ in pairs]
        batch_src_positions = [src for _, src in pairs]
    else:
        batch_token_positions = [sample_token_position(seq_len=l, token_selector=token_selector) for l in seq_lens]
    signals_activations = extract_activations_batch(
        model=model,
        batch_tokens=encoded,
        signal_specs=signals,
        batch_token_positions=batch_token_positions,
        seq_lens=seq_lens,
        batch_src_positions=batch_src_positions,
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
        id_counts = Counter(row_input_ids)
        for signal in signals:
            activation = signals_activations[signal][data_idx]
            # Subsample BEFORE building the row: at 100 features x 150k texts the dense
            # table is 15M dicts, nearly all of them inactive rows that would be dropped.
            if negative_keep is not None:
                u = subsample_key(seed, signal.type, signal.layer, signal.index,
                                  truncated_texts[data_idx])
                if activation <= 0 and u >= negative_keep:
                    continue
            token_position = batch_token_positions[data_idx]
            token_str = model.tokenizer.decode(encoded["input_ids"][data_idx][token_position], clean_up_tokenization_spaces=False)
            # SOURCE token, attention signals only. `token_position`/`token` keep meaning
            # the DESTINATION for every signal type, so nothing that reads them changes.
            # The columns are emitted on EVERY row (None for non-attention): rows
            # of one Dataset.from_list must share a key set or Arrow's schema inference
            # breaks, and nothing on the MLP path reads them.
            if not is_attention_type(signal.type):
                src_offset = src_position = src_token_str = None
                dst_unique = src_unique = None
            else:
                if batch_src_positions is not None:
                    # pair mode: the row's REALIZED offset, so offset-keyed readers
                    # (report_signal_statistics, the alignment check) still see one
                    src_position = batch_src_positions[data_idx]
                    src_offset = src_position - token_position
                else:
                    src_offset = signal.src_offset
                    src_position = token_position + src_offset
                src_token_str = model.tokenizer.decode(
                    encoded["input_ids"][data_idx][src_position], clean_up_tokenization_spaces=False)
                # PROMPT AMBIGUITY: the prompt names both tokens by string only, so a
                # token that occurs more than once in the text does not say which
                # occurrence is meant. Recorded per row so the analysis can split on it.
                dst_unique = id_counts[row_input_ids[token_position]] == 1
                src_unique = id_counts[row_input_ids[src_position]] == 1
            row = {
                "text": truncated_texts[data_idx],
                "doc_index": doc_offset + data_idx,
                "signal_type": signal.type,
                "signal_layer": signal.layer,
                "signal_index": signal.index,
                "token_position": batch_token_positions[data_idx],
                "token": token_str,
                "src_offset": src_offset,
                "src_token_position": src_position,
                "src_token": src_token_str,
                "dst_token_unique": dst_unique,
                "src_token_unique": src_unique,
                "text_input_ids": row_input_ids,
                "activation": activation,
                "target": float_to_str_target(activation, signal.precision) #f"{'+' if signals_activations[signal][data_idx]>0 else ''}{signals_activations[signal][data_idx]:.{signal.precision}f}",
            }
            if negative_keep is not None:
                row["subsample_u"] = u
            out.append(row)

    return out


def extract_attention_grid(model, batch_tokens, signal_specs, pairs, seq_lens) -> dict:
    """{spec: [B][P] values}: every (dst, src) of `pairs` for every pair-mode spec, read
    out of ONE forward pass by one AttentionCellGridTap per (site, layer).

    Runs the dispatched HF module directly rather than through model.trace: the taps do
    all the reading, so there is nothing for an nnsight trace body to do, and an empty
    traced block is not something nnsight's source tracing is guaranteed to accept.
    """
    assert signal_specs, "no signals"
    for spec in signal_specs:
        assert is_attention_type(spec.type) and spec.src_offset is None, (
            f"pair_list caches hold pair-mode attention specs only (e.g. 'P6.3@pairs'), "
            f"got {spec}")
    max_d = max(d for d, _ in pairs)
    assert int(seq_lens.min()) > max_d, (
        f"a document has {int(seq_lens.min())} tokens but the grid reaches dst={max_d}")
    dst = torch.tensor([d for d, _ in pairs], dtype=torch.long)       # [P]
    src = torch.tensor([s_ for _, s_ in pairs], dtype=torch.long)     # [P]

    groups: dict = {}                                    # (site, layer) -> sorted heads
    for spec in signal_specs:
        groups.setdefault((port_site(spec.type), spec.layer), set()).add(spec.index)
    groups = {k: sorted(v) for k, v in groups.items()}
    taps = {"logit": {}, "weight": {}}                   # site -> {layer: tap}
    for (site, layer), heads in groups.items():
        taps[site][layer] = AttentionCellGridTap(torch.tensor(heads, dtype=torch.long), dst, src)

    with torch.no_grad(), \
         installed_ports(model, taps["logit"], site="logit"), \
         installed_ports(model, taps["weight"], site="weight"):
        hf = model._model
        hf(input_ids=batch_tokens["input_ids"].to(hf.device),
           attention_mask=batch_tokens["attention_mask"].to(hf.device))

    bs = int(batch_tokens["input_ids"].shape[0])
    out = {}
    for spec in signal_specs:
        site = port_site(spec.type)
        tap = taps[site][spec.layer]
        assert tap.read is not None, (
            f"{site} grid tap for {spec} never fired -- the model is not running the "
            f"introspection attention implementation")
        assert tuple(tap.read.shape) == (bs, len(groups[(site, spec.layer)]), len(pairs)), (
            f"grid read {tuple(tap.read.shape)} != (B={bs}, H, P={len(pairs)})")
        h = groups[(site, spec.layer)].index(spec.index)
        out[spec] = tap.read[:, h, :].cpu().tolist()     # [B][P]
    return out


def create_pair_list_examples(texts, signals, token_selector, model, max_seq_length,
                              doc_offset=0) -> List[dict[str, Any]]:
    """PAIR MODE evaluation rows: one row per (document, signal, pair), same keys as
    create_batch_example so the two can share the downstream code."""
    encoded = model.tokenizer(texts, truncation=True, padding=True, return_tensors="pt",
                              max_length=max_seq_length, padding_side="right")
    seq_lens = encoded["attention_mask"].sum(1)
    pairs = [tuple(p) for p in token_selector.pairs]
    values = extract_attention_grid(model, encoded, signals, pairs, seq_lens)

    out = []
    truncated_texts = model.tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
    for data_idx in range(len(texts)):
        row_input_ids = encoded["input_ids"][data_idx][:int(seq_lens[data_idx])].tolist()
        id_counts = Counter(row_input_ids)
        # decode each position once; a 527-pair grid would otherwise decode ~1K times/doc
        tok_str = [model.tokenizer.decode([t], clean_up_tokenization_spaces=False)
                   for t in row_input_ids]
        for signal in signals:
            vals = values[signal][data_idx]
            for (d, src), v in zip(pairs, vals):
                out.append({
                    "text": truncated_texts[data_idx],
                    "doc_index": doc_offset + data_idx,
                    "signal_type": signal.type,
                    "signal_layer": signal.layer,
                    "signal_index": signal.index,
                    "token_position": d,
                    "token": tok_str[d],
                    "src_offset": src - d,
                    "src_token_position": src,
                    "src_token": tok_str[src],
                    "dst_token_unique": id_counts[row_input_ids[d]] == 1,
                    "src_token_unique": id_counts[row_input_ids[src]] == 1,
                    "text_input_ids": row_input_ids,
                    "activation": v,
                    "target": float_to_str_target(v, signal.precision),
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
    batch_size = config.training_config.eval_batch_size
    n_batches = math.ceil(dataset_config.max_samples / batch_size)

    # Two independent bounds, because the failure mode is expensive and silent.
    # `i >= n_batches` is the structural one: it holds even if the caller forgets to
    # .take() the stream, which has happened -- an unbounded test stream walks the whole
    # split (364,608 documents for C4-en validation) without erroring.
    #
    # Rows are flushed to Arrow in chunks rather than held as Python dicts to the end.
    # A row carries `text` and `text_input_ids` duplicated once PER SIGNAL, so a 200-neuron
    # cache over 3k texts is 600k dicts -- ~1.7 GB as Python objects, and Dataset.from_list
    # then builds the Arrow table on top of that, roughly doubling the peak. Arrow holds
    # the same content several times smaller, so flushing bounds the peak at one chunk.
    # A pair-list split is documents x signals x pairs rows (3M at 2K docs, 3 heads,
    # 527 pairs); without flushing that costs tens of GB.
    shards, rows = [], []

    def _flush():
        if rows:
            shards.append(Dataset.from_list(rows))
            rows.clear()

    seen = 0
    for i, batch in enumerate(tqdm(hf_dataset.batch(batch_size), total=n_batches)):
        if seen >= dataset_config.max_samples:
            print(f"[cache] stream yielded more than {dataset_config.max_samples} samples; "
                  f"stopping at the configured limit")
            break
        text_batch = batch[dataset_config.text_field]
        # trim the final batch: max_samples need not be a multiple of the batch size, and
        # at many signals per text an overshoot of a few texts is a few thousand rows
        room = dataset_config.max_samples - seen
        if len(text_batch) > room:
            text_batch = text_batch[:room]
        batch_row = create_batch_example(
            texts=text_batch,
            signals=signals,
            token_selector=token_selector,
            model=model,
            max_seq_length=dataset_config.max_seq_length,
            negative_keep=dataset_config.negative_keep,
            seed=config.seed,
            doc_offset=seen,
        )
        seen += len(text_batch)

        rows.extend(batch_row)
        if len(rows) >= ROWS_PER_CHUNK:
            _flush()
    _flush()

    if not shards:
        return Dataset.from_list([])
    return shards[0] if len(shards) == 1 else concatenate_datasets(shards)

def report_signal_statistics(dataset: Dataset, pair_mode: bool = False):
    df = dataset.to_pandas()
    assert len(df) > 0, "dataset is empty!"
    # src_offset joins the key when present: two attention signals of the same
    # (layer, head) at different source offsets are DIFFERENT signals, and grouping
    # without it would average them into one meaningless std.
    keys = ["signal_type", "signal_layer", "signal_index"]
    # Not in pair mode, where src_offset is each row's REALIZED offset (hundreds of
    # values, one signal) rather than part of the signal identity.
    if "src_offset" in df.columns and df["src_offset"].notna().all() and not pair_mode:
        keys.append("src_offset")
    dscr = df.groupby(keys)["activation"].describe()
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


def validate_attention_offsets(config, signal_specs) -> None:
    """Refuse an attention config whose source offsets cannot be honoured.

    A cached document is `seq_len` tokens; the destination sits at some position `dst`
    and the source at `dst + src_offset`. If `min_seq_length` is unset, C4 supplies
    documents shorter than `max_seq_length`, and on those `sample_token_position` clamps
    the destination toward the front -- so an offset of -31 silently resolves to position
    0 (BOS) for a short document and to a real token for a long one. The BOS/attention-
    sink measurement would then be right only by accident.

    Requiring `min_seq_length >= |min offset| + 1` makes every document long enough for
    the deepest offset. Setting min == max (e.g. both 32) additionally pins every
    document to exactly that length, so dst=-1 is position L-1 and offset -(L-1) is
    exactly BOS. Config input, so this raises rather than asserts.
    """
    att = [s for s in signal_specs if is_attention_type(s.type)]
    selectors = (("train", config.train_dataset, config.train_token_selector),
                 ("dev", config.dev_dataset, config.dev_token_selector),
                 ("test", config.test_dataset, config.test_token_selector))
    pair_specs = [s for s in att if s.src_offset is None]
    pair_sels = [name for name, _, sel in selectors if is_pair_selector(sel)]
    if pair_specs or pair_sels:
        _validate_pair_mode(att, pair_specs, selectors)
        return
    if not att:
        return
    deepest = min(s.src_offset for s in att)          # most negative
    need = abs(deepest) + 1                            # source, destination, and between
    for name, ds_cfg, sel in (
        ("train", config.train_dataset, config.train_token_selector),
        ("dev", config.dev_dataset, config.dev_token_selector),
        ("test", config.test_dataset, config.test_token_selector),
    ):
        floor = ds_cfg.min_seq_length
        if not floor or floor < need:
            raise ValueError(
                f"attention signals reach {deepest} tokens back, which needs every "
                f"{name} document to be at least {need} tokens, but "
                f"min_seq_length={floor!r}. Pass --min-seq-length {need} (or more) so "
                f"short documents are filtered out instead of having their source "
                f"position silently clamped onto a different token.")
        if sel.type == "fixed" and sel.position >= 0 and sel.position + deepest < 0:
            raise ValueError(
                f"{name} token selector is fixed at position {sel.position}, so a source "
                f"offset of {deepest} lands at {sel.position + deepest} (before BOS). "
                f"Use a negative (end-relative) destination or a shallower offset.")


def _validate_pair_mode(att, pair_specs, selectors) -> None:
    """PAIR MODE config checks (experiment 3-1). Config input, so raise.

    A pair-mode spec ("P6.3@pairs") has no offset of its own, so it is meaningless
    without a pair selector, and a pair selector has nothing to read from a fixed-offset
    spec. Mixing the two would cache rows whose `src_offset` means different things.
    """
    if len(pair_specs) != len(att):
        raise ValueError(
            f"pair-mode attention specs (@pairs) cannot be mixed with fixed-offset ones in "
            f"one config: {[str(s) for s in att if s.src_offset is not None][:5]}")
    for name, ds_cfg, sel in selectors:
        if not is_pair_selector(sel):
            raise ValueError(
                f"{name} token selector is {sel.type!r}, but @pairs signals need a pair "
                f"selector (random_pair or pair_grid:<m>) on every split")
        floor, cap = ds_cfg.min_seq_length, ds_cfg.max_seq_length
        if sel.type == "random_pair":
            # Every document must be exactly `cap` tokens: then all of them share one
            # pair set and the uniform-over-pairs draw means what it says. A shorter
            # document would draw from a smaller set and skew dst toward the front.
            if not floor or floor < cap:
                raise ValueError(
                    f"{name}: random_pair needs min_seq_length >= max_seq_length "
                    f"({cap}) so every document is exactly {cap} tokens; got "
                    f"min_seq_length={floor!r}")
        else:
            max_d = max(d for d, _ in sel.pairs)
            if not (max_d < cap and floor and cap <= floor):
                raise ValueError(
                    f"{name}: pair_list reaches dst={max_d}, which needs "
                    f"max_d < max_seq_length <= min_seq_length; got max_seq_length={cap}, "
                    f"min_seq_length={floor!r}")


def _raw_span(n_filtered: int, floor, config, tokenizer, probe: int = 2000) -> int:
    """How many RAW documents a take(n_filtered) actually consumes.

    Without a floor the two are equal. With one, the filter's pass rate is estimated on
    a short prefix and the span is n/pass_rate. This is what makes dev's raw offset
    checkable: train takes n FILTERED documents but walks past far more raw ones, and
    an offset that looked safe against max_samples can land inside train's raw span.
    """
    if not floor or not n_filtered:
        return n_filtered or 0
    ds_cfg = config.train_dataset
    kwargs = dict(ds_cfg.hf_data_kwargs)
    name_config = kwargs.pop("config", None)
    args = (ds_cfg.name, name_config) if name_config else (ds_cfg.name,)
    stream = load_dataset(*args, split=ds_cfg.split, **kwargs)
    kept = seen = 0
    for ex in stream:
        seen += 1
        if len(tokenizer(ex[ds_cfg.text_field], add_special_tokens=True)["input_ids"]) >= floor:
            kept += 1
        if seen >= probe:
            break
    rate = max(kept / seen, 1e-3) if seen else 1.0
    span = int(n_filtered / rate)
    print(f"[cache] min_seq_length={floor}: ~{rate*100:.1f}% of documents pass "
          f"(probe n={seen}); take({n_filtered}) spans ~{span} raw documents")
    return span


def _assert_dev_disjoint(dev_skip: int, train_raw_span: int, floor) -> None:
    """Dev's raw offset must clear the raw span train consumed, or the splits overlap."""
    if dev_skip >= train_raw_span:
        return
    raise ValueError(
        f"dev offset ({dev_skip} raw documents) lands inside train's raw span "
        f"(~{train_raw_span} documents: max_samples filtered at min_seq_length={floor}). "
        f"Train and dev would overlap. Raise --dev-offset-from-train above "
        f"{train_raw_span}, or point dev at a different split.")


def _realized_stats(dataset) -> dict:
    """Observed sequence-length and introspected-position distribution of a built split.

    `distance_from_end` is the gap between the introspected token and the end of the
    text, i.e. how far back the readout has to reach -- the axis a length-generalization
    run is actually varying, and the one that quietly collapses when documents are
    shorter than max_seq_length.
    """
    import numpy as np
    if len(dataset) == 0:
        return {}
    if "text_input_ids" in dataset.column_names:
        seq_len = np.array([len(x) for x in dataset["text_input_ids"]])
    else:
        return {}
    pos = np.array(dataset["token_position"])
    dist = seq_len - 1 - pos
    pct = [5, 25, 50, 75, 95]
    out = {
        "seq_len": {"min": int(seq_len.min()), "max": int(seq_len.max()),
                    "mean": round(float(seq_len.mean()), 1),
                    "pct": {p: int(np.percentile(seq_len, p)) for p in pct}},
        "token_position": {"min": int(pos.min()), "max": int(pos.max()),
                           "pct": {p: int(np.percentile(pos, p)) for p in pct}},
        "distance_from_end": {"min": int(dist.min()), "max": int(dist.max()),
                              "mean": round(float(dist.mean()), 1),
                              "pct": {p: int(np.percentile(dist, p)) for p in pct}},
    }
    if "src_token_position" in dataset.column_names:
        src = [x for x in dataset["src_token_position"] if x is not None]
        if src:
            src = np.array(src)
            out["src_token_position"] = {
                "min": int(src.min()), "max": int(src.max()),
                "pct": {p: int(np.percentile(src, p)) for p in pct}}
            # PROMPT AMBIGUITY, measured rather than assumed. The prompt names the
            # source token by its string only, so when that token appears more than once
            # in the 32-token context the prompt does not identify WHICH occurrence is
            # meant. That is a confound the MLP experiments never had (they name one
            # token). Recording the rate here lets the analysis split corr by
            # unique/duplicated instead of guessing how much it costs.
            # create_batch_example records the flags per row; recounting here would
            # walk text_input_ids once per row (900K rows x 512 tokens on a pair grid).
            for col in ("src_token_unique", "dst_token_unique"):
                if col in dataset.column_names:
                    flags = [x for x in dataset[col] if x is not None]
                    out[f"{col}_rate"] = round(float(np.mean(flags)), 4) if flags else None
    return out


_SIGNAL_SITES = {
    "mlp": "mlp.act_fn output (post-activation neuron)",
    "attention_score": "pre-softmax attention logit q.k/sqrt(d_head), after RoPE, "
                       "before causal mask and softmax",
    "attention_weight": "post-softmax attention weight (probability over the causal "
                        "row, read in fp32 before the dtype cast and before dropout; "
                        "rows sum to 1)",
    "residual_stream": "decoder layer output (residual stream)",
    "sae": "SAE feature (encode of the decoder-layer residual output)",
}


def _signal_site(specs) -> str:
    """Human description of WHAT was captured. Was hardcoded to the MLP string, which
    would have mislabelled every attention cache built after this."""
    kinds = sorted({s.type for s in specs})
    return " + ".join(_SIGNAL_SITES.get(k, k) for k in kinds) if kinds else "none"


def write_dataset_metadata(config: ExperimentConfig, output_dir: Path, model,
                           row_counts: dict, realized: dict | None = None):
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
        "capture": {"model_dtype": dtype, "signal_site": _signal_site(config.train_signals),
                    "attn_implementation": getattr(model.config, "_attn_implementation", None)},
        "row_counts": row_counts,
        # What the cache ACTUALLY contains, not what was requested. A length sweep is
        # only meaningful if the documents really reach the nominal length: max_seq_length
        # truncates but does not guarantee, and sample_token_position clamps a range into
        # whatever each document has. Comparing realized[] against the configured
        # max_seq_length / token_selector is how a silently collapsed length or distance
        # axis becomes visible on disk instead of showing up as a flat result.
        "realized": realized,
        "columns_include_text_input_ids": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by_experiment": config.experiment_name,
        "git_sha": _git_sha(),
    }
    path = output_dir / "metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Wrote dataset metadata to: {path}")


def build_pair_pool(config: ExperimentConfig, out_dir: Path) -> None:
    """Write the PAIR MODE injection pool (see dataset_utils.POOL_DOCS for what and why).

    Compact columns only -- no text, no token ids: 4K documents x 3 heads x ~600 pairs is
    7M rows, which with the usual per-row text_input_ids would be ~50 GB of copies that
    nothing reads. Written to a temporary directory and renamed, so a reader can never
    see a half-written pool.
    """
    import numpy as np
    from src.introspector.dataset_utils import POOL_DOCS, POOL_SEQ_LEN, pair_pool_identity
    from src.introspector.signals.pair_grid import pool_pairs

    specs = list(config.train_signals)
    if not specs or any(not is_attention_type(s.type) or s.src_offset is not None for s in specs):
        raise ValueError(f"--pair-pool needs pair-mode attention signals (e.g. P6.3@pairs), "
                         f"got {[str(s) for s in specs]}")
    floor = config.train_dataset.min_seq_length
    if not floor or floor < POOL_SEQ_LEN:
        raise ValueError(f"the pool reads {POOL_SEQ_LEN}-token documents, so the corpus floor "
                         f"must be >= {POOL_SEQ_LEN}; got min_seq_length={floor!r}")
    pairs = list(pool_pairs())
    set_seed(config.seed)

    ds_cfg = config.train_dataset
    kwargs = dict(ds_cfg.hf_data_kwargs)
    name_config = kwargs.pop("config", None)
    args = (ds_cfg.name, name_config) if name_config else (ds_cfg.name,)
    stream = load_dataset(*args, split=ds_cfg.split, **kwargs)
    tok = AutoTokenizer.from_pretrained(config.model_name)
    stream = stream.filter(lambda ex: len(tok(ex[ds_cfg.text_field],
                                              add_special_tokens=True)["input_ids"]) >= floor)
    stream = stream.take(POOL_DOCS)

    model = LanguageModel(config.model_name, device_map="auto",
                          quantization_config=get_quantization_config(config),
                          **attention_model_kwargs(specs))
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    cols = {k: [] for k in ("doc_index", "signal_layer", "signal_index", "token_position",
                            "src_token_position", "activation")}
    n_docs, bs = 0, config.training_config.eval_batch_size
    for batch in tqdm(stream.batch(bs), total=POOL_DOCS // bs):
        texts = batch[ds_cfg.text_field]
        encoded = model.tokenizer(texts, truncation=True, padding=True, return_tensors="pt",
                                  max_length=POOL_SEQ_LEN, padding_side="right")
        seq_lens = encoded["attention_mask"].sum(1)
        assert bool((seq_lens == POOL_SEQ_LEN).all()), (
            f"pool documents must be {POOL_SEQ_LEN} tokens, got {sorted(set(seq_lens.tolist()))}")
        values = extract_attention_grid(model, encoded, specs, pairs, seq_lens)
        B, P = len(texts), len(pairs)
        d_col = np.array([d for d, _ in pairs] * B)                  # [B*P]
        s_col = np.array([s_ for _, s_ in pairs] * B)                # [B*P]
        doc_col = np.repeat(np.arange(n_docs, n_docs + B), P)        # [B*P]
        for spec in specs:
            v = np.asarray(values[spec], dtype=np.float32)           # [B, P]
            assert v.shape == (B, P), f"{spec}: {v.shape} != {(B, P)}"
            cols["doc_index"].append(doc_col)
            cols["signal_layer"].append(np.full(B * P, spec.layer))
            cols["signal_index"].append(np.full(B * P, spec.index))
            cols["token_position"].append(d_col)
            cols["src_token_position"].append(s_col)
            cols["activation"].append(v.reshape(-1))
        n_docs += B
    if n_docs != POOL_DOCS:
        raise RuntimeError(f"pool stream yielded {n_docs} documents, wanted {POOL_DOCS}")

    data = {k: np.concatenate(v) for k, v in cols.items()}
    n = len(data["activation"])
    data["signal_type"] = [specs[0].type] * n
    data["src_offset"] = data["src_token_position"] - data["token_position"]
    lo, hi = float(data["activation"].min()), float(data["activation"].max())
    assert 0.0 <= lo and hi <= 1.0, f"pool values outside [0, 1]: [{lo}, {hi}]"

    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    Dataset.from_dict(data).save_to_disk(str(tmp))
    with open(tmp / "pool_metadata.json", "w") as f:
        json.dump({"identity": pair_pool_identity(config), "n_rows": n, "n_docs": n_docs,
                   "n_pairs": len(pairs), "value_range": [lo, hi],
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "git_sha": _git_sha()}, f, indent=2, default=str)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():                 # only reachable under --overwrite-cache
        import shutil
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    print(f"[pool] wrote {n} rows ({n_docs} docs x {len(specs)} signals x {len(pairs)} pairs) "
          f"-> {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="Whether to overwrite cached dataset files if there exist one")
    parser.add_argument("--pair-pool", action="store_true",
                        help="PAIR MODE: build ONLY the injection pool for this config's "
                             "signals (dataset_utils.pair_pool_dir), not the train/dev/test "
                             "cache. Skips if the pool exists.")

    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(Path(args.config).read_text())

    if args.pair_pool:
        from src.introspector.dataset_utils import pair_pool_dir
        pool_dir = pair_pool_dir(config)
        if pool_dir.exists() and not args.overwrite_cache:
            print(f"Pair pool cached, skipping ... {pool_dir}")
            return
        build_pair_pool(config, pool_dir)
        return

    output_dir = config.dataset_cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train"
    dev_path = output_dir / "dev"
    test_path = output_dir / "test"
    # metadata.json is written after all three splits, so it marks a COMPLETE cache; a
    # build killed midway (train saved, test not) is rebuilt instead of skipped.
    if (output_dir / "metadata.json").exists() and not args.overwrite_cache:
        print("Signal dataset cached, skipping ...")
        return
    if train_path.exists() and not args.overwrite_cache:
        print("Incomplete cache (no metadata.json); rebuilding ...")

    set_seed(config.seed)

    all_signals = list(config.train_signals) + list(config.dev_signals) + list(config.test_signals)
    validate_attention_offsets(config, all_signals)

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

    # Length floor, applied BEFORE skip/take. Order matters twice over:
    #   - take() before filter() would keep only the long documents *within* the first
    #     max_samples, silently returning far fewer rows than asked for (at a 512-token
    #     floor on C4, ~25% of them).
    #   - dev's skip() counts documents, so it has to count in the SAME (filtered) space
    #     as train's take(), or train and dev stop being disjoint.
    # Costs a longer stream, not more compute: a 512-token floor on C4 reads ~4x the
    # documents to yield the same number of rows.
    min_len_cfgs = {config.train_dataset.min_seq_length, config.dev_dataset.min_seq_length,
                    config.test_dataset.min_seq_length}
    if len({m for m in min_len_cfgs if m} ) > 1:
        # A per-split floor would make train and test draw from different document
        # populations -- a content shift masquerading as a generalization result.
        raise ValueError(f"min_seq_length must match across splits, got {min_len_cfgs}")

    def _apply_min_len(ds, ds_cfg, tokenizer):
        if not ds_cfg.min_seq_length:
            return ds
        field, floor = ds_cfg.text_field, ds_cfg.min_seq_length
        return ds.filter(
            lambda ex: len(tokenizer(ex[field], add_special_tokens=True)["input_ids"]) >= floor)

    length_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    floor = config.train_dataset.min_seq_length

    # DEV's skip happens on the RAW stream, BEFORE the filter; train and test filter
    # first. The asymmetry is deliberate and it is a large constant factor:
    # .skip(n) on a filtered stream has to TOKENIZE every document it walks past, at
    # roughly 1.1k docs/s versus 2.9k docs/s for the raw stream -- and it walks past
    # n/pass_rate of them, not n. Measured on C4-en at a 128-token floor, skipping a
    # million documents costs ~21 min filtered against ~6 min raw. That reads as a hang.
    # Filtering AFTER the skip is still correct as long as the raw offset clears the raw
    # span train consumed, which _assert_dev_disjoint checks below instead of assuming.
    hf_dataset_train = _apply_min_len(hf_dataset_train, config.train_dataset, length_tokenizer)
    hf_dataset_test = _apply_min_len(hf_dataset_test, config.test_dataset, length_tokenizer)

    # TEST was never truncated. Train and dev are take()n below, but the test split fell
    # through: under streaming, `for batch in hf_dataset.batch(...)` in build_split runs
    # to the END OF THE CORPUS, so --test-max-samples 3000 actually cached all ~45K
    # documents of c4 validation. The tqdm total is computed from max_samples, so the bar
    # just runs past 100% -- no error, only a much larger and slower test split than the
    # config asked for. It has to be truncated the same way train and dev are.
    # (This does not move any cache key, so caches already on disk are still found and
    # reused unchanged; only freshly built ones get the configured size.)
    if config.test_dataset.max_samples:
        hf_dataset_test = (hf_dataset_test.take(config.test_dataset.max_samples)
                           if isinstance(hf_dataset_test, IterableDataset)
                           else hf_dataset_test.select(
                               range(min(config.test_dataset.max_samples, len(hf_dataset_test)))))

    # assume dataset is large and is not truncated so using streaming mode
    if isinstance(hf_dataset_train, IterableDataset):
        hf_dataset_train = hf_dataset_train.take(config.train_dataset.max_samples)

        # Only guard against train when dev is drawn from the SAME split. Previously the
        # max() applied unconditionally, so pointing dev at a different split still paid
        # a max_samples-sized skip it had no reason to pay.
        same_stream = config.dev_dataset.split == config.train_dataset.split
        if same_stream:
            train_raw_span = _raw_span(config.train_dataset.max_samples, floor,
                                       config, length_tokenizer)
            dev_skip = max(train_raw_span, config.dev_dataset.offset)
            _assert_dev_disjoint(dev_skip, train_raw_span, floor)
        else:
            dev_skip = config.dev_dataset.offset
            # Dev on a different split from train usually means the TEST split, and then
            # the collision to worry about is with test, not train. Same trap as above:
            # test takes max_samples FILTERED rows and walks past more raw ones, so an
            # offset equal to test's max_samples still lands inside it.
            if (config.dev_dataset.split == config.test_dataset.split
                    and config.train_dataset.name == config.test_dataset.name):
                test_raw_span = _raw_span(config.test_dataset.max_samples, floor,
                                          config, length_tokenizer)
                if dev_skip < test_raw_span:
                    raise ValueError(
                        f"dev offset ({dev_skip}) lands inside TEST's raw span "
                        f"(~{test_raw_span} documents) -- dev and test share split "
                        f"'{config.test_dataset.split}' of {config.test_dataset.name} and "
                        f"would overlap. Raise --dev-offset-from-train above {test_raw_span}.")
        print(f"[cache] dev: skip {dev_skip} raw documents"
              f"{' (same split as train)' if same_stream else ' (separate split)'}")
        hf_dataset_dev = hf_dataset_dev.skip(dev_skip)
        hf_dataset_dev = _apply_min_len(hf_dataset_dev, config.train_dataset, length_tokenizer)
        hf_dataset_dev = hf_dataset_dev.take(config.dev_dataset.max_samples)

        # DO NOT REMOVE. Without this the test stream is unbounded and build_split walks
        # the entire split -- 364,608 documents for C4-en validation instead of the
        # configured few thousand. It does not error; it just runs ~100x too long and,
        # at many signals per text, accumulates enough rows to exhaust memory.
        hf_dataset_test = hf_dataset_test.take(config.test_dataset.max_samples)
    else:
        hf_dataset_dev = _apply_min_len(hf_dataset_dev, config.train_dataset, length_tokenizer)


    # load model and tokenizer
    #
    # attention_model_kwargs returns {} unless an attention signal is being cached, so a
    # neuron/SAE cache is still built under sdpa and its values are bit-identical to what
    # they always were. When it does fire, the SAME implementation must be used by the
    # trainer, or the cached value and the live value disagree at the bf16 level and the
    # alignment check will not reach corr ~0.9999.
    model = LanguageModel(
        config.model_name,
        device_map="auto",
        quantization_config=get_quantization_config(config),
        **attention_model_kwargs(all_signals),
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
    pair_mode = is_pair_selector(config.train_token_selector)
    train_stats = report_signal_statistics(train_dataset, pair_mode)
    print("Dev data signal statistics:")
    dev_stats = report_signal_statistics(dev_dataset, pair_mode)
    print("Test data signal statistics:")
    test_stats = report_signal_statistics(test_dataset, pair_mode)

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
                           {"train": len(train_dataset), "dev": len(dev_dataset), "test": len(test_dataset)},
                           realized={"train": _realized_stats(train_dataset),
                                     "dev": _realized_stats(dev_dataset),
                                     "test": _realized_stats(test_dataset)})


if __name__ == "__main__":
    main()