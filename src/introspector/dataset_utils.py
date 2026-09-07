"""
Shared-cache plumbing: canonical dataset names, dataset identity, signal filtering.

WHY
---
Caches used to be keyed by experiment_name, so every hyperparameter arm rebuilt an
identical dataset -- one full copy of the same 1M rows per learning rate. The
`dataset_name` field in ExperimentConfig fixes that (runs sharing a name share a
cache), but only if the name is derived RELIABLY from what actually determines the
cached content. Hand-written names silently collide (two configs, one name,
different data) or silently duplicate (one dataset, two names).

canonical_dataset_name() derives the name from the config: a readable prefix so the
directory is browsable, plus a short hash of the FULL identity so two datasets that
differ in any content-determining field can never share a directory.

    llama-3.2-1b/c4-en_N10.24_pos-1_seq32_300K_3f9a1c2b

Signals live in the same cache: caching 10 neurons of one layer costs ONE forward
pass per text (create_batch_example loops signals inside the batch), so a
(layer, position) cell of a sweep should be cached once with all its neurons and
filtered at train time -- see apply_signal_filter.
"""
from __future__ import annotations

import re
import json
import hashlib
from typing import Any, Optional

_TYPE_CODE = {"mlp": "N", "attention_score": "A", "residual_stream": "R", "sae": "S"}
# "N10.24" / "S12.4567" / bare "10.24" (assumed mlp)
_FILTER_RE = re.compile(r"^(?P<code>[NARS])?(?P<layer>\d+)\.(?P<index>\d+)$")


def _short_model(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def _short_dataset(ds_cfg) -> str:
    base = ds_cfg.name.split("/")[-1].lower()
    conf = ds_cfg.hf_data_kwargs.get("config")
    return f"{base}-{conf}" if conf else base


def _short_signals(specs) -> str:
    """Compact, readable signal descriptor: exact ids when there are few, a
    layer-and-count summary when there are many (the hash keeps it unambiguous)."""
    if not specs:
        return "nosig"
    codes = {_TYPE_CODE.get(s.type, "X") for s in specs}
    code = codes.pop() if len(codes) == 1 else "mix"
    if len(specs) <= 3:
        ordered = sorted(specs, key=lambda s: (s.layer, s.index))
        return "+".join(f"{_TYPE_CODE.get(s.type, 'X')}{s.layer}.{s.index}" for s in ordered)
    layers = sorted({s.layer for s in specs})
    span = f"L{layers[0]}" if len(layers) == 1 else f"L{layers[0]}-{layers[-1]}"
    return f"{code}{span}x{len(specs)}"


def _short_selector(sel) -> str:
    if sel.type == "fixed":
        return f"pos{sel.position}"
    if sel.type == "random_range":
        lo, hi = sorted((sel.start, sel.stop))
        return f"posR{lo}to{hi}"
    return f"pos{sel.type}"


def _short_count(n: Optional[int]) -> str:
    if not n:
        return "all"
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1_000 == 0:
        return f"{n // 1_000}K"
    return str(n)


def _spec_identity(spec) -> dict[str, Any]:
    """Every field of a signal spec that changes the cached value."""
    d = {"type": spec.type, "layer": spec.layer, "index": spec.index, "precision": spec.precision}
    for extra in ("sae_release", "sae_site", "sae_width"):
        if hasattr(spec, extra):
            d[extra] = getattr(spec, extra)
    return d


def _split_identity(ds_cfg) -> dict[str, Any]:
    return {
        "name": ds_cfg.name,
        "split": ds_cfg.split,
        "text_field": ds_cfg.text_field,
        "max_samples": ds_cfg.max_samples,
        "max_seq_length": ds_cfg.max_seq_length,
        # Part of the IDENTITY, not just a build option: a floor changes which documents
        # are in the cache, so a filtered and an unfiltered build at the same
        # max_seq_length are different datasets. Without this they hash alike and the
        # second run silently reuses the first one's cache.
        "min_seq_length": ds_cfg.min_seq_length,
        "offset": ds_cfg.offset,
        "hf_data_kwargs": ds_cfg.hf_data_kwargs,
    }


def dataset_identity(config) -> dict[str, Any]:
    """
    Everything that determines the CONTENT of a cached dataset -- and nothing that
    does not. Two configs with equal identities may share a cache directory; two
    with different identities must not. Deliberately excludes experiment_name,
    output_dir, and every training hyperparameter.

    NOTE: also excludes the tokenization convention, which cache_signals.py fixes
    (add_special_tokens=True, i.e. BOS). If that ever becomes configurable it MUST
    be added here, or old and new caches will collide.
    """
    return {
        "model_name": config.model_name,
        "seed": config.seed,
        "load_in_4bit": config.load_in_4bit,
        "load_in_8bit": config.load_in_8bit,
        "splits": {
            "train": _split_identity(config.train_dataset),
            "dev": _split_identity(config.dev_dataset),
            "test": _split_identity(config.test_dataset),
        },
        "signals": {
            "train": [_spec_identity(s) for s in config.train_signals],
            "dev": [_spec_identity(s) for s in config.dev_signals],
            "test": [_spec_identity(s) for s in config.test_signals],
        },
        "token_selectors": {
            "train": config.train_token_selector.model_dump(),
            "dev": config.dev_token_selector.model_dump(),
            "test": config.test_token_selector.model_dump(),
        },
    }


def identity_hash(config, length: int = 8) -> str:
    payload = json.dumps(dataset_identity(config), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:length]


def canonical_dataset_name(config) -> str:
    """
    Deterministic cache key: `<model>/<dataset>_<signals>_<selector>_seq<L>_<n>_<hash>`.

    The prefix is for humans (browse `data/llama-3.2-1b/` and see what exists); the
    hash is for correctness. Runs that differ only in training hyperparameters get
    the SAME name and reuse one cache; runs differing in any content field get
    different names automatically.
    """
    return "/".join([
        _short_model(config.model_name),
        "_".join([
            _short_dataset(config.train_dataset),
            _short_signals(config.train_signals),
            _short_selector(config.train_token_selector),
            f"seq{config.train_dataset.max_seq_length}",
            _short_count(config.train_dataset.max_samples),
            identity_hash(config),
        ]),
    ])


def parse_signal_filter(spec: str) -> dict[str, Any]:
    """'N10.24' / '10.24' -> {'signal_type': 'mlp', 'signal_layer': 10, 'signal_index': 24}.

    ONE signal. Callers that select a single feature (the SAE detector) depend on that,
    so a comma-separated list is rejected rather than silently truncated: _FILTER_RE is
    unanchored at the end and would happily match the first entry of 'N1.2,N3.4' and
    drop the rest. Use parse_signal_filters for a list.
    """
    if "," in spec:
        raise ValueError(
            f"parse_signal_filter takes ONE signal, got {spec!r}. Use parse_signal_filters().")
    m = _FILTER_RE.match(spec.strip())
    if not m:
        raise ValueError(f"Bad signal filter {spec!r}; expected e.g. 'N10.24' (type letter optional).")
    code = m.group("code") or "N"
    type_by_code = {v: k for k, v in _TYPE_CODE.items()}
    if code not in type_by_code:
        raise ValueError(f"Unknown signal type code {code!r} in {spec!r}; one of {sorted(type_by_code)}.")
    return {"signal_type": type_by_code[code],
            "signal_layer": int(m.group("layer")),
            "signal_index": int(m.group("index"))}


def parse_signal_filters(spec: str) -> list[dict[str, Any]]:
    """Comma-separated signals -> list of match dicts. 'N10.24' gives a 1-element list."""
    parts = [p for p in (x.strip() for x in spec.split(",")) if p]
    if not parts:
        raise ValueError(f"Empty signal filter {spec!r}.")
    return [parse_signal_filter(p) for p in parts]


def apply_signal_filter(dataset, spec: Optional[str], split_name: str = ""):
    """
    Keep only the rows for the named signal(s) out of a shared multi-signal cache.

    This is what makes per-neuron runs affordable in a sweep: cache a (layer,
    position) cell ONCE with all its neurons -- one forward pass per text serves
    every neuron -- then give each run its own signal here. Returns the dataset
    unchanged when spec is None.

    Accepts a COMMA-SEPARATED list as well as a single signal, which is what lets a
    multi-neuron run train on a subset of a shared cache, and lets an eval split the
    same cached test split into trained and held-out neurons without building a second
    dataset. Rows matching ANY listed signal are kept.
    """
    if not spec:
        return dataset
    wants = parse_signal_filters(spec)
    keys = {(w["signal_type"], w["signal_layer"], w["signal_index"]) for w in wants}
    before = len(dataset)
    filtered = dataset.filter(
        lambda ex: (ex["signal_type"], ex["signal_layer"], ex["signal_index"]) in keys)
    if len(filtered) == 0:
        present = sorted({(t, l, i) for t, l, i in zip(dataset["signal_type"], dataset["signal_layer"],
                                                       dataset["signal_index"])})
        raise SystemExit(f"signal_filter {spec!r} matched 0 of {before} rows in {split_name or 'dataset'}. "
                         f"Signals present: {present[:20]}{' ...' if len(present) > 20 else ''}")
    # A partial match is worth naming: asking for 100 neurons and getting 60 means the
    # cache was built with a different set, and the run would quietly train on fewer.
    got = {(t, l, i) for t, l, i in zip(filtered["signal_type"], filtered["signal_layer"],
                                        filtered["signal_index"])}
    missing = keys - got
    label = spec if len(wants) <= 3 else f"{len(wants)} signals"
    if missing:
        raise SystemExit(
            f"signal_filter ({label}) in {split_name or 'dataset'}: "
            f"{len(missing)} of {len(keys)} requested signals are absent from the cache, "
            f"e.g. {sorted(missing)[:5]}. The cache was built with a different signal set.")
    print(f"[signal_filter] {split_name or 'dataset'}: {label} -> {len(filtered)}/{before} rows "
          f"({len(got)} signals)")
    return filtered
