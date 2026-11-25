#!/usr/bin/env python3
"""
Point a freshly generated run config at a reference run's EXISTING signal cache.

WHY. A config's dataset_name ends in a hash of its dataset identity
(src/introspector/dataset_utils.canonical_dataset_name). On 2026-09-06 the split
identity gained `min_seq_length` (always written, None by default), which moved the
hash of every MLP cache built before then without changing a single cached value. A
config generated today for a main-grid cell therefore names a cache that does not exist,
and cache_signals would rebuild 300K rows per cell -- a duplicate of data already on
disk. This copies the reference run's dataset_name instead, after checking that every
field that determines the cache CONTENT is equal.

CHECKED (refuses on any difference): model, seed, quantization, mlp_site, the
train/dev/test signal lists and token selectors, and each split's corpus, split,
text field, sample count, sequence length, offset and HF kwargs; min_seq_length must be
unset in both. NOT checked: anything training-only (LR, schedule, loss weights, ...),
which is the point -- those are what the new run changes.

FALLS BACK (keeps the generated name, so cache_signals builds it) when the reference
config or its cache is missing, printing why.

Usage
  python data_prep/pin_dataset_cache.py --config <new run config> --reference <old run config>
"""
import argparse
import json
import sys
from pathlib import Path

CONTENT_FIELDS = ("model_name", "seed", "load_in_4bit", "load_in_8bit", "signal_filter",
                  "train_signals", "dev_signals", "test_signals",
                  "train_token_selector", "dev_token_selector", "test_token_selector")
SPLIT_FIELDS = ("name", "split", "text_field", "max_samples", "max_seq_length", "offset",
                "hf_data_kwargs", "min_seq_length")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="Run config to rewrite in place.")
    p.add_argument("--reference", type=Path, required=True,
                   help="Run config whose cache to reuse.")
    return p.parse_args()


def content_mismatches(cfg: dict, ref: dict) -> list[str]:
    bad = [k for k in CONTENT_FIELDS if cfg.get(k) != ref.get(k)]
    if (cfg.get("mlp_site") or "act_fn") != (ref.get("mlp_site") or "act_fn"):
        bad.append("mlp_site")
    for split in ("train_dataset", "dev_dataset", "test_dataset"):
        a, b = cfg.get(split) or {}, ref.get(split) or {}
        bad += [f"{split}.{k}" for k in SPLIT_FIELDS if a.get(k) != b.get(k)]
    return bad


def main():
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    if not args.reference.exists():
        print(f"[pin] no reference config {args.reference}; keeping {cfg['dataset_name']}")
        return
    ref = json.loads(args.reference.read_text())
    bad = content_mismatches(cfg, ref)
    if bad:
        sys.exit(f"[pin] {args.config.name} and {args.reference.name} differ in cache content "
                 f"({', '.join(bad)}); refusing to share a cache")
    if cfg["dataset_name"] == ref["dataset_name"]:
        print(f"[pin] already on the reference cache {ref['dataset_name']}")
        return
    meta = Path(cfg.get("cache_dir") or "data") / ref["dataset_name"] / "metadata.json"
    if not meta.exists():
        print(f"[pin] reference cache {meta.parent} is not on disk; keeping "
              f"{cfg['dataset_name']} (cache_signals will build it)")
        return
    print(f"[pin] {cfg['dataset_name']} -> {ref['dataset_name']} (cache of {args.reference.name})")
    cfg["dataset_name"] = ref["dataset_name"]
    args.config.write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
