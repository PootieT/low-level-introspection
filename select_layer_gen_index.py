#!/usr/bin/env python3
"""
Pick the neuron index for the layer-generalization sweep.

The sweep holds ONE index fixed across all 15 layers so the only coordinate that varies
is the layer. But index i at layer 0 and index i at layer 12 are unrelated neurons, and
an arbitrary index will be near-silent at some layer -- which reads as an undefined
correlation there, indistinguishable from a generalization failure unless you already
know to look. So the index is chosen by scanning, not picked.

Reads a scan pool (a few candidate indices x every layer, test split) and ranks
candidates by their MINIMUM activation std across layers: the index whose weakest layer
is strongest. That is the right criterion because the sweep's result at every layer
rests on this one index, and one dead layer poisons the per-layer picture.

Usage
-----
    python select_layer_gen_index.py --config <scan pool config> [--min-std 1e-3]
Prints the chosen index on the last line (for capture by a shell script) and the full
table on stderr.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk

from src.introspector.config import ExperimentConfig


def rank_candidates(test_raw, min_std: float) -> pd.DataFrame:
    df = pd.DataFrame({
        "layer": test_raw["signal_layer"], "index": test_raw["signal_index"],
        "activation": np.asarray(test_raw["activation"], dtype=float),
    })
    per = (df.groupby(["index", "layer"])["activation"]
             .agg(std="std", mean="mean",
                  frac_zero=lambda v: float(np.mean(np.abs(v) < 1e-6)))
             .reset_index())
    # per index: the weakest layer decides
    rank = (per.groupby("index")
              .agg(min_std=("std", "min"), median_std=("std", "median"),
                   n_layers=("layer", "nunique"),
                   n_dead=("std", lambda s: int((s <= min_std).sum())),
                   weakest_layer=("std", lambda s: int(per.loc[s.idxmin(), "layer"])))
              .reset_index()
              .sort_values(["n_dead", "min_std"], ascending=[True, False])
              .reset_index(drop=True))
    return rank, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="the SCAN pool config")
    ap.add_argument("--min-std", type=float, default=1e-3,
                    help="a (layer, index) at or below this is counted as dead")
    args = ap.parse_args()

    cfg = ExperimentConfig.model_validate_json(args.config.read_text())
    path = cfg.dataset_cache_dir / "test"
    if not path.exists():
        sys.exit(f"scan pool not cached: {path}\n    python cache_signals.py --config {args.config}")
    rank, per = rank_candidates(load_from_disk(str(path)), args.min_std)

    print("[layergen] candidate indices, ranked by their weakest layer:", file=sys.stderr)
    print(rank.to_string(index=False), file=sys.stderr)
    best = rank.iloc[0]
    if best["n_dead"] > 0:
        print(f"[layergen] WARNING: every candidate is dead at >= {int(best['n_dead'])} "
              f"layer(s); the chosen index will have undefined correlation there. "
              f"Scan more candidates.", file=sys.stderr)
    print(f"[layergen] chosen: index {int(best['index'])}  (min std {best['min_std']:.4f} "
          f"at layer {int(best['weakest_layer'])})", file=sys.stderr)
    # last line of stdout is the machine-readable answer
    print(int(best["index"]))


if __name__ == "__main__":
    main()
