#!/usr/bin/env python3
"""
Fetch the Neuronpedia explanation of each named SAE feature and write a labels file
{"labels": {"S<layer>.<index>": label}} -- the format prompt_sae_label reads (via
--feature-labels-file, or evals/score_sae_prompt_baseline.py --labels-file).

Explanations are chosen and normalized exactly as for the feature-generalization pool
(neuronpedia.parse_feature, build_sae_featgen_pool.normalize_label), so a feature has
the same label in every experiment. Network only; responses are cached on disk.

  python data_prep/fetch_sae_feature_labels.py --features-file scripts/sae/sweep_features.txt \
      --features S9.1568,S9.4854,S9.4120,S9.978,S9.12408 \
      --out scripts/sae/sweep_feature_labels.json
"""
from __future__ import annotations

import sys
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from data_prep.build_sae_featgen_pool import normalize_label
from src.introspector.signals.neuronpedia import fetch_feature, source_id_for


def read_features_file(path: Path) -> list[str]:
    """Every S<layer>.<index> token on non-comment lines of sweep_features.txt."""
    out = []
    for line in path.read_text().splitlines():
        if line.strip().startswith("#"):
            continue
        out += re.findall(r"S\d+\.\d+", line)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="", help="comma-separated S<layer>.<index>")
    p.add_argument("--features-file", type=Path, default=None)
    p.add_argument("--model", default="gemma-2-2b")
    p.add_argument("--site", default="res")
    p.add_argument("--width", default="16k")
    p.add_argument("--cache-dir", type=Path, default=Path("data/neuronpedia_cache"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    feats = [f for f in args.features.split(",") if f.strip()]
    if args.features_file:
        feats += read_features_file(args.features_file)
    feats = list(dict.fromkeys(f.strip() for f in feats))
    labels, missing = {}, []
    for f in feats:
        layer, index = map(int, f[1:].split("."))
        info = fetch_feature(args.model, source_id_for(layer, args.site, args.width), index,
                             cache_dir=args.cache_dir)
        if info is None or not info.has_explanation:
            missing.append(f)
            continue
        labels[f] = normalize_label(info.explanation)
        print(f"{f:>10}  {labels[f]}")
    if missing:
        raise SystemExit(f"no explanation for {missing}; nothing written")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "labels": labels, "source": f"neuronpedia {args.model} {args.site}-{args.width}",
        "fetched_at": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"wrote {len(labels)} labels -> {args.out}")


if __name__ == "__main__":
    main()
