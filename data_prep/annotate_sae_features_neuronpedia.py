#!/usr/bin/env python3
"""
Annotate an SAE feature-stats parquet with Neuronpedia metadata (explanation +
top-activating tokens) so the steering judge has a CONCEPT to check each feature
against. The feature's `explanation` is the ground-truth concept used downstream by
data_prep/generate_sae_steering_samples.py.

NOTE on interp scores: Neuronpedia exposes descriptions but NOT auto-interp scores
for the Gemma Scope res-16k features (`scores: []`). So we do NOT select features by
interp score -- selection is decided by the steering judge. `interp_score` is still
recorded here when a release happens to provide one; `--min-interp-score` is a no-op
for releases without scores (features with no score pass).

This is a standalone, NETWORK-only step (no GPU / no model). It sits between
data_prep/select_sae_features.py (Step 1) and the steering + training steps. It reads the stats
parquet, hits the public Neuronpedia feature API for the density-selected features,
and writes an enriched parquet (a superset of the input columns).

Run (login node with internet; the cluster's compute nodes may be offline):

  python data_prep/annotate_sae_features_neuronpedia.py \
      --stats data/sae_stats/gemma2-2b_res9_16k.parquet \
      --model gemma-2-2b --layer 9 --width 16k --site res \
      --out data/sae_stats/gemma2-2b_res9_16k.annotated.parquet
"""
from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path

import pandas as pd
import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from src.introspector.signals.neuronpedia import (
    fetch_feature, source_id_for, FeatureInfo,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", type=Path, required=True,
                   help="Feature-stats parquet from data_prep/select_sae_features.py.")
    p.add_argument("--model", default="gemma-2-2b",
                   help="Neuronpedia model id (NOTE: 'gemma-2-2b', no HF org prefix).")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--width", default="16k")
    p.add_argument("--site", default="res", choices=["res", "mlp", "attn"])
    p.add_argument("--only-selected", action="store_true", default=True,
                   help="Only fetch features with selected==True (default).")
    p.add_argument("--all-features", dest="only_selected", action="store_false",
                   help="Fetch every row instead of just selected ones (many requests).")
    p.add_argument("--max-fetch", type=int, default=None,
                   help="Only fetch the top-N selected features by projected positives "
                        "(proj_pos_at_fixed_pos). Bounds network work -- the density band can "
                        "select thousands of features, but only the top candidates get steered.")
    p.add_argument("--min-interp-score", type=float, default=None,
                   help="Shortlist floor: keep features whose interp_score >= this OR is "
                        "unavailable (NaN). Wires SAEFeatureSampler.min_interp_score.")
    p.add_argument("--cache-dir", type=Path, default=Path("data/neuronpedia_cache"))
    p.add_argument("--no-cache", action="store_true",
                   help="Ignore the on-disk cache and always re-fetch.")
    p.add_argument("--top-k", type=int, default=40, help="How many shortlisted rows to print.")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


_NEW_COLS = [
    "explanation", "explanation_model", "interp_score", "n_explanations",
    "has_explanation", "np_pos_str", "np_neg_str", "np_frac_nonzero",
    "np_max_act", "np_has_vector", "np_default_steer_strength",
]


def _row_from_info(info: FeatureInfo | None) -> dict:
    if info is None:
        return {c: (None if c not in ("has_explanation", "n_explanations") else
                    (False if c == "has_explanation" else 0)) for c in _NEW_COLS} | {
            "np_pos_str": json.dumps([]), "np_neg_str": json.dumps([])}
    return {
        "explanation": info.explanation,
        "explanation_model": info.explanation_model,
        "interp_score": info.interp_score,
        "n_explanations": info.n_explanations,
        "has_explanation": info.has_explanation,
        "np_pos_str": json.dumps(info.pos_str),
        "np_neg_str": json.dumps(info.neg_str),
        "np_frac_nonzero": info.frac_nonzero,
        "np_max_act": info.max_act,
        "np_has_vector": info.has_vector,
        "np_default_steer_strength": info.default_steer_strength,
    }


def main():
    args = parse_args()
    df = pd.read_parquet(args.stats)
    if "feature" not in df.columns:
        raise SystemExit(f"{args.stats} has no 'feature' column; is it a select_sae_features output?")

    source_id = source_id_for(args.layer, site=args.site, width=args.width)
    print(f"[annotate] model={args.model} source={source_id} rows={len(df)}")

    targets = df[df["selected"]] if (args.only_selected and "selected" in df.columns) else df
    if args.max_fetch and "proj_pos_at_fixed_pos" in targets.columns:
        targets = targets.sort_values("proj_pos_at_fixed_pos", ascending=False).head(args.max_fetch)
    print(f"[annotate] fetching {len(targets)} features "
          f"({'selected only' if args.only_selected else 'all'}"
          f"{f', top {args.max_fetch} by proj_pos' if args.max_fetch else ''})")

    rows = {}
    for feat in tqdm.tqdm(targets["feature"].tolist()):
        info = fetch_feature(
            args.model, source_id, int(feat),
            cache_dir=args.cache_dir, use_cache=not args.no_cache,
        )
        rows[int(feat)] = _row_from_info(info)

    for c in _NEW_COLS:
        df[c] = df["feature"].map(lambda f: rows.get(int(f), {}).get(c))
    df["has_explanation"] = df["has_explanation"].fillna(False).astype(bool)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    n_expl = int(df["has_explanation"].sum())
    n_scored = int(df["interp_score"].notna().sum())
    print(f"[annotate] wrote {len(df)} rows -> {args.out}  "
          f"(with_explanation={n_expl}, with_interp_score={n_scored})")

    chk = df[df["np_frac_nonzero"].notna() & df.get("density", pd.Series(dtype=float)).notna()]
    if len(chk):
        rel = (chk["np_frac_nonzero"] - chk["density"]).abs() / chk["np_frac_nonzero"].clip(lower=1e-9)
        bad = chk[rel > 0.5]
        if len(bad):
            print(f"[annotate] WARNING: {len(bad)} features differ >50% between our density and "
                  f"Neuronpedia frac_nonzero -- possible hook/tokenization mismatch. Sample:")
            print(bad[["feature", "density", "np_frac_nonzero"]].head().to_string(index=False))

    short = df[df["has_explanation"]].copy()
    if args.min_interp_score is not None:
        keep = short["interp_score"].isna() | (short["interp_score"] >= args.min_interp_score)
        short = short[keep]
    sort_cols = [c for c in ("interp_score", "proj_pos_at_fixed_pos") if c in short.columns]
    short = short.sort_values(sort_cols, ascending=False, na_position="last").head(args.top_k)

    print(f"\n=== {len(short)} EXPLAINED CANDIDATES"
          f"{f' (interp_score>={args.min_interp_score} or NaN)' if args.min_interp_score is not None else ''} ===")
    print("  next: generate steering samples for these with data_prep/generate_sae_steering_samples.py,")
    print("  then judge them blind (selection is decided by the steering judge).")
    show = [c for c in ("feature", "density", "proj_pos_at_fixed_pos", "interp_score",
                        "explanation") if c in short.columns]
    with pd.option_context("display.max_colwidth", 60, "display.float_format", lambda v: f"{v:.4g}"):
        print(short[show].to_string(index=False))


if __name__ == "__main__":
    main()
