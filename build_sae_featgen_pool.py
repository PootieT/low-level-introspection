#!/usr/bin/env python3
"""
Build the SAE-feature POOL for the feature-generalization experiment: an ordered list
of features with natural-language labels. A run with N features trains the first N;
every pool feature is evaluated (score_sae_feature_generalization.py), so the rest are
the held-out arm.

Two steps around the Neuronpedia annotation (annotate_sae_features_neuronpedia.py),
because that step needs internet and fetches one feature per request:

  candidates   stats parquet (select_sae_features.py) -> the features worth labelling.
               Keeps the density band (`selected`, unchanged) AND a floor on positives
               at the readout position: pos_density x --n-train-texts >= --min-positives.
               A feature in the band can still almost never fire at the last token of
               a 32-token text, and then it has nothing to balance and no 'on'
               magnitude to steer to. Draws a SEEDED RANDOM sample of those (not the
               top by density, which would skew the pool to common features), plus the
               --include features, and marks them `selected` so the annotator fetches
               exactly them. The original band flag is kept as `in_density_band`.

  build        annotated parquet -> pool json. Order: --include first (the steering-
               judged features, so the smallest runs line up with existing single-
               feature runs), then the candidates in seeded random order. Drops
               features without an explanation and NEAR-DUPLICATE labels: a feature is
               skipped if its label's cosine similarity to an already-kept label is
               >= --sim-threshold (sentence embeddings; default 0.7), since a held-out feature whose
               label paraphrases a trained one is not really unseen. Writes the full
               label similarity matrix so the analysis can relate held-out fidelity to
               the nearest trained label.

Run (see scripts/sae/build_featgen_pool_layer9.sh):
  python build_sae_featgen_pool.py candidates --stats S.parquet --out C.parquet ...
  python annotate_sae_features_neuronpedia.py --stats C.parquet --out A.parquet ...
  python build_sae_featgen_pool.py build --annotated A.parquet --out pool.json ...
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_feature(s: str, layer: int) -> int:
    m = re.fullmatch(rf"S{layer}\.(\d+)", s.strip())
    if not m:
        raise ValueError(f"{s!r} is not an S{layer}.<index> feature")
    return int(m.group(1))


def normalize_label(s) -> str:
    """One line, no double quotes (the prompt wraps the label in them)."""
    return " ".join(str(s).split()).replace('"', "'").strip()


def dedupe_key(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def embed_labels(labels: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """L2-normalized mean-pooled sentence embeddings (sentence-transformers models
    loaded through plain transformers, so no extra dependency)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            enc = tok(labels[i:i + batch_size], padding=True, truncation=True,
                      max_length=64, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


def greedy_dedupe(order: list[int], labels: dict[int, str], emb: dict[int, np.ndarray],
                  threshold: float, size: int, forced: set[int]):
    """Walk `order`, keeping a feature unless its label duplicates (same normalized
    string, or cosine >= threshold) one already kept. `forced` features are kept
    regardless, with a warning. Stops at `size`. Returns (kept, dropped records)."""
    kept, dropped, keys = [], [], {}
    for f in order:
        if len(kept) >= size:
            break
        k = dedupe_key(labels[f])
        sims = [(float(emb[f] @ emb[g]), g) for g in kept]
        best = max(sims) if sims else (-1.0, None)
        dup_of = keys.get(k)
        if dup_of is None and best[0] >= threshold:
            dup_of = best[1]
        if dup_of is not None:
            rec = {"feature": f, "label": labels[f], "dup_of": dup_of,
                   "dup_label": labels[dup_of], "sim": round(best[0], 4)}
            if f in forced:
                print(f"[pool] WARNING: forced feature {f} duplicates {dup_of}: {rec}")
            else:
                dropped.append(rec)
                continue
        kept.append(f)
        keys[k] = f
    return kept, dropped


def cmd_candidates(args):
    df = pd.read_parquet(args.stats)
    need = {"feature", "selected", "pos_density"}
    if not need <= set(df.columns):
        raise SystemExit(f"{args.stats} lacks {need - set(df.columns)}; is it a select_sae_features output?")
    if "pos_window" in df.columns:
        print(f"[candidates] pos_density measured at the last token of a "
              f"{int(df['pos_window'].iloc[0])}-token window")
    else:
        print("[candidates] WARNING: no pos_window column -- pos_density was measured at the "
              "last token of the FULL selection length, not of the training cache's window")
    include = [parse_feature(s, args.layer) for s in args.include.split(",")] if args.include else []
    df["in_density_band"] = df["selected"].astype(bool)
    df["proj_pos_train"] = df["pos_density"] * args.n_train_texts
    eligible = df[df["in_density_band"] & (df["proj_pos_train"] >= args.min_positives)]
    pool = sorted(set(eligible["feature"].astype(int)) - set(include))
    rng = np.random.RandomState(args.seed)
    pick = sorted(rng.choice(pool, size=min(args.n_candidates, len(pool)), replace=False).tolist())
    chosen = set(pick) | set(include)
    df["selected"] = df["feature"].astype(int).isin(chosen)
    for f in include:
        r = df[df["feature"] == f].iloc[0]
        if not (r["in_density_band"] and r["proj_pos_train"] >= args.min_positives):
            print(f"[candidates] WARNING: included feature {f} fails the filters "
                  f"(density={r['density']:.2e}, proj_pos_train={r['proj_pos_train']:.0f}); kept anyway")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"[candidates] band={int(df['in_density_band'].sum())}  "
          f"+positives floor ({args.min_positives} at {args.n_train_texts} texts)={len(eligible)}  "
          f"sampled={len(pick)} + included={len(include)} -> {args.out}")


def cmd_build(args):
    df = pd.read_parquet(args.annotated)
    if "explanation" not in df.columns:
        raise SystemExit(f"{args.annotated} has no explanation column; run the annotator first")
    include = [parse_feature(s, args.layer) for s in args.include.split(",")] if args.include else []
    cand = df[df["selected"].astype(bool)].copy()
    cand["label"] = cand["explanation"].map(lambda s: normalize_label(s) if isinstance(s, str) else "")
    no_label = cand[cand["label"] == ""]
    cand = cand[cand["label"] != ""]
    labels = dict(zip(cand["feature"].astype(int), cand["label"]))
    missing_inc = [f for f in include if f not in labels]
    if missing_inc:
        raise SystemExit(f"included feature(s) {missing_inc} have no explanation in {args.annotated}")
    rest = sorted(set(labels) - set(include))
    rest = [rest[i] for i in np.random.RandomState(args.seed).permutation(len(rest))]
    order = include + rest

    print(f"[pool] embedding {len(order)} labels with {args.embed_model}")
    E = embed_labels([labels[f] for f in order], args.embed_model)
    emb = {f: E[i] for i, f in enumerate(order)}
    kept, dropped = greedy_dedupe(order, labels, emb, args.sim_threshold, args.pool_size, set(include))
    if len(kept) < args.pool_size:
        raise SystemExit(f"only {len(kept)} features survive (need {args.pool_size}); "
                         f"label more candidates (raise --n-candidates in the candidates step)")

    K = np.stack([emb[f] for f in kept])
    sim = np.clip(K @ K.T, -1, 1)
    stats_cols = [c for c in ("density", "pos_density", "proj_pos_train", "active_mean",
                              "active_max", "np_frac_nonzero") if c in df.columns]
    by_f = df.set_index(df["feature"].astype(int))
    pool = {
        "layer": args.layer,
        "sae": {"release": args.sae_release, "site": "res", "width": args.width},
        "features": [f"S{args.layer}.{f}" for f in kept],
        "labels": {f"S{args.layer}.{f}": labels[f] for f in kept},
        "judged": [f"S{args.layer}.{f}" for f in include],
        "stats": {f"S{args.layer}.{f}": {c: float(by_f.loc[f, c]) for c in stats_cols} for f in kept},
        "label_similarity": np.round(sim, 4).tolist(),
        "selection": {"annotated": str(args.annotated), "seed": args.seed,
                      "sim_threshold": args.sim_threshold, "embed_model": args.embed_model,
                      "n_candidates_labelled": int(len(cand)),
                      "n_candidates_without_explanation": int(len(no_label)),
                      "n_dropped_duplicates": len(dropped)},
        "dropped_duplicates": [{k: (f"S{args.layer}.{v}" if k in ("feature", "dup_of") else v)
                                for k, v in d.items()} for d in dropped],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pool, indent=2))
    off = sim[~np.eye(len(kept), dtype=bool)]
    print(f"[pool] {len(kept)} features -> {args.out}  ({len(dropped)} near-duplicates dropped, "
          f"{len(no_label)} candidates without explanation); kept-label cosine "
          f"median={np.median(off):.2f} max={off.max():.2f}")
    with pd.option_context("display.max_colwidth", 80, "display.width", 140):
        print(pd.DataFrame({"rank": range(len(kept)), "feature": pool["features"],
                            "label": [labels[f] for f in kept]}).head(40).to_string(index=False))
    if dropped:
        print("\n[pool] sample of dropped near-duplicates:")
        print(pd.DataFrame(dropped).head(15).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--stats", type=Path, required=True)
    c.add_argument("--layer", type=int, required=True)
    c.add_argument("--include", default="", help="comma-separated S<layer>.<index>, always kept")
    c.add_argument("--n-train-texts", type=int, default=150_000)
    c.add_argument("--min-positives", type=int, default=50)
    c.add_argument("--n-candidates", type=int, default=600)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--annotated", type=Path, required=True)
    b.add_argument("--layer", type=int, required=True)
    b.add_argument("--include", default="")
    b.add_argument("--pool-size", type=int, default=200)
    b.add_argument("--sim-threshold", type=float, default=0.7,
                   help="all-MiniLM-L6-v2 cosine: same-concept paraphrases of Neuronpedia labels "
                        "score ~0.65-0.82 ('payments and financial transactions' vs 'financial "
                        "transactions and agreements' = 0.75), unrelated ones ~0.1-0.2")
    b.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    b.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    b.add_argument("--width", default="16k")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    (cmd_candidates if args.cmd == "candidates" else cmd_build)(args)


if __name__ == "__main__":
    main()
