#!/usr/bin/env python3
"""
Aggregate the SAE feature-generalization sweep (scripts/sae/train_sae_feature_
generalization.sh) and plot it.

Reads, per run directory (each written by score_sae_feature_generalization.py, plus the
run_spec.json the launcher writes):
  feature_generalization_by_feature.csv     every pool feature x {endogenous, interventional}
  feature_generalization_distractor.csv     steer feature j, prompt names feature i

SERIES. A run belongs to one series, from its table group:
  label / index      300k rows TOTAL at every N (rows per feature fall as 1/N)
  label_d100k        label prompt at 100k rows PER FEATURE (N=3's budget) at every N.
                     Its N=3 point is the label series' N=3 run, which has exactly that
                     budget. Kept apart from the label series so the two never average.

FIGURES (PNG + PDF, each with the CSV it was drawn from)
  featgen_by_n             balanced accuracy vs N, trained vs held-out, label vs index
                           (300k-total series): endogenous (can rise from the label
                           describing the text alone), then interventional (the headline)
  featgen_data_matched     the same panels, label prompt: 300k total vs 100k per feature.
                           Is the seen-feature interventional collapse at N>=10 starvation?
  featgen_distractor       INTERVENTIONAL eval, P("active") with the prompt naming feature i
                           while i is steered on (higher is better), a different feature j is
                           steered on (lower is better), or i is clamped to 0 -- the "off"
                           half of the interventional test, a no-op on texts where i is
                           already inactive (lower is better). A readout that reads the
                           named feature is high on the first and near the third on the
                           second. 300k-total series.
  featgen_distractor_data_matched   the same, label prompt: 300k total vs 100k per feature
  featgen_unseen_vs_similarity
                           held-out interventional BA per feature against the cosine of its
                           label to the NEAREST TRAINED label (pool label_similarity). If the
                           label arm generalizes by meaning, near-paraphrases should go first.

Every number is MACRO over features (each feature weighs the same), with SEM over
features as the error bar.

  python analysis/plot_sae_feature_generalization.py \
      [--runs outputs/runs/sae_featgen] [--out outputs/plots/sae_featgen]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.sae_style import apply_rc, chance_line, save, style_axes   # noqa: E402
from src.introspector.sae_featgen import load_pool, nearest_label_similarity   # noqa: E402

# fixed per series (validated trio, all pairs: CVD dE >= 16.8); trained/held-out by line
# style, never color
SERIES_COLOR = {"label": "#1f77b4", "index": "#ff7f0e", "label_d100k": "#d62728"}
SERIES_NAME = {"label": "label prompt", "index": "index prompt",
               "label_d100k": "label prompt, 100k rows/feature"}
DATA_NAME = {"label": "300k rows total", "label_d100k": "100k rows per feature"}
FIXED = ("label", "index")
MATCHED = ("label", "label_d100k")
SPLIT_STYLE = {"seen": dict(ls="-", marker="o", mfc=None),
               "unseen": dict(ls="--", marker="o", mfc="white")}
SPLIT_NAME = {"seen": "trained", "unseen": "held out"}
STIM_STYLE = {"diagonal": dict(ls="-", marker="o", name="steer the named feature ↑"),
              "own": dict(ls=":", marker="s", name="steer a different feature ↓"),
              "off": dict(ls="--", marker="^", name="named feature clamped to 0 ↓")}
D100K_ROWS = 100_000


def _series(spec):
    group = spec.get("group", spec["arm"])
    if group == "data100k":
        return "label_d100k"
    return spec["arm"]


def load_runs(runs_dir: Path):
    per, dist = [], []
    for spec_path in sorted(runs_dir.glob("*/run_spec.json")):
        d = spec_path.parent
        f = d / "feature_generalization_by_feature.csv"
        if not f.exists():
            print(f"[skip] {d.name}: no feature_generalization_by_feature.csv yet")
            continue
        spec = json.loads(spec_path.read_text())
        n = int(spec["n_features"])
        total = int(spec.get("total_rows", 300_000))
        meta = {"run": d.name, "arm": spec["arm"], "series": _series(spec), "n_features": n,
                "total_rows": total, "rows_per_feature": total // n,
                "pool_json": spec.get("pool_json")}
        df = pd.read_csv(f).assign(**meta)
        pool_path = spec.get("pool_json")
        if pool_path and Path(pool_path).exists():
            near = nearest_label_similarity(load_pool(pool_path), spec["train_features"])
            df["nearest_trained_sim"] = df["signal"].map(near)
        per.append(df)
        g = d / "feature_generalization_distractor.csv"
        if g.exists():
            dist.append(pd.read_csv(g).assign(**meta))
    if not per:
        raise SystemExit(f"no finished runs under {runs_dir}")
    per = with_shared_point(pd.concat(per, ignore_index=True))
    dist = with_shared_point(pd.concat(dist, ignore_index=True)) if dist else None
    return per, dist


def with_shared_point(df):
    """The label series' runs that already have 100k rows per feature (N=3) are also the
    first point of the 100k-per-feature series; copy them in under that series."""
    if not (df["series"] == "label_d100k").any():
        return df
    shared = df[(df["series"] == "label") & (df["rows_per_feature"] == D100K_ROWS)]
    return pd.concat([df, shared.assign(series="label_d100k")], ignore_index=True)


def macro(df, by, col):
    g = df.dropna(subset=[col]).groupby(by)[col]
    out = g.agg(["mean", "count", "std"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def n_axis(ax, ns):
    """Log x with the actual N values as ticks (1, 3, 10, ...), not powers of ten."""
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(int(n)) for n in ns])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("features trained (N)")


def plot_by_n(per, out_dir, series, names, fname, title):
    per = per[per["series"].isin(series)]
    if per.empty:
        return
    per = per.assign(split=np.where(per["is_trained"], "seen", "unseen"))
    s = macro(per, ["eval_type", "series", "split", "n_features"], "balanced_accuracy")
    s.to_csv(out_dir / f"{fname}.csv", index=False)
    ns = sorted(s["n_features"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, layout="constrained")
    for ax, kind in zip(axes, ("endogenous", "interventional")):
        for ser in series:
            for split in ("seen", "unseen"):
                g = s[(s.eval_type == kind) & (s.series == ser) & (s.split == split)].sort_values("n_features")
                if g.empty:
                    continue
                st = SPLIT_STYLE[split]
                ax.errorbar(g["n_features"], g["mean"], yerr=g["sem"], color=SERIES_COLOR[ser],
                            ls=st["ls"], marker=st["marker"], ms=8, lw=2, capsize=3,
                            mfc=st["mfc"] or SERIES_COLOR[ser], mec=SERIES_COLOR[ser],
                            label=f"{names[ser]}, {SPLIT_NAME[split]}")
        chance_line(ax)
        n_axis(ax, ns)
        ax.set_title(kind)
        style_axes(ax)
    axes[0].set_ylabel("balanced accuracy (macro over features)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=True)
    fig.suptitle(title)
    save(fig, out_dir / fname)


def plot_distractor(dist, out_dir, series, names, fname, title):
    d = dist[dist["series"].isin(series)]
    if d.empty:
        return
    d = d.assign(ask_split=np.where(d["ask_trained"], "seen", "unseen"))
    off = (d.drop_duplicates(["run", "series", "ask"])
           .assign(stimulus="off", p_active=lambda x: x["ask_off_p_active"]))
    d = pd.concat([d[d["stimulus"].isin(["diagonal", "own"])], off], ignore_index=True)
    # one value per ask feature first, so features with more partners do not weigh more
    per_ask = (d.groupby(["series", "n_features", "ask_split", "stimulus", "ask"])["p_active"]
               .mean().reset_index())
    s = macro(per_ask, ["series", "n_features", "ask_split", "stimulus"], "p_active")
    s.to_csv(out_dir / f"{fname}.csv", index=False)
    ns = sorted(s["n_features"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True, layout="constrained")
    for ax, split in zip(axes, ("seen", "unseen")):
        for ser in series:
            for stim, st in STIM_STYLE.items():
                g = s[(s.series == ser) & (s.ask_split == split) & (s.stimulus == stim)].sort_values("n_features")
                if g.empty:
                    continue
                ax.errorbar(g["n_features"], g["mean"], yerr=g["sem"], color=SERIES_COLOR[ser],
                            ls=st["ls"], marker=st["marker"], ms=7, lw=2, capsize=3,
                            label=f"{names[ser]}: {st['name']}")
        n_axis(ax, ns)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Seen Feature" if split == "seen" else "Unseen Feature")
        style_axes(ax)
    axes[0].set_ylabel('P(model answers "active")')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=True, fontsize=11)
    fig.suptitle(title)
    save(fig, out_dir / fname)


def plot_similarity(per, out_dir):
    if "nearest_trained_sim" not in per:
        print("[skip] similarity plot: no pool file to read label similarities from")
        return
    u = per[(~per["is_trained"]) & (per["eval_type"] == "interventional")].dropna(
        subset=["balanced_accuracy", "nearest_trained_sim"])
    u = u.drop_duplicates(["run", "signal"])   # the shared N=3 run once, as the label series
    u.to_csv(out_dir / "featgen_unseen_vs_similarity.csv", index=False)
    ns = sorted(u["n_features"].unique())
    if not ns:
        return
    fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 4.6), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    for ax, n in zip(axes[0], ns):
        for ser in SERIES_COLOR:
            g = u[(u.n_features == n) & (u.series == ser)]
            if g.empty:
                continue
            ax.scatter(g["nearest_trained_sim"], g["balanced_accuracy"], s=22, alpha=0.6,
                       color=SERIES_COLOR[ser], edgecolor="white", linewidth=0.5,
                       label=SERIES_NAME[ser])
        chance_line(ax)
        ax.set_title(f"N = {n}")
        ax.set_xlabel("label cosine to nearest\ntrained label")
        style_axes(ax)
    axes[0][0].set_ylabel("held-out interventional BA")
    handles, labels = [], []
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=True, fontsize=10)
    fig.suptitle("Held-out features vs how close their label is to a trained one")
    save(fig, out_dir / "featgen_unseen_vs_similarity")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("outputs/runs/sae_featgen"))
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "outputs/plots/sae_featgen")
    args = p.parse_args()
    apply_rc()
    args.out.mkdir(parents=True, exist_ok=True)
    per, dist = load_runs(args.runs)
    per.to_csv(args.out / "featgen_all_features.csv", index=False)
    plot_by_n(per, args.out, FIXED, SERIES_NAME, "featgen_by_n",
              "SAE feature generalization: does the readout reach held-out features?")
    matched = (per["series"] == "label_d100k").any()
    if matched:
        plot_by_n(per, args.out, MATCHED, DATA_NAME, "featgen_data_matched",
                  "Label prompt: 300k rows total vs 100k rows per feature")
    if dist is not None:
        dist.to_csv(args.out / "featgen_distractor_all.csv", index=False)
        plot_distractor(dist, args.out, FIXED, SERIES_NAME, "featgen_distractor",
                        "Interventional distractor control: is the readout addressed to the named feature?")
        if matched:
            plot_distractor(dist, args.out, MATCHED, DATA_NAME, "featgen_distractor_data_matched",
                            "Distractor control, label prompt: 300k rows total vs 100k rows per feature")
    plot_similarity(per, args.out)


if __name__ == "__main__":
    main()
