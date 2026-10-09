#!/usr/bin/env python3
"""
Aggregate the SAE feature-generalization sweep (scripts/sae/train_sae_feature_
generalization.sh) and plot it.

Reads, per run directory (each written by score_sae_feature_generalization.py, plus the
run_spec.json the launcher writes):
  feature_generalization_by_feature.csv     every pool feature x {endogenous, interventional}
  feature_generalization_distractor.csv     steer feature j, prompt names feature i

FIGURES (PNG + PDF, each with the CSV it was drawn from)
  featgen_by_n             balanced accuracy vs N trained features, seen vs held-out,
                           label vs index arm; interventional (the headline) and
                           endogenous (can rise from the label describing the text alone)
  featgen_distractor       P("active") when the prompt names i and the steered feature is
                           i (diagonal), another feature j (off-diagonal), or none ("off").
                           Held-out interventional accuracy only means "reads the named
                           feature" if the off-diagonal stays down at the "off" rate.
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
from analysis.sae_style import apply_rc, chance_line, save, style_axes, BASELINE, INK2   # noqa: E402
from src.introspector.sae_featgen import load_pool, nearest_label_similarity           # noqa: E402

# fixed per arm (validated pair: CVD dE 24.6); seen/held-out by line style, never color
ARM_COLOR = {"label": "#1f77b4", "index": "#ff7f0e"}
ARM_NAME = {"label": "label prompt", "index": "index prompt"}
SPLIT_STYLE = {"seen": dict(ls="-", marker="o", mfc=None),
               "unseen": dict(ls="--", marker="o", mfc="white")}
SPLIT_NAME = {"seen": "trained", "unseen": "held out"}


def load_runs(runs_dir: Path):
    per, dist = [], []
    for spec_path in sorted(runs_dir.glob("*/run_spec.json")):
        d = spec_path.parent
        f = d / "feature_generalization_by_feature.csv"
        if not f.exists():
            print(f"[skip] {d.name}: no feature_generalization_by_feature.csv yet")
            continue
        spec = json.loads(spec_path.read_text())
        meta = {"run": d.name, "arm": spec["arm"], "n_features": int(spec["n_features"]),
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
    return pd.concat(per, ignore_index=True), (pd.concat(dist, ignore_index=True) if dist else None)


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


def end_labels(ax, items, min_gap=0.045):
    """Direct labels at line ends, nudged apart vertically so crossing lines do not
    print on top of each other. items: (x, y, text) in data coordinates."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    items = sorted(items, key=lambda t: t[1])
    ys = [(y - lo) / (hi - lo) for _, y, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    for (x, _, text), yf in zip(items, ys):
        ax.annotate(text, (x, lo + yf * (hi - lo)), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK2, annotation_clip=False)


def plot_by_n(per, out_dir):
    per = per.assign(split=np.where(per["is_trained"], "seen", "unseen"))
    s = macro(per, ["eval_type", "arm", "split", "n_features"], "balanced_accuracy")
    s.to_csv(out_dir / "featgen_by_n.csv", index=False)
    ns = sorted(s["n_features"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, layout="constrained")
    for ax, kind in zip(axes, ("interventional", "endogenous")):
        ends = []
        for arm in ("label", "index"):
            for split in ("seen", "unseen"):
                g = s[(s.eval_type == kind) & (s.arm == arm) & (s.split == split)].sort_values("n_features")
                if g.empty:
                    continue
                st = SPLIT_STYLE[split]
                ax.errorbar(g["n_features"], g["mean"], yerr=g["sem"], color=ARM_COLOR[arm],
                            ls=st["ls"], marker=st["marker"], ms=8, lw=2, capsize=3,
                            mfc=st["mfc"] or ARM_COLOR[arm], mec=ARM_COLOR[arm],
                            label=f"{ARM_NAME[arm]}, {SPLIT_NAME[split]}")
                last = g.iloc[-1]
                ends.append((last["n_features"], last["mean"], f"{arm}, {SPLIT_NAME[split]}"))
        chance_line(ax)
        n_axis(ax, ns)
        ax.set_title(f"{kind}")
        style_axes(ax)
        end_labels(ax, ends)
    axes[0].set_ylabel("balanced accuracy (macro over features)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=True)
    fig.suptitle("SAE feature generalization: does the readout reach held-out features?")
    save(fig, out_dir / "featgen_by_n")


def plot_distractor(dist, out_dir):
    d = dist.assign(ask_split=np.where(dist["ask_trained"], "seen", "unseen"))
    off = (d.drop_duplicates(["run", "ask"])
           .assign(stimulus="off", p_active=lambda x: x["ask_off_p_active"]))
    d = pd.concat([d[d["stimulus"].isin(["diagonal", "own"])], off], ignore_index=True)
    # one value per ask feature first, so features with more partners do not weigh more
    per_ask = d.groupby(["arm", "n_features", "ask_split", "stimulus", "ask"])["p_active"].mean().reset_index()
    s = macro(per_ask, ["arm", "n_features", "ask_split", "stimulus"], "p_active")
    s.to_csv(out_dir / "featgen_distractor.csv", index=False)
    stim_style = {"diagonal": dict(ls="-", marker="o", name="steer the named feature"),
                  "own": dict(ls=":", marker="s", name="steer a different feature"),
                  "off": dict(ls="--", marker="^", name="named feature steered to 0")}
    ns = sorted(s["n_features"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True, layout="constrained")
    for ax, split in zip(axes, ("seen", "unseen")):
        for arm in ("label", "index"):
            for stim, st in stim_style.items():
                g = s[(s.arm == arm) & (s.ask_split == split) & (s.stimulus == stim)].sort_values("n_features")
                if g.empty:
                    continue
                ax.errorbar(g["n_features"], g["mean"], yerr=g["sem"], color=ARM_COLOR[arm],
                            ls=st["ls"], marker=st["marker"], ms=7, lw=2, capsize=3,
                            label=f"{ARM_NAME[arm]}: {st['name']}")
        n_axis(ax, ns)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"prompt names a {SPLIT_NAME[split]} feature")
        style_axes(ax)
    axes[0].set_ylabel('P(model answers "active")')
    handles, labels = axes[0].get_legend_handles_labels() or axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=True, fontsize=11)
    fig.suptitle("Distractor control: is the readout addressed to the named feature?")
    save(fig, out_dir / "featgen_distractor")


def plot_similarity(per, out_dir):
    if "nearest_trained_sim" not in per:
        print("[skip] similarity plot: no pool file to read label similarities from")
        return
    u = per[(~per["is_trained"]) & (per["eval_type"] == "interventional")].dropna(
        subset=["balanced_accuracy", "nearest_trained_sim"])
    u.to_csv(out_dir / "featgen_unseen_vs_similarity.csv", index=False)
    ns = sorted(u["n_features"].unique())
    if not ns:
        return
    fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 4.6), sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    for ax, n in zip(axes[0], ns):
        for arm in ("label", "index"):
            g = u[(u.n_features == n) & (u.arm == arm)]
            if g.empty:
                continue
            ax.scatter(g["nearest_trained_sim"], g["balanced_accuracy"], s=22, alpha=0.6,
                       color=ARM_COLOR[arm], edgecolor="white", linewidth=0.5, label=ARM_NAME[arm])
        chance_line(ax)
        ax.set_title(f"N = {n}")
        ax.set_xlabel("label cosine to nearest\ntrained label")
        style_axes(ax)
    axes[0][0].set_ylabel("held-out interventional BA")
    axes[0][0].legend(loc="lower left", frameon=True, fontsize=10)
    fig.suptitle("Held-out features vs how close their label is to a trained one")
    save(fig, out_dir / "featgen_unseen_vs_similarity")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("outputs/runs/sae_featgen"))
    p.add_argument("--out", type=Path, default=Path("outputs/plots/sae_featgen"))
    args = p.parse_args()
    apply_rc()
    args.out.mkdir(parents=True, exist_ok=True)
    per, dist = load_runs(args.runs)
    per.to_csv(args.out / "featgen_all_features.csv", index=False)
    plot_by_n(per, args.out)
    if dist is not None:
        dist.to_csv(args.out / "featgen_distractor_all.csv", index=False)
        plot_distractor(dist, args.out)
    plot_similarity(per, args.out)


if __name__ == "__main__":
    main()
