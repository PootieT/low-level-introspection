#!/usr/bin/env python3
"""
Plot SAE detection metrics across token positions, one line per feature.

Point it at the PARENT dir of a position sweep, e.g.
    outputs/runs/single_sae_detection_sweep_pos/
whose children look like
    sae_S9.12408_POS-10_lr5e-5_train300K/
each containing sae_detection_metrics.json ({"dev":..., "test":..., [test_interventional]}).

Parses feature (S<layer>.<idx>) and position (POS<n>) from the dir name, tabulates
the metrics, and draws a seaborn facet (one panel per metric) with position on x and
one line per feature. Interventional columns are included automatically if present.
"""
from __future__ import annotations

import os
import re
import pdb
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# metrics whose natural range is [0,1] (get a 0..1 y-axis)
RANGE01 = {"ap", "auroc", "f1_at_0", "acc_at_0", "paired_directional_acc"}
# metrics where 0.5 is the chance baseline (get a dashed reference line)
REF05 = {"auroc", "paired_directional_acc"}


def _style_ax(ax, metric):
    if metric in REF05:
        ax.axhline(0.5, color="grey", ls="--", lw=1)
    if metric in RANGE01:
        ax.set_ylim(0, 1)

# dir name -> feature + position (lr / train captured for optional faceting)
RUN_RE = re.compile(r"sae_sample(?P<sample>\d+)_S(?P<layer>\d+)\.(?P<index>\d+)_POS(?P<pos>-?\d+)"
                    r"(?:_lr(?P<lr>[0-9.e-]+))?(?:_train(?P<train>\w+))?")

# metrics to pull from each split's dict (skipped silently if absent)
METRICS = [
    "ap", "auroc", "f1_at_0", "acc_at_0", "base_rate_positive", # non intervention metrics
    "paired_directional_acc", "on_value", "mean_gap_on", "mean_gap_off" # interventional metrics
]


def load_sweep(parent_dir: str) -> pd.DataFrame:
    parent = Path(parent_dir)
    rows, skipped = [], 0
    for d in tqdm(sorted(parent.iterdir())):
        if not d.is_dir():
            continue
        m = RUN_RE.search(d.name)
        mpath = d / "sae_detection_metrics.json"
        if m is None or not mpath.exists():
            skipped += 1
            continue
        metrics = json.loads(mpath.read_text())
        base = {
            "feature": f"S{m['layer']}.{m['index']}",
            "layer": int(m["layer"]), "index": int(m["index"]),
            # sample index == density stratum from the sweep picker (0=rarest -> 9=most frequent)
            "sample": int(m["sample"]),
            "position": int(m["pos"]),
            "lr": m.group("lr"), "train": m.group("train"),
        }
        for split, split_metrics in metrics.items():          # dev / test / test_interventional
            if not isinstance(split_metrics, dict):
                continue
            for metric in METRICS:
                if metric in split_metrics and split_metrics[metric] is not None:
                    rows.append({**base, "split": split, "metric": metric,
                                 "value": float(split_metrics[metric])})
    if skipped:
        print(f"[plot] skipped {skipped} dirs (no match / no metrics json -- unfinished runs?)")
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No metrics parsed under {parent_dir}")
    return df


def plot_overview(sub, out_dir, split, by="feature", by_label=None,
                  palette_name=None, errorbar=None):
    """Combined facet (one panel per metric) with a swappable hue/aggregation:
      by='feature' -> one line per feature (no aggregation);
      by='layer'   -> one line per layer, mean±SE over its features (errorbar='se');
      by='sample'  -> one line per density bin, mean±SE over layers."""
    by_label = by_label or by
    data = sub.copy()
    palette = None
    if by in ("layer", "sample"):
        levels = sorted(data[by].unique())
        data[by] = pd.Categorical(data[by], categories=levels, ordered=True)
        palette = dict(zip(levels, sns.color_palette(palette_name or "viridis", len(levels))))
    g = sns.relplot(
        data=data, x="position", y="value", hue=by, palette=palette,
        col="metric", col_wrap=min(3, data["metric"].nunique()),
        kind="line", marker="o", errorbar=errorbar,
        facet_kws={"sharey": False}, height=3.5, aspect=1.3,
        legend="brief" if by != "feature" else "auto",
    )
    g.set_titles("{col_name}")
    tail = " (mean±SE)" if errorbar else f" — {data['feature'].nunique()} features"
    g.figure.suptitle(f"SAE detection ({split}) vs token position — by {by_label}{tail}", y=1.02)
    for ax in g.axes.flat:
        _style_ax(ax, ax.get_title())
    g.tight_layout()
    out_path = out_dir / f"{split}__overview_{by}.png"
    g.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.figure)
    print(f"[plot] wrote {out_path}")


def plot_per_metric_by_feature(sub, out_dir, split):
    """One file per metric: every feature is its own line over position."""
    for metric in sorted(sub["metric"].unique()):
        m = sub[sub["metric"] == metric]
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.lineplot(data=m, x="position", y="value", hue="feature",
                     marker="o", ax=ax, legend="brief")
        _style_ax(ax, metric)
        ax.set_title(f"SAE detection ({split}) — {metric} — {m['feature'].nunique()} features")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, ncol=1)
        fig.tight_layout()
        out_path = out_dir / f"{split}__by_feature__{metric}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out_path}")


def plot_per_metric_agg(sub, out_dir, split, by="layer", by_label=None, palette_name="viridis"):
    """One file per metric: one line per level of `by` (e.g. layer, or sample =
    density bin), aggregating over all other features at each (level, position)
    with a shaded ±1 standard-error band (seaborn errorbar='se'). This is the same
    idea as the by-layer view but with a swappable aggregation column."""
    by_label = by_label or by
    agg = sub.copy()
    levels = sorted(agg[by].unique())
    agg[by] = pd.Categorical(agg[by], categories=levels, ordered=True)
    palette = dict(zip(levels, sns.color_palette(palette_name, len(levels))))
    for metric in sorted(agg["metric"].unique()):
        m = agg[agg["metric"] == metric]
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.lineplot(
            data=m, x="position", y="value", hue=by, palette=palette,
            errorbar="se", marker="o", ax=ax,
        )
        _style_ax(ax, metric)
        n_per = m.groupby(by, observed=True)["feature"].nunique().max()
        ax.set_title(f"SAE detection ({split}) — {metric} — mean±SE by {by_label} "
                     f"(≤{n_per} feats/level)")
        ax.legend(title=by_label, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        fig.tight_layout()
        out_path = out_dir / f"{split}__agg_{by}__{metric}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out_path}")


SAMPLE_LABEL = "density bin (0=rare→9=frequent)"


def plot_sweep(parent_dir: str, split: str = "test", df=None, individual_metrics: bool = False):
    df = load_sweep(parent_dir) if df is None else df
    sub = df[df["split"] == split].copy()
    if sub.empty:
        raise SystemExit(f"No rows for split={split!r}; available: {sorted(df['split'].unique())}")
    sub = sub.sort_values(["layer", "feature", "metric", "position"])

    out_dir = Path(parent_dir).parents[1] / "plots" / Path(parent_dir).name
    os.makedirs(out_dir, exist_ok=True)

    # faceted overviews: raw (per feature) + the two aggregations (mean±SE)
    plot_overview(sub, out_dir, split, by="feature")
    plot_overview(sub, out_dir, split, by="layer", by_label="layer", errorbar="se")
    plot_overview(sub, out_dir, split, by="sample", by_label=SAMPLE_LABEL,
                  palette_name="rocket", errorbar="se")

    # per-metric individual files (one PNG per metric) -- off by default
    if individual_metrics:
        plot_per_metric_by_feature(sub, out_dir, split)
        plot_per_metric_agg(sub, out_dir, split, by="layer", by_label="layer")
        plot_per_metric_agg(sub, out_dir, split, by="sample",
                            by_label=SAMPLE_LABEL, palette_name="rocket")


def plot_all_splits(parent_dir: str, individual_metrics: bool = False):
    """Plot every split present except dev (test, test_interventional, ...)."""
    df = load_sweep(parent_dir)
    for split in sorted(df["split"].unique()):
        if split == "dev":
            continue
        print(f"\n===== plotting split={split} =====")
        plot_sweep(parent_dir, split=split, df=df, individual_metrics=individual_metrics)


if __name__ == "__main__":
    # plot_sweep("/projectnb/llamagrp/peter/low-level-introspection/outputs/runs/single_sae_detection_sweep_pos")
    plot_all_splits("/projectnb/llamagrp/peter/low-level-introspection/outputs/runs/single_sae_detection_sweep_layer_pos")
