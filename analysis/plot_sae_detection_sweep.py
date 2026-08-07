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

import re
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# dir name -> feature + position (lr / train captured for optional faceting)
RUN_RE = re.compile(r"sae_S(?P<layer>\d+)\.(?P<index>\d+)_POS(?P<pos>-?\d+)"
                    r"(?:_lr(?P<lr>[0-9.e-]+))?(?:_train(?P<train>\w+))?")

# metrics to pull from each split's dict (skipped silently if absent)
METRICS = ["ap", "auroc", "f1_at_0", "acc_at_0", "base_rate_positive"]


def load_sweep(parent_dir: str) -> pd.DataFrame:
    parent = Path(parent_dir)
    rows, skipped = [], 0
    for d in sorted(parent.iterdir()):
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


def plot_sweep(parent_dir: str, split: str = "test", out_name: str = "sweep_pos_metrics.png"):
    df = load_sweep(parent_dir)
    sub = df[df["split"] == split].copy()
    if sub.empty:
        raise SystemExit(f"No rows for split={split!r}; available: {sorted(df['split'].unique())}")
    sub = sub.sort_values(["feature", "metric", "position"])

    # one panel per metric; one line per feature over position. If the sweep spans
    # multiple lr/train configs, distinguish them with line style.
    style = None
    if sub[["lr", "train"]].drop_duplicates().shape[0] > 1:
        sub["config"] = "lr" + sub["lr"].astype(str) + "_" + sub["train"].astype(str)
        style = "config"

    n_feat = sub["feature"].nunique()
    g = sns.relplot(
        data=sub, x="position", y="value", hue="feature", style=style,
        col="metric", col_wrap=min(3, sub["metric"].nunique()),
        kind="line", marker="o", facet_kws={"sharey": False},
        height=3.5, aspect=1.3,
    )
    g.set_titles("{col_name}")
    g.figure.suptitle(f"SAE detection ({split}) vs token position — {n_feat} features", y=1.02)
    # reference lines where meaningful
    for ax in g.axes.flat:
        title = ax.get_title()
        if title == "auroc":
            ax.axhline(0.5, color="grey", ls="--", lw=1)
        if title in ("ap", "auroc"):
            ax.set_ylim(0, 1)
    g.tight_layout()

    out_path = Path(parent_dir) / out_name
    g.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")
    print(sub.pivot_table(index=["feature", "position"], columns="metric",
                          values="value").to_string())
    plt.close(g.figure)


if __name__ == "__main__":
    plot_sweep(
        "/projectnb/llamagrp/peter/low-level-introspection/outputs/runs/single_sae_detection_sweep_pos"
    )
