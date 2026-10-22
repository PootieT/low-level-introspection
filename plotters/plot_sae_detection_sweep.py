#!/usr/bin/env python3
"""
Plot SAE-feature detection metrics across token positions.

Point it at the PARENT dir of a sweep, e.g.
    outputs/runs/sae_detect_sweep_layer_token/     (endogenous grid, gate on + off)
    outputs/runs/sae_intervention_pilot/           (interventional training)

Handles both run-name conventions:
    saedet_sample0_S9.12408_POS-1_gateon_300K      (sweep)
    saeInterv_S9.12408_POS-1_scale1_300K           (interventional pilot)

and both metrics files:
    sae_detection_metrics.json          {dev, test, test_interventional}
    sae_intervention_train_metrics.json {dev_endogenous, test_endogenous, test_interventional}

Split names are NORMALIZED (`test_endogenous` -> `test`) so an endogenously-trained
run and an interventionally-trained one can be compared on the same axes; which
trainer produced a row is kept in `train_regime`.

Views per split (each written to plots/<sweep>/):
  <split>__overview_<by>.png     all metrics in one figure, hue = <by>
  and, with individual_metrics=True, one PNG per metric for each view.

Aggregated views (`layer`, `sample`, `gate`, `train_regime`) shade +-1 standard error
across the runs collapsed into each line.
"""
from __future__ import annotations

import os
import re
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Tolerant of both naming conventions: sample/gate/scale are optional groups.
#   saedet_sample0_S9.12408_POS-1_gateon_300K
#   saeInterv_S9.12408_POS-13_scale1_300K
RUN_RE = re.compile(
    r"(?:sample(?P<sample>\d+)_)?"
    r"S(?P<layer>\d+)\.(?P<index>\d+)"
    r"_POS(?P<pos>-?\d+)"
    r"(?:_gate(?P<gate>on|off))?"
    r"(?:_scale(?P<scale>[0-9.eE+-]+))?"
)

# metrics file -> which trainer wrote it
METRICS_FILES = {
    "sae_detection_metrics.json": "endogenous",
    "sae_intervention_train_metrics.json": "interventional",
}

# split-name normalization, so the two trainers land on shared axes
SPLIT_ALIASES = {"test_endogenous": "test", "dev_endogenous": "dev"}

METRICS = [
    "ap", "auroc", "f1_at_0", "acc_at_0", "base_rate_positive",          # detection
    "paired_directional_acc", "on_value", "mean_gap_on", "mean_gap_off",  # interventional
]

# metrics whose natural range is [0,1] (get a 0..1 y-axis)
RANGE01 = {"ap", "auroc", "f1_at_0", "acc_at_0", "paired_directional_acc"}
# metrics where 0.5 is the chance baseline (get a dashed reference line)
REF05 = {"auroc", "paired_directional_acc"}
# aggregation columns that get a mean+-SE band rather than one line per run
AGG_PALETTES = {"layer": "viridis", "sample": "rocket", "gate": "Set2", "train_regime": "Set1"}
SAMPLE_LABEL = "density bin (0=rare→9=frequent)"


def _style_ax(ax, metric):
    if metric in REF05:
        ax.axhline(0.5, color="grey", ls="--", lw=1)
    if metric in RANGE01:
        ax.set_ylim(0, 1)


def load_sweep(parent_dir: str) -> pd.DataFrame:
    parent = Path(parent_dir)
    rows, skipped = [], 0
    for d in tqdm(sorted(parent.iterdir())):
        if not d.is_dir():
            continue
        m = RUN_RE.search(d.name)
        found = [(f, regime) for f, regime in METRICS_FILES.items() if (d / f).exists()]
        if m is None or not found:
            skipped += 1
            continue
        sample = m.group("sample")
        base = {
            "run": d.name,
            "feature": f"S{m['layer']}.{m['index']}",
            "layer": int(m["layer"]), "index": int(m["index"]),
            # sample index == density stratum from the sweep picker (0=rarest)
            "sample": int(sample) if sample is not None else -1,
            "position": int(m["pos"]),
            "gate": m.group("gate") or "na",
            "scale": m.group("scale"),
        }
        for fname, regime in found:
            metrics = json.loads((d / fname).read_text())
            for split, split_metrics in metrics.items():
                if not isinstance(split_metrics, dict):
                    continue
                split = SPLIT_ALIASES.get(split, split)
                for metric in METRICS:
                    v = split_metrics.get(metric)
                    if v is not None:
                        rows.append({**base, "train_regime": regime, "split": split,
                                     "metric": metric, "value": float(v)})
    if skipped:
        print(f"[plot] skipped {skipped} dirs (name did not match / no metrics json -- "
              f"unfinished runs?)")
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No metrics parsed under {parent_dir}")
    return df


def _prep_hue(data, by, palette_name=None):
    """Discrete, numerically/alphabetically ordered hue + palette for an agg column."""
    levels = sorted(data[by].unique())
    data = data.copy()
    data[by] = pd.Categorical(data[by], categories=levels, ordered=True)
    palette = dict(zip(levels, sns.color_palette(palette_name or AGG_PALETTES.get(by, "viridis"),
                                                 len(levels))))
    return data, palette


def plot_overview(sub, out_dir, split, by="feature", by_label=None,
                  palette_name=None, errorbar=None, style=None):
    """One panel per metric; hue = `by`. Aggregated views pass errorbar='se' so the
    band is +-1 standard error over the runs collapsed into each line. `style` adds a
    second (dashed/solid) dimension, e.g. gate on vs off within each layer."""
    by_label = by_label or by
    data, palette = (_prep_hue(sub, by, palette_name) if by in AGG_PALETTES else (sub.copy(), None))
    g = sns.relplot(
        data=data, x="position", y="value", hue=by, palette=palette, style=style,
        col="metric", col_wrap=min(3, data["metric"].nunique()),
        kind="line", marker="o", errorbar=errorbar,
        facet_kws={"sharey": False}, height=3.5, aspect=1.3,
        legend="brief" if by != "feature" else "auto",
    )
    g.set_titles("{col_name}")
    tail = " (mean±SE)" if errorbar else f" — {data['feature'].nunique()} features"
    style_tail = f", style={style}" if style else ""
    g.figure.suptitle(f"SAE detection ({split}) vs token position — by {by_label}"
                      f"{tail}{style_tail}", y=1.02)
    for ax in g.axes.flat:
        _style_ax(ax, ax.get_title())
    g.tight_layout()
    suffix = f"_x_{style}" if style else ""
    out_path = out_dir / f"{split}__overview_{by}{suffix}.png"
    g.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.figure)
    print(f"[plot] wrote {out_path}")


def plot_per_metric(sub, out_dir, split, by="feature", by_label=None,
                    palette_name=None, errorbar=None, style=None):
    """One PNG per metric for the given view."""
    by_label = by_label or by
    data, palette = (_prep_hue(sub, by, palette_name) if by in AGG_PALETTES else (sub.copy(), None))
    for metric in sorted(data["metric"].unique()):
        m = data[data["metric"] == metric]
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.lineplot(data=m, x="position", y="value", hue=by, palette=palette,
                     style=style, errorbar=errorbar, marker="o", ax=ax)
        _style_ax(ax, metric)
        n_per = m.groupby(by, observed=True)["feature"].nunique().max()
        ax.set_title(f"SAE detection ({split}) — {metric} — by {by_label} "
                     f"(≤{n_per} feats/level)")
        ax.legend(title=by_label, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        fig.tight_layout()
        suffix = f"_x_{style}" if style else ""
        out_path = out_dir / f"{split}__{by}{suffix}__{metric}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out_path}")


def plot_sweep(parent_dir: str, split: str = "test", df=None, individual_metrics: bool = False):
    df = load_sweep(parent_dir) if df is None else df
    sub = df[df["split"] == split].copy()
    if sub.empty:
        raise SystemExit(f"No rows for split={split!r}; available: {sorted(df['split'].unique())}")
    sub = sub.sort_values(["layer", "feature", "metric", "position"])

    out_dir = Path(parent_dir).parents[1] / "plots" / Path(parent_dir).name
    os.makedirs(out_dir, exist_ok=True)

    # only add a dimension when the sweep actually varies it
    has_gate = sub["gate"].nunique() > 1
    has_sample = sub["sample"].nunique() > 1 and (sub["sample"] >= 0).any()
    has_regime = sub["train_regime"].nunique() > 1
    gate_style = "gate" if has_gate else None

    views = [dict(by="feature")]
    views.append(dict(by="layer", by_label="layer", errorbar="se", style=gate_style))
    if has_sample:
        views.append(dict(by="sample", by_label=SAMPLE_LABEL, errorbar="se", style=gate_style))
    if has_gate:
        # the headline arm comparison: gated (cannot read the feature => whatever it
        # scores is context bypass) vs ungated (adapter reaches the site)
        views.append(dict(by="gate", by_label="LoRA input gate", errorbar="se"))
    if has_regime:
        views.append(dict(by="train_regime", by_label="training regime", errorbar="se"))

    for v in views:
        plot_overview(sub, out_dir, split, **v)
        if individual_metrics:
            plot_per_metric(sub, out_dir, split, **v)


def plot_all_splits(parent_dir: str, individual_metrics: bool = False):
    """Plot every split present except dev (test, test_interventional, ...)."""
    df = load_sweep(parent_dir)
    for split in sorted(df["split"].unique()):
        if split == "dev":
            continue
        print(f"\n===== plotting split={split} =====")
        plot_sweep(parent_dir, split=split, df=df, individual_metrics=individual_metrics)


if __name__ == "__main__":
    plot_all_splits("/projectnb/llamagrp/peter/low-level-introspection/outputs/runs/sae_detect_sweep_layer_token")
    # plot_all_splits("/projectnb/llamagrp/peter/low-level-introspection/outputs/runs/sae_intervention_pilot")
