#!/usr/bin/env python3
"""
Plot attention-logit introspection across (layer, head, source-token offset).

Point it at the PARENT dir of a sweep, e.g.
    outputs/runs/attn_endo_sweep_layer_src/            (experiment 1, sampled heads)
    outputs/runs/attn_endo_sweep_layer_src_fixedhead/  (experiment 2, pinned heads)

A NEW regex rather than an edit to the neuron ones. Seven analysis scripts parse run
directory names with a hardcoded `N(\\d+)\\.(\\d+)` and skip a non-matching directory
WITHOUT warning (e.g. compare_sft_vs_joint.py:51,118). Widening those in place would
silently change which runs the existing MLP figures include; this follows the precedent
set by plot_sae_detection_sweep.py and owns its own pattern.

Run names look like
    endo_A8.5@-3_POS-1_gateon_w1.0_300K
i.e. A<layer>.<head>@<source offset>, destination pinned by POS.

Outputs (under plots/<sweep>/):
    corr_vs_src_offset.png       corr against source offset, one line per layer,
                                 gate on/off side by side; BOS marked separately
    corr_vs_src_offset_by_head.png   the same, faceted by head
    summary.csv                  one row per run, everything the plots draw from

TWO THINGS THE PLOTS DELIBERATELY SEPARATE, both of which would otherwise be misread:

  layer 0 is a CEILING CONTROL. Its pre-softmax logit is a function of embeddings and
    RoPE only, so with the offset fixed and the destination pinned it is fully determined
    by the two token strings the prompt already names -- a model can score well without
    reading anything internal. Drawn dashed.

  the deepest offset is BOS, the attention sink. Its logits have a very different mean
    and often a much smaller spread, and both the training loss (MSE/var) and the metric
    (Pearson corr) are scale invariant, so a near-constant signal yields nan or noise
    rather than a low score. Drawn as a separate marker, and `target_std` /
    `low_variance` in summary.csv say when a point should not be read at all.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# endo_A8.5@-3_POS-1_gateon_w1.0_300K   (P<layer>.<head> for post-softmax weights)
RUN_RE = re.compile(
    r"(?P<code>[AP])(?P<layer>\d+)\.(?P<head>\d+)@(?P<src>-?\d+)"
    r"_POS(?P<dst>-?\d+)"
    r"(?:_gate(?P<gate>on|off))?"
    r"(?:_w(?P<w>[0-9.eE+-]+))?"
)

METRICS = "endogenous_test_metrics_endogenous.json"
PREDICTIONS = "endogenous_test_predictions_endogenous.csv"
DRIFT = "activation_drift.json"

# Below this target standard deviation a correlation is not measuring introspection, it
# is dividing by ~0. Reported, not dropped: a vanishing denominator must stay visible.
LOW_VARIANCE_STD = 1e-3


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", help="parent directory holding the run directories")
    ap.add_argument("--out", default=None, help="output dir (default plots/<sweep name>)")
    ap.add_argument("--metric", default="corr", help="metric column to plot")
    return ap.parse_args()


def collect(sweep_dir: Path) -> pd.DataFrame:
    rows, skipped = [], []
    for run in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        m = RUN_RE.search(run.name)
        if not m:
            skipped.append(run.name)
            continue
        metrics_path = run / METRICS
        if not metrics_path.exists():
            skipped.append(f"{run.name} (no {METRICS})")
            continue
        row = {
            "run": run.name,
            "signal_code": m.group("code"),
            "layer": int(m.group("layer")),
            "head": int(m.group("head")),
            "src_offset": int(m.group("src")),
            "dst_pos": int(m.group("dst")),
            "gate": m.group("gate") or "off",
            "w": float(m.group("w")) if m.group("w") else float("nan"),
        }
        row.update(json.loads(metrics_path.read_text()))

        # Spread of the TARGET on the eval split. corr is scale invariant, so this is
        # the only thing that distinguishes "the model cannot read this signal" from
        # "there was nothing to read".
        pred_path = run / PREDICTIONS
        if pred_path.exists():
            act = pd.read_csv(pred_path)["activation"].to_numpy()
            row["target_std"] = float(np.std(act))
            row["target_mean"] = float(np.mean(act))
            row["low_variance"] = bool(row["target_std"] < LOW_VARIANCE_STD)

        # Whether the cached target still equals what the model computes. A cell whose
        # alignment failed is not a result, it is a bug -- surface it next to the score.
        drift_path = run / DRIFT
        if drift_path.exists():
            d = json.loads(drift_path.read_text())
            row["alignment_ok"] = d.get("alignment_ok")
            row["alignment_corr"] = d.get("best_alignment_corr")
        rows.append(row)

    # Loud about what was ignored: silence here is exactly what makes the neuron
    # scripts drop attention runs unnoticed.
    if skipped:
        print(f"[plot] skipped {len(skipped)} directories, e.g. {skipped[:5]}")
    if not rows:
        raise SystemExit(f"no attention runs found under {sweep_dir}")
    return pd.DataFrame(rows)


def _plot_axis(ax, df, metric, title):
    """corr vs source offset, one line per layer; BOS and layer 0 called out."""
    offsets = sorted(df["src_offset"].unique())
    bos = min(offsets) if offsets else None
    layers = sorted(df["layer"].unique())
    cmap = plt.get_cmap("viridis")
    for i, layer in enumerate(layers):
        g = df[df["layer"] == layer].groupby("src_offset")[metric]
        mean, sem = g.mean(), g.sem()
        x = list(mean.index)
        color = cmap(i / max(len(layers) - 1, 1))
        # layer 0 is a text-only ceiling, not a comparable data point
        style = "--" if layer == 0 else "-"
        label = f"L{layer}" + (" (text ceiling)" if layer == 0 else "")
        ax.plot(x, mean.values, style, marker="o", ms=4, color=color, label=label)
        ax.fill_between(x, mean.values - sem.values, mean.values + sem.values,
                        color=color, alpha=0.15)
        # the BOS point gets its own marker: it is a different regime (attention sink),
        # not the far end of a smooth axis
        if bos in mean.index:
            ax.plot([bos], [mean.loc[bos]], marker="*", ms=13, color=color,
                    linestyle="none")
    if df.get("low_variance") is not None:
        bad = df[df["low_variance"].fillna(False)]
        if len(bad):
            ax.scatter(bad["src_offset"], bad[metric], marker="x", s=60, color="red",
                       zorder=5, label=f"target std < {LOW_VARIANCE_STD:g}")
    ax.set_xlabel("source token offset from destination  (0 = self, min = BOS)")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.grid(alpha=0.25)


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    out = Path(args.out) if args.out else Path("plots") / sweep_dir.name
    out.mkdir(parents=True, exist_ok=True)

    df = collect(sweep_dir)
    df.to_csv(out / "summary.csv", index=False)
    print(f"[plot] {len(df)} runs -> {out/'summary.csv'}")
    if "alignment_ok" in df.columns:
        bad = df[df["alignment_ok"] == False]  # noqa: E712  (NaN must not match)
        if len(bad):
            print(f"[plot] *** {len(bad)} runs FAILED cache alignment; their scores are "
                  f"not interpretable: {sorted(bad['run'])[:5]}")

    gates = sorted(df["gate"].unique())
    fig, axes = plt.subplots(1, len(gates), figsize=(7 * len(gates), 5.5), squeeze=False)
    for ax, gate in zip(axes[0], gates):
        _plot_axis(ax, df[df["gate"] == gate], args.metric,
                   f"gate {gate} " + ("(adapter cannot touch the text)" if gate == "on"
                                      else "(adapter active over the text)"))
    axes[0][-1].legend(fontsize=8, ncol=2)
    fig.suptitle(f"{sweep_dir.name}: attention-logit introspection vs source offset")
    fig.tight_layout()
    fig.savefig(out / "corr_vs_src_offset.png", dpi=150)
    plt.close(fig)

    heads = sorted(df["head"].unique())
    ncol = min(4, len(heads))
    nrow = int(np.ceil(len(heads) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    for ax, head in zip(axes.ravel(), heads):
        _plot_axis(ax, df[(df["head"] == head) & (df["gate"] == gates[-1])],
                   args.metric, f"head {head} (gate {gates[-1]})")
    for ax in axes.ravel()[len(heads):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "corr_vs_src_offset_by_head.png", dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out}/corr_vs_src_offset.png and corr_vs_src_offset_by_head.png")


if __name__ == "__main__":
    main()
