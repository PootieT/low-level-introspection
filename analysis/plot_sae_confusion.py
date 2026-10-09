#!/usr/bin/env python3
"""Endogenous confusion matrices for the layer-9 SAE joint runs. Reads the per-class
eval JSON (from _scratch_equiv/endo_perclass.py) and plots, per feature at a chosen
joint weight, the 2x3 confusion matrix -- true {active, inactive} x predicted
{active, inactive, neither}; the neither column is dropped when every panel has zero
neither predictions -- colored by row-normalized rate (recall view, robust to
the ~99% inactive imbalance) with raw counts annotated. Writes
figures/sae/confusion_endogenous_w<W>.{png,pdf}."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sae_style import (INK, ORDER, SMALL, apply_rc, box, feature_cmap, feature_color, feature_label,
                       fig_title, save)

ROOT = Path(__file__).parents[1]
apply_rc()
PRED = ["active", "inactive", "neither"]
TRUE = ["active", "inactive"]


def confusion_counts(r):
    return np.array([[r["act_pred_active"], r["act_pred_inactive"], r["act_pred_neither"]],
                     [r["ina_pred_active"], r["ina_pred_inactive"], r["ina_pred_neither"]]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=ROOT / "_scratch_equiv/endo_perclass.json")
    p.add_argument("--weight", type=float, default=0.9)
    p.add_argument("--out-dir", type=Path, default=ROOT / "figures/sae")
    args = p.parse_args()

    rows = json.loads(Path(args.json).read_text())
    by = {(r["feature"], r["w"]): r for r in rows}
    feats = [f"S9.{f}" for f in ORDER if (f"S9.{f}", args.weight) in by]
    if not feats:
        raise SystemExit(f"no rows at w={args.weight} in {args.json}")

    show_neither = any(confusion_counts(by[(f, args.weight)])[:, 2].any() for f in feats)
    k = 3 if show_neither else 2
    fig, axes = plt.subplots(1, len(feats), figsize=((4.3 if show_neither else 3.4) * len(feats), 3.9))
    if len(feats) == 1:
        axes = [axes]
    for i_ax, (ax, feat) in enumerate(zip(axes, feats)):
        r = by[(feat, args.weight)]
        M = confusion_counts(r)[:, :k]
        Mn = M / M.sum(axis=1, keepdims=True)
        annot = np.array([[f"{M[i, j]:,}\n{Mn[i, j] * 100:.1f}%" for j in range(k)] for i in range(2)])
        sns.heatmap(Mn, annot=annot, fmt="", cmap=feature_cmap(feat.replace("S9.", "")), vmin=0, vmax=1,
                    cbar=False, xticklabels=PRED[:k], yticklabels=TRUE, linewidths=0.5, linecolor="white",
                    annot_kws={"fontsize": SMALL}, ax=ax)
        for t, v in zip(ax.texts, Mn.flatten()):
            t.set_color("white" if v > 0.55 else INK)
        ax.set_title(f"{feature_label(feat, ' ')}\nMCC = {r.get('mcc', float('nan')):.3f}",
                     color=feature_color(feat))
        ax.set_xlabel("predicted")
        ax.set_ylabel("true" if i_ax == 0 else "")
        box(ax)
    fig_title(fig, f"Endogenous confusion matrices (gate off, w={args.weight:g}), "
                   f"row-normalized shading")
    fig.tight_layout()
    save(fig, args.out_dir / f"confusion_endogenous_w{args.weight:g}")


if __name__ == "__main__":
    main()
