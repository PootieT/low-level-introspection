#!/usr/bin/env python3
"""Balanced-accuracy-only figures for the gate-off JOINT SAE-detection sweep.

Reads every sae_joint_train_metrics.json and plots ONLY balanced accuracy:
per-feature endogenous vs interventional bars at a chosen weight, and balanced
accuracy vs joint weight (endogenous + interventional) as line panels.

  python plotters/plot_sae_balanced_accuracy.py \
      --runs outputs/runs/sae_l9_joint --weight 0.9

Writes outputs/plots/sae_l9_joint/balanced_accuracy_w<W>.{png,pdf} and balanced_accuracy_vs_weight.{png,pdf}.
"""
import argparse, glob, json, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sae_style import (INK, INK2, ORDER, SMALL, apply_rc, chance_line, feature_color,
                       feature_label, fig_title, save, style_axes, tint)

ROOT = Path(__file__).parents[1]
apply_rc()


def load_joint(runs_dir):
    out = {}
    for f in glob.glob(f"{runs_dir}/saeJoint_S9.*/sae_joint_train_metrics.json"):
        fi = re.search(r"S9\.(\d+)", f).group(1)
        m = json.load(open(f))
        out.setdefault(fi, {})[float(m["joint_loss_weight"])] = m
    return out


def bacc(m, arm):
    d = m.get(arm) or {}
    v = d.get("balanced_accuracy")
    return float(v) if v is not None else float("nan")


def grouped_barh(feats, series, title, fname, xlabel):
    n = len(feats); y = np.arange(n)[::-1]
    k = len(series); h = 0.8 / k
    offs = [(k - 1) / 2 * h - i * h for i in range(k)]
    fig, ax = plt.subplots(figsize=(8.4, 1.4 + 0.95 * n))
    for (lab, amount, hatch, vals), off in zip(series, offs):
        cols = [tint(feature_color(f), amount) for f in feats]
        ax.barh(y + off, vals, h, color=cols, hatch=hatch, edgecolor="white", lw=0)
        for yi, v in zip(y + off, vals):
            if v == v:
                ax.text(v + 0.012, yi, f"{v:.2f}", va="center", ha="left", fontsize=SMALL, color=INK)
    chance_line(ax, axis="v")
    ax.set_yticks(y); ax.set_yticklabels([feature_label(f) for f in feats])
    ax.set_ylim(-1.25, n - 0.4); ax.set_xlim(0, 1.12); ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel(xlabel)
    style_axes(ax, grid="x")
    handles = [Patch(facecolor=tint(INK2, amount), hatch=hatch, edgecolor="white", label=lab)
               for lab, amount, hatch, _ in series]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=k)
    fig_title(fig, title)
    fig.tight_layout()
    save(fig, fname)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(ROOT / "outputs/runs/sae_l9_joint"))
    p.add_argument("--weight", type=float, default=0.9)
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/plots/sae_l9_joint")
    args = p.parse_args()

    joint = load_joint(args.runs)
    if not joint:
        raise SystemExit(f"no sae_joint_train_metrics.json under {args.runs}")
    feats = [f for f in ORDER if f in joint] + [f for f in joint if f not in ORDER]

    w = args.weight
    grouped_barh(
        feats,
        [("endogenous", 0.0, None, [bacc(joint[f].get(w, {}), "test_endogenous") for f in feats]),
         ("interventional", 0.5, "//", [bacc(joint[f].get(w, {}), "test_interventional") for f in feats])],
        f"Balanced accuracy (gate off, w={w:g})",
        args.out_dir / f"balanced_accuracy_w{w:g}", xlabel="balanced accuracy (chance = 0.5)")

    weights = sorted({wt for f in joint for wt in joint[f]})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, arm, ttl in zip(axes, ["test_endogenous", "test_interventional"],
                            ["Endogenous", "Interventional"]):
        for f in feats:
            ys = [bacc(joint[f].get(wt, {}), arm) for wt in weights]
            ax.plot(weights, ys, "-o", color=feature_color(f), label=feature_label(f, " "))
        chance_line(ax)
        ax.set_xlabel("joint weight w")
        ax.set_ylim(0.4, 1.03); ax.set_title(ttl)
        style_axes(ax)
    axes[0].set_ylabel("balanced accuracy")
    axes[1].legend(loc="lower left")
    fig_title(fig, "Balanced accuracy across the joint sweep (gate off)")
    fig.tight_layout()
    save(fig, args.out_dir / "balanced_accuracy_vs_weight")


if __name__ == "__main__":
    main()
