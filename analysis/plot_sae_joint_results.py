#!/usr/bin/env python3
"""Slide figures for the gate-off JOINT SAE-detection sweep.

Reads the joint trainer's `sae_joint_train_metrics.json` (written by
train_sae_detector_joint.py) across the weight sweep and plots the CURRENT metric
set only -- balanced accuracy + precision for the endogenous and interventional
readouts, and interventional balanced accuracy vs weight (faithfulness across the
sweep). AUROC/AP are deliberately not plotted; they were retired.

  python analysis/plot_sae_joint_results.py \
      --runs outputs/runs/sae_l9_joint --weight 0.9

Writes PNGs to figures/. Static, one 0-1 axis each, colorblind-safe hues, direct
value labels, chance line at 0.5 for the interventional metric.
"""
import argparse, glob, json, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#8a8a85", "#fcfcfb"
plt.rcParams.update({"font.size": 12, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF})
CONCEPT = {"12408": "appreciation", "1568": "limitations", "978": "statistics",
           "4120": "affiliations", "4854": "identifiers"}
ORDER = ["12408", "1568", "978", "4120", "4854"]


def load_joint(runs_dir):
    """{feature_id: {weight: metrics_dict}} from every sae_joint_train_metrics.json."""
    out = {}
    for f in glob.glob(f"{runs_dir}/saeJoint_S9.*/sae_joint_train_metrics.json"):
        fi = re.search(r"S9\.(\d+)", f).group(1)
        m = json.load(open(f))
        out.setdefault(fi, {})[float(m["joint_loss_weight"])] = m
    return out


def grouped_barh(feats, series, title, fname, chance=None, xlabel=None):
    """series = [(label, color, [values per feature]), ...] (2-3 series)."""
    names = [f"S9.{f}\n{CONCEPT.get(f, '')}" for f in feats]
    n = len(feats); y = np.arange(n)[::-1]
    k = len(series); h = 0.8 / k
    offs = [(k - 1) / 2 * h - i * h for i in range(k)]
    fig, ax = plt.subplots(figsize=(8.4, 1.4 + 0.95 * n))
    for (lab, col, vals), off in zip(series, offs):
        ax.barh(y + off, vals, h, color=col, label=lab)
        for yi, v in zip(y + off, vals):
            if v == v:
                ax.text(v + 0.012, yi, f"{v:.2f}", va="center", ha="left",
                        fontsize=10, color=INK)
    if chance is not None:
        ax.axvline(chance, color=MUTED, ls="--", lw=1.3)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=11, color=INK)
    ax.set_ylim(-1.25, n - 0.4); ax.set_xlim(0, 1.12); ax.set_xticks([0, .25, .5, .75, 1])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5, color=INK2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cfcfca")
    ax.tick_params(colors=INK2, length=0)
    ax.set_axisbelow(True); ax.grid(axis="x", color="#e9e9e5", lw=1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=k,
              frameon=False, fontsize=10.5)
    ax.set_title(title, fontsize=13, color=INK, loc="left", pad=10)
    fig.tight_layout(); fig.savefig(fname, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("wrote", fname)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="outputs/runs/sae_l9_joint")
    p.add_argument("--weight", type=float, default=0.9,
                   help="Joint weight to show in the per-feature bars.")
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    joint = load_joint(args.runs)
    if not joint:
        raise SystemExit(f"no sae_joint_train_metrics.json under {args.runs}")
    feats = [f for f in ORDER if f in joint] + [f for f in joint if f not in ORDER]

    def at(f, arm, key):
        m = joint[f].get(args.weight, {}).get(arm, {})
        v = m.get(key)
        return float(v) if v is not None else float("nan")

    w = args.weight
    grouped_barh(
        feats,
        [("balanced acc", BLUE, [at(f, "test_endogenous", "balanced_accuracy") for f in feats]),
         ("precision", ORANGE, [at(f, "test_endogenous", "precision") for f in feats])],
        f"Endogenous detection (gate off, w={w:g})",
        args.out_dir / f"joint_endogenous_w{w:g}.png", xlabel="score")

    grouped_barh(
        feats,
        [("balanced acc", BLUE, [at(f, "test_interventional", "balanced_accuracy") for f in feats]),
         ("precision", ORANGE, [at(f, "test_interventional", "precision") for f in feats])],
        f"Interventional faithfulness (gate off, w={w:g})",
        args.out_dir / f"joint_interventional_w{w:g}.png", chance=0.5,
        xlabel="score (balanced acc: faithful=1.0, bypass=0.5)")

    weights = sorted({wt for f in joint for wt in joint[f]})
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for f in feats:
        ys = [joint[f].get(wt, {}).get("test_interventional", {}).get("balanced_accuracy", np.nan)
              for wt in weights]
        ax.plot(weights, ys, "-o", label=f"S9.{f} {CONCEPT.get(f, '')}")
    ax.axhline(0.5, color=MUTED, ls="--", lw=1.3)
    ax.set_xlabel("joint weight w (endogenous term)", color=INK2)
    ax.set_ylabel("interventional balanced accuracy", color=INK2)
    ax.set_ylim(0.4, 1.03); ax.set_title("Faithfulness across the joint sweep", loc="left", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e9e9e5", lw=1); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig(args.out_dir / "joint_faithfulness_vs_weight.png", dpi=160)
    plt.close(fig); print("wrote", args.out_dir / "joint_faithfulness_vs_weight.png")


if __name__ == "__main__":
    main()
