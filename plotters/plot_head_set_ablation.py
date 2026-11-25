#!/usr/bin/env python3
"""
Which value does the report follow when the endogenous / interventional head SETS are
ablated? -- pooled over a sweep's mechanistic_analysis/score_head_set_ablation.py outputs.

One figure per head set definition (mass0.8, top30, ...): rows are (signal layer, token
position) cells, columns the three readouts, x the ablation conditions. Each blue dot is
one run (one neuron) with the real head set ablated; each gray dot is the same run's
mean over its layer- and size-matched random controls. Bars mark the median over runs.

  Endogenous             corr(prediction, natural activation a), no injection
  Interventional         corr(prediction, injected value v), x1 injection
  Natural | Injection    corr(prediction, a) under x1 injection: the report following
                         the natural value although the site carries v (context bypass)

Also writes head_set_ablation_runs.csv (every row of every run) and
head_set_ablation_summary.csv (median and IQR over runs per cell x condition x control).

Usage
-----
python plotters/plot_head_set_ablation.py --group mlp_joint_grid_layer_token
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEQ_LEN = 32
CONDITIONS = ["none", "endo_only", "interv_only", "joint", "keep_endo", "keep_interv"]
COND_LABEL = {"none": "None", "endo_only": "Endo Only\n(E∖I)",
              "interv_only": "Interv. Only\n(I∖E)", "joint": "Joint\n(E∩I)",
              "keep_endo": "All but\nEndo", "keep_interv": "All but\nInterv."}
READOUTS = [("corr_endogenous", "Endogenous: corr(pred, a)"),
            ("corr_interv", "Interventional: corr(pred, v)"),
            ("corr_natural_under_injection", "Natural | Injection: corr(pred, a)")]
REAL, RANDOM = "#2a78d6", "#9a9893"       # series-1 blue; recessive gray for the control
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", default="mlp_joint_grid_layer_token")
    p.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    p.add_argument("--run-glob", default="*",
                   help="Glob over run directory names within the group.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: outputs/plots/<group>/head_set_ablation/")
    return p.parse_args()


def load(group_dir: Path, run_glob: str) -> pd.DataFrame:
    files = sorted(group_dir.glob(f"{run_glob}/head_set_ablation/head_set_ablation.csv"))
    if not files:
        raise SystemExit(f"no head_set_ablation.csv under {group_dir}/{run_glob}/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["t"] = np.where(df["position"] < 0, df["position"] + SEQ_LEN, df["position"])
    df["cell"] = "L=" + df["signal_layer"].astype(str) + ", t=" + df["t"].astype(str)
    print(f"[plot] {len(files)} run(s), {df['cell'].nunique()} cell(s)")
    return df


def per_run(df: pd.DataFrame, head_def: str) -> pd.DataFrame:
    """One value per (run, condition, control): real as is, random averaged over draws.
    The no-ablation row is shared by every head set definition."""
    d = df[(df["head_def"] == head_def) | (df["condition"] == "none")]
    keys = ["run", "cell", "signal_layer", "t", "condition", "control"]
    cols = [c for c, _ in READOUTS] + ["n_ablated", "empty"]
    return d.groupby(keys, as_index=False)[cols].mean()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hd in sorted(set(df["head_def"]) - {"none"}):
        r = per_run(df, hd)
        for (cell, cond, ctrl), g in r.groupby(["cell", "condition", "control"]):
            row = {"head_def": hd, "cell": cell, "condition": cond, "control": ctrl,
                   "n_runs": len(g), "n_empty": int((g["empty"] > 0).sum()),
                   "n_ablated_median": float(g["n_ablated"].median())}
            for c, _ in READOUTS:
                v = g[c].dropna()
                row.update({f"{c}_median": v.median() if len(v) else np.nan,
                            f"{c}_q25": v.quantile(0.25) if len(v) else np.nan,
                            f"{c}_q75": v.quantile(0.75) if len(v) else np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


def plot_head_def(df: pd.DataFrame, head_def: str, out: Path):
    r = per_run(df, head_def)
    cells = (r[["signal_layer", "t", "cell"]].drop_duplicates()
             .sort_values(["signal_layer", "t"])["cell"].tolist())
    fig, axes = plt.subplots(len(cells), len(READOUTS), squeeze=False, sharey=True,
                             figsize=(4.4 * len(READOUTS), 2.5 * len(cells) + 0.6))
    rng = np.random.default_rng(0)
    for i, cell in enumerate(cells):
        for j, (col, title) in enumerate(READOUTS):
            ax = axes[i, j]
            g = r[r["cell"] == cell]
            for k, cond in enumerate(CONDITIONS):
                for ctrl, color, dx, fill in (("real", REAL, -0.16, True),
                                              ("random", RANDOM, 0.16, False)):
                    v = g[(g["condition"] == cond) & (g["control"] == ctrl)][col].dropna()
                    if cond == "none" and ctrl == "random":
                        continue
                    x = k + (0 if cond == "none" else dx)
                    if len(v):
                        ax.scatter(x + rng.uniform(-0.05, 0.05, len(v)), v, s=22,
                                   facecolors=color if fill else "white", edgecolors=color,
                                   linewidths=1.2, zorder=3)
                        ax.hlines(v.median(), x - 0.12, x + 0.12, color=color, lw=2.5,
                                  zorder=4)
                n_empty = int(((g["condition"] == cond) & (g["control"] == "real")
                               & (g["empty"] > 0)).sum())
                if n_empty:
                    ax.text(k, -0.27, f"{n_empty} empty", ha="center", fontsize=7,
                            color=MUTED)
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.grid(axis="y", alpha=0.25)
            ax.set_ylim(-0.32, 1.05)
            ax.set_xticks(range(len(CONDITIONS)))
            ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS] if i == len(cells) - 1
                               else [])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"{cell}\nPearson r")
    handles = [plt.Line2D([], [], marker="o", ls="", color=REAL, label="Head set ablated"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", color=RANDOM,
                          label="Random heads, same count per layer")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Head Set Ablation ({head_def}); dot = one neuron, bar = median",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def main():
    args = parse_args()
    df = load(args.runs_root / args.group, args.run_glob)
    out_dir = args.out_dir or ROOT / "outputs" / "plots" / args.group / "head_set_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "head_set_ablation_runs.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out_dir / "head_set_ablation_summary.csv", index=False)
    real = summary[summary["control"] == "real"]
    show = ["head_def", "cell", "condition", "n_runs", "n_empty", "n_ablated_median"] + \
        [f"{c}_median" for c, _ in READOUTS]
    print(real[show].round(3).to_string(index=False))
    for hd in sorted(set(df["head_def"]) - {"none"}):
        plot_head_def(df, hd, out_dir / f"head_set_ablation_{hd}.png")


if __name__ == "__main__":
    main()
