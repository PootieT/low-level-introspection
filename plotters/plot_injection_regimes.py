#!/usr/bin/env python3
"""
SFT vs joint loss across injection regimes (rebuttal: is the interventional advantage
specific to the training intervention?).

Reads every <run>/injection_regimes/injection_regimes.csv written by
evals/score_injection_regimes.py for the two arms and draws one panel per
(layer, token) cell: correlation per regime, mean +/- std over the cell's neurons, one
line per method. The x axis runs from the smallest perturbation to the largest:

    local@0.25s, local@1s   each example's live value + delta, scored within example
                            as corr(pred - pred_clean, delta)
    x1 .. x100              gaussian replacement at N x the natural std, corr(pred, v)

Regimes whose draws partly leave the joint runs' training range (uniform over 10x the
natural range) are shaded, using the measured fraction outside it (median over the
joint runs of the panel). The endogenous correlation is printed in each panel title.

Only neurons present in BOTH arms are drawn (the SFT sweep has more per cell), so the
two lines in a panel always average over the same neurons.

Outputs (outputs/plots/injection_regimes/ by default):
  injection_regimes_runs.csv     every run x arm row
  injection_regimes_summary.csv  mean / std / n per method x cell x arm
  injection_regimes_tables.md    the same numbers as one markdown table per cell (the
                                 rebuttal allows no images)
  injection_regimes.png / .pdf

Usage
-----
python plotters/plot_injection_regimes.py
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
# validated categorical slots 1 and 2 (dataviz validator, light surface): joint keeps
# the blue the head-set-ablation figure uses
METHODS = {"joint": ("Joint Loss (KL=0.1)", "#2a78d6"), "sft": ("SFT", "#eb6834")}
INK, MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
OUTSIDE_SHADE_MIN = 0.01      # shade a regime once >= 1% of its draws leave training range

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    p.add_argument("--joint", default="mlp_joint_grid_layer_token:joint_*_kl0.1_*",
                   help="'<group>:<run glob>' of the joint-loss arm.")
    p.add_argument("--sft", default="mlp_endo_sweep_layer_token:endo_*_gateoff_w1.0_300K",
                   help="'<group>:<run glob>' of the SFT arm.")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "plots" / "injection_regimes")
    return p.parse_args()


def load_arm(runs_root: Path, spec: str, method: str) -> pd.DataFrame:
    group, glob = spec.split(":", 1)
    files = sorted((runs_root / group).glob(f"{glob}/injection_regimes/injection_regimes.csv"))
    if not files:
        print(f"[plot] WARNING no injection_regimes.csv for {method} under {runs_root / group}/{glob}")
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["method"] = method
    print(f"[plot] {method}: {len(files)} run(s)")
    return df


def arm_order(arms) -> list[str]:
    """local arms by delta, then global arms by scale (the endogenous arm is not on x)."""
    local = sorted((a for a in arms if a.startswith("local@")),
                   key=lambda a: float(a[len("local@"):].rstrip("s")))
    glob = sorted((a for a in arms if a.startswith("x")), key=lambda a: float(a[1:]))
    return local + glob


def arm_label(arm: str) -> str:
    if arm.startswith("local@"):
        return f"Local\n±{arm[len('local@'):].rstrip('s')}σ"
    return f"{arm[1:]}×"


def paired_neurons(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only (layer, index, position) present in every method."""
    keys = ["layer", "index", "position"]
    per = [set(map(tuple, g[keys].drop_duplicates().to_numpy())) for _, g in df.groupby("method")]
    common = set.intersection(*per) if per else set()
    dropped = df[~df[keys].apply(tuple, axis=1).isin(common)]["run"].unique()
    if len(dropped):
        print(f"[plot] {len(dropped)} run(s) without a partner in the other arm left out")
    return df[df[keys].apply(tuple, axis=1).isin(common)].copy()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["method", "layer", "t", "arm"])
    s = g["corr"].agg(["mean", "std", "count"]).reset_index()
    if "frac_outside_train_support" in df:
        s = s.merge(g["frac_outside_train_support"].median().rename("frac_outside_median")
                    .reset_index(), on=["method", "layer", "t", "arm"])
    return s


def table_order(arms) -> list[str]:
    """Rows of the markdown tables: endogenous, gaussian by scale, local by delta (largest
    first, so the table reads from the paper's eval outwards)."""
    glob = sorted((a for a in arms if a.startswith("x")), key=lambda a: float(a[1:]))
    local = sorted((a for a in arms if a.startswith("local@")),
                   key=lambda a: -float(a[len("local@"):].rstrip("s")))
    return (["endogenous"] if "endogenous" in set(arms) else []) + glob + local


def table_row_label(arm: str, frac_out: float) -> str:
    if arm == "endogenous":
        return "Endogenous (no injection)"
    if arm.startswith("local@"):
        return f"Local ±{arm[len('local@'):].rstrip('s')}σ"
    label = f"Gaussian {arm[1:]}×"
    if np.isfinite(frac_out) and frac_out >= OUTSIDE_SHADE_MIN:
        label += f" ({100 * frac_out:.0f}% outside training range)"
    return label


def write_markdown(summary: pd.DataFrame, out_dir: Path) -> str:
    """One markdown table per (layer, token) cell: rows = regimes, one column per method,
    each 'mean ± std' over the cell's neurons."""
    fmt = lambda r: "–" if r is None else f"{r['mean']:.2f} ± {r['std']:.2f}"
    methods = [m for m in METHODS if m in set(summary["method"])]
    blocks = []
    for (L, t), cell in summary.groupby(["layer", "t"]):
        by = {(r["method"], r["arm"]): r for _, r in cell.iterrows()}
        n = int(cell["count"].max())
        joint = cell[cell["method"] == "joint"].set_index("arm")
        lines = [f"**Layer {L}, token t={t}** ({n} neurons, mean ± std)", "",
                 "| Regime | " + " | ".join(METHODS[m][0] for m in methods) + " |",
                 "|---|" + "---|" * len(methods)]
        for a in table_order(cell["arm"].unique()):
            f = (joint["frac_outside_median"].get(a, np.nan)
                 if "frac_outside_median" in joint else np.nan)
            lines.append(f"| {table_row_label(a, f)} | "
                         + " | ".join(fmt(by.get((m, a))) for m in methods) + " |")
        blocks.append("\n".join(lines))
    note = ("Gaussian N×: the neuron is replaced by v ~ N(μ, (Nσ)²) of its natural "
            "activations; corr(prediction, v). Local ±Dσ: each example's own value plus "
            "δ ~ N(0, (Dσ)²); corr(change in prediction, δ) within example. Joint-loss "
            "training replaced the neuron with values uniform over 10× its natural range.")
    md = "\n\n".join(blocks) + "\n\n" + note + "\n"
    (out_dir / "injection_regimes_tables.md").write_text(md)
    return md


def plot(summary: pd.DataFrame, out_dir: Path):
    layers, ts = sorted(summary["layer"].unique()), sorted(summary["t"].unique())
    arms = arm_order(summary["arm"].unique())
    x = np.arange(len(arms))
    fig, axes = plt.subplots(len(layers), len(ts), figsize=(3.6 * len(ts), 2.9 * len(layers)),
                             sharey=True, squeeze=False)
    for i, L in enumerate(layers):
        for j, t in enumerate(ts):
            ax = axes[i, j]
            cell = summary[(summary["layer"] == L) & (summary["t"] == t)]
            # shade the regimes that partly leave the joint runs' training range
            jt = cell[(cell["method"] == "joint")].set_index("arm")
            if "frac_outside_median" in jt:
                for k, a in enumerate(arms):
                    f = jt["frac_outside_median"].get(a, np.nan)
                    if np.isfinite(f) and f >= OUTSIDE_SHADE_MIN:
                        ax.axvspan(k - 0.5, k + 0.5, color=SHADE, lw=0, zorder=0)
                        ax.text(k, -0.12, f"{100 * f:.0f}% out", ha="center", va="bottom",
                                fontsize=7, color=MUTED)
            # local | global divider
            n_local = sum(a.startswith("local@") for a in arms)
            if 0 < n_local < len(arms):
                ax.axvline(n_local - 0.5, color=MUTED, lw=0.8, ls=":")
            endo = []
            for m, (label, color) in METHODS.items():
                d = cell[cell["method"] == m].set_index("arm")
                if d.empty:
                    continue
                y = d["mean"].reindex(arms).to_numpy()
                e = d["std"].reindex(arms).to_numpy()
                # local and global arms use different metrics: no line across the divider
                for seg, lab in ((slice(0, n_local), None), (slice(n_local, None), label)):
                    ax.errorbar(x[seg], y[seg], yerr=e[seg], color=color, lw=2, marker="o",
                                ms=5, capsize=2.5, elinewidth=1, label=lab, zorder=3)
                if "endogenous" in d.index:
                    endo.append(f"{label.split(' ')[0]} {d.loc['endogenous', 'mean']:.2f}")
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_xticks(x, [arm_label(a) for a in arms])
            ax.set_ylim(-0.15, 1.05)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            title = f"Layer {L}, Token t={t}"
            if endo:
                title += "\nEndogenous: " + ", ".join(endo)
            ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel("Correlation")
    for ax in axes[-1]:
        ax.set_xlabel("Injection Regime")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.02,
             "Local: corr(Δprediction, δ) within example around the live value.  "
             "N×: corr(prediction, injected value), gaussian at N× the natural std.  "
             "Shaded: draws partly outside the joint runs' training range.",
             ha="center", va="top", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"injection_regimes.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    df = pd.concat([load_arm(args.runs_root, args.joint, "joint"),
                    load_arm(args.runs_root, args.sft, "sft")], ignore_index=True)
    if df.empty:
        raise SystemExit("no injection_regimes results found")
    df["t"] = np.where(df["position"] < 0, df["position"] + SEQ_LEN, df["position"])
    df = paired_neurons(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "injection_regimes_runs.csv", index=False)
    summary = summarize(df)
    summary.to_csv(args.out_dir / "injection_regimes_summary.csv", index=False)
    print(write_markdown(summary, args.out_dir))
    plot(summary, args.out_dir)
    print(f"[plot] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
