#!/usr/bin/env python3
"""
Joint-loss weight (W) sweep: which mix of endogenous and interventional objectives?

Each (cell, W) is run on 5 neurons, so every point is a mean over neurons and the
band/bars are the standard error ACROSS NEURONS -- heterogeneity, not seed noise.

  W = 1  pure endogenous (predict the true activation; nothing stops context bypass)
  W = 0  pure interventional (report the injected value; no natural-scale target)

Figures
  1 pareto     endogenous vs interventional, one point per W, per cell. Top-right is
               a faithful reader; high-endogenous/low-interventional is the bypass
               corner. This is the plot that picks W.
  2 vs_weight  each metric against W directly, with SE bands -- easier to read off a
               chosen value, and shows whether the optimum is flat or sharp.
  3 cost       LM degradation (measured KL) against W, since a W that buys reading
               by damaging the model more is not free.

CELLS DIFFER IN KL BUDGET (the hard cell is floor-limited at a tight budget and runs
at 3.0). Compare W-SHAPES within a cell; do not compare absolute correlations across
cells.
"""
import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 11, "legend.title_fontsize": 12, "figure.titlesize": 18,
})

ROOT = Path(__file__).parents[1]
NEURON_RE = re.compile(r"N(?P<layer>\d+)\.(?P<index>\d+)")
POS_RE = re.compile(r"POS(?P<pos>-?\d+)")
KL_RE = re.compile(r"_kl(?P<kl>[0-9.]+)")
W_RE = re.compile(r"_w(?P<w>[0-9.]+)")
TRAIN_LEN = 32


def load_weight_sweep(exp_group="mlp_joint_weight_sweep") -> pd.DataFrame:
    rows, incomplete = [], []
    for d in sorted(Path(f"{ROOT}/outputs/runs/{exp_group}").iterdir()):
        if not d.is_dir():
            continue
        n, p, kl, w = (NEURON_RE.search(d.name), POS_RE.search(d.name),
                       KL_RE.search(d.name), W_RE.search(d.name))
        if not (n and p and w):
            continue

        def load(fname, key=None):
            f = d / fname
            if not f.exists():
                return None
            payload = json.loads(f.read_text())
            return payload[key] if key else payload

        interv = load("interventional_test_metrics_gaussian.json", "test")
        endo = load("endogenous_test_metrics_endogenous.json")
        if interv is None or endo is None:
            incomplete.append(d.name)
            continue
        drift, lm = load("activation_drift.json"), load("lm_degradation.json")
        row = {
            "run": d.name,
            "neuron": f"N{n['layer']}.{n['index']}",
            "layer": int(n["layer"]), "position": int(p["pos"]),
            "kl_target": float(kl["kl"]) if kl else np.nan,
            "w": float(w["w"]),
            "interventional_corr": interv.get("corr"),
            "endogenous_corr": endo.get("corr"),
        }
        if drift:
            row["drift_corr"] = drift.get("drift_tuned_vs_base", {}).get("corr")
            row["coupling_gap"] = (drift.get("readout_tracking") or {}).get("coupling_gap")
        if lm:
            for e in lm.get("lengths", []):
                row[f"kl@{e['length']}"] = e.get("kl_mean")
                row[f"top1@{e['length']}"] = e.get("top1_agreement")
        rows.append(row)

    if incomplete:
        print(f"[w] {len(incomplete)} run(s) still training / missing metrics "
              f"(first few: {incomplete[:4]})")
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No finished runs under outputs/runs/{exp_group}")
    df["cell"] = ("L" + df["layer"].astype(str) + " POS" + df["position"].astype(str)
                  + " (kl=" + df["kl_target"].map(lambda v: f"{v:g}") + ")")
    return df


def summarize(df: pd.DataFrame, metrics) -> pd.DataFrame:
    """mean / SE across neurons per (cell, W). SE is heterogeneity across neurons."""
    out = []
    for (cell, w), sub in df.groupby(["cell", "w"]):
        row = {"cell": cell, "w": w, "n_neurons": sub["neuron"].nunique()}
        for m in metrics:
            if m in sub:
                vals = sub[m].dropna()
                row[f"{m}_mean"] = vals.mean() if len(vals) else np.nan
                row[f"{m}_se"] = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
        out.append(row)
    return pd.DataFrame(out).sort_values(["cell", "w"]).reset_index(drop=True)


def plot_pareto(summary, out_dir):
    """Endogenous vs interventional, one point per W, error bars = SE over neurons."""
    cells = sorted(summary["cell"].unique())
    fig, axes = plt.subplots(1, len(cells), figsize=(5.8 * len(cells), 5.4), squeeze=False)
    ws = sorted(summary["w"].unique())
    colors = dict(zip(ws, sns.color_palette("viridis", len(ws))))
    for ax, cell in zip(axes.flat, cells):
        sub = summary[summary["cell"] == cell].sort_values("w")
        ax.plot(sub["interventional_corr_mean"], sub["endogenous_corr_mean"],
                "-", color="grey", alpha=0.5, zorder=1)
        for _, r in sub.iterrows():
            ax.errorbar(r["interventional_corr_mean"], r["endogenous_corr_mean"],
                        xerr=r.get("interventional_corr_se"), yerr=r.get("endogenous_corr_se"),
                        fmt="o", ms=11, color=colors[r["w"]], capsize=3, zorder=3,
                        label=f"w={r['w']:g} (n={int(r['n_neurons'])})")
            ax.annotate(f"{r['w']:g}", (r["interventional_corr_mean"], r["endogenous_corr_mean"]),
                        textcoords="offset points", xytext=(7, 5), fontsize=10)
        ax.set_title(cell)
        ax.set_xlabel("interventional corr (causal reading)")
        ax.set_ylabel("endogenous corr")
        ax.set_xlim(-0.1, 1.05); ax.set_ylim(-0.1, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="lower left")
    fig.suptitle("joint-loss weight: endogenous vs interventional (mean ± SE over neurons)",
                 y=1.02)
    fig.tight_layout()
    path = Path(out_dir) / "weight_pareto.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[w] wrote {path}")


def plot_vs_weight(summary, out_dir, metrics=("interventional_corr", "endogenous_corr")):
    """Each metric against W, so the chosen value can be read off directly."""
    cells = sorted(summary["cell"].unique())
    fig, axes = plt.subplots(1, len(cells), figsize=(5.8 * len(cells), 5), squeeze=False)
    for ax, cell in zip(axes.flat, cells):
        sub = summary[summary["cell"] == cell].sort_values("w")
        # w=0 on a log axis needs a stand-in; keep the order, label honestly
        xs = np.arange(len(sub))
        for m, marker in zip(metrics, ("o", "s")):
            mean, se = sub[f"{m}_mean"].to_numpy(), sub[f"{m}_se"].to_numpy()
            ax.plot(xs, mean, marker=marker, label=m.replace("_corr", ""))
            ax.fill_between(xs, mean - np.nan_to_num(se), mean + np.nan_to_num(se), alpha=0.2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{w:g}" for w in sub["w"]])
        ax.set_xlabel("W (endogenous weight)"); ax.set_ylabel("correlation")
        ax.set_ylim(-0.1, 1.05); ax.set_title(cell)
        ax.grid(alpha=0.25); ax.legend(fontsize=10)
    fig.suptitle("metric vs joint-loss weight (band = SE over neurons)", y=1.02)
    fig.tight_layout()
    path = Path(out_dir) / "weight_vs_metric.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[w] wrote {path}")


def plot_cost(summary, out_dir, length=TRAIN_LEN):
    """Does a given W buy its reading by damaging the model more?"""
    col = f"kl@{length}_mean"
    if col not in summary or summary[col].notna().sum() == 0:
        print(f"[w] no measured kl@{length}; skipping cost panel")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for cell, sub in summary.dropna(subset=[col]).groupby("cell"):
        sub = sub.sort_values("w")
        xs = np.arange(len(sub))
        ax.errorbar(xs, sub[col], yerr=sub.get(f"kl@{length}_se"), marker="o",
                    capsize=3, label=cell)
        ax.set_xticks(xs); ax.set_xticklabels([f"{w:g}" for w in sub["w"]])
    ax.set_xlabel("W (endogenous weight)")
    ax.set_ylabel(f"measured KL @L={length} (nats/token)")
    ax.set_yscale("log"); ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=10)
    ax.set_title("LM cost vs weight")
    fig.tight_layout()
    path = Path(out_dir) / "weight_cost.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[w] wrote {path}")


def report(summary):
    print("\n[w] weight sweep (mean ± SE over neurons):")
    for cell, sub in summary.groupby("cell"):
        print(f"\n--- {cell} ---")
        cols = ["w", "n_neurons", "interventional_corr_mean", "interventional_corr_se",
                "endogenous_corr_mean", "endogenous_corr_se"]
        cols += [c for c in (f"kl@{TRAIN_LEN}_mean", "drift_corr_mean") if c in sub]
        print(sub[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        # a defensible pick: best interventional, then prefer the simpler (larger) W
        # among those within one SE of it
        best = sub.loc[sub["interventional_corr_mean"].idxmax()]
        thresh = best["interventional_corr_mean"] - (best.get("interventional_corr_se") or 0)
        tied = sub[sub["interventional_corr_mean"] >= thresh]
        print(f"    best interventional: w={best['w']:g} "
              f"({best['interventional_corr_mean']:.3f} ± {best.get('interventional_corr_se', float('nan')):.3f});"
              f" within 1 SE: {[f'{w:g}' for w in tied['w']]}")


if __name__ == "__main__":
    EXP_GROUP = "mlp_joint_weight_sweep"
    OUT_DIR = f"{ROOT}/outputs/plots/{EXP_GROUP}"
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_weight_sweep(EXP_GROUP)
    metrics = ["interventional_corr", "endogenous_corr", "drift_corr", "coupling_gap",
               f"kl@{TRAIN_LEN}", f"top1@{TRAIN_LEN}"]
    summary = summarize(df, [m for m in metrics if m in df])
    df.to_csv(f"{OUT_DIR}/weight_sweep_runs.csv", index=False)
    summary.to_csv(f"{OUT_DIR}/weight_sweep_summary.csv", index=False)
    report(summary)
    plot_pareto(summary, OUT_DIR)
    plot_vs_weight(summary, OUT_DIR)
    plot_cost(summary, OUT_DIR)
