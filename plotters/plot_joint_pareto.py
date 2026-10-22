#!/usr/bin/env python3
"""
Pareto plot for the joint endogenous+interventional training sweep.

Each run dir (named ..._w<weight>) has two metric files written by
trainers/train_introspector_joint.py:
  - interventional_test_metrics_gaussian.json : {"distribution":..,"test":{"corr":..}}
  - endogenous_test_metrics_endogenous.json   : {"corr":..}

Plots endogenous corr (y) vs interventional@gaussian corr (x), one point per w,
colored by w, with the Pareto-optimal front marked.

Read it as:
  * a point in the TOP-RIGHT (both high)  -> a faithful single-forward reader exists (win)
  * a downward-sloping Pareto front        -> the two readouts conflict (trade-off);
                                              faithful reading is not representable here.

Example
-------
python plotters/plot_joint_pareto.py \
    --runs-glob 'outputs/runs/intervention_train_regressor_joint_sweep/intervention_train_reg_joint_scale10_w*'
"""
import re
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Poster-scale fonts: large enough to read from a few feet away.
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "legend.title_fontsize": 20,
    "figure.titlesize": 26,
})

W_RE = re.compile(r"_w([0-9]*\.?[0-9]+)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-glob", required=True,
                   help="Glob matching per-w run dirs, e.g. '.../intervention_train_reg_joint_scale10_w*'")
    p.add_argument("--interv-metrics", default="interventional_test_metrics_gaussian.json")
    p.add_argument("--endo-metrics", default="endogenous_test_metrics_endogenous.json")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    p.add_argument("--out-name", default="joint_pareto")
    p.add_argument("--title", default="Joint sweep: endogenous vs interventional (Pareto)")
    return p.parse_args()


def _corr(path, nested_key=None):
    d = json.load(open(path))
    if nested_key and nested_key in d:
        d = d[nested_key]
    return d.get("corr")


def pareto_mask(interv, endo):
    """Non-dominated points when MAXIMIZING both interv and endo."""
    pts = np.column_stack([np.asarray(interv, float), np.asarray(endo, float)])
    mask = np.ones(len(pts), bool)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i == j:
                continue
            if pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1] and np.any(pts[j] > pts[i]):
                mask[i] = False
                break
    return mask


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rd in sorted(glob.glob(args.runs_glob)):
        rdp = Path(rd)
        m = W_RE.search(rdp.name)
        if not m:
            print(f"[warn] no _w<val> in {rdp.name}; skipping")
            continue
        ip, ep = rdp / args.interv_metrics, rdp / args.endo_metrics
        if not ip.exists() or not ep.exists():
            print(f"[warn] {rdp.name}: missing metrics (interv={ip.exists()} endo={ep.exists()}); skipping")
            continue
        rows.append({"w_endo": float(m.group(1)), "run": rdp.name,
                     "interv_corr": _corr(ip, nested_key="test"), "endo_corr": _corr(ep)})
    if not rows:
        raise SystemExit(f"No runs with both metric files matched {args.runs_glob!r}")

    df = pd.DataFrame(rows).dropna(subset=["interv_corr", "endo_corr"]).sort_values("w_endo").reset_index(drop=True)
    df["pareto"] = pareto_mask(df["interv_corr"], df["endo_corr"])
    df.to_csv(args.output_dir / f"{args.out_name}.csv", index=False)
    print(df[["w_endo", "interv_corr", "endo_corr", "pareto"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 9))
    # trajectory in order of w (shows how the balance moves the operating point)
    ax.plot(df["interv_corr"], df["endo_corr"], "-", color="0.7", lw=1, zorder=1)
    sc = ax.scatter(df["interv_corr"], df["endo_corr"], c=df["w_endo"], cmap="viridis",
                    s=200, zorder=2, edgecolor="k", linewidth=0.5)
    for _, r in df.iterrows():
        ax.annotate(f"{r.w_endo:.2f}", (r.interv_corr, r.endo_corr),
                    textcoords="offset points", xytext=(6, 4), fontsize=16)
    # Pareto-optimal frontier
    pf = df[df["pareto"]].sort_values("interv_corr")
    if len(pf):
        ax.plot(pf["interv_corr"], pf["endo_corr"], "r--", lw=2.5, zorder=1, label="Pareto front")
        ax.scatter(pf["interv_corr"], pf["endo_corr"], marker="*", s=600, facecolor="none",
                   edgecolor="red", linewidth=2.5, zorder=3, label="Pareto-optimal")
        ax.legend(loc="lower left")

    cbar = fig.colorbar(sc, ax=ax, label="w (endogenous weight)")
    cbar.ax.tick_params(labelsize=18)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("interventional@gaussian corr (causal reading)")
    ax.set_ylabel("endogenous corr (natural-value reading)")
    ax.set_title(args.title)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{args.out_name}.png", dpi=150)
    plt.close(fig)
    print(f"[joint-pareto] wrote {len(df)} points to {args.output_dir}/{args.out_name}.png (+ csv)")


if __name__ == "__main__":
    main()
