#!/usr/bin/env python3
"""
Which MLP quantity would be easiest to introspect CAUSALLY? For each SFT grid neuron,
in the base model, three candidate targets:
  g   gate pre-activation  (w_gate . x)
  a   silu(g)              (what the grid trains on and injects into)
  h   a * u                (down_proj input; one neuron's post-down_proj write is
                            h * W_down[:, i], so it carries exactly h's information)

Two panels, per layer (mean and SE over the grid's neurons):
  bypass         held-out ridge R^2 of the target from the residual BEFORE the MLP --
                 how well SFT could fit the target without reading the neuron's write
  causal ceiling (E k)^2 / E k^2, k = per-token gain from an injected target value to
                 the write along W_down[:, i] -- the best corr^2 any fixed linear
                 readout of the write can reach under injection (h: exactly 1)

Faithful SFT is favored where the ceiling is high and the bypass is low.

Input: outputs/plots/mlp_endo_sweep_layer_token/write_stats/write_stats.csv
  (mechanistic_analysis/score_neuron_write_stats.py)

Usage
  python plotters/plot_target_site_readability.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "outputs/plots/mlp_endo_sweep_layer_token/write_stats"
TARGETS = [("g", "Gate Pre-Act. g", "tab:red"), ("a", "silu(g) = a  (Current)", "tab:blue"),
           ("h", "a · up = h  (Down-Proj Input)", "tab:green")]
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14})


def main():
    d = pd.read_csv(WS / "write_stats.csv")
    g = d.groupby("layer")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    rows = []
    for ax, (col, title, ylab) in zip(axes, [
            ("bypass_r2", "Bypass: Target From Pre-MLP Residual", "Held-Out Ridge R²"),
            ("causal_ceiling", "Causal Ceiling: Injected Target vs. MLP Write",
             "corr²(Injected Target, MLP Write)")]):
        for t, lab, c in TARGETS:
            m, se = g[f"{col}_{t}"].mean(), g[f"{col}_{t}"].sem()
            ax.errorbar(m.index, m.values, yerr=se.values, marker="o", color=c, label=lab,
                        capsize=3, lw=2)
            rows += [dict(metric=col, target=t, layer=L, mean=m[L], se=se[L]) for L in m.index]
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylab)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(sorted(d["layer"].unique()))
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(WS / "target_site_readability.png", dpi=150, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(WS / "target_site_readability.csv", index=False)
    print(pd.DataFrame(rows).pivot_table(index="layer", columns=["metric", "target"],
                                         values="mean").round(3).to_string())
    print(f"[plot] wrote {WS / 'target_site_readability.png'}")


if __name__ == "__main__":
    main()
