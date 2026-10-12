#!/usr/bin/env python3
"""
How much of the language model does introspection cost, as a function of what it is
asked to introspect?

KL(base || tuned) on the input text, in nats/token, from score_lm_degradation.py. Two
questions, two halves:

  1 SINGLE LOCALE   the layer/token grid in the UNANCHORED regime (the KL-0.1 regime
                    holds KL near 0.1 by construction and says nothing about need).
                    Which locales are expensive? Earlier tokens and later layers were
                    the expectation; the heatmap is the answer. KL against
                    interventional corr per cell shows whether cost buys fidelity.

  2 MULTI LOCALE    the length sweep's diagonal (every position trained), the neuron
                    sweep's cells that were given enough data to learn, and the layer
                    sweep's cells above its data floor -- each against the single-
                    locale grid cell at the same layer and position (layer 4, pos -1).
                    KL against number of locales is the capacity-scaling read.

KL is reported at each eval length score_lm_degradation ran (32/128/512). The 32-token
number is primary for runs trained at 32; the length sweep's runs train longer, so the
full profile is kept and the plots say which length they show.

The multi-locale sweeps did not score KL in their run scripts. Runs without
lm_degradation.json are listed with the exact command to score them, and every plot is
drawn from whatever is present.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parents[1]))
from analysis.run_catalog import (grid_runs, multi_locale_runs, missing_kl_commands,  # noqa: E402
                                  ROOT)

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
                     "legend.fontsize": 11})
GRID_GROUP = "mlp_joint_grid_layer_token"
LENGTHS = (32, 128, 512)
REF_CELL = {"layer": 4, "position": -1}    # where the multi-locale sweeps live
L_TRAIN = 32                               # training window (positions shown absolute)


def kl_cols(df):
    return [f"kl@{L}" for L in LENGTHS if f"kl@{L}" in df.columns]


def plot_grid_heatmaps(g: pd.DataFrame, out_dir: Path, L: int):
    """Per (layer, position): mean KL over neurons, interventional corr, top-1 agreement."""
    col = f"kl@{L}"
    if col not in g.columns:
        return
    cells = (g.groupby(["layer", "position"])
               .agg(kl=(col, "mean"), kl_std=(col, "std"), n=(col, "count"),
                    interv=("interventional_corr", "mean"), top1=(f"top1@{L}", "mean"))
               .reset_index())
    cells.to_csv(out_dir / f"grid_cells_L{L}.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (val, title, cmap, vmin, vmax) in zip(axes, (
            ("kl", f"KL(base‖tuned) nats/token @{L}", "magma", 0, None),
            ("interv", "interventional corr", "viridis", 0, 1),
            ("top1", f"top-1 agreement @{L}", "viridis", 0.5, 1))):
        piv = cells.pivot(index="layer", columns="position", values=val)
        sns.heatmap(piv, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, ax=ax,
                    linewidths=0.5, linecolor="white", annot_kws={"size": 8})
        ax.set_title(title); ax.set_xlabel("position"); ax.set_ylabel("layer")
    fig.tight_layout()
    fig.savefig(out_dir / f"grid_kl_heatmaps_L{L}.png", dpi=150)
    plt.close(fig)
    return cells


def plot_grid_kl_vs_fidelity(g: pd.DataFrame, out_dir: Path, L: int):
    """Does spending KL buy fidelity? Endogenous left, interventional right, one point
    per run, coloured by layer, marker size by distance from the end of the text."""
    col = f"kl@{L}"
    if col not in g.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    size = 18 + 3 * g["position"].abs()
    for ax, ycol, ylab in ((axes[0], "endogenous_corr", "endogenous corr"),
                           (axes[1], "interventional_corr", "interventional corr")):
        sc = ax.scatter(g[col], g[ycol], c=g["layer"], cmap="viridis", s=size,
                        edgecolors="k", linewidths=0.4, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel(f"KL(base‖tuned) nats/token @{L}")
        ax.set_ylabel(ylab)
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02); cb.set_label("layer")
    axes[1].text(0.98, 0.02, "marker size = |position| (distance from the end)",
                 transform=axes[1].transAxes, ha="right", fontsize=9, color="dimgrey")
    fig.savefig(out_dir / f"grid_kl_vs_fidelity_L{L}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multi_locale(m: pd.DataFrame, ref: pd.DataFrame, out_dir: Path, L: int):
    """KL against number of locales, one panel per family, with the single-locale grid
    cell (same layer, same position) as the reference bar at n=1."""
    col = f"kl@{L}"
    if col not in m.columns and (ref is None or col not in ref.columns):
        return
    fams = [f for f in ("length", "neuron", "layer") if f in set(m["family"])]
    if not fams:
        return
    fig, axes = plt.subplots(1, len(fams), figsize=(6.5 * len(fams), 6), sharey=True, squeeze=False)
    ref_kl = float(ref[col].mean()) if ref is not None and col in ref.columns and len(ref) else np.nan
    ref_sd = float(ref[col].std()) if ref is not None and col in ref.columns and len(ref) > 1 else 0.0
    # reference cell as an absolute 0-indexed position (-1 -> 31 in the 32-token window)
    ref_t = REF_CELL["position"] + L_TRAIN if REF_CELL["position"] < 0 else REF_CELL["position"]
    for i, (ax, fam) in enumerate(zip(axes[0], fams)):
        q = m[m["family"] == fam]
        agg = (q.groupby("n_locales")[col].agg(["mean", "std", "count"]).reset_index()
                 .sort_values("n_locales"))
        ax.errorbar(agg["n_locales"], agg["mean"], yerr=agg["std"].fillna(0), marker="o", ms=9,
                    capsize=3, lw=1.8, color="#d62728", label="Multi-Locale")
        for _, r in agg.iterrows():
            ax.annotate(f"n={int(r['count'])}", (r["n_locales"], r["mean"]),
                        textcoords="offset points", xytext=(6, -16), fontsize=13, color="grey")
        if not np.isnan(ref_kl):
            ax.axhline(ref_kl, color="#1f77b4", lw=1.6, ls="--",
                       label=f"Single Locale (L{REF_CELL['layer']}, t={ref_t}, n={len(ref)})")
            ax.axhspan(ref_kl - ref_sd, ref_kl + ref_sd, color="#1f77b4", alpha=0.12)
        ax.set_xscale("log", base=2)
        ax.set_xlabel({"length": "Positions Trained (Window = Length)",
                       "neuron": "Neurons Trained", "layer": "Layers Trained"}[fam], fontsize=22)
        # the title names the sweep; the curve itself is the same thing in every panel
        ax.set_title(f"{fam.capitalize()} Sweep", fontsize=24)
        ax.tick_params(labelsize=20)
        if i == 0:                      # identical in every panel, so drawn once
            ax.legend(fontsize=16, loc="upper left")
    axes[0][0].set_ylabel(f"KL(Base \u2016 Tuned) @ L={L}\n(nats/token)", fontsize=22)
    fig.tight_layout()
    fig.savefig(out_dir / f"multi_locale_kl_L{L}.png", dpi=150)
    plt.close(fig)


def plot_length_profile(m: pd.DataFrame, ref: pd.DataFrame, out_dir: Path):
    """KL at each eval length, per run label. The anchor is applied at the training
    length, so this shows whether longer contexts drift more."""
    cols = kl_cols(m)
    if len(cols) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = sorted(m["label"].unique()) if "label" in m else []
    pal = dict(zip(labels, sns.color_palette("husl", max(1, len(labels)))))
    for lab in labels:
        q = m[m["label"] == lab]
        ys = [q[c].mean() for c in cols]
        ax.plot([int(c.split("@")[1]) for c in cols], ys, marker="o", lw=1.5, color=pal[lab], label=lab)
    if ref is not None and len(ref):
        ax.plot([int(c.split("@")[1]) for c in cols], [ref[c].mean() for c in cols],
                marker="s", lw=2.2, color="k", ls="--", label="single locale (ref cell)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("eval length (tokens)"); ax.set_ylabel("KL(base‖tuned) nats/token")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "multi_locale_kl_by_length.png", dpi=150)
    plt.close(fig)


def main(out_dir=None, grid_group=GRID_GROUP, active_family=None):
    out_dir = Path(out_dir or ROOT / "outputs" / "plots" / "kl_capacity")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- single locale: the unanchored grid ----
    g = grid_runs(grid_group, gate="off")
    if not g.empty:
        g = g[(g["w"] == 0.01) & (g["kl_tag"] == "none")]
    have_g = not g.empty and g["has_kl"].any()
    if have_g:
        gk = g[g["has_kl"]]
        print(f"[kl] grid (unanchored): {len(gk)} runs with KL of {len(g)}")
        gk.to_csv(out_dir / "grid_runs.csv", index=False)
        for L in LENGTHS:
            plot_grid_heatmaps(gk, out_dir, L)
            plot_grid_kl_vs_fidelity(gk, out_dir, L)
        ref = gk[(gk["layer"] == REF_CELL["layer"]) & (gk["position"] == REF_CELL["position"])]
    else:
        print(f"[kl] no unanchored grid runs with lm_degradation.json under {grid_group}")
        ref = None

    # ---- multi locale ----
    m = multi_locale_runs()
    if m.empty:
        print("[kl] no multi-locale runs found (mlp_lengen / mlp_neurongen / mlp_layergen)")
    else:
        m.to_csv(out_dir / "multi_locale_runs.csv", index=False)
        have = m[m["has_kl"].fillna(False)]
        print(f"[kl] multi-locale: {len(m)} selected runs, {len(have)} with KL")
        cmds = missing_kl_commands(m)
        if cmds:
            lines = ["#!/bin/bash", "set -e",
                     "# runs selected for the KL-capacity comparison that have no lm_degradation.json",
                     "# families other than the active one are left in but commented out"]
            n_active = 0
            for fam in ("neuron", "length", "layer"):
                fam_cmds = [c for f, c in cmds if f == fam]
                if not fam_cmds:
                    continue
                on = (active_family is None) or (fam == active_family)
                lines.append(f"\n# ---- {fam} ({len(fam_cmds)} runs){'' if on else '  [inactive]'} ----")
                lines += [c if on else f"# {c}" for c in fam_cmds]
                n_active += len(fam_cmds) if on else 0
            (out_dir / "score_kl_missing.sh").write_text("\n".join(lines) + "\n")
            print(f"[kl] {len(cmds)} selected runs have no lm_degradation.json ({n_active} active); "
                  f"commands written to {out_dir / 'score_kl_missing.sh'}")
        if len(have):
            for L in LENGTHS:
                plot_multi_locale(have, ref, out_dir, L)
            plot_length_profile(have, ref, out_dir)
            cols = ["family", "label", "n_locales"] + kl_cols(have) + \
                   [c for c in ("interventional_corr", "endogenous_corr_macro") if c in have.columns]
            print("\n[kl] multi-locale KL (mean over runs of each label):")
            print(have.groupby(["family", "label", "n_locales"])[[c for c in cols if c not in
                  ("family", "label", "n_locales")]].mean().round(3).to_string())
    print(f"\n[kl] wrote {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default=None)
    ap.add_argument("--grid-group", default=GRID_GROUP)
    ap.add_argument("--active-family", default=None, choices=["neuron", "length", "layer"],
                    help="In score_kl_missing.sh, leave only this family's commands uncommented.")
    a = ap.parse_args()
    main(a.out_dir, grid_group=a.grid_group, active_family=a.active_family)
