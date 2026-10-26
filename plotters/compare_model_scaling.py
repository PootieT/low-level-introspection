#!/usr/bin/env python3
"""
Does introspectability change with model scale?

Aggregates the SFT and joint-loss layer x position sweeps across Llama-3.2-1B,
Llama-3.2-3B and Llama-3.1-8B. Reads the paired tables compare_sft_vs_joint.py writes
per model (outputs/plots/sft_vs_joint[_<model>]/paired_runs_*.csv), so it needs no run
directories and no weights.

THE COMPARISON IS NOT FREE -- the three sweeps do not share a grid:

  model   n_layers   layers trained        -> relative depth        positions
  1B      16         0,2,4,...,14             0.000 .. 0.875        8 positions
  3B      28         6,12,18,24               0.214 .. 0.857        4 positions
  8B      32         4,12,20,28               0.125 .. 0.875        4 positions

So:
  * LAYER INDEX IS NOT COMPARABLE. Layer 4 is a quarter of the way through the 1B model
    and an eighth of the way through the 8B. Everything here is keyed on RELATIVE DEPTH
    (layer / n_layers), which is the only coordinate the three share.
  * A mean over each model's own grid mixes the scale effect with which depths that
    model happened to sample -- the 1B mean includes depth 0.0, where introspection is
    easiest, and the 8B mean does not. 1B and 8B land on exactly the same four depths
    (0.125/0.375/0.625/0.875), so those give a clean matched comparison; 3B's four
    depths line up with neither, and it is reported on its own depths only.
  * The POSITIONS are intersected to the four all three sweeps ran (-1,-8,-20,-28).
  * The NEURONS differ between models by construction (different intermediate sizes, so
    the per-layer sampler draws different indices). This compares distributions of
    randomly chosen neurons at matched depths, never the same neuron.

Rank metrics (spearman / kendall_tau / monotone_r2) are picked up automatically for any
model whose runs were evaluated after the default eval started writing them (2026-10-12);
Pearson is always present and is what the figures default to.

MODEL SETS (--set). "base" is the three Llama grids above, all at the 1B-tuned LR 5e-5.
"hyper_tuned" is every model at the LR its own sweep chose (plotters/plot_lr_sweep.py),
one LR shared by the joint (KL 0.1) and SFT (gate off) arms -- the *_hyper_tuned grids,
which only run those two arms -- plus Llama-3.2-1B's original grid, tuned at 1B. Those
grids sit on depths 1/8, 3/8, 5/8, 7/8 by construction (layer = round(depth x n_layers),
halves up), so a model's layer is matched to the nearest shared depth within DEPTH_TOL
(at most 0.018 off for every hyper-tuned grid). The base 3B layers are up to 0.09 off,
so only two of them match.

DOES THE JOINT LOSS'S ADVANTAGE OVER SFT SHRINK WITH SCALE? (scale_trend; fixed on
2026-10-23, before the hyper-tuned grids had results.) The LR sweeps hinted at it -- in
Qwen the paired joint - SFT interventional gain fell with size, while joint stayed flat
and SFT's own interventional corr rose -- but on ~6 neurons per model. The test:
  1. per neuron, gain = joint_kl0.1 - sft_gateoff, PAIRED (both arms train the same
     neurons); interventional corr is the primary readout, endogenous secondary;
  2. per model, the mean over the depth-matched (depth, position) cells that EVERY model
     has, cells weighted equally -- so no model's mean mixes in depths others lack;
  3. OLS across models: mean ~ log10(params) + family (Llama / Qwen);
  4. two uncertainties, reported side by side: the OLS standard error (scatter BETWEEN
     models, the one that decides a trend) and a bootstrap over neurons resampled within
     cells (noise WITHIN each model's mean);
  5. the same fit on joint alone and on SFT alone. The sweep's reading predicts the
     slope is in SFT (rising) rather than joint (flat).
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from compare_sft_vs_joint import pretty_arm  # noqa: E402
from plot_single_neuron_results import SEQ_LEN  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 17, "axes.titlesize": 22, "axes.labelsize": 20,
                     "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 14})

# display name -> (plot-dir suffix, n_layers, params in billions)
MODEL_SETS = {
    "base": {
        "Llama-3.2-1B": ("", 16, 1.24),
        "Llama-3.2-3B": ("_llama323b", 28, 3.21),
        "Llama-3.1-8B": ("_llama318b", 32, 8.03),
    },
    "hyper_tuned": {
        "Qwen3-0.6B": ("_qwen3_06b_hyper_tuned", 28, 0.60),
        "Llama-3.2-1B": ("", 16, 1.24),   # its 5e-5 grid IS the 1B-tuned one
        "Qwen3-1.7B": ("_qwen3_17b_hyper_tuned", 28, 1.72),
        "Llama-3.2-3B": ("_llama323b_hyper_tuned", 28, 3.21),
        "Qwen3-4B": ("_qwen3_4b_hyper_tuned", 36, 4.02),
        "Llama-3.1-8B": ("_llama318b_hyper_tuned", 32, 8.03),
        "Qwen3-8B": ("_qwen3_8b_hyper_tuned", 36, 8.19),
        "Qwen3-14B": ("_qwen3_14b_hyper_tuned", 40, 14.8),
    },
}
# arms each set compares; scale_trend always compares joint_kl0.1 with sft_gateoff.
# SFT gate on (the bypass baseline, run for the Llama hyper-tuned grids only) is left
# out of every figure and aggregate: GATES below never loads its paired files.
SET_ARMS = {"base": None, "hyper_tuned": ("joint_kl0.1", "sft_gateoff")}
MODELS = MODEL_SETS["base"]      # main() rebinds this to the chosen set
# joint arm label -> paired-file stem
JOINT_ARMS = {"joint_kl0.1": "joint_kl0.1_vs_sft", "joint_free": "joint_free_vs_sft"}
GATES = ("gateoff",)
READOUTS = ("endogenous", "interventional")
METRICS = ("corr", "spearman", "kendall_tau", "monotone_r2")
# the depths 1B and 8B both trained, and every *_hyper_tuned grid's design depths
SHARED_DEPTHS = (0.125, 0.375, 0.625, 0.875)
# a layer counts as sitting on a shared depth when it is this close (relative depth)
DEPTH_TOL = 0.03
N_BOOT = 2000
ARM_ORDER = ["sft_gateoff", "joint_kl0.1", "joint_free"]


def family(model: str) -> str:
    return model.split("-")[0]


def depth_bin(depth: float):
    """The shared depth this layer stands for, or NaN if none is within DEPTH_TOL."""
    near = min(SHARED_DEPTHS, key=lambda d: abs(d - depth))
    return near if abs(near - depth) <= DEPTH_TOL else np.nan


def load_all(models=None, common_positions=None, arms=None) -> pd.DataFrame:
    """Long-form: one row per (model, arm, readout, metric, cell, neuron)."""
    models = MODELS if models is None else models
    rows, missing = [], []
    for name, (suffix, n_layers, params) in models.items():
        for joint, stem in JOINT_ARMS.items():
            for gate in GATES:
                # a set that compares only some arms neither needs nor misses the others
                if arms is not None and not {joint, f"sft_{gate}"} <= set(arms):
                    continue
                f = ROOT / f"outputs/plots/sft_vs_joint{suffix}" / f"paired_runs_{stem}_{gate}.csv"
                if not f.exists():
                    missing.append(str(f.relative_to(ROOT)))
                    continue
                t = pd.read_csv(f)
                # both arms live in one file, suffixed by their label
                for arm in (joint, f"sft_{gate}"):
                    if arms is not None and arm not in arms:
                        continue
                    for readout in READOUTS:
                        for metric in METRICS:
                            col = f"{readout}_{metric}_{arm}"
                            if col not in t.columns:
                                continue
                            sub = t[["layer", "position", "neuron", col]].dropna(subset=[col])
                            for _, r in sub.iterrows():
                                rows.append({
                                    "model": name, "params_b": params, "n_layers": n_layers,
                                    "arm": arm, "readout": readout, "metric": metric,
                                    "layer": int(r["layer"]),
                                    "depth": round(int(r["layer"]) / n_layers, 3),
                                    "position": int(r["position"]), "neuron": r["neuron"],
                                    "value": float(r[col]),
                                })
    if missing:
        print(f"[scale] {len(missing)} paired file(s) absent, e.g. {missing[:2]}")
    if not rows:
        sys.exit("[scale] no paired_runs_*.csv found. Run compare_sft_vs_joint.py per model first.")
    df = pd.DataFrame(rows).drop_duplicates(
        ["model", "arm", "readout", "metric", "layer", "position", "neuron"])
    df["depth_bin"] = df["depth"].map(depth_bin)
    df["family"] = df["model"].map(family)
    # the same arm appears in both joint files (sft_gateoff is paired against each joint
    # arm), so de-duplication above is load-bearing, not hygiene
    if common_positions is None:
        common_positions = sorted(set.intersection(
            *[set(g["position"]) for _, g in df.groupby("model")]))
    before = len(df)
    df = df[df["position"].isin(common_positions)]
    print(f"[scale] positions intersected to {common_positions}: "
          f"{before} -> {len(df)} rows")
    return df


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    cov = (df[df["metric"] == "corr"]
           .groupby(["model", "arm", "readout"])
           .agg(n=("value", "size"), depths=("depth", lambda s: sorted(s.unique())),
                matched_depths=("depth_bin", lambda s: sorted(float(x) for x in s.dropna().unique())))
           .reset_index())
    return cov


def _agg(df, by):
    g = df.groupby(by)["value"]
    out = g.agg(mean="mean", std="std", n="count").reset_index()
    out["se"] = out["std"] / np.sqrt(out["n"].clip(lower=1))
    return out


FAMILY_MARKER = {"Llama": "o", "Qwen3": "^"}
FAMILY_COLOR = {"Llama": "#2471a3", "Qwen3": "#c0392b"}
READOUT_COLOR = {"endogenous": "#1b9e77", "interventional": "#d95f02"}
METRIC_NAME = {"corr": "Correlation", "spearman": "Spearman \u03c1",
               "kendall_tau": "Kendall \u03c4", "monotone_r2": "Monotone R\u00b2"}


def model_styles(models):
    """Colour and marker per model: Llama in blues, Qwen in reds (darker = larger), Qwen
    drawn with triangles so the families stay apart in greyscale too."""
    out = {}
    for fam, cmap in (("Llama", "Blues"), ("Qwen3", "Reds")):
        ms = sorted((m for m in models if family(m) == fam), key=lambda m: MODELS[m][2])
        for m, c in zip(ms, sns.color_palette(cmap, len(ms) + 2)[2:]):
            out[m] = dict(color=c, marker=FAMILY_MARKER[fam])
    for m in models:
        out.setdefault(m, dict(color="grey", marker="s"))
    return out


def _family_handles(families, color="k"):
    from matplotlib.lines import Line2D
    return [Line2D([], [], ls="", marker=FAMILY_MARKER.get(f, "s"), ms=10, color=color,
                   label=f) for f in families]


def _size_ticks(ax, params):
    """Round size ticks on the log axis. With three models each size gets its own tick;
    with more, sizes crowd (1.24 / 1.72, 3.21 / 4.02, 8.03 / 8.19), so round ticks are
    used and the points carry the model names instead."""
    params = sorted(params)
    if len(params) <= 3:
        ticks = params
    else:
        ticks = [t for t in (0.5, 1, 2, 4, 8, 16, 32, 70)
                 if params[0] / 1.3 <= t <= params[-1] * 1.3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}B" for t in ticks])
    ax.minorticks_off()


def plot_depth_profile(df: pd.DataFrame, out_dir: Path, metric="corr"):
    """The primary figure: metric against RELATIVE DEPTH, one line per model.

    No cross-depth aggregation at all, so it carries no choice about which depths to
    average -- the reader sees the profile each model actually has, and whether the
    shape is preserved as scale grows.
    """
    _profile(df[df["metric"] == metric], "depth", out_dir / f"depth_profile_{metric}.png",
             metric, xlabel="Relative Depth (Layer / # Layers)")


def _profile(d, xcol, path, metric, xlabel, xticks=None, xmap=None):
    """Shared body of the depth and position profiles: readouts x arms, one line per
    model (Llama blues / circles, Qwen reds / triangles), mean +- SE over neurons."""
    if d.empty:
        return
    arms = [a for a in ARM_ORDER if a in set(d["arm"])]
    # family-major order, so the two-column key reads Llama | Qwen
    models = sorted((m for m in MODELS if m in set(d["model"])),
                    key=lambda m: (family(m), MODELS[m][2]))
    style = model_styles(models)
    fig, axes = plt.subplots(len(READOUTS), len(arms),
                             figsize=(6.4 * len(arms), 5.4 * len(READOUTS)),
                             squeeze=False, sharex=True, sharey="row")
    for r, readout in enumerate(READOUTS):
        lowest = -0.1     # extended below when a mean - SE falls under it
        for c, arm in enumerate(arms):
            ax = axes[r][c]
            for model in models:
                q = _agg(d[(d["readout"] == readout) & (d["arm"] == arm)
                           & (d["model"] == model)], [xcol]).sort_values(xcol)
                if q.empty:
                    continue
                lowest = min(lowest, (q["mean"] - q["se"].fillna(0)).min() - 0.06)
                x = q[xcol].map(xmap) if xmap else q[xcol]
                ax.errorbar(x, q["mean"], yerr=q["se"], marker=style[model]["marker"],
                            ms=9, capsize=3, lw=2, color=style[model]["color"], label=model)
            ax.axhline(0, color="k", lw=0.8, alpha=0.4)
            ax.grid(alpha=0.25)
            if xticks is not None:
                ax.set_xticks(xticks)
            if r == 0:
                ax.set_title(pretty_arm(arm))
            if r == len(READOUTS) - 1:
                ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel(f"{readout.capitalize()} {METRIC_NAME.get(metric, metric)}\n"
                              f"(Mean \u00b1 SE)")
        axes[r][0].set_ylim(lowest, 1.02)   # sharey="row": one call sets the row
    # the endogenous panels sit high (~0.6-1), so their lower half is free for the key
    axes[0][0].legend(loc="lower left", ncol=2, fontsize=13, frameon=False,
                      handlelength=1.6, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_position_profile(df: pd.DataFrame, out_dir: Path, metric="corr"):
    """The depth profile's companion: metric against TOKEN POSITION, one line per model,
    each point averaged over depths.

    The average runs over the depth-matched layers only (within DEPTH_TOL of the shared
    depths), so every model averages over the same depths and the lines differ by model,
    not by which depths each model happened to train. All *_hyper_tuned grids have all
    four; a model with fewer (the base 3B grid has two) is named in the panel note and
    printed (no longer in a figure title). Error bars are the SE over the pooled neurons
    at that position.
    """
    d = df[(df["metric"] == metric) & df["depth_bin"].notna()]
    if d.empty:
        return
    arms = [a for a in ARM_ORDER if a in set(d["arm"])]
    n_bins = d.groupby("model")["depth_bin"].nunique()
    partial = {m: int(n) for m, n in n_bins.items() if n < len(SHARED_DEPTHS)}
    if partial:
        print(f"[scale] position profile: averaged over fewer than {len(SHARED_DEPTHS)} "
              f"matched depths for {partial}")
    # absolute token index (t = SEQ_LEN + position, last token = SEQ_LEN - 1), as in the
    # per-model heatmaps
    to_abs = lambda p: SEQ_LEN + int(p)  # noqa: E731
    _profile(d, "position", out_dir / f"position_profile_{metric}.png", metric,
             xlabel="Token Position", xmap=to_abs,
             xticks=sorted(to_abs(p) for p in d["position"].unique()))


def plot_scaling(df: pd.DataFrame, out_dir: Path, metric="corr", shared_depths_only=True):
    """Metric against parameter count, one line per arm.

    shared_depths_only restricts to the depths 1B and 8B both trained, so the trend is
    not an artefact of which depths each model sampled. A model whose mean covers FEWER
    of the shared depths than the others (the base 3B grid matches two of four) is drawn
    hollow. Layers rounded onto a shared depth (<= 0.018 off in every *_hyper_tuned grid,
    by design) count as on it and are drawn solid.
    """
    d = df[df["metric"] == metric]
    if d.empty:
        return
    matched = d[d["depth_bin"].notna()] if shared_depths_only else d
    # models averaged over fewer shared depths than the best-covered model
    n_bins = matched.groupby("model")["depth_bin"].nunique()
    partial = set(n_bins[n_bins < n_bins.max()].index)
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, len(READOUTS), figsize=(7.4 * len(READOUTS), 5.8), sharey=True)
    arms = [a for a in ARM_ORDER if a in set(d["arm"])]
    colors = dict(zip(arms, ["#2471a3", "#5dade2", "#c0392b", "#e67e22"]
                      if len(arms) == 4 else sns.color_palette("Set1", len(arms))))
    colors.update({"sft_gateoff": "#2471a3", "joint_kl0.1": "#c0392b"})
    for ax, readout in zip(axes, READOUTS):
        for arm in arms:
            q = _agg(matched[(matched["readout"] == readout) & (matched["arm"] == arm)],
                     ["model", "params_b"]).sort_values("params_b")
            if q.empty:
                continue
            q["family"] = q["model"].map(family)
            # lines join models of one family only: across families they would zigzag
            for fam, g in q.groupby("family"):
                mk = FAMILY_MARKER.get(fam, "s")
                ax.errorbar(g["params_b"], g["mean"], yerr=g["se"], marker=mk, ms=10,
                            capsize=3, lw=1.6, color=colors[arm])
                # fewer shared depths than the others: hollow, so the caveat is visible
                off = g[g["model"].isin(partial)]
                if len(off):
                    ax.scatter(off["params_b"], off["mean"], s=130, marker=mk,
                               facecolors="white", edgecolors=colors[arm], linewidths=2,
                               zorder=5)
        ax.set_xscale("log")
        _size_ticks(ax, matched["params_b"].unique())
        ax.axhline(0, color="k", lw=0.8, alpha=0.4)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Parameters")
        ax.set_title(readout.capitalize())
    axes[0].set_ylabel(f"{METRIC_NAME.get(metric, metric)} (Mean \u00b1 SE)")
    handles = [Line2D([], [], color=colors[a], lw=2.5, label=pretty_arm(a)) for a in arms]
    handles += _family_handles(sorted(set(matched["family"])))
    if partial:
        handles.append(Line2D([], [], ls="", marker="o", ms=10, mfc="white", mec="k",
                              mew=2, label="Fewer Shared Depths"))
    # endogenous sits near the top, leaving its lower half empty for the key
    axes[0].legend(handles=handles, fontsize=14, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"scaling_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def joint_minus_sft(df: pd.DataFrame, out_dir: Path, metric="corr",
                    joint="joint_kl0.1", base="sft_gateoff"):
    """Does the joint loss's advantage over SFT hold up with scale?

    Computed per (model, depth, position) cell, since the neurons differ between models
    and cannot be paired across them; within a model the two arms DO share neurons, so
    the cell delta is paired there.
    """
    d = df[(df["metric"] == metric) & (df["arm"].isin([joint, base]))]
    if d.empty:
        return None
    key = ["model", "params_b", "readout", "depth", "position", "neuron"]
    w = d.pivot_table(index=key, columns="arm", values="value", aggfunc="first").reset_index()
    if joint not in w.columns or base not in w.columns:
        return None
    w["delta"] = w[joint] - w[base]
    w = w.dropna(subset=["delta"])
    matched = w[w["depth"].map(depth_bin).notna()]
    tbl = (matched.groupby(["readout", "model", "params_b"])["delta"]
           .agg(mean="mean", std="std", n="count").reset_index())
    tbl["se"] = tbl["std"] / np.sqrt(tbl["n"].clip(lower=1))
    tbl["family"] = tbl["model"].map(family)
    tbl.to_csv(out_dir / f"joint_minus_sft_{metric}.csv", index=False)
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    handles, fit_rows = [], []
    for readout in READOUTS:
        q = tbl[tbl["readout"] == readout].sort_values("params_b").reset_index(drop=True)
        if q.empty:
            continue
        col = READOUT_COLOR[readout]
        # points only, no line through all models: Llama and Qwen alternate in size
        for fam, g in q.groupby("family"):
            ax.errorbar(g["params_b"], g["mean"], yerr=g["se"], ls="",
                        marker=FAMILY_MARKER.get(fam, "s"), ms=11, capsize=3, color=col)
        label = readout.capitalize()
        if len(q) >= 3:
            # same fit as scale_trend: mean ~ log10(params) [+ family], drawn as the pooled
            # slope through each family's own mean (where the OLS line for that family runs)
            x = np.log10(q["params_b"].values)
            fit = _ols(x, q["family"].values, q["mean"].values)
            xs = np.array([x.min(), x.max()])
            for fam, g in q.groupby("family"):
                x0, y0 = np.log10(g["params_b"]).mean(), g["mean"].mean()
                ax.plot(10 ** xs, y0 + fit["slope"] * (xs - x0), ls="--", lw=1.6,
                        color=col, alpha=0.8)
            label += f": {fit['slope']:+.3f}/decade, p={fit['p']:.2f}"
            fit_rows.append({"readout": readout, "n_models": len(q),
                             "slope_per_decade": fit["slope"], "ols_se": fit["se"],
                             "ols_p": fit["p"], "ols_dof": fit["dof"],
                             "family_term": fit["family_term"]})
        handles.append(Line2D([], [], color=col, ls="--", lw=2, label=label))
    if fit_rows:
        fits = pd.DataFrame(fit_rows)
        fits.to_csv(out_dir / f"joint_minus_sft_fits_{metric}.csv", index=False)
        print(f"\n[scale] joint - SFT ({metric}) vs log10 params:")
        print(fits.round(3).to_string(index=False))
    handles += _family_handles(sorted(tbl["family"].unique()))
    ax.axhline(0, color="k", lw=0.9, alpha=0.5)
    ax.set_xscale("log")
    _size_ticks(ax, tbl["params_b"].unique())
    ax.grid(alpha=0.25)
    ax.set_xlabel("Parameters")
    ax.set_ylabel(f"\u0394 {METRIC_NAME.get(metric, metric)} (Joint \u2212 SFT)\n"
                  f"(Mean \u00b1 SE)")
    ax.legend(handles=handles, fontsize=13, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"joint_minus_sft_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return tbl


def _paired_cells(df, metric, readout, joint, base):
    """Per-neuron joint, sft and gain on depth-matched cells, keyed by (model, cell)."""
    d = df[(df["metric"] == metric) & (df["readout"] == readout)
           & df["arm"].isin([joint, base]) & df["depth_bin"].notna()]
    key = ["model", "family", "params_b", "depth_bin", "position", "neuron"]
    w = d.pivot_table(index=key, columns="arm", values="value", aggfunc="first").reset_index()
    if joint not in w.columns or base not in w.columns:
        return None
    w = w.dropna(subset=[joint, base]).rename(columns={joint: "joint", base: "sft"})
    w["gain"] = w["joint"] - w["sft"]
    return w


def _ols(x_logp, fam, y):
    """y ~ a + b log10(params) [+ c Qwen]. Returns slope, its SE and p, and coefficients.
    The family term is used only when each family has 2+ models (otherwise it is
    collinear with the intercept, or eats a whole family)."""
    from scipy import stats
    fams = pd.Series(fam)
    use_fam = fams.nunique() == 2 and fams.value_counts().min() >= 2
    X = [np.ones_like(x_logp), x_logp]
    if use_fam:
        X.append((fams == sorted(fams.unique())[-1]).astype(float).values)
    X = np.column_stack(X)
    dof = len(y) - X.shape[1]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    if dof <= 0:
        return dict(slope=beta[1], se=np.nan, p=np.nan, dof=dof, family_term=use_fam, beta=beta)
    resid = y - X @ beta
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = float(np.sqrt(cov[1, 1]))
    t = beta[1] / se if se > 0 else np.nan
    pval = float(2 * stats.t.sf(abs(t), dof)) if se > 0 else np.nan
    return dict(slope=float(beta[1]), se=se, p=pval, dof=dof, family_term=use_fam, beta=beta)


def scale_trend(df: pd.DataFrame, out_dir: Path, metric="corr",
                joint="joint_kl0.1", base="sft_gateoff", seed=0):
    """The test fixed in the module docstring: does the paired joint - SFT gain (and do
    joint and SFT on their own) move with log model size?"""
    from scipy import stats
    rng = np.random.default_rng(seed)
    model_rows, fit_rows = [], []
    for readout in READOUTS:
        w = _paired_cells(df, metric, readout, joint, base)
        if w is None or w["model"].nunique() < 3:
            continue
        # cells every model has: no model's mean may mix in depths the others lack
        cells_by_model = w.groupby("model").apply(
            lambda g: set(zip(g["depth_bin"], g["position"])), include_groups=False)
        common = set.intersection(*cells_by_model)
        dropped = {m: len(c - common) for m, c in cells_by_model.items() if c - common}
        if not common:
            print(f"[trend] {readout}: no (depth, position) cell is shared by every model")
            continue
        w = w[[c in common for c in zip(w["depth_bin"], w["position"])]]
        models = (w[["model", "family", "params_b"]].drop_duplicates()
                  .sort_values("params_b").reset_index(drop=True))
        x = np.log10(models["params_b"].values)
        for q in ("gain", "joint", "sft"):
            # per model: equal-weight mean over common cells, and N_BOOT resampled means
            # (neurons resampled with replacement WITHIN each cell)
            point, boots = [], []
            for m in models["model"]:
                g = w[w["model"] == m]
                cells = [c[q].values for _, c in g.groupby(["depth_bin", "position"])]
                point.append(np.mean([v.mean() for v in cells]))
                draws = np.mean([v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(axis=1)
                                 for v in cells], axis=0)
                boots.append(draws)
            point, boots = np.array(point), np.array(boots)          # (M,), (M, B)
            fit = _ols(x, models["family"].values, point)
            bslopes = np.array([_ols(x, models["family"].values, boots[:, b])["slope"]
                                for b in range(N_BOOT)])
            rho, rho_p = stats.spearmanr(x, point)
            fit_rows.append({
                "readout": readout, "quantity": q, "n_models": len(models),
                "n_cells": len(common), "slope_per_decade": fit["slope"],
                "ols_se": fit["se"], "ols_p": fit["p"], "ols_dof": fit["dof"],
                "family_term": fit["family_term"],
                "boot_lo": np.percentile(bslopes, 2.5), "boot_hi": np.percentile(bslopes, 97.5),
                "spearman_rho": rho, "spearman_p": rho_p})
            for fam_name, fm in models.groupby("family"):
                # per-family slopes only add information when there is more than one family
                if models["family"].nunique() > 1 and len(fm) >= 3:
                    idx = fm.index.values
                    ff = _ols(x[idx], fm["family"].values, point[idx])
                    fit_rows.append({"readout": readout, "quantity": q,
                                     "n_models": len(fm), "n_cells": len(common),
                                     "slope_per_decade": ff["slope"], "ols_se": ff["se"],
                                     "ols_p": ff["p"], "ols_dof": ff["dof"],
                                     "family_term": f"{fam_name} only"})
            for i, m in models.iterrows():
                model_rows.append({
                    "readout": readout, "quantity": q, "model": m["model"],
                    "family": m["family"], "params_b": m["params_b"], "mean": point[i],
                    "boot_lo": np.percentile(boots[i], 2.5),
                    "boot_hi": np.percentile(boots[i], 97.5),
                    "n_neurons": int((w["model"] == m["model"]).sum()),
                    "n_cells": len(common)})
        if dropped:
            print(f"[trend] {readout}: kept the {len(common)} cells every model has; "
                  f"cells dropped per model: {dropped}")
    if not fit_rows:
        print("[trend] fewer than 3 models with both arms -- no trend to fit")
        return None
    per_model, fits = pd.DataFrame(model_rows), pd.DataFrame(fit_rows)
    per_model.to_csv(out_dir / f"scale_trend_models_{metric}.csv", index=False)
    fits.to_csv(out_dir / f"scale_trend_fits_{metric}.csv", index=False)

    print(f"\n[trend] ===== {joint} vs {base}, {metric}: slope per decade of parameters =====")
    print("        ols_se/ols_p: scatter BETWEEN models (decides the trend);  boot_lo/hi: "
          "95% CI from neuron noise WITHIN models")
    print(fits.round(3).to_string(index=False))
    print(f"\n[trend] per-model means (cells weighted equally; 95% bootstrap CI):")
    order = {m: i for i, m in enumerate(MODELS)}
    tbl = (per_model.assign(_o=per_model["model"].map(order))
           .pivot_table(index=["readout", "_o", "model"], columns="quantity", values="mean")
           .reindex(columns=["joint", "sft", "gain"]).droplevel("_o"))
    print(tbl.round(3).to_string())

    # figure: one panel per quantity, endogenous on top, interventional below
    quantities = [("gain", "Joint \u2212 SFT"), ("joint", pretty_arm(joint)),
                  ("sft", pretty_arm(base))]
    readouts = [r for r in ("endogenous", "interventional") if r in set(per_model["readout"])]
    fig, axes = plt.subplots(len(readouts), 3, figsize=(19, 6.2 * len(readouts)),
                             squeeze=False, sharex=True)
    marker, color = FAMILY_MARKER, FAMILY_COLOR
    for r, readout in enumerate(readouts):
        for c, (q, label) in enumerate(quantities):
            ax = axes[r][c]
            pm = per_model[(per_model["readout"] == readout) & (per_model["quantity"] == q)]
            for fam_name, g in pm.groupby("family"):
                ax.errorbar(g["params_b"], g["mean"],
                            yerr=[g["mean"] - g["boot_lo"], g["boot_hi"] - g["mean"]],
                            fmt=marker.get(fam_name, "o"), ms=11, capsize=3,
                            color=color.get(fam_name, "grey"), label=fam_name)
                # Qwen labels left of the point, Llama right: the families interleave in
                # size, so same-side labels collide (3.2-3B next to 4B). The smallest
                # model's label stays right, clear of the y tick labels.
                for _, row in g.iterrows():
                    left = fam_name == "Qwen3" and row["params_b"] > pm["params_b"].min()
                    ax.annotate(row["model"].split("-", 1)[1], (row["params_b"], row["mean"]),
                                textcoords="offset points", xytext=(-7 if left else 7, 4),
                                ha="right" if left else "left", fontsize=13)
            f = fits[(fits["readout"] == readout) & (fits["quantity"] == q)
                     & fits["family_term"].isin([True, False])]
            if len(f):
                f = f.iloc[0]
                xs = np.log10(np.array([pm["params_b"].min(), pm["params_b"].max()]))
                # draw the pooled slope through each family's own mean
                for fam_name, g in pm.groupby("family"):
                    x0, y0 = np.log10(g["params_b"]).mean(), g["mean"].mean()
                    ax.plot(10 ** xs, y0 + f["slope_per_decade"] * (xs - x0), ls="--",
                            lw=1.2, color=color.get(fam_name, "grey"), alpha=0.7)
                ax.set_title(f"{label}\nSlope {f['slope_per_decade']:+.3f}/decade "
                             f"(p={f['ols_p']:.2f})", fontsize=20)
            if q == "gain":
                ax.axhline(0, color="k", lw=0.8, alpha=0.5)
            ax.set_xscale("log")
            _size_ticks(ax, pm["params_b"].unique())
            # start at the smallest round tick, not matplotlib's padded margin
            ax.set_xlim(left=min(0.5, pm["params_b"].min() / 1.15))
            ax.grid(alpha=0.25)
            if r == len(readouts) - 1:
                ax.set_xlabel("Parameters")
            if c == 0:
                ax.set_ylabel(f"{readout.capitalize()} {METRIC_NAME.get(metric, metric)}\n"
                              f"(Mean, 95% CI)")
    axes[0][0].legend(fontsize=14, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / f"scale_trend_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fits


def report(df: pd.DataFrame, metric="corr"):
    d = df[df["metric"] == metric]
    print(f"\n[scale] ===== {metric} =====")
    print("\nAll depths each model trained (NOT matched -- shown for completeness):")
    a = _agg(d, ["readout", "arm", "model"])
    print(a.pivot_table(index=["readout", "arm"], columns="model", values="mean")
           .round(3).to_string())
    m = d[d["depth_bin"].notna()]
    print(f"\nDepth-MATCHED (layers within {DEPTH_TOL} of depths {list(SHARED_DEPTHS)}):")
    b = _agg(m, ["readout", "arm", "model"])
    piv = b.pivot_table(index=["readout", "arm"], columns="model", values="mean").round(3)
    sepiv = b.pivot_table(index=["readout", "arm"], columns="model", values="se").round(3)
    piv = piv[[c for c in MODELS if c in piv.columns]]
    cols = [c for c in ("Llama-3.2-1B", "Llama-3.1-8B") if c in piv.columns]
    if len(cols) == 2:
        piv["delta_8B_minus_1B"] = (piv[cols[1]] - piv[cols[0]]).round(3)
        piv["se_1B"], piv["se_8B"] = sepiv[cols[0]].round(3), sepiv[cols[1]].round(3)
    print(piv.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="model_set", choices=sorted(MODEL_SETS), default="base",
                    help="base: the three Llama grids at 5e-5. hyper_tuned: every model at "
                         "its own swept LR (joint KL 0.1 and SFT gate off only).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default outputs/plots/model_scaling[_hyper_tuned]")
    ap.add_argument("--metrics", type=str, default=None,
                    help="Comma list; default: every metric present in the tables.")
    args = ap.parse_args()
    global MODELS
    MODELS = MODEL_SETS[args.model_set]
    if args.out_dir is None:
        args.out_dir = ROOT / "outputs" / "plots" / (
            "model_scaling" + ("" if args.model_set == "base" else f"_{args.model_set}"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all(arms=SET_ARMS[args.model_set])
    df.to_csv(args.out_dir / "model_scaling_runs.csv", index=False)
    cov = coverage(df)
    cov.to_csv(args.out_dir / "coverage.csv", index=False)
    print("\n[scale] coverage (Pearson rows):")
    print(cov.to_string(index=False))

    present = [m for m in METRICS if m in set(df["metric"])]
    wanted = args.metrics.split(",") if args.metrics else present
    for metric in [m for m in wanted if m in present]:
        report(df, metric)
        plot_depth_profile(df, args.out_dir, metric)
        plot_position_profile(df, args.out_dir, metric)
        plot_scaling(df, args.out_dir, metric)
        joint_minus_sft(df, args.out_dir, metric)
        scale_trend(df, args.out_dir, metric)
    skipped = [m for m in METRICS if m not in present]
    if skipped:
        print(f"\n[scale] not in the paired tables yet: {skipped} "
              f"(their runs predate the rank metrics in the default eval; re-evaluate "
              f"them, then rebuild the paired tables)")
    print(f"\n[scale] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
