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
model whose paired table was rebuilt after score_rank_metrics.py ran; Pearson is always
present and is what the figures default to.
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

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
                     "legend.fontsize": 11})

# display name -> (plot-dir suffix, n_layers, params in billions)
MODELS = {
    "Llama-3.2-1B": ("", 16, 1.24),
    "Llama-3.2-3B": ("_llama323b", 28, 3.21),
    "Llama-3.1-8B": ("_llama318b", 32, 8.03),
}
# joint arm label -> paired-file stem
JOINT_ARMS = {"joint_kl0.1": "joint_kl0.1_vs_sft", "joint_free": "joint_free_vs_sft"}
GATES = ("gateoff", "gateon")
READOUTS = ("endogenous", "interventional")
METRICS = ("corr", "spearman", "kendall_tau", "monotone_r2")
# the depths 1B and 8B both trained; 3B shares none of them
SHARED_DEPTHS = (0.125, 0.375, 0.625, 0.875)
ARM_ORDER = ["sft_gateon", "sft_gateoff", "joint_kl0.1", "joint_free"]


def load_all(models=MODELS, common_positions=None) -> pd.DataFrame:
    """Long-form: one row per (model, arm, readout, metric, cell, neuron)."""
    rows, missing = [], []
    for name, (suffix, n_layers, params) in models.items():
        for joint, stem in JOINT_ARMS.items():
            for gate in GATES:
                f = ROOT / f"outputs/plots/sft_vs_joint{suffix}" / f"paired_runs_{stem}_{gate}.csv"
                if not f.exists():
                    missing.append(str(f.relative_to(ROOT)))
                    continue
                t = pd.read_csv(f)
                # both arms live in one file, suffixed by their label
                for arm in (joint, f"sft_{gate}"):
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
           .agg(n=("value", "size"), depths=("depth", lambda s: sorted(s.unique())))
           .reset_index())
    return cov


def _agg(df, by):
    g = df.groupby(by)["value"]
    out = g.agg(mean="mean", std="std", n="count").reset_index()
    out["se"] = out["std"] / np.sqrt(out["n"].clip(lower=1))
    return out


def plot_depth_profile(df: pd.DataFrame, out_dir: Path, metric="corr"):
    """The primary figure: metric against RELATIVE DEPTH, one line per model.

    No cross-depth aggregation at all, so it carries no choice about which depths to
    average -- the reader sees the profile each model actually has, and whether the
    shape is preserved as scale grows.
    """
    d = df[df["metric"] == metric]
    if d.empty:
        return
    arms = [a for a in ARM_ORDER if a in set(d["arm"])]
    fig, axes = plt.subplots(len(READOUTS), len(arms),
                             figsize=(4.3 * len(arms), 3.9 * len(READOUTS)),
                             squeeze=False, sharex=True, sharey="row")
    colors = dict(zip(MODELS, sns.color_palette("viridis", len(MODELS))))
    for r, readout in enumerate(READOUTS):
        for c, arm in enumerate(arms):
            ax = axes[r][c]
            for model in MODELS:
                q = _agg(d[(d["readout"] == readout) & (d["arm"] == arm)
                           & (d["model"] == model)], ["depth"]).sort_values("depth")
                if q.empty:
                    continue
                ax.errorbar(q["depth"], q["mean"], yerr=q["se"], marker="o", ms=6,
                            capsize=3, lw=1.8, color=colors[model], label=model)
            ax.axhline(0, color="k", lw=0.8, alpha=0.4)
            ax.set_ylim(-0.1, 1.02)
            if r == 0:
                ax.set_title(arm)
            if r == len(READOUTS) - 1:
                ax.set_xlabel("relative depth (layer / n_layers)")
            if c == 0:
                ax.set_ylabel(f"{readout}\n{metric}")
    axes[0][-1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"depth_profile_{metric}.png", dpi=150)
    plt.close(fig)


def plot_scaling(df: pd.DataFrame, out_dir: Path, metric="corr", shared_depths_only=True):
    """Metric against parameter count, one line per arm.

    shared_depths_only restricts to the depths 1B and 8B both trained, so the trend is
    not an artefact of which depths each model sampled. 3B shares none of them and is
    drawn as an open marker at ITS OWN depths -- comparable in spirit, not matched.
    """
    d = df[df["metric"] == metric]
    if d.empty:
        return
    matched = d[d["depth"].isin(SHARED_DEPTHS)] if shared_depths_only else d
    fig, axes = plt.subplots(1, len(READOUTS), figsize=(6.4 * len(READOUTS), 5), sharey=True)
    arms = [a for a in ARM_ORDER if a in set(d["arm"])]
    colors = dict(zip(arms, sns.color_palette("Set1", len(arms))))
    for ax, readout in zip(axes, READOUTS):
        for arm in arms:
            q = _agg(matched[(matched["readout"] == readout) & (matched["arm"] == arm)],
                     ["model", "params_b"]).sort_values("params_b")
            if q.empty:
                continue
            exact = q["model"].isin([m for m in MODELS if any(
                round(l / MODELS[m][1], 3) in SHARED_DEPTHS for l in range(MODELS[m][1]))])
            ax.errorbar(q["params_b"], q["mean"], yerr=q["se"], marker="o", ms=7,
                        capsize=3, lw=1.8, color=colors[arm], label=arm)
            # 3B does not sit on the shared depths: hollow it so the caveat is visible
            off = q[~q["model"].isin(["Llama-3.2-1B", "Llama-3.1-8B"])]
            if len(off):
                ax.scatter(off["params_b"], off["mean"], s=120, facecolors="white",
                           edgecolors=colors[arm], linewidths=2, zorder=5)
        ax.set_xscale("log")
        ax.set_xticks([m[2] for m in MODELS.values()])
        ax.set_xticklabels([f"{m[2]:g}B" for m in MODELS.values()])
        ax.axhline(0, color="k", lw=0.8, alpha=0.4)
        ax.set_xlabel("parameters")
        ax.set_title(readout)
    axes[0].set_ylabel(metric)
    axes[-1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"scaling_{metric}.png", dpi=150)
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
    matched = w[w["depth"].isin(SHARED_DEPTHS)]
    tbl = (matched.groupby(["readout", "model", "params_b"])["delta"]
           .agg(mean="mean", std="std", n="count").reset_index())
    tbl["se"] = tbl["std"] / np.sqrt(tbl["n"].clip(lower=1))
    tbl.to_csv(out_dir / f"joint_minus_sft_{metric}.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = dict(zip(READOUTS, sns.color_palette("Set2", len(READOUTS))))
    for readout in READOUTS:
        q = tbl[tbl["readout"] == readout].sort_values("params_b")
        if q.empty:
            continue
        ax.errorbar(q["params_b"], q["mean"], yerr=q["se"], marker="o", ms=7, capsize=3,
                    lw=1.8, color=colors[readout], label=readout)
    ax.axhline(0, color="k", lw=0.9, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xticks([m[2] for m in MODELS.values()])
    ax.set_xticklabels([f"{m[2]:g}B" for m in MODELS.values()])
    ax.set_xlabel("parameters")
    ax.set_ylabel(f"{joint} - {base}  ({metric})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"joint_minus_sft_{metric}.png", dpi=150)
    plt.close(fig)
    return tbl


def report(df: pd.DataFrame, metric="corr"):
    d = df[df["metric"] == metric]
    print(f"\n[scale] ===== {metric} =====")
    print("\nAll depths each model trained (NOT matched -- shown for completeness):")
    a = _agg(d, ["readout", "arm", "model"])
    print(a.pivot_table(index=["readout", "arm"], columns="model", values="mean")
           .round(3).to_string())
    m = d[d["depth"].isin(SHARED_DEPTHS)]
    print(f"\nDepth-MATCHED (depths {list(SHARED_DEPTHS)}; 1B and 8B only, 3B shares none):")
    b = _agg(m, ["readout", "arm", "model"])
    piv = b.pivot_table(index=["readout", "arm"], columns="model", values="mean").round(3)
    sepiv = b.pivot_table(index=["readout", "arm"], columns="model", values="se").round(3)
    cols = [c for c in ("Llama-3.2-1B", "Llama-3.1-8B") if c in piv.columns]
    if len(cols) == 2:
        piv = piv[cols]
        piv["delta_8B_minus_1B"] = (piv[cols[1]] - piv[cols[0]]).round(3)
        piv["se_1B"], piv["se_8B"] = sepiv[cols[0]].round(3), sepiv[cols[1]].round(3)
    print(piv.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "outputs" / "plots" / "model_scaling")
    ap.add_argument("--metrics", type=str, default=None,
                    help="Comma list; default: every metric present in the tables.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all()
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
        plot_scaling(df, args.out_dir, metric)
        joint_minus_sft(df, args.out_dir, metric)
    skipped = [m for m in METRICS if m not in present]
    if skipped:
        print(f"\n[scale] not in the paired tables yet: {skipped} "
              f"(run score_rank_metrics.py --update-metrics-json, then rebuild them)")
    print(f"\n[scale] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
