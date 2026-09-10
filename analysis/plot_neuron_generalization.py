#!/usr/bin/env python3
"""
Does the readout carry neurons it was never trained on, and what does adding neurons
cost the ones already there?

Reads the neuron-generalization sweep (runs named neurongen_arm<A|B>_L<layer>_n<N>_r<rows>_...)
and answers three separate questions that the same runs happen to support. Sources,
written by score_neuron_generalization.py into each run directory:

  neuron_generalization_by_signal.csv   per-neuron corr, with is_trained and
                                        activation_std alongside
  neuron_generalization.csv             the per-arm macro summary

THE THREE QUESTIONS
  1 SCALING      seen-neuron fidelity as N grows, at a FIXED total row budget. Falling
                 could mean capacity OR starvation -- which is why (2) exists.
  2 SHARING      arm A at N=k against arm B (1 neuron) at the SAME rows-per-neuron. If
                 they coincide, neurons were trained privately and data does not
                 transfer between them; if A sits well above B, examples for one neuron
                 helped the others, i.e. a shared code.
  3 TRANSFER     held-out neurons. Interventional corr is the one to believe: the
                 injected value is independent per row, so a readout that ignores which
                 neuron the prompt names scores 0 by construction. A positive ENDOGENOUS
                 score on unseen neurons can just mean "unseen neurons look like seen
                 ones, so predict a typical activation for this context".

  plus INTERFERENCE: the neuron sets are nested prefixes, so the SAME neuron appears in
  every cell and can be followed as N grows. That isolates the cost of adding neighbours
  from neuron-to-neuron heterogeneity.

ALWAYS MACRO, NEVER POOLED. Every correlation here is averaged over neurons. A pooled
correlation across neurons with different activation scales counts the between-neuron
variance as signal and is badly inflated at 100 neurons.

WATCH FOR: a neuron that barely fires has undefined, not poor, correlation.
n_undefined_corr is carried through and reported.
"""
import os
import re
import sys
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
    "legend.fontsize": 12, "legend.title_fontsize": 13, "figure.titlesize": 18,
})

ROOT = Path(__file__).parents[1]
ARM_RE = re.compile(r"_arm(?P<arm>[AB])")
N_RE = re.compile(r"_n(?P<n>\d+)")
R_RE = re.compile(r"_r(?P<r>\d+)")
LAYER_RE = re.compile(r"_L(?P<layer>\d+)")
EVAL_TYPES = ("interventional", "endogenous")


def load_sweep(exp_group: str):
    """One frame of per-neuron rows and one of per-arm summaries, across all runs."""
    base = Path(f"{ROOT}/outputs/runs/{exp_group}")
    if not base.exists():
        sys.exit(f"No such experiment group: {base}")
    per_signal, summary, skipped = [], [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        m_arm, m_n, m_r = ARM_RE.search(d.name), N_RE.search(d.name), R_RE.search(d.name)
        if not (m_arm and m_n and m_r):
            continue                       # the shared pool config dir, etc.
        m_l = LAYER_RE.search(d.name)
        n_neurons, n_rows = int(m_n["n"]), int(m_r["r"])
        meta = {
            "run": d.name, "arm": m_arm["arm"], "n_neurons": n_neurons, "n_rows": n_rows,
            "layer": int(m_l["layer"]) if m_l else None,
            # the budget axis that makes arm A and arm B comparable
            "rows_per_neuron": n_rows // max(1, n_neurons),
        }

        def find(fname):
            direct = d / fname
            if direct.exists():
                return direct
            sub = sorted(d.glob(f"diag_*/{fname}"))
            return sub[-1] if sub else None

        f_sig, f_sum = find("neuron_generalization_by_signal.csv"), find("neuron_generalization.csv")
        if f_sig is None:
            skipped.append((d.name, "no neuron_generalization_by_signal.csv"))
            continue
        per_signal.append(pd.read_csv(f_sig).assign(**meta))
        if f_sum is not None:
            summary.append(pd.read_csv(f_sum).assign(**meta))

    if not per_signal:
        sys.exit(f"No neuron-generalization results under {base}. Rsync the runs first.")
    ps = pd.concat(per_signal, ignore_index=True)
    sm = pd.concat(summary, ignore_index=True) if summary else pd.DataFrame()
    if skipped:
        print(f"[neurongen] skipped {len(skipped)} directories:")
        for name, why in skipped[:10]:
            print(f"    {name}: {why}")
    ps["arm_label"] = np.where(ps["is_trained"], "seen", "unseen")
    return ps, sm


def report(ps: pd.DataFrame):
    runs = ps[["run", "arm", "n_neurons", "n_rows", "rows_per_neuron"]].drop_duplicates()
    print(f"\n[neurongen] {len(runs)} runs")
    print(runs.sort_values(["arm", "n_neurons", "n_rows"]).to_string(index=False))
    dead = ps[ps.get("activation_std", pd.Series(1, index=ps.index)).fillna(1) <= 1e-6]
    if len(dead):
        print(f"[neurongen] {dead['signal'].nunique()} neuron(s) have near-zero activation "
              f"variance; their correlations are undefined, not poor, and are excluded "
              f"from every macro average below")
    a = runs[runs["arm"] == "A"]
    if len(a) and a["n_rows"].nunique() > 1:
        print(f"[neurongen] WARNING: arm A rows are not constant across cells "
              f"({sorted(a['n_rows'].unique())}); the fixed-budget comparison assumes they are")


def _macro(ps: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Macro-average over neurons. Undefined correlations are excluded from the mean and
    counted, so a shrinking denominator cannot pass for a stable score."""
    g = ps.groupby(by)["corr"]
    out = g.agg(corr_macro="mean", corr_std="std", n_defined="count").reset_index()
    out["n_signals"] = ps.groupby(by)["corr"].size().values
    out["n_undefined"] = out["n_signals"] - out["n_defined"]
    return out


def plot_scaling(ps: pd.DataFrame, out_dir):
    """Q1+Q3: seen and unseen fidelity as the number of trained neurons grows (arm A)."""
    d = ps[ps["arm"] == "A"]
    if d.empty:
        return
    m = _macro(d, ["eval_type", "arm_label", "n_neurons"])
    m.to_csv(Path(out_dir) / "scaling_by_n.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, et in zip(axes, EVAL_TYPES):
        for lbl, colour, style in (("seen", "#1f77b4", "-"), ("unseen", "#d62728", "--")):
            q = m[(m["eval_type"] == et) & (m["arm_label"] == lbl)].sort_values("n_neurons")
            if q.empty:
                continue
            ax.errorbar(q["n_neurons"], q["corr_macro"], yerr=q["corr_std"].fillna(0),
                        marker="o", ms=6, capsize=3, lw=1.8, ls=style, color=colour,
                        label=f"{lbl} (n={q['n_defined'].iloc[0]} neurons)")
        ax.set_xscale("log", base=10)
        ax.axhline(0, color="k", lw=0.9, alpha=0.4)
        ax.set_xlabel("neurons trained (total rows held fixed)")
        ax.set_title(et)
        ax.legend(fontsize=10)
    axes[0].set_ylabel("macro corr over neurons")
    fig.suptitle("Scaling: fidelity as neurons are added at a fixed row budget")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "scaling_by_n.png", dpi=150)
    plt.close(fig)
    return m


def plot_sharing(ps: pd.DataFrame, out_dir):
    """Q2: arm A against the data-matched null, on rows-per-neuron.

    Arm A at N=k and arm B at the same rows-per-neuron give each neuron the same amount
    of data; they differ only in whether other neurons were trained alongside. Curves
    coinciding means the data did not transfer -- neurons were learned privately.
    """
    d = ps[ps["arm_label"] == "seen"]
    if d.empty or d["arm"].nunique() < 2:
        return
    # Arm A's N=1 cell IS arm B's full-data point (one neuron, the whole budget) and the
    # sweep runs it once. Count it in both series so the two curves are anchored at the
    # same left-hand point instead of arm B appearing to stop short.
    anchor = d[(d["arm"] == "A") & (d["n_neurons"] == 1)]
    if len(anchor):
        d = pd.concat([d, anchor.assign(arm="B")], ignore_index=True)
    m = _macro(d, ["eval_type", "arm", "rows_per_neuron", "n_neurons"])
    m.to_csv(Path(out_dir) / "sharing_arm_a_vs_b.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, et in zip(axes, EVAL_TYPES):
        for arm, colour, style, lab in (("A", "#1f77b4", "-", "arm A (N neurons)"),
                                        ("B", "#7f7f7f", "--", "arm B (1 neuron, matched data)")):
            q = m[(m["eval_type"] == et) & (m["arm"] == arm)].sort_values("rows_per_neuron")
            if q.empty:
                continue
            ax.errorbar(q["rows_per_neuron"], q["corr_macro"], yerr=q["corr_std"].fillna(0),
                        marker="o" if arm == "A" else "s", ms=6, capsize=3, lw=1.8,
                        ls=style, color=colour, label=lab)
            if arm == "A":
                for _, r in q.iterrows():
                    ax.annotate(f"N={int(r['n_neurons'])}",
                                (r["rows_per_neuron"], r["corr_macro"]),
                                textcoords="offset points", xytext=(0, 9),
                                ha="center", fontsize=9, color=colour)
        ax.set_xscale("log", base=10)
        ax.set_xlabel("training rows per neuron")
        ax.set_title(et)
        ax.legend(fontsize=10)
    axes[0].set_ylabel("macro corr, trained neurons")
    # Arm B is ONE neuron per point, so it has no spread to show -- its wiggle is a single
    # run's noise, not a measured error bar. Arm A's bars are across its N neurons. Saying
    # so on the figure keeps the two from being read as comparable uncertainty.
    axes[0].text(0.02, 0.02, "arm B = 1 neuron per point (no spread to show)",
                 transform=axes[0].transAxes, fontsize=9, color="dimgrey")
    fig.suptitle("Sharing: does training other neurons help, at matched data per neuron?\n"
                 "curves coinciding = no transfer between neurons")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "sharing_arm_a_vs_b.png", dpi=150)
    plt.close(fig)
    return m


def plot_interference(ps: pd.DataFrame, out_dir, n_track: int = 6):
    """Does adding neurons damage the ones already there?

    The training sets are nested prefixes, so a neuron trained at N=1 is also trained at
    every larger N. Following those specific neurons separates the cost of adding
    neighbours from neuron-to-neuron heterogeneity, which the macro average blends in.
    """
    d = ps[(ps["arm"] == "A") & ps["is_trained"]]
    if d.empty:
        return
    counts = d.groupby("signal")["n_neurons"].nunique()
    tracked = counts[counts > 1].sort_values(ascending=False).head(n_track).index.tolist()
    if not tracked:
        print("[neurongen] interference: no neuron appears in more than one cell")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colours = dict(zip(tracked, sns.color_palette("husl", len(tracked))))
    for ax, et in zip(axes, EVAL_TYPES):
        sub = d[d["eval_type"] == et]
        for sig in tracked:
            q = sub[sub["signal"] == sig].sort_values("n_neurons")
            ax.plot(q["n_neurons"], q["corr"], marker="o", ms=5, lw=1.4,
                    color=colours[sig], alpha=0.85, label=sig)
        agg = _macro(sub, ["n_neurons"]).sort_values("n_neurons")
        ax.plot(agg["n_neurons"], agg["corr_macro"], lw=2.8, color="k", alpha=0.85,
                zorder=5, label="all trained (macro)")
        ax.set_xscale("log", base=10)
        ax.axhline(0, color="k", lw=0.9, alpha=0.4)
        ax.set_xlabel("neurons trained")
        ax.set_title(et)
        ax.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("corr (single neuron)")
    fig.suptitle("Interference: the SAME neurons followed as neighbours are added")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "interference.png", dpi=150)
    plt.close(fig)


def plot_distribution(ps: pd.DataFrame, out_dir):
    """Per-neuron spread behind each macro average.

    A macro mean of 0.5 built from neurons at 0.9 and 0.1 is a different result from one
    built from neurons all near 0.5, and the mean alone cannot tell them apart.
    """
    d = ps[ps["arm"] == "A"].dropna(subset=["corr"])
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, et in zip(axes, EVAL_TYPES):
        sub = d[d["eval_type"] == et]
        if sub.empty:
            continue
        sns.stripplot(data=sub, x="n_neurons", y="corr", hue="arm_label",
                      dodge=True, alpha=0.45, size=3, ax=ax,
                      palette={"seen": "#1f77b4", "unseen": "#d62728"}, legend=(ax is axes[0]))
        sns.boxplot(data=sub, x="n_neurons", y="corr", hue="arm_label", dodge=True,
                    showfliers=False, boxprops=dict(alpha=0.25), ax=ax,
                    palette={"seen": "#1f77b4", "unseen": "#d62728"}, legend=False)
        ax.axhline(0, color="k", lw=0.9, alpha=0.4)
        ax.set_title(et)
        ax.set_xlabel("neurons trained")
        ax.set_ylabel("corr (one point per neuron)" if ax is axes[0] else "")
    fig.suptitle("Per-neuron spread behind each macro average")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "per_neuron_distribution.png", dpi=150)
    plt.close(fig)


def main(exp_group="mlp_neurongen"):
    out_dir = Path(f"{ROOT}/outputs/plots/{exp_group}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ps, sm = load_sweep(exp_group)
    report(ps)
    ps.to_csv(out_dir / "neuron_generalization_all_signals.csv", index=False)
    if not sm.empty:
        sm.to_csv(out_dir / "neuron_generalization_summary.csv", index=False)
    m = plot_scaling(ps, out_dir)
    plot_sharing(ps, out_dir)
    plot_interference(ps, out_dir)
    plot_distribution(ps, out_dir)
    if m is not None:
        print("\n[neurongen] macro corr by (eval_type, arm, N):")
        print(m.sort_values(["eval_type", "arm_label", "n_neurons"]).to_string(index=False))
    print(f"\n[neurongen] wrote plots to {out_dir}")


if __name__ == "__main__":
    main(exp_group=sys.argv[1] if len(sys.argv) > 1 else "mlp_neurongen")
