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
    "legend.fontsize": 12, "legend.title_fontsize": 13, "figure.titlesize": 18,
})

ROOT = Path(__file__).parents[1]
# Legacy directory names from the arm-A/arm-B version of the sweep script. New runs carry a
# run_spec.json sidecar with exact fields; these regexes are only the fallback for runs
# that predate it.
ARM_RE = re.compile(r"_arm(?P<arm>[AB])")
N_RE = re.compile(r"_n(?P<n>\d+)")
R_RE = re.compile(r"_r(?P<r>\d+)")
LAYER_RE = re.compile(r"_L(?P<layer>\d+)")
EVAL_TYPES = ("endogenous", "interventional")
# seen = filled circle, solid; unseen = HOLLOW circle, dotted. A hollow glyph survives
# in a legend where a cross or a dash pattern does not.
STYLE = {"seen":   dict(marker="o", ls="-"),
         "unseen": dict(marker="o", ls=":", mfc="none", mew=1.8)}
# "single", not "null": pandas reads the literal string "null" back from any CSV as NaN
LEGACY_GROUP = {"A": "scaling", "B": "single"}


def _run_meta(d: Path):
    """Fields describing a run: from run_spec.json, else parsed from a legacy name."""
    spec = d / "run_spec.json"
    if spec.exists():
        j = json.loads(spec.read_text())
        return {
            "run": d.name, "group": j["group"], "n_neurons": int(j["neurons"]),
            "offset": int(j.get("offset", 0)), "rows_per_neuron": int(j["rows_per_neuron"]),
            "n_rows": int(j.get("rows", j["rows_per_neuron"] * j["neurons"])),
            "epochs": int(j.get("epochs", 1)), "lora_r": int(j.get("lora_r", 16)),
            "layer": j.get("layer"),
        }
    m_arm, m_n, m_r = ARM_RE.search(d.name), N_RE.search(d.name), R_RE.search(d.name)
    if not (m_arm and m_n and m_r):
        return None
    m_l = LAYER_RE.search(d.name)
    n, rows = int(m_n["n"]), int(m_r["r"])
    return {
        "run": d.name, "group": LEGACY_GROUP[m_arm["arm"]], "n_neurons": n, "offset": 0,
        "rows_per_neuron": rows // max(1, n), "n_rows": rows, "epochs": 1, "lora_r": 16,
        "layer": int(m_l["layer"]) if m_l else None,
    }


def load_sweep(exp_group: str):
    """One frame of per-neuron rows and one of per-arm summaries, across all runs."""
    base = Path(f"{ROOT}/outputs/runs/{exp_group}")
    if not base.exists():
        # Plots get rsynced more often than runs. The all-signals CSV this script writes
        # is the complete per-neuron frame, so re-plotting from it loses nothing.
        csv = Path(f"{ROOT}/outputs/plots/{exp_group}/neuron_generalization_all_signals.csv")
        if csv.exists():
            print(f"[neurongen] no run directories under {base}; re-plotting from {csv.name}")
            ps = pd.read_csv(csv)
            ps["arm_label"] = np.where(ps["is_trained"], "seen", "unseen")
            ps["is_baseline_cfg"] = (ps["epochs"] == 1) & (ps["lora_r"] == 16)
            return ps, pd.DataFrame()
        sys.exit(f"No such experiment group: {base}")
    per_signal, summary, skipped = [], [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        meta = _run_meta(d)
        if meta is None:
            continue                       # the shared pool config dir, etc.

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
    # the reference configuration every sweep varies away from
    ps["is_baseline_cfg"] = (ps["epochs"] == 1) & (ps["lora_r"] == 16)
    return ps, sm


def report(ps: pd.DataFrame):
    runs = ps[["run", "group", "n_neurons", "offset", "rows_per_neuron", "epochs", "lora_r"]
              ].drop_duplicates()
    print(f"\n[neurongen] {len(runs)} runs")
    print(runs.sort_values(["group", "n_neurons", "rows_per_neuron", "offset"]).to_string(index=False))
    dead = ps[ps.get("activation_std", pd.Series(1, index=ps.index)).fillna(1) <= 1e-6]
    if len(dead):
        print(f"[neurongen] {dead['signal'].nunique()} neuron(s) have near-zero activation "
              f"variance; their correlations are undefined, not poor, and are excluded "
              f"from every macro average below")
    a = ps[ps["group"] == "scaling"].drop_duplicates("run")
    if len(a) and a["n_rows"].nunique() > 1:
        print(f"[neurongen] WARNING: scaling rows are not constant across cells "
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
    d = ps[(ps["group"] == "scaling") & ps["is_baseline_cfg"]]
    if d.empty:
        return
    m = _macro(d, ["eval_type", "arm_label", "n_neurons"])
    m.to_csv(Path(out_dir) / "scaling_by_n.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, et in zip(axes, EVAL_TYPES):
        for lbl, colour in (("seen", "#1f77b4"), ("unseen", "#d62728")):
            q = m[(m["eval_type"] == et) & (m["arm_label"] == lbl)].sort_values("n_neurons")
            if q.empty:
                continue
            ax.errorbar(q["n_neurons"], q["corr_macro"], yerr=q["corr_std"].fillna(0),
                        ms=7, capsize=3, lw=1.8, color=colour, **STYLE[lbl],
                        label=f"{lbl} (n={q['n_defined'].iloc[0]} neurons)")
        ax.set_xscale("log", base=10)
        ax.axhline(0, color="k", lw=0.9, alpha=0.4)
        ax.set_xlabel("neurons trained (total rows held fixed)")
        ax.set_title(et)
        ax.legend(fontsize=10)
    axes[0].set_ylabel("macro corr over neurons")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "scaling_by_n.png", dpi=150)
    plt.close(fig)
    return m


def plot_sharing(ps: pd.DataFrame, out_dir):
    """Q2: the scaling curve against the single-neuron null, on rows-per-neuron.

    A scaling cell at N=k and a single-neuron run at the same rows-per-neuron give each
    neuron the same amount of data; they differ only in whether other neurons were
    trained alongside. Curves coinciding means the data did not transfer -- neurons were
    learned privately.

    The null at a given budget pools EVERY 1-neuron baseline-config run at that budget,
    whatever its group: the legacy single row and the single_n10 offsets 1-9 are the same
    kind of observation, and together they give the null an across-neuron spread where
    it previously had a single point.
    """
    d = ps[(ps["arm_label"] == "seen") & ps["is_baseline_cfg"]]
    scal = d[d["group"] == "scaling"]
    null = d[d["n_neurons"] == 1]
    if scal.empty or null.empty:
        return
    a = _macro(scal, ["eval_type", "rows_per_neuron", "n_neurons"]).assign(series="scaling")
    b = _macro(null, ["eval_type", "rows_per_neuron"]).assign(series="single-neuron", n_neurons=1)
    b["n_runs"] = null.groupby(["eval_type", "rows_per_neuron"])["run"].nunique().values
    m = pd.concat([a, b], ignore_index=True)
    m.to_csv(Path(out_dir) / "sharing_scaling_vs_null.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, et in zip(axes, EVAL_TYPES):
        qa = a[a["eval_type"] == et].sort_values("rows_per_neuron")
        qb = b[b["eval_type"] == et].sort_values("rows_per_neuron")
        ax.errorbar(qa["rows_per_neuron"], qa["corr_macro"], yerr=qa["corr_std"].fillna(0),
                    marker="o", ms=6, capsize=3, lw=1.8, color="#1f77b4", label="scaling (N neurons)")
        for _, r in qa.iterrows():
            ax.annotate(f"N={int(r['n_neurons'])}", (r["rows_per_neuron"], r["corr_macro"]),
                        textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=9, color="#1f77b4")
        ax.errorbar(qb["rows_per_neuron"], qb["corr_macro"], yerr=qb["corr_std"].fillna(0),
                    marker="s", ms=6, capsize=3, lw=1.8, ls="--", color="#7f7f7f",
                    label="single-neuron (matched data)")
        # how many single-neuron runs stand behind each null point -- the spread is only
        # real where this exceeds 1
        for _, r in qb.iterrows():
            ax.annotate(f"n={int(r['n_runs'])}", (r["rows_per_neuron"], r["corr_macro"]),
                        textcoords="offset points", xytext=(0, -14), ha="center",
                        fontsize=8, color="#7f7f7f")
        ax.set_xscale("log", base=10)
        ax.set_xlabel("training rows per neuron")
        ax.set_title(et)
        ax.legend(fontsize=10)
    axes[0].set_ylabel("macro corr, trained neurons")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "sharing_scaling_vs_null.png", dpi=150)
    plt.close(fig)
    return m


def plot_paired_null(ps: pd.DataFrame, out_dir, n_cell: int = 10):
    """The N=10 cell against its own neurons trained alone -- neuron for neuron.

    Stronger than comparing two means: each of the 10 neurons has a fidelity when
    trained with 9 others and a fidelity when trained by itself on the same data, so
    the per-neuron DIFFERENCE cancels neuron heterogeneity entirely. The sign and
    consistency of those ten differences is the sharing result at this budget.
    """
    d = ps[ps["is_baseline_cfg"] & ps["is_trained"]]
    cell = d[(d["group"] == "scaling") & (d["n_neurons"] == n_cell)]
    if cell.empty:
        return
    rpn = int(cell["rows_per_neuron"].iloc[0])
    alone = d[(d["n_neurons"] == 1) & (d["rows_per_neuron"] == rpn)]
    if alone.empty:
        return
    # a single-neuron run's trained neuron is the one it was trained on
    alone = alone[alone["signal"].isin(cell["signal"].unique())]
    merged = cell.merge(alone[["eval_type", "signal", "corr"]], on=["eval_type", "signal"],
                        suffixes=("_together", "_alone"))
    if merged.empty:
        print(f"[neurongen] paired null: no single-neuron runs match the N={n_cell} "
              f"cell's neurons at {rpn} rows/neuron")
        return
    merged["delta"] = merged["corr_together"] - merged["corr_alone"]
    merged.to_csv(Path(out_dir) / f"paired_null_n{n_cell}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, et in zip(axes, EVAL_TYPES):
        q = merged[merged["eval_type"] == et].sort_values("corr_alone")
        if q.empty:
            continue
        x = np.arange(len(q))
        ax.scatter(x, q["corr_alone"], marker="s", s=55, color="#7f7f7f", label="trained alone", zorder=3)
        ax.scatter(x, q["corr_together"], marker="o", s=55, color="#1f77b4",
                   label=f"trained with {n_cell - 1} others", zorder=3)
        for xi, (_, r) in zip(x, q.iterrows()):
            ax.plot([xi, xi], [r["corr_alone"], r["corr_together"]], color="k", lw=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(q["signal"], rotation=45, ha="right", fontsize=8)
        md, sd = q["delta"].mean(), q["delta"].std()
        n_pos = int((q["delta"] > 0).sum())
        ax.set_title(f"{et}\nΔ(together − alone) = {md:+.3f} ± {sd:.3f};  "
                     f"{n_pos}/{len(q)} improved", fontsize=13)
        ax.axhline(0, color="k", lw=0.8, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("corr (this neuron)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"paired_null_n{n_cell}.png", dpi=150)
    plt.close(fig)
    return merged


def plot_sweep(ps: pd.DataFrame, out_dir, group: str, x: str, xlabel: str, fname: str,
               xlog: bool = True):
    """Generic one-factor sweep: seen and unseen macro corr against `x` within a group,
    with the matching baseline-config run (same neurons and budget) drawn as a reference
    point so the sweep is read as a change FROM something."""
    d = ps[ps["group"] == group]
    if d.empty:
        return
    m = _macro(d, ["eval_type", "arm_label", "n_neurons", "rows_per_neuron", x])
    m.to_csv(Path(out_dir) / f"{fname}.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cells = sorted(d[["n_neurons", "rows_per_neuron"]].drop_duplicates().itertuples(index=False))
    palette = dict(zip(cells, sns.color_palette("tab10", len(cells))))
    for ax, et in zip(axes, EVAL_TYPES):
        for cell in cells:
            n, rpn = cell
            for lbl in ("seen", "unseen"):
                q = m[(m["eval_type"] == et) & (m["arm_label"] == lbl) &
                      (m["n_neurons"] == n) & (m["rows_per_neuron"] == rpn)].sort_values(x)
                if q.empty:
                    continue
                ax.errorbar(q[x], q["corr_macro"], yerr=q["corr_std"].fillna(0),
                            ms=7, capsize=3, lw=1.8, color=palette[cell], **STYLE[lbl],
                            label=f"N={n} @ {rpn//1000}k/neuron, {lbl}")
            # the baseline-config run for this cell, as a reference
            ref = ps[ps["is_baseline_cfg"] & (ps["n_neurons"] == n) & (ps["rows_per_neuron"] == rpn)
                     & (ps["group"] != group) & (ps["eval_type"] == et) & ps["is_trained"]]
            if len(ref):
                base_x = {"epochs": 1, "lora_r": 16}[x]
                # Say WHICH baseline: the same N exists at more than one budget (N=30 at
                # 10k/neuron in the scaling sweep, at 30k/neuron in data30k), and the star
                # is the one matching THIS sweep's budget, not the scaling cell.
                ax.scatter([base_x], [ref["corr"].mean()], marker="*", s=180, color=palette[cell],
                           edgecolors="k", zorder=5,
                           label=f"baseline: N={n} @ {rpn//1000}k/neuron, "
                                 f"{ {'epochs': 1, 'lora_r': 16}[x] } {x} (seen)")
        if xlog:
            ax.set_xscale("log", base=2 if x == "lora_r" else 10)
        ax.axhline(0, color="k", lw=0.9, alpha=0.4)
        ax.set_xlabel(xlabel)
        ax.set_title(et)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("macro corr over neurons")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{fname}.png", dpi=150)
    plt.close(fig)
    return m


def plot_interference(ps: pd.DataFrame, out_dir, n_track: int = 6):
    """Does adding neurons damage the ones already there?

    The training sets are nested prefixes, so a neuron trained at N=1 is also trained at
    every larger N. Following those specific neurons separates the cost of adding
    neighbours from neuron-to-neuron heterogeneity, which the macro average blends in.
    """
    d = ps[(ps["group"] == "scaling") & ps["is_baseline_cfg"] & ps["is_trained"]]
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
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "interference.png", dpi=150)
    plt.close(fig)


def plot_distribution(ps: pd.DataFrame, out_dir):
    """Per-neuron spread behind each macro average.

    A macro mean of 0.5 built from neurons at 0.9 and 0.1 is a different result from one
    built from neurons all near 0.5, and the mean alone cannot tell them apart.
    """
    d = ps[(ps["group"] == "scaling") & ps["is_baseline_cfg"]].dropna(subset=["corr"])
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
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "per_neuron_distribution.png", dpi=150)
    plt.close(fig)


TRAIN_BATCH = 32     # every run in these sweeps; updates = rows x epochs / batch


def plot_updates(ps: pd.DataFrame, out_dir):
    """Seen-neuron fidelity against TOTAL optimizer updates, coloured by rows per neuron.

    Every update trains the same shared adapter and head -- there is no per-neuron
    parameter, the neuron only enters through the prompt -- so "how many updates" and
    "how much distinct data per neuron" are separable axes, and the sweeps happen to
    populate both. This plot puts every baseline-config run on those two axes at once,
    so the question "is it updates, or is it data per neuron?" is answered by whether
    the points line up on x (updates) or on colour (rows/neuron).

    Rank rows are excluded (they change neither axis); the ten 30k single-neuron runs
    and the three 10-epoch runs are each pooled into one point.
    """
    d = ps[ps["is_trained"] & (ps["lora_r"] == 16)].copy()
    if d.empty:
        return
    d["updates"] = d["rows_per_neuron"] * d["n_neurons"] * d["epochs"] / TRAIN_BATCH
    key = ["eval_type", "n_neurons", "rows_per_neuron", "epochs"]
    m = (d.groupby(key)["corr"].agg(corr_macro="mean", corr_std="std", n_neurons_eval="count")
           .reset_index())
    m["updates"] = m["rows_per_neuron"] * m["n_neurons"] * m["epochs"] / TRAIN_BATCH
    m["n_runs"] = d.groupby(key)["run"].nunique().values
    m.to_csv(Path(out_dir) / "updates_vs_fidelity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    rpns = sorted(m["rows_per_neuron"].unique())
    norm = matplotlib.colors.LogNorm(vmin=min(rpns), vmax=max(rpns))
    cmap = plt.get_cmap("viridis")
    for ax, et in zip(axes, EVAL_TYPES):
        q = m[m["eval_type"] == et]
        sc = ax.scatter(q["updates"], q["corr_macro"], c=q["rows_per_neuron"], cmap=cmap, norm=norm,
                        s=60 + 18 * np.log2(q["n_neurons"]), edgecolors="k", linewidths=0.7, zorder=3)
        ax.errorbar(q["updates"], q["corr_macro"], yerr=q["corr_std"].fillna(0), fmt="none",
                    ecolor="grey", elinewidth=0.8, capsize=2, zorder=2)
        for _, r in q.iterrows():
            ep = f" x{int(r['epochs'])}ep" if r["epochs"] > 1 else ""
            ax.annotate(f"N={int(r['n_neurons'])} @{int(r['rows_per_neuron'])//1000}k{ep}",
                        (r["updates"], r["corr_macro"]), textcoords="offset points",
                        xytext=(6, 4), fontsize=8)
        ax.set_xscale("log", base=10)
        ax.axhline(0, color="k", lw=0.9, alpha=0.4)
        ax.set_xlabel(f"total optimizer updates  (rows x epochs / {TRAIN_BATCH})")
        ax.set_title(et)
    axes[0].set_ylabel("macro corr, trained neurons")
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("distinct rows per neuron")
    fig.savefig(Path(out_dir) / "updates_vs_fidelity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return m


REF_UPDATES = 300_000 / TRAIN_BATCH     # the N=10 cell: 9,375 optimizer updates


def plot_data30k(ps: pd.DataFrame, out_dir, rpn: int = 30_000):
    """The fixed-budget scaling curve against a series that holds rows/neuron fixed.

    The second series is assembled from EVERY r=16 run at `rpn` rows/neuron, whatever
    group it came from -- the N=10 scaling cell, the data30k rows, the 10-epoch singles.
    Where a given N exists at several epoch settings, the one whose total updates are
    closest to the N=10 cell's is used, and points that needed epochs > 1 to get there
    are annotated. That matters because at 30k rows/neuron and one epoch, N=1 gets 937
    updates and sits at chance: without matching updates the series would RISE with N,
    which reads as "more neurons helps" when it is "more updates helps".

    A flat series then says the fixed-budget fall was data starvation, full stop.
    """
    scal = ps[(ps["group"] == "scaling") & ps["is_baseline_cfg"]]
    pool = ps[(ps["rows_per_neuron"] == rpn) & (ps["lora_r"] == 16)].copy()
    if pool.empty or scal.empty:
        return
    pool["updates"] = pool["rows_per_neuron"] * pool["n_neurons"] * pool["epochs"] / TRAIN_BATCH
    # per N, the epoch setting closest to the reference update count
    pick = (pool.drop_duplicates(["n_neurons", "epochs"])
                .assign(gap=lambda d: (d["updates"] - REF_UPDATES).abs())
                .sort_values("gap").drop_duplicates("n_neurons")[["n_neurons", "epochs"]])
    matched = pool.merge(pick, on=["n_neurons", "epochs"])

    a = _macro(scal, ["eval_type", "arm_label", "n_neurons"]).assign(series="300k total (scaling)")
    b = _macro(matched, ["eval_type", "arm_label", "n_neurons"]).assign(series=f"{rpn // 1000}k per neuron")
    ep = matched.drop_duplicates("n_neurons").set_index("n_neurons")["epochs"]
    b["epochs"] = b["n_neurons"].map(ep)
    b["updates"] = b["rows_per_neuron"] if "rows_per_neuron" in b else rpn * b["n_neurons"] * b["epochs"] / TRAIN_BATCH
    m = pd.concat([a, b], ignore_index=True)
    m.to_csv(Path(out_dir) / "data30k_vs_scaling.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, et in zip(axes, EVAL_TYPES):
        for series, colour in (("300k total (scaling)", "#1f77b4"), (f"{rpn // 1000}k per neuron", "#d62728")):
            for lbl in ("seen", "unseen"):
                q = m[(m["eval_type"] == et) & (m["series"] == series) &
                      (m["arm_label"] == lbl)].sort_values("n_neurons")
                if q.empty:
                    continue
                ax.errorbar(q["n_neurons"], q["corr_macro"], yerr=q["corr_std"].fillna(0),
                            ms=7, capsize=3, lw=1.8, color=colour, **STYLE[lbl],
                            label=f"{series}, {lbl}")
                if lbl == "seen" and "epochs" in q and series != "300k total (scaling)":
                    for _, r in q[q["epochs"] > 1].iterrows():
                        ax.annotate(f"x{int(r['epochs'])} ep", (r["n_neurons"], r["corr_macro"]),
                                    textcoords="offset points", xytext=(7, 6), fontsize=9,
                                    color=colour)
        ax.set_xscale("log", base=10)
        ax.axhline(0, color="k", lw=0.9, alpha=0.4)
        ax.set_xlabel("neurons trained")
        ax.set_title(et)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("macro corr over neurons")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "data30k_vs_scaling.png", dpi=150)
    plt.close(fig)
    return m


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
    plot_paired_null(ps, out_dir, n_cell=10)
    plot_interference(ps, out_dir)
    plot_distribution(ps, out_dir)
    # data30k joins the scaling picture on the same axes as a second budget
    plot_sweep(ps, out_dir, "epochs", "epochs", "training epochs (same 30k rows/neuron)",
               "sweep_epochs")
    plot_sweep(ps, out_dir, "rank", "lora_r", "LoRA rank (N=30 @ 30k rows/neuron)",
               "sweep_rank")
    plot_data30k(ps, out_dir)
    plot_updates(ps, out_dir)
    if m is not None:
        print("\n[neurongen] macro corr by (eval_type, arm, N):")
        print(m.sort_values(["eval_type", "arm_label", "n_neurons"]).to_string(index=False))
    print(f"\n[neurongen] wrote plots to {out_dir}")


if __name__ == "__main__":
    main(exp_group=sys.argv[1] if len(sys.argv) > 1 else "mlp_neurongen")
