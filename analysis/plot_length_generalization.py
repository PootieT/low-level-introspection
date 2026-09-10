#!/usr/bin/env python3
"""
Does the readout reach further back than it was trained to reach?

Reads the length-generalization sweep (runs named lengen_N<sig>_len<L>_win<W>_...) and
plots fidelity against DISTANCE FROM THE INTROSPECTED TOKEN TO THE READOUT, which is
the quantity the training window restricts. Sources, written by
score_length_generalization.py into each run directory:

  length_generalization_by_distance.csv   corr per (eval_len, eval_type, distance bin),
                                          with in_train_range already labelled
  length_generalization.csv               pooled-over-distance summary per eval_len

WHY DISTANCE AND NOT EVAL LENGTH. The introspection prompt sits AFTER the text, so
reading position p means reaching back (text_len - 1 - p) tokens. A run trained on the
last 16 tokens has seen reaches of 0-15 and nothing else, whatever the sequence length
was. Plotting against eval length instead would mix that reach with absolute position
and distractor load, and the three move together.

WHY THE SHADING MATTERS. A learned recency prior ("attend to the last few tokens") and
a genuine inability to reach produce the SAME falling curve. They separate only on
whether the IN-RANGE bins hold up while the out-of-range ones fall, so in-range bins are
shaded on every panel and the two are never averaged into one number.

WHICH METRIC. Interventional is the headline: the injected value is independent of
context, so contextual bypass scores exactly 0 and any correlation is a real causal
read. Endogenous is plotted alongside because the gap between them at long distance is
itself informative -- endogenous holding up while interventional collapses means the
readout fell back on predicting the activation from context.

ERROR BARS are spread across the 5 neurons of a cell (heterogeneity), not run-to-run
noise -- there is one run per (cell, neuron).

Panels
  1 reach curves     corr vs distance, one line per training window, faceted by eval len
  2 matched reach    fixed window across training lengths -- does a longer training
                     sequence change the mechanism, holding reach fixed?
  3 gap heatmap      (train_len x window) -> in-range corr minus out-of-range corr
  4 ceiling check    diagonal cells (window == train_len): raw reach with no
                     generalization gap, which bounds everything off the diagonal
"""
import os
import re
import sys
import glob
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
LEN_RE = re.compile(r"_len(?P<len>\d+)")
WIN_RE = re.compile(r"_win(?P<win>\d+)")
GATE_RE = re.compile(r"gate(?P<gate>on|off)")
NEURON_RE = re.compile(r"N(?P<layer>\d+)\.(?P<index>\d+)")

DISTANCE_BINS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)  # must match the scorer
MIN_BIN_N = 30      # below this a bin's correlation is mostly noise; drawn hollow


def load_sweep(exp_group: str, gate: str | None = "off") -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per (run, eval_len, eval_type, distance bin), plus the pooled summary.

    Reads from the run directory and, failing that, from a diag_* subdirectory -- a
    --length-only rerun on an explicit checkpoint writes there.
    """
    base = Path(f"{ROOT}/outputs/runs/{exp_group}")
    if not base.exists():
        sys.exit(f"No such experiment group: {base}")
    bins, pooled, skipped = [], [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        g = GATE_RE.search(d.name)
        if gate and (not g or g["gate"] != gate):
            continue
        m_len, m_win, m_n = LEN_RE.search(d.name), WIN_RE.search(d.name), NEURON_RE.search(d.name)
        if not (m_len and m_win):
            skipped.append((d.name, "name has no _len/_win tag"))
            continue

        def find(fname):
            direct = d / fname
            if direct.exists():
                return direct
            sub = sorted(d.glob(f"diag_*/{fname}"))
            return sub[-1] if sub else None

        meta = {
            "run": d.name,
            "train_len": int(m_len["len"]),
            "window": int(m_win["win"]),
            "neuron": f"N{m_n['layer']}.{m_n['index']}" if m_n else "?",
            "cell": f"len{m_len['len']}/win{m_win['win']}",
        }
        fb, fp = find("length_generalization_by_distance.csv"), find("length_generalization.csv")
        if fb is None:
            skipped.append((d.name, "no length_generalization_by_distance.csv"))
            continue
        b = pd.read_csv(fb).assign(**meta)
        bins.append(b)
        if fp is not None:
            pooled.append(pd.read_csv(fp).assign(**meta))

    if not bins:
        sys.exit(f"No length-generalization results under {base} "
                 f"(gate={gate}). Rsync the runs first.")
    b = pd.concat(bins, ignore_index=True)
    p = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()
    if skipped:
        print(f"[lengen] skipped {len(skipped)} directories:")
        for name, why in skipped[:10]:
            print(f"    {name}: {why}")
    # in_train_range comes from the scorer, but recompute as a fallback: an older run may
    # predate it, and silently averaging seen with unseen bins is the one error that
    # would make a null result look real.
    if "in_train_range" not in b or b["in_train_range"].isna().all():
        b["in_train_range"] = b["max_distance"] <= (b["window"] - 1)
    b["in_train_range"] = b["in_train_range"].astype(bool)
    # A bin whose span crosses the edge of the training window is neither in nor out.
    # The scorer flags these; older result files predate the flag, so derive it from the
    # span when it is absent. Assigning a straddling bin to a side inflates the gap.
    if "min_distance" not in b:
        # Reconstruct each bin's lower edge from the fixed ladder. Falling back to
        # max_distance instead would make the straddle test (lo <= e) & (hi > e)
        # identically False and quietly disable the exclusion.
        prev = {hi: (lo + 1) for lo, hi in zip([-1] + list(DISTANCE_BINS[:-1]), DISTANCE_BINS)}
        b["min_distance"] = b["max_distance"].map(prev).fillna(b["max_distance"])
    if "straddles_window" not in b:
        edge = b["window"] - 1          # trained distances are 0 .. window-1
        b["straddles_window"] = (b["min_distance"] <= edge) & (b["max_distance"] > edge)
    b["straddles_window"] = b["straddles_window"].fillna(False).astype(bool)
    return b, p


def report(b: pd.DataFrame, expected_cells=None):
    cells = sorted(b["cell"].unique())
    print(f"\n[lengen] {b['run'].nunique()} runs, {len(cells)} cells, "
          f"eval lengths {sorted(b['eval_len'].unique())}")
    per_cell = b.groupby("cell")["neuron"].nunique()
    thin = per_cell[per_cell < per_cell.max()]
    if len(thin):
        print(f"[lengen] cells with fewer neurons than the fullest ({per_cell.max()}):")
        for c, n in thin.items():
            print(f"    {c}: {n}")
    if expected_cells:
        missing = sorted(set(expected_cells) - set(cells))
        if missing:
            print(f"[lengen] MISSING cells: {missing}")
    noisy = b[(b["n"] < MIN_BIN_N) & b["corr"].notna()]
    if len(noisy):
        thin_bins = sorted(noisy["max_distance"].unique())
        print(f"[lengen] {len(noisy)} bins have n<{MIN_BIN_N} (distances {thin_bins}); "
              f"their correlations are mostly noise. n_rows is carried into every "
              f"aggregate CSV so these stay checkable.")


def _agg(b: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean and spread ACROSS NEURONS. Bins whose corr is undefined are dropped from the
    mean and counted, so a shrinking denominator cannot pass for a stable curve."""
    g = b.groupby(by)
    out = g["corr"].agg(mean="mean", std="std", n_neurons="count").reset_index()
    out["n_rows"] = g["n"].sum().values
    return out


def _draw(ax, s, color, label, train_len):
    """One curve, with points BEYOND THE TRAINING LENGTH drawn as crosses.

    Three regimes, and only two of them were previously distinguishable:
      o  within the training window          -- trained reach
      o  beyond the window, <= train_len     -- unseen reach, but sequences that long
                                                were seen during training
      X  beyond train_len                    -- unseen reach AND unseen sequence length
    The middle case turns out to be the hard one, so collapsing it with the third (as a
    single marker does) hides the actual result.
    """
    ax.errorbar(s["max_distance"], s["mean"], yerr=s["std"].fillna(0),
                marker="", capsize=3, lw=1.8, color=color, label=label)
    within = s["max_distance"] <= train_len
    ax.scatter(s.loc[within, "max_distance"], s.loc[within, "mean"],
               marker="o", s=34, color=color, zorder=3)
    ax.scatter(s.loc[~within, "max_distance"], s.loc[~within, "mean"],
               marker="X", s=80, color=color, zorder=3, linewidths=0.6, edgecolors="white")


def plot_reach_curves(b: pd.DataFrame, out_dir, eval_type="interventional"):
    d = b[b["eval_type"] == eval_type]
    if d.empty:
        return
    eval_lens = sorted(d["eval_len"].unique())
    train_lens = sorted(d["train_len"].unique())
    fig, axes = plt.subplots(len(train_lens), len(eval_lens),
                             figsize=(4.6 * len(eval_lens), 3.9 * len(train_lens)),
                             squeeze=False, sharex=True, sharey=True)
    windows = sorted(d["window"].unique())
    colors = dict(zip(windows, sns.color_palette("viridis", len(windows))))
    for r, tl in enumerate(train_lens):
        for c, el in enumerate(eval_lens):
            ax = axes[r][c]
            sub = d[(d["train_len"] == tl) & (d["eval_len"] == el)]
            for w in sorted(sub["window"].unique()):
                s = _agg(sub[sub["window"] == w], ["max_distance"]).sort_values("max_distance")
                if s.empty:
                    continue
                _draw(ax, s, colors[w], f"win {w}", tl)
                # the training window's own edge: everything right of it is unseen reach
                ax.axvline(w - 1, color=colors[w], ls=":", lw=1, alpha=0.6)
            ax.set_xscale("log", base=2)
            ax.axhline(0, color="k", lw=0.8, alpha=0.4)
            ax.set_ylim(-0.25, 1.02)
            if r == 0:
                ax.set_title(f"eval len {el}")
            if c == 0:
                ax.set_ylabel(f"train len {tl}\ncorr")
            if r == len(train_lens) - 1:
                ax.set_xlabel("distance from readout (tokens, log2)")
    # Build the legend from the full colour map, not from one panel's lines: the
    # top-left cell trains on a single window, so a panel-derived legend would list one.
    handles = [plt.Line2D([], [], color=colors[w], marker="o", lw=1.8, label=f"win {w}")
               for w in windows]
    axes[0][-1].legend(handles=handles, title="train window", fontsize=10)
    fig.suptitle(f"{eval_type} readout vs reach — dotted line = edge of training "
             f"window; X = distance beyond the training LENGTH")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"reach_curves_{eval_type}.png", dpi=150)
    plt.close(fig)


def plot_matched_reach(b: pd.DataFrame, out_dir, eval_type="interventional"):
    """Same training window across training lengths: does more context change the
    mechanism, holding reach fixed? Only windows present at >1 train length."""
    d = b[b["eval_type"] == eval_type]
    shared = [w for w, g in d.groupby("window") if g["train_len"].nunique() > 1]
    if not shared:
        return
    shared = sorted(shared)
    fig, axes = plt.subplots(1, len(shared), figsize=(4.8 * len(shared), 4.2),
                             squeeze=False, sharey=True)
    for i, w in enumerate(shared):
        ax = axes[0][i]
        sub = d[d["window"] == w]
        tls = sorted(sub["train_len"].unique())
        colors = dict(zip(tls, sns.color_palette("rocket", len(tls))))
        for tl in tls:
            s = _agg(sub[sub["train_len"] == tl], ["max_distance"]).sort_values("max_distance")
            _draw(ax, s, colors[tl], f"train len {tl}", tl)
        ax.axvline(w - 1, color="k", ls=":", lw=1, alpha=0.6)
        ax.axhline(0, color="k", lw=0.8, alpha=0.4)
        ax.set_xscale("log", base=2)
        ax.set_title(f"training window = {w}")
        ax.set_xlabel("distance from readout")
        if i == 0:
            ax.set_ylabel(f"{eval_type} corr")
        ax.legend(fontsize=10)
    fig.suptitle("Matched reach across training lengths (pooled over eval lengths)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"matched_reach_{eval_type}.png", dpi=150)
    plt.close(fig)


def plot_gap_heatmap(b: pd.DataFrame, out_dir, eval_type="interventional"):
    """(train_len x window) -> in-range corr, out-of-range corr, and the gap.

    The gap is the generalization cost. A cell on the diagonal (window == train_len) has
    no out-of-range bins at its own eval length, so its gap is blank rather than 0.
    """
    d = b[(b["eval_type"] == eval_type) & (~b["straddles_window"])]
    if d.empty:
        return
    n_strad = int((b["eval_type"] == eval_type).sum() - len(d))
    if n_strad:
        print(f"[lengen] gap heatmap ({eval_type}): excluded {n_strad} bins that straddle "
              f"the training-window edge; they belong to neither side.")
    agg = _agg(d, ["train_len", "window", "in_train_range"])
    piv = {k: agg[agg["in_train_range"] == k]
           .pivot(index="train_len", columns="window", values="mean")
           for k in (True, False)}
    gap = piv[True] - piv[False]
    panels = [("in training range", piv[True], "viridis"),
              ("beyond training range", piv[False], "viridis"),
              ("gap (in − out)", gap, "rocket_r")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (title, mat, cmap) in zip(axes, panels):
        sns.heatmap(mat, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                    vmin=None if title.startswith("gap") else -0.2,
                    vmax=None if title.startswith("gap") else 1.0,
                    cbar_kws={"shrink": 0.8}, linewidths=0.5, linecolor="white")
        ax.set_title(f"{title}")
        ax.set_xlabel("training window")
        ax.set_ylabel("training length")
    fig.suptitle(f"{eval_type} readout: generalization gap by cell "
                 f"(blank = no bins of that kind)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"gap_heatmap_{eval_type}.png", dpi=150)
    plt.close(fig)
    return agg


def plot_ceiling(b: pd.DataFrame, out_dir):
    """Diagonal cells (window == train_len): raw reach with no generalization gap.

    This bounds every off-diagonal result. If the ceiling itself falls with distance,
    an off-diagonal failure at that distance is not a generalization failure -- the
    model cannot reach that far even when trained to.
    """
    d = b[b["window"] == b["train_len"]]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, et in zip(axes, ("interventional", "endogenous")):
        sub = d[d["eval_type"] == et]
        tls = sorted(sub["train_len"].unique())
        colors = dict(zip(tls, sns.color_palette("mako", len(tls))))
        for tl in tls:
            s = _agg(sub[sub["train_len"] == tl], ["max_distance"]).sort_values("max_distance")
            _draw(ax, s, colors[tl], f"len {tl}", tl)
        ax.set_xscale("log", base=2)
        ax.axhline(0, color="k", lw=0.8, alpha=0.4)
        ax.set_title(et)
        ax.set_xlabel("distance from readout")
        ax.legend(fontsize=10)
    axes[0].set_ylabel("corr")
    fig.suptitle("Ceiling: cells trained on the whole sequence (window = length)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "ceiling_diagonal.png", dpi=150)
    plt.close(fig)


def variance_decomposition(b: pd.DataFrame, out_dir) -> pd.DataFrame:
    """Which axis the spread in the aggregated curves actually comes from.

    Every point in ceiling_diagonal and matched_reach averages over BOTH neurons and eval
    lengths, so its error bar mixes two different things: how much neurons differ from
    each other, and how much the same neuron differs across eval lengths (i.e. how much
    context sits in front of the target). Those have opposite implications -- the first is
    heterogeneity to report, the second is a systematic effect being hidden inside a
    spread estimate.

    Two-way sum-of-squares split within each (eval_type, train_len, window, distance) cell
    over the balanced neuron x eval_len block:
        SS_total = SS_neuron + SS_eval_len + SS_residual
    Reported as shares. Cells with fewer than 2 neurons or 2 eval lengths are skipped --
    the decomposition is undefined there, and the count of usable cells is reported so a
    thin table is visible as thin.
    """
    rows = []
    for (et, tl, w, md), g in b.groupby(["eval_type", "train_len", "window", "max_distance"]):
        piv = g.pivot_table(index="neuron", columns="eval_len", values="corr")
        piv = piv.dropna(axis=1, how="any").dropna(axis=0, how="any")
        n_r, n_c = piv.shape
        if n_r < 2 or n_c < 2:
            continue
        v = piv.to_numpy()
        m = v.mean()
        ss_total = ((v - m) ** 2).sum()
        ss_neuron = n_c * ((v.mean(axis=1) - m) ** 2).sum()
        ss_evallen = n_r * ((v.mean(axis=0) - m) ** 2).sum()
        rows.append({
            "eval_type": et, "train_len": tl, "window": w, "max_distance": md,
            "n_neurons": n_r, "n_eval_lens": n_c,
            "sd_total": float(np.sqrt(ss_total / max(1, v.size - 1))),
            "ss_total": ss_total, "ss_neuron": ss_neuron, "ss_eval_len": ss_evallen,
            "ss_residual": ss_total - ss_neuron - ss_evallen,
        })
    if not rows:
        print("[lengen] variance decomposition: no cell has >=2 neurons AND >=2 eval "
              "lengths; nothing to decompose")
        return pd.DataFrame()

    cells = pd.DataFrame(rows)
    for c in ("neuron", "eval_len", "residual"):
        cells[f"pct_{c}"] = 100 * cells[f"ss_{c}"] / cells["ss_total"].replace(0, np.nan)
    cells.to_csv(Path(out_dir) / "variance_by_cell.csv", index=False)

    def _roll(df, by):
        g = df.groupby(by)[["ss_total", "ss_neuron", "ss_eval_len", "ss_residual"]].sum()
        out = pd.DataFrame({
            "pct_neuron": 100 * g["ss_neuron"] / g["ss_total"],
            "pct_eval_len": 100 * g["ss_eval_len"] / g["ss_total"],
            "pct_residual": 100 * g["ss_residual"] / g["ss_total"],
        }).reset_index()
        out["n_cells"] = df.groupby(by).size().values
        out["mean_sd"] = df.groupby(by)["sd_total"].mean().values
        return out

    cells["scope"] = np.where(cells["train_len"] == cells["window"], "diagonal", "off-diagonal")
    overall = pd.concat([_roll(cells, ["eval_type"]).assign(scope="all cells"),
                         _roll(cells[cells["scope"] == "diagonal"], ["eval_type"])
                         .assign(scope="diagonal only")], ignore_index=True)
    by_dist = _roll(cells, ["eval_type", "max_distance"])
    overall.to_csv(Path(out_dir) / "variance_overall.csv", index=False)
    by_dist.to_csv(Path(out_dir) / "variance_by_distance.csv", index=False)

    print("\n[lengen] where the spread comes from (share of SS, pooled over cells):")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("\n[lengen] by distance:")
    print(by_dist.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    return cells


def plot_variance_sources(b: pd.DataFrame, out_dir, eval_type="interventional"):
    """The same diagonal curves, drawn once per neuron and once per eval length.

    Companion to the table: whichever panel fans out more is the axis carrying the
    spread. The aggregate is overlaid in black so the thin lines can be read against the
    curve the other plots show.
    """
    d = b[(b["eval_type"] == eval_type) & (b["window"] == b["train_len"])]
    if d.empty:
        return
    tls = sorted(d["train_len"].unique())
    fig, axes = plt.subplots(len(tls), 2, figsize=(11, 3.3 * len(tls)),
                             squeeze=False, sharex=True, sharey=True)
    for r, tl in enumerate(tls):
        sub = d[d["train_len"] == tl]
        for c, (key, pal) in enumerate((("neuron", "husl"), ("eval_len", "crest"))):
            ax = axes[r][c]
            levels = sorted(sub[key].unique())
            colors = dict(zip(levels, sns.color_palette(pal, len(levels))))
            for lv in levels:
                q = _agg(sub[sub[key] == lv], ["max_distance"]).sort_values("max_distance")
                ax.plot(q["max_distance"], q["mean"], lw=1.0, alpha=0.75,
                        color=colors[lv], marker="o", ms=3, label=str(lv))
            agg = _agg(sub, ["max_distance"]).sort_values("max_distance")
            ax.plot(agg["max_distance"], agg["mean"], lw=2.6, color="k", alpha=0.85,
                    zorder=5, label="aggregate")
            ax.set_xscale("log", base=2)
            ax.axhline(0, color="k", lw=0.8, alpha=0.3)
            ax.axvline(tl, color="grey", ls=":", lw=1)
            if r == 0:
                ax.set_title(f"one line per {key}")
            if c == 0:
                ax.set_ylabel(f"train len {tl}\ncorr")
            if r == len(tls) - 1:
                ax.set_xlabel("distance from readout")
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"{eval_type}: sources of spread on the diagonal cells "
                 f"(dotted line = training length)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"variance_sources_{eval_type}.png", dpi=150)
    plt.close(fig)


def plot_relative_generalization(pooled: pd.DataFrame, out_dir, eval_type="interventional"):
    """Does degradation track the LENGTH RATIO rather than the absolute length?

    x  eval_len / train_len -- how far past its training length the run is being pushed
    y  (corr - corr_ID) / corr_ID, where corr_ID is that same run evaluated at its OWN
       training length. So a run scoring 0.80 in-distribution and 0.70 at 2x reads -12.5%.

    Normalising against each run's own in-distribution score is what makes the training
    lengths comparable: a run that starts at 0.5 and one that starts at 0.9 cannot be
    compared on raw corr, and the absolute drop would flatter whichever started lower.

    If the curves LIE ON TOP OF EACH OTHER, degradation is a function of the ratio and
    training short is free -- push any length by 4x and pay the same fraction. If they
    SEPARATE, absolute length is what matters and a short-trained run cannot be rescued
    by the ratio being modest.

    Diagonal cells only (window == train_len), matching ceiling_diagonal: off the diagonal
    the in-distribution reference is itself already a generalization result, so the
    normalisation would divide one unknown by another.

    Uses the POOLED per-eval-length correlation (over that eval set's whole distance
    range), not a per-bin number: the question is about the eval length as a whole.
    """
    col = {"interventional": "interventional_corr", "endogenous": "endogenous_corr"}[eval_type]
    if pooled.empty or col not in pooled.columns:
        return
    d = pooled[pooled["window"] == pooled["train_len"]].dropna(subset=[col]).copy()
    if d.empty:
        return

    ref = (d[d["eval_len"] == d["train_len"]]
           .drop_duplicates(subset=["run"]).set_index("run")[col])
    d["id_corr"] = d["run"].map(ref)
    missing = int(d["id_corr"].isna().sum())
    # A near-zero in-distribution score makes the ratio explode and means the run failed
    # at its own training length -- there is no generalization to measure from there.
    tiny = int((d["id_corr"].abs() <= 0.05).sum())
    d = d[d["id_corr"].notna() & (d["id_corr"].abs() > 0.05)]
    if missing or tiny:
        print(f"[lengen] relative generalization ({eval_type}): dropped {missing} rows with "
              f"no in-distribution reference and {tiny} whose reference |corr| <= 0.05")
    if d.empty:
        return

    d["ratio"] = d["eval_len"] / d["train_len"]
    d["rel"] = (d[col] - d["id_corr"]) / d["id_corr"]
    d = d[d["ratio"] >= 1]

    g = d.groupby(["train_len", "ratio"])["rel"].agg(mean="mean", std="std", n="count").reset_index()
    g["mean_pct"], g["std_pct"] = 100 * g["mean"], 100 * g["std"]
    g.to_csv(Path(out_dir) / f"relative_generalization_{eval_type}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    tls = sorted(g["train_len"].unique())
    colors = dict(zip(tls, sns.color_palette("mako", len(tls))))
    for tl in tls:
        q = g[g["train_len"] == tl].sort_values("ratio")
        ax.errorbar(q["ratio"], q["mean_pct"], yerr=q["std_pct"].fillna(0),
                    marker="o", ms=6, capsize=3, lw=1.8, color=colors[tl],
                    label=f"train len {tl}  (ID corr {d[d.train_len==tl]['id_corr'].mean():.2f})")
    ax.axhline(0, color="k", lw=0.9, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("eval length / training length")
    ax.set_ylabel(f"{eval_type} corr, relative to in-distribution (%)")
    ax.set_title("Does the drop track the length RATIO?\n"
                 "overlapping curves = yes; separated = absolute length is what matters")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"relative_generalization_{eval_type}.png", dpi=150)
    plt.close(fig)
    return g


def main(exp_group="mlp_lengen", gate="off", expected_cells=None):
    out_dir = Path(f"{ROOT}/outputs/plots/{exp_group}")
    out_dir.mkdir(parents=True, exist_ok=True)
    b, pooled = load_sweep(exp_group, gate=gate)
    report(b, expected_cells)
    b.to_csv(out_dir / "length_generalization_bins.csv", index=False)
    if not pooled.empty:
        pooled.to_csv(out_dir / "length_generalization_pooled.csv", index=False)
    for et in ("interventional", "endogenous"):
        plot_reach_curves(b, out_dir, et)
        plot_matched_reach(b, out_dir, et)
        agg = plot_gap_heatmap(b, out_dir, et)
        if agg is not None:
            agg.to_csv(out_dir / f"gap_by_cell_{et}.csv", index=False)
    plot_ceiling(b, out_dir)
    for et in ("interventional", "endogenous"):
        plot_variance_sources(b, out_dir, et)
        if not pooled.empty:
            plot_relative_generalization(pooled, out_dir, et)
    variance_decomposition(b, out_dir)
    print(f"\n[lengen] wrote plots to {out_dir}")


if __name__ == "__main__":
    EXPECTED = [f"len{l}/win{w}" for l, ws in
                ((16, [16]), (32, [16, 32]), (64, [16, 32, 64]), (128, [16, 32, 64, 128]))
                for w in ws]
    main(exp_group=sys.argv[1] if len(sys.argv) > 1 else "mlp_lengen",
         expected_cells=EXPECTED)
