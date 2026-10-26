#!/usr/bin/env python3
"""
Do the endogenous and interventional evaluations rely on the same heads? -- pooled over
a sweep's mechanistic_analysis/score_head_knockout.py outputs.

Per run (one point each), against signal layer and position. Sign convention
everywhere: a head is important when its knockout HURTS (positive drop); heads whose
knockout improves an eval never count as important.
  TOP-HEAD OVERLAP  each arm's q-MASS SET -- the fewest heads whose positive drops sum
                    to q (default 0.5) of that arm's total positive drop -- and the
                    overlap coefficient |A & B| / min(|A|, |B|) between the two arms'
                    sets, vs a layer-matched null (heads shuffled within their layer).
                    q is kept well below 1: the endogenous drop has a long tail of tiny
                    precisely-measured effects, so a 95% set can hold most heads and the
                    null overlap then sits near 0.7.
  TOP-k OVERLAP     Jaccard of the two arms' top-k heads vs the same null.
  TRANSFER          cumulative knockout down arm B's ranking, evaluated on arm A, as a
                    fraction of the damage done by the BETTER of the two rankings on A
                    (1 = B's heads break A at least as well as A's own ranking; random
                    orderings give the floor). In [0, 1] by construction.
  EFFECTIVE HEADS   participation ratio of the drops, positive drops only (the heads
                    that carry the eval) and |drop| (also counting heads whose knockout
                    helps -- most heads in the interventional arm, consistent with them
                    adding text-dependent noise to a prediction whose target ignores the
                    text).

Across runs:
  GENERIC HEADS     for every (layer, head), the fraction of runs it is eligible in
                    (layer > signal layer) where it lands in an arm's top-k. A head in the
                    top-k of most runs, whatever the signal, is a candidate signal-agnostic
                    (task) head; chance is k / (eligible heads) per run.

Usage
-----
python plotters/plot_head_knockout.py --group mlp_joint_grid_layer_token --kl kl0.1
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotters.run_catalog import NEURON_RE, POS_RE, KL_RE  # noqa: E402  (one name parser)
from src.introspector.knockout_stats import agreement_from_tables  # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
A, B = "endogenous", "x1"
LABEL = {A: "Endogenous", B: "Interventional"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", required=True, help="Experiment group under the runs root.")
    p.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    p.add_argument("--kl", type=str, default=None,
                   help="Keep only runs whose name carries this KL tag (e.g. kl0.1).")
    p.add_argument("--analysis", type=str, default="head_knockout",
                   help="Per-run subdirectory (head_knockout_<window> for other windows).")
    p.add_argument("--k", type=int, default=10, help="Top-k for overlap and prevalence.")
    p.add_argument("--mass", type=float, default=0.5,
                   help="Mass fraction q of the head sets whose overlap is plotted (0.8 "
                        "and 0.95 are always recorded in the CSV too).")
    p.add_argument("--n-perm", type=int, default=1000,
                   help="Permutations for the layer-matched nulls.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: outputs/plots/<group>/head_knockout")
    p.add_argument("--replot", action="store_true",
                   help="Redraw the figures from knockout_runs.csv / "
                        "knockout_head_prevalence.csv in the output dir (no run dirs).")
    return p.parse_args()


# ======================================================================
# Loading
# ======================================================================
def load(args):
    root = args.runs_root / args.group
    if not root.exists():
        raise SystemExit(f"No such group: {root}")
    runs, heads, missing = [], [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        n, pz = NEURON_RE.search(d.name), POS_RE.search(d.name)
        if not (n and pz):
            continue
        if args.kl:
            m = KL_RE.search(d.name)
            if not m or f"kl{m['kl']}" != args.kl:
                continue
        kd = d / args.analysis
        s_p, i_p = kd / "head_knockout_summary.json", kd / "head_knockout_individual.csv"
        if not (s_p.exists() and i_p.exists()):
            missing.append(d.name)
            continue
        s = json.loads(s_p.read_text())
        ind = pd.read_csv(i_p)
        c_p = kd / "head_knockout_curves.csv"
        # recomputed from the tables, not read from the summary, so every run is scored
        # under the CURRENT definitions whatever version wrote it
        t = agreement_from_tables(ind, pd.read_csv(c_p) if c_p.exists() else None,
                                  mass_qs=sorted({args.mass, 0.5, 0.8, 0.95}),
                                  ks=(5, 10, 20), n_perm=args.n_perm)
        loc = {"run": d.name, "signal_layer": int(n["layer"]),
               "signal_index": int(n["index"]), "position": int(pz["pos"])}
        row = dict(loc, n_heads=s["n_heads"], n_examples=s["n_examples"],
                   **{f"baseline_{a}": s["baseline"].get(a) for a in (A, B)},
                   **{f"reliability_{a}": s.get(f"reliability_{a}", np.nan) for a in (A, B)})
        for key, v in t.items():
            if not isinstance(v, (list, dict)):
                row[key] = v
        for m in t.get(f"mass_overlap_{A}_vs_{B}", []):
            q = f"{m['q']:g}"
            row.update({f"mass{q}_overlap": m["overlap"], f"mass{q}_null": m["null_mean"],
                        f"mass{q}_p": m["p"]})
        for tk in t.get(f"topk_{A}_vs_{B}", []):
            row[f"jaccard_top{tk['k']}"] = tk["jaccard"]
            row[f"null_top{tk['k']}"] = tk["null_mean"]
            row[f"p_top{tk['k']}"] = tk["p"]
        runs.append(row)
        heads.append(ind.assign(**loc))
    print(f"[knockout-agg] {len(runs)} run(s) with knockout results under {root}")
    if missing:
        print(f"[knockout-agg] {len(missing)} run(s) without results, e.g. {missing[:3]}")
    if not runs:
        raise SystemExit("Nothing to aggregate.")
    return pd.DataFrame(runs), pd.concat(heads, ignore_index=True)


# ======================================================================
# Per-run agreement against locale
# ======================================================================
def _agreement_panels(k: int, q: float) -> list:
    qs = f"{q:g}"
    rnd = "Random Layer-Matched Heads"
    return [
        ("Overlap", f"Top-Head Overlap Coeff.\n({100 * q:g}% Mass)",
         [(f"mass{qs}_overlap", "Observed", "#c0392b"),
          (f"mass{qs}_null", rnd, "#bdc3c7")], (-0.05, 1.05)),
        ("Overlap", f"Top-{k} Jaccard", [(f"jaccard_top{k}", "Observed", "#2471a3"),
                              (f"null_top{k}", rnd, "#bdc3c7")], (-0.05, 1.05)),
        ("Functional Similarity", "Transfer Ratio", [(f"transfer_{A}_to_{B}", f"{LABEL[A]}-Ranked, on {LABEL[B]}", "#c0392b"),
                            (f"transfer_{B}_to_{A}", f"{LABEL[B]}-Ranked, on {LABEL[A]}", "#2471a3"),
                            (f"transfer_random_to_{B}", f"Random, on {LABEL[B]}", "#bdc3c7")],
         (-0.05, 1.05)),
        ("Sparsity", "Effective Heads\n(Positive Drops)",
         [(f"effective_heads_pos_{A}", LABEL[A], "#c0392b"),
          (f"effective_heads_pos_{B}", LABEL[B], "#2471a3")], None),
        ("Sparsity", "Effective Heads\n(|Drop|)",
         [(f"effective_heads_abs_{A}", LABEL[A], "#c0392b"),
          (f"effective_heads_abs_{B}", LABEL[B], "#2471a3")], None),
    ]


# font sizes: the combined overview figure vs the per-axis figures meant for the paper
FONTS_OVERVIEW = {"title": 10, "label": 11, "tick": 9, "legend": 7, "point": 8, "median": 200}
FONTS_PAPER = {"title": 18, "label": 18, "tick": 16, "legend": 15, "point": 30, "median": 500}


def _draw_row(axes, runs, xcol, xlabel, panels, fonts, titles=True, legends=True,
              legend_below=False, one_line_titles=False, seed=0):
    """One row of panels: every run a point (jittered), the median a bar."""
    rng = np.random.default_rng(seed)
    levels = sorted(runs[xcol].unique())
    pos = {v: i for i, v in enumerate(levels)}
    for ax, (title, ylabel, series, ylim) in zip(axes, panels):
        width = 0.8 / len(series)
        for si, (col, label, color) in enumerate(series):
            if col not in runs:
                continue
            off = (si - (len(series) - 1) / 2) * width
            x = runs[xcol].map(pos) + off
            ax.scatter(x + rng.uniform(-width / 4, width / 4, len(x)), runs[col],
                       s=fonts["point"], color=color, alpha=0.5, label=label)
            med = runs.groupby(xcol)[col].median()
            mx = [pos[v] + off for v in med.index]
            lw = fonts.get("median_lw", 2.5)
            if fonts.get("median_edge"):
                # dark outline under the bar, so it stands out of the point cloud
                ax.scatter(mx, med.values, marker="_", s=fonts["median"] * 1.08, color="k",
                           linewidths=lw + 3, zorder=4)
            ax.scatter(mx, med.values, marker="_", s=fonts["median"], color=color,
                       linewidths=lw, zorder=5)
        ax.set_xticks(range(len(levels)), levels)
        ax.set_xlabel(xlabel, fontsize=fonts["label"])
        ax.set_ylabel(ylabel, fontsize=fonts["label"])
        ax.tick_params(labelsize=fonts["tick"])
        if titles:
            ax.set_title(title.replace("\n", " ") if one_line_titles else title,
                         fontsize=fonts["title"])
        if ylim is not None:
            # bounded metrics: fixed axis so one outlier cannot set the scale
            ax.set_ylim(*ylim)
            ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.25)
        if legends:
            if legend_below:
                # below the axes, so a larger legend never sits on the points
                ax.legend(fontsize=fonts["legend"], loc="upper center",
                          bbox_to_anchor=(0.5, -0.3), frameon=False, markerscale=1.5)
            else:
                ax.legend(fontsize=fonts["legend"], loc="best")


SEQ_LEN = 32      # token positions are drawn as 0-indexed absolute indices (-1 -> 31)


def agreement_by_locale(runs: pd.DataFrame, k: int, q: float, out: Path):
    """The overview (both locale axes as two rows) plus one figure per locale axis, all in
    the paper style: large fonts, two-line titles, legends below, no suptitle."""
    panels = _agreement_panels(k, q)
    runs = runs.assign(position=runs["position"].where(runs["position"] >= 0,
                                                       runs["position"] + SEQ_LEN))
    xs = [("signal_layer", "Signal Layer", "knockout_agreement_by_signal_layer.png"),
          ("position", "Token Position", "knockout_agreement_by_position.png")]

    fig, axes = plt.subplots(len(xs), len(panels), figsize=(4.6 * len(panels), 5.2 * len(xs)),
                             squeeze=False)
    for r, (xcol, xlabel, _) in enumerate(xs):
        last = r == len(xs) - 1
        # titles once (top row); legends once per column, below the bottom row
        _draw_row(axes[r], runs, xcol, xlabel, panels, FONTS_PAPER,
                  titles=r == 0, legends=last, legend_below=last)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.2, wspace=0.48, hspace=0.42)
    fig.savefig(out / "knockout_agreement_by_locale.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    agreement_grid(runs, k, q, out)

    for xcol, xlabel, fname in xs:
        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 6.2),
                                 squeeze=False)
        _draw_row(axes[0], runs, xcol, xlabel, panels, FONTS_PAPER, legend_below=True)
        # explicit margins: tight_layout shrinks the axes to make room for the legends
        # below them, which squashes the plots; bbox_inches="tight" trims the rest
        fig.subplots_adjust(left=0.04, right=0.99, top=0.86, bottom=0.36, wspace=0.48)
        fig.savefig(out / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)


FONTS_GRID = {"title": 32, "label": 28, "tick": 25, "legend": 26, "point": 55,
              "median": 900, "median_lw": 6, "median_edge": True, "letter": 50}


def _panel_letter(ax, letter, fonts):
    """Bold lower-case panel letter tucked into the axes' bottom-left corner: below the
    y tick labels and left of the x tick labels, so it adds no margin of its own."""
    ax.annotate(letter, xy=(0, 0), xycoords="axes fraction", xytext=(-8, -6),
                textcoords="offset points", fontsize=fonts["letter"], fontweight="bold",
                ha="right", va="top")


def _header(fig, x, y, title, handles, labels, fonts, below=False):
    """Bold title at (x, y) (figure fraction, text baseline), its legend on the same line
    right after it -- or, below=True, stacked one entry per line under it."""
    t = fig.text(x, y, title, fontsize=fonts["title"], fontweight="bold",
                 ha="left", va="baseline")
    kw = dict(fontsize=fonts["legend"], frameon=False, markerscale=1.6, borderpad=0,
              borderaxespad=0, handletextpad=0.2, columnspacing=1.0)
    if below:
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x, y - 0.012),
                   ncol=1, labelspacing=0.25, **kw)
        return
    bb = t.get_window_extent(fig.canvas.get_renderer())
    fig.legend(handles, labels, loc="lower left",
               bbox_to_anchor=(bb.x1 / fig.bbox.width + 0.012, y - 0.006),
               ncol=len(labels), **kw)


def agreement_grid(runs: pd.DataFrame, k: int, q: float, out: Path,
                   xcol="signal_layer", xlabel="Signal Layer",
                   fname="knockout_agreement_by_signal_layer_grid.png"):
    """The same five panels as knockout_agreement_by_signal_layer.png, larger and grouped
    by purpose: (a, b) the two overlap panels top left, (c, d) the two sparsity panels
    below them, (e) transfer -- functional similarity -- as one big panel spanning both
    rows on the right. Each group has one bold title with its legend beside it; the two
    overlap panels draw "Observed" in one colour so one legend serves both.

    Laid out by hand, in inches, because constrained layout cannot set a legend on the
    same line as a title."""
    over_mass, over_jac, transfer, sp_pos, sp_abs = _agreement_panels(k, q)
    observed = "#7d3c98"     # not red / blue, which mean Endogenous / Interventional here
    recolor = lambda pnl: (pnl[0], pnl[1], [(c, l, observed if l == "Observed" else col)  # noqa: E731
                                            for c, l, col in pnl[2]], pnl[3])
    over_mass, over_jac = recolor(over_mass), recolor(over_jac)
    f = FONTS_GRID

    W, H = 28.8, 12.8
    aw, ah = 6.4, 4.0                      # small panels
    x_a, x_b, x_e = 2.1, 10.9, 19.6        # left edges (room for 2-line y-labels)
    y_bot, y_top = 1.4, 7.4                # bottom edges of the two rows
    top = y_top + ah
    fig = plt.figure(figsize=(W, H))
    rect = lambda x, y, w, h: [x / W, y / H, w / W, h / H]  # noqa: E731
    ax_a, ax_b = fig.add_axes(rect(x_a, y_top, aw, ah)), fig.add_axes(rect(x_b, y_top, aw, ah))
    ax_c = fig.add_axes(rect(x_a, y_bot, aw, ah))
    ax_d = fig.add_axes(rect(x_b, y_bot, aw, ah), sharey=ax_c)
    e_top = top - 1.5                      # the stacked 3-entry legend sits above (e)
    ax_e = fig.add_axes(rect(x_e, y_bot, W - x_e - 0.35, e_top - y_bot))

    _draw_row([ax_a, ax_b], runs, xcol, xlabel, [over_mass, over_jac], f,
              titles=False, legends=False)
    _draw_row([ax_c, ax_d], runs, xcol, xlabel, [sp_pos, sp_abs], f,
              titles=False, legends=False)
    _draw_row([ax_e], runs, xcol, xlabel, [transfer], f, titles=False, legends=False)

    for x0, y0, title, ax in ((x_a, top, "Overlap", ax_a), (x_a, y_bot + ah, "Sparsity", ax_c)):
        _header(fig, x0 / W, (y0 + 0.3) / H, title, *ax.get_legend_handles_labels(), f)
    _header(fig, x_e / W, (top + 0.3) / H, transfer[0], *ax_e.get_legend_handles_labels(),
            f, below=True)
    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d, ax_e), "abcde"):
        _panel_letter(ax, letter, f)
    fig.savefig(out / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pooled_table(runs: pd.DataFrame, k: int, q: float) -> pd.DataFrame:
    qs = f"{q:g}"
    cols = [f"baseline_{A}", f"baseline_{B}", f"reliability_{A}", f"reliability_{B}",
            f"mass{qs}_size_{A}", f"mass{qs}_size_{B}", f"mass{qs}_overlap", f"mass{qs}_null",
            f"jaccard_top{k}", f"null_top{k}", f"transfer_{A}_to_{B}", f"transfer_{B}_to_{A}",
            f"transfer_random_to_{A}", f"transfer_random_to_{B}",
            f"effective_heads_pos_{A}", f"effective_heads_pos_{B}",
            f"effective_heads_abs_{A}", f"effective_heads_abs_{B}",
            f"n_helped_by_knockout_{A}", f"n_helped_by_knockout_{B}"]
    cols = [c for c in cols if c in runs]
    tab = runs[cols].quantile([0.25, 0.5, 0.75]).T
    tab.columns = ["q25", "median", "q75"]
    tab["n_runs"] = runs[cols].notna().sum()
    if f"mass{qs}_p" in runs:
        tab.loc[f"frac_runs_mass{qs}_p<0.05", ["median", "n_runs"]] = [
            float((runs[f"mass{qs}_p"] < 0.05).mean()), int(runs[f"mass{qs}_p"].notna().sum())]
    if f"p_top{k}" in runs:
        tab.loc[f"frac_runs_top{k}_p<0.05", ["median", "n_runs"]] = [
            float((runs[f"p_top{k}"] < 0.05).mean()), int(runs[f"p_top{k}"].notna().sum())]
    return tab


# ======================================================================
# Generic heads: prevalence in the top-k across runs
# ======================================================================
def prevalence(heads: pd.DataFrame, k: int, out: Path) -> pd.DataFrame:
    rows = []
    for arm, g in heads.groupby("arm"):
        top = g[g["rank"] <= k]
        n_elig = g.groupby(["layer", "head"])["run"].nunique().rename("n_eligible")
        n_top = top.groupby(["layer", "head"])["run"].nunique().rename("n_top")
        # chance of a head landing in a run's top-k = k / that run's eligible heads,
        # averaged over the runs it is eligible in
        per_run = g.groupby("run")["head"].size().rename("n_heads_run")
        chance = (g[["run", "layer", "head"]].merge(per_run, on="run")
                  .assign(c=lambda d: np.minimum(1, k / d["n_heads_run"]))
                  .groupby(["layer", "head"])["c"].mean().rename("chance"))
        d = pd.concat([n_elig, n_top, chance], axis=1).fillna({"n_top": 0}).reset_index()
        d["arm"] = arm
        d["prevalence"] = d["n_top"] / d["n_eligible"]
        rows.append(d)
    prev = pd.concat(rows, ignore_index=True)
    plot_prevalence(prev, k, out)
    return prev


def plot_prevalence(prev: pd.DataFrame, k: int, out: Path):
    """Heatmap per arm of how often each head is in a run's top-k. Separate from
    prevalence() so the figure can be redrawn from knockout_head_prevalence.csv."""
    arms = [a for a in (A, B) if a in set(prev["arm"])]
    n_layers = int(prev["layer"].max()) + 1
    n_heads = int(prev["head"].max()) + 1
    lo = int(prev["layer"].min())
    fig, axes = plt.subplots(len(arms), 1, squeeze=False,
                             figsize=(max(6, 0.3 * n_heads), 0.6 + len(arms) * (0.3 * (n_layers - lo) + 1)))
    for ax, arm in zip(axes[:, 0], arms):
        m = (prev[prev["arm"] == arm].pivot(index="layer", columns="head", values="prevalence")
             .reindex(index=range(lo, n_layers), columns=range(n_heads)))
        hm = sns.heatmap(m, ax=ax, cmap="magma_r", vmin=0, vmax=1,
                         cbar_kws={"label": f"Fraction of Runs in Top-{k}"})
        hm.figure.axes[-1].yaxis.label.set_size(16)          # colorbar label
        hm.figure.axes[-1].tick_params(labelsize=13)
        # short title names the arm; "of runs where the head is above the signal layer"
        # (the denominator) belongs in the caption
        ax.set_title(f"{LABEL.get(arm, arm)} Head Prevalence", fontsize=20)
        ax.set_ylabel("Layer", fontsize=17); ax.set_xlabel("Head", fontsize=17)
        ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(out / "knockout_head_prevalence.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out = args.out_dir or ROOT / "outputs" / "plots" / args.group / args.analysis
    out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        agreement_by_locale(pd.read_csv(out / "knockout_runs.csv"), args.k, args.mass, out)
        if (out / "knockout_head_prevalence.csv").exists():
            plot_prevalence(pd.read_csv(out / "knockout_head_prevalence.csv"), args.k, out)
        print(f"[knockout-agg] redrew figures in {out}")
        return
    runs, heads = load(args)
    runs.to_csv(out / "knockout_runs.csv", index=False)
    agreement_by_locale(runs, args.k, args.mass, out)
    tab = pooled_table(runs, args.k, args.mass)
    tab.to_csv(out / "knockout_pooled.csv")
    print("\n[knockout-agg] pooled over runs:\n" + tab.to_string(float_format=lambda v: f"{v:.3f}"))
    prev = prevalence(heads, args.k, out)
    prev.to_csv(out / "knockout_head_prevalence.csv", index=False)
    for arm, g in prev.groupby("arm"):
        g = g[g["n_eligible"] >= 5].sort_values("prevalence", ascending=False).head(8)
        print(f"\n[knockout-agg] {arm}: most prevalent top-{args.k} heads (>= 5 eligible runs)")
        print(g[["layer", "head", "n_eligible", "n_top", "prevalence", "chance"]]
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n[knockout-agg] wrote {out}")


if __name__ == "__main__":
    main()
