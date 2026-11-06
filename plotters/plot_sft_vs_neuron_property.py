#!/usr/bin/env python3
"""
Does a property of the introspected neuron predict how causally faithful SFT
introspection on it is?

Joins one per-neuron property with each grid neuron's SFT results on the layer x token
grid and correlates them. Two properties so far:

  context      (--metric context, default x: n_eff) how much preceding context the
               neuron needs: mechanistic_analysis/score_neuron_context_dependence.py.
               Other columns of that file can be chosen with --x (r2_1: R2 with BOS +
               the current token only; r2_token_table: token-identity lookup table;
               r2_nmax: R2 at the longest truncation).
  redundancy   (--metric redundancy, x: R2) leave-one-out R2 of the neuron from every
               other neuron of its layer: mechanistic_analysis/score_neuron_redundancy.py
               via scripts/neuron/score_neuron_redundancy_sft_grid.sh.

THE CONFOUND IS LAYER. Interventional corr falls steeply with depth and these
properties change with depth too, so a pooled correlation can be nothing but "early
layers differ". The headline is therefore WITHIN-LAYER: layer means removed from both
variables (layer fixed effects). The partial correlation additionally controls the
neuron's scale (log std) and how often it fires (firing rate for redundancy, fraction
positive for context), so a sparsity effect cannot pass for the property.

UNIT OF ANALYSIS: the neuron, averaged over the 8 token positions. Position explains
~5% of the SFT interventional variance and neuron identity ~67%, so averaging loses
little and stops 8 correlated cells per neuron from passing as independent points.

Inputs
  the metric's score file(s) (see METRICS)
  outputs/plots/mlp_endo_sweep_layer_token_n10/sweep_runs_both_gates.csv  (80 neurons;
      falls back to the 5-neuron table)
Outputs (in the metric's directory unless --out-dir)
  sft_vs_<metric>_<x>.csv             one row per neuron
  sft_vs_<metric>_<x>_stats.csv       pooled / within-layer / partial / per-layer stats
  sft_vs_<metric>_<x>_gate<g>.png     pooled and within-layer scatters per outcome
  <metric>_<x>_by_layer.png           each layer's distribution, grid neurons marked
  <metric>_<x>_density_by_layer.png   one density line per layer on a shared axis

Usage
  python plotters/plot_sft_vs_neuron_property.py --metric context [--x n_eff] [--gate off]
  python plotters/plot_sft_vs_neuron_property.py --metric redundancy
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, gaussian_kde, t as tdist

ROOT = Path(__file__).resolve().parents[1]
GRID_PLOTS = ROOT / "outputs/plots/mlp_endo_sweep_layer_token"
SFT_TABLES = [ROOT / "outputs/plots/mlp_endo_sweep_layer_token_n10/sweep_runs_both_gates.csv",
              ROOT / "outputs/plots/mlp_endo_sweep_layer_token/sweep_runs_both_gates.csv"]
OUTCOMES = [("interventional", "Interventional Corr"), ("endogenous", "Endogenous Corr"),
            ("mean", "Mean of Both")]
X_LABELS = {"R2": "Same-Layer Redundancy R²",
            "n_eff": "Effective Context Length n_eff",
            "r2_1": "R² with Current Token Only",
            "r2_token_table": "Token-Identity R² (Lookup Table)",
            "r2_nmax": "R² at Longest Truncation"}
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14})


# ======================================================================
# Metrics: each loader returns one row per scored neuron with columns
#   layer, index, <x columns>, std, sparsity, is_reference
# is_reference marks neurons that only feed the layer distribution (all neurons of
# the layer for redundancy, the random neurons for context).
# ======================================================================
def load_redundancy(d: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(d.glob("L*/neuron_redundancy.csv")):
        x = pd.read_csv(f).rename(columns={"neuron": "index", "firing_rate": "sparsity"})
        x["layer"] = int(f.parent.name[1:])
        x.loc[~x["kept"], "R2"] = np.nan
        rows.append(x)
    if not rows:
        raise SystemExit(f"no neuron_redundancy.csv under {d}/L*/ -- run "
                         f"scripts/neuron/score_neuron_redundancy_sft_grid.sh first")
    out = pd.concat(rows, ignore_index=True)
    out["is_reference"] = True          # every neuron of the layer; grid ones picked by join
    return out


def load_context(d: Path) -> pd.DataFrame:
    f = d / "context_dependence.csv"
    if not f.exists():
        raise SystemExit(f"no {f} -- run mechanistic_analysis/score_neuron_context_dependence.py first")
    x = pd.read_csv(f).rename(columns={"frac_positive": "sparsity"})
    x["is_reference"] = ~x["is_grid"]
    return x


METRICS = {
    "redundancy": dict(dir=GRID_PLOTS / "redundancy", load=load_redundancy, x="R2",
                       sparsity="Firing Rate"),
    "context": dict(dir=GRID_PLOTS / "context_dependence", load=load_context, x="n_eff",
                    sparsity="Fraction Positive"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric", default="context", choices=sorted(METRICS))
    p.add_argument("--x", default=None, help="Column of the metric's file to correlate.")
    p.add_argument("--gate", default="off", choices=["off", "on"])
    p.add_argument("--score-dir", type=Path, default=None)
    p.add_argument("--sft-table", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def load_sft(path, gate) -> pd.DataFrame:
    paths = [path] if path else [p for p in SFT_TABLES if p.exists()]
    if not paths:
        raise SystemExit("no SFT sweep table found")
    d = pd.read_csv(paths[0])
    d = d[(d["metric"] == "corr") & (d["gate"] == gate)]
    per = (d.pivot_table(index=["neuron", "layer", "index"], columns="eval_type",
                         values="value", aggfunc="mean").reset_index())
    per["mean"] = per[["endogenous", "interventional"]].mean(axis=1)
    print(f"[sft-vs-prop] {Path(paths[0]).name}: {len(per)} neurons, gate {gate}")
    return per


def residualize(y, X):
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def partial_corr(x, y, Z):
    rx, ry = residualize(x, Z), residualize(y, Z)
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = len(x) - 2 - Z.shape[1]
    tstat = r * np.sqrt(dof / max(1 - r * r, 1e-12))
    return r, float(2 * tdist.sf(abs(tstat), dof)), dof


def layer_dummies(layers):
    u = sorted(set(layers))
    return np.column_stack([(layers == l).astype(float) for l in u[1:]])


FE, PARTIAL = "within-layer (layer FE)", "partial (layer FE + log std + sparsity)"


def stats_table(m: pd.DataFrame, xcol: str) -> pd.DataFrame:
    out = []
    x = m[xcol].values
    D = layer_dummies(m["layer"].values)
    cov = np.column_stack([D, np.log(m["std"].values), m["sparsity"].values])
    for y_col, _ in OUTCOMES:
        y = m[y_col].values
        rho, p = spearmanr(x, y)
        out.append(dict(outcome=y_col, test="pooled spearman", r=rho, p=p, n=len(m)))
        r, p, dof = partial_corr(x, y, D)
        out.append(dict(outcome=y_col, test=FE, r=r, p=p, n=len(m), dof=dof))
        r, p, dof = partial_corr(x, y, cov)
        out.append(dict(outcome=y_col, test=PARTIAL, r=r, p=p, n=len(m), dof=dof))
        for L, g in m.groupby("layer"):
            if len(g) >= 4:
                rho, p = spearmanr(g[xcol], g[y_col])
                out.append(dict(outcome=y_col, test=f"layer {L} spearman", r=rho, p=p, n=len(g)))
    return pd.DataFrame(out)


def plot_scatter(m, st, xcol, out_png, gate):
    layers = sorted(m["layer"].unique())
    col = dict(zip(layers, plt.cm.viridis(np.linspace(0, 0.95, len(layers)))))
    xlab = X_LABELS.get(xcol, xcol)
    fig, axes = plt.subplots(2, len(OUTCOMES), figsize=(5.6 * len(OUTCOMES), 10))
    xd = m[xcol] - m.groupby("layer")[xcol].transform("mean")
    for j, (y_col, y_lab) in enumerate(OUTCOMES):
        yd = m[y_col] - m.groupby("layer")[y_col].transform("mean")
        for L in layers:
            g = m["layer"] == L
            axes[0, j].scatter(m.loc[g, xcol], m.loc[g, y_col], color=col[L], s=40,
                               label=f"L{L}", edgecolors="k", linewidths=0.3)
            axes[1, j].scatter(xd[g], yd[g], color=col[L], s=40, edgecolors="k",
                               linewidths=0.3)
        s = st[st["outcome"] == y_col].set_index("test")
        a, b, c = s.loc["pooled spearman"], s.loc[FE], s.loc[PARTIAL]
        axes[0, j].set_title(f"{y_lab}\nPooled ρ = {a.r:+.2f} (p = {a.p:.2g})")
        axes[1, j].set_title(f"Within Layer r = {b.r:+.2f} (p = {b.p:.2g})\n"
                             f"+ Scale, Sparsity r = {c.r:+.2f} (p = {c.p:.2g})")
        for ax, yl in ((axes[0, j], y_lab), (axes[1, j], f"{y_lab} (Layer Mean Removed)")):
            ax.set_ylabel(yl)
            ax.grid(alpha=0.25)
        axes[0, j].set_xlabel(xlab)
        axes[1, j].set_xlabel(f"{xlab}\n(Layer Mean Removed)")
        axes[1, j].axhline(0, color="k", lw=0.7)
        axes[1, j].axvline(0, color="k", lw=0.7)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False,
               bbox_to_anchor=(0.5, -0.02), title="Signal Layer")
    fig.suptitle(f"SFT (Gate {gate.capitalize()}): Faithfulness vs {xlab}, "
                 f"{len(m)} Neurons", fontsize=16)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_distributions(scores, m, xcol, out_png):
    layers = sorted(m["layer"].unique())
    ref = scores[scores["is_reference"]]
    fig, ax = plt.subplots(figsize=(1.3 * len(layers) + 3, 5))
    data = [ref[ref["layer"] == L][xcol].dropna().values for L in layers]
    ok = [k for k, v in enumerate(data) if len(v)]
    if ok:
        ax.violinplot([data[k] for k in ok], positions=ok, showmedians=True, widths=0.8)
    for k, L in enumerate(layers):
        g = m[m["layer"] == L]
        ax.scatter(np.full(len(g), k) + np.random.default_rng(k).uniform(-.08, .08, len(g)),
                   g[xcol], color="tab:red", s=22, zorder=3,
                   label="Grid Neurons" if k == 0 else None)
    ax.set_xticks(range(len(layers)), [f"L{L}" for L in layers])
    ax.set_ylabel(X_LABELS.get(xcol, xcol))
    ax.set_title("Reference Neurons of Each Layer (Violins) vs the Introspected Ones")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_layer_density_lines(scores, m, xcol, out_png):
    """One density line per layer over that layer's reference neurons, so layers can be
    compared on one axis; the grid neurons as same-coloured ticks underneath. For an
    R2-type score bunched just below 1, a second panel shows the unexplained fraction
    1 - R2 on a log axis, where the layers actually separate."""
    layers = sorted(scores["layer"].unique())
    col = dict(zip(layers, plt.cm.viridis(np.linspace(0, 0.95, len(layers)))))
    ref = scores[scores["is_reference"]]
    panels = [(xcol, X_LABELS.get(xcol, xcol), False)]
    if xcol in ("R2",) or xcol.startswith("r2"):
        panels.append((xcol, f"1 − {X_LABELS.get(xcol, xcol)} (Unexplained, Log Scale)", True))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 5.2), squeeze=False)
    for ax, (c, lab, log_resid) in zip(axes[0], panels):
        tf = (lambda v: np.log10(np.clip(1 - v, 1e-6, None))) if log_resid else (lambda v: v)
        allv = tf(ref[c].dropna().values)
        lo, hi = np.percentile(allv, [0.5, 99.5])
        grid_x = np.linspace(lo, hi, 400)
        ymax = 0
        for L in layers:
            v = tf(ref[ref["layer"] == L][c].dropna().values)
            if len(v) < 5 or np.std(v) == 0:
                continue
            dens = gaussian_kde(v)(grid_x)
            ymax = max(ymax, dens.max())
            ax.plot(10 ** grid_x if log_resid else grid_x, dens, color=col[L], lw=2,
                    label=f"L{L} (median {np.median(ref[ref['layer'] == L][c]):.3g})")
        for k, L in enumerate(layers):         # grid neurons as ticks below zero
            g = tf(m[m["layer"] == L][c].dropna().values)
            y = -ymax * (0.04 + 0.035 * k)
            ax.scatter(10 ** g if log_resid else g, np.full(len(g), y), marker="|", s=90,
                       color=col[L], lw=1.6)
        ax.axhline(0, color="k", lw=0.6)
        if log_resid:      # the density is of log10(1 - R2), drawn against 1 - R2
            ax.set_xscale("log")
            ax.set_xlim(10 ** lo, 10 ** hi)
            ticks = [t for t in (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1)
                     if 10 ** lo <= t <= 10 ** hi]
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.set_xlim(lo, hi)
        ax.set_xlabel(lab)
        ax.set_ylabel("Density (Reference Neurons)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=10, title="Layer", loc="upper left")
    fig.suptitle("Lines: All Reference Neurons per Layer | Ticks Below: Grid Neurons",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    spec = METRICS[args.metric]
    xcol = args.x or spec["x"]
    score_dir = args.score_dir or spec["dir"]
    out_dir = args.out_dir or score_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = spec["load"](score_dir)
    if xcol not in scores:
        raise SystemExit(f"--x {xcol!r} not in {sorted(scores.columns)}")
    sft = load_sft(args.sft_table, args.gate)
    keep = ["layer", "index", xcol, "std", "sparsity"]
    m = sft.merge(scores[keep].drop_duplicates(["layer", "index"]), on=["layer", "index"],
                  how="left")
    missing = m[m[xcol].isna()]
    if len(missing):
        print(f"[sft-vs-prop] {len(missing)} neurons without {xcol}: "
              f"{missing['neuron'].tolist()[:10]}")
    m = m.dropna(subset=[xcol, "endogenous", "interventional"])
    if m["layer"].nunique() < 2 or len(m) < 6:
        raise SystemExit(f"only {len(m)} neurons over {m['layer'].nunique()} layer(s) have "
                         f"both scores")
    stem = f"sft_vs_{args.metric}_{xcol}"
    m.to_csv(out_dir / f"{stem}.csv", index=False)
    st = stats_table(m, xcol)
    st.to_csv(out_dir / f"{stem}_stats.csv", index=False)
    print(st[~st["test"].str.startswith("layer")].to_string(
        index=False, float_format=lambda v: f"{v:.3g}"))
    plot_scatter(m, st, xcol, out_dir / f"{stem}_gate{args.gate}.png", args.gate)
    plot_layer_distributions(scores, m, xcol, out_dir / f"{args.metric}_{xcol}_by_layer.png")
    plot_layer_density_lines(scores, m, xcol,
                             out_dir / f"{args.metric}_{xcol}_density_by_layer.png")
    print(f"[sft-vs-prop] wrote {out_dir}")


if __name__ == "__main__":
    main()
