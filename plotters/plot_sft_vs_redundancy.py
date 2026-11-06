#!/usr/bin/env python3
"""
Does same-layer redundancy predict how causally faithful SFT introspection is?

Joins each grid neuron's same-layer redundancy (leave-one-out R^2 from every other
neuron of its layer; mechanistic_analysis/score_neuron_redundancy.py via
scripts/neuron/score_neuron_redundancy_sft_grid.sh) with its SFT results on the
layer x token grid, and correlates them.

THE CONFOUND IS LAYER. Interventional corr falls steeply with depth and R^2 may well
change with depth too, so a pooled correlation can be nothing but "early layers
differ". The headline is therefore WITHIN-LAYER: layer means removed from both
variables (equivalently, layer fixed effects). The partial correlation additionally
controls the neuron's scale (log std) and firing rate, the covariates the earlier
redundancy analysis used.

UNIT OF ANALYSIS: the neuron, averaged over the 8 token positions. Position explains
~5% of the SFT interventional variance and neuron identity ~67%, so averaging loses
little and stops 8 correlated cells per neuron from passing as independent points.

Inputs
  outputs/plots/mlp_endo_sweep_layer_token/redundancy/L*/neuron_redundancy.csv
  outputs/plots/mlp_endo_sweep_layer_token_n10/sweep_runs_both_gates.csv  (80 neurons;
      falls back to the 5-neuron table)
Outputs (--out-dir, default outputs/plots/mlp_endo_sweep_layer_token/redundancy/)
  sft_vs_redundancy.csv          one row per neuron
  sft_vs_redundancy_stats.csv    pooled / within-layer / partial / per-layer stats
  sft_vs_redundancy_gate<g>.png  pooled and within-layer scatters per outcome
  redundancy_by_layer.png        each layer's R^2 distribution, grid neurons marked

Usage
  python plotters/plot_sft_vs_redundancy.py [--gate off|on]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, t as tdist

ROOT = Path(__file__).resolve().parents[1]
RED_DIR = ROOT / "outputs/plots/mlp_endo_sweep_layer_token/redundancy"
SFT_TABLES = [ROOT / "outputs/plots/mlp_endo_sweep_layer_token_n10/sweep_runs_both_gates.csv",
              ROOT / "outputs/plots/mlp_endo_sweep_layer_token/sweep_runs_both_gates.csv"]
OUTCOMES = [("interventional", "Interventional Corr"), ("endogenous", "Endogenous Corr"),
            ("mean", "Mean of Both")]
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate", default="off", choices=["off", "on"])
    p.add_argument("--red-dir", type=Path, default=RED_DIR)
    p.add_argument("--sft-table", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=RED_DIR)
    return p.parse_args()


def load_redundancy(red_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(red_dir.glob("L*/neuron_redundancy.csv")):
        d = pd.read_csv(f).rename(columns={"neuron": "index"})
        d["layer"] = int(f.parent.name[1:])
        # where each neuron sits in its own layer's R^2 distribution
        d["R2_pct"] = d["R2"].rank(pct=True)
        rows.append(d)
    if not rows:
        raise SystemExit(f"no neuron_redundancy.csv under {red_dir}/L*/ -- run "
                         f"scripts/neuron/score_neuron_redundancy_sft_grid.sh first")
    return pd.concat(rows, ignore_index=True)


def load_sft(path, gate) -> pd.DataFrame:
    paths = [path] if path else [p for p in SFT_TABLES if p.exists()]
    if not paths:
        raise SystemExit("no SFT sweep table found")
    d = pd.read_csv(paths[0])
    d = d[(d["metric"] == "corr") & (d["gate"] == gate)]
    per = (d.pivot_table(index=["neuron", "layer", "index"], columns="eval_type",
                         values="value", aggfunc="mean").reset_index())
    per["mean"] = per[["endogenous", "interventional"]].mean(axis=1)
    per["n_positions"] = d.groupby("neuron")["position"].nunique().reindex(per["neuron"]).values
    print(f"[sft-vs-red] {paths[0].relative_to(ROOT)}: {len(per)} neurons, gate {gate}")
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


def stats_table(m: pd.DataFrame) -> pd.DataFrame:
    out = []
    x = m["R2"].values
    D = layer_dummies(m["layer"].values)
    cov = np.column_stack([D, np.log(m["std"].values), m["firing_rate"].values])
    for y_col, _ in OUTCOMES:
        y = m[y_col].values
        rho, p = spearmanr(x, y)
        out.append(dict(outcome=y_col, test="pooled spearman", r=rho, p=p, n=len(m)))
        r, p, dof = partial_corr(x, y, D)
        out.append(dict(outcome=y_col, test="within-layer (layer FE)", r=r, p=p, n=len(m), dof=dof))
        r, p, dof = partial_corr(x, y, cov)
        out.append(dict(outcome=y_col, test="partial (layer FE + log std + firing)",
                        r=r, p=p, n=len(m), dof=dof))
        for L, g in m.groupby("layer"):
            if len(g) >= 4:
                rho, p = spearmanr(g["R2"], g[y_col])
                out.append(dict(outcome=y_col, test=f"layer {L} spearman", r=rho, p=p, n=len(g)))
    return pd.DataFrame(out)


def plot_scatter(m, st, out_png, gate):
    layers = sorted(m["layer"].unique())
    cmap = plt.cm.viridis(np.linspace(0, 0.95, len(layers)))
    col = dict(zip(layers, cmap))
    fig, axes = plt.subplots(2, len(OUTCOMES), figsize=(5.6 * len(OUTCOMES), 10))
    xd = m["R2"] - m.groupby("layer")["R2"].transform("mean")
    for j, (y_col, y_lab) in enumerate(OUTCOMES):
        yd = m[y_col] - m.groupby("layer")[y_col].transform("mean")
        for L in layers:
            g = m["layer"] == L
            axes[0, j].scatter(m.loc[g, "R2"], m.loc[g, y_col], color=col[L], s=40,
                               label=f"L{L}", edgecolors="k", linewidths=0.3)
            axes[1, j].scatter(xd[g], yd[g], color=col[L], s=40, edgecolors="k",
                               linewidths=0.3)
        s = st[st["outcome"] == y_col].set_index("test")
        a = s.loc["pooled spearman"]
        axes[0, j].set_title(f"{y_lab}\nPooled ρ = {a.r:+.2f} (p = {a.p:.2g})")
        b, c = s.loc["within-layer (layer FE)"], s.loc["partial (layer FE + log std + firing)"]
        axes[1, j].set_title(f"Within Layer r = {b.r:+.2f} (p = {b.p:.2g})\n"
                             f"+ Scale, Firing r = {c.r:+.2f} (p = {c.p:.2g})")
        for ax, yl in ((axes[0, j], y_lab), (axes[1, j], f"{y_lab} (Layer Mean Removed)")):
            ax.set_ylabel(yl)
            ax.grid(alpha=0.25)
        axes[0, j].set_xlabel("Same-Layer Redundancy R²")
        axes[1, j].set_xlabel("R² (Layer Mean Removed)")
        axes[1, j].axhline(0, color="k", lw=0.7)
        axes[1, j].axvline(0, color="k", lw=0.7)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False,
               bbox_to_anchor=(0.5, -0.02), title="Signal Layer")
    fig.suptitle(f"SFT (Gate {gate.capitalize()}): Faithfulness vs Same-Layer Redundancy, "
                 f"{len(m)} Neurons", fontsize=16)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_distributions(red, m, out_png):
    layers = sorted(red["layer"].unique())
    fig, ax = plt.subplots(figsize=(1.3 * len(layers) + 3, 5))
    data = [red[(red["layer"] == L) & red["kept"]]["R2"].dropna().values for L in layers]
    ax.violinplot(data, positions=range(len(layers)), showmedians=True, widths=0.8)
    for k, L in enumerate(layers):
        g = m[m["layer"] == L]
        ax.scatter(np.full(len(g), k) + np.random.default_rng(k).uniform(-.08, .08, len(g)),
                   g["R2"], color="tab:red", s=22, zorder=3,
                   label="Grid Neurons" if k == 0 else None)
    ax.set_xticks(range(len(layers)), [f"L{L}" for L in layers])
    ax.set_ylabel("Same-Layer Redundancy R²")
    ax.set_title("All Neurons of Each Layer vs the Introspected Ones")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    red = load_redundancy(args.red_dir)
    sft = load_sft(args.sft_table, args.gate)
    m = sft.merge(red[["layer", "index", "R2", "R2_pct", "std", "firing_rate", "kept"]],
                  on=["layer", "index"], how="left")
    missing = m[m["R2"].isna()]
    if len(missing):
        print(f"[sft-vs-red] {len(missing)} neurons without R2 (layer not scored, or dropped "
              f"as rarely firing): {missing['neuron'].tolist()[:10]}")
    m = m.dropna(subset=["R2", "endogenous", "interventional"])
    if m["layer"].nunique() < 2 or len(m) < 6:
        raise SystemExit(f"only {len(m)} neurons over {m['layer'].nunique()} layer(s) have "
                         f"both scores -- run the remaining layers first")
    m.to_csv(args.out_dir / "sft_vs_redundancy.csv", index=False)
    st = stats_table(m)
    st.to_csv(args.out_dir / "sft_vs_redundancy_stats.csv", index=False)
    print(st[~st["test"].str.startswith("layer")].to_string(
        index=False, float_format=lambda v: f"{v:.3g}"))
    plot_scatter(m, st, args.out_dir / f"sft_vs_redundancy_gate{args.gate}.png", args.gate)
    plot_layer_distributions(red, m, args.out_dir / "redundancy_by_layer.png")
    print(f"[sft-vs-red] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
