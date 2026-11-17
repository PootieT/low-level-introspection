#!/usr/bin/env python3
"""
Which MLP site does SFT introspect most causally? Plots the mlp_site grid
(scripts/neuron/train_single_mlp_introspector_endogenous_sweep_mlp_site.sh) against the
main grid's act_fn runs at the same cells and neurons:

  gate     g = gate_proj(x)
  act_fn   a = silu(g)            main grid (mlp_endo_sweep_layer_token), filtered here
  down_in  h = a * up_proj(x)     the neuron's write is h * W_down[:, i]

Figures (outputs/plots/mlp_site_sweep_layer_token/):
  site_lines_<eval>_<metric>.png   rows = LoRA gate on / off, columns = site,
                                   x = token position, one line per layer (mean +- SE
                                   over the cell's 5 neurons), shared y
  site_summary_<metric>.png        x = site, one line per layer, averaged over positions
                                   (SE over neurons); endogenous and interventional
                                   side by side, one row per gate
  site_runs.csv                    the long table behind them (redraws without run dirs)

act_fn comes from outputs/runs/mlp_endo_sweep_layer_token when the run dirs are present,
otherwise from that grid's saved outputs/plots/.../sweep_runs_both_gates.csv. Either way
it is cut to the (neuron, position) pairs the site runs cover, so every site is the same
set of neurons.

Usage
  python plotters/plot_mlp_site_sweep.py                 # load runs, write csv + plots
  python plotters/plot_mlp_site_sweep.py --from-csv      # redraw from site_runs.csv
  python plotters/plot_mlp_site_sweep.py --metrics corr spearman
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_single_neuron_results import SEQ_LEN, load_endogenous_sweep  # noqa: E402

SITE_GROUP = "mlp_site_sweep_layer_token"
MAIN_GROUP = "mlp_endo_sweep_layer_token"
OUT = ROOT / "outputs/plots" / SITE_GROUP
SITES = ["gate", "act_fn", "down_in"]
SITE_WORDS = {"gate": "Gate Pre-Act. g", "act_fn": "silu(g) = a  (Main Grid)",
              "down_in": "a · up = h  (Down-Proj Input)"}
GATES = ["on", "off"]
EVALS = ["endogenous", "interventional"]
COLS = ["site", "gate", "neuron", "layer", "position", "eval_type", "metric", "value", "run"]
SITE_SHORT = {"gate": "g (gate)", "act_fn": "a (act_fn)", "down_in": "h (down_in)"}
# the imported module sets poster-size fonts globally; reset the ones it touches
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
                     "legend.title_fontsize": 13, "figure.titlesize": 15})


def _load_group(group, site):
    parts = []
    for gate in GATES:
        try:
            parts.append(load_endogenous_sweep(group, gate=gate, site=site))
        except SystemExit as e:
            print(f"[site] {group} site={site} gate={gate}: {e}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)


def load_all():
    if not (ROOT / "outputs/runs" / SITE_GROUP).is_dir():
        raise SystemExit(f"no run dirs under outputs/runs/{SITE_GROUP}; sync them, or pass "
                         f"--from-csv to redraw from {OUT / 'site_runs.csv'}")
    new = pd.concat([_load_group(SITE_GROUP, s) for s in ("gate", "down_in")],
                    ignore_index=True)
    if new.empty:
        raise SystemExit(f"no finished runs under outputs/runs/{SITE_GROUP}")

    main_runs = ROOT / "outputs/runs" / MAIN_GROUP
    if main_runs.is_dir() and any(main_runs.glob("endo_*")):
        act = _load_group(MAIN_GROUP, "act_fn")
        print(f"[site] act_fn from run dirs under {main_runs}")
    else:
        csv = ROOT / "outputs/plots" / MAIN_GROUP / "sweep_runs_both_gates.csv"
        act = pd.read_csv(csv).assign(site="act_fn")
        print(f"[site] act_fn from {csv} (run dirs not synced)")
    # same neurons and positions as the site runs (gate arm matched too)
    keep = new[["gate", "neuron", "position"]].drop_duplicates()
    act = act.merge(keep, on=["gate", "neuron", "position"])
    df = pd.concat([new[COLS], act[COLS]], ignore_index=True)
    df = df[df["eval_type"].isin(EVALS)]
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "site_runs.csv", index=False)
    print(f"[site] wrote {OUT / 'site_runs.csv'}")
    return df


def coverage(df):
    c = (df[(df["eval_type"] == "endogenous") & (df["metric"] == "corr")]
         .groupby(["gate", "site", "layer", "position"])["neuron"].nunique()
         .unstack(["layer", "position"]))
    print("[site] neurons per cell (endogenous corr):")
    print(c.reindex(pd.MultiIndex.from_product([GATES, SITES])).to_string())


def _palette(df):
    layers = sorted(df["layer"].unique())
    return dict(zip(layers, sns.color_palette("viridis", len(layers))))


def plot_lines(df, eval_type, metric):
    sub = df[(df["eval_type"] == eval_type) & (df["metric"] == metric)]
    if sub.empty:
        return
    sub = sub.assign(position=sub["position"].map(lambda p: p + SEQ_LEN if p < 0 else p))
    pal = _palette(sub)
    fig, axes = plt.subplots(len(GATES), len(SITES), figsize=(5.2 * len(SITES), 4.2 * len(GATES)),
                             sharex=True, sharey=True, squeeze=False)
    for i, gate in enumerate(GATES):
        for j, site in enumerate(SITES):
            ax = axes[i, j]
            d = sub[(sub["gate"] == gate) & (sub["site"] == site)]
            if d.empty:
                ax.text(0.5, 0.5, "no runs", ha="center", va="center", transform=ax.transAxes)
            else:
                sns.lineplot(data=d, x="position", y="value", hue="layer", palette=pal,
                             marker="o", errorbar="se", ax=ax, legend=(i == 0 and j == len(SITES) - 1))
            ax.set_xticks(sorted(sub["position"].unique()))
            ax.axhline(0, color="grey", ls="--", lw=1)
            ax.grid(alpha=0.25)
            ax.set_title(f"{SITE_WORDS[site]}" if i == 0 else "")
            ax.set_xlabel("Token Position" if i == len(GATES) - 1 else "")
            ax.set_ylabel(f"LoRA Gate-{gate.capitalize()}\n{eval_type.title()} {metric}" if j == 0 else "")
    if "corr" in metric or metric in ("spearman", "kendall_tau"):
        axes[0, 0].set_ylim(min(-0.1, sub.groupby(['gate','site','layer','position'])['value'].mean().min() - 0.05), 1.05)
    leg_ax = axes[0, -1]
    if leg_ax.get_legend() is not None:
        sns.move_legend(leg_ax, "center left", bbox_to_anchor=(1.02, 0.5), title="Layer", frameon=False)
    fig.tight_layout()
    path = OUT / f"site_lines_{eval_type}_{metric}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[site] wrote {path}")


def plot_summary(df, metric):
    sub = df[df["metric"] == metric]
    if sub.empty:
        return
    # one value per neuron: mean over positions, so the SE is across neurons
    per = (sub.groupby(["gate", "site", "layer", "neuron", "eval_type"])["value"].mean()
           .reset_index())
    per["site"] = pd.Categorical(per["site"], SITES, ordered=True)
    pal = _palette(per)
    fig, axes = plt.subplots(len(GATES), len(EVALS), figsize=(6 * len(EVALS), 4.2 * len(GATES)),
                             sharey=True, squeeze=False)
    for i, gate in enumerate(GATES):
        for j, ev in enumerate(EVALS):
            ax = axes[i, j]
            d = per[(per["gate"] == gate) & (per["eval_type"] == ev)]
            if not d.empty:
                sns.pointplot(data=d, x="site", y="value", hue="layer", palette=pal, order=SITES,
                              errorbar="se", dodge=0.25, capsize=0.08, ax=ax,
                              legend=(i == 0 and j == len(EVALS) - 1))
            ax.set_xticks(range(len(SITES)), [SITE_SHORT[s] for s in SITES])
            ax.axhline(0, color="grey", ls="--", lw=1)
            ax.grid(alpha=0.25)
            ax.set_title(f"{ev.title()} {metric}" if i == 0 else "")
            ax.set_xlabel("Introspected Site" if i == len(GATES) - 1 else "")
            ax.set_ylabel(f"LoRA Gate-{gate.capitalize()}" if j == 0 else "")
    if "corr" in metric or metric in ("spearman", "kendall_tau"):
        axes[0, 0].set_ylim(min(-0.1, per.groupby(['gate','site','layer','eval_type'], observed=True)['value'].mean().min() - 0.05), 1.05)
    leg_ax = axes[0, -1]
    if leg_ax.get_legend() is not None:
        sns.move_legend(leg_ax, "center left", bbox_to_anchor=(1.02, 0.5), title="Layer", frameon=False)
    fig.tight_layout()
    path = OUT / f"site_summary_{metric}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[site] wrote {path}")
    tab = per.groupby(["gate", "eval_type", "layer", "site"], observed=True)["value"].mean().unstack("site")
    print(f"[site] {metric}, mean over positions and neurons:")
    print(tab.reindex(columns=SITES).round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-csv", action="store_true", help="redraw from site_runs.csv")
    ap.add_argument("--metrics", nargs="+", default=["corr"],
                    help="any of corr spearman kendall_tau monotone_r2 l2_dist")
    args = ap.parse_args()
    df = pd.read_csv(OUT / "site_runs.csv") if args.from_csv else load_all()
    coverage(df)
    for metric in args.metrics:
        for ev in EVALS:
            plot_lines(df, ev, metric)
        plot_summary(df, metric)


if __name__ == "__main__":
    main()
