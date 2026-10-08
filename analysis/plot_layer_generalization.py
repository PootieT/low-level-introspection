#!/usr/bin/env python3
"""
Does the readout carry a LAYER it was never trained on?

Reads the layer-generalization sweep (outputs/runs/mlp_layergen/layergen_*/) and answers
the layer-axis version of the neuron question. Every run trains ONE index at some set of
layers and is evaluated on that index at all 15 layers; each locale is seen or unseen by
layer. Sources, per run directory:

  run_spec.json                          train_layers, group, index
  neuron_generalization_by_signal.csv    per-locale corr with is_trained (the eval is
                                         shared with the neuron sweep; only the axis
                                         being held out differs)

WHAT THE LAYER AXIS HAS THAT THE OTHERS DO NOT: test layers sit BETWEEN train layers.
The prompt names the layer as a number, and at N >= 2 every unseen number lies between
two seen ones -- interpolation. At N=1 it is extrapolation for six of the seven. That
is a free second axis, and it is flagged per (run, layer) as `bracketed`.

CEILINGS. A low score at layer 13 could mean "did not generalize there" or "cannot be
read there at all" -- deeper layers have fewer layers above them to carry the signal to
the readout position, and that is inseparable from generalization failure without a
reference. Two single-layer ceilings per test layer, same index:
  ceil37k    matched to N=8's per-layer data. The reference for the generalization
             claim: of what this layer can do with this much data, how much came back?
  ceil300k   full budget. How far from the best achievable?
Unseen results are reported as a FRACTION of ceiling, not raw.

k=1 by design (one index, so only the layer varies), which means every per-layer number
is one neuron. The spread across the seven test layers is the only spread there is.

Panels
  layer_heatmap        every run x every layer, trained cells outlined -- the picture
  unseen_by_n          seen / unseen macro over layers as N grows
  fraction_of_ceiling  unseen corr / ceiling, per test layer, one line per N
  bracketed            interpolation vs extrapolation, per N
  interference         each train layer followed as neighbours are added (nested sets)
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from plot_neuron_generalization import series_style_handles  # noqa: E402  (same legend)

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "legend.title_fontsize": 13, "figure.titlesize": 18,
})

ROOT = Path(__file__).parents[1]
EVAL_TYPES = ("endogenous", "interventional")
CEILINGS = ("ceil37k", "ceil300k")


def load_sweep(exp_group: str):
    base = Path(f"{ROOT}/outputs/runs/{exp_group}")
    if not base.exists():
        # Plots get rsynced more often than runs; the all-signals CSV is the complete
        # frame. train_layers is not stored in it, but it is exactly the set of layers
        # marked is_trained for that run, so it is rebuilt from the rows.
        csv = Path(f"{ROOT}/outputs/plots/{exp_group}/layer_generalization_all.csv")
        if csv.exists():
            print(f"[layergen] no run directories under {base}; re-plotting from {csv.name}")
            ps = pd.read_csv(csv)
            tl = (ps[ps["is_trained"]].groupby("run")["signal_layer"]
                    .apply(lambda v: tuple(sorted(int(x) for x in v.unique()))))
            ps["train_layers"] = ps["run"].map(tl)
            ps["arm_label"] = np.where(ps["is_trained"], "seen", "unseen")
            return ps
        sys.exit(f"No such experiment group: {base}")
    frames, skipped = [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        spec = d / "run_spec.json"
        if not spec.exists():
            continue                      # scan/pool config dirs, etc.
        j = json.loads(spec.read_text())
        f = d / "neuron_generalization_by_signal.csv"
        if not f.exists():
            sub = sorted(d.glob("diag_*/neuron_generalization_by_signal.csv"))
            f = sub[-1] if sub else None
        if f is None:
            skipped.append((d.name, "no neuron_generalization_by_signal.csv"))
            continue
        df = pd.read_csv(f)
        df["run"] = d.name
        df["group"] = j["group"]
        df["index"] = int(j["index"])
        df["n_layers"] = int(j["n_layers"])
        df["rows_per_locale"] = int(j["rows_per_locale"])
        tl = sorted(int(x) for x in j["train_layers"])
        df["train_layers"] = [tuple(tl)] * len(df)
        # interpolation vs extrapolation, per evaluated layer
        df["bracketed"] = [any(a < L for a in tl) and any(b > L for b in tl)
                           for L in df["signal_layer"]]
        frames.append(df)
    if not frames:
        sys.exit(f"No layer-generalization results under {base}. Rsync the runs first.")
    ps = pd.concat(frames, ignore_index=True)
    ps["arm_label"] = np.where(ps["is_trained"], "seen", "unseen")
    if skipped:
        print(f"[layergen] skipped {len(skipped)} directories:")
        for name, why in skipped[:10]:
            print(f"    {name}: {why}")
    return ps


def report(ps: pd.DataFrame):
    runs = ps.drop_duplicates("run")[["run", "group", "n_layers", "rows_per_locale", "train_layers"]]
    print(f"\n[layergen] {len(runs)} runs, index {sorted(ps['index'].unique())}")
    print(runs.sort_values(["group", "n_layers"]).to_string(index=False))
    if ps["index"].nunique() > 1:
        print("[layergen] WARNING: more than one index across runs; the sweep assumes one")
    dead = ps[ps["activation_std"].fillna(1) <= 1e-6]
    if len(dead):
        print(f"[layergen] {sorted(dead['signal_layer'].unique())}: layers where the index is "
              f"near-silent; correlations there are undefined, not poor")
    for c in CEILINGS:
        have = sorted(ps[ps["group"] == c]["signal_layer"][ps["is_trained"]].unique())
        want = [1, 3, 5, 7, 9, 11, 13]
        missing = [L for L in want if L not in have]
        if missing:
            print(f"[layergen] {c}: missing ceilings for layers {missing}")


def _ceilings(ps: pd.DataFrame) -> pd.DataFrame:
    """(eval_type, ceiling group, layer) -> corr of the single-layer run trained there."""
    c = ps[ps["group"].isin(CEILINGS) & ps["is_trained"]]
    return (c.groupby(["eval_type", "group", "signal_layer"])["corr"].mean()
             .rename("ceiling").reset_index())


def plot_layer_heatmap(ps: pd.DataFrame, out_dir):
    """Every run x every layer. Trained cells are outlined, so the eye can separate
    'trained here' from 'generalized here' without a second panel."""
    order = (ps.drop_duplicates("run")
               .assign(_g=lambda d: d["group"].map({"scaling": 0, "data300k": 1, "ceil37k": 2, "ceil300k": 3}))
               .sort_values(["_g", "n_layers", "train_layers"])["run"].tolist())
    fig, axes = plt.subplots(1, 2, figsize=(17, 0.42 * len(order) + 2.5), sharey=True)
    for ax, et in zip(axes, EVAL_TYPES):
        sub = ps[ps["eval_type"] == et]
        piv = sub.pivot_table(index="run", columns="signal_layer", values="corr").reindex(order)
        tr = sub.pivot_table(index="run", columns="signal_layer", values="is_trained",
                             aggfunc="max").reindex(order).fillna(False).astype(bool)
        sns.heatmap(piv, annot=True, fmt=".2f", cmap="viridis", vmin=-0.2, vmax=1.0, ax=ax,
                    linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.6},
                    annot_kws={"size": 8})
        for r, run in enumerate(piv.index):
            for c, L in enumerate(piv.columns):
                if tr.loc[run, L]:
                    ax.add_patch(mpatches.Rectangle((c, r), 1, 1, fill=False, ec="red", lw=2))
        ax.set_title(et)
        ax.set_xlabel("layer")
        ax.set_ylabel("")
    axes[0].set_yticklabels([r.replace("layergen_", "") for r in order], rotation=0, fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "layer_heatmap.png", dpi=150)
    plt.close(fig)


def plot_unseen_by_n(ps: pd.DataFrame, out_dir):
    d = ps[ps["group"] == "scaling"]
    if d.empty:
        return
    STYLE = {"seen": dict(marker="o", ls="-"), "unseen": dict(marker="o", ls=":", mfc="none", mew=1.8)}
    def agg(sub, series):
        return (sub.groupby(["eval_type", "arm_label", "n_layers"])["corr"]
                   .agg(corr_macro="mean", corr_std="std", n_layers_in="count")
                   .reset_index().assign(series=series))
    m = agg(d, "300k total (scaling)")
    # the same N re-run at 300k rows PER LAYER: same layers as the scaling cell, so the
    # only difference is data. Drawn as a separate point, not joined to the curve.
    more = ps[ps["group"] == "data300k"]
    if not more.empty:
        rpl = int(more["rows_per_locale"].iloc[0]) // 1000
        m = pd.concat([m, agg(more, f"{rpl}k per layer")], ignore_index=True)
    m.to_csv(Path(out_dir) / "unseen_by_n.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    colours = {"300k total (scaling)": "#1f77b4"}
    for ax, et in zip(axes, EVAL_TYPES):                   # endogenous first
        for series in m["series"].unique():
            colour = colours.get(series, "#d62728")
            for lbl in ("seen", "unseen"):
                q = m[(m["eval_type"] == et) & (m["series"] == series) &
                      (m["arm_label"] == lbl)].sort_values("n_layers")
                if q.empty:
                    continue
                ax.errorbar(q["n_layers"], q["corr_macro"], yerr=q["corr_std"].fillna(0),
                            ms=9, capsize=3, lw=1.8, color=colour, **STYLE[lbl])
        ax.set_xscale("log", base=2)
        ax.axhline(0, color="k", lw=0.9, alpha=0.4)
        ax.set_xlabel("Layers Trained", fontsize=26)
        ax.set_title(et.capitalize(), fontsize=28)
        ax.tick_params(labelsize=22)
    axes[0].set_ylabel("Macro Correlation\nover Layers", fontsize=26)
    # compact legend, in the right panel only: colour = data budget, style = seen/unseen
    handles = series_style_handles(
        [(series, colours.get(series, "#d62728")) for series in m["series"].unique()], STYLE)
    axes[-1].legend(handles=handles, ncol=2, fontsize=16, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "unseen_by_n.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return m


def plot_fraction_of_ceiling(ps: pd.DataFrame, out_dir):
    """Unseen corr / ceiling corr, per test layer, one line per N. Both ceilings."""
    ceil = _ceilings(ps)
    if ceil.empty:
        print("[layergen] no ceiling runs yet; fraction-of-ceiling plots skipped")
        return
    d = ps[ps["group"].isin(["scaling", "data300k"]) & ~ps["is_trained"]].copy()
    # label the high-data cell so its line is not confused with the scaling N=4 line
    d["n_layers"] = np.where(d["group"] == "data300k", d["n_layers"].astype(str) + " @300k/layer",
                             d["n_layers"].astype(str))
    m = d.merge(ceil, on=["eval_type", "signal_layer"], how="inner", suffixes=("", "_c"))
    m = m.rename(columns={"group_c": "ceiling_group"}) if "group_c" in m else m
    # merge brought in the ceiling's `group` under a suffix; keep it explicit
    if "group_y" in m.columns:
        m = m.rename(columns={"group_x": "group", "group_y": "ceiling_group"})
    m["fraction"] = m["corr"] / m["ceiling"]
    # a weak ceiling makes the fraction meaningless; keep the row, flag it
    m["ceiling_weak"] = m["ceiling"].abs() <= 0.05
    m.to_csv(Path(out_dir) / "fraction_of_ceiling.csv", index=False)

    for cg in CEILINGS:
        mm = m[m["ceiling_group"] == cg]
        if mm.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        ns = sorted(mm["n_layers"].unique(), key=lambda v: (int(str(v).split()[0]), str(v)))
        colours = dict(zip(ns, sns.color_palette("mako", len(ns))))
        for ax, et in zip(axes, EVAL_TYPES):
            for n in ns:
                q = mm[(mm["eval_type"] == et) & (mm["n_layers"] == n)].sort_values("signal_layer")
                ax.plot(q["signal_layer"], q["fraction"], marker="o", ms=6, lw=1.8,
                        color=colours[n], label=f"N={n}")
                w = q[q["ceiling_weak"]]
                if len(w):
                    ax.scatter(w["signal_layer"], w["fraction"], marker="x", s=90, color="k", zorder=5)
            ax.axhline(1, color="k", lw=0.9, alpha=0.5)
            ax.axhline(0, color="k", lw=0.9, alpha=0.3)
            ax.set_xlabel("unseen (test) layer")
            ax.set_title(et)
            ax.legend(fontsize=10, title="train layers")
        lo = max(-0.6, float(np.nanmin(mm["fraction"])) - 0.1)
        axes[0].set_ylim(lo, max(1.15, float(np.nanpercentile(mm["fraction"], 97)) + 0.1))
        axes[0].set_ylabel(f"unseen corr / {cg} ceiling")
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"fraction_of_ceiling_{cg}.png", dpi=150)
        plt.close(fig)
    return m


def plot_bracketed(ps: pd.DataFrame, out_dir):
    """Interpolation (test layer between two train layers) vs extrapolation, per N."""
    ceil = _ceilings(ps)
    d = ps[(ps["group"] == "scaling") & ~ps["is_trained"]]
    if d.empty:
        return
    if not ceil.empty:
        c37 = ceil[ceil["group"] == "ceil37k"][["eval_type", "signal_layer", "ceiling"]]
        d = d.merge(c37, on=["eval_type", "signal_layer"], how="left")
        d["value"] = np.where(d["ceiling"].abs() > 0.05, d["corr"] / d["ceiling"], np.nan)
        ylabel = "unseen corr / ceil37k"
    else:
        d = d.assign(value=d["corr"])
        ylabel = "unseen corr (no ceilings yet)"
    m = (d.groupby(["eval_type", "n_layers", "bracketed"])["value"]
           .agg(mean="mean", std="std", n="count").reset_index())
    m.to_csv(Path(out_dir) / "bracketed.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, et in zip(axes, EVAL_TYPES):
        q = m[m["eval_type"] == et]
        ns = sorted(q["n_layers"].unique()); x = np.arange(len(ns)); w = 0.36
        for k, (b, colour, lab) in enumerate(((True, "#2ca02c", "interpolation (bracketed)"),
                                              (False, "#ff7f0e", "extrapolation"))):
            qq = q[q["bracketed"] == b].set_index("n_layers").reindex(ns)
            ax.bar(x + (k - 0.5) * w, qq["mean"], w, yerr=qq["std"].fillna(0), capsize=3,
                   color=colour, label=lab)
            for xi, n in zip(x, ns):
                if not np.isnan(qq.loc[n, "n"] if n in qq.index else np.nan):
                    ax.text(xi + (k - 0.5) * w, 0.02, f"n={int(qq.loc[n, 'n'])}", ha="center",
                            fontsize=8, color="white", transform=ax.get_xaxis_transform())
        ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in ns])
        ax.axhline(0, color="k", lw=0.8, alpha=0.4)
        ax.set_title(et)
        ax.legend(fontsize=9)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "bracketed.png", dpi=150)
    plt.close(fig)
    return m


def plot_interference(ps: pd.DataFrame, out_dir):
    """Each train layer followed as neighbours are added. Sets are nested, so layer 8 is
    in every run and layer 0 in every run from N=2 on."""
    d = ps[(ps["group"] == "scaling") & ps["is_trained"]]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    layers = sorted(d["signal_layer"].unique())
    colours = dict(zip(layers, sns.color_palette("husl", len(layers))))
    for ax, et in zip(axes, EVAL_TYPES):
        sub = d[d["eval_type"] == et]
        for L in layers:
            q = sub[sub["signal_layer"] == L].sort_values("n_layers")
            if len(q) < 2:
                continue
            ax.plot(q["n_layers"], q["corr"], marker="o", ms=5, lw=1.4, color=colours[L],
                    alpha=0.85, label=f"layer {L}")
        agg = sub.groupby("n_layers")["corr"].mean().reset_index()
        ax.plot(agg["n_layers"], agg["corr"], lw=2.8, color="k", alpha=0.85, zorder=5,
                label="all trained (macro)")
        ax.set_xscale("log", base=2)
        ax.axhline(0, color="k", lw=0.9, alpha=0.4)
        ax.set_xlabel("layers trained")
        ax.set_title(et)
        ax.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("corr (this layer)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "interference.png", dpi=150)
    plt.close(fig)


def main(exp_group="mlp_layergen"):
    out_dir = Path(f"{ROOT}/outputs/plots/{exp_group}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ps = load_sweep(exp_group)
    report(ps)
    ps.drop(columns=["train_layers"]).to_csv(out_dir / "layer_generalization_all.csv", index=False)
    plot_layer_heatmap(ps, out_dir)
    m = plot_unseen_by_n(ps, out_dir)
    plot_fraction_of_ceiling(ps, out_dir)
    plot_bracketed(ps, out_dir)
    plot_interference(ps, out_dir)
    if m is not None:
        print("\n[layergen] macro corr over layers by (eval_type, arm, N):")
        print(m.to_string(index=False))
    print(f"\n[layergen] wrote plots to {out_dir}")


if __name__ == "__main__":
    main(exp_group=sys.argv[1] if len(sys.argv) > 1 else "mlp_layergen")
