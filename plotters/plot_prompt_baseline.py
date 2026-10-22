#!/usr/bin/env python3
"""
Aggregate the prompted baseline (evals/score_prompt_baseline.py) over the main grid, and set
it against the trained arms.

The baseline is the UNTRAINED model, asked with the grid's own prompts and read out by
ridge probes on its final hidden state, pooled over k greedily generated tokens. Each
probe becomes an ARM shaped exactly like compare_sft_vs_joint.load_group's frames, so
pairing, the OOD ladder and the heatmaps are that script's own functions:

  prompt_endo    ridge on the true activation           (head-only analogue of SFT)
  prompt_joint   ridge on the grid's joint objective    (the joint recipe, LoRA lr=0)
  prompt_verbal  the number the model WRITES in its k_max-token continuation, parsed
                 as a float -- no probe at all. corr is over the rows that parse
                 (parse rate is reported), and is NaN when every row writes the same
                 number; those runs drop out of the mean rather than counting as 0.

The probe arms are read at --k (default 0, the trained head's own readout position).
The verbal arm has no OOD scores (the OOD pass scores the probes), so it is left out of
the ladder.

Reads, per <baseline-root>/prompt_<signal>_POS<p>/:
  prompt_baseline_by_k.csv     the (prompt, probe, pool, k) table
  {endo,joint}_probe/ood_generalization.csv
  verbal_report.json           the number the model WRITES, no probe
  learning_curve.csv           k=0 metrics vs probe training rows
and the learning-curve pilot's root, if present.

Writes to --out-dir (default outputs/plots/prompt_baseline):
  baseline_by_k_all.csv                   every row of every run
  corr_vs_k.png / corr_vs_k_summary.csv   mean +- SE over runs, per (probe, pool), with
                                          the trained arms and the no-prompt control as
                                          reference lines (on the SAME runs), and the
                                          verbal readout as a star at k_max
  corr_vs_k_by_layer.png                  each metric from the probe aimed at it
  heatmap_prompt_{endo,joint,verbal}_<metric>.png
  paired_summary.csv                      trained arm minus baseline arm, within neuron
  ood_ladder_{endogenous,interventional}.{png,csv}
  verbal_summary.csv
  learning_curve.png / learning_curve_summary.csv
  missing_runs.json

Example
-------
python plotters/plot_prompt_baseline.py
python plotters/plot_prompt_baseline.py --k 16       # the arms read at k=16 instead
"""
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import compare_sft_vs_joint as cmp     # also applies the shared poster rcParams
from compare_sft_vs_joint import (
    KEY, ROOT, expected_grid, load_group, pair, paired_stats, plot_arm_heatmap,
    plot_delta_heatmap, plot_ood_ladders, report_missing,
)

TRAINED_ARMS = {
    "sft_gateon":  dict(group="mlp_endo_sweep_layer_token", filters=dict(gate="on", w=1.0)),
    "sft_gateoff": dict(group="mlp_endo_sweep_layer_token", filters=dict(gate="off", w=1.0)),
    "joint_kl0.1": dict(group="mlp_joint_grid_layer_token",
                        filters=dict(gate="off", w=0.01, kl="0.1")),
    "joint_free":  dict(group="mlp_joint_grid_layer_token",
                        filters=dict(gate="off", w=0.01, kl="none")),
}
BASELINE_PROBES = ("endo", "joint")
PROBE_COLORS = {"endo": "tab:blue", "joint": "tab:orange"}
POOL_STYLES = {"mean": "-", "last": "--"}

_pretty_arm = cmp.pretty_arm


def pretty_arm(label: str) -> str:
    if label.startswith("prompt_"):
        return {"prompt_endo": "Prompted (Endo Probe)",
                "prompt_joint": "Prompted (Joint Probe)",
                "prompt_verbal": "Prompted (Verbal, No Probe)"}.get(label, label)
    return _pretty_arm(label)


cmp.pretty_arm = pretty_arm             # the ladder's legend resolves names through it


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-root", type=Path,
                   default=ROOT / "outputs/runs/mlp_prompt_baseline_layer_token")
    p.add_argument("--lc-root", type=Path,
                   default=ROOT / "outputs/runs/mlp_prompt_baseline_learning_curve",
                   help="Learning-curve pilot (MODE=lc of the launch script); optional.")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/plots/prompt_baseline")
    p.add_argument("--k", type=int, default=0, help="k at which the baseline ARMS are read.")
    p.add_argument("--pool", default="mean")
    p.add_argument("--delta-heatmaps", action="store_true",
                   help="Also write a trained-minus-baseline (layer, position) map per pair.")
    return p.parse_args()


# ======================================================================
# Loading
# ======================================================================

def _run_dirs(root: Path):
    if not root.exists():
        raise SystemExit(f"No baseline results under {root}")
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "prompt_baseline_by_k.csv").exists():
            yield d, json.loads((d / "run_spec.json").read_text())


def load_by_k(root: Path) -> pd.DataFrame:
    frames = []
    for d, spec in _run_dirs(root):
        t = pd.read_csv(d / "prompt_baseline_by_k.csv")
        t = t.assign(run=d.name, layer=int(spec["layer"]), index=int(spec["index"]),
                     neuron=f"N{spec['layer']}.{spec['index']}",
                     position=int(spec["position"]))
        frames.append(t)
    if not frames:
        raise SystemExit(f"No finished baseline runs under {root}")
    df = pd.concat(frames, ignore_index=True)
    print(f"[base] {df['run'].nunique()} runs, {df['neuron'].nunique()} neurons, "
          f"{df['position'].nunique()} positions, {df['layer'].nunique()} layers")
    return df


def baseline_arm(root: Path, by_k: pd.DataFrame, probe: str, k: int, pool: str) -> pd.DataFrame:
    """One baseline probe as an arm: the columns compare_sft_vs_joint.load_group emits."""
    sel = by_k[(by_k["prompt"] == "templated") & (by_k["probe"] == probe)
               & (by_k["pool"] == pool) & (by_k["k"] == k)]
    if sel.empty:
        raise SystemExit(f"no rows for probe={probe} pool={pool} k={k}")
    rows = []
    for _, r in sel.iterrows():
        row = {"group": root.name, "run": r["run"], "layer": r["layer"], "index": r["index"],
               "neuron": r["neuron"], "position": r["position"],
               "interventional_corr": r["interv_corr"], "endogenous_corr": r["endo_corr"],
               "lambda_at_edge": r["lambda_at_edge"]}
        # the OOD pass scores the k=0 probes only
        ood_path = root / r["run"] / f"{probe}_probe" / "ood_generalization.csv"
        if k == 0 and ood_path.exists():
            for _, o in pd.read_csv(ood_path).iterrows():
                c = o.get("corpus")
                if isinstance(c, str):
                    row[f"ood_{c}_interventional"] = o.get("interventional_corr")
                    row[f"ood_{c}_endogenous"] = o.get("endogenous_corr")
                    row[f"ood_{c}_activation_std"] = o.get("activation_std")
        rows.append(row)
    return pd.DataFrame(rows)


def verbal_arm(root: Path):
    """The verbal readout as an arm: per run, corr of the parsed number with the true
    activation (endogenous) and with the injected value (interventional)."""
    rows = []
    for d, spec in _run_dirs(root):
        f = d / "verbal_report.json"
        if not f.exists():
            continue
        rep = json.loads(f.read_text())
        rows.append({"group": root.name, "run": d.name, "layer": int(spec["layer"]),
                     "index": int(spec["index"]), "neuron": f"N{spec['layer']}.{spec['index']}",
                     "position": int(spec["position"]), "k_max": max(spec["k_values"]),
                     "endogenous_corr": rep.get("endogenous", {}).get("corr"),
                     "interventional_corr": rep.get("interventional", {}).get("corr"),
                     "parse_rate_endogenous": rep.get("endogenous", {}).get("parse_rate"),
                     "parse_rate_interventional": rep.get("interventional", {}).get("parse_rate")})
    if not rows:
        print("[base] no verbal reports; no verbal arm")
        return None
    df = pd.DataFrame(rows)
    n_nan = int(df["endogenous_corr"].isna().sum())
    if n_nan:
        print(f"[base] verbal: {n_nan}/{len(df)} runs have undefined endogenous corr "
              f"(the model wrote one number for every row); excluded from means")
    return df


def load_trained_arms() -> dict:
    frames = {}
    for label, spec in TRAINED_ARMS.items():
        try:
            frames[label] = load_group(spec["group"], **spec["filters"])
        except SystemExit as e:
            print(f"[base] trained arm {label}: not available ({e}); skipped")
    return frames


# ======================================================================
# Figures
# ======================================================================

def _mean_se(v):
    v = pd.Series(v).dropna()
    return (v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan, len(v))


def plot_corr_vs_k(by_k, trained, out_dir, verbal=None):
    """Mean +- SE over runs vs k. Reference lines (trained arms, no-prompt control) are
    averaged over the SAME runs as the curves, so a line is never lifted by cells the
    baseline does not have. The verbal readout is a star at k_max: the number it parses
    comes from the whole k_max-token continuation."""
    tmpl = by_k[by_k["prompt"] == "templated"]
    runs = set(map(tuple, tmpl[KEY].drop_duplicates().values))
    rows = []
    for (probe, pool, k), g in tmpl.groupby(["probe", "pool", "k"]):
        for metric in ("endo_corr", "interv_corr"):
            m, se, n = _mean_se(g[metric])
            rows.append({"probe": probe, "pool": pool, "k": k, "metric": metric,
                         "mean": m, "se": se, "n": n})
    summ = pd.DataFrame(rows)
    summ.to_csv(out_dir / "corr_vs_k_summary.csv", index=False)

    ks = sorted(tmpl["k"].unique())
    xpos = {k: i for i, k in enumerate(ks)}
    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5), sharey=True)
    refs = {}
    for label, f in trained.items():
        f = f[[tuple(r) in runs for r in f[KEY].values]]
        if len(f):
            refs[label] = {"endo_corr": f["endogenous_corr"].mean(),
                           "interv_corr": f["interventional_corr"].mean(), "n": len(f)}
    ctrl = by_k[(by_k["prompt"] == "none") & (by_k["k"] == 0)]
    for ax, metric, title in zip(axes, ("endo_corr", "interv_corr"),
                                 ("Endogenous Correlation", "Interventional Correlation")):
        for (probe, pool), g in summ[summ["metric"] == metric].groupby(["probe", "pool"]):
            g = g.sort_values("k")
            ax.errorbar([xpos[k] for k in g["k"]], g["mean"], yerr=g["se"], marker="o",
                        markersize=9, linewidth=2.5, capsize=4, color=PROBE_COLORS[probe],
                        linestyle=POOL_STYLES.get(pool, ":"),
                        label=f"{probe} probe, {pool} pool")
        for (label, ref), color in zip(refs.items(), plt.get_cmap("Greys")(np.linspace(0.45, 0.9, 4))):
            ax.axhline(ref[metric], color=color, linestyle=":", linewidth=2.5,
                       label=f"{pretty_arm(label)} (n={ref['n']})")
        if verbal is not None and len(verbal):
            v = verbal[[tuple(r) in runs for r in verbal[KEY].values]]
            col = "endogenous_corr" if metric == "endo_corr" else "interventional_corr"
            m, se, n = _mean_se(v[col])
            k_max = int(v["k_max"].max())
            ax.errorbar([xpos.get(k_max, len(ks) - 1)], [m], yerr=[se], marker="*",
                        markersize=24, color="k", capsize=4, linestyle="none", zorder=6,
                        label=f"verbal: written number, no probe (n={n})")
        for probe, g in ctrl.groupby("probe"):
            ax.scatter([xpos[0]], [g[metric].mean()], marker="X", s=220, zorder=5,
                       color=PROBE_COLORS[probe], edgecolor="k",
                       label=f"no prompt, {probe} probe")
        ax.set_xticks(range(len(ks))); ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("Generated Tokens Pooled (k)"); ax.set_title(title)
        ax.axhline(0, color="grey", lw=1); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Correlation (Mean ± SE over Neurons)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=15,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out_dir / "corr_vs_k.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[base] wrote {out_dir / 'corr_vs_k.png'}")
    print(summ[summ["pool"] == "mean"].pivot_table(index="k", columns=["metric", "probe"],
                                                   values="mean").round(4).to_string())


def plot_corr_vs_k_by_layer(by_k, out_dir, pool="mean"):
    """Each metric from the probe aimed at it: endogenous from the endo probe,
    interventional from the joint probe -- the baseline's best shot at each."""
    tmpl = by_k[(by_k["prompt"] == "templated") & (by_k["pool"] == pool)]
    ks = sorted(tmpl["k"].unique())
    layers = sorted(tmpl["layer"].unique())
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(layers)))
    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5), sharey=True)
    for ax, (metric, probe, title) in zip(axes, (("endo_corr", "endo", "Endogenous (Endo Probe)"),
                                                 ("interv_corr", "joint", "Interventional (Joint Probe)"))):
        sub = tmpl[tmpl["probe"] == probe]
        for layer, color in zip(layers, colors):
            g = sub[sub["layer"] == layer].groupby("k")[metric]
            m, se = g.mean().reindex(ks), (g.std(ddof=1) / np.sqrt(g.count())).reindex(ks)
            ax.errorbar(range(len(ks)), m, yerr=se, marker="o", linewidth=2.2, capsize=3,
                        color=color, label=f"L{layer}")
        ax.set_xticks(range(len(ks))); ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("Generated Tokens Pooled (k)"); ax.set_title(title)
        ax.axhline(0, color="grey", lw=1); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Correlation (Mean ± SE)")
    axes[1].legend(ncol=2, fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / "corr_vs_k_by_layer.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[base] wrote {out_dir / 'corr_vs_k_by_layer.png'}")


def verbal_summary(root, out_dir):
    rows = []
    for d, spec in _run_dirs(root):
        f = d / "verbal_report.json"
        if not f.exists():
            continue
        for arm, v in json.loads(f.read_text()).items():
            rows.append({"run": d.name, "layer": spec["layer"], "position": spec["position"],
                         "readout": arm, **v})
    if not rows:
        print("[base] no verbal reports")
        return
    df = pd.DataFrame(rows)
    out = []
    for (readout,), g in df.groupby(["readout"]):
        m, se, n = _mean_se(g["corr"])
        out.append({"readout": readout, "n_runs": n, "corr_mean": m, "corr_se": se,
                    "parse_rate_mean": g["parse_rate"].mean(),
                    "distinct_values_median": g["n_distinct_values"].median()})
    summ = pd.DataFrame(out)
    df.to_csv(out_dir / "verbal_by_run.csv", index=False)
    summ.to_csv(out_dir / "verbal_summary.csv", index=False)
    print("[base] verbal readout (number the model writes):")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def plot_learning_curve(root, lc_root, out_dir):
    rows = []
    for source, r in (("grid", root), ("pilot", lc_root)):
        if not r.exists():
            continue
        for d, spec in _run_dirs(r):
            f = d / "learning_curve.csv"
            if f.exists():
                rows.append(pd.read_csv(f).assign(source=source, run=d.name))
    if not rows:
        print("[base] no learning curves")
        return
    df = pd.concat(rows, ignore_index=True)
    summ = (df.groupby(["source", "probe", "n_train"])[["endo_corr", "interv_corr"]]
            .agg(["mean", "sem", "count"]).reset_index())
    summ.columns = ["_".join(c).rstrip("_") for c in summ.columns]
    summ.to_csv(out_dir / "learning_curve_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5), sharey=True)
    for ax, metric, title in zip(axes, ("endo_corr", "interv_corr"),
                                 ("Endogenous Correlation", "Interventional Correlation")):
        for (source, probe), g in summ.groupby(["source", "probe"]):
            ax.errorbar(g["n_train"], g[f"{metric}_mean"], yerr=g[f"{metric}_sem"],
                        marker="o" if source == "grid" else "s", linewidth=2.5, capsize=4,
                        color=PROBE_COLORS[probe], linestyle="-" if source == "grid" else "--",
                        label=f"{probe} probe ({source}, n={int(g[f'{metric}_count'].max())} runs)")
        ax.set_xscale("log"); ax.set_xlabel("Probe Training Rows"); ax.set_title(f"{title}, k=0")
        ax.axhline(0, color="grey", lw=1); ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Correlation (Mean ± SE)")
    axes[1].legend(fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[base] wrote {out_dir / 'learning_curve.png'}")


# ======================================================================

def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    by_k = load_by_k(args.baseline_root)
    by_k.to_csv(out_dir / "baseline_by_k_all.csv", index=False)
    edge = by_k[by_k["k"] == 0].groupby(["prompt", "probe"])["lambda_at_edge"].mean()
    print(f"[base] fraction of k=0 probe fits whose ridge penalty sits at the grid edge "
          f"(the upper edge = 'nothing decodable, predict the mean'):\n{edge.round(3).to_string()}")

    arms = {f"prompt_{p}": baseline_arm(args.baseline_root, by_k, p, args.k, args.pool)
            for p in BASELINE_PROBES if p in set(by_k["probe"])}
    verbal = verbal_arm(args.baseline_root)
    if verbal is not None:
        arms["prompt_verbal"] = verbal
    missing = {label: report_missing(f, label, expected_grid(model="llama321b"), model="llama321b")
               for label, f in arms.items()}
    with open(out_dir / "missing_runs.json", "w") as f:
        json.dump({k: [list(m) for m in v] for k, v in missing.items()}, f, indent=2)

    trained = load_trained_arms()
    plot_corr_vs_k(by_k, trained, out_dir, verbal=verbal)
    plot_corr_vs_k_by_layer(by_k, out_dir, pool=args.pool)
    for label, f in arms.items():
        for m in ("interventional_corr", "endogenous_corr"):
            plot_arm_heatmap(f, m, label, out_dir)

    # trained minus baseline, within neuron
    tables = []
    for t_label, t in trained.items():
        for b_label, b in arms.items():
            print(f"\n[base] ---------- {t_label} vs {b_label} ----------")
            try:
                merged = pair(t, b, t_label, b_label)
            except SystemExit as e:
                print(f"[base] {e}")
                continue
            metrics = ["interventional_corr", "endogenous_corr"]
            metrics += sorted({c.rsplit("_", 1)[0] for c in merged.columns
                               if c.startswith("ood_") and c.endswith(f"_{t_label}")
                               and "activation_std" not in c})
            stats = [s for s in (paired_stats(merged, m, t_label, b_label) for m in metrics) if s]
            if stats:
                table = pd.DataFrame(stats).assign(arm=t_label, baseline=b_label)
                tables.append(table)
                print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            if args.delta_heatmaps:
                for m in ("interventional_corr", "endogenous_corr"):
                    plot_delta_heatmap(merged, m, t_label, b_label, out_dir)
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(out_dir / "paired_summary.csv", index=False)

    frames = {**trained, **{a: f for a, f in arms.items() if a != "prompt_verbal"}}
    arm_colors = {a: c for a, c in zip(frames, plt.get_cmap("tab10").colors)}
    plot_ood_ladders(frames, out_dir, arm_colors=arm_colors)

    verbal_summary(args.baseline_root, out_dir)
    plot_learning_curve(args.baseline_root, args.lc_root, out_dir)


if __name__ == "__main__":
    main()
