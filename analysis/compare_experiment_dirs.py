#!/usr/bin/env python3
"""
Paired comparison of two experiment groups: joint training vs endogenous-only SFT.

The two sweeps use neuron sets seeded per (seed, layer), and the 5-neuron grid's
neurons are a prefix of the 10-neuron endogenous sweep's, so runs pair EXACTLY on
(layer, position, neuron). Pairing matters: neuron-to-neuron variance is large
(interventional corr swings ~+-0.1 between neurons in the same cell), so comparing two
group means throws away most of the power. Every number here is a within-neuron delta.

Reads, per run:
  interventional_test_metrics_gaussian.json   causal reading, in-domain
  endogenous_test_metrics_endogenous.json     endogenous readout, in-domain
  ood_generalization.csv                      both metrics per OOD corpus
  activation_drift.json                       drift + readout coupling

The headline is the OOD ladder: in-domain both arms look similar, and the claim is that
SFT's endogenous score falls further under shift because it was reading context rather
than the neuron. A near-flat INTERVENTIONAL line across the ladder says the transmission
channel itself is indifferent to the text distribution.

Usage: edit the two group names at the bottom and run.
"""
import os
import re
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
                     "legend.fontsize": 11, "figure.titlesize": 17})

ROOT = Path(__file__).parents[1]
NEURON_RE = re.compile(r"N(?P<layer>\d+)\.(?P<index>\d+)")
POS_RE = re.compile(r"POS(?P<pos>-?\d+)")
GATE_RE = re.compile(r"gate(?P<gate>on|off)")
W_RE = re.compile(r"_w(?P<w>[0-9.]+)")
KL_RE = re.compile(r"_kl(?P<kl>[0-9.a-z]+)")
KEY = ["layer", "position", "neuron"]


def load_group(exp_group: str, gate="off", w=None, kl=None) -> pd.DataFrame:
    """One row per run, with in-domain metrics and one column per OOD corpus."""
    rows = []
    root = Path(f"{ROOT}/outputs/runs/{exp_group}")
    if not root.exists():
        raise SystemExit(f"No such group: {root}")
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        n, p = NEURON_RE.search(d.name), POS_RE.search(d.name)
        if not (n and p):
            continue
        g, mw, mkl = GATE_RE.search(d.name), W_RE.search(d.name), KL_RE.search(d.name)
        if gate and (not g or g["gate"] != gate):
            continue
        if w is not None and (not mw or float(mw["w"]) != w):
            continue
        if kl is not None and (not mkl or mkl["kl"] != kl):
            continue

        def js(fname, key=None):
            f = d / fname
            if not f.exists():
                return None
            payload = json.loads(f.read_text())
            return payload[key] if key else payload

        interv, endo = (js("interventional_test_metrics_gaussian.json", "test"),
                        js("endogenous_test_metrics_endogenous.json"))
        if interv is None or endo is None:
            continue
        row = {"group": exp_group, "run": d.name,
               "layer": int(n["layer"]), "index": int(n["index"]),
               "neuron": f"N{n['layer']}.{n['index']}", "position": int(p["pos"]),
               "w": float(mw["w"]) if mw else None, "kl": mkl["kl"] if mkl else None,
               "interventional_corr": interv.get("corr"),
               "endogenous_corr": endo.get("corr")}
        drift = js("activation_drift.json")
        if drift:
            row["drift_corr"] = drift.get("drift_tuned_vs_base", {}).get("corr")
            row["coupling_gap"] = (drift.get("readout_tracking") or {}).get("coupling_gap")
        ood_path = d / "ood_generalization.csv"
        if ood_path.exists():
            ood = pd.read_csv(ood_path)
            for _, r in ood.iterrows():
                c = r.get("corpus")
                if not isinstance(c, str):
                    continue
                row[f"ood_{c}_interventional"] = r.get("interventional_corr")
                row[f"ood_{c}_endogenous"] = r.get("endogenous_corr")
                row[f"ood_{c}_activation_std"] = r.get("activation_std")
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No runs matched in {exp_group} (gate={gate}, w={w}, kl={kl})")
    print(f"[cmp] {exp_group}: {len(df)} runs, {df['neuron'].nunique()} neurons, "
          f"{df['position'].nunique()} positions, {df['layer'].nunique()} layers")
    return df


def pair(a: pd.DataFrame, b: pd.DataFrame, label_a="A", label_b="B") -> pd.DataFrame:
    """Inner join on (layer, position, neuron). Unmatched runs are reported, since a
    silently dropped arm would bias every average that follows."""
    merged = a.merge(b, on=KEY, suffixes=(f"_{label_a}", f"_{label_b}"))
    only_a, only_b = len(a) - len(merged), len(b) - len(merged)
    print(f"[cmp] paired {len(merged)} run(s); unmatched: {only_a} in {label_a}, "
          f"{only_b} in {label_b}")
    if len(merged) == 0:
        raise SystemExit("No runs paired -- do the two groups share neurons/positions?")
    return merged


def paired_stats(merged, metric, label_a, label_b, min_activation_std=1e-6):
    """Within-neuron delta with a paired SE. Rows whose activation variance is ~0 are
    dropped: correlation is undefined there and would read as a generalization failure."""
    ca, cb = f"{metric}_{label_a}", f"{metric}_{label_b}"
    if ca not in merged or cb not in merged:
        return None
    sub = merged[[*KEY, ca, cb]].dropna()
    std_col = None
    if metric.startswith("ood_"):
        corpus = metric.split("_")[1]
        for cand in (f"ood_{corpus}_activation_std_{label_a}", f"ood_{corpus}_activation_std_{label_b}"):
            if cand in merged:
                std_col = cand
                break
    if std_col is not None:
        keep = merged.loc[sub.index, std_col].fillna(0) > min_activation_std
        dropped = int((~keep).sum())
        if dropped:
            print(f"[cmp] {metric}: dropped {dropped} row(s) with ~zero activation variance")
        sub = sub[keep.values]
    if sub.empty:
        return None
    d = sub[ca] - sub[cb]
    return {"metric": metric, "n": len(sub),
            f"mean_{label_a}": sub[ca].mean(), f"mean_{label_b}": sub[cb].mean(),
            "delta_mean": d.mean(),
            "delta_se": d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan,
            "frac_a_higher": float((d > 0).mean())}


def plot_ood_ladder(merged, label_a, label_b, out_dir, readout="endogenous"):
    """Both arms across the corpus ladder, mean ± SE over paired neurons."""
    corpora = sorted({c.split("_")[1] for c in merged.columns
                      if c.startswith("ood_") and c.endswith(f"_{readout}_{label_a}")})
    if not corpora:
        print(f"[cmp] no OOD {readout} columns; skipping ladder")
        return
    rows = []
    for corpus in corpora:
        for label in (label_a, label_b):
            col = f"ood_{corpus}_{readout}_{label}"
            if col in merged:
                vals = merged[col].dropna()
                rows.append({"corpus": corpus, "arm": label, "mean": vals.mean(),
                             "se": vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan})
    # in-domain anchor first
    for label in (label_a, label_b):
        col = f"{readout}_corr_{label}"
        if col in merged:
            vals = merged[col].dropna()
            rows.append({"corpus": "in-domain", "arm": label, "mean": vals.mean(),
                         "se": vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan})
    df = pd.DataFrame(rows)
    order = ["in-domain"] + corpora
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, sub in df.groupby("arm"):
        sub = sub.set_index("corpus").reindex(order).reset_index()
        x = np.arange(len(order))
        ax.errorbar(x, sub["mean"], yerr=sub["se"], marker="o", capsize=3, label=label)
    ax.set_xticks(np.arange(len(order))); ax.set_xticklabels(order, rotation=20)
    ax.set_ylabel(f"{readout} corr"); ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="grey", ls="--", lw=1); ax.grid(alpha=0.25); ax.legend()
    ax.set_title(f"{readout} readout across the corpus ladder (mean ± SE over paired neurons)")
    fig.tight_layout()
    path = Path(out_dir) / f"ood_ladder_{readout}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[cmp] wrote {path}")


def plot_delta_heatmap(merged, metric, label_a, label_b, out_dir):
    """Within-neuron delta by (layer, position) -- where does one arm win?"""
    ca, cb = f"{metric}_{label_a}", f"{metric}_{label_b}"
    if ca not in merged or cb not in merged:
        return
    sub = merged[[*KEY, ca, cb]].dropna()
    if sub.empty:
        return
    sub = sub.assign(delta=sub[ca] - sub[cb])
    pivot = sub.pivot_table(index="layer", columns="position", values="delta", aggfunc="mean")
    lim = float(np.nanmax(np.abs(pivot.to_numpy()))) or 1.0
    plt.figure(figsize=(1.3 * pivot.shape[1] + 4, 0.7 * pivot.shape[0] + 3))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, vmin=-lim, vmax=lim, annot=True, fmt=".2f",
                cbar_kws={"label": f"{metric}: {label_a} - {label_b}"})
    plt.xlabel("token position"); plt.ylabel("layer")
    plt.title(f"{metric}: {label_a} minus {label_b} (paired by neuron)")
    plt.tight_layout()
    path = Path(out_dir) / f"delta_{metric}.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"[cmp] wrote {path}")


def main(group_a, group_b, label_a, label_b, out_dir, filters_a=None, filters_b=None):
    os.makedirs(out_dir, exist_ok=True)
    a = load_group(group_a, **(filters_a or {}))
    b = load_group(group_b, **(filters_b or {}))
    merged = pair(a, b, label_a, label_b)
    merged.to_csv(Path(out_dir) / "paired_runs.csv", index=False)

    metrics = ["interventional_corr", "endogenous_corr"]
    metrics += sorted({c.rsplit("_", 1)[0] for c in merged.columns
                       if c.startswith("ood_") and c.endswith(f"_{label_a}")
                       and "activation_std" not in c})
    stats = [s for s in (paired_stats(merged, m, label_a, label_b) for m in metrics) if s]
    table = pd.DataFrame(stats)
    table.to_csv(Path(out_dir) / "paired_summary.csv", index=False)
    print(f"\n[cmp] paired summary ({label_a} vs {label_b}):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plot_ood_ladder(merged, label_a, label_b, out_dir, readout="endogenous")
    plot_ood_ladder(merged, label_a, label_b, out_dir, readout="interventional")
    for m in ("interventional_corr", "endogenous_corr"):
        plot_delta_heatmap(merged, m, label_a, label_b, out_dir)
    return merged, table


if __name__ == "__main__":
    # joint training (w=0.01, KL budget 0.1) vs endogenous-only SFT (w=1.0), gate off
    main(group_a="mlp_joint_grid_layer_token", group_b="mlp_endo_sweep_layer_token",
         label_a="joint", label_b="sft",
         filters_a={"gate": "off", "w": 0.01, "kl": "0.1"},
         filters_b={"gate": "off", "w": 1.0},
         out_dir=f"{ROOT}/outputs/plots/joint_vs_sft")
