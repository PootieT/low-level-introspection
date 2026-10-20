#!/usr/bin/env python3
"""
Read the 8B LR sweep and apply its decision rule: did the 1B learning rate transfer?

Runs come from scripts/neuron/train_single_mlp_introspector_joint_lr_sweep_llama318b.sh
(group mlp_joint_lr_sweep_llama318b): LR x {kl0.1, unanchored} x 3 cells x 2 tuning
neurons that the grid never trained.

For each (cell, regime) it reports the corr at the grid's LR (5e-5), the best LR and its
gain, and -- as the target -- Llama-3.2-1B at the same relative depth and position, so
the question "how much of the 1B-vs-8B gap does tuning close?" has a number.

NOISE IS LARGE: two neurons per point. The spread between them is printed beside every
gain; a gain smaller than that spread is not evidence the optimum moved.

Reads run directories when they are present and writes lr_sweep_runs.csv; without them
it redraws from that CSV (outputs/runs is partial on a laptop).
"""
from __future__ import annotations

import re
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
GROUP = "mlp_joint_lr_sweep_llama318b"
N_LAYERS = 32                  # Llama-3.1-8B
GRID_LR = 5e-5
REF_MODEL = "Llama-3.2-1B"
REGIME_OF_TAG = {"kl0.1": "kl0.1", "klnone": "unanchored"}
REF_ARM = {"kl0.1": "joint_kl0.1", "unanchored": "joint_free"}
READOUTS = ("endogenous", "interventional")
COLORS = {"kl0.1": "#1f77b4", "unanchored": "#d62728"}
NAME_RE = re.compile(r"lrsweep_N(?P<layer>\d+)\.(?P<index>\d+)_POS(?P<pos>-?\d+)_.*?_"
                     r"(?P<kl>kl[0-9.]+|klnone)_w[0-9.]+_lr(?P<lr>[0-9.eE+-]+)$")


def _js(p: Path):
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:
        return None


def load_runs(group=GROUP) -> pd.DataFrame:
    base = ROOT / "outputs" / "runs" / group
    cache = ROOT / "outputs" / "plots" / group / "lr_sweep_runs.csv"
    if not base.exists():
        if cache.exists():
            print(f"[lr] no run directories under {base}; redrawing from {cache.name}")
            return pd.read_csv(cache)
        sys.exit(f"[lr] neither {base} nor {cache} exists -- rsync the sweep first")
    rows, unfinished = [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        m = NAME_RE.match(d.name)
        if not m:
            continue
        interv = _js(d / "interventional_test_metrics_gaussian.json")
        endo = _js(d / "endogenous_test_metrics_endogenous.json")
        if interv is None or endo is None:
            unfinished.append(d.name)
            continue
        it = interv.get("test", interv)
        row = {"run": d.name, "layer": int(m["layer"]), "index": int(m["index"]),
               "position": int(m["pos"]), "regime": REGIME_OF_TAG.get(m["kl"], m["kl"]),
               "lr": float(m["lr"])}
        for metric in ("corr", "spearman"):
            row[f"interventional_{metric}"] = it.get(metric)
            row[f"endogenous_{metric}"] = endo.get(metric)
        kl = _js(d / "lm_degradation.json")
        if kl:
            for e in kl.get("lengths", []):
                row[f"kl@{e['length']}"] = e.get("kl_mean")
        rows.append(row)
    if unfinished:
        print(f"[lr] {len(unfinished)} run(s) without test metrics yet, e.g. {unfinished[:2]}")
    if not rows:
        sys.exit(f"[lr] no finished runs under {base}")
    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def ref_1b(df: pd.DataFrame) -> pd.DataFrame:
    """1B at the matched relative depth and position, per regime -- the target line."""
    f = ROOT / "outputs" / "plots" / "model_scaling" / "model_scaling_runs.csv"
    if not f.exists():
        print(f"[lr] {f.name} absent; run analysis/compare_model_scaling.py for the 1B reference")
        return pd.DataFrame()
    s = pd.read_csv(f)
    s = s[(s["model"] == REF_MODEL) & (s["metric"] == "corr")]
    out = []
    for (layer, pos), _ in df.groupby(["layer", "position"]):
        depth = round(layer / N_LAYERS, 3)
        for regime, arm in REF_ARM.items():
            for readout in READOUTS:
                v = s[(s["depth"] == depth) & (s["position"] == pos) & (s["arm"] == arm)
                      & (s["readout"] == readout)]["value"]
                if len(v):
                    out.append({"layer": layer, "position": pos, "regime": regime,
                                "readout": readout, "ref_1b": float(v.mean()),
                                "depth": depth})
    return pd.DataFrame(out)


def decision_table(df: pd.DataFrame, ref: pd.DataFrame, readout="interventional") -> pd.DataFrame:
    col = f"{readout}_corr"
    g = (df.groupby(["layer", "position", "regime", "lr"])[col]
           .agg(mean="mean", spread=lambda v: float(v.max() - v.min()), n="count")
           .reset_index())
    rows = []
    for (layer, pos, regime), q in g.groupby(["layer", "position", "regime"]):
        at_grid = q[np.isclose(q["lr"], GRID_LR)]
        best = q.loc[q["mean"].idxmax()]
        r = {"cell": f"L{layer} pos{pos}", "regime": regime,
             "corr_at_5e-5": float(at_grid["mean"].iloc[0]) if len(at_grid) else np.nan,
             "best_lr": best["lr"], "corr_at_best": best["mean"],
             "neuron_spread_at_best": best["spread"], "n_lrs": len(q)}
        r["gain"] = r["corr_at_best"] - r["corr_at_5e-5"]
        if not ref.empty:
            rr = ref[(ref["layer"] == layer) & (ref["position"] == pos)
                     & (ref["regime"] == regime) & (ref["readout"] == readout)]
            if len(rr):
                r["1B_ref"] = float(rr["ref_1b"].iloc[0])
                gap = r["1B_ref"] - r["corr_at_5e-5"]
                r["gap_to_1B_at_5e-5"] = gap
                # only meaningful when 8B is actually behind; otherwise there is no gap
                r["frac_gap_closed"] = r["gain"] / gap if gap > 0.02 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, ref: pd.DataFrame, out_dir: Path, metric="corr"):
    cells = sorted(df.groupby(["layer", "position"]).groups,
                   key=lambda c: (c != (12, -8), c))       # the main cell first
    fig, axes = plt.subplots(len(READOUTS), len(cells), figsize=(5 * len(cells), 8.2),
                             squeeze=False, sharex=True, sharey="row")
    for r, readout in enumerate(READOUTS):
        col = f"{readout}_{metric}"
        if col not in df.columns:
            continue
        for c, (layer, pos) in enumerate(cells):
            ax = axes[r][c]
            for regime in ("kl0.1", "unanchored"):
                q = df[(df["layer"] == layer) & (df["position"] == pos) & (df["regime"] == regime)]
                if q.empty:
                    continue
                ax.scatter(q["lr"], q[col], s=16, alpha=0.45, color=COLORS[regime])
                m = q.groupby("lr")[col].mean().reset_index().sort_values("lr")
                ax.plot(m["lr"], m[col], marker="o", ms=6, lw=1.8, color=COLORS[regime],
                        label=regime)
                if not ref.empty and metric == "corr":
                    rr = ref[(ref["layer"] == layer) & (ref["position"] == pos)
                             & (ref["regime"] == regime) & (ref["readout"] == readout)]
                    if len(rr):
                        ax.axhline(rr["ref_1b"].iloc[0], color=COLORS[regime], ls="--",
                                   lw=1.2, alpha=0.7)
            ax.axvline(GRID_LR, color="grey", ls=":", lw=1)
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(f"L{layer} pos {pos}  (depth {layer / N_LAYERS:.3f})")
            if r == len(READOUTS) - 1:
                ax.set_xlabel("learning rate")
            if c == 0:
                ax.set_ylabel(f"{readout} {metric}")
    axes[0][-1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"lr_sweep_{metric}.png", dpi=150)
    plt.close(fig)

    if "kl@32" in df.columns and df["kl@32"].notna().any():
        fig, axes = plt.subplots(1, len(cells), figsize=(5 * len(cells), 4.2),
                                 squeeze=False, sharey=True)
        for c, (layer, pos) in enumerate(cells):
            ax = axes[0][c]
            for regime in ("kl0.1", "unanchored"):
                q = (df[(df["layer"] == layer) & (df["position"] == pos) & (df["regime"] == regime)]
                     .dropna(subset=["kl@32"]).sort_values("lr"))
                if len(q):
                    ax.plot(q["lr"], q["kl@32"], marker="o", lw=1.8, color=COLORS[regime],
                            label=regime)
            ax.axvline(GRID_LR, color="grey", ls=":", lw=1)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("learning rate")
            ax.set_title(f"L{layer} pos {pos}")
        axes[0][0].set_ylabel("KL(base‖tuned) nats/token @32")
        axes[0][-1].legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "lr_sweep_kl.png", dpi=150)
        plt.close(fig)


def _fmt(t: pd.DataFrame) -> pd.DataFrame:
    """Round scores to 3 decimals but print learning rates in %g."""
    t = t.copy()
    for c in t.columns:
        if c.endswith("lr"):
            t[c] = t[c].map(lambda v: f"{v:g}")
        elif t[c].dtype.kind == "f":
            t[c] = t[c].round(3)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default=GROUP)
    args = ap.parse_args()
    out_dir = ROOT / "outputs" / "plots" / args.group
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(args.group)
    print(f"[lr] {len(df)} finished run(s): "
          + ", ".join(f"{k}={v}" for k, v in df.groupby('regime').size().items()))
    ref = ref_1b(df)
    for metric in ("corr", "spearman"):
        if f"interventional_{metric}" in df.columns and df[f"interventional_{metric}"].notna().any():
            plot(df, ref, out_dir, metric)

    t = decision_table(df, ref)
    t.to_csv(out_dir / "lr_sweep_decision.csv", index=False)
    print("\n[lr] interventional corr -- decision table (fixed before the results):")
    # LRs must not go through round(3): 5e-5 would print as 0.0
    print(_fmt(t).to_string(index=False))
    moved = t[(t["best_lr"] != GRID_LR) & (t["gain"] > t["neuron_spread_at_best"])]
    print(f"\n[lr] (cell, regime) pairs where the best LR is not {GRID_LR:g} AND the gain "
          f"exceeds the spread between the two tuning neurons: {len(moved)} of {len(t)}")
    if len(moved):
        print(_fmt(moved[["cell", "regime", "best_lr", "gain", "neuron_spread_at_best"]])
              .to_string(index=False))
        print("[lr] -> the 1B LR did not transfer cleanly here; re-run the 8B grid at the "
              "better LR before reading the scale trend")
    else:
        print(f"[lr] -> no evidence the optimum moved; the 8B grid at {GRID_LR:g} stands")
    print(f"\n[lr] wrote {out_dir}")


if __name__ == "__main__":
    main()
