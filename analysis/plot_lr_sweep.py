#!/usr/bin/env python3
"""
Read an LR sweep and apply its decision rule: did the 1B learning rate transfer?

Runs come from scripts/neuron/train_single_mlp_introspector_joint_lr_sweep.sh (group
mlp_joint_lr_sweep_<model>): LR x {kl0.1, unanchored} x 3 cells x 2 tuning neurons that
no grid trains. The 8B runs of the older ..._lr_sweep_llama318b.sh share its group/names.

For each (cell, regime) it reports the corr at the grid's LR (5e-5), the best LR and its
gain, and -- as the target -- Llama-3.2-1B at the nearest relative depth and the same
position, so "how much of the gap to 1B does tuning close?" has a number. (For Qwen that
target is cross-family: what the protocol reached at 1B, not a same-family baseline.)

NOISE IS LARGE: two neurons per point. The spread between them is printed beside every
gain; a gain smaller than that spread is not evidence the optimum moved.

Reads run directories when they are present and writes lr_sweep_runs.csv; without them
it redraws from that CSV (outputs/runs is partial on a laptop).

  python analysis/plot_lr_sweep.py                          # llama318b
  python analysis/plot_lr_sweep.py --model llama323b
  python analysis/plot_lr_sweep.py --model qwen25_05b,llama323b,qwen25_7b,llama318b
With several models it also overlays their LR curves per cell role and writes the best
LR per (model, role, regime) to outputs/plots/lr_sweep_compare/.
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
# sweep key -> (display name, n_layers, params in B). The key is SWEEP_MODEL in the script
MODELS = {
    "qwen25_05b": ("Qwen2.5-0.5B", 24, 0.49),
    "qwen25_15b": ("Qwen2.5-1.5B", 28, 1.54),
    "qwen25_3b": ("Qwen2.5-3B", 36, 3.09),
    "llama323b": ("Llama-3.2-3B", 28, 3.21),
    "qwen25_7b": ("Qwen2.5-7B", 28, 7.62),
    "llama318b": ("Llama-3.1-8B", 32, 8.03),
    "qwen25_14b": ("Qwen2.5-14B", 48, 14.7),
}
# every model's three cells play the same role, keyed by position (see the sweep script)
ROLE = {-8: "mid (~3/8)", -1: "shallow (~1/8)", -28: "deep (~7/8)"}
ROLE_ORDER = (-8, -1, -28)
GRID_LR = 5e-5
REF_MODEL = "Llama-3.2-1B"
REF_DEPTH_TOL = 0.1            # nearest 1B depth must be this close to count as a reference
REGIME_OF_TAG = {"kl0.1": "kl0.1", "klnone": "unanchored"}
REF_ARM = {"kl0.1": "joint_kl0.1", "unanchored": "joint_free"}
READOUTS = ("endogenous", "interventional")
COLORS = {"kl0.1": "#1f77b4", "unanchored": "#d62728"}
NAME_RE = re.compile(r"lrsweep_N(?P<layer>\d+)\.(?P<index>\d+)_POS(?P<pos>-?\d+)_.*?_"
                     r"(?P<kl>kl[0-9.]+|klnone)_w[0-9.]+_lr(?P<lr>[0-9.eE+-]+)$")


def group_of(key: str) -> str:
    return f"mlp_joint_lr_sweep_{key}"


def _js(p: Path):
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:
        return None


def load_runs(group: str) -> pd.DataFrame | None:
    base = ROOT / "outputs" / "runs" / group
    cache = ROOT / "outputs" / "plots" / group / "lr_sweep_runs.csv"
    if not base.exists():
        if cache.exists():
            print(f"[lr] no run directories under {base}; redrawing from {cache.name}")
            return pd.read_csv(cache)
        print(f"[lr] neither {base} nor {cache} exists -- rsync the sweep first")
        return None
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
        print(f"[lr] no finished runs under {base}")
        return None
    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def ref_1b(df: pd.DataFrame, n_layers: int) -> pd.DataFrame:
    """1B at the NEAREST relative depth and the same position, per regime -- the target
    line. Depths match exactly for 8B; 3B and Qwen fall between 1B layers, so the 1B
    depth actually used is kept in ref_depth."""
    f = ROOT / "outputs" / "plots" / "model_scaling" / "model_scaling_runs.csv"
    if not f.exists():
        print(f"[lr] {f.name} absent; run analysis/compare_model_scaling.py for the 1B reference")
        return pd.DataFrame()
    s = pd.read_csv(f)
    s = s[(s["model"] == REF_MODEL) & (s["metric"] == "corr")]
    out = []
    for (layer, pos), _ in df.groupby(["layer", "position"]):
        depth = layer / n_layers
        for regime, arm in REF_ARM.items():
            for readout in READOUTS:
                q = s[(s["position"] == pos) & (s["arm"] == arm) & (s["readout"] == readout)]
                if q.empty:
                    continue
                near = q["depth"].iloc[(q["depth"] - depth).abs().argmin()]
                if abs(near - depth) > REF_DEPTH_TOL:
                    continue
                out.append({"layer": layer, "position": pos, "regime": regime,
                            "readout": readout, "depth": round(depth, 3),
                            "ref_depth": float(near),
                            "ref_1b": float(q[q["depth"] == near]["value"].mean())})
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
        swept = sorted(q["lr"])
        r = {"cell": f"L{layer} pos{pos}", "position": pos, "regime": regime,
             "corr_at_5e-5": float(at_grid["mean"].iloc[0]) if len(at_grid) else np.nan,
             "best_lr": best["lr"], "corr_at_best": best["mean"],
             "neuron_spread_at_best": best["spread"], "n_lrs": len(q),
             # how many tuning neurons stand behind each side of the comparison. With
             # ONE, the spread is 0 by definition and the noise guard below is vacuous
             "n_at_best": int(best["n"]),
             "n_at_5e-5": int(at_grid["n"].iloc[0]) if len(at_grid) else 0,
             # the optimum sits on the boundary of what was tried: the true best LR may
             # lie beyond it, and "best" is then a lower bound on how far it moved
             "at_edge": bool(len(swept) > 1 and best["lr"] in (swept[0], swept[-1])
                             and not np.isclose(best["lr"], GRID_LR))}
        r["gain"] = r["corr_at_best"] - r["corr_at_5e-5"]
        if not ref.empty:
            rr = ref[(ref["layer"] == layer) & (ref["position"] == pos)
                     & (ref["regime"] == regime) & (ref["readout"] == readout)]
            if len(rr):
                r["1B_ref"] = float(rr["ref_1b"].iloc[0])
                r["1B_ref_depth"] = float(rr["ref_depth"].iloc[0])
                gap = r["1B_ref"] - r["corr_at_5e-5"]
                r["gap_to_1B_at_5e-5"] = gap
                # only meaningful when the model is actually behind; otherwise no gap
                r["frac_gap_closed"] = r["gain"] / gap if gap > 0.02 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def _cells(df: pd.DataFrame):
    """(layer, position) cells in role order: the main (pos -8) cell first."""
    rank = {p: i for i, p in enumerate(ROLE_ORDER)}
    return sorted(df.groupby(["layer", "position"]).groups,
                  key=lambda c: (rank.get(c[1], len(rank)), c))


def plot(df: pd.DataFrame, ref: pd.DataFrame, out_dir: Path, n_layers: int, name: str,
         metric="corr"):
    cells = _cells(df)
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
                ax.set_title(f"L{layer} pos {pos}  (depth {layer / n_layers:.3f})")
            if r == len(READOUTS) - 1:
                ax.set_xlabel("learning rate")
            if c == 0:
                ax.set_ylabel(f"{readout} {metric}")
    axes[0][-1].legend(fontsize=9)
    fig.suptitle(f"{name}: LR sweep (dashed = {REF_MODEL} at nearest depth)", fontsize=11)
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


def plot_compare(runs: dict[str, pd.DataFrame], out_dir: Path, metric="corr",
                 readout="interventional"):
    """One panel per (regime, cell role), one LR curve per model, models ordered by size:
    does the optimal LR drift with scale, and does it drift the same way in both
    families?"""
    col = f"{readout}_{metric}"
    keys = sorted(runs, key=lambda k: MODELS[k][2])
    cmap = plt.get_cmap("viridis")
    regimes = ("kl0.1", "unanchored")
    fig, axes = plt.subplots(len(regimes), len(ROLE_ORDER), figsize=(5 * len(ROLE_ORDER), 8.2),
                             squeeze=False, sharex=True, sharey="row")
    for i, key in enumerate(keys):
        df = runs[key]
        if col not in df.columns:
            continue
        name, _, _ = MODELS[key]
        color = cmap(i / max(len(keys) - 1, 1))
        ls = "-" if name.startswith("Llama") else "--"
        for r, regime in enumerate(regimes):
            for c, pos in enumerate(ROLE_ORDER):
                q = df[(df["position"] == pos) & (df["regime"] == regime)]
                if q.empty:
                    continue
                m = q.groupby("lr")[col].mean().reset_index().sort_values("lr")
                axes[r][c].plot(m["lr"], m[col], marker="o", ms=5, lw=1.8, ls=ls,
                                color=color, label=name)
                best = m.loc[m[col].idxmax()]
                axes[r][c].scatter([best["lr"]], [best[col]], s=110, facecolors="none",
                                   edgecolors=color, lw=1.6, zorder=3)
    for r, regime in enumerate(regimes):
        for c, pos in enumerate(ROLE_ORDER):
            ax = axes[r][c]
            ax.axvline(GRID_LR, color="grey", ls=":", lw=1)
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(f"{ROLE[pos]}, pos {pos}")
            if r == len(regimes) - 1:
                ax.set_xlabel("learning rate")
            if c == 0:
                ax.set_ylabel(f"{regime}\n{readout} {metric}")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9,
                   frameon=False)
    fig.suptitle("LR curves by model (circle = best LR; solid Llama, dashed Qwen)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_dir / f"lr_curves_by_model_{readout}_{metric}.png", dpi=150)
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


def report(t: pd.DataFrame):
    """Apply the decision rule fixed before the results and print the verdict."""
    print("\n[lr] interventional corr -- decision table (fixed before the results):")
    # LRs must not go through round(3): 5e-5 would print as 0.0
    print(_fmt(t.drop(columns="position")).to_string(index=False))
    # The noise guard needs BOTH tuning neurons on both sides of the comparison; with one,
    # neuron_spread is 0 by definition and any gain "exceeds" it. Rows short of that are
    # reported as provisional rather than counted either way.
    judgeable = t[(t["n_at_best"] >= 2) & (t["n_at_5e-5"] >= 2)]
    provisional = t.drop(judgeable.index)
    moved = judgeable[(~np.isclose(judgeable["best_lr"], GRID_LR))
                      & (judgeable["gain"] > judgeable["neuron_spread_at_best"])]
    print(f"\n[lr] judgeable (both tuning neurons finished at the best LR and at "
          f"{GRID_LR:g}): {len(judgeable)} of {len(t)} (cell, regime) pairs")
    if len(provisional):
        print(f"[lr] PROVISIONAL -- only one neuron behind the comparison, so the noise guard "
              f"cannot apply:")
        print(_fmt(provisional[["cell", "regime", "best_lr", "gain", "n_at_best", "n_at_5e-5"]])
              .to_string(index=False))
    edge = t[t["at_edge"]]
    if len(edge):
        print(f"[lr] AT THE EDGE of the swept LR range (the optimum may lie beyond it -- "
              f"extend the grid before trusting 'best_lr'):")
        print(_fmt(edge[["cell", "regime", "best_lr", "gain"]]).to_string(index=False))
    if len(judgeable):
        print(f"\n[lr] of the judgeable pairs, {len(moved)} have a best LR other than "
              f"{GRID_LR:g} with a gain above the spread between the two tuning neurons")
        if len(moved):
            print(_fmt(moved[["cell", "regime", "best_lr", "gain", "neuron_spread_at_best"]])
                  .to_string(index=False))
            print("[lr] -> the 1B LR did not transfer cleanly here; re-run the grid at the "
                  "better LR before reading the scale trend")
        else:
            print(f"[lr] -> no evidence the optimum moved; the grid at {GRID_LR:g} stands")
    else:
        print("[lr] -> nothing judgeable yet; wait for the second tuning neuron")


def analyze(key: str, group: str | None = None):
    name, n_layers, _ = MODELS[key]
    group = group or group_of(key)
    print(f"\n{'=' * 78}\n[lr] {name}  ({group})\n{'=' * 78}")
    out_dir = ROOT / "outputs" / "plots" / group
    df = load_runs(group)
    if df is None:
        return None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[lr] {len(df)} finished run(s): "
          + ", ".join(f"{k}={v}" for k, v in df.groupby('regime').size().items()))
    ref = ref_1b(df, n_layers)
    for metric in ("corr", "spearman"):
        if f"interventional_{metric}" in df.columns and df[f"interventional_{metric}"].notna().any():
            plot(df, ref, out_dir, n_layers, name, metric)
    t = decision_table(df, ref)
    t.to_csv(out_dir / "lr_sweep_decision.csv", index=False)
    report(t)
    print(f"\n[lr] wrote {out_dir}")
    return df, t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama318b",
                    help=f"sweep key(s), comma-separated: {', '.join(MODELS)}")
    ap.add_argument("--group", default=None,
                    help="override the run group (single --model only)")
    args = ap.parse_args()
    keys = [k.strip() for k in args.model.split(",") if k.strip()]
    bad = [k for k in keys if k not in MODELS]
    if bad:
        sys.exit(f"[lr] unknown model key(s) {bad}; known: {list(MODELS)}")
    if args.group and len(keys) > 1:
        sys.exit("[lr] --group only makes sense with a single --model")

    runs, tables = {}, []
    for key in keys:
        df, t = analyze(key, args.group)
        if df is None:
            continue
        runs[key] = df
        name, n_layers, params = MODELS[key]
        tables.append(t.assign(model=name, params_b=params,
                               role=t["position"].map(ROLE)))
    if len(runs) < 2:
        return

    out_dir = ROOT / "outputs" / "plots" / "lr_sweep_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("corr", "spearman"):
        plot_compare(runs, out_dir, metric)
    s = (pd.concat(tables, ignore_index=True)
           .sort_values(["regime", "position", "params_b"],
                        key=lambda c: c.map({p: i for i, p in enumerate(ROLE_ORDER)})
                        if c.name == "position" else c))
    s.to_csv(out_dir / "best_lr_by_model.csv", index=False)
    cols = ["regime", "role", "model", "best_lr", "corr_at_5e-5", "corr_at_best", "gain",
            "n_at_best", "at_edge"]
    print(f"\n{'=' * 78}\n[lr] best LR by model (interventional corr), smallest model first\n"
          f"{'=' * 78}")
    print(_fmt(s[cols]).to_string(index=False))
    print(f"\n[lr] wrote {out_dir}")


if __name__ == "__main__":
    main()
