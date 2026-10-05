#!/usr/bin/env python3
"""
Do the endogenous and interventional evaluations rely on the same heads? -- pooled over
a sweep's score_head_knockout.py outputs.

Per run (one point each), against signal layer and position:
  RANK AGREEMENT    Spearman of the two arms' per-head corr drops, raw and
                    DISATTENUATED (divided by the geometric mean of each arm's split-half
                    reliability). Raw Spearman over every head is dragged down by the
                    hundreds of heads with no effect, whose ranks are noise; the
                    disattenuated value asks whether the arms disagree MORE than each
                    disagrees with itself.
  TOP-k OVERLAP     Jaccard of the two arms' top-k heads vs the layer-matched null (heads
                    shuffled within their layer), so agreeing on DEPTH alone earns nothing.
  TRANSFER          cumulative knockout down arm B's ranking, evaluated on arm A, as a
                    fraction of the damage A's own ranking does (1 = B's heads are as
                    good as A's own at breaking A; random orderings give the floor).
  HOW MANY HEADS    effective head count (participation ratio of the drops) per arm.

Across runs:
  GENERIC HEADS     for every (layer, head), the fraction of runs it is eligible in
                    (layer > signal layer) where it lands in an arm's top-k. A head in the
                    top-k of most runs, whatever the signal, is a candidate signal-agnostic
                    (task) head; chance is k / (eligible heads) per run.

Usage
-----
python analysis/plot_head_knockout.py --group mlp_joint_grid_layer_token --kl kl0.1
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
from analysis.run_catalog import NEURON_RE, POS_RE, KL_RE  # noqa: E402  (one name parser)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
A, B = "endogenous", "x1"


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
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: outputs/plots/<group>/head_knockout")
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
        loc = {"run": d.name, "signal_layer": int(n["layer"]),
               "signal_index": int(n["index"]), "position": int(pz["pos"])}
        row = dict(loc, n_heads=s["n_heads"], n_examples=s["n_examples"],
                   **{f"baseline_{a}": s["baseline"].get(a) for a in (A, B)})
        for key in (f"spearman_{A}_vs_{B}", f"disattenuated_{A}_vs_{B}",
                    f"reliability_{A}", f"reliability_{B}", f"effective_heads_{A}",
                    f"effective_heads_{B}", f"n_significant_{A}", f"n_significant_{B}",
                    f"max_drop_{A}", f"max_drop_{B}", f"top_head_{A}", f"top_head_{B}",
                    f"transfer_{A}_to_{B}", f"transfer_{B}_to_{A}",
                    f"transfer_random_to_{A}", f"transfer_random_to_{B}",
                    f"significant_jaccard_{A}_vs_{B}"):
            row[key] = s.get(key, np.nan)
        for t in s.get(f"topk_{A}_vs_{B}", []):
            row[f"jaccard_top{t['k']}"] = t["jaccard"]
            row[f"null_top{t['k']}"] = t["null_mean"]
            row[f"p_top{t['k']}"] = t["p"]
        runs.append(row)
        heads.append(pd.read_csv(i_p).assign(**loc))
    print(f"[knockout-agg] {len(runs)} run(s) with knockout results under {root}")
    if missing:
        print(f"[knockout-agg] {len(missing)} run(s) without results, e.g. {missing[:3]}")
    if not runs:
        raise SystemExit("Nothing to aggregate.")
    return pd.DataFrame(runs), pd.concat(heads, ignore_index=True)


# ======================================================================
# Per-run agreement against locale
# ======================================================================
def agreement_by_locale(runs: pd.DataFrame, k: int, out: Path):
    panels = [
        ("rank agreement", [(f"spearman_{A}_vs_{B}", "Spearman (raw)", "#7f8c8d"),
                            (f"disattenuated_{A}_vs_{B}", "Spearman (disattenuated)", "#c0392b")]),
        (f"top-{k} Jaccard", [(f"jaccard_top{k}", "observed", "#2471a3"),
                              (f"null_top{k}", "layer-matched null", "#bdc3c7")]),
        ("transfer ratio", [(f"transfer_{A}_to_{B}", f"{A}-ranked, on {B}", "#c0392b"),
                            (f"transfer_{B}_to_{A}", f"{B}-ranked, on {A}", "#2471a3"),
                            (f"transfer_random_to_{B}", f"random, on {B}", "#bdc3c7")]),
        ("effective heads", [(f"effective_heads_{A}", A, "#c0392b"),
                             (f"effective_heads_{B}", B, "#2471a3")]),
    ]
    xs = [("signal_layer", "signal layer"), ("position", "token position")]
    fig, axes = plt.subplots(len(xs), len(panels), figsize=(4.0 * len(panels), 3.3 * len(xs)),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for r, (xcol, xlabel) in enumerate(xs):
        levels = sorted(runs[xcol].unique())
        pos = {v: i for i, v in enumerate(levels)}
        for c, (title, series) in enumerate(panels):
            ax = axes[r, c]
            width = 0.8 / len(series)
            for si, (col, label, color) in enumerate(series):
                if col not in runs:
                    continue
                off = (si - (len(series) - 1) / 2) * width
                x = runs[xcol].map(pos) + off
                ax.scatter(x + rng.uniform(-width / 4, width / 4, len(x)), runs[col],
                           s=8, color=color, alpha=0.5, label=label)
                med = runs.groupby(xcol)[col].median()
                ax.scatter([pos[v] + off for v in med.index], med.values, marker="_",
                           s=200, color=color, linewidths=2)
            ax.set_xticks(range(len(levels)), levels)
            ax.set_xlabel(xlabel)
            if r == 0:
                ax.set_title(title)
            if "transfer" in title or "Spearman" in title or "rank" in title:
                ax.axhline(0, color="k", lw=0.5)
            if "transfer" in title or "rank" in title:
                ax.axhline(1, color="k", lw=0.5, ls=":")
            ax.grid(alpha=0.25)
            if r == 0:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"Endogenous vs interventional head importance, per run (bar = median)",
                 y=1.01)
    fig.tight_layout()
    fig.savefig(out / "knockout_agreement_by_locale.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def pooled_table(runs: pd.DataFrame, k: int) -> pd.DataFrame:
    cols = [f"baseline_{A}", f"baseline_{B}", f"reliability_{A}", f"reliability_{B}",
            f"spearman_{A}_vs_{B}", f"disattenuated_{A}_vs_{B}", f"jaccard_top{k}",
            f"null_top{k}", f"transfer_{A}_to_{B}", f"transfer_{B}_to_{A}",
            f"transfer_random_to_{A}", f"transfer_random_to_{B}",
            f"effective_heads_{A}", f"effective_heads_{B}"]
    cols = [c for c in cols if c in runs]
    q = runs[cols].quantile([0.25, 0.5, 0.75]).T
    q.columns = ["q25", "median", "q75"]
    q["n_runs"] = runs[cols].notna().sum()
    if f"p_top{k}" in runs:
        q.loc[f"frac_runs_top{k}_p<0.05", ["median", "n_runs"]] = [
            float((runs[f"p_top{k}"] < 0.05).mean()), int(runs[f"p_top{k}"].notna().sum())]
    return q


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

    arms = [a for a in (A, B) if a in set(prev["arm"])]
    n_layers = int(prev["layer"].max()) + 1
    n_heads = int(prev["head"].max()) + 1
    lo = int(prev["layer"].min())
    fig, axes = plt.subplots(len(arms), 1, squeeze=False,
                             figsize=(max(6, 0.3 * n_heads), 0.6 + len(arms) * (0.3 * (n_layers - lo) + 1)))
    for ax, arm in zip(axes[:, 0], arms):
        m = (prev[prev["arm"] == arm].pivot(index="layer", columns="head", values="prevalence")
             .reindex(index=range(lo, n_layers), columns=range(n_heads)))
        sns.heatmap(m, ax=ax, cmap="magma_r", vmin=0, vmax=1, cbar_kws={"label": f"frac runs in top-{k}"})
        ax.set_title(f"{arm}: how often each head is in a run's top-{k} "
                     f"(of runs where it is above the signal)", fontsize=9)
        ax.set_ylabel("layer"); ax.tick_params(labelsize=6)
    axes[-1, 0].set_xlabel("head")
    fig.tight_layout()
    fig.savefig(out / "knockout_head_prevalence.png", dpi=150)
    plt.close(fig)
    return prev


def main():
    args = parse_args()
    out = args.out_dir or ROOT / "outputs" / "plots" / args.group / args.analysis
    out.mkdir(parents=True, exist_ok=True)
    runs, heads = load(args)
    runs.to_csv(out / "knockout_runs.csv", index=False)
    agreement_by_locale(runs, args.k, out)
    q = pooled_table(runs, args.k)
    q.to_csv(out / "knockout_pooled.csv")
    print("\n[knockout-agg] pooled over runs:\n" + q.to_string(float_format=lambda v: f"{v:.3f}"))
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
