#!/usr/bin/env python3
"""
Seed variance of the joint - SFT gap (rebuttal: "means +/- std over 3-5 neurons ... no
seeds"), from scripts/neuron/train_seed_variance_grid.sh.

Per cell (layers {2, 8} x t {4, 31}) every neuron has up to 5 replicates per arm: train
seeds 1-4 (group mlp_seed_variance_layer_token) and the original grid run (unseeded; see
src/introspector/training/seeding.py). A replicate pairs joint seed s with SFT seed s
(the original with the original) into one delta. Reported per cell and metric:

  Joint / SFT          mean over neurons x replicates, +/- the SEED SD
  delta                mean joint - SFT
  seed SD (delta)      spread across replicates of the SAME neuron: the pooled
                       within-neuron SD, sqrt(mean over neurons of var over seeds)
  neuron SD (delta)    spread across neurons of each neuron's replicate-averaged delta --
                       the +/- Fig. 3 prints
  delta > 0            how many of the (neuron, replicate) deltas are positive
  p                    two-sided paired t-test over the neurons on the replicate-averaged
                       delta (the test plotters/stats_sft_vs_joint.py runs on one
                       replicate), Holm-corrected over the cells of the table
If the seed SD is small next to both the delta and the neuron SD, Fig. 3's per-cell
spread is neuron-to-neuron variation, not training noise, and one run per neuron was
enough to see the gap.

Every replicate is evaluated on the same test rows and injected values (the configs share
ExperimentConfig.seed), so the seed SD is training variance only.

Outputs (outputs/plots/seed_variance/ by default):
  seed_variance_runs.csv       every run: arm, cell, neuron, replicate, metric, corr
  seed_variance_summary.csv    the table below, per cell x metric
  seed_variance_tables.md      markdown for the rebuttal

Usage (reads run dirs, so on SCC):
python plotters/table_seed_variance.py
"""
from __future__ import annotations

import re
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEQ_LEN = 32
METRICS = {"endogenous": ("endogenous_test_metrics_endogenous.json", None),
           "interventional": ("interventional_test_metrics_gaussian.json", "test")}
NAME_RE = re.compile(r"^(?P<arm>joint|endo)_N(?P<layer>\d+)\.(?P<index>\d+)_POS(?P<pos>-?\d+)_"
                     r"(?P<rest>.*?)(?:_seed(?P<seed>\d+))?$")
ARM = {"joint": "joint", "endo": "sft"}
ARM_LABEL = {"joint": "Joint (KL=0.1)", "sft": "SFT"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    p.add_argument("--seed-group", default="mlp_seed_variance_layer_token")
    p.add_argument("--original-groups", default="joint:mlp_joint_grid_layer_token,"
                                                 "endo:mlp_endo_sweep_layer_token",
                   help="'<name prefix>:<group>' of the runs the seed replicates copy.")
    p.add_argument("--no-original", action="store_true",
                   help="Leave the original (unseeded) grid runs out of the replicates.")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "plots" / "seed_variance")
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def read_corr(run_dir: Path, metric: str) -> float | None:
    fname, key = METRICS[metric]
    p = run_dir / fname
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return (d.get(key, d) if key else d).get("corr")


def load_runs(runs_root: Path, seed_group: str, original_groups: dict, with_original: bool
              ) -> pd.DataFrame:
    rows, originals = [], set()
    for run_dir in sorted((runs_root / seed_group).iterdir()) if (runs_root / seed_group).exists() else []:
        m = NAME_RE.match(run_dir.name)
        if not m or m["seed"] is None:
            continue
        originals.add((m["arm"], run_dir.name[: -len(f"_seed{m['seed']}")]))
        rows += _rows(run_dir, m, f"seed{m['seed']}")
    if with_original:
        for prefix, orig in sorted(originals):
            run_dir = runs_root / original_groups[prefix] / orig
            m = NAME_RE.match(orig)
            if run_dir.exists():
                rows += _rows(run_dir, m, "original")
            else:
                print(f"[seeds] WARNING original run missing: {run_dir}")
    df = pd.DataFrame(rows, columns=["run", "arm", "layer", "index", "position", "t",
                                     "replicate", "metric", "corr"])
    return df.dropna(subset=["corr"])


def _rows(run_dir: Path, m, replicate: str) -> list[dict]:
    pos = int(m["pos"])
    return [{"run": run_dir.name, "arm": ARM[m["arm"]], "layer": int(m["layer"]),
             "index": int(m["index"]), "position": pos, "t": pos + SEQ_LEN if pos < 0 else pos,
             "replicate": replicate, "metric": metric, "corr": read_corr(run_dir, metric)}
            for metric in METRICS]


def holm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = len(ok)
    if m:
        order = ok[np.argsort(p[ok], kind="stable")]
        out[order] = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[order]))
    return out


def seed_and_neuron_sd(wide: pd.DataFrame) -> tuple[float, float]:
    """wide: rows = neurons, columns = replicates. (pooled within-neuron SD across
    replicates, SD across neurons of the replicate means)."""
    within = wide.var(axis=1, ddof=1)
    seed_sd = float(np.sqrt(within.dropna().mean())) if within.notna().any() else np.nan
    means = wide.mean(axis=1)
    neuron_sd = float(means.std(ddof=1)) if len(means) > 1 else np.nan
    return seed_sd, neuron_sd


def summarize(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for (metric, L, t), g in df.groupby(["metric", "layer", "t"]):
        wide = {a: g[g["arm"] == a].pivot_table(index="index", columns="replicate",
                                                values="corr", aggfunc="first")
                for a in ARM_LABEL}
        if any(w.empty for w in wide.values()):
            continue
        reps = sorted(set(wide["joint"].columns) & set(wide["sft"].columns))
        neurons = sorted(set(wide["joint"].index) & set(wide["sft"].index))
        delta = wide["joint"].loc[neurons, reps] - wide["sft"].loc[neurons, reps]
        row = {"metric": metric, "layer": L, "t": t, "n_neurons": len(neurons),
               "n_replicates": len(reps), "n_deltas": int(delta.notna().sum().sum())}
        for a, w in wide.items():
            row[f"{a}_mean"] = float(np.nanmean(w.to_numpy()))
            row[f"{a}_seed_sd"], row[f"{a}_neuron_sd"] = seed_and_neuron_sd(w)
        row["delta_mean"] = float(np.nanmean(delta.to_numpy()))
        row["delta_seed_sd"], row["delta_neuron_sd"] = seed_and_neuron_sd(delta)
        row["delta_positive"] = int((delta > 0).sum().sum())
        d = delta.mean(axis=1).dropna().to_numpy()       # replicate-averaged, per neuron
        if len(d) > 1 and d.std(ddof=1) > 1e-12:
            tt = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            row["t_stat"], row["p"] = float(tt), float(2 * student_t.sf(abs(tt), len(d) - 1))
        else:
            row["t_stat"], row["p"] = np.nan, np.nan
        rows.append(row)
    s = pd.DataFrame(rows)
    if s.empty:
        return s
    s["p_holm"] = np.nan
    for _, idx in s.groupby("metric").groups.items():
        s.loc[idx, "p_holm"] = holm(s.loc[idx, "p"])
    s["significant"] = s["p_holm"] < alpha
    return s


def fmt_p(p) -> str:
    if not np.isfinite(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write_markdown(s: pd.DataFrame, out: Path, alpha: float) -> str:
    md = []
    for metric in METRICS:
        sub = s[s["metric"] == metric].sort_values(["layer", "t"])
        if sub.empty:
            continue
        md += [f"**{metric.capitalize()} correlation, joint − SFT across training seeds**", "",
               "| Layer, token | Joint | SFT | Δ | Δ SD across seeds | Δ SD across neurons "
               "| Δ > 0 | p (Holm) |",
               "|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            d = f"{r['delta_mean']:+.2f}"
            md.append(
                f"| L{int(r['layer'])}, t={int(r['t'])} "
                f"| {r['joint_mean']:.2f} ± {r['joint_seed_sd']:.2f} "
                f"| {r['sft_mean']:.2f} ± {r['sft_seed_sd']:.2f} "
                f"| {'**' + d + '**' if r['significant'] else d} "
                f"| {r['delta_seed_sd']:.2f} | {r['delta_neuron_sd']:.2f} "
                f"| {int(r['delta_positive'])}/{int(r['n_deltas'])} | {fmt_p(r['p_holm'])} |")
        md.append("")
    n_rep = int(s["n_replicates"].max()) if len(s) else 0
    n_neu = int(s["n_neurons"].max()) if len(s) else 0
    md.append(f"{n_neu} neurons per cell x {n_rep} training replicates per arm (seeds 1–4 "
              f"and the original run), all evaluated on the same test rows and injected values. "
              f"Joint / SFT: mean ± SD across seeds (pooled within neuron). Δ SD across neurons: "
              f"of each neuron's seed-averaged Δ. p: paired t-test over neurons on the "
              f"seed-averaged Δ, Holm over the cells; **bold**: p < {alpha:g}.")
    text = "\n".join(md) + "\n"
    (out / "seed_variance_tables.md").write_text(text)
    return text


def main():
    args = parse_args()
    groups = dict(t.split(":", 1) for t in args.original_groups.split(","))
    df = load_runs(args.runs_root, args.seed_group, groups, not args.no_original)
    if df.empty:
        raise SystemExit(f"no finished runs under {args.runs_root / args.seed_group}")
    counts = df[df["metric"] == "endogenous"].groupby(["arm", "replicate"]).size()
    print("[seeds] finished runs per arm x replicate:\n" + counts.to_string())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "seed_variance_runs.csv", index=False)
    s = summarize(df, args.alpha)
    s.to_csv(args.out_dir / "seed_variance_summary.csv", index=False)
    print(write_markdown(s, args.out_dir, args.alpha))
    print(f"[seeds] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
