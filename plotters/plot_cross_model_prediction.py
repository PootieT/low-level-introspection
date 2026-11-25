#!/usr/bin/env python3
"""
Privileged-access baseline: can ANOTHER model predict a model's neuron from the text as
well as the model's own introspector does? (rebuttal W4; Li et al. 2025)

A 3x3 grid per metric: ROWS = predictor, COLUMNS = source (whose neuron is predicted).
  diagonal      the source's own SFT run (w=1, no KL, gate off) from its grid
  off-diagonal  scripts/neuron/train_cross_model_sft.sh: the predictor trained on the
                SOURCE's cached rows (same texts, same activations, same test split),
                SFT at the predictor's own LR, endogenous eval only
Each cell: mean +/- std over the column's signals (4 cells x 5 neurons of the source).
Only signals that every predictor of a column has are kept, so the three cells of a
column average over exactly the same signals.

SIGNIFICANCE. Within a column, each cross predictor vs the self run is a two-sided
PAIRED t-test over the column's signals (pairs share the neuron, the rows and the
labels), on the metric as shown, Holm-corrected over every (column, cross predictor,
metric) test. Significant cross cells carry an asterisk in the figure and are bold in
the markdown. Same convention as plotters/plot_injection_regimes.py.

Interventional correlation is deliberately absent: a model that only reads the text
cannot follow an injection into another model, so its interventional corr is 0 by
construction -- which is the point of the interventional eval.

Outputs (outputs/plots/cross_model_prediction/ by default):
  cross_model_runs.csv      one row per run: predictor, source, signal, pearson, spearman
  cross_model_summary.csv   mean / std / n per predictor x source x metric
  cross_model_tests.csv     self vs cross per column: n, means, t, p, Holm p, significant
  cross_model_tables.md     one markdown table per metric (the rebuttal allows no images)
  cross_model_heatmap.png / .pdf

Usage
-----
python plotters/plot_cross_model_prediction.py
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotters.plot_injection_regimes import holm  # noqa: E402

# key -> (display name, the model's own SFT grid)
MODELS = {
    "llama1b": ("Llama-3.2-1B", "mlp_endo_sweep_layer_token"),
    "qwen06b": ("Qwen3-0.6B", "mlp_endo_sweep_layer_token_qwen3_06b_hyper_tuned"),
    "qwen17b": ("Qwen3-1.7B", "mlp_endo_sweep_layer_token_qwen3_17b_hyper_tuned"),
}
CROSS_GROUP = "mlp_cross_model_sft"
METRICS = {"pearson": ("corr", "Pearson"), "spearman": ("spearman", "Spearman")}
SELF_RE = re.compile(r"^endo_N(?P<layer>\d+)\.(?P<index>\d+)_POS(?P<pos>-?\d+)_gateoff_w1\.0_300K$")
CROSS_RE = re.compile(r"^cross_(?P<pred>[a-z0-9]+)_from_(?P<src>[a-z0-9]+)_N(?P<layer>\d+)\."
                      r"(?P<index>\d+)_POS(?P<pos>-?\d+)_gateoff_w1\.0_300K$")

# dataviz reference sequential ramp (blue, light -> dark), steps 100-700
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED = "#0b0b0b", "#52514e"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    p.add_argument("--models", default=",".join(MODELS),
                   help="Comma list of model keys, in row/column order.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--correction", choices=("holm", "none"), default="holm")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "outputs" / "plots" / "cross_model_prediction")
    return p.parse_args()


def read_metrics(run_dir: Path) -> dict | None:
    p = run_dir / "endogenous_test_metrics_endogenous.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {m: d.get(k) for m, (k, _) in METRICS.items()}


def collect(runs_root: Path, models: list[str]) -> pd.DataFrame:
    """Cross runs first; they define each source's signal set. Self runs are then read
    for exactly those signals (each SFT grid holds more cells and neurons)."""
    rows = []
    for d in sorted((runs_root / CROSS_GROUP).glob("cross_*")):
        m = CROSS_RE.match(d.name)
        if not m or m["pred"] not in models or m["src"] not in models:
            continue
        met = read_metrics(d)
        if met is None:
            continue
        rows.append({"predictor": m["pred"], "source": m["src"], "layer": int(m["layer"]),
                     "index": int(m["index"]), "position": int(m["pos"]), "run": d.name, **met})
    cross = pd.DataFrame(rows)
    if cross.empty:
        return cross
    for src in models:
        signals = cross.loc[cross["source"] == src, ["layer", "index", "position"]].drop_duplicates()
        group = runs_root / MODELS[src][1]
        for L, i, pos in signals.itertuples(index=False):
            d = group / f"endo_N{L}.{i}_POS{pos}_gateoff_w1.0_300K"
            met = read_metrics(d)
            if met is None:
                print(f"[cross] WARNING self run missing: {d}")
                continue
            rows.append({"predictor": src, "source": src, "layer": L, "index": i,
                         "position": pos, "run": d.name, **met})
    return pd.DataFrame(rows)


def complete_columns(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Per source column, keep the signals every predictor in `models` has."""
    keys = ["layer", "index", "position"]
    kept = []
    for src, g in df.groupby("source"):
        n_pred = g.groupby(keys)["predictor"].nunique()
        full = n_pred[n_pred == len(models)].index
        dropped = len(n_pred) - len(full)
        if dropped:
            print(f"[cross] {src}: {dropped} signal(s) without all {len(models)} predictors left out")
        kept.append(g[g.set_index(keys).index.isin(full)])
    return pd.concat(kept, ignore_index=True) if kept else df.iloc[:0]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    long = df.melt(id_vars=["predictor", "source", "layer", "index", "position"],
                   value_vars=list(METRICS), var_name="metric", value_name="value")
    return (long.groupby(["predictor", "source", "metric"])["value"]
            .agg(["mean", "std", "count"]).reset_index())


def paired_tests(df: pd.DataFrame, alpha: float = 0.05, correction: str = "holm") -> pd.DataFrame:
    """Self vs each cross predictor, per source column and metric: paired over signals."""
    keys = ["layer", "index", "position"]
    rows = []
    for src, g in df.groupby("source"):
        for metric in METRICS:
            w = g.pivot_table(index=keys, columns="predictor", values=metric, aggfunc="first")
            if src not in w:
                continue
            for pred in (c for c in w.columns if c != src):
                pair = w[[src, pred]].dropna()
                d = (pair[pred] - pair[src]).to_numpy()
                if len(d) < 2 or np.std(d, ddof=1) < 1e-12:
                    stat, p = np.nan, np.nan
                else:
                    stat, p = (float(v) for v in ttest_rel(pair[pred], pair[src]))
                rows.append({"source": src, "predictor": pred, "metric": metric,
                             "n_pairs": len(pair), "mean_self": float(pair[src].mean()),
                             "mean_cross": float(pair[pred].mean()),
                             "mean_diff": float(d.mean()) if len(d) else np.nan,
                             "t_stat": stat, "p": p})
    res = pd.DataFrame(rows, columns=["source", "predictor", "metric", "n_pairs", "mean_self",
                                      "mean_cross", "mean_diff", "t_stat", "p"])
    res["p_adj"] = holm(res["p"]) if correction == "holm" else res["p"]
    res["significant"] = res["p_adj"] < alpha
    res.attrs.update(alpha=alpha, correction=correction,
                     n_tests=int(np.isfinite(res["p"]).sum()))
    return res


def _sig_set(tests):
    if tests is None or tests.empty:
        return set()
    return set(map(tuple, tests.loc[tests["significant"], ["predictor", "source", "metric"]]
                   .to_numpy()))


def write_markdown(summary: pd.DataFrame, models: list[str], out_dir: Path,
                   tests: pd.DataFrame | None = None) -> str:
    sig = _sig_set(tests)
    s = summary.set_index(["predictor", "source", "metric"])
    names = [MODELS[m][0] for m in models]
    blocks = []
    for metric, (_, label) in METRICS.items():
        n = int(summary.loc[summary["metric"] == metric, "count"].max()) if len(summary) else 0
        lines = [f"**Endogenous {label} correlation** (rows: predictor, columns: model whose "
                 f"neuron is predicted; mean ± std over {n} neurons per column)", "",
                 "| Predictor \\ Source | " + " | ".join(names) + " |",
                 "|---|" + "---|" * len(models)]
        for pred in models:
            cells = []
            for src in models:
                key = (pred, src, metric)
                if key not in s.index:
                    cells.append("–")
                    continue
                txt = f"{s.loc[key, 'mean']:.2f} ± {s.loc[key, 'std']:.2f}"
                if pred == src:
                    txt += " (self)"
                elif key in sig:
                    txt = f"**{txt}**"
                cells.append(txt)
            lines.append(f"| {MODELS[pred][0]} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))
    note = ("Off-diagonal: the predictor is trained (SFT) on the source model's own cached "
            "activations and texts, and evaluated on the same test rows as the source's self "
            "run, so it can only use the text.")
    if tests is not None and not tests.empty:
        corr_txt = (f"Holm-corrected over all {tests.attrs.get('n_tests')} comparisons"
                    if tests.attrs.get("correction") == "holm" else "uncorrected")
        note += (f" Bold: differs from the column's self run (two-sided paired t-test over "
                 f"neurons, {corr_txt}, p < {tests.attrs.get('alpha'):g}).")
    md = "\n\n".join(blocks) + "\n\n" + note + "\n"
    (out_dir / "cross_model_tables.md").write_text(md)
    return md


def plot(summary: pd.DataFrame, models: list[str], out_dir: Path, tests=None):
    sig = _sig_set(tests)
    s = summary.set_index(["predictor", "source", "metric"])
    cmap = LinearSegmentedColormap.from_list("seq_blue", RAMP)
    vals = summary["mean"].to_numpy()
    # one scale for both panels, starting a step below the lowest cell so the self vs
    # cross contrast uses the whole ramp
    vmin = max(0.0, float(np.floor(np.nanmin(vals) * 10) / 10) - 0.1) if len(vals) else 0.0
    names = [MODELS[m][0] for m in models]
    k = len(models)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.4 * len(METRICS) + 1.2, 3.8),
                             squeeze=False, sharey=True, constrained_layout=True)
    for c, (ax, (metric, (_, label))) in enumerate(zip(axes[0], METRICS.items())):
        M = np.full((k, k), np.nan)
        for i, pred in enumerate(models):
            for j, src in enumerate(models):
                if (pred, src, metric) in s.index:
                    M[i, j] = s.loc[(pred, src, metric), "mean"]
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=1.0)
        for i, pred in enumerate(models):
            for j, src in enumerate(models):
                key = (pred, src, metric)
                if key not in s.index:
                    ax.text(j, i, "–", ha="center", va="center", color=MUTED)
                    continue
                m_, sd = s.loc[key, "mean"], s.loc[key, "std"]
                star = "*" if key in sig else ""
                dark = (m_ - vmin) / (1.0 - vmin) > 0.55
                ax.text(j, i, f"{m_:.2f}{star}\n± {sd:.2f}", ha="center", va="center",
                        fontsize=9, color="#ffffff" if dark else INK)
            # the self cell of this column's row: outlined, since it is the reference
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.6))
        ax.set_xticks(range(k), names, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(k), names, fontsize=8)
        ax.set_xlabel("Source (Neuron Predicted)", fontsize=9)
        if c == 0:
            ax.set_ylabel("Predictor", fontsize=9)
        ax.set_title(f"Endogenous {label} Correlation", fontsize=10)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.04, pad=0.02, label="Mean Correlation")
    corr_txt = ("Holm-corrected" if tests is not None and tests.attrs.get("correction") == "holm"
                else "uncorrected")
    fig.supxlabel("Boxed: the source model's own (self) run.  *: cross predictor differs from "
                  f"self (paired t-test over neurons, {corr_txt}).", fontsize=7, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"cross_model_heatmap.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model keys {unknown}; known: {list(MODELS)}")
    df = collect(args.runs_root, models)
    if df.empty:
        raise SystemExit(f"no finished cross-model runs under {args.runs_root / CROSS_GROUP}")
    df = complete_columns(df, models)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "cross_model_runs.csv", index=False)
    summary = summarize(df)
    summary.to_csv(args.out_dir / "cross_model_summary.csv", index=False)
    tests = paired_tests(df, alpha=args.alpha, correction=args.correction)
    tests.to_csv(args.out_dir / "cross_model_tests.csv", index=False)
    print(write_markdown(summary, models, args.out_dir, tests))
    plot(summary, models, args.out_dir, tests)
    print(f"[cross] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
