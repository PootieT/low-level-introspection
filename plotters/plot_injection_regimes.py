#!/usr/bin/env python3
"""
SFT vs joint loss across injection regimes (rebuttal: is the interventional advantage
specific to the training intervention?).

Reads every <run>/injection_regimes/injection_regimes.csv written by
evals/score_injection_regimes.py for the two arms and draws one panel per
(layer, token) cell: correlation per regime, mean +/- std over the cell's neurons, one
line per method. The x axis runs from the smallest perturbation to the largest:

    local@0.25s, local@1s   each example's live value + delta, scored within example
                            as corr(pred - pred_clean, delta)
    x1 .. x100              gaussian replacement at N x the natural std, corr(pred, v)

Regimes whose draws partly leave the joint runs' training range (uniform over 10x the
natural range) are shaded, using the measured fraction outside it (median over the
joint runs of the panel). The endogenous correlation is printed in each panel title.

Only neurons present in BOTH arms are drawn (the SFT sweep has more per cell), so the
two lines in a panel always average over the same neurons.

SIGNIFICANCE (the bold joint cells of the markdown tables). Per cell and regime, joint vs
SFT is a two-sided PAIRED t-test over the cell's neurons -- the same unit the mean +/- std
is taken over; the pairs share the neuron, the test rows and the injected values -- on
the correlations as shown. NOT on Fisher z: that stabilizes the SAMPLING variance of one
r, but the variance tested here is BETWEEN neurons, and z stretches the joint runs'
near-1 values apart (0.97 vs 0.999 is a z gap of 1.7, more than 0.2 vs 0.6), which buries
a large joint-SFT gap in noise. p-values are Holm-corrected across every comparison in
the file (--correction none for raw). With 5 neurons a
signed-rank test cannot reach p < 0.05 at all (its smallest two-sided p is 0.0625), which
is why the test is parametric. A pair whose difference is identical for every neuron
(e.g. both arms scored 0) is untestable and never bolded.

Outputs (outputs/plots/injection_regimes/ by default):
  injection_regimes_runs.csv     every run x arm row
  injection_regimes_summary.csv  mean / std / n per method x cell x arm
  injection_regimes_tests.csv    joint vs SFT per cell x arm: n pairs, mean difference,
                                 t, p, Holm-adjusted p, significant
  injection_regimes_pooled_summary.csv / _pooled_tests.csv
                                 the same, POOLED over every locale per arm: each regime
                                 tested once over all of its paired runs (20 = 10 neurons
                                 x 2 token positions for the rebuttal grid), Holm over the
                                 regimes. This is the headline comparison: 5 neurons per
                                 locale cannot carry a strict per-locale test when SFT's
                                 between-neuron spread is 0.3-0.5.
  injection_regimes_tables.md    the same numbers as markdown tables (the rebuttal allows
                                 no images): the pooled table first, then one per locale;
                                 joint bold where it differs significantly from SFT, each
                                 table under its own tests
  injection_regimes.png / .pdf

Usage
-----
python plotters/plot_injection_regimes.py
python plotters/plot_injection_regimes.py \\
    --from-runs-csv outputs/plots/injection_regimes/injection_regimes_runs.csv
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEQ_LEN = 32
# validated categorical slots 1 and 2 (dataviz validator, light surface): joint keeps
# the blue the head-set-ablation figure uses
METHODS = {"joint": ("Joint Loss (KL=0.1)", "#2a78d6"), "sft": ("SFT", "#eb6834")}
INK, MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
OUTSIDE_SHADE_MIN = 0.01      # shade a regime once >= 1% of its draws leave training range
# grouping for the summaries / joint-vs-SFT tests: per locale, or every locale at once
PER_LOCALE = ("layer", "t", "arm")
POOLED = ("arm",)
PAIR_KEY = ["layer", "index", "t"]      # one trained run per method: the unit that is paired

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    p.add_argument("--joint", default="mlp_joint_grid_layer_token:joint_*_kl0.1_*",
                   help="'<group>:<run glob>' of the joint-loss arm.")
    p.add_argument("--sft", default="mlp_endo_sweep_layer_token:endo_*_gateoff_w1.0_300K",
                   help="'<group>:<run glob>' of the SFT arm.")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "plots" / "injection_regimes")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level for bolding a joint cell.")
    p.add_argument("--correction", choices=("holm", "none"), default="holm",
                   help="Multiple-comparison correction, applied separately to the pooled "
                        "tests (one per regime) and the per-locale tests (locale x regime).")
    p.add_argument("--from-runs-csv", type=Path, default=None,
                   help="Redraw from a saved injection_regimes_runs.csv (e.g. the one SCC "
                        "wrote) instead of the per-run folders, which are only partly "
                        "synced locally. That file is read, never overwritten.")
    return p.parse_args()


def load_arm(runs_root: Path, spec: str, method: str) -> pd.DataFrame:
    group, glob = spec.split(":", 1)
    files = sorted((runs_root / group).glob(f"{glob}/injection_regimes/injection_regimes.csv"))
    if not files:
        print(f"[plot] WARNING no injection_regimes.csv for {method} under {runs_root / group}/{glob}")
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["method"] = method
    print(f"[plot] {method}: {len(files)} run(s)")
    return df


def arm_order(arms) -> list[str]:
    """local arms by delta, then global arms by scale (the endogenous arm is not on x)."""
    local = sorted((a for a in arms if a.startswith("local@")),
                   key=lambda a: float(a[len("local@"):].rstrip("s")))
    glob = sorted((a for a in arms if a.startswith("x")), key=lambda a: float(a[1:]))
    return local + glob


def arm_label(arm: str) -> str:
    if arm.startswith("local@"):
        return f"Local\n±{arm[len('local@'):].rstrip('s')}σ"
    return f"{arm[1:]}×"


def paired_neurons(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only (layer, index, position) present in every method."""
    keys = ["layer", "index", "position"]
    per = [set(map(tuple, g[keys].drop_duplicates().to_numpy())) for _, g in df.groupby("method")]
    common = set.intersection(*per) if per else set()
    dropped = df[~df[keys].apply(tuple, axis=1).isin(common)]["run"].unique()
    if len(dropped):
        print(f"[plot] {len(dropped)} run(s) without a partner in the other arm left out")
    return df[df[keys].apply(tuple, axis=1).isin(common)].copy()


def summarize(df: pd.DataFrame, by: tuple = PER_LOCALE) -> pd.DataFrame:
    """mean / std / count of the correlation per method x `by` (default: per locale;
    POOLED: every locale's runs together)."""
    keys = ["method"] + list(by)
    g = df.groupby(keys)
    s = g["corr"].agg(["mean", "std", "count"]).reset_index()
    if "frac_outside_train_support" in df:
        s = s.merge(g["frac_outside_train_support"].median().rename("frac_outside_median")
                    .reset_index(), on=keys)
    return s


def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values. NaNs (untestable pairs) pass through and do not
    count towards the number of comparisons."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = len(ok)
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    out[order] = np.maximum.accumulate(adj)
    return out


def paired_tests(df: pd.DataFrame, alpha: float = 0.05, correction: str = "holm",
                 by: tuple = PER_LOCALE) -> pd.DataFrame:
    """Joint vs SFT within each `by` group: two-sided paired t-test on the raw
    correlations, pairs = the same (layer, neuron, token) run in both methods, then
    corrected across every test returned. by=PER_LOCALE tests each locale over its
    neurons; by=POOLED tests each regime over every locale's runs at once."""
    by = list(by)
    rows = []
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        w = g.pivot_table(index=PAIR_KEY, columns="method", values="corr", aggfunc="first")
        if not {"joint", "sft"} <= set(w.columns):
            continue
        w = w[["joint", "sft"]].dropna()
        d = (w["joint"] - w["sft"]).to_numpy()
        if len(d) < 2 or np.std(d, ddof=1) < 1e-12:
            stat, p = np.nan, np.nan       # one pair, or a constant shift: untestable
        else:
            stat, p = (float(v) for v in ttest_rel(w["joint"], w["sft"]))
        rows.append({**dict(zip(by, key)), "n_pairs": len(w),
                     "mean_joint": float(w["joint"].mean()), "mean_sft": float(w["sft"].mean()),
                     "mean_diff": float(d.mean()), "t_stat": stat, "p": p})
    res = pd.DataFrame(rows, columns=by + ["n_pairs", "mean_joint", "mean_sft",
                                           "mean_diff", "t_stat", "p"])
    res["p_adj"] = holm(res["p"]) if correction == "holm" else res["p"]
    res["significant"] = res["p_adj"] < alpha          # NaN compares False
    res.attrs.update(alpha=alpha, correction=correction,
                     n_tests=int(np.isfinite(res["p"]).sum()))
    return res


def table_order(arms) -> list[str]:
    """Rows of the markdown tables: endogenous, gaussian by scale, local by delta (largest
    first, so the table reads from the paper's eval outwards)."""
    glob = sorted((a for a in arms if a.startswith("x")), key=lambda a: float(a[1:]))
    local = sorted((a for a in arms if a.startswith("local@")),
                   key=lambda a: -float(a[len("local@"):].rstrip("s")))
    return (["endogenous"] if "endogenous" in set(arms) else []) + glob + local


def table_row_label(arm: str, frac_out: float) -> str:
    if arm == "endogenous":
        return "Endogenous (no injection)"
    if arm.startswith("local@"):
        return f"Local ±{arm[len('local@'):].rstrip('s')}σ"
    label = f"Gaussian {arm[1:]}×"
    if np.isfinite(frac_out) and frac_out >= OUTSIDE_SHADE_MIN:
        label += f" ({100 * frac_out:.0f}% outside training range)"
    return label


def _table_block(title: str, cell: pd.DataFrame, bold_arms: set, methods: list) -> str:
    """One markdown table: rows = regimes, one 'mean ± std' column per method; the joint
    cell of every arm in `bold_arms` is bold."""
    def fmt(r, bold=False):
        if r is None:
            return "–"
        s = f"{r['mean']:.2f} ± {r['std']:.2f}"
        return f"**{s}**" if bold else s

    by = {(r["method"], r["arm"]): r for _, r in cell.iterrows()}
    joint = cell[cell["method"] == "joint"].set_index("arm")
    lines = [title, "",
             "| Regime | " + " | ".join(METHODS[m][0] for m in methods) + " |",
             "|---|" + "---|" * len(methods)]
    for a in table_order(cell["arm"].unique()):
        f = (joint["frac_outside_median"].get(a, np.nan)
             if "frac_outside_median" in joint else np.nan)
        lines.append(f"| {table_row_label(a, f)} | "
                     + " | ".join(fmt(by.get((m, a)), bold=(m == "joint" and a in bold_arms))
                                  for m in methods) + " |")
    return "\n".join(lines)


def _test_note(tests: pd.DataFrame, what: str) -> str:
    corr_txt = (f"Holm-corrected over its {tests.attrs.get('n_tests')} comparisons"
                if tests.attrs.get("correction") == "holm" else "uncorrected")
    return (f"{what}: bold = Joint differs from SFT (two-sided paired t-test, {corr_txt}, "
            f"p < {tests.attrs.get('alpha'):g}).")


def write_markdown(summary: pd.DataFrame, out_dir: Path, tests: pd.DataFrame | None = None,
                   pooled: pd.DataFrame | None = None,
                   pooled_tests: pd.DataFrame | None = None, pooled_title: str = "") -> str:
    """Markdown tables: the POOLED table first (every locale's runs together, when given),
    then one table per (layer, token) locale. Each table's joint cells are bold where its
    own tests mark them significantly different from SFT; the two sets of tests are
    corrected separately."""
    methods = [m for m in METHODS if m in set(summary["method"])]
    sig = lambda tst, cols: (set() if tst is None or tst.empty else
                             set(map(tuple, tst.loc[tst["significant"], cols].to_numpy())))
    blocks = []
    if pooled is not None:
        bold = {a for (a,) in sig(pooled_tests, ["arm"])}
        blocks.append(_table_block(pooled_title or "**All locales pooled** (mean ± std)",
                                   pooled, bold, methods))
    loc_sig = sig(tests, ["layer", "t", "arm"])
    for (L, t), cell in summary.groupby(["layer", "t"]):
        n = int(cell["count"].max())
        bold = {a for (l, tt, a) in loc_sig if (l, tt) == (L, t)}
        blocks.append(_table_block(f"**Layer {L}, token t={t}** ({n} neurons, mean ± std)",
                                   cell, bold, methods))
    note = ("Gaussian N×: the neuron is replaced by v ~ N(μ, (Nσ)²) of its natural "
            "activations; corr(prediction, v). Local ±Dσ: each example's own value plus "
            "δ ~ N(0, (Dσ)²); corr(change in prediction, δ) within example. Joint-loss "
            "training replaced the neuron with values uniform over 10× its natural range.")
    if pooled_tests is not None and not pooled_tests.empty:
        note += " " + _test_note(pooled_tests, "Pooled table, pairs = the same "
                                               "(layer, neuron, token) run in both methods")
    if tests is not None and not tests.empty:
        note += " " + _test_note(tests, "Per-locale tables, pairs = the locale's neurons")
    md = "\n\n".join(blocks) + "\n\n" + note + "\n"
    (out_dir / "injection_regimes_tables.md").write_text(md)
    return md


def plot(summary: pd.DataFrame, out_dir: Path):
    layers, ts = sorted(summary["layer"].unique()), sorted(summary["t"].unique())
    arms = arm_order(summary["arm"].unique())
    x = np.arange(len(arms))
    fig, axes = plt.subplots(len(layers), len(ts), figsize=(3.6 * len(ts), 2.9 * len(layers)),
                             sharey=True, squeeze=False)
    for i, L in enumerate(layers):
        for j, t in enumerate(ts):
            ax = axes[i, j]
            cell = summary[(summary["layer"] == L) & (summary["t"] == t)]
            # shade the regimes that partly leave the joint runs' training range
            jt = cell[(cell["method"] == "joint")].set_index("arm")
            if "frac_outside_median" in jt:
                for k, a in enumerate(arms):
                    f = jt["frac_outside_median"].get(a, np.nan)
                    if np.isfinite(f) and f >= OUTSIDE_SHADE_MIN:
                        ax.axvspan(k - 0.5, k + 0.5, color=SHADE, lw=0, zorder=0)
                        ax.text(k, -0.12, f"{100 * f:.0f}% out", ha="center", va="bottom",
                                fontsize=7, color=MUTED)
            # local | global divider
            n_local = sum(a.startswith("local@") for a in arms)
            if 0 < n_local < len(arms):
                ax.axvline(n_local - 0.5, color=MUTED, lw=0.8, ls=":")
            endo = []
            for m, (label, color) in METHODS.items():
                d = cell[cell["method"] == m].set_index("arm")
                if d.empty:
                    continue
                y = d["mean"].reindex(arms).to_numpy()
                e = d["std"].reindex(arms).to_numpy()
                # local and global arms use different metrics: no line across the divider
                for seg, lab in ((slice(0, n_local), None), (slice(n_local, None), label)):
                    ax.errorbar(x[seg], y[seg], yerr=e[seg], color=color, lw=2, marker="o",
                                ms=5, capsize=2.5, elinewidth=1, label=lab, zorder=3)
                if "endogenous" in d.index:
                    endo.append(f"{label.split(' ')[0]} {d.loc['endogenous', 'mean']:.2f}")
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_xticks(x, [arm_label(a) for a in arms])
            ax.set_ylim(-0.15, 1.05)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            title = f"Layer {L}, Token t={t}"
            if endo:
                title += "\nEndogenous: " + ", ".join(endo)
            ax.set_title(title, fontsize=9)
            if j == 0:
                ax.set_ylabel("Correlation")
    for ax in axes[-1]:
        ax.set_xlabel("Injection Regime")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.02,
             "Local: corr(Δprediction, δ) within example around the live value.  "
             "N×: corr(prediction, injected value), gaussian at N× the natural std.  "
             "Shaded: draws partly outside the joint runs' training range.",
             ha="center", va="top", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"injection_regimes.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def pooled_title(df: pd.DataFrame) -> str:
    """'**All 4 locales pooled** (20 runs per method = 10 neurons x 2 token positions ...)'"""
    runs = df[df["method"] == "joint"][PAIR_KEY].drop_duplicates()
    n_loc = len(runs[["layer", "t"]].drop_duplicates())
    n_neu = len(runs[["layer", "index"]].drop_duplicates())
    n_pos = runs["t"].nunique()
    return (f"**All {n_loc} locales pooled** ({len(runs)} runs per method = {n_neu} neurons "
            f"× {n_pos} token positions, mean ± std)")


def main():
    args = parse_args()
    if args.from_runs_csv is not None:
        df = pd.read_csv(args.from_runs_csv)
        print(f"[plot] {args.from_runs_csv}: {df['run'].nunique()} run(s) "
              f"({df.groupby('method')['run'].nunique().to_dict()})")
    else:
        df = pd.concat([load_arm(args.runs_root, args.joint, "joint"),
                        load_arm(args.runs_root, args.sft, "sft")], ignore_index=True)
    if df.empty:
        raise SystemExit("no injection_regimes results found")
    df["t"] = np.where(df["position"] < 0, df["position"] + SEQ_LEN, df["position"])
    df = paired_neurons(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = args.out_dir / "injection_regimes_runs.csv"
    if args.from_runs_csv is None or args.from_runs_csv.resolve() != runs_csv.resolve():
        df.to_csv(runs_csv, index=False)
    summary = summarize(df)
    summary.to_csv(args.out_dir / "injection_regimes_summary.csv", index=False)
    tests = paired_tests(df, alpha=args.alpha, correction=args.correction)
    tests.to_csv(args.out_dir / "injection_regimes_tests.csv", index=False)
    pooled = summarize(df, by=POOLED)
    pooled.to_csv(args.out_dir / "injection_regimes_pooled_summary.csv", index=False)
    pooled_tests = paired_tests(df, alpha=args.alpha, correction=args.correction, by=POOLED)
    pooled_tests.to_csv(args.out_dir / "injection_regimes_pooled_tests.csv", index=False)
    print(write_markdown(summary, args.out_dir, tests, pooled=pooled,
                         pooled_tests=pooled_tests, pooled_title=pooled_title(df)))
    plot(summary, args.out_dir)
    print(f"[plot] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
