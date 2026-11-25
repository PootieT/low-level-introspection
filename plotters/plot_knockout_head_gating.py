#!/usr/bin/env python3
"""
Are the heads the INTERVENTIONAL evaluation needs the ones that respond to anomalous
injections, while the heads only the ENDOGENOUS evaluation needs do not?

For every run with both mechanistic_analysis/score_head_knockout.py and mechanistic_analysis/trace_mover_head.py outputs, every
head above the signal layer is put in one column by which arms' top-k (single-knockout
drop) it is in --
    Top Endogenous Only | Top Both | Top Interventional Only | Other Heads (base rate)
-- and one row by the shape of its attention-to-t curve over the x10 injection grid
(src/introspector/head_taxonomy.gating_shape: rises relative to the centre point):
    2 rows: Not Gating / Gating
    3 rows: Not Gating / 1-Way (L-shaped, one sign) / 2-Way (U-shaped, both signs)
The unit is a (run, head) pair. Cells show the count and the fraction of the column, so
each column reads as "of the heads in this category, how many gate".

Every figure is drawn at the taxonomy's STATIC threshold (a rise below 0.05 = flat) and
again at its SWITCHING threshold (0.3), so a conclusion that only holds at one cut is
visible as such.

TESTS. Three comparisons of gating rate: Top Interventional Only vs Top Endogenous Only,
any interventional (Only + Both) vs Top Endogenous Only, and Top Endogenous Only vs
Other Heads. The (run, head) pairs are NOT independent, so a pooled 2x2 test (Fisher,
kept in the summary only as the number the paper first reported) overstates the
evidence: every run is fine-tuned from the same base model, so one base head (e.g.
L15H14) recurs as a top head in dozens of runs and whether it gates is largely a
property of that head; the 320 runs are 40 neurons x 8 positions; the top heads of a run
come from one ranking; and "Other" pools runs with 30-478 candidate heads whose base
rate differs by layer. Each comparison is therefore tested three ways that respect that
structure:
  mixed model    logistic regression gate ~ comparison, crossed random intercepts for
                 run and base head (statsmodels BinomialBayesMixedGLM, variational Bayes:
                 log-odds ratio +/- posterior sd, p from the normal approximation)
  head bootstrap rate difference with whole BASE heads resampled (every run a head
                 appears in goes in or out together); percentile CI and p
  within run     per run holding both categories, the difference of its gating rates;
                 Wilcoxon signed-rank over runs (each run is compared to itself)
statsmodels is imported lazily: without it the mixed model is skipped with a warning.

Usage
-----
python plotters/plot_knockout_head_gating.py --group mlp_joint_grid_layer_token --kl kl0.1 --k 2
python plotters/plot_knockout_head_gating.py --group mlp_joint_grid_layer_token \\
    --from-csv outputs/plots/mlp_joint_grid_layer_token/head_knockout/knockout_head_gating.csv
"""
from __future__ import annotations

import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, norm, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotters.run_catalog import NEURON_RE, POS_RE, KL_RE  # noqa: E402
from src.introspector.head_taxonomy import (  # noqa: E402
    gating_shape, GATING_SHAPES, DEFAULT_STATIC_TAU, DEFAULT_ATT_TAU)
from src.introspector.knockout_stats import top_k_arms, top_category, TOP_CATEGORIES  # noqa: E402

COLUMNS = list(TOP_CATEGORIES) + ["other"]
COL_LABEL = {"top endogenous only": "Top Endogenous\nOnly", "top both": "Top Both",
             "top interventional only": "Top Interventional\nOnly", "other": "Other Heads"}
ROW_LABEL = {"not gating": "Not Gating", "gating": "Gating", "1-way": "1-Way",
             "2-way": "2-Way"}
EO, BOTH, IO, OTHER = "top endogenous only", "top both", "top interventional only", "other"
# (key, label, category set A, category set B): is the gating rate of A different from B?
COMPARISONS = [
    ("interv_only_vs_endo_only", "Interventional-only vs endogenous-only", [IO], [EO]),
    ("any_interv_vs_endo_only", "Any interventional (only + both) vs endogenous-only",
     [IO, BOTH], [EO]),
    ("endo_only_vs_other", "Endogenous-only vs other heads", [EO], [OTHER]),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", required=True)
    p.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    p.add_argument("--kl", type=str, default=None)
    p.add_argument("--k", type=int, default=3,
                   help="Top-k per arm (matches mechanistic_analysis/trace_mover_head.py --top-heads).")
    p.add_argument("--regime", type=str, default="x10",
                   help="Injection regime whose attention curve is classified.")
    p.add_argument("--taus", type=str, default=f"{DEFAULT_STATIC_TAU},{DEFAULT_ATT_TAU}",
                   help="Rise thresholds to tabulate (static, switching).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: outputs/plots/<group>/head_knockout")
    p.add_argument("--from-csv", type=Path, default=None,
                   help="Redraw and re-test from a saved knockout_head_gating.csv instead of "
                        "the per-run folders (only a few are synced locally). The file is "
                        "read, never overwritten; its categories keep the k they were built "
                        "with (taken from the sibling summary JSON when present).")
    p.add_argument("--n-boot", type=int, default=10000,
                   help="Head-bootstrap resamples.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load(args, taus) -> pd.DataFrame:
    root = args.runs_root / args.group
    rows, missing = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        n, pz = NEURON_RE.search(d.name), POS_RE.search(d.name)
        if not (n and pz):
            continue
        if args.kl:
            m = KL_RE.search(d.name)
            if not m or f"kl{m['kl']}" != args.kl:
                continue
        ko_p = d / "head_knockout" / "head_knockout_individual.csv"
        dose_p = d / "mover_trace" / "mover_head_dose_all.csv"
        if not (ko_p.exists() and dose_p.exists()):
            missing.append(d.name)
            continue
        ko = pd.read_csv(ko_p)
        arms_of = top_k_arms(ko, args.k)
        drop = {(int(r.layer), int(r.head), r.arm): (-float(r.delta), int(r.rank))
                for r in ko.itertuples()}
        dose = pd.read_csv(dose_p)
        dose = dose[dose["regime"] == args.regime]
        if dose.empty:
            missing.append(f"{d.name} (no {args.regime} regime)")
            continue
        for (L, h), g in dose.groupby(["layer", "head"]):
            key = (int(L), int(h))
            row = {"run": d.name, "signal_layer": int(n["layer"]),
                   "signal_index": int(n["index"]), "position": int(pz["pos"]),
                   "layer": key[0], "head": key[1],
                   "category": top_category(arms_of.get(key))}
            for arm in ("endogenous", "x1"):
                if (*key, arm) in drop:
                    row[f"drop_{arm}"], row[f"rank_{arm}"] = drop[(*key, arm)]
            for tau in taus:
                gs = gating_shape(g["v"], g["attn_to_t"], tau=tau)
                row.update({"attn_at_centre": gs["attn_at_centre"],
                            "rise_lo": gs["rise_lo"], "rise_hi": gs["rise_hi"],
                            f"shape_tau{tau:g}": gs["shape"]})
            rows.append(row)
    print(f"[gating] {len({r['run'] for r in rows})} run(s) with knockout + mover outputs")
    if missing:
        print(f"[gating] skipped {len(missing)}, e.g. {missing[:3]}")
    if not rows:
        raise SystemExit("Nothing to aggregate.")
    return pd.DataFrame(rows)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (c - hw, c + hw)


def tables(df: pd.DataFrame, tau: float):
    shape = df[f"shape_tau{tau:g}"]
    t3 = pd.crosstab(shape, df["category"]).reindex(index=list(GATING_SHAPES),
                                                    columns=COLUMNS, fill_value=0)
    gating = np.where(shape == "not gating", "not gating", "gating")
    t2 = pd.crosstab(pd.Series(gating, index=df.index), df["category"]).reindex(
        index=["not gating", "gating"], columns=COLUMNS, fill_value=0)
    return t2, t3


def gate_flags(df: pd.DataFrame, tau: float) -> pd.Series:
    return (df[f"shape_tau{tau:g}"] != "not gating").astype(int)


def head_ids(df: pd.DataFrame) -> pd.Series:
    """The BASE-model head: the same (layer, head) in every run is the same head."""
    return "L" + df["layer"].astype(str) + "H" + df["head"].astype(str)


def pooled_fisher(gate: pd.Series, in_a: pd.Series, in_b: pd.Series) -> dict:
    """The pooled 2x2 test the paper first reported -- treats every (run, head) as
    independent, so kept for reference only."""
    a, b = gate[in_a], gate[in_b]
    odds, p = fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
    return {"odds_ratio": float(odds), "p": float(p)}


def within_run_wilcoxon(df: pd.DataFrame, gate: pd.Series, in_a: pd.Series,
                        in_b: pd.Series) -> dict:
    """Per run holding both categories: gating rate of A minus that of B. Wilcoxon
    signed-rank over runs (ties dropped), so each run is compared only to itself."""
    d = pd.DataFrame({"run": df["run"], "gate": gate, "a": in_a, "b": in_b})
    ra = d[d["a"]].groupby("run")["gate"].mean()
    rb = d[d["b"]].groupby("run")["gate"].mean()
    diff = (ra - rb).dropna().to_numpy()
    nz = diff[diff != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 6 else np.nan
    return {"n_runs": int(len(diff)), "n_up": int((diff > 0).sum()),
            "n_down": int((diff < 0).sum()), "n_tied": int((diff == 0).sum()),
            "mean_diff": float(diff.mean()) if len(diff) else np.nan, "p": p}


def head_bootstrap(df: pd.DataFrame, gate: pd.Series, in_a: pd.Series, in_b: pd.Series,
                   n_boot: int = 10000, seed: int = 0) -> dict:
    """rate(A) - rate(B) with whole BASE heads resampled with replacement: a head that
    is top in 50 runs enters or leaves the sample as one unit. Percentile 95% CI;
    two-sided p = 2 x the smaller tail beyond 0 (floored at 1 / n_boot)."""
    d = pd.DataFrame({"hid": head_ids(df), "gate": gate, "a": in_a.astype(int),
                      "b": in_b.astype(int)})
    d = d[(d["a"] + d["b"]) > 0]
    agg = pd.DataFrame({"gA": (d["gate"] * d["a"]).groupby(d["hid"]).sum(),
                        "nA": d["a"].groupby(d["hid"]).sum(),
                        "gB": (d["gate"] * d["b"]).groupby(d["hid"]).sum(),
                        "nB": d["b"].groupby(d["hid"]).sum()}).to_numpy(float)
    H = len(agg)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(H, np.full(H, 1.0 / H), size=n_boot).astype(float)   # [B, H]
    s = w @ agg                                                             # [B, 4]
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = s[:, 0] / s[:, 1] - s[:, 2] / s[:, 3]
    diff = diff[np.isfinite(diff)]
    obs = agg[:, 0].sum() / agg[:, 1].sum() - agg[:, 2].sum() / agg[:, 3].sum()
    tail = min((diff <= 0).mean(), (diff >= 0).mean())
    p = max(2 * tail, 1.0 / len(diff))
    return {"n_heads": int(H), "diff": float(obs),
            "ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
            "p": float(p), "p_at_floor": bool(2 * tail < 1.0 / len(diff)),
            "n_boot": int(len(diff))}


def mixed_model(df: pd.DataFrame, gate: pd.Series, in_a: pd.Series, in_b: pd.Series,
                starts: tuple = (0, 1, 2)) -> dict | None:
    """gate ~ in_A over the rows of A and B, with crossed random intercepts for run and
    base head (variational Bayes). None when statsmodels is not installed.

    statsmodels draws the optimizer's starting point from fresh OS entropy unless given a
    seed, and one unseeded fit of endogenous-only vs other landed at 1.08 +/- 0.40 where
    every seeded start gives 1.68 +/- 0.19. So the model is fitted from several SEEDED
    starts and the best objective kept; `converged` and the spread of the estimate over
    the starts are recorded. (The Laplace fit, fit_map, is not used: on the small
    top-vs-top comparisons its variance parameters run off to a degenerate solution.)"""
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError:
        print("[gating] statsmodels not installed: mixed model skipped")
        return None
    keep = in_a | in_b
    d = pd.DataFrame({"gate": gate[keep], "x": in_a[keep].astype(float),
                      "run": df.loc[keep, "run"], "hid": head_ids(df[keep])})
    model = BinomialBayesMixedGLM.from_formula(
        "gate ~ x", {"run": "0 + C(run)", "head": "0 + C(hid)"}, d)
    fits = []
    for seed in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit_vb(rng=seed)
        i = list(res.model.exog_names).index("x")
        fits.append((float(res.optim_retvals.fun), float(res.fe_mean[i]), float(res.fe_sd[i]),
                     bool(res.optim_retvals.success)))
    obj, est, sd, ok = min(fits)
    return {"log_odds_ratio": est, "sd": sd, "z": est / sd,
            "p": float(2 * norm.sf(abs(est / sd))), "converged": ok,
            "n_starts": len(fits),
            "start_spread": float(max(f[1] for f in fits) - min(f[1] for f in fits)),
            "n_rows": int(keep.sum()), "n_runs": int(d["run"].nunique()),
            "n_heads": int(d["hid"].nunique())}


def stats(df: pd.DataFrame, t2: pd.DataFrame, tau: float, n_boot: int = 10000,
          seed: int = 0, with_mixed: bool = True) -> dict:
    """Per-column gating rates (Wilson CI) and, per comparison, the pooled Fisher test
    (reference only) plus the three tests that respect the run / base-head structure."""
    out = {}
    for c in COLUMNS:
        k, n = int(t2.loc["gating", c]), int(t2[c].sum())
        lo, hi = wilson(k, n)
        out[c] = {"n": n, "gating": k, "rate": k / n if n else np.nan, "ci95": [lo, hi]}
    gate = gate_flags(df, tau)
    for key, label, A, B in COMPARISONS:
        in_a, in_b = df["category"].isin(A), df["category"].isin(B)
        if not (in_a.any() and in_b.any()):
            continue
        out[key] = {"label": label,
                    "rate_a": float(gate[in_a].mean()), "rate_b": float(gate[in_b].mean()),
                    "n_a": int(in_a.sum()), "n_b": int(in_b.sum()),
                    "pooled_fisher": pooled_fisher(gate, in_a, in_b),
                    "within_run": within_run_wilcoxon(df, gate, in_a, in_b),
                    "head_bootstrap": head_bootstrap(df, gate, in_a, in_b, n_boot, seed),
                    "mixed_model": mixed_model(df, gate, in_a, in_b) if with_mixed else None}
    return out


def fmt_p(p: float, floor: bool = False) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    if floor:
        return f"< {p:.0e}".replace("e-0", "e-")
    if p < 1e-3:
        return f"{p:.0e}".replace("e-0", "e-")
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def tests_markdown(s: dict, tau: float, k, regime: str, n_runs: int) -> str:
    """One rebuttal-ready markdown table of the comparisons at this threshold."""
    lines = [f"**Gating rate by knockout category** (attention rise ≥ {tau:g} under "
             f"{regime} injection, top-{k} heads per evaluation, {n_runs} models)", "",
             "| Comparison | Gating rate | Pooled Fisher (as in paper) | Within-model Wilcoxon "
             "| Base-head bootstrap: Δ rate [95% CI] | Mixed model: log-OR ± sd |",
             "|---|---|---|---|---|---|"]
    for key, label, _, _ in COMPARISONS:
        c = s.get(key)
        if c is None:
            continue
        wr, hb, mm = c["within_run"], c["head_bootstrap"], c["mixed_model"]
        mm_txt = ("n/a" if mm is None else
                  f"{mm['log_odds_ratio']:.2f} ± {mm['sd']:.2f}, p = {fmt_p(mm['p'])}"
                  + ("" if mm.get("converged", True) else "†"))
        lines.append(
            f"| {label} | {c['rate_a']:.2f} vs {c['rate_b']:.2f} "
            f"| p = {fmt_p(c['pooled_fisher']['p'])} "
            f"| p = {fmt_p(wr['p'])} ({wr['n_up']} up / {wr['n_down']} down, "
            f"{wr['n_runs']} models) "
            f"| {hb['diff']:+.2f} [{hb['ci95'][0]:+.2f}, {hb['ci95'][1]:+.2f}], "
            f"p {fmt_p(hb['p'], hb['p_at_floor']) if hb['p_at_floor'] else '= ' + fmt_p(hb['p'])} "
            f"| {mm_txt} |")
    mm_any = next((s[k_]["mixed_model"] for k_, *_ in COMPARISONS
                   if s.get(k_) and s[k_]["mixed_model"]), None)
    note = ("Pooled Fisher treats every (model, head) pair as independent. All models are "
            "fine-tuned from the same base model, so the same base head recurs across models "
            "and whether it gates is largely a property of that head. Within-model Wilcoxon: "
            "per model, the difference between the two categories' gating rates, "
            "signed-rank test over models. Base-head bootstrap: base-model heads resampled "
            "as whole units, so a head that appears in many models counts once per draw. "
            "Mixed model: logistic regression with crossed random intercepts for model and "
            "base head (variational Bayes; p from the normal approximation).")
    if mm_any:
        note += (f" The mixed model for the first comparison uses {mm_any['n_rows']} rows from "
                 f"{mm_any['n_runs']} models and {mm_any['n_heads']} distinct base heads.")
    unconv = [s[k_]["mixed_model"] for k_, *_ in COMPARISONS
              if s.get(k_) and s[k_]["mixed_model"] and not s[k_]["mixed_model"]["converged"]]
    if unconv:
        spread = max(m["start_spread"] for m in unconv)
        note += (f" † The variational optimizer stopped before its convergence criterion; the "
                 f"estimate agrees to within {spread:.2g} across {unconv[0]['n_starts']} "
                 f"random starts.")
    return "\n".join(lines) + "\n\n" + note + "\n"


def plot_table(tab: pd.DataFrame, path: Path, tau: float):
    col = tab.sum(axis=0).astype(float)
    frac = tab.astype(float) / col.where(col > 0)
    fig, ax = plt.subplots(figsize=(1.9 * tab.shape[1] + 1.6, 1.0 * tab.shape[0] + 1.9))
    im = ax.imshow(frac.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            f = frac.iat[i, j]
            ax.text(j, i, f"{tab.iat[i, j]}\n({100 * f:.0f}%)" if np.isfinite(f) else "0",
                    ha="center", va="center", fontsize=14,
                    color="white" if np.isfinite(f) and f > 0.55 else "black")
    ax.set_xticks(range(tab.shape[1]), [COL_LABEL[c] for c in tab.columns], fontsize=13)
    ax.set_yticks(range(tab.shape[0]), [ROW_LABEL[r] for r in tab.index], fontsize=14)
    # the base-rate column is a reference, not one of the compared categories
    ax.axvline(len(TOP_CATEGORIES) - 0.5, color="k", lw=2)
    ax.set_xlabel("Knockout Importance", fontsize=15)
    ax.set_ylabel(f"Attention Response\n(Rise ≥ {tau:g})", fontsize=15)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Fraction of Column", fontsize=13)
    cb.ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    taus = [float(x) for x in args.taus.split(",") if x.strip()]
    out = args.out_dir or ROOT / "outputs" / "plots" / args.group / "head_knockout"
    out.mkdir(parents=True, exist_ok=True)
    k, regime = args.k, args.regime
    if args.from_csv is not None:
        df = pd.read_csv(args.from_csv)
        prev = args.from_csv.parent / "knockout_head_gating_summary.json"
        if prev.exists():
            old = json.loads(prev.read_text())
            k, regime = old.get("k", k), old.get("regime", regime)
        print(f"[gating] {args.from_csv}: {df['run'].nunique()} run(s), {len(df)} (run, head) "
              f"rows; categories built with top-{k}, regime {regime}")
    else:
        df = load(args, taus)
        df.to_csv(out / "knockout_head_gating.csv", index=False)
    summary = {"k": k, "regime": regime, "n_runs": int(df["run"].nunique()),
               "n_boot": args.n_boot}
    md = []
    for tau in taus:
        t2, t3 = tables(df, tau)
        tag = f"_tau{tau:g}"
        plot_table(t2, out / f"knockout_gating_2way{tag}.png", tau)
        plot_table(t3, out / f"knockout_gating_3way{tag}.png", tau)
        s = stats(df, t2, tau, n_boot=args.n_boot, seed=args.seed)
        summary[f"tau{tau:g}"] = s
        print(f"\n[gating] rise threshold {tau:g} ({regime}):")
        print(t3.rename(index=ROW_LABEL, columns=lambda c: COL_LABEL[c].replace("\n", " "))
              .to_string())
        for c in COLUMNS:
            r = s[c]
            print(f"[gating]   {COL_LABEL[c].replace(chr(10), ' '):26s} gating "
                  f"{r['gating']}/{r['n']} = {r['rate']:.2f} "
                  f"(95% CI {r['ci95'][0]:.2f}-{r['ci95'][1]:.2f})")
        md.append(tests_markdown(s, tau, k, regime, summary["n_runs"]))
    (out / "knockout_head_gating_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    text = "\n".join(md)
    (out / "knockout_head_gating_tests.md").write_text(text)
    print("\n" + text)
    print(f"[gating] wrote {out}")


if __name__ == "__main__":
    main()
