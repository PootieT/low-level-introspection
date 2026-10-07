#!/usr/bin/env python3
"""
Are the heads the INTERVENTIONAL evaluation needs the ones that respond to anomalous
injections, while the heads only the ENDOGENOUS evaluation needs do not?

For every run with both score_head_knockout.py and trace_mover_head.py outputs, every
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
visible as such. Fisher exact tests compare the gating rate of the interventional
columns against Top Endogenous Only, and of Top Endogenous Only against Other Heads.

Usage
-----
python analysis/plot_knockout_head_gating.py --group mlp_joint_grid_layer_token --kl kl0.1
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
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.run_catalog import NEURON_RE, POS_RE, KL_RE  # noqa: E402
from src.introspector.head_taxonomy import (  # noqa: E402
    gating_shape, GATING_SHAPES, DEFAULT_STATIC_TAU, DEFAULT_ATT_TAU)
from src.introspector.knockout_stats import top_k_arms, top_category, TOP_CATEGORIES  # noqa: E402

COLUMNS = list(TOP_CATEGORIES) + ["other"]
COL_LABEL = {"top endogenous only": "Top Endogenous\nOnly", "top both": "Top Both",
             "top interventional only": "Top Interventional\nOnly", "other": "Other Heads"}
ROW_LABEL = {"not gating": "Not Gating", "gating": "Gating", "1-way": "1-Way",
             "2-way": "2-Way"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", required=True)
    p.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    p.add_argument("--kl", type=str, default=None)
    p.add_argument("--k", type=int, default=3,
                   help="Top-k per arm (matches trace_mover_head.py --top-heads).")
    p.add_argument("--regime", type=str, default="x10",
                   help="Injection regime whose attention curve is classified.")
    p.add_argument("--taus", type=str, default=f"{DEFAULT_STATIC_TAU},{DEFAULT_ATT_TAU}",
                   help="Rise thresholds to tabulate (static, switching).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: outputs/plots/<group>/head_knockout")
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


def stats(t2: pd.DataFrame) -> dict:
    out = {}
    for c in COLUMNS:
        k, n = int(t2.loc["gating", c]), int(t2[c].sum())
        lo, hi = wilson(k, n)
        out[c] = {"n": n, "gating": k, "rate": k / n if n else np.nan, "ci95": [lo, hi]}

    def fisher(a_cols, b_cols, name):
        a = [int(t2.loc["gating", a_cols].sum()), int(t2.loc["not gating", a_cols].sum())]
        b = [int(t2.loc["gating", b_cols].sum()), int(t2.loc["not gating", b_cols].sum())]
        if min(sum(a), sum(b)) == 0:
            return
        odds, p = fisher_exact([a, b])
        out[name] = {"odds_ratio": float(odds), "p": float(p)}

    fisher(["top interventional only"], ["top endogenous only"],
           "fisher_interv_only_vs_endo_only")
    fisher(["top interventional only", "top both"], ["top endogenous only"],
           "fisher_any_interv_vs_endo_only")
    fisher(["top endogenous only"], ["other"], "fisher_endo_only_vs_other")
    return out


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
    df = load(args, taus)
    df.to_csv(out / "knockout_head_gating.csv", index=False)
    summary = {"k": args.k, "regime": args.regime, "n_runs": int(df["run"].nunique())}
    for tau in taus:
        t2, t3 = tables(df, tau)
        tag = f"_tau{tau:g}"
        plot_table(t2, out / f"knockout_gating_2way{tag}.png", tau)
        plot_table(t3, out / f"knockout_gating_3way{tag}.png", tau)
        s = stats(t2)
        summary[f"tau{tau:g}"] = s
        print(f"\n[gating] rise threshold {tau:g} ({args.regime}):")
        print(t3.rename(index=ROW_LABEL, columns=lambda c: COL_LABEL[c].replace("\n", " "))
              .to_string())
        for c in COLUMNS:
            r = s[c]
            print(f"[gating]   {COL_LABEL[c].replace(chr(10), ' '):26s} gating "
                  f"{r['gating']}/{r['n']} = {r['rate']:.2f} "
                  f"(95% CI {r['ci95'][0]:.2f}-{r['ci95'][1]:.2f})")
        for name in ("fisher_interv_only_vs_endo_only", "fisher_any_interv_vs_endo_only",
                     "fisher_endo_only_vs_other"):
            if name in s:
                print(f"[gating]   {name}: odds ratio {s[name]['odds_ratio']:.2f}, "
                      f"p = {s[name]['p']:.3g}")
    (out / "knockout_head_gating_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[gating] wrote {out}")


if __name__ == "__main__":
    main()
