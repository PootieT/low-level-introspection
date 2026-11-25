#!/usr/bin/env python3
"""
Which value does the report follow when the endogenous / interventional head SETS are
ablated? -- pooled over a sweep's mechanistic_analysis/score_head_set_ablation.py outputs.

One figure per head set definition (mass0.8, top30, ...): rows are (signal layer, token
position) cells, columns the three readouts, x the ablation conditions. Each blue dot is
one run (one neuron) with the real head set ablated; each gray dot is the same run's
mean over its layer- and size-matched random controls. Bars mark the median over runs.

  Endogenous             corr(prediction, natural activation a), no injection
  Interventional         corr(prediction, injected value v), x1 injection
  Natural | Injection    corr(prediction, a) under x1 injection: the report following
                         the natural value although the site carries v (context bypass)

Also writes head_set_ablation_runs.csv (every row of every run),
head_set_ablation_summary.csv (median and IQR over runs per cell x condition x control) and
head_set_ablation_tables.md: per head set definition, one pooled table (mean +/- std over
runs, random control in parentheses) and one per-cell table, for text-only replies.

The tables end with an SFT reference row: the matching SFT runs (--sft-group, same
neurons and positions, gate off), unablated, read from their own eval CSVs on the full
test split. Their endogenous and interventional prediction files cover the same rows in
the same order, so corr(injected-run prediction, natural activation) needs no GPU. SFT
bypasses the site, so this row shows what the natural-value column looks like when it does.

Usage
-----
python plotters/plot_head_set_ablation.py --group mlp_joint_grid_layer_token
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEQ_LEN = 32
CONDITIONS = ["none", "endo_only", "interv_only", "joint", "keep_endo", "keep_interv"]
COND_LABEL = {"none": "None", "endo_only": "Endo Only\n(E∖I)",
              "interv_only": "Interv. Only\n(I∖E)", "joint": "Joint\n(E∩I)",
              "keep_endo": "All but\nEndo", "keep_interv": "All but\nInterv."}
READOUTS = [("corr_endogenous", "Endogenous: corr(pred, a)"),
            ("corr_interv", "Interventional: corr(pred, v)"),
            ("corr_natural_under_injection", "Natural | Injection: corr(pred, a)")]
REAL, RANDOM = "#2a78d6", "#9a9893"       # series-1 blue; recessive gray for the control
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", default="mlp_joint_grid_layer_token")
    p.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    p.add_argument("--run-glob", default="*",
                   help="Glob over run directory names within the group.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: outputs/plots/<group>/head_set_ablation/")
    p.add_argument("--sft-group", default="mlp_endo_sweep_layer_token",
                   help="Group of the matching SFT runs for the reference row ('' to omit).")
    return p.parse_args()


def load(group_dir: Path, run_glob: str) -> pd.DataFrame:
    files = sorted(group_dir.glob(f"{run_glob}/head_set_ablation/head_set_ablation.csv"))
    if not files:
        raise SystemExit(f"no head_set_ablation.csv under {group_dir}/{run_glob}/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["t"] = np.where(df["position"] < 0, df["position"] + SEQ_LEN, df["position"])
    df["cell"] = "L=" + df["signal_layer"].astype(str) + ", t=" + df["t"].astype(str)
    print(f"[plot] {len(files)} run(s), {df['cell'].nunique()} cell(s)")
    return df


def per_run(df: pd.DataFrame, head_def: str) -> pd.DataFrame:
    """One value per (run, condition, control): real as is, random averaged over draws.
    The no-ablation row is shared by every head set definition."""
    d = df[(df["head_def"] == head_def) | (df["condition"] == "none")]
    keys = ["run", "cell", "signal_layer", "t", "condition", "control"]
    cols = [c for c, _ in READOUTS] + ["n_ablated", "empty"]
    return d.groupby(keys, as_index=False)[cols].mean()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hd in sorted(set(df["head_def"]) - {"none"}):
        r = per_run(df, hd)
        for (cell, cond, ctrl), g in r.groupby(["cell", "condition", "control"]):
            row = {"head_def": hd, "cell": cell, "condition": cond, "control": ctrl,
                   "n_runs": len(g), "n_empty": int((g["empty"] > 0).sum()),
                   "n_ablated_median": float(g["n_ablated"].median())}
            for c, _ in READOUTS:
                v = g[c].dropna()
                row.update({f"{c}_median": v.median() if len(v) else np.nan,
                            f"{c}_q25": v.quantile(0.25) if len(v) else np.nan,
                            f"{c}_q75": v.quantile(0.75) if len(v) else np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


def plot_head_def(df: pd.DataFrame, head_def: str, out: Path):
    r = per_run(df, head_def)
    cells = (r[["signal_layer", "t", "cell"]].drop_duplicates()
             .sort_values(["signal_layer", "t"])["cell"].tolist())
    fig, axes = plt.subplots(len(cells), len(READOUTS), squeeze=False, sharey=True,
                             figsize=(4.4 * len(READOUTS), 2.5 * len(cells) + 0.6))
    rng = np.random.default_rng(0)
    for i, cell in enumerate(cells):
        for j, (col, title) in enumerate(READOUTS):
            ax = axes[i, j]
            g = r[r["cell"] == cell]
            for k, cond in enumerate(CONDITIONS):
                for ctrl, color, dx, fill in (("real", REAL, -0.16, True),
                                              ("random", RANDOM, 0.16, False)):
                    v = g[(g["condition"] == cond) & (g["control"] == ctrl)][col].dropna()
                    if cond == "none" and ctrl == "random":
                        continue
                    x = k + (0 if cond == "none" else dx)
                    if len(v):
                        ax.scatter(x + rng.uniform(-0.05, 0.05, len(v)), v, s=22,
                                   facecolors=color if fill else "white", edgecolors=color,
                                   linewidths=1.2, zorder=3)
                        ax.hlines(v.median(), x - 0.12, x + 0.12, color=color, lw=2.5,
                                  zorder=4)
                n_empty = int(((g["condition"] == cond) & (g["control"] == "real")
                               & (g["empty"] > 0)).sum())
                if n_empty:
                    ax.text(k, -0.27, f"{n_empty} empty", ha="center", fontsize=7,
                            color=MUTED)
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.grid(axis="y", alpha=0.25)
            ax.set_ylim(-0.32, 1.05)
            ax.set_xticks(range(len(CONDITIONS)))
            ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS] if i == len(cells) - 1
                               else [])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"{cell}\nPearson r")
    handles = [plt.Line2D([], [], marker="o", ls="", color=REAL, label="Head set ablated"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", color=RANDOM,
                          label="Random heads, same count per layer")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Head Set Ablation ({head_def}); dot = one neuron, bar = median",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


# ======================================================================
# Markdown tables (text-only replies)
# ======================================================================
TABLE_LABEL = {"none": "None",
               "endo_only": "Endogenous-only heads (E∖I)",
               "interv_only": "Interventional-only heads (I∖E)",
               "joint": "Shared heads (E∩I)",
               "keep_endo": "All but endogenous heads (keep E)",
               "keep_interv": "All but interventional heads (keep I)"}
TABLE_COLS = ["Endogenous r(pred, a)", "Interventional r(pred, v)",
              "Under injection r(pred, a)"]
SFT_LABEL = "SFT, no ablation (reference)"


def _corr(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else np.nan


def sft_reference(df: pd.DataFrame, runs_root: Path, sft_group: str) -> pd.DataFrame:
    """The matching SFT run of every locale in `df`, unablated, from its eval CSVs."""
    rows, missing = [], []
    locs = df[["signal", "position", "cell"]].drop_duplicates()
    for _, r in locs.iterrows():
        pairs = [(d / "endogenous_test_predictions_endogenous.csv",
                  d / "interventional_test_predictions_gaussian.csv")
                 for d in sorted((runs_root / sft_group).glob(
                     f"endo_{r['signal']}_POS{r['position']}_gateoff_*"))]
        pairs = [p for p in pairs if p[0].exists() and p[1].exists()]
        if not pairs:
            missing.append(f"{r['signal']} POS{r['position']}")
            continue
        e, i = pd.read_csv(pairs[0][0]), pd.read_csv(pairs[0][1])
        if len(e) != len(i):
            raise SystemExit(f"{pairs[0][0].parent.name}: {len(e)} endogenous vs {len(i)} "
                             f"interventional rows; the files are not row-aligned")
        rows.append({"cell": r["cell"], "signal": r["signal"], "run": pairs[0][0].parent.name,
                     "corr_endogenous": _corr(e["predictions"], e["activation"]),
                     "corr_interv": _corr(i["predictions"], i["injected"]),
                     "corr_natural_under_injection": _corr(i["predictions"], e["activation"])})
    if missing:
        print(f"[table] no SFT eval CSVs under {runs_root / sft_group} for {len(missing)} "
              f"locale(s), e.g. {missing[:3]}")
    return pd.DataFrame(rows)


def _pm(v: pd.Series) -> str:
    v = v.dropna()
    if not len(v):
        return "–"
    return f"{v.mean():.2f} ± {(v.std() if len(v) > 1 else 0.0):.2f}"


def _definition(head_def: str) -> str:
    if head_def.startswith("mass"):
        return (f"the fewest heads holding {100 * float(head_def[4:]):g}% of the summed "
                f"single-head knockout drop")
    return f"the {head_def[3:]} heads with the largest single-head knockout drop"


def pooled_table(df: pd.DataFrame, head_def: str, sft: pd.DataFrame, meta: dict) -> str:
    r = per_run(df, head_def)
    n_runs = r["run"].nunique()
    lines = ["| Heads ablated | # ablated | " + " | ".join(TABLE_COLS) + " |",
             "|---|---:|---:|---:|---:|"]
    empties = []
    for cond in CONDITIONS:
        real = r[(r["condition"] == cond) & (r["control"] == "real")]
        rand = r[(r["condition"] == cond) & (r["control"] == "random")]
        n_empty = int((real["empty"] > 0).sum())
        if n_empty:
            empties.append(f"{TABLE_LABEL[cond].split(' (')[-1].rstrip(')')} in "
                           f"{n_empty}/{n_runs}")
        live = real[real["empty"] == 0]
        cells = []
        for c, _ in READOUTS:
            s = _pm(live[c])
            if cond != "none" and rand[c].notna().any():
                s += f" ({rand[c].mean():.2f})"
            cells.append(s)
        n_abl = int(live["n_ablated"].median()) if len(live) else 0
        dagger = " †" if n_empty else ""    # outside the f-string: py<3.12 rejects \ there
        lines.append(f"| {TABLE_LABEL[cond]}{dagger} | {n_abl} | " + " | ".join(cells) + " |")
    if len(sft):
        lines.append(f"| {SFT_LABEL} | – | "
                     + " | ".join(_pm(sft[c]) for c, _ in READOUTS) + " |")
    layers = sorted(r["signal_layer"].unique())
    ts = sorted(r["t"].unique())
    notes = [
        f"Mean ± std over {n_runs} joint-loss models (KL = 0.1; layers "
        f"{', '.join(map(str, layers))} × token positions {', '.join(map(str, ts))}; "
        f"one model per neuron), on the {meta.get('n_eval_rows', '~2.8K')} held-out C4 test "
        f"rows not used to rank heads. E / I: {_definition(head_def)} for the endogenous / "
        f"interventional correlation (Sec. 6.2); heads are mean-ablated from the signal token "
        f"onward. a: natural activation; v: injected value (Gaussian at the neuron's own "
        f"mean and std, as in the interventional eval). The last column correlates the "
        f"prediction under injection with the a that the injection replaced. In "
        f"parentheses: the same number of heads per layer chosen at random (mean over "
        f"{meta.get('n_random', 5)} draws)."]
    if empties:
        notes.append(f"† Empty set in some models ({'; '.join(empties)}); those models "
                     f"are left out of that row.")
    if len(sft):
        notes.append(f"SFT row: the same {len(sft)} locales trained with SFT only, no "
                     f"ablation, on the full test split.")
    return "\n".join(lines) + "\n\n" + "\n\n".join(notes)


def per_cell_table(df: pd.DataFrame, head_def: str, sft: pd.DataFrame) -> str:
    r = per_run(df, head_def)
    real = r[r["control"] == "real"]
    cells = (real[["signal_layer", "t", "cell"]].drop_duplicates()
             .sort_values(["signal_layer", "t"])["cell"].tolist())

    def trio(g):
        m = [g[c].mean() for c, _ in READOUTS]
        return " / ".join("–" if np.isnan(x) else f"{x:.2f}" for x in m)

    lines = ["| Heads ablated | " + " | ".join(cells) + " |",
             "|---|" + "---:|" * len(cells)]
    for cond in CONDITIONS:
        g = real[(real["condition"] == cond) & (real["empty"] == 0)]
        lines.append(f"| {TABLE_LABEL[cond]} | "
                     + " | ".join(trio(g[g["cell"] == c]) for c in cells) + " |")
    if len(sft):
        lines.append(f"| {SFT_LABEL} | "
                     + " | ".join(trio(sft[sft["cell"] == c]) for c in cells) + " |")
    return ("\n".join(lines) + "\n\nEach entry: endogenous r(pred, a) / interventional "
            "r(pred, v) / under-injection r(pred, a), mean over the neurons of that locale.")


def write_tables(df: pd.DataFrame, sft: pd.DataFrame, meta: dict, out: Path):
    parts = []
    for hd in sorted(set(df["head_def"]) - {"none"}):
        parts += [f"## Head sets: {hd}", pooled_table(df, hd, sft, meta),
                  "### By locale", per_cell_table(df, hd, sft)]
    text = "\n\n".join(parts) + "\n"
    out.write_text(text)
    print(text)
    print(f"[table] wrote {out}")


def run_meta(group_dir: Path, run_glob: str) -> dict:
    """Held-out row count and random draws, as the runs recorded them."""
    import json
    metas = [json.loads(p.read_text()) for p in
             sorted(group_dir.glob(f"{run_glob}/head_set_ablation/head_set_ablation_meta.json"))]
    if not metas:
        return {}
    rows = sorted({m["n_eval_rows"] for m in metas})
    return {"n_eval_rows": f"{rows[0]:,}" if len(rows) == 1 else f"{rows[0]:,}–{rows[-1]:,}",
            "n_random": max(m["n_random"] for m in metas)}


def main():
    args = parse_args()
    group_dir = args.runs_root / args.group
    df = load(group_dir, args.run_glob)
    out_dir = args.out_dir or ROOT / "outputs" / "plots" / args.group / "head_set_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "head_set_ablation_runs.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out_dir / "head_set_ablation_summary.csv", index=False)
    for hd in sorted(set(df["head_def"]) - {"none"}):
        plot_head_def(df, hd, out_dir / f"head_set_ablation_{hd}.png")
    sft = (sft_reference(df, args.runs_root, args.sft_group) if args.sft_group
           else pd.DataFrame())
    if len(sft):
        sft.to_csv(out_dir / "head_set_ablation_sft_reference.csv", index=False)
    write_tables(df, sft, run_meta(group_dir, args.run_glob),
                 out_dir / "head_set_ablation_tables.md")


if __name__ == "__main__":
    main()
