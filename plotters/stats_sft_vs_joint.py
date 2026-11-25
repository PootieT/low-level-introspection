#!/usr/bin/env python3
"""
Significance of the paper's Fig. 3 joint - SFT deltas, for all three signal types
(rebuttal: "values are means +/- std over 3-5 neurons ... no CIs, seeds or tests").

ONE TEST FOR EVERY SIGNAL TYPE. Each unit -- a neuron (5 per cell), an attention head
(3 per cell) or an SAE feature (3 per cell) -- contributes one PAIR of runs on the same
locale, joint and SFT. The test is the two-sided paired t-test over the pairs of a
slice, i.e. a one-sample t-test on the per-unit deltas Fig. 3 averages:
    t = mean(delta) / (sd(delta) / sqrt(n)),   df = n - 1,   sd with ddof = 1.
It only needs (n, mean, sd) per slice, which is what lets the attention panel, whose
per-head runs are not on this machine, go through the same test as the other two.

SLICES, each Holm-corrected within its family (signal type x metric x slice kind):
  cell       one (layer, position) cell: n = 5 / 3 / 3 pairs
  position   one position, all layers (different layers = different units)
  layer      one layer, all positions
  signal     everything of one signal type
Neurons and SAE features are REUSED across the positions of a layer (each cell trains
fresh runs on the same 5 neurons / 3 features), so the layer and signal slices pool
pairs that share a unit. `layer_unit_avg` / `signal_unit_avg` average each unit over
positions first and test the units (5 neurons or 3 features per layer, 40 / 12 per
signal) -- the robustness check for that dependence. Head identities are not readable
from the attention panel, so it has no unit-averaged rows.

DATA
  neuron     outputs/plots/sft_vs_joint/paired_runs_joint_kl0.1_vs_sft_gateoff.csv
             (joint KL=0.1, w=0.01 vs SFT w=1.0, gate off; 64 cells x 5 neurons)
  SAE        outputs/runs/sae_layerpos_joint/*/sae_joint_train_metrics.json
             (balanced accuracy, joint w=0.9 vs SFT w=1.0; 16 cells x 3 features)
  attention  ATTENTION_ARXIV below: transcribed from panel c of the arXiv version's
             Fig. 3 (~/Downloads/joint_loss.pdf, 2026-10-28 slide export; attention
             hyperparameters fixed after submission, panels a/b/d/e/f pixel-identical
             to the submission). Each cell is mean +/- SE (std ddof=1 / sqrt 3) over
             3 heads -- the format plotters/plot_paper_joint_loss_panel_c.py draws --
             so sd = SE * sqrt(3). Pooled slices are rebuilt EXACTLY from the per-cell
             (n, mean, sd) by the law of total variance.
The neuron and SAE tables reproduce the published cells to two decimals (checked
2026-11-25), so their tests are on exactly the numbers in the figure.

ROUNDING. The transcribed attention numbers have two decimals, so attention also gets
a CONSERVATIVE p: each cell's |mean| shrunk and SE grown by the rounding half-width
0.005 (a "0.01 +/- 0.00" cell cannot then pass on rounding alone). Attention cells and
slices count as significant only when the Holm-corrected CONSERVATIVE p does. A slice
whose deltas are all exactly 0 (e.g. both arms at chance) gets t = 0, p = 1; one whose
deltas are identical but non-zero has no variance, is untestable and never significant.

Outputs (outputs/plots/sft_vs_joint/significance/):
  significance_tests.csv     every slice: n, mean, sd, t, p, Holm p, conservative p,
                             significant
  significance_tables.md     markdown tables for the rebuttal (no images allowed)

Usage
-----
python plotters/stats_sft_vs_joint.py
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
ROUND = 0.005           # half-width of a value printed at two decimals
ALPHA = 0.05
SIGNALS = ("neuron", "attention", "sae")
SIGNAL_LABEL = {"neuron": "Neuron activation", "attention": "Attention weight",
                "sae": "SAE feature"}
METRICS = ("endogenous", "interventional")
UNIT = {"neuron": "neurons", "attention": "heads", "sae": "features"}
POS_LABEL = {"neuron": "Token position", "attention": "Source offset", "sae": "Token position"}

# Fig. 3 panel c, arXiv version: (mean, SE) per (layer, source offset), 3 heads per cell.
ATTN_OFFSETS = (-27, -19, -11, -3, -1)
ATTENTION_ARXIV = {
    "endogenous": {
        2: [(0.02, 0.01), (0.01, 0.01), (0.03, 0.02), (0.03, 0.01), (0.01, 0.00)],
        6: [(0.03, 0.04), (0.00, 0.02), (-0.02, 0.01), (-0.02, 0.01), (0.00, 0.00)],
        10: [(0.01, 0.01), (-0.03, 0.02), (-0.04, 0.04), (0.00, 0.00), (0.01, 0.01)],
        14: [(-0.01, 0.00), (-0.02, 0.00), (0.00, 0.01), (0.00, 0.01), (0.01, 0.01)],
    },
    "interventional": {
        2: [(0.56, 0.12), (0.52, 0.14), (0.71, 0.04), (0.68, 0.18), (0.65, 0.14)],
        6: [(0.49, 0.25), (0.48, 0.09), (0.44, 0.22), (0.55, 0.12), (0.39, 0.19)],
        10: [(0.64, 0.02), (0.41, 0.21), (0.24, 0.25), (0.61, 0.05), (0.40, 0.24)],
        14: [(0.23, 0.23), (0.00, 0.00), (0.00, 0.00), (0.00, 0.00), (0.00, 0.00)],
    },
}
ATTN_HEADS = 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--neuron-table", type=Path,
                   default=ROOT / "outputs/plots/sft_vs_joint/paired_runs_joint_kl0.1_vs_sft_gateoff.csv")
    p.add_argument("--sae-runs", type=Path, default=ROOT / "outputs/runs/sae_layerpos_joint")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "outputs/plots/sft_vs_joint/significance")
    p.add_argument("--alpha", type=float, default=ALPHA)
    return p.parse_args()


# ======================================================================
# Data: per-unit deltas (neuron, SAE) and per-cell summaries (attention)
# ======================================================================
def neuron_deltas(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["pos"] = np.where(d["position"] < 0, d["position"] + SEQ_LEN, d["position"])
    rows = []
    for m in METRICS:
        rows.append(pd.DataFrame({
            "signal": "neuron", "metric": m, "layer": d["layer"], "pos": d["pos"],
            "unit": d["layer"].astype(str) + "." + d["neuron"].astype(str),
            "delta": d[f"{m}_corr_joint_kl0.1"] - d[f"{m}_corr_sft_gateoff"]}))
    return pd.concat(rows, ignore_index=True)


SAE_RE = re.compile(r"saeJoint_S(?P<layer>\d+)\.(?P<feat>\d+)_POS(?P<pos>\d+)_gateoff_w(?P<w>\d+)$")


def sae_deltas(runs: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(runs.glob("*/sae_joint_train_metrics.json")):
        m = SAE_RE.match(f.parent.name)
        if not m:
            continue
        j = json.loads(f.read_text())
        rows.append({"layer": int(m["layer"]), "feat": int(m["feat"]), "pos": int(m["pos"]),
                     "arm": {"09": "joint", "10": "sft"}[m["w"]],
                     "endogenous": j["test_endogenous"]["balanced_accuracy"],
                     "interventional": j["test_interventional"]["balanced_accuracy"]})
    s = pd.DataFrame(rows)
    w = s.pivot_table(index=["layer", "pos", "feat"], columns="arm", values=list(METRICS))
    out = []
    for m in METRICS:
        dl = (w[(m, "joint")] - w[(m, "sft")]).dropna().reset_index(name="delta")
        out.append(dl.assign(signal="sae", metric=m,
                             unit=dl["layer"].astype(str) + "." + dl["feat"].astype(str))
                   .drop(columns="feat"))
    return pd.concat(out, ignore_index=True)


def attention_cells(conservative: bool) -> pd.DataFrame:
    """Per-cell (n, mean, sd) of the transcribed panel; `conservative` moves every cell
    to the edge of its rounding interval that weakens the test."""
    rows = []
    for m, grid in ATTENTION_ARXIV.items():
        for L, cells in grid.items():
            for pos, (mean, se) in zip(ATTN_OFFSETS, cells):
                if conservative:
                    mean = np.sign(mean) * max(abs(mean) - ROUND, 0.0)
                    se = se + ROUND
                rows.append({"signal": "attention", "metric": m, "layer": L, "pos": pos,
                             "n": ATTN_HEADS, "mean": mean, "sd": se * np.sqrt(ATTN_HEADS)})
    return pd.DataFrame(rows)


# ======================================================================
# Statistics
# ======================================================================
def summarize_units(x: np.ndarray) -> tuple[int, float, float]:
    x = np.asarray(x, dtype=float)
    return len(x), float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else np.nan


def pool_cells(cells: pd.DataFrame) -> tuple[int, float, float]:
    """(N, mean, sd) of all units of several cells, from each cell's (n, mean, sd): the
    law of total variance, exact up to the rounding of the inputs."""
    n, m, s = (cells[c].to_numpy(dtype=float) for c in ("n", "mean", "sd"))
    N = int(n.sum())
    M = float((n * m).sum() / N)
    ss = float(((n - 1) * s ** 2).sum() + (n * (m - M) ** 2).sum())
    return N, M, float(np.sqrt(ss / (N - 1))) if N > 1 else np.nan


def t_test(n: int, mean: float, sd: float) -> tuple[float, float]:
    """Two-sided one-sample t-test of mean = 0 from summary statistics. A slice with no
    variance is untestable (NaN), unless its mean is exactly 0 too (t = 0, p = 1)."""
    if n < 2 or not np.isfinite(sd):
        return np.nan, np.nan
    if sd < 1e-12:
        return (0.0, 1.0) if abs(mean) < 1e-12 else (np.nan, np.nan)
    t = mean / (sd / np.sqrt(n))
    return float(t), float(2 * student_t.sf(abs(t), n - 1))


def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values; NaNs pass through and do not count."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = len(ok)
    if m:
        order = ok[np.argsort(p[ok], kind="stable")]
        out[order] = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * p[order]))
    return out


SLICE_KEYS = {"cell": ["layer", "pos"], "position": ["pos"], "layer": ["layer"], "signal": []}


def unit_slices(units: pd.DataFrame) -> list[dict]:
    """Every slice of per-unit deltas (neuron, SAE), plus the unit-averaged robustness rows."""
    rows = []
    for (sig, m), g in units.groupby(["signal", "metric"]):
        for kind, keys in SLICE_KEYS.items():
            for key, s in (g.groupby(keys) if keys else [((), g)]):
                key = key if isinstance(key, tuple) else (key,)
                rows.append(_row(sig, m, kind, dict(zip(keys, key)), *summarize_units(s["delta"])))
        avg = g.groupby(["layer", "unit"])["delta"].mean().reset_index()
        for L, s in avg.groupby("layer"):
            rows.append(_row(sig, m, "layer_unit_avg", {"layer": L}, *summarize_units(s["delta"])))
        rows.append(_row(sig, m, "signal_unit_avg", {}, *summarize_units(avg["delta"])))
    return rows


def cell_slices(nominal: pd.DataFrame, conservative: pd.DataFrame) -> list[dict]:
    """Every slice of per-cell summaries (attention), nominal and conservative."""
    rows = []
    for (sig, m), g in nominal.groupby(["signal", "metric"]):
        gc = conservative[(conservative["signal"] == sig) & (conservative["metric"] == m)]
        for kind, keys in SLICE_KEYS.items():
            groups = g.groupby(keys) if keys else [((), g)]
            for key, s in groups:
                key = key if isinstance(key, tuple) else (key,)
                sel = dict(zip(keys, key))
                sc = gc
                for k, v in sel.items():
                    sc = sc[sc[k] == v]
                row = _row(sig, m, kind, sel, *pool_cells(s))
                row["t_conservative"], row["p_conservative"] = t_test(*pool_cells(sc))
                rows.append(row)
    return rows


def _row(sig, metric, kind, sel, n, mean, sd) -> dict:
    t, p = t_test(n, mean, sd)
    return {"signal": sig, "metric": metric, "slice": kind,
            "layer": sel.get("layer"), "pos": sel.get("pos"),
            "n": n, "mean": mean, "sd": sd, "t": t, "p": p}


def run_tests(units: pd.DataFrame, alpha: float) -> pd.DataFrame:
    res = pd.DataFrame(unit_slices(units) + cell_slices(attention_cells(False),
                                                       attention_cells(True)))
    for c in ("t_conservative", "p_conservative"):
        if c not in res:
            res[c] = np.nan
    # the t / p that decide significance: conservative for the transcribed attention panel
    att = res["signal"] == "attention"
    res["t_decide"] = np.where(att, res["t_conservative"], res["t"])
    res["p_decide"] = np.where(att, res["p_conservative"], res["p"])
    res["p_holm"] = np.nan
    for _, idx in res.groupby(["signal", "metric", "slice"]).groups.items():
        res.loc[idx, "p_holm"] = holm(res.loc[idx, "p_decide"])
    res["significant"] = res["p_holm"] < alpha
    return res


# ======================================================================
# Markdown
# ======================================================================
def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_delta(r, with_sd=True) -> str:
    s = f"{r['mean']:+.2f}" + (f" ± {r['sd']:.2f}" if with_sd and np.isfinite(r["sd"]) else "")
    return f"**{s}**" if r["significant"] else s


def slice_table(res, sig, kind, keyname, header) -> list[str]:
    sub = res[(res["signal"] == sig) & (res["slice"] == kind)]
    keys = sorted(sub[keyname].dropna().unique())
    lines = [f"| {header} | n | Endogenous Δ | p (Holm) | Interventional Δ | p (Holm) |",
             "|---|---|---|---|---|---|"]
    for k in keys:
        cells = []
        n = None
        for m in METRICS:
            r = sub[(sub["metric"] == m) & (sub[keyname] == k)].iloc[0]
            n = int(r["n"])
            cells += [fmt_delta(r), fmt_p(r["p_holm"])]
        lines.append(f"| {int(k)} | {n} | " + " | ".join(cells) + " |")
    return lines


def cell_grid(res, sig, metric, alpha: float = ALPHA) -> list[str]:
    sub = res[(res["signal"] == sig) & (res["metric"] == metric) & (res["slice"] == "cell")]
    layers, poss = sorted(sub["layer"].unique()), sorted(sub["pos"].unique())
    lines = [f"| Layer \\ {POS_LABEL[sig]} | " + " | ".join(str(int(p)) for p in poss) + " |",
             "|---|" + "---|" * len(poss)]
    for L in layers:
        row = []
        for p in poss:
            r = sub[(sub["layer"] == L) & (sub["pos"] == p)]
            row.append(fmt_delta(r.iloc[0], with_sd=False) if len(r) else "")
        lines.append(f"| {int(L)} | " + " | ".join(row) + " |")
    k, tot = int(sub["significant"].sum()), int(np.isfinite(sub["p_decide"]).sum())
    raw = int((sub["p_decide"] < alpha).sum())
    note = (f"{k} of {tot} cells significant after Holm over the {tot} cells "
            f"({raw} at uncorrected p < {alpha:g}); {int((sub['mean'] > 0).sum())} of "
            f"{len(sub)} cells have Δ > 0")
    if tot < len(sub):
        note += f" ({len(sub) - tot} untestable: identical non-zero Δ across {UNIT[sig]})"
    lines += ["", note + "."]
    return lines


def write_markdown(res: pd.DataFrame, out: Path, alpha: float) -> str:
    md = ["## Joint − SFT, all signal types", "",
          "| Signal | Metric | n pairs | Δ (mean ± sd) | t | p | n units (avg. over positions) | p (unit-avg.) |",
          "|---|---|---|---|---|---|---|---|"]
    for sig in SIGNALS:
        for m in METRICS:
            r = res[(res["signal"] == sig) & (res["metric"] == m) & (res["slice"] == "signal")].iloc[0]
            u = res[(res["signal"] == sig) & (res["metric"] == m) & (res["slice"] == "signal_unit_avg")]
            u_txt = ((str(int(u.iloc[0]["n"])), fmt_p(u.iloc[0]["p"])) if len(u) else ("–", "–"))
            md.append(f"| {SIGNAL_LABEL[sig]} | {m.capitalize()} | {int(r['n'])} | "
                      f"{fmt_delta(r)} | {r['t_decide']:.1f} | {fmt_p(r['p_decide'])} | "
                      f"{u_txt[0]} | {u_txt[1]} |")
    for sig in SIGNALS:
        md += ["", f"## {SIGNAL_LABEL[sig]}: per layer", ""]
        md += slice_table(res, sig, "layer", "layer", "Layer")
        md += ["", f"## {SIGNAL_LABEL[sig]}: per {POS_LABEL[sig].lower()}", ""]
        md += slice_table(res, sig, "position", "pos", POS_LABEL[sig])
        for m in METRICS:
            md += ["", f"## {SIGNAL_LABEL[sig]}: per cell, {m} Δ", ""]
            md += cell_grid(res, sig, m, alpha)
    md += ["", f"Two-sided paired t-test of joint vs SFT over the {{neurons, heads, features}} "
               f"of each slice (one-sample t on the per-unit Δ), Holm-corrected within each "
               f"table; **bold**: p < {alpha:g}. Neuron and SAE from the per-run results; "
               f"attention from the published cells (mean ± SE over 3 heads), significant only "
               f"if it survives moving every printed value to the weak end of its rounding "
               f"interval. Unit-averaged p: each neuron/feature averaged over positions first, "
               f"since the same units are retrained at every position of a layer."]
    text = "\n".join(md) + "\n"
    (out / "significance_tables.md").write_text(text)
    return text


def main():
    args = parse_args()
    units = pd.concat([neuron_deltas(args.neuron_table), sae_deltas(args.sae_runs)],
                      ignore_index=True)
    for sig, g in units.groupby("signal"):
        print(f"[stats] {sig}: {g.groupby('metric').size().to_dict()} unit pairs")
    res = run_tests(units, args.alpha)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_dir / "significance_tests.csv", index=False)
    print(write_markdown(res, args.out_dir, args.alpha))
    print(f"[stats] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
