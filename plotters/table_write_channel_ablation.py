#!/usr/bin/env python3
"""
Which part of the target neuron's write does the report read? -- markdown tables over a
sweep's mechanistic_analysis/score_write_channel_ablation.py outputs.

Pooled table: one row per write condition, mean +/- std over runs of
  endogenous r(pred, a)         no injection
  interventional r(pred, v)     x1 injection
  under injection r(pred, a)    x1 injection, against the natural value it replaced
then a per-locale table (endo / interv / natural means) and the column-norm line (how big
the LoRA-added W_down column is next to the base one, and their cosine).

Usage
-----
python plotters/table_write_channel_ablation.py --group mlp_joint_grid_layer_token
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEQ_LEN = 32
LABEL = {"none": "Unchanged",
         "remove_native_write": "Native write removed (base W_down column)",
         "remove_added_write": "LoRA-added write removed (ΔW_down column)",
         "base_up": "W_up edit undone for this neuron",
         "base_write": "Base-model write (both edits undone)",
         "remove_write": "Whole write removed"}
READOUTS = [("corr_endogenous", "Endogenous r(pred, a)"),
            ("corr_interv", "Interventional r(pred, v)"),
            ("corr_natural_under_injection", "Under injection r(pred, a)")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", default="mlp_joint_grid_layer_token")
    p.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    p.add_argument("--run-glob", default="*")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: outputs/plots/<group>/write_channel_ablation/")
    return p.parse_args()


def _pm(v: pd.Series) -> str:
    v = v.dropna()
    if not len(v):
        return "–"
    return f"{v.mean():.2f} ± {(v.std() if len(v) > 1 else 0.0):.2f}"


def load(group_dir: Path, run_glob: str):
    files = sorted(group_dir.glob(f"{run_glob}/write_channel_ablation/write_channel_ablation.csv"))
    if not files:
        raise SystemExit(f"no write_channel_ablation.csv under {group_dir}/{run_glob}/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["t"] = np.where(df["position"] < 0, df["position"] + SEQ_LEN, df["position"])
    df["cell"] = "L=" + df["signal_layer"].astype(str) + ", t=" + df["t"].astype(str)
    metas = pd.DataFrame([json.loads((f.parent / "write_channel_ablation_meta.json").read_text())
                          for f in files
                          if (f.parent / "write_channel_ablation_meta.json").exists()])
    print(f"[table] {df['run'].nunique()} run(s), {df['cell'].nunique()} cell(s)")
    return df, metas


def pooled_table(df: pd.DataFrame, metas: pd.DataFrame) -> str:
    conds = [c for c in LABEL if c in set(df["condition"])]
    lines = ["| Write of the target neuron at the signal token | "
             + " | ".join(h for _, h in READOUTS) + " |", "|---|---:|---:|---:|"]
    for c in conds:
        g = df[df["condition"] == c]
        lines.append(f"| {LABEL[c]} | " + " | ".join(_pm(g[col]) for col, _ in READOUTS) + " |")
    n = df["run"].nunique()
    layers = sorted(df["signal_layer"].unique())
    ts = sorted(df["t"].unique())
    rows = (f"{int(metas['n_eval_rows'].iloc[0]):,}" if len(metas) and
            metas["n_eval_rows"].nunique() == 1 else "all")
    note = (f"Mean ± std over {n} joint-loss models (KL = 0.1; signal layers "
            f"{', '.join(map(str, layers))} × token positions {', '.join(map(str, ts))}; "
            f"one model per neuron), on {rows} C4 test rows. Only the target neuron's write at "
            f"the signal layer and token is edited. Its finetuned write is h · (W_down + "
            f"ΔW_down)[:, i] with h = a · u; native = the base-weight part, LoRA-added "
            f"= the Δ part. a: natural activation; v: injected value (x1 Gaussian, as in "
            f"the interventional eval).")
    parts = ["\n".join(lines), note]
    if len(metas) and "added_over_native" in metas:
        parts.append(f"Column sizes: |ΔW_down[:, i]| = {_pm(metas['added_over_native'])} × "
                     f"|W_down[:, i]|, cosine {_pm(metas['cos_added_native'])}.")
    return "\n\n".join(parts)


def per_cell_table(df: pd.DataFrame) -> str:
    conds = [c for c in LABEL if c in set(df["condition"])]
    cells = (df[["signal_layer", "t", "cell"]].drop_duplicates()
             .sort_values(["signal_layer", "t"])["cell"].tolist())
    lines = ["| Write of the target neuron | " + " | ".join(cells) + " |",
             "|---|" + "---:|" * len(cells)]
    for c in conds:
        g = df[df["condition"] == c]
        vals = []
        for cell in cells:
            h = g[g["cell"] == cell]
            vals.append(" / ".join(f"{h[col].mean():.2f}" for col, _ in READOUTS)
                        if len(h) else "–")
        lines.append(f"| {LABEL[c]} | " + " | ".join(vals) + " |")
    return ("\n".join(lines) + "\n\nEach entry: endogenous r(pred, a) / interventional "
            "r(pred, v) / under-injection r(pred, a), mean over the neurons of that locale.")


def main():
    args = parse_args()
    df, metas = load(args.runs_root / args.group, args.run_glob)
    out_dir = args.out_dir or ROOT / "outputs" / "plots" / args.group / "write_channel_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "write_channel_ablation_runs.csv", index=False)
    text = "\n\n".join(["## Write-channel ablation", pooled_table(df, metas),
                        "### By locale", per_cell_table(df)]) + "\n"
    (out_dir / "write_channel_ablation_tables.md").write_text(text)
    print(text)
    print(f"[table] wrote {out_dir / 'write_channel_ablation_tables.md'}")


if __name__ == "__main__":
    main()
