#!/usr/bin/env python3
"""
Where does the signal layer's attention LoRA help the report? -- markdown tables over a
sweep's mechanistic_analysis/score_attention_lora_positions.py outputs.

Pooled table: one row per token group whose layer-L attention output is taken with its
LoRA zeroed, mean +/- std over runs of
  endogenous r(pred, a)         no injection
  interventional r(pred, v)     x1 injection
  write r(v, h)                 x1: how faithfully the neuron's write h = v * u carries v
  report r(pred, h)             x1: the report against what was actually written
then the gain lines (the target's u at t in the base model, finetuned, and with layer L's
attention LoRA removed at t; how its boost ranks among all neurons) and a per-locale table.

Usage
-----
python plotters/table_attn_lora_positions.py --group mlp_joint_grid_layer_token
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
         "all": "All tokens (= zeroing layer L's attention LoRA)",
         "t": "Signal token t only",
         "text_before_t": "Text before t",
         "text_after_t": "Text after t",
         "prompt": "Prompt (readout excluded)",
         "readout": "Readout token only",
         "all_but_t": "All tokens except t"}
READOUTS = [("corr_endogenous", "Endogenous r(pred, a)"),
            ("corr_interv", "Interventional r(pred, v)"),
            ("corr_v_h", "Write r(v, h)"),
            ("corr_interv_pred_h", "Report r(pred, h)")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", default="mlp_joint_grid_layer_token")
    p.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    p.add_argument("--run-glob", default="*")
    p.add_argument("--analysis-name", default="attn_lora_positions")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: outputs/plots/<group>/<analysis-name>/")
    return p.parse_args()


def _pm(v: pd.Series) -> str:
    v = v.dropna()
    if not len(v):
        return "–"
    return f"{v.mean():.2f} ± {(v.std() if len(v) > 1 else 0.0):.2f}"


def load(args) -> pd.DataFrame:
    files = sorted((args.runs_root / args.group).glob(
        f"{args.run_glob}/{args.analysis_name}/attn_lora_positions.csv"))
    if not files:
        raise SystemExit(f"no {args.analysis_name}/attn_lora_positions.csv under "
                         f"{args.runs_root / args.group}/{args.run_glob}/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["t"] = np.where(df["position"] < 0, df["position"] + SEQ_LEN, df["position"])
    df["cell"] = "L=" + df["signal_layer"].astype(str) + ", t=" + df["t"].astype(str)
    metas = pd.DataFrame([json.loads((f.parent / "attn_lora_positions_meta.json").read_text())
                          for f in files
                          if (f.parent / "attn_lora_positions_meta.json").exists()])
    print(f"[table] {df['run'].nunique()} run(s), {df['cell'].nunique()} cell(s)")
    return df, metas


def pooled_table(df: pd.DataFrame, metas: pd.DataFrame) -> str:
    conds = [c for c in LABEL if c in set(df["condition"])]
    lines = ["| Layer-L attention LoRA removed at | " + " | ".join(h for _, h in READOUTS)
             + " | Models |", "|---|" + "---:|" * (len(READOUTS) + 1)]
    for c in conds:
        g = df[df["condition"] == c]
        lines.append(f"| {LABEL[c]} | " + " | ".join(_pm(g[col]) for col, _ in READOUTS)
                     + f" | {g['run'].nunique()} |")
    n = df["run"].nunique()
    layers = sorted(df["signal_layer"].unique())
    ts = sorted(df["t"].unique())
    rows = (f"{int(metas['n_eval_rows'].iloc[0]):,} C4 test rows" if len(metas) and
            metas["n_eval_rows"].nunique() == 1 else "the C4 test split")
    note = (f"Mean ± std over models ({n} joint-loss models in all, KL = 0.1; signal layers "
            f"{', '.join(map(str, layers))} × token positions {', '.join(map(str, ts))}; one "
            f"model per neuron), on {rows}. Layer L's attention is run with and without its LoRA (q, k, "
            f"v, o) on the same input; the LoRA-off output replaces the finetuned one only "
            f"at the listed tokens. It runs before the signal MLP, so at t it changes the "
            f"MLP's input. The neuron writes h · W_down[:, i] with h = a · u (u = up_proj's "
            f"output at t; a replaced by v under injection). Text after t is empty when "
            f"t = 31. Injection: x1 Gaussian, as in the interventional eval.")
    return "\n".join(lines) + "\n\n" + note


def gain_lines(df: pd.DataFrame, metas: pd.DataFrame) -> list[str]:
    out = []
    if len(metas) and "u_base_mean" in metas:
        m = metas.dropna(subset=["u_base_mean"])
        out.append(f"Target neuron's gain u at t (per-model mean, then mean ± std over "
                   f"{len(m)} models): base model {_pm(m['u_base_mean'])}; finetuned "
                   f"{_pm(m['u_finetuned_mean'])}; finetuned with layer L's attention LoRA "
                   f"removed at t {_pm(m['u_attn_off_t_mean'])}.")
        for other, label in (("base", "the base model"),
                             ("attn_off_t", "layer L's attention LoRA removed at t")):
            r, rank = m[f"abs_u_ratio_vs_{other}"], m[f"abs_u_ratio_vs_{other}_rank"]
            out.append(f"Mean |u| at t, finetuned over {label}: target {r.median():.1f}× "
                       f"(median over models; median neuron "
                       f"{m[f'abs_u_ratio_vs_{other}_median'].median():.2f}×); the target's "
                       f"boost ranks first of {int(m['n_neurons'].iloc[0]):,} neurons in "
                       f"{int((rank == 1).sum())} of {len(m)} models (worst rank "
                       f"{int(rank.max())}).")
        conds = ()
    else:
        conds = ("none", "t", "all")
    for c in conds:
        g = df[df["condition"] == c]
        if not len(g):
            continue
        out.append(f"u at t, {LABEL[c].lower()}: mean {_pm(g['u_mean'])}, std "
                   f"{_pm(g['u_std'])}, positive in {_pm(100 * g['u_frac_positive'])} % of "
                   f"rows; row-by-row r with the unedited u {_pm(g['u_corr_none'])}.")
    g = df[df["condition"] == "t"]
    if len(g):
        out.append(f"Neuron's own activation with the LoRA removed at t: r(live a, cached a) "
                   f"{_pm(g['corr_a_live_cached'])} (unedited "
                   f"{_pm(df[df['condition'] == 'none']['corr_a_live_cached'])}).")
    return out


def per_cell_table(df: pd.DataFrame) -> str:
    conds = [c for c in LABEL if c in set(df["condition"])]
    cells = (df[["signal_layer", "t", "cell"]].drop_duplicates()
             .sort_values(["signal_layer", "t"])["cell"].tolist())
    lines = ["| Layer-L attention LoRA removed at | " + " | ".join(cells) + " |",
             "|---|" + "---:|" * len(cells)]
    for c in conds:
        g = df[df["condition"] == c]
        vals = []
        for cell in cells:
            h = g[g["cell"] == cell]
            vals.append(f"{h['corr_endogenous'].mean():.2f} / {h['corr_interv'].mean():.2f}"
                        if len(h) else "–")
        lines.append(f"| {LABEL[c]} | " + " | ".join(vals) + " |")
    return ("\n".join(lines) + "\n\nEach entry: endogenous r(pred, a) / interventional "
            "r(pred, v), mean over the neurons of that locale.")


def main():
    args = parse_args()
    df, metas = load(args)
    out_dir = args.out_dir or ROOT / "outputs" / "plots" / args.group / args.analysis_name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "attn_lora_positions_runs.csv", index=False)
    text = "\n\n".join(["## Signal-layer attention LoRA by token", pooled_table(df, metas),
                        "\n".join(f"- {s}" for s in gain_lines(df, metas)),
                        "### By locale", per_cell_table(df)]) + "\n"
    (out_dir / "attn_lora_positions_tables.md").write_text(text)
    print(text)
    print(f"[table] wrote {out_dir / 'attn_lora_positions_tables.md'}")


if __name__ == "__main__":
    main()
