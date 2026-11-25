#!/usr/bin/env python3
"""
Do the LoRA modules finetuning changed most also matter most? -- markdown tables over a
sweep's mechanistic_analysis/ablate_lora_modules.py outputs, next to Fig. 7's LoRA norms.

Inputs
  <runs-root>/<group>/<run>/<analysis-name>/lora_ablation.csv   (run with --granularity both)
  outputs/plots/<group>/lora_weights/lora_profiles.csv          (per-module Frobenius norms,
                                                                 plotters/compare_lora_weight_patterns.py)

Table: by layer relative to the signal layer L, for each layer's whole MLP LoRA and whole
attention LoRA:
  LoRA norm      the block's Frobenius norm over the run's largest block norm (Fig. 7's
                 quantity), mean over the layers in the bin, then over runs
  delta r        change in interventional / endogenous r when that block's LoRA is zeroed
                 (one layer at a time), averaged the same way
Summary lines: where each run's largest MLP / attention drop sits; Spearman, per run,
between a module's LoRA norm and the interventional drop when it alone is zeroed (and the
same over blocks); the baseline and all-LoRA-off references.

Usage
-----
python plotters/table_lora_ablation.py --group mlp_joint_grid_layer_token
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEQ_LEN = 32
BINS = ["< L", "L", "L+1", "L+2", "L+3 to L+6", "≥ L+7"]
BLOCK_OF = {"q_proj": "attn", "k_proj": "attn", "v_proj": "attn", "o_proj": "attn",
            "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", default="mlp_joint_grid_layer_token")
    p.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    p.add_argument("--run-glob", default="*")
    p.add_argument("--analysis-name", default="lora_ablation_x1")
    p.add_argument("--norms", type=Path, default=None,
                   help="Default: outputs/plots/<group>/lora_weights/lora_profiles.csv")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: outputs/plots/<group>/<analysis-name>/")
    return p.parse_args()


def _pm(v: pd.Series) -> str:
    v = v.dropna()
    return f"{v.mean():.2f} ± {(v.std() if len(v) > 1 else 0.0):.2f}"


def rel_bin(rel: int) -> str:
    if rel < 0:
        return BINS[0]
    if rel <= 2:
        return BINS[rel + 1]
    return BINS[4] if rel <= 6 else BINS[5]


def rel_label(rel: int) -> str:
    return "L" if rel == 0 else (f"L+{rel}" if rel > 0 else f"L{rel}")


def load(args) -> pd.DataFrame:
    files = sorted((args.runs_root / args.group).glob(
        f"{args.run_glob}/{args.analysis_name}/lora_ablation.csv"))
    if not files:
        raise SystemExit(f"no {args.analysis_name}/lora_ablation.csv under "
                         f"{args.runs_root / args.group}/{args.run_glob}/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "granularity" not in df or not (df["granularity"] == "block").any():
        raise SystemExit("no block rows: run ablate_lora_modules.py with --granularity both")
    df["t"] = np.where(df["position"] < 0, df["position"] + SEQ_LEN, df["position"])
    df["cell"] = "L=" + df["signal_layer"].astype(str) + ", t=" + df["t"].astype(str)
    metas = [json.loads((f.parent / "ablation_meta.json").read_text()) for f in files
             if (f.parent / "ablation_meta.json").exists()]
    df.attrs["n_examples"] = sorted({m.get("n_examples") for m in metas} - {None})
    print(f"[table] {df['run'].nunique()} run(s), {df['cell'].nunique()} cell(s)")
    return df


def block_norms(norms_csv: Path, runs: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """((run, layer, block) -> block Frobenius norm over the run's largest block,
    the per-module norm rows of these runs)."""
    n = pd.read_csv(norms_csv)
    n = n[n["condition"].isin(runs)].copy()
    if n.empty:
        raise SystemExit(f"{norms_csv} has none of the ablated runs")
    n["block"] = n["module"].map(BLOCK_OF)
    b = (n.assign(sq=n["frob"] ** 2).groupby(["condition", "layer", "block"], as_index=False)
         ["sq"].sum())
    b["norm"] = np.sqrt(b["sq"])
    b["rel_norm"] = b["norm"] / b.groupby("condition")["norm"].transform("max")
    return b.rename(columns={"condition": "run"})[["run", "layer", "block", "rel_norm"]], n


def block_table(df: pd.DataFrame, bn: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    blk = df[df["granularity"] == "block"].copy()
    blk["layer"] = blk["layer"].astype(int)
    blk = blk.rename(columns={"module": "block"}).merge(bn, on=["run", "layer", "block"],
                                                       how="left")
    blk["rel"] = blk["layer"] - blk["signal_layer"]
    blk["bin"] = blk["rel"].map(rel_bin)
    per_run = (blk.groupby(["run", "block", "bin"], as_index=False)
               [["rel_norm", "d_interv", "d_endo"]].mean())
    agg = per_run.groupby(["block", "bin"])[["rel_norm", "d_interv", "d_endo"]].mean()
    n_runs = per_run.groupby(["block", "bin"])["run"].nunique()
    head = ("| Layer | MLP: LoRA norm | MLP: Δ interv. r | MLP: Δ endo. r | "
            "Attn: LoRA norm | Attn: Δ interv. r | Attn: Δ endo. r | Models |")
    lines = [head, "|---|" + "---:|" * 7]
    for b in BINS:
        cells = []
        for blk_name in ("mlp", "attn"):
            if (blk_name, b) in agg.index:
                r = agg.loc[(blk_name, b)]
                cells += [f"{r['rel_norm']:.2f}", f"{r['d_interv']:+.2f}", f"{r['d_endo']:+.2f}"]
            else:
                cells += ["–"] * 3
        n = int(n_runs.get(("mlp", b), n_runs.get(("attn", b), 0)))
        lines.append(f"| {b} | " + " | ".join(cells) + f" | {n} |")
    return "\n".join(lines), blk


def summary_lines(df: pd.DataFrame, blk: pd.DataFrame, norms: pd.DataFrame) -> list[str]:
    out = []
    n_runs = df["run"].nunique()
    for blk_name, label in (("mlp", "MLP"), ("attn", "attention")):
        g = blk[blk["block"] == blk_name]
        top = g.loc[g.groupby("run")["d_interv"].idxmin()]          # most negative delta
        counts = top["rel"].value_counts().sort_index()
        where = ", ".join(f"{rel_label(int(r))} in {c}" for r, c in counts.items())
        out.append(f"Largest interventional drop from removing one layer's {label} LoRA: "
                   f"{where} (of {n_runs} models; median drop "
                   f"{-top['d_interv'].median():.2f}).")
    # does a module's update size predict how much the report needs it?
    mod = df[df["granularity"] == "module"].copy()
    rhos = {"module": [], "block": []}
    if len(mod):
        mod["layer"] = mod["layer"].astype(int)
        m = mod.merge(norms[["condition", "layer", "module", "frob"]].rename(
            columns={"condition": "run"}), on=["run", "layer", "module"], how="left")
        for _, g in m.dropna(subset=["frob"]).groupby("run"):
            rhos["module"].append(spearmanr(g["frob"], -g["d_interv"])[0])
    for _, g in blk.dropna(subset=["rel_norm"]).groupby("run"):
        rhos["block"].append(spearmanr(g["rel_norm"], -g["d_interv"])[0])
    for k, v in rhos.items():
        v = np.array([x for x in v if np.isfinite(x)])
        if len(v):
            out.append(f"Spearman between a {k}'s LoRA norm and the interventional drop when "
                       f"it is zeroed: {v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f} "
                       f"(mean ± std over {len(v)} models).")
    ref = df[df["layer"].astype(str).isin(["baseline", "all_off"])]
    for key, label in (("baseline", "No ablation"), ("all_off", "All LoRA zeroed")):
        r = ref[ref["layer"].astype(str) == key]
        if len(r):
            out.append(f"{label}: interventional r {_pm(r['interv'])}, "
                       f"endogenous r {_pm(r['endo'])}.")
    return out


def main():
    args = parse_args()
    df = load(args)
    norms_csv = args.norms or ROOT / "outputs" / "plots" / args.group / "lora_weights" / "lora_profiles.csv"
    bn, norms = block_norms(norms_csv, set(df["run"]))
    table, blk = block_table(df, bn)
    layers = sorted(df["signal_layer"].unique())
    ts = sorted(df["t"].unique())
    n_ex = df.attrs.get("n_examples") or []
    rows = f"{n_ex[0]:,} C4 test rows" if len(n_ex) == 1 else "the C4 test split"
    note = (f"Mean over {df['run'].nunique()} joint-loss models (KL = 0.1; signal layers "
            f"{', '.join(map(str, layers))} × token positions {', '.join(map(str, ts))}; "
            f"one model per neuron), on {rows}. Each row bins layers by their "
            f"offset from the signal layer L; a block's LoRA (attention: q, k, v, o; MLP: "
            f"gate, up, down) is zeroed one layer at a time and the change in r is averaged "
            f"over the layers in the bin. LoRA norm: the block's Frobenius norm over the "
            f"model's largest block (Fig. 7). Interventional r: x1 Gaussian injection, as in "
            f"the interventional eval. Row L's attention runs before the signal MLP in that "
            f"layer, so it cannot read the neuron's write; its effect on the report goes "
            f"through what it writes elsewhere (other tokens, or the signal token's residual).")
    text = "\n\n".join(["## LoRA ablation by layer", table, note,
                        "\n".join(f"- {s}" for s in summary_lines(df, blk, norms))]) + "\n"
    out_dir = args.out_dir or ROOT / "outputs" / "plots" / args.group / args.analysis_name
    out_dir.mkdir(parents=True, exist_ok=True)
    blk.to_csv(out_dir / "lora_ablation_blocks.csv", index=False)
    (out_dir / "lora_ablation_tables.md").write_text(text)
    print(text)
    print(f"[table] wrote {out_dir / 'lora_ablation_tables.md'}")


if __name__ == "__main__":
    main()
