#!/usr/bin/env python3
"""
Aggregate the SAE prompted baseline (score_sae_prompt_baseline.py) and set it against
the trained SAE runs on the SAME (feature, position) runs.

Baseline arms (outputs/runs/sae_prompt_baseline/<prompt>/<S..>_POS<p>/baseline_by_k.csv):
  verbal@k      the words the untrained model writes, matched to the paraphrase lists
  probe@k       logistic probe on its final hidden state, pooled over k generated tokens
  logit_lex/tok next-token logits of active- vs inactive-like words (k=0)
  noprompt      the probe on the text alone
each for prompt = index (the trained prompt) and label (Neuronpedia explanation).
Trained arms (outputs/runs/{sae_layerpos_joint,sae_l9_joint}/saeJoint_*/sae_joint_train_metrics.json):
  sft = w 1.0, joint = w 0.9 -- scored on the generated token, so balanced accuracy (BA)
  is the shared metric; the baseline's thresholded readouts use a threshold picked on a
  validation slice of train, never on test.

Writes to --out-dir (default outputs/plots/sae_prompt_baseline):
  baseline_all.csv, trained_all.csv     every row read
  ba_vs_k.{png,pdf,csv}                 BA vs k per readout x prompt, trained arms as references
  heatmap_ba.{png,pdf}                  layer x position: baseline probe@0 vs trained arms
  paired_summary.csv                    trained minus baseline, within (feature, position)
  learning_curve.{png,csv}              probe@0 AUROC vs active training rows
  scale_sweep.{png,csv}                 interventional AUROC vs steering scale
  missing_runs.json

  python analysis/plot_sae_prompt_baseline.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.sae_style import apply_rc, chance_line, save, style_axes, INK, INK2   # noqa: E402

PROMPT_COLOR = {"label": "#1f77b4", "index": "#ff7f0e"}   # same pair as the feature-gen plots
READOUT_STYLE = {"probe": dict(ls="-", marker="o"), "verbal": dict(ls="--", marker="s"),
                 "logit_lex": dict(ls="", marker="D"), "noprompt_probe": dict(ls="", marker="^")}
TRAINED_STYLE = {"sft": dict(color=INK, ls="-"), "joint": dict(color=INK2, ls="-.")}
NAME_RE = re.compile(r"saeJoint_(S(\d+)\.\d+)_POS(-?\d+)_gateoff_w(\d+)")
W_OF = {"10": 1.0, "09": 0.9, "075": 0.75, "05": 0.5}


def load_baseline(root: Path):
    frames = []
    for f in sorted(root.glob("*/*/baseline_by_k.csv")):
        frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit(f"no baseline results under {root}")
    df = pd.concat(frames, ignore_index=True)
    df["layer"] = df["signal"].str.extract(r"S(\d+)\.")[0].astype(int)
    return df


def load_trained(groups: list[Path]):
    rows = []
    for g in groups:
        for f in sorted(g.glob("saeJoint_*/sae_joint_train_metrics.json")):
            m = NAME_RE.match(f.parent.name)
            if not m:
                continue
            w = W_OF.get(m.group(4))
            arm = {1.0: "sft", 0.9: "joint"}.get(w)
            if arm is None:
                continue
            met = json.loads(f.read_text())
            for et in ("endogenous", "interventional"):
                t = met.get(f"test_{et}", {})
                rows.append({"arm": arm, "signal": m.group(1), "layer": int(m.group(2)),
                             "position": int(m.group(3)), "eval_type": et,
                             "balanced_accuracy": t.get("balanced_accuracy"),
                             "precision": t.get("precision")})
    return pd.DataFrame(rows)


def mean_sem(df, by, col="balanced_accuracy"):
    g = df.dropna(subset=[col]).groupby(by)[col]
    out = g.agg(["mean", "count", "std"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def plot_ba_vs_k(base, trained, out_dir):
    keys = set(map(tuple, base[["signal", "position"]].drop_duplicates().values))
    tr = trained[[k in keys for k in zip(trained["signal"], trained["position"])]]
    s = mean_sem(base, ["prompt", "readout", "k", "eval_type"])
    s.to_csv(out_dir / "ba_vs_k.csv", index=False)
    ks = sorted(base["k"].unique())
    xpos = {k: i for i, k in enumerate(ks)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True, layout="constrained")
    for ax, et in zip(axes, ("endogenous", "interventional")):
        for arm, st in TRAINED_STYLE.items():
            t = tr[(tr.arm == arm) & (tr.eval_type == et)]["balanced_accuracy"].dropna()
            if len(t):
                ax.axhline(t.mean(), lw=1.6, label=f"trained {arm} (n={len(t)})", **st)
        for prompt in ("index", "label"):
            for ro, st in READOUT_STYLE.items():
                g = s[(s.prompt == prompt) & (s.readout == ro) & (s.eval_type == et)].sort_values("k")
                if g.empty:
                    continue
                ax.errorbar([xpos[k] for k in g["k"]], g["mean"], yerr=g["sem"], ms=7, lw=2,
                            capsize=3, color=PROMPT_COLOR[prompt], **st,
                            label=f"{prompt} prompt: {ro}")
        chance_line(ax)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("generated tokens k")
        ax.set_title(et)
        style_axes(ax)
    axes[0].set_ylabel("balanced accuracy (mean over feature x position)")
    seen = {}
    for ax in axes:                       # a series can exist in one panel only (NaN BA)
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(list(seen.values()), list(seen), loc="outside lower center", ncol=4,
               frameon=True, fontsize=10)
    fig.suptitle("Untrained gemma-2-2b vs trained introspectors (SAE detection)")
    save(fig, out_dir / "ba_vs_k")


def plot_heatmaps(base, trained, out_dir):
    pb = base[(base.readout == "probe") & (base.k == 0) & (base.prompt == "index")]
    grid = pb[pb.position >= 0]
    if grid.empty:
        return
    panels = [("baseline probe@0 (index)", grid)]
    for arm in ("sft", "joint"):
        panels.append((f"trained {arm}", trained[(trained.arm == arm) & (trained.position >= 0)]))
    layers = sorted(grid["layer"].unique())
    positions = sorted(grid["position"].unique())
    fig, axes = plt.subplots(2, len(panels), figsize=(4.4 * len(panels), 7.6), layout="constrained")
    for r, et in enumerate(("endogenous", "interventional")):
        for c, (title, df) in enumerate(panels):
            ax = axes[r, c]
            piv = (df[df.eval_type == et].groupby(["layer", "position"])["balanced_accuracy"].mean()
                   .unstack().reindex(index=layers, columns=positions))
            im = ax.imshow(piv.values, vmin=0.5, vmax=1.0, cmap="Blues", origin="lower", aspect="auto")
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                                color="white" if v > 0.8 else INK)
            ax.set_xticks(range(len(positions)), [str(p) for p in positions])
            ax.set_yticks(range(len(layers)), [str(l) for l in layers])
            ax.set_xlabel("token position")
            if c == 0:
                ax.set_ylabel(f"{et}\nlayer")
            ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.6, label="balanced accuracy (mean over 3 features)")
    save(fig, out_dir / "heatmap_ba")


def paired_summary(base, trained, out_dir):
    rows = []
    for (prompt, ro, k, et), g in base.groupby(["prompt", "readout", "k", "eval_type"]):
        for arm in ("sft", "joint"):
            t = trained[(trained.arm == arm) & (trained.eval_type == et)]
            m = g.merge(t, on=["signal", "position"], suffixes=("_base", "_trained"))
            d = (m["balanced_accuracy_trained"] - m["balanced_accuracy_base"]).dropna()
            if len(d) == 0:
                continue
            rows.append({"prompt": prompt, "readout": ro, "k": k, "eval_type": et, "trained_arm": arm,
                         "n": len(d), "mean_trained": float(m["balanced_accuracy_trained"].mean()),
                         "mean_baseline": float(m["balanced_accuracy_base"].mean()),
                         "delta_mean": float(d.mean()), "delta_se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
                         "frac_trained_higher": float((d > 0).mean())})
    pd.DataFrame(rows).to_csv(out_dir / "paired_summary.csv", index=False)


def plot_learning_curve(root, out_dir):
    frames = []
    for f in root.glob("*/*/learning_curve.csv"):
        frames.append(pd.read_csv(f).assign(prompt=f.parts[-3], run=f.parts[-2]))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_dir / "learning_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5), layout="constrained")
    for prompt, g in df.groupby("prompt"):
        for col, ls in (("endo_auroc", "-"), ("interv_auroc", "--")):
            s = g.groupby("n_active")[col].agg(["mean", "sem"]).reset_index()
            ax.errorbar(s["n_active"], s["mean"], yerr=s["sem"], ls=ls, marker="o", capsize=3,
                        color=PROMPT_COLOR.get(prompt, INK2),
                        label=f"{prompt} prompt, {col.split('_')[0]}ogenous" if col.startswith("endo")
                        else f"{prompt} prompt, interventional")
    chance_line(ax)
    ax.set_xscale("log")
    ax.set_xlabel("active training rows (probe@0)")
    ax.set_ylabel("AUROC (mean over runs)")
    style_axes(ax)
    ax.legend(frameon=True, fontsize=10)
    ax.set_title("Is the untrained probe data-limited?")
    save(fig, out_dir / "learning_curve")


def plot_scale_sweep(root, out_dir):
    frames = []
    for f in root.glob("*/*/scale_sweep.csv"):
        frames.append(pd.read_csv(f).assign(prompt=f.parts[-3], run=f.parts[-2]))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_dir / "scale_sweep.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5), layout="constrained")
    styles = {"probe_endo": dict(ls="-", marker="o"), "probe_fresh": dict(ls=":", marker="s"),
              "logit_lex": dict(ls="--", marker="D")}
    names = {"probe_endo": "endogenous probe", "probe_fresh": "probe fit on steered rows",
             "logit_lex": "logit (lexicon)"}
    for prompt, g in df.groupby("prompt"):
        for ro, st in styles.items():
            s = g.groupby("scale")[f"{ro}_auroc"].agg(["mean", "sem"]).reset_index()
            ax.errorbar(s["scale"], s["mean"], yerr=s["sem"], capsize=3, ms=7, lw=2,
                        color=PROMPT_COLOR.get(prompt, INK2), label=f"{prompt} prompt: {names[ro]}", **st)
    chance_line(ax)
    ax.set_xscale("log")
    ax.set_xticks(sorted(df["scale"].unique()), [f"{s:g}x" for s in sorted(df["scale"].unique())])
    ax.set_xlabel("steering value (x median active value)")
    ax.set_ylabel("interventional AUROC (on vs off)")
    style_axes(ax)
    ax.legend(frameon=True, fontsize=9)
    ax.set_title("Is the injected feature visible at the readout at all?")
    save(fig, out_dir / "scale_sweep")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-root", type=Path, default=Path("outputs/runs/sae_prompt_baseline"))
    p.add_argument("--trained", type=Path, nargs="+",
                   default=[Path("outputs/runs/sae_layerpos_joint"), Path("outputs/runs/sae_l9_joint")])
    p.add_argument("--out-dir", type=Path, default=Path("outputs/plots/sae_prompt_baseline"))
    args = p.parse_args()
    apply_rc()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = load_baseline(args.baseline_root)
    trained = load_trained(args.trained)
    base.to_csv(args.out_dir / "baseline_all.csv", index=False)
    trained.to_csv(args.out_dir / "trained_all.csv", index=False)
    have = set(map(tuple, base[["signal", "position"]].drop_duplicates().values))
    want = set(map(tuple, trained[["signal", "position"]].drop_duplicates().values))
    missing = sorted(f"{s}@{p}" for s, p in want - have)
    (args.out_dir / "missing_runs.json").write_text(json.dumps(missing, indent=2))
    if missing:
        print(f"[plot] {len(missing)} trained (feature, position) without a baseline, e.g. {missing[:4]}")
    plot_ba_vs_k(base, trained, args.out_dir)
    plot_heatmaps(base, trained, args.out_dir)
    paired_summary(base, trained, args.out_dir)
    plot_learning_curve(args.baseline_root, args.out_dir)
    plot_scale_sweep(args.baseline_root, args.out_dir)


if __name__ == "__main__":
    main()
