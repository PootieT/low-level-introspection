#!/usr/bin/env python3
"""
Aggregate the SAE prompted baseline (evals/score_sae_prompt_baseline.py) and set it against
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

GRID ONLY by default: the 48 (feature, position) runs of the layer x position grid. The
5 layer-9 reference runs (position -1) are dropped from the baseline AND the trained arms
(--include-l9 keeps them).

Writes to --out-dir (default outputs/plots/sae_prompt_baseline):
  baseline_all.csv, trained_all.csv     every row read
  ba_vs_k.{png,pdf,csv}                 BA vs k per readout x prompt, trained arms as references
  heatmap_ba.{png,pdf}                  layer x position: baseline probe@0 vs trained arms
  paired_summary.csv                    trained minus baseline, within (feature, position)
  learning_curve.{png,csv}              probe@0 AUROC vs active training rows
  scale_sweep.{png,csv}                 interventional AUROC vs steering scale
  missing_runs.json

  python plotters/plot_sae_prompt_baseline.py
  # re-plot from the tables a previous run wrote (e.g. a synced plot folder):
  python plotters/plot_sae_prompt_baseline.py --from-tables outputs/plots/sae_prompt_baseline
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
from plotters.sae_style import apply_rc, chance_line, save, style_axes, INK, INK2   # noqa: E402

PROMPT_COLOR = {"label": "#1f77b4", "index": "#ff7f0e"}   # same pair as the feature-gen plots
# trained arms, named as in the neuron figures (plotters/compare_sft_vs_joint.pretty_arm);
# both SAE arms ran gate off with the KL-0.1 anchor
ARM_NAME = {"sft": "SFT (Gate-off, KL=0.1)", "joint": "Joint Loss (W=0.9, KL=0.1)"}
# poster scale, identical to the neuron prompt-baseline figure (compare_sft_vs_joint.py)
POSTER_RC = {"font.size": 20, "axes.titlesize": 26, "axes.labelsize": 22,
             "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18,
             "legend.title_fontsize": 20, "figure.titlesize": 28}
NAME_RE = re.compile(r"saeJoint_(S(\d+)\.\d+)_POS(-?\d+)_gateoff_w(\d+)")
W_OF = {"10": 1.0, "09": 0.9, "075": 0.75, "05": 0.5}


RUN_DIR_RE = re.compile(r"(S(\d+)\.\d+)_POS(-?\d+)")


def load_baseline(root: Path):
    """Every run's table, identified by its DIRECTORY (<prompt>/<S..>_POS<p>/, which
    evals/score_sae_prompt_baseline.py names) rather than by the columns inside the file.
    A file whose own signal/position/prompt columns disagree with its directory is
    reported, so a stray or hand-copied table cannot silently join the wrong cell."""
    frames = []
    for f in sorted(root.glob("*/*/baseline_by_k.csv")):
        m = RUN_DIR_RE.fullmatch(f.parent.name)
        if m is None:
            print(f"[plot] SKIPPED {f}: directory is not named <S<layer>.<index>>_POS<position>")
            continue
        df = pd.read_csv(f)
        ident = {"prompt": f.parent.parent.name, "signal": m.group(1), "position": int(m.group(3))}
        for col, want in ident.items():
            if col in df:
                got = df[col]
                off = got.isna() | (got.astype(str) != str(want))
                if off.any():
                    print(f"[plot] WARNING {f}: {int(off.sum())}/{len(df)} rows have {col}="
                          f"{sorted(map(str, got[off].unique()))[:3]}, directory says {want!r}; "
                          f"using the directory")
            df[col] = want
        df["layer"] = int(m.group(2))
        frames.append(df)
    if not frames:
        raise SystemExit(f"no baseline results under {root}")
    return pd.concat(frames, ignore_index=True)


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
    """Mean +- SE over runs vs k, laid out and worded like the neuron baseline's
    corr_vs_k.png. Trained-arm reference lines average over the SAME runs as the curves."""
    runs = set(map(tuple, base[["signal", "position"]].drop_duplicates().values))
    tr = trained[[k in runs for k in zip(trained["signal"], trained["position"])]]
    s = mean_sem(base, ["prompt", "readout", "k", "eval_type"])
    s.to_csv(out_dir / "ba_vs_k.csv", index=False)
    ks = sorted(base["k"].unique())
    xpos = {k: i for i, k in enumerate(ks)}
    with plt.rc_context(POSTER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(22, 8.5), sharey=True)
        for ax, et, title in zip(axes, ("endogenous", "interventional"),
                                 ("Endogenous Balanced Accuracy", "Interventional Balanced Accuracy")):
            for prompt in ("index", "label"):
                c = PROMPT_COLOR[prompt]
                g = s[(s.prompt == prompt) & (s.eval_type == et)]
                pr = g[g.readout == "probe"].sort_values("k")
                ax.errorbar([xpos[k] for k in pr["k"]], pr["mean"], yerr=pr["sem"], marker="o",
                            markersize=9, linewidth=2.5, capsize=4, color=c, linestyle="-",
                            label=f"endo probe, mean pool ({prompt} prompt)")
                vb = g[g.readout == "verbal"].sort_values("k")
                ax.errorbar([xpos[k] for k in vb["k"]], vb["mean"], yerr=vb["sem"], marker="*",
                            markersize=18, linewidth=2.5, capsize=4, color=c, linestyle="--",
                            label=f"verbal: written answer, no probe ({prompt} prompt)")
                lg = g[(g.readout == "logit_lex") & (g.k == 0)]
                if len(lg):
                    ax.errorbar([xpos[0]], lg["mean"], yerr=lg["sem"], marker="D", markersize=12,
                                capsize=4, color=c, markeredgecolor="k", linestyle="none", zorder=6,
                                label=f"logit: active vs inactive words, no probe ({prompt} prompt)")
            for (arm, name), color in zip(ARM_NAME.items(), plt.get_cmap("Greys")([0.55, 0.9])):
                t = tr[(tr.arm == arm) & (tr.eval_type == et)]["balanced_accuracy"].dropna()
                if len(t):
                    ax.axhline(t.mean(), color=color, linestyle=":", linewidth=2.5,
                               label=f"{name} (n={len(t)})")
            # the control reads the text alone, so it is the same for both prompt arms
            ctrl = base[(base.readout == "noprompt_probe") & (base.k == 0) & (base.eval_type == et)
                        & (base.prompt == "index")]["balanced_accuracy"].dropna()
            if len(ctrl):
                ax.scatter([xpos[0]], [ctrl.mean()], marker="X", s=220, zorder=5, color="0.6",
                           edgecolor="k", label="no prompt, endo probe")
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels([str(k) for k in ks])
            ax.set_xlabel("Generated Tokens Pooled (k)")
            ax.set_title(title)
            ax.axhline(0.5, color="grey", lw=1)
            ax.grid(alpha=0.25)
        n_runs = len(runs)
        axes[0].set_ylabel("Balanced Accuracy (Mean ± SE over Features)")
        seen = {}
        for ax in axes:                   # a series can exist in one panel only (NaN BA)
            for h, l in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(l, h)
        fig.legend(list(seen.values()), list(seen), loc="lower center", ncol=3, fontsize=15,
                   bbox_to_anchor=(0.5, -0.2))
        fig.tight_layout()
        save(fig, out_dir / "ba_vs_k")
    print(f"[plot] ba_vs_k over {n_runs} runs; mean BA (mean pool / verbal):")
    print(s[s.readout.isin(["probe", "verbal"])].pivot_table(
        index="k", columns=["eval_type", "prompt", "readout"], values="mean").round(3).to_string())


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


def read_per_run(root: Path, name: str) -> pd.DataFrame:
    """<root>/<prompt>/<run>/<name> for every run, tagged with prompt, run, position."""
    frames = [pd.read_csv(f).assign(prompt=f.parts[-3], run=f.parts[-2])
              for f in sorted(root.glob(f"*/*/{name}"))]
    return with_position(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()


def with_position(df):
    if len(df) and "position" not in df:
        df["position"] = df["run"].str.extract(r"_POS(-?\d+)$")[0].astype(int)
    return df


def plot_learning_curve(df, out_dir):
    if df.empty:
        return
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


def plot_scale_sweep(df, out_dir):
    if df.empty:
        return
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
    p.add_argument("--from-tables", type=Path, default=None,
                   help="Re-plot from baseline_all.csv / trained_all.csv / learning_curve.csv / "
                        "scale_sweep.csv in this directory instead of reading every run.")
    p.add_argument("--include-l9", action="store_true",
                   help="Keep the 5 layer-9 reference runs (position -1); default is the grid only.")
    args = p.parse_args()
    apply_rc()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.from_tables:
        t = args.from_tables
        base = pd.read_csv(t / "baseline_all.csv")
        trained = pd.read_csv(t / "trained_all.csv")
        lc = with_position(pd.read_csv(t / "learning_curve.csv")) if (t / "learning_curve.csv").exists() else pd.DataFrame()
        sweep = with_position(pd.read_csv(t / "scale_sweep.csv")) if (t / "scale_sweep.csv").exists() else pd.DataFrame()
    else:
        base = load_baseline(args.baseline_root)
        trained = load_trained(args.trained)
        lc = read_per_run(args.baseline_root, "learning_curve.csv")
        sweep = read_per_run(args.baseline_root, "scale_sweep.csv")
        # unfiltered, so --from-tables can re-plot either selection later
        base.to_csv(args.out_dir / "baseline_all.csv", index=False)
        trained.to_csv(args.out_dir / "trained_all.csv", index=False)
        for name, df in (("learning_curve", lc), ("scale_sweep", sweep)):
            if len(df):
                df.to_csv(args.out_dir / f"{name}.csv", index=False)
    if not args.include_l9:
        # the layer-9 reference runs are the only ones at position -1
        keep = lambda df: df[df["position"] >= 0] if len(df) else df
        n0 = base[["signal", "position"]].drop_duplicates().shape[0]
        base, trained, lc, sweep = keep(base), keep(trained), keep(lc), keep(sweep)
        print(f"[plot] grid only: {base[['signal', 'position']].drop_duplicates().shape[0]} of "
              f"{n0} baseline runs (layer-9 reference runs dropped; --include-l9 keeps them)")
    have = set(map(tuple, base[["signal", "position"]].drop_duplicates().values))
    want = set(map(tuple, trained[["signal", "position"]].drop_duplicates().values))
    missing = sorted(f"{s}@{p}" for s, p in want - have)
    (args.out_dir / "missing_runs.json").write_text(json.dumps(missing, indent=2))
    if missing:
        print(f"[plot] {len(missing)} trained (feature, position) without a baseline, e.g. {missing[:4]}")
    plot_ba_vs_k(base, trained, args.out_dir)
    plot_heatmaps(base, trained, args.out_dir)
    paired_summary(base, trained, args.out_dir)
    plot_learning_curve(lc, args.out_dir)
    plot_scale_sweep(sweep, args.out_dir)


if __name__ == "__main__":
    main()
