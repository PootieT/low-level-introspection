#!/usr/bin/env python3
"""Figures for the layer x position SAE joint sweep (scripts/sae/train_sae_sweep.sh).

Reads every sae_joint_train_metrics.json under --runs (run names
saeJoint_S<L>.<i>_POS<p>_gateoff_w<W>), ranks features by scripts/sae/sweep_features.txt,
labels them with their Neuronpedia concept (annotated parquets), and writes, per weight W:

  heatmap_<metric>_w<W>          layer x position, mean over the 3 features, endo | interv
  heatmap_<metric>_byfeat_w<W>   same, one row per feature rank (F1-F3)
  vs_position_bacc_w<W>          balanced acc vs position, one line per layer (mean +- SE)
  vs_layer_bacc_w<W>             balanced acc vs layer, one line per position (mean +- SE);
                                 --anchor runs (layer 9, POS -1) drawn as reference points
  base_rate_w<W>                 natural active rate on the test split, feature x position
  confusion_L<L>_w<W>            endogenous confusion, feature x position (runs that logged
                                 per-class counts)

plus sweep_summary.csv (one row per run). Metrics are the current set only: balanced
accuracy + precision on the generated token.

  python analysis/plot_sae_sweep.py --runs outputs/runs/sae_layerpos_joint \
      --anchor outputs/runs/sae_l9_joint
"""
import argparse, glob, json, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sae_style import INK, SMALL, apply_rc, box, chance_line, fig_title, save, style_axes

ROOT = Path(__file__).parents[1]
apply_rc()
RUN_RE = re.compile(r"saeJoint_S(?P<layer>\d+)\.(?P<index>\d+)_POS(?P<pos>-?\d+)_gateoff_w(?P<w>\d+)")
ARMS = [("test_endogenous", "endogenous"), ("test_interventional", "interventional")]
METRICS = {"balanced_accuracy": "balanced accuracy", "precision": "precision"}
RANK_COLOR = dict(zip(["F1", "F2", "F3"], sns.color_palette("tab10", 3)))


def parse_w(tag):
    return float(tag) / (10 ** (len(tag) - 1))


def load_ranks(path):
    ranks = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split("#")[0].split()
        if len(parts) >= 2 and "TODO" not in parts:
            for k, f in enumerate(parts[1:4]):
                ranks[f] = f"F{k + 1}"
    return ranks


def load_concepts(layers):
    out = {}
    for L in layers:
        cands = [ROOT / f"data/sae_stats/gemma2-2b_res{L}_pos-1_16k.annotated.parquet",
                 ROOT / f"data/sae_stats/gemma2-2b_res{L}_16k.annotated.parquet"]
        p = next((c for c in cands if c.exists()), None)
        if p is None:
            continue
        d = pd.read_parquet(p, columns=["feature", "explanation"]).dropna()
        out.update({f"S{L}.{int(r.feature)}": str(r.explanation).strip() for r in d.itertuples()})
    return out


def load_runs(runs_dir, source):
    rows = []
    for f in glob.glob(f"{runs_dir}/saeJoint_S*_POS*_gateoff_w*/sae_joint_train_metrics.json"):
        m = RUN_RE.search(Path(f).parent.name)
        if not m:
            continue
        met = json.loads(Path(f).read_text())
        row = dict(source=source, layer=int(m["layer"]), feature=f"S{m['layer']}.{m['index']}",
                   position=int(m["pos"]), w=float(met.get("joint_loss_weight", parse_w(m["w"]))))
        for arm, short in ARMS:
            for k, v in (met.get(arm) or {}).items():
                if isinstance(v, (int, float)):
                    row[f"{short}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def heat(ax, piv, title, vmin=0.5):
    sns.heatmap(piv, ax=ax, vmin=vmin, vmax=1.0, cmap="Blues", annot=True, fmt=".2f",
                annot_kws={"fontsize": SMALL - 2}, cbar=False, linewidths=0.5, linecolor="white")
    ax.set_title(title); ax.set_xlabel("token position"); ax.set_ylabel("layer")
    ax.tick_params(axis="y", rotation=0)
    box(ax)


def plot_heatmaps(df, w, metric, out):
    vmin = 0.5 if metric == "balanced_accuracy" else 0.0
    d = df[df.w == w]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (_, short) in zip(axes, ARMS):
        piv = d.pivot_table(index="layer", columns="position", values=f"{short}_{metric}", aggfunc="mean")
        heat(ax, piv, short, vmin)
    fig_title(fig, f"{METRICS[metric]}, mean over 3 features per layer (w={w:g})")
    fig.tight_layout(); save(fig, out / f"heatmap_{metric}_w{w:g}")

    ranks = sorted(d["rank"].dropna().unique())
    if not ranks:
        return
    fig, axes = plt.subplots(len(ranks), 2, figsize=(11, 4.0 * len(ranks)), squeeze=False)
    for i, r in enumerate(ranks):
        for j, (_, short) in enumerate(ARMS):
            piv = d[d["rank"] == r].pivot_table(index="layer", columns="position",
                                                values=f"{short}_{metric}", aggfunc="mean")
            heat(axes[i, j], piv, f"{r}: {short}", vmin)
    fig_title(fig, f"{METRICS[metric]} by feature rank (w={w:g}); feature differs per layer")
    fig.tight_layout(); save(fig, out / f"heatmap_{metric}_byfeat_w{w:g}")


def mean_se(d, by, col):
    g = d.groupby(by)[col]
    return g.mean(), g.sem().fillna(0)


def plot_lines(df, anchor, w, out):
    d = df[df.w == w]
    layers, positions = sorted(d.layer.unique()), sorted(d.position.unique())
    for x, hue, hues, cmap, fname in [("position", "layer", layers, "viridis", "vs_position"),
                                      ("layer", "position", positions, "rocket", "vs_layer")]:
        colors = dict(zip(hues, sns.color_palette(cmap, len(hues))))
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
        for ax, (_, short) in zip(axes, ARMS):
            col = f"{short}_balanced_accuracy"
            for h in hues:
                mu, se = mean_se(d[d[hue] == h], x, col)
                ax.plot(mu.index, mu.values, "-o", color=colors[h], label=f"{hue} {h}")
                ax.fill_between(mu.index, mu - se, mu + se, color=colors[h], alpha=0.15, lw=0)
            if x == "layer" and anchor is not None and len(anchor):
                a = anchor[anchor.w == w]
                if len(a):
                    ax.errorbar([9], [a[col].mean()], yerr=[a[col].sem() if len(a) > 1 else 0],
                                fmt="D", mfc="white", color=INK, ms=8, capsize=3,
                                label="layer 9 ref (pos -1)")
            chance_line(ax)
            ax.set_xlabel(x); ax.set_title(short); ax.set_ylim(0.4, 1.03)
            ax.set_xticks(mu.index if x == "position" else layers)
            style_axes(ax)
        axes[0].set_ylabel("balanced accuracy")
        axes[1].legend(loc="lower left", fontsize=SMALL - 2)
        fig_title(fig, f"Balanced accuracy vs {x}, mean ± SE over features (gate off, w={w:g})")
        fig.tight_layout(); save(fig, out / f"{fname}_bacc_w{w:g}")


def feat_label(f, concepts, rank):
    c = concepts.get(f, "")
    c = (c[:28] + "...") if len(c) > 31 else c
    return f"{rank} {f}\n{c}" if c else f"{rank} {f}"


def plot_base_rate(df, concepts, w, out):
    d = df[df.w == w].copy()
    if "endogenous_base_rate_positive" not in d:
        return
    d["label"] = [feat_label(f, concepts, r) for f, r in zip(d.feature, d["rank"].fillna(""))]
    d = d.sort_values(["layer", "rank"])
    piv = d.pivot_table(index="label", columns="position", values="endogenous_base_rate_positive",
                        aggfunc="mean", sort=False) * 100
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(piv) + 1.5))
    sns.heatmap(piv, ax=ax, cmap="Blues", annot=True, fmt=".2f", cbar=False,
                annot_kws={"fontsize": SMALL - 2}, linewidths=0.5, linecolor="white")
    ax.set_xlabel("token position"); ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=SMALL - 3)
    box(ax)
    fig_title(fig, "Natural active rate on the test split (%)")
    fig.tight_layout(); save(fig, out / f"base_rate_w{w:g}")


def plot_confusion(df, concepts, w, out):
    keys = [f"endogenous_{t}_pred_{p}" for t in ("act", "ina") for p in ("active", "inactive", "neither")]
    d = df[(df.w == w)].dropna(subset=[k for k in keys if k in df]) if all(k in df for k in keys) else None
    if d is None or not len(d):
        print(f"no per-class counts at w={w:g}; skipping confusion matrices")
        return
    for L in sorted(d.layer.unique()):
        dl = d[d.layer == L].sort_values(["rank", "position"])
        feats, positions = list(dict.fromkeys(dl.feature)), sorted(dl.position.unique())
        show_n = bool(dl[[keys[2], keys[5]]].to_numpy().any())
        k = 3 if show_n else 2
        fig, axes = plt.subplots(len(feats), len(positions), squeeze=False,
                                 figsize=((2.9 if show_n else 2.3) * len(positions), 2.6 * len(feats)))
        for i, f in enumerate(feats):
            for j, p in enumerate(positions):
                ax = axes[i, j]
                r = dl[(dl.feature == f) & (dl.position == p)]
                if not len(r):
                    ax.axis("off"); continue
                M = r[keys].to_numpy(float).reshape(2, 3)[:, :k]
                Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
                annot = np.array([[f"{int(M[a, b]):,}\n{Mn[a, b] * 100:.1f}%" for b in range(k)]
                                  for a in range(2)])
                sns.heatmap(Mn, ax=ax, annot=annot, fmt="", vmin=0, vmax=1, cmap="Blues", cbar=False,
                            xticklabels=["active", "inactive", "neither"][:k] if i == len(feats) - 1 else False,
                            yticklabels=["active", "inactive"] if j == 0 else False,
                            annot_kws={"fontsize": SMALL - 4}, linewidths=0.5, linecolor="white")
                for t, v in zip(ax.texts, Mn.flatten()):
                    t.set_color("white" if v > 0.55 else INK)
                if i == 0:
                    ax.set_title(f"pos {p}")
                if j == 0:
                    ax.set_ylabel(feat_label(f, concepts, r["rank"].iloc[0] or ""), fontsize=SMALL - 3)
                box(ax)
        fig_title(fig, f"Layer {L}: endogenous confusion (rows true, cols predicted; w={w:g})")
        fig.tight_layout(); save(fig, out / f"confusion_L{L}_w{w:g}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(ROOT / "outputs/runs/sae_layerpos_joint"))
    p.add_argument("--anchor", default=str(ROOT / "outputs/runs/sae_l9_joint"),
                   help="Layer-9 POS -1 runs drawn as reference points ('' to skip).")
    p.add_argument("--features-file", default=str(ROOT / "scripts/sae/sweep_features.txt"))
    p.add_argument("--out-dir", type=Path, default=ROOT / "figures/sae_sweep")
    args = p.parse_args()

    df = load_runs(args.runs, "sweep")
    if df.empty:
        raise SystemExit(f"no sae_joint_train_metrics.json under {args.runs}")
    anchor = load_runs(args.anchor, "anchor") if args.anchor else None
    ranks = load_ranks(args.features_file)
    df["rank"] = df.feature.map(ranks)
    concepts = load_concepts(sorted(set(df.layer)))
    df["concept"] = df.feature.map(concepts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat([df, anchor], ignore_index=True) if anchor is not None else df
    summary.sort_values(["source", "rank", "layer", "position", "w"]).to_csv(
        args.out_dir / "sweep_summary.csv", index=False)
    print("wrote", args.out_dir / "sweep_summary.csv", f"({len(df)} sweep runs)")

    for w in sorted(df.w.unique(), reverse=True):
        for metric in METRICS:
            plot_heatmaps(df, w, metric, args.out_dir)
        plot_lines(df, anchor, w, args.out_dir)
        plot_base_rate(df, concepts, w, args.out_dir)
        plot_confusion(df, concepts, w, args.out_dir)


if __name__ == "__main__":
    main()
