#!/usr/bin/env python3
"""Dataset statistics of the SAE features.

DEFAULT -- the layer x position sweep (4 layers x 4 positions x 3 features per layer,
scripts/sae/sweep_common.sh). Appendix figure in the layout of
analysis/plot_neuron_activation_stats.py: one column per layer, x = token position, one
marker per feature. Two rows, because an SAE feature is sparse and the statistics that
matter for a detection readout are how OFTEN it fires and how STRONGLY:
  row 1  active token rate (%)            -- also the detection task's class balance
  row 2  activation when active, mean ± std over the active tokens
(The per-cache dataset_signal_stats.json is computed over ALL tokens, mostly zeros,
so its mean / std mostly measure sparsity; this script reads the raw activations.)

Cells are found from the sweep's experiment configs (one cache per (layer, position),
shared by its 3 features). Writes to outputs/plots/sae_layerpos_joint/:
  sae_feature_stats.{png,pdf}   the figure (--split, default train)
  sae_feature_stats.csv         every (layer, feature, position, split) row

--layer9 -- the older layer-9 figures for the 5 features at POS -1 (sae_l9_joint):
  outputs/plots/sae_l9_joint/dataset_stats_rates.{png,pdf}, dataset_stats_table.{png,pdf}

  python analysis/plot_sae_dataset_stats.py                 # sweep figure
  python analysis/plot_sae_dataset_stats.py --layer9        # layer-9 figures
"""
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from datasets import load_from_disk                                   # noqa: E402
from src.introspector.config import ExperimentConfig                  # noqa: E402
from src.introspector.dataset_utils import apply_signal_filter        # noqa: E402
from sae_style import (INK, INK2, ORDER, SMALL, SPINE, apply_rc, feature_color,  # noqa: E402
                       feature_label, fig_title, save, style_axes, sweep_feature_label, tint)

SPLITS = ["train", "dev", "test"]
RUN_RE = re.compile(r"saeJoint_S(?P<layer>\d+)\.(?P<index>\d+)_POS(?P<pos>-?\d+)_")
# colour by feature RANK (F1-F3, sweep_features.txt order) so the scheme is the same in
# every layer even though the features themselves differ per layer
RANK_COLORS = sns.color_palette("tab10", 3)


# ======================================================================
# Layer x position sweep
# ======================================================================
def sweep_cells(config_dir: Path):
    """{cache_dir: (layer, position, [feature ids in rank order])}, one per (layer, pos)
    cell. Every run of a cell shares one cache, so the first config of a cell is used."""
    cells = {}
    for p in sorted(config_dir.glob("saeJoint_*.json")):
        m = RUN_RE.search(p.stem)
        if not m:
            continue
        cfg = ExperimentConfig.model_validate_json(p.read_text())
        cache = Path(cfg.dataset_cache_dir)
        cache = cache if cache.is_absolute() else ROOT / cache
        if cache in cells:
            continue
        feats = [f"S{s.layer}.{s.index}" for s in cfg.train_signals]
        cells[cache] = (int(m["layer"]), int(m["pos"]), feats)
    return cells


def feature_stats(cells, splits=SPLITS) -> pd.DataFrame:
    """One row per (layer, feature, position, split): active rate and activation stats."""
    rows = []
    for cache, (layer, pos, feats) in sorted(cells.items(), key=lambda kv: kv[1][:2]):
        rank = {f: k for k, f in enumerate(feats)}
        for split in splits:
            if not (cache / split).exists():
                print(f"[stats] missing {cache / split}; skipping")
                continue
            ds = load_from_disk(str(cache / split))
            d = ds.select_columns(["signal_layer", "signal_index", "activation"]).to_pandas()
            for (L, idx), g in d.groupby(["signal_layer", "signal_index"]):
                f = f"S{int(L)}.{int(idx)}"
                a = g["activation"].astype(float).to_numpy()
                act = a[a > 0]
                rows.append({"layer": layer, "position": pos, "feature": f,
                             "rank": rank.get(f), "split": split, "n": len(a),
                             "n_active": len(act), "active_rate_pct": 100 * len(act) / max(len(a), 1),
                             "active_mean": act.mean() if len(act) else np.nan,
                             "active_std": act.std(ddof=1) if len(act) > 1 else np.nan,
                             "mean_all": a.mean(), "std_all": a.std(ddof=1) if len(a) > 1 else np.nan})
    return pd.DataFrame(rows)


def report_coverage(df):
    layers, positions = sorted(df["layer"].unique()), sorted(df["position"].unique())
    have = set(zip(df["layer"], df["position"]))
    missing = [(l, p) for l in layers for p in positions if (l, p) not in have]
    print(f"[stats] {len(have)} cell(s): layers {[int(v) for v in layers]}, "
          f"positions {[int(v) for v in positions]}")
    if missing:
        print("[stats] MISSING cell(s): " + ", ".join(f"L{l}/pos{p}" for l, p in missing))


def plot_sweep(df: pd.DataFrame, out_dir: Path, split="train"):
    """Rows = statistic, columns = layer; sizes match plot_neuron_activation_stats."""
    d = df[df["split"] == split]
    layers = sorted(d["layer"].unique())
    positions = sorted(d["position"].unique())
    xpos = {p: k for k, p in enumerate(positions)}             # evenly spaced, labelled by value
    rows = [("active_rate_pct", None, "Active Token Rate (%)"),
            ("active_mean", "active_std", "Activation When Active\n(Mean ± Std)")]
    fig, axes = plt.subplots(len(rows), len(layers), figsize=(4.4 * len(layers), 3.9 * len(rows)),
                             squeeze=False)
    offsets = np.linspace(-0.2, 0.2, 3)
    for j, layer in enumerate(layers):
        sub = d[d["layer"] == layer]
        feats = sub.sort_values("rank").drop_duplicates("feature")[["feature", "rank"]]
        for i, (val, err, ylabel) in enumerate(rows):
            ax = axes[i, j]
            for f, r in feats.itertuples(index=False):
                s = sub[sub["feature"] == f].sort_values("position")
                r = int(r) if pd.notna(r) else 0
                ax.errorbar(s["position"].map(xpos) + offsets[r], s[val],
                            yerr=s[err] if err else None, fmt="o", ms=5, capsize=2, lw=1.2,
                            color=RANK_COLORS[r], label=sweep_feature_label(f))
            ax.set_xticks(range(len(positions)))
            ax.set_xticklabels(positions, fontsize=10)
            ax.tick_params(axis="y", labelsize=10)
            ax.set_xlim(-0.5, len(positions) - 0.5)
            ax.grid(alpha=0.25)
            if i == 0:
                ax.set_title(f"Layer {layer}", fontsize=14)
                ax.set_ylim(bottom=0)
            if i == len(rows) - 1:
                ax.set_xlabel("Token Position", fontsize=12)
                # below the bottom panel, so it never covers the error bars
                # one column: 'S10.13054 (Reviews)' x3 is wider than a panel
                ax.legend(title="SAE Feature", fontsize=9, title_fontsize=9, ncol=1,
                          loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"sae_feature_stats.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[stats] wrote {path}")
    plt.close(fig)


def run_sweep(args):
    cells = sweep_cells(args.config_root / args.group)
    if not cells:
        raise SystemExit(f"No saeJoint_*.json configs under {args.config_root / args.group}")
    df = feature_stats(cells)
    if df.empty:
        raise SystemExit("No cached splits found for any cell (build the caches first).")
    out = args.out_dir or ROOT / "outputs/plots" / args.group
    out.mkdir(parents=True, exist_ok=True)
    report_coverage(df)
    df.sort_values(["layer", "rank", "position", "split"]).to_csv(
        out / "sae_feature_stats.csv", index=False)
    print(f"[stats] wrote {out / 'sae_feature_stats.csv'}")
    plot_sweep(df, out, split=args.split)


# ======================================================================
# Layer-9 reference set (the earlier figures, unchanged)
# ======================================================================
SPLIT_STYLE = {"train": (0.0, None), "dev": (0.35, "//"), "test": (0.65, None)}
L9_CFG = ROOT / "experiment_configs/sae_l9_joint/saeJoint_S9.1568_POS-1_gateoff_w09.json"


def run_layer9():
    apply_rc()
    cfg = ExperimentConfig.model_validate_json(L9_CFG.read_text())
    cache = cfg.dataset_cache_dir
    raw = {s: load_from_disk(str(cache / s)) for s in SPLITS}
    counts = {}
    for fi in ORDER:
        counts[fi] = {}
        for s in SPLITS:
            ds = apply_signal_filter(raw[s], f"S9.{fi}", s)
            a = np.array([float(x) for x in ds["activation"]])
            act = int((a > 0).sum())
            counts[fi][s] = (act, len(a) - act, len(a))

    out = ROOT / "outputs/plots/sae_l9_joint"; out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(ORDER)); w = 0.26
    for i, s in enumerate(SPLITS):
        amount, hatch = SPLIT_STYLE[s]
        rates = [100 * counts[f][s][0] / counts[f][s][2] for f in ORDER]
        ax.bar(x + (i - 1) * w, rates, w, color=[tint(feature_color(f), amount) for f in ORDER],
               hatch=hatch, edgecolor="white", lw=0)
        for xi, r in zip(x + (i - 1) * w, rates):
            ax.text(xi, r + 0.02, f"{r:.2f}%", ha="center", va="bottom", fontsize=SMALL, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([feature_label(f) for f in ORDER])
    ax.set_ylabel("active token rate (%)")
    ax.set_ylim(0, 1.15 * max(100 * counts[f][s][0] / counts[f][s][2] for f in ORDER for s in SPLITS))
    style_axes(ax, grid="y")
    ax.legend(handles=[Patch(facecolor=tint(INK2, SPLIT_STYLE[s][0]), hatch=SPLIT_STYLE[s][1],
                             edgecolor="white", label=s) for s in SPLITS],
              ncol=3, loc="upper right")
    fig_title(fig, "SAE feature activation rate by split")
    fig.tight_layout()
    save(fig, out / "dataset_stats_rates")

    fig, ax2 = plt.subplots(figsize=(12, 2.4)); ax2.axis("off")
    col_labels = ["feature"] + [f"{s} (active / inactive)" for s in SPLITS]
    cell_text = [[feature_label(f, " ")] + [f"{counts[f][s][0]:,} / {counts[f][s][1]:,}" for s in SPLITS]
                 for f in ORDER]
    tbl = ax2.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(SMALL); tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(SPINE)
        cell.get_text().set_color(INK)
        if r == 0:
            cell.set_text_props(fontweight="bold"); cell.set_facecolor(tint(INK2, 0.88))
        elif c == 0:
            cell.set_facecolor(tint(feature_color(ORDER[r - 1]), 0.75))
    t = counts[ORDER[0]]
    fig_title(fig, f"Active / inactive token counts (split totals: train {t['train'][2]:,}, "
                   f"dev {t['dev'][2]:,}, test {t['test'][2]:,})")
    fig.tight_layout()
    save(fig, out / "dataset_stats_table")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer9", action="store_true",
                   help="Draw the older layer-9 figures instead of the sweep figure.")
    p.add_argument("--group", default="sae_layerpos_joint",
                   help="Sweep config group (subdirectory of --config-root).")
    p.add_argument("--config-root", type=Path, default=ROOT / "experiment_configs")
    p.add_argument("--split", default="train", choices=SPLITS,
                   help="Split drawn in the figure (the CSV has all three).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: outputs/plots/<group>/")
    args = p.parse_args()
    if args.layer9:
        run_layer9()
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
