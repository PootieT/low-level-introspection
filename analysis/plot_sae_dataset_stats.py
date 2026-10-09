#!/usr/bin/env python3
"""Dataset-statistics figures for the layer-9 SAE joint runs: the natural activation
rate per feature and split (bar plot) and the active / inactive token counts (table).
Writes figures/sae/dataset_stats_rates.{png,pdf} and dataset_stats_table.{png,pdf}. Counts are
read straight from the cache (no model needed)."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from datasets import load_from_disk
from src.introspector.config import ExperimentConfig
from src.introspector.dataset_utils import apply_signal_filter
from sae_style import (INK, INK2, ORDER, SMALL, SPINE, apply_rc, feature_color, feature_label,
                       fig_title, save, style_axes, tint)

apply_rc()
SPLITS = ["train", "dev", "test"]
SPLIT_STYLE = {"train": (0.0, None), "dev": (0.35, "//"), "test": (0.65, None)}
CFG = ROOT / "experiment_configs/sae_l9_joint/saeJoint_S9.1568_POS-1_gateoff_w09.json"

cfg = ExperimentConfig.model_validate_json(CFG.read_text())
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

out = ROOT / "figures/sae"; out.mkdir(parents=True, exist_ok=True)

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
