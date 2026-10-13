#!/usr/bin/env python3
"""
Panel c of paper_plots/sft.pdf: the attention-weight grid of paper experiment 1-1
(scripts/attention_weight/paper/1-1_grid_endogenous_logistic_norm_dev.sh).

Two heatmaps, layer {2 6 10 14} x source offset {-27 -19 -11 -3 -1}, each cell the mean
+- SE over its three hand-picked heads:
    Attn Endogenous Correlation      corr                       (test 30K)
    Attn Interventional Correlation  empirical_prob injection   (interventional_test_metrics_empirical_prob.json,
                                                                 filled by 2-1-0_reeval_1-1_empirical_prob.sh)
The numbers are those of notebook/attention_weight/paper/1-1_..._analysis.ipynb section 2-2
(plots/paper/1-1_grid_endogenous_logistic_norm_dev_corr3_noalign.pdf), redrawn at two
decimals to match panels a/b/d.

sft.pdf is a Google Slides export whose panels are raster images, so the deliverable for
the slide is the standalone PNG. The panel is laid out on sft.pdf's own page coordinates,
copying panel d's geometry shifted up to c's slot, so it lines up with d when dropped in.

Run from the repo root:
    python analysis/plot_paper_sft_panel_c.py
Outputs (paper_plots/):
    sft_c_attn.{png,pdf}   panel c alone, for the slide (no "c" letter; the slide has it)
    sft_with_c.pdf         preview: sft.pdf with panel c stamped over the placeholder
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

# The sibling owns the attention run-name pattern; import rather than restate.
sys.path.insert(0, str(Path(__file__).parent))
from plot_attention_srctoken_sweep import collect  # noqa: E402

RUNS = Path("outputs/runs/attn_weight_paper_grid_endo_logistic_ndev_300K")
CONFIGS = Path("experiment_configs/attn_weight_paper_grid_endo_logistic_ndev_300K")
EMP_FILE = "interventional_test_metrics_empirical_prob.json"
SFT = Path("paper_plots/sft.pdf")
OUT = Path("paper_plots")

LAYERS = [2, 6, 10, 14]
OFFSETS_LR = [-27, -19, -11, -3, -1]     # -1 (next to the destination) rightmost, as in the notebook
N_HEADS = 3
ARM = "gateoff/none"
EMP = "interventional_corr_emp"
METRICS = {"corr": "Attn Endogenous\nCorrelation",
           EMP: "Attn Interventional\nCorrelation"}

# ---- sft.pdf geometry, in points from the page's top-left (measured off the export) ----
PAGE_W, PAGE_H = 1755.0, 517.0
C_BOX = (1179.0, 0.0, PAGE_W, 259.0)     # placeholder incl. its border; d's title starts at y=262.7
# White cover for the stamped preview. The placeholder's 0.75pt border is stroked ON the page's top
# edge (y -0.375..0.375), so a cover starting exactly at y=0 left a hairline in some viewers.
# Overshoot the page (clipped anyway) and stop short of a/b's colorbar labels (x<=1175).
COVER = (1177.0, -5.0, PAGE_W + 5.0, 260.0)
# Panel d's heatmaps sit at y 312.7-470 with d's slot starting at y=260, so c's are at -260.
HEAT_Y = (52.7, 210.0)
HEAT_X = [(1229.0, 1462.0), (1470.0, 1703.0)]
CBAR_X = (1711.7, 1725.7)
C_LABEL_XY = (1193.3, 240.0)             # baseline-left of the slide's "c" (Arial Bold)
# DejaVu sizes whose rendered widths match panel d's text ("SAE Endogenous", "16", "1.0"). The cell
# text is a notch under d's 13.4 because c has 5 columns in the width d gives 4.
FS_TITLE, FS_TICK, FS_CELL, FS_CBAR = 21.5, 16.5, 12.5, 16.0

CMAP = plt.get_cmap("viridis").copy()
CMAP.set_bad("#f0efec")
NORM = Normalize(vmin=0, vmax=1)         # a/b's range, so c's colors read on the same scale


def load() -> pd.DataFrame:
    df = collect(RUNS)
    assert (df["arm"] == ARM).all(), f"arms {df['arm'].unique()}, expected only {ARM}"
    assert (df["dst_pos"] == -1).all(), f"dst_pos {df['dst_pos'].unique()}, expected -1"
    # Run names carry neither head activation nor loss, so a stray run from experiment 1
    # would be indistinguishable here without reading its config.
    for run in df["run"]:
        tc = json.loads((CONFIGS / f"{run}.json").read_text())["training_config"]
        assert (tc.get("regression_head_activation"), tc.get("regression_loss")) == ("sigmoid", "norm_deviance"), (
            f"{run}: {tc.get('regression_head_activation')}, {tc.get('regression_loss')}")
    assert not df.duplicated(["layer", "head", "src_offset"]).any(), "duplicate runs per (layer, head, offset)"

    def emp_corr(run):
        p = RUNS / run / EMP_FILE
        if not p.exists():
            return np.nan
        payload = json.loads(p.read_text())
        # a file of another distribution under this name would silently change the panel
        assert payload.get("distribution") == "empirical_prob", f"{p}: {payload.get('distribution')!r}"
        return payload["test"]["corr"]

    df[EMP] = df["run"].apply(emp_corr)                                # [n_runs]
    return df


def cell_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per cell mean / SE (std ddof=1 / sqrt n) / n, same as the notebook's cell_stats."""
    rows = []
    for L in LAYERS:
        for o in OFFSETS_LR:
            g = df[(df["layer"] == L) & (df["src_offset"] == o)]
            v = g[metric].dropna().to_numpy(dtype=float)               # [n <= 3]
            rows.append({"layer": L, "src_offset": o, "n": len(v),
                         "mean": v.mean() if len(v) else np.nan,
                         "se": v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                         "low_var": bool((g["low_variance"] == True).any()) if "low_variance" in g else False})  # noqa: E712
    return pd.DataFrame(rows).set_index(["layer", "src_offset"])


def fmt2(x: float) -> str:
    s = f"{x:.2f}"
    return "0.00" if s == "-0.00" else s   # a/b never print a signed zero


def text_color(rgba):
    r, g, b = rgba[:3]
    return "white" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5 else "black"


def page_rect(x0, x1, y0, y1):
    """Page points (top-left origin) -> figure fraction [left, bottom, width, height]."""
    return [x0 / PAGE_W, (PAGE_H - y1) / PAGE_H, (x1 - x0) / PAGE_W, (y1 - y0) / PAGE_H]


def draw(stats: dict[str, pd.DataFrame], with_c_label: bool):
    fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))
    fig.patch.set_alpha(0)
    # Cover the placeholder (light blue box, its border, its "c") in the stamped preview.
    fig.add_artist(Rectangle((COVER[0] / PAGE_W, (PAGE_H - COVER[3]) / PAGE_H),
                             (COVER[2] - COVER[0]) / PAGE_W, (COVER[3] - COVER[1]) / PAGE_H,
                             transform=fig.transFigure, facecolor="white", edgecolor="none", zorder=-10))
    for k, ((m, title), (x0, x1)) in enumerate(zip(METRICS.items(), HEAT_X)):
        ax = fig.add_axes(page_rect(x0, x1, *HEAT_Y))
        s = stats[m]
        mat = np.full((len(LAYERS), len(OFFSETS_LR)), np.nan)          # [4, 5], rows = layer, cols = offset
        for (L, o), r in s.iterrows():
            mat[LAYERS.index(L), OFFSETS_LR.index(o)] = r["mean"]
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=CMAP, norm=NORM, aspect="auto",
                       interpolation="nearest")
        for (L, o), r in s.iterrows():
            i, j = LAYERS.index(L), OFFSETS_LR.index(o)
            if np.isnan(r["mean"]):
                ax.text(j, i, "--", ha="center", va="center", fontsize=FS_CELL, color="#8a8a85")
                continue
            txt = fmt2(r["mean"]) + ("" if np.isnan(r["se"]) else f"\n±{fmt2(r['se'])}")
            if r["n"] != N_HEADS:
                txt += f"\n(n={r['n']})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=FS_CELL,
                    color=text_color(CMAP(NORM(r["mean"]))))
            if r["low_var"]:
                ax.add_patch(Rectangle((j - .47, i - .47), .94, .94, fill=False, edgecolor="black", lw=1, ls="--"))
        ax.set_xticks(range(len(OFFSETS_LR)), [str(o) for o in OFFSETS_LR], fontsize=FS_TICK)
        ax.set_xlabel("Source Offset", fontsize=FS_TITLE, labelpad=2)
        if k == 0:
            ax.set_yticks(range(len(LAYERS)), [str(L) for L in LAYERS], fontsize=FS_TICK,
                          rotation=90, va="center")
            ax.set_ylabel("Layer", fontsize=FS_TITLE, labelpad=2)
        else:
            ax.set_yticks([])
        ax.tick_params(length=4, pad=2)
        ax.set_title(title, fontsize=FS_TITLE, linespacing=1.1, pad=6)
        for side in ax.spines.values():
            side.set_visible(False)
    cax = fig.add_axes(page_rect(*CBAR_X, *HEAT_Y))
    cb = fig.colorbar(im, cax=cax, ticks=np.linspace(0, 1, 6))
    # d's labels start ~2.6pt right of its bar; any more and "1.0" runs off the page edge
    cb.ax.tick_params(labelsize=FS_CBAR, length=2, pad=0.6)
    cb.ax.set_yticklabels([f"{t:.1f}" for t in np.linspace(0, 1, 6)])
    cb.outline.set_visible(False)
    if with_c_label:
        fig.text(C_LABEL_XY[0] / PAGE_W, (PAGE_H - C_LABEL_XY[1]) / PAGE_H, "c", fontsize=41,
                 fontfamily="Liberation Sans", fontweight="bold", ha="left", va="baseline")
    return fig


def main():
    df = load()
    print(f"[panel c] {len(df)} runs, {int(df[EMP].notna().sum())} with the empirical_prob eval")
    stats = {m: cell_stats(df, m) for m in METRICS}
    for m, s in stats.items():
        short = s[s["n"] != N_HEADS]
        if len(short):
            print(f"[panel c] WARNING {m}: {len(short)} cells with < {N_HEADS} heads:\n{short}")
        print(f"[panel c] {m}\n{s['mean'].unstack('src_offset').round(3)}")

    plt.rcParams["pdf.fonttype"] = 42
    # standalone: crop to c's slot (inches, bottom-left origin)
    crop = Bbox.from_extents(C_BOX[0] / 72, (PAGE_H - C_BOX[3]) / 72, C_BOX[2] / 72, PAGE_H / 72)
    fig = draw(stats, with_c_label=False)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"sft_c_attn.{ext}", dpi=300, bbox_inches=crop, facecolor="white")
    plt.close(fig)

    fig = draw(stats, with_c_label=True)
    with tempfile.TemporaryDirectory() as tmp:
        overlay = Path(tmp) / "overlay.pdf"
        fig.savefig(overlay, transparent=True)
        subprocess.run(["pdftk", str(SFT), "stamp", str(overlay), "output", str(OUT / "sft_with_c.pdf")],
                       check=True)
    plt.close(fig)
    print(f"[panel c] wrote {OUT}/sft_c_attn.{{png,pdf}} and {OUT}/sft_with_c.pdf")


if __name__ == "__main__":
    main()
