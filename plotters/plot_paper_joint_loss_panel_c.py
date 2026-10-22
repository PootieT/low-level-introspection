#!/usr/bin/env python3
"""
Panel c of paper_plots/joint_loss.pdf: the paired-diff attention-weight grid of paper experiment 2-2
(scripts/attention_weight/paper/2-2_grid_endogenous_logistic_norm_dev_prob_uniform.sh) at w = 0.3.

Two heatmaps, layer {2 6 10 14} x source offset {-27 -19 -11 -3 -1}. Each cell is the mean +- SE over its
three hand-picked heads of the per-head diff  exp 2-2 (w = 0.3) - exp 1-1  (same layer, head, offset):
    Attn Endogenous Delta Correlation      corr
    Attn Interventional Delta Correlation  empirical_prob injection (interventional_test_metrics_empirical_prob.json)
The numbers are those of notebook/attention_weight/paper/2-2_..._analysis.ipynb section 4-2
(plots/paper/2-2_grid_endogenous_logistic_norm_dev_prob_uniform_paired_diff3_noalign_w0.3.pdf, its 1st and
3rd grids), redrawn at two decimals to match panels a/b/d.

Colors: RdBu_r on +-max|cell mean| over the two grids, the same rule a/b (+-0.82) and d (+-0.5) each use
for their own range. (The notebook's range is shared over 3 metrics x 3 w, so its colors differ.)

joint_loss.pdf is a Google Slides export whose panels are raster images, so the deliverable for the slide
is the standalone PNG. The panel is laid out on joint_loss.pdf's own page coordinates, copying panel d's
geometry shifted up to c's slot, so it lines up with d when dropped in.

Run from the repo root:
    python plotters/plot_paper_joint_loss_panel_c.py
Outputs (paper_plots/):
    joint_loss_c_attn.{png,pdf}   panel c alone, for the slide (no "c" letter; the slide has it)
    joint_loss_with_c.pdf         preview: joint_loss.pdf with panel c stamped over the placeholder
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

# The siblings own the attention run-name pattern and the two-decimal format; import rather than restate.
sys.path.insert(0, str(Path(__file__).parent))
from plot_attention_srctoken_sweep import collect  # noqa: E402
from plot_paper_sft_panel_c import fmt2  # noqa: E402

GROUP_ENDO = "attn_weight_paper_grid_endo_logistic_ndev_300K"            # exp 1-1 (w = 1.0)
GROUP_INTERV = "attn_weight_paper_grid_interv_logistic_ndev_punif_300K"  # exp 2-2 (U(0, 1) injection)
RUNS_ENDO, CONFIGS_ENDO = Path("outputs/runs") / GROUP_ENDO, Path("experiment_configs") / GROUP_ENDO
RUNS_INTERV, CONFIGS_INTERV = Path("outputs/runs") / GROUP_INTERV, Path("experiment_configs") / GROUP_INTERV
EMP_FILE = "interventional_test_metrics_empirical_prob.json"
SRC = Path("paper_plots/joint_loss.pdf")
OUT = Path("paper_plots")

W = 0.3
LAYERS = [2, 6, 10, 14]
OFFSETS_LR = [-27, -19, -11, -3, -1]     # -1 (next to the destination) rightmost, as in the notebook
N_HEADS = 3
ARM = "gateoff/none"
KEYS = ["layer", "head", "src_offset"]
EMP = "interventional_corr_emp"
METRICS = {"corr": "Attn Endogenous\nΔ Correlation",
           EMP: "Attn Interventional\nΔ Correlation"}

# ---- joint_loss.pdf geometry, in points from the page's top-left (read off the export's content stream) ----
PAGE_W, PAGE_H = 1498.0, 436.0
C_TOP = 0.0
D_TOP = 214.94                           # d's image spans y 214.94-435.48; the placeholder ends where it starts
C_BOX = (1010.67, C_TOP, PAGE_W, D_TOP)  # placeholder x 1010.67-1403.86; d's colorbar labels reach x~1496.5
# White cover for the stamped preview: the placeholder (fill, 0.62pt border, "Attention Joint 4X4", its "c").
# Overshoot the page top (clipped anyway); start right of b's colorbar labels (x<=1000.2) and stop before
# d's title (y>=218.3).
COVER = (1009.0, -5.0, PAGE_W + 5.0, 216.0)
# d's image is 2048x921 px at 0.239458 pt/px from (1010.67, 214.94); its heatmaps are px y 191-748,
# x 173-977 and 1005-1811, colorbar x 1839-1888. c's are the same, shifted up by D_TOP - C_TOP.
_PX, _X0 = 0.239458, 1010.67
HEAT_Y = (C_TOP + 191 * _PX, C_TOP + 748 * _PX)
HEAT_X = [(_X0 + 173 * _PX, _X0 + 977 * _PX), (_X0 + 1005 * _PX, _X0 + 1811 * _PX)]
CBAR_X = (_X0 + 1839 * _PX, _X0 + 1888 * _PX)
# "c" at d's offset within its slot (d's baseline-left is (1020.18, 425.8)). The slide's own "c" sits at
# baseline y 197.1, which would collide with c's "14" tick label.
C_LABEL_XY = (1020.18, 425.8 - D_TOP + C_TOP)
C_LABEL_FS = 33.6                        # slide's Arial Bold "c"/"d": 37.5pt bbox / 1.117 em
# DejaVu sizes whose rendered ink widths match panel d's text ("SAE Endogenous", "Token Position", digit
# heights of the ticks / colorbar). The cell text is a notch under d's 10.9 because c has 5 columns in
# the width d gives 4; at 10.9 "-0.25 ±0.11" ran into its neighbors.
FS_TITLE, FS_LABEL, FS_TICK, FS_CELL, FS_CBAR = 17.7, 18.6, 14.9, 10.0, 13.0
# d's title baselines, in points above the heatmap top (px 150 and 68 vs the top at 191). Each line is
# placed on its own baseline: as one multi-line title, "Endogenous"'s descender lifted the left title
# above the right one.
TITLE_BASELINES = (41 * _PX, 123 * _PX)
TEXT_DARK = "#262626"                    # d's (seaborn) dark annotation gray

CMAP = plt.get_cmap("RdBu_r").copy()
CMAP.set_bad("#f0efec")


def emp_corr(runs_dir: Path, run: str) -> float:
    p = runs_dir / run / EMP_FILE
    if not p.exists():
        return np.nan
    payload = json.loads(p.read_text())
    # a file of another distribution under this name would silently change the panel
    assert payload.get("distribution") == "empirical_prob", f"{p}: {payload.get('distribution')!r}"
    return payload["test"]["corr"]


def load(runs_dir: Path, configs_dir: Path, w: float, interv_dist: str | None) -> pd.DataFrame:
    df = collect(runs_dir)
    df = df[df["w"] == w].reset_index(drop=True)
    assert len(df), f"{runs_dir.name}: no runs with w={w}"
    assert (df["arm"] == ARM).all(), f"{runs_dir.name}: arms {df['arm'].unique()}, expected only {ARM}"
    assert (df["dst_pos"] == -1).all(), f"{runs_dir.name}: dst_pos {df['dst_pos'].unique()}, expected -1"
    # Run names carry neither head activation, loss nor injection distribution, so a stray run from
    # another experiment would be indistinguishable here without reading its config.
    for run in df["run"]:
        tc = json.loads((configs_dir / f"{run}.json").read_text())["training_config"]
        got = (tc.get("regression_head_activation"), tc.get("regression_loss"))
        assert got == ("sigmoid", "norm_deviance"), f"{run}: {got}"
        if interv_dist is not None:
            assert tc.get("intervention_signal_distribution") == interv_dist, (
                f"{run}: {tc.get('intervention_signal_distribution')!r} != {interv_dist!r}")
    assert not df.duplicated(KEYS).any(), f"{runs_dir.name}: duplicate runs per (layer, head, offset)"
    df[EMP] = df["run"].apply(lambda r: emp_corr(runs_dir, r))         # [n_runs]
    df["low_var"] = (df["low_variance"] == True) if "low_variance" in df else False  # noqa: E712
    return df


def paired_diffs() -> pd.DataFrame:
    endo = load(RUNS_ENDO, CONFIGS_ENDO, 1.0, None)
    interv = load(RUNS_INTERV, CONFIGS_INTERV, W, "prob_uniform")
    cols = KEYS + list(METRICS) + ["low_var"]
    paired = interv[cols].merge(endo[cols], on=KEYS, how="inner", suffixes=("", "_endo"))
    assert len(paired) == len(interv), (
        f"{len(interv) - len(paired)} exp-2-2 runs without an exp-1-1 partner")
    for m in METRICS:
        paired[f"diff_{m}"] = paired[m] - paired[f"{m}_endo"]           # [n_pairs], NaN if either side is
    paired["low_var"] = paired["low_var"] | paired["low_var_endo"]
    return paired


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
                         "low_var": bool(g["low_var"].any())})
    return pd.DataFrame(rows).set_index(["layer", "src_offset"])


def text_color(rgba):
    r, g, b = rgba[:3]
    return "white" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5 else TEXT_DARK


def page_rect(x0, x1, y0, y1):
    """Page points (top-left origin) -> figure fraction [left, bottom, width, height]."""
    return [x0 / PAGE_W, (PAGE_H - y1) / PAGE_H, (x1 - x0) / PAGE_W, (y1 - y0) / PAGE_H]


def draw(stats: dict[str, pd.DataFrame], norm: Normalize, ticks: np.ndarray, with_c_label: bool):
    fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))
    fig.patch.set_alpha(0)
    fig.add_artist(Rectangle((COVER[0] / PAGE_W, (PAGE_H - COVER[3]) / PAGE_H),
                             (COVER[2] - COVER[0]) / PAGE_W, (COVER[3] - COVER[1]) / PAGE_H,
                             transform=fig.transFigure, facecolor="white", edgecolor="none", zorder=-10))
    for k, ((m, title), (x0, x1)) in enumerate(zip(METRICS.items(), HEAT_X)):
        ax = fig.add_axes(page_rect(x0, x1, *HEAT_Y))
        s = stats[m]
        mat = np.full((len(LAYERS), len(OFFSETS_LR)), np.nan)          # [4, 5], rows = layer, cols = offset
        for (L, o), r in s.iterrows():
            mat[LAYERS.index(L), OFFSETS_LR.index(o)] = r["mean"]
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=CMAP, norm=norm, aspect="auto",
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
                    color=text_color(CMAP(norm(r["mean"]))))
            if r["low_var"]:
                ax.add_patch(Rectangle((j - .47, i - .47), .94, .94, fill=False, edgecolor="black", lw=1, ls="--"))
        ax.set_xticks(range(len(OFFSETS_LR)), [str(o) for o in OFFSETS_LR], fontsize=FS_TICK)
        ax.set_xlabel("Source Offset", fontsize=FS_LABEL, labelpad=1)
        if k == 0:
            ax.set_yticks(range(len(LAYERS)), [str(L) for L in LAYERS], fontsize=FS_TICK,
                          rotation=90, va="center")
            ax.set_ylabel("Layer", fontsize=FS_LABEL, labelpad=1)
        else:
            ax.set_yticks([])
        ax.tick_params(length=4, pad=2)
        for line, dy in zip(reversed(title.split("\n")), TITLE_BASELINES):
            ax.annotate(line, (0.5, 1), xycoords="axes fraction", xytext=(0, dy), textcoords="offset points",
                        ha="center", va="baseline", fontsize=FS_TITLE)
        for side in ax.spines.values():
            side.set_visible(False)
    cax = fig.add_axes(page_rect(*CBAR_X, *HEAT_Y))
    cb = fig.colorbar(im, cax=cax, ticks=ticks)
    cb.ax.tick_params(labelsize=FS_CBAR, length=2, pad=1.5)
    cb.ax.set_yticklabels([f"{t:.1f}".replace("-", "−") for t in ticks])
    cb.outline.set_visible(False)
    if with_c_label:
        fig.text(C_LABEL_XY[0] / PAGE_W, (PAGE_H - C_LABEL_XY[1]) / PAGE_H, "c", fontsize=C_LABEL_FS,
                 fontfamily="Liberation Sans", fontweight="bold", ha="left", va="baseline")
    return fig


def main():
    paired = paired_diffs()
    print(f"[panel c] {len(paired)} paired runs (w={W}), "
          f"{int(paired[f'diff_{EMP}'].notna().sum())} with the empirical_prob eval on both sides")
    stats = {m: cell_stats(paired, f"diff_{m}") for m in METRICS}
    for m, s in stats.items():
        short = s[s["n"] != N_HEADS]
        if len(short):
            print(f"[panel c] WARNING {m}: {len(short)} cells with < {N_HEADS} heads:\n{short}")
        print(f"[panel c] diff {m}\n{s['mean'].unstack('src_offset')[OFFSETS_LR].round(3)}")

    means = np.concatenate([s["mean"].to_numpy() for s in stats.values()])  # [2 * 20]
    assert np.isfinite(means).any(), "every cell is empty"
    vmax = float(np.nanmax(np.abs(means)))
    norm = Normalize(vmin=-vmax, vmax=vmax)
    # 0.4 steps: five labels over the same bar height as d's five (0.2 steps over +-0.5)
    step = 0.4 if vmax >= 0.8 else 0.2
    ticks = np.arange(-np.floor(vmax / step), np.floor(vmax / step) + 1) * step   # [n_ticks]
    print(f"[panel c] color range ±{vmax:.3f}, colorbar ticks {ticks.round(1).tolist()}")

    plt.rcParams["pdf.fonttype"] = 42
    # standalone: crop to c's slot (inches, bottom-left origin)
    crop = Bbox.from_extents(C_BOX[0] / 72, (PAGE_H - C_BOX[3]) / 72, C_BOX[2] / 72, (PAGE_H - C_BOX[1]) / 72)
    fig = draw(stats, norm, ticks, with_c_label=False)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"joint_loss_c_attn.{ext}", dpi=300, bbox_inches=crop, facecolor="white")
    plt.close(fig)

    fig = draw(stats, norm, ticks, with_c_label=True)
    with tempfile.TemporaryDirectory() as tmp:
        overlay = Path(tmp) / "overlay.pdf"
        fig.savefig(overlay, transparent=True)
        subprocess.run(["pdftk", str(SRC), "stamp", str(overlay), "output", str(OUT / "joint_loss_with_c.pdf")],
                       check=True)
    plt.close(fig)
    print(f"[panel c] wrote {OUT}/joint_loss_c_attn.{{png,pdf}} and {OUT}/joint_loss_with_c.pdf")


if __name__ == "__main__":
    main()
