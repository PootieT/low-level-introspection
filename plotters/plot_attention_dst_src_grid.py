#!/usr/bin/env python3
"""
Stage 7: attention-weight introspection over a (destination, source) pair grid.

Point it at the PARENT dir of the sweep:
    python plotters/plot_attention_dst_src_grid.py outputs/runs/attn_weight_dst_src_grid_300K

Every pair (dst, src) with src <= dst over ABSOLUTE offsets
    OFFSETS = {0 -1 -3 -7 -11 -15 -19 -23 -27 -31}     (0 = last token, -31 = BOS)
was trained at the same three heads (L2 h20, L7 h9, L11 h25; one per layer band) by
scripts/attention_weight/7-1_train_dst_src_grid_endogenous_300K.sh. The headline number
for a pair is the mean over those three heads.

RUN NAMES KEEP THE CODE'S CONVENTION, NOT THE GRID'S. `endo_P7.9@-4_POS-28_...` is the
pair (dst, src) = (-27, -31): POS is the token selector (-1 = last token, so dst = POS+1)
and @ is the source offset RELATIVE to the destination (src = dst + @). Both conversions
are asserted to land on OFFSETS, because an off-by-one there produces a plausible-looking
cell at the wrong place rather than an error.

THE RUN-NAME PATTERN IS NOT RESTATED. collect() and RUN_RE come from
plot_attention_srctoken_sweep (as summarize_attention_weight_stage2.py does), so the `_kl`
pitfall documented there cannot come back through a third copy of the regex.

(-31,-31) IS NOT A RUN. With the destination at BOS the causal row has length 1 and the
weight is exactly 1.0 everywhere, so it is drawn as a hatched "const" cell.

Outputs (under outputs/plots/<sweep>/):
    dst_src_grid_corr.png                 endogenous Pearson corr, 3-head mean +- SE
    dst_src_grid_spearman.png             endogenous Spearman, same layout
    dst_src_grid_interventional_corr.png  interventional@gaussian corr, same layout
    dst_src_grid_corr_by_layer.png        endogenous corr per head (L2 / L7 / L11): the
                                          mean hides layer structure, this shows it
    dst_src_grid_runs.csv                 one row per run
    dst_src_grid_by_pair.csv              one row per pair: mean / se / n per metric

Cell borders: dashed = at least one head's target is low variance (corr there divides by
~0, see flag_low_variance); red = at least one head FAILED cache alignment (not a result).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# The sibling owns the attention run-name pattern; import rather than restate.
sys.path.insert(0, str(Path(__file__).parent))
from plot_attention_srctoken_sweep import collect  # noqa: E402

# Must match OFFSETS / HEADS in 7-1_train_dst_src_grid_endogenous_300K.sh.
OFFSETS = (0, -1, -3, -7, -11, -15, -19, -23, -27, -31)
BOS_OFFSET = -31
HEADS = ((2, 20), (7, 9), (11, 25))
ARM = "gateoff/none"

EXPECTED_PAIRS = [(d, s) for d in OFFSETS for s in OFFSETS
                  if s <= d and not (d == s == BOS_OFFSET)]
assert len(EXPECTED_PAIRS) == 54, len(EXPECTED_PAIRS)

METRICS = {
    "corr": "endogenous Pearson corr",
    "spearman": "endogenous Spearman corr",
    "interventional_corr": "interventional@gaussian corr",
}

# Axis order, BOS first: rows top->bottom move the destination toward the end of the text,
# columns left->right move the source, so the valid pairs (src <= dst) fill the lower-left
# triangle and the diagonal is the self pairs.
AXIS = tuple(sorted(OFFSETS))   # (-31, ..., 0)

# One-hue sequential ramp (light = low corr, dark = high), dataviz reference palette blue.
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SEQ_CMAP.set_bad("#f0efec")
INK, MUTED = "#1a1a19", "#8a8a85"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", help="parent directory holding the run directories")
    ap.add_argument("--out", default=None,
                    help="output dir (default outputs/plots/<sweep name>)")
    return ap.parse_args()


def load(sweep_dir: Path) -> pd.DataFrame:
    df = collect(sweep_dir)
    off_arm = df[df["arm"] != ARM]
    if len(off_arm):
        print(f"[grid] ignoring {len(off_arm)} runs outside arm {ARM}: "
              f"{sorted(off_arm['arm'].unique())}")
    df = df[df["arm"] == ARM].copy()
    if df.empty:
        raise SystemExit(f"no {ARM} runs under {sweep_dir}")

    # selector -> absolute offset, relative source -> absolute source
    df["dst"] = df["dst_pos"] + 1
    df["src"] = df["dst"] + df["src_offset"]

    bad_pos = df[~df["dst"].isin(OFFSETS) | ~df["src"].isin(OFFSETS) | (df["src"] > df["dst"])]
    assert bad_pos.empty, (
        f"{len(bad_pos)} runs do not decode onto the grid {OFFSETS} with src <= dst, e.g.\n"
        f"{bad_pos[['run', 'dst_pos', 'src_offset', 'dst', 'src']].head().to_string()}")
    heads = set(zip(df["layer"], df["head"]))
    assert heads <= set(HEADS), (
        f"runs at heads {sorted(heads - set(HEADS))} are not the pinned heads {HEADS}; "
        f"this is not a 7-1 sweep, or HEADS drifted from the launch script")
    assert set(df["signal_code"]) == {"P"}, f"signal codes {set(df['signal_code'])}"
    dup = df[df.duplicated(["dst", "src", "layer", "head"], keep=False)]
    assert dup.empty, f"duplicate runs per (pair, head):\n{dup['run'].to_string()}"
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    present = [m for m in METRICS if m in df.columns]
    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        print(f"[grid] *** metric columns absent from every run, not drawn: {missing}")

    rows = []
    for (d, s) in EXPECTED_PAIRS:
        g = df[(df["dst"] == d) & (df["src"] == s)]
        row = {"dst": d, "src": s, "n_runs": len(g)}
        for m in present:
            v = g[m].dropna().to_numpy(dtype=float)
            row[f"{m}_n"] = len(v)
            row[f"{m}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{m}_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
        row["any_low_variance"] = (bool((g["low_variance"] == True).any())  # noqa: E712
                                   if "low_variance" in g else False)
        row["any_alignment_fail"] = (bool((g["alignment_ok"] == False).any())  # noqa: E712
                                     if "alignment_ok" in g else False)
        rows.append(row)
    agg = pd.DataFrame(rows)

    # Loud about holes: a partly finished sweep must not read as a finished one.
    absent = agg[agg["n_runs"] == 0]
    thin = agg[(agg["n_runs"] > 0) & (agg["n_runs"] < len(HEADS))]
    if len(absent):
        print(f"[grid] *** {len(absent)}/{len(agg)} pairs have NO runs: "
              f"{list(zip(absent['dst'], absent['src']))}")
    if len(thin):
        print(f"[grid] *** {len(thin)} pairs have fewer than {len(HEADS)} heads: "
              f"{list(zip(thin['dst'], thin['src'], thin['n_runs']))}")
    fails = agg[agg["any_alignment_fail"]]
    if len(fails):
        print(f"[grid] *** {len(fails)} pairs include a run that FAILED cache alignment: "
              f"{list(zip(fails['dst'], fails['src']))}")
    return agg


def _matrix(values: dict) -> np.ndarray:
    """{(dst, src): value} -> [len(AXIS), len(AXIS)], rows = dst, cols = src, NaN elsewhere."""
    mat = np.full((len(AXIS), len(AXIS)), np.nan)          # [10, 10]
    for (d, s), v in values.items():
        mat[AXIS.index(d), AXIS.index(s)] = v
    return mat


def _text_color(rgba):
    r, g, b = rgba[:3]
    return "white" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5 else INK


def _draw_grid(ax, mean, labels, flags, norm, title):
    """One heatmap panel. `mean`/`labels`/`flags` are dicts keyed by (dst, src)."""
    mat = _matrix(mean)
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=SEQ_CMAP, norm=norm, aspect="equal")
    for (d, s) in EXPECTED_PAIRS + [(BOS_OFFSET, BOS_OFFSET)]:
        r, c = AXIS.index(d), AXIS.index(s)
        if (d, s) == (BOS_OFFSET, BOS_OFFSET):
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, facecolor="none",
                                   edgecolor=MUTED, hatch="///", linewidth=0))
            ax.text(c, r, "const\n=1", ha="center", va="center", fontsize=6, color=MUTED)
            continue
        v = mean.get((d, s), np.nan)
        if np.isnan(v):
            ax.text(c, r, "--", ha="center", va="center", fontsize=7, color=MUTED)
        else:
            ax.text(c, r, labels[(d, s)], ha="center", va="center", fontsize=5.5,
                    color=_text_color(SEQ_CMAP(norm(v))))
        low_var, align_fail = flags.get((d, s), (False, False))
        if align_fail:
            ax.add_patch(Rectangle((c - .45, r - .45), .9, .9, fill=False,
                                   edgecolor="#d03b3b", linewidth=1.5))
        elif low_var:
            ax.add_patch(Rectangle((c - .45, r - .45), .9, .9, fill=False,
                                   edgecolor=INK, linewidth=0.8, linestyle="--"))
    ax.set_xticks(range(len(AXIS)), [str(o) for o in AXIS], fontsize=7)
    ax.set_yticks(range(len(AXIS)), [str(o) for o in AXIS], fontsize=7)
    ax.set_xlabel("source offset (-31 = BOS)", fontsize=8)
    ax.set_ylabel("destination offset (0 = last token)", fontsize=8)
    ax.set_title(title, fontsize=9)
    for side in ax.spines.values():
        side.set_visible(False)
    return im


def _norm(values):
    finite = [v for v in values if np.isfinite(v)]
    lo = min(0.0, min(finite)) if finite else 0.0
    return plt.Normalize(vmin=lo, vmax=1.0)


def plot_mean(agg: pd.DataFrame, metric: str, out_path: Path, sweep_name: str):
    if f"{metric}_mean" not in agg:
        return
    keys = list(zip(agg["dst"], agg["src"]))
    mean = dict(zip(keys, agg[f"{metric}_mean"]))
    labels = {}
    for k, m, se, n in zip(keys, agg[f"{metric}_mean"], agg[f"{metric}_se"], agg[f"{metric}_n"]):
        txt = f"{m:.2f}" + ("" if np.isnan(se) else f"\n±{se:.2f}")
        labels[k] = txt + (f"\nn={n}" if 0 < n < len(HEADS) else "")
    flags = dict(zip(keys, zip(agg["any_low_variance"], agg["any_alignment_fail"])))
    norm = _norm(mean.values())

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = _draw_grid(ax, mean, labels, flags, norm,
                    f"{METRICS[metric]}, mean ± SE over {len(HEADS)} heads "
                    f"(L{HEADS[0][0]}h{HEADS[0][1]}, L{HEADS[1][0]}h{HEADS[1][1]}, "
                    f"L{HEADS[2][0]}h{HEADS[2][1]})\n{sweep_name}")
    fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[grid] wrote {out_path}")


def plot_by_layer(df: pd.DataFrame, metric: str, out_path: Path, sweep_name: str):
    if metric not in df:
        return
    norm = _norm(df[metric].to_numpy(dtype=float))
    fig, axes = plt.subplots(1, len(HEADS), figsize=(5.2 * len(HEADS), 5.0))
    for ax, (layer, head) in zip(axes, HEADS):
        g = df[(df["layer"] == layer) & (df["head"] == head)]
        keys = list(zip(g["dst"], g["src"]))
        mean = dict(zip(keys, g[metric].astype(float)))
        labels = {k: f"{v:.2f}" for k, v in mean.items()}
        low = g["low_variance"] == True if "low_variance" in g else [False] * len(g)  # noqa: E712
        fail = g["alignment_ok"] == False if "alignment_ok" in g else [False] * len(g)  # noqa: E712
        flags = dict(zip(keys, zip(low, fail)))
        im = _draw_grid(ax, mean, labels, flags, norm, f"L{layer} h{head}")
    fig.colorbar(im, ax=list(axes), shrink=0.8).ax.tick_params(labelsize=7)
    fig.suptitle(f"{METRICS[metric]} per head -- {sweep_name}", fontsize=10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid] wrote {out_path}")


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    out = Path(args.out) if args.out else Path("outputs/plots") / sweep_dir.name
    out.mkdir(parents=True, exist_ok=True)

    df = load(sweep_dir)
    print(f"[grid] {len(df)} runs over {df.groupby(['dst', 'src']).ngroups} pairs")
    df.sort_values(["dst", "src", "layer"], ascending=[False, False, True]).to_csv(
        out / "dst_src_grid_runs.csv", index=False)

    agg = aggregate(df)
    agg.to_csv(out / "dst_src_grid_by_pair.csv", index=False)

    plot_mean(agg, "corr", out / "dst_src_grid_corr.png", sweep_dir.name)
    plot_mean(agg, "spearman", out / "dst_src_grid_spearman.png", sweep_dir.name)
    plot_mean(agg, "interventional_corr", out / "dst_src_grid_interventional_corr.png",
              sweep_dir.name)
    plot_by_layer(df, "corr", out / "dst_src_grid_corr_by_layer.png", sweep_dir.name)


if __name__ == "__main__":
    main()
