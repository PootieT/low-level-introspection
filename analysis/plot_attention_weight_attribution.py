#!/usr/bin/env python3
"""Draw the attention-weight attribution maps written by
attribute_attention_weight_readout.py.

Point it at that script's output directory:
    python analysis/plot_attention_weight_attribution.py <attribution dir>

FIGURES
  attr_triangles_<method>_<group>_targethead.png   a 4x4 small multiple over the 16
  attr_triangles_<method>_<group>_otherheads.png    layers: one figure for the TARGET
                                        head, one for the mean of the other 31, sharing a
                                        color scale. Axis 0 is the destination token,
                                        axis 1 the source token, both RIGHT-ALIGNED
                                        (0 = the readout token). A star marks the
                                        (destination, source) cell the model was trained
                                        to report.
  attr_layer_profile_<method>.png       total |attribution| per layer, target head against
                                        the other-head mean -- the one-line summary of
                                        "which layers carry the report".
  attr_target_cell_<method>.png         the trained cell's own attribution per layer,
                                        against the mean over that layer's other cells.
                                        THIS is the figure that answers the experiment's
                                        first question: is the reported weight used?
  attr_tokens_<method>.png              per-token attribution at the input embeddings.

TWO THINGS ABOUT THE COLOR, BOTH LOAD-BEARING
  1. Attribution is SIGNED, so the maps use a diverging scale with its neutral midpoint
     pinned to EXACTLY zero (`TwoSlopeNorm`-style symmetric limits). Letting matplotlib
     autoscale a diverging colormap puts the neutral color at the data's midpoint
     instead, so cells that push the prediction up and cells that push it down get
     colored as if they agreed. Symmetric limits are what make "blue = negative" true.
  2. The limits are a robust PERCENTILE, not the max. One saturated cell otherwise
     flattens every other cell to the neutral color and the map reads as empty.

The masked region (source after destination) is not drawn at all. Its attribution is
exactly 0 by construction, and coloring it would show structure where there is no data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Diverging: two poles and a NEUTRAL midpoint, never a hue at the middle and never a
# rainbow. RdBu_r is red-positive / blue-negative through near-white.
DIVERGING = "RdBu_r"
# Sequential: one hue, light to dark, for magnitudes that have no sign.
SEQUENTIAL = "Blues"
ROBUST_PCT = 99.0      # percentile for the symmetric color limit


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("attr_dir", type=Path, help="directory written by the attribution script")
    ap.add_argument("--out", type=Path, default=None, help="default: <attr_dir>")
    ap.add_argument("--max-layers-per-fig", type=int, default=16)
    return ap.parse_args()


def symmetric_limit(a: np.ndarray) -> float:
    """A robust, symmetric color limit so zero sits at the neutral midpoint."""
    v = np.abs(a[np.isfinite(a)])
    if v.size == 0:
        return 1.0
    lim = float(np.percentile(v, ROBUST_PCT))
    return lim if lim > 0 else float(v.max()) or 1.0


def causal_mask(N: int) -> np.ndarray:
    """True where the cell does not exist (source strictly after destination)."""
    return np.triu(np.ones((N, N), dtype=bool), 1)


def target_indices(meta, N):
    """Absolute (destination, source) indices of the cell the model was trained on.

    The stored indices are RIGHT-ALIGNED with 0 = the last token, so absolute is
    rel + (N - 1). Asserted rather than trusted: an off-by-one here moves the star onto a
    neighbouring cell whose value is a perfectly ordinary number, so the figure would
    look right and be wrong.

    The destination is the last token of the INPUT TEXT, not the readout token: an
    earlier writer hardcoded dst_rel = 0 and starred the readout token's attention to
    the prompt token before it, a cell nothing was trained on.
    """
    # Metas from that writer have no absolute pair. Refuse them rather than draw the
    # wrong star again.
    assert "dst_abs" in meta and "src_abs" in meta, (
        "attribution_meta.json has no dst_abs/src_abs: it was written with the old "
        "dst_rel=0 convention, whose star sits on the readout token. Re-run "
        "attribute_attention_weight_readout.py (or rewrite the meta) first.")
    di = meta["dst_rel"] + (N - 1)
    si = meta["src_rel"] + (N - 1)
    # Check the right-aligned pair against the absolute one, and check dst FIRST and on
    # its own: shifting both indices by one keeps (src - dst) equal to src_offset, so an
    # offset-only check passes on a star that sits one cell off the trained one.
    assert di == meta["dst_abs"], (
        f"dst_rel {meta['dst_rel']} + (N-1={N - 1}) = {di} != dst_abs {meta['dst_abs']}: "
        f"the writer and this plotter disagree on where rel 0 is")
    assert si == meta["src_abs"], (
        f"src_rel {meta['src_rel']} + (N-1={N - 1}) = {si} != src_abs {meta['src_abs']}")
    assert di < N - 1, (
        f"dst {di} is the readout token (N-1={N - 1}); the trained destination is in the "
        f"text, before the introspection prompt")
    assert 0 <= si <= di < N, (
        f"target cell out of range or non-causal: dst_rel={meta['dst_rel']} "
        f"src_rel={meta['src_rel']} -> (dst={di}, src={si}) with N={N}")
    assert si - di == meta["src_offset"], (
        f"target cell is {si - di} tokens behind the destination but src_offset is "
        f"{meta['src_offset']}")
    return di, si


def draw_triangles(mean, meta, out_dir, method, group):
    """mean: [n_layers, n_heads, N, N] -> two small-multiple grids of triangular maps.

    ONE FIGURE PER PANEL KIND, each a 4 x 4 grid over the 16 layers -- not a single
    16 x 2 column, which would be four feet tall and unreadable. The two figures share
    one color scale so the target head and the other-head mean stay directly comparable;
    scaling them separately would make a weak target head look as strong as a strong one.
    """
    n_layers, n_heads, N, _ = mean.shape
    target_head = meta["signal_head"]
    others = [h for h in range(n_heads) if h != target_head]

    tgt = mean[:, target_head]                      # [n_layers, N, N]
    oth = mean[:, others].mean(1)                   # [n_layers, N, N]
    mask = causal_mask(N)
    keep = ~np.broadcast_to(mask, tgt.shape)
    lim = symmetric_limit(np.concatenate([tgt[keep], oth[keep]]))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    # right-aligned tick labels: 0 is the readout token, negative going back
    rel = np.arange(N) - (N - 1)
    step = max(1, N // 5)
    ticks = list(range(0, N, step))
    ticklabels = [str(rel[t]) for t in ticks]

    ncol = 4
    nrow = int(np.ceil(n_layers / ncol))
    paths = []
    for data, kind, label in ((tgt, "targethead", f"head {target_head} (the target head)"),
                              (oth, "otherheads", f"mean of the other {len(others)} heads")):
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.2 * nrow),
                                 squeeze=False)
        for L in range(n_layers):
            ax = axes[L // ncol][L % ncol]
            im = ax.imshow(np.ma.array(data[L], mask=mask), cmap=DIVERGING, norm=norm,
                           aspect="equal", interpolation="nearest")
            star = (L == meta["signal_layer"] and kind == "targethead")
            ax.set_title(f"L{L}" + ("  <- target" if star else ""), fontsize=9,
                         fontweight="bold" if star else "normal")
            ax.set_xticks(ticks); ax.set_xticklabels(ticklabels, fontsize=6)
            ax.set_yticks(ticks); ax.set_yticklabels(ticklabels, fontsize=6)
            ax.tick_params(length=2)
            if L // ncol == nrow - 1:
                ax.set_xlabel("source", fontsize=7)
            if L % ncol == 0:
                ax.set_ylabel("destination", fontsize=7)
            if star:
                di, si = target_indices(meta, N)
                ax.plot(si, di, marker="*", ms=13, mfc="none", mec="k", mew=1.5)
        for k in range(n_layers, nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.colorbar(im, ax=axes, shrink=0.55, pad=0.02,
                     label=f"{method} attribution to p (signed)")
        fig.suptitle(f"{label}  |  {group} examples, n={meta['groups'][group]['n']}, "
                     f"N={meta['N']}  (0 = readout token)", fontsize=11)
        p = out_dir / f"attr_triangles_{method}_{group}_{kind}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def draw_layer_profile(mean, meta, out_path, method):
    n_layers, n_heads, N, _ = mean.shape
    target_head = meta["signal_head"]
    others = [h for h in range(n_heads) if h != target_head]
    mask = ~causal_mask(N)
    tgt = np.array([np.abs(mean[L, target_head][mask]).sum() for L in range(n_layers)])
    oth = np.array([np.abs(mean[L, others][:, mask]).sum(1).mean() for L in range(n_layers)])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(n_layers), tgt, "-o", lw=2, ms=6, label=f"head {target_head} (target)")
    ax.plot(range(n_layers), oth, "--s", lw=2, ms=6, label="mean of other heads")
    ax.axvline(meta["signal_layer"], color="k", ls=":", lw=1.2, alpha=0.7)
    ax.annotate(f"target layer {meta['signal_layer']}",
                xy=(meta["signal_layer"], ax.get_ylim()[1]), fontsize=8,
                ha="center", va="bottom")
    ax.set_xlabel("layer"); ax.set_ylabel(f"total |{method} attribution|")
    ax.set_title("where the report is carried")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def draw_target_cell(mean, meta, out_path, method):
    """The trained cell against everything else in its layer.

    The experiment's first question in one figure: if the model really reads the weight
    it reports, the starred cell should stand out from the other cells of the same head.
    """
    n_layers, n_heads, N, _ = mean.shape
    h = meta["signal_head"]
    di, si = target_indices(meta, N)
    mask = ~causal_mask(N)
    cell = np.array([mean[L, h, di, si] for L in range(n_layers)])
    rest = []
    for L in range(n_layers):
        m = mask.copy(); m[di, si] = False
        rest.append(np.abs(mean[L, h][m]).mean())
    rest = np.array(rest)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.bar(range(n_layers), cell, color="#B33A3A", label="the reported cell (signed)")
    ax.plot(range(n_layers), rest, "--s", color="#4A6FA5", lw=2, ms=5,
            label="mean |attribution| of the head's other cells")
    ax.axvline(meta["signal_layer"], color="k", ls=":", lw=1.2, alpha=0.7)
    ax.set_xlabel("layer"); ax.set_ylabel(f"{method} attribution")
    ax.set_title(f"does the prediction use the weight it reports?  "
                 f"(P{meta['signal_layer']}.{h}@{meta['src_offset']})")
    ax.grid(alpha=0.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def draw_tokens(token, meta, out_path, method):
    N = token.shape[0]
    rel = np.arange(N) - (N - 1)
    lim = symmetric_limit(token)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    # keywords, always: TwoSlopeNorm's FIRST positional argument is vcenter, not vmin,
    # so TwoSlopeNorm(-lim, 0, lim) silently means "centre at -lim" and raises only
    # because the three then fail the ascending check.
    colors = plt.get_cmap(DIVERGING)(TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)(token))
    ax.bar(rel, token, color=colors, width=0.85)
    ax.axhline(0, color="k", lw=0.8)
    # the input text is pinned to 32 tokens, so everything at or after -32+... is prompt
    ax.set_xlabel("token position (0 = readout token)")
    ax.set_ylabel(f"{method} attribution")
    ax.set_title(f"per-token attribution at the input embeddings (n={meta['N']} tokens)")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main():
    args = parse_args()
    d = args.attr_dir
    out = args.out or d
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((d / "attribution_meta.json").read_text())
    method = meta["method"]

    print(f"[plot] {d}  method={method}  target cell "
          f"P{meta['signal_layer']}.{meta['signal_head']}@{meta['src_offset']}  "
          f"N={meta['N']}  gate={meta['gate']}")
    if meta.get("pin_template") is None:
        print("[plot] *** WARNING: the template was NOT pinned when this was computed, "
              "so averaged per-token maps are not interpretable position by position.")

    for npz in sorted(d.glob(f"attr_attn_probs_{method}_*.npz")):
        group = npz.stem.rsplit("_", 1)[-1]
        z = np.load(npz)
        mean, token = z["mean"], z["token"]
        print(f"[plot] {group}: mean{tuple(mean.shape)} n={int(z['n'])}")
        draw_triangles(mean, meta, out, method, group)
        if group == "best":
            draw_layer_profile(mean, meta, out / f"attr_layer_profile_{method}.png", method)
            draw_target_cell(mean, meta, out / f"attr_target_cell_{method}.png", method)
            draw_tokens(token, meta, out / f"attr_tokens_{method}.png", method)

    print(f"[plot] wrote {len(list(out.glob('attr_*.png')))} figures under {out}")


if __name__ == "__main__":
    main()
