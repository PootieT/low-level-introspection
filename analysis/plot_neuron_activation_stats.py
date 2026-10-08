#!/usr/bin/env python3
"""
Appendix figure: training-data activation statistics of every introspected neuron.

The layer x token-position sweep trains on one dataset per (layer, position) cell, each
holding the activations of that layer's sampled neurons (5 by default). The same
neurons are reused at every position of a layer, so the grid is really
(layers x neurons) measured at each position. Each cell's dataset_signal_stats.json
already stores count / mean / std / min / quartiles / max per neuron and split, so
nothing is recomputed here.

Outputs (in --out-dir):
  neuron_activation_stats.png / .pdf   one panel per layer: x = token position,
                                       y = mean +/- 1 std, one marker per neuron
  neuron_activation_stats.csv          every (layer, neuron, position) row

It also prints a POSITION CHECK: for each neuron, how far its mean moves across
positions relative to its own std. A small spread means position barely changes the
distribution, so a pooled per-neuron table is enough; a large one means the
per-position values matter.

Usage
-----
python analysis/plot_neuron_activation_stats.py
python analysis/plot_neuron_activation_stats.py --model llama-3.2-3b
"""
import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parents[1]
# NL<layer>x<neurons per cell>_pos<position>_ ; multi-layer datasets such as
# NL0-12x4 do not match (the layer field must be a single integer)
CELL_RE = re.compile(r"_NL(?P<layer>\d+)x(?P<k>\d+)_pos(?P<pos>-?\d+)_")
STAT_COLS = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
# a neuron whose mean moves by more than this many of its own stds across positions
# is flagged as position-dependent in the printed check
POSITION_FLAG = 0.25


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="llama-3.2-1b",
                   help="Subdirectory of --data-root holding the model's datasets.")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--glob", default="c4-en_NL*x5_pos*_seq32_300K_*",
                   help="Dataset directory glob selecting the sweep's cell datasets.")
    p.add_argument("--split", default="train", choices=["train", "dev", "test"])
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "outputs/plots/neuron_activation_stats")
    return p.parse_args()


def load_stats(data_dir: Path, pattern: str, split: str) -> pd.DataFrame:
    """One row per (layer, neuron, position) from every matching cell dataset."""
    rows, seen = [], {}
    for d in sorted(data_dir.glob(pattern)):
        m = CELL_RE.search(d.name)
        f = d / "dataset_signal_stats.json"
        if not (m and f.exists()):
            continue
        cell = (int(m["layer"]), int(m["pos"]))
        if cell in seen:
            # two datasets for one cell (e.g. a rebuilt dataset): keep the newer one
            print(f"[stats] WARNING: duplicate cell L{cell[0]} pos{cell[1]}: "
                  f"{seen[cell].name} and {d.name}; keeping the newer")
            if f.stat().st_mtime < (seen[cell] / "dataset_signal_stats.json").stat().st_mtime:
                continue
            rows = [r for r in rows if (r["layer"], r["position"]) != cell]
        seen[cell] = d
        for s in json.loads(f.read_text())[split]:
            rows.append({"layer": int(s["signal_layer"]), "index": int(s["signal_index"]),
                         "position": cell[1], "dataset": d.name,
                         **{c: s.get(c) for c in STAT_COLS}})
    return pd.DataFrame(rows)


def report_coverage(df: pd.DataFrame):
    layers = sorted(int(v) for v in df["layer"].unique())
    positions = sorted(int(v) for v in df["position"].unique())
    have = set(zip(df["layer"], df["position"]))
    missing = [(l, p) for l in layers for p in positions if (l, p) not in have]
    print(f"[stats] {len(have)} cell(s): layers {layers}, positions {positions}, "
          f"{df.groupby('layer')['index'].nunique().to_dict()} neurons per layer")
    if missing:
        print(f"[stats] MISSING {len(missing)} cell(s): "
              + ", ".join(f"L{l}/pos{p}" for l, p in missing))


def position_check(df: pd.DataFrame) -> pd.DataFrame:
    """Per neuron: spread of the mean across positions, in units of its average std."""
    g = df.groupby(["layer", "index"])
    chk = g.agg(n_positions=("position", "nunique"), mean_min=("mean", "min"),
                mean_max=("mean", "max"), std_avg=("std", "mean"),
                std_min=("std", "min"), std_max=("std", "max")).reset_index()
    chk["mean_shift_in_std"] = (chk["mean_max"] - chk["mean_min"]) / chk["std_avg"]
    chk["std_ratio"] = chk["std_max"] / chk["std_min"]
    multi = chk[chk["n_positions"] > 1]
    print(f"\n[stats] POSITION CHECK over {len(multi)} neuron(s) seen at >1 position:")
    if multi.empty:
        print("[stats]   no neuron is seen at more than one position; nothing to compare")
        return chk
    print(f"[stats]   mean shift across positions (in units of the neuron's std): "
          f"median {multi['mean_shift_in_std'].median():.3f}, "
          f"max {multi['mean_shift_in_std'].max():.3f}")
    print(f"[stats]   std ratio across positions (max/min): "
          f"median {multi['std_ratio'].median():.3f}, max {multi['std_ratio'].max():.3f}")
    flagged = multi[multi["mean_shift_in_std"] > POSITION_FLAG]
    if len(flagged):
        print(f"[stats]   {len(flagged)} neuron(s) shift by > {POSITION_FLAG} std:")
        for r in flagged.sort_values("mean_shift_in_std", ascending=False).itertuples():
            print(f"[stats]     N{r.layer}.{r.index}: shift {r.mean_shift_in_std:.3f} std, "
                  f"std ratio {r.std_ratio:.2f}")
    else:
        print(f"[stats]   every neuron moves by <= {POSITION_FLAG} std across positions: "
              f"a pooled per-neuron table describes the data")
    return chk


def plot(df: pd.DataFrame, out_dir: Path):
    layers = sorted(df["layer"].unique())
    positions = sorted(df["position"].unique())
    xpos = {p: k for k, p in enumerate(positions)}   # evenly spaced, labelled by value
    ncol = min(4, len(layers))
    nrow = int(np.ceil(len(layers) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.1 * nrow), squeeze=False)
    for k, (ax, layer) in enumerate(zip(axes.flat, layers)):
        sub = df[df["layer"] == layer]
        neurons = sorted(sub["index"].unique())
        colors = sns.color_palette("tab10", len(neurons))
        # dodge the neurons sideways so their error bars do not sit on top of each other
        offsets = np.linspace(-0.28, 0.28, len(neurons)) if len(neurons) > 1 else [0.0]
        for neuron, color, off in zip(neurons, colors, offsets):
            s = sub[sub["index"] == neuron].sort_values("position")
            ax.errorbar(s["position"].map(xpos) + off, s["mean"], yerr=s["std"],
                        fmt="o", ms=4, capsize=2, lw=1.2, color=color, label=f"{neuron}")
        ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(positions, fontsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_title(f"Layer {layer}", fontsize=14)
        ax.set_xlabel("Token Position", fontsize=12)
        if k % ncol == 0:                               # left column only
            ax.set_ylabel("Activation (Mean ± Std)", fontsize=12)
        ax.grid(alpha=0.25)
        # below the panel, so it never covers the error bars
        ax.legend(title="Neuron Index", fontsize=9, title_fontsize=9, ncol=3,
                  loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False)
    for ax in axes.flat[len(layers):]:
        ax.set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"neuron_activation_stats.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[stats] wrote {path}")
    plt.close(fig)


def main():
    args = parse_args()
    data_dir = args.data_root / args.model
    df = load_stats(data_dir, args.glob, args.split)
    if df.empty:
        raise SystemExit(f"No cell datasets matched {data_dir / args.glob}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_coverage(df)
    df.sort_values(["layer", "index", "position"]).to_csv(
        args.out_dir / "neuron_activation_stats.csv", index=False)
    position_check(df).to_csv(args.out_dir / "neuron_position_check.csv", index=False)
    plot(df, args.out_dir)


if __name__ == "__main__":
    main()
