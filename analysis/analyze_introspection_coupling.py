#!/usr/bin/env python3
"""
Introspection coupling: does the readout track the model's OWN live activation, or a
memorised context-to-value map?

Under gate-off training the adapter changes the neuron it is reporting on, so the
neuron's live value drifts away from the cached target the readout was trained on. That
drift is the natural experiment: a readout that has coupled to the live state follows
the neuron to where it moved; one that memorised the target keeps predicting the
cached value. activation_drift.json records both:

  corr_pred_cached       readout vs the cached (training) target
  corr_pred_tuned_live   readout vs the neuron's live value in the tuned model
  coupling_gap           tuned_live - cached.  > 0: follows the neuron.  < 0: follows
                         the target it was trained on.
  drift_vs_cached_corr   how far the neuron itself moved (1 = did not move)

The gap is only informative where the neuron actually drifted -- with drift corr 0.99
the two references coincide and the gap is noise around zero. So the gap is always
read AGAINST drift, never alone.

Interventional corr is the causal version of the same question (an injected value can
only be reported by reading it live) and is carried alongside as the other coupling
measure.

Arms compared: endogenous-only SFT (w=1) and the joint loss (w=0.01, both KL regimes),
on the same (layer, position, neuron) cells. The question the paper asks is whether
adding the interventional term changes coupling -- either way.

Sources: run directories (full readout block) or, failing that, the paired CSVs
compare_sft_vs_joint.py writes (gap and drift only; the scatter is then skipped).
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parents[1]))
from analysis.run_catalog import grid_runs, ROOT  # noqa: E402

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
                     "legend.fontsize": 11})

ARMS = {                      # label -> (group, w, kl_tag)
    "sft":          ("mlp_endo_sweep_layer_token", 1.0,  None),
    "joint_kl0.1":  ("mlp_joint_grid_layer_token", 0.01, "0.1"),
    "joint_free":   ("mlp_joint_grid_layer_token", 0.01, "none"),
}
NEEDED = ["coupling_gap", "drift_vs_cached_corr", "interventional_corr", "endogenous_corr"]
# figure names for the arm labels (the labels stay the keys in every CSV)
ARM_TITLE = {"sft": "SFT", "joint_kl0.1": "Joint Loss (KL=0.1)", "joint_free": "Joint Loss (KL=None)"}
BASELINE = "sft"      # paired deltas are every other arm minus this one

# --preset: which arms to compare. Each preset = (arms, titles, baseline arm, out subdir).
# weight_sweep: the 3 cells x 5 neurons of mlp_joint_weight_sweep at the extreme weights,
# so the drift caused by the INTERVENTIONAL loss alone (w=0) can be read against the
# endogenous-only model (w=1) on the same neurons. kl_tag is not filtered: the anchor
# budget is per cell (0.1 or 3.0) and identical across weights within a cell.
LABEL_ARM = {"weight_sweep": "w0"}   # readout scatter: label this arm's points by neuron
ANNOTATE = None
MAX_SHAPED_CELLS = 6                  # marker shape per (layer, position) cell up to this many
PRESETS = {
    "sft_vs_joint": (dict(ARMS), dict(ARM_TITLE), "sft", "introspection_coupling"),
    "weight_sweep": ({"w1":    ("mlp_joint_weight_sweep", 1.0,  None),
                      "w0.01": ("mlp_joint_weight_sweep", 0.01, None),
                      "w0":    ("mlp_joint_weight_sweep", 0.0,  None)},
                     {"w1": "Endogenous Only (w=1)", "w0.01": "Joint Loss (w=0.01)",
                      "w0": "Interventional Only (w=0)"},
                     "w1", "mlp_joint_weight_sweep/introspection_coupling"),
}
SEQ_LEN = 32      # heatmap columns are drawn as 0-indexed absolute positions (-1 -> 31)
FS = dict(title=22, label=20, ticks=16, legend=15, annot=12)


def _arm(a):
    return ARM_TITLE.get(a, a)


def load_arms() -> pd.DataFrame:
    frames = []
    for arm, (group, w, kl) in ARMS.items():
        df = grid_runs(group, gate="off")
        if df.empty:
            continue
        df = df[df["w"] == w]
        if kl is not None:
            df = df[df["kl_tag"] == kl]
        if df.empty:
            continue
        df = df[df["has_drift"]].assign(arm=arm)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_from_paired_csvs() -> pd.DataFrame:
    """Fallback: compare_sft_vs_joint's paired frames carry gap and drift per arm."""
    pdir = ROOT / "outputs" / "plots" / "sft_vs_joint"
    rows = []
    for f in sorted(pdir.glob("paired_runs_*_vs_sft_gateoff.csv")):
        df = pd.read_csv(f)
        # the arm label is everything after the metric prefix: "joint_free", "sft_gateoff"
        arms = sorted({c[len("coupling_gap_"):] for c in df.columns
                       if c.startswith("coupling_gap_")})
        for arm in arms:
            sub = pd.DataFrame({
                "arm": arm.replace("_gateoff", ""),
                "layer": df["layer"], "position": df["position"], "neuron": df["neuron"],
                "coupling_gap": df.get(f"coupling_gap_{arm}"),
                # compare_sft_vs_joint stores drift_tuned_vs_BASE; base-live and cached agree
                # to alignment corr ~0.998, so it stands in for vs-cached here
                "drift_vs_cached_corr": df.get(f"drift_corr_{arm}"),
                "interventional_corr": df.get(f"interventional_corr_{arm}"),
                "endogenous_corr": df.get(f"endogenous_corr_{arm}"),
            })
            rows.append(sub)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True).drop_duplicates(["arm", "layer", "position", "neuron"])
    return out.dropna(subset=["coupling_gap"])


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, layer, position): mean/std over neurons of every coupling number."""
    cols = [c for c in NEEDED + ["corr_pred_cached", "corr_pred_tuned_live", "corr_pred_base_live"]
            if c in df.columns]
    g = df.groupby(["arm", "layer", "position"])[cols]
    m = g.mean().add_suffix("_mean")
    s = g.std().add_suffix("_std")
    n = g.size().rename("n_neurons")
    return pd.concat([m, s, n], axis=1).reset_index()


def plot_gap_vs_drift(df: pd.DataFrame, out_dir: Path):
    """THE plot. x = how far the neuron moved, y = which reference the readout follows.
    Coupled readouts stay near y=0 or go positive as x falls; memorising readouts go
    negative as the neuron leaves the target behind."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    arms = [a for a in ARMS if a in set(df["arm"])] or sorted(df["arm"].unique())
    pal = dict(zip(arms, sns.color_palette("Set1", len(arms))))
    for arm in arms:
        q = df[df["arm"] == arm]
        ax.scatter(q["drift_vs_cached_corr"], q["coupling_gap"], s=28, alpha=0.65,
                   color=pal[arm], label=f"{_arm(arm)} (n={len(q)})", edgecolors="none")
    ax.axhline(0, color="k", lw=0.9, alpha=0.5)
    ax.set_xlabel("Neuron Drift: Corr(Live Tuned, Cached)", fontsize=FS["label"])
    ax.set_ylabel("Coupling Gap", fontsize=FS["label"])
    ax.tick_params(labelsize=FS["ticks"])
    ax.invert_xaxis()
    ax.legend(fontsize=FS["legend"])
    fig.tight_layout()
    fig.savefig(out_dir / "coupling_gap_vs_drift.png", dpi=150)
    plt.close(fig)


def plot_readout_scatter(df: pd.DataFrame, out_dir: Path):
    """corr(pred, cached) against corr(pred, live). On the diagonal the readout tracks
    the live neuron exactly as well as the target it was trained on."""
    if "corr_pred_tuned_live" not in df.columns or df["corr_pred_tuned_live"].isna().all():
        print("[ic] readout scatter skipped: full readout block not available (needs run dirs)")
        return
    arms = [a for a in ARMS if a in set(df["arm"])] or sorted(df["arm"].unique())
    fig, axes = plt.subplots(1, len(arms), figsize=(6 * len(arms), 5.6), sharey=True, squeeze=False)
    # few cells (the weight sweep has 3): one marker shape per (layer, position) cell in
    # every panel, so a locale can be followed across arms; the grid's 64 cells stay circles
    cells = sorted({(int(l), int(p)) for l, p in zip(df["layer"], df["position"])})
    shaped = len(cells) <= MAX_SHAPED_CELLS
    markers = dict(zip(cells, ["o", "s", "^", "D", "v", "P"])) if shaped else {}
    for ax, arm in zip(axes[0], arms):
        q = df[df["arm"] == arm]
        groups = q.groupby(["layer", "position"]) if shaped else [(None, q)]
        for cell, g in groups:
            sc = ax.scatter(g["corr_pred_cached"], g["corr_pred_tuned_live"],
                            c=g["drift_vs_cached_corr"], cmap="viridis", vmin=0.5, vmax=1.0,
                            s=90 if shaped else 30, marker=markers.get(cell, "o"),
                            edgecolors="k", linewidths=0.6 if shaped else 0.4)
        if arm == ANNOTATE:
            # neuron index next to each point, to see whether a locale is consistently in a
            # corner (the marker shape says which cell it is); labels that would collide
            # are pushed down and get a leader line back to their point
            _label_points(ax, q["corr_pred_cached"], q["corr_pred_tuned_live"],
                          [str(n).split(".")[-1] for n in q["neuron"]])
        ax.plot([-0.2, 1], [-0.2, 1], "k--", lw=0.9, alpha=0.6)
        ax.set_xlim(-0.2, 1.02); ax.set_ylim(-0.2, 1.02)
        ax.set_title(_arm(arm), fontsize=FS["title"])
        ax.tick_params(labelsize=FS["ticks"])
    if shaped:
        handles = [plt.Line2D([], [], ls="", marker=markers[c], ms=11, mfc="white", mec="k",
                              label=f"L{c[0]}, t={c[1] + SEQ_LEN if c[1] < 0 else c[1]}")
                   for c in cells]
        axes[0][0].legend(handles=handles, title="Locale", fontsize=FS["legend"],
                          title_fontsize=FS["legend"], loc="lower right")
    axes[0][0].set_ylabel("Corr(Prediction, Live Activation)", fontsize=FS["label"])
    # one shared x-label: per-panel copies are wider than a panel and collide
    fig.supxlabel("Corr(Prediction, Cached Target)", fontsize=FS["label"], y=-0.02)
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Neuron Drift Corr.", fontsize=FS["label"])
    cb.ax.tick_params(labelsize=FS["ticks"])
    fig.savefig(out_dir / "readout_cached_vs_live.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _label_points(ax, xs, ys, labels, w=0.12, h=0.05, dx=0.03, dy=0.015):
    """Greedy non-overlapping labels in data units (w, h ~ a 4-digit label at size 11 on
    these [-0.2, 1] axes): place each label up-right of its point, moving it down by one
    label height until it clears every label already placed."""
    placed = []
    for x, y, t in sorted(zip(xs, ys, labels), key=lambda r: -r[1]):
        lx, ly = x + dx, y + dy
        while any(abs(lx - px) < w and abs(ly - py) < h for px, py in placed):
            ly -= h
        placed.append((lx, ly))
        moved = abs(ly - (y + dy)) > 1e-9
        ax.annotate(t, (x, y), xytext=(lx, ly), textcoords="data", fontsize=11,
                    va="bottom", arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4")
                    if moved else None)


def plot_cell_heatmaps(cells: pd.DataFrame, out_dir: Path, metric: str, vmin=None, vmax=None,
                       cmap="viridis"):
    arms = [a for a in ARMS if a in set(cells["arm"])] or sorted(cells["arm"].unique())
    col = f"{metric}_mean"
    if col not in cells.columns:
        return
    fig, axes = plt.subplots(1, len(arms), figsize=(7.5 * len(arms), 6.6), squeeze=False)
    for i, (ax, arm) in enumerate(zip(axes[0], arms)):
        piv = cells[cells["arm"] == arm].pivot(index="layer", columns="position", values=col)
        piv = piv.rename(columns=lambda c: c + SEQ_LEN if c < 0 else c)   # absolute positions
        sns.heatmap(piv, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, ax=ax,
                    linewidths=0.5, linecolor="white", annot_kws={"size": FS["annot"]},
                    cbar=(ax is axes[0][-1]), yticklabels=(i == 0))
        ax.set_title(_arm(arm), fontsize=FS["title"])
        ax.set_xlabel("Token Position", fontsize=FS["label"])
        ax.set_ylabel("Layer" if i == 0 else "", fontsize=FS["label"])
        ax.tick_params(labelsize=FS["ticks"])
    fig.axes[-1].tick_params(labelsize=FS["ticks"])                        # colorbar
    fig.tight_layout()
    fig.savefig(out_dir / f"heatmap_{metric}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_arm_deltas(df: pd.DataFrame, out_dir: Path):
    """Each arm minus the BASELINE arm on the SAME cell and neuron, for gap, drift and
    both correlations. A paired delta near zero on the gap is the 'this objective does
    not change coupling' result."""
    if BASELINE not in set(df["arm"]):
        return
    key = ["layer", "position", "neuron"]
    base = df[df["arm"] == BASELINE].set_index(key)
    rows = []
    for arm in [a for a in df["arm"].unique() if a != BASELINE]:
        o = df[df["arm"] == arm].set_index(key)
        j = o.join(base, lsuffix="_j", rsuffix="_s", how="inner")
        for metric in ("coupling_gap", "interventional_corr", "endogenous_corr", "drift_vs_cached_corr"):
            a, b = j.get(f"{metric}_j"), j.get(f"{metric}_s")
            if a is None or b is None:
                continue
            d = (a - b).dropna()
            if len(d) == 0:
                continue
            rows.append({"arm": arm, "metric": metric, "n_pairs": len(d),
                         "delta_mean": float(d.mean()), "delta_std": float(d.std()),
                         "delta_sem": float(d.std() / np.sqrt(len(d))),
                         "frac_positive": float((d > 0).mean())})
    if not rows:
        return
    t = pd.DataFrame(rows)
    t.to_csv(out_dir / f"arm_deltas_vs_{BASELINE}.csv", index=False)
    print(f"\n[ic] each arm − {BASELINE}, paired on (layer, position, neuron):")
    print(t.round(3).to_string(index=False))
    return t


def main(out_dir=None, preset="sft_vs_joint"):
    global ARMS, ARM_TITLE, BASELINE, ANNOTATE
    ARMS, ARM_TITLE, BASELINE, subdir = PRESETS[preset]
    ANNOTATE = LABEL_ARM.get(preset)
    out_dir = Path(out_dir or ROOT / "outputs" / "plots" / subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_arms()
    source = "run directories"
    if df.empty:
        df = load_from_paired_csvs()
        source = "paired CSVs (gap and drift only)"
    if df.empty:
        sys.exit("[ic] nothing to analyse: no grid runs with activation_drift.json, and no "
                 "outputs/plots/sft_vs_joint/paired_runs_*.csv")
    print(f"[ic] loaded {len(df)} runs from {source}: "
          + ", ".join(f"{a}={n}" for a, n in df["arm"].value_counts().items()))
    df.to_csv(out_dir / "coupling_runs.csv", index=False)

    cells = cell_table(df)
    cells.to_csv(out_dir / "coupling_by_cell.csv", index=False)
    plot_gap_vs_drift(df, out_dir)
    plot_readout_scatter(df, out_dir)
    plot_cell_heatmaps(cells, out_dir, "coupling_gap", vmin=-0.5, vmax=0.5, cmap="RdBu_r")
    plot_cell_heatmaps(cells, out_dir, "drift_vs_cached_corr", vmin=0, vmax=1)
    if "corr_pred_tuned_live" in cells.columns or "corr_pred_tuned_live_mean" in cells.columns:
        plot_cell_heatmaps(cells, out_dir, "corr_pred_tuned_live", vmin=0, vmax=1)
    plot_arm_deltas(df, out_dir)

    # the one-line summary per arm
    print("\n[ic] per arm, over all cells and neurons:")
    s = (df.groupby("arm")[[c for c in NEEDED if c in df.columns]]
           .agg(["mean", "std"]).round(3))
    print(s.to_string())
    print(f"\n[ic] wrote {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir", nargs="?", default=None,
                    help="Default: outputs/plots/<preset's subdir>.")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="sft_vs_joint",
                    help="sft_vs_joint (default): SFT vs the two joint arms on the layer x "
                         "position grid. weight_sweep: w=1 / 0.01 / 0 on the weight sweep.")
    a = ap.parse_args()
    main(a.out_dir, a.preset)
