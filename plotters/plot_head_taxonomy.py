#!/usr/bin/env python3
"""
What kinds of heads move the introspected value, where do they sit, and are they the
same heads for different signals?

Aggregates every run's mechanistic_analysis/trace_mover_head.py outputs across a sweep. Each run's heads are
classified by src/introspector/head_taxonomy.py -- the same function the ablation uses,
so a head's class here is its class there -- and then pooled.

  Q1  WHERE EACH CLASS LIVES
      class composition by rel_depth (= head layer - signal layer), pooled over runs.
      rel_depth, not absolute layer: the grid's eight signal layers are what separate
      "the layer just above the signal" from "layer 1".

  Q2  ARE THEY THE SAME HEADS ACROSS LOCALES
      Jaccard overlap of each class's head set between pairs of runs that differ in
      exactly one coordinate:
        across_index     same (layer, position), different neuron
        across_position  same (layer, neuron), different position -- the grid holds the
                         SAME five neurons at every position of a layer, so this is clean
        across_layer     different signal layer. In the main grid this also changes the
                         neuron (neurons are sampled per layer); the same-neuron version
                         needs the mlp_layergen sweep, which holds one index fixed.
      Heads are compared by ABSOLUTE identity (layer, head): head 31 at layer 1 and head
      31 at layer 5 are unrelated, so there is no rel_depth version of identity.

      Every overlap is reported against a LAYER-MATCHED base rate: the Jaccard two random
      sets would have if each drew the same number of heads PER LAYER as observed, from
      that layer's heads. A uniform null over all heads is the wrong comparison, because
      depth is already known -- the threshold heads all sit one layer above the signal,
      and any 11 of that layer's 32 heads overlap another 11 by ~3.8 heads by chance. A
      uniform null (drawn from ~480) calls that a large excess and mistakes "same LAYER"
      for "same HEADS". The uniform rate is kept as a column, but only the excess over
      the layer-matched rate is evidence that the same heads recur.

  HOW MANY HEADS
      effective head count, (sum|s|)^2 / sum s^2 over the heads' slopes -- a threshold-
      free participation ratio -- separately for the total slope and the slope at t.
      These have very different shapes (at-t transport concentrated in a handful, total
      contribution diffuse), and the contrast is part of the finding.

  WRITER VS MOVER (from mover_transport_summary.csv, where present)
      writer_share      how much of a head's transport rides on the direction training
                        ADDED to the neuron's down_proj column
      z_edit_vs_random  is the head's OV aligned to that trained direction?
      ov_share          OV-mediated fraction of the measured slope at t (rel_depth 1)

  ABLATION (from head_ablation/, where present)
      class-group knockout deltas per evaluation arm, pooled over runs.

Usage
-----
python plotters/plot_head_taxonomy.py --group mlp_joint_grid_layer_token
python plotters/plot_head_taxonomy.py --group mlp_joint_grid_layer_token --kl kl0.1
"""
from __future__ import annotations

import sys
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotters.run_catalog import NEURON_RE, POS_RE, KL_RE  # noqa: E402  (one name parser)
from src.introspector.head_taxonomy import (  # noqa: E402
    classify_heads, HEAD_CLASSES, DEFAULT_ATT_TAU, DEFAULT_STATIC_TAU,
    DEFAULT_FRAC_T_MIN, DEFAULT_T_SHARE_MIN, DEFAULT_TOTAL_SHARE_MIN)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
CLASS_COLORS = {"threshold_grabber": "#c0392b", "threshold_other": "#e59866",
                "graded_transporter": "#8e44ad", "linear_transporter": "#2471a3",
                "re_reader": "#27ae60", "inert": "#bdc3c7"}
ACTIVE_CLASSES = [c for c in HEAD_CLASSES if c != "inert"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", required=True, help="Experiment group under the runs root.")
    p.add_argument("--runs-root", type=Path, default=ROOT / "outputs" / "runs")
    p.add_argument("--kl", type=str, default=None,
                   help="Keep only runs whose name carries this KL tag (e.g. kl0.1).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: outputs/plots/<group>/head_taxonomy")
    p.add_argument("--att-tau", type=float, default=DEFAULT_ATT_TAU)
    p.add_argument("--static-tau", type=float, default=DEFAULT_STATIC_TAU)
    p.add_argument("--frac-t-min", type=float, default=DEFAULT_FRAC_T_MIN)
    p.add_argument("--t-share-min", type=float, default=DEFAULT_T_SHARE_MIN)
    p.add_argument("--total-share-min", type=float, default=DEFAULT_TOTAL_SHARE_MIN)
    return p.parse_args()


# ======================================================================
# Loading
# ======================================================================
def load_runs(args) -> tuple[pd.DataFrame, dict]:
    """Classified heads for every run with a mover trace, plus that run's raw frames."""
    root = args.runs_root / args.group
    if not root.exists():
        raise SystemExit(f"No such group: {root}")
    classified, raw, skipped = [], {}, []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        n, pz = NEURON_RE.search(d.name), POS_RE.search(d.name)
        if not (n and pz):
            continue
        if args.kl:
            m = KL_RE.search(d.name)
            if not m or f"kl{m['kl']}" != args.kl:
                continue
        mt = d / "mover_trace"
        heads_csv = mt / "mover_heads.csv"
        if not heads_csv.exists():
            skipped.append(d.name)
            continue
        heads = pd.read_csv(heads_csv)
        if "regime" not in heads or "attn_t_delta" not in heads:
            # a trace from before the regime / all-head recording: it cannot be
            # classified consistently, and silently mixing it in would bias the pool
            skipped.append(f"{d.name} (pre-taxonomy trace)")
            continue
        prof_p = mt / "mover_attention_profile_all_heads.csv"
        prof = pd.read_csv(prof_p) if prof_p.exists() else None
        c = classify_heads(heads, prof, att_tau=args.att_tau, static_tau=args.static_tau,
                           frac_t_min=args.frac_t_min, t_share_min=args.t_share_min,
                           total_share_min=args.total_share_min)
        loc = {"run": d.name, "signal_layer": int(n["layer"]), "signal_index": int(n["index"]),
               "position": int(pz["pos"])}
        c = c.assign(**loc)
        classified.append(c)
        tg_p = mt / "mover_transport_summary.csv"
        raw[d.name] = {"heads": heads, "loc": loc,
                       "transport": pd.read_csv(tg_p) if tg_p.exists() else None,
                       "ablation": (pd.read_csv(d / "head_ablation" / "head_ablation_groups.csv")
                                    if (d / "head_ablation" / "head_ablation_groups.csv").exists()
                                    else None)}
    print(f"[taxonomy] {len(classified)} run(s) classified from {root}")
    if skipped:
        print(f"[taxonomy] skipped {len(skipped)} without a usable mover trace, e.g. "
              f"{skipped[:3]}")
    if not classified:
        raise SystemExit("Nothing to aggregate.")
    return pd.concat(classified, ignore_index=True), raw


# ======================================================================
# Q1: composition by depth
# ======================================================================
def composition(cls: pd.DataFrame, out: Path):
    """Class counts per rel_depth, pooled over runs, as counts and per-run means."""
    n_runs = cls["run"].nunique()
    ct = (cls.groupby(["rel_depth", "head_class"]).size().unstack(fill_value=0)
          .reindex(columns=list(HEAD_CLASSES), fill_value=0))
    per_run = ct / n_runs
    per_run.to_csv(out / "class_by_rel_depth_per_run.csv")
    print("\n[taxonomy] mean heads per run, by rel_depth x class (inert omitted):")
    print(per_run[ACTIVE_CLASSES].round(2).to_string())

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bottom = np.zeros(len(per_run))
    for c in ACTIVE_CLASSES:
        ax.bar(per_run.index, per_run[c], bottom=bottom, color=CLASS_COLORS[c], label=c)
        bottom += per_run[c].values
    ax.set_xlabel("rel_depth (head layer - signal layer)")
    ax.set_ylabel("heads per run")
    ax.set_title(f"Where each head class sits ({n_runs} runs)")
    ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "class_by_rel_depth.png", dpi=150); plt.close(fig)

    # the same, split by signal layer: is "threshold heads at rel_depth 1" a property of
    # the layer above the signal, or of absolute layer 1?
    if cls["signal_layer"].nunique() > 1:
        thr = cls[cls["head_class"].str.startswith("threshold")]
        tab = (thr.groupby(["signal_layer", "rel_depth"]).size()
               / cls.groupby("signal_layer")["run"].nunique()).unstack(fill_value=0)
        tab.to_csv(out / "threshold_heads_by_signal_layer.csv")
        fig, ax = plt.subplots(figsize=(7, 3.4))
        im = ax.imshow(tab.values, aspect="auto", cmap="Reds", origin="lower")
        ax.set_yticks(range(len(tab.index))); ax.set_yticklabels(tab.index)
        ax.set_xticks(range(len(tab.columns))); ax.set_xticklabels(tab.columns)
        ax.set_xlabel("rel_depth"); ax.set_ylabel("signal layer")
        ax.set_title("threshold heads per run, by signal layer x rel_depth")
        fig.colorbar(im, ax=ax); fig.tight_layout()
        fig.savefig(out / "threshold_heads_by_signal_layer.png", dpi=150); plt.close(fig)


def static_targets(cls: pd.DataFrame, out: Path):
    if "static_target" not in cls:
        return
    tab = (cls[cls["head_class"] != "inert"]
           .groupby(["head_class", "static_target"]).size().unstack(fill_value=0))
    tab.to_csv(out / "static_target_by_class.csv")
    print("\n[taxonomy] endogenous attention target (argmax excl. BOS), by class:")
    print(tab.to_string())


# ======================================================================
# Q2: locale invariance
# ======================================================================
def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else np.nan


def _expected_jaccard(na: int, nb: int, pool: int) -> float:
    """E[Jaccard] for two uniform random subsets of sizes na, nb from `pool` items."""
    if pool <= 0 or (na + nb) == 0:
        return np.nan
    inter = na * nb / pool
    return inter / (na + nb - inter)


def _expected_jaccard_layer_matched(sa: set, sb: set, n_heads: int) -> float:
    """E[Jaccard] when each set keeps its per-layer COUNTS but its heads within a layer
    are random: expected intersection is sum over layers of a_l * b_l / n_heads."""
    layers = {L for L, _ in sa} | {L for L, _ in sb}
    inter = sum(sum(1 for x in sa if x[0] == L) * sum(1 for x in sb if x[0] == L) / n_heads
                for L in layers)
    denom = len(sa) + len(sb) - inter
    return inter / denom if denom > 0 else np.nan


def overlaps(cls: pd.DataFrame, n_heads: int, n_layers: int, out: Path):
    runs = cls.drop_duplicates("run").set_index("run")[["signal_layer", "signal_index",
                                                        "position"]]
    sets = {(r, c): {(int(L), int(h)) for L, h in zip(g["layer"], g["head"])}
            for (r, c), g in cls.groupby(["run", "head_class"])}
    rows = []
    for ra, rb in itertools.combinations(runs.index, 2):
        a, b = runs.loc[ra], runs.loc[rb]
        same = (a == b)
        if same["signal_layer"] and same["position"] and not same["signal_index"]:
            axis = "across_index"
        elif same["signal_layer"] and same["signal_index"] and not same["position"]:
            axis = "across_position"
        elif not same["signal_layer"]:
            axis = "across_layer"
        else:
            continue
        # only heads that could have appeared in BOTH runs: layers above both signals
        floor = max(int(a["signal_layer"]), int(b["signal_layer"]))
        pool = (n_layers - 1 - floor) * n_heads
        for c in ACTIVE_CLASSES:
            sa = {x for x in sets.get((ra, c), set()) if x[0] > floor}
            sb = {x for x in sets.get((rb, c), set()) if x[0] > floor}
            if not sa and not sb:
                continue
            rows.append({"axis": axis, "head_class": c, "run_a": ra, "run_b": rb,
                         "n_a": len(sa), "n_b": len(sb), "pool": pool,
                         "jaccard": _jaccard(sa, sb),
                         "expected": _expected_jaccard_layer_matched(sa, sb, n_heads),
                         "expected_uniform": _expected_jaccard(len(sa), len(sb), pool)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "overlap_pairs.csv", index=False)
    if df.empty:
        print("\n[taxonomy] no run pairs differing in exactly one coordinate yet")
        return df
    summ = (df.groupby(["axis", "head_class"])
            .agg(n_pairs=("jaccard", "size"), jaccard=("jaccard", "mean"),
                 expected=("expected", "mean"),
                 expected_uniform=("expected_uniform", "mean"))
            .assign(excess=lambda t: t["jaccard"] - t["expected"]).reset_index())
    summ.to_csv(out / "overlap_summary.csv", index=False)
    print("\n[taxonomy] head-set overlap vs LAYER-MATCHED base rate "
          "(Jaccard; excess = observed - layer-matched random):")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    axes_present = [a for a in ("across_index", "across_position", "across_layer")
                    if a in set(summ["axis"])]
    fig, axs = plt.subplots(1, len(axes_present), figsize=(1 + 3.3 * len(axes_present), 3.3),
                            squeeze=False, sharey=True)
    for ax, axis in zip(axs.flat, axes_present):
        s = summ[summ["axis"] == axis].set_index("head_class").reindex(ACTIVE_CLASSES).dropna(
            subset=["jaccard"])
        x = np.arange(len(s))
        ax.bar(x - 0.2, s["jaccard"], 0.4, label="observed",
               color=[CLASS_COLORS[c] for c in s.index])
        ax.bar(x + 0.2, s["expected"], 0.4, label="layer-matched random", color="#999999")
        ax.set_xticks(x); ax.set_xticklabels(s.index, rotation=45, ha="right", fontsize=8)
        ax.set_title(axis.replace("_", " "), fontsize=10); ax.grid(axis="y", alpha=0.25)
    axs.flat[0].set_ylabel("Jaccard"); axs.flat[0].legend(fontsize=8)
    fig.suptitle("Are they the same heads? (vs random sets of the same size)", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summ


# ======================================================================
# How many heads
# ======================================================================
def effective_counts(raw: dict, out: Path):
    """Participation ratio per run per regime, for the total and the at-t slope."""
    rows = []
    for run, d in raw.items():
        for rg, g in d["heads"].groupby("regime"):
            for col, label in (("mean_abs_slope", "total"), ("mean_abs_slope_at_t", "at_t")):
                s = g[col].abs().values
                pr = (s.sum() ** 2 / (s ** 2).sum()) if (s ** 2).sum() > 0 else np.nan
                rows.append({"run": run, **d["loc"], "regime": rg, "slope": label,
                             "effective_heads": pr})
    df = pd.DataFrame(rows)
    df.to_csv(out / "effective_head_counts.csv", index=False)
    summ = df.groupby(["regime", "slope"])["effective_heads"].agg(["mean", "std", "count"])
    print("\n[taxonomy] effective number of heads (participation ratio):")
    print(summ.to_string(float_format=lambda v: f"{v:.1f}"))
    return df


# ======================================================================
# Writer vs mover, and ablation
# ======================================================================
def writer_mover(cls: pd.DataFrame, raw: dict, out: Path):
    frames = [d["transport"].assign(run=r) for r, d in raw.items()
              if d["transport"] is not None and not d["transport"].empty]
    if not frames:
        return
    tg = pd.concat(frames, ignore_index=True).merge(
        cls[["run", "layer", "head", "head_class"]], on=["run", "layer", "head"], how="left")
    cols = [c for c in ("writer_share", "z_edit_vs_random", "z_base_vs_random", "ov_share")
            if c in tg]
    summ = tg[tg["head_class"] != "inert"].groupby("head_class")[cols].median()
    summ.to_csv(out / "writer_vs_mover_by_class.csv")
    print("\n[taxonomy] writer vs mover, median per class (non-inert heads):")
    print(summ.to_string(float_format=lambda v: f"{v:.3f}"))


def ablation(raw: dict, out: Path):
    frames = [d["ablation"].assign(run=r) for r, d in raw.items() if d["ablation"] is not None]
    if not frames:
        return
    ab = pd.concat(frames, ignore_index=True)
    summ = ab.groupby(["group", "arm"])["delta"].agg(["mean", "std", "count"]).unstack("arm")
    summ.to_csv(out / "ablation_group_deltas.csv")
    print("\n[taxonomy] class-group knockout deltas, pooled over runs:")
    print(ab.pivot_table(index="group", columns="arm", values="delta", aggfunc="mean")
          .to_string(float_format=lambda v: f"{v:+.4f}"))


def main():
    args = parse_args()
    out = args.out_dir or (ROOT / "outputs" / "plots" / args.group / "head_taxonomy")
    out.mkdir(parents=True, exist_ok=True)
    cls, raw = load_runs(args)
    cls.to_csv(out / "head_classes_all_runs.csv", index=False)
    # model geometry, from the traces themselves rather than assumed
    any_heads = next(iter(raw.values()))["heads"]
    n_heads = int(any_heads["head"].max()) + 1
    n_layers = int(any_heads["layer"].max()) + 1

    composition(cls, out)
    static_targets(cls, out)
    overlaps(cls, n_heads, n_layers, out)
    effective_counts(raw, out)
    writer_mover(cls, raw, out)
    ablation(raw, out)
    print(f"\n[taxonomy] wrote {out}")


if __name__ == "__main__":
    main()
