#!/usr/bin/env python3
"""
Paired comparison of endogenous-only SFT against BOTH joint-training regimes.

    sft          w=1.0, gate off      no interventional loss at all
    joint_kl0.1  w=0.01, KL budget .1 joint training at small behavioural cost
    joint_free   w=0.01, unanchored   joint training with nothing holding the LM

The two joint arms answer different questions, so both belong here: joint_kl0.1 is the
usable operating point, joint_free is the ceiling. If joint_kl0.1 tracks joint_free,
the constraint is free; if it tracks sft instead, the anchor is buying LM preservation
at the cost of the mechanism.

Runs pair EXACTLY on (layer, position, neuron): both sweeps resolve neurons from the
same per-(seed, layer) sampler and train the same first five. Pairing matters --
neuron-to-neuron spread is ~+-0.1, so comparing group means throws away most of the
power. Every number here is a within-neuron delta.

Reads, per run:
  interventional_test_metrics_gaussian.json   causal reading, in-domain
  endogenous_test_metrics_endogenous.json     endogenous readout, in-domain
  ood_generalization.csv                      both metrics per OOD corpus
  activation_drift.json                       drift + readout coupling

The headline is the OOD ladder: in-domain the arms look similar, and the claim is that
SFT's endogenous score falls further under shift because it was reading context rather
than the neuron. A near-flat INTERVENTIONAL line says the transmission channel itself
is indifferent to the text distribution.

MISSING CELLS are reported per arm at load time, as exact (layer, position, neuron)
triples, so unfinished or crashed runs can be resubmitted rather than silently shrinking
an average.
"""
import os
import re
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent))
# the SFT sweep's heatmap style, shared so both scripts' heatmaps are identical
from plot_single_neuron_results import SEQ_LEN, draw_layer_position_heatmap  # noqa: E402

# Poster-scale fonts: kept identical to plot_single_neuron_results.py so the two
# scripts' figures share one look.
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "legend.title_fontsize": 20,
    "figure.titlesize": 28,
})

ROOT = Path(__file__).parents[1]
NEURON_RE = re.compile(r"N(?P<layer>\d+)\.(?P<index>\d+)")
POS_RE = re.compile(r"POS(?P<pos>-?\d+)")
GATE_RE = re.compile(r"gate(?P<gate>on|off)")
W_RE = re.compile(r"_w(?P<w>[0-9.]+)")
KL_RE = re.compile(r"_kl(?P<kl>[0-9.a-z]+)")
KEY = ["layer", "position", "neuron"]

# Fidelity measures carried through the whole comparison. "corr" is PEARSON and keeps
# that name, so every existing figure, CSV column and downstream reader is unchanged;
# the rank measures are added beside it.
#
# The default eval writes them into the metrics JSONs (compute_metrics_from_numericals
# in src/introspector/training/metrics.py); runs evaluated before 2026-10-12 lack them.
# A run without them yields NaN for those columns and is skipped by
# paired_stats/plot_delta_heatmap rather than dropping the run, so a partially-scored
# sweep still produces its Pearson figures.
#
#   spearman     monotone, rank-based: a prediction far outside the activation's range
#                costs its rank, not its magnitude
#   kendall_tau  tau-b, which handles the heavy TIES post-SiLU activations create
#                (15-36% distinct values across these runs). Smaller than r by
#                construction -- rank arms with it, do not read it as a fraction
#   monotone_r2  cross-fit isotonic R^2, on Pearson r^2's scale: where r^2 ~= this, the
#                relationship really is linear and nothing was lost by measuring it so
FIDELITY_METRICS = ("corr", "spearman", "kendall_tau", "monotone_r2")
# what the delta heatmaps are drawn for (one figure per readout per metric)
DELTA_METRICS = ("corr", "spearman")


# The sweeps' grid, and the neuron sampler from src/introspector/signals/resolution.py.
#
# EVERY FIELD IS PER MODEL. The three sweeps differ in all three: the layer list (a
# 16-layer model cannot use layer 28), the position list, and -- the one that bites
# silently -- the intermediate size, which is the pool the neuron indices are SAMPLED
# FROM. Llama-3.1-8B's pool is 14336, so seed 21 draws entirely different indices there
# than at 8192; an expected grid built with the wrong pool names neurons that do not
# exist and reports every real run as missing while missing nothing real.
#
# Verified against the synced configs: these reproduce the actual trained indices
# exactly for all four layers of both the 3B and 8B sweeps.
MODELS = {
    "llama321b": dict(layers=[0, 2, 4, 6, 8, 10, 12, 14],
                      positions=[-1, -4, -8, -12, -16, -20, -24, -28],
                      pool=8192, n_layers=16),
    "llama323b": dict(layers=[6, 12, 18, 24],
                      positions=[-1, -8, -20, -28],
                      pool=8192, n_layers=28),
    "llama318b": dict(layers=[4, 12, 20, 28],
                      positions=[-1, -8, -20, -28],
                      pool=14336, n_layers=32),
    # re-runs at the LR the per-model sweep chose (1e-4). 3B moves to the depths the 1B
    # and 8B grids share (round(depth x 28) at 1/8, 3/8, 5/8, 7/8); 8B keeps its layers
    "llama323b_hyper_tuned": dict(layers=[4, 11, 18, 25],
                                  positions=[-1, -8, -20, -28],
                                  pool=8192, n_layers=28),
    "llama318b_hyper_tuned": dict(layers=[4, 12, 20, 28],
                                  positions=[-1, -8, -20, -28],
                                  pool=14336, n_layers=32),
    "qwen3_06b_hyper_tuned": dict(layers=[4, 11, 18, 25],
                                  positions=[-1, -8, -20, -28],
                                  pool=3072, n_layers=28),
    "qwen3_17b_hyper_tuned": dict(layers=[4, 11, 18, 25],
                                  positions=[-1, -8, -20, -28],
                                  pool=6144, n_layers=28),
    "qwen3_4b_hyper_tuned": dict(layers=[5, 14, 23, 32],
                                 positions=[-1, -8, -20, -28],
                                 pool=9728, n_layers=36),
    "qwen3_8b_hyper_tuned": dict(layers=[5, 14, 23, 32],
                                 positions=[-1, -8, -20, -28],
                                 pool=12288, n_layers=36),
    "qwen3_14b_hyper_tuned": dict(layers=[5, 15, 25, 35],
                                  positions=[-1, -8, -20, -28],
                                  pool=17408, n_layers=40),
}
DEFAULT_MODEL = "llama321b"
N_TRAINED = 5
SAMPLER_SEED = 21

# kept for callers that still read them; the model registry is the source of truth
GRID_LAYERS = MODELS[DEFAULT_MODEL]["layers"]
GRID_POSITIONS = MODELS[DEFAULT_MODEL]["positions"]
INTERMEDIATE_SIZE = MODELS[DEFAULT_MODEL]["pool"]


def model_for_group(group: str) -> str:
    """Which model a run group belongs to, from its name suffix.

    Group names carry the model (mlp_endo_sweep_layer_token_llama318b); anything with no
    suffix is the original 1B sweep. Longest key first, so ..._llama323b_hyper_tuned is
    not read as llama323b (whose grid has different layers).
    """
    for key in sorted(MODELS, key=len, reverse=True):
        if group.endswith(key):
            return key
    return DEFAULT_MODEL


def expected_grid(layers=None, positions=None, n_neurons=None,
                  seed=None, pool=None, model=None):
    """Every (layer, position, neuron) the sweeps should contain, for one MODEL.

    Mirrors resolve_signal_source: a per-(seed, layer) RNG draws the layer's signal
    list, and the scripts train its first n_neurons. Deriving it rather than inferring
    it from what exists is the point -- a cell missing from BOTH arms is invisible to
    any inference from the data.

    `model` selects layers/positions/pool from MODELS; explicit arguments override it.
    """
    import hashlib
    import random as _random
    spec = MODELS[model or DEFAULT_MODEL]
    layers = spec["layers"] if layers is None else layers
    positions = spec["positions"] if positions is None else positions
    pool = spec["pool"] if pool is None else pool
    n_neurons = N_TRAINED if n_neurons is None else n_neurons
    seed = SAMPLER_SEED if seed is None else seed
    out = []
    for layer in layers:
        layer_seed = int(hashlib.sha256(f"{seed}:{layer}".encode()).hexdigest()[:8], 16)
        neurons = _random.Random(layer_seed).sample(range(pool), 10)[:n_neurons]
        for pos in positions:
            for idx in neurons:
                out.append((layer, pos, f"N{layer}.{idx}"))
    return out


def report_missing(df: pd.DataFrame, label: str, expected=None, model=None):
    """Which runs of the grid are absent from this arm, and which present runs are not
    in the grid at all.

    UNEXPECTED matters as much as missing: a sweep whose layer list was edited between
    submissions leaves partially-finished runs at the old layers, and those inflate the
    run count while contributing nothing to the grid. The 8B endo directory has exactly
    that -- layers 6 and 18 from an earlier grid, 4 and 2 neurons deep.
    """
    expected = expected or expected_grid(model=model)
    exp_set = set(expected)
    have = set(zip(df["layer"], df["position"], df["neuron"]))
    missing = [e for e in expected if e not in have]
    unexpected = sorted(have - exp_set)
    print(f"[cmp] {label}: {len(have & exp_set)}/{len(expected)} expected runs present"
          + (f", plus {len(unexpected)} not in the grid" if unexpected else ""))
    if unexpected:
        by_cell = {}
        for layer, pos, neuron in unexpected:
            by_cell.setdefault((layer, pos), []).append(neuron)
        print(f"[cmp] {label}: UNEXPECTED {len(unexpected)} run(s) across {len(by_cell)} cell(s) "
              f"(not in this model's grid -- leftovers from an earlier layer/position list?):")
        for (layer, pos), neurons in sorted(by_cell.items()):
            print(f"        layer {layer:2d} pos {pos:4d}: {sorted(neurons)}")
    if not missing:
        return []
    by_cell = {}
    for layer, pos, neuron in missing:
        by_cell.setdefault((layer, pos), []).append(neuron)
    print(f"[cmp] {label}: MISSING {len(missing)} run(s) across {len(by_cell)} cell(s):")
    for (layer, pos), neurons in sorted(by_cell.items()):
        full = " (whole cell)" if len(neurons) == N_TRAINED else ""
        print(f"        layer {layer:2d} pos {pos:4d}{full}: {sorted(neurons)}")
    return missing


def load_group(exp_group: str, gate="off", w=None, kl=None) -> pd.DataFrame:
    """One row per run, with in-domain metrics and one column per OOD corpus."""
    rows = []
    root = Path(f"{ROOT}/outputs/runs/{exp_group}")
    if not root.exists():
        raise SystemExit(f"No such group: {root}")
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        n, p = NEURON_RE.search(d.name), POS_RE.search(d.name)
        if not (n and p):
            continue
        g, mw, mkl = GATE_RE.search(d.name), W_RE.search(d.name), KL_RE.search(d.name)
        if gate and (not g or g["gate"] != gate):
            continue
        if w is not None and (not mw or float(mw["w"]) != w):
            continue
        if kl is not None and (not mkl or mkl["kl"] != kl):
            continue

        def js(fname, key=None):
            f = d / fname
            if not f.exists():
                return None
            payload = json.loads(f.read_text())
            return payload[key] if key else payload

        interv, endo = (js("interventional_test_metrics_gaussian.json", "test"),
                        js("endogenous_test_metrics_endogenous.json"))
        if interv is None or endo is None:
            continue
        row = {"group": exp_group, "run": d.name,
               "layer": int(n["layer"]), "index": int(n["index"]),
               "neuron": f"N{n['layer']}.{n['index']}", "position": int(p["pos"]),
               "w": float(mw["w"]) if mw else None, "kl": mkl["kl"] if mkl else None}
        for metric in FIDELITY_METRICS:
            # the legacy column names stay exactly as they were for Pearson
            suffix = "corr" if metric == "corr" else metric
            row[f"interventional_{suffix}"] = interv.get(metric)
            row[f"endogenous_{suffix}"] = endo.get(metric)
        # the diagnostic, not a third score: where Pearson and Spearman disagree the
        # cause in this project has been out-of-range predictions, not curvature
        for readout in ("interventional", "endogenous"):
            r, rho = row.get(f"{readout}_corr"), row.get(f"{readout}_spearman")
            row[f"{readout}_corr_minus_spearman"] = (r - rho if r is not None and rho is not None
                                                     else None)
        drift = js("activation_drift.json")
        if drift:
            row["drift_corr"] = drift.get("drift_tuned_vs_base", {}).get("corr")
            row["coupling_gap"] = (drift.get("readout_tracking") or {}).get("coupling_gap")
        ood_path = d / "ood_generalization.csv"
        if ood_path.exists():
            ood = pd.read_csv(ood_path)
            for _, r in ood.iterrows():
                c = r.get("corpus")
                if not isinstance(c, str):
                    continue
                row[f"ood_{c}_interventional"] = r.get("interventional_corr")
                row[f"ood_{c}_endogenous"] = r.get("endogenous_corr")
                row[f"ood_{c}_activation_std"] = r.get("activation_std")
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No runs matched in {exp_group} (gate={gate}, w={w}, kl={kl})")
    print(f"[cmp] {exp_group}: {len(df)} runs, {df['neuron'].nunique()} neurons, "
          f"{df['position'].nunique()} positions, {df['layer'].nunique()} layers")
    return df


def pair(a: pd.DataFrame, b: pd.DataFrame, label_a="A", label_b="B") -> pd.DataFrame:
    """Inner join on (layer, position, neuron). Unmatched runs are reported, since a
    silently dropped arm would bias every average that follows."""
    merged = a.merge(b, on=KEY, suffixes=(f"_{label_a}", f"_{label_b}"))
    only_a, only_b = len(a) - len(merged), len(b) - len(merged)
    print(f"[cmp] paired {len(merged)} run(s); unmatched: {only_a} in {label_a}, "
          f"{only_b} in {label_b}")
    if len(merged) == 0:
        raise SystemExit("No runs paired -- do the two groups share neurons/positions?")
    return merged


def paired_stats(merged, metric, label_a, label_b, min_activation_std=1e-6):
    """Within-neuron delta with a paired SE. Rows whose activation variance is ~0 are
    dropped: correlation is undefined there and would read as a generalization failure."""
    ca, cb = f"{metric}_{label_a}", f"{metric}_{label_b}"
    if ca not in merged or cb not in merged:
        return None
    sub = merged[[*KEY, ca, cb]].dropna()
    std_col = None
    if metric.startswith("ood_"):
        corpus = metric.split("_")[1]
        for cand in (f"ood_{corpus}_activation_std_{label_a}", f"ood_{corpus}_activation_std_{label_b}"):
            if cand in merged:
                std_col = cand
                break
    if std_col is not None:
        keep = merged.loc[sub.index, std_col].fillna(0) > min_activation_std
        dropped = int((~keep).sum())
        if dropped:
            print(f"[cmp] {metric}: dropped {dropped} row(s) with ~zero activation variance")
        sub = sub[keep.values]
    if sub.empty:
        return None
    d = sub[ca] - sub[cb]
    return {"metric": metric, "n": len(sub),
            f"mean_{label_a}": sub[ca].mean(), f"mean_{label_b}": sub[cb].mean(),
            "delta_mean": d.mean(),
            "delta_se": d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan,
            "frac_a_higher": float((d > 0).mean())}


def ood_ladder_table(frames: dict, readout="endogenous", common_only=True):
    """Per (corpus, arm): mean ± SE over neurons of one readout across the corpus ladder,
    or None if there is nothing to plot.

    common_only restricts to runs present in ALL arms, so a line is not lifted by a
    cell another arm never finished. The dropped count is printed."""
    arms = list(frames)
    if common_only and len(arms) > 1:
        keys = set.intersection(*[set(map(tuple, f[KEY].values)) for f in frames.values()])
        frames = {a: f[[tuple(r) in keys for r in f[KEY].values]] for a, f in frames.items()}
        print(f"[cmp] ladder ({readout}): {len(keys)} run(s) common to all {len(arms)} arm(s)")
    corpora = sorted({c.split("_", 1)[1].rsplit("_", 1)[0]
                      for f in frames.values() for c in f.columns
                      if c.startswith("ood_") and c.endswith(f"_{readout}")})
    if not corpora:
        print(f"[cmp] no OOD {readout} columns; skipping ladder")
        return None
    rows = []
    for arm, f in frames.items():
        for corpus in ["in-domain"] + corpora:
            col = f"{readout}_corr" if corpus == "in-domain" else f"ood_{corpus}_{readout}"
            if col not in f:
                continue
            vals = f[col].dropna()
            if len(vals) == 0:
                continue
            rows.append({"corpus": corpus, "arm": arm, "n": len(vals), "mean": vals.mean(),
                         "se": vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan})
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[cmp] ladder ({readout}): no data; skipping")
        return None
    return df


def plot_ood_ladders(frames: dict, out_dir, arm_colors=None, common_only=True):
    """Both readouts across the corpus ladder in ONE figure: endogenous left,
    interventional right, shared y; grouped bars, mean ± SE over runs (as in the CSVs).

    The x-order is set by the endogenous panel -- in-domain pinned leftmost, then OOD
    corpora by decreasing endogenous performance (averaged over arms) -- and reused for
    the interventional panel so the two line up column-for-column."""
    tables = {}
    for readout in ("endogenous", "interventional"):
        t = ood_ladder_table(frames, readout, common_only)
        if t is not None:
            tables[readout] = t
            t.to_csv(Path(out_dir) / f"ood_ladder_{readout}.csv", index=False)
    if not tables:
        return None
    ref = tables.get("endogenous", next(iter(tables.values())))
    order = ["in-domain"] + (ref[ref["corpus"] != "in-domain"].groupby("corpus")["mean"].mean()
                             .sort_values(ascending=False).index.tolist())
    # any corpus present only in the other readout goes last, by its own performance
    for t in tables.values():
        extra = (t[~t["corpus"].isin(order)].groupby("corpus")["mean"].mean()
                 .sort_values(ascending=False).index.tolist())
        order += extra

    arms = list(frames)
    # Grouped bars, mean ± SE over runs (the same numbers ood_ladder_*.csv holds). Sized
    # for a half-width paper column: titles / y-label larger than the generalization
    # figures, unrotated two-line tick labels.
    fig, axes = plt.subplots(1, 2, figsize=(19, 6.5), sharey=True)
    xticklabels = [pretty_corpus(c).replace(" (", "\n(") for c in order]
    width = 0.8 / len(arms)
    for ax, readout in zip(axes, ("endogenous", "interventional")):
        t = tables.get(readout)
        for i, arm in enumerate(arms):
            if t is None:
                continue
            sub = t[t["arm"] == arm].set_index("corpus").reindex(order)
            xs = np.arange(len(order)) + (i - (len(arms) - 1) / 2) * width
            color = arm_colors.get(arm) if arm_colors else None
            ax.bar(xs, sub["mean"], width * 0.92, yerr=sub["se"], capsize=3, color=color,
                   label=pretty_arm(arm), error_kw=dict(lw=1.2))
            print(f"[cmp] ladder ({readout}) {pretty_arm(arm)}: n={int(sub['n'].max())} runs")
        ax.set_title(f"OOD {readout.capitalize()}", fontsize=36)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(xticklabels, fontsize=24)
        ax.tick_params(axis="y", labelsize=26)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Correlation\n(Mean \u00b1 SE)", fontsize=34)   # one line overruns the axis
    # bars fill the panel from 0, so the legend needs headroom: the axis runs to 1.25
    # but is labelled 0-1 only, and the 2 x 2 legend sits above the tallest bars
    axes[0].set_ylim(0, 1.25)
    axes[0].set_yticks(np.arange(0, 1.01, 0.2))
    handles, labels = next((ax.get_legend_handles_labels() for ax in axes
                            if ax.get_legend_handles_labels()[0]), ([], []))
    if handles:
        axes[0].legend(handles, labels, loc="upper center", ncol=2, fontsize=20,
                       frameon=False, handlelength=1.0, columnspacing=0.8,
                       borderaxespad=0.2)
    fig.tight_layout()
    path = Path(out_dir) / "ood_ladder.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[cmp] wrote {path}")
    return order


def pretty_arm(label: str) -> str:
    """Short, poster-friendly arm name, e.g. joint_free -> 'Joint Loss (KL=None)',
    sft_gateoff -> 'SFT (Gate-off)'."""
    if label.startswith("joint"):
        if "free" in label:
            kl = "None"
        else:
            m = KL_RE.search(label)
            kl = m["kl"] if m else "?"
        return f"Joint Loss (KL={kl})"
    if label.startswith("sft"):
        gate = "on" if "gateon" in label else "off" if "gateoff" in label else "?"
        return f"SFT (Gate-{gate})"
    return label


CORPUS_LABELS = {
    "in-domain": "C4 (ID)",
    "wikipedia": "Wiki",
    "shuffled": "C4 (Shuf.)",
    "c4_zh": "C4 (Chinese)",
    "c4_bn": "C4 (Bengali)",
}


def pretty_corpus(corpus: str) -> str:
    """Display label for an OOD corpus key, e.g. 'c4_zh' -> 'C4 (Chinese)'."""
    return CORPUS_LABELS.get(corpus, corpus.replace("_", " ").title())


def pretty_metric(metric: str) -> str:
    """Readable metric name for the colorbar, e.g. interventional_corr ->
    'Interventional Correlation', ood_<corpus>_endogenous -> 'Endogenous Correlation (<corpus>)'."""
    readout_names = {"interventional": "Interventional Correlation",
                     "endogenous": "Endogenous Correlation"}
    metric_words = {"corr": "Pearson r", "spearman": "Spearman rho",
                    "kendall_tau": "Kendall tau-b (smaller than r by construction)",
                    "monotone_r2": "Monotone R2 (isotonic, cross-fit)",
                    "corr_minus_spearman": "Pearson r - Spearman rho"}
    for readout in ("interventional", "endogenous"):
        if metric.startswith(f"{readout}_"):
            tail = metric[len(readout) + 1:]
            if tail == "corr":
                return readout_names[readout]
            if tail in metric_words:
                return f"{readout.capitalize()} {metric_words[tail]}"
    if metric.startswith("ood_"):
        parts = metric.split("_")
        readout, corpus = parts[-1], "_".join(parts[1:-1])
        return f"{readout_names.get(readout, readout.title() + ' Correlation')} ({corpus})"
    return metric.replace("_", " ").title()


def delta_pivots(merged, metric, label_a, label_b):
    """(mean, std) of the within-neuron delta a - b per (layer, position), or None.
    The std is the spread across the cell's neurons: a +0.3 mean built from neurons at
    +0.9 and -0.3 is a different claim from one where every neuron moved +0.3."""
    ca, cb = f"{metric}_{label_a}", f"{metric}_{label_b}"
    if ca not in merged or cb not in merged:
        return None
    sub = merged[[*KEY, ca, cb]].dropna()
    if sub.empty:
        return None
    sub = sub.assign(delta=sub[ca] - sub[cb])
    mean = sub.pivot_table(index="layer", columns="position", values="delta", aggfunc="mean")
    std = sub.pivot_table(index="layer", columns="position", values="delta", aggfunc="std")
    return mean, std


def delta_color_limit(merged, metrics, label_a, label_b) -> float:
    """One symmetric colour limit across all `metrics` of a pair, so the endogenous and
    interventional delta maps share a colorbar and read on the same scale."""
    lims = [float(np.nanmax(np.abs(p[0].to_numpy())))
            for p in (delta_pivots(merged, m, label_a, label_b) for m in metrics) if p]
    lims = [v for v in lims if np.isfinite(v) and v > 0]
    return max(lims) if lims else 1.0


def plot_delta_heatmap(merged, metric, label_a, label_b, out_dir, lim=None):
    """Within-neuron delta by (layer, position) -- where does one arm win?"""
    piv = delta_pivots(merged, metric, label_a, label_b)
    if piv is None:
        return
    mean, std = piv
    if lim is None:
        lim = float(np.nanmax(np.abs(mean.to_numpy()))) or 1.0
    # cells are annotated mean ± std (std over the cell's neurons, aggfunc="std")
    title = f"Neuron \u0394 {pretty_metric(metric)}"   # \u0394 = label_a - label_b, paired within neuron
    path = Path(out_dir) / f"delta_{metric}_{label_a}_vs_{label_b}.png"
    draw_layer_position_heatmap(mean, std, title, path, vmin=-lim, vmax=lim,
                                cmap="RdBu_r", center=0, seq_len=SEQ_LEN)
    print(f"[cmp] wrote {path}")


def plot_arm_heatmap(df, metric, label, out_dir):
    """Absolute per-(layer, position) heatmap of ONE arm's metric (not a difference),
    mean ± std over the cell's neurons -- what the arm scores on its own, so the delta
    maps can be read against the levels they came from."""
    if metric not in df:
        return
    sub = df[[*KEY, metric]].dropna()
    if sub.empty:
        return
    mean = sub.pivot_table(index="layer", columns="position", values=metric, aggfunc="mean")
    std = sub.pivot_table(index="layer", columns="position", values=metric, aggfunc="std")
    # correlations get the SFT maps' fixed [0, 1] viridis scale, which also makes the
    # endogenous and interventional maps share one colorbar
    vmin, vmax = (0, 1) if metric.endswith("corr") else (None, None)
    title = f"Neuron {pretty_metric(metric)}"
    path = Path(out_dir) / f"heatmap_{label}_{metric}.png"
    draw_layer_position_heatmap(mean, std, title, path, vmin=vmin, vmax=vmax,
                                seq_len=SEQ_LEN)
    print(f"[cmp] wrote {path}")


def main(arms: dict, out_dir, pairs=None, model=None):
    """
    arms:  {label: {"group": ..., "filters": {...}}}
    pairs: [(a, b), ...] to compare as a-minus-b. Default: every joint arm against
           every sft arm.
    model: key into MODELS. None infers it per arm from the group name suffix, so a
           mixed call still measures each arm against ITS OWN grid.

    Pairwise rather than one joint table: a join across all arms would drop any cell
    missing from any one of them, which is exactly the cell the missing-run report
    exists to surface.
    """
    os.makedirs(out_dir, exist_ok=True)
    frames, missing, models = {}, {}, {}
    for label, spec in arms.items():
        print(f"\n[cmp] ===== {label} =====")
        try:
            frames[label] = load_group(spec["group"], **spec.get("filters", {}))
        except SystemExit as e:
            print(f"[cmp] {label}: no runs ({e}); skipping this arm")
            continue
        m = model or model_for_group(spec["group"])
        models[label] = m
        spec_m = MODELS[m]
        print(f"[cmp] {label}: grid for {m} -- layers {spec_m['layers']}, "
              f"positions {spec_m['positions']}, neuron pool {spec_m['pool']}")
        missing[label] = report_missing(frames[label], label, expected_grid(model=m), model=m)
    with open(Path(out_dir) / "missing_runs.json", "w") as f:
        json.dump({"models": models,
                   "missing": {k: [list(m) for m in v] for k, v in missing.items()}}, f, indent=2)

    if pairs is None:
        pairs = [(a, b) for a in frames if a.startswith("joint")
                 for b in frames if b.startswith("sft")]
    tables = []
    for label_a, label_b in pairs:
        if label_a not in frames or label_b not in frames:
            print(f"[cmp] skipping {label_a} vs {label_b}: an arm is missing")
            continue
        print(f"\n[cmp] ---------- {label_a} vs {label_b} ----------")
        merged = pair(frames[label_a], frames[label_b], label_a, label_b)
        merged.to_csv(Path(out_dir) / f"paired_runs_{label_a}_vs_{label_b}.csv", index=False)
        metrics = [f"{r}_{m if m != 'corr' else 'corr'}"
                   for r in ("interventional", "endogenous") for m in FIDELITY_METRICS]
        metrics += [f"{r}_corr_minus_spearman" for r in ("interventional", "endogenous")]
        metrics = [m for m in metrics if f"{m}_{label_a}" in merged.columns]
        metrics += sorted({c.rsplit("_", 1)[0] for c in merged.columns
                           if c.startswith("ood_") and c.endswith(f"_{label_a}")
                           and "activation_std" not in c})
        stats = [st for st in (paired_stats(merged, m, label_a, label_b) for m in metrics) if st]
        table = pd.DataFrame(stats).assign(arm=label_a, baseline=label_b)
        tables.append(table)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        # one colour limit for both readouts, so the two delta maps share a colorbar
        # one colour limit per METRIC family, shared by its two readouts, so the
        # interventional and endogenous maps of the same metric are comparable but a
        # metric on a different scale (tau-b, R^2) is not forced onto Pearson's limits
        for fam in DELTA_METRICS:
            fam_metrics = tuple(f"{r}_{fam}" for r in ("interventional", "endogenous")
                                if f"{r}_{fam}_{label_a}" in merged.columns)
            if not fam_metrics:
                continue
            lim = delta_color_limit(merged, fam_metrics, label_a, label_b)
            for m in fam_metrics:
                plot_delta_heatmap(merged, m, label_a, label_b, out_dir, lim=lim)

    if tables:
        pd.concat(tables, ignore_index=True).to_csv(Path(out_dir) / "paired_summary.csv",
                                                    index=False)
    # the joint arms' absolute (layer, position) maps on their own, so the joint results
    # can be read without reference to SFT
    for label, f in frames.items():
        if not label.startswith("joint"):
            continue
        for fam in FIDELITY_METRICS:
            for r in ("interventional", "endogenous"):
                col = f"{r}_{fam}"
                if col in f.columns and f[col].notna().any():
                    plot_arm_heatmap(f, col, label, out_dir)
    # one color per arm, shared by both readout panels so one legend serves both
    arm_colors = {arm: c for arm, c in zip(frames, plt.get_cmap("tab10").colors)}
    plot_ood_ladders(frames, out_dir, arm_colors=arm_colors)
    return frames, tables


if __name__ == "__main__":
    # SFT appears TWICE, once per gate arm. gate ON is the purest bypass baseline: the
    # adapter cannot touch the introspected token, so interventional corr is 0 by
    # construction and whatever endogenous score it has is context prediction. That
    # makes it the arm whose endogenous score SHOULD collapse hardest off-distribution,
    # and therefore the sharpest test of the OOD logic itself. gate OFF isolates the
    # objective (joint vs endogenous-only) with access held constant.
    ARMS = {
        "sft_gateon":  dict(group="mlp_endo_sweep_layer_token",
                            filters=dict(gate="on", w=1.0)),
        "sft_gateoff": dict(group="mlp_endo_sweep_layer_token",
                            filters=dict(gate="off", w=1.0)),
        "joint_kl0.1": dict(group="mlp_joint_grid_layer_token",
                            filters=dict(gate="off", w=0.01, kl="0.1")),
        "joint_free":  dict(group="mlp_joint_grid_layer_token",
                            filters=dict(gate="off", w=0.01, kl="none")),
    }
    # --model-suffix runs the same four arms for another model: the group names gain the
    # suffix and the expected grid follows from it, so nothing else changes.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-suffix", default="",
                    help="e.g. _llama318b, _llama323b or _llama323b_hyper_tuned; appended "
                         "to every group name")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if a.model_suffix:
        for spec in ARMS.values():
            spec["group"] = spec["group"] + a.model_suffix
    out = a.out_dir or f"{ROOT}/outputs/plots/sft_vs_joint{a.model_suffix}"
    # default pairs = every joint arm against every sft arm (4 comparisons)
    main(ARMS, out_dir=out)
