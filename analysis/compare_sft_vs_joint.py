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
    suffix is the original 1B sweep.
    """
    for key in MODELS:
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
               "w": float(mw["w"]) if mw else None, "kl": mkl["kl"] if mkl else None,
               "interventional_corr": interv.get("corr"),
               "endogenous_corr": endo.get("corr")}
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


def plot_ood_ladder(frames: dict, out_dir, readout="endogenous", common_only=True,
                    arm_colors=None):
    """Every arm across the corpus ladder, mean ± SE over neurons.

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
        return
    order = ["in-domain"] + corpora
    rows = []
    for arm, f in frames.items():
        for corpus in order:
            col = f"{readout}_corr" if corpus == "in-domain" else f"ood_{corpus}_{readout}"
            if col not in f:
                continue
            vals = f[col].dropna()
            if len(vals) == 0:
                continue
            rows.append({"corpus": corpus, "arm": arm, "n": len(vals), "mean": vals.mean(),
                         "se": vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(13, 8))
    for arm, sub in df.groupby("arm"):
        sub = sub.set_index("corpus").reindex(order).reset_index()
        # arm_colors keeps each arm the same color in BOTH the endogenous and
        # interventional figures, so the two eval settings are read off one legend.
        color = arm_colors.get(arm) if arm_colors else None
        ax.errorbar(np.arange(len(order)), sub["mean"], yerr=sub["se"], marker="o",
                    markersize=10, linewidth=2.5, capsize=4, color=color,
                    label=f"{arm} (n={int(sub['n'].max())})")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=20, fontsize=26)
    ax.tick_params(axis="y", labelsize=26)
    ax.set_ylabel(f"{readout} corr", fontsize=32); ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="grey", ls="--", lw=1); ax.grid(alpha=0.25)
    ax.legend(fontsize=22)
    ax.set_title(f"{readout} readout across the corpus ladder (mean ± SE over neurons)",
                 fontsize=30)
    fig.tight_layout()
    path = Path(out_dir) / f"ood_ladder_{readout}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    df.to_csv(Path(out_dir) / f"ood_ladder_{readout}.csv", index=False)
    print(f"[cmp] wrote {path}")


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


def pretty_metric(metric: str) -> str:
    """Short metric name for the colorbar, e.g. interventional_corr ->
    'Interventional Corr.', ood_<corpus>_endogenous -> 'Endogenous Corr. (<corpus>)'."""
    readout_names = {"interventional": "Interventional Corr.",
                     "endogenous": "Endogenous Corr."}
    if metric in ("interventional_corr", "endogenous_corr"):
        return readout_names[metric.split("_")[0]]
    if metric.startswith("ood_"):
        parts = metric.split("_")
        readout, corpus = parts[-1], "_".join(parts[1:-1])
        return f"{readout_names.get(readout, readout.title() + ' Corr.')} ({corpus})"
    return metric.replace("_", " ").title()


def plot_delta_heatmap(merged, metric, label_a, label_b, out_dir):
    """Within-neuron delta by (layer, position) -- where does one arm win?"""
    ca, cb = f"{metric}_{label_a}", f"{metric}_{label_b}"
    if ca not in merged or cb not in merged:
        return
    sub = merged[[*KEY, ca, cb]].dropna()
    if sub.empty:
        return
    sub = sub.assign(delta=sub[ca] - sub[cb])
    pivot = sub.pivot_table(index="layer", columns="position", values="delta", aggfunc="mean")
    # spread across the cell's neurons: a +0.3 mean built from neurons at +0.9 and -0.3
    # is a different claim from one where every neuron moved +0.3
    std = sub.pivot_table(index="layer", columns="position", values="delta", aggfunc="std")
    annot = (pivot.round(2).astype(str) + "\n±" +
             std.round(2).fillna(0).astype(str))
    lim = float(np.nanmax(np.abs(pivot.to_numpy()))) or 1.0
    plt.figure(figsize=(2.0 * pivot.shape[1] + 5, 1.1 * pivot.shape[0] + 4))
    # cells are annotated mean ± std (std over the cell's neurons, aggfunc="std")
    hm = sns.heatmap(pivot, cmap="RdBu_r", center=0, vmin=-lim, vmax=lim, annot=annot, fmt="",
                annot_kws={"size": 20},
                cbar_kws={"label": f"Paired Diff. {pretty_metric(metric)} (Mean ± Std)"})
    hm.figure.axes[-1].yaxis.label.set_size(24)  # colorbar label
    hm.figure.axes[-1].tick_params(labelsize=22)  # colorbar ticks
    hm.tick_params(axis="x", labelsize=30)
    hm.tick_params(axis="y", labelsize=30)
    plt.xlabel("token position", fontsize=38); plt.ylabel("layer", fontsize=38)
    plt.title(f"{pretty_arm(label_a)} vs {pretty_arm(label_b)}", fontsize=36)
    plt.tight_layout()
    path = Path(out_dir) / f"delta_{metric}_{label_a}_vs_{label_b}.png"
    plt.savefig(path, dpi=150); plt.close()
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
        metrics = ["interventional_corr", "endogenous_corr"]
        metrics += sorted({c.rsplit("_", 1)[0] for c in merged.columns
                           if c.startswith("ood_") and c.endswith(f"_{label_a}")
                           and "activation_std" not in c})
        stats = [st for st in (paired_stats(merged, m, label_a, label_b) for m in metrics) if st]
        table = pd.DataFrame(stats).assign(arm=label_a, baseline=label_b)
        tables.append(table)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        for m in ("interventional_corr", "endogenous_corr"):
            plot_delta_heatmap(merged, m, label_a, label_b, out_dir)

    if tables:
        pd.concat(tables, ignore_index=True).to_csv(Path(out_dir) / "paired_summary.csv",
                                                    index=False)
    # one color per arm, shared by both readouts so their legends line up
    arm_colors = {arm: c for arm, c in zip(frames, plt.get_cmap("tab10").colors)}
    plot_ood_ladder(frames, out_dir, readout="endogenous", arm_colors=arm_colors)
    plot_ood_ladder(frames, out_dir, readout="interventional", arm_colors=arm_colors)
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
                    help="e.g. _llama318b or _llama323b; appended to every group name")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if a.model_suffix:
        for spec in ARMS.values():
            spec["group"] = spec["group"] + a.model_suffix
    out = a.out_dir or f"{ROOT}/outputs/plots/sft_vs_joint{a.model_suffix}"
    # default pairs = every joint arm against every sft arm (4 comparisons)
    main(ARMS, out_dir=out)
