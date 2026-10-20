#!/usr/bin/env python3
"""
Which attention heads does each evaluation need? -- every head, knocked out alone.

The circuit trace's per-head map is a DIRECT-attribution slope measured under
injection: it says which heads deliver the injected value to the readout, is blind to
heads that act through later layers, and has no endogenous counterpart. This script
asks the plainer causal question for both evaluations at once: knock out one head and
see how much each evaluation's correlation falls.

WHAT IS ABLATED. Every head in the layers ABOVE the signal (layer L+1 .. last; layer
L's attention runs before MLP L writes the neuron, so it cannot see the signal), one at
a time, mean-ablated (the per-position-bucket mean over 500 successful TRAIN examples,
see src/introspector/head_ablation.py) over the token window from the introspected
token t through the readout. Positions before t cannot see the signal under causal
attention, so the window removes every write that could carry it while leaving the
context the neuron is computed from intact.

WHAT IS EVALUATED. The same --n-examples (200) test examples the model gets right
endogenously (|prediction - activation| <= 0.02, the circuit trace's selection), under
  endogenous   no injection; corr(prediction, cached activation)
  x1           gaussian injection at the neuron's own std, sampler re-seeded to
               config.seed for every pass -> identical injected values everywhere
Each knockout is therefore a PAIRED difference on fixed examples and fixed values.

OUTPUTS (per run, <run>/head_knockout/)
  head_knockout_individual.csv   per (head, arm): corr, delta, frac_drop, paired
                                 bootstrap CI, p_hurts, significant, rank
  head_knockout_curves.csv       cumulative knockout down each arm's ranking and down
                                 random orderings, evaluated in EVERY arm (so the
                                 endogenous-ranked curve evaluated on x1 is the cross-
                                 transfer test)
  head_knockout_summary.json     baselines, ranking reliability (split-half), cross-arm
                                 Spearman raw and disattenuated, top-k Jaccard vs a
                                 layer-matched null, curve areas and transfer ratios
  head_knockout_preds.npz        every per-example prediction, for re-analysis
  head_knockout_{maps,agreement,curves}.png
  head_knockout_meta.json        provenance; written LAST (the --skip-existing sentinel)

Usage
-----
python score_head_knockout.py --config <run config>
python score_head_knockout.py --group mlp_joint_grid_layer_token \\
    --run-glob 'joint_*kl0.1*' --skip-existing          # shards over $SGE_TASK_ID
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from tqdm import tqdm

from src.introspector.analysis_loading import run_per_locale
from src.introspector.head_ablation import (
    WINDOWS, parse_arms, HeadAblator, build_mean_cache, eval_arm_outputs, safe_corr,
    successful_train_examples)
from src.introspector.knockout_stats import (
    knockout_deltas, bootstrap_deltas, bootstrap_summary, split_half,
    jaccard, ranks_desc, rowwise_corr, agreement_from_tables)
from src.introspector.run_discovery import add_discovery_args, runs_from_args
from trace_introspection_circuit import endogenous_predictions, select_success_indices
from analysis.run_catalog import NEURON_RE, POS_RE, KL_RE


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--arms", type=str, default="endogenous,x1",
                   help="'endogenous' and/or 'xN' (gaussian injection at N x the std).")
    p.add_argument("--window", choices=WINDOWS, default="t_to_last",
                   help="Token window the knockout applies to. Non-default windows write "
                        "to head_knockout_<window>/.")
    p.add_argument("--ablation", choices=("mean", "zero"), default="mean")
    p.add_argument("--n-examples", type=int, default=200,
                   help="Successful test examples evaluated.")
    p.add_argument("--success-threshold", type=float, default=0.02)
    p.add_argument("--max-scan", type=int, default=None,
                   help="Max test rows scanned for successes (default: all).")
    p.add_argument("--mean-examples", type=int, default=500,
                   help="Successful TRAIN examples the per-head mean is taken over.")
    p.add_argument("--mean-scan", type=int, default=4000)
    p.add_argument("--curve-k", type=int, default=30,
                   help="Longest cumulative knockout.")
    p.add_argument("--n-random-orders", type=int, default=3,
                   help="Random orderings for the cumulative-curve baseline.")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--n-splits", type=int, default=100,
                   help="Random example halvings for split-half reliability.")
    p.add_argument("--topk", type=str, default="5,10,20")
    p.add_argument("--mass", type=str, default="0.5,0.8,0.95",
                   help="Mass fractions for the head sets compared across arms: the "
                        "fewest heads whose positive drops sum to q of the arm's total. "
                        "The FIRST is the one plotted.")
    p.add_argument("--n-perm", type=int, default=1000,
                   help="Permutations for the layer-matched top-k null.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Default: the run's own eval_batch_size.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for bootstrap/permutations/random orders (the injection RNG "
                        "always uses config.seed, to match the reported eval).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Single-run override; a sweep writes to <run>/head_knockout/.")
    add_discovery_args(p)
    return p.parse_args()


def analysis_name(window: str) -> str:
    return "head_knockout" if window == "t_to_last" else f"head_knockout_{window}"


def parse_floats(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def parse_ints(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def head_label(L: int, h: int) -> str:
    return f"L{L}H{h}"


# ======================================================================
# Measurement
# ======================================================================
def knockout_run(run, spec, args, out_dir: Path):
    cfg = run.base.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    bs = args.batch_size or run.config.training_config.eval_batch_size
    arms = parse_arms(args.arms)
    names = [a.name for a in arms]
    heads = [(L, h) for L in range(run.layer + 1, n_layers) for h in range(n_heads)]
    meta = {"run": run.config.experiment_name, "checkpoint": str(run.checkpoint),
            "signal": f"N{run.layer}.{run.index}", "signal_layer": run.layer,
            "ablation": args.ablation, "window": args.window, "arms": names,
            "n_heads_ablated": len(heads), "head_layers": [run.layer + 1, n_layers - 1]}
    if not heads:
        print(f"[knockout] signal at the last layer ({run.layer}): no heads above it")
        (out_dir / "head_knockout_meta.json").write_text(json.dumps(meta, indent=2))
        return

    # ---- eval set: successful endogenous test examples ----
    preds, acts = endogenous_predictions(run.model, run.dataset, run.collator, bs)
    idx = select_success_indices(preds, acts, args.success_threshold, args.n_examples,
                                 args.max_scan)
    if len(idx) < 20:
        raise SystemExit(f"only {len(idx)} successful examples; loosen --success-threshold")
    dataset = run.dataset.select(idx)
    print(f"[knockout] {len(heads)} heads (layers {run.layer + 1}-{n_layers - 1}) x "
          f"{len(names)} arms on {len(idx)} examples, window={args.window}")

    ablator = HeadAblator(run, n_heads, head_dim, mode=args.ablation)
    mean_info = {}
    try:
        if args.ablation == "mean":
            cache_ds, mean_info = successful_train_examples(run, args)
            ablator.means = build_mean_cache(run, cache_ds, bs, n_layers)

        def evaluate(ko_heads):
            ablator.set_heads(ko_heads)
            try:
                return {a.name: eval_arm_outputs(run, dataset, a, ablator, bs, args.window)
                        for a in arms}
            finally:
                ablator.set_heads([])

        # ---- baseline ----
        base_out = evaluate([])
        p0 = {a: base_out[a][0] for a in names}
        tgt = {a: base_out[a][1] for a in names}
        base = {a: safe_corr(p0[a], tgt[a]) for a in names}
        print("[knockout] baseline: " + ", ".join(f"{a}={v:.4f}" for a, v in base.items()))

        def check_targets(out):
            # the pairing everything below relies on: same examples, same injected values
            for a in names:
                if not np.array_equal(out[a][1], tgt[a]):
                    raise RuntimeError(f"{a}: targets changed between passes -- the "
                                       f"injection sampler is not reproducible, so the "
                                       f"knockout deltas would not be paired")

        # ---- every head alone ----
        P = {a: np.empty((len(heads), len(idx)), dtype=np.float32) for a in names}
        for j, hd in enumerate(tqdm(heads, desc="[knockout] heads", mininterval=30)):
            out = evaluate([hd])
            check_targets(out)
            for a in names:
                P[a][j] = out[a][0]

        imp = {a: -knockout_deltas(P[a], p0[a], tgt[a]) for a in names}

        # ---- cumulative knockouts: each arm's ranking + random orderings ----
        K = min(args.curve_k, len(heads))
        rng = np.random.default_rng(args.seed + 1000 * run.layer + run.index)
        orders = {f"by_{a}": list(np.argsort(-imp[a], kind="stable")[:K]) for a in names}
        for r in range(args.n_random_orders):
            orders[f"random{r}"] = list(rng.permutation(len(heads))[:K])
        C = {a: np.empty((len(orders), K, len(idx)), dtype=np.float32) for a in names}
        for oi, (oname, order) in enumerate(orders.items()):
            for k in tqdm(range(1, K + 1), desc=f"[knockout] curve {oname}", mininterval=30):
                out = evaluate([heads[j] for j in order[:k]])
                check_targets(out)
                for a in names:
                    C[a][oi, k - 1] = out[a][0]
    finally:
        ablator.remove()

    # ---- per-example predictions: everything below can be recomputed from these ----
    np.savez_compressed(
        out_dir / "head_knockout_preds.npz",
        heads=np.array(heads), examples=np.array(idx), arms=np.array(names),
        orders=np.array(list(orders)), order_heads=np.array([o for o in orders.values()]),
        **{f"target_{a}": tgt[a] for a in names}, **{f"base_{a}": p0[a] for a in names},
        **{f"knockout_{a}": P[a] for a in names}, **{f"curve_{a}": C[a] for a in names})

    ind_df, summary, curves_df = summarize(heads, names, P, p0, tgt, C, orders, args)
    ind_df.to_csv(out_dir / "head_knockout_individual.csv", index=False)
    curves_df.to_csv(out_dir / "head_knockout_curves.csv", index=False)
    summary.update({"baseline": base, "n_examples": len(idx), "signal": meta["signal"],
                    "signal_layer": run.layer})
    (out_dir / "head_knockout_summary.json").write_text(json.dumps(summary, indent=2))
    _report(summary, names)
    plot_all(meta["run"], ind_df, curves_df, summary, out_dir)
    meta.update({"n_examples": len(idx), "baseline": base, "mean_cache": mean_info,
                 "batch_size": bs, "curve_k": K, "orders": list(orders)})
    (out_dir / "head_knockout_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[knockout] wrote {out_dir}")


# ======================================================================
# Statistics
# ======================================================================
def summarize(heads, names, P, p0, tgt, C, orders, args):
    rows, imp, sig = [], {}, {}
    for ai, a in enumerate(names):
        base = rowwise_corr(p0[a][None], tgt[a])[0]
        corr = rowwise_corr(P[a], tgt[a])
        delta = corr - base
        bs = bootstrap_summary(bootstrap_deltas(P[a], p0[a], tgt[a], n_boot=args.n_boot,
                                                seed=args.seed + ai))
        imp[a], sig[a] = -delta, bs["significant"]
        rk = ranks_desc(-delta)
        for j, (L, h) in enumerate(heads):
            rows.append({"layer": L, "head": h, "arm": a, "corr": corr[j], "baseline": base,
                         "delta": delta[j], "frac_drop": -delta[j] / base,
                         "ci_lo": bs["ci_lo"][j], "ci_hi": bs["ci_hi"][j],
                         "p_hurts": bs["p_hurts"][j], "significant": bool(bs["significant"][j]),
                         "rank": int(rk[j])})
    ind_df = pd.DataFrame(rows)

    summary = {"n_heads": len(heads), "n_boot": args.n_boot}
    for a in names:
        summary[f"n_significant_{a}"] = int(sig[a].sum())
        summary[f"max_drop_{a}"] = float(np.nanmax(imp[a]))
        summary[f"top_head_{a}"] = head_label(*heads[int(np.nanargmax(imp[a]))])
    summary.update(split_half(P, p0, tgt, n_splits=args.n_splits, seed=args.seed))
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    for a, b in pairs:
        summary[f"spearman_{a}_vs_{b}"] = float(spearmanr(imp[a], imp[b])[0])
        sa, sb = set(np.where(sig[a])[0]), set(np.where(sig[b])[0])
        summary[f"significant_jaccard_{a}_vs_{b}"] = jaccard(sa, sb)
        summary[f"significant_overlap_{a}_vs_{b}"] = len(sa & sb)

    # ---- curves: each ordering evaluated in every arm ----
    crow = []
    for a in names:
        base = rowwise_corr(p0[a][None], tgt[a])[0]
        for oi, (oname, order) in enumerate(orders.items()):
            corr = rowwise_corr(C[a][oi], tgt[a])                       # [K]
            for k in range(len(order)):
                crow.append({"order_by": oname, "k": k + 1,
                             "added": head_label(*heads[order[k]]), "arm": a,
                             "corr": corr[k], "baseline": base, "delta": corr[k] - base,
                             "frac_drop": -(corr[k] - base) / base})
    curves_df = pd.DataFrame(crow)
    summary.update(agreement_from_tables(
        ind_df, curves_df, mass_qs=parse_floats(args.mass), ks=parse_ints(args.topk),
        n_perm=args.n_perm, seed=args.seed))
    return ind_df, summary, curves_df


def _report(s, names):
    for a in names:
        print(f"[knockout] {a}: {s[f'n_significant_{a}']} significant heads, top "
              f"{s[f'top_head_{a}']} (drop {s[f'max_drop_{a}']:+.4f}), effective heads "
              f"{s[f'effective_heads_pos_{a}']:.1f} (positive drops) / "
              f"{s[f'effective_heads_abs_{a}']:.1f} (|drop|), knockout helps "
              f"{s[f'n_helped_by_knockout_{a}']} of {s['n_heads']}, split-half reliability "
              f"{s[f'reliability_{a}']:.2f}")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            mo = ", ".join(f"q={m['q']:g}: {m['n_a']}/{m['n_b']} heads, overlap "
                           f"{m['overlap']:.2f} (null {m['null_mean']:.2f}, p={m['p']:.3f})"
                           for m in s[f"mass_overlap_{a}_vs_{b}"])
            tk = ", ".join(f"J@{t['k']}={t['jaccard']:.2f} (null {t['null_mean']:.2f}, "
                           f"p={t['p']:.3f})" for t in s[f"topk_{a}_vs_{b}"])
            print(f"[knockout] {a} vs {b} mass sets: {mo}")
            print(f"[knockout] {a} vs {b} top-k: {tk}")
            print(f"[knockout] transfer (vs the better ranking): {b}-ranked on {a} "
                  f"{s.get(f'transfer_{b}_to_{a}', np.nan):.2f}, {a}-ranked on {b} "
                  f"{s.get(f'transfer_{a}_to_{b}', np.nan):.2f} (random "
                  f"{s.get(f'transfer_random_to_{a}', np.nan):.2f} / "
                  f"{s.get(f'transfer_random_to_{b}', np.nan):.2f})")


# ======================================================================
# Plots
# ======================================================================
def arm_label(arm: str) -> str:
    """Display name of an evaluation arm."""
    if arm == "endogenous":
        return "Endogenous"
    if arm == "x1":
        return "Interventional"
    return f"Interventional ({arm})" if arm.startswith("x") else arm.capitalize()


SEQ_LEN = 32      # training window length (positions in titles are absolute)


def run_title(name: str, signal: str | None = None) -> str:
    """'joint_N12.1235_POS-12_..._kl0.1_...' -> 'Joint Loss (KL=0.1), Neuron Activation,
    L=12, i=1235, t=20'. `signal` (e.g. 'N12.1235') overrides the name's neuron, which
    a multi-locale run's name does not carry. Unparseable pieces are left out."""
    parts = []
    kl = KL_RE.search(name)
    kl_txt = None if not kl else ("Unanchored" if kl["kl"] == "none" else f"KL={kl['kl']}")
    if name.startswith("joint"):
        parts.append(f"Joint Loss ({kl_txt})" if kl_txt else "Joint Loss")
    elif kl_txt:
        parts.append(kl_txt)
    n = NEURON_RE.search(signal or "") or NEURON_RE.search(name)
    if n:
        parts.append(f"Neuron Activation, L={n['layer']}, i={n['index']}")
    pz = POS_RE.search(name)
    if pz:
        # 0-indexed absolute position in the 32-token window (POS-1 -> t=31), as in
        # every other figure
        pos = int(pz["pos"])
        parts.append(f"t={pos + SEQ_LEN if pos < 0 else pos}")
    return ", ".join(parts) or name


def plot_all(name, ind_df, curves_df, summary, out_dir: Path):
    names = list(dict.fromkeys(ind_df["arm"]))
    title = run_title(name, summary.get("signal"))
    _plot_maps(title, ind_df, names, out_dir)
    if len(names) >= 2:
        _plot_agreement(title, ind_df, summary, names[:2], out_dir)
    _plot_curves(title, curves_df, names, out_dir)


def _plot_maps(title, ind_df, names, out_dir):
    """Drop per head (layer x head), one panel per arm, significant heads dotted."""
    layers = sorted(ind_df["layer"].unique())
    n_heads = int(ind_df["head"].max()) + 1
    fig, axes = plt.subplots(len(names), 1, squeeze=False,
                             figsize=(max(6, 0.3 * n_heads), 0.5 + len(names) * (0.35 * len(layers) + 1)))
    for ax, a in zip(axes[:, 0], names):
        d = ind_df[ind_df["arm"] == a]
        m = (-d.pivot(index="layer", columns="head", values="delta")
             .reindex(index=layers, columns=range(n_heads))).to_numpy()
        lim = np.nanmax(np.abs(m)) or 1.0
        sns.heatmap(m, ax=ax, cmap="coolwarm", center=0, vmin=-lim, vmax=lim,
                    yticklabels=layers, xticklabels=list(range(n_heads)),
                    cbar_kws={"label": "Correlation Drop", "pad": 0.01})
        s = d[d["significant"]]
        ax.scatter(s["head"] + 0.5, s["layer"].map({L: i for i, L in enumerate(layers)}) + 0.5,
                   s=6, c="k", marker="o")
        base = d["baseline"].iloc[0]
        ax.set_title(f"{arm_label(a)} (Baseline Correlation {base:.3f}; "
                     f"Dot = 95% CI Excludes 0)", fontsize=9)
        ax.set_ylabel("Layer"); ax.tick_params(labelsize=6)
    axes[-1, 0].set_xlabel("Head")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "head_knockout_maps.png", dpi=150)
    plt.close(fig)


def _plot_agreement(title, ind_df, s, pair, out_dir):
    a, b = pair
    w = ind_df.pivot_table(index=["layer", "head"], columns="arm",
                           values=["delta", "significant"], aggfunc="first")
    x, y = -w[("delta", a)].astype(float), -w[("delta", b)].astype(float)
    sig_any = w[("significant", a)].astype(bool) | w[("significant", b)].astype(bool)
    layers = w.index.get_level_values("layer")
    fig, ax = plt.subplots(figsize=(4.6, 5.0))
    sc = ax.scatter(x, y, c=layers, cmap="viridis", s=np.where(sig_any, 22, 8),
                    edgecolors=np.where(sig_any, "k", "none"), linewidths=0.5)
    fig.colorbar(sc, ax=ax, label="Layer")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    for (L, h) in pd.concat([x.nlargest(4), y.nlargest(4)]).index.unique():
        ax.annotate(head_label(L, h), (x[(L, h)], y[(L, h)]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    tk = {t["k"]: t for t in s.get(f"topk_{a}_vs_{b}", [])}
    t10 = tk.get(10)
    dis = s.get(f"disattenuated_{a}_vs_{b}")
    dis_txt = (f"Disattenuated {dis:.2f}" if dis is not None and np.isfinite(dis)
               else "Disattenuated n/a: Split-Half < 0.2")
    txt = (f"Spearman {s[f'spearman_{a}_vs_{b}']:.2f} ({dis_txt})\n"
           f"Reliability: {arm_label(a)} {s[f'reliability_{a}']:.2f}, "
           f"{arm_label(b)} {s[f'reliability_{b}']:.2f}")
    if t10:
        txt += f"\nTop-10 Jaccard {t10['jaccard']:.2f} (Random Layer-Matched Heads {t10['null_mean']:.2f})"
    mo = (s.get(f"mass_overlap_{a}_vs_{b}") or [None])[0]
    if mo:
        txt += (f"\n{100 * mo['q']:g}%-Mass Sets ({mo['n_a']} / {mo['n_b']} Heads): Overlap "
                f"{mo['overlap']:.2f} (Random Layer-Matched Heads {mo['null_mean']:.2f})")
    # below the axes: inside, it covers whichever corner the top heads land in
    ax.text(0.0, -0.2, txt, transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(fc="white", ec="0.8"))
    ax.set_xlabel(f"Correlation Drop ({arm_label(a)})")
    ax.set_ylabel(f"Correlation Drop ({arm_label(b)})")
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "head_knockout_agreement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_curves(title, curves_df, names, out_dir):
    fig, axes = plt.subplots(1, len(names), figsize=(1 + 3.6 * len(names), 3.3),
                             squeeze=False, sharey=False)
    palette = {f"by_{a}": c for a, c in zip(names, ("#c0392b", "#2471a3", "#27ae60"))}
    for ax, a in zip(axes[0], names):
        d = curves_df[curves_df["arm"] == a]
        rand = d[d["order_by"].str.startswith("random")]
        for oname, g in rand.groupby("order_by"):
            g = g.sort_values("k")
            ax.plot([0] + g["k"].tolist(), [0] + (-g["delta"]).tolist(), color="0.7",
                    lw=0.8, label="Random Order" if oname == "random0" else None)
        for oname in palette:
            g = d[d["order_by"] == oname].sort_values("k")
            if len(g):
                ax.plot([0] + g["k"].tolist(), [0] + (-g["delta"]).tolist(), "o-", ms=2.5,
                        color=palette[oname], label=f"Ranked by {arm_label(oname[3:])}")
        ax.axhline(0, color="k", lw=0.6)
        # drop grows DOWNWARD: knocking out more heads reads as the curve falling
        ax.invert_yaxis()
        ax.set_title(arm_label(a), fontsize=9)
        ax.set_xlabel("Heads Knocked Out (Cumulative)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Correlation Drop", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title, fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "head_knockout_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def replot_knockout(spec, out_dir: Path, args) -> bool:
    """Redraw from cache, first refreshing the table-derived summary metrics (mass-set
    overlap, effective heads, transfer) so results computed under older definitions
    pick up the current ones without re-running the knockouts."""
    i, c, s = (out_dir / "head_knockout_individual.csv", out_dir / "head_knockout_curves.csv",
               out_dir / "head_knockout_summary.json")
    if not (i.exists() and c.exists() and s.exists()):
        return False
    ind_df, curves_df, summary = pd.read_csv(i), pd.read_csv(c), json.loads(s.read_text())
    for stale in [k for k in summary if k.startswith("effective_heads_") and
                  not k.startswith(("effective_heads_pos_", "effective_heads_abs_"))]:
        summary.pop(stale)
    summary.update(agreement_from_tables(
        ind_df, curves_df, mass_qs=parse_floats(args.mass), ks=parse_ints(args.topk),
        n_perm=args.n_perm, seed=args.seed))
    s.write_text(json.dumps(summary, indent=2))
    plot_all(spec.name, ind_df, curves_df, summary, out_dir)
    return True


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    runs = runs_from_args(args)
    if not runs:
        raise SystemExit("No runs with checkpoints matched.")
    run_per_locale(args, runs, analysis_name(args.window), "head_knockout_meta.json",
                   analyze=lambda run, spec, out_dir: knockout_run(run, spec, args, out_dir),
                   replot=replot_knockout, log="[knockout]")


if __name__ == "__main__":
    main()
