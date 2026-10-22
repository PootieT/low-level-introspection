#!/usr/bin/env python3
"""
Which attention heads is each evaluation actually relying on?

mechanistic_analysis/trace_mover_head.py classifies heads by what they DO when the signal is swept --
threshold detectors that switch their attention onto an anomalous token, linear
transporters that fetch the value at constant attention, re-readers that pick it up
downstream (src/introspector/head_taxonomy.py). None of that says what the model NEEDS.
This script knocks heads out and measures the evaluations the paper reports.

THE ARMS match the main evaluation exactly, so the unablated baseline reproduces the
numbers already in each run directory -- and that is checked, not assumed:

  endogenous   no injection; corr(prediction, cached activation)
               == endogenous_test_metrics_endogenous.json
  x1           gaussian injection at the neuron's own std, RNG seeded with config.seed
               == interventional_test_metrics_gaussian.json
  x10          the same at 10x the std. Not a reported metric: it is here because the
               threshold detectors only switch at large injections, and a knockout that
               moves NOTHING in x1 or endogenous is only interpretable if it moves
               something somewhere.

Every interventional pass re-seeds the sampler, so every ablation config sees exactly
the same injected values and the deltas are paired.

WHAT IT DISCRIMINATES. The reported interventional eval draws from N(mean, std), and
the detectors' attention SWITCH sits past ~3 sigma, where under 0.3% of those draws
land -- so it was natural to predict that knocking out the threshold heads leaves both
reported metrics intact and moves only x10.

That prediction FAILED on the first cell measured (joint_N0.1377_POS-12, 256 eval rows,
provisional classes): the 11 threshold heads cost x1 -0.23 against +0.0005 for a
size-matched random control, while costing endogenous only -0.045. Switching is not
all these heads do: at natural magnitudes they still attend to t weakly (1e-3 to 3e-2)
and fetch the value through that small attention -- one of them carried a third of the
at-t slope mass locally. The class is named for its x10 behaviour; at x1 the same heads
ARE the transporters. The endogenous/interventional asymmetry is then the finding to
chase, and the arms are read against each other:
  same heads carry both arms   -> shared transport; the interventional term's benefit is
                                  anti-bypass pressure on the readout, not new machinery
  interventional needs heads   -> endogenous is carried by another route (re-readers,
  endogenous does not             or context), and the two evals exercise different
                                  circuits

MEAN ABLATION (default), not zero. A head's output is replaced with its mean over a
cache of examples the model gets RIGHT, so the knockout removes the head's
example-specific contribution without pushing the residual stream off-distribution.
Zeroing an attention output is the classic knockout but is itself an out-of-
distribution perturbation, and part of any effect it shows is the model reacting to a
zero. The mean is conditional on WHERE the token is -- input text, introspection prompt,
or the readout token -- since a head's typical output at the readout is nothing like
its output mid-document. The cache comes from the TRAIN split, so the examples
evaluated and the examples the mean is taken from never overlap.

FULL KNOCKOUT at every position, so indirect effects -- a knocked-out head's absence
propagating through the layers above it -- are part of what is measured.

OUTPUTS (per run, <run>/head_ablation/)
  head_classes.csv                the classification the groups were built from
  head_ablation_groups.csv        each head class knocked out together, per arm
  head_ablation_individual.csv    every non-inert head knocked out alone, per arm
  head_ablation_curves.csv        cumulative knockout in order of each arm's own
                                  individual importance: how many heads it needs
  head_ablation_meta.json         baseline vs reported, mean-cache provenance

Usage
-----
python mechanistic_analysis/score_head_ablation.py --config <run config>
python mechanistic_analysis/score_head_ablation.py --group mlp_joint_grid_layer_token \\
    --run-glob 'joint_*kl0.1*' --skip-existing
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from src.introspector.analysis_loading import load_run_for_analysis, run_per_locale
from src.introspector.head_ablation import (
    parse_arms, HeadAblator, build_mean_cache, eval_all, successful_train_examples)
from src.introspector.head_taxonomy import (
    classify_heads, class_members, DEFAULT_ATT_TAU, DEFAULT_STATIC_TAU,
    DEFAULT_FRAC_T_MIN, DEFAULT_T_SHARE_MIN, DEFAULT_TOTAL_SHARE_MIN)
from src.introspector.run_discovery import (
    add_discovery_args, runs_from_args, resolve_run_action, locale_out_dir)

# class groups knocked out together. The two unions are the comparisons the prediction
# is phrased in; the singletons show which sub-class carries it.
GROUPS = {
    "threshold_all": ("threshold_grabber", "threshold_other"),
    "threshold_grabber": ("threshold_grabber",),
    "transporter_all": ("graded_transporter", "linear_transporter"),
    "linear_transporter": ("linear_transporter",),
    "re_reader": ("re_reader",),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--mover-dir", type=Path, default=None,
                   help="Where mechanistic_analysis/trace_mover_head.py wrote this run's outputs (default: "
                        "<run>/mover_trace). The head classes are read from there.")
    p.add_argument("--arms", type=str, default="endogenous,x1,x10",
                   help="Evaluation arms: 'endogenous' and/or 'xN' (gaussian injection at "
                        "N times the neuron's std). endogenous and x1 are the reported "
                        "metrics; x10 is where the threshold heads actually switch.")
    p.add_argument("--ablation", choices=("mean", "zero"), default="mean")
    p.add_argument("--mean-examples", type=int, default=500,
                   help="Successful TRAIN examples the per-head mean is taken over.")
    p.add_argument("--mean-scan", type=int, default=4000,
                   help="Train rows scanned to find --mean-examples successful ones.")
    p.add_argument("--success-threshold", type=float, default=0.02,
                   help="|endogenous prediction - cached activation| for a train example "
                        "to count as successful (same default as the circuit trace).")
    p.add_argument("--n-eval", type=int, default=None,
                   help="Test rows evaluated (default: the whole split, which is what "
                        "makes the baseline comparable to the reported metrics).")
    p.add_argument("--max-individual", type=int, default=40,
                   help="Cap on heads knocked out one at a time (non-inert heads only).")
    p.add_argument("--curve-k", type=int, default=15,
                   help="Longest cumulative knockout curve.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Default: the run's own eval_batch_size.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the random-head control group (the injection RNG always "
                        "uses config.seed, to match the reported eval).")
    # classification thresholds, passed straight to head_taxonomy.classify_heads
    p.add_argument("--att-tau", type=float, default=DEFAULT_ATT_TAU)
    p.add_argument("--static-tau", type=float, default=DEFAULT_STATIC_TAU)
    p.add_argument("--frac-t-min", type=float, default=DEFAULT_FRAC_T_MIN)
    p.add_argument("--t-share-min", type=float, default=DEFAULT_T_SHARE_MIN)
    p.add_argument("--total-share-min", type=float, default=DEFAULT_TOTAL_SHARE_MIN)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Single-run override; a sweep always writes to <run>/head_ablation/.")
    add_discovery_args(p)
    return p.parse_args()


# ======================================================================
# Main measurement
# ======================================================================
def reported_metrics(run_dir: Path, locale=None) -> dict:
    """The run's own reported numbers, for the baseline check.

    For a multi-locale run the metrics JSONs are POOLED over every trained signal, which
    no single-locale view can reproduce (and pooling inflates corr with between-signal
    variance anyway). The saved PREDICTION CSVs carry signal_layer/signal_index, so the
    per-signal number is recomputed from them instead.
    """
    out = {}
    if locale is None:
        e = run_dir / "endogenous_test_metrics_endogenous.json"
        i = run_dir / "interventional_test_metrics_gaussian.json"
        if e.exists():
            out["endogenous"] = json.loads(e.read_text()).get("corr")
        if i.exists():
            out["x1"] = json.loads(i.read_text()).get("test", {}).get("corr")
        return out
    _, L, idx = locale
    for key, fname, target in (("endogenous", "endogenous_test_predictions_endogenous.csv",
                                "activation"),
                               ("x1", "interventional_test_predictions_gaussian.csv",
                                "injected")):
        f = run_dir / fname
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if not {"signal_layer", "signal_index", target}.issubset(df.columns):
            continue
        d = df[(df["signal_layer"] == L) & (df["signal_index"] == idx)]
        if len(d) > 2 and d["predictions"].std() > 0 and d[target].std() > 0:
            out[key] = float(pearsonr(d["predictions"], d[target])[0])
    return out


def ablate_run(run, spec, args, out_dir: Path):
    cfg = run.base.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    bs = args.batch_size or run.config.training_config.eval_batch_size
    arms = parse_arms(args.arms)

    # ---- classes, from the mover trace ----
    # a multi-locale run's mover trace is per signal, in mover_trace/<signal>/
    mover_dir = (Path(args.mover_dir) if args.mover_dir else
                 locale_out_dir(spec.run_dir / "mover_trace", spec,
                                (run.signal_type, run.layer, run.index)))
    heads_csv = mover_dir / "mover_heads.csv"
    if not heads_csv.exists():
        raise SystemExit(f"{heads_csv} not found -- run mechanistic_analysis/trace_mover_head.py on this run "
                         f"first (the knockout groups are built from its head classes).")
    prof = mover_dir / "mover_attention_profile_all_heads.csv"
    classes = classify_heads(
        pd.read_csv(heads_csv), pd.read_csv(prof) if prof.exists() else None,
        att_tau=args.att_tau, static_tau=args.static_tau, frac_t_min=args.frac_t_min,
        t_share_min=args.t_share_min, total_share_min=args.total_share_min)
    classes.to_csv(out_dir / "head_classes.csv", index=False)
    print("[ablate] head classes: " + ", ".join(
        f"{k}={v}" for k, v in classes["head_class"].value_counts().items()))

    # ---- eval set, ablator, mean cache ----
    dataset = run.dataset if args.n_eval is None else run.dataset.select(
        range(min(args.n_eval, len(run.dataset))))
    ablator = HeadAblator(run, n_heads, head_dim, mode=args.ablation)
    mean_info = {}
    try:
        if args.ablation == "mean":
            cache_ds, mean_info = successful_train_examples(run, args)
            ablator.means = build_mean_cache(run, cache_ds, bs, n_layers)

        # ---- baseline, and the check that it IS the reported run ----
        base = eval_all(run, dataset, arms, ablator, bs, [])
        print(f"[ablate] baseline: " + ", ".join(f"{k}={v:.4f}" for k, v in base.items()))
        this_locale = (run.signal_type, run.layer, run.index) if run.is_multi_locale else None
        reported = reported_metrics(spec.run_dir, this_locale)
        full_split = args.n_eval is None
        for k, v in reported.items():
            if k in base and v is not None:
                gap = abs(base[k] - v)
                # subset first: a subset baseline is a different number, and must not be
                # reported as "OK" just because it happens to land close
                if not full_split:
                    flag = "(subset; not comparable)"
                elif this_locale is not None and k != "endogenous":
                    # the reported run drew its injected values sequentially over ALL
                    # signals' rows; re-seeding over this signal's rows alone draws a
                    # different (equally distributed) sequence, so agreement is only
                    # statistical here. Endogenous involves no draws and must match.
                    flag = ("(per-signal re-draw; statistically comparable only)"
                            if gap < 0.05 else "LARGE GAP -- check before interpreting")
                elif gap < 0.01:
                    flag = "OK"
                else:
                    flag = ("MISMATCH -- the ablation harness is not reproducing the "
                            "reported eval; do not interpret knockouts until resolved")
                print(f"[ablate] baseline {k}={base[k]:.4f} vs reported {v:.4f} "
                      f"(|diff|={gap:.4f}) {flag}")

        # ---- class groups + a size-matched random control ----
        group_rows = []
        for gname, members in tqdm(GROUPS.items()):
            heads = class_members(classes, members)
            if not heads:
                continue
            res = eval_all(run, dataset, arms, ablator, bs, heads)
            for a in arms:
                group_rows.append({"group": gname, "n_heads": len(heads), "arm": a.name,
                                   "corr": res[a.name], "baseline": base[a.name],
                                   "delta": res[a.name] - base[a.name],
                                   "heads": ";".join(f"L{L}h{h}" for L, h in heads)})
        # the null: ablating k heads the taxonomy calls inert, SIZE-MATCHED to each group.
        # Without it a group's drop could just be what ablating any k heads does, and a
        # single control at the largest group's size would be too harsh a null for the
        # smaller groups. Size-1 groups are left to the individual knockouts.
        inert = class_members(classes, "inert")
        sizes = sorted({len(class_members(classes, m)) for m in GROUPS.values()} - {0, 1})
        rng = np.random.RandomState(args.seed + 1000 * run.layer + run.index)
        for k_ctrl in tqdm(sizes):
            if len(inert) < k_ctrl:
                continue
            pick = [inert[j] for j in rng.choice(len(inert), size=k_ctrl, replace=False)]
            res = eval_all(run, dataset, arms, ablator, bs, pick)
            for a in arms:
                group_rows.append({"group": f"random_inert_n{k_ctrl}", "n_heads": k_ctrl,
                                   "arm": a.name, "corr": res[a.name],
                                   "baseline": base[a.name],
                                   "delta": res[a.name] - base[a.name],
                                   "heads": ";".join(f"L{L}h{h}" for L, h in pick)})
        groups_df = pd.DataFrame(group_rows)
        groups_df.to_csv(out_dir / "head_ablation_groups.csv", index=False)
        print("\n[ablate] class-group knockouts (delta vs baseline):")
        if not groups_df.empty:
            print(groups_df.pivot_table(index=["group", "n_heads"], columns="arm",
                                        values="delta").to_string(float_format=lambda v: f"{v:+.4f}"))

        # ---- individual knockouts ----
        cand = classes[classes["head_class"] != "inert"].sort_values("best_rank")
        cand = cand.head(args.max_individual)
        ind_rows = []
        for n, r in tqdm(enumerate(cand.itertuples(), 1)):
            res = eval_all(run, dataset, arms, ablator, bs, [(r.layer, r.head)])
            for a in arms:
                ind_rows.append({"layer": int(r.layer), "head": int(r.head),
                                 "head_class": r.head_class,
                                 "rel_depth": getattr(r, "rel_depth", np.nan),
                                 "arm": a.name, "corr": res[a.name],
                                 "baseline": base[a.name], "delta": res[a.name] - base[a.name]})
            if n % 10 == 0 or n == len(cand):
                print(f"[ablate] individual knockouts: {n}/{len(cand)}")
        ind_df = pd.DataFrame(ind_rows)
        ind_df.to_csv(out_dir / "head_ablation_individual.csv", index=False)

        # ---- cumulative curves, one ordering per arm ----
        # ordered by THAT arm's own individual drop, so each curve answers "how many
        # heads does this evaluation need" rather than inheriting another arm's ranking
        curve_rows = []
        if not ind_df.empty:
            for order_arm in arms:
                order = (ind_df[ind_df["arm"] == order_arm.name]
                         .sort_values("delta")[["layer", "head"]].values.tolist())
                order = [(int(L), int(h)) for L, h in order][:args.curve_k]
                for k in range(1, len(order) + 1):
                    res = eval_all(run, dataset, arms, ablator, bs, order[:k])
                    for a in arms:
                        curve_rows.append({"order_by": order_arm.name, "k": k,
                                           "added": f"L{order[k-1][0]}h{order[k-1][1]}",
                                           "arm": a.name, "corr": res[a.name],
                                           "baseline": base[a.name],
                                           "delta": res[a.name] - base[a.name]})
                print(f"[ablate] cumulative curve ordered by {order_arm.name}: done")
        curves_df = pd.DataFrame(curve_rows)
        curves_df.to_csv(out_dir / "head_ablation_curves.csv", index=False)
    finally:
        ablator.remove()

    with open(out_dir / "head_ablation_meta.json", "w") as f:
        json.dump({"run": run.config.experiment_name, "checkpoint": str(run.checkpoint),
                   "signal": f"N{run.layer}.{run.index}", "ablation": args.ablation,
                   "arms": [a.name for a in arms], "baseline": base,
                   "reported": reported_metrics(
                       spec.run_dir,
                       (run.signal_type, run.layer, run.index) if run.is_multi_locale else None),
                   "n_eval": len(dataset), "full_split": args.n_eval is None,
                   "mean_cache": mean_info, "mover_dir": str(mover_dir),
                   "class_counts": classes["head_class"].value_counts().to_dict()},
                  f, indent=2)
    _plot(run.config.experiment_name, groups_df, curves_df, out_dir)
    print(f"[ablate] wrote {out_dir}")


# ======================================================================
# Plots
# ======================================================================
def _plot(name, groups_df, curves_df, out_dir: Path):
    if groups_df is not None and not groups_df.empty:
        piv = groups_df.pivot_table(index="group", columns="arm", values="delta")
        order = ([g for g in GROUPS if g in piv.index]
                 + sorted(g for g in piv.index if g.startswith("random_inert")))
        piv = piv.loc[order]
        sizes = groups_df.drop_duplicates("group").set_index("group")["n_heads"]
        piv.index = [f"{g}\n(n={int(sizes[g])})" for g in piv.index]
        fig, ax = plt.subplots(figsize=(1.2 + 1.3 * len(piv), 3.4))
        piv.plot.bar(ax=ax, width=0.8, rot=0)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("corr change vs baseline", fontsize=9)
        ax.tick_params(labelsize=7); ax.legend(fontsize=7, title="arm", title_fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"Knocking out each head class\n{name}", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "head_ablation_groups.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if curves_df is not None and not curves_df.empty:
        orders = list(dict.fromkeys(curves_df["order_by"]))
        fig, axes = plt.subplots(1, len(orders), figsize=(1 + 3.4 * len(orders), 3.2),
                                 squeeze=False, sharey=True)
        for ax, ob in zip(axes.flat, orders):
            d = curves_df[curves_df["order_by"] == ob]
            for arm, g in d.groupby("arm", sort=False):
                g = g.sort_values("k")
                ax.plot([0] + g["k"].tolist(), [0] + g["delta"].tolist(), "o-", ms=3,
                        label=arm)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xlabel("heads knocked out (cumulative)", fontsize=8)
            ax.set_title(f"ordered by {ob} importance", fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
        axes.flat[0].set_ylabel("corr change vs baseline", fontsize=8)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"How many heads does each evaluation need? -- {name}", fontsize=8, y=1.03)
        fig.tight_layout()
        fig.savefig(out_dir / "head_ablation_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def replot_ablation(spec, out_dir: Path, args) -> bool:
    g, c = out_dir / "head_ablation_groups.csv", out_dir / "head_ablation_curves.csv"
    if not g.exists():
        return False
    _plot(spec.name, pd.read_csv(g), pd.read_csv(c) if c.exists() else None, out_dir)
    return True


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    runs = runs_from_args(args)
    if not runs:
        raise SystemExit("No runs with checkpoints matched.")
    run_per_locale(args, runs, "head_ablation", "head_ablation_meta.json",
                   analyze=lambda run, spec, out_dir: ablate_run(run, spec, args, out_dir),
                   replot=replot_ablation, log="[ablate]")


if __name__ == "__main__":
    main()
