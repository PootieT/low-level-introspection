#!/usr/bin/env python3
"""
Do endogenous reports read the signal site, or a separate context path? -- ablate the
endogenous and interventional head SETS together and see which value the report follows.

The single-head knockout (mechanistic_analysis/score_head_knockout.py) ranks every head
above the signal by how much its knockout hurts the endogenous correlation (E) and the
x1-injected correlation (I). This script takes those two rankings as given, builds head
SETS from them, mean-ablates each set in one go, and reads out, on rows the rankings were
NOT chosen on:

  corr_endogenous               no injection: corr(prediction, natural activation a)
  corr_interv                   x1 injection: corr(prediction, injected value v)
  corr_natural_under_injection  x1 injection: corr(prediction, a), the natural value the
                                injection REPLACED. v is drawn independently of a, so this
                                is how much the report still follows a once the site no
                                longer carries it: a context path that bypasses the site.

A report that bypasses the site on natural inputs but reads it under injection (the
"separate injection detector" account) predicts: ablating I leaves corr_endogenous
intact and RAISES corr_natural_under_injection; keeping only E makes the injected report
follow a; keeping only I gives a low corr_endogenous. One shared reader predicts
corr_natural_under_injection stays low everywhere and keeping only I still reports
natural values.

HEAD SETS (per --head-defs, from the knockout's single-head drops; positive drops only)
  massQ   the fewest heads whose positive drops sum to Q of the arm's total
          (knockout_stats.mass_set, the sets the knockout summary reports)
  topK    the K heads with the largest positive drops
CONDITIONS (per head set definition)
  endo_only     ablate E \\ I
  interv_only   ablate I \\ E
  joint         ablate E & I
  keep_endo     ablate every candidate head except E ("all but endo")
  keep_interv   ablate every candidate head except I
plus `none` (nothing ablated), and for every condition --n-random random controls with
the same number of heads PER LAYER (for keep_*, the KEPT set is randomized), so a drop
can be told apart from "this many heads were removed". An empty set (e.g. I \\ E when I
sits inside E) is recorded as empty and not evaluated.

Ablation, token window and mean cache are the knockout's (mean over 500 successful train
rows, tokens t..last, heads in layers L+1..last), and must match its meta. Every pass
re-seeds the injection sampler to config.seed, so all conditions see the same v.

EVALUATION ROWS: the run's test split MINUS the knockout's ranking rows (the 200
successful rows in head_knockout_preds.npz), i.e. ~2.8K of the 3K.

OUTPUTS (per run, <run>/head_set_ablation/)
  head_set_ablation.csv         one row per (head_def, condition, control, draw)
  head_set_ablation_sets.json   E, I and every ablated set, as "L<layer>H<head>"
  head_set_ablation_preds.npz   per-example predictions per config, v, a, row indices
  head_set_ablation_meta.json   provenance; written LAST (the --skip-existing sentinel)

Usage
-----
python mechanistic_analysis/score_head_set_ablation.py --config <run config>
python mechanistic_analysis/score_head_set_ablation.py --group mlp_joint_grid_layer_token \\
    --run-glob 'joint_*_kl0.1_*' --cells 2:-1,2:-28,8:-1,8:-28 --skip-existing
"""
from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from src.introspector.analysis_loading import run_per_locale
from src.introspector.head_ablation import (
    WINDOWS, Arm, HeadAblator, build_mean_cache, eval_arm_with_activation, safe_corr,
    successful_train_examples)
from src.introspector.knockout_stats import mass_set
from src.introspector.run_discovery import add_discovery_args, discover_runs, load_run, shard

ANALYSIS = "head_set_ablation"
SENTINEL = "head_set_ablation_meta.json"
KNOCKOUT = "head_knockout"
ARMS = (Arm("endogenous", None), Arm("x1", 1.0))
CONDITIONS = ("endo_only", "interv_only", "joint", "keep_endo", "keep_interv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--cells", type=str, default=None,
                   help="Comma list of 'layer:position' cells to keep after discovery, e.g. "
                        "'2:-1,2:-28,8:-1,8:-28' (positions as in the run names: -1 = t 31, "
                        "-28 = t 4). Default: every run --run-glob matches.")
    p.add_argument("--head-defs", type=str, default="mass0.8,top30",
                   help="Head set definitions: 'massQ' (fewest heads holding Q of the "
                        "positive single-head drop) and/or 'topK'.")
    p.add_argument("--n-random", type=int, default=5,
                   help="Layer- and size-matched random controls per condition.")
    p.add_argument("--window", choices=WINDOWS, default="t_to_last",
                   help="Must match the knockout the head sets come from.")
    p.add_argument("--ablation", choices=("mean", "zero"), default="mean",
                   help="Must match the knockout the head sets come from.")
    p.add_argument("--success-threshold", type=float, default=0.02,
                   help="Defines the successful TRAIN rows the mean cache is taken over.")
    p.add_argument("--mean-examples", type=int, default=500)
    p.add_argument("--mean-scan", type=int, default=4000)
    p.add_argument("--max-eval-rows", type=int, default=None,
                   help="Evaluate only the first N held-out rows (smoke tests).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Default: the run's own eval_batch_size.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the random controls (the injection RNG always uses "
                        "config.seed, to match the reported eval).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Single-run override; a sweep writes to <run>/head_set_ablation/.")
    add_discovery_args(p)
    return p.parse_args()


# ======================================================================
# Head sets (pure; unit-tested)
# ======================================================================
def parse_cells(spec: str | None) -> set[tuple[int, int]] | None:
    if not spec:
        return None
    out = set()
    for tok in (t.strip() for t in spec.split(",")):
        if tok:
            L, pos = tok.split(":")
            out.add((int(L), int(pos)))
    return out


def head_label(L: int, h: int) -> str:
    return f"L{L}H{h}"


def importance_vectors(ind_df: pd.DataFrame, heads: list[tuple[int, int]]) -> dict:
    """{arm: importance aligned to `heads`}: importance = correlation DROP (-delta)."""
    out = {}
    for arm, d in ind_df.groupby("arm"):
        s = d.set_index(["layer", "head"])["delta"]
        missing = [hd for hd in heads if hd not in s.index]
        if missing:
            raise ValueError(f"knockout table lacks {len(missing)} head(s) for arm {arm}, "
                             f"e.g. {missing[:3]}")
        out[arm] = -s.loc[heads].to_numpy(dtype=float)
    return out


def top_positive(importance: np.ndarray, k: int) -> set:
    """Indices of the k largest POSITIVE drops (fewer if fewer heads hurt when removed)."""
    imp = np.nan_to_num(importance, nan=0.0)
    order = [int(j) for j in np.argsort(-imp, kind="stable") if imp[j] > 0]
    return set(order[:k])


def build_head_set(importance: np.ndarray, definition: str) -> set:
    if definition.startswith("mass"):
        return mass_set(importance, float(definition[4:]))
    if definition.startswith("top"):
        return top_positive(importance, int(definition[3:]))
    raise SystemExit(f"Unknown head set definition {definition!r}; use 'massQ' or 'topK'.")


def condition_sets(E: set, I: set, n_heads: int) -> dict[str, set]:
    """Condition -> the head indices it ABLATES."""
    every = set(range(n_heads))
    return {"endo_only": E - I, "interv_only": I - E, "joint": E & I,
            "keep_endo": every - E, "keep_interv": every - I}


def kept_set(condition: str, E: set, I: set) -> set | None:
    return {"keep_endo": E, "keep_interv": I}.get(condition)


def layer_matched_random(chosen: set, layers: np.ndarray, rng) -> set:
    """A random head set with exactly `chosen`'s number of heads in every layer."""
    out = set()
    for L in np.unique(layers[list(chosen)]) if chosen else []:
        pool = np.where(layers == L)[0]
        n = int(sum(layers[j] == L for j in chosen))
        out.update(int(j) for j in rng.choice(pool, size=n, replace=False))
    return out


def random_control(condition: str, ablated: set, E: set, I: set, layers: np.ndarray,
                   rng) -> set:
    """Ablated set of one random control: for keep_* the KEPT set is randomized (same
    per-layer counts as E or I) and everything else is ablated."""
    kept = kept_set(condition, E, I)
    if kept is None:
        return layer_matched_random(ablated, layers, rng)
    return set(range(len(layers))) - layer_matched_random(kept, layers, rng)


# ======================================================================
# Measurement
# ======================================================================
def load_knockout(run_dir: Path, args):
    ko = run_dir / KNOCKOUT
    paths = {k: ko / f"head_knockout_{k}" for k in ("individual.csv", "preds.npz", "meta.json")}
    for p in paths.values():
        if not p.exists():
            return None, f"missing {p}"
    meta = json.loads(paths["meta.json"].read_text())
    for key in ("ablation", "window"):
        if meta.get(key) != getattr(args, key):
            raise SystemExit(f"{run_dir.name}: knockout used {key}={meta.get(key)!r}, this run "
                             f"asked for {getattr(args, key)!r}; the head sets would not mean "
                             f"what they say")
    return {"individual": pd.read_csv(paths["individual.csv"]),
            "npz": np.load(paths["preds.npz"]), "meta": meta}, None


def set_ablation_run(run, spec, args, out_dir: Path):
    t0 = time.time()
    cfg = run.base.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    bs = args.batch_size or run.config.training_config.eval_batch_size

    ko, why = load_knockout(spec.run_dir, args)
    if ko is None:
        print(f"[set-ablation] {spec.name}: no knockout results ({why}); run "
              f"mechanistic_analysis/score_head_knockout.py first. Skipping.")
        return

    heads = [(L, h) for L in range(run.layer + 1, n_layers) for h in range(n_heads)]
    ko_heads = [tuple(int(x) for x in hd) for hd in ko["npz"]["heads"]]
    if ko_heads != heads:
        raise SystemExit(f"{spec.name}: knockout covered {len(ko_heads)} heads, expected "
                         f"{len(heads)} (layers {run.layer + 1}..{n_layers - 1})")
    layers = np.array([L for L, _ in heads])
    imp = importance_vectors(ko["individual"], heads)
    if set(imp) != {"endogenous", "x1"}:
        raise SystemExit(f"{spec.name}: knockout arms {sorted(imp)}, need endogenous and x1")

    # ---- held-out rows: everything the rankings were NOT chosen on ----
    ranked = np.asarray(ko["npz"]["examples"], dtype=int)
    acts_all = np.asarray(run.dataset["activation"], dtype=np.float32)
    if ranked.max() >= len(acts_all) or not np.allclose(
            acts_all[ranked], ko["npz"]["target_endogenous"], atol=1e-6):
        raise SystemExit(f"{spec.name}: the knockout's ranking rows do not line up with "
                         f"this test split (different cache or preprocessing?)")
    eval_idx = np.setdiff1d(np.arange(len(acts_all)), ranked)
    if args.max_eval_rows is not None:
        eval_idx = eval_idx[:args.max_eval_rows]
    dataset = run.dataset.select(eval_idx.tolist())
    print(f"[set-ablation] {len(heads)} candidate heads (layers {run.layer + 1}-"
          f"{n_layers - 1}); {len(eval_idx)} held-out rows (test {len(acts_all)} minus "
          f"{len(ranked)} ranking rows)")

    # ---- head sets and the configurations to evaluate ----
    rng = np.random.default_rng(args.seed + 1000 * run.layer + run.index)
    defs = [d.strip() for d in args.head_defs.split(",") if d.strip()]
    sets_json, configs = {}, [{"head_def": "none", "condition": "none", "control": "real",
                               "draw": 0, "ablated": set(), "kept": None}]
    for d in defs:
        E, I = build_head_set(imp["endogenous"], d), build_head_set(imp["x1"], d)
        conds = condition_sets(E, I, len(heads))
        sets_json[d] = {"E": sorted(head_label(*heads[j]) for j in E),
                        "I": sorted(head_label(*heads[j]) for j in I),
                        **{c: sorted(head_label(*heads[j]) for j in s) for c, s in conds.items()}}
        print(f"[set-ablation] {d}: |E|={len(E)} |I|={len(I)} |E&I|={len(E & I)} "
              f"|E-I|={len(E - I)} |I-E|={len(I - E)}")
        for c in CONDITIONS:
            ablated, kept = conds[c], kept_set(c, E, I)
            empty = (len(kept) == 0) if kept is not None else (len(ablated) == 0)
            base = {"head_def": d, "condition": c, "n_E": len(E), "n_I": len(I)}
            configs.append({**base, "control": "real", "draw": 0, "ablated": ablated,
                            "kept": kept, "empty": empty})
            if empty:
                continue
            for r in range(args.n_random):
                configs.append({**base, "control": "random", "draw": r, "kept": kept,
                                "ablated": random_control(c, ablated, E, I, layers, rng)})

    # ---- evaluate (identical ablated sets are evaluated once) ----
    ablator = HeadAblator(run, n_heads, head_dim, mode=args.ablation)
    memo, ref_targets = {}, {}
    try:
        mean_info = {}
        if args.ablation == "mean":
            cache_ds, mean_info = successful_train_examples(run, args)
            ablator.means = build_mean_cache(run, cache_ds, bs, n_layers)

        def evaluate(ablated: set):
            key = frozenset(ablated)
            if key not in memo:
                ablator.set_heads([heads[j] for j in sorted(ablated)])
                try:
                    out = {a.name: eval_arm_with_activation(run, dataset, a, ablator, bs,
                                                            args.window) for a in ARMS}
                finally:
                    ablator.set_heads([])
                for name, (_, tgt, _) in out.items():
                    # the pairing every comparison relies on: same rows, same injected v
                    ref = ref_targets.setdefault(name, tgt)
                    if not np.array_equal(ref, tgt):
                        raise RuntimeError(f"{name}: targets changed between passes -- the "
                                           f"injection sampler is not reproducible")
                memo[key] = out
            return memo[key]

        n_eval = sum(not c.get("empty") for c in configs)
        print(f"[set-ablation] {n_eval} configurations ({len(configs) - n_eval} empty), "
              f"{len(ARMS)} arms each")
        rows, P = [], {a.name: [] for a in ARMS}
        for k, c in enumerate(configs):
            row = {k2: v for k2, v in c.items() if k2 not in ("ablated", "kept")}
            row.update(n_ablated=len(c["ablated"]),
                       n_kept=len(heads) - len(c["ablated"]), empty=bool(c.get("empty")))
            if c.get("empty"):
                row.update(corr_endogenous=np.nan, corr_interv=np.nan,
                           corr_natural_under_injection=np.nan,
                           pred_std_endogenous=np.nan, pred_std_x1=np.nan)
                for a in P:
                    P[a].append(np.full(len(eval_idx), np.nan, dtype=np.float32))
            else:
                out = evaluate(c["ablated"])
                pe, ae, _ = out["endogenous"]
                px, v, a_nat = out["x1"]
                row.update(corr_endogenous=safe_corr(pe, ae), corr_interv=safe_corr(px, v),
                           corr_natural_under_injection=safe_corr(px, a_nat),
                           pred_std_endogenous=float(np.std(pe)), pred_std_x1=float(np.std(px)))
                P["endogenous"].append(pe.astype(np.float32))
                P["x1"].append(px.astype(np.float32))
            rows.append(row)
            if c["control"] == "real":
                print(f"[set-ablation] {c['head_def']:>8} {c['condition']:<12} "
                      f"ablated {row['n_ablated']:>3}  endo {row['corr_endogenous']:+.3f}  "
                      f"interv {row['corr_interv']:+.3f}  natural|inj "
                      f"{row['corr_natural_under_injection']:+.3f}"
                      + ("  (empty set)" if row["empty"] else ""))
    finally:
        ablator.remove()

    df = pd.DataFrame(rows)
    df.insert(0, "run", spec.name)
    df.insert(1, "signal", f"N{run.layer}.{run.index}")
    df.insert(2, "signal_layer", run.layer)
    df.insert(3, "position", spec.position)
    df.to_csv(out_dir / "head_set_ablation.csv", index=False)
    (out_dir / "head_set_ablation_sets.json").write_text(json.dumps(sets_json, indent=2))
    np.savez_compressed(
        out_dir / "head_set_ablation_preds.npz", examples=eval_idx,
        configs=np.array([f"{c['head_def']}|{c['condition']}|{c['control']}|{c['draw']}"
                          for c in configs]),
        injected=ref_targets["x1"], activation=ref_targets["endogenous"],
        **{f"pred_{a}": np.stack(v) for a, v in P.items()})
    meta = {"run": spec.name, "checkpoint": str(run.checkpoint),
            "signal": f"N{run.layer}.{run.index}", "signal_layer": run.layer,
            "position": spec.position, "head_defs": defs, "conditions": list(CONDITIONS),
            "n_random": args.n_random, "ablation": args.ablation, "window": args.window,
            "n_candidate_heads": len(heads), "n_eval_rows": int(len(eval_idx)),
            "n_ranking_rows_excluded": int(len(ranked)), "max_eval_rows": args.max_eval_rows,
            "mean_cache": mean_info, "batch_size": bs, "n_forward_configs": len(memo),
            "knockout_baseline": ko["meta"].get("baseline"),
            "seconds": round(time.time() - t0, 1)}
    (out_dir / SENTINEL).write_text(json.dumps(meta, indent=2))
    print(f"[set-ablation] wrote {out_dir} ({meta['seconds']:.0f}s)")


def select_runs(args):
    """--config, or --group/--run-glob narrowed to --cells BEFORE sharding, so the array
    splits only the runs it will actually analyze."""
    if args.config:
        return [load_run(args.config, output_root=args.output_root)]
    runs = discover_runs(group=args.group, run_glob=args.run_glob,
                         config_root=args.config_root, output_root=args.output_root)
    cells = parse_cells(args.cells)
    if cells is not None:
        runs = [r for r in runs if (r.layer, r.position) in cells]
        print(f"[set-ablation] {len(runs)} run(s) in cells {sorted(cells)}")
    return shard(runs, args.task_index, args.n_tasks)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    runs = select_runs(args)
    if not runs:
        raise SystemExit("No runs with checkpoints matched.")
    run_per_locale(args, runs, ANALYSIS, SENTINEL,
                   analyze=lambda run, spec, out_dir: set_ablation_run(run, spec, args, out_dir),
                   log="[set-ablation]")


if __name__ == "__main__":
    main()
