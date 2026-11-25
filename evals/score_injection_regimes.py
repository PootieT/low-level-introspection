#!/usr/bin/env python3
"""
Does interventional reporting hold up under interventions the model was NOT trained on?

The joint-loss runs are trained on context-independent REPLACEMENTS of the neuron drawn
uniformly over 10x its natural range ('extreme'), and the paper's interventional eval
is the same kind of replacement (a gaussian at the natural std). So a reviewer can ask
whether the joint model learned to read the site, or only to undo the training
intervention. This script scores ONE trained run under several regimes, the same way
for SFT and joint runs, on the run's own test split, so the two can be compared regime
by regime:

  endogenous   no injection. corr(prediction, cached activation): the paper's
               endogenous number (up to the prompt-template seeding, see below).
  xN           replace the neuron at (L, i, t) with v ~ N(mean, N * std) of its TRAIN
               activations, through the model's own sampler re-seeded to config.seed,
               exactly as the reported eval does. corr(prediction, v). x1-x10 stay inside
               the training range; x30 / x100 put part of the draws outside it
               (frac_outside_train_support, per arm).
  local@Ds     add delta = D * std * z, z ~ N(0, 1), to each example's LIVE activation --
               the per-example centring of the gating-heads experiment
               (mechanistic_analysis/trace_mover_head.py: centres from
               capture_live_centres, written back with the training InterventionHook).
               Context-DEPENDENT, unlike anything in training. Scored WITHIN example:
                   corr(pred_local - pred_clean, delta_realized)
               A model that ignores the site scores ~0, one that reads it ~1. The pooled
               corr(pred, centre + delta) is NOT used: a model that ignores the site
               already gets ~0.7 from the spread of the natural value alone. When the
               prediction does not move at all the correlation is undefined and is
               recorded as 0.

CHECKS, printed and stored in the summary:
  * delta = 0 identity: writing the live value back must reproduce the clean
    prediction (src/introspector/regimes.identity_check). Its max |dpred| is the
    numerical floor the local arms are read against.
  * reported numbers: the run's own endogenous / gaussian / scale-sweep correlations
    are stored next to ours. They agree only approximately: the analysis loader seeds
    each row's prompt template from the row, the trainer's eval did not. Within this
    script every run -- SFT and joint -- sees the same rows with the same prompts.

OUTPUTS (per run, <run>/injection_regimes/)
  injection_regimes.csv           one row per arm: corr, n, frac_outside_train_support,
                                  dpred_std (local), identity floor
  injection_regimes_summary.json  the same, plus the checks and the reported numbers
  injection_regimes_preds.npz     every per-example prediction / target / delta
  injection_regimes_meta.json     provenance; written LAST (the --skip-existing sentinel)

Usage
-----
python evals/score_injection_regimes.py --config <run config>
python evals/score_injection_regimes.py --group mlp_joint_grid_layer_token \\
    --run-glob 'joint_*_kl0.1_*' --cells 2:-1,2:-28,8:-1,8:-28 --skip-existing
python evals/score_injection_regimes.py --group mlp_endo_sweep_layer_token \\
    --run-glob 'endo_*_gateoff_w1.0_300K' --cells 2:-1,2:-28,8:-1,8:-28 \\
    --neurons 2.1447,2.1886,2.2566,2.5949,2.6998,8.3852,8.4144,8.4956,8.5777,8.6370
"""
from __future__ import annotations

import sys
import json
import time
import argparse
import operator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from src.introspector.analysis_loading import absolute_positions, run_per_locale
from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.regimes import Regime, identity_check
from src.introspector.run_discovery import add_discovery_args, discover_runs, load_run, shard
from src.introspector.training.model import InterventionHook
from mechanistic_analysis.trace_introspection_circuit import capture_live_centres

ANALYSIS = "injection_regimes"
SENTINEL = "injection_regimes_meta.json"
DEFAULT_ARMS = "x1,x3,x10,x30,x100,local@1,local@0.25"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--arms", type=str, default=DEFAULT_ARMS,
                   help="Comma list of 'xN' (gaussian replacement at N x the std) and "
                        "'local@D' (each example's live value + D x std x z). The "
                        "endogenous arm is always run.")
    p.add_argument("--cells", type=str, default=None,
                   help="Comma list of 'layer:position' cells kept after discovery, e.g. "
                        "'2:-1,2:-28,8:-1,8:-28' (positions as in the run names: -1 = t 31, "
                        "-28 = t 4). Default: every run --run-glob matches.")
    p.add_argument("--neurons", type=str, default=None,
                   help="Comma list of 'layer.index' neurons kept after discovery, e.g. "
                        "'2.1447,8.3852'. The SFT sweep has 10 neurons per cell and the "
                        "joint grid 5, so the SFT arm passes the joint grid's neurons.")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Contiguous prefix of the test split (default: all of it).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Default: the run's own eval_batch_size.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Single-run override; a sweep writes to <run>/injection_regimes/.")
    add_discovery_args(p)
    return p.parse_args()


# ======================================================================
# Arms and pure helpers (unit-tested in tests/test_evals.py)
# ======================================================================
@dataclass(frozen=True)
class Arm:
    name: str
    kind: str          # "endogenous" | "global" | "local"
    sigma: float       # std multiplier: the gaussian's spread (global) or delta's (local)


def parse_arms(spec: str) -> list[Arm]:
    """'x1,local@0.25' -> [endogenous, x1, local@0.25s]; endogenous always comes first."""
    arms = [Arm("endogenous", "endogenous", 0.0)]
    for tok in (t.strip().lower() for t in spec.split(",")):
        if not tok or tok in ("endogenous", "endo"):
            continue
        if tok.startswith("local"):
            d = float(tok.split("@", 1)[1]) if "@" in tok else 1.0
            if d <= 0:
                raise SystemExit(f"{tok!r}: a local arm needs a positive delta")
            arms.append(Arm(f"local@{d:g}s", "local", d))
        elif tok.startswith("x"):
            arms.append(Arm(f"x{float(tok[1:]):g}", "global", float(tok[1:])))
        else:
            raise SystemExit(f"Unknown arm {tok!r}; use 'xN' or 'local@D'.")
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate arms in {spec!r}")
    return arms


def parse_cells(spec: str | None) -> set[tuple[int, int]] | None:
    if not spec:
        return None
    return {(int(L), int(pos)) for L, pos in
            (tok.strip().split(":") for tok in spec.split(",") if tok.strip())}


def parse_neurons(spec: str | None) -> set[tuple[int, int]] | None:
    if not spec:
        return None
    return {(int(L), int(i)) for L, i in
            (tok.strip().split(".") for tok in spec.split(",") if tok.strip())}


def corr_or_zero(x: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    """(Pearson r, degenerate). A constant input has no correlation; it is scored 0 and
    flagged -- for a local arm a prediction that never moves IS the 'ignores the site'
    outcome, so dropping it would hide exactly the result being tested for."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0, True
    return float(np.corrcoef(x, y)[0, 1]), False


def train_support(stats: dict, distribution: str, extreme_scale: float) -> tuple[float, float] | None:
    """[lo, hi] of the TRAINING injection distribution for this neuron, or None when it
    is unbounded/unknown. 'extreme' = uniform over extreme_scale x the empirical range,
    centred on its midpoint (IntrospectionRegressionModel.sample_intervention_values)."""
    if distribution == "extreme":
        c = (stats["min"] + stats["max"]) / 2
        h = extreme_scale * (stats["max"] - stats["min"]) / 2
        return c - h, c + h
    if distribution == "uniform":
        return stats["min"], stats["max"]
    return None


def frac_outside(values: np.ndarray, support: tuple[float, float] | None) -> float:
    if support is None:
        return float("nan")
    lo, hi = support
    v = np.asarray(values)
    return float(((v < lo) | (v > hi)).mean())


# ======================================================================
# Forward passes (mirror the reported evals)
# ======================================================================
@torch.no_grad()
def predict(run, dataset, batch_size: int, scale: float | None = None,
            local_values: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(predictions, targets) over `dataset`, in order.

    scale=None, local_values=None: clean forward; targets are the cached activations.
    scale: gaussian replacement through the model's OWN sampler, re-seeded to
           config.seed so every run and every pass draws the same sequence; targets are
           the injected values.
    local_values: one value per row written at (L, i, t) with the training
           InterventionHook on the act_fn output (as trace_mover_head does); targets are
           the cached activations (unused by the caller).
    """
    assert scale is None or local_values is None
    model = run.model
    model.eval()
    prev = (model.intervention_distribution, model.intervention_scale)
    if scale is not None:
        model.intervention_distribution = "gaussian"
        model.intervention_scale = float(scale)
        model.seed_intervention_rng(run.config.seed)
    site = operator.attrgetter(resolve_mlp_signal_path(model, run.layer, path_to_output=False))(model)
    head = model.regression_head
    preds, targets, row0 = [], [], 0
    try:
        for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=run.collator,
                                shuffle=False):
            batch = batch.to(model.device)
            act = batch.pop("activation")
            batch.pop("anchor_region_mask", None)   # bookkeeping, not a model input
            batch.pop("labels", None)               # token labels: an LM loss we do not need
            n = batch["input_ids"].shape[0]
            handle = None
            if local_values is not None:
                rows = torch.arange(n, device=model.device)
                hook = InterventionHook(
                    row_indices=rows,
                    token_positions=absolute_positions(batch["token_position"],
                                                       batch["attention_mask"]),
                    signal_indices=torch.full((n,), run.index, device=model.device,
                                              dtype=torch.long),
                    intervention_values=local_values[row0:row0 + n].to(model.device))
                handle = site.register_forward_hook(hook)
            try:
                out = model(**batch, do_intervention=scale is not None,
                            output_hidden_states=True)
            finally:
                if handle is not None:
                    handle.remove()
            h = out.hidden_states[-1][:, -1, :]
            preds.append(head(h.to(device=head.weight.device, dtype=head.weight.dtype))
                         .squeeze(-1).float().cpu())
            targets.append(out.intervention_values.float().cpu() if scale is not None
                           else act.float().cpu())
            row0 += n
    finally:
        model.intervention_distribution, model.intervention_scale = prev
    if local_values is not None and row0 != len(local_values):
        raise RuntimeError(f"{row0} rows evaluated but {len(local_values)} local values")
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def reported_numbers(run_dir: Path) -> dict:
    """The run's own eval numbers, for the side-by-side check (None when absent)."""
    def corr(name, key=None):
        p = run_dir / name
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return (d.get(key, d) if key else d).get("corr")
    out = {"endogenous": corr("endogenous_test_metrics_endogenous.json"),
           "x1": corr("interventional_test_metrics_gaussian.json", "test")}
    sweep = run_dir / "interventional_scale_sweep.csv"
    if sweep.exists():
        for _, r in pd.read_csv(sweep).iterrows():
            out[f"x{float(r['scale']):g}_scale_sweep"] = float(r["head_corr"])
    return out


# ======================================================================
# One run
# ======================================================================
def score_run(run, spec, args, out_dir: Path):
    t0 = time.time()
    cfg = run.config
    bs = args.batch_size or cfg.training_config.eval_batch_size
    arms = parse_arms(args.arms)
    dataset = run.dataset
    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    N = len(dataset)
    stats = run.stats
    tc = cfg.training_config
    support = train_support(stats, tc.intervention_signal_distribution,
                            tc.intervention_extreme_scale)
    print(f"[regimes] {cfg.experiment_name}: N{run.layer}.{run.index} pos={spec.position} | "
          f"{N} test rows | neuron mean={stats['mean']:.4f} std={stats['std']:.4f} | "
          f"training injections: {tc.intervention_signal_distribution} "
          f"x{tc.intervention_extreme_scale:g} -> support {support}")

    # ---- clean pass: endogenous arm, and the reference for every local delta ----
    pred_clean, act = predict(run, dataset, bs)
    rows, saved = [], {"activation": act, "pred_endogenous": pred_clean}
    r, deg = corr_or_zero(pred_clean, act)
    rows.append({"arm": "endogenous", "kind": "endogenous", "sigma": 0.0, "corr": r,
                 "degenerate": deg})
    print(f"[regimes] endogenous: corr={r:.4f}")

    # ---- global replacements: the model's own sampler, as in the reported eval ----
    for a in (a for a in arms if a.kind == "global"):
        p, v = predict(run, dataset, bs, scale=a.sigma)
        r, deg = corr_or_zero(p, v)
        out = frac_outside(v, support)
        rows.append({"arm": a.name, "kind": "global", "sigma": a.sigma, "corr": r,
                     "degenerate": deg, "frac_outside_train_support": out})
        saved[f"pred_{a.name}"], saved[f"target_{a.name}"] = p, v
        print(f"[regimes] {a.name}: corr={r:.4f} (injected std {v.std():.4f}, "
              f"{100 * out:.1f}% outside the training range)")

    # ---- local perturbations around each example's LIVE value ----
    local_arms = [a for a in arms if a.kind == "local"]
    identity_err = None
    if local_arms:
        # as trace_mover_head.prepare_inputs: the live (adapter-on) activation, not the
        # cached column, which drifts in gate-off runs
        centres = capture_live_centres(run.model, dataset, run.collator, bs,
                                       run.layer, run.index)
        inject_dtype = next(run.model.parameters()).dtype
        print(f"[regimes] live centres: mean={centres.mean():.4f} std={centres.std():.4f} "
              f"(cached mean={act.mean():.4f} std={act.std():.4f}, corr with cached "
              f"{np.corrcoef(centres.numpy(), act)[0, 1]:.3f})")
        p0, _ = predict(run, dataset, bs, local_values=centres.to(inject_dtype))
        identity_err = identity_check(Regime(name="local", kind="local", grid=np.zeros(1),
                                             delta_sigma=0.0),
                                      p0, pred_clean, label=cfg.experiment_name)
        floor_std = float(np.std(p0 - pred_clean))
        # one z per example, shared by every local arm (and by every run of this cell:
        # same seed, same rows), so the arms differ only in magnitude
        z = torch.from_numpy(np.random.default_rng(cfg.seed).standard_normal(N)).float()
        saved.update({"centres": centres.numpy(), "z": z.numpy()})
        for a in local_arms:
            # the value actually written is rounded to the model dtype; score against the
            # delta that was REALLY applied
            vals = (centres + a.sigma * stats["std"] * z).to(inject_dtype)
            delta = vals.float().numpy() - centres.numpy()
            p, _ = predict(run, dataset, bs, local_values=vals)
            dpred = p - pred_clean
            r, deg = corr_or_zero(dpred, delta)
            rows.append({"arm": a.name, "kind": "local", "sigma": a.sigma, "corr": r,
                         "degenerate": deg, "frac_outside_train_support":
                         frac_outside(vals.float().numpy(), support),
                         "dpred_std": float(dpred.std()), "identity_dpred_std": floor_std,
                         "identity_max_abs_dpred": identity_err})
            saved[f"pred_{a.name}"], saved[f"delta_{a.name}"] = p, delta
            print(f"[regimes] {a.name}: within-example corr(dpred, delta)={r:.4f} "
                  f"(dpred std {dpred.std():.2e} vs delta=0 floor {floor_std:.2e})")

    # ---- write ----
    reported = reported_numbers(spec.run_dir)
    df = pd.DataFrame(rows)
    df.insert(0, "run", cfg.experiment_name)
    df.insert(1, "group", spec.group)
    df.insert(2, "layer", run.layer)
    df.insert(3, "index", run.index)
    df.insert(4, "position", spec.position)
    df["n"] = N
    df.to_csv(out_dir / "injection_regimes.csv", index=False)
    np.savez_compressed(out_dir / "injection_regimes_preds.npz", **saved)
    summary = {"run": cfg.experiment_name, "group": spec.group,
               "signal": f"N{run.layer}.{run.index}", "position": spec.position, "n": N,
               "arms": {r["arm"]: {k: v for k, v in r.items() if k != "arm"} for r in rows},
               "train_support": support, "identity_max_abs_dpred": identity_err,
               "reported": reported}
    (out_dir / "injection_regimes_summary.json").write_text(json.dumps(summary, indent=2))
    for name in ("endogenous", "x1"):
        ours, theirs = summary["arms"].get(name, {}).get("corr"), reported.get(name)
        if ours is not None and theirs is not None:
            print(f"[regimes] {name}: ours {ours:.4f} vs reported {theirs:.4f} "
                  f"(differ by {abs(ours - theirs):.4f}; prompt templates are seeded "
                  f"per row here, so expect small gaps)")
    meta = {"run": cfg.experiment_name, "checkpoint": str(run.checkpoint),
            "arms": [a.name for a in arms], "n_examples": N, "batch_size": bs,
            "seed": cfg.seed, "seconds": round(time.time() - t0, 1)}
    (out_dir / SENTINEL).write_text(json.dumps(meta, indent=2))
    print(f"[regimes] wrote {out_dir} ({meta['seconds']:.0f}s)")


def select_runs(args):
    """--config, or --group/--run-glob narrowed to --cells / --neurons BEFORE sharding,
    so the array splits only the runs it will actually score."""
    if args.config:
        return [load_run(args.config, output_root=args.output_root)]
    runs = discover_runs(group=args.group, run_glob=args.run_glob,
                         config_root=args.config_root, output_root=args.output_root)
    cells, neurons = parse_cells(args.cells), parse_neurons(args.neurons)
    if cells is not None:
        runs = [r for r in runs if (r.layer, r.position) in cells]
    if neurons is not None:
        runs = [r for r in runs if (r.layer, r.index) in neurons]
    print(f"[regimes] {len(runs)} run(s) after --cells {args.cells} / --neurons {args.neurons}")
    return shard(runs, args.task_index, args.n_tasks)


def main():
    args = parse_args()
    parse_arms(args.arms)    # fail on a bad spec before any model loads
    runs = select_runs(args)
    if not runs:
        raise SystemExit("No runs with checkpoints matched.")
    run_per_locale(args, runs, ANALYSIS, SENTINEL,
                   analyze=lambda run, spec, out_dir: score_run(run, spec, args, out_dir),
                   log="[regimes]")


if __name__ == "__main__":
    main()
