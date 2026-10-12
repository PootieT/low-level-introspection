#!/usr/bin/env python3
"""
Rank-based (non-linear) fidelity metrics for runs that have already been evaluated.

Pearson r assumes the readout is a LINEAR function of the activation. These do not:

  spearman      monotone, rank-based. Immune to how far an outlying prediction lies,
                only to where it ranks.
  kendall_tau   tau-b, which handles TIES explicitly. Post-SiLU activations pile up at
                the floor -- 15-36% distinct values across this project's runs -- and
                Spearman's tie correction is cruder. Smaller in magnitude than r or rho
                by construction: rank runs with it, do not read it as a fraction.
  monotone_r2   cross-fit isotonic R^2: the variance a best-case MONOTONE readout could
                explain, on the same scale as Pearson r^2. Where r^2 ~= monotone_r2 the
                relationship really is linear and nothing was being lost.

NO MODEL, NO CHECKPOINT, NO GPU. Every eval already wrote its per-example predictions
next to its metrics (endogenous_test_predictions_*.csv, interventional_test_predictions_*.csv),
so this recomputes from those files. It therefore covers the OOD arms too, for free, and
runs on a laptop over a synced tree.

WHAT THE GAP MEANS. On healthy runs all three agree to ~0.05. Where they diverge the
cause in this project is not curvature: the gate-ON arm scores Pearson 0.33 against
Spearman 0.72 because a handful of predictions land an order of magnitude outside the
activation's range (predictions -3.8..5.5 for activations -0.28..0.73); trimming 1% of
the prediction extremes restores Pearson to 0.71. `pred_out_of_range_frac` is reported
beside the metrics so that reading is available rather than inferred.

Usage
-----
    python score_rank_metrics.py --group mlp_endo_sweep_layer_token
    python score_rank_metrics.py --group mlp_joint_grid_layer_token --update-metrics-json
    python score_rank_metrics.py --group mlp_lengen --run-glob 'lengen_*win16*'
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.introspector.run_discovery import (  # noqa: E402
    add_discovery_args, discover_runs, load_run, REPO_ROOT,
)
from src.introspector.training.metrics import rank_metrics  # noqa: E402

# (readout, prediction file stem, target column). The suffix after the stem is the eval
# arm -- "_endogenous", "_gaussian", "_ood_<corpus>", "_len<N>", "_neurongen", ...
PRED_FILES = (
    ("endogenous", "endogenous_test_predictions", "activation"),
    ("interventional", "interventional_test_predictions", "injected"),
)


def _arm(path: Path, stem: str) -> str:
    """'endogenous_test_predictions_ood_c4_bn.csv' -> 'ood_c4_bn'."""
    return path.stem[len(stem):].lstrip("_") or "default"


def score_one(path: Path, target_col: str) -> dict | None:
    """Rank metrics for one predictions CSV, or None if it cannot be scored."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if "predictions" not in df.columns or target_col not in df.columns:
        return {"error": f"missing column(s); has {list(df.columns)[:6]}"}
    p = pd.to_numeric(df["predictions"], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(p) & np.isfinite(a)
    n_drop = int((~ok).sum())
    p, a = p[ok], a[ok]
    if len(a) < 2:
        return {"error": f"only {len(a)} finite row(s)"}
    if float(np.std(a)) < 1e-12:
        # a constant target makes every correlation undefined, not poor
        return {"n": len(a), "n_nonfinite": n_drop, "degenerate_target": True,
                "activation_std": 0.0}
    from scipy.stats import pearsonr
    out = {"n": len(a), "n_nonfinite": n_drop, "degenerate_target": False,
           "pearson": float(pearsonr(p, a)[0]), "activation_std": float(np.std(a)),
           **rank_metrics(p, a)}
    out["pearson_minus_spearman"] = out["pearson"] - out["spearman"]
    # per-signal breakdown when the eval wrote one (multi-locale runs): a pooled
    # correlation over signals with different activation scales is inflated
    if {"signal_layer", "signal_index"} <= set(df.columns):
        per = []
        for (L, i), g in df.loc[ok].groupby(["signal_layer", "signal_index"]):
            gp = pd.to_numeric(g["predictions"], errors="coerce").to_numpy(float)
            ga = pd.to_numeric(g[target_col], errors="coerce").to_numpy(float)
            if len(ga) >= 2 and np.std(ga) > 1e-12:
                per.append(rank_metrics(gp, ga)["spearman"])
        if per:
            out["spearman_macro"] = float(np.mean(per))
            out["spearman_macro_std"] = float(np.std(per))
            out["n_signals"] = len(per)
    return out


def score_run(run) -> list[dict]:
    rows = []
    for readout, stem, target_col in PRED_FILES:
        for path in sorted(run.run_dir.glob(f"{stem}*.csv")):
            res = score_one(path, target_col)
            if res is None:
                continue
            rows.append({"group": run.group, "run": run.name, "layer": run.layer,
                         "index": run.index, "position": run.position,
                         "cell": run.cell, "n_locales": len(run.locales) or 1,
                         "readout": readout, "arm": _arm(path, stem),
                         "file": path.name, **res})
    return rows


def update_metrics_json(run, rows) -> int:
    """Merge the new keys into the eval's own metrics JSON, so existing loaders that
    read those files pick them up without being changed. Pearson stays under 'corr'."""
    written = 0
    for r in rows:
        if "error" in r:
            continue
        stem = ("endogenous_test_metrics" if r["readout"] == "endogenous"
                else "interventional_test_metrics")
        suffix = "" if r["arm"] == "default" else f"_{r['arm']}"
        f = run.run_dir / f"{stem}{suffix}.json"
        if not f.exists():
            continue
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        add = {k: r[k] for k in ("spearman", "spearman_p_value", "kendall_tau",
                                 "kendall_p_value", "monotone_r2",
                                 "pearson_minus_spearman", "pred_out_of_range_frac",
                                 "target_unique_frac") if k in r}
        # the interventional file nests its numbers under "test"
        if r["readout"] == "interventional" and isinstance(payload.get("test"), dict):
            payload["test"].update(add)
        else:
            payload.update(add)
        f.write_text(json.dumps(payload, indent=2))
        written += 1
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_discovery_args(ap)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where the sweep table goes (default outputs/plots/<group>/).")
    ap.add_argument("--update-metrics-json", action="store_true",
                    help="Also merge the new keys into each run's *_test_metrics_*.json.")
    args = ap.parse_args()

    if args.config:
        runs = [load_run(Path(args.config), output_root=args.output_root)]
    else:
        # require_checkpoint=False ON PURPOSE: this reads saved predictions, so a run
        # whose weights were deleted (or were never synced) is still scorable.
        runs = discover_runs(group=args.group, run_glob=args.run_glob,
                             config_root=args.config_root, output_root=args.output_root,
                             require_checkpoint=False)
    rows, no_preds, updated = [], [], 0
    for run in runs:
        r = score_run(run)
        if not r:
            no_preds.append(run.name)
            continue
        rows.extend(r)
        if args.update_metrics_json:
            updated += update_metrics_json(run, r)
    if not rows:
        sys.exit(f"[rank] no prediction CSVs found under any of {len(runs)} run(s). "
                 f"Looked for {[s for _, s, _ in PRED_FILES]} in each run_dir.")

    df = pd.DataFrame(rows)
    group = args.group or (runs[0].group if runs else "runs")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "outputs" / "plots" / group
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "rank_metrics.csv"
    df.to_csv(path, index=False)

    if no_preds:
        print(f"[rank] {len(no_preds)} run(s) had no prediction CSVs, e.g. {no_preds[:3]}")
    bad = df[df.get("error").notna()] if "error" in df else df.iloc[0:0]
    if len(bad):
        print(f"[rank] {len(bad)} file(s) unscorable:")
        for _, r in bad.head(5).iterrows():
            print(f"    {r['run'][:50]} {r['file']}: {r['error']}")
    deg = df[df.get("degenerate_target") == True] if "degenerate_target" in df else df.iloc[0:0]
    if len(deg):
        print(f"[rank] {len(deg)} file(s) have a constant target; their correlations are "
              f"undefined, not poor")

    good = df[df.get("spearman").notna()] if "spearman" in df else df.iloc[0:0]
    if len(good):
        print(f"\n[rank] {len(good)} scored eval(s) across {good['run'].nunique()} run(s):")
        summ = (good.groupby(["readout", "arm"])
                    [["pearson", "spearman", "kendall_tau", "monotone_r2",
                      "pearson_minus_spearman", "pred_out_of_range_frac"]]
                    .agg(["mean", "std"]).round(3))
        print(summ.to_string())
        flag = good[good["pearson_minus_spearman"].abs() > 0.1]
        if len(flag):
            print(f"\n[rank] {len(flag)} eval(s) where Pearson and Spearman disagree by >0.1 "
                  f"-- check pred_out_of_range_frac before reading it as non-linearity:")
            for _, r in flag.reindex(flag["pearson_minus_spearman"].abs()
                                     .sort_values(ascending=False).index).head(10).iterrows():
                print(f"    {r['run'][:46]:48s} {r['readout'][:5]}/{r['arm'][:12]:14s} "
                      f"r={r['pearson']:+.3f} rho={r['spearman']:+.3f} "
                      f"oor={r['pred_out_of_range_frac']:.3f}")
    if args.update_metrics_json:
        print(f"\n[rank] merged new keys into {updated} metrics JSON file(s)")
    print(f"\n[rank] wrote {path}")


if __name__ == "__main__":
    main()
