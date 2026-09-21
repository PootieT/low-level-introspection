#!/usr/bin/env python3
"""
One place that knows how to find a run and read its metric files.

Every sweep writes the same per-run files (interventional/endogenous test metrics,
activation_drift.json, lm_degradation.json) but names its runs differently: the
layer/token grid encodes the locale in the directory name, the length sweep encodes
train length and window, the neuron and layer sweeps carry a run_spec.json sidecar.
This module normalises all of them into one frame with a `family` column, so an
analysis that cuts across sweeps (KL against number of locales, say) reads one table.

Nothing here aggregates. Aggregation belongs to the analysis that owns the question.
"""
from __future__ import annotations

import re
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
RUNS = ROOT / "outputs" / "runs"

NEURON_RE = re.compile(r"N(?P<layer>\d+)\.(?P<index>\d+)")
POS_RE = re.compile(r"POS(?P<pos>-?\d+)")
GATE_RE = re.compile(r"gate(?P<gate>on|off)")
W_RE = re.compile(r"_w(?P<w>[0-9.]+)")
KL_RE = re.compile(r"_kl(?P<kl>[0-9.a-z]+)")
LEN_RE = re.compile(r"_len(?P<len>\d+)_win(?P<win>\d+)")


def _js(d: Path, fname: str):
    f = d / fname
    if not f.exists():
        sub = sorted(d.glob(f"diag_*/{fname}"))
        f = sub[-1] if sub else None
    if f is None:
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def read_metrics(d: Path) -> dict:
    """Flat dict of every per-run number the cross-sweep analyses use. Missing files
    leave their columns absent rather than raising, so a partial sync still loads."""
    out = {}
    interv = _js(d, "interventional_test_metrics_gaussian.json")
    endo = _js(d, "endogenous_test_metrics_endogenous.json")
    if interv:
        out["interventional_corr"] = interv.get("test", {}).get("corr")
    if endo:
        out["endogenous_corr"] = endo.get("corr")
        # multi-locale runs: the macro average is the honest number, pooled is inflated
        if "corr_macro" in endo:
            out["endogenous_corr_macro"] = endo["corr_macro"]

    drift = _js(d, "activation_drift.json")
    if drift:
        out["alignment_ok"] = drift.get("alignment_ok")
        for k, blk in (("drift_vs_base", "drift_tuned_vs_base"),
                       ("drift_vs_cached", "drift_tuned_vs_cached")):
            b = drift.get(blk) or {}
            out[f"{k}_corr"] = b.get("corr")
            out[f"{k}_std_ratio"] = b.get("std_ratio")
        rt = drift.get("readout_tracking") or {}
        for k in ("corr_pred_cached", "corr_pred_tuned_live", "corr_pred_base_live", "coupling_gap"):
            out[k] = rt.get(k)

    kl = _js(d, "lm_degradation.json")
    if kl:
        out["kl_negligible"] = kl.get("negligible")
        for e in kl.get("lengths", []):
            L = e.get("length")
            out[f"kl@{L}"] = e.get("kl_mean")
            out[f"kl_p95@{L}"] = e.get("kl_p95")
            out[f"top1@{L}"] = e.get("top1_agreement")
            if e.get("ppl_base"):
                out[f"ppl_ratio@{L}"] = e["ppl_tuned"] / e["ppl_base"]
    out["has_kl"] = kl is not None
    out["has_drift"] = drift is not None
    return out


def grid_runs(exp_group: str, gate: str | None = "off") -> pd.DataFrame:
    """The layer/token grid: one locale per run, encoded in the directory name."""
    rows = []
    root = RUNS / exp_group
    if not root.exists():
        return pd.DataFrame()
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        n, p = NEURON_RE.search(d.name), POS_RE.search(d.name)
        if not (n and p):
            continue
        g, mw, mkl = GATE_RE.search(d.name), W_RE.search(d.name), KL_RE.search(d.name)
        if gate and (not g or g["gate"] != gate):
            continue
        rows.append({
            "family": "grid", "group": exp_group, "run": d.name, "config_hint": d.name,
            "layer": int(n["layer"]), "index": int(n["index"]),
            "neuron": f"N{n['layer']}.{n['index']}", "position": int(p["pos"]),
            "gate": g["gate"] if g else None,
            "w": float(mw["w"]) if mw else None,
            "kl_tag": mkl["kl"] if mkl else "none",
            "n_locales": 1, **read_metrics(d),
        })
    return pd.DataFrame(rows)


def length_runs(exp_group: str = "mlp_lengen", diagonal_only: bool = True) -> pd.DataFrame:
    """The length sweep. A run's locales are the positions in its training window, so
    n_locales = window. The DIAGONAL (window == train_len) is the one where every
    position of the sequence was trained, which is the clean multi-locale comparison."""
    rows = []
    root = RUNS / exp_group
    if not root.exists():
        return pd.DataFrame()
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m, n = LEN_RE.search(d.name), NEURON_RE.search(d.name)
        if not (m and n):
            continue
        L, W = int(m["len"]), int(m["win"])
        if diagonal_only and L != W:
            continue
        rows.append({
            "family": "length", "group": exp_group, "run": d.name, "config_hint": d.name,
            "layer": int(n["layer"]), "index": int(n["index"]),
            "neuron": f"N{n['layer']}.{n['index']}",
            "train_len": L, "window": W, "n_locales": W,
            "label": f"positions={W}", **read_metrics(d),
        })
    return pd.DataFrame(rows)


def _sidecar_runs(exp_group: str, family: str, keep) -> pd.DataFrame:
    rows = []
    root = RUNS / exp_group
    if not root.exists():
        return pd.DataFrame()
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        spec = d / "run_spec.json"
        if not spec.exists():
            continue
        j = json.loads(spec.read_text())
        if not keep(j):
            continue
        rows.append({"family": family, "group": exp_group, "run": d.name,
                     "config_hint": d.name, "spec": j, **read_metrics(d)})
    return pd.DataFrame(rows)


def neuron_runs(exp_group: str = "mlp_neurongen") -> pd.DataFrame:
    """Neuron sweep, restricted BY CONFIG to cells that were given enough data to learn.

    Selection is on the configuration, never on the outcome: the fixed-budget cells at
    N=30 and N=100 sit below the ~30k rows/neuron floor and learned nothing, and a
    'capacity used' number for a run that learned nothing is not informative. That is a
    known fact about the design, not a threshold on a result.
    """
    def keep(j):
        g, n, ep = j["group"], int(j["neurons"]), float(j.get("epochs", 1))
        return ((g == "scaling" and n in (1, 3, 10))
                or (g == "data30k" and n == 30 and ep == 1 and int(j.get("offset", 0)) == 0))
    df = _sidecar_runs(exp_group, "neuron", keep)
    if df.empty:
        return df
    df["n_locales"] = df["spec"].map(lambda j: int(j["neurons"]))
    df["rows_per_locale"] = df["spec"].map(lambda j: int(j["rows_per_neuron"]))
    df["layer"] = df["spec"].map(lambda j: int(j.get("layer", 4)))
    df["label"] = df.apply(lambda r: f"neurons={r.n_locales} @{r.rows_per_locale // 1000}k", axis=1)
    return df.drop(columns=["spec"])


def layer_runs(exp_group: str = "mlp_layergen") -> pd.DataFrame:
    """Layer sweep, by config: the cells at or above the per-layer data floor."""
    def keep(j):
        g, n = j["group"], int(j["n_layers"])
        return (g == "scaling" and n in (1, 2)) or (g == "data300k")
    df = _sidecar_runs(exp_group, "layer", keep)
    if df.empty:
        return df
    df["n_locales"] = df["spec"].map(lambda j: int(j["n_layers"]))
    df["rows_per_locale"] = df["spec"].map(lambda j: int(j["rows_per_locale"]))
    df["train_layers"] = df["spec"].map(lambda j: tuple(j["train_layers"]))
    df["label"] = df.apply(lambda r: f"layers={r.n_locales} @{r.rows_per_locale // 1000}k", axis=1)
    return df.drop(columns=["spec"])


def multi_locale_runs() -> pd.DataFrame:
    frames = [f for f in (length_runs(), neuron_runs(), layer_runs()) if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def missing_kl_commands(df: pd.DataFrame, config_root: str = "experiment_configs") -> list[str]:
    """Shell lines to score the runs that have no lm_degradation.json yet."""
    if df.empty or "has_kl" not in df:
        return []
    cmds = []
    for _, r in df[~df["has_kl"].fillna(False)].iterrows():
        cmds.append(f"python score_lm_degradation.py --config {config_root}/{r['group']}/{r['run']}.json")
    return cmds
