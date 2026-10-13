"""Per-(dst, src) fidelity for PAIR MODE runs (experiment 3-1).

WHY NOT JUST THE POOLED CORR. A pair-mode test split mixes cells whose typical weights
differ by orders of magnitude: the BOS sink sits near 0.5-0.9, a distant ordinary token
near 1e-3. Most of the pooled variance is therefore BETWEEN pairs, and a readout that
only learned "BOS is high, far is low" -- without reading this document's attention at
all -- scores a pooled corr near 0.97. The numbers here separate the two:

  corr_within_pair_mean   corr over documents inside each pair, averaged over pairs.
                          A pair-mean-only readout scores 0.
  corr_demeaned           pooled corr after subtracting each pair's mean from both
                          prediction and target. Also 0 for a pair-mean readout.
  pair_mean_baseline_corr pooled corr of the target against that pair-mean readout
                          (means from the injection pool, i.e. train documents). The
                          pooled corr has to be read against this floor.

Distant pairs have tiny within-pair variance (weights packed near 1e-3), where a
probability-space corr is dominated by noise; logodds_corr is reported next to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import signal_range_key

PAIR_KEYS = ["signal_layer", "signal_index", "token_position", "src_token_position"]
LOGODDS_EPS = 1e-6


def _corr(x, y) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _logodds(p):
    p = np.clip(np.asarray(p, dtype=np.float64), LOGODDS_EPS, 1 - LOGODDS_EPS)
    return np.log(p / (1 - p))


def pair_breakdown(df: pd.DataFrame, value_col: str, target_col: str,
                   signal_ranges: dict | None = None):
    """(by_pair frame, summary dict) for a predictions frame carrying PAIR_KEYS."""
    missing = [k for k in PAIR_KEYS if k not in df.columns]
    assert not missing, f"pair_breakdown needs columns {missing}"
    rows = []
    for key, g in df.groupby(PAIR_KEYS):
        pred, tgt = g[value_col].to_numpy(), g[target_col].to_numpy()
        rows.append({**dict(zip(PAIR_KEYS, (int(k) for k in key))),
                     "n": int(len(g)),
                     "target_mean": float(tgt.mean()), "target_std": float(tgt.std()),
                     "pred_mean": float(pred.mean()), "pred_std": float(pred.std()),
                     "corr": _corr(pred, tgt),
                     "logodds_corr": _corr(_logodds(pred), _logodds(tgt))})
    by_pair = pd.DataFrame(rows)
    by_pair["src_offset"] = by_pair["src_token_position"] - by_pair["token_position"]

    grp = df.groupby(PAIR_KEYS)
    pred_dm = df[value_col] - grp[value_col].transform("mean")
    tgt_dm = df[target_col] - grp[target_col].transform("mean")
    finite = by_pair["corr"].notna()
    summary = {
        "n_pairs": int(len(by_pair)),
        "n_pairs_corr_undefined": int((~finite).sum()),
        "corr_pooled": _corr(df[value_col].to_numpy(), df[target_col].to_numpy()),
        "corr_within_pair_mean": float(by_pair.loc[finite, "corr"].mean()) if finite.any() else float("nan"),
        "corr_within_pair_median": float(by_pair.loc[finite, "corr"].median()) if finite.any() else float("nan"),
        "logodds_corr_within_pair_mean": float(by_pair["logodds_corr"].mean(skipna=True)),
        "corr_demeaned": _corr(pred_dm.to_numpy(), tgt_dm.to_numpy()),
    }
    if signal_ranges:
        # the readout that knows each pair's typical weight and nothing about this document
        means = []
        for l, h, d, s in df[PAIR_KEYS].itertuples(index=False):
            st = signal_ranges.get(signal_range_key(l, h, pair=(d, s)))
            means.append(np.nan if st is None else st["mean"])
        means = np.asarray(means, dtype=np.float64)
        ok = ~np.isnan(means)
        summary["pair_mean_baseline_corr"] = _corr(means[ok], df[target_col].to_numpy()[ok])
        summary["pair_mean_baseline_coverage"] = float(ok.mean())
    return by_pair, summary


def attach_provenance(df: pd.DataFrame, provenance: pd.DataFrame | None) -> pd.DataFrame:
    """Add per-row provenance (doc_index, *_token_unique) to a predictions frame.

    These columns cannot ride through the collated batch: every batch key reaches
    model(**batch) in the readout extraction. So the caller builds them from the raw
    split, aligned to the preprocessed rows, and they are joined here by ROW ORDER --
    which is why the lengths must agree exactly.
    """
    if provenance is None:
        return df
    assert len(provenance) == len(df), (
        f"provenance has {len(provenance)} rows, predictions {len(df)} -- the eval "
        f"dataset and its provenance were filtered differently")
    for c in provenance.columns:
        if c not in df.columns:
            df[c] = provenance[c].to_numpy()
    return df


def write_pair_outputs(df: pd.DataFrame, value_col: str, target_col: str, model,
                       out_path) -> dict | None:
    """PAIR MODE only: write the by-pair CSV and return the summary (None otherwise)."""
    if not getattr(model, "_pair_ranges", False) or "src_token_position" not in df.columns:
        return None
    by_pair, summary = pair_breakdown(df, value_col, target_col, model.signal_ranges)
    by_pair.to_csv(out_path, index=False)
    print(f"[pairs] {out_path.name}: {summary['n_pairs']} pairs, "
          f"within-pair corr {summary['corr_within_pair_mean']:.4f}, "
          f"demeaned {summary['corr_demeaned']:.4f}, pooled {summary['corr_pooled']:.4f}, "
          f"pair-mean baseline {summary.get('pair_mean_baseline_corr', float('nan')):.4f}")
    return summary
