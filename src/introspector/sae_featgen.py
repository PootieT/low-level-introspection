"""
Shared pieces of the SAE FEATURE-GENERALIZATION experiment: one introspector trained
on N SAE features at once, evaluated on features it never trained on.

Used by trainers/train_sae_detector_joint.py (multi-feature path), score_sae_feature_
generalization.py (the seen/unseen + distractor eval), data_prep/build_sae_featgen_pool.py
(the 200-feature pool) and plotters/plot_sae_feature_generalization.py.

ROW BUDGET (balance_rows_per_feature). Every run trains on the same TOTAL number of
rows, split equally over its N features (total // N each), and each feature's share
is balanced to pos_frac:

    inactive rows   SUBSAMPLED to the quota: the `quota` rows with the smallest
                    subsample_u (dataset_utils.subsample_key), so a feature's inactive
                    rows at N=100 are a subset of its rows at N=10.
    active rows     REPEATED to the quota: whole copies of every active row, then the
                    smallest-u ones for the remainder. Exact counts, unlike the integer
                    repeat factor of the single-feature oversample_positives.

Consequence worth stating wherever results are reported: distinct ACTIVE examples per
feature are fixed by the feature's density (always drawn from every cached text),
while distinct INACTIVE examples shrink as 1/N. The active:inactive ROW ratio is the
same at every N; what falls with N is how often each active example is repeated.

When a feature has fewer inactive rows than its quota (N=1: the quota is 150k and the
cache holds ~150k texts), all of them are used and the active count is scaled down to
keep pos_frac -- the single-feature recipe, which also keeps every negative.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset_utils import subsample_key


def ensure_subsample_u(ds, seed: int):
    """Add `subsample_u` if the cache was built without negative_keep (keep=None)."""
    if "subsample_u" in ds.column_names:
        return ds
    return ds.map(lambda ex: {"subsample_u": subsample_key(
        seed, ex["signal_type"], ex["signal_layer"], ex["signal_index"], ex["text"])})


def balance_rows_per_feature(ds, features: list[int], total_rows: int, pos_frac: float,
                             seed: int):
    """Resample a multi-feature detection train split to the row budget.

    ds        HF dataset with signal_index, activation, subsample_u columns
    features  the trained feature indices, in pool order (all must be present)
    Returns (resampled dataset, shuffled; per-feature stats DataFrame).
    """
    if not 0 < pos_frac < 1:
        raise ValueError(f"pos_frac must be in (0, 1), got {pos_frac}")
    n = len(features)
    quota = total_rows // n
    if quota < 2:
        raise ValueError(f"budget {total_rows} rows is < 2 per feature at N={n}")
    sig = np.asarray(ds["signal_index"])
    act = np.asarray(ds["activation"], dtype=float)
    u = np.asarray(ds["subsample_u"], dtype=float)

    picked, stats = [], []
    for f in features:
        rows = np.flatnonzero(sig == f)
        pos = rows[act[rows] > 0]
        neg = rows[act[rows] <= 0]
        if len(pos) == 0:
            raise SystemExit(f"feature index {f} has no active rows in the train cache; it "
                             f"cannot be balanced (nor steered: no 'on' magnitude). Drop it "
                             f"from the pool -- data_prep/build_sae_featgen_pool.py's positives floor "
                             f"exists to prevent this.")
        n_pos_want = int(round(quota * pos_frac))
        n_neg_want = quota - n_pos_want
        neg_sorted = neg[np.argsort(u[neg], kind="stable")]
        n_neg = min(n_neg_want, len(neg_sorted))
        n_pos = n_pos_want if n_neg == n_neg_want else int(round(n_neg * pos_frac / (1 - pos_frac)))
        pos_sorted = pos[np.argsort(u[pos], kind="stable")]
        reps, rem = divmod(n_pos, len(pos_sorted))
        pos_rows = np.concatenate([np.tile(pos_sorted, reps), pos_sorted[:rem]])
        picked += [neg_sorted[:n_neg], pos_rows]
        stats.append({"feature": int(f), "quota": quota,
                      "active_unique": int(len(pos)), "inactive_available": int(len(neg)),
                      "active_rows": int(n_pos), "inactive_rows": int(n_neg),
                      "active_repeat": n_pos / len(pos),
                      "inactive_short": int(n_neg_want - n_neg)})
    idx = np.concatenate(picked)
    idx = idx[np.random.RandomState(seed).permutation(len(idx))]
    stats = pd.DataFrame(stats)
    short = stats[stats["inactive_short"] > 0]
    if len(short):
        print(f"[balance] {len(short)} feature(s) had fewer inactive rows than the quota and "
              f"use all of them (active rows scaled to keep pos_frac={pos_frac}); expected at "
              f"N=1, a cache-size problem at larger N")
    print(f"[balance] N={n} features x quota {quota} -> {len(idx)} rows; active repeat "
          f"min/median/max = {stats['active_repeat'].min():.1f}/"
          f"{stats['active_repeat'].median():.1f}/{stats['active_repeat'].max():.1f}")
    return ds.select(idx.tolist()), stats


def load_pool(path: str | Path) -> dict:
    """The pool file data_prep/build_sae_featgen_pool.py writes. `features` is in pool order
    (a run with N features trains the first N); `labels` covers all of them;
    `label_similarity[i][j]` is the cosine between the labels of features[i] and [j]."""
    pool = json.loads(Path(path).read_text())
    for key in ("features", "labels"):
        if key not in pool:
            raise ValueError(f"{path} is not a feature-gen pool file (no {key!r})")
    missing = [f for f in pool["features"] if f not in pool["labels"]]
    if missing:
        raise ValueError(f"{path}: {len(missing)} pool features have no label, e.g. {missing[:5]}")
    return pool


def nearest_label_similarity(pool: dict, trained: list[str]) -> dict[str, float]:
    """For every pool feature, the cosine to its most similar TRAINED feature's label
    (excluding itself). NaN when there is no other trained feature or no matrix."""
    sim = pool.get("label_similarity")
    feats = pool["features"]
    if sim is None:
        return {f: float("nan") for f in feats}
    sim = np.asarray(sim, dtype=float)
    pos = {f: i for i, f in enumerate(feats)}
    t_idx = [pos[t] for t in trained if t in pos]
    out = {}
    for f, i in pos.items():
        others = [j for j in t_idx if j != i]
        out[f] = float(sim[i, others].max()) if others else float("nan")
    return out
