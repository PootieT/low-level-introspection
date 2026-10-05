"""
Statistics for single-head knockouts (score_head_knockout.py and its aggregator).

Everything works on per-example PREDICTIONS, not on summary correlations, because
every knockout is evaluated on the same examples with the same injected values: a
head's effect is a PAIRED difference, and resampling examples jointly for the
baseline and every knockout is what makes the noise floor honest.

Conventions: P is [n_heads, N] (one row of predictions per knockout), p0 [N] the
unablated predictions, t [N] the targets. delta = corr(knockout) - corr(baseline), so
a head that matters has delta < 0; importance = -delta.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr


# ======================================================================
# Correlations
# ======================================================================
def rowwise_corr(P: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pearson corr of each row of P ([..., N]) with t ([..., N], broadcast)."""
    Pc = P - P.mean(-1, keepdims=True)
    tc = t - t.mean(-1, keepdims=True)
    num = (Pc * tc).sum(-1)
    den = np.sqrt((Pc ** 2).sum(-1) * (tc ** 2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def knockout_deltas(P: np.ndarray, p0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """[n_heads] corr(knockout) - corr(baseline) on the full example set."""
    return rowwise_corr(P, t) - rowwise_corr(p0[None], t)[0]


# ======================================================================
# Paired bootstrap
# ======================================================================
def bootstrap_deltas(P: np.ndarray, p0: np.ndarray, t: np.ndarray, n_boot: int = 1000,
                     seed: int = 0, chunk: int = 100) -> np.ndarray:
    """[n_boot, n_heads] bootstrap distribution of every head's delta.

    PAIRED: each replicate draws ONE resample of example indices and recomputes the
    baseline corr and every knockout corr on it, so the variation shared by baseline
    and knockout (which examples happen to be in the set) cancels in the difference.
    """
    rng = np.random.default_rng(seed)
    N = len(t)
    out = np.empty((n_boot, P.shape[0]))
    for lo in range(0, n_boot, chunk):
        b = min(chunk, n_boot - lo)
        idx = rng.integers(0, N, size=(b, N))                    # [b, N]
        tb = t[idx]                                              # [b, N]
        base = rowwise_corr(p0[idx], tb)                         # [b]
        ko = rowwise_corr(P[:, idx], tb[None])                   # [H, b]
        out[lo:lo + b] = (ko - base[None]).T
    return out


def bootstrap_summary(boot: np.ndarray, alpha: float = 0.05) -> dict:
    """Per-head percentile CI and a one-sided bootstrap p for 'knockout hurts'.

    p_hurts = fraction of replicates with delta >= 0 (with the +1 correction), i.e.
    how often the resampled data fail to show a drop. `significant` = the upper CI
    bound is below zero.
    """
    lo = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)
    p = ((boot >= 0).sum(0) + 1) / (boot.shape[0] + 1)
    return {"ci_lo": lo, "ci_hi": hi, "p_hurts": p, "significant": hi < 0}


# ======================================================================
# Ranking reliability and cross-arm agreement
# ======================================================================
def spearman_brown(r: float) -> float:
    """Reliability of the full set from the split-half correlation r."""
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan


def split_half(P_by_arm: dict, p0_by_arm: dict, t_by_arm: dict, n_splits: int = 100,
               seed: int = 0, min_split_half: float = 0.2) -> dict:
    """
    How reproducible is each arm's head RANKING, and how much of the cross-arm
    disagreement is just that noise?

    Each split halves the examples (the SAME halves for every arm, since the arms are
    evaluated on the same examples). Within an arm, Spearman between the two halves'
    deltas is the split-half reliability (Spearman-Brown corrected to the full set).
    Cross-arm agreement is measured on opposite halves too (arm A on half 1 vs arm B on
    half 2, and the reverse), so it carries the same noise as the within-arm numbers
    and the disattenuated ratio cross / sqrt(rel_A * rel_B) compares like with like:
    ~1 means the arms rank heads as similarly as each arm agrees with itself.

    The ratio is left NaN when either arm's split-half correlation is below
    `min_split_half`: dividing by a near-zero reliability turns noise into numbers
    anywhere in (-inf, inf), and a ranking that does not reproduce itself has no
    "true" agreement to recover. Fall back on top-k overlap and transfer then.
    """
    arms = list(P_by_arm)
    rng = np.random.default_rng(seed)
    N = len(next(iter(t_by_arm.values())))
    within = {a: [] for a in arms}
    cross = {(a, b): [] for i, a in enumerate(arms) for b in arms[i + 1:]}
    for _ in range(n_splits):
        perm = rng.permutation(N)
        halves = (perm[: N // 2], perm[N // 2:])
        d = {a: [knockout_deltas(P_by_arm[a][:, h], p0_by_arm[a][h], t_by_arm[a][h])
                 for h in halves] for a in arms}
        for a in arms:
            within[a].append(_spearman(d[a][0], d[a][1]))
        for (a, b) in cross:
            cross[(a, b)].append(np.nanmean([_spearman(d[a][0], d[b][1]),
                                             _spearman(d[a][1], d[b][0])]))
    out = {"n_splits": n_splits}
    for a in arms:
        r = float(np.nanmean(within[a]))
        out[f"split_half_{a}"] = r
        out[f"reliability_{a}"] = spearman_brown(r)
    for (a, b), v in cross.items():
        c = float(np.nanmean(v))
        ra, rb = out[f"split_half_{a}"], out[f"split_half_{b}"]
        out[f"cross_half_{a}_vs_{b}"] = c
        ok = ra >= min_split_half and rb >= min_split_half
        out[f"disattenuated_{a}_vs_{b}"] = c / np.sqrt(ra * rb) if ok else np.nan
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(spearmanr(x[ok], y[ok])[0])


# ======================================================================
# Top-k overlap against a layer-matched null
# ======================================================================
def top_k(importance: np.ndarray, k: int) -> set:
    """Indices of the k most important heads (largest importance = largest drop)."""
    order = np.argsort(-np.nan_to_num(importance, nan=-np.inf), kind="stable")
    return set(order[:k].tolist())


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else np.nan


def topk_jaccard_null(imp_a: np.ndarray, imp_b: np.ndarray, layers: np.ndarray, k: int,
                      n_perm: int = 1000, seed: int = 0) -> dict:
    """
    Observed top-k Jaccard between two importance vectors, against a LAYER-MATCHED null:
    arm B's importances are shuffled among the heads of the same layer, so the null
    keeps "which layers matter" and only randomizes "which heads within them". A
    uniform null would credit two arms for agreeing on depth alone.
    """
    obs = jaccard(top_k(imp_a, k), top_k(imp_b, k))
    rng = np.random.default_rng(seed)
    groups = [np.where(layers == L)[0] for L in np.unique(layers)]
    ta = top_k(imp_a, k)
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuf = imp_b.copy()
        for g in groups:
            shuf[g] = imp_b[rng.permutation(g)]
        null[i] = jaccard(ta, top_k(shuf, k))
    return {"k": k, "jaccard": obs, "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p": float(((null >= obs).sum() + 1) / (n_perm + 1))}


# ======================================================================
# Cumulative-knockout curves
# ======================================================================
def curve_area(drops: np.ndarray) -> float:
    """Mean drop over k = 1..K: how fast an ordering destroys the eval."""
    return float(np.nanmean(drops))


def participation_ratio(importance: np.ndarray) -> float:
    """(sum|s|)^2 / sum s^2 over heads: a threshold-free effective head count."""
    s = np.abs(np.nan_to_num(importance))
    q = (s ** 2).sum()
    return float(s.sum() ** 2 / q) if q > 0 else np.nan


def ranks_desc(importance: np.ndarray) -> np.ndarray:
    """1 = most important."""
    return rankdata(-np.nan_to_num(importance, nan=-np.inf), method="min").astype(int)
