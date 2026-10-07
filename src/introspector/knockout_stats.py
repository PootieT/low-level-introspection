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
# Mass sets: the heads that carry most of an arm's total drop
# ======================================================================
def mass_set(importance: np.ndarray, q: float) -> set:
    """Smallest set of heads whose POSITIVE drops sum to >= q of the arm's total positive
    drop, taken in order of drop. Heads whose knockout helps never enter."""
    pos = np.clip(np.nan_to_num(importance), 0, None)
    tot = pos.sum()
    if tot <= 0:
        return set()
    order = np.argsort(-pos, kind="stable")
    n = int(np.searchsorted(np.cumsum(pos[order]) / tot, q - 1e-12) + 1)
    return set(order[:n].tolist())


def overlap_coefficient(a: set, b: set) -> float:
    """|A & B| / min(|A|, |B|): 1 when the smaller set sits inside the larger."""
    return len(a & b) / min(len(a), len(b)) if a and b else np.nan


def mass_overlap_null(imp_a: np.ndarray, imp_b: np.ndarray, layers: np.ndarray, q: float,
                      n_perm: int = 1000, seed: int = 0) -> dict:
    """
    Overlap coefficient of the two arms' q-mass sets, against a LAYER-MATCHED null (arm
    B's drops shuffled among heads of the same layer, so depth agreement earns nothing).

    The null matters more here than for top-k: the sets are not size-matched, and when
    one is large the min-normalized overlap approaches 1 by chance -- at q=0.95 the
    endogenous set can hold most heads, and the null then sits near 0.7.
    """
    A, B = mass_set(imp_a, q), mass_set(imp_b, q)
    obs = overlap_coefficient(A, B)
    rng = np.random.default_rng(seed)
    groups = [np.where(layers == L)[0] for L in np.unique(layers)]
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuf = imp_b.copy()
        for g in groups:
            shuf[g] = imp_b[rng.permutation(g)]
        null[i] = overlap_coefficient(A, mass_set(shuf, q))
    ok = np.isfinite(null)
    return {"q": q, "n_a": len(A), "n_b": len(B), "overlap": obs,
            "null_mean": float(null[ok].mean()) if ok.any() else np.nan,
            "null_p95": float(np.percentile(null[ok], 95)) if ok.any() else np.nan,
            "p": (float(((null[ok] >= obs).sum() + 1) / (ok.sum() + 1))
                  if ok.any() and np.isfinite(obs) else np.nan)}


# ======================================================================
# Cumulative-knockout curves
# ======================================================================
def curve_area(drops: np.ndarray) -> float:
    """Mean drop over k = 1..K: how fast an ordering destroys the eval."""
    return float(np.nanmean(drops))


def transfer_ratios(area: dict, arms: list, random_orders: list) -> dict:
    """
    transfer_{b}_to_{a} = area(b's ordering, evaluated on a) / max over the two arms'
    orderings of their area on a. 1 = b's top heads break a at least as well as a's own
    ranking does. Normalizing by the BETTER of the two (not by a's own) keeps it in
    [0, 1]: a's own single-head ranking is not a ceiling -- when a's single drops are tiny
    and redundant, the other arm's ranking can break a several times faster.
    `area` is keyed (order_name, arm); random orderings are averaged.
    """
    out = {}
    for a in arms:
        best = max(area.get((f"by_{b}", a), np.nan) for b in arms)
        if not np.isfinite(best) or best <= 0:
            continue
        for b in arms:
            if b != a:
                out[f"transfer_{b}_to_{a}"] = area[(f"by_{b}", a)] / best
        out[f"transfer_own_to_{a}"] = area[(f"by_{a}", a)] / best
        if random_orders:
            out[f"transfer_random_to_{a}"] = float(
                np.mean([area[(r, a)] for r in random_orders])) / best
    return out


def participation_ratio(importance: np.ndarray, mode: str = "pos") -> float:
    """(sum s)^2 / sum s^2 over heads: a threshold-free effective head count.

    mode="pos": s = max(drop, 0), the heads whose knockout HURTS -- the same sign
                convention as every ranking here (top-k sets, cumulative orderings).
    mode="abs": s = |drop|, also counting heads whose knockout IMPROVES the eval. In the
                interventional arm those are most heads (removing any head's text-
                dependent write takes noise out of a prediction whose target ignores the
                text), so this mode measures how much the eval is touched at all, not how
                many heads carry it.
    """
    x = np.nan_to_num(importance)
    s = np.clip(x, 0, None) if mode == "pos" else np.abs(x)
    q = (s ** 2).sum()
    return float(s.sum() ** 2 / q) if q > 0 else np.nan


def ranks_desc(importance: np.ndarray) -> np.ndarray:
    """1 = most important."""
    return rankdata(-np.nan_to_num(importance, nan=-np.inf), method="min").astype(int)


# ======================================================================
# Everything that needs only the per-head and curve TABLES
# ======================================================================
def agreement_from_tables(ind_df, curves_df, mass_qs=(0.5, 0.8, 0.95), ks=(5, 10, 20),
                          n_perm: int = 1000, seed: int = 0) -> dict:
    """
    The cross-arm metrics computable from head_knockout_individual.csv and
    head_knockout_curves.csv alone, so the per-run summary and the sweep aggregator
    compute them identically (and old results can be re-summarized without the GPU).

    Sign convention throughout: importance = -delta, and a head is "important" when its
    knockout HURTS (positive importance). Top-k sets, mass sets and the cumulative
    orderings all rank by this signed value; effective head counts are reported both
    positive-only (the same convention) and on |drop|.
    """
    import pandas as pd  # noqa: F401  (tables in, plain dict out)
    w = ind_df.pivot_table(index=["layer", "head"], columns="arm", values="delta")
    arms = [a for a in ("endogenous", "x1") if a in w.columns] + \
           [a for a in w.columns if a not in ("endogenous", "x1")]
    layers = w.index.get_level_values("layer").to_numpy()
    imp = {a: -w[a].to_numpy(dtype=float) for a in arms}
    out = {}
    for a in arms:
        out[f"effective_heads_pos_{a}"] = participation_ratio(imp[a], "pos")
        out[f"effective_heads_abs_{a}"] = participation_ratio(imp[a], "abs")
        out[f"n_helped_by_knockout_{a}"] = int((imp[a] < 0).sum())
        for q in mass_qs:
            out[f"mass{q:g}_size_{a}"] = len(mass_set(imp[a], q))
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            out[f"topk_{a}_vs_{b}"] = [
                topk_jaccard_null(imp[a], imp[b], layers, k, n_perm=n_perm, seed=seed)
                for k in ks if k < len(layers)]
            out[f"mass_overlap_{a}_vs_{b}"] = [
                mass_overlap_null(imp[a], imp[b], layers, q, n_perm=n_perm, seed=seed)
                for q in mass_qs]
    if curves_df is not None and len(curves_df):
        area = {(o, a): curve_area(-g.sort_values("k")["delta"].to_numpy())
                for (o, a), g in curves_df.groupby(["order_by", "arm"])}
        out["curve_area"] = {f"{o}|{a}": v for (o, a), v in area.items()}
        rand = sorted({o for o, _ in area if o.startswith("random")})
        out.update(transfer_ratios(area, arms, rand))
    return out


# ======================================================================
# Which arms' top-k a head belongs to
# ======================================================================
TOP_CATEGORIES = ("top endogenous only", "top both", "top interventional only")


def top_k_arms(ko_df, k: int) -> dict:
    """{(layer, head): [arms]} for every head in at least one arm's top k by single-
    knockout drop (rank 1 = largest drop, so heads whose knockout helps never enter)."""
    out = {}
    for arm, g in ko_df.groupby("arm", sort=False):
        for r in g.nsmallest(k, "rank").itertuples():
            out.setdefault((int(r.layer), int(r.head)), []).append(arm)
    return out


def top_category(arms) -> str:
    """'top endogenous only' / 'top both' / 'top interventional only' / 'other'."""
    s = set(arms or ())
    if {"endogenous", "x1"} <= s:
        return "top both"
    if "endogenous" in s:
        return "top endogenous only"
    if "x1" in s:
        return "top interventional only"
    return "other"
