"""Unit tests for the single-head knockout statistics and token window."""
import numpy as np
import torch

from src.introspector.head_ablation import window_mask
from src.introspector.knockout_stats import (
    rowwise_corr, knockout_deltas, bootstrap_deltas, bootstrap_summary, split_half,
    topk_jaccard_null, participation_ratio, ranks_desc)


def _synthetic(n=200, n_heads=64, important=(3, 17), seed=0):
    rng = np.random.default_rng(seed)
    t = rng.normal(size=n)
    p0 = t + 0.1 * rng.normal(size=n)
    eff = np.zeros(n_heads)
    eff[list(important)] = 0.4
    P = np.stack([(1 - e) * p0 + e * rng.normal(size=n) for e in eff])
    return P, p0, t


def test_rowwise_corr_matches_numpy():
    rng = np.random.default_rng(1)
    P, t = rng.normal(size=(5, 50)), rng.normal(size=50)
    expected = [np.corrcoef(p, t)[0, 1] for p in P]
    np.testing.assert_allclose(rowwise_corr(P, t), expected, atol=1e-12)


def test_untouched_head_has_zero_delta_and_no_significance():
    P, p0, t = _synthetic()
    P[0] = p0                                    # a knockout that changes nothing
    d = knockout_deltas(P, p0, t)
    assert d[0] == 0
    bs = bootstrap_summary(bootstrap_deltas(P, p0, t, n_boot=200))
    # paired: the identical row has a zero-width distribution, not a wide one
    assert abs(bs["ci_lo"][0]) < 1e-12 and abs(bs["ci_hi"][0]) < 1e-12
    assert not bs["significant"][0]
    assert bs["significant"][3] and bs["significant"][17]


def test_split_half_recovers_shared_ranking():
    P, p0, t = _synthetic(n_heads=32, important=(1, 2, 3, 4, 5, 6))
    P2, p02, t2 = P.copy(), p0.copy(), t.copy()
    out = split_half({"a": P, "b": P2}, {"a": p0, "b": p02}, {"a": t, "b": t2}, n_splits=20)
    assert out["reliability_a"] > 0
    # identical arms: cross-half agreement equals within-arm agreement
    np.testing.assert_allclose(out["disattenuated_a_vs_b"], 1.0, atol=1e-9)


def test_topk_null_is_layer_matched():
    layers = np.repeat([0, 1], 8)
    imp_a = np.r_[np.arange(8, 0, -1), np.zeros(8)]      # layer 0 matters
    imp_b = imp_a.copy()
    r = topk_jaccard_null(imp_a, imp_b, layers, k=3, n_perm=200)
    assert r["jaccard"] == 1.0
    # shuffling within layer 0 still lands top-3 in layer 0, so the null is well above
    # a uniform draw from 16 heads (~0.11)
    assert r["null_mean"] > 0.15 and r["p"] < 0.05


def test_participation_ratio_and_ranks():
    assert participation_ratio(np.array([1.0, 0, 0])) == 1.0
    assert participation_ratio(np.array([1.0, 1, 1, 1])) == 4.0
    assert list(ranks_desc(np.array([0.1, 0.5, -0.2]))) == [2, 1, 3]


def test_window_mask_accounts_for_left_padding():
    am = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    tp = torch.tensor([1, 3])                    # row 0: two left pads -> t at index 3
    assert window_mask("t_to_last", am, tp).int().tolist() == [[0, 0, 0, 1, 1, 1],
                                                               [0, 0, 0, 1, 1, 1]]
    assert window_mask("t", am, tp).int().tolist() == [[0, 0, 0, 1, 0, 0],
                                                       [0, 0, 0, 1, 0, 0]]
    assert window_mask("all", am, tp) is None
