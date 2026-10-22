"""Unit tests for the multi-feature SAE (feature-generalization) plumbing."""
import random

import numpy as np
import pytest
import torch
from datasets import Dataset

from src.introspector.dataset_utils import _split_identity, subsample_key
from src.introspector.config import DatasetConfig
from src.introspector.prompts import sae_label_prompt, sae_prompt
from src.introspector.sae_featgen import balance_rows_per_feature, nearest_label_similarity
from src.introspector.signals.sae import SAEBundle


def _rows(spec, n_texts=2000, seed=0):
    """spec: {feature: active fraction}. One row per (text, feature), like a cache."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n_texts):
        text = f"text {t}"
        for f, p in spec.items():
            a = float(rng.uniform(1, 5)) if rng.random() < p else 0.0
            out.append({"text": text, "signal_type": "sae", "signal_layer": 9, "signal_index": f,
                        "activation": a, "subsample_u": subsample_key(42, "sae", 9, f, text)})
    return Dataset.from_list(out)


def test_subsample_key_is_content_keyed_and_nested():
    a = subsample_key(42, "sae", 9, 7, "hello")
    assert a == subsample_key(42, "sae", 9, 7, "hello")
    assert a != subsample_key(42, "sae", 9, 8, "hello")
    assert a != subsample_key(43, "sae", 9, 7, "hello")
    u = np.array([subsample_key(0, "sae", 9, 1, f"t{i}") for i in range(20000)])
    assert 0 <= u.min() and u.max() < 1 and abs(u.mean() - 0.5) < 0.01
    keep_lo, keep_hi = set(np.flatnonzero(u < 0.05)), set(np.flatnonzero(u < 0.2))
    assert keep_lo < keep_hi


def test_balance_exact_quota_and_pos_frac():
    ds = _rows({1: 0.05, 2: 0.01, 3: 0.2})
    out, stats = balance_rows_per_feature(ds, [1, 2, 3], total_rows=3000, pos_frac=0.5, seed=0)
    assert len(out) == 3000
    sig, act = np.array(out["signal_index"]), np.array(out["activation"])
    for f in (1, 2, 3):
        assert (sig == f).sum() == 1000
        assert (act[sig == f] > 0).sum() == 500
    # active rows are repeated, never invented
    assert set(stats["active_unique"]) == {int(((np.array(ds["signal_index"]) == f)
                                               & (np.array(ds["activation"]) > 0)).sum()) for f in (1, 2, 3)}


def test_balance_inactive_rows_are_nested_across_n():
    ds = _rows({1: 0.05, 2: 0.05, 3: 0.05})

    def inactive_texts(feats, total):
        out, _ = balance_rows_per_feature(ds, feats, total, 0.5, seed=0)
        s, a, t = np.array(out["signal_index"]), np.array(out["activation"]), np.array(out["text"])
        return set(t[(s == 1) & (a <= 0)])

    big, small = inactive_texts([1], 2000), inactive_texts([1, 2, 3], 2000)
    assert small < big and len(small) > 0


def test_balance_short_on_negatives_keeps_pos_frac():
    ds = _rows({1: 0.1}, n_texts=1000)
    out, stats = balance_rows_per_feature(ds, [1], total_rows=10000, pos_frac=0.5, seed=0)
    n_neg = int((np.array(ds["activation"]) <= 0).sum())
    a = np.array(out["activation"])
    assert (a <= 0).sum() == n_neg and (a > 0).sum() == n_neg   # all negatives, matched actives
    assert stats["inactive_short"].iloc[0] > 0


def test_balance_refuses_a_feature_without_actives():
    ds = _rows({1: 0.05, 2: 0.0}, n_texts=500)
    with pytest.raises(SystemExit):
        balance_rows_per_feature(ds, [1, 2], 1000, 0.5, seed=0)


class _LinearSAE(torch.nn.Module):
    def __init__(self, d=8, k=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W_enc = torch.randn(d, k, generator=g)
        self.W_dec = torch.randn(k, d, generator=g)
        self.b_dec = torch.randn(d, generator=g)

    def encode(self, x):
        return torch.relu(x @ self.W_enc)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec


def test_steer_delta_per_row_index_matches_scalar_index():
    sae = _LinearSAE()
    b = SAEBundle(sae=sae, release="r", sae_id="i", site="res", width="16k", layer=9,
                  hook_name="h", d_sae=16, device="cpu", dtype=torch.float32)
    resid = torch.randn(5, 8)
    idx = torch.tensor([3, 0, 7, 3, 15])
    vals = torch.tensor([1.0, 2.0, 0.0, 4.0, 0.5])
    per_row = b.steer_delta(resid, idx, vals)
    looped = torch.stack([b.steer_delta(resid[r:r + 1], int(idx[r]), vals[r])[0] for r in range(5)])
    torch.testing.assert_close(per_row, looped)
    # delta lies along the steered feature's decoder row, scaled by (target - current)
    f = sae.encode(resid)
    want = (vals - f[torch.arange(5), idx]).unsqueeze(1) * sae.W_dec[idx]
    torch.testing.assert_close(per_row, want, atol=1e-5, rtol=1e-5)


def test_label_prompt_differs_from_index_prompt_only_in_the_name(monkeypatch):
    labels = {"S9.12": "references to dogs"}
    for k in range(4):
        monkeypatch.setattr(random, "choice", lambda seq, k=k: seq[k])
        lab = sae_label_prompt(9, 12, "dog", 5, labels=labels)
        idx = sae_prompt(9, 12, "dog", 5)
        if k < 3:
            assert lab.replace('SAE feature "references to dogs"', "SAE feature 12") == idx
        else:
            assert lab.replace('"references to dogs" (layer 9)', "9.12") == idx
    with pytest.raises(KeyError):
        sae_label_prompt(9, 13, "dog", 5, labels=labels)


def test_negative_keep_only_enters_the_cache_identity_when_set():
    base = DatasetConfig(name="allenai/c4", split="train", max_samples=10)
    assert "negative_keep" not in _split_identity(base)
    assert _split_identity(base.model_copy(update={"negative_keep": 0.1}))["negative_keep"] == 0.1


def test_greedy_dedupe_drops_paraphrases_but_keeps_forced():
    from data_prep.build_sae_featgen_pool import greedy_dedupe
    e = {1: np.array([1.0, 0.0]), 2: np.array([0.99, 0.141]), 3: np.array([0.0, 1.0]),
         4: np.array([0.0, 1.0])}
    labels = {1: "dogs", 2: "canines", 3: "numbers", 4: "Numbers!"}
    kept, dropped = greedy_dedupe([1, 2, 3, 4], labels, e, 0.9, size=10, forced=set())
    assert kept == [1, 3] and {d["feature"] for d in dropped} == {2, 4}
    kept, _ = greedy_dedupe([1, 2, 3], labels, e, 0.9, size=10, forced={2})
    assert kept == [1, 2, 3]


def test_multi_sampler_draws_from_each_features_own_range():
    from trainers.train_sae_detector_joint import MultiOnOffSampler
    s = MultiOnOffSampler({1: np.array([1.0, 2.0]), 2: np.array([10.0, 20.0])},
                          distribution="extreme", extreme_scale=1.0, seed=0)
    for f, (lo, hi) in ((1, (1, 2)), (2, (10, 20))):
        on = [v for on_, v in (s.draw(f) for _ in range(400)) if on_]
        assert on and min(on) >= lo and max(on) <= hi
    assert s.samplers[1].rng is s.samplers[2].rng


def test_natural_precision_matches_full_population():
    from evals.score_sae_feature_generalization import natural_precision
    # 100 actives (TPR .8), 900 inactives in the population, FPR .1; eval set kept 90 inactives
    m = {"tpr": 0.8, "ina_pred_active": 9, "ina_pred_inactive": 81, "ina_pred_neither": 0}
    assert natural_precision(m, base_rate=0.1) == pytest.approx(80 / (80 + 90))


def test_nearest_label_similarity_excludes_self():
    pool = {"features": ["S9.1", "S9.2", "S9.3"],
            "label_similarity": [[1, .5, .2], [.5, 1, .9], [.2, .9, 1]]}
    out = nearest_label_similarity(pool, ["S9.1", "S9.2"])
    assert out == {"S9.1": .5, "S9.2": .5, "S9.3": .9}
