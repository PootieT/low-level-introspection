"""Unit tests for the prompted baseline's weighted ridge probe (score_prompt_baseline.py)."""
import numpy as np
import torch
from sklearn.linear_model import Ridge

import score_prompt_baseline as spb


def _blocks(n=500, H=24, seed=0):
    rng = np.random.RandomState(seed)
    Xe = torch.tensor(rng.randn(n, H) * 3 + 1, dtype=torch.float16)
    Xi = torch.tensor(rng.randn(n, H) * 2 - 1, dtype=torch.float16)
    ye = torch.tensor(Xe.float().numpy() @ rng.randn(H) + rng.randn(n), dtype=torch.float32)
    yi = torch.tensor(Xi.float().numpy() @ rng.randn(H) * 5 + rng.randn(n), dtype=torch.float32)
    return [(Xe, ye, 0.3), (Xi, yi, 0.7)]


def test_chunked_weighted_ridge_matches_sklearn(monkeypatch):
    """The two-block, variance-normalized objective, accumulated over many row chunks,
    is sklearn's weighted ridge on the same standardized features."""
    monkeypatch.setattr(spb, "LINALG_CHUNK", 37)
    blocks = _blocks()
    n = blocks[0][0].shape[0]
    tr, val = spb.split_indices(n, 0.1, 0)
    probe, diag = spb.fit_probe(blocks, tr, val, lambdas=np.array([1e-3, 1e-1, 10.0]))

    idx = np.concatenate([tr, val])
    Xtr = np.concatenate([b[0].double().numpy()[tr] for b in blocks])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Z = np.concatenate([(b[0].double().numpy()[idx] - mu) / sd for b in blocks])
    Y = np.concatenate([b[1].double().numpy()[idx] for b in blocks])
    w = np.concatenate([np.full(len(idx), wb / (len(idx) * b[1].double().numpy()[idx].var(ddof=1)))
                        for b in blocks for wb in [b[2]]])
    ref = Ridge(alpha=probe.lam).fit(Z, Y, sample_weight=w / w.sum())

    X_new = torch.tensor(np.random.RandomState(1).randn(50, Z.shape[1]), dtype=torch.float16)
    np.testing.assert_allclose(probe.predict(X_new),
                               ref.predict((X_new.double().numpy() - mu) / sd), atol=1e-5)
    assert diag["n_fit"] == n


def test_penalty_selection_prefers_shrinkage_on_noise():
    """With targets independent of the features, the held-out objective should pick the
    largest penalty -- the probe falls back to predicting the mean."""
    rng = np.random.RandomState(2)
    X = torch.tensor(rng.randn(400, 32), dtype=torch.float16)
    y = torch.tensor(rng.randn(400), dtype=torch.float32)
    tr, val = spb.split_indices(400, 0.25, 0)
    probe, diag = spb.fit_probe([(X, y, 1.0)], tr, val, lambdas=np.array([1e-4, 1e-2, 1e2]))
    assert probe.lam == 1e2 and diag["lambda_at_edge"]
