"""trainers/: every training script loads the way scripts/ runs it, and the post-training
bookkeeping helpers (eval breakdowns, on/off intervention draws, linear probes) behave.

No model is loaded; the end-to-end gated-LoRA check that needs Llama-3.2-1B lives in
test_unchanged_signal_with_gated_lora.py.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from script_smoke import run_smoke, scripts_in, smoke_failure

TRAINERS = scripts_in("trainers")


@pytest.fixture(scope="module")
def smoke():
    return run_smoke(TRAINERS)


@pytest.mark.parametrize("path", TRAINERS, ids=lambda p: p.name)
def test_script_loads(smoke, path):
    proc = smoke[path]
    assert proc.returncode == 0, smoke_failure(proc)


def test_text_len_from_mask_handles_left_padding_and_both_mask_names():
    from trainers.train_introspector_joint import _text_len_from_mask

    # row 0: 2 left pads, 3 text tokens, then the prompt; row 1: no pad, 5 text tokens
    attention = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]])
    prompt = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1]])
    ids = torch.zeros_like(attention)
    for key in ("lora_token_mask", "anchor_region_mask"):     # gate on / gate off
        out = _text_len_from_mask({"input_ids": ids, "attention_mask": attention, key: prompt})
        assert out.tolist() == [3, 5]
    no_prompt = {"input_ids": ids, "attention_mask": attention,
                 "lora_token_mask": torch.zeros_like(prompt)}
    assert _text_len_from_mask(no_prompt).tolist() == [-1, -1]
    assert _text_len_from_mask({"input_ids": ids, "attention_mask": attention}).tolist() == [-1, -1]


def test_per_signal_metrics_reports_rank_measures_per_neuron():
    from trainers.train_introspector_joint import _per_signal_metrics

    rng = np.random.default_rng(0)
    act = rng.normal(size=400)
    df = pd.DataFrame({"activation": act, "predictions": act + 0.1 * rng.normal(size=400),
                       "signal_layer": 10, "signal_index": np.repeat([24, 99], 200)})
    df.loc[df["signal_index"] == 99, "predictions"] = rng.normal(size=200)   # unreadable one
    assert _per_signal_metrics(df[df["signal_index"] == 24]) is None         # single signal
    per = _per_signal_metrics(df).set_index("signal_index")
    assert list(per.index) == [24, 99] and (per["n"] == 200).all()
    assert per.loc[24, "corr"] > 0.95 and abs(per.loc[99, "corr"]) < 0.2
    for k in ("spearman", "kendall_tau", "monotone_r2"):                    # default eval
        assert per.loc[24, k] > 0.8, k


def test_multi_on_off_sampler_uses_each_features_own_range():
    from trainers.train_sae_detector_joint import MultiOnOffSampler

    raw = {"signal_index": [1] * 4 + [2] * 4,
           "activation": [0.0, 1.0, 2.0, 3.0, 0.0, 10.0, 20.0, 30.0]}
    s = MultiOnOffSampler.from_rows(raw, distribution="extreme", seed=0)
    draws = {f: [s.draw(f) for _ in range(2000)] for f in (1, 2)}
    for f, (lo, hi) in {1: (1.0, 3.0), 2: (10.0, 30.0)}.items():
        on = [v for is_on, v in draws[f] if is_on]
        assert 0.45 < len(on) / 2000 < 0.55
        assert lo <= min(on) and max(on) <= hi
        assert all(v == 0.0 for is_on, v in draws[f] if not is_on)
    with pytest.raises(SystemExit):
        MultiOnOffSampler.from_rows({"signal_index": [3, 3], "activation": [0.0, 0.0]})


def test_fit_ridge_recovers_a_linear_readout():
    from trainers.train_introspector_with_intervention import fit_ridge

    g = torch.Generator().manual_seed(0)
    X = torch.randn(500, 8, generator=g)
    w = torch.randn(8, generator=g)
    probe = fit_ridge(X, X @ w + 0.5, lam=1e-6)
    torch.testing.assert_close(probe.coef, w, atol=1e-3, rtol=0)
    assert abs(probe.intercept - 0.5) < 1e-3
    torch.testing.assert_close(probe.predict(X[:5]), X[:5] @ w + 0.5, atol=1e-3, rtol=0)
