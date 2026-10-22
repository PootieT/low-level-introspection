"""mechanistic_analysis/: every script loads the way scripts/ runs it, and the readout
decomposition the circuit traces rest on adds up.

mechanistic_analysis/trace_introspection_circuit.py attributes the regression readout to
residual writes through a FROZEN final RMSNorm. Every per-layer and per-head number
downstream is only meaningful if those pieces sum back to the prediction, so that
identity is checked here on random tensors. The knockout statistics have their own file
(test_knockout_stats.py).
"""
import numpy as np
import pytest
import torch

from script_smoke import run_smoke, scripts_in, smoke_failure

MECH = scripts_in("mechanistic_analysis")


@pytest.fixture(scope="module")
def smoke():
    return run_smoke(MECH)


@pytest.mark.parametrize("path", MECH, ids=lambda p: p.name)
def test_script_loads(smoke, path):
    proc = smoke[path]
    assert proc.returncode == 0, smoke_failure(proc)


def test_frozen_norm_contributions_sum_to_the_prediction():
    from mechanistic_analysis.trace_introspection_circuit import contribution, readout_scale

    g = torch.Generator().manual_seed(0)
    N, H, eps, bias = 6, 16, 1e-5, 0.3
    writes = torch.randn(5, N, H, generator=g, dtype=torch.float64)    # 5 residual writers
    resid = writes.sum(0)
    w, gamma = torch.randn(H, generator=g, dtype=torch.float64), torch.rand(H, generator=g,
                                                                          dtype=torch.float64)
    rms = resid.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    pred = (gamma * resid / rms) @ w + bias                           # RMSNorm -> linear head
    a = readout_scale(resid, w, gamma, eps)
    total = sum(contribution(a, writes[i]) for i in range(len(writes)))
    torch.testing.assert_close(total, pred - bias)


def test_per_head_contributions_sum_to_the_attention_write():
    from mechanistic_analysis.trace_introspection_circuit import (contribution,
                                                                  per_head_contributions)

    g = torch.Generator().manual_seed(1)
    N, n_heads, head_dim, H = 4, 3, 5, 12
    z = torch.randn(N, n_heads * head_dim, generator=g)
    W_O = torch.randn(H, n_heads * head_dim, generator=g)
    a = torch.randn(N, H, generator=g)
    per_head = per_head_contributions(z, W_O, n_heads, head_dim, a)
    assert per_head.shape == (N, n_heads)
    torch.testing.assert_close(per_head.sum(1), contribution(a, z @ W_O.T), atol=1e-4, rtol=1e-4)


def test_effective_weight_includes_the_lora_update():
    from peft import LoraConfig, get_peft_model

    from mechanistic_analysis.trace_introspection_circuit import effective_weight

    torch.manual_seed(0)
    base = torch.nn.Sequential(torch.nn.Linear(8, 6, bias=False))
    assert torch.equal(effective_weight(base[0]), base[0].weight.detach())
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["0"],
                                            init_lora_weights=False))
    layer = model.base_model.model[0]
    x = torch.randn(3, 8)
    torch.testing.assert_close(x @ effective_weight(layer).T, layer(x).detach())
    assert not torch.allclose(effective_weight(layer), layer.base_layer.weight.detach())


def test_linfit_slope_r2():
    from mechanistic_analysis.trace_introspection_circuit import linfit_slope_r2

    x = np.linspace(0, 1, 50)
    assert linfit_slope_r2(x, 3 * x + 1) == pytest.approx((3.0, 1.0))
    assert linfit_slope_r2(np.ones(5), np.arange(5.0)) == (0.0, 0.0)   # no spread in x
