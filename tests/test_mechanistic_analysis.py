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


def test_signal_subspace_principal_is_sign_invariant():
    """The write-in's sign flips with up_proj(x) across texts; the principal direction
    must recover the axis where the signed mean cancels."""
    import numpy as np
    from mechanistic_analysis.trace_signal_subspace import principal, orthobasis
    rng = np.random.default_rng(0)
    axis = rng.normal(size=64); axis /= np.linalg.norm(axis)
    signs = np.r_[np.ones(10), -np.ones(10)]
    X = signs[:, None] * axis[None] * rng.uniform(0.5, 2, 20)[:, None]
    v, e = principal(X)
    assert abs(abs(v @ axis) - 1) < 1e-6 and abs(e - 1) < 1e-6
    assert np.linalg.norm(X.mean(0)) < 0.5 * np.linalg.norm(X, axis=1).mean()
    U = orthobasis(np.vstack([axis, axis, rng.normal(size=64)]))
    assert U.shape == (64, 2) and np.allclose(U.T @ U, np.eye(2), atol=1e-8)


def test_subspace_patch_is_exact():
    """h <- h - (h - h_clean) U U^T inside the window: a full basis restores h_clean,
    an empty basis changes nothing, and positions outside the window are untouched."""
    import torch
    from mechanistic_analysis.trace_signal_subspace import SubspacePatch
    torch.manual_seed(0)
    B, S, d = 2, 5, 8
    h, h0 = torch.randn(B, S, d), torch.randn(B, S, d)
    mask = torch.zeros(B, S); mask[:, 2:] = 1                  # window = positions 2..4
    full = SubspacePatch(torch.eye(d), h0, mask, is_tuple=False)(None, None, h)
    assert torch.allclose(full[:, 2:], h0[:, 2:], atol=1e-6)
    assert torch.equal(full[:, :2], h[:, :2])
    empty = SubspacePatch(torch.zeros(d, 0), h0, mask, is_tuple=True)(None, None, (h, "x"))
    assert torch.equal(empty[0], h) and empty[1] == "x"
    # a 1-d subspace removes exactly the difference's component along it
    u = torch.zeros(d, 1); u[3] = 1
    one = SubspacePatch(u, h0, mask, is_tuple=False)(None, None, h)
    exp = h.clone(); exp[:, 2:, 3] = h0[:, 2:, 3]
    assert torch.allclose(one, exp, atol=1e-6)


def test_head_set_conditions_partition_and_keep():
    from mechanistic_analysis.score_head_set_ablation import condition_sets, kept_set
    E, I = {0, 1, 2, 3}, {2, 3, 4}
    c = condition_sets(E, I, n_heads=8)
    assert c["endo_only"] == {0, 1} and c["interv_only"] == {4} and c["joint"] == {2, 3}
    assert c["keep_endo"] == {4, 5, 6, 7} and c["keep_interv"] == {0, 1, 5, 6, 7}
    assert kept_set("keep_endo", E, I) == E and kept_set("joint", E, I) is None


def test_head_set_definitions_use_positive_drops_only():
    from mechanistic_analysis.score_head_set_ablation import build_head_set, parse_cells
    imp = np.array([0.5, -0.2, 0.3, 0.0, 0.2, np.nan])
    assert build_head_set(imp, "top2") == {0, 2}
    assert build_head_set(imp, "top30") == {0, 2, 4}           # only 3 heads hurt
    assert build_head_set(imp, "mass0.8") == {0, 2}            # 0.8 of 1.0 total
    assert parse_cells("2:-1, 8:-28") == {(2, -1), (8, -28)} and parse_cells(None) is None


def test_random_controls_match_per_layer_counts():
    from mechanistic_analysis.score_head_set_ablation import random_control
    layers = np.repeat([3, 4, 5], 6)                              # 18 heads, 3 layers
    E, I = {0, 1, 6}, {7, 12}
    rng = np.random.default_rng(0)
    count = lambda s: np.bincount(layers[sorted(s)] - 3, minlength=3).tolist()
    for _ in range(20):
        r = random_control("endo_only", E - I, E, I, layers, rng)
        assert count(r) == count(E - I)
        kept = set(range(18)) - random_control("keep_endo", set(range(18)) - E, E, I, layers, rng)
        assert count(kept) == count(E)


def test_write_channel_ablator_partitions_the_write():
    """At the window only: removing the native and the added write sums to removing the
    whole write, other positions are untouched, and base_write reproduces the base model's
    write of the neuron (base u, base W_down column)."""
    from types import SimpleNamespace
    from peft import LoraConfig, get_peft_model
    from mechanistic_analysis.score_write_channel_ablation import WriteChannelAblator
    from mechanistic_analysis.trace_introspection_circuit import effective_weight

    torch.manual_seed(0)
    d, f, j = 8, 12, 3

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(d, f, bias=False)
            self.up_proj = torch.nn.Linear(d, f, bias=False)
            self.down_proj = torch.nn.Linear(f, d, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP()

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Body()

    root = Root()
    get_peft_model(root, LoraConfig(r=2, lora_alpha=4, target_modules=["up_proj", "down_proj"],
                                    init_lora_weights=False))
    mlp = root.model.layers[0].mlp
    ab = WriteChannelAblator(SimpleNamespace(base=root), layer=0, index=j)
    x = torch.randn(2, 5, d)
    window = torch.zeros(2, 5, dtype=torch.bool)
    window[0, 2] = window[1, 4] = True

    def run(cond):
        ab.set_condition(cond)
        ab.window = window
        try:
            with torch.no_grad():
                return mlp(x)
        finally:
            ab.set_condition("none")
            ab.window = None

    full, nat, add, rem, bw = (run(c) for c in ("none", "remove_native_write",
                                                "remove_added_write", "remove_write",
                                                "base_write"))
    torch.testing.assert_close(nat + add - full, rem, atol=1e-5, rtol=1e-5)
    assert torch.equal(nat[~window], full[~window]) and torch.equal(bw[~window], full[~window])
    assert not torch.allclose(nat[window], full[window])
    with torch.no_grad():
        a = torch.nn.functional.silu(mlp.gate_proj(x))[..., j]
        u_eff, u_base = mlp.up_proj(x)[..., j], x @ mlp.up_proj.base_layer.weight[j]
        w_eff = effective_weight(mlp.down_proj)[:, j]
        w_base = mlp.down_proj.base_layer.weight[:, j]
        expected = full - (a * u_eff)[..., None] * w_eff + (a * u_base)[..., None] * w_base
    torch.testing.assert_close(bw[window], expected[window], atol=1e-5, rtol=1e-5)
    with torch.no_grad():                         # outside a batch the hooks do nothing
        ab.set_condition("remove_write")
        assert torch.equal(mlp(x), full)
    ab.remove_hooks()


def test_attention_lora_position_ablator_splices_lora_off_output():
    """mechanistic_analysis/score_attention_lora_positions.py: the token groups partition
    the real tokens, the hook swaps in the LoRA-off attention output exactly at the
    group, "all" equals zeroing the layer's attention LoRA, and the u/h/a recorders read
    the same neuron at the same token (h = a * u)."""
    from types import SimpleNamespace
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    from mechanistic_analysis.score_attention_lora_positions import (
        GROUPS, AttentionLoRAPositionAblator, group_mask)

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                      max_position_embeddings=16)
    base = LlamaForCausalLM(cfg).eval()
    get_peft_model(base, LoraConfig(r=2, lora_alpha=4, init_lora_weights=False,
                                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                    "up_proj", "down_proj"]))
    # row 0 is left-padded once; text, then one prompt token, then the readout
    am = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    buckets = torch.tensor([[-1, 0, 0, 0, 1, 2], [0, 0, 0, 0, 1, 2]])
    t_mask = torch.zeros(2, 6, dtype=torch.bool)
    t_mask[0, 2] = t_mask[1, 3] = True
    real = buckets >= 0
    parts = torch.stack([group_mask(g, buckets, t_mask) for g in GROUPS]).long().sum(0)
    assert torch.equal(parts.bool(), real) and int(parts.max()) == 1
    assert torch.equal(group_mask("all_but_t", buckets, t_mask), real & ~t_mask)
    assert not group_mask("none", buckets, t_mask).any()
    assert not group_mask("text_after_t", buckets, t_mask)[1].any()    # t is the last text token

    ids = torch.randint(0, 32, (2, 6))
    attn = base.model.layers[0].self_attn
    ab = AttentionLoRAPositionAblator(SimpleNamespace(base=base), attn_layer=0,
                                      signal_layer=1, index=3)
    seen = {}
    cap = attn.register_forward_hook(lambda m, a, out: seen.__setitem__("attn", out[0].clone()))

    def run(cond, lora_off=False):
        ab.condition, ab.buckets, ab.window = cond, buckets, t_mask
        saved = [lin.weight.data.clone() for lin in ab.lora_B]
        if lora_off:
            for lin in ab.lora_B:
                lin.weight.data.zero_()
        try:
            with torch.no_grad():
                logits = base(input_ids=ids, attention_mask=am, use_cache=False).logits
        finally:
            for lin, w in zip(ab.lora_B, saved):
                lin.weight.data = w
            ab.condition, ab.buckets, ab.window = "none", None, None
        return seen["attn"], logits

    try:
        on_attn, on_logits = run("none")
        off_attn, off_logits = run("none", lora_off=True)
        assert not torch.allclose(on_attn[real], off_attn[real])          # the LoRA matters
        for cond in ("t", "prompt", "text_before_t", "all_but_t"):
            mixed, _ = run(cond)
            m = group_mask(cond, buckets, t_mask)
            torch.testing.assert_close(mixed[m], off_attn[m])
            torch.testing.assert_close(mixed[~m], on_attn[~m])
        _, all_logits = run("all")
        torch.testing.assert_close(all_logits[real], off_logits[real])
        torch.testing.assert_close(run("none")[1], on_logits)              # hooks restore LoRA
        ab.reset_records()
        run("none")
        rec = ab.records()
        assert len(rec["u"]) == 2
        np.testing.assert_allclose(rec["h"], rec["a"] * rec["u"], rtol=1e-5, atol=1e-6)
    finally:
        cap.remove()
        ab.remove_hooks()
