"""ExperimentConfig.mlp_site: the cached value, the injected value and the drift capture
must all be the SAME tensor at every site (act_fn / gate / down_in), and the default
site must leave every existing cache key untouched.

A tiny randomly initialized Llama (no weights downloaded) stands in for the real model:
the site is a property of the MLP's module structure, which it shares.

    python -m pytest tests/test_mlp_site.py
"""
import json
import operator
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from src.introspector.models.signal_paths import (
    MLP_SITES, register_mlp_site_hook, resolve_mlp_signal_path)
from src.introspector.training.model import InterventionHook

REPO = Path(__file__).resolve().parents[1]
LAYER, IDX, POS = 1, 5, 3


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=4,
                      architectures=["LlamaForCausalLM"])
    model = LlamaForCausalLM(cfg).eval()
    ids = torch.randint(0, 64, (2, 7))
    return model, ids


def _mlp_parts(model, ids):
    """x (MLP input), g, u at LAYER from one clean forward."""
    mlp = model.model.layers[LAYER].mlp
    store = {}
    h = mlp.register_forward_pre_hook(lambda m, a: store.__setitem__("x", a[0].detach()))
    with torch.no_grad():
        model(input_ids=ids)
    h.remove()
    x = store["x"]
    return x, mlp.gate_proj(x).detach(), mlp.up_proj(x).detach()


def _truth(site, g, u):
    a = F.silu(g)
    return {"act_fn": a, "gate": g, "down_in": a * u}[site]


def test_default_path_is_unchanged(tiny):
    model, _ = tiny
    assert resolve_mlp_signal_path(model, 2) == "model.layers.2.mlp.act_fn.output"
    assert resolve_mlp_signal_path(model, 2, path_to_output=False) == "model.layers.2.mlp.act_fn"
    assert resolve_mlp_signal_path(model, 2, site="gate") == "model.layers.2.mlp.gate_proj.output"
    assert resolve_mlp_signal_path(model, 2, site="down_in") == "model.layers.2.mlp.down_proj.input"
    assert resolve_mlp_signal_path(model, 2, False, site="down_in") == "model.layers.2.mlp.down_proj"
    with pytest.raises(ValueError):
        resolve_mlp_signal_path(model, 2, site="up_proj")


@pytest.mark.parametrize("site", sorted(MLP_SITES))
def test_capture_hook_reads_the_site(tiny, site):
    """What the drift check records through register_mlp_site_hook == the site's value."""
    model, ids = tiny
    _, g, u = _mlp_parts(model, ids)
    path = resolve_mlp_signal_path(model, LAYER, path_to_output=False, site=site)
    got = {}

    def cap(m, inputs, value):
        got["v"] = value.detach().clone()
        return None
    handle = register_mlp_site_hook(operator.attrgetter(path)(model), site, cap)
    with torch.no_grad():
        model(input_ids=ids)
    handle.remove()
    torch.testing.assert_close(got["v"], _truth(site, g, u))


@pytest.mark.parametrize("site", sorted(MLP_SITES))
def test_injection_replaces_the_site(tiny, site):
    """InterventionHook at the site sets exactly one entry of the site's tensor, and the
    MLP output is what the MLP computes from the edited tensor."""
    model, ids = tiny
    x, g, u = _mlp_parts(model, ids)
    mlp = model.model.layers[LAYER].mlp
    rows, pos, idx = torch.tensor([0, 1]), torch.tensor([POS, POS - 1]), torch.tensor([IDX, IDX])
    vals = torch.tensor([3.0, -2.0])
    hook = InterventionHook(row_indices=rows, token_positions=pos, signal_indices=idx,
                            intervention_values=vals)
    path = resolve_mlp_signal_path(model, LAYER, path_to_output=False, site=site)
    handle = register_mlp_site_hook(operator.attrgetter(path)(model), site, hook)
    out = {}
    h2 = mlp.register_forward_hook(lambda m, a, o: out.__setitem__("y", o.detach()))
    with torch.no_grad():
        model(input_ids=ids)
    handle.remove()
    h2.remove()

    g2, a2 = g.clone(), F.silu(g)
    if site == "gate":
        g2[rows, pos, idx] = vals
        hidden = F.silu(g2) * u
    elif site == "act_fn":
        a2[rows, pos, idx] = vals
        hidden = a2 * u
    else:
        hidden = F.silu(g) * u
        hidden[rows, pos, idx] = vals
    with torch.no_grad():
        expect = mlp.down_proj(hidden)
    # the edit is local to the injected token: rows before POS-1 are untouched upstream
    torch.testing.assert_close(out["y"][rows, pos], expect[rows, pos])


@pytest.mark.parametrize("site", sorted(MLP_SITES))
def test_nnsight_cache_path_reads_the_site(tiny, site):
    """data_prep/cache_signals.py reads attrgetter(path)(model)[range(bs), positions]
    inside an nnsight trace; for down_in the path ends in .input."""
    nnsight = pytest.importorskip("nnsight")
    model, ids = tiny
    _, g, u = _mlp_parts(model, ids)
    lm = nnsight.NNsight(model)
    path = resolve_mlp_signal_path(model, LAYER, site=site)
    bs = ids.shape[0]
    positions = [POS, POS - 1]
    with torch.no_grad(), lm.trace(ids):
        saved = operator.attrgetter(path)(lm)[range(bs), positions][:, IDX].save()
    got = saved.value if hasattr(saved, "value") else saved
    torch.testing.assert_close(got, _truth(site, g, u)[range(bs), positions][:, IDX])


def test_default_site_keeps_cache_identity():
    from src.introspector.config import ExperimentConfig
    from src.introspector.dataset_utils import (
        canonical_dataset_name, dataset_identity, identity_hash)

    cfgs = sorted((REPO / "experiment_configs/mlp_endo_sweep_layer_token").glob("endo_N0.*_POS-1_gateoff*.json"))
    if not cfgs:
        pytest.skip("no grid config synced locally")
    cfg = ExperimentConfig(**json.loads(cfgs[0].read_text()))
    assert cfg.mlp_site == "act_fn"
    # the default contributes NOTHING to the identity, so every hash computed before the
    # field existed is reproduced (stored dataset_names can still differ from a fresh
    # derivation because of earlier identity changes; they are pinned in the config)
    assert "mlp_site" not in dataset_identity(cfg)
    assert "-act_fn" not in canonical_dataset_name(cfg)
    other = cfg.model_copy(update={"mlp_site": "down_in"})
    assert identity_hash(other) != identity_hash(cfg)
    assert "-down_in_" in canonical_dataset_name(other)
