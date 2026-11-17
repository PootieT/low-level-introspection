
from nnsight.modeling.language import LanguageModel

from ..signals.schema import SignalSpec, MLPNeuronSpec, ResidualStreamSpec


# Architectures whose decoder MLP is down_proj(act_fn(gate_proj(x)) * up_proj(x)) at
# model.layers.<i>.mlp -- so the act_fn output is the same "neuron" in each.
GATED_SILU_MLP_ARCHS = {"LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"}

# WHICH per-neuron MLP quantity an "mlp" signal is (ExperimentConfig.mlp_site), as
# (module under model.layers.<i>.mlp, whether the quantity is its output or its input).
# With g = gate_proj(x), u = up_proj(x):
#   act_fn   a = silu(g)   the default, and every run before mlp_site existed
#   gate     g             the gate pre-activation. Linear in the MLP input, so its input
#                          bypass is exact (ridge R^2 ~0.99 from the pre-MLP residual)
#   down_in  h = a * u     down_proj's input. One neuron's write to the residual is
#                          h * W_down[:, i], so a fixed readout of the write recovers h
#                          exactly; a and g reach the write only through u(x), which
#                          flips sign with context (see mechanistic_analysis/
#                          score_neuron_write_stats.py)
# Caching reads the site and the injection hook replaces it, so for every site the
# introspected value and the intervened value are the same tensor.
MLP_SITES = {
    "act_fn": ("act_fn", "output"),
    "gate": ("gate_proj", "output"),
    "down_in": ("down_proj", "input"),
}


def check_mlp_site(site: str) -> str:
    if site not in MLP_SITES:
        raise ValueError(f"mlp_site {site!r} not in {sorted(MLP_SITES)}")
    return site


def register_mlp_site_hook(module, site: str, hook):
    """Attach a FORWARD-HOOK-shaped callable `hook(module, inputs, value) -> new value or
    None` to the module that resolve_mlp_signal_path(..., path_to_output=False,
    site=site) returns. For an output site that is an ordinary forward hook. For an input
    site (down_in) it runs as a forward PRE-hook on the module's first positional
    argument, so InterventionHook / ActivationCaptureHook work unchanged at every site.
    Returns the removable handle."""
    if MLP_SITES[check_mlp_site(site)][1] == "output":
        return module.register_forward_hook(hook)

    def pre(mod, args):
        new = hook(mod, args, args[0])
        return None if new is None else (new,) + tuple(args[1:])
    return module.register_forward_pre_hook(pre)


def resolve_mlp_signal_path(
    model: LanguageModel,
    layer: int,
    path_to_output: bool=True,
    site: str = "act_fn",
)-> str:
    """
    Return model specific path to MLP signals
    :param model:
    :param layer: which layer
    :param path_to_output: suffix .output (or .input, for an input site) at end of path,
        otherwise return just the path to the module -- hook it with
        register_mlp_site_hook, which knows whether the site is an output or an input
    :param site: which MLP quantity, see MLP_SITES
    :return: MLP signal path for NNsight LanguageModel
    """
    prefix_path = ""
    if model.config is None: # peft model wrapper running with nnsight
        model = model.model
        prefix_path = "model."
    elif model.config is not None and hasattr(model.model, "model"): # peft model, running without nnsight (pytorch hook)
        prefix_path = "model."

    arch = model.config.architectures[0]
    if arch in GATED_SILU_MLP_ARCHS:
        # MLP: output = down_proj(act_fn(gate_proj(x)) * up_proj(x)). We return the output
        # of act_fn(gate_proj(x)) as the "neuron activation". Verified identical in
        # transformers 5.8.1 for LlamaMLP, Qwen2MLP and Qwen3MLP: same module names,
        # same forward, so the same hook site means the same quantity.
        mod, port = MLP_SITES[check_mlp_site(site)]
        path = f"model.layers.{layer}.mlp.{mod}.{port}"
    else:
        raise NotImplementedError(
            f"No MLP signal path for architecture {arch!r}. If its MLP computes "
            f"down_proj(act_fn(gate_proj(x)) * up_proj(x)) under model.layers.<i>.mlp, add "
            f"it to GATED_SILU_MLP_ARCHS -- after checking, not by assumption.")
    if not path_to_output:
        path = path.rsplit(".", 1)[0]
    return f"{prefix_path}{path}"

def resolve_resid_signal_path(
    model: LanguageModel,
    layer: int
):
    """
    Return model specific path to residual stream signals
    :param model:
    :param layer: which layer
    :return: residual stream signal path for NNsight LanguageModel
    """
    return f"model.model.layers[{layer}].output"


def resolve_sae_hook_path(
    model: LanguageModel,
    layer: int,
    site: str = "res",
    path_to_output: bool = True,
) -> str:
    """
    Path (from the nnsight/HF root) to the residual-stream site a Gemma Scope
    `res` SAE was trained on. This is the decoder-layer OUTPUT for `layer`, which
    equals TransformerLens `blocks.{layer}.hook_resid_post`. The SAME site must be
    used for (a) caching the feature activation and (b) registering the steering
    forward hook -- otherwise the SAE encodes an off-distribution residual.

    Mirrors resolve_mlp_signal_path's prefix handling so it works whether the
    object is a raw nnsight LanguageModel, a peft-wrapped model under nnsight, or a
    peft model driven by plain pytorch hooks.

    IMPORTANT: a decoder layer's output is a TUPLE (hidden_states, ...) for both
    Llama and Gemma2. With `path_to_output=True` this path ends at `.output`, so
    the caller must take element [0] to get the residual tensor (nnsight:
    `attrgetter(path)(m)[0]`; pytorch forward-hook: `output[0]`). With
    `path_to_output=False` it points at the layer MODULE, for
    `module.register_forward_hook(...)`.
    """
    if site != "res":
        raise NotImplementedError(
            f"SAE hook site '{site}' not wired yet; only 'res' (decoder layer "
            f"output) is supported. Add mlp/attn sites here to match the release."
        )

    prefix_path = ""
    if model.config is None:  # peft model wrapper running with nnsight
        model = model.model
        prefix_path = "model."
    elif model.config is not None and hasattr(model.model, "model"):  # peft + pytorch hooks
        prefix_path = "model."

    arch = model.config.architectures[0]
    if not any(a in arch for a in ("LlamaForCausalLM", "Gemma2ForCausalLM", "Gemma3ForCausalLM")):
        raise NotImplementedError(f"SAE hook path not verified for architecture {arch!r}")

    # residual site == decoder layer output, architecture-agnostic
    path = f"model.layers.{layer}.output"
    if not path_to_output:
        path = path.removesuffix(".output")
    return f"{prefix_path}{path}"