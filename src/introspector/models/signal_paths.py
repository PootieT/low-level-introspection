
from nnsight.modeling.language import LanguageModel

from ..signals.schema import SignalSpec, MLPNeuronSpec, ResidualStreamSpec


# Architectures whose decoder MLP is down_proj(act_fn(gate_proj(x)) * up_proj(x)) at
# model.layers.<i>.mlp -- so the act_fn output is the same "neuron" in each.
GATED_SILU_MLP_ARCHS = {"LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"}


def resolve_mlp_signal_path(
    model: LanguageModel,
    layer: int,
    path_to_output: bool=True,
)-> str:
    """
    Return model specific path to MLP signals
    :param model:
    :param layer: which layer
    :param path_to_output: suffix .output at end of path, otherwise return just the path to theMLP module
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
        path = f"model.layers.{layer}.mlp.act_fn.output"
    else:
        raise NotImplementedError(
            f"No MLP signal path for architecture {arch!r}. If its MLP computes "
            f"down_proj(act_fn(gate_proj(x)) * up_proj(x)) under model.layers.<i>.mlp, add "
            f"it to GATED_SILU_MLP_ARCHS -- after checking, not by assumption.")
    if not path_to_output:
        path = path.removesuffix(".output")
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