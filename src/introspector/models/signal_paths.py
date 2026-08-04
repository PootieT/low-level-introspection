
from nnsight.modeling.language import LanguageModel

from ..signals.schema import SignalSpec, MLPNeuronSpec, ResidualStreamSpec


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

    if "LlamaForCausalLM" in model.config.architectures[0]:
        # Llama MLP: output = down_proj (Swish(gate_proj(x)) * up_proj(x))
        # we return the output of Swish(gate_proj(x)) as this is the "neuron activation"
        path =  f"model.layers.{layer}.mlp.act_fn.output"
    else:
        raise NotImplementedError
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