
from nnsight.modeling.language import LanguageModel

from ..signals.schema import SignalSpec, MLPNeuronSpec, ResidualStreamSpec


def resolve_mlp_signal_path(
    model: LanguageModel,
    layer: int
)-> str:
    """
    Return model specific path to MLP signals
    :param model:
    :param layer: which layer
    :return: MLP signal path for NNsight LanguageModel
    """
    prefix_path = ""
    if model.config is None: # peft model wrapper
        model = model.model
        prefix_path = "model."
    if "LlamaForCausalLM" in model.config.architectures[0]:
        # Llama MLP: output = down_proj (Swish(gate_proj(x)) * up_proj(x))
        # we return the output of Swish(gate_proj(x)) as this is the "neuron activation"
        path =  f"model.layers.{layer}.mlp.act_fn.output"
    else:
        raise NotImplementedError
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