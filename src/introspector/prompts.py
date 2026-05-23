import random
from typing import Literal


def mlp_prompt_with_token_index(
    layer: int,
    signal_index: int,
    token: str,
    token_index: int,
) -> str:
    templates = [
        "At token {TOKEN} (token #{TOKEN_IDX}), the MLP activation for neuron {SIGNAL_IDX} at layer {LAYER} is",
        "The MLP activation for neuron {SIGNAL_IDX} at layer {LAYER} for {TOKEN} (token #{TOKEN_IDX}) is",
        "Neuron {SIGNAL_IDX} at layer {LAYER} and {TOKEN} (token #{TOKEN_IDX}) has activation level of",
        "Neuron {LAYER}.{SIGNAL_IDX} at {TOKEN} (token #{TOKEN_IDX}) has activation of",
    ]
    prompt = random.choice(templates).format(
        TOKEN=token, SIGNAL_IDX=signal_index, LAYER=layer, TOKEN_IDX=token_index,
    )
    return prompt

def mlp_prompt(
    layer: int,
    signal_index: int,
    token: str,
    token_index: int,
) -> str:
    templates = [
        'At token "{TOKEN}", the MLP activation for neuron {SIGNAL_IDX} at layer {LAYER} is',
        'The MLP activation for neuron {SIGNAL_IDX} at layer {LAYER} for token "{TOKEN}" is',
        'Neuron {SIGNAL_IDX} at layer {LAYER} and token "{TOKEN}" has activation level of',
        'Neuron {LAYER}.{SIGNAL_IDX} at token "{TOKEN}" has activation of',
    ]
    prompt = random.choice(templates).format(
        TOKEN=token, SIGNAL_IDX=signal_index, LAYER=layer
    )
    return prompt


def attention_score_prompt(
    text: str,
    layer: int,
    signal_index: int,
    src_token: str,
    src_token_index: int,
    dst_token: str,
    dst_token_index: int,
) -> str:
    templates = [
        "At token {DST_TOKEN}, the attention score for head {SIGNAL_IDX} on {TGT_TOKEN} at layer {LAYER} is",
        "The attention score for head {SIGNAL_IDX} from {SRC_TOKEN} to {DST_TOKEN} at layer {LAYER} is",
        "Attention head {LAYER}.{SIGNAL_IDX} from {SRC_TOKEN} to {DST_TOKEN} has attention score of",
        "Attention head {SIGNAL_IDX} at layer {LAYER}, from token {SRC_TOKEN} to {DST_TOKEN} has attention score of",
    ]
    prompt = random.choice(templates).format(
        SRC_TOKEN=src_token, SIGNAL_IDX=signal_index, LAYER=layer, DST_TOKEN=dst_token,
    )
    return prompt


def residual_stream_prompt(
    text: str,
    layer: int,
    signal_index: int,
    token: str,
    token_index: int,
) -> str:
    templates = [
        "At token {TOKEN}, the residual stream activation at layer {LAYER}, dimension {SIGNAL_IDX} is",
        "The hidden state activation for channel {SIGNAL_IDX} at layer {LAYER} for {TOKEN} is",
        "Residual stream neuron {SIGNAL_IDX} at layer {LAYER} and {TOKEN} has activation level of",
        "Residual stream index {LAYER}.{SIGNAL_IDX} at {TOKEN} has activation of",
    ]
    prompt = random.choice(templates).format(
        TOKEN=token, SIGNAL_IDX=signal_index, LAYER=layer
    )
    return prompt


def prompt_no_token_index(
    signal_type: Literal["mlp", "attention_score", "residual_stream"],
    **kwargs
):
    if signal_type == "mlp":
        return mlp_prompt(**kwargs)
    elif signal_type == "attention_score":
        return attention_score_prompt(**kwargs)
    elif signal_type == "residual_stream":
        return residual_stream_prompt(**kwargs)
    else:
        raise ValueError()

def prompt_with_token_index(
        signal_type: Literal["mlp", "attention_score", "residual_stream"],
        **kwargs
):
    if signal_type == "mlp":
        return mlp_prompt_with_token_index(**kwargs)
    elif signal_type == "attention_score":
        raise NotImplementedError()
    elif signal_type == "residual_stream":
        raise NotImplementedError()
    else:
        raise ValueError()

PROMPT_REGISTRY = {
    "prompt_no_token_index": prompt_no_token_index,
    "prompt_with_token_index": prompt_with_token_index,
}