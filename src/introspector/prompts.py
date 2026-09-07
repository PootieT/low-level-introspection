import random
from typing import Literal


# The index the PROMPT shows is not the raw cached `token_position`.
# cache_signals.py tokenizes with add_special_tokens=True, so position 0 is BOS and
# the first *visible* text token sits at position 1. Displaying the raw position
# would ask the model to count a token it cannot see. Subtracting the BOS makes the
# displayed number 0-based over exactly the text the model reads.
# Only the DISPLAYED number shifts -- `token_position` itself still indexes the
# BOS-prefixed sequence and is what the intervention hook injects at. Do not
# "fix" one without the other.
BOS_OFFSET = 1


def _display_token_index(token_index: int) -> int:
    return max(0, int(token_index) - BOS_OFFSET)


def mlp_prompt_with_token_index(
    layer: int,
    signal_index: int,
    token: str,
    token_index: int,
) -> str:
    # These MUST be mlp_prompt()'s templates with only the ` (token #N)` clause
    # added. They previously also dropped the quotes around {TOKEN} and the word
    # "token" before it, so switching arms changed the tokenization of the prompt
    # and not just the presence of the index -- any index-vs-no-index gap would
    # have been confounded by that.
    templates = [
        'At token "{TOKEN}" (token #{TOKEN_IDX}), the MLP activation for neuron {SIGNAL_IDX} at layer {LAYER} is',
        'The MLP activation for neuron {SIGNAL_IDX} at layer {LAYER} for token "{TOKEN}" (token #{TOKEN_IDX}) is',
        'Neuron {SIGNAL_IDX} at layer {LAYER} and token "{TOKEN}" (token #{TOKEN_IDX}) has activation level of',
        'Neuron {LAYER}.{SIGNAL_IDX} at token "{TOKEN}" (token #{TOKEN_IDX}) has activation of',
    ]
    prompt = random.choice(templates).format(
        TOKEN=token, SIGNAL_IDX=signal_index, LAYER=layer,
        TOKEN_IDX=_display_token_index(token_index),
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


def sae_prompt(
    layer: int,
    signal_index: int,
    token: str,
    token_index: int,
) -> str:
    templates = [
        'At token "{TOKEN}", SAE feature {SIGNAL_IDX} at layer {LAYER} is',
        'The SAE feature {SIGNAL_IDX} at layer {LAYER} for token "{TOKEN}" is',
        'SAE feature {SIGNAL_IDX} at layer {LAYER} and token "{TOKEN}" has activation level of',
        'SAE feature {LAYER}.{SIGNAL_IDX} at token "{TOKEN}" has activation of',
    ]
    prompt = random.choice(templates).format(
        TOKEN=token, SIGNAL_IDX=signal_index, LAYER=layer
    )
    return prompt


def prompt_no_token_index(
    signal_type: Literal["mlp", "attention_score", "residual_stream", "sae"],
    **kwargs
):
    if signal_type == "mlp":
        return mlp_prompt(**kwargs)
    elif signal_type == "attention_score":
        return attention_score_prompt(**kwargs)
    elif signal_type == "residual_stream":
        return residual_stream_prompt(**kwargs)
    elif signal_type == "sae":
        return sae_prompt(**kwargs)
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