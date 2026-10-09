import random
from functools import partial
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


def feature_key(layer: int, signal_index: int) -> str:
    """Key into TrainingConfig.feature_labels for an SAE feature."""
    return f"S{int(layer)}.{int(signal_index)}"


def sae_label_prompt(
    layer: int,
    signal_index: int,
    token: str,
    token_index: int,
    labels: dict[str, str],
) -> str:
    """sae_prompt() with the feature NAMED BY ITS LABEL instead of its index.

    These MUST stay sae_prompt()'s templates with only `SAE feature {SIGNAL_IDX}` (or
    `{LAYER}.{SIGNAL_IDX}`) swapped for `SAE feature "{LABEL}"`, so the label and
    index arms of the feature-generalization experiment differ in how the feature is
    addressed and in nothing else (same rule as mlp_prompt_with_token_index). The
    fourth template's `{LAYER}.{SIGNAL_IDX}` becomes `"{LABEL}" (layer {LAYER})` --
    its index-only form names the layer, so the label form keeps it.

    A feature with no label raises rather than falling back to its index: a silent
    fallback would put index-addressed rows into a label-addressed run.
    """
    key = feature_key(layer, signal_index)
    if key not in labels:
        raise KeyError(f"no label for SAE feature {key}; the config's feature_labels "
                       f"must cover every feature the run trains or evaluates")
    templates = [
        'At token "{TOKEN}", SAE feature "{LABEL}" at layer {LAYER} is',
        'The SAE feature "{LABEL}" at layer {LAYER} for token "{TOKEN}" is',
        'SAE feature "{LABEL}" at layer {LAYER} and token "{TOKEN}" has activation level of',
        'SAE feature "{LABEL}" (layer {LAYER}) at token "{TOKEN}" has activation of',
    ]
    return random.choice(templates).format(TOKEN=token, LABEL=labels[key], LAYER=layer)


def prompt_sae_label(signal_type: str, labels: dict[str, str], **kwargs):
    if signal_type != "sae":
        raise ValueError(f"prompt_sae_label is for SAE features, got signal_type={signal_type!r}")
    return sae_label_prompt(labels=labels, **kwargs)


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
    # needs labels=TrainingConfig.feature_labels bound first; use get_prompt_fn
    "prompt_sae_label": prompt_sae_label,
}

# prompt types whose function needs the config's feature_labels bound in
_LABEL_PROMPTS = {"prompt_sae_label"}


def get_prompt_fn(training_config):
    """PROMPT_REGISTRY lookup that binds feature_labels for the label prompts, so
    preprocess_example can call every prompt function with the same row fields."""
    fn = PROMPT_REGISTRY[training_config.prompt_type]
    if training_config.prompt_type in _LABEL_PROMPTS:
        if not training_config.feature_labels:
            raise ValueError(f"prompt_type={training_config.prompt_type!r} needs "
                             f"training_config.feature_labels (--feature-labels-file)")
        return partial(fn, labels=training_config.feature_labels)
    return fn