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


def _pick_template(templates, template_idx):
    """Draw a template, or take a pinned one.

    `None` keeps `random.choice` -- byte-identical to what every prompt family did before
    pinning existed, so no cached dataset or trained run changes. An explicit index is
    for analyses that average over examples position by position.
    """
    if template_idx is None:
        return random.choice(templates)
    if not 0 <= template_idx < len(templates):
        raise ValueError(
            f"template_idx {template_idx} out of range for {len(templates)} templates")
    return templates[template_idx]


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
    layer: int,
    signal_index: int,
    src_token: str,
    dst_token: str,
) -> str:
    """Ask for the pre-softmax attention logit of head `signal_index` at `layer`,
    from the destination token to the source token.

    `signal_index` IS the query head. Two fixes against the previous version:
      - the first template referenced {TGT_TOKEN}, a placeholder .format() was never
        given, so one draw in four raised KeyError;
      - the tokens were unquoted, unlike every other prompt family here. Quoting them
        matters more for attention than for MLP, because the prompt names TWO tokens and
        without delimiters the boundary between them and the template words is guesswork
        for a token like " the".

    KNOWN AMBIGUITY (accepted deliberately): the source is named by its string only, so
    when the same token occurs several times in the context the prompt does not say
    which occurrence is meant. cache_signals records `src_token_unique_rate` in the
    dataset metadata so the effect can be measured instead of assumed.
    """
    templates = [
        'At token "{DST_TOKEN}", the attention score for head {HEAD} at layer {LAYER} on token "{SRC_TOKEN}" is',
        'The attention score for head {HEAD} at layer {LAYER}, from token "{DST_TOKEN}" to token "{SRC_TOKEN}", is',
        'Attention head {LAYER}.{HEAD} at token "{DST_TOKEN}", attending to token "{SRC_TOKEN}", has attention score of',
        'Head {HEAD} at layer {LAYER} gives token "{SRC_TOKEN}" an attention score, at token "{DST_TOKEN}", of',
    ]
    return random.choice(templates).format(
        SRC_TOKEN=src_token, HEAD=signal_index, LAYER=layer, DST_TOKEN=dst_token,
    )


def attention_weight_prompt(
    layer: int,
    signal_index: int,
    src_token: str,
    dst_token: str,
    template_idx: int | None = None,
) -> str:
    """Post-softmax counterpart of attention_score_prompt. Worded as a *weight* (a
    probability), not a score, so the two signal types are not asked for in the same
    words -- a model trained on one should not be able to mistake it for the other.

    `template_idx` pins the wording instead of drawing it. Training leaves it None and
    keeps drawing at random; ATTRIBUTION sets it, because an averaged per-token
    attribution map needs the same token index to mean the same word in every example,
    and four templates of different lengths put different words at the same offset.
    """
    templates = [
        'At token "{DST_TOKEN}", the attention weight for head {HEAD} at layer {LAYER} on token "{SRC_TOKEN}" is',
        'The attention weight for head {HEAD} at layer {LAYER}, from token "{DST_TOKEN}" to token "{SRC_TOKEN}", is',
        'Attention head {LAYER}.{HEAD} at token "{DST_TOKEN}", attending to token "{SRC_TOKEN}", has attention weight of',
        'Head {HEAD} at layer {LAYER} puts, at token "{DST_TOKEN}", an attention weight on token "{SRC_TOKEN}" of',
    ]
    return _pick_template(templates, template_idx).format(
        SRC_TOKEN=src_token, HEAD=signal_index, LAYER=layer, DST_TOKEN=dst_token,
    )


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
    signal_type: Literal["mlp", "attention_score", "attention_weight", "residual_stream", "sae"],
    **kwargs
):
    if signal_type == "mlp":
        return mlp_prompt(**kwargs)
    elif signal_type == "attention_score":
        return attention_score_prompt(**kwargs)
    elif signal_type == "attention_weight":
        return attention_weight_prompt(**kwargs)
    elif signal_type == "residual_stream":
        return residual_stream_prompt(**kwargs)
    elif signal_type == "sae":
        return sae_prompt(**kwargs)
    else:
        raise ValueError()

def prompt_with_token_index(
        signal_type: Literal["mlp", "attention_score", "attention_weight", "residual_stream"],
        **kwargs
):
    if signal_type == "mlp":
        return mlp_prompt_with_token_index(**kwargs)
    elif signal_type in ("attention_score", "attention_weight"):
        raise NotImplementedError()
    elif signal_type == "residual_stream":
        raise NotImplementedError()
    else:
        raise ValueError()

PROMPT_REGISTRY = {
    "prompt_no_token_index": prompt_no_token_index,
    "prompt_with_token_index": prompt_with_token_index,
}

# Signal types whose prompt names a source token as well as a destination token.
_ATTENTION_TYPES = ("attention_score", "attention_weight")


def build_prompt_kwargs(example) -> dict:
    """Cache row -> the keyword arguments that row's prompt family takes.

    THE point of this function is that it is the ONLY place that knows a prompt's
    argument list depends on the signal type. preprocess_example used to hardcode the
    five MLP kwargs, so an attention row raised TypeError there; and preprocess_example
    is the shared entry point for a dozen call sites, so widening it in place would have
    meant touching all of them.

    The MLP/SAE/residual branch reproduces the old argument list exactly, so existing
    runs build byte-identical prompts.
    """
    signal_type = example["signal_type"]
    if signal_type in _ATTENTION_TYPES:
        # `token`/`token_position` are the DESTINATION for attention rows; the source
        # comes from the columns cache_signals adds for these signal types.
        src_token = example.get("src_token")
        if src_token is None:
            raise ValueError(
                f"{signal_type} row has no `src_token`; the cache predates attention "
                f"signals or was built by a different code path. Re-cache it.")
        return {
            "layer": example["signal_layer"],
            "signal_index": example["signal_index"],   # query head
            "src_token": src_token,
            "dst_token": example["token"],
        }
    return {
        "layer": example["signal_layer"],
        "signal_index": example["signal_index"],
        "token": example["token"],
        "token_index": example["token_position"],
    }
