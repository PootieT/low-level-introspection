import pdb
import random
import re
import hashlib
from typing import List, Optional

from transformers import PretrainedConfig

from .schema import (
    SignalSpec, MLPNeuronSpec, AttentionScoreSpec, AttentionWeightSpec,
    ResidualStreamSpec, SAEFeatureSpec, SignalSource,
)

# Signal name grammar. The "@<src>" suffix is the attention source-token offset and is
# only meaningful for the attention codes. Parsed with a regex rather than the old
# `signal[1:].split(".")`, which assumed exactly two dot-separated parts and blew up with
# an unhelpful unpack error on anything richer.
_SIGNAL_RE = re.compile(
    r"^(?P<code>[NARSP])(?P<layer>\d+)\.(?P<index>\d+)(?:@(?P<src>[+-]?\d+|pairs))?$"
)
# "@pairs": PAIR MODE. The spec names only (layer, head); the (dst, src) pair of each row
# comes from a pair token selector (see schema.RandomPairTokenSelector).
PAIRS_SUFFIX = "pairs"

_CODE_TO_SPEC = {
    "N": MLPNeuronSpec,
    "A": AttentionScoreSpec,
    "P": AttentionWeightSpec,
    "R": ResidualStreamSpec,
    "S": SAEFeatureSpec,
}

# Codes whose signal is a cell of an attention matrix and therefore carries a source
# offset. Single source of truth for "does @<src> belong on this name".
ATTENTION_CODES = ("A", "P")


def parse_signal(signal: str, precision: int) -> SignalSpec:
    """Parse a signal name into a spec.

      N10.24     MLP neuron 24 at layer 10
      A10.5@-3   attention logit, layer 10 head 5, source 3 tokens before the destination
      P10.5@0    post-softmax attention weight, layer 10 head 5, source == destination
      P6.3@pairs post-softmax attention weight, layer 6 head 3, (dst, src) from the
                 pair token selector, one pair per row (PAIR MODE)
      S9.12102   SAE feature (identity fields take SAEFeatureSpec defaults)

    This is user input (CLI / config), so failures raise rather than assert.
    """
    m = _SIGNAL_RE.match(signal.strip())
    if m is None:
        raise ValueError(
            f"{signal!r} is not a signal name. Expected <code><layer>.<index> with an "
            f"optional @<src_offset> for attention, e.g. 'N10.24', 'A10.5@-3'."
        )
    code = m.group("code").upper()
    layer, index, src = int(m.group("layer")), int(m.group("index")), m.group("src")
    if code in ATTENTION_CODES:
        # src is optional so a bare "A10.5" still parses, taking the spec default.
        if src is None:
            kwargs = {}
        elif src == PAIRS_SUFFIX:
            kwargs = {"src_offset": None}
        else:
            kwargs = {"src_offset": int(src)}
        return _CODE_TO_SPEC[code](layer=layer, index=index, precision=precision, **kwargs)
    if src is not None:
        raise ValueError(
            f"{signal!r}: a source offset (@{src}) only applies to attention signals "
            f"{ATTENTION_CODES}, not to code {code!r}."
        )
    return _CODE_TO_SPEC[code](layer=layer, index=index, precision=precision)


def get_available_indices(layer: int, model_config, no_sample_signals: List[SignalSpec]) -> List[int]:
    available_indices = list(range(model_config.intermediate_size))  # llama3.1 schema
    for no_sample_sig in no_sample_signals:
        if no_sample_sig.layer == layer:
            available_indices.remove(no_sample_sig.index)
    return available_indices


def _layer_rng(seed: int, layer: int) -> random.Random:
    """RNG seeded per (seed, layer).

    Extracted so the attention sampler draws heads the same way resolve_signal_source
    draws neurons. See the comment at its call site for why the layer is mixed in.
    """
    layer_seed = int(hashlib.sha256(f"{seed}:{layer}".encode()).hexdigest()[:8], 16)
    return random.Random(layer_seed)


def resolve_attention_sampler(sampler, model_config, precision: int) -> List[SignalSpec]:
    """Expand an AttentionLayerSampler into concrete specs, heads-major.

    Heads-major ordering matters: a sweep trains "the first N heads of the cell", and
    that phrase must mean the same set at every source offset.
    """
    if (sampler.heads is None) == (sampler.heads_per_layer is None):
        raise ValueError(
            "attention_layer_sampler needs exactly one of `heads` (pinned) or "
            f"`heads_per_layer` (sampled); got heads={sampler.heads!r}, "
            f"heads_per_layer={sampler.heads_per_layer!r}."
        )
    if not sampler.src_offsets:
        raise ValueError("attention_layer_sampler needs a non-empty `src_offsets`.")
    if any(off > 0 for off in sampler.src_offsets):
        raise ValueError(
            f"src_offsets must be <= 0 (source cannot follow the destination under a "
            f"causal mask); got {sampler.src_offsets}."
        )

    n_heads = model_config.num_attention_heads
    spec_cls = _CODE_TO_SPEC["A" if sampler.signal_type == "attention_score" else "P"]

    if sampler.heads is not None:
        bad = [h for h in sampler.heads if not 0 <= h < n_heads]
        if bad:
            raise ValueError(f"heads {bad} outside [0, {n_heads}) for this model.")
    elif not 0 < sampler.heads_per_layer <= n_heads:
        raise ValueError(
            f"heads_per_layer={sampler.heads_per_layer} outside (0, {n_heads}]."
        )

    signals: List[SignalSpec] = []
    for layer in sampler.layers:
        if sampler.heads is not None:
            heads = list(sampler.heads)
        else:
            # Same per-(seed, layer) seeding as the neuron sampler below, so a layer's
            # heads are FIXED across every source offset (and across reruns) and the
            # offset axis stays paired head by head.
            heads = sorted(_layer_rng(sampler.seed, layer).sample(
                range(n_heads), sampler.heads_per_layer))
        for head in heads:
            for off in sampler.src_offsets:
                signals.append(spec_cls(layer=layer, index=head,
                                        src_offset=off, precision=precision))
    return signals

def resolve_signal_source(
    signal_source: SignalSource,
    model_config: PretrainedConfig,
    precision:int,
    no_sample_signals:Optional[List[SignalSpec]]=None,
    same_train_test_signals:bool=False
) -> List[SignalSpec]:
    """
    Resolve specifications of signal source sampler to actual signal specifications
    :param signal_source: signal source sampling parameter config
    :param model_config: HF model config
    :param precision: precision of signal source
    :param no_sample_signals: list of signals that is already sampled from a different data split
    :param same_train_test_signals: if using the same signals for train/test splits
    :return:
    """
    no_sample_signals = no_sample_signals or []
    if signal_source.type == "attention_layer_sampler":
        if same_train_test_signals:
            return no_sample_signals
        return resolve_attention_sampler(signal_source, model_config, precision)
    if signal_source.type == "explicit":
        sampled_signals = [parse_signal(sig, precision) for sig in signal_source.signals if sig not in no_sample_signals]
    elif signal_source.type == "uniform_layer_sampler":
        if same_train_test_signals:
            sampled_signals = no_sample_signals
        else:
            sampled_signals = []
            for layer in signal_source.layers:
                available_indices = get_available_indices(layer, model_config, no_sample_signals)
                # Seed PER (seed, layer) instead of drawing from the global RNG.
                # Previously signal_source.seed was declared and never read, so every
                # config generation sampled different neurons from OS entropy: a sweep
                # got a different neuron at every token position, which unpairs the
                # position axis (neuron variance leaks into the position effect) and
                # makes the choice unrecoverable from the spec. Mixing the layer in
                # keeps layers from all drawing the SAME indices, since the candidate
                # pool is identical across layers.
                sampled_indices = _layer_rng(signal_source.seed, layer).sample(
                    available_indices, signal_source.signals_per_layer)
                if signal_source.signal_type == "mlp":
                    layer_signals = [MLPNeuronSpec(layer=layer,index=i, precision=precision) for i in sampled_indices]
                elif signal_source.signal_type in ("attention_score", "attention_weight"):
                    # get_available_indices draws from intermediate_size (8192, the MLP
                    # width), so this branch used to hand out "head" indices far outside
                    # the model's 32 attention heads -- and it has no way to express a
                    # source offset. Refuse instead of producing nonsense.
                    raise ValueError(
                        f"uniform_layer_sampler cannot select {signal_source.signal_type} "
                        f"signals (it samples from the MLP width and has no source-offset "
                        f"axis). Use attention_layer_sampler."
                    )
                elif signal_source.signal_type == "residual_stream":
                    layer_signals = [ResidualStreamSpec(layer=layer,index=i, precision=precision) for i in sampled_indices]
                else:
                    raise NotImplementedError(f"{signal_source.signal_type} is not supported")
                sampled_signals.extend(layer_signals)
    else:
        raise NotImplementedError(f"{signal_source.type} is not supported")

    return sampled_signals

