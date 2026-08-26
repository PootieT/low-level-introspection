import pdb
import random
import hashlib
from typing import List, Optional

from transformers import PretrainedConfig

from .schema import (
    SignalSpec, MLPNeuronSpec, AttentionScoreSpec, ResidualStreamSpec,
    SAEFeatureSpec, SignalSource,
)


def parse_signal(signal: str, precision: int) -> SignalSpec:
    layer, index = signal[1:].split(".")
    if signal.upper().startswith("N"):
        return MLPNeuronSpec(layer=int(layer),index=int(index),precision=precision)
    elif signal.upper().startswith("A"):
        return AttentionScoreSpec(layer=int(layer),index=int(index),precision=precision)
    elif signal.upper().startswith("R"):
        return ResidualStreamSpec(layer=int(layer),index=int(index),precision=precision)
    elif signal.upper().startswith("S"):
        # e.g. "S9.12102"; SAE identity (release/site/width) uses SAEFeatureSpec
        # defaults (Gemma Scope res 16k). Override those in the generated config if
        # you point at a different SAE.
        return SAEFeatureSpec(layer=int(layer),index=int(index),precision=precision)
    else:
        raise NotImplementedError(f"{signal} is not supported")


def get_available_indices(layer: int, model_config, no_sample_signals: List[SignalSpec]) -> List[int]:
    available_indices = list(range(model_config.intermediate_size))  # llama3.1 schema
    for no_sample_sig in no_sample_signals:
        if no_sample_sig.layer == layer:
            available_indices.remove(no_sample_sig.index)
    return available_indices

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
                layer_seed = int(hashlib.sha256(
                    f"{signal_source.seed}:{layer}".encode()).hexdigest()[:8], 16)
                sampled_indices = random.Random(layer_seed).sample(
                    available_indices, signal_source.signals_per_layer)
                if signal_source.signal_type == "mlp":
                    layer_signals = [MLPNeuronSpec(layer=layer,index=i, precision=precision) for i in sampled_indices]
                elif signal_source.signal_type == "attention_score":
                    layer_signals = [AttentionScoreSpec(layer=layer,index=i, precision=precision) for i in sampled_indices]
                elif signal_source.signal_type == "residual_stream":
                    layer_signals = [ResidualStreamSpec(layer=layer,index=i, precision=precision) for i in sampled_indices]
                else:
                    raise NotImplementedError(f"{signal_source.signal_type} is not supported")
                sampled_signals.extend(layer_signals)
    else:
        raise NotImplementedError(f"{signal_source.type} is not supported")

    return sampled_signals

