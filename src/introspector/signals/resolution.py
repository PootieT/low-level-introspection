import random
from typing import List, Optional

from transformers import PretrainedConfig

from .schema import SignalSpec, MLPNeuronSpec, AttentionScoreSpec, ResidualStreamSpec, SignalSource


def parse_signal(signal: str, precision: int) -> SignalSpec:
    layer, index = signal[1:].split(".")
    if signal.upper().startswith("N"):
        return MLPNeuronSpec(layer=int(layer),index=int(index),precision=precision)
    elif signal.upper().startswith("A"):
        return AttentionScoreSpec(layer=int(layer),index=int(index),precision=precision)
    elif signal.upper().startswith("R"):
        return ResidualStreamSpec(layer=int(layer),index=int(index),precision=precision)
    else:
        raise NotImplementedError(f"{signal} is not supported")


def get_available_indices(layer: int, model_config, no_sample_signals: List[SignalSpec]) -> List[int]:
    available_indices = list(range(model_config.intermediate_size))  # llama3.1 schema
    for no_sample_sig in no_sample_signals:
        if no_sample_sig.layer == layer:
            available_indices.remove(no_sample_sig.index)
    return available_indices

def resolve_signal_source(signal_source: SignalSource, model_config: PretrainedConfig, precision:int, no_sample_signals:Optional[List[SignalSpec]]=None) -> List[SignalSpec]:
    no_sample_signals = no_sample_signals or []
    if signal_source.type == "explicit":
        sampled_signals = [parse_signal(sig, precision) for sig in signal_source.signals if sig not in no_sample_signals]
    elif signal_source.type == "uniform_layer_sampler":
        sampled_signals = []
        for layer in signal_source.layers:
            available_indices = get_available_indices(layer, model_config, no_sample_signals)
            sampled_indices = random.sample(available_indices, signal_source.signals_per_layer)
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

