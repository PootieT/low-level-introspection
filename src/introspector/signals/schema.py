from typing import Literal, Union

from pydantic import BaseModel

class SpecModel(BaseModel):
    type: str
    layer: int
    index: int
    precision: int=3

    def __hash__(self):
        return hash(f"{self.type[0]}{self.layer}{self.index}")

    def __eq__(self, other):
        if not isinstance(other, SpecModel):
            return False
        return (self.type == other.type) and (self.layer == other.layer) and (self.index == other.index)

class MLPNeuronSpec(SpecModel):
    type: Literal["mlp"] = "mlp"

class AttentionScoreSpec(SpecModel):
    type: Literal["attention_score"] = "attention_score"

class ResidualStreamSpec(SpecModel):
    type: Literal["residual_stream"] = "residual_stream"


SignalSpec = Union[MLPNeuronSpec, AttentionScoreSpec, ResidualStreamSpec]

class ExplicitSignalSpec(BaseModel):
    type: Literal["explicit"]
    signals: list[str]  # A2.1, N19.2, R21.232

class UniformLayerSampler(BaseModel):
    type: Literal["uniform_layer_sampler"]
    signal_type: Literal["mlp", "attention_score", "residual_stream"]
    layers: list[int]
    signals_per_layer: int
    seed: int = 0

SignalSource = Union[
    ExplicitSignalSpec,
    UniformLayerSampler,
]


class FixedTokenSelector(BaseModel):
    type: Literal["fixed"] = "fixed"
    position: int

class RandomRangeTokenSelector(BaseModel):
    type: Literal["random_range"] = "random_range"
    start: int
    stop: int


TokenSelector = Union[FixedTokenSelector,RandomRangeTokenSelector]
