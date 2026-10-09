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


class SAEFeatureSpec(SpecModel):
    """A single SAE feature target. `layer`/`index` are the SAE's layer and the
    feature id; the extra fields pin down *which* SAE produced it so the cache
    rows, the encoder used for capture, and the steering hook all agree.

    `sae_release`/`sae_site`/`sae_width` are resolved to a concrete loader id by
    src.introspector.signals.sae.load_sae. Defaults target Gemma Scope's curated
    ("canonical") residual-stream SAEs for gemma-2-2b.
    """
    type: Literal["sae"] = "sae"
    # provider/release family; see signals/sae.py for how this maps to a loader id
    sae_release: str = "gemma-scope-2b-pt-res-canonical"
    # which activation site the SAE was trained on. MUST match the hook the
    # residual is captured from AND steered at (see resolve_sae_hook_path).
    sae_site: Literal["res", "mlp", "attn"] = "res"
    # dictionary width label as used by the release (e.g. "16k", "65k", "1m")
    sae_width: str = "16k"

    def __hash__(self):
        # distinguish two different SAEs at the same (layer, index)
        return hash(f"s{self.sae_release}{self.sae_site}{self.sae_width}{self.layer}{self.index}")

    def __eq__(self, other):
        if not isinstance(other, SAEFeatureSpec):
            return False
        return (self.sae_release == other.sae_release and self.sae_site == other.sae_site
                and self.sae_width == other.sae_width and self.layer == other.layer
                and self.index == other.index)


SignalSpec = Union[MLPNeuronSpec, AttentionScoreSpec, ResidualStreamSpec, SAEFeatureSpec]

class ExplicitSignalSpec(BaseModel):
    type: Literal["explicit"]
    signals: list[str]  # A2.1, N19.2, R21.232

class UniformLayerSampler(BaseModel):
    type: Literal["uniform_layer_sampler"]
    signal_type: Literal["mlp", "attention_score", "residual_stream", "sae"]
    layers: list[int]
    signals_per_layer: int
    seed: int = 0


class SAEFeatureSampler(BaseModel):
    """Select SAE features to introspect on by interpretability-friendly criteria
    instead of listing raw indices. Feature stats (activation density) come from a
    pre-computed stats file (see scripts/select_sae_features.py); interp scores /
    labels come from Neuronpedia. Selection deliberately avoids dead features (no
    signal) and ultra-dense ones (trivially always-on)."""
    type: Literal["sae_feature_sampler"]
    sae_release: str = "gemma-scope-2b-pt-res-canonical"
    sae_site: Literal["res", "mlp", "attn"] = "res"
    sae_width: str = "16k"
    layers: list[int]
    features_per_layer: int
    # Keep features whose firing rate is within [min_density, max_density], with
    # density computed BOS-excluded (matching Neuronpedia's "ignore BOS" default).
    #   max_density=0.01 mirrors Neuronpedia's default "hide >1% density" filter,
    #     which drops trivial position/always-on features (e.g. fires on every
    #     "The"). Do not raise it without a reason.
    #   min_density guards against features so rare they yield ~no active examples
    #     in the training corpus (the sparse-target problem): with N training
    #     tokens you want density * N active examples to be enough to learn from.
    #     Tune this to the dataset size, not blindly.
    min_density: float = 1e-3
    max_density: float = 0.01
    # optional Neuronpedia auto-interp score floor (None => ignore). NOTE: Neuronpedia
    # exposes descriptions but NOT auto-interp scores for the Gemma Scope res-16k
    # features (scores: []), so this floor is effectively inert for that release --
    # feature selection is instead decided by the steering judge
    # (generate_sae_steering_samples.py + out-of-band judging). annotate_sae_features_
    # neuronpedia.py still records interp_score when a release does provide one.
    min_interp_score: float | None = None
    seed: int = 0


SignalSource = Union[
    ExplicitSignalSpec,
    UniformLayerSampler,
    SAEFeatureSampler,
]


class FixedTokenSelector(BaseModel):
    type: Literal["fixed"] = "fixed"
    position: int

class RandomRangeTokenSelector(BaseModel):
    type: Literal["random_range"] = "random_range"
    start: int
    stop: int


TokenSelector = Union[FixedTokenSelector,RandomRangeTokenSelector]
