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

class _AttentionSpecBase(SpecModel):
    """Common shape of an attention signal: a single cell of one head's attention
    matrix, addressed by (layer, head, destination token, source token).

    `index` IS THE QUERY HEAD index -- not a neuron. prompts.attention_score_prompt
    already reads SIGNAL_IDX as the head, and reusing `index` keeps the whole
    signal_layer/signal_index plumbing (cache columns -> collator -> model kwargs)
    working unchanged.

    The destination token comes from the experiment's token_selector, exactly as it
    does for an MLP neuron. Only the SOURCE token is carried here, as an offset from
    the destination.
    """
    # Source token offset relative to the destination token: 0 == the destination
    # attending to itself, -1 == the token immediately before it.
    #
    # None == PAIR MODE ("P6.3@pairs"): the (dst, src) pair is not part of the signal but
    # comes from a pair token selector (RandomPairTokenSelector / PairListTokenSelector),
    # one pair per cached row. The spec then names only (layer, head), and each row's
    # realized offset (src - dst) is written into its `src_offset` column. An int and
    # None never mix in one config; cache_signals.validate_attention_offsets refuses it.
    #
    # WHY a position lives in the signal IDENTITY. One forward pass produces the whole
    # [B, Hq, L_dst, L_src] logit tensor, so every source offset is free once the
    # destination is fixed. Putting src_offset in the spec lets ONE cached dataset hold
    # many offsets, split per run by --signal-filter (which is deliberately not part of
    # the cache identity). Keeping it in the token selector instead would force one
    # caching pass per offset.
    src_offset: int | None = -1

    def __hash__(self):
        # MUST override: SpecModel.__hash__ keys on (type[0], layer, index) only, and
        # extract_activations_batch collects results in a dict keyed by spec. Without
        # this, the 8 heads x 10 offsets of a cell collapse into 8 entries WITHOUT AN
        # ERROR and 9 of every 10 signals silently take another signal's values.
        # Same reason SAEFeatureSpec overrides it.
        return hash(f"{self.type}{self.layer}.{self.index}@{self.src_offset}")

    def __eq__(self, other):
        if not isinstance(other, _AttentionSpecBase):
            return False
        return (self.type == other.type and self.layer == other.layer
                and self.index == other.index and self.src_offset == other.src_offset)


class AttentionScoreSpec(_AttentionSpecBase):
    """Pre-softmax attention logit: q.k/sqrt(d_head) after RoPE, before the causal
    mask and the softmax. Type code "A", e.g. "A10.5@-3"."""
    type: Literal["attention_score"] = "attention_score"


class AttentionWeightSpec(_AttentionSpecBase):
    """Post-softmax attention weight (a probability in [0, 1]). Type code "P".

    Declared now so the spec/parse/identity plumbing is in place; the extraction branch
    reads the same port one step later in introspection_attention_forward.
    """
    type: Literal["attention_weight"] = "attention_weight"

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


SignalSpec = Union[MLPNeuronSpec, AttentionScoreSpec, AttentionWeightSpec,
                   ResidualStreamSpec, SAEFeatureSpec]

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


class AttentionLayerSampler(BaseModel):
    """Select attention cells to introspect on, as heads x source offsets per layer.

    UniformLayerSampler cannot express this: it draws ONE index per signal out of
    `model_config.intermediate_size` (the MLP width, 8192 -- not the 32 attention
    heads), and it has no notion of a source offset at all.

    Two modes, exactly one of which must be set:
      heads=[...]        pin the head set (the fixed-head sweep, e.g. stride 4 over the
                         32 query heads visits each of the 8 GQA key/value groups once)
      heads_per_layer=n  sample n heads per layer, seeded per (seed, layer) the same way
                         resolve_signal_source seeds neurons -- so a layer's heads are
                         IDENTICAL at every source offset and the offset curve can be
                         read per head. Drawing fresh heads per offset would let head
                         variance leak into the offset effect, which is the bug the
                         neuron sweep's position axis already had once.
    """
    type: Literal["attention_layer_sampler"]
    signal_type: Literal["attention_score", "attention_weight"] = "attention_score"
    layers: list[int]
    heads: list[int] | None = None
    heads_per_layer: int | None = None
    # Source offsets relative to the destination token; 0 == self-attention.
    src_offsets: list[int]
    seed: int = 0


SignalSource = Union[
    ExplicitSignalSpec,
    UniformLayerSampler,
    AttentionLayerSampler,
    SAEFeatureSampler,
]


class FixedTokenSelector(BaseModel):
    type: Literal["fixed"] = "fixed"
    position: int

class RandomRangeTokenSelector(BaseModel):
    type: Literal["random_range"] = "random_range"
    start: int
    stop: int


class RandomPairTokenSelector(BaseModel):
    """PAIR MODE, training: each document gets ONE (dst, src) pair drawn uniformly from
    all causal pairs 0 <= src <= dst < seq_len except (0, 0) (BOS -> BOS, always 1.0).

    Uniform over PAIRS, not over dst: P(dst = d) is proportional to d + 1. Requires every
    document to be exactly max_seq_length tokens (min_seq_length >= max_seq_length), so
    the pair set is the same for every document; validate_attention_offsets enforces it.
    """
    type: Literal["random_pair"] = "random_pair"


class PairListTokenSelector(BaseModel):
    """PAIR MODE, evaluation: EVERY listed absolute (dst, src) pair on every document, one
    cached row per (document, signal, pair), all read from a single forward pass.

    Absolute positions (0 == BOS), so the same pair means the same attention cell at
    every eval length: by the causal mask the weight at (d, s) depends only on tokens
    0..d, and a length-m cache of right-truncated documents holds the same value there
    for every m > d. That is what makes the length sweep a paired comparison.
    """
    type: Literal["pair_list"] = "pair_list"
    pairs: list[tuple[int, int]]


TokenSelector = Union[FixedTokenSelector, RandomRangeTokenSelector,
                      RandomPairTokenSelector, PairListTokenSelector]

PAIR_SELECTOR_TYPES = ("random_pair", "pair_list")


def is_pair_selector(sel) -> bool:
    return getattr(sel, "type", None) in PAIR_SELECTOR_TYPES
