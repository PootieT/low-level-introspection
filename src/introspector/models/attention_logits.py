"""Read/write access to PRE-SOFTMAX attention logits.

WHY THIS FILE EXISTS
--------------------
The attention logit -- q.k/sqrt(d_head), after RoPE and the 1/sqrt(d) scaling, BEFORE
the causal mask and the softmax -- is not the output of any module. In HuggingFace's
`transformers` it is a local variable inside `eager_attention_forward`
(transformers/models/llama/modeling_llama.py), so `register_forward_hook` can neither
read it nor write it. Two further traps:

  * The models here load with `attn_implementation=None`, which `transformers` resolves
    to **sdpa**. Under sdpa the attention function returns `attn_weights=None`: there is
    nothing to hook at all.
  * `output_attentions=True` is the wrong tool twice over -- it forces eager AND it
    hands back POST-softmax probabilities, never the logits.

So we go through the extension point `transformers` actually provides for this:
`AttentionInterface.register(name, fn)` plus `config._attn_implementation = name`. One
custom attention function gives BOTH the read (caching, alignment/drift checks) and the
write (intervention) on the same tensor, at the same point in the computation, which is
what makes the cached value and the injected value refer to the same quantity.

The rejected alternative was recomputing the logits from `q_proj`/`k_proj` outputs (what
`src/introspector/utils.py:get_l2_attn_weights` does). It re-implements RoPE and GQA by
hand -- that copy is in fact broken for Llama-3.2-1B, which is GQA (32 query heads, 8 KV
heads) and needs a `repeat_kv` the copy skips -- and, decisively, it cannot express an
intervention: changing q or k moves EVERY (dst, src) pair of that head, not one cell.

NUMERICS. This function is eager-math, not sdpa. Switching implementations perturbs
activations at the ~1e-3 level in bf16, and that perturbation accumulates into the q/k of
later layers. Caching and training must therefore use the SAME implementation or the
alignment check (want corr ~0.9999) will not pass. `attention_model_kwargs()` is the one
place that decides this, and it returns {} when no attention signal is involved, so MLP
runs keep loading under sdpa and their numbers do not move.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import AttentionMaskInterface, eager_mask
from transformers.modeling_utils import AttentionInterface
from transformers.models.llama.modeling_llama import repeat_kv

# Registered name for our attention implementation. Passed as `attn_implementation=` at
# model load time and stored in `config._attn_implementation`.
ATTN_IMPL_NAME = "introspection_logit"

# Attribute name the port is installed under on a `LlamaAttention` module. Underscore-
# prefixed so it cannot collide with a real transformers field.
PORT_ATTR = "_introspection_port"


def introspection_attention_forward(
    module,
    query,            # [B, Hq, L, d]
    key,              # [B, Hkv, L, d]
    value,            # [B, Hkv, L, d]
    attention_mask,   # [B, 1, L, L] additive, or None
    scaling,
    dropout: float = 0.0,
    **kwargs,
):
    """`eager_attention_forward` with a read/write port on the pre-softmax logits.

    Kept deliberately line-for-line with the upstream eager implementation apart from
    the port call, so that a `transformers` upgrade that changes the attention math
    shows up as a diff against a short, readable function.
    """
    key_states = repeat_kv(key, module.num_key_value_groups)        # [B, Hq, L, d]
    value_states = repeat_kv(value, module.num_key_value_groups)    # [B, Hq, L, d]

    # THE SIGNAL. [B, Hq, L_dst, L_src]
    attn_logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    port = getattr(module, PORT_ATTR, None)
    if port is not None:
        # Called BEFORE the mask is added, on purpose. In the causal region (src <= dst)
        # the mask adds 0, so masked and unmasked logits are identical there and the
        # signal is the same quantity either way; but an injected value added before the
        # mask survives, while one written after would be a value the softmax never sees
        # in the same form. Reading pre-mask also keeps the definition finite for every
        # (dst, src) pair instead of -inf outside the causal region.
        attn_logits = port(attn_logits)                             # [B, Hq, L_dst, L_src]

    attn_weights = attn_logits
    if attention_mask is not None:
        # slice to the kv length, matching upstream's handling of cached keys
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)          # [B, Hq, L, d]
    attn_output = attn_output.transpose(1, 2).contiguous()          # [B, L, Hq, d]
    return attn_output, attn_weights


AttentionInterface.register(ATTN_IMPL_NAME, introspection_attention_forward)

# REGISTERING THE ATTENTION FUNCTION ALONE IS NOT ENOUGH -- and the failure is silent.
# transformers builds the causal mask through a SECOND registry keyed by the same
# implementation name (masking_utils.create_causal_mask -> ALL_MASK_ATTENTION_FUNCTIONS),
# and `_preprocess_mask_arguments` treats a name that is missing there as a backend that
# does its own masking: it returns attention_mask=None. Our function would then be handed
# None, add no mask, and run FULLY BIDIRECTIONAL attention -- every token attending to
# its own future -- while raising nothing and producing plausible-looking numbers.
# `eager_mask` is the additive float mask the eager implementation uses, which is exactly
# what the math above expects. validate_attention_logits.py is what caught this.
AttentionMaskInterface.register(ATTN_IMPL_NAME, eager_mask)


def attention_model_kwargs(signal_specs) -> dict:
    """`from_pretrained` kwargs needed to make attention logits reachable.

    Returns {} unless some spec in `signal_specs` is an attention signal, so a neuron or
    SAE run is loaded exactly as before (sdpa) and its numbers cannot move.
    """
    if any(is_attention_type(getattr(s, "type", None)) for s in signal_specs):
        return {"attn_implementation": ATTN_IMPL_NAME}
    return {}


def is_attention_type(signal_type) -> bool:
    """True for the signal types served by this file."""
    return signal_type in ("attention_score", "attention_weight")


def ensure_dispatched(model):
    """Materialize an nnsight LanguageModel's real weights before touching its modules.

    nnsight loads lazily: until the first trace, `LanguageModel._model` is a meta model,
    and the first trace REPLACES it with the dispatched one. Anything setattr'd onto a
    module before that -- a port, a hook -- is attached to the discarded copy and is
    simply never called. Nothing raises; the port just reports nothing, which is how
    this was found. Call before installing ports on an nnsight model.
    """
    dispatch = getattr(model, "dispatch", None)
    if callable(dispatch):
        dispatch()
    return model


def resolve_attention_module(model, layer: int):
    """The real `nn.Module` that computes attention for `layer`, however it is wrapped.

    The port is installed by setattr, so this MUST return the torch module the forward
    pass actually runs -- not a proxy of it. That is the trap here: an nnsight
    LanguageModel answers `getattr(x, "layers")` with an `Envoy`, so a naive walk finds
    something that looks exactly right, accepts the attribute, and is never consulted
    during the forward. The failure is silent at install time; it only shows up as
    "the port never fired". Hence: unwrap to the underlying model FIRST
    (`nnsight.LanguageModel._model`), then walk with nn.Module-only checks.

    Handles a raw HF model, an nnsight LanguageModel, and a peft-wrapped model (each of
    which adds a `.model` level).
    """
    node = model
    # nnsight LanguageModel keeps the real HF model here; Envoy keeps it in ._module.
    for attr in ("_model", "_module"):
        inner = getattr(node, attr, None)
        if isinstance(inner, nn.Module):
            node = inner
            break

    for _ in range(6):  # deep enough for peft(HF); cheap and bounded
        layers = getattr(node, "layers", None)
        if isinstance(layers, nn.ModuleList) and hasattr(layers[0], "self_attn"):
            assert 0 <= layer < len(layers), f"layer {layer} out of range [0, {len(layers)})"
            attn = layers[layer].self_attn
            assert isinstance(attn, nn.Module), (
                f"resolved a {type(attn).__name__}, not an nn.Module -- a proxy/envoy "
                f"would accept the port and never run it")
            return attn
        node = getattr(node, "model", None)
        if not isinstance(node, nn.Module):
            break
    raise ValueError(
        f"could not find a decoder-layer stack with .self_attn under {type(model).__name__}"
    )


def assert_attention_impl(model) -> None:
    """Fail loudly if the model was not loaded with our attention implementation.

    Without this the port is simply never called: sdpa ignores it, every cached value
    would be missing and every intervention a no-op -- a silent failure, not an error.
    """
    impl = getattr(model.config, "_attn_implementation", None)
    assert impl == ATTN_IMPL_NAME, (
        f"attention signals need attn_implementation={ATTN_IMPL_NAME!r}, got {impl!r}. "
        f"Pass **attention_model_kwargs(specs) at model load."
    )


class AttentionLogitPort:
    """Batch-wise read/write port on ONE layer's pre-softmax attention logits.

    Installed as `self_attn._introspection_port` and called by
    `introspection_attention_forward` with the full [B, Hq, L_dst, L_src] tensor.

    READ mode  (`values is None`): records the logit at each example's own
        (head, dst, src) into `self.read` -- shape [B] -- and returns the tensor
        untouched.
    WRITE mode (`values` given): additionally overwrites that one cell per example.

    `heads`, `dst`, `src` (and `values`) are all per-example, length B. This is exactly
    the batch-indexing shape CLAUDE.md flags: `logits[bidx, heads, dst, src]` with index
    tensors of differing lengths broadcasts silently instead of raising, so the lengths
    are asserted on every call rather than assumed.
    """

    def __init__(self, heads, dst, src, values=None):
        self.heads = heads          # [B] long
        self.dst = dst              # [B] long
        self.src = src              # [B] long
        self.values = values        # [B] float, or None for read-only
        self.read = None            # [B] float, filled on each call

    def __call__(self, attn_logits):                       # [B, Hq, L_dst, L_src]
        B, n_heads, seq_dst, seq_src = attn_logits.shape
        assert self.heads.shape == (B,), f"heads {tuple(self.heads.shape)} != ({B},)"
        assert self.dst.shape == (B,), f"dst {tuple(self.dst.shape)} != ({B},)"
        assert self.src.shape == (B,), f"src {tuple(self.src.shape)} != ({B},)"
        if self.values is not None:
            assert self.values.shape == (B,), f"values {tuple(self.values.shape)} != ({B},)"

        # Bounds are asserted, never clamped. A clamp here would turn an out-of-range
        # source offset into a DIFFERENT, valid-looking position -- the exact shape of
        # the off-by-one that produced scripts_old_with_off_by_one_bug/.
        assert int(self.heads.min()) >= 0 and int(self.heads.max()) < n_heads, (
            f"head out of range: [{int(self.heads.min())}, {int(self.heads.max())}] "
            f"not within [0, {n_heads})")
        assert int(self.dst.min()) >= 0 and int(self.dst.max()) < seq_dst, (
            f"dst out of range: [{int(self.dst.min())}, {int(self.dst.max())}] "
            f"not within [0, {seq_dst})")
        assert int(self.src.min()) >= 0 and int(self.src.max()) < seq_src, (
            f"src out of range: [{int(self.src.min())}, {int(self.src.max())}] "
            f"not within [0, {seq_src})")
        # src > dst is outside the causal region: its logit exists but never reaches the
        # softmax, so reporting it (or injecting into it) would be measuring nothing.
        assert bool((self.src <= self.dst).all()), (
            f"src must not be after dst; worst case src={int((self.src - self.dst).argmax())}")

        device = attn_logits.device
        bidx = torch.arange(B, device=device)                       # [B]
        heads = self.heads.to(device)
        dst = self.dst.to(device)
        src = self.src.to(device)

        self.read = attn_logits[bidx, heads, dst, src].detach().float()   # [B]

        if self.values is None:
            return attn_logits

        out = attn_logits.clone()
        out[bidx, heads, dst, src] = self.values.to(device=device, dtype=out.dtype)
        return out


class PortFanOut:
    """Feed one layer's logit tensor to several ports.

    installed_ports attaches exactly ONE callable per attention module, but a layer is
    routinely read at many cells at once -- a cached cell holds heads x source offsets,
    and the alignment check reads several destination offsets. This calls each port in
    turn and returns the last one's tensor, so reads compose and a single write (there
    is at most one during intervention) still takes effect.
    """

    def __init__(self, ports):
        self.ports = list(ports)

    def __call__(self, attn_logits):
        for port in self.ports:
            attn_logits = port(attn_logits)
        return attn_logits


def fan_out_ports(ports_by_spec: dict) -> dict:
    """{spec: port} -> {layer: PortFanOut}, grouping specs that share a layer."""
    by_layer: dict = {}
    for spec, port in ports_by_spec.items():
        by_layer.setdefault(spec.layer, []).append(port)
    return {layer: PortFanOut(ports) for layer, ports in by_layer.items()}


class installed_ports:
    """Context manager: attach ports to layers, guarantee removal.

    `ports` is {layer_index: AttentionLogitPort}. Removal happens in `finally` so a raised
    assertion inside the forward cannot leave a stale port attached to the model -- which
    would silently intervene on the NEXT, unrelated forward pass.
    """

    def __init__(self, model, ports: dict):
        self.model = model
        self.ports = ports
        self._modules = []

    def __enter__(self):
        ensure_dispatched(self.model)
        assert_attention_impl(self.model)
        for layer, port in self.ports.items():
            module = resolve_attention_module(self.model, layer)
            assert getattr(module, PORT_ATTR, None) is None, (
                f"layer {layer} already has a port installed; nested/leaked port")
            setattr(module, PORT_ATTR, port)
            self._modules.append(module)
        return self.ports

    def __exit__(self, *exc):
        for module in self._modules:
            setattr(module, PORT_ATTR, None)
        self._modules = []
        return False
