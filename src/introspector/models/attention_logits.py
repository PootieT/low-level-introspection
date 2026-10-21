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

# Attribute names the ports are installed under on a `LlamaAttention` module. Underscore-
# prefixed so they cannot collide with a real transformers field.
#
# TWO ports, not one, because the two signals live at DIFFERENT points of the same
# forward: the logit before the mask and the softmax, the weight after both. A single
# attribute would force the A and P ports of one layer to evict each other, and
# `installed_ports` asserts the slot is empty -- so the collision would at least be
# loud. It is still wrong: caching A and P in one pass is the whole reason a cell costs
# one forward.
PORT_ATTR = "_introspection_port"                 # pre-softmax attention logit (code A)
PORT_ATTR_WEIGHT = "_introspection_weight_port"   # post-softmax attention weight (code P)
PORT_ATTRS = {"logit": PORT_ATTR, "weight": PORT_ATTR_WEIGHT}


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

    # THE OTHER SIGNAL. [B, Hq, L_dst, L_src], a probability over the causal row.
    #
    # DO NOT fold these three statements back into the one-liner they came from
    # (`F.softmax(..., dtype=torch.float32).to(query.dtype)`). Split this way the kernel
    # sequence is identical when `wport is None`, so every attention_score run stays
    # bit-for-bit what it was -- that is what validate_attention_logits.py checks and
    # what keeps the A sweep's cached numbers comparable to new ones.
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # [B, Hq, L, L] fp32
    wport = getattr(module, PORT_ATTR_WEIGHT, None)
    if wport is not None:
        # Read in fp32, BEFORE the cast and BEFORE dropout, both deliberately.
        #   cast: bf16's relative precision is ~2^-8 = 4e-3, and a mid-context weight is
        #     ~1e-3 -- quantizing here would leave 2-3 significant digits, i.e. the
        #     quantization noise would be the same size as the signal's variation.
        #   dropout: with module.training=True it zeroes values or scales them by
        #     1/(1-p). Llama's attention_dropout is 0 today; this does not depend on that.
        attn_weights = wport(attn_weights)                               # [B, Hq, L, L]
    attn_weights = attn_weights.to(query.dtype)
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


def port_site(signal_type) -> str:
    """Which of the two port sites a signal type is read/written at.

    Single source of truth, so `cache_signals` and the trainer cannot disagree about
    where a signal lives. They CAN disagree silently otherwise: both branch on
    `is_attention_type`, which is True for either type, so a P config routed through the
    logit port would cache the pre-softmax value under a `P` name and an "attention
    weight" prompt without raising anything.
    """
    assert is_attention_type(signal_type), f"not an attention signal: {signal_type!r}"
    return "logit" if signal_type == "attention_score" else "weight"


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

    def _resolve(self, attn_logits):
        """Validate the index tensors against this batch and move them to its device.

        Returns (bidx, heads, dst, src), each [B]. Split out of __call__ so the
        post-softmax subclass reuses the checks verbatim instead of copying them --
        every one of them is as valid on a probability tensor as on a logit tensor.
        """
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
        return bidx, self.heads.to(device), self.dst.to(device), self.src.to(device)

    def __call__(self, attn_logits):                       # [B, Hq, L_dst, L_src]
        bidx, heads, dst, src = self._resolve(attn_logits)

        self.read = attn_logits[bidx, heads, dst, src].detach().float()   # [B]

        if self.values is None:
            return attn_logits

        out = attn_logits.clone()
        out[bidx, heads, dst, src] = self.values.to(device=attn_logits.device,
                                                    dtype=out.dtype)
        return out


class AttentionWeightPort(AttentionLogitPort):
    """Read/write port on ONE layer's POST-softmax attention weights (code P).

    READ is the parent's: gather p[b, head, dst, src] into `self.read`. The tensor is a
    probability row rather than a logit row, but every bound and the causal `src <= dst`
    check mean the same thing, so `_resolve` is reused unchanged.

    WRITE is NOT the parent's. Overwriting one cell with t leaves the row summing to
    1 - p_j + t, and no logit row softmaxes to that -- it would inject a state the model
    cannot produce and then ask it to report it. Instead the row is RENORMALIZED:

        p'_j = t,    p'_k = p_k * (1 - t) / (1 - p_j)    for k != j

    which is exactly equal, in every cell, to shifting the single pre-softmax logit z_j
    by delta = logit(t) - logit(p_j), where logit(u) = log(u/(1-u)) is the inverse of
    sigmoid -- NOT the attention logit z, which is the other thing "logit" means here.
    Sketch: with R = sum_{k != j} exp(z_k), p_j / (1 - p_j) = exp(z_j) / R, so
    logit(p_j) = z_j - log R; moving z_j alone leaves R fixed, so logit(p_j) moves by
    exactly the same delta. (Full derivation: md/attention_weight/
    1_attention_weight_experiment_plan.md, section 3-2 (a)-(c). Checked numerically by
    validate_attention_weights.py, block EQUIVALENCE.)

    That equality is what lets the A and P experiments be read side by side: both
    intervene by moving one attention logit and letting the row follow.

    The other cells of the row moving is intended, not a leak. A probability row cannot
    have one entry changed in isolation, and it mirrors A, where changing one logit also
    moves every weight in its row.
    """

    # Guard band on p_j only: the renormalization divides by (1 - p_j). The target t
    # needs no band -- see the assert in __call__.
    EPS = 1e-6

    def __call__(self, attn_probs):                        # [B, Hq, L_dst, L_src] fp32
        bidx, heads, dst, src = self._resolve(attn_probs)

        p_j = attn_probs[bidx, heads, dst, src]                      # [B]
        self.read = p_j.detach().float()                             # [B], read BEFORE the write

        if self.values is None:
            return attn_probs

        t = self.values.to(device=attn_probs.device, dtype=attn_probs.dtype)   # [B]
        # Internal invariants -> assert, and NEVER clamp. The sampler, not the user,
        # produces these. If t >= 1 or p_j >= 1 the scale (1-t)/(1-p_j) goes negative
        # and builds a row with negative probabilities -- and since the softmax is
        # already behind us, nothing downstream raises. Clamping would instead inject a
        # silently different value than the one recorded as the regression target.
        # t is checked against the CLOSED [0, 1]. It enters the row only through the
        # linear (1 - t), so t = 0 (others scaled by 1/(1-p_j)) and t = 1 (others 0) are
        # both valid probability rows; only the logit-shift reading of the write needs
        # 0 < t < 1, and nothing computes logit(t) here. This used to be (EPS, 1-EPS),
        # which killed `empirical_prob` runs: real train weights go down to 1e-9
        # (0.7% of L14 h26 @-19 sit below 1e-6), and that sampler injects them as-is.
        assert bool(((t >= 0) & (t <= 1)).all()), (
            f"attention weight target outside [0,1]: min={float(t.min())}, "
            f"max={float(t.max())}")
        assert bool((p_j < 1 - self.EPS).all()), (
            f"p_j too close to 1 to renormalize: max={float(p_j.max())}")

        scale = ((1 - t) / (1 - p_j)).view(-1, 1)                    # [B, 1]
        out = attn_probs.clone()
        row = out[bidx, heads, dst] * scale                          # [B, L_src]
        row[torch.arange(row.shape[0], device=row.device), src] = t  # target cell exactly t
        out[bidx, heads, dst] = row
        return out


class AttentionRowTap:
    """Read-only tap on a WHOLE destination row, at several source offsets and all heads.

    A port reads one cell per example, so scanning 16 layers x 32 heads x 10 offsets
    through the ordinary machinery would need 5120 ports. This reads the same
    [B, Hq, L_dst, L_src] tensor the forward already computed and gathers
    [B, Hq, n_offsets] out of it, which is what makes a full-grid spread scan cost one
    forward per batch instead of 5120.

    `dst` is per-example ([B]); `src_offsets` is a fixed list of non-positive offsets
    shared by the batch. Out-of-range (src < 0) is asserted, not clamped, for the same
    reason as in AttentionLogitPort.
    """

    def __init__(self, dst, src_offsets):
        self.dst = dst                       # [B] long
        self.src_offsets = list(src_offsets)  # e.g. [0, -1, -3, ..., -31]
        self.read = None                     # [B, Hq, n_offsets] float32

    def __call__(self, attn_probs):                        # [B, Hq, L_dst, L_src]
        B, n_heads, seq_dst, seq_src = attn_probs.shape
        assert self.dst.shape == (B,), f"dst {tuple(self.dst.shape)} != ({B},)"
        device = attn_probs.device
        dst = self.dst.to(device)
        assert int(dst.min()) >= 0 and int(dst.max()) < seq_dst, (
            f"dst out of range: [{int(dst.min())}, {int(dst.max())}] not within [0, {seq_dst})")

        # The destination row is the same for every offset, so gather it once.
        rows = attn_probs[torch.arange(B, device=device), :, dst]    # [B, Hq, L_src]
        cols = []
        for off in self.src_offsets:
            src = dst + off                                          # [B]
            assert bool(((src >= 0) & (src <= dst)).all()), (
                f"offset {off} leaves the causal range for some example: "
                f"src min={int(src.min())}, dst min={int(dst.min())}")
            cols.append(rows.gather(2, src.view(B, 1, 1).expand(B, n_heads, 1)))  # [B,Hq,1]
        self.read = torch.cat(cols, dim=2).detach().float()          # [B, Hq, n_offsets]
        return attn_probs


class AttentionCellGridTap:
    """Read-only tap on a FIXED list of absolute (dst, src) cells, for a few heads.

    PAIR MODE evaluation reads every pair of a grid (up to 599) on every document. One
    AttentionLogitPort per (head, pair) would be ~1.8K ports per forward; this gathers
    the same [B, H, P] block out of the tensor the forward already computed, in one go.

    `heads` [H] and `dst`/`src` [P] are shared by the whole batch, because a pair-list
    cache asks for the same cells in every document. Bounds and causality are asserted,
    never clamped, as in AttentionLogitPort.
    """

    def __init__(self, heads, dst, src):
        assert dst.shape == src.shape and dst.ndim == 1, (
            f"dst {tuple(dst.shape)} / src {tuple(src.shape)} must both be [P]")
        assert bool((src <= dst).all()), "src must not be after dst"
        self.heads = heads                    # [H] long
        self.dst = dst                        # [P] long
        self.src = src                        # [P] long
        self.read = None                      # [B, H, P] float32

    def __call__(self, attn_probs):                        # [B, Hq, L_dst, L_src]
        B, n_heads, seq_dst, seq_src = attn_probs.shape
        device = attn_probs.device
        heads, dst, src = (self.heads.to(device), self.dst.to(device), self.src.to(device))
        assert int(heads.min()) >= 0 and int(heads.max()) < n_heads, (
            f"head out of range: [{int(heads.min())}, {int(heads.max())}] not within [0, {n_heads})")
        assert int(src.min()) >= 0 and int(dst.max()) < seq_dst, (
            f"pair out of range: src min {int(src.min())}, dst max {int(dst.max())}, "
            f"L={seq_dst} -- a document is shorter than the grid needs")
        sub = attn_probs[:, heads]                                   # [B, H, L_dst, L_src]
        self.read = sub[:, :, dst, src].detach().float()             # [B, H, P]
        return attn_probs


class AttentionProbGradTap:
    """Make one layer's post-softmax weights a differentiable leaf, for attribution.

    Returns `alpha * p` where p is detached and marked as requiring grad, so that after
    one backward from the readout, `self.leaf.grad` holds d(prediction)/d(p) for EVERY
    (head, dst, src) of that layer at once -- 512 head-matrices from a single backward
    across the 16 layers.

    `alpha` is the Integrated Gradients path coefficient against the zero-attention
    baseline (Hao et al. 2021, "Self-Attention Attribution"): alpha = 1 gives plain
    gradient x activation, which is IG's single-step approximation, and averaging
    gradients over an alpha grid gives IG proper. IG is what the final figures use,
    because a softmax row saturates and gradient x activation under-attributes exactly
    there.

    Detaching means the scaled row does not sum to 1 for alpha < 1. That is the intended
    IG baseline (no attention at all), not a bug -- unlike AttentionWeightPort's write,
    this is not pretending to be a reachable model state.

    CONSEQUENCE OF DETACHING, and the reason token attribution needs its own pass: the
    gradient path from p BACK to q, k and ultimately the input embeddings is severed at
    every tapped layer. d(prediction)/d(p) stays correct for each layer, because the
    path FORWARD from each leaf is intact, but a backward taken with these taps
    installed reaches the embeddings only through the residual stream. Run the
    embedding-gradient pass with no taps installed.
    """

    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.leaf = None      # [B, Hq, L, L] fp32, requires_grad
        self.value = None     # [B, Hq, L, L] fp32, detached copy for grad x act

    def __call__(self, attn_probs):                        # [B, Hq, L_dst, L_src]
        leaf = attn_probs.detach().clone().requires_grad_(True)
        self.leaf = leaf
        self.value = leaf.detach()
        if self.alpha == 1.0:
            return leaf
        return leaf * self.alpha


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

    `ports` is {layer_index: port}. Removal happens in `finally` so a raised assertion
    inside the forward cannot leave a stale port attached to the model -- which would
    silently intervene on the NEXT, unrelated forward pass.

    `site` picks which of the two attributes to attach to: "logit" (pre-softmax, the
    default and what every pre-existing call site means) or "weight" (post-softmax).
    Two instances of this manager can therefore be nested to read both sites in one
    forward, because they claim different attributes.
    """

    def __init__(self, model, ports: dict, site: str = "logit"):
        assert site in PORT_ATTRS, f"unknown port site {site!r}, want one of {sorted(PORT_ATTRS)}"
        self.model = model
        self.ports = ports
        self.site = site
        self._attr = PORT_ATTRS[site]
        self._modules = []

    def __enter__(self):
        # NO PORTS -> TOUCH NOTHING. cache_signals enters this manager for every batch of
        # every signal type, with an empty dict whenever no attention signal is being
        # cached -- and attention_model_kwargs deliberately loads such models with plain
        # sdpa. Asserting the attention implementation regardless therefore failed every
        # MLP and SAE cache build on the first batch. It went unnoticed because the sweeps
        # since this manager landed all reused caches built before it. Attention runs
        # always pass ports, so they still get the assert below.
        if not self.ports:
            return self.ports
        ensure_dispatched(self.model)
        assert_attention_impl(self.model)
        for layer, port in self.ports.items():
            module = resolve_attention_module(self.model, layer)
            assert getattr(module, self._attr, None) is None, (
                f"layer {layer} already has a {self.site!r} port installed; nested/leaked port")
            setattr(module, self._attr, port)
            self._modules.append(module)
        return self.ports

    def __exit__(self, *exc):
        for module in self._modules:
            setattr(module, self._attr, None)
        self._modules = []
        return False
