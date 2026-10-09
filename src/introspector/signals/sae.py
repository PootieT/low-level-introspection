"""Gemma Scope (and compatible) SAE loading for introspection targets.

We deliberately do NOT run the base model in TransformerLens. We load only the
SAE weights + encode/decode math + normalization via `sae_lens`, then apply them
to residual-stream activations captured from the HF model with nnsight (during
caching) or modified in a forward hook (during steering). This keeps the whole
introspection pipeline on HF + nnsight, reusing the community-validated SAE
conventions without a second model runtime.

Two invariants EVERY downstream component must share, or the targets are silently
wrong (garbage feature activations, mislabeled steering):
  1. the hook SITE the SAE was trained on -- see `resolve_sae_hook_path` in
     models/signal_paths.py; for Gemma Scope res SAEs this is the decoder layer
     output (== TransformerLens `blocks.{L}.hook_resid_post`);
  2. this `encode()` -- which folds in Gemma Scope's input normalization and the
     JumpReLU threshold. Do not re-implement the encoder by hand elsewhere.

Nothing here can be exercised on the Mac editing box (no torch / sae_lens); it is
meant to run on the cluster. Imports are kept lazy so the module is importable
for schema/config work without the heavy deps present.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# torch is imported lazily inside methods so this module imports without it.


def default_sae_id(site: str, width: str, layer: int) -> str:
    """sae_lens `sae_id` for a Gemma Scope *canonical* SAE at (site, width, layer).

    Canonical = the single curated L0 pick per (site, width, layer) that
    sae_lens/Neuronpedia expose as the default. Non-canonical variants (specific
    l0 buckets) use a different id and are out of scope for the first pass.
    """
    # NOTE: for canonical releases the width label in the release string and the
    # sae_id both carry the width; keep them consistent (e.g. "16k").
    return f"layer_{layer}/width_{width}/canonical"


@dataclass
class SAEBundle:
    """Thin wrapper around a loaded sae_lens SAE, exposing exactly what caching,
    steering, and validation need. All tensors live on `device` in `dtype`
    (default float32 -- Gemma Scope weights are float32 and encoding in bf16 loses
    the small-magnitude features we care about)."""

    sae: object              # the underlying sae_lens.SAE
    release: str
    sae_id: str
    site: str
    width: str
    layer: int
    hook_name: str           # sae_lens cfg hook name, e.g. "blocks.9.hook_resid_post"
    d_sae: int
    device: str
    dtype: object

    # ---- core ops (apply to residual captured at the trained hook site) ----
    def encode(self, resid):
        """resid [..., d_model] (raw HF residual at the hook site) -> feature acts
        [..., d_sae]. Casts to the SAE dtype/device first."""
        x = resid.to(device=self.device, dtype=self.dtype)
        return self.sae.encode(x)

    def decode(self, feats):
        """feature acts [..., d_sae] -> reconstructed residual [..., d_model] in
        RAW activation space (sae_lens un-normalizes internally if the SAE was
        trained with input normalization)."""
        f = feats.to(device=self.device, dtype=self.dtype)
        return self.sae.decode(f)

    def reconstruct(self, resid):
        """Round-trip encode->decode; used by the validation gate for FVU."""
        return self.decode(self.encode(resid))

    # ---- per-feature accessors (steering / diagnostics) ----
    def decoder_direction(self, index: int):
        """Raw decoder row W_dec[index] : [d_model]. This is the direction (and
        magnitude) a unit of feature activation writes into the residual. Steering
        uses the difference-of-decode below rather than this directly, so any
        output un-normalization is handled correctly; this is for diagnostics."""
        return self.sae.W_dec[index].detach()

    def threshold(self, index: Optional[int] = None):
        """JumpReLU threshold(s). Below it the feature is exactly 0, at/above it
        the activation jumps to a positive value -- relevant for the sparse
        sampler and detection binning. Returns a scalar if `index` given, else the
        full [d_sae] vector. None if the SAE is not JumpReLU."""
        thr = getattr(self.sae, "threshold", None)
        if thr is None:
            return None
        thr = thr.detach()
        return thr if index is None else thr[index]

    def steer_delta(self, resid, index: int, target_value):
        """Residual delta (RAW space) that clamps feature `index` to
        `target_value`, leaving every other feature's contribution unchanged:

            delta = decode(f{index:=target}) - decode(f)

        computed as a difference so any input/output normalization sae_lens
        applies cancels correctly. Returns [..., d_model] to be ADDED to resid.

        resid: [..., d_model] raw residual at the hook site.
        index: one feature id for every row, or a [bs] integer tensor giving each row
            of a [bs, d_model] resid its OWN feature (multi-feature training/eval).
        target_value: scalar or broadcastable to resid[..., 0].
        """
        import torch

        feats = self.encode(resid)                       # [..., d_sae]
        base = self.decode(feats)                        # [..., d_model]
        feats_clamped = feats.clone()
        tv = torch.as_tensor(target_value, device=feats.device, dtype=feats.dtype)
        if torch.is_tensor(index) and index.ndim == 1:
            if feats.ndim != 2 or index.shape[0] != feats.shape[0]:
                raise ValueError(f"per-row index {tuple(index.shape)} needs a [bs, d_model] "
                                 f"resid with bs={index.shape[0]}; got feats {tuple(feats.shape)}")
            rows = torch.arange(feats.shape[0], device=feats.device)
            feats_clamped[rows, index.to(feats.device)] = tv.expand(feats.shape[0])
        else:
            feats_clamped[..., index] = tv
        steered = self.decode(feats_clamped)             # [..., d_model]
        return (steered - base).to(dtype=resid.dtype, device=resid.device)


def load_sae(
    release: str = "gemma-scope-2b-pt-res-canonical",
    *,
    site: str = "res",
    width: str = "16k",
    layer: int = 9,
    sae_id: Optional[str] = None,
    device: Optional[str] = None,
    dtype=None,
) -> SAEBundle:
    """Load a Gemma Scope SAE via sae_lens and wrap it.

    `release`/`site`/`width`/`layer` mirror SAEFeatureSpec. `sae_id` overrides the
    default canonical id if you need a specific L0 variant.

    sae_lens changed the `from_pretrained` return signature across versions
    (SAE vs (SAE, cfg, sparsity)); we normalize both. Verify the resolved
    `hook_name` against `resolve_sae_hook_path` before trusting any activations --
    that check is the whole point of the validation gate.
    """
    import torch
    from sae_lens import SAE

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float32
    if sae_id is None:
        sae_id = default_sae_id(site, width, layer)

    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    # normalize (SAE) vs (SAE, cfg_dict, sparsity)
    sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    sae = sae.to(device=device, dtype=dtype)

    cfg = getattr(sae, "cfg", None)
    hook_name = getattr(cfg, "hook_name", None) or getattr(cfg, "metadata", None) and \
        getattr(cfg.metadata, "hook_name", None)
    d_sae = int(getattr(sae, "d_sae", None) or getattr(cfg, "d_sae"))

    return SAEBundle(
        sae=sae, release=release, sae_id=sae_id, site=site, width=width,
        layer=layer, hook_name=hook_name or "", d_sae=d_sae,
        device=device, dtype=dtype,
    )


def load_sae_for_spec(spec, device: Optional[str] = None, dtype=None) -> SAEBundle:
    """Convenience: build a bundle from an SAEFeatureSpec (or the sampler config).
    Any object exposing sae_release/sae_site/sae_width/layer works."""
    return load_sae(
        release=spec.sae_release, site=spec.sae_site, width=spec.sae_width,
        layer=spec.layer, device=device, dtype=dtype,
    )
