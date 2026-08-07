#!/usr/bin/env python3
"""
GATE: validate that a Gemma Scope SAE is being applied to the right residual site
before any SAE feature is cached or steered.

Everything in the SAE pipeline (caching, steering, training targets) is silently
wrong if the residual we feed the SAE encoder is not the site the SAE was trained
on. This script forces that check to the surface:

  1. resolve the HF hook path (resolve_sae_hook_path) and print it next to the
     SAE's own recorded `hook_name` (sae_lens cfg) -- they must describe the same
     site (e.g. HF layer-9 output  <->  blocks.9.hook_resid_post).
  2. run a batch of real text, encode the captured residual, and report:
       - FVU (fraction of variance unexplained) of the SAE reconstruction. A
         correctly-aligned Gemma Scope res SAE reconstructs well (rule-of-thumb
         FVU well under ~0.35); a misaligned hook gives near-1 / nonsense.
       - L0 (mean active features/token). Should be a modest count (order tens),
         not ~0 (dead / wrong site) and not ~d_sae (no thresholding / wrong site).
  3. dump the top-activating features with their peak token + context, so you can
     open those feature ids on Neuronpedia and confirm the concept matches.

Run on the cluster (needs torch + sae_lens + the 2b model), NOT the Mac box:

  python validate_sae_hook.py \
      --model google/gemma-2-2b \
      --sae-release gemma-scope-2b-pt-res-canonical \
      --layer 9 --width 16k --site res \
      --n-texts 64 --max-seq-length 64
"""
from __future__ import annotations

import pdb
import argparse
import operator

from nnsight import LanguageModel
import torch
from datasets import load_dataset

from src.introspector.models.signal_paths import resolve_sae_hook_path
from src.introspector.signals.sae import load_sae


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-2b",
                   help="MUST be the base model the SAE was trained on (not -it).")
    p.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--width", default="16k")
    p.add_argument("--site", default="res", choices=["res", "mlp", "attn"])
    p.add_argument("--dataset", default="allenai/c4")
    p.add_argument("--dataset-config", default="en")
    p.add_argument("--split", default="validation")
    p.add_argument("--n-texts", type=int, default=64)
    p.add_argument("--max-seq-length", type=int, default=64)
    p.add_argument("--top-features", type=int, default=10)
    p.add_argument("--context", type=int, default=6,
                   help="tokens of left-context to print around a feature's peak token.")
    # rough pass thresholds (layer-dependent; treat as smell test, not law)
    p.add_argument("--fvu-warn", type=float, default=0.35)
    p.add_argument("--l0-min", type=float, default=2.0)
    return p.parse_args()


@torch.no_grad()
def capture_residual(model, hook_path, tokens):
    """nnsight trace: grab the decoder-layer output (residual post-block) at every
    position."""
    getter = operator.attrgetter(hook_path)
    with model.trace(tokens):
        resid = getter(model).detach().save()   # [B, T, D]
    return resid.value if hasattr(resid, "value") else resid


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[gate] loading model {args.model}")
    model = LanguageModel(args.model, device_map="auto")
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[gate] loading SAE {args.sae_release} layer={args.layer} width={args.width}")
    bundle = load_sae(release=args.sae_release, site=args.site, width=args.width,
                      layer=args.layer, device=device, dtype=torch.float32)

    hook_path = resolve_sae_hook_path(model, args.layer, site=args.site, path_to_output=True)
    print("\n=== HOOK-SITE ALIGNMENT ===")
    print(f"  HF capture path      : {hook_path}")
    print(f"  SAE recorded hook    : {bundle.hook_name or '(none in cfg -- verify manually!)'}")
    print(f"  expected TL name     : blocks.{args.layer}.hook_resid_post")
    print(f"  d_sae                : {bundle.d_sae}")
    print("  -> the HF path and the SAE recorded hook must describe the SAME site.")

    # ---- data ----
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    texts = [ex["text"] for ex in ds.take(args.n_texts)]
    print("=== Example texts ===")
    print(f"1: {texts[0]}")
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=args.max_seq_length, padding_side="right")
    mask = enc["attention_mask"].bool()                    # [B, T]

    resid = capture_residual(model, hook_path, enc).float() # [B, T, D]
    feats = bundle.encode(resid)                            # [B, T, d_sae]
    recon = bundle.decode(feats)                            # [B, T, D]

    # ---- exclude BOS before scoring. gemma-2 parks a massive-activation
    #      attention sink on <bos> (residual norm ~thousands) that the SAE is not
    #      trained to reconstruct; left in, its squared error dwarfs everything and
    #      drives FVU > 1 while every "top feature" is just the sink. Standard
    #      Gemma Scope reconstruction evals drop it. ----
    keep = mask.clone()
    if tok.bos_token_id is not None:
        keep &= (enc["input_ids"] != tok.bos_token_id)
    keep = keep.to(resid.device)

    R = resid[keep]                                        # [N, D]
    Rhat = recon[keep].to(R.device, R.dtype)
    Fk = feats[keep]                                       # [N, d_sae]
    num = ((R - Rhat) ** 2).sum().item()
    den = ((R - R.mean(0, keepdim=True)) ** 2).sum().item()
    fvu = num / den if den > 0 else float("nan")
    l0 = (Fk > 0).float().sum(-1).mean().item()
    resid_norm = R.norm(dim=-1).mean().item()

    # confirm the sink is what was inflating FVU: report bos residual norm for contrast
    bos_sel = (mask & (enc["input_ids"] == tok.bos_token_id)).to(resid.device)
    bos_norm = resid[bos_sel].norm(dim=-1).mean().item() if bos_sel.any() else float("nan")

    print("\n=== RECONSTRUCTION / SPARSITY (BOS excluded) ===")
    print(f"  tokens kept          : {R.shape[0]}  (bos/pad dropped)")
    print(f"  mean |resid| (kept)  : {resid_norm:.3f}")
    print(f"  mean |resid| (bos)   : {bos_norm:.3f}   <- sink, excluded")
    print(f"  FVU (1-EV)           : {fvu:.4f}   (want << {args.fvu_warn})")
    print(f"  L0 (active/token)    : {l0:.2f}    (want > {args.l0_min}, << {bundle.d_sae})")

    # ---- top features + peak token for Neuronpedia eyeballing ----
    print("\n=== TOP FEATURES (open these ids on Neuronpedia and confirm the concept) ===")
    max_per_feat, _ = feats.masked_fill(~keep.unsqueeze(-1), 0.0).reshape(-1, bundle.d_sae).max(0)
    top_vals, top_ids = max_per_feat.topk(min(args.top_features, bundle.d_sae))
    ids_flat = enc["input_ids"]
    for rank, (fid, val) in enumerate(zip(top_ids.tolist(), top_vals.tolist())):
        fmap = feats[..., fid].masked_fill(~keep, -1.0)     # [B, T]
        b, t = divmod(int(fmap.argmax().item()), fmap.shape[1])
        lo = max(0, t - args.context)
        ctx = tok.decode(ids_flat[b, lo:t + 1])
        peak_tok = tok.decode(ids_flat[b, t])
        print(f"  #{rank:<2} feature {fid:>6}  act={val:7.3f}  peak='{peak_tok}'  ctx=...{ctx!r}")

    # ---- verdict ----
    ok_fvu = fvu < args.fvu_warn
    ok_l0 = args.l0_min < l0 < 0.5 * bundle.d_sae
    print("\n=== VERDICT ===")
    print(f"  FVU ok : {ok_fvu}   L0 ok : {ok_l0}")
    if ok_fvu and ok_l0:
        print("  PASS (smell test). Still confirm the top-feature concepts match "
              "Neuronpedia before caching at scale.")
    else:
        print("  FAIL. Likely wrong hook site / normalization / model variant. "
              "Do NOT cache features until this passes. Check: base (not -it) model, "
              "the resolved hook path vs the SAE's recorded hook, and the sae_id.")


if __name__ == "__main__":
    main()
