#!/usr/bin/env python3
"""Validate the pre-softmax attention-logit read/write port.

The attention logit is not a module output, so it is captured by a custom attention
implementation rather than a hook (see src/introspector/models/attention_logits.py).
That means the capture is a REIMPLEMENTATION of attention, and a reimplementation needs
a reference to be checked against -- otherwise a wrong-but-plausible number (a missing
`repeat_kv` under GQA, a missing RoPE, the wrong scaling) flows all the way into the
cache and looks like a hard-to-learn signal instead of a bug.

Three checks, in order of what they would catch:

  1. READ    softmax(port_logits + causal_mask) must equal the attention probabilities
             that `attn_implementation="eager"` reports via output_attentions. This pins
             the whole expression -- RoPE, GQA expansion, the 1/sqrt(d) scaling and the
             mask -- against transformers' own implementation.
  2. DRIFT   how far our implementation moves the model's output logits relative to
             sdpa. Not a pass/fail, a number to record: it is why caching and training
             must use the SAME implementation.
  3. WRITE   injecting a value into one cell must change that head's attention row in
             the expected direction AND leave every other head's row bit-identical.

Run (from the repo root):
    python validate_attention_logits.py
    python validate_attention_logits.py --model meta-llama/Llama-3.2-1B --layer 8 --head 5
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.introspector.models.attention_logits import (
    ATTN_IMPL_NAME, AttentionLogitPort, installed_ports,
)

TEXTS = [
    "The quick brown fox jumps over the lazy dog and then keeps running down the road.",
    "Attention heads route information between tokens; some of them attend to the first token.",
    "introspection research is cool, and the model should be able to report its own state",
    "banana apple quizzes love fruits the the the the the repeated tokens here",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--head", type=int, default=5)
    ap.add_argument("--src-offset", type=int, default=-3)
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max |our softmax(logits) - eager attn_weights| allowed")
    return ap.parse_args()


def encode(tokenizer, device):
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    # right padding, add_special_tokens=True: same convention as cache_signals
    return tokenizer(TEXTS, return_tensors="pt", padding=True, truncation=True,
                     max_length=32, padding_side="right").to(device)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp32 throughout: the point is to compare two implementations of the same formula,
    # and bf16 rounding would blur the very difference we are trying to bound.
    load = dict(dtype=torch.float32, device_map=device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"[validate] loading {args.model} three times (eager / {ATTN_IMPL_NAME} / sdpa)")
    eager = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager", **load).eval()
    ours = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation=ATTN_IMPL_NAME, **load).eval()

    batch = encode(tokenizer, device)
    B, L = batch["input_ids"].shape
    seq_lens = batch["attention_mask"].sum(1)
    dst = (seq_lens - 1).long()                        # last real token
    src = (dst + args.src_offset).clamp(min=0).long()
    print(f"[validate] B={B} L={L} seq_lens={seq_lens.tolist()} "
          f"dst={dst.tolist()} src={src.tolist()} layer={args.layer} head={args.head}")

    # ---- 1. READ ----
    with torch.no_grad():
        ref = eager(**batch, output_attentions=True)
    ref_attn = ref.attentions[args.layer].float()      # [B, Hq, L_dst, L_src] post-softmax

    # a port that reads the WHOLE matrix, not one cell: keep it for the comparison
    captured = {}

    def capture_all(attn_logits):
        captured["logits"] = attn_logits.detach().float().clone()   # [B, Hq, L, L]
        return attn_logits

    module = ours.model.layers[args.layer].self_attn
    setattr(module, "_introspection_port", capture_all)
    try:
        with torch.no_grad():
            out_ours = ours(**batch)
    finally:
        setattr(module, "_introspection_port", None)

    logits = captured["logits"]                        # [B, Hq, L, L]
    # Rebuild the additive mask transformers would have applied: -inf where the source
    # is after the destination (causal) OR is a padding token. Getting only the causal
    # half right is not enough -- with right padding the pad COLUMNS would otherwise
    # take softmax mass here and not in the reference, which is the difference this
    # check would then report as a capture bug.
    keep = batch["attention_mask"][:, None, None, :].bool()        # [B, 1, 1, L_src]
    causal = torch.ones(L, L, dtype=torch.bool, device=logits.device).tril()[None, None]
    allowed = keep & causal                                        # [B, 1, L_dst, L_src]
    mask = torch.where(allowed, 0.0, float("-inf")).to(logits.dtype)
    ours_attn = torch.softmax(logits + mask, dim=-1)               # [B, Hq, L, L]

    # Compare only rows the model actually reads: padding rows are garbage in BOTH
    # implementations (they differ in how the pad mask is built), and disagreeing there
    # would be noise, not a bug.
    row_ok = torch.zeros(B, L, dtype=torch.bool, device=logits.device)
    for b, n in enumerate(seq_lens.tolist()):
        row_ok[b, :n] = True
    m = row_ok[:, None, :, None].expand_as(ours_attn)
    diff = (ours_attn - ref_attn).abs()[m]
    print(f"[validate] READ   max|softmax(port logits) - eager attentions| = {diff.max():.3e} "
          f"(mean {diff.mean():.3e}, tol {args.tol})")
    assert diff.max() < args.tol, (
        f"attention logit capture disagrees with eager attention by {diff.max():.3e}; "
        f"the recomputation (RoPE / repeat_kv / scaling / mask) is wrong")

    # the cell the experiment actually reads
    bidx = torch.arange(B, device=logits.device)
    cell = logits[bidx, args.head, dst, src]
    print(f"[validate] READ   logit[layer={args.layer}, head={args.head}, dst, src] = "
          f"{[round(v, 4) for v in cell.tolist()]}")

    # ---- 2. DRIFT vs sdpa ----
    sdpa = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="sdpa", **load).eval()
    with torch.no_grad():
        out_sdpa = sdpa(**batch)
    d = (out_ours.logits - out_sdpa.logits).abs()
    print(f"[validate] DRIFT  |ours - sdpa| output logits: max={d.max():.3e} mean={d.mean():.3e} "
          f"(fp32; expect larger in bf16 -- this is why caching and training must use "
          f"the SAME implementation)")

    # ---- 3. WRITE ----
    inject = torch.full((B,), 12.5, device=device)
    port = AttentionLogitPort(heads=torch.full((B,), args.head, dtype=torch.long),
                              dst=dst, src=src, values=inject)
    after = {}

    def capture_after(attn_logits):
        out = port(attn_logits)
        after["logits"] = out.detach().float().clone()
        return out

    setattr(module, "_introspection_port", capture_after)
    try:
        with torch.no_grad():
            ours(**batch)
    finally:
        setattr(module, "_introspection_port", None)

    written = after["logits"][bidx, args.head, dst, src]
    assert torch.allclose(written, inject, atol=1e-4), f"write did not land: {written.tolist()}"
    print(f"[validate] WRITE  target cell now {written.tolist()[:2]}... (injected {inject[0].item()})")

    # everything except the written cells must be untouched, bit for bit
    delta = (after["logits"] - logits)
    expected = torch.zeros_like(delta)
    expected[bidx, args.head, dst, src] = inject - cell
    assert torch.equal(delta, expected), (
        f"write touched cells it should not: {int((delta != expected).sum())} extra cells changed")
    print("[validate] WRITE  every other (head, dst, src) cell is bit-identical")

    print("[validate] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
