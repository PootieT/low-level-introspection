#!/usr/bin/env python3
"""Validate the POST-softmax attention-weight read/write port (signal code `P`).

The counterpart of tests/validate_attention_logits.py, for the other port site. The read side
is easier to check than A's: `attn_implementation="eager"` with `output_attentions=True`
returns exactly the post-softmax probabilities, so this is not comparing a
reimplementation against a reference, it is comparing the same softmax to itself. The
tolerance is correspondingly tighter (1e-6, against A's 1e-3).

The write side is the hard part, and is the reason this file exists. A cell of a
probability row cannot simply be overwritten: the row would stop summing to 1 and no
logit row softmaxes to that, so the model would be asked to report a state it cannot
produce. AttentionWeightPort renormalizes instead. This script checks that the
renormalization is what it claims to be.

Five checks, in order of what they would catch:

  1. NO-REGRESSION  with no weight port installed, the model's output logits must be
                    BIT-IDENTICAL to sdpa-free eager... no: to the same implementation
                    before this port existed. Approximated here by checking that
                    installing a read-only port changes nothing at all, which is the
                    property the A sweep depends on.
  2. READ           port.read must equal eager's output_attentions probabilities.
  3. WRITE          row sums stay 1, the target cell is exactly t, no probability goes
                    negative, and no other head or destination row moves.
  4. EQUIVALENCE    THE IMPORTANT ONE. Renormalizing the row to put t at the target cell
                    must produce the same attention output as shifting that cell's
                    pre-softmax logit z_j by logit(t) - logit(p_j) and letting the
                    softmax renormalize. This is the empirical proof of the derivation
                    in md/attention_weight/1_attention_weight_experiment_plan.md 3-2,
                    and it is what licenses reading the A and P interventional numbers
                    side by side.
  5. DOMAIN         a target outside [0, 1] must RAISE. Passing silently is the
                    failure: the renormalization scale (1-t)/(1-p_j) goes negative and
                    builds a row of negative probabilities, and the softmax is already
                    behind us so nothing downstream complains. The endpoints t = 0 and
                    t = 1 must be ACCEPTED and give a valid row: the empirical_prob
                    sampler injects real train weights, which go down to 1e-9.

Run (from the repo root):
    python tests/validate_attention_weights.py
    python tests/validate_attention_weights.py --model meta-llama/Llama-3.2-1B --layer 8 --head 5
"""

import sys
import argparse
from pathlib import Path
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from src.introspector.models.attention_logits import (
    ATTN_IMPL_NAME, PORT_ATTR, PORT_ATTR_WEIGHT, AttentionLogitPort, AttentionWeightPort,
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
    ap.add_argument("--read-tol", type=float, default=1e-6,
                    help="max |port.read - eager attention probs|. Tighter than the A "
                         "validator's because this is the same softmax, not a rebuild")
    ap.add_argument("--equiv-tol", type=float, default=1e-5,
                    help="max |renormalized row - single-logit-shift row| (check 4)")
    return ap.parse_args()


def encode(tokenizer, device):
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    # right padding, add_special_tokens=True: same convention as cache_signals
    return tokenizer(TEXTS, return_tensors="pt", padding=True, truncation=True,
                     max_length=32, padding_side="right").to(device)


def _run_with(model, module, attr, port, batch):
    """One forward with `port` attached at `attr`, removed unconditionally afterwards."""
    setattr(module, attr, port)
    try:
        with torch.no_grad():
            return model(**batch)
    finally:
        setattr(module, attr, None)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp32 throughout. The port reads in fp32 on purpose (bf16's ~4e-3 relative precision
    # would leave 2-3 significant digits on a 1e-3 weight), so validating in bf16 would
    # measure the cast rather than the capture.
    load = dict(dtype=torch.float32, device_map=device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"[validate] loading {args.model} twice (eager / {ATTN_IMPL_NAME})")
    eager = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", **load).eval()
    ours = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation=ATTN_IMPL_NAME, **load).eval()

    batch = encode(tokenizer, device)
    B, L = batch["input_ids"].shape
    seq_lens = batch["attention_mask"].sum(1)
    dst = (seq_lens - 1).long()                        # last real token
    src = (dst + args.src_offset).clamp(min=0).long()
    bidx = torch.arange(B, device=device)
    heads = torch.full((B,), args.head, dtype=torch.long, device=device)
    print(f"[validate] B={B} L={L} seq_lens={seq_lens.tolist()} dst={dst.tolist()} "
          f"src={src.tolist()} layer={args.layer} head={args.head}")

    module = ours.model.layers[args.layer].self_attn

    # ---- 0. baseline: no ports at all ----
    with torch.no_grad():
        out_clean = ours(**batch)

    # ---- 1. NO-REGRESSION: a read-only weight port must change nothing ----
    probs_seen = {}

    def capture_all(p):
        probs_seen["p"] = p.detach().float().clone()    # [B, Hq, L_dst, L_src]
        return p

    out_read = _run_with(ours, module, PORT_ATTR_WEIGHT, capture_all, batch)
    same = torch.equal(out_clean.logits, out_read.logits)
    print(f"[validate] NO-REG read-only weight port leaves output logits bit-identical: {same}")
    assert same, "installing a read-only weight port perturbed the forward"

    # ---- 2. READ vs eager output_attentions ----
    with torch.no_grad():
        ref = eager(**batch, output_attentions=True)
    ref_attn = ref.attentions[args.layer].float()       # [B, Hq, L_dst, L_src] post-softmax
    ours_attn = probs_seen["p"]

    # Compare only rows the model actually reads. Padding ROWS are garbage in both
    # implementations and disagreeing there would be noise, not a bug.
    row_ok = torch.zeros(B, L, dtype=torch.bool, device=device)
    for b, n in enumerate(seq_lens.tolist()):
        row_ok[b, :n] = True
    m = row_ok[:, None, :, None].expand_as(ours_attn)
    diff = (ours_attn - ref_attn).abs()[m]
    print(f"[validate] READ   max|port probs - eager attentions| = {diff.max():.3e} "
          f"(mean {diff.mean():.3e}, tol {args.read_tol})")
    assert diff.max() < args.read_tol, (
        f"post-softmax capture disagrees with eager by {diff.max():.3e}")

    port_r = AttentionWeightPort(heads=heads, dst=dst, src=src)
    _run_with(ours, module, PORT_ATTR_WEIGHT, port_r, batch)
    p_j = port_r.read.to(device)
    print(f"[validate] READ   p[layer={args.layer}, head={args.head}, dst, src] = "
          f"{[f'{v:.6f}' for v in p_j.tolist()]}")
    assert torch.allclose(p_j, ours_attn[bidx, args.head, dst, src], atol=1e-6)

    # ---- 3. WRITE: row sum, exactness, non-negativity, locality ----
    t = torch.tensor([0.5, 0.25, 0.05, 0.8], device=device)[:B]
    after = {}

    def capture_written(p):
        out = AttentionWeightPort(heads=heads, dst=dst, src=src, values=t)(p)
        after["p"] = out.detach().float().clone()
        return out

    _run_with(ours, module, PORT_ATTR_WEIGHT, capture_written, batch)
    wrote = after["p"]
    rows = wrote[bidx, args.head, dst]                  # [B, L_src]
    sums = rows.sum(-1)
    print(f"[validate] WRITE  row sums after injection: {[f'{v:.6f}' for v in sums.tolist()]}")
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"row sum broke: {sums}"
    got = wrote[bidx, args.head, dst, src]
    assert torch.allclose(got, t, atol=1e-6), f"target cell is {got.tolist()}, wanted {t.tolist()}"
    assert float(wrote.min()) >= 0.0, f"negative probability produced: {float(wrote.min())}"
    print(f"[validate] WRITE  target cell exactly t, min probability {float(wrote.min()):.3e}")

    # locality: zero the touched rows in both tensors and demand bitwise equality
    a, b = wrote.clone(), ours_attn.clone()
    a[bidx, args.head, dst] = 0
    b[bidx, args.head, dst] = 0
    assert torch.equal(a, b), (
        f"write leaked outside the target row: {int((a != b).sum())} cells changed")
    print("[validate] WRITE  every other head and destination row is bit-identical")

    # the intended coupling INSIDE the row: p'_k = p_k (1-t)/(1-p_j)
    scale = ((1 - t) / (1 - p_j)).view(-1, 1)
    want = ours_attn[bidx, args.head, dst] * scale
    want[bidx, src] = t
    assert torch.allclose(rows, want, atol=1e-6), "in-row renormalization is not p_k(1-t)/(1-p_j)"
    print("[validate] WRITE  in-row cells follow p_k(1-t)/(1-p_j) (intended coupling)")

    # ---- 4. EQUIVALENCE: renormalization == shifting one attention logit ----
    # Capture the pre-softmax logits so delta can be computed, then inject
    # z_j + logit(t) - logit(p_j) through the LOGIT port and compare the resulting
    # attention output of this layer against the renormalized one.
    logits_seen = {}

    def capture_logits(z):
        logits_seen["z"] = z.detach().float().clone()
        return z

    _run_with(ours, module, PORT_ATTR, capture_logits, batch)
    z = logits_seen["z"]
    z_j = z[bidx, args.head, dst, src]
    delta = torch.log(t / (1 - t)) - torch.log(p_j / (1 - p_j))
    port_z = AttentionLogitPort(heads=heads, dst=dst, src=src, values=z_j + delta)
    via_logit = {}

    def capture_after_logit_write(p):                   # read the POST-softmax result
        via_logit["p"] = p.detach().float().clone()
        return p

    setattr(module, PORT_ATTR, port_z)
    setattr(module, PORT_ATTR_WEIGHT, capture_after_logit_write)
    try:
        with torch.no_grad():
            out_via_logit = ours(**batch)
    finally:
        setattr(module, PORT_ATTR, None)
        setattr(module, PORT_ATTR_WEIGHT, None)

    d_rows = (via_logit["p"][bidx, args.head, dst] - rows).abs().max()
    print(f"[validate] EQUIV  max|renormalized row - single-logit-shift row| = {d_rows:.3e} "
          f"(tol {args.equiv_tol})")
    assert d_rows < args.equiv_tol, (
        f"renormalization is NOT a single-logit shift ({d_rows:.3e}); the A and P "
        f"interventions are then different operations and cannot be compared")
    print(f"[validate] EQUIV  delta = logit(t) - logit(p_j) = "
          f"{[round(v, 4) for v in delta.tolist()]}")

    # ---- 5. DOMAIN: the closed [0, 1] is accepted, anything outside must raise ----
    for edge_val, what in ((0.0, "t = 0"), (1e-9, "t = 1e-9"), (1.0, "t = 1")):
        t = torch.full((B,), edge_val, device=device)
        out = AttentionWeightPort(heads=heads, dst=dst, src=src, values=t)(ours_attn)
        rows = out[bidx, args.head, dst]                     # [B, L_src]
        sums = rows.sum(-1)                                  # [B]
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"{what}: row sums {sums}"
        assert bool((rows >= 0).all()), f"{what}: negative probability {float(rows.min())}"
        assert torch.equal(rows[bidx, src], t.to(rows.dtype)), f"{what}: target cell is not t"
        print(f"[validate] DOMAIN {what:8s} -> accepted, valid row")
    for bad_val, what in ((-0.1, "t < 0"), (1.5, "t > 1")):
        bad = torch.full((B,), bad_val, device=device)
        try:
            AttentionWeightPort(heads=heads, dst=dst, src=src, values=bad)(ours_attn)
        except AssertionError:
            print(f"[validate] DOMAIN {what:8s} -> raised, as it must")
        else:
            raise SystemExit(
                f"[validate] FAILED: {what} was accepted. A target outside [0,1] makes "
                f"(1-t)/(1-p_j) negative and produces a row of negative probabilities, "
                f"and nothing downstream would raise.")

    print("[validate] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
