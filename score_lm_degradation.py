#!/usr/bin/env python3
"""
Did introspection training damage the model's language modelling?

Runs the SAME held-out tokens through the checkpoint twice -- adapter ON, then
inside `disable_adapter()` -- and compares the two next-token distributions. Both
passes share the base weights, the tokens, and the dtype, so any difference is the
adapter and nothing else.

Metrics, in order of sensitivity:

  KL(base || tuned)   mean nats per token. The sharp instrument: an NLL difference
                      can cancel across tokens (the adapter helps here, hurts there)
                      while KL cannot. This is what detects the damage a benchmark
                      score at 1B never will.
  top-1 agreement     fraction of positions where the argmax token is unchanged.
                      Interpretable, and the number to quote when someone asks
                      whether "the model still behaves the same".
  NLL / perplexity    on and off, plus the delta. Familiar, but the least sensitive.

LENGTH SPLITS matter here: introspection training runs on 32-token texts, so the
adapter has never seen a long context. Degradation can be invisible at the trained
length and obvious past it, which is why KL is also reported split at position 32.

ONLY MEANINGFUL FOR GATE-OFF CHECKPOINTS. With the input gate on, the adapter is
masked off over ordinary text during training, so a plain PEFT forward here would
apply LoRA in a regime it was never trained in -- the resulting numbers describe a
model that does not exist. The script refuses those unless --force.

Example
-------
python score_lm_degradation.py --config experiment_configs/<group>/<run>.json
"""
from __future__ import annotations

import os
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.introspector.config import ExperimentConfig
from train_introspector import INTROSPECTION_SEPARATOR
from src.introspector.training.model import load_added_token_embeddings

# KL below this (nats/token) and top-1 agreement above the second value => the
# adapter left ordinary language modelling essentially untouched. Calibration, not law.
KL_NEGLIGIBLE = 0.01
TOP1_NEGLIGIBLE = 0.99


def load_adapter_model(config: ExperimentConfig, ckpt, trim_vocab: bool = True):
    """
    Load an introspection checkpoint for ORDINARY-text evaluation.

    PEFT's save_pretrained uses save_embedding_layers="auto", which stores
    embed_tokens/lm_head whenever the vocab was resized -- and every run here resizes
    by one for the introspection separator. So the adapter cannot load onto a stock
    base; resize first, exactly as training did.

    Those rows were FROZEN during training (get_peft_model marks only LoRA
    trainable), so the extra one is still the random init from
    resize_token_embeddings: a junk logit for a token that never occurs in real text.
    It is shared by both conditions, so KL and top-1 agreement are unaffected either
    way, but it does sit in every softmax and perturbs absolute NLL/perplexity.
    trim_vocab drops it again after loading, restoring the exact base vocabulary
    (Llama-3.2 ties embed/lm_head, so both shrink together).

    Returns (model, tokenizer) with the tokenizer matching the model's final vocab.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    orig_vocab = len(tokenizer)
    added = tokenizer.add_special_tokens({"additional_special_tokens": [INTROSPECTION_SEPARATOR]})
    base = AutoModelForCausalLM.from_pretrained(
        config.model_name, device_map="auto",
        torch_dtype=torch.bfloat16 if config.training_config.bf16 else torch.float32)
    if added:
        base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, str(ckpt), autocast_adapter_dtype=False)
    # restores the separator row if the checkpoint's frozen embeddings were
    # stripped. Irrelevant once trim_vocab drops that row, but keeps this loader
    # correct for callers that do not trim.
    load_added_token_embeddings(model, str(ckpt))
    if added and trim_vocab:
        model.get_base_model().resize_token_embeddings(orig_vocab)
        print(f"[load] trimmed vocab {len(tokenizer)} -> {orig_vocab} "
              f"(dropped the untrained separator row)")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model.eval()
    return model, tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Adapter checkpoint (default: latest under the run's output dir).")
    p.add_argument("--lengths", type=str, default="32,128,512",
                   help="Context lengths to evaluate. 32 is the training length; the longer "
                        "ones test context the adapter never saw.")
    p.add_argument("--n-docs", type=int, default=200,
                   help="Documents per length (only docs long enough for that length are kept, "
                        "so no padding and no packing across document boundaries).")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--kl-chunk", type=int, default=128,
                   help="Positions per chunk when computing KL, to bound the fp32 softmax memory.")
    p.add_argument("--no-trim-vocab", action="store_true",
                   help="Keep the extra (untrained) separator row instead of trimming back to the "
                        "base vocabulary. Shared by both conditions either way, so KL/top-1 are "
                        "unaffected; only absolute NLL/perplexity shift.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if lm_degradation.json already exists (default: skip, "
                        "matching cache_signals.py / train_introspector_joint.py).")
    p.add_argument("--force", action="store_true",
                   help="Run even on a gated checkpoint (the numbers will not describe the "
                        "trained model -- see the module docstring).")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def load_docs(config: ExperimentConfig, tokenizer, lengths, n_docs):
    """
    Held-out C4 documents, tokenized WITH BOS (matching cache_signals.py and
    preprocess_example). Skips past the slice the test split consumes so this is not
    scored on text the run already evaluated on. For each length, keeps only
    documents with at least that many tokens and truncates to exactly it -- no
    padding, no cross-document packing, so every scored position is real context.
    """
    ds_cfg = config.test_dataset
    kwargs = {k: v for k, v in ds_cfg.hf_data_kwargs.items() if k != "config"}
    hf_config = ds_cfg.hf_data_kwargs.get("config")
    stream = (load_dataset(ds_cfg.name, hf_config, split=ds_cfg.split, **kwargs) if hf_config
              else load_dataset(ds_cfg.name, split=ds_cfg.split, **kwargs))
    skip = ds_cfg.max_samples or 0
    if hasattr(stream, "skip"):
        stream = stream.skip(skip)

    max_len = max(lengths)
    need = {L: n_docs for L in lengths}
    out = {L: [] for L in lengths}
    for ex in stream:
        ids = tokenizer(ex[ds_cfg.text_field], truncation=True, max_length=max_len,
                        add_special_tokens=True)["input_ids"]
        for L in lengths:
            if len(out[L]) < need[L] and len(ids) >= L:
                out[L].append(ids[:L])
        if all(len(v) >= need[L] for L, v in out.items()):
            break
    for L, docs in out.items():
        print(f"[lm] length {L}: {len(docs)} docs"
              + ("" if len(docs) == n_docs else f"  (WANTED {n_docs}; C4 ran out of long enough docs)"))
    return out


@torch.no_grad()
def compare_one_length(model, docs, length, batch_size, kl_chunk):
    """Adapter ON vs OFF on identical tokens. Returns per-position sums so the
    caller can split by position without a second pass."""
    device = model.device
    n_pos = length - 1                      # position i predicts token i+1
    kl_sum = torch.zeros(n_pos, dtype=torch.float64)
    nll_on = torch.zeros(n_pos, dtype=torch.float64)
    nll_off = torch.zeros(n_pos, dtype=torch.float64)
    top1_same = torch.zeros(n_pos, dtype=torch.float64)
    n_seq = 0

    for start in range(0, len(docs), batch_size):
        batch = torch.tensor(docs[start:start + batch_size], device=device)
        attn = torch.ones_like(batch)
        logits_on = model(input_ids=batch, attention_mask=attn).logits[:, :-1, :]
        with model.disable_adapter():
            logits_off = model(input_ids=batch, attention_mask=attn).logits[:, :-1, :]
        targets = batch[:, 1:]
        n_seq += batch.shape[0]

        # chunk over positions: a full-sequence fp32 softmax over a 128k vocab is
        # several GB, and we need two of them
        for s in range(0, n_pos, kl_chunk):
            e = min(s + kl_chunk, n_pos)
            lp_on = F.log_softmax(logits_on[:, s:e, :].float(), dim=-1)
            lp_off = F.log_softmax(logits_off[:, s:e, :].float(), dim=-1)
            # KL(base || tuned): how much probability mass the base model puts where
            # the tuned model no longer does. Forward KL from the reference.
            kl = (lp_off.exp() * (lp_off - lp_on)).sum(-1)                  # [b, chunk]
            tgt = targets[:, s:e].unsqueeze(-1)
            kl_sum[s:e] += kl.sum(0).double().cpu()
            nll_on[s:e] += (-lp_on.gather(-1, tgt).squeeze(-1)).sum(0).double().cpu()
            nll_off[s:e] += (-lp_off.gather(-1, tgt).squeeze(-1)).sum(0).double().cpu()
            top1_same[s:e] += (lp_on.argmax(-1) == lp_off.argmax(-1)).sum(0).double().cpu()

    return {"kl": kl_sum / n_seq, "nll_on": nll_on / n_seq, "nll_off": nll_off / n_seq,
            "top1_same": top1_same / n_seq, "n_seq": n_seq}


def summarize(per_pos, length, train_len=32):
    """Collapse per-position sums, and split at the training length: damage that
    only appears past position `train_len` is the adapter failing to generalize to
    context it never saw."""
    kl, on, off, top1 = (per_pos[k].numpy() for k in ("kl", "nll_on", "nll_off", "top1_same"))
    row = {
        "length": length, "n_seq": per_pos["n_seq"],
        "kl_mean": float(kl.mean()), "kl_p95": float(np.percentile(kl, 95)),
        "top1_agreement": float(top1.mean()),
        "nll_base": float(off.mean()), "nll_tuned": float(on.mean()),
        "nll_delta": float(on.mean() - off.mean()),
        "ppl_base": float(np.exp(off.mean())), "ppl_tuned": float(np.exp(on.mean())),
    }
    if length > train_len:
        row["kl_mean_within_trainlen"] = float(kl[:train_len - 1].mean())
        row["kl_mean_beyond_trainlen"] = float(kl[train_len - 1:].mean())
    return row


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(args.config.read_text())
    run_dir = Path(config.output_dir) / config.experiment_name
    out_dir = args.output_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "lm_degradation.json").exists() and not args.overwrite:
        print(f"[lm] already scored: {out_dir}/lm_degradation.json (--overwrite to redo)")
        return

    if config.training_config.disable_adapter_in_input and not args.force:
        raise SystemExit(
            "This is a GATED checkpoint: during training the adapter was masked off over "
            "ordinary text, so a plain PEFT forward applies LoRA where it never ran. Its LM "
            "behaviour is base behaviour by construction. Pass --force only if you want the "
            "counterfactual anyway.")

    ckpts = glob.glob(f"{run_dir}/checkpoint-*")
    ckpt = Path(args.checkpoint) if args.checkpoint else (
        Path(max(ckpts, key=os.path.getctime)) if ckpts else None)
    if ckpt is None:
        raise SystemExit(f"No checkpoint found under {run_dir}; pass --checkpoint.")
    print(f"[lm] config={args.config}\n[lm] checkpoint={ckpt}")

    model, tokenizer = load_adapter_model(config, ckpt, trim_vocab=not args.no_trim_vocab)
    # NB: deliberately NOT patching GatedLinear -- a gate-off adapter's training-time
    # behaviour IS plain LoRA everywhere, which is exactly what plain PEFT does here.

    lengths = [int(x) for x in args.lengths.split(",")]
    docs = load_docs(config, tokenizer, lengths, args.n_docs)

    rows = []
    for L in lengths:
        if not docs[L]:
            print(f"[lm] length {L}: no documents, skipping")
            continue
        per_pos = compare_one_length(model, docs[L], L, args.batch_size, args.kl_chunk)
        rows.append(summarize(per_pos, L))
        np.savetxt(out_dir / f"lm_kl_by_position_L{L}.csv", per_pos["kl"].numpy(),
                   delimiter=",", header="kl_nats", comments="")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "lm_degradation.csv", index=False)
    print("\n[lm] adapter ON vs OFF on identical held-out tokens:")
    print(df.to_string(index=False))

    worst = df.loc[df["kl_mean"].idxmax()]
    ok = bool(worst["kl_mean"] < KL_NEGLIGIBLE and worst["top1_agreement"] > TOP1_NEGLIGIBLE)
    verdict = (
        f"negligible: worst case (L={int(worst['length'])}) KL={worst['kl_mean']:.5f} nats/token, "
        f"top-1 agreement {worst['top1_agreement']:.4f}. Introspection training left ordinary "
        f"language modelling intact."
        if ok else
        f"NOT negligible: at L={int(worst['length'])} KL={worst['kl_mean']:.5f} nats/token and "
        f"top-1 agreement {worst['top1_agreement']:.4f} (thresholds {KL_NEGLIGIBLE}, "
        f"{TOP1_NEGLIGIBLE}). The adapter changed the model's behaviour on ordinary text, so "
        f"'introspects on itself' needs qualifying -- and if kl_mean_beyond_trainlen far exceeds "
        f"kl_mean_within_trainlen, the damage is specific to context lengths training never saw.")
    print(f"\n[lm] VERDICT: {verdict}")
    with open(out_dir / "lm_degradation.json", "w") as f:
        json.dump({"checkpoint": str(ckpt), "config": str(args.config),
                   "gated": config.training_config.disable_adapter_in_input,
                   "lengths": rows, "verdict": verdict, "negligible": ok}, f, indent=2)
    print(f"[lm] wrote {out_dir}/lm_degradation.{{csv,json}}")


if __name__ == "__main__":
    main()
