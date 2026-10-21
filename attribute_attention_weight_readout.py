#!/usr/bin/env python3
"""Attribute a trained introspector's prediction back to the model's own computation.

The question this exists to answer is the first one in the experiment plan: when the
model reports an attention weight, does the prediction actually DEPEND on that weight, or
has it inferred the number from the context? A high endogenous correlation cannot tell
those apart -- the attention-logit sweep reached corr 0.894 against a layer-0 text
ceiling of 0.987, i.e. nearly all of it was reachable without reading anything internal.

WHAT IS ATTRIBUTED TO, AND WHY IT IS ACTIVATIONS
------------------------------------------------
Attribution here is to ACTIVATIONS (the attention probability matrix, the input
embeddings), never to weight matrices (W_Q/W_K/W_V, W_gate/W_up/W_down). Four reasons:

  1. The question is example-conditional. "Which computation carried THIS weight into
     THIS report" is per-example and per-position; a weight matrix is shared across every
     position and example, so attributing to W_Q answers a different question -- which
     parameter matters on average -- and that one ablate_lora_modules.py already covers.
  2. Only activations admit an exact decomposition. The residual stream is a SUM of
     component outputs, so with the final RMSNorm frozen the per-component contributions
     add to the prediction exactly. There is no such decomposition over W_Q or W_gate.
  3. It is the convention (Elhage et al. 2021; Wang et al. 2022; Hao et al. 2021;
     Nanda 2023 attribution patching; Syed et al. 2023 EAP).
  4. W_Q and W_K reach the prediction only through q, k and p, so attributing to p
     captures their whole routing effect -- and p is what this experiment is about.
     Their scales are arbitrary relative to each other, so a Layer x Head grid built from
     the two matrices would not be comparable across them anyway.

METHOD PER FIGURE
-----------------
  attention-weight triangles   gradient x activation (a fast screen) and INTEGRATED
                               GRADIENTS (the figure). There is no exact decomposition
                               over individual attention edges, and a softmax row
                               saturates -- exactly where gradient x activation
                               under-attributes -- so the published panel uses IG against
                               the zero-attention baseline (Hao et al. 2021, "Self-
                               Attention Attribution").
  token attribution            gradient x embedding, and IG to a zero-embedding baseline.
                               Run in its OWN backward with no taps installed: the taps
                               detach p, which severs the gradient path from p back to
                               the embeddings.

  The Layer x Head and Layer x Position grids are deliberately NOT produced here. They
  come from the EXACT additive residual decomposition already in
  trace_introspection_circuit.py, which needs no gradients and is not an approximation.
  Running an approximate method for a quantity available in closed form is strictly worse.

AGGREGATING ACROSS EXAMPLES
---------------------------
An [N, N] triangle only means something averaged if position i is the same thing in every
example. The input text is pinned to 32 tokens, so all length variation comes from the
introspection prompt: one of four templates is drawn at random per example, and the
source/destination token strings differ in length. Hence:

  1. --pin-template fixes the wording (attention_weight_prompt takes a `template_idx`;
     None keeps the random draw, so training behaviour is unchanged).
  2. examples are bucketed by EXACT tokenized length and only the modal bucket is
     averaged -- even with the template pinned, " the" is one token and " Brixton" three.
  3. positions are labelled RIGHT-ALIGNED (0 = the readout token). With N fixed and the
     text at absolute [0, 31], left and right alignment coincide, so the text block and
     the prompt block are aligned at once. That would fail if the text length were free.

Run (from the repo root):
    python attribute_attention_weight_readout.py --config <config.json>
    python attribute_attention_weight_readout.py --config <config.json> \\
        --attr-method ig --ig-steps 20 --n-examples 200
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.introspector.config import ExperimentConfig
from src.introspector.dataset_utils import apply_signal_filter
from src.introspector.models.attention_logits import (
    AttentionProbGradTap, attention_model_kwargs, installed_ports, is_attention_type,
    port_site,
)
from src.introspector.prompts import PROMPT_REGISTRY
from src.introspector.training.collator import LoRAGatingCollator
from src.introspector.training.model import (
    IntrospectionRegressionModel, logit_moments, signal_range_key,
    load_added_token_embeddings, load_regression_head,
    replace_lora_with_gated_version, replace_model_with_gated_version,
)
from train_introspector import (
    INTROSPECTION_SEPARATOR, drop_invalid_positions, preprocess_example,
)
from train_introspector_joint import UngatedCollator
from src.introspector.training.model import resize_embeddings_for_tokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="defaults to the newest checkpoint under the config's output dir")
    ap.add_argument("--attr-method", choices=["gradxact", "ig"], default="ig")
    ap.add_argument("--ig-steps", type=int, default=20,
                    help="IG path steps; 1 reduces to gradient x activation")
    ap.add_argument("--n-examples", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pin-template", type=int, default=0,
                    help="index into the prompt template list; -1 leaves it random "
                         "(which makes an averaged per-token map meaningless)")
    ap.add_argument("--success-threshold-std", type=float, default=0.25,
                    help="keep examples whose |pred - target| is under this many TARGET "
                         "STDS. Relative, not absolute: on a cell with p~1e-3 an absolute "
                         "0.02 admits every example, and on a BOS cell almost none")
    ap.add_argument("--no-contrast-worst", action="store_true",
                    help="skip the worst-n map. It is on by default because a mean "
                         "triangle hides a bimodal one")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# ----------------------------------------------------------------------
# Loading. Deliberately NOT trace_introspection_circuit.load_model_and_data: that one
# resolves the cache as cache_dir/experiment_name (an older layout), and its preprocess
# emits only the MLP columns, so an attention row would reach the forward without
# src_token_position. This mirrors train_introspector_joint.main() instead, which is the
# code path these checkpoints were actually trained under.
# ----------------------------------------------------------------------
def load_everything(config: ExperimentConfig, checkpoint: Path, pin_template: int | None):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [INTROSPECTION_SEPARATOR]})

    cache = config.dataset_cache_dir
    train_raw = load_from_disk(str(cache / "train"))
    test_raw = load_from_disk(str(cache / "test"))
    if config.signal_filter:
        train_raw = apply_signal_filter(train_raw, config.signal_filter)
        test_raw = apply_signal_filter(test_raw, config.signal_filter)

    df = train_raw.to_pandas()
    signal_type = df["signal_type"].iloc[0]
    assert signal_type == "attention_weight", (
        f"this script attributes an attention_weight readout; the cache holds "
        f"{signal_type!r}")
    group_cols = ["signal_layer", "signal_index", "src_offset"]
    signal_ranges = {}
    for key, vals in df.groupby(group_cols)["activation"]:
        assert float(vals.min()) >= 0.0 and float(vals.max()) <= 1.0, (
            f"cell {key} has activations outside [0,1]; the cache holds pre-softmax "
            f"logits under a 'P' name")
        signal_ranges[signal_range_key(*key)] = {
            "min": float(vals.min()), "max": float(vals.max()),
            "mean": float(vals.mean()), "std": float(vals.std()),
            **logit_moments(vals.tolist())}

    prompt_fn = PROMPT_REGISTRY[config.training_config.prompt_type]
    if pin_template is not None:
        # Pin the wording so a given token index is the same word in every example.
        # Threaded as a kwarg rather than by monkeypatching random.choice, so nothing
        # else in the process is affected.
        base_fn = prompt_fn

        def prompt_fn(signal_type, **kwargs):      # noqa: F811 (deliberate shadow)
            return base_fn(signal_type, template_idx=pin_template, **kwargs)

    def preprocess(ex):
        out = preprocess_example(ex, tokenizer=tokenizer, prompt_fn=prompt_fn,
                                 max_length=config.training_config.max_seq_length,
                                 return_frozen_input_len=True, return_position_valid=True,
                                 for_prediction=True)
        out["token_position"] = ex["token_position"]
        out["signal_layer"] = ex["signal_layer"]
        out["signal_index"] = ex["signal_index"]
        out["activation"] = ex["activation"]
        out["src_token_position"] = ex["src_token_position"]
        out["signal_src_offset"] = ex["src_offset"]
        return out

    test_dataset = test_raw.map(preprocess, remove_columns=test_raw.column_names)
    # Same max_length as the preprocess above, as in train_introspector_joint.py:1380-1383.
    # It only names the cause in the error message, but it is a required argument: omitting
    # it crashed every cell of the first analysis job (logs/attn_w_analysis.o7605926)
    # before the model had even loaded.
    test_dataset = drop_invalid_positions(test_dataset, "test",
                                          config.training_config.max_seq_length)

    base = AutoModelForCausalLM.from_pretrained(
        config.model_name, device_map="auto",
        dtype=torch.bfloat16 if config.training_config.bf16 else torch.float32,
        **attention_model_kwargs(config.train_signals))
    resize_embeddings_for_tokenizer(base, tokenizer)
    peft_model = PeftModel.from_pretrained(base, str(checkpoint),
                                           autocast_adapter_dtype=False)
    model = IntrospectionRegressionModel(
        peft_model, signal_ranges=signal_ranges,
        intervention_distribution=config.training_config.intervention_signal_distribution,
        intervention_extreme_scale=config.training_config.intervention_extreme_scale,
        attention_site=port_site(signal_type))
    load_regression_head(model, str(checkpoint))
    load_added_token_embeddings(model, str(checkpoint))
    replace_model_with_gated_version(model)
    replace_lora_with_gated_version(model)
    model.eval()

    # The collator must match how the run was TRAINED: with the gate on, the mask stays
    # under `lora_token_mask` and the patched forward gates the adapter; with it off the
    # mask moves to `anchor_region_mask` and LoRA applies everywhere. Using the wrong one
    # silently changes the computation being attributed.
    gate_input = config.training_config.disable_adapter_in_input
    collator_cls = LoRAGatingCollator if gate_input else UngatedCollator
    collator = collator_cls(tokenizer=tokenizer, model=model, padding=True,
                            label_pad_token_id=-100, return_tensors="pt",
                            for_prediction=True)
    return model, tokenizer, test_dataset, collator, gate_input


def latest_checkpoint(config: ExperimentConfig, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    run_dir = Path(config.output_dir) / config.experiment_name
    ckpts = sorted(run_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_ctime)
    if not ckpts:
        raise SystemExit(f"no checkpoint under {run_dir}; pass --checkpoint")
    return ckpts[-1]


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------
@torch.no_grad()
def endogenous_predictions(model, dataset, collator, batch_size):
    """Prediction and target for every example, with no intervention."""
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    preds, acts, lens = [], [], []
    head = model.regression_head
    for batch in dl:
        batch = batch.to(model.device)
        activation = batch.pop("activation")
        # real (unpadded) length per row: prediction batches are LEFT padded
        lens.extend((batch["attention_mask"].sum(1)).tolist())
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    lora_token_mask=batch.get("lora_token_mask"),
                    do_intervention=False, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]
        pred = head(h.to(device=head.weight.device, dtype=head.weight.dtype)).squeeze(-1)
        preds.append(pred.float().cpu())
        acts.append(activation.float().cpu())
    return torch.cat(preds), torch.cat(acts), lens


def modal_length_bucket(lengths):
    """The most common tokenized length, the indices that have it, and their share.

    Only within one exact length do [N, N] triangles share a grid.
    """
    counts = Counter(lengths)
    N, n = counts.most_common(1)[0]
    idx = [i for i, L in enumerate(lengths) if L == N]
    return N, idx, n / max(len(lengths), 1)


# ----------------------------------------------------------------------
# Attribution
# ----------------------------------------------------------------------
def attribute_batch(model, batch, n_layers, method, ig_steps):
    """d(prediction)/d(p) x p for every layer of one batch -> [n_layers, B, H, N, N].

    Both methods are one path integral over alpha in (0, 1]: `gradxact` is the single
    endpoint alpha = 1; `ig` averages `ig_steps` of them. The baseline is zero attention,
    which is where a saturated softmax row differs most from its local gradient -- the
    reason the published figure uses ig.
    """
    alphas = [1.0] if method == "gradxact" else [(k + 1) / ig_steps
                                                 for k in range(ig_steps)]
    total, pred_out = None, None
    head = model.regression_head
    for alpha in alphas:
        taps = {L: AttentionProbGradTap(alpha=alpha) for L in range(n_layers)}
        with installed_ports(model, taps, site="weight"):
            with torch.enable_grad():
                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            lora_token_mask=batch.get("lora_token_mask"),
                            do_intervention=False, output_hidden_states=True)
                h = out.hidden_states[-1][:, -1, :]
                pred = head(h.to(device=head.weight.device, dtype=head.weight.dtype)
                            ).squeeze(-1)
                model.zero_grad(set_to_none=True)
                pred.sum().backward()
        if pred_out is None:
            pred_out = pred.detach().float().cpu()
        for L, tap in taps.items():
            g = tap.leaf.grad
            assert g is not None, (
                f"layer {L} tap received no gradient -- the backward did not reach it")
            # grad x activation at this alpha; `value` is the unscaled p, so averaging
            # over the alpha grid is IG's Riemann sum
            contrib = (g * tap.value).detach().float().cpu()
            if total is None:
                total = torch.zeros((n_layers,) + tuple(contrib.shape),
                                    dtype=torch.float32)
            total[L] += contrib
    total /= len(alphas)
    return total, pred_out


def token_attribution(model, batch, method, ig_steps):
    """grad x embedding at the input, summed over the hidden dimension -> [B, N].

    Its own backward, with NO attention taps installed: the taps detach p, which severs
    the gradient path from the attention probabilities back to the embeddings.
    """
    head = model.regression_head
    embed = model.get_input_embeddings()
    base_emb = embed(batch["input_ids"]).detach()
    alphas = [1.0] if method == "gradxact" else [(k + 1) / ig_steps
                                                 for k in range(ig_steps)]
    total = torch.zeros(base_emb.shape[:2], dtype=torch.float32)
    for alpha in alphas:
        emb = (base_emb * alpha).clone().requires_grad_(True)
        with torch.enable_grad():
            out = model(inputs_embeds=emb, attention_mask=batch["attention_mask"],
                        lora_token_mask=batch.get("lora_token_mask"),
                        do_intervention=False, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :]
            pred = head(h.to(device=head.weight.device, dtype=head.weight.dtype)).squeeze(-1)
            model.zero_grad(set_to_none=True)
            pred.sum().backward()
        total += (emb.grad * base_emb).sum(-1).detach().float().cpu()
    return total / len(alphas)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = ExperimentConfig.model_validate_json(args.config.read_text())
    if not config.use_regressor:
        raise SystemExit("attribution targets the regressor head (use_regressor must be True)")
    ckpt = latest_checkpoint(config, args.checkpoint)
    # plots/<exp_group>/attribution/<run>, next to the group's other figures rather than
    # inside the run directory. output_dir is outputs/runs/<exp_group>, so its last
    # component is the group.
    out = args.out or (Path("plots") / Path(config.output_dir).name / "attribution"
                       / config.experiment_name)
    out.mkdir(parents=True, exist_ok=True)

    pin = None if args.pin_template < 0 else args.pin_template
    print(f"[attr] config={args.config}")
    print(f"[attr] checkpoint={ckpt}")
    print(f"[attr] method={args.attr_method}"
          + (f" steps={args.ig_steps}" if args.attr_method == "ig" else "")
          + f"  pin_template={pin}  -> {out}")
    if pin is None:
        print("[attr] *** WARNING: the prompt template is NOT pinned. A given token index "
              "means a different word in different examples, so the averaged maps below "
              "are not interpretable position by position.")

    model, tokenizer, dataset, collator, gate_input = load_everything(config, ckpt, pin)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"[attr] gate={'on' if gate_input else 'off'}  layers={n_layers} heads={n_heads}")

    # ---- select examples ----
    preds, acts, lens = endogenous_predictions(model, dataset, collator, args.batch_size)
    target_std = float(acts.std())
    err = (preds - acts).abs()
    thresh = args.success_threshold_std * target_std
    ok = (err <= thresh).numpy()
    print(f"[attr] target std={target_std:.6g}  threshold={thresh:.6g} "
          f"({args.success_threshold_std} std)  -> {int(ok.sum())}/{len(ok)} examples pass")

    N, bucket_idx, share = modal_length_bucket(lens)
    print(f"[attr] modal tokenized length N={N} covers {len(bucket_idx)}/{len(lens)} "
          f"examples ({100 * share:.1f}%)")
    if share < 0.2:
        print("[attr] *** WARNING: the modal bucket is small. With --pin-template set "
              "this usually means the source/destination token strings vary a lot in "
              "length; the maps are still valid, just averaged over fewer examples.")

    cand = [i for i in bucket_idx if ok[i]]
    order = sorted(cand, key=lambda i: float(err[i]))
    best = order[:args.n_examples]
    worst_pool = [i for i in bucket_idx if i not in set(best)]
    worst = sorted(worst_pool, key=lambda i: -float(err[i]))[:args.n_examples]
    print(f"[attr] BEST n={len(best)} (of {len(cand)} eligible in the bucket); "
          f"WORST n={len(worst)}")
    if not best:
        raise SystemExit("no example passed both the success threshold and the length "
                         "bucket; loosen --success-threshold-std")

    # ---- attribute ----
    groups = {"best": best} if args.no_contrast_worst else {"best": best, "worst": worst}
    summary = {}
    for name, idx in groups.items():
        if not idx:
            continue
        sub = dataset.select(idx)
        dl = DataLoader(sub, batch_size=args.batch_size, collate_fn=collator, shuffle=False)
        acc = None          # [n_layers, H, N, N] running sum
        acc_sq = None
        tok_acc = None
        n_seen = 0
        for batch in dl:
            batch = batch.to(model.device)
            batch.pop("activation", None)
            attr, _ = attribute_batch(model, batch, n_layers, args.attr_method,
                                      args.ig_steps)          # [L, B, H, N, N]
            tok = token_attribution(model, batch, args.attr_method, args.ig_steps)  # [B,N]
            b = attr.shape[1]
            s = attr.sum(1)                                     # [L, H, N, N]
            acc = s if acc is None else acc + s
            acc_sq = (attr ** 2).sum(1) if acc_sq is None else acc_sq + (attr ** 2).sum(1)
            tok_acc = tok.sum(0) if tok_acc is None else tok_acc + tok.sum(0)
            n_seen += b
        mean = (acc / n_seen).numpy()                           # [L, H, N, N]
        var = (acc_sq / n_seen).numpy() - mean ** 2
        sem = np.sqrt(np.maximum(var, 0) / n_seen)
        tok_mean = (tok_acc / n_seen).numpy()                   # [N]

        np.savez_compressed(out / f"attr_attn_probs_{args.attr_method}_{name}.npz",
                            mean=mean.astype(np.float32), sem=sem.astype(np.float32),
                            token=tok_mean.astype(np.float32), n=n_seen, N=N)
        summary[name] = dict(n=int(n_seen), N=int(N))
        print(f"[attr] {name}: wrote attr_attn_probs_{args.attr_method}_{name}.npz "
              f"(n={n_seen}, N={N})")

    # ---- the target cell, for the star marker in the figures ----
    # The destination is the cached token_position -- the last token of the INPUT TEXT
    # (--train-token-selector -1 selects within the text) -- NOT the readout token at the
    # end of the introspection prompt. This used to be hardcoded as dst_rel=0, which put
    # every star on the readout token attending to the prompt token before it, a cell
    # nothing was trained on (60K P2.31@-1: trained (31, 30) of N=56, starred (55, 54)).
    # Read from the attributed rows instead, and require one cell for all of them: the
    # maps average over these rows, so a star is only meaningful if they agree.
    attributed = dataset.select(sorted({i for idx in groups.values() for i in idx}))
    dst_set = set(attributed["token_position"])
    src_set = set(attributed["src_token_position"])
    assert len(dst_set) == 1 and len(src_set) == 1, (
        f"attributed rows disagree on the target cell: token_position {sorted(dst_set)}, "
        f"src_token_position {sorted(src_set)}")
    # Absolute in the introspection sequence as-is: token_position indexes the
    # BOS-prefixed text, which is also the prefix of that sequence, and every attributed
    # row is in the length-N bucket, so a batch of them has no left padding to add
    # (train_introspector_joint.py adds num_left_pad for mixed-length batches).
    dst_abs, src_abs = dst_set.pop(), src_set.pop()
    row0 = attributed[0]
    assert 0 <= src_abs <= dst_abs < N - 1, (
        f"target cell (dst={dst_abs}, src={src_abs}) must sit in the text, before the "
        f"readout token at {N - 1}")
    assert src_abs - dst_abs == row0["signal_src_offset"], (
        f"src {src_abs} - dst {dst_abs} != src_offset {row0['signal_src_offset']}")
    meta = dict(
        signal_layer=int(row0["signal_layer"]), signal_head=int(row0["signal_index"]),
        src_offset=int(row0["signal_src_offset"]),
        # Right-aligned indices, with 0 = THE LAST TOKEN (the readout), negative going
        # back; absolute index = rel + (N - 1). The absolute pair is stored too so the
        # plotter can check the two conventions against each other.
        dst_abs=int(dst_abs), src_abs=int(src_abs),
        dst_rel=int(dst_abs - (N - 1)), src_rel=int(src_abs - (N - 1)),
        N=int(N), n_layers=int(n_layers), n_heads=int(n_heads),
        method=args.attr_method, ig_steps=args.ig_steps if args.attr_method == "ig" else 1,
        pin_template=pin, gate="on" if gate_input else "off",
        target_std=target_std, success_threshold_std=args.success_threshold_std,
        n_pass=int(ok.sum()), n_total=int(len(ok)), bucket_share=share,
        checkpoint=str(ckpt), groups=summary,
    )
    (out / "attribution_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[attr] target cell: layer {meta['signal_layer']} head {meta['signal_head']} "
          f"dst_rel {meta['dst_rel']} src_rel {meta['src_rel']}")
    print(f"[attr] wrote attribution_meta.json under {out}")
    print(f"[attr] next: python analysis/plot_attention_weight_attribution.py {out}")


if __name__ == "__main__":
    main()
