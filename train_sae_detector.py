#!/usr/bin/env python3
"""
Phase-1 SAE introspection: fixed-position DETECTION of a single sparse feature.

The model reads the introspection prompt and emits a word ("active"/"inactive")
saying whether the SAE feature fired at the target token. This is the generative
analog of a classifier head -- it keeps the answer in the model's own tokens
(natural, and it sets up the semantic evals) -- but sparse features make plain
accuracy meaningless (always-"inactive" scores ~99%). So we:

  * oversample positive (feature-active) rows in TRAIN up to
    detection_balance_pos_frac, so the concept is learnable;
  * score threshold-free on the held-out NATURAL distribution via the
    active-vs-inactive first-token logit gap -> AUROC + Average Precision (AP,
    the imbalance-honest headline) + F1/accuracy at gap=0.

Reuses the LoRA-gating stack from train_introspector.py unchanged. Endogenous
only (no steering). Run on the cluster:

  python train_sae_detector.py --config experiment_configs/<sae_detection>.json
"""
from __future__ import annotations

import os
import glob
import json
import math
import argparse
import operator
from functools import partial
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_from_disk, Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.introspector.config import ExperimentConfig
from src.introspector.dataset_utils import apply_signal_filter, parse_signal_filter
from src.introspector.models.signal_paths import resolve_sae_hook_path
from src.introspector.signals.sae import load_sae_for_spec
from src.introspector.training.collator import LoRAGatingCollator
from src.introspector.training.model import (
    StripEmbeddingSaveCallback,
    replace_model_with_gated_version, replace_lora_with_gated_version, SAEInterventionHook,
)
from src.introspector.prompts import PROMPT_REGISTRY
from train_introspector import preprocess_example, INTROSPECTION_SEPARATOR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--overwrite-cache", action="store_true",
                   help="Re-run even if metrics exist, and start training fresh instead of "
                        "resuming from the latest checkpoint.")
    p.add_argument("--ood-eval", action="store_true",
                   help="After training, re-evaluate detection on the OOD corpus ladder "
                        "(see score_ood_generalization.py). An endogenous score built on "
                        "context should fall off-distribution; a causal read should not.")
    p.add_argument("--ood-corpora", type=str, default=None,
                   help="Comma-separated subset of the OOD ladder (default: all).")
    p.add_argument("--force-drift-eval", action="store_true",
                   help="Run the cache-alignment check (measure_sae_alignment). Same role as "
                        "--force-drift-eval in the neuron trainers: it is the regression test "
                        "for the off-by-one/BOS class of bug. Want best offset +0, corr ~1.0, "
                        "'alignment OK'. Cheap; run it once per dataset in a sweep.")
    return p.parse_args()


def is_active(activation) -> bool:
    return float(activation) > 0.0


def resolve_active_sae_spec(config):
    """The ONE SAE feature this run trains on.

    Under the shared-cache convention a (layer, position) cell is cached once with
    all its features and `--signal-filter S<layer>.<index>` splits it per run, so
    config.train_signals carries the WHOLE cell. Taking train_signals[0] would then
    train on one feature while steering another -- silently evaluating the wrong
    thing. Resolve through the filter instead, and refuse to guess when a
    multi-feature cache is used without one.
    """
    sae_specs = [s for s in config.train_signals if getattr(s, "type", None) == "sae"]
    if not sae_specs:
        raise SystemExit("config.train_signals contains no SAE feature")
    if config.signal_filter:
        want = parse_signal_filter(config.signal_filter)
        match = [s for s in sae_specs
                 if s.layer == want["signal_layer"] and s.index == want["signal_index"]]
        if not match:
            raise SystemExit(
                f"--signal-filter {config.signal_filter!r} names no SAE feature in "
                f"train_signals: {[(s.layer, s.index) for s in sae_specs]}")
        return match[0]
    if len(sae_specs) > 1:
        raise SystemExit(
            f"{len(sae_specs)} SAE features cached but no signal_filter set; ambiguous. "
            f"Pass --signal-filter S<layer>.<index> when generating the config.")
    return sae_specs[0]


def oversample_positives(ds: Dataset, target_pos_frac: float, seed: int) -> Dataset:
    """Duplicate feature-active rows so positives make up ~target_pos_frac of the
    TRAIN split. Negatives are all kept. No-op / warning at the degenerate ends."""
    acts = np.array([float(a) for a in ds["activation"]])
    pos_idx = np.where(acts > 0)[0]
    neg_idx = np.where(acts <= 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        print(f"[balance] WARNING: pos={len(pos_idx)} neg={len(neg_idx)}; "
              f"cannot balance, using as-is.")
        return ds
    if target_pos_frac <= 0 or target_pos_frac >= 1:
        return ds
    # want P/(P+N)=f with N fixed => P = f*N/(1-f); how many total pos rows needed
    desired_pos = int(round(target_pos_frac * len(neg_idx) / (1 - target_pos_frac)))
    reps = max(1, round(desired_pos / len(pos_idx)))
    parts = [ds]
    for _ in range(reps - 1):
        parts.append(ds.select(pos_idx.tolist()))
    out = concatenate_datasets(parts).shuffle(seed=seed)
    new_frac = (len(pos_idx) * reps) / (len(pos_idx) * reps + len(neg_idx))
    print(f"[balance] pos {len(pos_idx)}->x{reps}, neg {len(neg_idx)}; "
          f"train pos_frac {len(pos_idx)/len(ds):.3f} -> {new_frac:.3f} (n={len(out)})")
    return out


def make_preprocess(config, tokenizer):
    tc = config.training_config
    return lambda ex, for_prediction=False: preprocess_example(
        ex, tokenizer=tokenizer, prompt_fn=PROMPT_REGISTRY[tc.prompt_type],
        max_length=tc.max_seq_length, return_frozen_input_len=tc.disable_adapter_in_input,
        for_prediction=for_prediction, detection=True,
        detection_active_word=tc.detection_active_word,
        detection_inactive_word=tc.detection_inactive_word,
    )


def label_token_ids(tokenizer, tc):
    """First-token ids of the active/inactive words AS EMITTED (leading space, so
    " active" != "active"). Shared by the post-hoc eval and the in-training hooks."""
    active_id = tokenizer(" " + tc.detection_active_word.strip(), add_special_tokens=False)["input_ids"][0]
    inactive_id = tokenizer(" " + tc.detection_inactive_word.strip(), add_special_tokens=False)["input_ids"][0]
    if active_id == inactive_id:
        raise ValueError(f"active/inactive share first token id {active_id}; pick more distinct words.")
    return active_id, inactive_id


def detection_metrics_from_scores(scores, labels):
    """scores: active-minus-inactive logit gap; labels: binary (1=active). Returns
    the threshold-free detection metrics (AP is the imbalance-honest headline)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    base_rate = float(labels.mean()) if len(labels) else float("nan")
    res = {"n": int(len(labels)), "base_rate_positive": base_rate}
    if len(labels) == 0 or labels.min() == labels.max():
        res.update({"auroc": float("nan"), "ap": float("nan"),
                    "f1_at_0": float("nan"), "acc_at_0": float("nan"),
                    "note": "single-class eval set"})
        return res
    res["auroc"] = float(roc_auc_score(labels, scores))
    res["ap"] = float(average_precision_score(labels, scores))          # imbalance-honest headline
    preds = (scores > 0).astype(int)
    res["f1_at_0"] = float(f1_score(labels, preds, zero_division=0))
    res["acc_at_0"] = float((preds == labels).mean())
    return res


def preprocess_logits_for_detection(logits, labels, active_id, inactive_id):
    """Trainer hook: reduce eval logits [B,T,V] to just the (active, inactive)
    logits at the row that predicts the first target token -> [B, 2]. Keeps eval
    memory tiny (same reason train_introspector argmaxes), and is all we need for
    AUROC/AP on the dev split during training."""
    is_tgt = labels != -100                                 # [B,T]
    first_j = is_tgt.float().argmax(dim=1)                  # first target-token position
    pred_row = (first_j - 1).clamp(min=0)                   # logits[:, t] predicts token t+1
    b = torch.arange(logits.shape[0], device=logits.device)
    rows = logits[b, pred_row].float()                      # [B, V]
    return torch.stack([rows[:, active_id], rows[:, inactive_id]], dim=-1)  # [B, 2]


def compute_detection_metrics(eval_pred, active_id, inactive_id):
    """Trainer compute_metrics for the dev split during training. preds are the
    [B,2] (active,inactive) logits from preprocess_logits_for_detection; the gold
    label is recovered from the first target token id in `labels`."""
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    gap = preds[:, 0] - preds[:, 1]
    first_j = (labels != -100).argmax(axis=1)
    gold = labels[np.arange(len(labels)), first_j]          # first target token id
    valid = (gold == active_id) | (gold == inactive_id)
    y = (gold == active_id).astype(int)
    return detection_metrics_from_scores(gap[valid], y[valid])


@torch.no_grad()
def auroc_eval(model, ds_pred, activations, collator, tokenizer, tc, batch_size):
    """Threshold-free detection metrics on the natural distribution (post-hoc).
    ds_pred: for_prediction dataset (prompt only, left-padded).
    activations: aligned list of true feature activations (same order, no shuffle)."""
    model.eval()
    active_id, inactive_id = label_token_ids(tokenizer, tc)
    dl = DataLoader(ds_pred, batch_size=batch_size, collate_fn=collator, shuffle=False)
    scores = []
    for batch in dl:
        batch = batch.to(model.device)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    lora_token_mask=batch.get("lora_token_mask"))
        last = out.logits[:, -1, :].float()                 # left-padded -> last col = last real token
        scores.append((last[:, active_id] - last[:, inactive_id]).cpu())
    scores = torch.cat(scores).numpy()
    labels = np.array([1 if is_active(a) else 0 for a in activations])
    return detection_metrics_from_scores(scores, labels), scores, labels


def active_on_value(train_raw):
    """A representative 'on' magnitude to steer to: the median of the feature's
    naturally-active activations. Robust to the sparse zero mass and to outliers."""
    acts = np.array([float(a) for a in train_raw["activation"]])
    active = acts[acts > 0]
    return float(np.median(active)) if len(active) else 1.0


@torch.no_grad()
def interventional_detection_eval(model, spec, test_raw, ds_pred, collator, tokenizer, tc,
                                  on_value, batch_size):
    """Paired counterfactual faithfulness test. For each context we steer the SAE
    feature to `on_value` and to 0 at the SAME target token, and read the report's
    active-vs-inactive logit gap each time. If the detector causally reads the
    feature, gap(on) > gap(off); if it rides context correlations, the two are
    indistinguishable. Reports interventional AUROC/AP (on vs off across all
    contexts) and a paired directional accuracy (fraction with gap_on > gap_off).

    ds_pred: for_prediction (prompt-only, left-padded) dataset, SAME row order as
    test_raw (shuffle=False), so token positions line up by a running pointer.
    """
    model.eval()
    active_id, inactive_id = label_token_ids(tokenizer, tc)
    bundle = load_sae_for_spec(spec, device=str(model.device))
    module_path = resolve_sae_hook_path(model, spec.layer, site=spec.sae_site, path_to_output=False)
    module = operator.attrgetter(module_path)(model)
    token_positions_all = np.asarray(test_raw["token_position"])

    dl = DataLoader(ds_pred, batch_size=batch_size, collate_fn=collator, shuffle=False)
    gaps = {"on": [], "off": []}
    ptr = 0
    for batch in dl:
        batch = batch.to(model.device)
        n = batch["input_ids"].shape[0]
        # token_position is relative to the (unpadded) text, which starts after the
        # left pad; shift by the leading pad count to hit the intended token.
        tok_pos = torch.tensor(token_positions_all[ptr:ptr + n], device=model.device)
        ptr += n
        num_left_pad = (batch["attention_mask"] == 0).sum(dim=1)
        abs_pos = tok_pos + num_left_pad
        for state in ("on", "off"):
            tv = torch.full((n,), on_value if state == "on" else 0.0,
                            device=model.device, dtype=torch.float32)
            hook = SAEInterventionHook(bundle, abs_pos, spec.index, tv)
            handle = module.register_forward_hook(hook)
            try:
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            lora_token_mask=batch.get("lora_token_mask"))
            finally:
                handle.remove()
            last = out.logits[:, -1, :].float()
            gaps[state].append((last[:, active_id] - last[:, inactive_id]).cpu())

    gap_on = torch.cat(gaps["on"]).numpy()
    gap_off = torch.cat(gaps["off"]).numpy()
    scores = np.concatenate([gap_on, gap_off])
    labels = np.concatenate([np.ones_like(gap_on), np.zeros_like(gap_off)])
    res = detection_metrics_from_scores(scores, labels)
    res["paired_directional_acc"] = float((gap_on > gap_off).mean())
    res["on_value"] = float(on_value)
    res["mean_gap_on"] = float(gap_on.mean())
    res["mean_gap_off"] = float(gap_off.mean())
    return res, gap_on, gap_off


# ======================================================================
# Cache alignment (SAE analogue of measure_activation_drift / probe_cache_alignment)
#
# Same regression test the neuron pipeline runs via --force-drift-eval, which is what
# caught the off-by-one: cache_signals tokenizes WITH BOS, so a cached token_position
# only indexes the training sequence correctly if preprocess_example also tokenizes
# with BOS. If it does not, position p points one token to the RIGHT of the token the
# prompt names and the steering hook injects into the neighbour.
#
# Running with adapters DISABLED reproduces the base model that built the cache, so
# the offset whose recomputed feature matches the cached column is the true
# convention. WANT: best offset +0, corr ~1.0, token_match ~1.0 -> "alignment OK".
# ======================================================================
DRIFT_OFFSETS = (-1, 0, 1)
TRAIN_OFFSET = 0        # what the steering hook / trainer assume


class _ResidCaptureHook:
    """Records the residual at pos+offset for several offsets without modifying it
    (returns None, so the layer output passes through untouched)."""

    def __init__(self, token_positions, offsets=DRIFT_OFFSETS):
        self.token_positions = token_positions
        self.offsets = tuple(offsets)
        self.captured = {}

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        seq_len = hidden.shape[1]
        b = torch.arange(hidden.shape[0], device=hidden.device)
        for off in self.offsets:
            pos = (self.token_positions + off).clamp(0, seq_len - 1)
            self.captured[off] = hidden[b, pos].detach().float()
        return None


@torch.no_grad()
def measure_sae_alignment(model, spec, test_raw, ds_pred, collator, tokenizer,
                          batch_size, out_dir, max_examples=512):
    """Recompute the SAE feature at the cached position (adapters off) and compare to
    the cached activation, across offsets. Writes sae_cache_alignment.json."""
    model.eval()
    bundle = load_sae_for_spec(spec, device=str(model.device))
    module_path = resolve_sae_hook_path(model, spec.layer, site=spec.sae_site, path_to_output=False)
    module = operator.attrgetter(module_path)(model)
    token_positions_all = np.asarray(test_raw["token_position"])
    cached_tokens = list(test_raw["token"])
    cached_acts = np.asarray(test_raw["activation"], dtype=float)

    n = min(max_examples, len(ds_pred))
    dl = DataLoader(ds_pred.select(range(n)), batch_size=batch_size, collate_fn=collator, shuffle=False)
    live = {off: [] for off in DRIFT_OFFSETS}
    tok_match, ptr = [], 0
    for batch in dl:
        batch = batch.to(model.device)
        bs = batch["input_ids"].shape[0]
        tok_pos = torch.tensor(token_positions_all[ptr:ptr + bs], device=model.device)
        num_left_pad = (batch["attention_mask"] == 0).sum(dim=1)
        abs_pos = tok_pos + num_left_pad                       # left-padded pred batch
        hook = _ResidCaptureHook(abs_pos)
        handle = module.register_forward_hook(hook)
        try:
            with model.disable_adapter():                      # reproduce the caching model
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                      lora_token_mask=batch.get("lora_token_mask"))
        finally:
            handle.remove()
        for off, resid in hook.captured.items():
            live[off].append(bundle.encode(resid)[:, spec.index].float().cpu())
        # token-string match at offset 0: does the cached `token` sit where we index?
        ids = batch["input_ids"]
        for i in range(bs):
            got = tokenizer.decode(ids[i, abs_pos[i]])
            tok_match.append(got == cached_tokens[ptr + i])
        ptr += bs

    cached = cached_acts[:ptr]
    rows = {}
    for off in DRIFT_OFFSETS:
        v = torch.cat(live[off]).numpy()
        ok = np.std(v) > 1e-9 and np.std(cached) > 1e-9
        rows[off] = float(np.corrcoef(v, cached)[0, 1]) if ok else float("nan")
    best = max(rows, key=lambda o: (-np.inf if np.isnan(rows[o]) else rows[o]))
    match_rate = float(np.mean(tok_match))
    aligned = (best == TRAIN_OFFSET) and (rows[TRAIN_OFFSET] > 0.99) and (match_rate > 0.99)

    print("\n=== SAE CACHE ALIGNMENT (adapters off) ===")
    print(f"  n={ptr}  token_match={match_rate:.4f}")
    for off in DRIFT_OFFSETS:
        print(f"  offset {off:+d}: corr(live, cached) = {rows[off]:+.4f}"
              + ("   <-- train assumes this" if off == TRAIN_OFFSET else ""))
    print(f"  best offset = {best:+d}  ->  {'alignment OK' if aligned else 'MISALIGNED'}")
    if not aligned:
        print("  ^ do NOT trust interventional numbers from this run: the steering hook "
              "is injecting into a different token than the prompt names, or the cached "
              "feature is not reproducible (check BOS/tokenization convention).")
    res = {"n": ptr, "token_match": match_rate, "corr_by_offset": {str(k): v for k, v in rows.items()},
           "best_offset": int(best), "train_offset": TRAIN_OFFSET, "aligned": bool(aligned)}
    with open(Path(out_dir) / "sae_cache_alignment.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(args.config.read_text())
    tc = config.training_config
    assert tc.objective == "detection", "train_sae_detector expects objective='detection'"
    cache_dir = config.dataset_cache_dir
    out_dir = Path(config.output_dir) / config.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "sae_interventional_scores.png").exists() and not args.overwrite_cache:
        print(f"Already ran, skipping {out_dir}")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_added_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": [INTROSPECTION_SEPARATOR]})

    train_raw = load_from_disk(str(cache_dir / "train"))
    dev_raw = load_from_disk(str(cache_dir / "dev"))
    test_raw = load_from_disk(str(cache_dir / "test"))
    # a (layer, position) cell is cached once with ALL its features; keep only this
    # run's rows (no-op when signal_filter is unset)
    train_raw = apply_signal_filter(train_raw, config.signal_filter, "train")
    dev_raw = apply_signal_filter(dev_raw, config.signal_filter, "dev")
    test_raw = apply_signal_filter(test_raw, config.signal_filter, "test")
    train_bal = oversample_positives(train_raw, tc.detection_balance_pos_frac, config.seed)

    pp = make_preprocess(config, tokenizer)
    train_ds = train_bal.map(pp, remove_columns=train_bal.column_names)
    dev_ds = dev_raw.map(pp, remove_columns=dev_raw.column_names)
    # for_prediction datasets (prompt only) for the AUROC eval, natural distribution
    dev_pred = dev_raw.map(lambda ex: pp(ex, for_prediction=True), remove_columns=dev_raw.column_names)
    test_pred = test_raw.map(lambda ex: pp(ex, for_prediction=True), remove_columns=test_raw.column_names)

    print("===== sample train rows =====")
    for i in range(min(3, len(train_ds))):
        print(tokenizer.decode(train_ds[i]["input_ids"]))

    # OOD caches MUST be built before we take the GPU: cache_signals runs as a
    # subprocess and a second CUDA context fails once this process holds the device.
    ood_corpora, ood_prepared = None, None
    if args.ood_eval:
        from score_ood_generalization import prepare_ood_caches, DEFAULT_CORPORA
        ood_corpora = (tuple(c.strip() for c in args.ood_corpora.split(","))
                       if args.ood_corpora else DEFAULT_CORPORA)
        ood_prepared = prepare_ood_caches(config, ood_corpora)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if tc.bf16 else torch.float32, device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))
    peft_config = LoraConfig(
        r=tc.lora_r, lora_alpha=tc.lora_alpha, target_modules=tc.lora_target_modules,
        lora_dropout=tc.lora_dropout, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.resize_token_embeddings(len(tokenizer))

    # gate comes from the CONFIG, not from code, so both arms of a sweep come from one
    # script (see scripts/README.md). UngatedCollator drops lora_token_mask, which makes
    # GatedLinear reduce to stock LoRA over the whole sequence, input text included.
    from train_introspector_joint import UngatedCollator
    replace_model_with_gated_version(model)
    replace_lora_with_gated_version(model)
    gate_input = tc.disable_adapter_in_input
    collator_cls = LoRAGatingCollator if gate_input else UngatedCollator
    print(f"[sae-detect] LoRA input gate: "
          f"{'ON -- adapter gated to the introspection prompt' if gate_input else 'OFF -- adapter active over the input text too'} "
          f"(disable_adapter_in_input={gate_input}); lora_dropout={tc.lora_dropout}")
    collator = collator_cls(tokenizer=tokenizer, model=model, padding=True,
                            label_pad_token_id=-100, return_tensors="pt")
    pred_collator = collator_cls(tokenizer=tokenizer, model=model, padding=True,
                                 label_pad_token_id=-100, return_tensors="pt",
                                 for_prediction=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir), run_name=config.experiment_name,
        learning_rate=tc.learning_rate,
        per_device_train_batch_size=tc.train_batch_size,
        per_device_eval_batch_size=tc.eval_batch_size,
        num_train_epochs=tc.num_epochs,
        logging_steps=math.ceil(100 / tc.train_batch_size),
        eval_strategy="steps", eval_steps=math.ceil(400 // tc.train_batch_size),
        save_strategy="epoch", bf16=tc.bf16, remove_unused_columns=False, report_to="wandb",
    )
    # dev AUROC/AP/F1 logged every eval step (wandb + history), not just eval_loss.
    # preprocess_logits_for_metrics shrinks eval logits to [B,2] so this is cheap.
    active_id, inactive_id = label_token_ids(tokenizer, tc)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds,
        eval_dataset=dev_ds, data_collator=collator,
        compute_metrics=partial(compute_detection_metrics, active_id=active_id, inactive_id=inactive_id),
        preprocess_logits_for_metrics=partial(
            preprocess_logits_for_detection, active_id=active_id, inactive_id=inactive_id),
        # drop the frozen resized embeddings from each checkpoint (~1.0 GB, 98% of it)
        callbacks=[StripEmbeddingSaveCallback(n_added_tokens)],
    )
    ckpts = glob.glob(f"{out_dir}/checkpoint-*")
    if ckpts and not args.overwrite_cache:
        trainer.train(resume_from_checkpoint=max(ckpts, key=os.path.getctime))
    else:
        trainer.train()

    # ---- threshold-free detection eval on the natural distribution ----
    dev_metrics, dev_scores, dev_labels = auroc_eval(
        model, dev_pred, dev_raw["activation"], pred_collator, tokenizer, tc, tc.eval_batch_size)
    test_metrics, test_scores, test_labels = auroc_eval(
        model, test_pred, test_raw["activation"], pred_collator, tokenizer, tc, tc.eval_batch_size)
    metrics = {"dev": dev_metrics, "test": test_metrics}

    # ---- interventional (causal) faithfulness: steer the feature on vs off on the
    #      SAME contexts and see whether the report follows. Only for SAE specs. ----
    spec = resolve_active_sae_spec(config)   # honours signal_filter (shared cache)

    # regression test for the off-by-one/BOS class of bug -- run BEFORE trusting any
    # interventional number, since a misaligned position invalidates all of them
    if args.force_drift_eval:
        measure_sae_alignment(model, spec, test_raw, test_pred, pred_collator,
                              tokenizer, tc.eval_batch_size, out_dir)

    if spec is not None:
        on_value = active_on_value(train_raw)
        interv_metrics, gap_on, gap_off = interventional_detection_eval(
            model, spec, test_raw, test_pred, pred_collator, tokenizer, tc, on_value, tc.eval_batch_size)
        metrics["test_interventional"] = interv_metrics
        # paired on/off gap distribution: separation == causal readability
        plt.figure(figsize=(6, 4))
        plt.hist(gap_off, bins=40, alpha=0.5, label=f"steered OFF (n={len(gap_off)})", density=True)
        plt.hist(gap_on, bins=40, alpha=0.5, label=f"steered ON (n={len(gap_on)})", density=True)
        plt.axvline(0, color="k", lw=1)
        plt.xlabel("active - inactive logit gap"); plt.ylabel("density")
        plt.title(f"interventional AUROC={interv_metrics.get('auroc', float('nan')):.3f} "
                  f"paired_dir_acc={interv_metrics.get('paired_directional_acc', float('nan')):.3f}")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "sae_interventional_scores.png", dpi=150)
        plt.close()

        if args.ood_eval:
            from score_ood_generalization import evaluate_ood_detection
            evaluate_ood_detection(
                model, config, tokenizer, pred_collator,
                preprocess_fn=lambda ex: pp(ex, for_prediction=True), spec=spec,
                on_value=on_value, output_dir=out_dir, corpora=ood_corpora,
                prepared=ood_prepared, batch_size=tc.eval_batch_size)

    print(json.dumps(metrics, indent=2))
    with open(out_dir / "sae_detection_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # score distribution by true label (sanity: separation between active/inactive)
    plt.figure(figsize=(6, 4))
    for lab, name in [(1, "active"), (0, "inactive")]:
        s = test_scores[test_labels == lab]
        if len(s):
            plt.hist(s, bins=40, alpha=0.5, label=f"{name} (n={len(s)})", density=True)
    plt.axvline(0, color="k", lw=1)
    plt.xlabel("active - inactive logit gap"); plt.ylabel("density")
    plt.title(f"test AUROC={test_metrics.get('auroc', float('nan')):.3f} "
              f"AP={test_metrics.get('ap', float('nan')):.3f}")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "sae_detection_scores.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
