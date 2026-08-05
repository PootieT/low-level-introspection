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
from src.introspector.training.collator import LoRAGatingCollator
from src.introspector.training.model import (
    replace_model_with_gated_version, replace_lora_with_gated_version,
)
from src.introspector.prompts import PROMPT_REGISTRY
from train_introspector import preprocess_example, INTROSPECTION_SEPARATOR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def is_active(activation) -> bool:
    return float(activation) > 0.0


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


@torch.no_grad()
def auroc_eval(model, ds_pred, activations, collator, tokenizer, tc, batch_size):
    """Threshold-free detection metrics on the natural distribution.
    ds_pred: for_prediction dataset (prompt only, left-padded).
    activations: aligned list of true feature activations (same order, no shuffle)."""
    model.eval()
    active_id = tokenizer(tc.detection_active_word, add_special_tokens=False)["input_ids"][0]
    inactive_id = tokenizer(tc.detection_inactive_word, add_special_tokens=False)["input_ids"][0]
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

    base_rate = float(labels.mean())
    res = {"n": int(len(labels)), "base_rate_positive": base_rate}
    if labels.min() == labels.max():
        res.update({"auroc": float("nan"), "ap": float("nan"), "note": "single-class eval set"})
        return res, scores, labels
    res["auroc"] = float(roc_auc_score(labels, scores))
    res["ap"] = float(average_precision_score(labels, scores))          # imbalance-honest headline
    preds = (scores > 0).astype(int)
    res["f1_at_0"] = float(f1_score(labels, preds, zero_division=0))
    res["acc_at_0"] = float((preds == labels).mean())
    return res, scores, labels


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(args.config.read_text())
    tc = config.training_config
    assert tc.objective == "detection", "train_sae_detector expects objective='detection'"
    cache_dir = config.dataset_cache_dir
    out_dir = Path(config.output_dir) / config.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "sae_detection_metrics.json").exists() and not args.overwrite:
        print(f"Already ran, skipping {out_dir}")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [INTROSPECTION_SEPARATOR]})

    train_raw = load_from_disk(str(cache_dir / "train"))
    dev_raw = load_from_disk(str(cache_dir / "dev"))
    test_raw = load_from_disk(str(cache_dir / "test"))
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

    assert tc.disable_adapter_in_input, "detection run assumes the LoRA input gate is on"
    replace_model_with_gated_version(model)
    replace_lora_with_gated_version(model)
    collator = LoRAGatingCollator(tokenizer=tokenizer, model=model, padding=True,
                                  label_pad_token_id=-100, return_tensors="pt")
    pred_collator = LoRAGatingCollator(tokenizer=tokenizer, model=model, padding=True,
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
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                      eval_dataset=dev_ds, data_collator=collator)
    ckpts = glob.glob(f"{out_dir}/checkpoint-*")
    if ckpts and not args.overwrite:
        trainer.train(resume_from_checkpoint=max(ckpts, key=os.path.getctime))
    else:
        trainer.train()

    # ---- threshold-free detection eval on the natural distribution ----
    dev_metrics, dev_scores, dev_labels = auroc_eval(
        model, dev_pred, dev_raw["activation"], pred_collator, tokenizer, tc, tc.eval_batch_size)
    test_metrics, test_scores, test_labels = auroc_eval(
        model, test_pred, test_raw["activation"], pred_collator, tokenizer, tc, tc.eval_batch_size)
    metrics = {"dev": dev_metrics, "test": test_metrics}
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
