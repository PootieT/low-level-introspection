#!/usr/bin/env python3
"""
Joint endogenous + interventional introspection training.

Per batch we run TWO forwards on the same examples and step on a weighted sum:

    L = w * L_endo + (1 - w) * L_interv

  - L_endo:   no injection, target = the true activation      (natural-scale target)
  - L_interv: inject the signal, target = the injected value  (large learnable scale)

Both losses are VARIANCE-NORMALIZED (MSE / var(target) ~= 1 - R^2) so their
magnitudes are comparable despite the ~10x scale difference, and `w` genuinely
controls the balance rather than the accident of scale.

Rationale (see design notes):
  * The faithful absolute-value reader is the UNIQUE joint optimum: a bypass head
    fails L_interv (context is uncorrelated with the injected value, so it adds
    noise), a perturbation-only head fails L_endo (no perturbation endogenously).
    So L_interv is the anti-bypass regularizer -- keep `w` (endo weight) < 1 and
    CONSTANT (do NOT anneal the mix toward endogenous; that removes the pressure).
  * Interventional injection must stay at a LEARNABLE scale (>=10x, the natural
    scale is at the bf16 floor), configured via the usual distribution/scale; the
    endogenous term supplies the natural-scale target.
  * Two forwards per batch => both gradients enter every step (no tug-of-war).

Sweep `--joint-loss-weight` across runs and read the (endogenous, interventional)
FRONTIER: a weight with BOTH high => a faithful reader exists; a pure trade-off
=> it is not representable in this single-forward regime.
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
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.introspector.config import ExperimentConfig
from src.introspector.prompts import PROMPT_REGISTRY
from src.introspector.training.collator import LoRAGatingCollator
from src.introspector.training.metrics import compute_metrics
from src.introspector.training.model import (
    IntrospectionRegressionModel,
    RegressionHeadSaveCallback,
    InterventionAnnealCallback,
    load_regression_head,
    replace_model_with_gated_version,
    replace_lora_with_gated_version,
)
from train_introspector import preprocess_example, INTROSPECTION_SEPARATOR
from train_introspector_with_intervention import (
    InterventionTrainer,
    evaluate_interventional_test,
    get_argmax_from_logits,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--joint-loss-weight", type=float, default=0.5,
                   help="Weight w on the ENDOGENOUS term; interventional gets (1-w). "
                        "Keep constant; sweep across runs to map the frontier.")
    return p.parse_args()


def _var_norm_mse(pred, target):
    """Variance-normalized MSE = MSE / var(target) (~ 1 - R^2, scale-invariant)."""
    pred = pred.float()
    target = target.float()
    var = torch.clamp(target.var(), min=1e-8)
    return nn.functional.mse_loss(pred, target) / var


class JointInterventionTrainer(InterventionTrainer):
    """Two-forward joint loss: w*endogenous + (1-w)*interventional (both var-normed)."""

    def __init__(self, *args, joint_loss_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_loss_weight = joint_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = inputs.copy()

        # --- endogenous forward: no injection, target = true activation ---
        endo_inputs = inputs.copy()
        endo_inputs["labels"] = inputs["activation"]
        endo_inputs["do_intervention"] = False
        endo_out = model(**endo_inputs)
        L_endo = _var_norm_mse(endo_out.logits, inputs["activation"])

        # --- interventional forward: inject, target = injected value ---
        interv_inputs = inputs.copy()
        interv_inputs["labels"] = inputs["activation"]   # forward overwrites with injected value
        interv_inputs["do_intervention"] = True
        interv_out = model(**interv_inputs)
        L_interv = _var_norm_mse(interv_out.logits, interv_out.intervention_values)

        w = self.joint_loss_weight
        loss = w * L_endo + (1.0 - w) * L_interv
        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            self.log({"L_endo": float(L_endo.detach()), "L_interv": float(L_interv.detach())})
        return (loss, interv_out) if return_outputs else loss


@torch.no_grad()
def eval_endogenous_test(model, dataset, collator, config, output_dir, suffix="_endogenous"):
    """corr(head(no-injection), true activation) on the (for_prediction) test set."""
    dl = DataLoader(dataset, batch_size=config.training_config.eval_batch_size,
                    collate_fn=collator, shuffle=False)
    head = model.regression_head
    preds, acts = [], []
    for batch in dl:
        batch = batch.to(model.device)
        activation = batch["activation"]
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    lora_token_mask=batch.get("lora_token_mask"),
                    do_intervention=False, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]
        pred = head(h.to(device=head.weight.device, dtype=head.weight.dtype)).squeeze(-1)
        preds.append(pred.float().cpu()); acts.append(activation.float().cpu())
    p, a = torch.cat(preds).numpy(), torch.cat(acts).numpy()
    corr = float(pearsonr(p, a)[0]) if (np.std(p) > 1e-12 and np.std(a) > 1e-12) else float("nan")
    metrics = {"corr": corr}
    with open(Path(output_dir) / f"endogenous_test_metrics{suffix}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[joint] endogenous test corr = {corr:.4f}")
    return corr


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(args.config.read_text())
    if not config.use_regressor:
        raise SystemExit("Joint training targets the regressor head (use_regressor must be True).")

    cache_dir = Path(config.cache_dir) / config.experiment_name
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "interventional_test_metrics_gaussian.json").exists() and not args.overwrite_cache:
        print(f"Experiments ran already, skipping.. {output_dir}")
        return

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [INTROSPECTION_SEPARATOR]})

    # ---- data ----
    train_raw = load_from_disk(str(cache_dir / "train"))
    dev_raw = load_from_disk(str(cache_dir / "dev"))
    test_raw = load_from_disk(str(cache_dir / "test"))

    df = train_raw.to_pandas()
    signal_ranges = {}
    for key, vals in df.groupby(["signal_layer", "signal_index"])["activation"]:
        signal_ranges[key] = {"min": float(vals.min()), "max": float(vals.max()),
                              "mean": float(vals.mean()), "std": float(vals.std())}

    def preprocess(ex):
        out = preprocess_example(ex, tokenizer=tokenizer,
                                 prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
                                 max_length=config.training_config.max_seq_length,
                                 return_frozen_input_len=True, for_prediction=True)
        out["token_position"] = ex["token_position"]
        out["signal_layer"] = ex["signal_layer"]
        out["signal_index"] = ex["signal_index"]
        out["activation"] = ex["activation"]
        return out

    train_dataset = train_raw.map(preprocess, remove_columns=train_raw.column_names)
    dev_dataset = dev_raw.map(preprocess, remove_columns=dev_raw.column_names)
    test_dataset_pred = test_raw.map(preprocess, remove_columns=test_raw.column_names)

    # ---- model (regressor + gated LoRA) ----
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, device_map="auto",
        torch_dtype=torch.bfloat16 if config.training_config.bf16 else torch.float32)
    model.resize_token_embeddings(len(tokenizer))
    peft_config = LoraConfig(
        r=config.training_config.lora_r, lora_alpha=config.training_config.lora_alpha,
        target_modules=config.training_config.lora_target_modules,
        lora_dropout=config.training_config.lora_dropout, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    model.resize_token_embeddings(len(tokenizer))
    model = IntrospectionRegressionModel(
        model, signal_ranges=signal_ranges,
        intervention_distribution=config.training_config.intervention_signal_distribution,
        intervention_extreme_scale=config.training_config.intervention_extreme_scale)
    replace_model_with_gated_version(model)
    replace_lora_with_gated_version(model)

    pred_collator = LoRAGatingCollator(tokenizer=tokenizer, model=model, padding=True,
                                       label_pad_token_id=-100, return_tensors="pt", for_prediction=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir), run_name=config.experiment_name,
        learning_rate=config.training_config.learning_rate,
        per_device_train_batch_size=config.training_config.train_batch_size,
        per_device_eval_batch_size=config.training_config.eval_batch_size,
        num_train_epochs=config.training_config.num_epochs,
        logging_steps=math.ceil(100 / config.training_config.train_batch_size),
        eval_strategy="steps", eval_steps=math.ceil(400 // config.training_config.train_batch_size),
        save_strategy="epoch", bf16=config.training_config.bf16,
        remove_unused_columns=False, report_to="wandb")

    callbacks = [RegressionHeadSaveCallback(model)]
    if config.training_config.intervention_anneal:
        # NOTE: this anneals the injection SCALE (magnitude), NOT the mix weight.
        callbacks.append(InterventionAnnealCallback(
            model,
            scale_start=config.training_config.intervention_scale_start,
            scale_end=config.training_config.intervention_scale_end,
            schedule=config.training_config.intervention_anneal_schedule,
            start_frac=config.training_config.intervention_anneal_start_frac,
            end_frac=config.training_config.intervention_anneal_end_frac,
            log_every=math.ceil(400 / config.training_config.train_batch_size)))

    trainer = JointInterventionTrainer(
        model=model, tokenizer=tokenizer, signal_ranges=signal_ranges, args=training_args,
        train_dataset=train_dataset, eval_dataset=dev_dataset, data_collator=pred_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        preprocess_logits_for_metrics=None, experiment_config=config,
        do_intervention=True, callbacks=callbacks, joint_loss_weight=args.joint_loss_weight)
    print(f"[joint] joint_loss_weight (endo)={args.joint_loss_weight}  "
          f"interv_distribution={config.training_config.intervention_signal_distribution} "
          f"scale={model.intervention_scale}")

    ckpt_dirs = glob.glob(f"{output_dir}/checkpoint-*")
    if ckpt_dirs and not args.overwrite_cache:
        latest = max(ckpt_dirs, key=os.path.getctime)
        print(f"Found checkpoint, resuming/loading from {latest}")
        train_results = trainer.train(resume_from_checkpoint=latest)
        load_regression_head(model, latest)
    else:
        train_results = trainer.train()
        trainer.save_metrics("train", train_results.metrics)

    # ---- frontier metrics: interventional@gaussian(scale=1) AND endogenous ----
    evaluate_interventional_test(model, eval_dataset=test_dataset_pred, collator=pred_collator,
                                 config=config, output_dir=output_dir,
                                 distribution="gaussian", suffix="_gaussian")
    eval_endogenous_test(model, test_dataset_pred, pred_collator, config, output_dir)


if __name__ == "__main__":
    main()
