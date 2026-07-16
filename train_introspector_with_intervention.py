#!/usr/bin/env python3

"""
Train an introspector model WITH causal intervention training.

Key idea:
-----------
During forward pass, we intervene on the target neuron activation
and replace it with a random value sampled from the neuron's empirical range.

The model is then trained to predict THIS INTERVENED VALUE.

This prevents causal bypass where the model predicts activations
from contextual correlations instead of actually reading the neuron.

Architecture:
--------------
- frozen base model
- LoRA adapters
- token-wise LoRA gating
- native torch forward hooks for intervention
- normal teacher-forced LM training

"""

from __future__ import annotations

import os
import pdb
import glob
import argparse
import json
import math
import random
import operator
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.introspector.utils import float_to_str_target
from src.introspector.config import ExperimentConfig
from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.prompts import PROMPT_REGISTRY
from src.introspector.training.collator import LoRAGatingCollator
from src.introspector.training.metrics import (
    compute_metrics,
    compute_metrics_from_numericals,
    get_numericals,
)
from src.introspector.training.model import (
    replace_model_with_gated_version,
    replace_lora_with_gated_version,
    IntrospectionRegressionModel,
    RegressionHeadSaveCallback,
    load_regression_head,
)
from train_introspector import (
    preprocess_example,
    INTROSPECTION_SEPARATOR,
    evaluate_test_split
)

# ============================================================
# Utilities
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="Whether to overwrite cached experiment files if there exist one")
    parser.add_argument("--no-intervention", action="store_true",
                        help="Disable causal intervention. For the regressor this trains/evaluates against "
                             "the ORIGINAL activation instead of a random intervened value, which is a "
                             "sanity check that the regression head can read a scalar out of the hidden "
                             "states at all.")
    parser.add_argument("--refit-head", action="store_true",
                        help="Post-hoc diagnostic: freeze the LoRA/base model and solve for the OPTIMAL "
                             "linear regression head (closed-form ridge/OLS) that predicts the true "
                             "activation from the frozen representation. Answers 'what is the best a "
                             "linear head could do on the normal activation range given this model'.")
    parser.add_argument("--refit-lambda", type=float, default=1.0,
                        help="Ridge regularization strength for --refit-head.")


    return parser.parse_args()


def get_argmax_from_logits(logits, labels):
    return torch.argmax(logits, dim=-1)


# ============================================================
# Post-hoc optimal regression head (closed-form ridge / OLS)
# ============================================================

@torch.no_grad()
def _extract_readout_features(model, dataset, collator, config, readout_pos=-1):
    """
    Run the frozen model WITHOUT intervention and collect, per example:
      - the last-layer hidden state at the readout position (the vector the
        regression head reads), and
      - the true activation target.
    Returns (X [N, H], y [N]) as float32 CPU tensors.
    """
    dl = DataLoader(
        dataset,
        batch_size=config.training_config.eval_batch_size,
        collate_fn=collator,
        shuffle=False,
    )
    feats, targets = [], []
    model.eval()
    for batch in dl:
        batch = batch.to(model.device)
        activation = batch.pop("activation")
        outputs = model(**batch, do_intervention=False, output_hidden_states=True)
        h = outputs.hidden_states[-1][:, readout_pos, :].float().cpu()
        feats.append(h)
        targets.append(activation.float().cpu())
    return torch.cat(feats), torch.cat(targets)


def refit_regression_head(model, fit_dataset, eval_dataset, collator, config,
                          output_dir, ridge_lambda=1.0):
    """
    Freeze everything and solve for the optimal linear regression head via ridge
    regression (closed form) that maps the frozen representation -> true
    activation. Loads the solution into model.regression_head and reports how
    well this *best-case* linear head does on the normal activation range.

    NB: this is a linear probe (correlational). High correlation here means the
    representation linearly encodes the activation / the head was mis-aimed, NOT
    that the model causally reads the neuron.
    """
    Xtr, ytr = _extract_readout_features(model, fit_dataset, collator, config)
    Xte, yte = _extract_readout_features(model, eval_dataset, collator, config)

    # augment with a bias column and solve (X'X + lambda I) w = X'y
    n, hdim = Xtr.shape
    Xtr_b = torch.cat([Xtr, torch.ones(n, 1)], dim=1)
    A = Xtr_b.T @ Xtr_b + ridge_lambda * torch.eye(hdim + 1)
    w = torch.linalg.solve(A, Xtr_b.T @ ytr)  # [H+1]
    weight, bias = w[:-1], w[-1]

    # load the optimal head back into the model (keep head dtype/device)
    head = model.regression_head
    with torch.no_grad():
        head.weight.copy_(weight.view(1, -1).to(head.weight.dtype).to(head.weight.device))
        head.bias.copy_(bias.view(1).to(head.bias.dtype).to(head.bias.device))

    # evaluate the fitted head
    pred_tr = (Xtr_b @ w)
    Xte_b = torch.cat([Xte, torch.ones(Xte.shape[0], 1)], dim=1)
    pred_te = (Xte_b @ w)
    train_metrics = compute_metrics_from_numericals(pred_tr.numpy(), ytr.numpy())
    test_metrics = compute_metrics_from_numericals(pred_te.numpy(), yte.numpy())
    print(f"[refit_regression_head] ridge_lambda={ridge_lambda}")
    print(f"  train-fit  corr={train_metrics['corr']:.4f}  l2={train_metrics['l2_dist']:.4f}")
    print(f"  test       corr={ test_metrics['corr']:.4f}  l2={ test_metrics['l2_dist']:.4f}")

    out = {"ridge_lambda": ridge_lambda, "train_fit": train_metrics, "test": test_metrics}
    with open(str(Path(output_dir) / "refit_head_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    df = pd.DataFrame({"predictions": pred_te.numpy(), "activation": yte.numpy()})
    df.to_csv(str(Path(output_dir) / "refit_head_predictions.csv"), index=False)
    sns.scatterplot(data=df,x="activation", y="predictions")
    plt.savefig(str(Path(output_dir) / "refit_head_predictions_scatter.png"))
    return test_metrics



# ============================================================
# Custom Trainer
# ============================================================

class InterventionTrainer(Trainer):

    def __init__(
        self,
        *args,
        signal_ranges,
        tokenizer: AutoTokenizer,
        experiment_config: ExperimentConfig,
        do_intervention: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.signal_ranges = signal_ranges
        self.tokenizer = tokenizer
        self.experiment_config = experiment_config
        self.do_intervention = do_intervention

    def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            **kwargs,
    ):
        # clone to avoid HF mutation weirdness
        inputs = inputs.copy()
        # ====================================================
        # REGRESSION MODE
        # ====================================================
        if self.experiment_config.use_regressor:
            # labels should be original activations. When intervention is on,
            # model.forward overwrites them with the sampled intervention value
            # internally; when off, they stay as the true activation target.
            inputs["labels"] = inputs["activation"]

        # ====================================================
        # TEXT GENERATION MODE
        # ====================================================
        else:
            # standard LM labels
            pass

        if not self.do_intervention:
            inputs["do_intervention"] = False

        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        # for the text-generation model there is no regression head /
        # intervention_values, so use the default HF prediction path
        if not self.experiment_config.use_regressor:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

        # make evaluation/predict, use EXACT SAME LOGIC as compute_loss
        inputs = inputs.copy()
        inputs["labels"] = inputs["activation"]
        if not self.do_intervention:
            inputs["do_intervention"] = False

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs.loss.detach()
        preds = outputs.logits.detach().float()
        if self.do_intervention:
            # IMPORTANT: compare against the intervened values, NOT original activations
            labels = outputs.intervention_values.detach().float()
        else:
            # sanity-check mode: target is the true, un-intervened activation
            labels = inputs["activation"].detach().float()
        return loss, preds, labels

# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    config = ExperimentConfig.model_validate_json(args.config.read_text())

    cache_dir = Path(config.cache_dir) / config.experiment_name
    train_path = cache_dir / "train"
    dev_path = cache_dir / "dev"
    test_path = cache_dir / "test"
    output_dir = Path(config.output_dir) / f"{config.experiment_name}"

    if (output_dir / "refit_head_predictions_scatter.png").exists() and not args.overwrite_cache:
        print(f"Experiments ran already, skipping.. {output_dir}")
        return

    # update tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {"additional_special_tokens": [INTROSPECTION_SEPARATOR]}
    tokenizer.add_special_tokens(special_tokens)

    # ========================================================
    # Load datasets
    # ========================================================

    train_dataset_raw = load_from_disk(str(train_path))
    dev_dataset_raw = load_from_disk(str(dev_path))
    test_dataset_raw = load_from_disk(str(test_path))

    # ========================================================
    # Compute signal ranges
    # ========================================================
    df = train_dataset_raw.to_pandas() # TODO might be computationally expensive, could load cache stats from data
    grouped = df.groupby(["signal_layer", "signal_index"])["activation"]
    signal_ranges = {}
    for key, vals in grouped:
        signal_ranges[key] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }

    # ========================================================
    # Preprocess datasets
    # ========================================================
    def preprocess(ex, for_prediction=False, keep_signal_specs=True):
        out = preprocess_example(
            ex,
            tokenizer=tokenizer,
            prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
            max_length=config.training_config.max_seq_length,
            return_frozen_input_len=True,
            for_prediction=for_prediction,
        )
        if keep_signal_specs:
            out["token_position"] = ex["token_position"]
            out["signal_layer"] = ex["signal_layer"]
            out["signal_index"] = ex["signal_index"]
            if config.use_regressor:
                out["activation"] = ex["activation"]
        return out
    
    training_preprocess_func = partial(preprocess, for_prediction=True) if config.use_regressor else preprocess
    train_dataset = train_dataset_raw.map(
        training_preprocess_func,
        remove_columns=train_dataset_raw.column_names,
    )
    dev_dataset = dev_dataset_raw.map(
        training_preprocess_func,
        remove_columns=dev_dataset_raw.column_names,
    )
    test_dataset = test_dataset_raw.map(
        partial(preprocess, keep_signal_specs=True),
        remove_columns=test_dataset_raw.column_names,
    )
    test_dataset_pred = test_dataset_raw.map(
        partial(preprocess, for_prediction=True, keep_signal_specs=True),
        remove_columns=test_dataset_raw.column_names,
    )
    
    # Load model add peft
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.training_config.bf16 else torch.float32,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=config.training_config.lora_r,
        lora_alpha=config.training_config.lora_alpha,
        target_modules=config.training_config.lora_target_modules,
        lora_dropout=config.training_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.resize_token_embeddings(len(tokenizer))

    if config.use_regressor:
        model = IntrospectionRegressionModel(
            model,
            signal_ranges=signal_ranges,
            intervention_distribution=config.training_config.intervention_signal_distribution,
            intervention_extreme_scale=config.training_config.intervention_extreme_scale,
        )
        # model.__class__ = IntrospectionRegressionModel
        # model.regression_head = nn.Linear(model.config.hidden_size,1,bias=False,)

    # Token-wise LoRA gating
    replace_model_with_gated_version(model)
    replace_lora_with_gated_version(model)


    collator = LoRAGatingCollator(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    pred_collator = LoRAGatingCollator(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
        for_prediction=True
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=config.experiment_name,
        learning_rate=config.training_config.learning_rate,
        per_device_train_batch_size=config.training_config.train_batch_size,
        per_device_eval_batch_size=config.training_config.eval_batch_size,
        num_train_epochs=config.training_config.num_epochs,
        logging_steps=math.ceil(100/config.training_config.train_batch_size),
        eval_strategy="steps",
        eval_steps=math.ceil(400//config.training_config.train_batch_size),
        save_strategy="epoch",
        bf16=config.training_config.bf16,
        remove_unused_columns=False,
        report_to="wandb",
    )

    trainer = InterventionTrainer(
        model=model,
        tokenizer=tokenizer,
        signal_ranges=signal_ranges,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=pred_collator if config.use_regressor else collator,
        compute_metrics=partial(compute_metrics,tokenizer=tokenizer),
        preprocess_logits_for_metrics=None if config.use_regressor else get_argmax_from_logits,
        experiment_config=config,
        do_intervention=not args.no_intervention,
        # persist the custom regression head next to every checkpoint (PEFT won't)
        callbacks=[RegressionHeadSaveCallback(model)] if config.use_regressor else None,
    )

    ckpt_dirs = glob.glob(f"{output_dir}/checkpoint-*")
    if ckpt_dirs and not args.overwrite_cache:
        print(f"Found Trained model, skipping training and loading directly ..")
        latest_checkpoint_path = max(ckpt_dirs, key=os.path.getctime)
        # resume loads the adapter/embeddings; the regression head is NOT in the
        # PEFT checkpoint, so restore it explicitly afterwards
        train_results = trainer.train(resume_from_checkpoint=latest_checkpoint_path)
        if config.use_regressor:
            load_regression_head(model, latest_checkpoint_path)
    else:
        train_results = trainer.train()
        trainer.save_metrics( "train", train_results.metrics)
    # seed the intervention sampler so eval draws are reproducible across reloads
    if config.use_regressor:
        model.seed_intervention_rng(config.seed)
    dev_results = trainer.evaluate()
    trainer.save_metrics("dev", dev_results,)

    # Eval with default trainer setup (if train with intervention then test with intervention)
    if config.use_regressor:
        model.seed_intervention_rng(config.seed)
    evaluate_test_split(
        config,
        model,
        pred_collator,
        test_dataset,
        test_dataset_pred,
        test_dataset_raw,
        tokenizer,
        train_dataset_raw,
        trainer
    )

    # alternative eval setting (flip the intervention flag)
    trainer.do_intervention = args.no_intervention
    if config.use_regressor:
        model.seed_intervention_rng(config.seed)
    evaluate_test_split(
        config,
        model,
        pred_collator,
        test_dataset,
        test_dataset_pred,
        test_dataset_raw,
        tokenizer,
        train_dataset_raw,
        trainer,
        suffix="_alt_intervention"
    )

    # Post-hoc diagnostic: solve for the optimal linear head on the normal
    # activation range given the frozen representation (mutates the head, so run
    # it LAST, after the standard evals).
    if args.refit_head and config.use_regressor:
        refit_regression_head(
            model,
            fit_dataset=train_dataset,       # for_prediction=True, has "activation"
            eval_dataset=test_dataset_pred,  # for_prediction=True, has "activation"
            collator=pred_collator,
            config=config,
            output_dir=output_dir,
            ridge_lambda=args.refit_lambda,
        )

if __name__ == "__main__":
    main()

