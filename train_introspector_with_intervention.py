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

import pdb
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

from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.introspector.config import ExperimentConfig
from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.prompts import PROMPT_REGISTRY
from src.introspector.training.collator import LoRAGatingCollator
from src.introspector.training.metrics import (
    compute_metrics,
    get_numericals,
)
from src.introspector.training.model import (
    replace_model_with_gated_version,
    replace_lora_with_gated_version,
)
from train_introspector import (
    preprocess_example,
    INTROSPECTION_SEPARATOR,
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

    return parser.parse_args()


def get_argmax_from_logits(logits, labels):
    return torch.argmax(logits, dim=-1)


# ============================================================
# Hook helpers
# ============================================================

# def resolve_mlp_module(model, layer_idx):
#     """
#     Adjust this for your architecture.
#
#     For Llama:
#         model.model.layers[i].mlp.down_proj
#
#     We intervene AFTER the MLP output projection.
#     """
#
#     return model.base_model.model.model.layers[layer_idx].mlp.down_proj


class InterventionHook:
    # TODO extend to other types of signals
    def __init__(self, token_positions, signal_indices, intervention_values):
        """
        token_positions: [bs]
        signal_indices: [bs]
        intervention_values: [bs]
        """
        self.token_positions = token_positions
        self.neuron_indices = signal_indices
        self.intervention_values = intervention_values

    def __call__(self, module, inputs, outputs):
        # outputs shape:
        # [bs, seq, hidden]
        outputs = outputs.clone()
        bs = outputs.shape[0]
        batch_idx = torch.arange(bs, device=outputs.device)

        outputs[
            batch_idx,
            self.token_positions,
            self.neuron_indices,
        ] = self.intervention_values.to(outputs.dtype)

        return outputs


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
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.signal_ranges = signal_ranges
        self.tokenizer = tokenizer
        self.experiment_config = experiment_config

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        token_positions = inputs.pop("token_position")
        signal_layers = inputs.pop("signal_layer")
        signal_indices = inputs.pop("signal_index")

        bs = token_positions.shape[0]
        device = token_positions.device

        # Sample random intervention values
        intervention_values = []

        for layer, idx in zip(signal_layers.tolist(), signal_indices.tolist()):
            low, high = self.signal_ranges[(layer, idx)]
            # TODO can change to something more natural instead of uniform
            val = random.uniform(low, high)
            intervention_values.append(val)

        intervention_values = torch.tensor(intervention_values, device=device, dtype=torch.float32)

        # Replace target text with intervention values
        precision = self.experiment_config.training_config.precision
        target_strings = [f"{v.item():.{precision}f}{self.tokenizer.eos_token}" for v in intervention_values]
        tokenized_targets = self.tokenizer(target_strings, add_special_tokens=False)["input_ids"]
        labels = inputs["labels"].clone()
        attn_mask = inputs["attention_mask"].clone()
        lora_mask = inputs["lora_token_mask"].clone()
        for i in range(bs):
            tgt = tokenized_targets[i]
            non_ignore = (labels[i] != -100).nonzero(as_tuple=True)[0]
            if len(non_ignore) == 0:
                continue
            start = non_ignore[0].item()
            # if len(tgt) > len(non_ignore): # if new label is longer (e.g. 0.123 -> -0.322)
            #     pdb.set_trace()
            #     end = min(labels.shape[1], len(labels[i, start:]))
            #     labels[i, start:end] = torch.tensor(tgt[: end - start], device=labels.device)
            #     attn_mask[i, start:end] = torch.tensor(1, device=labels.device)
            #     lora_mask[i, start:end] = torch.tensor(1, device=labels.device)
            # else: # if new label is shorter (e.g. -0.322 -> 0.123), need to flip the other digits
            #     end = min(start + len(tgt), labels.shape[1])
            #     labels[i, start:end] = torch.tensor(tgt[: end - start], device=labels.device)
            #     labels[i, end:] = torch.tensor(-100, device=labels.device)
            #     attn_mask[i, end:] = torch.tensor(0, device=labels.device)
            #     lora_mask[i, end:] = torch.tensor(0, device=labels.device)
            # with normalization of digit str (+0.123 or -0.211), length of new and old string should be same len
            assert len(tgt) == len(non_ignore), "new random signal not same length as original signal"
            end = labels.shape[1]
            labels[i, start:end] = torch.tensor(tgt[: end - start], device=labels.device)
            attn_mask[i, start:end] = torch.tensor(1, device=labels.device)
            lora_mask[i, start:end] = torch.tensor(1, device=labels.device)

        inputs["labels"] = labels
        inputs["attention_mask"] = attn_mask
        inputs["lora_attention_mask"] = lora_mask

        # Register hooks
        hooks = []
        unique_layers = torch.unique(signal_layers)

        for layer in unique_layers.tolist():
            mask = signal_layers == layer
            layer_token_positions = token_positions[mask]
            layer_signal_indices = signal_indices[mask]
            layer_intervention_values = intervention_values[mask]

            hook = InterventionHook(
                token_positions=layer_token_positions,
                signal_indices=layer_signal_indices,
                intervention_values=layer_intervention_values,
            )

            # TODO extend to other signal types
            module_path = resolve_mlp_signal_path(model, layer, path_to_output=False)
            module = operator.attrgetter(module_path)(model)
            handle = module.register_forward_hook(hook)
            hooks.append(handle)

        # Forward model and compute loss
        outputs = model(**inputs)
        loss = outputs.loss

        # Cleanup hooks
        for h in hooks:
            h.remove()

        return (loss, outputs) if return_outputs else loss


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
    output_dir = Path(config.output_dir) / f"{config.experiment_name}_intervention"

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
        signal_ranges[key] = (float(vals.min()),float(vals.max()),)

    # ========================================================
    # Preprocess datasets
    # ========================================================
    def preprocess(ex):
        out = preprocess_example(
            ex,
            tokenizer=tokenizer,
            prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
            max_length=config.training_config.max_seq_length,
            return_frozen_input_len=True,
        )
        out["token_position"] = ex["token_position"]
        out["signal_layer"] = ex["signal_layer"]
        out["signal_index"] = ex["signal_index"]
        return out

    train_dataset = train_dataset_raw.map(
        preprocess,
        remove_columns=train_dataset_raw.column_names,
    )
    dev_dataset = dev_dataset_raw.map(
        preprocess,
        remove_columns=dev_dataset_raw.column_names,
    )
    test_dataset = test_dataset_raw.map(
        preprocess,
        remove_columns=test_dataset_raw.column_names,
    )
    # TODO test prediction data

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
    model = get_peft_model(model,peft_config)
    model.resize_token_embeddings(len(tokenizer))

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
        data_collator=collator,
        compute_metrics=partial(compute_metrics,tokenizer=tokenizer),
        preprocess_logits_for_metrics=get_argmax_from_logits,
        experiment_config=config
    )

    train_results = trainer.train()
    trainer.save_metrics( "train", train_results.metrics)
    dev_results = trainer.evaluate()
    trainer.save_metrics("dev", dev_results,)


if __name__ == "__main__":
    main()

