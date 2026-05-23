#!/usr/bin/env python3

"""
Train an introspector model on cached activation datasets.

This script:
1. Loads frozen experiment config
2. Loads cached HF datasets
3. Formats prompts + targets
4. Trains a causal LM to predict signals

Example:

python scripts/train_introspector.py \
    --config configs/experiments/mlp_generalization.json
"""

from __future__ import annotations

import argparse
import json
import math
import pdb
from functools import partial
from pathlib import Path
from typing import Tuple

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments, PreTrainedTokenizerBase, DataCollatorForSeq2Seq,
)

from src.introspector.config import ExperimentConfig
from src.introspector.training.metrics import compute_metrics
from src.introspector.prompts import PROMPT_REGISTRY

INTROSPECTION_SEPARATOR = "<|introspection|>"


def preprocess_example(
    example,
    tokenizer,
    prompt_fn,
    max_length,
):

    introspection_prompt = prompt_fn(
        signal_type=example["signal_type"],
        layer=example["signal_layer"],
        signal_index=example["signal_index"],
        token=example["token"],
        token_index=example["token_position"],
    )
    full_prompt = (f"{example['text']} "
                   f"{INTROSPECTION_SEPARATOR}"
                   f"{introspection_prompt}")

    target = example["target"]

    full_text = f"{full_prompt} {target}{tokenizer.eos_token}"

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    prompt_tokens = tokenizer(
        full_prompt,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    input_ids = full_tokens["input_ids"]
    labels = input_ids.copy()
    prompt_len = len(prompt_tokens["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len
    return {
        "input_ids": input_ids,
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels,
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main():

    args = parse_args()

    config = ExperimentConfig.model_validate_json(args.config.read_text())

    cache_dir = Path(config.cache_dir) / config.experiment_name
    train_path = cache_dir / "train"
    test_path = cache_dir / "test"

    # load tokenizer, add special token
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {
        "additional_special_tokens": [
            INTROSPECTION_SEPARATOR
        ]
    }
    num_added = tokenizer.add_special_tokens(
        special_tokens
    )

    # load cached signal datasets
    train_dataset_raw = load_from_disk(str(train_path))
    train_dataset = train_dataset_raw.map(
        lambda ex: preprocess_example(
            ex,
            tokenizer=tokenizer,
            prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
            max_length=config.training_config.max_seq_length,
        ),
        remove_columns=train_dataset_raw.column_names,
    )
    test_dataset_raw = load_from_disk(str(test_path))
    test_dataset = test_dataset_raw.map(
        lambda ex: preprocess_example(
            ex,
            tokenizer=tokenizer,
            prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
            max_length=config.training_config.max_seq_length,
        ),
        remove_columns=test_dataset_raw.column_names,
    )

    # split test into several versions controlling seen vs. unseen tokens
    train_tokens = set(train_dataset_raw["token"])
    test_dataset_unseen_raw = test_dataset_raw.filter(lambda ex: ex["token"] not in train_tokens)
    test_dataset_seen_raw = test_dataset_raw.filter(lambda ex: ex["token"] in train_tokens)
    test_dataset_unseen = test_dataset_unseen_raw.map(
        lambda ex: preprocess_example(
            ex,
            tokenizer=tokenizer,
            prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
            max_length=config.training_config.max_seq_length,
        ),
        remove_columns=test_dataset_unseen_raw.column_names,
    )
    test_dataset_seen = test_dataset_seen_raw.map(
        lambda ex: preprocess_example(
            ex,
            tokenizer=tokenizer,
            prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
            max_length=config.training_config.max_seq_length,
        ),
        remove_columns=test_dataset_seen_raw.column_names,
    )

    for i in range(3):
        print(f"============= Training example row {i} =============")
        print(tokenizer.decode(train_dataset[i]["input_ids"]))

    for i in range(3):
        print(f"============= Test example row {i} =============")
        print(tokenizer.decode(test_dataset[i]["input_ids"]))

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # wrap model in lora config
    peft_config = LoraConfig(
        r=config.training_config.lora_r, # rank
        lora_alpha=config.training_config.lora_alpha,  # Scaling factor
        target_modules="all-linear",  # Apply to all linear layers (or specify specific ones like ["q_proj", "v_proj"])
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    output_dir = Path(config.output_dir) / config.experiment_name

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config.training_config.learning_rate,
        per_device_train_batch_size=config.training_config.train_batch_size,
        per_device_eval_batch_size=config.training_config.eval_batch_size,
        num_train_epochs=config.training_config.num_epochs,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        # report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
    )
    train_results = trainer.train()

    trainer.save_metrics("train", train_results.metrics)
    eval_results = trainer.evaluate()
    trainer.save_metrics("test", eval_results)
    history = trainer.state.log_history
    with open(str(output_dir / "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # evaluate test split carefully by looking at instances with seen tokens vs not
    other_eval_metrics = trainer.evaluate(
        eval_dataset={
            "test_seen_tokens": test_dataset_seen,
            "test_unseen_tokens": test_dataset_unseen,
        }
    )
    trainer.save_metrics("test_others", other_eval_metrics)

    final_dir = output_dir / "adapter_weights"

    trainer.save_model(str(final_dir))
    print(f"Saved model to: {final_dir}")


if __name__ == "__main__":
    main()