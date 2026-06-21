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
import regex as re
import threading
from functools import partial
from pathlib import Path
from typing import Tuple

from tqdm import tqdm
import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments, PreTrainedTokenizerBase, DataCollatorForSeq2Seq,
)

from src.introspector.config import ExperimentConfig
from src.introspector.training.collator import LoRAGatingCollator
from src.introspector.training.model import replace_model_with_gated_version, replace_lora_with_gated_version
from src.introspector.training.metrics import compute_metrics, get_numericals, compute_metrics_from_numericals
from src.introspector.prompts import PROMPT_REGISTRY

INTROSPECTION_SEPARATOR = "<|introspection|>"


def preprocess_example(
    example,
    tokenizer,
    prompt_fn,
    max_length,
    return_frozen_input_len=False,
    for_prediction=False,
):
    """
    preprocess dataset for training or prediction
    :param example: Dict of examples
    :param tokenizer:
    :param prompt_fn: function to piece together the prompt
    :param max_length:
    :param return_frozen_input_len: used for lora token masking
    :param for_prediction: if True, do not suffix the target activation, and left-pad the batches
    :return:
    """
    introspection_prompt = prompt_fn(
        signal_type=example["signal_type"],
        layer=example["signal_layer"],
        signal_index=example["signal_index"],
        token=example["token"],
        token_index=example["token_position"],
    )
    full_prompt = f"{example['text']} {INTROSPECTION_SEPARATOR}{introspection_prompt}"

    target = example["target"]

    if for_prediction:
        full_text = f"{full_prompt}"
    else:
        full_text = f"{full_prompt} {target}{tokenizer.eos_token}"

    input_tokens = tokenizer(
        example['text'],
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

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
    out =  {
        "input_ids": input_ids,
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels,
    }
    if return_frozen_input_len:
        out["frozen_input_len"] = len(input_tokens["input_ids"])

    return out


def get_argmax_from_logits(logits, labels):
    return torch.argmax(logits, dim=-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="Whether to overwrite cached experiment files if there exist one")

    return parser.parse_args()


def main():

    args = parse_args()

    config = ExperimentConfig.model_validate_json(args.config.read_text())

    cache_dir = Path(config.cache_dir) / config.experiment_name
    train_path = cache_dir / "train"
    dev_path = cache_dir / "dev"
    test_path = cache_dir / "test"
    output_dir = Path(config.output_dir) / config.experiment_name

    if (output_dir / "test_results.json").exists() and not args.overwrite_cache:
        print(f"Experiments ran already, skipping.. {output_dir}")
        return

    # load tokenizer, add special token
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {"additional_special_tokens": [INTROSPECTION_SEPARATOR]}
    num_added = tokenizer.add_special_tokens(special_tokens)

    # load cached signal datasets
    train_dataset_raw = load_from_disk(str(train_path))
    train_dataset = train_dataset_raw.map(
        lambda ex: preprocess_example(
            ex,
            tokenizer=tokenizer,
            prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
            max_length=config.training_config.max_seq_length,
            return_frozen_input_len=config.training_config.disable_adapter_in_input,
        ),
        remove_columns=train_dataset_raw.column_names,
    )
    dev_dataset_raw = load_from_disk(str(dev_path))
    dev_dataset = dev_dataset_raw.map(
        lambda ex: preprocess_example(
            ex,
            tokenizer=tokenizer,
            prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
            max_length=config.training_config.max_seq_length,
            return_frozen_input_len=config.training_config.disable_adapter_in_input,
        ),
        remove_columns=dev_dataset_raw.column_names,
    )
    test_dataset_raw = load_from_disk(str(test_path))
    test_dataset = test_dataset_raw.map(
        lambda ex: preprocess_example(
            ex,
            tokenizer=tokenizer,
            prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
            max_length=config.training_config.max_seq_length,
            return_frozen_input_len=config.training_config.disable_adapter_in_input,
        ),
        remove_columns=test_dataset_raw.column_names,
    )
    test_dataset_pred = test_dataset_raw.map(
        lambda ex: preprocess_example(
            ex,
            tokenizer=tokenizer,
            prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
            max_length=config.training_config.max_seq_length,
            return_frozen_input_len=config.training_config.disable_adapter_in_input,
            for_prediction=True,
        ),
        remove_columns=test_dataset_raw.column_names,
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
    model.resize_token_embeddings(len(tokenizer))

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
    model.resize_token_embeddings(len(tokenizer)) # https://discuss.huggingface.co/t/resize-embeddings-on-peft-model/136761/3

    if config.training_config.disable_adapter_in_input:
        # monkey patch forward to take in mask
        replace_model_with_gated_version(model)

        # modify lora linear so it takes in mask and applies across tokens
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
    else:
        collator = DataCollatorForSeq2Seq(
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
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        # report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        preprocess_logits_for_metrics=get_argmax_from_logits  # otherwise during test prediction it's too much mem
    )
    train_results = trainer.train()

    trainer.save_metrics("train", train_results.metrics)

    # get dev eval split final metrics
    dev_results = trainer.evaluate()
    trainer.save_metrics("dev", dev_results)

    # for test split (splitting eval into different subsets to control for e.g.token overlap)
    test_outputs = trainer.predict(test_dataset)
    test_predictions = test_outputs.predictions
    test_label_ids = test_outputs.label_ids
    full_test_results = compute_metrics((test_predictions, test_label_ids), tokenizer=tokenizer)
    train_tokens = set(train_dataset_raw["token"])
    seen_token_mask = test_dataset_raw.map(lambda ex: {"seen":True if ex["token"] in train_tokens else False}, remove_columns=test_dataset_raw.column_names)
    seen_token_mask = np.array(list(seen_token_mask["seen"]))
    seen_token_test_results = compute_metrics([test_predictions[seen_token_mask], test_label_ids[seen_token_mask]], tokenizer=tokenizer)
    unseen_token_test_results = compute_metrics([test_predictions[~seen_token_mask], test_label_ids[~seen_token_mask]], tokenizer=tokenizer)
    trainer.save_metrics("test", full_test_results)
    trainer.save_metrics("test_seen_token", seen_token_test_results)
    trainer.save_metrics("test_unseen_token", unseen_token_test_results)

    history = trainer.state.log_history
    with open(str(output_dir / "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    final_dir = output_dir / "adapter_weights"

    # trainer.save_model(str(final_dir))
    # print(f"Saved model to: {final_dir}")

    # now evaluate using prediction (autoregressive generation)
    if not config.training_config.disable_adapter_in_input:
        exit() # Not implemented yet

    dataloader = DataLoader(
        test_dataset_pred,
        batch_size=config.training_config.train_batch_size,
        collate_fn=pred_collator,
        shuffle=False
    )
    model.eval()
    gen_nums = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(model.device)
            gen_tokens_batch = model.generate(**batch, max_new_tokens=config.training_config.precision+2)
            input_seq_len = batch["input_ids"].shape[1]
            gen_strs_batch = tokenizer.batch_decode(gen_tokens_batch[:, input_seq_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for gen_str in gen_strs_batch:
                try:
                    gen_str_clean = re.sub(r'[^0-9.\-+]', '', gen_str)
                    gen_num = float(gen_str_clean)
                except:
                    gen_num = np.nan
                gen_nums.append(gen_num)

    labels = np.array([float(a) for a in test_dataset_raw["target"]])
    gen_nums = np.array(gen_nums)
    test_pred_metrics = compute_metrics_from_numericals(gen_nums, labels)
    seen_token_test_pred_results = compute_metrics_from_numericals(gen_nums[seen_token_mask], labels[seen_token_mask])
    unseen_token_test_pred_results = compute_metrics_from_numericals(gen_nums[~seen_token_mask], labels[~seen_token_mask])
    trainer.save_metrics("test_pred", test_pred_metrics)
    trainer.save_metrics("test_pred_seen_token", seen_token_test_pred_results)
    trainer.save_metrics("test_pred_unseen_token", unseen_token_test_pred_results)



if __name__ == "__main__":
    main()