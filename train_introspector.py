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

import os
import glob
import argparse
import json
import math
import pdb

import pandas as pd
import regex as re
import threading
from functools import partial
from pathlib import Path
from typing import Tuple

from tqdm import tqdm
import numpy as np
import torch
from datasets import load_from_disk, Dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments, PreTrainedTokenizerBase, DataCollatorForSeq2Seq,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.introspector.config import ExperimentConfig
from src.introspector.training.collator import LoRAGatingCollator
from src.introspector.training.model import (
    replace_model_with_gated_version, replace_lora_with_gated_version,
    StripEmbeddingSaveCallback, load_added_token_embeddings,
)
from src.introspector.training.metrics import compute_metrics, get_numericals, compute_metrics_from_numericals
from src.introspector.prompts import PROMPT_REGISTRY, build_prompt_kwargs

INTROSPECTION_SEPARATOR = "<|introspection|>"


def preprocess_example(
    example,
    tokenizer,
    prompt_fn,
    max_length,
    return_frozen_input_len=False,
    return_position_valid=False,
    for_prediction=False,
    detection=False,
    detection_active_word="active",
    detection_inactive_word="inactive",
    separator=INTROSPECTION_SEPARATOR,
):
    """
    preprocess dataset for training or prediction
    :param example: Dict of examples
    :param tokenizer:
    :param prompt_fn: function to piece together the prompt
    :param max_length:
    :param return_frozen_input_len: used for lora token masking
    :param return_position_valid: emit a per-row flag for whether max_length kept the
        introspected token. Opt-in: it is an extra column, and callers that map with
        remove_columns=<original> would otherwise hand it to the collator as a model kwarg.
    :param for_prediction: if True, do not suffix the target activation, and left-pad the batches
    :param separator: what sits between the text and the introspection prompt. Trained
        runs use the added special token; the untrained prompt baseline
        (score_prompt_baseline.py) passes plain text, because to a model that never
        trained on it the special token is a random, frozen embedding.
    :return:
    """
    # Argument list depends on the signal type (attention names a source token as well
    # as a destination one), so the mapping lives in prompts.build_prompt_kwargs rather
    # than being spelled out here -- this function is the shared entry point for a dozen
    # callers. For mlp/sae/residual it produces exactly the arguments this used to pass.
    introspection_prompt = prompt_fn(
        signal_type=example["signal_type"],
        **build_prompt_kwargs(example),
    )
    full_prompt = f"{example['text']} {separator}{introspection_prompt}"

    if detection:
        # binary readout for sparse features: the feature is active iff its
        # (JumpReLU-thresholded) activation is > 0. strip() so the template's
        # single separator space is the only space, regardless of whether the
        # configured word carries a leading space.
        word = detection_active_word if float(example["activation"]) > 0 else detection_inactive_word
        target = word.strip()
    else:
        target = example["target"]

    if for_prediction:
        full_text = f"{full_prompt}"
    else:
        full_text = f"{full_prompt} {target}{tokenizer.eos_token}"

    # add_special_tokens=True (i.e. a leading BOS) on ALL THREE tokenizations, to
    # match cache_signals.py, which tokenizes with the default add_special_tokens=True.
    # Without BOS here the pipeline was wrong twice over:
    #   POSITION  cached token_position indexes a BOS-prefixed sequence, so it pointed
    #             one token to the RIGHT of the token the prompt names -- the
    #             intervention hook injected into the neighbour. At pos=-1 that landed
    #             one past the text, i.e. the first LoRA-ACTIVE token under gating,
    #             which is why pos=-1 worked while earlier (frozen) positions did not.
    #   FIDELITY  BOS is Llama's attention sink; dropping it perturbs every downstream
    #             activation, so even at the right position the live neuron correlated
    #             only ~0.81 with the cached target instead of ~1.0.
    # Verified by probe_cache_alignment.py: (bos=True, offset=0) gives token-match 1.000
    # and corr 0.9999 vs the cache; every other convention fails one or both. All three
    # calls must stay consistent -- frozen_input_len and prompt_len both shift by the
    # same one token, so the gate mask and label masking keep their boundaries.
    input_tokens = tokenizer(
        example['text'],
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

    prompt_tokens = tokenizer(
        full_prompt,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
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

    # Truncation guard. `token_position` indexes the FULL cached sequence; here the
    # text is re-tokenized under `max_length`, which can cut the introspected token
    # off entirely. Nothing downstream would notice: InterventionHook and the capture
    # hooks .clamp() the position into range, so a truncated target silently becomes
    # "inject at the last surviving token" -- a wrong-position run that still looks
    # healthy. Harmless while every run used pos=-1 on short texts; a real hazard for
    # long-context runs, where max_length must cover text + prompt + target, not just
    # the text. Flagged per row so the caller can refuse to train on a poisoned split
    # rather than discovering it in the results.
    if return_position_valid:
        n_text = len(input_tokens["input_ids"])
        valid = int(example["token_position"]) < n_text
        # An attention row addresses TWO tokens. The source is always at or before the
        # destination, so it survives whenever the destination does -- but only if it is
        # also >= 0, which a re-tokenization that shortened the text can break. Checking
        # it explicitly costs nothing and stops a row with an unaddressable source from
        # reaching the port's assert mid-training.
        src_pos = example.get("src_token_position")
        if src_pos is not None:
            valid = valid and 0 <= int(src_pos) < n_text
        out["position_valid"] = valid
        # Distinguishes the TWO ways a position can go out of range, which need opposite
        # responses. Truncation is systematic and fixable (raise max_length); a
        # decode/re-encode round trip that lands on a different token count is neither.
        out["text_truncated"] = n_text >= max_length

    return out


# A handful of rows per split is expected (decode/re-encode round trips), a systematic
# fraction is a misconfiguration. Anything at or below this is dropped with a warning.
POSITION_DROP_TOLERANCE = 0.001


def drop_invalid_positions(dataset, split: str, max_length: int,
                           tolerance: float = POSITION_DROP_TOLERANCE):
    """Drop rows whose introspected token is not addressable, or refuse if there are many.

    A row is unusable when token_position indexes past the re-tokenized text: the
    intervention and capture hooks .clamp() the position into range, so it would train
    on the WRONG activation while looking perfectly healthy.

    There are two causes and they want opposite responses:

      TRUNCATION      max_length cut the text short. Systematic, scales with the data,
                      and fixable by raising training_config.max_seq_length to cover
                      text + introspection prompt + target.
      ROUND TRIP      cache_signals stores `text` as a DECODED string, so training
                      re-tokenizes it. For a small number of documents that round trip
                      does not reproduce the original token count, and the cached
                      token_position no longer addresses the intended token. Not
                      fixable by any max_length; those rows are simply unusable.
                      (The cache also stores text_input_ids -- training from those
                      directly would remove the round trip and this whole failure mode.)

    So: drop a tolerable fraction with a warning naming the dominant cause, and raise
    only when enough rows fail that something is actually misconfigured.
    """
    cols = getattr(dataset, "column_names", [])
    if "position_valid" not in cols:
        return dataset
    valid = np.asarray(dataset["position_valid"], dtype=bool)
    n_bad = int((~valid).sum())
    drop_cols = [c for c in ("position_valid", "text_truncated") if c in cols]
    if n_bad == 0:
        return dataset.remove_columns(drop_cols)

    frac = n_bad / len(dataset)
    n_trunc = 0
    if "text_truncated" in cols:
        n_trunc = int((np.asarray(dataset["text_truncated"], dtype=bool) & ~valid).sum())
    cause = (f"{n_trunc} truncated by max_length={max_length}, "
             f"{n_bad - n_trunc} from decode/re-encode round trips")
    if frac > tolerance:
        raise ValueError(
            f"[{split}] {n_bad}/{len(dataset)} ({frac:.2%}) examples have an "
            f"unaddressable introspected token ({cause}). The hooks would clamp these to "
            f"the last surviving token and train on the wrong activation. "
            + (f"Raise training_config.max_seq_length above {max_length} to cover text + "
               f"introspection prompt + target."
               if n_trunc > n_bad - n_trunc else
               "This is NOT a max_length problem -- the text re-tokenizes to a different "
               "length than the cache recorded. Rebuild the cache, or train from the "
               "cached text_input_ids.")
        )
    print(f"[{split}] dropping {n_bad}/{len(dataset)} ({frac:.3%}) rows with an "
          f"unaddressable introspected token ({cause}); under the {tolerance:.1%} "
          f"tolerance, so continuing.")
    kept = dataset.filter(lambda ex: ex["position_valid"])
    return kept.remove_columns(drop_cols)


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

    cache_dir = config.dataset_cache_dir
    train_path = cache_dir / "train"
    dev_path = cache_dir / "dev"
    test_path = cache_dir / "test"
    output_dir = Path(config.output_dir) / config.experiment_name

    if (output_dir / "test_pred_results.json").exists() and not args.overwrite_cache:
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
        torch_dtype=torch.bfloat16 if config.training_config.bf16 else torch.float32,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))

    # wrap model in lora config
    peft_config = LoraConfig(
        r=config.training_config.lora_r, # rank
        lora_alpha=config.training_config.lora_alpha,  # Scaling factor
        target_modules=config.training_config.lora_target_modules,  # "all-linear", Apply to all linear layers (or specify specific ones like ["q_proj", "v_proj"])
        lora_dropout=config.training_config.lora_dropout,
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
        bf16=config.training_config.bf16,
        remove_unused_columns=False,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        preprocess_logits_for_metrics=get_argmax_from_logits,  # otherwise during test prediction it's too much mem
        # PEFT saves the whole resized embed_tokens/lm_head (~1.0 GB, 98% of the
        # checkpoint) because the vocab grew by one; drop them at save time and keep
        # only the added rows, which load_added_token_embeddings restores.
        callbacks=[StripEmbeddingSaveCallback(num_added)],
    )
    ckpt_dirs = glob.glob(f"{output_dir}/checkpoint-*")
    if ckpt_dirs and not args.overwrite_cache:
        print(f"Found Trained model, skipping training and loading directly ..")
        latest_checkpoint_path = max(ckpt_dirs, key=os.path.getctime)
        train_results = trainer.train(resume_from_checkpoint=latest_checkpoint_path)
        load_added_token_embeddings(model, latest_checkpoint_path)
    else:
        train_results = trainer.train()
    trainer.save_metrics("train", train_results.metrics)

    # get dev eval split final metrics
    dev_results = trainer.evaluate()
    trainer.save_metrics("dev", dev_results)

    history = trainer.state.log_history
    with open(str(output_dir / "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # final_dir = output_dir / "adapter_weights"
    # trainer.save_model(str(final_dir))
    # print(f"Saved model to: {final_dir}")

    # for test split (splitting eval into different subsets to control for e.g.token overlap)
    evaluate_test_split(config, model, pred_collator, test_dataset, test_dataset_pred, test_dataset_raw, tokenizer,
                        train_dataset_raw, trainer)


def evaluate_test_split(
    config: ExperimentConfig,
    model: AutoModelForCausalLM,
    pred_collator: DataCollatorForSeq2Seq,
    test_dataset: Dataset,
    test_dataset_pred: Dataset,
    test_dataset_raw: Dataset,
    tokenizer: AutoTokenizer,
    train_dataset_raw: Dataset,
    trainer: Trainer,
    suffix: str = "",
):
    """
    All evaluation logic for test split (next-token-prediction eval + generation eval)

    Args:
        suffix: experiment artifact suffix
    """
    test_outputs = trainer.predict(test_dataset)
    test_predictions = test_outputs.predictions
    test_label_ids = test_outputs.label_ids
    full_test_results = compute_metrics((test_predictions, test_label_ids), tokenizer=tokenizer)
    train_tokens = set(train_dataset_raw["token"])
    seen_token_mask = test_dataset_raw.map(lambda ex: {"seen": True if ex["token"] in train_tokens else False},
                                           remove_columns=test_dataset_raw.column_names)
    seen_token_mask = np.array(list(seen_token_mask["seen"]))
    seen_token_test_results = compute_metrics([test_predictions[seen_token_mask], test_label_ids[seen_token_mask]],
                                              tokenizer=tokenizer)
    unseen_token_test_results = compute_metrics([test_predictions[~seen_token_mask], test_label_ids[~seen_token_mask]],
                                                tokenizer=tokenizer)
    trainer.save_metrics(f"test{suffix}", full_test_results)
    trainer.save_metrics(f"test_seen_token{suffix}", seen_token_test_results)
    trainer.save_metrics(f"test_unseen_token{suffix}", unseen_token_test_results)
    output_dir = Path(config.output_dir) / config.experiment_name
    pred_df = pd.DataFrame({"predictions": test_predictions, "labels": test_label_ids, "activation": test_dataset["activation"]})
    pred_df.to_csv(str(output_dir / f"test_predictions{suffix}.json"), index=False)
    sns.scatterplot(x="labels", y="predictions", hue="activation", data=pred_df)
    plt.savefig(str(output_dir / f"test_predictions_scatter{suffix}.png"))
    plt.close()
    # now evaluate using prediction (autoregressive generation)
    if not config.training_config.disable_adapter_in_input or config.use_regressor:
        return  # Not implemented yet, and doesn't apply for regressor

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
            gen_tokens_batch = model.generate(
                **batch, max_new_tokens=config.training_config.precision + 2, pad_token_id=tokenizer.pad_token_id,
                do_sample=config.eval_config.do_sample, temperature=config.eval_config.temperature,
                top_k=config.eval_config.top_k, top_p=config.eval_config.top_p
            )
            input_seq_len = batch["input_ids"].shape[1]
            gen_strs_batch = tokenizer.batch_decode(gen_tokens_batch[:, input_seq_len:], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
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
    unseen_token_test_pred_results = compute_metrics_from_numericals(gen_nums[~seen_token_mask],
                                                                     labels[~seen_token_mask])
    trainer.save_metrics(f"test_pred{suffix}", test_pred_metrics)
    trainer.save_metrics(f"test_pred_seen_token{suffix}", seen_token_test_pred_results)
    trainer.save_metrics(f"test_pred_unseen_token{suffix}", unseen_token_test_pred_results)


if __name__ == "__main__":
    main()