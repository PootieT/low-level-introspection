"""
This file takes in a finetuned introspector, and intervenes at
the low-level signal it is trained on, and caches model outputs
after intervention
"""
import os
import glob
import pdb
import json
import regex as re
import operator
from pathlib import Path
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import load_from_disk
from nnsight.modeling.language import LanguageModel
from peft import PeftModel
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.data import data_collator

from generate_experiment_config import load_spec
from src.introspector.config import ExperimentConfig
from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.prompts import PROMPT_REGISTRY
from src.introspector.signals.resolution import resolve_signal_source
from src.introspector.training.collator import LoRAGatingCollator
from src.introspector.training.metrics import compute_metrics_from_numericals
from src.introspector.training.model import replace_model_with_gated_version, replace_lora_with_gated_version
from src.introspector.utils import rsetattr
from train_introspector import preprocess_example, INTROSPECTION_SEPARATOR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",  # experiment config used to train introspector
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--intervention-signals",
        type=str,
        default=None,
        help="Inline or path to json configuration of intervention signals."
             'For fixed signals it might look like {"type":"explicit", "signals":["N12.1","N12.2"]}, or simply "N12.1,N12.2"'
             'For sampling random sets it might look like {"type":"uniform_layer_sampler", '
             '  "signal_type":"attention_score","layers":[1,2,3,4], "signals_per_layer"=2, "seed"=21}'
             'default to None, the training signals'
    )
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="Whether to overwrite cached dataset files if there exist one")

    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(Path(args.config).read_text())
    if args.intervention_signals is None:
        signals = config.train_signals
    else:
        model_config = AutoConfig.from_pretrained(config.model_name, local_files_only=True)
        signals = resolve_signal_source(load_spec(args.train_signals), model_config=model_config, precision=args.precision)

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {"additional_special_tokens": [INTROSPECTION_SEPARATOR]}
    num_added = tokenizer.add_special_tokens(special_tokens)

    model = AutoModelForCausalLM.from_pretrained(config.model_name,device_map="auto", local_files_only=True, torch_dtype=torch.bfloat16,) #
    model.resize_token_embeddings(len(tokenizer))
    device = model.device
    checkpoint_paths = glob.glob(str(Path(config.output_dir) / config.experiment_name / "checkpoint-*"))
    latest_checkpoint_path = max(checkpoint_paths, key=os.path.getctime)
    model = PeftModel.from_pretrained(model, latest_checkpoint_path, autocast_adapter_dtype=False)

    # monkey patch to get the introspection token masking
    replace_model_with_gated_version(model)
    replace_lora_with_gated_version(model)
    gated_model = LanguageModel(model, tokenizer=tokenizer, )

    # load test dataset, get signal ranges
    test_data_path = Path(config.cache_dir) / config.experiment_name / "test"
    test_dataset_raw = load_from_disk(str(test_data_path))
    df = test_dataset_raw.to_pandas()
    dscr = df.groupby(["signal_type", "signal_layer", "signal_index"])["activation"].describe()
    dscr = dscr.to_dict()
    test_dataset = test_dataset_raw.map(
        lambda ex: preprocess_example(
            ex,
            tokenizer=tokenizer,
            prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
            max_length=config.training_config.max_seq_length,
            return_frozen_input_len=config.training_config.disable_adapter_in_input,
            for_prediction=True
        ),
        remove_columns=test_dataset_raw.column_names,
    )

    collator = LoRAGatingCollator(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
        for_prediction=True
    )

    # for each signal forward whole dataset, for each datapoint, sample
    # a random activation for the neuron, obtain generation
    # compare sampled random activation to generated activation
    signal_output = []
    all_metrics = {}
    for signal in signals:
        print(f"Processing signal: {signal}")
        dataloader = DataLoader(
            test_dataset,
            batch_size=config.training_config.train_batch_size,
            collate_fn=collator,
            shuffle=False
        )
        unparsable_cnt = 0
        # uniformly sample a random signal as replacement (TODO could make it more realistic)
        high, low = dscr["min"][(signal.type, signal.layer, signal.index)], dscr["max"][(signal.type, signal.layer, signal.index)]
        random_signals = torch.rand(len(test_dataset)).to(torch.bfloat16).to(device)*(high-low) + low  #
        data_idx = 0
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            bs, input_seq_len = batch["input_ids"].shape[0], batch["input_ids"].shape[1]
            random_signal_batch = random_signals[data_idx:data_idx + bs]
            og_token_positions = torch.tensor(test_dataset_raw["token_position"][data_idx:data_idx + bs]).to(device)
            # prediction requires left padding, so shift by the leading pad count.
            # NOTE: the extra -1 that used to sit here was compensating for training
            # tokenizing without BOS while the cache tokenized with it. preprocess_example
            # now keeps BOS, so cached token_position maps directly and no offset applies.
            token_positions = og_token_positions + (batch["input_ids"] == tokenizer.pad_token_id).sum(dim=1).to(device)
            with torch.no_grad(), gated_model.generate(batch, max_new_tokens=signal.precision+2, pad_token_id=tokenizer.pad_token_id):
                # TODO extend to other signal types
                mlp_layer_path = resolve_mlp_signal_path(gated_model, signal.layer)
                getter = operator.attrgetter(mlp_layer_path)
                # assuming dimensions are [bs, token, h], output dim is [bs, h]
                og_output = getter(gated_model)
                og_signal_batch = og_output[range(bs), token_positions, signal.index].detach().cpu().tolist().save()
                og_output[range(bs), token_positions, signal.index] = random_signal_batch
                output = gated_model.generator.output.save()

            # parse output to number and save
            gen_strs_batch = tokenizer.batch_decode(output[:, input_seq_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, pred_str in enumerate(gen_strs_batch):
                try:
                    pred_str_cleaned = re.sub(r'[^0-9.\-+]', '', pred_str)
                    pred_num = float(pred_str_cleaned)
                    signal_output.append({"data_idx": data_idx+i,"signal": str(signal), "intervened_signal": random_signal_batch[i].cpu().item(), "pred_num": pred_num, "og_signal": og_signal_batch[i]})
                except:
                    unparsable_cnt += 1
                    signal_output.append({"data_idx": data_idx + i, "signal": str(signal),"intervened_signal": random_signal_batch[i].cpu().item(), "pred_num": np.nan, "og_signal": og_signal_batch[i]})
            data_idx += bs

        print(f"unparsable predictions cnt = {unparsable_cnt}")
        all_metrics[str(signal)] = compute_metrics_from_numericals([r["pred_num"] for r in signal_output], [r["intervened_signal"] for r in signal_output])

    output_df = pd.DataFrame(signal_output)
    output_df.to_parquet(Path(config.output_dir) / config.experiment_name / "intervention_results.parquet")
    with open(str(Path(config.output_dir) / config.experiment_name / "intervention_results_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)



if __name__ == "__main__":
    main()