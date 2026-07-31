#!/usr/bin/env python3
"""
Score A: base-model bypass ceiling (faithful context-predictability at the readout).

For each neuron, measure how well a linear probe can recover the neuron's true
activation from the BASE model's readout-position hidden state, running the real
{input, introspection_prompt} sequence (LoRA off). This is the endogenous
decodability available AT THE READOUT before any training -- the contextual
shortcut that competes with causal reading.

Prediction (deviation hypothesis): a neuron's end-of-training ENDOGENOUS corr
should be NEGATIVELY related to Score A -- high bypass ceiling => the neuron sits
at its context-expected value => nothing for the deviation-reader to read
endogenously.

Method (per neuron):
  1. run the base model over the neuron's test cache with its introspection prompt
  2. take hidden_states[-1][:, -1] (final layer, last token = readout position)
  3. ridge-probe (train/eval split) -> the neuron's true activation; report test corr

Notes
-----
- Uses the BASE model (no PEFT): Score A is bypass availability *before* training,
  the quantity that predicts whether training relies on bypass.
- The <|introspection|> token gets a fresh (untrained) embedding after resize; it
  is constant across examples so a probe absorbs it -- variation comes from the
  input text and the per-example token/position in the prompt.
- Score A entangles context-predictability with retrieval (the value must reach
  the readout via attention); that entangled quantity IS the operative bypass.

Example
-------
python score_A_bypass_ceiling.py --config experiment_configs/.../<ref>.json \
    --cache-glob 'data/.../*/test' --n-examples 2000
"""
from __future__ import annotations

import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

from src.introspector.config import ExperimentConfig
from src.introspector.prompts import PROMPT_REGISTRY
from src.introspector.training.collator import LoRAGatingCollator
from train_introspector import preprocess_example, INTROSPECTION_SEPARATOR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True,
                   help="Reference config (model_name, prompt_type, max_seq_length, seed).")
    p.add_argument("--cache-glob", type=str, required=True,
                   help="Glob matching per-neuron test cache dirs (HF datasets), e.g. 'data/.../*/test'.")
    p.add_argument("--n-examples", type=int, default=2000,
                   help="Examples per neuron used to fit+eval the bypass probe.")
    p.add_argument("--split-frac", type=float, default=0.7, help="Train fraction for the probe.")
    p.add_argument("--ridge", type=float, default=10.0,
                   help="Ridge alpha for the bypass probe (readout features are high-variance).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    return p.parse_args()


def bypass_ceiling(X, y, ridge, split_frac, seed):
    """Ridge probe X->y with a shuffled row split; return held-out Pearson corr."""
    n = len(y)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    ntr = int(n * split_frac)
    if ntr < 5 or n - ntr < 5:
        return float("nan")
    # standardize features on train (readout has massive-activation dims)
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = (X - mu) / sd
    reg = Ridge(alpha=ridge).fit(Xs[:ntr], y[:ntr])
    pred = reg.predict(Xs[ntr:])
    if np.std(pred) < 1e-12 or np.std(y[ntr:]) < 1e-12:
        return float("nan")
    return float(pearsonr(pred, y[ntr:])[0])


@torch.no_grad()
def readout_features(model, dataset, collator, batch_size):
    """Base-model final-layer last-token hidden state + true activation per example."""
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    feats, acts = [], []
    for batch in dl:
        batch = batch.to(model.device)
        activation = batch["activation"]
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    output_hidden_states=True)
        feats.append(out.hidden_states[-1][:, -1, :].float().cpu())
        acts.append(activation.float().cpu())
    return torch.cat(feats).numpy(), torch.cat(acts).numpy()


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(args.config.read_text())
    seed = config.seed
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [INTROSPECTION_SEPARATOR]})
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, device_map="auto", local_files_only=True,
        torch_dtype=torch.bfloat16 if config.training_config.bf16 else torch.float32,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    collator = LoRAGatingCollator(
        tokenizer=tokenizer, model=model, padding=True,
        label_pad_token_id=-100, return_tensors="pt", for_prediction=True,
    )
    prompt_fn = PROMPT_REGISTRY[config.training_config.prompt_type]
    max_len = config.training_config.max_seq_length

    def preprocess(ex):
        out = preprocess_example(ex, tokenizer=tokenizer, prompt_fn=prompt_fn,
                                 max_length=max_len, return_frozen_input_len=True,
                                 for_prediction=True)
        out["activation"] = ex["activation"]
        return out

    cache_dirs = sorted(glob.glob(args.cache_glob))
    if not cache_dirs:
        raise SystemExit(f"No caches matched {args.cache_glob!r}")
    print(f"[scoreA] {len(cache_dirs)} neuron caches; base model {config.model_name}")

    rows = []
    for cdir in cache_dirs:
        ds_raw = load_from_disk(cdir)
        layer = int(ds_raw["signal_layer"][0])
        index = int(ds_raw["signal_index"][0])
        n = min(args.n_examples, len(ds_raw))
        ds = ds_raw.select(range(n)).map(preprocess, remove_columns=ds_raw.column_names)
        X, y = readout_features(model, ds, collator, args.batch_size)
        score = bypass_ceiling(X, y, args.ridge, args.split_frac, seed)
        rows.append({"neuron": f"N{layer}.{index}", "layer": layer,
                     "neuron_index": index, "score_A": score, "n": n})
        print(f"[scoreA] N{layer}.{index}: score_A={score:+.4f} (n={n})")

    df = pd.DataFrame(rows).sort_values("score_A").reset_index(drop=True)
    df.to_csv(args.output_dir / "score_A_bypass_ceiling.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.hist(df["score_A"].dropna(), bins=20)
    plt.xlabel("Score A (base-model bypass ceiling corr)"); plt.ylabel("neuron count")
    plt.title("Per-neuron bypass ceiling (Score A)")
    plt.tight_layout()
    plt.savefig(args.output_dir / "score_A_hist.png", dpi=150)
    plt.close()
    print(f"[scoreA] wrote {len(df)} neuron scores to {args.output_dir}")


if __name__ == "__main__":
    main()
