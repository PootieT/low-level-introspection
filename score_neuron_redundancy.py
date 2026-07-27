#!/usr/bin/env python3
"""
Score B: same-layer neuron redundancy (context-predictability proxy).

For a target layer L, estimate for every MLP neuron how well its activation is
linearly predicted by the *other neurons at the same layer* (leave-one-out R^2),
computed in one shot from the precision matrix:

    standardize activations -> correlation matrix R (+ ridge)
    P = (R + lambda*I)^{-1}
    R2_i = 1 - 1 / P_ii

High R2 => redundant (the co-active population already encodes it) => predicted
(deviation hypothesis) to FAIL end-of-training endogenous eval. Low R2 => unique
=> predicted to succeed. Only SAME-LAYER siblings are used on purpose: upstream
layers determine the neuron trivially and downstream layers carry its propagated
value, both of which would inflate R2 and destroy the range.

Activations are collected from the base model over the introspection TRAINING
texts. Dead / rarely-firing neurons are dropped before inverting.

Outputs: per-neuron scores + covariates (std, firing rate, mean), an R2
histogram, and a variance/firing-matched stratified candidate set spanning the
R2 range (ready to drive a per-neuron training sweep).

Example
-------
python score_neuron_redundancy.py --config experiment_configs/.../<run>.json \
    --layer 10 --positions all --per-bin 4
"""
from __future__ import annotations

import os
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.introspector.config import ExperimentConfig


# ======================================================================
# Pure math (unit-tested)
# ======================================================================
def redundancy_from_covariance(Sigma, ridge):
    """
    Leave-one-out R^2 of each variable regressed on all the others, from the
    covariance matrix, via the precision-matrix identity.

    Sigma: [p, p] covariance. Returns R2: [p].
      R2_i = 1 - 1 / (Sigma_ii * P_ii),  P = (corr + ridge*I)^{-1}
    (standardizing to the correlation matrix so Sigma_ii -> 1).
    """
    std = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
    corr = Sigma / np.outer(std, std)
    P = np.linalg.inv(corr + ridge * np.eye(corr.shape[0]))
    Pii = np.diag(P)
    return 1.0 - 1.0 / Pii


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--layer", type=int, default=None,
                   help="Layer to score (default: config.train_signals[0].layer).")
    p.add_argument("--positions", choices=["all", "introspected"], default="all",
                   help="'all' = every non-pad token in the training texts (robust, more samples); "
                        "'introspected' = only each example's token_position.")
    p.add_argument("--n-texts", type=int, default=None,
                   help="Cap on number of training texts to process (default: all).")
    p.add_argument("--min-firing-rate", type=float, default=0.01,
                   help="Drop neurons active (|act|>fire-thresh) in fewer than this fraction of tokens.")
    p.add_argument("--fire-thresh", type=float, default=1e-4)
    p.add_argument("--ridge", type=float, default=1e-2,
                   help="Ridge added to the correlation matrix before inversion (mandatory: the layer "
                        "covariance is overcomplete/ill-conditioned).")
    p.add_argument("--per-bin", type=int, default=4,
                   help="Neurons to select per R2 decile for the stratified candidate set.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


# ======================================================================
# Activation collection (same-layer act_fn, base model, forward hook)
# ======================================================================
@torch.no_grad()
def collect_layer_stats(model, tokenizer, texts, token_positions, layer, positions_mode,
                        max_seq_length, batch_size, fire_thresh):
    """
    Stream the base model over `texts`, hook layer-L mlp.act_fn, and accumulate
    sufficient statistics for the same-layer activation covariance:
      n, s = sum_t x_t, G = sum_t x_t x_t^T, fire = sum_t 1[|x_t| > fire_thresh].
    Returns (n, s, G, fire) with G,s,fire on CPU float64.
    """
    device = model.device
    act_fn = model.model.layers[layer].mlp.act_fn
    captured = {}

    def hook(_m, _inp, out):
        captured["h"] = out  # [b, seq, intermediate]
    handle = act_fn.register_forward_hook(hook)

    p = None  # intermediate size, discovered on first batch
    n = 0
    s = G = fire = None
    order = list(range(len(texts)))
    try:
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch_texts = [texts[i] for i in idx]
            enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_seq_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            h = captured["h"].float()                       # [b, seq, p]

            if positions_mode == "all":
                mask = enc["attention_mask"].bool()         # [b, seq]
                X = h[mask]                                  # [num_tokens, p]
            else:  # introspected: one position per example
                sel = []
                for bi, i in enumerate(idx):
                    pos = int(token_positions[i])
                    pos = max(0, min(pos, h.shape[1] - 1))
                    sel.append(h[bi, pos])
                X = torch.stack(sel, 0)                      # [b, p]

            if p is None:
                p = X.shape[1]
                s = torch.zeros(p, dtype=torch.float64, device=device)
                G = torch.zeros(p, p, dtype=torch.float64, device=device)
                fire = torch.zeros(p, dtype=torch.float64, device=device)
            Xd = X.double()
            n += X.shape[0]
            s += Xd.sum(0)
            G += Xd.T @ Xd
            fire += (X.abs() > fire_thresh).double().sum(0)
    finally:
        handle.remove()

    return n, s.cpu(), G.cpu(), fire.cpu()


def stratified_select(df_kept, per_bin, n_bins=10):
    """Bin kept neurons by R2 into deciles; per bin pick `per_bin` neurons whose
    std is closest to the bin's median std (control the scale confound)."""
    df = df_kept.dropna(subset=["R2"]).copy()
    if df.empty:
        return df
    df["bin"] = pd.qcut(df["R2"].rank(method="first"), q=min(n_bins, len(df)), labels=False)
    picks = []
    for b, g in df.groupby("bin"):
        med = g["std"].median()
        picks.append(g.iloc[(g["std"] - med).abs().argsort()[:per_bin]])
    return pd.concat(picks).sort_values("R2").reset_index(drop=True)


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(args.config.read_text())
    layer = args.layer if args.layer is not None else config.train_signals[0].layer
    max_seq_length = args.max_seq_length or config.training_config.max_seq_length
    out_dir = args.output_dir or (Path(config.output_dir) / config.experiment_name / f"redundancy_L{layer}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, device_map="auto", local_files_only=True,
        torch_dtype=torch.bfloat16 if config.training_config.bf16 else torch.float32,
    )
    model.eval()

    train_raw = load_from_disk(str(Path(config.cache_dir) / config.experiment_name / "train"))
    texts = list(train_raw["text"])
    token_positions = list(train_raw["token_position"])
    if args.n_texts:
        texts, token_positions = texts[:args.n_texts], token_positions[:args.n_texts]
    print(f"[redundancy] layer={layer} positions={args.positions} n_texts={len(texts)} "
          f"model={config.model_name}")

    n, s, G, fire = collect_layer_stats(
        model, tokenizer, texts, token_positions, layer, args.positions,
        max_seq_length, args.batch_size, args.fire_thresh)
    p = s.shape[0]
    print(f"[redundancy] collected n={n} token-activations over {p} neurons")
    if n < 2 * p:
        print(f"[redundancy] WARNING: n={n} < 2*p={2*p}; covariance is under-sampled. "
              f"Use --positions all and/or more texts; results rely heavily on --ridge.")

    s, G, fire = s.numpy(), G.numpy(), fire.numpy()
    mean = s / n
    Sigma = G / n - np.outer(mean, mean)
    std = np.sqrt(np.clip(np.diag(Sigma), 0.0, None))
    firing_rate = fire / n

    # drop dead / rarely-firing neurons before inverting (they make corr singular)
    keep = (firing_rate >= args.min_firing_rate) & (std > 1e-8)
    kept_idx = np.where(keep)[0]
    print(f"[redundancy] keeping {len(kept_idx)}/{p} neurons "
          f"(firing_rate>={args.min_firing_rate}, std>0)")

    R2 = np.full(p, np.nan)
    if len(kept_idx) >= 2:
        R2[kept_idx] = redundancy_from_covariance(Sigma[np.ix_(kept_idx, kept_idx)], args.ridge)

    df = pd.DataFrame({"neuron": np.arange(p), "R2": R2, "std": std,
                       "firing_rate": firing_rate, "mean": mean, "kept": keep})
    df.to_csv(out_dir / "neuron_redundancy.csv", index=False)

    kept_df = df[df["kept"]].copy()
    cand = stratified_select(kept_df, args.per_bin)
    cand.to_csv(out_dir / "candidate_neurons.csv", index=False)

    # R2 histogram over kept neurons
    plt.figure(figsize=(7, 4))
    plt.hist(kept_df["R2"].dropna(), bins=40)
    plt.xlabel("same-layer redundancy R2"); plt.ylabel("neuron count")
    plt.title(f"Layer {layer} neuron redundancy (n_tokens={n}, ridge={args.ridge})")
    plt.tight_layout()
    plt.savefig(out_dir / "redundancy_hist.png", dpi=150)
    plt.close()

    with open(out_dir / "redundancy_meta.json", "w") as f:
        json.dump({"layer": layer, "n_tokens": int(n), "n_neurons": int(p),
                   "n_kept": int(len(kept_idx)), "ridge": args.ridge,
                   "positions": args.positions,
                   "R2_quantiles": {q: float(np.nanquantile(R2, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}},
                  f, indent=2)
    print(f"[redundancy] R2 kept quantiles: "
          f"{ {q: round(float(np.nanquantile(R2, q)),3) for q in (0.1,0.5,0.9)} }")
    print(f"[redundancy] wrote scores + {len(cand)} candidate neurons to {out_dir}")


if __name__ == "__main__":
    main()
