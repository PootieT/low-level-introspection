#!/usr/bin/env python3
"""
Activation statistics of the SFT grid's neurons in the base model: how often they
fire, how strongly, and how heavy-tailed they are. Inputs to the per-neuron feature
regression (plotters/fit_sft_feature_regression.py), alongside redundancy and context
dependence.

One full-context forward per cached 32-token window; every position except BOS.
The hooked quantity is act_fn(gate_proj(x)) -- the "neuron activation" the
introspection task reports -- so for SiLU "firing" means a positive activation
(negative values are bounded below by about -0.28).

Per neuron:
  mean, std, mean_abs        location and scale
  frac_positive              fraction of tokens with activation > 0 (firing rate)
  frac_strong                fraction of tokens above 25% of the neuron's own 99th
                             percentile: a scale-free sparsity -- how often it fires
                             near its typical peak
  mean_pos                   mean activation when positive (firing magnitude)
  p99                        99th percentile (typical peak)
  skew, kurtosis             shape (excess kurtosis; heavy tails = rare large firing)

Output: <output-dir>/activation_stats.csv (default
outputs/plots/mlp_endo_sweep_layer_token/activation_stats/).

Usage
  python mechanistic_analysis/score_neuron_activation_stats.py --dataset-dir <cache dir> [--n-texts 10000]
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, skew
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_neuron_context_dependence import grid_neurons, load_windows, Capture, run

ROOT = Path(__file__).resolve().parents[1]
STRONG_FRAC = 0.25


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--neurons", default=None, help="Default: the SFT grid's neurons.")
    p.add_argument("--n-texts", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "outputs/plots/mlp_endo_sweep_layer_token/activation_stats")
    return p.parse_args()


def neuron_stats(a: np.ndarray) -> dict:
    p99 = float(np.percentile(a, 99))
    pos = a[a > 0]
    return dict(mean=a.mean(), std=a.std(), mean_abs=np.abs(a).mean(),
                frac_positive=(a > 0).mean(),
                frac_strong=(a > STRONG_FRAC * p99).mean() if p99 > 0 else 0.0,
                mean_pos=pos.mean() if len(pos) else 0.0, p99=p99,
                skew=skew(a), kurtosis=kurtosis(a))


def main():
    args = parse_args()
    neurons = grid_neurons(args.neurons)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    layers = sorted({L for L, _ in neurons})
    cols = {L: sorted({i for l2, i in neurons if l2 == L}) for L in layers}
    cap = Capture(model, {L: torch.as_tensor(cols[L], device=device) for L in layers})
    windows = np.array(load_windows(args.dataset_dir, args.split, args.n_texts, args.seed))
    try:
        acts = run(model, cap, [(torch.as_tensor(windows[s:s + args.batch_size]), None)
                                for s in range(0, len(windows), args.batch_size)], device)
    finally:
        cap.remove()
    rows = []
    for L in layers:
        A = acts[L][:, 1:].reshape(-1, len(cols[L]))         # drop BOS
        for k, i in enumerate(cols[L]):
            rows.append(dict(neuron=f"N{L}.{i}", layer=L, index=i, **neuron_stats(A[:, k])))
    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "activation_stats.csv", index=False)
    (args.output_dir / "activation_stats_meta.json").write_text(json.dumps(dict(
        dataset_dir=str(args.dataset_dir), split=args.split, model=args.model,
        n_windows=len(windows), n_samples_per_neuron=int(windows.shape[0] * (windows.shape[1] - 1)),
        strong_frac=STRONG_FRAC), indent=2))
    print(df.groupby("layer")[["frac_positive", "frac_strong", "mean_pos", "kurtosis"]]
          .mean().round(3).to_string())
    print(f"[stats] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
