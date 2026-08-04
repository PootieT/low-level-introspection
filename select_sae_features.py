#!/usr/bin/env python3
"""
Select SAE features worth introspecting on, by a streaming density pass over a
large corpus. This is the FEATURE-SELECTION artifact -- distinct from the
per-experiment training cache (cache_signals.py). It never materializes a
[tokens x d_sae] matrix; it accumulates O(d_sae) counters, so >=1M tokens is a
compute cost, not a memory cost.

Per feature it records (all BOS-excluded, matching Neuronpedia's "ignore BOS"):
  - density            : fraction of (non-bos, non-pad) tokens where the feature fires
  - active_mean/std/max: value distribution WHEN active (for the sparse sampler /
                         signal_ranges sanity, and to eyeball dynamic range)
  - pos_density        : fraction of SEQUENCES where the feature fires at the fixed
                         readout position (--fixed-position). This is what governs
                         how many positive examples a fixed-position DETECTION cache
                         will contain -- the number that actually matters for the
                         first single-feature run.

Selection keeps features with min_density <= density <= max_density
(max_density=0.01 == Neuronpedia's default "hide >1% density"), writes a stats
parquet, and prints ranked candidates with a projected positive count.

Run on the cluster:

  python select_sae_features.py \
      --model google/gemma-2-2b \
      --sae-release gemma-scope-2b-pt-res-canonical --layer 9 --width 16k \
      --n-tokens 1000000 --max-seq-length 128 --fixed-position -1 \
      --cache-size 20000 --out data/sae_stats/gemma2-2b_res9_16k.parquet
"""
from __future__ import annotations

import argparse
import operator
from pathlib import Path

import torch
import pandas as pd
from datasets import load_dataset
from nnsight import LanguageModel

from src.introspector.models.signal_paths import resolve_sae_hook_path
from src.introspector.signals.sae import load_sae


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--width", default="16k")
    p.add_argument("--site", default="res", choices=["res", "mlp", "attn"])
    p.add_argument("--dataset", default="allenai/c4")
    p.add_argument("--dataset-config", default="en")
    p.add_argument("--split", default="train")
    p.add_argument("--n-tokens", type=int, default=1_000_000,
                   help="Approx number of non-bos tokens to accumulate over.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--fixed-position", type=int, default=-1,
                   help="Readout position for the first fixed-position detection run; "
                        "pos_density is measured here.")
    p.add_argument("--min-density", type=float, default=1e-3)
    p.add_argument("--max-density", type=float, default=0.01)
    p.add_argument("--cache-size", type=int, default=20000,
                   help="Planned training-cache size; used only to project positive counts.")
    p.add_argument("--top-k", type=int, default=40, help="How many candidates to print.")
    p.add_argument("--out", type=Path, default=Path("data/sae_stats/feature_stats.parquet"))
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LanguageModel(args.model, device_map="auto")
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bundle = load_sae(release=args.sae_release, site=args.site, width=args.width,
                      layer=args.layer, device=device, dtype=torch.float32)
    hook_path = resolve_sae_hook_path(model, args.layer, site=args.site, path_to_output=True)
    d_sae = bundle.d_sae
    print(f"[select] hook={hook_path} d_sae={d_sae} target≈{args.n_tokens} tokens")

    # O(d_sae) accumulators (float64 to stay accurate over ~1M tokens)
    active_count = torch.zeros(d_sae, dtype=torch.float64, device=device)
    active_sum = torch.zeros(d_sae, dtype=torch.float64, device=device)
    active_sqsum = torch.zeros(d_sae, dtype=torch.float64, device=device)
    active_max = torch.zeros(d_sae, dtype=torch.float64, device=device)
    tok_count = 0
    # fixed-position accumulators (per sequence)
    pos_active_count = torch.zeros(d_sae, dtype=torch.float64, device=device)
    seq_count = 0

    getter = operator.attrgetter(hook_path)
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    buf = []
    for ex in ds:
        buf.append(ex["text"])
        if len(buf) < args.batch_size:
            continue
        enc = tok(buf, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_seq_length, padding_side="right")
        buf = []
        with model.trace(enc):
            resid = getter(model).save()                    # [B, T, D]
        resid = (resid.value if hasattr(resid, "value") else resid).float().to(device)
        feats = bundle.encode(resid)                        # [B, T, d_sae]

        am = enc["attention_mask"].bool().to(device)
        keep = am & (enc["input_ids"].to(device) != tok.bos_token_id)   # [B, T]
        active = (feats > 0)

        km = keep.unsqueeze(-1)
        active_count += (active & km).sum(dim=(0, 1)).double()
        vals = feats.masked_fill(~km, 0.0)
        active_sum += vals.sum(dim=(0, 1)).double()
        active_sqsum += (vals * vals).sum(dim=(0, 1)).double()
        active_max = torch.maximum(active_max, vals.amax(dim=(0, 1)).double())
        tok_count += int(keep.sum().item())

        # fixed-position (last real token per sequence, right-padded)
        lengths = am.sum(dim=1)                              # [B]
        pos = (lengths + args.fixed_position) if args.fixed_position < 0 else torch.full_like(lengths, args.fixed_position)
        pos = pos.clamp(0, feats.shape[1] - 1)
        bidx = torch.arange(feats.shape[0], device=device)
        feats_pos = feats[bidx, pos]                        # [B, d_sae]
        # only count sequences whose fixed position is a real, non-bos token
        pos_ok = keep[bidx, pos]                            # [B]
        pos_active_count += ((feats_pos > 0) & pos_ok.unsqueeze(-1)).sum(0).double()
        seq_count += int(pos_ok.sum().item())

        if tok_count >= args.n_tokens:
            break

    density = (active_count / max(tok_count, 1)).cpu()
    active_mean = (active_sum / active_count.clamp(min=1)).cpu()
    active_var = (active_sqsum / active_count.clamp(min=1)).cpu() - active_mean ** 2
    active_std = active_var.clamp(min=0).sqrt()
    pos_density = (pos_active_count / max(seq_count, 1)).cpu()

    df = pd.DataFrame({
        "feature": range(d_sae),
        "density": density.numpy(),
        "pos_density": pos_density.numpy(),
        "active_mean": active_mean.numpy(),
        "active_std": active_std.numpy(),
        "active_max": active_max.cpu().numpy(),
        "n_active": active_count.cpu().numpy().astype("int64"),
    })
    df["selected"] = (df["density"] >= args.min_density) & (df["density"] <= args.max_density)
    # projected positives if we build a fixed-position detection cache of --cache-size
    df["proj_pos_at_fixed_pos"] = (df["pos_density"] * args.cache_size).round().astype("int64")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"[select] wrote {len(df)} rows -> {args.out}  "
          f"(tokens={tok_count}, seqs={seq_count}, selected={int(df['selected'].sum())})")

    # rank selected candidates by projected positives at the fixed position (we
    # want enough positives to actually train a detector), then by dynamic range
    cand = df[df["selected"]].sort_values(
        ["proj_pos_at_fixed_pos", "active_max"], ascending=False).head(args.top_k)
    print(f"\n=== TOP {len(cand)} SELECTED FEATURES (density in [{args.min_density},{args.max_density}]) ===")
    print("  pick one with proj_pos_at_fixed_pos comfortably > a few hundred, then")
    print("  confirm its concept on Neuronpedia before caching.")
    with pd.option_context("display.float_format", lambda v: f"{v:.4g}"):
        print(cand[["feature", "density", "pos_density", "proj_pos_at_fixed_pos",
                    "active_mean", "active_max"]].to_string(index=False))


if __name__ == "__main__":
    main()
