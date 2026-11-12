#!/usr/bin/env python3
"""Step A1b: activation percentiles of every in-band feature, over the SAME token pass as the
A1 firing statistics (data_prep/select_sae_features.py: C4 train, streamed in order, 128-token texts, BOS
and padding excluded, the first --n-tokens tokens). The steering step scales its strengths by one
of these (config steering.magnitude, e.g. act_q99) rather than the single peak value, which is
an outlier.

n_active and max are recomputed here and checked against the A1 stats file: they must match,
which shows the two passes saw the same tokens.

    python data_prep/sae_activation_quantiles.py --layer 5 \
        --stats data/sae_stats/gemma2-2b_res5_pos-1_16k.parquet \
        --out data/sae_stats/gemma2-2b_res5_act_quantiles_16k.parquet
"""
import argparse, operator, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from datasets import load_dataset
from nnsight import LanguageModel

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from src.introspector.models.signal_paths import resolve_sae_hook_path
from src.introspector.signals.sae import load_sae

QUANTILES = {"act_q50": 0.50, "act_q90": 0.90, "act_q95": 0.95, "act_q99": 0.99, "act_q999": 0.999}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--stats", type=Path, required=True, help="A1 stats parquet of this layer")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    p.add_argument("--width", default="16k")
    p.add_argument("--n-tokens", type=int, default=2_000_000)   # as in precompute_sae_feature_stats.sh
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--density-band", type=float, nargs=2, default=[1e-3, 1e-2])
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stats = pd.read_parquet(args.stats)
    lo, hi = args.density_band
    feats_in_band = stats.feature[(stats.density >= lo) & (stats.density <= hi)].to_numpy()
    sub = torch.as_tensor(feats_in_band, device=device)
    print(f"[quantiles] layer {args.layer}: {len(feats_in_band)} in-band features")

    model = LanguageModel(args.model, device_map="auto")
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bundle = load_sae(release=args.sae_release, site="res", width=args.width, layer=args.layer,
                      device=device, dtype=torch.float32)
    getter = operator.attrgetter(resolve_sae_hook_path(model, args.layer, site="res", path_to_output=True))

    cols, vals = [], []   # (column in `sub`, activation) of every active entry
    tok_count, buf = 0, []
    bar = tqdm.tqdm(total=args.n_tokens)
    for ex in load_dataset("allenai/c4", "en", split="train", streaming=True):
        buf.append(ex["text"])
        if len(buf) < args.batch_size:
            continue
        enc = tok(buf, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_seq_length, padding_side="right")
        buf = []
        with model.trace(enc):
            resid = getter(model).save()
        resid = (resid.value if hasattr(resid, "value") else resid).float().to(device)
        f = bundle.encode(resid)[..., sub]                          # [B, T, S]
        keep = enc["attention_mask"].bool().to(device) & (enc["input_ids"].to(device) != tok.bos_token_id)
        f = f[keep]                                                 # [N, S]
        r, c = torch.nonzero(f > 0, as_tuple=True)
        cols.append(c.cpu().numpy().astype(np.int32))
        vals.append(f[r, c].cpu().numpy().astype(np.float32))
        n = int(keep.sum().item())
        tok_count += n
        bar.update(n)
        if tok_count >= args.n_tokens:
            break

    cols, vals = np.concatenate(cols), np.concatenate(vals)
    order = np.argsort(cols, kind="stable")
    cols, vals = cols[order], vals[order]
    bounds = np.searchsorted(cols, np.arange(len(feats_in_band) + 1))
    rows = []
    for j, f_idx in enumerate(feats_in_band):
        v = vals[bounds[j]:bounds[j + 1]]
        row = {"feature": int(f_idx), "n_active": len(v), "act_max": float(v.max()) if len(v) else np.nan}
        row.update({k: float(np.quantile(v, q)) if len(v) else np.nan for k, q in QUANTILES.items()})
        rows.append(row)
    out = pd.DataFrame(rows)

    # same tokens as A1: per-feature active counts and maxima must agree
    # (GPU numerics can flip a few activations that sit right at the JumpReLU threshold, so
    # counts may differ by a handful; a different token pass would differ by far more)
    ref = stats.set_index("feature").loc[out.feature]
    dn = np.abs(ref.n_active.to_numpy() - out.n_active.to_numpy())
    rel = dn / np.maximum(ref.n_active.to_numpy(), 1)
    max_err = float(np.nanmax(np.abs(ref.active_max.to_numpy() - out.act_max.to_numpy())))
    print(f"[quantiles] tokens {tok_count}; n_active differs for {int((dn > 0).sum())} features "
          f"(max {int(dn.max())} tokens, max rel {rel.max():.1e}); max |dmax| {max_err:.2e}")
    if rel.max() > 1e-3 or max_err > 1e-2:
        raise SystemExit("token pass differs from the A1 stats; not writing")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"[quantiles] wrote {args.out}")
    print(out[["n_active", "act_q50", "act_q99", "act_max"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
