#!/usr/bin/env python3
"""Scan post-softmax attention weights over the FULL (layer, head, source offset) grid.

Stage 1 of the attention-weight experiment: pick the (layer, head, source offset) cells
worth training on, instead of sampling heads at random as the attention-logit sweep did.
That sweep came out flat -- mean endogenous corr 0.894 with no variation across layer or
offset -- partly because a randomly chosen cell is usually an uninformative one. Here
every cell is measured first and three are chosen by hand.

WHY THIS IS NOT A cache_signals.py CONFIG. That path builds one port per signal spec, so
16 layers x 32 heads x 10 offsets would need 5120 ports and 5120 gathers per batch. An
AttentionRowTap reads a whole destination row instead, so the entire grid comes out of
ONE forward per batch. Nothing is written to data/; this only reads.

The model is loaded with the same attention implementation caching and training use, so
the cells this ranks are the cells those will later see (that implementation is
eager-math and differs from sdpa at ~1e-3 in bf16).

OUTPUTS (under --out, default plots/attn_weight_scan/)
  attn_weight_scan_raw.npy           [N, n_layers, n_heads, n_offsets] float32
  attn_weight_scan_by_layer_head.csv one row per (layer, head), pooled over ALL offsets
                                     and documents, sorted by `std` descending
  attn_weight_scan_by_cell.csv       one row per (layer, head, src_offset)
  spread_layer_head.png              layer x head heatmap of `std`
  spread_by_offset.png               `std` against offset, one line per layer

RANKING BY `std`, WITH `cv` ALONGSIDE. The training loss (_var_norm_mse) and the headline
metric (Pearson corr) are both scale invariant, so what decides whether a cell is
learnable at all is its absolute spread. Expect that to favour cells near the BOS
attention sink, whose mean is large: that is a finding, not a bug, and `cv` (= std/mean)
is in the same table so a reader can apply the scale-free view at the same time.

VETOES, applied to the RECOMMENDATION only -- every cell stays in the CSVs:
  p95 < --dead-p95      no appreciable attention mass, so neither an endogenous nor an
                        interventional result about it would mean anything
  offset == BOS         a separate regime (the sink), not the far end of the axis. For
                        the logit signal BOS was a problem because its variance nearly
                        vanished; for the weight it is the opposite, its mean dominates.
  layer 0               the text-only ceiling control, kept in the tables for comparison

Run (from the repo root):
    python scan_attention_weight_spread.py --n-docs 4000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.introspector.models.attention_logits import (
    ATTN_IMPL_NAME, AttentionRowTap, installed_ports,
)

# The same ten offsets the attention-logit sweep used, so the two experiments share an
# x axis. 0 is the destination attending to itself; the deepest is BOS at length 32.
DEFAULT_OFFSETS = (0, -1, -3, -7, -11, -15, -19, -23, -27, -31)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--n-docs", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=32,
                    help="documents are pinned to EXACTLY this many tokens (BOS included)")
    ap.add_argument("--offsets", type=int, nargs="+", default=list(DEFAULT_OFFSETS))
    ap.add_argument("--dataset", default="allenai/c4")
    ap.add_argument("--dataset-config", default="en")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--out", default="plots/attn_weight_scan")
    ap.add_argument("--dead-p95", type=float, default=1e-3,
                    help="veto a cell from the recommendation if its 95th percentile "
                         "weight is below this")
    ap.add_argument("--low-layers", type=int, nargs=2, default=[1, 4])
    ap.add_argument("--mid-layers", type=int, nargs=2, default=[6, 9])
    ap.add_argument("--high-layers", type=int, nargs=2, default=[11, 14])
    ap.add_argument("--fp32", action="store_true",
                    help="scan in fp32 instead of bf16. bf16 is the default because it "
                         "is what training uses; fp32 says how much of the spread is real")
    return ap.parse_args()


def stream_batches(args, tokenizer):
    """Yield right-padded batches of documents that are EXACTLY seq_len tokens long.

    Exactly, not at least. A post-softmax weight is normalized over the whole causal row,
    so two documents of different lengths do not carry the same quantity at the same
    offset -- the denominator differs. (An attention LOGIT does not have this problem,
    which is why the logit sweep could have been sloppier here and was not.) Pinning the
    length also makes the deepest offset land on BOS for every document rather than for
    most of them.
    """
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    buf, seen, kept = [], 0, 0
    for ex in ds:
        seen += 1
        ids = tokenizer(ex[args.text_field], add_special_tokens=True)["input_ids"]
        if len(ids) < args.seq_len:
            continue
        buf.append(ids[:args.seq_len])
        kept += 1
        if len(buf) == args.batch_size:
            yield torch.tensor(buf, dtype=torch.long), seen, kept
            buf = []
        if kept >= args.n_docs:
            break
    if buf:
        yield torch.tensor(buf, dtype=torch.long), seen, kept


def summarize(values, offsets_axis=None):
    """Per-cell statistics from a [N, ...] array of weights along axis 0."""
    v = values.astype(np.float64)
    mean = v.mean(0)
    std = v.std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mean > 0, std / mean, np.nan)
    # log-odds moments: the space the interventional sampler works in, so a cell with a
    # healthy std_logit is one where an injection is meaningful rather than saturating
    eps = 1e-6
    lv = np.log(np.clip(v, eps, 1 - eps) / (1 - np.clip(v, eps, 1 - eps)))
    # share of total variance contributed by the top 1% of documents: the direct
    # measurement of the skew that makes Pearson corr tail-dominated for this signal
    dev2 = (v - mean) ** 2
    k = max(1, int(round(0.01 * v.shape[0])))
    top = np.sort(dev2, axis=0)[-k:].sum(0)
    total = dev2.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        top_share = np.where(total > 0, top / total, np.nan)
    return {
        "n": v.shape[0],
        "mean": mean, "std": std, "cv": cv,
        "mean_logit": lv.mean(0), "std_logit": lv.std(0, ddof=1),
        "p05": np.percentile(v, 5, axis=0), "p25": np.percentile(v, 25, axis=0),
        "p50": np.percentile(v, 50, axis=0), "p75": np.percentile(v, 75, axis=0),
        "p95": np.percentile(v, 95, axis=0),
        "frac_lt_1e-3": (v < 1e-3).mean(0),
        "top1pct_var_share": top_share,
    }


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.fp32 else torch.bfloat16

    offsets = list(args.offsets)
    assert all(o <= 0 for o in offsets), f"offsets must be <= 0 (causal), got {offsets}"
    assert min(offsets) > -args.seq_len, (
        f"offset {min(offsets)} needs more than {args.seq_len} tokens")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"[scan] loading {args.model} with attn_implementation={ATTN_IMPL_NAME} "
          f"({dtype}) on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation=ATTN_IMPL_NAME, dtype=dtype, device_map=device).eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"[scan] grid = {n_layers} layers x {n_heads} heads x {len(offsets)} offsets "
          f"= {n_layers * n_heads * len(offsets)} cells, from ONE forward per batch")

    chunks = []
    seen = kept = 0
    for batch_ids, seen, kept in stream_batches(args, tokenizer):
        batch_ids = batch_ids.to(device)
        bs = batch_ids.shape[0]
        # every document is exactly seq_len tokens, so there is no padding and the
        # destination is the last position for all of them
        dst = torch.full((bs,), args.seq_len - 1, dtype=torch.long, device=device)
        taps = {L: AttentionRowTap(dst=dst, src_offsets=offsets) for L in range(n_layers)}
        with torch.no_grad(), installed_ports(model, taps, site="weight"):
            model(input_ids=batch_ids,
                  attention_mask=torch.ones_like(batch_ids))
        # [n_layers, B, n_heads, n_offsets] -> [B, n_layers, n_heads, n_offsets]
        stacked = torch.stack([taps[L].read for L in range(n_layers)], dim=0)
        chunks.append(stacked.permute(1, 0, 2, 3).cpu().numpy().astype(np.float32))
        if kept % (args.batch_size * 10) == 0:
            print(f"[scan] {kept}/{args.n_docs} documents ({seen} streamed)")

    values = np.concatenate(chunks, axis=0)           # [N, n_layers, n_heads, n_offsets]
    N = values.shape[0]
    print(f"[scan] collected {values.shape} from {seen} streamed documents "
          f"({100.0 * kept / max(seen, 1):.1f}% were long enough)")
    np.save(out / "attn_weight_scan_raw.npy", values)

    # ---- per (layer, head, offset) ----
    cell = summarize(values.reshape(N, -1))
    idx = [(L, h, o) for L in range(n_layers) for h in range(n_heads) for o in offsets]
    cell_df = pd.DataFrame({
        "layer": [i[0] for i in idx], "head": [i[1] for i in idx],
        "src_offset": [i[2] for i in idx],
        **{k: (v if np.isscalar(v) else np.asarray(v).ravel()) for k, v in cell.items()},
    })
    cell_df["dead"] = cell_df["p95"] < args.dead_p95
    cell_df = cell_df.sort_values("std", ascending=False)
    cell_df.to_csv(out / "attn_weight_scan_by_cell.csv", index=False)

    # ---- per (layer, head), pooled over every offset AND document ----
    # "the mean attention weight of this (layer, head), averaged over all data and all
    # offsets", which is what the experiment plan asks for.
    pooled = values.transpose(0, 3, 1, 2).reshape(N * len(offsets), n_layers * n_heads)
    lh = summarize(pooled)
    lh_df = pd.DataFrame({
        "layer": np.repeat(np.arange(n_layers), n_heads),
        "head": np.tile(np.arange(n_heads), n_layers),
        **{k: (v if np.isscalar(v) else np.asarray(v).ravel()) for k, v in lh.items()},
    })
    lh_df["dead"] = lh_df["p95"] < args.dead_p95
    lh_df = lh_df.sort_values("std", ascending=False)
    lh_df.to_csv(out / "attn_weight_scan_by_layer_head.csv", index=False)

    print(f"\n[scan] top 15 (layer, head) by std (pooled over all offsets):")
    cols = ["layer", "head", "mean", "std", "cv", "std_logit", "p95", "top1pct_var_share"]
    print(lh_df[cols].head(15).to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # ---- figures ----
    grid = lh_df.pivot(index="layer", columns="head", values="std").sort_index()
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(grid.values, aspect="auto", cmap="viridis")
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_title("std of post-softmax attention weight (pooled over offsets)")
    fig.colorbar(im, ax=ax, label="std")
    fig.tight_layout(); fig.savefig(out / "spread_layer_head.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    by_lo = cell_df.groupby(["layer", "src_offset"])["std"].mean().reset_index()
    for L in range(n_layers):
        s = by_lo[by_lo["layer"] == L].sort_values("src_offset")
        ax.plot(s["src_offset"], s["std"], marker="o", ms=3,
                label=f"L{L}", color=plt.cm.viridis(L / max(n_layers - 1, 1)))
    ax.set_xlabel("source offset from destination (0 = self, min = BOS)")
    ax.set_ylabel("std of attention weight (mean over heads)")
    ax.set_yscale("log")
    ax.set_title("spread by offset and layer")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "spread_by_offset.png", dpi=150); plt.close(fig)

    # ---- recommendation ----
    bos = min(offsets)
    pick_pool = cell_df[(~cell_df["dead"]) & (cell_df["src_offset"] != bos)
                        & (cell_df["layer"] != 0)]
    bands = {"low": args.low_layers, "mid": args.mid_layers, "upper": args.high_layers}
    picks = {}
    print(f"\n[scan] RECOMMENDED CELLS (ranked by std; layer 0, BOS offset {bos} and "
          f"cells with p95 < {args.dead_p95:g} excluded)")
    for name, (lo, hi) in bands.items():
        band = pick_pool[(pick_pool["layer"] >= lo) & (pick_pool["layer"] <= hi)]
        if not len(band):
            print(f"  {name:5s} layers {lo}-{hi}: NO eligible cell")
            continue
        # Bracket indexing, never attribute access: `r.head` is pandas' Series.head
        # METHOD, not the column, and int() of it raises somewhere far from the cause.
        # `r.mean`, `r.std` and `r.cv` shadow methods too.
        r = band.iloc[0]
        layer, head, off = int(r["layer"]), int(r["head"]), int(r["src_offset"])
        sig = f"P{layer}.{head}@{off}"
        picks[name] = dict(signal=sig, layer=layer, head=head, src_offset=off,
                           mean=float(r["mean"]), std=float(r["std"]), cv=float(r["cv"]),
                           std_logit=float(r["std_logit"]), p95=float(r["p95"]),
                           top1pct_var_share=float(r["top1pct_var_share"]))
        print(f"  {name:5s} layers {lo}-{hi}: {sig:16s} mean={r['mean']:.4g} "
              f"std={r['std']:.4g} cv={r['cv']:.3g} std_logit={r['std_logit']:.3g} "
              f"p95={r['p95']:.4g} top1%var={r['top1pct_var_share']:.3g}")

    meta = dict(model=args.model, n_docs=int(N), docs_streamed=int(seen),
                seq_len=args.seq_len, offsets=offsets, dtype=str(dtype),
                attn_implementation=ATTN_IMPL_NAME, dead_p95=args.dead_p95,
                n_layers=n_layers, n_heads=n_heads, picks=picks)
    (out / "scan_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[scan] wrote CSVs, figures and scan_meta.json under {out}")


if __name__ == "__main__":
    main()
