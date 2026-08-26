#!/usr/bin/env python3
"""
Probe: does the TRAINING-time tokenization reproduce the CACHED activations?

No training, no adapters -- just the base model, the cached dataset, and a forward
hook. It exists because measure_activation_drift (train_introspector_joint.py)
found that the adapters-OFF model does NOT reproduce the cached column: the best
offset was -1 (not the 0 that training assumes) and even there corr was only 0.80,
where an exact match should be ~1.0.

Two independent things can go wrong, and this script separates them:

  ALIGNMENT   cache_signals.py tokenizes with add_special_tokens defaulting to
              True (BOS at index 0); preprocess_example uses
              add_special_tokens=False. If that is the whole story, cached
              position p is training position p-1, and tokenizing WITH BOS makes
              offset 0 correct again. Measured by TOKEN-STRING MATCH RATE: the
              cache stores the decoded token at the introspected position, so the
              convention that reproduces those strings is the right one.

  FIDELITY    even at the correct position the activation can differ, because
              (a) no-BOS input removes Llama's attention sink and perturbs every
              downstream activation, (b) the cache stores decode(input_ids) and
              training re-tokenizes that string -- and encode(decode(x)) is not
              guaranteed to be identity for byte-level BPE, or (c) dtype (the
              cache ran the nnsight default, training runs bf16). Measured by
              corr / relative error RESTRICTED to token-matched rows, so a bad
              number cannot be blamed on misalignment.

Read the output as:
  token_match ~1.0 AND corr_matched ~1.0  -> one-line fix in preprocess_example
                                             (tokenize with BOS); caches are fine,
                                             retrain only, no re-caching.
  token_match ~1.0 BUT corr_matched < 1   -> position is right, the activation is
                                             not: check the --also-fp32 rows for
                                             dtype, else it is the BOS sink.
  token_match  < 1.0 for every condition  -> decode/re-encode does not round-trip;
                                             the cache must store input_ids and
                                             the datasets have to be rebuilt.

SELF-CHECK: the (bos=False, offset=-1) row reproduces the condition
measure_activation_drift already reported. Its corr should land near the number
you saw there (~0.80). If it does not, distrust this script before believing it.

Example
-------
python probe_cache_alignment.py \
    --config experiment_configs/<group>/<run>.json --n-examples 1024 --also-fp32
"""
from __future__ import annotations

import json
import operator
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.introspector.config import ExperimentConfig
from src.introspector.models.signal_paths import resolve_mlp_signal_path

# (add_bos, offset) pairs to test. offset is added to the CACHED token_position.
CONDITIONS = [(True, -1), (True, 0), (True, 1), (False, -1), (False, 0), (False, 1)]
# what preprocess_example + IntrospectionRegressionModel.forward currently do
TRAINING_CONVENTION = (False, 0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--n-examples", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-seq-length", type=int, default=None,
                   help="Truncation length. Default: the DATASET max_seq_length for this split, "
                        "which is what cache_signals.py truncated with (NOT training_config's).")
    p.add_argument("--also-fp32", action="store_true",
                   help="Re-run every condition with the model cast to fp32, to test whether a "
                        "residual mismatch is just bf16 rounding.")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


@torch.no_grad()
def capture(model, tokenizer, rows, layer, add_bos, offset, batch_size, max_length):
    """
    Tokenize each row's text (with/without BOS), run the base model, and read the
    target neuron at cached_position + offset. Right padding matches cache time and
    keeps positions valid as-is (causal attention ignores trailing pads).

    Returns a DataFrame with the live activation, the decoded token at the probed
    position, and a validity flag for rows whose position falls outside the text.
    """
    path = resolve_mlp_signal_path(model, layer, path_to_output=False)
    module = operator.attrgetter(path)(model)

    captured = {}

    def hook(mod, inputs, outputs):
        captured["v"] = outputs.detach()
        return None

    live, tok_str, valid = [], [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows.iloc[start:start + batch_size]
        enc = tokenizer(list(chunk["text"]), truncation=True, padding=True, max_length=max_length,
                        add_special_tokens=add_bos, return_tensors="pt", padding_side="right")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        seq_lens = enc["attention_mask"].sum(1)

        handle = module.register_forward_hook(hook)
        try:
            model(**enc)
        finally:
            handle.remove()
        acts = captured["v"]                       # [bs, seq, d_mlp]

        for i, (_, row) in enumerate(chunk.iterrows()):
            pos = int(row["token_position"]) + offset
            ok = 0 <= pos < int(seq_lens[i])
            valid.append(ok)
            if ok:
                live.append(float(acts[i, pos, int(row["signal_index"])]))
                tok_str.append(tokenizer.decode(enc["input_ids"][i, pos],
                                                clean_up_tokenization_spaces=False))
            else:
                live.append(float("nan"))
                tok_str.append(None)

    return pd.DataFrame({"live": live, "live_token": tok_str, "valid": valid}, index=rows.index)


def _corr(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(pearsonr(a, b)[0])


def score(df, cached_col="activation"):
    """Alignment (token-string match) and fidelity (agreement on matched rows)."""
    v = df[df["valid"]]
    matched = v[v["live_token"] == v["token"]]
    cached_m, live_m = matched[cached_col].to_numpy(float), matched["live"].to_numpy(float)
    err = np.abs(live_m - cached_m)
    std = float(np.std(cached_m)) if len(cached_m) else float("nan")
    # "close" = within the precision the cache was written at (3 decimals by default)
    tol = np.maximum(1e-2 * np.abs(cached_m), 1e-3)
    return {
        "n": int(len(df)),
        "n_valid": int(len(v)),
        "token_match_rate": float(len(matched) / len(v)) if len(v) else float("nan"),
        "corr_all": _corr(v["live"], v[cached_col]),
        "corr_matched": _corr(live_m, cached_m) if len(matched) else float("nan"),
        "rel_rmse_matched": float(np.sqrt(np.mean(err ** 2)) / std) if std > 1e-12 else float("nan"),
        "median_abs_err_matched": float(np.median(err)) if len(err) else float("nan"),
        "frac_close_matched": float(np.mean(err <= tol)) if len(err) else float("nan"),
    }


def verdict(df):
    """Turn the condition table into the decision: one-line fix, dtype/BOS problem,
    or rebuild the caches."""
    ok = df.dropna(subset=["token_match_rate"])
    if ok.empty:
        return "No valid conditions -- check --max-seq-length and the split."
    best = ok.sort_values(["token_match_rate", "corr_matched"], ascending=False).iloc[0]
    tag = f"bos={bool(best['add_bos'])}, offset={int(best['offset']):+d}, dtype={best['dtype']}"
    if best["token_match_rate"] < 0.99:
        return (f"NO convention aligns (best {tag}: token match "
                f"{best['token_match_rate']:.3f}). decode/re-encode does not round-trip, so the "
                f"stored text cannot recover the cached tokenization. FIX: cache input_ids in "
                f"cache_signals.py and REBUILD the datasets -- adding BOS alone will not do it.")
    if best["corr_matched"] > 0.99 and best["frac_close_matched"] > 0.95:
        return (f"ALIGNED and FAITHFUL at {tag}: token match {best['token_match_rate']:.3f}, "
                f"corr {best['corr_matched']:.4f}, {best['frac_close_matched']:.1%} of rows within "
                f"cache precision. FIX: tokenize the training input the same way "
                f"(add_special_tokens=True in preprocess_example) so cached position p maps to "
                f"training position p. Caches stay valid -- retrain only, no re-caching.")
    return (f"ALIGNED but NOT faithful at {tag}: token match {best['token_match_rate']:.3f} yet "
            f"corr {best['corr_matched']:.4f} (rel_rmse {best['rel_rmse_matched']:.3f}). The position "
            f"is right and the activation still differs. Compare the fp32 rows (--also-fp32): if they "
            f"lift, it is dtype; if not, the residual gap is the BOS attention sink or a partial "
            f"round-trip failure, and the cache should store input_ids.")


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(args.config.read_text())
    out_dir = args.output_dir or Path("outputs/probes") / config.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_cfg = {"train": config.train_dataset, "dev": config.dev_dataset, "test": config.test_dataset}[args.split]
    max_length = args.max_seq_length or ds_cfg.max_seq_length

    raw = load_from_disk(str(config.dataset_cache_dir / args.split))
    rows = raw.to_pandas().head(args.n_examples).reset_index(drop=True)
    layers = sorted(rows["signal_layer"].unique().tolist())
    print(f"[probe] {len(rows)} rows from {config.dataset_cache_dir / args.split} | "
          f"layers={layers} | truncation max_length={max_length}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # NOTE: deliberately NO introspection separator and NO resize_token_embeddings.
    # We feed the raw text exactly as cache_signals.py did; the prompt that follows
    # it at training time cannot affect these positions under causal attention.

    results, per_row = [], {}
    dtypes = [("bf16", torch.bfloat16)] + ([("fp32", torch.float32)] if args.also_fp32 else [])
    for dtype_name, dtype in dtypes:
        print(f"[probe] loading {config.model_name} in {dtype_name} ...")
        model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=dtype,
                                                     device_map="auto")
        model.eval()
        for add_bos, offset in CONDITIONS:
            frames = []
            for layer in layers:
                sub = rows[rows["signal_layer"] == layer]
                frames.append(capture(model, tokenizer, sub, layer, add_bos, offset,
                                      args.batch_size, max_length))
            got = pd.concat(frames).sort_index()
            merged = rows.join(got)
            m = score(merged)
            m.update({"add_bos": add_bos, "offset": offset, "dtype": dtype_name})
            results.append(m)
            per_row[(dtype_name, add_bos, offset)] = merged
            print(f"[probe] bos={str(add_bos):5s} offset={offset:+d} dtype={dtype_name}: "
                  f"token_match={m['token_match_rate']:.4f} corr_matched={m['corr_matched']:.4f} "
                  f"corr_all={m['corr_all']:.4f} rel_rmse={m['rel_rmse_matched']:.4f} "
                  f"close={m['frac_close_matched']:.3f}")
        del model
        torch.cuda.empty_cache()

    cols = ["dtype", "add_bos", "offset", "n", "n_valid", "token_match_rate", "corr_all",
            "corr_matched", "rel_rmse_matched", "median_abs_err_matched", "frac_close_matched"]
    df = pd.DataFrame(results)[cols]
    df.to_csv(out_dir / "cache_alignment_probe.csv", index=False)
    print("\n[probe] conditions (token_match_rate answers ALIGNMENT, corr_matched answers FIDELITY):")
    print(df.to_string(index=False))

    # self-check against the number measure_activation_drift already reported
    base = df[(df.dtype == "bf16") & (~df.add_bos) & (df.offset == -1)]
    if len(base):
        print(f"\n[probe] self-check -- (bos=False, offset=-1) corr_all={float(base['corr_all'].iloc[0]):.4f}; "
              f"this should be close to the best-offset corr from the drift eval (~0.80).")

    # dump per-row detail for the training convention and for the winner
    best_key = df.sort_values(["token_match_rate", "corr_matched"], ascending=False).iloc[0]
    for tag, key in [("training_convention", ("bf16",) + TRAINING_CONVENTION),
                     ("best", (best_key["dtype"], bool(best_key["add_bos"]), int(best_key["offset"])))]:
        if key in per_row:
            d = per_row[key]
            d["token_matches"] = d["valid"] & (d["live_token"] == d["token"])
            d[["text", "token_position", "token", "live_token", "token_matches",
               "activation", "live", "valid"]].to_csv(out_dir / f"cache_alignment_rows_{tag}.csv",
                                                      index=False)

    v = verdict(df)
    print(f"\n[probe] VERDICT: {v}")
    with open(out_dir / "cache_alignment_probe.json", "w") as f:
        json.dump({"conditions": results, "verdict": v,
                   "config": str(args.config), "split": args.split,
                   "max_length": max_length, "n_examples": int(len(rows))}, f, indent=2)
    print(f"[probe] wrote {out_dir}/cache_alignment_probe.{{csv,json}} + per-row CSVs")


if __name__ == "__main__":
    main()
