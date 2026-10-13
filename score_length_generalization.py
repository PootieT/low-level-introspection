#!/usr/bin/env python3
"""
Does the readout generalize to token positions further back than it was trained on?

The introspection prompt is appended AFTER the input text, so reading a neuron at
position p means reaching back (text_len - 1 - p) tokens from where the readout
happens. Training on a narrow window near the end of the text leaves that reach
untested: a model can solve the training distribution perfectly by attending to the
recent window and never learning to address a token by what the prompt names.

This module evaluates one trained checkpoint across an (eval length x distance) grid.

WHAT THE TWO AXES SEPARATE
    distance_from_end   how far back the readout reaches. This is the quantity the
                        training window restricts, and the one RoPE has to extrapolate.
    eval length         how much text sits in front of the target. Holding distance
                        fixed and varying length isolates absolute position and
                        distractor load from the reach itself -- a 128-token document
                        and a 16-token document with the target 8 tokens from the end
                        pose the same reach and different contexts.

A single "accuracy at length 128" number conflates the two, which is why results are
binned on distance and reported per eval length.

THE FAILURE MODE THIS IS BUILT TO DISTINGUISH
A learned recency prior ("attend to the last k tokens") and a RoPE extrapolation
failure produce the same falling accuracy-vs-distance curve. They are told apart by
whether the in-distribution bins (distance <= the training window) hold up while the
out-of-distribution bins fall -- so every bin is labeled in_train_range against the
training token selector, and the two are never averaged together.

WATCH FOR: bins are populated by sampling positions uniformly over the eval range, so
near-distance bins are thin at long eval lengths. `n` is reported per bin; a bin with
a handful of examples has a correlation that is mostly noise.

Usage
-----
At the end of training:  pass --length-eval to train_introspector_joint.py
On an existing checkpoint (no retraining):
    python train_introspector_joint.py --config <cfg> --length-only --checkpoint <ckpt>
"""
from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk

from src.introspector.config import ExperimentConfig
from src.introspector.signals.schema import (
    RandomRangeTokenSelector, PairListTokenSelector, is_pair_selector,
)
from src.introspector.signals.pair_grid import pair_grid
from src.introspector.dataset_utils import apply_signal_filter, canonical_dataset_name

# Eval lengths. Every one of these must be <= the min_seq_length the caches were built
# with, or the documents will not actually reach the nominal length and the distance
# axis collapses toward the short end (see DatasetConfig.min_seq_length).
DEFAULT_EVAL_LENGTHS = (16, 64, 128)
DEFAULT_N_TEST = 3000

# Log-spaced, because what matters is the ORDER of magnitude of the reach, and because
# uniform position sampling puts most mass at large distances. Upper edges, inclusive.
DISTANCE_BINS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

# preprocess_example truncates on training_config.max_seq_length, which has to cover
# text + introspection prompt + target. Too small and the introspected token is cut off
# entirely -- drop_invalid_positions drops those rows (or refuses, if there are enough
# to mean something is misconfigured) rather than let the hooks clamp them
# to the wrong token.
PROMPT_HEADROOM = 40


def training_window(config: ExperimentConfig) -> int | None:
    """Widest distance-from-end the run was TRAINED on, or None if not expressible.

    A fixed selector at -k trains one distance (k-1). A random_range over [start, stop]
    trains a span; negative bounds are distances from the end, which is what the length
    sweep varies. Returns the largest trained distance, so bins beyond it are the
    out-of-distribution ones.
    """
    sel = config.train_token_selector
    if getattr(sel, "type", None) == "fixed":
        pos = sel.position
        return abs(pos) - 1 if pos < 0 else None
    if getattr(sel, "type", None) == "random_range":
        lo, hi = sel.start, sel.stop
        if lo < 0 and hi < 0:
            return max(abs(lo), abs(hi)) - 1
    return None


def derive_config(base_config: ExperimentConfig, eval_len: int, config_dir: Path,
                  n_test: int = DEFAULT_N_TEST) -> Path:
    """A config identical to the run's except for the TEST split's length and positions.

    The test selector spans the WHOLE eval length rather than the training window: the
    point is to sample distances the run never saw. Positions are drawn uniformly over
    that span and binned afterwards, so one cache per length serves every distance bin
    -- a cache per (length, bin) would multiply the build cost for no extra information.

    training_config.max_seq_length is re-derived from THIS length. It is stored as
    train_len + headroom, so a checkpoint trained at 16 carries a cap of ~46 and would
    silently truncate a 128-token eval document down to nothing.
    """
    cfg = base_config.model_copy(deep=True)
    pair_mode = is_pair_selector(base_config.train_token_selector)
    if pair_mode:
        # PAIR MODE (3-1): the fixed grid of THIS length, on the SAME documents as the
        # run's own test split -- same corpus, same floor, same count, so the streamed
        # documents are identical and only their truncation length differs. That is what
        # makes (document, dst, src) comparable across lengths; the eval checks it.
        n_test = base_config.test_dataset.max_samples
    cfg.test_dataset = cfg.test_dataset.model_copy(update={
        "max_samples": n_test,
        "max_seq_length": eval_len,
    })
    if pair_mode:
        cfg.test_token_selector = PairListTokenSelector(pairs=list(pair_grid(eval_len)))
    else:
        # Sample positions across the entire document, expressed from the end so it is
        # the same quantity the training selector is written in. Always a range, whatever
        # the run trained on -- a fixed-position run is exactly the case whose
        # generalization to other positions is in question.
        cfg.test_token_selector = RandomRangeTokenSelector(start=-eval_len, stop=-1)

    # Nothing trains here, but cache_signals builds all three splits. Shrink them, and
    # keep dev's offset small: on a streaming dataset the run's usual .skip(1_000_000)
    # actually iterates a million documents per cache build, which reads as a hang.
    # Dev is only built because cache_signals builds all three splits; nothing here
    # trains or validates. Point it at the TRAIN split rather than inheriting the run's
    # dev split: the run's dev may live on the same split as this config's test (c4
    # validation for a length eval, and for the c4_zh / c4_bn corpora), and dev would
    # then have to clear TEST's raw span instead of a 64-row train span. Offset 2000
    # clears the shrunken train span (64 rows is ~90 raw documents under a length floor)
    # with a wide margin and still costs under a second.
    # The OFFSET matters as much as the size: inheriting the run's usual 1,000,000 would
    # make .skip() walk a million streaming documents per derived cache, which reads as
    # a hang rather than an error.
    cfg.train_dataset = cfg.train_dataset.model_copy(update={"max_samples": 64})
    cfg.dev_dataset = cfg.dev_dataset.model_copy(update={
        "max_samples": 64, "offset": 2000, "split": cfg.train_dataset.split})

    cfg.training_config = cfg.training_config.model_copy(
        update={"max_seq_length": eval_len + PROMPT_HEADROOM})
    cfg.dataset_name = None
    cfg.dataset_name = canonical_dataset_name(cfg)
    cfg.experiment_name = f"{base_config.experiment_name}__len{eval_len}"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{cfg.experiment_name}.json"
    path.write_text(json.dumps(cfg.model_dump(), indent=2, sort_keys=True, default=str))
    return path


def ensure_cache(config_path: Path):
    """Run cache_signals for this config (it no-ops when the dataset already exists).

    stderr is captured and its tail re-raised: a bare CalledProcessError says only that
    the child exited non-zero, which is useless when several lengths fail for different
    reasons.
    """
    proc = subprocess.run([sys.executable, "cache_signals.py", "--config", str(config_path)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-15:])
        raise RuntimeError(f"cache_signals failed for {config_path.name}:\n{tail}")
    return proc


def prepare_length_caches(base_config: ExperimentConfig,
                          eval_lengths=DEFAULT_EVAL_LENGTHS,
                          n_test: int = DEFAULT_N_TEST):
    """
    Build the eval caches. CALL THIS BEFORE THE CALLER LOADS ITS MODEL.

    cache_signals runs as a subprocess and needs the GPU to extract activations. If the
    parent already holds the device, that second CUDA context fails with "CUDA-capable
    device(s) is/are busy or unavailable".

    Returns {eval_len: config_path or None}; failures are recorded, not raised.
    """
    floor = base_config.test_dataset.min_seq_length
    config_dir = Path("experiment_configs") / "_lengen"
    prepared, errors = {}, {}
    for eval_len in eval_lengths:
        if floor and eval_len > floor:
            # Not a hard failure, but the resulting cache would not mean what its name
            # says: documents shorter than eval_len make the far distance bins empty
            # and pull the whole distribution toward the short end.
            msg = (f"eval length {eval_len} exceeds the cache min_seq_length floor "
                   f"({floor}); documents are not guaranteed to reach it")
            print(f"[lengen] SKIP len={eval_len}: {msg}")
            errors[eval_len] = msg
            prepared[eval_len] = None
            continue
        print(f"[lengen] preparing cache: len={eval_len}")
        try:
            cfg_path = derive_config(base_config, eval_len, config_dir, n_test)
            ensure_cache(cfg_path)
            prepared[eval_len] = cfg_path
        except Exception as e:
            print(f"[lengen] len={eval_len} cache FAILED ({type(e).__name__}: {e})")
            errors[eval_len] = f"{type(e).__name__}: {e}"
            prepared[eval_len] = None
    prepare_length_caches.errors = errors
    return prepared


def _bin_by_distance(df: pd.DataFrame, value_col: str, target_col: str,
                     train_window: int | None) -> pd.DataFrame:
    """Correlation within each distance bin, labeled in- or out-of-training-range."""
    from src.introspector.training.metrics import compute_metrics_from_numericals
    if "distance_from_end" not in df.columns:
        return pd.DataFrame()
    d = df[df["distance_from_end"] >= 0].copy()
    if d.empty:
        return pd.DataFrame()
    edges = list(DISTANCE_BINS)
    labels = [f"<={b}" for b in edges]
    # keep the numeric span; do not re-parse the label
    span_of = dict(zip(labels, zip([-1] + edges[:-1], edges)))
    d["bin"] = pd.cut(d["distance_from_end"], bins=[-1] + edges, labels=labels)
    rows = []
    for name, g in d.groupby("bin", observed=True):
        lo, hi = span_of[str(name)]
        # A bin covers distances (lo, hi]. Labelling it by its UPPER edge alone is wrong
        # at the boundary: with a window of 128 (trained distances 0-127) the "<=128" bin
        # spans 65-128 and is almost entirely in range, but hi=128 would mark the whole
        # bin out-of-range -- understating in-range fidelity and inflating the gap. So a
        # bin is in-range only if it lies WHOLLY inside the window, out-of-range only if
        # wholly outside, and straddling otherwise. Straddling bins are excluded from the
        # gap rather than assigned to a side.
        if train_window is None:
            in_range, straddles = None, False
        else:
            in_range = hi <= train_window
            straddles = (lo < train_window) and (hi > train_window)
        row = {"bin": str(name), "n": len(g), "min_distance": lo + 1, "max_distance": hi,
               "in_train_range": in_range, "straddles_window": straddles,
               "corr": np.nan, "l2_dist": np.nan}
        # corr over fewer than 2 points is undefined; the row is still emitted so a
        # thin or empty bin is visible rather than silently absent from the curve
        if len(g) >= 2:
            m = compute_metrics_from_numericals(g[value_col].to_numpy(),
                                                g[target_col].to_numpy())
            row.update({k: m[k] for k in ("corr", "l2_dist") if k in m})
        rows.append(row)
    return pd.DataFrame(rows)


def _doc_ids(raw, n_rows_per_doc):
    """{doc_index: text_input_ids} from a single-signal split ordered doc -> pair."""
    first = raw.select(range(0, len(raw), n_rows_per_doc))
    docs = first["doc_index"]
    assert docs == list(range(len(docs))), (
        f"rows are not grouped doc-major ({n_rows_per_doc} per doc): doc_index starts "
        f"{docs[:5]}")
    return dict(zip(docs, first["text_input_ids"]))


def _check_same_documents(raw, reference, eval_len):
    """PAIR MODE: assert this length's documents are the reference ones, longer.

    The length sweep is a paired comparison -- the weight at (d, s) of document i is the
    same number at every length -- only if every length streamed the same documents and
    truncated them differently. Nothing else enforces that; a changed filter or sample
    count would silently swap the documents and turn the comparison into a content shift.
    """
    ref_ids, ref_len = reference
    n_pairs = len(pair_grid(eval_len))
    ids = _doc_ids(raw, n_pairs)
    assert set(ids) == set(ref_ids), (
        f"len {eval_len}: {len(ids)} documents vs {len(ref_ids)} in the reference split")
    bad = [i for i in ids if ids[i][:ref_len] != ref_ids[i]]
    assert not bad, (f"len {eval_len}: {len(bad)} documents do not start with the reference "
                     f"split's tokens (e.g. doc {bad[0]}) -- the splits hold different documents")
    print(f"[lengen] len {eval_len}: same {len(ids)} documents as the reference "
          f"(len {ref_len}), prefix-checked")


def evaluate_length_grid(model, base_config: ExperimentConfig, tokenizer, collator,
                         output_dir, eval_lengths=DEFAULT_EVAL_LENGTHS,
                         n_test: int = DEFAULT_N_TEST, prepared=None,
                         interv_dists=("gaussian",)):
    """
    Evaluate an already-built model across eval lengths, binned by distance-to-readout.

    `prepared` comes from prepare_length_caches, which must have run BEFORE the model
    was loaded (see its docstring). Passing None falls back to preparing here, which
    works only if nothing else holds the GPU.

    The preprocess function is rebuilt per eval length rather than taken from the
    caller: it closes over training_config.max_seq_length, and the caller's is sized for
    the TRAINING length, which would truncate every longer eval document.
    """
    from train_introspector import preprocess_example, drop_invalid_positions
    from train_introspector_joint import eval_endogenous_test, eval_provenance
    from train_introspector_with_intervention import evaluate_interventional_test
    from src.introspector.prompts import PROMPT_REGISTRY
    from src.introspector.models.attention_logits import is_attention_type

    output_dir = Path(output_dir)
    if prepared is None:
        print("[lengen] WARNING: caches not prepared before the model was loaded; "
              "building them now. If the GPU is already held this fails with "
              "'CUDA-capable device(s) is/are busy or unavailable'.")
        prepared = prepare_length_caches(base_config, eval_lengths, n_test)
    errors = getattr(prepare_length_caches, "errors", {})
    train_window = training_window(base_config)
    print(f"[lengen] training window = {train_window} tokens from the end "
          f"(bins beyond this are out-of-distribution)")

    pair_mode = is_pair_selector(base_config.train_token_selector)
    reference = None
    if pair_mode:
        # the run's own test split (length n) is the reference every longer one must extend
        ref_raw = apply_signal_filter(load_from_disk(str(base_config.dataset_cache_dir / "test")),
                                      base_config.signal_filter, "lengen/reference")
        ref_len = base_config.test_dataset.max_seq_length
        reference = (_doc_ids(ref_raw, len(base_config.test_token_selector.pairs)), ref_len)

    summary_rows, bin_frames = [], []
    for eval_len in eval_lengths:
        cfg_path = prepared.get(eval_len)
        if cfg_path is None:
            summary_rows.append({"eval_len": eval_len,
                                 "error": errors.get(eval_len, "cache not prepared")})
            continue
        print(f"\n[lengen] ===== eval_len={eval_len} =====")
        try:
            cfg = ExperimentConfig.model_validate_json(Path(cfg_path).read_text())
            raw = load_from_disk(str(cfg.dataset_cache_dir / "test"))
            raw = apply_signal_filter(raw, base_config.signal_filter, f"lengen/{eval_len}")

            def preprocess(ex, _cfg=cfg):
                out = preprocess_example(
                    ex, tokenizer=tokenizer,
                    prompt_fn=PROMPT_REGISTRY[base_config.training_config.prompt_type],
                    max_length=_cfg.training_config.max_seq_length,
                    return_frozen_input_len=True, return_position_valid=True,
                    for_prediction=True)
                out["token_position"] = ex["token_position"]
                out["signal_layer"] = ex["signal_layer"]
                out["signal_index"] = ex["signal_index"]
                out["activation"] = ex["activation"]
                # Attention rows need the source as well, exactly as the trainer's own
                # preprocess emits it. Without these the interventional eval tripped the
                # "intervention site mismatch" assert on every attention run.
                if is_attention_type(ex["signal_type"]):
                    out["src_token_position"] = ex["src_token_position"]
                    out["signal_src_offset"] = ex["src_offset"]
                return out

            if pair_mode:
                _check_same_documents(raw, reference, eval_len)
            ds = raw.map(preprocess, remove_columns=raw.column_names)
            provenance = eval_provenance(raw, ds)
            ds = drop_invalid_positions(ds, f"lengen/len{eval_len}",
                                        cfg.training_config.max_seq_length)

            suffix = f"_len{eval_len}"
            interv = None
            for dist in interv_dists:
                # gaussian keeps the historical file names; any other distribution
                # gets its own suffix so the two never overwrite each other
                r = evaluate_interventional_test(
                    model, eval_dataset=ds, collator=collator, config=base_config,
                    output_dir=str(output_dir), distribution=dist,
                    suffix=suffix if dist == "gaussian" else f"{suffix}_{dist}",
                    provenance=provenance)
                interv = r if interv is None else interv
            endo = eval_endogenous_test(model, ds, collator, base_config, output_dir,
                                        suffix=suffix, provenance=provenance)

            # Read the per-example frames back rather than threading them through the
            # eval functions' return values: those return metric dicts and are called
            # by several other scripts, and the CSVs are written at a fixed path.
            for kind, fname, val, tgt in (
                ("endogenous", f"endogenous_test_predictions{suffix}.csv",
                 "predictions", "activation"),
                ("interventional", f"interventional_test_predictions{suffix}.csv",
                 "predictions", "injected"),
            ):
                fpath = output_dir / fname
                if not fpath.exists():
                    continue
                binned = _bin_by_distance(pd.read_csv(fpath), val, tgt, train_window)
                if not binned.empty:
                    binned.insert(0, "eval_len", eval_len)
                    binned.insert(1, "eval_type", kind)
                    bin_frames.append(binned)

            summary_rows.append({
                "eval_len": eval_len, "n": len(ds),
                "train_window": train_window,
                "interventional_corr": interv.get("corr"),
                "endogenous_corr": endo.get("corr"),
                "endogenous_l2": endo.get("l2_dist"),
                # PAIR MODE: the headline is within-pair, not pooled (pair_metrics)
                **{f"endogenous_{k}": v for k, v in endo.get("pairs", {}).items()
                   if k in ("corr_within_pair_mean", "corr_demeaned",
                            "pair_mean_baseline_corr", "n_pairs")},
            })
        except Exception as e:
            print(f"[lengen] len={eval_len} eval FAILED ({type(e).__name__}: {e}); continuing")
            summary_rows.append({"eval_len": eval_len, "error": f"{type(e).__name__}: {e}"})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "length_generalization.csv", index=False)
    bins = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()
    if not bins.empty:
        bins.to_csv(output_dir / "length_generalization_by_distance.csv", index=False)
    with open(output_dir / "length_generalization.json", "w") as f:
        json.dump({"train_window": train_window,
                   "eval_lengths": list(eval_lengths),
                   "signal_filter": base_config.signal_filter,
                   "summary": summary_rows}, f, indent=2, default=str)

    print("\n[lengen] summary (pooled over distances):")
    print(summary.to_string(index=False))
    if not bins.empty:
        print("\n[lengen] by distance bin:")
        print(bins.to_string(index=False))
        thin = bins[bins["n"] < 30]
        if len(thin):
            print(f"[lengen] NOTE: {len(thin)} bin(s) have n<30; their correlations are "
                  f"mostly noise.")
    return summary, bins
