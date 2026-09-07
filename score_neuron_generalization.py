#!/usr/bin/env python3
"""
Does the readout generalize to neurons it was never trained to report?

A run that trains on N neurons at one (layer, position) cell can be asked two
different questions on the same held-out text:

  SEEN    the trained neurons. How well does fidelity hold up as N grows? If every
          neuron gets its own private channel, they compete for the same residual
          bandwidth and per-neuron fidelity should fall as N rises. If one shared code
          serves all of them, it should stay flat.
  UNSEEN  neurons present in the cache but absent from training. Reachable only if the
          model learned an addressing scheme over neuron identity rather than N
          separate channels.

Every run evaluates EVERY neuron in the cache in one pass and labels each as trained
or held-out afterwards. The split is a property of the run, not of the data: the same
200-neuron test split serves the whole sweep, so a neuron held out at N=10 is trained
at N=100. Recording per-neuron fidelity once with is_trained alongside lets the
aggregation follow a single neuron across the sweep -- which is the interference
measurement, and is unrecoverable if the partition is baked into the files at eval time.
The unseen arm therefore costs no extra caching and no extra pass.

WHICH METRIC TO BELIEVE
  interventional corr   THE headline for the unseen arm. The injected value is drawn
                        independently per row, so a model that ignores which neuron the
                        prompt names cannot report it -- it scores 0 by construction.
                        That makes it immune to the artifact below.
  endogenous corr_macro macro-averaged OVER NEURONS, never pooled. Pooling mixes neurons
                        with different activation scales and counts the between-neuron
                        variance as signal, which inflates the number -- badly, at 100
                        neurons. Even macro-averaged, endogenous fidelity on unseen
                        neurons can be positive for an uninteresting reason: if unseen
                        neurons behave like seen ones, "predict a typical activation for
                        this context" scores above zero without any addressing at all.

So: endogenous high + interventional ~0 on the unseen arm means the readout is
predicting from context, not reading the named neuron. If those two disagree, the
follow-up is a name-shuffled control (prompt names a different neuron than the row's).

WATCH FOR: a neuron that barely fires has near-zero activation variance, which makes
its correlation undefined rather than bad. activation_std is reported per neuron.

Usage
-----
At the end of training:  pass --neuron-gen-eval to train_introspector_joint.py
On an existing checkpoint (no retraining):
    python train_introspector_joint.py --config <cfg> --neuron-gen-only --checkpoint <ckpt>
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk

from src.introspector.config import ExperimentConfig
from src.introspector.dataset_utils import apply_signal_filter, parse_signal_filters

_CODE = {"mlp": "N", "attention_score": "A", "residual_stream": "R", "sae": "S"}


def _spec_str(signal_type: str, layer: int, index: int) -> str:
    return f"{_CODE.get(signal_type, 'N')}{layer}.{index}"


def split_seen_unseen(config: ExperimentConfig, test_raw) -> tuple[list[str], list[str]]:
    """Partition the signals PRESENT IN THE CACHE into trained and held-out.

    Trained set comes from config.signal_filter when the run selected a subset of a
    shared cache, and from config.train_signals otherwise. Held-out is everything else
    in the cache -- derived from what is actually on disk rather than from the config,
    so a cache built with a different signal set shows up as a wrong count here instead
    of as a silently empty arm.
    """
    present = sorted({(t, l, i) for t, l, i in
                      zip(test_raw["signal_type"], test_raw["signal_layer"],
                          test_raw["signal_index"])})
    if config.signal_filter:
        trained = {(w["signal_type"], w["signal_layer"], w["signal_index"])
                   for w in parse_signal_filters(config.signal_filter)}
    else:
        trained = {(s.type, s.layer, s.index) for s in config.train_signals}
    seen = [k for k in present if k in trained]
    unseen = [k for k in present if k not in trained]
    return ([_spec_str(*k) for k in seen], [_spec_str(*k) for k in unseen])


def _activation_stats(raw) -> pd.DataFrame:
    df = pd.DataFrame({"signal_layer": raw["signal_layer"],
                       "signal_index": raw["signal_index"],
                       "activation": np.asarray(raw["activation"], dtype=float)})
    g = df.groupby(["signal_layer", "signal_index"])["activation"]
    return pd.DataFrame({"activation_std": g.std(), "activation_mean": g.mean(),
                         "activation_frac_zero": g.apply(lambda v: float(np.mean(np.abs(v) < 1e-6)))
                         }).reset_index()


def evaluate_neuron_generalization(model, config: ExperimentConfig, tokenizer, collator,
                                   preprocess_fn, output_dir, max_unseen: int | None = None):
    """
    Evaluate one trained checkpoint on EVERY neuron in the cache, in a single pass,
    labelling each as trained or held-out afterwards.

    One pass rather than one per arm, because the split is a property of the RUN, not of
    the data: the same 200-neuron test split serves every run in the sweep, and a neuron
    that is held out for the N=10 run is trained for the N=100 run. Recording each
    neuron's fidelity once, with is_trained alongside, lets the aggregation follow a
    SINGLE neuron across the whole sweep -- which is the interference measurement.
    Splitting at eval time instead would bake one run's partition into the files and
    make that comparison impossible to recover.

    Needs no cache preparation -- unlike the OOD and length evals, this reads the run's
    own test split, so there is no subprocess and no GPU-ordering constraint.
    """
    from train_introspector_joint import eval_endogenous_test, _per_signal_metrics
    from train_introspector_with_intervention import evaluate_interventional_test

    output_dir = Path(output_dir)
    test_raw = load_from_disk(str(config.dataset_cache_dir / "test"))
    seen, unseen = split_seen_unseen(config, test_raw)
    if max_unseen is not None and max_unseen < len(unseen):
        keep = set(seen) | set(unseen[:max_unseen])
        test_raw = apply_signal_filter(test_raw, ",".join(sorted(keep)), "neurongen/capped")
        unseen = unseen[:max_unseen]
    trained_set = set(seen)
    print(f"[neurongen] evaluating {len(seen) + len(unseen)} neurons in one pass: "
          f"{len(seen)} trained, {len(unseen)} held out")
    if not unseen:
        print("[neurongen] NOTE: no held-out neurons in this cache; there is no unseen "
              "arm. Cache the cell with more neurons than the run trains on.")

    act_stats = _activation_stats(test_raw)
    ds = test_raw.map(preprocess_fn, remove_columns=test_raw.column_names)
    suffix = "_neurongen"
    interv = evaluate_interventional_test(
        model, eval_dataset=ds, collator=collator, config=config,
        output_dir=str(output_dir), distribution="gaussian", suffix=suffix)
    endo = eval_endogenous_test(model, ds, collator, config, output_dir, suffix=suffix)
    # The POOLED numbers these return mix trained and held-out neurons and mean nothing
    # here; they are recomputed per arm below. Kept only in the json for traceability.

    frames = []
    for kind, fname, val, tgt in (
        ("endogenous", f"endogenous_test_predictions{suffix}.csv", "predictions", "activation"),
        ("interventional", f"interventional_test_predictions{suffix}.csv", "predictions", "injected"),
    ):
        fpath = output_dir / fname
        if not fpath.exists():
            continue
        per = _per_signal_metrics(pd.read_csv(fpath), value_col=val, target_col=tgt)
        if per is None or per.empty:
            continue
        per.insert(0, "eval_type", kind)
        frames.append(per)
    if not frames:
        print("[neurongen] no per-signal metrics produced; nothing to report")
        return pd.DataFrame(), pd.DataFrame()

    per_signal = pd.concat(frames, ignore_index=True)
    per_signal["signal"] = [_spec_str("mlp", l, i) for l, i in
                            zip(per_signal["signal_layer"], per_signal["signal_index"])]
    per_signal["is_trained"] = per_signal["signal"].isin(trained_set)
    per_signal["n_trained_in_run"] = len(seen)
    per_signal = per_signal.merge(act_stats, on=["signal_layer", "signal_index"], how="left")
    per_signal.to_csv(output_dir / "neuron_generalization_by_signal.csv", index=False)

    # Macro over neurons, per arm. Never pooled across neurons -- see module docstring.
    # Neurons with undefined correlation (near-constant activation) are excluded from the
    # mean and counted separately, so a shrinking denominator cannot pass for a good score.
    rows = []
    for (kind, is_tr), g in per_signal.groupby(["eval_type", "is_trained"]):
        valid = g["corr"].dropna()
        rows.append({
            "arm": "seen" if is_tr else "unseen", "eval_type": kind,
            "n_signals": len(g), "n_undefined_corr": int(len(g) - len(valid)),
            "n_trained_in_run": len(seen),
            "corr_macro": float(valid.mean()) if len(valid) else float("nan"),
            "corr_macro_std": float(valid.std(ddof=0)) if len(valid) else float("nan"),
            "corr_macro_min": float(valid.min()) if len(valid) else float("nan"),
            "corr_macro_max": float(valid.max()) if len(valid) else float("nan"),
        })
    summary = pd.DataFrame(rows).sort_values(["eval_type", "arm"]).reset_index(drop=True)
    summary.to_csv(output_dir / "neuron_generalization.csv", index=False)

    with open(output_dir / "neuron_generalization.json", "w") as f:
        json.dump({"n_trained": len(seen), "n_heldout": len(unseen),
                   "trained_signals": seen[:200], "heldout_signals": unseen[:200],
                   "pooled_ignore_me": {"endogenous": endo.get("corr"),
                                        "interventional": interv.get("corr")},
                   "summary": rows}, f, indent=2, default=str)

    print("\n[neurongen] summary (macro-averaged over neurons, per arm):")
    print(summary.to_string(index=False))
    dead = per_signal[per_signal["activation_std"].fillna(1) <= 1e-6]
    if len(dead):
        print(f"[neurongen] NOTE: {len(dead['signal'].unique())} neuron(s) have near-zero "
              f"activation variance; their correlations are undefined, not poor.")
    return summary, per_signal
