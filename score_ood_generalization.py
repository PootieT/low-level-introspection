#!/usr/bin/env python3
"""
Does the readout generalize off-distribution -- and does joint training generalize
better than plain endogenous SFT?

The claim the joint loss is supposed to buy is a CAUSAL readout rather than a
context-predicting one. Those two are hard to tell apart in-distribution: a model that
has learned "text like this implies activation like that" scores well on C4 either way.
Off-distribution they separate, because the context-to-activation mapping is what
breaks under shift while a causal read is indifferent to it.

  ENDOGENOUS corr OOD   the discriminating metric. SFT's in-domain score comes
                        substantially from context, so it should fall further.
  INTERVENTIONAL corr   tests something different: whether the transmission channel
                        itself survives shift. If the mechanism is the private relay
                        we measured, it should not care what language the text is in.

CORPUS LADDER (near -> far), so a drop can be read against how far the shift went.
Without the near anchor there is no way to tell "bypass collapsed" from "everything
degrades off-distribution".

  wikipedia   style only, same language
  code/math   domain shift, Latin script
  c4_zh       language shift, non-Latin
  c4_bn       low-resource language
  shuffled    THE CONTROL: the run's own test text with word order destroyed. Token
              statistics preserved, semantics removed -- a context-predicting readout
              has nothing left to use, a causal one is unaffected. Needs no download.

WATCH FOR: a neuron that barely fires on a corpus has near-zero activation variance,
which makes correlation unstable and reads as "generalization failure" when it is an
undefined metric. activation_std is reported per corpus for exactly this filtering.

Also note the introspection PROMPT stays English while the text does not, so the
language rows carry a second shift; the shuffled control does not.

Usage
-----
On an existing checkpoint (no retraining):
    python train_introspector_joint.py --config <cfg> --diag-only --checkpoint <ckpt> --ood-eval
At the end of training: pass --ood-eval to train_introspector_joint.py.
"""
from __future__ import annotations

import os
import sys
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk

from src.introspector.config import ExperimentConfig, DatasetConfig
from src.introspector.dataset_utils import apply_signal_filter, canonical_dataset_name

# name -> how to load it. Dataset ids drift between releases; each corpus is attempted
# independently and a failure is reported without taking the others down.
OOD_CORPORA = {
    "wikipedia": dict(name="wikimedia/wikipedia", split="train", text_field="text",
                      hf_data_kwargs={"config": "20231101.en", "streaming": True}),
    "code":      dict(name="codeparrot/codeparrot-clean-valid", split="train",
                      text_field="content", hf_data_kwargs={"streaming": True}),
    "math":      dict(name="open-web-math/open-web-math", split="train", text_field="text",
                      hf_data_kwargs={"streaming": True}),
    "c4_zh":     dict(name="allenai/c4", split="validation", text_field="text",
                      hf_data_kwargs={"config": "zh", "streaming": True}),
    "c4_bn":     dict(name="allenai/c4", split="validation", text_field="text",
                      hf_data_kwargs={"config": "bn", "streaming": True}),
    # built locally from the run's own cached test split -- see build_shuffled_corpus
    "shuffled":  dict(shuffled=True, name="json", split="train", text_field="text",
                      hf_data_kwargs={"streaming": True}),
}
DEFAULT_CORPORA = ("shuffled", "wikipedia", "c4_zh", "c4_bn", "math", "code")


def build_shuffled_corpus(base_config: ExperimentConfig, out_path: Path, seed: int = 0) -> Path:
    """Word-shuffle the run's own cached test texts into a JSONL.

    Same corpus, same vocabulary, same length distribution -- only the order carries no
    meaning any more. That isolates "predicts from context" from "reads the signal"
    better than any other corpus can, because nothing else about the distribution moved.
    """
    if out_path.exists():
        return out_path
    src = base_config.dataset_cache_dir / "test"
    texts = load_from_disk(str(src))["text"]
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(out_path, "w") as f:
        for t in texts:
            if t in seen:          # cached rows repeat once per signal
                continue
            seen.add(t)
            words = t.split()
            rng.shuffle(words)
            f.write(json.dumps({"text": " ".join(words)}) + "\n")
    print(f"[ood] wrote shuffled corpus ({len(seen)} texts) -> {out_path}")
    return out_path


DEFAULT_N_TEST = 3000        # same size as the in-domain test split


def derive_config(base_config: ExperimentConfig, corpus: str, spec: dict,
                  config_dir: Path, n_test: int = DEFAULT_N_TEST) -> Path:
    """A config identical to the run's except for the TEST corpus.

    Train/dev are shrunk to a token amount: nothing trains here, but cache_signals
    builds all three splits. dataset_name is cleared so the canonical key is derived
    from the new content -- which also means the 5 neurons of a cell share one OOD
    cache, and it is built once."""
    cfg = base_config.model_copy(deep=True)
    test = spec.copy()
    test.pop("shuffled", None)
    cfg.test_dataset = DatasetConfig(
        name=test["name"], split=test["split"], text_field=test["text_field"],
        max_samples=n_test,
        max_seq_length=base_config.test_dataset.max_seq_length,
        hf_data_kwargs=test["hf_data_kwargs"],
    )
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
    cfg.dataset_name = None
    cfg.dataset_name = canonical_dataset_name(cfg)
    cfg.experiment_name = f"{base_config.experiment_name}__ood_{corpus}"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{cfg.experiment_name}.json"
    path.write_text(json.dumps(cfg.model_dump(), indent=2, sort_keys=True, default=str))
    return path


def ensure_cache(config_path: Path):
    """Run cache_signals for this config (it no-ops when the dataset already exists).

    stderr is captured and its tail re-raised: a bare CalledProcessError says only that
    the child exited non-zero, which is useless when six corpora fail and the causes
    differ (a missing dataset config, a gated repo, a network timeout).
    """
    proc = subprocess.run([sys.executable, "cache_signals.py", "--config", str(config_path)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-15:])
        raise RuntimeError(f"cache_signals failed for {config_path.name}:\n{tail}")
    return proc


def prepare_ood_caches(base_config: ExperimentConfig, corpora=DEFAULT_CORPORA,
                       n_test=DEFAULT_N_TEST, seed=0):
    """
    Build the OOD activation caches. CALL THIS BEFORE THE CALLER LOADS ITS MODEL.

    cache_signals runs as a subprocess and needs the GPU to extract activations. If the
    parent already holds the device, that second CUDA context fails with "CUDA-capable
    device(s) is/are busy or unavailable" -- which is what happens if caching is
    deferred until evaluation time.

    Returns {corpus: config_path or None}; failures are recorded, not raised, so one
    unavailable dataset does not cost the others.
    """
    config_dir = Path("experiment_configs") / "_ood"
    prepared, errors = {}, {}
    for corpus in corpora:
        spec = OOD_CORPORA.get(corpus)
        if spec is None:
            errors[corpus] = f"unknown corpus (known: {sorted(OOD_CORPORA)})"
            prepared[corpus] = None
            continue
        print(f"[ood] preparing cache: {corpus}")
        try:
            spec = spec.copy()
            if spec.pop("shuffled", False):
                jsonl = Path("data/_ood_shuffled") / f"{base_config.dataset_name or 'run'}.jsonl"
                build_shuffled_corpus(base_config, jsonl, seed=seed)
                spec["hf_data_kwargs"] = {"data_files": str(jsonl), "streaming": True}
            cfg_path = derive_config(base_config, corpus, spec, config_dir, n_test)
            ensure_cache(cfg_path)
            prepared[corpus] = cfg_path
        except Exception as e:
            print(f"[ood] {corpus} cache FAILED ({type(e).__name__}: {e})")
            errors[corpus] = f"{type(e).__name__}: {e}"
            prepared[corpus] = None
    prepare_ood_caches.errors = errors
    return prepared


def evaluate_ood(model, base_config: ExperimentConfig, tokenizer, collator, preprocess_fn,
                 output_dir, corpora=DEFAULT_CORPORA, n_test=DEFAULT_N_TEST, seed=0,
                 prepared=None):
    """
    Evaluate an already-built model on each OOD corpus.

    `prepared` comes from prepare_ood_caches, which must have run BEFORE the model was
    loaded (see its docstring). Passing None falls back to preparing here, which works
    only if nothing else holds the GPU.
    """
    # imported here: the training script imports THIS module, so a module-level import
    # of its eval helpers would be circular
    from train_introspector_joint import eval_endogenous_test
    from train_introspector_with_intervention import evaluate_interventional_test

    output_dir = Path(output_dir)
    if prepared is None:
        print("[ood] WARNING: caches not prepared before the model was loaded; building "
              "them now. If the GPU is already held this fails with "
              "'CUDA-capable device(s) is/are busy or unavailable'.")
        prepared = prepare_ood_caches(base_config, corpora, n_test, seed)
    errors = getattr(prepare_ood_caches, "errors", {})

    rows = []
    for corpus in corpora:
        cfg_path = prepared.get(corpus)
        if cfg_path is None:
            rows.append({"corpus": corpus, "error": errors.get(corpus, "cache not prepared")})
            continue
        print(f"\n[ood] ===== {corpus} =====")
        try:
            cfg = ExperimentConfig.model_validate_json(Path(cfg_path).read_text())
            raw = load_from_disk(str(cfg.dataset_cache_dir / "test"))
            raw = apply_signal_filter(raw, base_config.signal_filter, f"ood/{corpus}")
            acts = np.asarray(raw["activation"], dtype=float)
            ds = raw.map(preprocess_fn, remove_columns=raw.column_names)

            interv = evaluate_interventional_test(
                model, eval_dataset=ds, collator=collator, config=base_config,
                output_dir=str(output_dir), distribution="gaussian",
                suffix=f"_ood_{corpus}")
            endo = eval_endogenous_test(model, ds, collator, base_config, output_dir,
                                        suffix=f"_ood_{corpus}")
            rows.append({
                "corpus": corpus, "n": len(ds),
                "interventional_corr": interv.get("corr"),
                "endogenous_corr": endo.get("corr"),
                "endogenous_l2": endo.get("l2_dist"),
                # a near-silent neuron makes correlation meaningless: filter on this
                "activation_std": float(np.std(acts)),
                "activation_mean": float(np.mean(acts)),
                "activation_frac_zero": float(np.mean(np.abs(acts) < 1e-6)),
            })
        except Exception as e:
            print(f"[ood] {corpus} eval FAILED ({type(e).__name__}: {e}); continuing")
            rows.append({"corpus": corpus, "error": f"{type(e).__name__}: {e}"})

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "ood_generalization.csv", index=False)
    with open(output_dir / "ood_generalization.json", "w") as f:
        json.dump({"signal_filter": base_config.signal_filter, "rows": rows}, f,
                  indent=2, default=str)
    print("\n[ood] generalization summary:")
    print(df.to_string(index=False))
    if "activation_std" in df and (df["activation_std"].fillna(1) <= 1e-6).any():
        print("[ood] NOTE: some corpora have near-zero activation variance for this "
              "neuron; their correlations are not interpretable.")
    return df


def evaluate_ood_detection(model, base_config: ExperimentConfig, tokenizer, collator,
                           preprocess_fn, spec, on_value, output_dir,
                           corpora=DEFAULT_CORPORA, n_test=DEFAULT_N_TEST, seed=0,
                           prepared=None, batch_size=128):
    """
    DETECTION counterpart of evaluate_ood, for SAE-feature runs.

    Same corpus ladder and same logic -- an endogenous score that comes from context
    should fall off-distribution while a causal read should not -- but scored with the
    detection metrics (AP is the imbalance-honest headline, AUROC secondary) instead of
    correlation, and with the paired steering test carried along so the causal channel
    can be tracked corpus by corpus.

    WATCH FOR (the detection analogue of the near-silent-neuron caveat): a feature that
    never fires on a corpus gives a single-class eval set, where AP/AUROC are undefined
    rather than bad. base_rate_positive is reported per corpus for exactly that filter.
    """
    from train_sae_detector import auroc_eval, interventional_detection_eval

    output_dir = Path(output_dir)
    if prepared is None:
        print("[ood] WARNING: caches not prepared before the model was loaded; building "
              "them now (fails if the GPU is already held).")
        prepared = prepare_ood_caches(base_config, corpora, n_test, seed)
    errors = getattr(prepare_ood_caches, "errors", {})
    tc = base_config.training_config

    rows = []
    for corpus in corpora:
        cfg_path = prepared.get(corpus)
        if cfg_path is None:
            rows.append({"corpus": corpus, "error": errors.get(corpus, "cache not prepared")})
            continue
        print(f"\n[ood-detect] ===== {corpus} =====")
        try:
            cfg = ExperimentConfig.model_validate_json(Path(cfg_path).read_text())
            raw = load_from_disk(str(cfg.dataset_cache_dir / "test"))
            raw = apply_signal_filter(raw, base_config.signal_filter, f"ood/{corpus}")
            acts = np.asarray(raw["activation"], dtype=float)
            ds = raw.map(preprocess_fn, remove_columns=raw.column_names)

            endo, _, _ = auroc_eval(model, ds, raw["activation"], collator, tokenizer,
                                    tc, batch_size)
            interv, _, _ = interventional_detection_eval(
                model, spec, raw, ds, collator, tokenizer, tc, on_value, batch_size)
            rows.append({
                "corpus": corpus, "n": len(ds),
                "endogenous_ap": endo.get("ap"),
                "endogenous_auroc": endo.get("auroc"),
                "endogenous_f1_at_0": endo.get("f1_at_0"),
                "interventional_auroc": interv.get("auroc"),
                "paired_directional_acc": interv.get("paired_directional_acc"),
                # single-class corpora make AP/AUROC undefined, not bad -- filter on this
                "base_rate_positive": endo.get("base_rate_positive"),
                "activation_std": float(np.std(acts)),
                "activation_frac_zero": float(np.mean(np.abs(acts) < 1e-6)),
            })
        except Exception as e:
            print(f"[ood-detect] {corpus} eval FAILED ({type(e).__name__}: {e}); continuing")
            rows.append({"corpus": corpus, "error": f"{type(e).__name__}: {e}"})

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "ood_generalization_detection.csv", index=False)
    with open(output_dir / "ood_generalization_detection.json", "w") as f:
        json.dump({"signal_filter": base_config.signal_filter, "rows": rows}, f,
                  indent=2, default=str)
    print("\n[ood-detect] generalization summary:")
    print(df.to_string(index=False))
    if "base_rate_positive" in df and (df["base_rate_positive"].fillna(1) <= 0).any():
        print("[ood-detect] NOTE: some corpora never fire this feature; their AP/AUROC "
              "are undefined, not zero.")
    return df


if __name__ == "__main__":
    print(__doc__)
    print("Run through train_introspector_joint.py --ood-eval (optionally with "
          "--diag-only --checkpoint <ckpt> for an existing run), or "
          "train_sae_detector.py --ood-eval for SAE detection runs.")
