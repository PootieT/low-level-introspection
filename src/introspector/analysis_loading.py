"""
Load a trained introspector for post-hoc circuit analysis.

WHY THIS EXISTS
---------------
Every circuit analysis needs the same object: the checkpoint's model wrapped as the
regressor it was trained as, its test split preprocessed exactly as training
preprocessed it, and a collator that reproduces the run's LoRA gating. Getting any of
those wrong does not crash -- it silently measures a different model. Three copies of
that sequence is three chances to drift apart, so there is one.

TWO THINGS THE OLD PER-SCRIPT COPIES GOT WRONG, and which are the reason this module
exists rather than a tidy-up:

  CACHE PATH.  `cache_dir / experiment_name` predates shared caches. Sweeps key the
    cache by `dataset_name` (config.dataset_cache_dir) so one (layer, position) cell is
    cached ONCE for all its neurons, and every run in the joint grid would fail to load
    -- or, worse, load a stale directory that happens to carry the run's own name.
    Along with it goes `signal_filter`: a grid run's cache holds all 5 neurons of its
    cell and the filter is the only thing that says which one this run reports.

  THE GATE.  LoRAGatingCollator emits `lora_token_mask`, which GatedLinear applies to
    zero the adapter over the input text. A run trained with `disable_adapter_in_input`
    FALSE never had that mask and its adapter is live over the text; analyzing it with
    the gated collator disables the very machinery under study, at the origin token,
    and still produces plausible-looking numbers. The whole joint grid is gate-off.

PRECISION. The analyses cast their own captures to fp32, but the FORWARD runs in the
run's own dtype (bf16 unless the config says otherwise). That is deliberate: the
quantity being localized is what the trained model actually computes, bf16 floor
included -- an fp32 forward is a different model, and a separate experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig
from .dataset_utils import apply_signal_filter
from .run_discovery import find_checkpoint, resolve_locale
from .training.collator import LoRAGatingCollator
from .training.model import (
    IntrospectionRegressionModel,
    load_added_token_embeddings,
    load_regression_head,
    replace_lora_with_gated_version,
    replace_model_with_gated_version,
)


@dataclass
class LoadedRun:
    """Everything an analysis needs, loaded once."""
    config: ExperimentConfig
    checkpoint: Path
    model: Any
    tokenizer: Any
    collator: Any
    dataset: Any                 # preprocessed test split (model kwargs)
    raw: Any                     # the unpreprocessed test split (text, activation, ...)
    signal_ranges: dict
    signal_type: str
    layer: int
    index: int
    preprocess_fn: Callable

    @property
    def base(self):
        """The underlying HF causal LM, through the PEFT/regressor wrappers."""
        return self.model.get_base_model()

    @property
    def stats(self) -> dict:
        """Empirical activation stats of THIS run's neuron (min/max/mean/std)."""
        return self.signal_ranges[(self.layer, self.index)]


def compute_signal_ranges(raw) -> dict:
    """{(layer, index): {min,max,mean,std}} from a cached split.

    Taken from the TRAIN split by every caller, because that is the distribution the
    intervention sampler was configured against during training; using test stats
    would quietly change the injected values relative to what the run saw.
    """
    # only the three columns the stats need: a cached split also carries `text` and
    # `text_input_ids`, and materializing those for 300k rows costs seconds and
    # hundreds of MB per run -- once per run, times every run in a sweep.
    cols = [c for c in ("signal_layer", "signal_index", "activation") if c in raw.column_names]
    df = raw.select_columns(cols).to_pandas()
    ranges = {}
    for key, vals in df.groupby(["signal_layer", "signal_index"])["activation"]:
        ranges[key] = {"min": float(vals.min()), "max": float(vals.max()),
                       "mean": float(vals.mean()), "std": float(vals.std())}
    return ranges


def layer_output_is_tuple(model) -> bool:
    """
    Does a decoder layer's forward return `(hidden, ...)` or a bare tensor?

    It depends on the transformers version -- 4.x returns a tuple, 5.x returns the
    tensor -- and every capture that reads a layer's residual has to index it the
    right way. Guessing from the version string is fragile across the cluster and
    this machine, so this runs ONE two-token forward with a hook and looks. Cheap,
    and it cannot disagree with the model actually being analyzed.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layer = base.model.layers[0]
    seen = {}

    def hook(_m, _inputs, output):
        seen["tuple"] = isinstance(output, tuple)

    handle = layer.register_forward_hook(hook)
    try:
        device = next(base.parameters()).device
        bos = getattr(base.config, "bos_token_id", None) or 0
        with torch.no_grad():
            model(input_ids=torch.tensor([[bos, bos]], device=device), do_intervention=False)
    finally:
        handle.remove()
    if "tuple" not in seen:
        raise RuntimeError("Could not determine decoder-layer output type (hook never fired).")
    return seen["tuple"]


def absolute_positions(token_position: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Cached token_position (indexes the unpadded, BOS-prefixed text) -> index into
    the LEFT-PADDED batch. Identical correction to the one the intervention hook and
    the drift measurement apply; every capture in every analysis must use it."""
    num_left_pad = (attention_mask == 0).sum(dim=1).to(token_position.device)
    return token_position + num_left_pad


def load_run_for_analysis(
    config: ExperimentConfig | str | Path,
    checkpoint: Optional[Path] = None,
    split: str = "test",
    max_examples: Optional[int] = None,
    device_map: str = "auto",
    local_files_only: bool = False,
) -> LoadedRun:
    """
    The trained model + its own test split, ready to measure.

    `checkpoint` defaults to the latest under the run's output dir. `max_examples`
    truncates the split as a CONTIGUOUS PREFIX (cache_signals writes every signal of a
    text consecutively, so a prefix stays balanced across signals where a random subset
    would not).
    """
    from train_introspector import (  # local: heavy module, and it imports this package
        INTROSPECTION_SEPARATOR, drop_invalid_positions, preprocess_example)
    from train_introspector_joint import UngatedCollator
    from .prompts import PROMPT_REGISTRY

    if not isinstance(config, ExperimentConfig):
        config = ExperimentConfig.model_validate_json(Path(config).read_text())
    if not config.use_regressor:
        raise SystemExit(f"{config.experiment_name}: the circuit analyses target the "
                         f"regression head (config.use_regressor must be True).")
    run_dir = Path(config.output_dir) / config.experiment_name
    checkpoint = Path(checkpoint) if checkpoint else find_checkpoint(run_dir)
    signal_type, layer, index = resolve_locale(config)

    # ---- tokenizer (the separator is part of the trained vocabulary) ----
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [INTROSPECTION_SEPARATOR]})

    # ---- data (shared cache + this run's signal) ----
    cache_dir = config.dataset_cache_dir
    train_raw = apply_signal_filter(load_from_disk(str(cache_dir / "train")),
                                    config.signal_filter, "train")
    raw = apply_signal_filter(load_from_disk(str(cache_dir / split)),
                              config.signal_filter, split)
    signal_ranges = compute_signal_ranges(train_raw)
    del train_raw
    if max_examples is not None and max_examples < len(raw):
        raw = raw.select(range(max_examples))

    def preprocess(ex):
        out = preprocess_example(
            ex, tokenizer=tokenizer,
            prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
            max_length=config.training_config.max_seq_length,
            return_frozen_input_len=True, return_position_valid=True,
            for_prediction=True)
        out["token_position"] = ex["token_position"]
        out["signal_layer"] = ex["signal_layer"]
        out["signal_index"] = ex["signal_index"]
        out["activation"] = ex["activation"]
        return out

    dataset = raw.map(preprocess, remove_columns=raw.column_names)
    # rows whose introspected token was truncated away are not addressable at all; the
    # hooks would clamp them onto the wrong token and the analysis would average that in
    dataset = drop_invalid_positions(dataset, split, config.training_config.max_seq_length)

    # ---- model ----
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        config.model_name, device_map=device_map, local_files_only=local_files_only,
        torch_dtype=torch.bfloat16 if config.training_config.bf16 else torch.float32)
    base.resize_token_embeddings(len(tokenizer))
    peft_model = PeftModel.from_pretrained(base, str(checkpoint), autocast_adapter_dtype=False)
    model = IntrospectionRegressionModel(
        peft_model, signal_ranges=signal_ranges,
        intervention_distribution=config.training_config.intervention_signal_distribution,
        intervention_extreme_scale=config.training_config.intervention_extreme_scale)
    load_regression_head(model, str(checkpoint))
    load_added_token_embeddings(model, str(checkpoint))   # no-op unless embeddings were stripped
    replace_model_with_gated_version(model)
    replace_lora_with_gated_version(model)
    model.eval()

    # ---- collator: reproduce the run's own gating, or the analysis measures a
    # different model than the one that was trained (see module docstring) ----
    gate_input = config.training_config.disable_adapter_in_input
    collator_cls = LoRAGatingCollator if gate_input else UngatedCollator
    print(f"[load] {config.experiment_name}: N{layer}.{index} | ckpt={checkpoint.name} | "
          f"gate={'ON' if gate_input else 'OFF'} ({collator_cls.__name__}) | "
          f"{len(dataset)} {split} rows")
    collator = collator_cls(tokenizer=tokenizer, model=model, padding=True,
                            label_pad_token_id=-100, return_tensors="pt", for_prediction=True)

    return LoadedRun(config=config, checkpoint=checkpoint, model=model, tokenizer=tokenizer,
                     collator=collator, dataset=dataset, raw=raw, signal_ranges=signal_ranges,
                     signal_type=signal_type, layer=layer, index=index, preprocess_fn=preprocess)
