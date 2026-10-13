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

import hashlib
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig
from .dataset_utils import apply_signal_filter
from .models.attention_logits import is_attention_type
from .run_discovery import find_checkpoint, resolve_locales, locale_tag, plan_locales
from .training.collator import LoRAGatingCollator
from .training.model import (
    IntrospectionRegressionModel,
    load_added_token_embeddings,
    load_regression_head,
    logit_moments,
    replace_lora_with_gated_version,
    replace_model_with_gated_version,
    signal_range_key,
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
    # every signal the run was trained on; `layer`/`index` above are the one this
    # object is currently VIEWING (see for_locale)
    locales: list = field(default_factory=list)

    @property
    def is_multi_locale(self) -> bool:
        return len(self.locales) > 1

    @property
    def tag(self) -> str:
        return locale_tag(self.signal_type, self.layer, self.index)

    def for_locale(self, locale) -> "LoadedRun":
        """A view of this run restricted to ONE trained signal, sharing the model.

        A multi-locale run is analyzed as one pass per trained signal. Loading the model
        once per signal would cost a full load each time (a hundred of them for the
        largest neuron-sweep run), so the view shares the model, tokenizer and collator
        and only narrows the test split to that signal's rows and re-points
        `layer`/`index` -- which is all the single-locale analyses read.
        """
        stype, L, i = locale
        keep = lambda ex: int(ex["signal_layer"]) == L and int(ex["signal_index"]) == i
        ds = self.dataset.filter(keep)
        raw = self.raw.filter(keep)
        if len(ds) == 0:
            raise ValueError(f"{self.config.experiment_name}: no test rows for "
                             f"{locale_tag(stype, L, i)} -- the test split was built with "
                             f"a different signal set than the run trained on")
        return replace(self, dataset=ds, raw=raw, signal_type=stype, layer=L, index=i)

    @property
    def base(self):
        """The underlying HF causal LM, through the PEFT/regressor wrappers."""
        return self.model.get_base_model()

    @property
    def stats(self) -> dict:
        """Empirical activation stats of THIS run's neuron (min/max/mean/std)."""
        return self.signal_ranges[(self.layer, self.index)]


def _row_seed(ex) -> int:
    """Stable per-row seed from the row's identity. hashlib, not hash(): Python's
    str hash is salted per process, which would reintroduce exactly the
    run-to-run nondeterminism this exists to remove."""
    key = (f"{ex['text']}|{ex['signal_type']}|{ex['signal_layer']}|"
           f"{ex['signal_index']}|{ex['token_position']}")
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def compute_signal_ranges(raw) -> dict:
    """{signal_range_key: {min,max,mean,std}} from a cached split.

    Taken from the TRAIN split by every caller, because that is the distribution the
    intervention sampler was configured against during training; using test stats
    would quietly change the injected values relative to what the run saw.

    Keyed exactly as the model's sampler looks it up (signal_range_key): (layer, index)
    for MLP/SAE, but (layer, head, source offset) for attention. A cell holds ten offsets
    whose ranges differ by an order of magnitude between the BOS sink and a mid-context
    token, and grouping on (layer, head) alone pools all ten into one range -- every
    intervention would be drawn from the wrong distribution, silently, because the
    resulting numbers are still plausible. For attention_weight the log-odds moments ride
    along too: that signal lives in (0,1) and make_bounded_prob_intervention_fn samples
    in log-odds space.
    """
    signal_type = raw[0]["signal_type"] if len(raw) else None
    attention = is_attention_type(signal_type)
    group_cols = ["signal_layer", "signal_index"] + (["src_offset"] if attention else [])
    # only the columns the stats need: a cached split also carries `text` and
    # `text_input_ids`, and materializing those for 300k rows costs seconds and
    # hundreds of MB per run -- once per run, times every run in a sweep.
    cols = [c for c in group_cols + ["activation"] if c in raw.column_names]
    df = raw.select_columns(cols).to_pandas()
    if attention:
        assert "src_offset" in df and df["src_offset"].notna().all(), (
            "attention rows without src_offset -- cache predates attention signals")
    ranges = {}
    for key, vals in df.groupby(group_cols)["activation"]:
        stats = {"min": float(vals.min()), "max": float(vals.max()),
                 "mean": float(vals.mean()), "std": float(vals.std())}
        if signal_type == "attention_weight":
            stats.update(logit_moments(vals.tolist()))
        ranges[signal_range_key(*key)] = stats
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
    locales = resolve_locales(config)
    # the first signal is only the INITIAL view; multi-locale callers iterate
    # for_locale over `locales` rather than treating this as "the" locale
    signal_type, layer, index = locales[0]

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
        # The prompt template is picked by prompts.py with the GLOBAL, unseeded
        # random.choice, so which template a row gets depended on how many rows were
        # mapped before it -- and on the process's entropy-seeded RNG. Two invocations
        # of the same analysis on the same run gave different prompts, and a multi-
        # locale run (15k interleaved rows) prompted a given example differently from
        # the single-locale run (3k filtered rows): 1 of 10 rows matched. Seeding the
        # draw from the row's own identity makes each row's prompt a fixed property of
        # the row. The global RNG is restored afterwards, so nothing else sees a change.
        state = random.getstate()
        random.seed(_row_seed(ex))
        try:
            out = preprocess_example(
                ex, tokenizer=tokenizer,
                prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
                max_length=config.training_config.max_seq_length,
                return_frozen_input_len=True, return_position_valid=True,
                for_prediction=True)
        finally:
            random.setstate(state)
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
                     signal_type=signal_type, layer=layer, index=index, preprocess_fn=preprocess,
                     locales=locales)


def run_per_locale(args, runs, analysis_name: str, sentinel: str, analyze,
                   replot=None, log: str = "[run]"):
    """
    The loop every circuit analysis runs: for each run, for each of its trained
    signals, compute into <run>/<analysis_name>/[<signal>/].

    A single-locale run behaves exactly as before (results directly in
    <run>/<analysis_name>/). A multi-locale run -- one readout trained on several
    signals, as in the neuron and layer generalization sweeps -- is analyzed once per
    trained signal, each into its own subdirectory, with the model LOADED ONCE and
    shared across them (LoadedRun.for_locale).

    Why every analysis goes through here rather than keeping its own loop: before this
    existed, multi-locale runs failed to load at all. Once loading succeeds, a script
    that still used the run's FIRST signal would inject there while evaluating rows that
    name every other signal -- a silently wrong result rather than an error. One loop
    means no script can be left half-converted.

    `analyze(run_view, spec_view, out_dir)` does the actual work for one locale.
    """
    import torch
    sweeping = len(runs) > 1 or getattr(args, "group", None) is not None
    for k, spec in enumerate(runs, 1):
        base = ((getattr(args, "output_dir", None) or spec.run_dir / analysis_name)
                if not sweeping else spec.run_dir / analysis_name)
        todo = plan_locales(spec, base, sentinel, args, replot, log)
        if not todo:
            continue
        extra = (f" -- {len(todo)} of {len(spec.locales)} trained signals"
                 if spec.is_multi_locale else "")
        print(f"\n{log} ===== [{k}/{len(runs)}] {spec.name} ({spec.cell}){extra} =====")
        run = load_run_for_analysis(spec.config,
                                    checkpoint=getattr(args, "checkpoint", None) or spec.checkpoint())
        try:
            for loc, out_dir in todo:
                view = run.for_locale(loc) if spec.is_multi_locale else run
                out_dir.mkdir(parents=True, exist_ok=True)
                if spec.is_multi_locale:
                    print(f"\n{log} --- signal {view.tag}: {len(view.dataset)} test rows ---")
                analyze(view, spec.for_locale(loc), out_dir)
        finally:
            # a sweep loads one model per run; without this the previous run's weights
            # stay resident and the next checkpoint OOMs on a single GPU
            del run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
