#!/usr/bin/env python3

"""
Schema classes for experiments, datasets, etc
"""
from pathlib import Path
from typing import List, Any, Optional, Union

from pydantic import BaseModel

from .signals.schema import SignalSpec, TokenSelector


class DatasetConfig(BaseModel):
    name: str
    split: str
    text_field: str = "text"
    max_samples: int | None = None
    max_seq_length: int = 128
    hf_data_kwargs: dict[str, Any] = {}
    offset: int = 0


class TrainingConfig(BaseModel):
    learning_rate: float = 1e-5
    train_batch_size: int = 8
    eval_batch_size: int = 16
    num_epochs: int = 3
    max_seq_length: int = 256
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: Union[str,List[str]] = "all-linear"
    lora_dropout: float = 0.05
    disable_adapter_in_input: bool = False
    prompt_type: str = "prompt_no_token_index"
    precision: int = 3
    bf16: bool = True
    # objective: "generate_value" (default numeric SFT) | "detection" (binary
    # active/inactive readout for sparse SAE features -- the model emits a word,
    # scored threshold-free by the active-vs-inactive token logits => AUROC/AP).
    objective: str = "generate_value"
    detection_active_word: str = " active"
    detection_inactive_word: str = " inactive"
    # oversample positive (feature-active) rows in the TRAIN split up to this
    # fraction so a sparse feature can't be learned as "always inactive". Eval/test
    # keep the natural distribution; metrics are AUROC/AP (imbalance-robust).
    detection_balance_pos_frac: float = 0.5
    # LR schedule
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"   # linear | cosine | constant | ...
    # checkpoint frequency: 0 => per epoch; >0 => every save_steps (a value <1 is a
    # ratio of total steps, e.g. 0.2 => 5 checkpoints/epoch for mid-run diagnostics)
    save_steps: float = 0.0
    # distribution used to sample the random value injected at intervention time
    # "uniform": U(min, max) of the neuron's empirical range
    # "gaussian": N(mean, std) fit to the neuron's empirical activations
    # "extreme": U over a range intervention_extreme_scale x wider than empirical (sanity check)
    intervention_signal_distribution: str = "uniform"
    # how much wider than the empirical range to sample from when
    # intervention_signal_distribution == "extreme"
    intervention_extreme_scale: float = 1000.0
    # anneal the intervention spread (range/std multiplier) from scale_start ->
    # scale_end over the training window [start_frac, end_frac]; scale_end=1 lands
    # on the endogenous-matched distribution the model must generalize to.
    intervention_anneal: bool = False
    intervention_scale_start: float = 100.0
    intervention_scale_end: float = 1.0
    intervention_anneal_schedule: str = "log"   # log | linear | cosine
    intervention_anneal_start_frac: float = 0.0
    intervention_anneal_end_frac: float = 1.0


class EvalConfig(BaseModel):
    # generation configs
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0

class ExperimentConfig(BaseModel):
    experiment_name: str
    created_at: str
    model_name: str
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    seed: int = 42

    train_dataset: DatasetConfig
    dev_dataset: DatasetConfig
    test_dataset: DatasetConfig

    train_signals: List[SignalSpec]
    dev_signals: List[SignalSpec]
    test_signals: List[SignalSpec]

    train_token_selector: TokenSelector
    dev_token_selector: TokenSelector
    test_token_selector: TokenSelector

    use_regressor: bool=False

    training_config: TrainingConfig
    eval_config: EvalConfig = EvalConfig(
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )

    cache_dir: str
    output_dir: str
    # identity of the cached signal dataset. Runs that share the SAME data (model,
    # dataset, signal, token selector, seq length, precision) can set the same
    # dataset_name to reuse one cache instead of re-caching per experiment_name.
    # None => falls back to experiment_name (backward compatible).
    dataset_name: Optional[str] = None

    @property
    def dataset_cache_dir(self) -> Path:
        """Where the cached signal dataset lives (keyed by dataset_name if set)."""
        return Path(self.cache_dir) / (self.dataset_name or self.experiment_name)