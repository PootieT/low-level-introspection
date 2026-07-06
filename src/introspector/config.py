#!/usr/bin/env python3

"""
Schema classes for experiments, datasets, etc
"""
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