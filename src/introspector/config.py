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
    # Minimum TOKEN length (BOS included) a document must have to be kept.
    # max_seq_length only truncates; without a floor, a "length 512" cache is really
    # "whatever the corpus gave us, capped at 512". Measured on C4-en (n=10k, Llama-3.2
    # tokenizer): median 247 tokens, and only 25.1% of documents reach 512 -- so an
    # unfiltered 512 cache is ~75% short documents, and sample_token_position clamps a
    # [-1,-512] range down to whatever each document actually has. The length axis and
    # the distance-to-prompt axis both silently collapse toward the short end, with
    # nothing in the config or the logs to say so. It never bit the pos=-1 runs because
    # 98.8% of C4 clears 32 tokens.
    #
    # Set this to the LONGEST length in a length sweep (not to each arm's own length),
    # so every arm draws from the same document population: long C4 documents are
    # systematically different prose from short ones, and per-arm floors would confound
    # the length effect with a content shift.
    min_seq_length: int | None = None
    hf_data_kwargs: dict[str, Any] = {}
    offset: int = 0
    # Keep each INACTIVE (activation <= 0) row with this probability, decided by a
    # deterministic hash of (seed, signal, text) -- dataset_utils.subsample_key, stored
    # per row as `subsample_u`. Active rows are always kept. None => keep every row and
    # write no `subsample_u` column (the behaviour of every cache built before this).
    #
    # For multi-feature SAE caches: a dense text x feature table is 15M rows at 100
    # features x 150k texts, almost all of them inactive rows the trainer would throw
    # away. Because the key is per (signal, text) and thresholded, a lower keep is a
    # strict SUBSET of a higher one -- a feature's inactive rows at N=100 are a prefix
    # of its rows at N=10 -- and the rows with u < q are a uniform sample of texts at
    # rate q, active and inactive alike.
    negative_keep: float | None = None


class TrainingConfig(BaseModel):
    learning_rate: float = 1e-5
    train_batch_size: int = 8
    # micro-batches accumulated per optimizer step. The EFFECTIVE batch is
    # train_batch_size * gradient_accumulation_steps -- halve one and double the
    # other to cut activation memory without changing the optimization.
    gradient_accumulation_steps: int = 1
    eval_batch_size: int = 16
    # float, not int: a fractional epoch (0.333 on a 900k-row cache) is how a large
    # multi-neuron run is held to the SAME optimizer-update count as a small one
    num_epochs: float = 3
    max_seq_length: int = 256
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: Union[str,List[str]] = "all-linear"
    # 0.0 by default: GatedLinear used to drop lora_dropout on the floor, so every
    # gated run to date trained at dropout=0 regardless of this value. The gate now
    # applies dropout correctly, and this default preserves the old behavior.
    lora_dropout: float = 0.0
    disable_adapter_in_input: bool = False
    prompt_type: str = "prompt_no_token_index"
    precision: int = 3
    bf16: bool = True
    # objective: "generate_value" (default numeric SFT) | "detection" (binary
    # active/inactive readout for sparse SAE features -- the model emits a word,
    # scored threshold-free by the active-vs-inactive token logits => AUROC/AP).
    objective: str = "generate_value"
    # no leading space: the prompt template inserts the separator space, exactly
    # like the numeric targets (float_to_str_target has no leading space either).
    detection_active_word: str = "active"
    detection_inactive_word: str = "inactive"
    # oversample positive (feature-active) rows in the TRAIN split up to this
    # fraction so a sparse feature can't be learned as "always inactive". Eval/test
    # keep the natural distribution; metrics are AUROC/AP (imbalance-robust).
    detection_balance_pos_frac: float = 0.5
    # MULTI-FEATURE detection (trainers/train_sae_detector_joint.py). When set, the train split
    # is resampled to exactly this many rows in total, split EQUALLY across the trained
    # features (budget // N each), each feature's share balanced to
    # detection_balance_pos_frac: inactive rows subsampled, active rows repeated. None
    # => the single-feature recipe (oversample_positives), unchanged.
    detection_total_train_rows: int | None = None
    # Natural-language name per SAE feature, keyed "S<layer>.<index>" (Neuronpedia
    # explanations). Read by prompt_type="prompt_sae_label", which names the feature by
    # this label instead of its index. Carries the WHOLE pool's labels, not just the
    # trained ones, so held-out features can be prompted at eval time.
    feature_labels: dict[str, str] | None = None
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
    # "empirical_prob": one of the cell's own train-split attention weights, as-is
    # (attention_weight only; trainers/train_introspector_joint.py)
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
    # regression head: "identity" = linear (the default every earlier run used),
    # "sigmoid" = logistic probe for probability targets (SigmoidRegressionHead)
    regression_head_activation: str = "identity"
    # regression loss, read ONLY by trainers/train_introspector_joint.py (other trainers refuse a
    # non-default value): "var_norm_mse" = MSE / var(target), "norm_deviance" = the
    # Bernoulli deviance KL(p||q) / KL(p||p_bar), for a sigmoid head on probability targets.
    # Both read as ~1 - R^2, so a joint loss mixes them at a balance set by w, not by scale.
    regression_loss: str = "var_norm_mse"


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
    # Restrict train/dev/test rows to ONE signal out of a shared multi-signal cache,
    # e.g. "N10.24" (see dataset_utils.apply_signal_filter). This is what lets a
    # sweep cache a (layer, position) cell ONCE with all its neurons -- one forward
    # pass per text serves every neuron -- and still train one introspector per
    # neuron. None => use every row in the cache (all signals jointly).
    signal_filter: Optional[str] = None

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