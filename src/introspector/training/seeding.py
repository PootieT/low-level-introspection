"""
Seeding a TRAINING run without touching what it is evaluated on.

ExperimentConfig.seed fixes the evaluation (the injected eval values every interventional
eval re-seeds to, the probe splits) and is part of how a run's cache was built. Before
TrainingConfig.train_seed existed nothing seeded training at all: the LoRA A matrices,
the regression head and the added separator-token row were initialized from an unseeded
RNG, the training-time intervention sampler (`random.Random()`) was unseeded, and the HF
Trainer -- never given a seed -- shuffled data with its default 42 and re-seeded the
global RNGs (which the KL position sampler draws from) with it too.

With train_seed set, all of those follow train_seed:
  seed_training(s)          before the base model loads (embedding resize samples the
                            new row) -- python / numpy / torch / CUDA
  seed_intervention_rng(s)  on the model, once it exists (training-time injections)
  trainer_seed_kwargs(s)    into TrainingArguments: Trainer re-seeds the globals with
                            `seed` at construction and shuffles with `data_seed`
The post-training evals still re-seed their own sampler to ExperimentConfig.seed, so two
runs that differ only in train_seed are evaluated on identical values.
"""
from __future__ import annotations

from typing import Optional

import transformers


def seed_training(train_seed: Optional[int]) -> None:
    """Seed every global RNG model construction draws from. No-op for None, so runs
    configured before train_seed existed behave exactly as they did."""
    if train_seed is not None:
        transformers.set_seed(int(train_seed))


def trainer_seed_kwargs(train_seed: Optional[int]) -> dict:
    """TrainingArguments kwargs: {} for None (keeps the HF default every earlier run used)."""
    if train_seed is None:
        return {}
    return {"seed": int(train_seed), "data_seed": int(train_seed)}
