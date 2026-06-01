# low-level-introspection
Investigating whether LMs can be finetuned to introspect low level signals, and if they can, how do they do it?

## Repo Structure

introspector/
├── README.md
├── pyproject.toml
├── configs/
│   ├── experiments/
│   └── templates/
├── scripts/
│   ├── generate_experiment_config.py
│   ├── cache_signals.py
│   └── train_introspector.py
├── src/
│   └── introspector/
│       ├── __init__.py
│       ├── config.py
│       ├── prompts.py
│       ├── datasets.py
│       ├── serialization.py
│       ├── signal_cache/
│       │   ├── __init__.py
│       │   ├── nnsight_backend.py
│       │   ├── extractors.py
│       │   └── schema.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── collators.py
│       │   └── metrics.py
│       ├── models/
│       │   ├── target_formatting.py
│       │   └── label_spaces.py
│       └── utils/
│           ├── logging.py
│           └── hf.py
│
├── data/
│   ├── raw/
│   ├── cached/
│   └── processed/
│
├── outputs/
│   ├── checkpoints/
│   ├── runs/
│   └── evals/
│
└── tests/

I am going to start a new project on finetuning hf models to introspect low-level signals (e.g. MLP activations). I think I want to start with first couple of files:
1. `generate_experiment_config.py`, this will take in arguments relevant to the experiment, and spit out a training json config, which can be consumed by trainer for reproducible experiments. Arguments should include 
- model_name (hf model), 
- dataset_name (hf dataset names), 
- signal type (mlp activation, residual stream activation, or attention scores)
- token_positions ()
- train_signal_sets (this would depend on the signal types, but for mlp activations for example, it would be specified in json as a 

2. `cache_signals.py`, this will take in the json file,