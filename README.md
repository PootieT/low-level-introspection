# low-level-introspection
Investigating whether LMs can be finetuned to introspect low level signals, and if they can, how do they do it?

## Signal Types
Currently we have the following types of signals written for training:
- MLP neuron (after non-linearity)
- Attention logit (pre-softmax `q.k/sqrt(d_head)`, after RoPE) at one
  `(layer, head, destination token, source token)` cell -- named `A<layer>.<head>@<src_offset>`
- Attention weight (post-softmax, code `P`): spec/parse/identity plumbing is in place,
  extraction reads the same port one step later (not wired yet)
- SAE feature
- Residual Stream (TODO)

## Repo Structure
Run everything from the repo root, as `python <folder>/<script>.py`.

| Folder | What is in it |
|---|---|
| `data_prep/` | build experiment configs, cache signals into datasets, pick neurons / SAE features |
| `trainers/` | train introspectors (SFT, with intervention, joint) and SAE detectors |
| `evals/` | evaluate trained introspectors: length / OOD / neuron generalization, LM degradation, benchmarks, prompted baselines |
| `mechanistic_analysis/` | how the introspector works: circuit tracing, head knockout / ablation, attribution, probes |
| `plotters/` | figures and tables from finished runs |
| `src/introspector/` | shared library: configs, signal extraction, models, training utilities |
| `scripts/` | shell drivers for every experiment (index in `scripts/README.md`) |
| `tests/` | unit and smoke tests (`python -m pytest tests/`), plus model-backed sanity checks (`tests/validate_*.py`) |
| `utils/` | housekeeping: checkpoint cleanup, relocating runs |

## Important Files
- `data_prep/generate_experiment_config.py`, this will take in arguments relevant to the experiment, and spit out a training json
  config, which can be consumed by trainer for reproducible experiments.
- `data_prep/cache_signals.py` this will take in the json file, and cache the signals into dataset files.
- `trainers/train_introspector.py` trains introspector with regular SFT objectives
- `trainers/train_introspector_with_intervention.py` trains introspector with intervention during training.

## Example Experiments
Checkout `.sh` files in `scripts/`; `scripts/README.md` lists what each one runs.

## Roadmap