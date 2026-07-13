# low-level-introspection
Investigating whether LMs can be finetuned to introspect low level signals, and if they can, how do they do it?

## Signal Types
Currently we have the following types of signals written for training:
- MLP neuron (after non-linearity)
- Attention Score (TOOD)
- Residual Stream (TODO)

## Repo Structure

## Important Files
- `generate_experiment_config.py`, this will take in arguments relevant to the experiment, and spit out a training json
  config, which can be consumed by trainer for reproducible experiments.
- `cache_signals.py` this will take in the json file, and cache the signals into dataset files.
- `train_intorspector.py` trains introspector with regular SFT objectives
- `train_introspector_with_intervention.py` trains introspector with intervention during training.

## Example Experiments
Checkout `.sh` files in `scripts. For example:
```commandline
./scripts/train_single_mlp_introspector.sh
```
train and test on an introspector for a single neuron.

## Roadmap