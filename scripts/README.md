# scripts/

Post-fix launch scripts. Anything under `scripts_old_with_off_by_one_bug/` produced
runs where the intervention hook injected into the token AFTER the one the prompt
named, and where the live neuron matched the cached endogenous target at only ~0.81
(training tokenized without BOS; the cache tokenized with it). **Numbers from those
runs are not comparable to numbers from here.** They are kept for provenance, not for
reuse — migrate a script into this directory rather than running it in place.

## Conventions for scripts in this directory

- **No `--dataset-name`.** `generate_experiment_config.py` derives the cache key from
  the fields that determine the data plus an identity hash, so runs differing only in
  hyperparameters share one dataset and runs differing in content can never collide.
  Pass an explicit name only to reuse a cache built before this existed.
- **`--cache-dir data`** — the shared root. Pointing it at a per-group subdirectory
  re-fragments the cache and defeats the sharing.
- **`--force-drift-eval`** on train and `--diag-only` invocations. The alignment check
  it prints is the regression test for this whole class of bug: want best offset `+0`,
  corr ~0.9999, `alignment OK`.
- **Input gate via config**, not code: pass `--disable-adapter-in-input` for the gated
  arm, omit it to leave the adapter active over the input text. Scripts expose this as
  a `GATE` variable so both arms come from one file.
- **`--lora-dropout` explicitly.** `GatedLinear` used to drop dropout on the floor, so
  every gated run trained at 0 whatever the config said. It is applied now; 0.0
  reproduces the old effective behavior.
- **`--signal-filter N<layer>.<index>`** when a cell of a sweep caches several neurons
  at once. Caching 10 neurons costs one forward pass per text, so a (layer, position)
  cell should be built once and split per neuron at train time.
- **`set -e`** after env setup, so a failed config generation cannot fall through to
  training on a stale config.

## Contents

| script | what it sweeps |
|---|---|
| `neuron/train_single_mlp_introspector_regressor_joint_loss_prev_token_hyper.sh` | learning rate (6 arms) for joint training on one neuron at random positions in [-31,-1]; `GATE` switches the gated control arm |

| `neuron/train_single_mlp_introspector_endogenous_sweep_layer_token.sh` | 8 layers x 8 positions x 10 neurons, endogenous only (`W=1.0`); one array task per (layer, position) cell, 10 neurons inside it sharing one cached dataset |

Others are migrated as the experiment list for the paper firms up.
