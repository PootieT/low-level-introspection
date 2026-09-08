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

| script | what it sweeps                                                                                                                                                                                                                  |
|---|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `neuron/train_single_mlp_introspector_regressor_joint_loss_prev_token_hyper.sh` | learning rate (6 arms) for joint training on one neuron at random positions in [-31,-1]; `GATE` switches the gated control arm; this is mostly a playground script to test out new features, and manual hyper-parameter tunning |
| `neuron/train_single_mlp_introspector_endogenous_sweep_layer_token.sh` | endogenous only (`W=1.0`): 8 layers x 8 positions x 5 neurons x {gate on, gate off}; the SFT baseline the joint grid is compared against |
| `neuron/train_single_mlp_introspector_joint_weight_sweep.sh` | joint-loss weight `W` at 3 (layer, position, KL-budget) cells x 5 neurons = 90 runs; fixes `W` for the main grid and gives the endogenous-vs-interventional pareto with error bars |
| `neuron/train_single_mlp_introspector_joint_sweep_layer_token.sh` | MAIN GRID: 8 layers x 8 positions x 5 neurons in two cost regimes (KL budget 0.1, and unanchored for the ceiling + measured cost) = 640 runs |
| `sae/precompute_sae_feature_stats.sh` | per-layer SAE feature density stats (one task per layer, skips existing); the sweep's feature picker needs these to avoid dead/punctuation features |
| `sae/train_single_sae_introspector_sweep_layer_position.sh` | SAE DETECTION grid: 8 layers x 8 positions x 5 features x {gate on, gate off} = 128 cells; the SAE analogue of the endogenous neuron sweep |
| `sae/train_sae_intervention_pilot.sh` | interventional SAE detection training on 3 features x 3 positions: does steering on/off during training break the bypass before the full grid is worth running |
| `attention/train_single_attn_logit_introspector_endogenous_sweep_layer_srctoken.sh` | ATTENTION LOGIT grid: 8 layers x 10 source offsets x 8 heads x {gate on, off} = 1280 runs, destination pinned to the last token. `HEAD_SELECT=random` (default) samples 8 heads per layer; `HEAD_SELECT=fixed` pins (0 4 8 12 16 20 24 28) -- the same file is both experiments |
| `attention/pilot/pilot_attn_logit_one_cell.sh` | **run this first.** One cell of the above at 3K samples / 2 heads, with a printed PASS/FAIL verdict on cache alignment, the injection-scale saturation table, and a wall-time extrapolation for sizing `h_rt` |

Others are migrated as the experiment list for the paper firms up.

## SAE-specific notes

SAE features are sparse (~1% density), so the readout is BINARY detection
("active"/"inactive" in the model's own tokens), not magnitude regression: at a fixed
position a magnitude target is ~all zeros and a regressor wins by predicting 0. Score
with **AP** (imbalance-honest) and AUROC; pooled accuracy is meaningless at a 1% base
rate. The same conventions above apply, plus:

- **`--signal-filter S<layer>.<index>`** (note the `S`) selects one feature out of a
  cell's shared cache. The SAE trainers resolve the steering target THROUGH this
  filter, so training one feature while steering another cannot happen silently.
- **`--force-drift-eval`** runs `measure_sae_alignment`: recomputes the feature at the
  cached position with adapters off and reports corr per offset. Want best offset
  `+0`, corr ~1.0, `alignment OK`. `train_sae_detector_intervention.py` runs it BEFORE
  training and aborts on failure, because steering a misaligned position invalidates
  the whole run rather than just its diagnostics.
- **`--ood-eval`** uses the detection branch (`evaluate_ood_detection`), reporting AP /
  paired_directional_acc per corpus. A corpus where the feature never fires gives an
  undefined AP, not a bad one — filter on `base_rate_positive`.
- **`score_lm_degradation.py` / `score_lighteval_benchmarks.py`** are signal-agnostic
  and apply unchanged, but only to **gate-off** checkpoints.

## Attention-specific notes

The signal is the PRE-SOFTMAX attention logit, `q.k/sqrt(d_head)` after RoPE and before
the causal mask and the softmax, at one `(layer, head, destination token, source token)`
cell. Signals are named **`A<layer>.<head>@<src_offset>`** (e.g. `A10.5@-3`), where the
offset is the source token's distance behind the destination token and the destination
comes from the token selector as usual. `P<layer>.<head>@<src>` is reserved for the
post-softmax weight. The same conventions as above apply, plus:

- **`--min-seq-length` must equal `--max-seq-length`** (32 in the sweep). The deepest
  offset is meant to be BOS, and `sample_token_position` CLAMPS a destination that a
  short document cannot support -- so without a floor, offset -31 silently resolves to
  position 0 on a short document and to an ordinary token on a long one, and the
  attention-sink measurement is right only by accident. `cache_signals.py` refuses an
  attention config whose offsets the floor cannot honour. Note the neuron sweeps set no
  floor, so the two families do not draw from an identical document population.
- **`--signal-filter A<layer>.<head>@<src>` must include the offset.** A cell caches 8
  heads at one offset and each run trains one of them; a filter without `@<src>` is
  rejected rather than matching every offset in the cell.
- **`--force-drift-eval` runs `measure_attention_alignment`**, the attention analogue of
  the neuron drift check. Want best offset `+0`, corr ~0.9999, `alignment OK`. This is
  the ONLY automatic check that the live logit is the quantity `cache_signals` wrote: a
  failure here usually means caching and training ran under different attention
  implementations (compare `capture.attn_implementation` in the cache's `metadata.json`)
  or the `(dst, src)` pair is off by one.
- **The attention implementation is chosen from the config's signal types**, not by a
  flag: attention runs load with `attn_implementation="introspection_logit"` (an eager
  variant carrying the read/write port), neuron and SAE runs keep loading under sdpa and
  their numbers do not move. Both cache and trainer derive it the same way, so they
  cannot disagree unless one of them is edited.
- **`--intervention-extreme-scale` does not mean the same thing here.** `extreme` draws
  uniformly from `scale x` the empirical RANGE, so a logit range of ~30 at scale 10
  injects +/-150 -- past which the softmax saturates and different injected values give
  the SAME forward pass, i.e. a value the model cannot report even in principle. At
  `W=1.0` the flag is inert (no interventional training; `evaluate_interventional_test`
  forces gaussian at scale 1.0) but it does reach `interventional_scale_sweep`. Read that
  sweep before choosing a scale for any joint (`W<1`) follow-up.
- **Reading the grid.** Layer 0 is a text-only CEILING CONTROL (its logit is a function of
  embeddings and RoPE alone, hence determined by the two token strings the prompt already
  names), and the BOS offset is a separate regime whose small variance can make corr
  meaningless -- check the per-signal `std` in `dataset_signal_stats.json`.
  `analysis/plot_attention_srctoken_sweep.py` marks both.
- **Prompt ambiguity is measured, not assumed.** The prompt names the source token by
  string only, so a token repeated in the 32-token context does not identify a position.
  `cache_signals` records `src_token_unique_rate` in the cache metadata; split on it
  rather than assuming the cost is zero.
- The neuron analysis scripts parse `N<layer>.<index>` out of run names and skip
  non-matching directories **without warning**, so they ignore attention runs. Use
  `analysis/plot_attention_srctoken_sweep.py`, which owns its own pattern; do not widen
  the neuron regexes in place.
