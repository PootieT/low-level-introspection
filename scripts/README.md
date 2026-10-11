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
| `neuron/score_prompt_baseline_layer_token.sh` | PROMPTED BASELINE over the main grid (320 runs, 64 tasks, no training): the untrained model asked with the grid's prompts, ridge probes on its final hidden state pooled over k=0..16 greedily generated tokens; endo + joint probes, verbal readout, no-prompt control, OOD. `MODE=lc` is the probe learning-curve pilot. Aggregate with `analysis/plot_prompt_baseline.py` |
| `sae/precompute_sae_feature_stats.sh` | per-layer SAE feature density stats (one task per layer, skips existing); the sweep's feature picker needs these to avoid dead/punctuation features |
| `sae/train_single_sae_introspector_sweep_layer_position.sh` | SAE DETECTION grid: 8 layers x 8 positions x 5 features x {gate on, gate off} = 128 cells; the SAE analogue of the endogenous neuron sweep |
| `sae/train_sae_intervention_pilot.sh` | interventional SAE detection training on 3 features x 3 positions: does steering on/off during training break the bypass before the full grid is worth running |
| `sae/build_featgen_pool_layer9.sh` | the 200-feature layer-9 POOL for feature generalization: density band + >=50 positives at the 32-token readout, Neuronpedia labels, near-duplicate labels removed, judged features first |
| `sae/cache_sae_feature_generalization.sh` + `sae/train_sae_feature_generalization.sh` (table in `sae/featgen_common.sh`) | SAE FEATURE GENERALIZATION: N in {1,3,10,30,100} layer-9 features at 300k total rows, label-prompt arm (tasks 1-5) then index-prompt control (6-10), plus the label arm at 100k rows per feature for N=10/30/100 (11-13); scored on all 200 pool features (seen / held out) + a steer x ask distractor matrix. Aggregate with `analysis/plot_sae_feature_generalization.py`; see `sae/README.md` section 8 |
| `sae/score_sae_prompt_baseline.sh` | SAE PROMPTED BASELINE: the untrained gemma-2-2b over the grid's 48 (feature, position) runs + the 5 layer-9 runs, index and label prompts (`PROMPT=`); verbal@k (paraphrase-matched words), logit, probe@k with val-chosen thresholds, no-prompt control, learning curve, steering-scale sweep. `STAGE=cache` builds the 17 caches first. Aggregate with `analysis/plot_sae_prompt_baseline.py`; see `sae/README.md` section 9 |

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
