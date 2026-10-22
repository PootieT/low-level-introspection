# scripts/

This directory contains the shell scripts that launch every experiment. Most run on BU's SCC,
an SGE cluster: `#$` headers, submit with `qsub`, array size in the `#$ -t` line. The SAE
scripts with `#SBATCH` headers run on a SLURM cluster: submit with `sbatch`, array size in
`--array`. All of them `cd` to (or are submitted from) the repo root and call
`python <folder>/<script>.py`.

Several scripts note their full array in a `# full run -t ...` comment, while the live
`#$ -t` line covers only the tasks being re-run. Check that line before submitting.

## Conventions for scripts in this directory

- **No `--dataset-name`.** `data_prep/generate_experiment_config.py` derives the cache key from
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

Which of these produce which paper figure: see "Example Experiments" in the top-level
`README.md`.

### `neuron/` -- MLP neurons, Llama-3.2-1B unless named

| script | what it runs |
|---|---|
| `train_single_mlp_introspector_endogenous_sweep_layer_token.sh` | endogenous only (`W=1.0`): 8 layers x 8 positions x 5 neurons x {gate on, gate off}; the SFT baseline the joint grid is compared against. `-t 1-128` |
| `train_single_mlp_introspector_joint_sweep_layer_token.sh` | MAIN GRID: 8 layers x 8 positions x 5 neurons in two cost regimes (KL budget 0.1, and unanchored for the ceiling + measured cost) = 640 runs |
| `train_single_mlp_introspector_joint_weight_sweep.sh` | joint-loss weight `W` at 3 (layer, position, KL-budget) cells x 6 weights x 5 neurons = 90 runs; fixes `W` for the main grid and gives the endogenous-vs-interventional pareto with error bars. Plot with `plotters/plot_weight_sweep.py` |
| `train_single_mlp_introspector_{endogenous,joint}_sweep_layer_token_{llama323b,llama318b}.sh` | the SFT baseline and the main grid on Llama-3.2-3B / Llama-3.1-8B: 4 layers x 4 positions x 5 neurons, at the 1B hyperparameters. `-t 1-32` |
| `train_single_mlp_introspector_joint_lr_sweep.sh` (+ `train_single_mlp_introspector_joint_lr_sweep_cache.sh`) | LR sweep for any model (`SWEEP_MODEL=`): did the 1B learning rate transfer? The `_cache` script builds its 3 caches first; `train_single_mlp_introspector_joint_lr_sweep_llama318b.sh` is the original 8B-only sweep. Read with `plotters/plot_lr_sweep.py --model <key>` |
| `train_single_mlp_introspector_joint_sweep_layer_token_<model>_hyper_tuned.sh` | the main grid at the LR the sweep picked, KL 0.1 only, 4 layers x 4 positions x 5 neurons = 80 runs, for `llama323b`, `llama318b`, `qwen3_06b`, `qwen3_17b`, `qwen3_4b`. Compare with `plotters/compare_sft_vs_joint.py --model-suffix _<model>_hyper_tuned`; scaling across models in `plotters/compare_model_scaling.py` |
| `train_single_mlp_introspector_length_generalization.sh` (+ `train_single_mlp_introspector_length_generalization_sft.sh`) | LENGTH GENERALIZATION: train on a narrow window at the end of the text, test how far back the readout reaches. 10 cells x 5 neurons, joint `W=0.01`; the `_sft` script is the `W=1.0` arm on identical data. Plot with `plotters/plot_length_generalization.py --compare mlp_lengen:Joint mlp_lengen_sft:SFT` |
| `train_single_mlp_introspector_layer_generalization.sh` | LAYER GENERALIZATION: one neuron index (picked by `data_prep/select_layer_gen_index.py`) trained on even layers 0-14, tested on odd layers 1-13. Groups via `GROUP=`. Plot with `plotters/plot_layer_generalization.py` |
| `train_single_mlp_introspector_neuron_generalization.sh` | NEURON (index) GENERALIZATION: does the readout carry neurons it never trained on, and what do extra neurons cost the trained ones? One table row per run, grouped (scaling, single, epochs, ...); add a row and bump `#$ -t`. Plot with `plotters/plot_neuron_generalization.py` |
| `score_prompt_baseline_layer_token.sh` | PROMPTED BASELINE over the main grid (320 runs, 64 tasks, no training): the untrained model asked with the grid's prompts, ridge probes on its final hidden state pooled over k=0..16 greedily generated tokens; endo + joint probes, verbal readout, no-prompt control, OOD. `MODE=lc` is the probe learning-curve pilot. Aggregate with `plotters/plot_prompt_baseline.py` |
| `score_head_knockout_joint_grid.sh` | single-head knockout over the main grid's KL 0.1 arm (320 runs, 64 tasks): which heads each evaluation needs. Plot with `plotters/plot_head_knockout.py` and `plotters/plot_knockout_head_gating.py` |
| `run_circuit_localization.sh` | one GPU job holding the other circuit analyses on the main grid (LoRA weight norms, direct attribution, injection specificity, gradient attribution, mover heads, head ablation, head taxonomy); uncomment the step to run |
| `train_single_mlp_introspector_regressor_joint_loss_prev_token_hyper.sh` | learning rate (6 arms) for joint training on one neuron at random positions in [-31,-1]; `GATE` switches the gated control arm. Mostly a playground for testing new features and manual hyperparameter tuning |

### `sae/` -- SAE features, gemma-2-2b (details in `sae/README.md`)

| script | what it runs |
|---|---|
| `precompute_sae_feature_stats.sh` | per-layer SAE feature density stats (one task per layer, skips existing); the sweep's feature picker needs these to avoid dead/punctuation features. SLURM |
| `annotate_sae_features_neuronpedia.sh` | attaches Neuronpedia labels to those stats parquets (layers 0 5 10 15 20). Plain bash |
| `generate_sae_steering_samples.sh` | blinded steered-vs-unsteered text samples for judging which features steer cleanly (the layer-9 selection). SLURM |
| `submit_sae_sweep.sh` -> `cache_sae_sweep.sh` + `train_sae_sweep.sh` (grid in `sweep_common.sh`, features in `sweep_features.txt`) | SAE LAYER x POSITION GRID: 4 layers x 4 positions x 3 features x `W` in {1.0, 0.9}, gate off, one cache per cell then 96 training tasks. Plot with `plotters/plot_sae_sweep.py`. SLURM |
| `train_sae_joint_layer9.sh` | the layer-9 reference: 5 judged features at the last token, joint gate-off training at `W` (default 0.5). Plot with `plotters/plot_sae_joint_results.py`, `plot_sae_balanced_accuracy.py`, `plot_sae_confusion.py`. SLURM |
| `train_single_sae_introspector_sweep_layer_position.sh` | SAE DETECTION grid on SCC: 8 layers x 8 positions x {gate on, gate off} = 128 cells, the features trained inside each cell; the SAE analogue of the endogenous neuron sweep |
| `train_sae_intervention_pilot.sh` | interventional SAE detection training on 3 features x 3 positions: does steering on/off during training break the bypass before the full grid is worth running |
| `build_featgen_pool_layer9.sh` | the 200-feature layer-9 POOL for feature generalization: density band + >=50 positives at the 32-token readout, Neuronpedia labels, near-duplicate labels removed, judged features first |
| `cache_sae_feature_generalization.sh` + `train_sae_feature_generalization.sh` (table in `featgen_common.sh`) | SAE FEATURE GENERALIZATION: N in {1,3,10,30,100} layer-9 features at 300k total rows, label-prompt arm (tasks 1-5) then index-prompt control (6-10), plus the label arm at 100k rows per feature for N=10/30/100 (11-13); scored on all 200 pool features (seen / held out) + a steer x ask distractor matrix. Aggregate with `plotters/plot_sae_feature_generalization.py`; see `sae/README.md` section 8 |
| `score_sae_prompt_baseline.sh` | SAE PROMPTED BASELINE: the untrained gemma-2-2b over the grid's 48 (feature, position) runs + the 5 layer-9 runs, index and label prompts (`PROMPT=`); verbal@k (paraphrase-matched words), logit, probe@k with val-chosen thresholds, no-prompt control, learning curve, steering-scale sweep. `STAGE=cache` builds the 17 caches first. Aggregate with `plotters/plot_sae_prompt_baseline.py`; see `sae/README.md` section 9 |

### `attention/` -- pre-softmax attention logits (`A`)

| script | what it runs |
|---|---|
| `pilot/pilot_attn_logit_one_cell.sh` | **run this first.** One cell of the grid at 3K samples / 2 heads, with a printed PASS/FAIL verdict on cache alignment, the injection-scale saturation table, and a wall-time extrapolation for sizing `h_rt` |
| `1-1_..._sweep_layer_srctoken.sh` + `1-2_..._sweep_layer_srctoken.sh` | ATTENTION LOGIT grid: 8 layers x 10 source offsets x 8 heads x {gate on, off} = 1280 runs, destination pinned to the last token. One script split in two array ranges (`-t 1-80`, `-t 81-160`). `HEAD_SELECT=random` (default) samples 8 heads per layer; `HEAD_SELECT=fixed` pins (0 4 8 12 16 20 24 28) -- the same file is both experiments |
| `analysis/1_attn_logit_analysis.sh` | plots the grid with `plotters/plot_attention_srctoken_sweep.py` |

### `attention_weight/` -- post-softmax attention weights (`P`), in stages

| script | what it runs |
|---|---|
| `1_scan.sh` | **run this first for `P`.** Measures the post-softmax weight over the WHOLE 16 layer x 32 head x 10 offset grid (one forward per batch) and lists the cells worth training on. Writes nothing to `data/` |
| `pilot/pilot_attn_weight_one_cell.sh` | the `P` pilot: one picked cell end to end, with verdicts on alignment, cached-target range, row-sum preservation, injection domain and saturation, plus `tests/validate_attention_weights.py` |
| `2_train_attn_weight_endogenous_picked_cells.sh` | STAGE 2, endogenous: 6 hand-picked cells ({lower, mid, upper layer} x {source near, far}) x 3 arms (gate off + KL 0.1, gate off unanchored, gate on) x 3 heads. `-t 1-6` |
| `2-0_cache_attn_weight_300K.sh` -> `2-1_..._300K.sh` | Stage 2 at 300K train / 30K test: 2-0 builds the six cell caches that 2-1, 3-1 and 4-1 share; 2-1 trains one arm per task. `-t 1-6`, then `-t 1-18` |
| `3_train_attn_weight_joint_weight_sweep.sh` | STAGE 3: joint-loss weight `W` at the 6 cells x 6 weights x 3 heads = 108 runs; the `P` analogue of the neuron weight sweep, and what fixes `W` for stage 4. `-t 1-36` |
| `3-1_..._300K.sh` / `3-2_..._unanchored_300K.sh` | Stage 3 at 300K on 2-0's caches, KL 0.1 and unanchored. Full run `-t 1-36` |
| `4_train_attn_weight_interventional_picked_cells.sh` (+ `4-1_..._300K.sh`) | STAGE 4: interventional training at the chosen `W`, 6 cells x {KL 0.1, unanchored} = 12 tasks x 3 heads. `-t 1-12` |
| `L7H4M1/{1,2,4}_*.sh` | L7 h4 @-1 only, W in {0, .01, .03, .1, .3}, one head, Stage 3 settings otherwise. 1 = sigmoid head, 2 = training injection U(0,1) in probability space (`prob_uniform`), 4 = 1 + 2 + empirical eval. Shared body in `_common.sh`. `-t 1-5`; `SMOKE=1` for a 2000-row check |
| `L7H4M1/3_empirical_logit_eval.sh` | no training: re-evaluates Stage 3's L7 h4 checkpoints with injection drawn from the cell's own log-odds histogram (`empirical_logit`). One job |
| `7-0_cache_dst_src_grid_300K.sh` | builds the nine per-DESTINATION 300K/30K caches for 7-1 (3 pinned heads x every reachable source offset each) by running 7-1 with `CACHE_ONLY=1`. `-t 1-9` |
| `7-1_train_dst_src_grid_endogenous_300K.sh` | (destination, source) PAIR GRID, endogenous, gate off unanchored: every `src <= dst` pair over absolute offsets {0 -1 -3 -7 -11 -15 -19 -23 -27 -31} (0 = last token, -31 = BOS) minus the constant (-31,-31) = 54 pairs x 3 pinned heads (L2 h20, L7 h9, L11 h25). Absolute dst `D` is token selector `D-1`, src `S` is `@(S-D)`. Holds on 7-0. `-t 1-54`; plot with `plotters/plot_attention_dst_src_grid.py` |
| `analysis/1_attn_weight_analysis.sh` | everything a Stage 2 group produced, in one GPU job: gradient attribution, then the tables and figures. `2_..._endo_60K.sh` / `2-1_..._endo_300K.sh` are thin wrappers that point it at a group |
| `analysis/2_transmission_probe.sh` | no training: does an injected weight reach the readout token at all (`mechanistic_analysis/probe_attention_weight_transmission.py`) |
| `analysis/7-1_dst_src_grid_analysis.sh` | CPU-only figures for the 7-1 pair grid; safe on a partial sweep |

### `attention_weight/paper/` -- the attention-weight experiments in the paper

The grid throughout: layers {2 6 10 14} x source offsets {-1 -3 -11 -19 -27}, 3 hand-picked
heads per cell, destination = last token, train 300K / test 30K.

| script | what it runs |
|---|---|
| `1_grid_endogenous.sh` | experiment 1: the grid, endogenous, linear readout. 20 cells x 3 heads = 60 runs. Full run `-t 1-20` |
| `1-1_grid_endogenous_logistic_norm_dev.sh` | experiment 1-1: experiment 1 with a sigmoid head and normalized-deviance loss. The SFT panel of the paper (`plotters/plot_paper_sft_panel_c.py`) |
| `2_grid_interventional.sh` | experiment 2: the grid with the joint loss, `W` in {0.01, 0.1, 0.3}, 180 runs, paired with experiment 1 |
| `2-1_..._prob_true_dist.sh` / `2-2_..._prob_uniform.sh` | experiment 2 with 1-1's readout, injection drawn from the cell's own distribution (2-1) or uniformly over [0, 1] (2-2). 2-2 at `W=0.3` is the joint-loss panel of the paper (`plotters/plot_paper_joint_loss_panel_c.py`) |
| `1-0_reeval_1_...sh`, `2-0_reeval_2_...sh`, `2-1-0_reeval_1-1_...sh` | no training: re-score finished runs of 1, 2 and 1-1 under the `empirical_prob` injection (`trainers/train_introspector_joint.py --interv-eval-only`), so they pair with 2-1 / 2-2 |
| `3-1-0_cache_multipairs_lengen.sh` -> `3-1_multipairs_lengen_endogenous_logistic_norm_dev.sh` | experiment 3-1: one run learns every (dst, src) pair of a head, across training lengths 16-128, then length generalization. Plot with `plotters/plot_multipairs_lengen.py` |
| `4-2-1_multipairs_lengen_interventional_..._prob_uniform.sh` | 3-1 with the joint loss at `W=0.3` and uniform injection |
| `5_train_attn_weight_joint_weight_sweep_unanchored_300K_..._prob_uniform.sh` | the unanchored `W` sweep of 3-2 with 2-2's readout and injection: 6 cells x 6 `W` x 3 heads |

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
  `+0`, corr ~1.0, `alignment OK`. `trainers/train_sae_detector_intervention.py` runs it BEFORE
  training and aborts on failure, because steering a misaligned position invalidates
  the whole run rather than just its diagnostics.
- **`--ood-eval`** uses the detection branch (`evaluate_ood_detection`), reporting AP /
  paired_directional_acc per corpus. A corpus where the feature never fires gives an
  undefined AP, not a bad one — filter on `base_rate_positive`.
- **`evals/score_lm_degradation.py` / `evals/score_lighteval_benchmarks.py`** are signal-agnostic
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
  attention-sink measurement is right only by accident. `data_prep/cache_signals.py` refuses an
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
  `plotters/plot_attention_srctoken_sweep.py` marks both.
- **Prompt ambiguity is measured, not assumed.** The prompt names the source token by
  string only, so a token repeated in the 32-token context does not identify a position.
  `cache_signals` records per-row `src_token_unique` / `dst_token_unique` columns in the
  cache; split on them rather than assuming the cost is zero.
- The neuron analysis scripts parse `N<layer>.<index>` out of run names and skip
  non-matching directories **without warning**, so they ignore attention runs. Use
  `plotters/plot_attention_srctoken_sweep.py`, which owns its own pattern; do not widen
  the neuron regexes in place.

## Attention-weight (`P`) specifics, on top of the above

The signal is the POST-SOFTMAX weight `p = softmax(z + mask)` at one
`(layer, head, destination, source)` cell -- a probability, and a function of the WHOLE
causal row rather than of two positions. Everything in the attention section still
applies; these are the places where being a probability changes something.

- **One custom attention implementation, two port sites.** `attn_implementation` is
  `introspection_logit` for BOTH codes; the difference is which attribute the port is
  installed under (`installed_ports(..., site="logit"|"weight")`). `port_site()` maps
  signal type to site and is the single source of truth. Routing a `P` spec through the
  logit port caches the PRE-softmax value under a `P` name and an "attention weight"
  prompt, and **raises nothing** -- the trainer now asserts the cached activations lie in
  [0,1] to catch exactly that.
- **`--precision 6`, not 3.** A weight of 0.0014 renders as the string `"+0.001"` at
  precision 3 and most cells collapse to `"+0.000"`. The regressor loss reads the float
  `activation` column, so the headline result is unaffected either way, but any
  generation-mode eval or `metrics.py` parsing path is silently gutted. Precision is part
  of the cache identity, so this forks a fresh cache -- free for a new experiment, and it
  must NOT be changed for the `A` runs, whose caches are already on disk.
- **Intervention is a row RENORMALIZATION, not an overwrite.** Setting one cell to `t`
  would leave the row summing to `1 - p_j + t`, which no logit row softmaxes to.
  `AttentionWeightPort` scales the rest by `(1-t)/(1-p_j)`, which is provably identical
  to shifting that cell's attention logit by `logit(t) - logit(p_j)` -- so `A` and `P`
  intervene in the same way and their numbers are comparable.
  `tests/validate_attention_weights.py` checks that equality numerically; run it before
  trusting any interventional `P` result.
- **Injected values are drawn in LOG-ODDS space.** All three built-in distributions leave
  `(0,1)`: `gaussian` on a mean-1e-3 cell draws negatives about half the time, and
  `extreme` is not a probability at any scale. The trainer installs
  `make_bounded_prob_intervention_fn`, which samples `u` and sets `t = sigmoid(u)`.
  **`--scale-sweep` is therefore in log-odds units** (`1,2,4,8,16`), not the `A`
  experiment's `1,3,10,30,100`. The sampler counts how often a draw hits its guard band,
  because at a large scale a rising correlation can mean the model learned the clamp.
- **KL only applies with the gate OFF**, and now says so. The anchor needs
  `anchor_region_mask`, which only `UngatedCollator` emits; with the gate on the term is
  dropped silently and the run still writes a directory named `_kl<budget>_`. It is also
  vacuous there -- the gate pins text behaviour to the base model, so KL is 0 by
  construction. Pass **`--require-kl-anchor`** on any arm that names a budget.
- **`low_variance` is judged per signal code.** `A` uses an absolute std threshold (its
  std is ~1.6); `P` uses the coefficient of variation plus a dead-cell floor on the mean,
  because a weight's std is ~1e-4 and the absolute rule would flag every cell.
- **Pick cells from the scan, do not sample them.** `mechanistic_analysis/scan_attention_weight_spread.py`
  ranks all 5120 cells by spread. Measured on C4 at length 32: the BOS offset carries a
  mean weight of ~0.65 against 0.005-0.07 elsewhere, and ~5% of cells are dead
  (`p95 < 1e-3`). Both are reasons a randomly chosen cell is usually not worth training.
- **Report Spearman next to Pearson.** `compute_metrics_from_numericals` now emits both.
  The weight distribution is heavily right-skewed and both the loss (`_var_norm_mse`) and
  Pearson are squared-scale, so a few large rows can decide the score; the gap between
  the two correlations measures how much the tail is carrying.
