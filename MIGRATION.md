# Migrating a branch from the old layout

On 2026-10-21 (commit `b409457`) every root-level script moved into a folder named for its
purpose, and `analysis/` became `plotters/`. The last commit with the old layout is `849be90`.
If your branch started before that, this page shows how to bring it across.

## Steps

1. **Merge `main` into your branch** (or rebase onto it). Git detects the renames, so your
   edits to a moved file follow it to its new path. Conflicts show up only where `main`
   changed the same lines you did. In practice that means the import block at the top of
   a moved file: keep `main`'s import lines, then re-apply your own changes below them.
2. **Rewrite references to old paths:**
   ```bash
   python utils/migrate_old_paths.py            # dry run
   python utils/migrate_old_paths.py --apply
   ```
   This fixes `python train_introspector_joint.py ...` in your shell scripts,
   `from train_introspector import ...` in your Python files, and `analysis/...` in your
   docs. It then lists anything it cannot fix: references to removed scripts, and new
   `.py` files sitting at the repo root.
3. **Move any new root-level script into its folder**, using the table below. Add the
   repo-root line after its standard-library imports, as every moved script already has:
   ```python
   sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
   ```
   Imports across folders use the full name: `from trainers.train_introspector import ...`.
4. **Check:**
   ```bash
   python -m pytest tests/test_repo_layout.py
   ```
   This fails on any import that doesn't resolve (including imports inside functions), on
   any `python <path>.py` that points at a missing file, and on any script left at the
   repo root.
5. **On the cluster, let queued jobs finish before you pull.** SGE and SLURM keep a copy
   of each job script from submission time. A task still waiting in the queue will run
   `python train_introspector_joint.py`, which no longer exists after the pull.

Run every script from the repo root as `python <folder>/<script>.py`. Nothing else about
how scripts are invoked has changed.

## Where things went

| Old | New | What it holds |
|---|---|---|
| `analysis/*.py` | `plotters/*.py` | figures and tables from finished runs |
| `train_*.py` | `trainers/` | training |
| `cache_signals.py`, `generate_experiment_config.py`, `select_layer_gen_index.py`, `select_sae_features.py`, `build_sae_featgen_pool.py`, `fetch_sae_feature_labels.py`, `annotate_sae_features_neuronpedia.py`, `generate_sae_steering_samples.py` | `data_prep/` | configs, signal caches, choosing neurons / SAE features |
| `score_length_generalization.py`, `score_lighteval_benchmarks.py`, `score_lm_degradation.py`, `score_neuron_generalization.py`, `score_ood_generalization.py`, `score_prompt_baseline.py`, `score_sae_prompt_baseline.py`, `score_sae_feature_generalization.py` | `evals/` | evaluating trained introspectors |
| `score_gradient_attribution.py`, `score_head_ablation.py`, `score_head_knockout.py`, `score_injection_specificity.py`, `score_neuron_bypass_ceiling.py`, `score_neuron_redundancy.py`, `trace_introspection_circuit.py`, `trace_mover_head.py`, `ablate_lora_modules.py`, `attribute_attention_weight_readout.py`, `probe_attention_weight_transmission.py`, `scan_attention_weight_spread.py` | `mechanistic_analysis/` | how the introspector works |
| `validate_attention_logits.py`, `validate_attention_weights.py`, `validate_sae_hook.py`, `probe_cache_alignment.py` | `tests/` | model-backed sanity checks (scripts, not pytest tests) |

`src/`, `scripts/` and `utils/` stayed where they were. The `scripts/*/analysis/` folders
(shell drivers) also kept their names.

## Removed

| Old | Use instead |
|---|---|
| `intervene_introspector.py` | Every trainer runs its own interventional eval. To re-score an existing checkpoint, use `trainers/train_introspector_joint.py --interv-eval-only`. |
| `score_rank_metrics.py` | The default eval writes `spearman`, `kendall_tau`, `monotone_r2` and `spearman_macro` into each metrics JSON. Runs evaluated before 2026-10-12 don't have these keys. |
| `utils/relocate_wrong_model_runs.py` | — |
| `scripts_old_with_off_by_one_bug/`, `outputs_old_with_off_by_one_bug/` | Runs from before the BOS / off-by-one fix; their numbers aren't comparable to current runs. |

To recover any removed file, take it from the last commit with the old layout:
`git show 849be90:<old path>`.

## Behaviour changes made in the same refactor

- `plotters/plot_lora_weight_pattern.py` now takes `--checkpoint` / `--output-dir` /
  `--highlight-layer`, as its usage line always said. Before, it ignored its arguments
  and ran on a hard-coded cluster checkpoint.
- `trainers/train_introspector_joint.py` now writes `spearman_macro` / `spearman_macro_std`
  next to `corr_macro` for runs with more than one signal.
