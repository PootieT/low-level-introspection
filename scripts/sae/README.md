# SAE-feature introspection — handoff

Companion track to the neuron (MLP) experiments. Same question — *can a model be
trained to report its own low-level internal signals, and is the report faithful or a
context bypass?* — but the target is a **Gemma Scope SAE feature** on
`google/gemma-2-2b` instead of an MLP neuron on Llama-3.2-1B.

Everything here is post-BOS-fix. The conventions in `../README.md` (shared cache keys,
`--signal-filter`, `--force-drift-eval`, gate-via-config, explicit `--lora-dropout`,
`set -e`) apply unchanged; this file only covers what is SAE-specific.

---

## 1. Why this track looks different from the neuron track

**The readout is binary detection, not magnitude regression.** SAE features are sparse
(~0.1–1% of tokens). At a fixed token position a magnitude target is ~all zeros, so a
regressor minimises loss by predicting 0 and every correlation-style metric is
meaningless. So the model instead emits one of two words — `active` / `inactive` — in
its own tokens, and we score:

| metric | why |
|---|---|
| **AP** (average precision) | the headline. Imbalance-honest at a ~1% base rate. |
| AUROC | secondary, threshold-free. |
| `f1_at_0`, `acc_at_0` | at the natural threshold (logit gap = 0). |
| `base_rate_positive` | context. **A 99% accuracy at a 1% base rate means nothing.** |

Scoring never samples generation: we do one forward pass and take the
`logit(" active") − logit(" inactive")` gap at the readout position. That gap is the
continuous score AP/AUROC are computed from.

**The steering primitive is different.** A neuron is a module output you can overwrite.
An SAE feature is not a module — it is `ReLU(W_enc·resid + b_enc)` computed on the
residual stream. To set it we clamp via *difference-of-decode*:

```
resid += decode(feats with feature := target) − decode(feats)
```

which preserves the SAE reconstruction error and moves only that feature's
contribution (`SAEBundle.steer_delta`, applied by `SAEInterventionHook`).

---

## 2. Where the code lives

### Library (shared, SAE-aware)
| file | role |
|---|---|
| `src/introspector/signals/sae.py` | `SAEBundle`: loads a Gemma Scope SAE through `sae_lens`, applies it to HF residuals. `encode` / `decode` / `steer_delta`. |
| `src/introspector/signals/schema.py` | `SAEFeatureSpec` (`type="sae"`, plus `sae_release` / `sae_site` / `sae_width`, which are part of the cache identity hash). |
| `src/introspector/signals/resolution.py` | `S<layer>.<index>` → `SAEFeatureSpec` (the `S` prefix, alongside `N`/`A`/`R`). |
| `src/introspector/models/signal_paths.py` | `resolve_sae_hook_path` — the residual site the SAE was trained on (decoder-layer output ≡ TransformerLens `blocks.L.hook_resid_post`). **Capture and steering must use this same site.** |
| `src/introspector/training/model.py` | `SAEInterventionHook` — the steering hook. |
| `src/introspector/prompts.py` | `sae_prompt` (the introspection prefill). |

### Entry points
| file | role |
|---|---|
| `validate_sae_hook.py` | **Gate 0.** Is the SAE being applied to the right residual? |
| `select_sae_features.py` | Streaming density pass → per-layer feature-stats parquet. |
| `cache_signals.py` | Shared cacher; the `sae` branch saves the residual in-trace and encodes after. |
| `train_sae_detector.py` | **Endogenous** detection training + all post-hoc evals. |
| `train_sae_detector_intervention.py` | **Interventional** detection training (label = the steered state). |
| `analysis/plot_sae_detection_sweep.py` | Plots a sweep directory. |
| `score_ood_generalization.py` | `evaluate_ood_detection` (detection branch of the OOD ladder). |
| `score_lm_degradation.py`, `score_lighteval_benchmarks.py` | Signal-agnostic; **gate-off checkpoints only.** |

---

## 3. Run order

### Step 0 — validate the SAE hook (once per model × SAE release × layer family)

```bash
python validate_sae_hook.py --model google/gemma-2-2b \
    --sae-release gemma-scope-2b-pt-res-canonical \
    --layer 9 --width 16k --site res --n-texts 64 --max-seq-length 64
```

**Everything downstream is garbage if this fails.** Pass criteria:
- printed HF hook path and the SAE's recorded `hook_name` describe the same site
  (`model.layers.9.output` ↔ `blocks.9.hook_resid_post`);
- **FVU ≪ 0.35** and **L0 in the tens** — both computed with **BOS excluded**;
- top-activating features look like real concepts (cross-check ids on Neuronpedia).

> Gemma-2 parks a massive-activation attention sink on `<bos>` (residual norm ~1500 vs
> ~140 elsewhere). If BOS is left in, FVU comes out **> 1** and every "top feature" is
> the sink. That is expected and already handled — mentioned so a re-run of an older
> script does not send you chasing it.

### Step 1 — per-layer feature stats

```bash
qsub scripts/sae/precompute_sae_feature_stats.sh          # 8 tasks, one per layer
```

Writes `data/sae_stats/gemma2-2b_res{L}_16k.parquet`. Skips layers already done.
Selection keeps `density ∈ [1e-3, 1e-2]`; the upper bound mirrors Neuronpedia's default
"hide >1% density", which is what keeps punctuation/position features out.

**Required before Step 2** — without a parquet the sweep's picker silently falls back to
uniform-random over 16k features, most of which are dead at a fixed position.

### Step 2 — the main endogenous grid

```bash
JID=$(qsub scripts/sae/precompute_sae_feature_stats.sh | awk '{print $3}' | cut -d. -f1)
qsub -hold_jid $JID scripts/sae/train_single_sae_introspector_sweep_layer_position.sh
```

8 layers × 8 positions × {gate on, gate off} = **128 array tasks**, 5 features trained
inside each → 640 runs. Per task: generate config → `cache_signals.py` → 5 ×
`train_sae_detector.py`.

- **Features are density-stratified and seeded by LAYER only**, so a layer's features are
  identical at every position → the position curve is paired per feature. `sample` index
  is a monotone density rank (0 = rarest → 9 = most frequent).
- **One dataset per (layer, position) cell** holds all 10 features (encoding 10 features
  costs one forward pass per text); `--signal-filter S<layer>.<index>` splits it per run.
- **GATE is the outer array dimension** (tasks 1–64 on, 65–128 off) so both arms of a cell
  never race to build the same cache directory.

### Step 3 — interventional pilot (can run concurrently; separate cache group)

```bash
qsub scripts/sae/train_sae_intervention_pilot.sh          # 3 features x 3 positions
```

Trains with steering: the on/off state is drawn 50/50 **independently of the text** and
the label *is* that steered state, so context cannot predict it. This is the
anti-bypass condition. Run this before committing to a full interventional grid.

### Step 4 — plots

```bash
python analysis/plot_sae_detection_sweep.py    # edit the path at the bottom
```

Writes to `outputs/plots/<sweep_name>/`, per split (`test`, `test_interventional`;
`dev` is skipped):

- `test__overview_feature.png` — one line per feature (busy but complete)
- `test__overview_layer_x_gate.png` — per layer, **mean ± 1 SE**, gate as line style
- `test__overview_sample_x_gate.png` — per density bin, mean ± SE
- `test__overview_gate.png` — **the headline arm comparison**

`plot_all_splits(path, individual_metrics=True)` additionally writes one PNG per metric.

---

## 4. How to read the results

**The alignment line comes first.** Every cell's `SAMPLE=0` run prints:

```
=== SAE CACHE ALIGNMENT (adapters off) ===
  offset +0: corr(live, cached) = +0.9999   <-- train assumes this
  best offset = +0  ->  alignment OK
```

`best offset = +0`, corr ≈ 1.0, `token_match` ≈ 1.0. **Anything else and the run's
interventional numbers are void** — it means the steering hook is injecting into a
different token than the prompt names (the off-by-one class of bug, now guarded).
Also saved as `sae_cache_alignment.json`.

**Then the two arms.** With **gate ON**, no trainable parameter touches the introspected
token, so the model *cannot* read the feature — whatever AP it scores is context bypass,
and that is the number to subtract. With **gate OFF** the adapter reaches the site and
causal reading becomes possible.

**Then faithfulness.** `test_interventional.paired_directional_acc` = fraction of contexts
where steering the feature ON raises the active-vs-inactive gap relative to steering it
OFF, on the *same* context.

| value | meaning |
|---|---|
| → 1.0 | the report causally tracks the feature — faithful |
| ≈ 0.5 | no causal effect: the report rides context (**bypass** — what we see endogenously, matching the neuron result**)** |
| < 0.5 | systematic inverse effect; most likely an out-of-distribution steering artifact, not genuine anti-reading |

Outputs per run: `sae_detection_metrics.json` (endogenous trainer),
`sae_intervention_train_metrics.json` (interventional trainer),
`sae_cache_alignment.json`, `ood_generalization_detection.csv`,
`sae_interventional_scores.png`.

---

## 5. Gotchas

- **A single-class eval set gives undefined AP/AUROC, not bad ones.** A feature that never
  fires on a corpus/position reports `nan`. Filter on `base_rate_positive` before
  concluding "generalization failure" — same trap as the near-silent-neuron caveat in the
  neuron OOD script.
- **`score_lm_degradation.py` / `score_lighteval_benchmarks.py` are gate-off only.** With
  the gate on, the adapter never ran over ordinary text during training, so a plain PEFT
  forward describes a model that does not exist. The scripts refuse gate-on checkpoints
  unless `--force`; the sweep only invokes them on the gate-off arm.
- **`--signal-filter` is not part of the cache identity** — that is deliberate (it is what
  lets 10 features share one dataset). It does mean that if you forget it on a
  multi-feature config, the trainer aborts rather than guessing which feature to steer.
- **`extreme` at scale 1 == uniform over the feature's empirical active range.** The name
  is kept for consistency with the neuron scripts; `uniform` would be the better name at
  publication-refactor time. Raise `--intervention-extreme-scale` (10, 100) for
  larger-than-natural injections. Because SAE activations are non-negative, "on" draws are
  floored at the smallest observed active value so an `active` label is always genuinely
  active — unlike the neuron sampler, which is symmetric around the midpoint.
- **Disk**: 640 runs × checkpoints. `StripEmbeddingSaveCallback` already strips the frozen
  embedding rows; watch the quota anyway on the first full grid.
- **Compute**: gemma-2-2b × 5 runs/task (each caches 300K + trains 1 epoch + evals) at
  `h_rt=18:00:00`. If tasks time out, lower `--train-max-samples` or `N_FEAT_TRAIN`.

---

## 6. Open question this track is meant to answer

The neuron track found that introspection reports do **not** causally track the signal at
earlier token positions — intervention does not move the prediction — i.e. the readout is
a context bypass. The SAE track asks whether that is a property of *the target* (an
arbitrary, uninterpretable neuron the model has no reason to propagate) or of *the
mechanism* (attention cannot transport an arbitrary internal scalar to a later readout).

Two things change here: the target is a semantically meaningful, load-bearing direction
the model actually uses, and the task is **detection** rather than magnitude — which plays
to what attention is structurally good at (identity/retrieval) rather than what it is bad
at (reading a scalar magnitude). If detection stays faithful at early positions where
neuron magnitude reading failed, that is the interesting positive result. If it collapses
the same way, the wall is about transport, not about the target — also a clean finding.
