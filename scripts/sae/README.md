# SAE-feature introspection — current pipeline

Companion to the neuron track. Same question — *can a model be trained to report its own
low-level internal signals, and is the report faithful or a context bypass?* — but the
target is a **Gemma Scope SAE feature** on `google/gemma-2-2b` instead of an MLP neuron.

This file documents the **current methodology**: gate OFF everywhere, LM-head readout,
and one joint trainer that spans endogenous + interventional. The shared conventions in
`../README.md` (cache keys, `--signal-filter`, gate-via-config, `set -e`) apply unchanged.

---

## 1. What is different from the neuron track

**The readout is binary detection, not magnitude regression.** SAE features are sparse
(~0.1–1% of tokens), so a magnitude target is ~all zeros and correlation metrics are
meaningless. The model instead answers in its own words: at the readout position we take
the token it actually **generates** — argmax over the full vocabulary, one greedy forward
— and check whether it is `active`, `inactive`, or neither. The prediction is that
generated token, not a logit comparison between the two words.

**We report balanced accuracy + precision** on the generated token, for both the
endogenous readout and the interventional test — those two metrics only. AUROC/AP and the
two-word logit gap are **not** used.

**The steering primitive is difference-of-decode.**

*The problem.* With a neuron, "set it to X" is easy — a neuron is a number the model
literally computes, so you overwrite it. An SAE feature is **not** something the model
computes. The model computes its residual-stream vector `resid`; a *separate* little
autoencoder (the SAE) reads that vector and decomposes it into ~16k features, where
feature *i*'s value is `ReLU(W_enc·resid + b_enc)`. So the feature is derived *from* the
residual — there is no slot inside the model to overwrite.

*The recipe.* To force feature *i* to a target value:
1. `feats = encode(resid)` — read the current feature values;
2. make a copy with just that one feature changed to the target;
3. take the difference of decodes: `delta = decode(feats_with_target) − decode(feats)`;
4. `resid += delta`.

(`SAEBundle.steer_delta`, applied by `SAEInterventionHook`.)

*Why the difference, not just `resid = decode(feats_with_target)`?* The SAE is lossy —
`decode(encode(resid)) ≈ resid`, not exactly. Replacing the residual with a fresh decode
would discard that reconstruction error and disturb *every* direction at once. Adding only
the difference cancels the reconstruction error (it appears in both decode terms), so the
net change is exactly feature *i*'s decoder direction scaled to hit the target — **only
that one feature moves in SAE coordinates; everything else is left untouched.** (Re-encoding
the steered residual does *not* return the target, because encoder and decoder are not
inverses; see "What the interventional test measures" in §6.)

**The gate is OFF.** The LoRA adapter runs over the whole sequence, so the readout can
causally read the feature at its origin. Because nothing else then preserves language
modelling over the input text, training adds an **adaptive KL anchor** to the base model
on the text region (`--kl-anchor-target 0.1`, matching the neuron track).

---

## 2. Where the code lives

### Library (shared, SAE-aware)
| file | role |
|---|---|
| `src/introspector/signals/sae.py` | `SAEBundle`: loads a Gemma Scope SAE via `sae_lens`; `encode`/`decode`/`steer_delta`. |
| `src/introspector/signals/schema.py` | `SAEFeatureSpec` (+`sae_release`/`sae_site`/`sae_width`), `SAEFeatureSampler`. |
| `src/introspector/signals/resolution.py` | `S<layer>.<index>` → `SAEFeatureSpec`. |
| `src/introspector/signals/neuronpedia.py` | Neuronpedia client for feature concepts (descriptions/top tokens). |
| `src/introspector/models/signal_paths.py` | `resolve_sae_hook_path` — the residual site the SAE was trained on (capture AND steer here). |
| `src/introspector/training/model.py` | `SAEInterventionHook`, the gated-LoRA wrappers. |
| `src/introspector/prompts.py` | the introspection prompt. |

### Entry points
| file | role |
|---|---|
| `validate_sae_hook.py` | **Gate 0.** Is the SAE applied to the right residual? (FVU ≪ 0.35, L0 in the tens, BOS excluded.) |
| `select_sae_features.py` | Streaming density pass → per-layer feature-stats parquet. |
| `annotate_sae_features_neuronpedia.py` | Attach Neuronpedia concepts to the density-selected features (network-only). |
| `generate_sae_steering_samples.py` | Blinded steered-vs-unsteered samples for the out-of-band steering judge. |
| `cache_signals.py` | Shared cacher; the `sae` branch encodes features after capturing the residual. |
| `train_sae_detector_joint.py` | **The trainer.** Joint endogenous + interventional loss, gate off, LM head, KL anchor. |
| `analysis/plot_sae_joint_results.py` | Figures from the joint metrics (balanced acc / precision). |
| `scripts/sae/{sweep_common,cache_sae_sweep,train_sae_sweep,submit_sae_sweep}.sh` | The layer x position sweep (§7). |
| `analysis/plot_sae_sweep.py` | Layer x position figures + `sweep_summary.csv` for the sweep. |

---

## 3. Run order

```bash
# 0. validate the hook (once per model × release × layer family)
python validate_sae_hook.py --model google/gemma-2-2b \
    --sae-release gemma-scope-2b-pt-res-canonical --layer 9 --width 16k --site res

# 1. per-layer density stats  (writes data/sae_stats/gemma2-2b_res{L}_16k.parquet)
sbatch scripts/sae/precompute_sae_feature_stats.sh

# 1.5 attach Neuronpedia concepts to the selected features (login node, needs internet)
bash scripts/sae/annotate_sae_features_neuronpedia.sh   # login node (needs internet), not sbatch

# 2. steering gate: generate blinded samples, then judge them OUT OF BAND
sbatch scripts/sae/generate_sae_steering_samples.sh
#     -> read the *.judge.md blind, pick the concept-expressing option(s) per prompt;
#        features whose STEERED options are consistently chosen (baseline not) qualify.

# 3. cache + train the joint gate-off sweep (per weight W)
W=0.9 sbatch scripts/sae/train_sae_joint_layer9.sh     # also W=1.0 (=endogenous), 0.75, 0.5

# 4. figures
python analysis/plot_sae_joint_results.py --runs outputs/runs/sae_l9_joint --weight 0.9
```

The steering gate decides which features to train on; the 5 layer-9 features that passed
are the default `FEATURES` in `train_sae_joint_layer9.sh`.

> **Note on layer 9.** The batch scripts in steps 1–2 now sweep the layer x position grid's
> layers `{0,5,10,15,20}` (§7; they originally swept even layers `{0,…,14}`) — layer 9
> (the layer actually trained) is **not**
> in that set. The five trained layer-9 features were selected by running the same three
> steps directly for layer 9 (this is what produced the `res9` artifacts in
> `data/sae_stats/` and `data/sae_steering_samples/`):
>
> ```bash
> python select_sae_features.py --model google/gemma-2-2b \
>     --sae-release gemma-scope-2b-pt-res-canonical --layer 9 --width 16k --site res \
>     --n-tokens 2000000 --max-seq-length 128 --fixed-position -1 \
>     --min-density 1e-3 --max-density 1e-2 \
>     --out data/sae_stats/gemma2-2b_res9_16k.parquet
> python annotate_sae_features_neuronpedia.py \
>     --stats data/sae_stats/gemma2-2b_res9_16k.parquet \
>     --model gemma-2-2b --layer 9 --width 16k --site res \
>     --out data/sae_stats/gemma2-2b_res9_16k.annotated.parquet
> python generate_sae_steering_samples.py \
>     --annotated data/sae_stats/gemma2-2b_res9_16k.annotated.parquet \
>     --model google/gemma-2-2b --layer 9 --width 16k --site res \
>     --out-dir data/sae_steering_samples
> ```
> (Or add `9` to the `LAYERS` arrays in the three batch scripts to fold it into the sweep.)

---

## 4. The joint trainer

`loss = w · endogenous_CE + (1 − w) · interventional_CE + kl_coef · KL_anchor` (two
forwards/step). `W=1.0` is pure endogenous (interventional forward skipped) through the
same code, so endogenous and joint are directly comparable. Sweep `w ∈ {1.0, 0.9, 0.75, 0.5}`.

- **endogenous** forward: no steering; target = the feature's natural firing at the token.
- **interventional** forward: steer the feature on/off 50/50, decorrelated from the text;
  target = the forced state. Context cannot predict it — the anti-bypass condition.
- **KL anchor**: adaptive KL-to-base penalty on the input-text tokens only. It logs
  `L_kl` and `kl_coef`; `L_kl` stuck at exactly 0 would mean `disable_adapter()` is not
  being honored. See below for the mechanics.

### The KL anchor, in detail

**What it is against.** The gate is off, so the adapter runs over the whole sequence — but
the detection loss only cares about the answer token, so nothing else stops the adapter
from scrambling the model's behaviour on ordinary text. The anchor is the leash: at a few
sampled **input-text** positions it measures how far the tuned model (adapter ON) has
drifted from the frozen base model (adapter OFF — for LoRA that *is* the original
gemma-2-2b) and penalises that drift. The introspection prompt is never anchored; that is
where the readout lives and is meant to change.

**The quantity.** For one text position `c`, summing over the vocabulary `v`:

    KL_at(c) = sum_v  p_base(v|c) * log( p_base(v|c) / p_tuned(v|c) )

    KL_anchor = average of KL_at(c) over the sampled text positions      (nats/token)

`p_base` = model with the adapter off, `p_tuned` = adapter on, so it isolates the drift
caused purely by the adapter. Both sides use gemma's softcapped logits (we reuse the
endogenous forward's `.logits` rather than re-running the LM head), so the KL measures real
drift, not the softcap.

**Why sample ~4 positions, not all.** A full KL needs the entire next-token distribution
(V ~= 256k) at every position, in fp32, with the backward graph — several GB over a full
sequence. Uniformly random text positions are an *unbiased* estimate of the mean per-token
KL, so a few per row (times the batch, times many steps) is plenty. `--kl-anchor-positions`
defaults to 4. The controller's dead-band (below) absorbs the resulting estimator noise.

**The penalty added to the loss:**

    total_loss = w * L_endo + (1 - w) * L_interv + kl_coef * KL_anchor

**How `kl_coef` adapts (a thermostat).** `kl_coef` (call it lambda) is *not* fixed and
*not* learned by gradient — a controller nudges it each step to hold the measured KL near a
target budget (`--kl-anchor-target`, default 0.1 nats/token). With lambda clamped to
`[1e-4, 1e3]`, starting at 1.0:

    if   measured_KL > 1.5 * target :  lambda = min(lambda * 1.5, 1e3)    # drift high -> punish harder
    elif measured_KL < target / 1.5 :  lambda = max(lambda / 1.5, 1e-4)   # drift low  -> ease off
    else                            :  leave lambda unchanged             # dead-band  -> do nothing

- Multiplicative (x1.5 / /1.5) so lambda can span its whole range in a few steps and adjust
  proportionally to how far off it is.
- The `1.5` dead-band is a *tolerance*: the KL is a noisy sampled estimate, so don't react
  unless it is more than ~50% off target. `1.5` is the PPO adaptive-KL convention
  (Schulman et al.), matched to the neuron track — a heuristic, not a derived value.
- lambda is a constant during backprop; it is updated *afterwards* from the detached
  measured KL.

Net effect is negative feedback: you specify the *amount of drift you tolerate* (the
target), and lambda auto-solves for whatever penalty strength delivers it — instead of
hand-picking a fixed weight. `--kl-anchor-target 0` disables the anchor;
`--kl-anchor-weight` (with no target) uses a fixed lambda that never adapts.

**Cost per step** (w<1, anchor on): three forwards — endogenous (grad), interventional
(grad), and one no-grad `disable_adapter` forward for the base distribution.

Outputs per run: `sae_joint_train_metrics.json` (endogenous + interventional balanced
accuracy / precision) and `sae_cache_alignment.json`. All metrics are on the model's
generated token.

---

## 5. How to read the results

**Alignment first — the sanity gate.** This catches one silent bug: *steering/reading the
wrong token.* Every `--force-drift-eval` run prints and saves `sae_cache_alignment.json`.

*What the cache is.* Training and eval read a dataset precomputed once by `cache_signals.py`.
Each row stores the text, `token_position` (which token we introspect on), the `token`
string there, and `activation` — the feature's value at that token, computed by the FROZEN
base model + SAE encoder. That `activation` is the endogenous label.

*The risk it guards.* `token_position` was computed on the *cache-time* tokenization. Eval
re-tokenizes the text and steers at that same index. If the two tokenizations disagree (a
stray BOS, an extra separator, a padding slip), the index now points at a *different* token
than the one the label describes — nothing crashes; the numbers are just wrong.

*The check* (`cache_alignment`). With the adapter **off** (reproducing the caching model),
it recomputes the feature at the cached position at offsets −1 / 0 / +1, correlates each
against the cached `activation`, and checks the decoded token equals the cached `token`.
Pass criteria:
- **best offset +0** — the offset whose recompute best matches the cache is 0 (no shift);
- **corr ≈ 1.0** — at offset 0 the recomputed value reproduces the cached one (same model,
  same token);
- **token_match ≈ 1.0** — the token string at that index equals the cached one.

```
offset -1: corr(live, cached) = +0.01
offset +0: corr(live, cached) = +0.9999   <-- what training assumes
offset +1: corr(live, cached) = -0.01
best offset = +0  ->  alignment OK
```

If the best correlation instead shows up at +1 or −1 (with `token_match` low), the hook is
injecting into the *neighbour* of the named token, so **every interventional number is
void.** It is a gate you pass, not a result you report.

*Scope.* This validates the cache against the BASE model (adapter off) — it only confirms
the positions are right. It does NOT measure how far the TUNED model's live activation has
drifted from the cached value during training (the introspection-vs-recompute question);
that is a separate diagnostic (cf. the neuron track's `measure_activation_drift`), not yet
built here.

**Then faithfulness — read it from `test_interventional.balanced_accuracy`.** The
interventional set steers the feature on/off 50/50, decorrelated from the text, and labels
each forward by the forced state. Because that label cannot be predicted from context, this
balanced accuracy *is* the causal test:

| test_interventional.balanced_accuracy | meaning |
|---|---|
| → 1.0 | steering flips the report — it causally tracks the feature — **faithful** |
| ≈ 0.5 | steering does nothing — the report rides context — **bypass** |
| < 0.5 | steering flips it the wrong way — usually an OOD steering artifact, not anti-reading |

(Contrast `test_endogenous.balanced_accuracy`, which only shows the report and the feature
*correlate* on natural data — it cannot tell faithful from bypass. The interventional
number is what separates them.)

---

## 6. Gotchas

- **A `nan` metric means "no data to score", not "the model failed".** SAE features are
  sparse, so at a fixed token position a feature can have *no active examples* in the test
  set — one class only. Balanced accuracy and precision are then undefined (averaging the
  per-class recalls hits 0/0 on the missing class), so the eval returns `nan` with a
  `single-class eval set` note instead of a misleading number. It is *per feature* (each
  feature is scored on its own run) and specific to the *tested position*. Before calling a
  feature a failure, check `base_rate_positive`: `0.0` (or `1.0`) means the set was
  single-class. Only the *endogenous* metric can hit this — the interventional set is 50/50
  by construction. With the selected 0.1–1% density features it should not occur (~hundreds
  of active examples in 30k); it is a guard for rarer features, other positions, or smaller
  corpora.

- **One cache holds several features; `--signal-filter` picks one, and forgetting it aborts.**
  To avoid re-encoding, a single cache stores all of a cell's features (one forward pass over
  the text encodes them all); each run trains on one via `--signal-filter S9.<index>`. The
  filter is deliberately *not* part of the cache identity, so every per-feature run reuses the
  same cache directory instead of rebuilding it. The catch: because the filter is not in the
  identity, a multi-feature cache with no filter is ambiguous — so the trainer **aborts**
  rather than silently training on the wrong feature (which would be an invisible bug). Always
  pass the filter when the cache holds more than one feature.

- **`extreme` scale 1 is the *natural* magnitude, not an extreme one.** The interventional
  steering magnitude comes from `OnOffSampler`; the distribution is named `extreme` only for
  parity with the neuron scripts. At `--intervention-extreme-scale 1`, an "on" draw is uniform
  over the feature's observed active range `[amin, amax]` — i.e. natural magnitudes. Because
  SAE activations are non-negative, every "on" draw is floored at `amin` (~the JumpReLU
  threshold), so an "on" label is active *in SAE coordinates* (re-encoded, it often is not;
  see the next item). To inject harder than natural, raise
  `--intervention-extreme-scale` (10, 100). Interpretation: faithfulness at scale 1 is a
  strong result (the model reads the feature at magnitudes it actually sees); needing a large
  scale to move the report is a weaker, more artificial effect.

- **What the interventional test measures: the decoder direction, not an encoder-read
  "active" feature.** Steering adds `(v - f) * W_dec[i]`. Re-encoded, feature *i*'s
  pre-activation rises by only ~0.85·v (`W_enc[:, i] · W_dec[i]` is 0.83–0.86 for the
  layer-9 features), and on a naturally inactive token it starts below zero, so a natural-
  magnitude "on" steer usually stays under the JumpReLU threshold (~2.5). Measured on the
  five layer-9 features in bf16 (`_scratch_equiv/check_steer_reencode.py`, 2026-10-09):

  | feature | train draws read active | eval value (median active) read active | ...on naturally inactive rows | "off" reads inactive |
  |---|---|---|---|---|
  | S9.1568 | 74% | 18% | 8% | 99.6% |
  | S9.4854 | 78% | 21% | 12% | 100% |
  | S9.4120 | 63% | 16% | 5% | 99.6% |
  | S9.978 | 55% | 11% | 5% | 100% |
  | S9.12408 | 75% | 24% | 7% | 99.8% |

  So the interventional result answers "can the model detect feature *i*'s decoder direction
  injected at natural magnitude", while the endogenous labels use the encoder's definition
  (activation > 0). The two coincide for "off" but not for "on". **Kept deliberately** (decided
  2026-10-09) so the layer x position sweep stays comparable with the layer-9 runs; describe
  interventional faithfulness in these terms. An encoder-consistent variant (per-row
  injection chosen so the re-encoded value equals the target) would be a separate follow-up.
  Same check: the bf16 model's activations agree with the fp32-cached labels on 100% of
  inactive and 98.4–100% of active rows.

---

## 7. Layer x position sweep

The layer-9 analysis repeated over **layers {0, 5, 10, 15, 20} x token positions
{1, 8, 16, 24, 31}**, 3 features per layer, W in {1.0, 0.9}: 5 x 5 x 3 x 2 = **150 runs**
(~1 h each on layer-9 timing). The layer-9 runs (position -1) are the reference.

**Features are selected per layer, not per position**, with exactly the layer-9 pipeline
(§3 steps 0–2, at position -1), then reused at every position of that layer. Gemma Scope
trains one SAE per layer, so features do not carry across layers by index; each layer gets
its own. The annotation step fetches only the top 200 candidates per layer
(`--max-fetch`, default `MAX_FETCH=200`): the steering step only takes the top 20 by
the same ranking, so this selects the same features as a full fetch as long as >= 20 of
the 200 have an explanation (at every layer, all 200 did).

**Layer 25 is excluded.** It is the last decoder layer: after its output only the final norm
and the LM head follow, neither of which mixes positions, so steering the text token cannot
reach the readout token and the interventional test is capped at chance by construction.

**Positions are absolute token indices** (0 is BOS, which Gemma Scope SAEs exclude, and
`--max-seq-length 32` makes 31 the last). The sweep sets `--min-seq-length 32`: without it
the cacher clamps a position past a short text to its last token, mixing positions.

```bash
# 1. selection (per layer, position -1) -- writes ..._res{L}_pos-1_16k.parquet
sbatch scripts/sae/precompute_sae_feature_stats.sh             # includes Gate 0 per layer
bash   scripts/sae/annotate_sae_features_neuronpedia.sh        # needs internet
sbatch scripts/sae/generate_sae_steering_samples.sh            # layer-9 settings -> judge.md per layer, judge blind

# 2. record the top 3 passing features per layer (ranked) in scripts/sae/sweep_features.txt

# 3. per-position density check of the chosen features (25 tasks)
POSITIONS="1 8 16 24 31" sbatch --array=1-25 scripts/sae/precompute_sae_feature_stats.sh

# 4. caches (25, one per layer x position) then training (150), chained with afterok
DRY_RUN=1 bash scripts/sae/submit_sae_sweep.sh                 # print the plan
bash scripts/sae/submit_sae_sweep.sh                           # WEIGHTS="..." / MAX_PARALLEL=N

# 5. figures -> outputs/plots/sae_layerpos_joint/
python analysis/plot_sae_sweep.py --runs outputs/runs/sae_layerpos_joint \
    --anchor outputs/runs/sae_l9_joint
```

- **Two stages, so tasks never race on a cache.** One cache per (layer, position) holds all
  3 of the layer's features (the signal list is the same for every run of the cell, so they
  resolve to one `dataset_name`); `cache_sae_sweep.sh` builds them, `train_sae_sweep.sh`
  starts only after all succeed.
- **Feature-major task order.** Training tasks 1–50 are F1 of every layer over all positions
  and weights, 51–100 F2, 101–150 F3. SLURM starts low indices first, so F1's complete
  layer x position grid arrives first.
- **Every run does the cache-alignment check** (`--force-drift-eval`) and stops if steering
  would hit a different token than the one cached. New positions are exactly where an
  off-by-one would show up.
- **Resubmitting is safe.** The trainer skips a finished run and resumes from the newest
  checkpoint (saved every 25%); the cacher skips a built cache.
- **Rows whose token changes on re-tokenization are dropped** (train and test). The prompt
  is `f"{text} {separator}..."`; a truncated text ending in a lone space token merges with
  the template's space, so the prompt would name and steer a different token than the
  label describes. ~1.3% of rows at the last position of a truncated text, none elsewhere;
  the trainer prints the count per split.
- **The alignment check samples active rows on purpose** (up to half of its 512), so a
  sparse feature cannot fail it just by having no positives among the first rows; below 10
  active test rows it falls back to the token-match check and warns.
- **Precision is `nan`, not 0, when a run never predicts `active`** (undefined, and kept
  out of the means in the heatmaps).
- **Metrics files now include confusion counts** (`tpr`, `tnr`, `act_pred_*`, `ina_pred_*`),
  so `plot_sae_sweep.py` draws confusion matrices without reloading checkpoints.


---

## 8. Feature generalization (features named by their label)

The SAE analogue of the neuron track's neuron-generalization sweep
(`scripts/neuron/train_single_mlp_introspector_neuron_generalization.sh`), where naming
neurons by index gave no generalization at all. Here a feature is named by its
Neuronpedia explanation. Can one introspector, trained on N layer-9 features, read
features it never trained on?

| file | role |
|---|---|
| `build_sae_featgen_pool.py` | the 200-feature pool: `candidates` (density band + positives floor, seeded random sample), `build` (labels, near-duplicate removal, ordering, label-similarity matrix) |
| `src/introspector/sae_featgen.py` | row budget (`balance_rows_per_feature`), pool loading, nearest-trained-label similarity |
| `train_sae_detector_joint.py` | multi-feature path (`main_multi`), taken when `detection_total_train_rows` is set. The single-feature `main()` is untouched. |
| `score_sae_feature_generalization.py` | eval on every pool feature (seen / held out) + the steer x ask distractor matrix |
| `scripts/sae/featgen_common.sh` | the run table + config generation |
| `analysis/plot_sae_feature_generalization.py` | figures + CSVs |

```bash
qsub scripts/sae/build_featgen_pool_layer9.sh        # gate 0, stats, candidates, Neuronpedia, pool
qsub scripts/sae/cache_sae_feature_generalization.sh # every cache, one job
qsub -hold_jid sae_featgen_cache -t 1-5 scripts/sae/train_sae_feature_generalization.sh   # label arm
qsub -t 6-10 scripts/sae/train_sae_feature_generalization.sh                              # index arm, later
# label arm at N=3's 100k rows PER FEATURE (N=10/30/100 = 1M/3M/10M rows), one shared cache:
qsub -v STAGE_GROUPS=data100k scripts/sae/cache_sae_feature_generalization.sh
qsub -hold_jid sae_featgen_cache -t 11 -l h_rt=24:00:00 scripts/sae/train_sae_feature_generalization.sh
qsub -hold_jid sae_featgen_cache -t 12-13 -l h_rt=48:00:00 scripts/sae/train_sae_feature_generalization.sh
python analysis/plot_sae_feature_generalization.py
```

**Design.**
- **Recipe.** Layer 9, position -1, 32-token texts, joint w=0.9, KL 0.1, gate off: the
  layer-9 recipe. N in {1, 3, 10, 30, 100} as nested prefixes of the pool.
- **Arms.** `label` uses `prompt_sae_label`: the existing SAE templates with only the
  feature's name swapped for its quoted label. `index` is the existing
  `prompt_no_token_index`, the control.
- **Row budget: 300k rows at every N, split equally across features (300k/N each),
  exactly half active.**
  - Inactive rows are subsampled to the quota: the smallest `subsample_u`, so they are
    nested across N.
  - Active rows are repeated to the quota.
  - So distinct active examples per feature are fixed by density at every N, while
    distinct inactive examples shrink as 1/N. **State this when reporting the N curve.**
  - Active repeats fall from 100-1000x at N=1 to at most 10x at N=100.
- **data100k rows (11-13): 100k rows PER FEATURE at N=10/30/100** (N=3's budget). With
  300k total, seen-feature interventional accuracy collapses at N>=10, and these runs
  test whether that is starvation. Totals and optimizer updates grow with N (1M/3M/10M
  rows), so they don't separate data from steps.
  - At a fixed per-feature quota the inactive keep rate is the same at every N (0.383, as
    for label_n3), and subsampling is keyed per (feature, text). So one 100-feature cache
    holds every row each of them needs, and each run selects its features with
    `--signal-filter`. The table's `cache_feats` column sets this.
  - Runs over 1M rows checkpoint every 10%. One that outlives its job resumes from the
    latest checkpoint when the same task is resubmitted.
  - Plotted in red on the same axes as the label/index series (`featgen_by_n`,
    `featgen_distractor`), only at N=10/30/100, like the extra-data series of the neuron
    and layer generalization figures. They are their own series, so they never average
    into the 300k-total curves.
- **Caches** store only what the trainer can use, via `DatasetConfig.negative_keep`
  (`--train-negative-keep`, `--test-negative-keep`):
  - Inactive rows are kept by a deterministic per-(signal, text) hash
    (`dataset_utils.subsample_key`). Thresholds nest, and rows below a threshold are a
    uniform sample of texts.
  - The field enters the cache hash only when set, so every existing cache keeps its
    name.
- **Pool selection.**
  - Same density band as section 3, plus >= 50 projected positives at the readout.
  - `pos_density` is measured at the last token of a 32-token window
    (`select_sae_features.py --pos-window 32`). At 128 tokens the last token is often
    the document's end, which is a different population.
  - Near-duplicate labels are dropped at all-MiniLM-L6-v2 cosine >= 0.7. Paraphrases
    measured 0.65-0.82, unrelated labels ~0.15.
  - The 5 steering-judged features come first, so the smallest runs line up with the
    existing layer-9 single-feature runs.

**Reading the results.**
- **Headline: held-out interventional balanced accuracy.** Each text is steered on
  (the feature's median active value) and off.
- **It needs the distractor matrix next to it.** At eval most texts are naturally
  inactive, so "off" changes nothing and "on" is the only perturbation. A readout that
  detects *any* injection scores well on held-out features too. The matrix steers
  feature j while the prompt names feature i: a readout addressed to i stays at its
  "off" rate off the diagonal.
- **Endogenous held-out accuracy is weak evidence**: the label describes text the
  feature fires on, so it can be guessed without introspection.
- **Precision is reported at each feature's natural base rate** (from TPR/FPR), since
  the endogenous eval set subsamples inactive rows.
