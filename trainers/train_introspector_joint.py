#!/usr/bin/env python3
"""
Joint endogenous + interventional introspection training.

Per batch we run TWO forwards on the same examples and step on a weighted sum:

    L = w * L_endo + (1 - w) * L_interv

  - L_endo:   no injection, target = the true activation      (natural-scale target)
  - L_interv: inject the signal, target = the injected value  (large learnable scale)

Both losses are VARIANCE-NORMALIZED (MSE / var(target) ~= 1 - R^2) so their
magnitudes are comparable despite the ~10x scale difference, and `w` genuinely
controls the balance rather than the accident of scale.

Rationale (see design notes):
  * The faithful absolute-value reader is the UNIQUE joint optimum: a bypass head
    fails L_interv (context is uncorrelated with the injected value, so it adds
    noise), a perturbation-only head fails L_endo (no perturbation endogenously).
    So L_interv is the anti-bypass regularizer -- keep `w` (endo weight) < 1 and
    CONSTANT (do NOT anneal the mix toward endogenous; that removes the pressure).
  * Interventional injection must stay at a LEARNABLE scale (>=10x, the natural
    scale is at the bf16 floor), configured via the usual distribution/scale; the
    endogenous term supplies the natural-scale target.
  * Two forwards per batch => both gradients enter every step (no tug-of-war).

Sweep `--joint-loss-weight` across runs and read the (endogenous, interventional)
FRONTIER: a weight with BOTH high => a faithful reader exists; a pure trade-off
=> it is not representable in this single-forward regime.
"""
from __future__ import annotations

import os
import sys
import glob
import json
import math
import argparse
import operator
from typing import Optional
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from src.introspector.config import ExperimentConfig
from src.introspector.dataset_utils import apply_signal_filter, pair_pool_dir
from src.introspector.signals.schema import is_pair_selector
from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.models.attention_logits import (
    AttentionLogitPort, AttentionWeightPort, PortFanOut, attention_model_kwargs,
    installed_ports, is_attention_type, port_site,
)
from src.introspector.prompts import PROMPT_REGISTRY
from src.introspector.training.collator import LoRAGatingCollator
from src.introspector.training.metrics import compute_metrics, compute_metrics_from_numericals
from src.introspector.training.pair_metrics import attach_provenance, write_pair_outputs
from src.introspector.training.model import (
    signal_range_key,
    logit_moments,
    logit_histogram,
    empirical_prob_values,
    EMPIRICAL_EPS,
    make_bounded_prob_intervention_fn,
    IntrospectionRegressionModel,
    RegressionHeadSaveCallback,
    InterventionAnnealCallback,
    StripEmbeddingSaveCallback,
    load_added_token_embeddings,
    load_regression_head,
    replace_model_with_gated_version,
    replace_lora_with_gated_version,
)
from trainers.train_introspector import preprocess_example, drop_invalid_positions, INTROSPECTION_SEPARATOR
from trainers.train_introspector_with_intervention import (
    InterventionTrainer,
    evaluate_interventional_test,
    get_argmax_from_logits,
)
from src.introspector.training.model import resize_embeddings_for_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--joint-loss-weight", type=float, default=0.5,
                   help="Weight w on the ENDOGENOUS term; interventional gets (1-w). "
                        "Keep constant; sweep across runs to map the frontier.")
    p.add_argument("--kl-anchor-target", type=float, default=None,
                   help="KL BUDGET in nats/token on the input text. The anchor coefficient is "
                        "adapted multiplicatively to hold measured KL(base||tuned) near this "
                        "value, so you set the behavioural drift you will tolerate instead of "
                        "sweeping a weight. Sweeping the BUDGET against interventional corr is "
                        "the 'cost of introspection' frontier. None => no adaptive anchor.")
    p.add_argument("--kl-anchor-weight", type=float, default=0.0,
                   help="Fixed anchor coefficient, used when --kl-anchor-target is not given. "
                        "0 disables the anchor entirely (the pre-anchor behaviour).")
    p.add_argument("--kl-anchor-positions", type=int, default=4,
                   help="Text positions sampled per sequence for the KL estimate. A full-vocab "
                        "fp32 KL over every position is ~GBs; a random subset is unbiased and "
                        "cheap. Raise for a lower-variance estimate.")
    p.add_argument("--require-kl-anchor", action="store_true",
                   help="Fail instead of training if a KL anchor was requested but cannot "
                        "apply. The anchor needs `anchor_region_mask`, which only "
                        "UngatedCollator emits, so with the input gate ON the KL term is "
                        "skipped SILENTLY and the run still writes a directory named "
                        "_kl<budget>_. Pass this on any gated arm that names a budget.")
    p.add_argument("--kl-anchor-coef-init", type=float, default=1.0)
    p.add_argument("--kl-anchor-coef-range", type=str, default="0.0001,1000",
                   help="Clamp for the adaptive coefficient (min,max).")
    p.add_argument("--ood-eval", action="store_true",
                   help="After the in-domain evals, score the readout on out-of-distribution "
                        "corpora (see evals/score_ood_generalization.py). Endogenous corr is the "
                        "discriminating metric: a context-predicting readout collapses under "
                        "shift, a causal one does not.")
    p.add_argument("--prepare-only", action="store_true",
                   help="Build the eval caches that --length-eval / --ood-eval would build "
                        "(see prepare_length_caches / prepare_ood_caches), then exit -- no "
                        "model, no training. Exits non-zero if any of them failed. Lets a "
                        "cache job build them once for several runs that share them, "
                        "instead of those runs racing to build them concurrently.")
    p.add_argument("--ood-only", action="store_true",
                   help="Load the checkpoint and run ONLY the OOD eval -- no training, and "
                        "none of the other diagnostics. Results are written into the RUN dir "
                        "(not a diag_* subdir) so they overwrite a previous ood_generalization.* "
                        "and stay where the aggregator looks. Use this to redo an OOD pass that "
                        "failed without retraining.")
    p.add_argument("--ood-corpora", type=str, default=None,
                   help="Comma-separated corpus names (default: shuffled,wikipedia,c4_zh,math). "
                        "'shuffled' is the run's own test text with word order destroyed -- the "
                        "cleanest control, and it needs no download.")
    p.add_argument("--length-eval", action="store_true",
                   help="After the in-domain evals, score the readout across eval lengths, "
                        "binned by distance from the introspected token to the readout "
                        "(see evals/score_length_generalization.py). The point is the bins BEYOND "
                        "the training window: a run trained on the last k tokens can solve "
                        "training with a recency prior and fail everywhere else.")
    p.add_argument("--eval-n-test", type=int, default=None,
                   help="Texts per length-eval cache. Default: inherit the run's own test "
                        "split size. Set this rather than editing --test-max-samples: that "
                        "is part of the dataset IDENTITY, so changing it renames the run's "
                        "TRAINING cache too and --length-only would then fail to load it.")
    p.add_argument("--length-only", action="store_true",
                   help="Load the checkpoint and run ONLY the length-generalization eval -- "
                        "no training, no other diagnostics. Writes into the RUN dir so results "
                        "land where the aggregator looks.")
    p.add_argument("--eval-lengths", type=str, default=None,
                   help="Comma-separated eval lengths (default: 16,64,128). Each must be <= the "
                        "min_seq_length the caches were built with, or documents will not "
                        "actually reach that length.")
    p.add_argument("--neuron-gen-eval", action="store_true",
                   help="After the in-domain evals, score the readout on the trained neurons "
                        "AND on neurons cached but never trained (see "
                        "evals/score_neuron_generalization.py). Both arms read this run's own test "
                        "split, so no extra caching. Interventional corr is the metric to "
                        "believe on the unseen arm: the injected value is independent per row, "
                        "so a readout that ignores which neuron the prompt names scores 0.")
    p.add_argument("--neuron-gen-only", action="store_true",
                   help="Load the checkpoint and run ONLY the neuron-generalization eval. "
                        "Writes into the RUN dir so results land where the aggregator looks.")
    p.add_argument("--neuron-gen-pool-config", type=str, default=None,
                   help="Config whose cached TEST split holds the full neuron pool "
                        "(trained + held-out). A run's own cache only has the neurons it "
                        "trained on -- sizing it to hold the whole pool would multiply its "
                        "train split by the pool size. Point this at the sweep's shared cell "
                        "config; without it there is no unseen arm.")
    p.add_argument("--max-unseen-neurons", type=int, default=None,
                   help="Cap the held-out neurons evaluated (default: every neuron in the "
                        "cache that the run did not train on).")
    p.add_argument("--evals-per-run", type=int, default=25,
                   help="How many dev evaluations to run over the whole training run. "
                        "eval_steps was previously a fixed 400-examples cadence, which is "
                        "781 evaluations on a 300k-row single-epoch run -- invisible when "
                        "dev was 200 rows, ruinous once dev scales with the neuron count. "
                        "Scale-invariant: the cadence follows the run length. Dev is a "
                        "monitor only (no load_best_model_at_end), so this cannot change "
                        "the trained model.")
    p.add_argument("--max-train-examples", type=int, default=None,
                   help="Cap TRAIN rows after the signal filter, without rebuilding the cache. "
                        "A row is one (text, neuron) pair, so with N neurons this caps rows, "
                        "not texts, and per-neuron data falls as 1/N -- which is what makes a "
                        "fixed total budget the sharing measurement. Also how the data-matched "
                        "control arm reaches a fraction of the budget on one neuron.")
    p.add_argument("--skip-drift-eval", action="store_true",
                   help="Skip the activation-drift measurement (see measure_activation_drift). It runs "
                        "automatically whenever the adapter is active over the input text, because that "
                        "is when the cached endogenous target can go stale.")
    p.add_argument("--force-drift-eval", action="store_true",
                   help="Run the drift measurement even when the input gate is ON. Drift is ~0 by "
                        "construction there (no trainable parameter touches the text), so this is a null "
                        "control -- and the cache-alignment check it prints is useful on any run.")
    p.add_argument("--routing-probe-scale", type=float, default=10.0,
                   help="Injection scale for the routing-probe diagnostic (use a LARGE learnable "
                        "scale so 'does it reach the readout' is detectable above the bf16 floor).")
    p.add_argument("--routing-probe-pca", type=int, default=256,
                   help="PCA components for the routing-probe ceiling (regularize when n ~ n_features; "
                        "0 disables PCA).")
    p.add_argument("--scale-sweep", type=str, default="1,3,10,30,100",
                   help="Comma-separated injection scales for the interventional scale sweep.")
    p.add_argument("--interv-eval-distributions", type=str, default="gaussian",
                   help="Comma-separated injection distributions for the headline interventional "
                        "test eval, each written as interventional_test_metrics_<dist>.json. "
                        "'empirical_logit' (attention_weight only) draws from the cell's own "
                        "train log-odds histogram and also writes "
                        "interventional_ceiling_empirical_logit.json. 'empirical_prob' "
                        "(attention_weight only) draws one of the cell's own train weights "
                        "as-is (bootstrap, no clamp).")
    p.add_argument("--interv-eval-only", action="store_true",
                   help="Load --checkpoint (e.g. another run's) and run ONLY the interventional "
                        "test evals, writing into THIS config's output dir. No training.")
    p.add_argument("--diag-fp32", action="store_true",
                   help="Also re-run the diagnostics (scale sweep + routing probe) with the model cast "
                        "to fp32, to test whether natural-scale failure is a bf16 numerical floor. "
                        "Cheap: resume the checkpoint (skips training) and pass this flag.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Specific checkpoint dir to load (default: latest under output_dir).")
    p.add_argument("--diag-only", action="store_true",
                   help="Load --checkpoint's weights and run evals/diagnostics WITHOUT training. Use to "
                        "eval a MID-RUN checkpoint (plain resume would continue training instead). "
                        "Outputs go to output_dir/diag_<checkpoint>/ so they don't clobber the final run.")
    return p.parse_args()


def _var_norm_mse(pred, target):
    """Variance-normalized MSE = MSE / var(target) (~ 1 - R^2, scale-invariant)."""
    pred = pred.float()
    target = target.float()
    var = torch.clamp(target.var(), min=1e-8)
    return nn.functional.mse_loss(pred, target) / var


def _norm_deviance(z, target):
    """Normalized Bernoulli deviance D / D_null = mean KL(p||sigmoid(z)) / mean KL(p||p_bar).

    The probability-target counterpart of _var_norm_mse: 0 = perfect, 1 = predicting the
    batch mean, so a joint loss mixes two terms at the balance set by w, not by scale.
    Takes the head's LOG-ODDS z, not its probability: BCE through a sigmoid has gradient
    (q - p) in z, where MSE through it has (q - p) q (1 - q) and stalls on the many
    attention weights near 0. H(p) is subtracted so the minimum is 0 for soft targets
    (plain BCE bottoms out at the targets' entropy).
    """
    z = z.float()                                                       # [B]
    p = target.float()                                                  # [B]
    assert z.shape == p.shape and z.ndim == 1, f"z {tuple(z.shape)} vs target {tuple(p.shape)}"
    assert ((p >= 0) & (p <= 1)).all(), (
        f"norm_deviance needs probability targets, got range [{p.min():.4g}, {p.max():.4g}]")
    ent = -(torch.special.xlogy(p, p) + torch.special.xlogy(1 - p, 1 - p))  # [B], H(p)
    kl = nn.functional.binary_cross_entropy_with_logits(z, p, reduction="none") - ent  # [B]
    # p_bar is the null model's prediction: a constant, so no gradient flows through it
    p_bar = p.mean().clamp(1e-7, 1 - 1e-7)
    kl_null = -(p * torch.log(p_bar) + (1 - p) * torch.log1p(-p_bar)) - ent  # [B]
    return kl.mean() / torch.clamp(kl_null.mean(), min=1e-12)


REGRESSION_LOSSES = ("var_norm_mse", "norm_deviance")


def _lm_head(model):
    """The base model's LM head, through whatever wrappers are in the way."""
    for get in (lambda m: m.get_base_model().lm_head,
                lambda m: m.base_model.model.lm_head,
                lambda m: m.lm_head):
        try:
            return get(model)
        except AttributeError:
            continue
    raise AttributeError("Could not locate lm_head on the model for the KL anchor.")


def _sample_text_positions(region_mask, attention_mask, k, generator=None):
    """
    Pick up to k INPUT-TEXT positions per sequence, uniformly at random.

    Text is region_mask == 0 (the collator marks the introspection prompt with 1) and
    attention_mask == 1 (prediction batches are left-padded, and padding also reads as
    0 in the region mask). Returns [B, k] indices plus a validity mask -- short
    sequences may have fewer than k text positions.
    """
    valid = (region_mask == 0) & (attention_mask == 1)                 # [B, L]
    scores = torch.rand(valid.shape, device=valid.device, generator=generator)
    scores = scores.masked_fill(~valid, -1.0)
    k = min(k, valid.shape[1])
    top = scores.topk(k, dim=1)
    return top.indices, top.values >= 0.0                              # [B,k], [B,k]


def kl_anchor_loss(model, inputs, tuned_hidden, region_mask, k):
    """
    Forward KL(base || tuned) over sampled input-text positions, in nats/token.

    The reference is this same model with adapters disabled -- for LoRA that IS the
    base model, so the anchor costs one no-grad forward and NO extra memory. The
    direction matches evals/score_lm_degradation.py, so the quantity trained against and the
    quantity reported are the same thing.

    Only the TEXT region is anchored. The introspection prompt is where the readout
    lives and is supposed to change; anchoring it would fight the task directly.
    """
    attention_mask = inputs["attention_mask"]
    idx, ok = _sample_text_positions(region_mask, attention_mask, k)
    if not bool(ok.any()):
        return None
    bidx = torch.arange(idx.shape[0], device=idx.device).unsqueeze(1)

    head = _lm_head(model)
    sel_tuned = tuned_hidden[bidx, idx]                                # [B,k,H]
    logits_tuned = head(sel_tuned.to(head.weight.dtype))               # [B,k,V]

    with torch.no_grad():
        with model.disable_adapter():
            # no labels => the regression branch is skipped and .logits are real LM logits
            base_out = model(input_ids=inputs["input_ids"], attention_mask=attention_mask,
                             do_intervention=False)
        logits_base = base_out.logits[bidx, idx]

    lp_tuned = torch.log_softmax(logits_tuned.float(), dim=-1)
    lp_base = torch.log_softmax(logits_base.float(), dim=-1)
    kl = (lp_base.exp() * (lp_base - lp_tuned)).sum(-1)                # [B,k]
    return (kl * ok.float()).sum() / ok.float().sum().clamp(min=1.0)


class UngatedCollator(LoRAGatingCollator):
    """LoRAGatingCollator minus the gate, selected when training_config.
    disable_adapter_in_input is False (matching trainers/train_introspector.py, where that
    flag has always chosen between gated and plain LoRA).

    It drops `lora_token_mask` from the batch, so replace_model_with_gated_version
    sets every LoRA layer's mask to None and GatedLinear reduces to stock LoRA over
    the WHOLE sequence, input text included. Everything else -- left padding for
    prediction, frozen_input_len popping -- is unchanged, so the trainer, evals and
    diagnostics need no modification.

    Why this matters here: with the gate ON, no trainable parameter touches the
    computation at the introspected token, so the injected signal is routed
    downstream at fixed base-model gain -- the suspected reason earlier token
    positions fail. With it OFF the adapter can boost that gain AT THE ORIGIN.
    Caveat: the model's own activation then drifts from the CACHED endogenous
    target; the interventional loss/metric is unaffected, since the injected value
    is ground truth by construction."""

    def __call__(self, features):
        batch = super().__call__(features)
        # keep the mask under a different key: gating must be OFF (the model's patched
        # forward only reads `lora_token_mask`), but the KL anchor still needs to know
        # which positions are input TEXT (0) versus introspection prompt (1)
        mask = batch.pop("lora_token_mask", None)
        if mask is not None:
            batch["anchor_region_mask"] = mask
        return batch


class JointInterventionTrainer(InterventionTrainer):
    """Two-forward joint loss: w*endogenous + (1-w)*interventional, both normalized
    (var-normed MSE, or normalized deviance for a sigmoid head; see `regression_loss`)."""

    def __init__(self, *args, joint_loss_weight=0.5, kl_anchor_target=None,
                 kl_anchor_weight=0.0, kl_anchor_positions=4, kl_anchor_coef_init=1.0,
                 kl_anchor_coef_range=(0.01, 1000.0), regression_loss="var_norm_mse", **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_loss_weight = joint_loss_weight
        assert regression_loss in REGRESSION_LOSSES, f"regression_loss {regression_loss!r}"
        self.regression_loss = regression_loss
        # anchor: adaptive to a KL budget if a target is set, else a fixed weight
        self.kl_anchor_target = kl_anchor_target
        self.kl_anchor_positions = kl_anchor_positions
        self._last_kl_update_step = -1   # see compute_loss: one controller update per optimizer step
        self.kl_anchor_range = kl_anchor_coef_range
        self.kl_anchor_coef = (kl_anchor_coef_init if kl_anchor_target is not None
                               else kl_anchor_weight)
        self.kl_anchor_on = self.kl_anchor_coef > 0 or kl_anchor_target is not None

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = inputs.copy()
        inputs.pop("anchor_region_mask", None)   # loss bookkeeping, not a model input
        return super().prediction_step(model, inputs, prediction_loss_only,
                                       ignore_keys=ignore_keys)

    def _update_kl_coef(self, kl_value):
        """PPO-style multiplicative controller: hold measured KL near the budget.
        Bands at 1.5x avoid oscillating on ordinary step-to-step noise."""
        if self.kl_anchor_target is None:
            return
        lo, hi = self.kl_anchor_range
        if kl_value > 1.5 * self.kl_anchor_target:
            self.kl_anchor_coef = min(self.kl_anchor_coef * 1.5, hi)
        elif kl_value < self.kl_anchor_target / 1.5:
            self.kl_anchor_coef = max(self.kl_anchor_coef / 1.5, lo)

    def _regression_loss(self, out, target):
        if self.regression_loss == "var_norm_mse":
            return _var_norm_mse(out.logits, target)
        z = out.get("regression_pre_activation")
        assert z is not None, "norm_deviance needs the sigmoid head's log-odds; the forward gave none"
        return _norm_deviance(z, target)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = inputs.copy()
        # not a model input: the patched forward only understands `lora_token_mask`
        region_mask = inputs.pop("anchor_region_mask", None)
        w = self.joint_loss_weight
        endo_out = interv_out = L_endo = L_interv = L_kl = None

        # A forward whose weight is 0 is skipped entirely. At w=1 (pure endogenous)
        # or w=0 (pure interventional) the zero-weighted term adds no gradient, but
        # `0.0 * L` still builds and traverses its graph on backward -- a silent 2x
        # on every step. Skipping makes the endpoints cost exactly what the
        # single-objective trainers cost, so one script can serve the whole frontier.
        if w > 0.0:
            # endogenous forward: no injection, target = true activation
            endo_inputs = inputs.copy()
            endo_inputs["labels"] = inputs["activation"]
            endo_inputs["do_intervention"] = False
            endo_out = model(**endo_inputs)
            L_endo = self._regression_loss(endo_out, inputs["activation"])

        if w < 1.0:
            # interventional forward: inject, target = injected value
            interv_inputs = inputs.copy()
            interv_inputs["labels"] = inputs["activation"]   # forward overwrites with injected value
            interv_inputs["do_intervention"] = True
            interv_out = model(**interv_inputs)
            L_interv = self._regression_loss(interv_out, interv_out.intervention_values)

        loss = 0.0
        if L_endo is not None:
            loss = loss + w * L_endo
        if L_interv is not None:
            loss = loss + (1.0 - w) * L_interv

        # KL anchor: keep next-token behaviour on the INPUT TEXT close to the base
        # model's. Nothing else in this objective preserves language modelling -- with
        # the adapter live over the text, an unanchored run reaches ~3 nats/token and
        # 24% top-1 agreement, i.e. it stops being a language model.
        if self.kl_anchor_on and region_mask is not None:
            anchor_out = endo_out
            if anchor_out is None:      # w == 0: no clean forward exists yet
                clean = inputs.copy()
                clean["do_intervention"] = False
                clean.pop("labels", None)
                anchor_out = model(**clean)
            L_kl = kl_anchor_loss(model, inputs, anchor_out.hidden_states[-1],
                                  region_mask, self.kl_anchor_positions)
            if L_kl is not None:
                loss = loss + self.kl_anchor_coef * L_kl
                # ONCE PER OPTIMIZER STEP, not once per micro-batch. compute_loss runs
                # per micro-batch, so under gradient accumulation the controller would
                # apply its multiplicative update N times between parameter updates --
                # the same measured KL driving the coefficient N-fold, which changes the
                # loop gain and can oscillate. global_step is constant within an
                # accumulation group, so this fires on the group's first micro-batch and
                # reduces to the previous behaviour when accumulation is 1.
                if self._last_kl_update_step != self.state.global_step:
                    self._update_kl_coef(float(L_kl.detach()))
                    self._last_kl_update_step = self.state.global_step

        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            log = {k: float(v.detach()) for k, v in
                   (("L_endo", L_endo), ("L_interv", L_interv), ("L_kl", L_kl)) if v is not None}
            if self.kl_anchor_on:
                log["kl_coef"] = float(self.kl_anchor_coef)
            self.log(log)
        out = interv_out if interv_out is not None else endo_out
        return (loss, out) if return_outputs else loss


def _eff_batch(config) -> int:
    """Rows consumed per optimizer step."""
    tc = config.training_config
    return max(1, tc.train_batch_size) * max(1, getattr(tc, "gradient_accumulation_steps", 1))


def _opt_steps_per_epoch(n_train_rows: int, config) -> int:
    """Optimizer steps in one epoch.

    Not rows/batch: with gradient accumulation an optimizer step consumes
    train_batch_size * gradient_accumulation_steps rows, and HF counts global_step,
    max_steps, eval_steps, save_steps and logging_steps in OPTIMIZER steps. Dividing by
    the micro-batch alone overstates the run length by the accumulation factor, which
    silently multiplies the eval count and, for a fractional-epoch run, the length of
    training itself.
    """
    tc = config.training_config
    eff = max(1, tc.train_batch_size) * max(1, getattr(tc, "gradient_accumulation_steps", 1))
    return max(1, math.ceil(n_train_rows / eff))


def _eval_steps(n_train_rows: int, config, evals_per_run: int) -> int:
    """Dev-eval cadence, expressed as a COUNT of evaluations rather than a fixed
    examples-per-eval interval.

    The old formula (every 400 examples) gives 781 evaluations on a 300k-row single-epoch
    run. That was cheap while dev was 200 rows; with a multi-neuron cache dev rows are
    texts x neurons, so at 100 neurons each evaluation is 157 batches and the run spends
    longer in dev than in training. Tying the cadence to run length keeps it bounded
    whatever the dataset size.
    """
    steps_per_epoch = _opt_steps_per_epoch(n_train_rows, config)
    # ceil, and no max(1, epochs): 0.333 epochs is a legitimate run length, and
    # clamping it to 1 would space the evals for a run 3x longer than the real one
    total = max(1, math.ceil(steps_per_epoch * float(config.training_config.num_epochs)))
    return max(1, total // max(1, evals_per_run))


def _checkpoint_is_complete(ckpt) -> bool:
    """True when this checkpoint already sits at the end of its training run.

    Matters because trainer.train(resume_from_checkpoint=...) on a FINISHED run is not
    a no-op: with ignore_data_skip off (the default) the Trainer walks the dataloader
    forward to restore data order before training zero further steps. At 300k examples
    that is many minutes of collating batches it will never use, and it presents as a
    progress bar frozen at 0/<max_steps> with no error -- indistinguishable from a hang.
    """
    state = Path(ckpt) / "trainer_state.json"
    if not state.exists():
        return False
    try:
        st = json.loads(state.read_text())
    except Exception:
        return False
    gs, ms = st.get("global_step"), st.get("max_steps")
    return bool(gs and ms and gs >= ms)


def _text_len_from_mask(batch) -> torch.Tensor:
    """Tokenized text length (BOS included) per row, recovered from the LoRA gate mask.

    The mask is 0 over the input text and 1 over the introspection prompt, with padding
    zeroed on whichever side the collator padded. Prediction batches are LEFT-padded, so
    the first 1 sits at (num_left_pad + text_len); subtracting the pad count leaves the
    text length.

    The key differs by collator and BOTH must be handled: UngatedCollator (gate off --
    the setting every current experiment runs in) moves the mask to `anchor_region_mask`,
    because the patched forward reads `lora_token_mask` and its mere presence would
    re-enable gating. The semantics are identical, only the name differs.

    Returns -1 per row when neither key is present, or where no prompt region exists.
    Deliberately not an error: this runs AFTER training, and a collator variant that
    carries the mask under some third name should cost the distance column, not the
    whole finished run.
    """
    mask = batch.get("lora_token_mask")
    if mask is None:
        mask = batch.get("anchor_region_mask")
    if mask is None:
        n = batch["input_ids"].shape[0]
        return torch.full((n,), -1, dtype=torch.long, device=batch["input_ids"].device)
    num_left_pad = (batch["attention_mask"] == 0).sum(dim=1)
    first_one = torch.argmax(mask, dim=1)
    has_prompt = mask.sum(dim=1) > 0
    return torch.where(has_prompt, first_one - num_left_pad,
                       torch.full_like(first_one, -1))


def _per_signal_metrics(df: pd.DataFrame, value_col: str = "predictions",
                        target_col: str = "activation") -> Optional[pd.DataFrame]:
    """Per-(layer, neuron) fidelity from a predictions frame.

    Returns None for a single-signal run, where the pooled metrics already say
    everything and an extra one-row file is just noise. A neuron whose activation
    is constant across the eval set has undefined corr; it is reported as NaN
    rather than dropped, so a capacity cliff cannot hide as a shrinking denominator.
    """
    if "signal_index" not in df.columns:
        return None
    keys = ["signal_layer", "signal_index"]
    if df.groupby(keys).ngroups < 2:
        return None
    rows = []
    for (layer, index), g in df.groupby(keys):
        m = compute_metrics_from_numericals(g[value_col].to_numpy(),
                                            g[target_col].to_numpy())
        rows.append({"signal_layer": int(layer), "signal_index": int(index),
                     "n": int(len(g)), **m})
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


@torch.no_grad()
def eval_endogenous_test(model, dataset, collator, config, output_dir, suffix="_endogenous",
                         provenance=None):
    """
    Endogenous eval (no injection): full metrics (corr, l2_dist, avg/std diff, ...)
    matching the interventional eval, plus a predicted-vs-actual scatter.

    `provenance` (PAIR MODE): per-row doc_index / *_token_unique, aligned to `dataset`
    (see pair_metrics.attach_provenance).
    """
    # dropout is a no-op under eval(); before the GatedLinear dropout fix it was a
    # no-op either way, so these forwards used to inherit train mode harmlessly.
    model.eval()
    dl = DataLoader(dataset, batch_size=config.training_config.eval_batch_size,
                    collate_fn=collator, shuffle=False)
    head = model.regression_head
    preds, acts, layers, idxs, tokpos, seqlens, srcpos = [], [], [], [], [], [], []
    for batch in dl:
        batch = batch.to(model.device)
        activation = batch["activation"]
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    lora_token_mask=batch.get("lora_token_mask"),
                    do_intervention=False, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]
        pred = head(h.to(device=head.weight.device, dtype=head.weight.dtype)).squeeze(-1)
        preds.append(pred.float().cpu()); acts.append(activation.float().cpu())
        # Per-signal provenance: with >1 neuron in the run, a single pooled corr is
        # not interpretable -- pooling across neurons with different activation
        # scales inflates it (between-neuron variance is counted as signal). The
        # multi-neuron scaling curve needs the per-neuron breakdown, and the pooled
        # number stays only for continuity with single-neuron runs.
        layers.append(batch["signal_layer"].cpu()); idxs.append(batch["signal_index"].cpu())
        # token_position + the text length behind it: without these, a length-generalization
        # run cannot be binned by distance-to-readout after the fact, and every distance
        # bin would need its own eval cache.
        # The collator POPS frozen_input_len and folds it into lora_token_mask (0 over the
        # text, 1 over the introspection prompt), so the length is recovered from the mask
        # rather than read off the batch -- adding a key here would also hand it to
        # model(**inputs) during training.
        tokpos.append(batch["token_position"].cpu())
        seqlens.append(_text_len_from_mask(batch).cpu())
        # attention: the SOURCE position too. With a fixed offset it was recoverable as
        # dst + offset; in pair mode every row has its own, and without it the by-pair
        # breakdown (and any dst x src plot) cannot be built.
        if "src_token_position" in batch:
            srcpos.append(batch["src_token_position"].cpu())
    pred, act = torch.cat(preds).numpy(), torch.cat(acts).numpy()
    layer_col = torch.cat(layers).numpy(); index_col = torch.cat(idxs).numpy()
    pos_col = torch.cat(tokpos).numpy(); len_col = torch.cat(seqlens).numpy()

    metrics = compute_metrics_from_numericals(pred, act)   # corr, l2_dist, avg_diff, std_diff, ...
    df = pd.DataFrame({"predictions": pred, "activation": act,
                       "signal_layer": layer_col, "signal_index": index_col,
                       "token_position": pos_col, "text_len": len_col})
    df["distance_from_end"] = np.where(df["text_len"] > 0,
                                       df["text_len"] - 1 - df["token_position"], -1)
    if srcpos:
        df["src_token_position"] = torch.cat(srcpos).numpy()
        assert len(df["src_token_position"]) == len(df), "src positions misaligned"
        df["src_offset"] = df["src_token_position"] - df["token_position"]
    df = attach_provenance(df, provenance)
    pair_summary = write_pair_outputs(df, "predictions", "activation", model,
                                      Path(output_dir) / f"endogenous_test_by_pair{suffix}.csv")
    if pair_summary is not None:
        metrics["pairs"] = pair_summary
    per_signal = _per_signal_metrics(df)
    if per_signal is not None:
        per_signal.to_csv(
            Path(output_dir) / f"endogenous_test_by_signal{suffix}.csv", index=False)
        # Macro-average over neurons: the honest headline when n_signals > 1.
        metrics["n_signals"] = int(len(per_signal))
        metrics["corr_macro"] = float(per_signal["corr"].mean())
        metrics["corr_macro_std"] = float(per_signal["corr"].std(ddof=0))
        metrics["corr_macro_min"] = float(per_signal["corr"].min())
        if "spearman" in per_signal:
            metrics["spearman_macro"] = float(per_signal["spearman"].mean())
            metrics["spearman_macro_std"] = float(per_signal["spearman"].std(ddof=0))
    with open(Path(output_dir) / f"endogenous_test_metrics{suffix}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    df.to_csv(Path(output_dir) / f"endogenous_test_predictions{suffix}.csv", index=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(act, pred, s=12, alpha=0.5)
    lim = [float(min(act.min(), pred.min())), float(max(act.max(), pred.max()))]
    plt.plot(lim, lim, "k--", lw=1, alpha=0.6)   # y = x reference
    plt.xlabel("true activation"); plt.ylabel("prediction")
    plt.title(f"endogenous: pred vs actual "
              f"(corr={metrics['corr']:.3f}, l2={metrics['l2_dist']:.3f})")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"endogenous_test_scatter{suffix}.png", dpi=150)
    plt.close()
    print(f"[joint] endogenous test: corr={metrics['corr']:.4f} l2={metrics['l2_dist']:.4f} "
          f"avg_diff={metrics['avg_diff']:.4f} std_diff={metrics['std_diff']:.4f}")
    return metrics


# ======================================================================
# Activation drift + cache alignment
#
# With the adapter gated to the introspection prompt, no trainable parameter
# touches the input text, so the neuron the model computes at the introspected
# token IS the cached (base-model) value and the endogenous target is exact.
# With the gate off that guarantee is gone, and a high endogenous corr has two
# readings that the interventional metric cannot tell apart:
#   (a) drift is small -- the model reads its own, barely-moved neuron;
#   (b) drift is large and the adapter RECOMPUTES what the base model's neuron
#       would have been, as a side channel, instead of reading the live one.
# corr(tuned_live, base_live) at the same site separates them.
#
# The same pass also settles a cache/training ALIGNMENT question. data_prep/cache_signals.py
# tokenizes with add_special_tokens defaulting to True (BOS at index 0), while
# preprocess_example tokenizes with add_special_tokens=False, so a cached
# token_position p may correspond to training position p-1 -- exactly the -1 that the
# old post-hoc intervention script applied ("offset of -1 is label shift I think") and
# that IntrospectionRegressionModel.forward does NOT. Running the model with
# adapters DISABLED reproduces the model that built the cache, so the offset whose
# captured activation correlates ~1.0 with the cached column is the true
# convention. Anything else is an off-by-one in the intervention site.
# ======================================================================

DRIFT_OFFSETS = (-1, 0, 1)
# what IntrospectionRegressionModel.forward currently assumes (left-pad shift only)
TRAIN_OFFSET = 0


class ActivationCaptureHook:
    """Forward hook that RECORDS the target neuron without modifying it (returns
    None, so the module output passes through untouched).

    Indexing mirrors InterventionHook -- absolute, left-pad-adjusted positions --
    except that it handles a SUBSET of the batch (`row_indices`, the rows whose
    signal lives in this layer) and captures several `offsets` around the position
    in one forward, which is what the alignment check needs. Offsets are clamped
    into range, so a row whose introspected token sits at a sequence edge may
    report a padded/edge position for the outer offsets; that only affects the
    diagnostic offsets, never offset 0.
    """

    def __init__(self, row_indices, token_positions, signal_indices, offsets=DRIFT_OFFSETS):
        self.row_indices = row_indices
        self.token_positions = token_positions
        self.neuron_indices = signal_indices
        self.offsets = tuple(offsets)
        self.captured = {}

    def __call__(self, module, inputs, outputs):
        seq_len = outputs.shape[1]
        for off in self.offsets:
            pos = (self.token_positions + off).clamp(0, seq_len - 1)
            self.captured[off] = outputs[self.row_indices, pos, self.neuron_indices].detach().float().cpu()
        return None


# The Llama MLP is  down_proj( act_fn(gate_proj(x)) * up_proj(x) ), so for the target
# neuron n these three tensors play different roles, and the weight analysis
# (plotters/compare_lora_weight_patterns.py) found the adapter targets only two:
#   gate_proj row n  -> the neuron's OWN value (what we intervene on).  NOT targeted.
#   up_proj   row n  -> the multiplicative gate on its contribution.    rank 1 of 8192.
#   down_proj col n  -> the residual direction it writes.               rank 1 of 8192.
# up_proj's update came back ~14 sigma ANTI-aligned with the base row, which would
# flatten the gate's input-dependence -- i.e. make transmission less context-dependent,
# as a reliable relay would. TRANSMISSION_SITES measures that directly: a std ratio
# well below 1 for up_proj is the signature.
TRANSMISSION_SITES = ("up_proj", "gate_proj")


class VectorCaptureHook:
    """Like ActivationCaptureHook but keeps the WHOLE vector at the position, not one
    neuron. Used for the MLP's output: the target neuron's share of its layer's write
    is only meaningful against the size of that write."""

    def __init__(self, row_indices, token_positions):
        self.row_indices = row_indices
        self.token_positions = token_positions
        self.captured = None

    def __call__(self, module, inputs, outputs):
        out = outputs[0] if isinstance(outputs, tuple) else outputs
        pos = self.token_positions.clamp(0, out.shape[1] - 1)
        self.captured = out[self.row_indices, pos].detach().float().cpu()
        return None


def _capture_activations(model, batch, abs_pos, signal_layer, signal_index, offsets):
    """One forward over `batch`, capturing the target neuron at every offset AND the
    regression head's prediction from the same pass (the head reads the last-token
    final hidden state, exactly as eval_endogenous_test does -- so the prediction
    here is the endogenous readout, free of a second forward).

    Returns ({offset: [bs] float tensor}, preds [bs]). One hook per distinct signal
    layer, so a multi-neuron / multi-layer batch still needs only one forward."""
    bs = abs_pos.shape[0]
    out = {off: torch.full((bs,), float("nan")) for off in offsets}
    trans = {site: torch.full((bs,), float("nan")) for site in TRANSMISSION_SITES}
    mlp_norm = torch.full((bs,), float("nan"))
    hooks, handles, trans_hooks, mlp_hooks = [], [], [], []
    try:
        for layer in torch.unique(signal_layer).tolist():
            m = signal_layer == layer
            rows = torch.nonzero(m, as_tuple=True)[0]
            hook = ActivationCaptureHook(rows, abs_pos[m], signal_index[m], offsets)
            act_path = resolve_mlp_signal_path(model, layer, path_to_output=False)
            module = operator.attrgetter(act_path)(model)
            handles.append(module.register_forward_hook(hook))
            hooks.append((rows, hook))
            # same positions, same neuron index, on the transmission tensors
            for site in TRANSMISSION_SITES:
                site_hook = ActivationCaptureHook(rows, abs_pos[m], signal_index[m], (0,))
                site_module = operator.attrgetter(
                    act_path.rsplit(".", 1)[0] + "." + site)(model)
                handles.append(site_module.register_forward_hook(site_hook))
                trans_hooks.append((rows, site, site_hook))
            # the whole MLP write at this token, for the neuron's share of it
            mlp_hook = VectorCaptureHook(rows, abs_pos[m])
            mlp_module = operator.attrgetter(act_path.rsplit(".", 1)[0])(model)
            handles.append(mlp_module.register_forward_hook(mlp_hook))
            mlp_hooks.append((rows, mlp_hook))
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        lora_token_mask=batch.get("lora_token_mask"), do_intervention=False)
    finally:
        for h in handles:
            h.remove()
    for rows, hook in hooks:
        for off in offsets:
            out[off][rows.cpu()] = hook.captured[off]
    for rows, site, hook in trans_hooks:
        trans[site][rows.cpu()] = hook.captured[0]
    for rows, hook in mlp_hooks:
        if hook.captured is not None:
            mlp_norm[rows.cpu()] = hook.captured.norm(dim=-1)
    out["_transmission"] = trans
    out["_mlp_norm"] = mlp_norm
    head = model.regression_head
    h = outputs.hidden_states[-1][:, -1, :]
    preds = head(h.to(device=head.weight.device, dtype=head.weight.dtype)).squeeze(-1).float().cpu()
    return out, preds


def _capture_attention_logits(model, batch, abs_dst, abs_src, signal_layer,
                              signal_index, offsets, port_cls=AttentionLogitPort,
                              site="logit"):
    """One forward capturing the introspected attention signal at several offsets.

    The attention analogue of _capture_activations. It cannot reuse that function at
    all: an attention logit is not a module output (see models/attention_logits.py), so
    there is nothing to register a forward hook on. Ports do the reading instead, one
    per (layer, offset).

    An offset shifts the destination AND the source together, keeping their relative
    distance -- that distance IS the signal's identity, so moving only one would read a
    different signal rather than a mis-positioned one. Shifting both is exactly what an
    off-by-one in `token_position` would look like, which is what this check exists to
    catch.

    Positions are clamped, and only for the diagnostic offsets: at offset +1 the
    destination of a pos=-1 row runs past the end of the sequence. Offset 0 is never
    clamped in practice, and it is the only offset the verdict depends on.

    `port_cls`/`site` select which of the two attention sites to read: the pre-softmax
    logit (default, and what every caller meant before the weight port existed) or the
    post-softmax weight. Reading a P run through the logit port would compare a cached
    probability against a live logit, so the alignment corr would be near zero and the
    verdict would read as "the positions are off by one" -- the wrong diagnosis.
    """
    bs = abs_dst.shape[0]
    seq_len = batch["input_ids"].shape[1]
    out = {off: torch.full((bs,), float("nan")) for off in offsets}
    layer_ports, port_index = {}, {}
    for layer in torch.unique(signal_layer).tolist():
        m = signal_layer == layer
        rows = torch.nonzero(m, as_tuple=True)[0]
        ports = []
        for off in offsets:
            dst = (abs_dst[m] + off).clamp(0, seq_len - 1)
            # keep the source at the same distance behind the destination
            src = (abs_src[m] + off).clamp(0, seq_len - 1).minimum(dst)
            ports.append(port_cls(heads=signal_index[m], dst=dst, src=src))
        layer_ports[layer] = PortFanOut(ports)
        port_index[layer] = (rows, ports)
    with installed_ports(model, layer_ports, site=site):
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        lora_token_mask=batch.get("lora_token_mask"), do_intervention=False)
    for layer, (rows, ports) in port_index.items():
        for off, port in zip(offsets, ports):
            out[off][rows.cpu()] = port.read.cpu()
    head = model.regression_head
    h = outputs.hidden_states[-1][:, -1, :]
    # .detach() BEFORE .cpu(): moving a tensor to CPU does NOT break the autograd graph,
    # and the caller keeps every batch's `preds` in a list for the whole loop. Without
    # this, each batch's graph -- and every GPU activation it references -- stays alive
    # until the loop ends. That OOMed at 42GB on a 44GB card the first time this ran.
    # The decorator on the caller makes it moot in practice; keep it so this helper is
    # also safe if it is ever called from outside a no_grad context.
    preds = head(h.to(device=head.weight.device, dtype=head.weight.dtype)).squeeze(-1)
    preds = preds.float().detach().cpu()
    return out, preds


# no_grad, exactly as on measure_activation_drift below. Without it this walks the whole
# test split with autograd enabled, running TWO forwards per batch (tuned + adapters
# off), and nothing is freed until the loop ends -- 42GB on a 44GB card. It is a
# read-only diagnostic; there is never a backward pass here.
@torch.no_grad()
def measure_attention_alignment(model, dataset, collator, config, output_dir, suffix="",
                                offsets=DRIFT_OFFSETS, signal_type="attention_score"):
    """Attention counterpart of measure_activation_drift.

    Answers the same two questions with the same numbers, minus the MLP-specific
    transmission analysis (up_proj/gate_proj have no attention analogue):

      ALIGNMENT  corr(base_live, cached) per offset. Exactly one offset must be ~1.0,
                 and it must be 0. This is the ONLY automatic check that the new
                 extraction path reads the same quantity cache_signals wrote -- if
                 caching and training ended up on different attention implementations,
                 or the source position is off by one, it shows up here and nowhere
                 else. Do not skip it.
      DRIFT      corr(tuned_live, base_live) at offset 0: has the adapter moved the
                 signal the endogenous target was frozen from.

    Writes activation_drift{suffix}.{json,csv,png} under the same names as the MLP
    version so the existing analysis scripts keep finding it.

    `signal_type` picks the port site. It matters MORE for attention_weight than for
    attention_score: a post-softmax value depends on the whole row, so if caching and
    training ran under different attention implementations or different padding, what
    breaks is the NORMALIZING CONSTANT, not the offset. That failure keeps the best
    offset at 0 and only pulls the correlation down a little -- it looks like noise, not
    like a bug, which is exactly why this has to run.
    """
    site = port_site(signal_type)
    port_cls = AttentionWeightPort if site == "weight" else AttentionLogitPort
    quantity = "attention weight" if site == "weight" else "attention logit"
    model.eval()
    dl = DataLoader(dataset, batch_size=config.training_config.eval_batch_size,
                    collate_fn=collator, shuffle=False)
    cached_all, pos_all, pred_all = [], [], []
    tuned_all = {off: [] for off in offsets}
    base_all = {off: [] for off in offsets}

    for batch in dl:
        batch = batch.to(model.device)
        token_position = batch["token_position"]
        # same left-pad correction the intervention path applies
        num_left_pad = (batch["attention_mask"] == 0).sum(dim=1).to(token_position.device)
        abs_dst = token_position + num_left_pad
        abs_src = batch["src_token_position"] + num_left_pad
        signal_layer, signal_index = batch["signal_layer"], batch["signal_index"]

        tuned, preds = _capture_attention_logits(model, batch, abs_dst, abs_src,
                                                 signal_layer, signal_index, offsets,
                                                 port_cls=port_cls, site=site)
        with model.disable_adapter():
            base, _ = _capture_attention_logits(model, batch, abs_dst, abs_src,
                                                signal_layer, signal_index, offsets,
                                                port_cls=port_cls, site=site)
        cached_all.append(batch["activation"].float().cpu())
        pred_all.append(preds)
        pos_all.append(token_position.cpu())
        for off in offsets:
            tuned_all[off].append(tuned[off])
            base_all[off].append(base[off])

    cached = torch.cat(cached_all).numpy()
    pred = torch.cat(pred_all).numpy()
    pos = torch.cat(pos_all).numpy()
    tuned = {off: torch.cat(v).numpy() for off, v in tuned_all.items()}
    base = {off: torch.cat(v).numpy() for off, v in base_all.items()}

    alignment = [dict(offset=off, **_drift_stats(base[off], cached)) for off in offsets]
    align_df = pd.DataFrame(alignment)
    corrs = {o: _corr_or_nan(base[o], cached) for o in offsets}
    best = max(offsets, key=lambda o: -np.inf if np.isnan(corrs[o]) else corrs[o])
    best_corr = corrs[best]
    drift_vs_base = _drift_stats(tuned[TRAIN_OFFSET], base[TRAIN_OFFSET])
    drift_vs_cached = _drift_stats(tuned[TRAIN_OFFSET], cached)
    readout = {
        "corr_pred_cached": _corr_or_nan(pred, cached),
        "corr_pred_tuned_live": _corr_or_nan(pred, tuned[TRAIN_OFFSET]),
        "corr_pred_base_live": _corr_or_nan(pred, base[TRAIN_OFFSET]),
    }
    readout["coupling_gap"] = readout["corr_pred_tuned_live"] - readout["corr_pred_cached"]

    metrics = {
        "signal_family": "attention",
        "signal_type": signal_type,
        "train_offset": TRAIN_OFFSET,
        "alignment": alignment,
        "best_alignment_offset": int(best),
        "best_alignment_corr": best_corr,
        "alignment_ok": bool(best == TRAIN_OFFSET and best_corr > 0.99),
        "drift_tuned_vs_base": drift_vs_base,
        "drift_tuned_vs_cached": drift_vs_cached,
        "readout_tracking": readout,
        # no transmission_path: up_proj/gate_proj are MLP-only, and there is no
        # attention analogue of "the neuron's share of its layer's write".
        "transmission_path": None,
    }
    out = Path(output_dir)
    with open(out / f"activation_drift{suffix}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"token_position": pos, "cached": cached, "prediction": pred,
                  **{f"base_off{off}": base[off] for off in offsets},
                  **{f"tuned_off{off}": tuned[off] for off in offsets}}
                 ).to_csv(out / f"activation_drift{suffix}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].scatter(cached, base[best], s=10, alpha=0.4)
    axes[0].set_xlabel(f"cached {quantity}")
    axes[0].set_ylabel(f"live, adapters OFF (offset {best:+d})")
    axes[0].set_title(f"cache alignment (corr={best_corr:.4f})")
    axes[1].scatter(base[TRAIN_OFFSET], tuned[TRAIN_OFFSET], s=10, alpha=0.4)
    axes[1].set_xlabel("live, adapters OFF"); axes[1].set_ylabel("live, adapter ON")
    axes[1].set_title(f"drift (corr={drift_vs_base['corr']:.4f}, "
                      f"std ratio={drift_vs_base['std_ratio']:.3f})")
    for ax in axes:
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out / f"activation_drift{suffix}.png", dpi=150)
    plt.close()

    print(f"[drift] {quantity} cache alignment (adapters OFF vs cached column):")
    print(align_df.to_string(index=False))
    if metrics["alignment_ok"]:
        print(f"[drift] alignment OK: offset {TRAIN_OFFSET:+d} reproduces the cache "
              f"(corr={best_corr:.4f}).")
    else:
        print(f"[drift] *** ALIGNMENT MISMATCH *** best offset {best:+d} (corr={best_corr:.4f}), "
              f"expected {TRAIN_OFFSET:+d}. Either the (dst, src) pair is off by "
              f"{TRAIN_OFFSET - best:+d} tokens, or caching and training ran under "
              f"different attention implementations -- check capture.attn_implementation "
              f"in the cache metadata.json against this run.")
    print(f"[drift] tuned vs base @offset {TRAIN_OFFSET:+d}: corr={drift_vs_base['corr']:.4f} "
          f"std_ratio={drift_vs_base['std_ratio']:.3f}")
    return metrics


def _drift_stats(x, ref):
    """Agreement of `x` with reference `ref`: correlation, spread ratio, mean
    shift, and RMSE in units of the reference's std (so it reads as 'how big is
    the discrepancy compared to the signal we are trying to predict')."""
    ref_std = float(np.std(ref))
    return {
        "corr": _corr_or_nan(x, ref),
        "std_ratio": float(np.std(x) / ref_std) if ref_std > 1e-12 else float("nan"),
        # raw means as well as the shift: "is the tuned value dominated by a constant?"
        # needs mean and std on the same footing, not a difference
        "mean_tuned": float(np.mean(x)),
        "mean_base": float(np.mean(ref)),
        "mean_shift": float(np.mean(x) - np.mean(ref)),
        "std_tuned": float(np.std(x)),
        "std_base": ref_std,
        "rel_rmse": float(np.sqrt(np.mean((x - ref) ** 2)) / ref_std) if ref_std > 1e-12 else float("nan"),
    }


@torch.no_grad()
def measure_activation_drift(model, dataset, collator, config, output_dir, suffix="",
                             offsets=DRIFT_OFFSETS, n_bins=6):
    """
    Capture the introspected neuron's LIVE activation twice on the test split --
    once with the adapter active (tuned) and once with adapters disabled (base) --
    and compare both against the cached activation the endogenous loss trains on.

    Writes activation_drift{suffix}.{json,csv,png} and returns the metrics dict.
    Two questions answered:

    ALIGNMENT  corr(base, cached) per offset. The adapter-disabled model IS the one
               that built the cache, so exactly one offset should be ~1.0. If that
               offset is not TRAIN_OFFSET, the intervention hook has been injecting
               into the wrong token.
    DRIFT      corr(tuned, base) at TRAIN_OFFSET -- alignment-independent, since
               both are read at the same site. ~1.0 with std_ratio ~1.0 means the
               endogenous target is still valid and reading (a) is right; a low
               corr with a still-high endogenous score points at recompute (b).
    """
    model.eval()
    dl = DataLoader(dataset, batch_size=config.training_config.eval_batch_size,
                    collate_fn=collator, shuffle=False)
    cached_all, pos_all, pred_all = [], [], []
    tuned_all = {off: [] for off in offsets}
    base_all = {off: [] for off in offsets}
    trans_all = {"tuned": {s: [] for s in TRANSMISSION_SITES},
                 "base": {s: [] for s in TRANSMISSION_SITES}}
    mlp_all = {"tuned": [], "base": []}

    for batch in dl:
        batch = batch.to(model.device)
        token_position = batch["token_position"]
        # same left-pad correction the intervention hook applies (prediction
        # batches are left-padded, so the text starts at index num_left_pad)
        num_left_pad = (batch["attention_mask"] == 0).sum(dim=1).to(token_position.device)
        abs_pos = token_position + num_left_pad
        signal_layer, signal_index = batch["signal_layer"], batch["signal_index"]

        tuned, preds = _capture_activations(model, batch, abs_pos, signal_layer, signal_index, offsets)
        with model.disable_adapter():
            base, _ = _capture_activations(model, batch, abs_pos, signal_layer, signal_index, offsets)

        cached_all.append(batch["activation"].float().cpu())
        pred_all.append(preds)
        pos_all.append(token_position.cpu())
        for site in TRANSMISSION_SITES:
            trans_all["tuned"][site].append(tuned["_transmission"][site])
            trans_all["base"][site].append(base["_transmission"][site])
        mlp_all["tuned"].append(tuned["_mlp_norm"])
        mlp_all["base"].append(base["_mlp_norm"])
        for off in offsets:
            tuned_all[off].append(tuned[off])
            base_all[off].append(base[off])

    cached = torch.cat(cached_all).numpy()
    pred = torch.cat(pred_all).numpy()
    pos = torch.cat(pos_all).numpy()
    tuned = {off: torch.cat(v).numpy() for off, v in tuned_all.items()}
    base = {off: torch.cat(v).numpy() for off, v in base_all.items()}
    transmission, site_vals = {}, {}
    for site in TRANSMISSION_SITES:
        t, b = (torch.cat(trans_all["tuned"][site]).numpy(),
                torch.cat(trans_all["base"][site]).numpy())
        site_vals[site] = (t, b)
        transmission[site] = _drift_stats(t, b)

    # What actually reaches the residual stream is the PRODUCT a[n]*u[n], scaled by the
    # (fixed) down_proj column. Measuring the factors separately understates the
    # channel: a 30% smaller signal times a 20x larger gate is still a large boost.
    prod_t = tuned[TRAIN_OFFSET] * site_vals["up_proj"][0]
    prod_b = base[TRAIN_OFFSET] * site_vals["up_proj"][1]
    transmission["product_act_x_up"] = _drift_stats(prod_t, prod_b)
    denom = float(np.mean(np.abs(prod_b)))
    transmission["product_act_x_up"]["abs_mean_ratio"] = (
        float(np.mean(np.abs(prod_t)) / denom) if denom > 1e-12 else float("nan"))

    # ... and against the size of the whole MLP write at that token: if the layer's
    # output grew by the same factor, nothing was singled out.
    mlp_t, mlp_b = (torch.cat(mlp_all["tuned"]).numpy(), torch.cat(mlp_all["base"]).numpy())
    mlp_ok = np.isfinite(mlp_t) & np.isfinite(mlp_b) & (mlp_b > 1e-12)
    transmission["mlp_output_norm"] = {
        "mean_tuned": float(np.mean(mlp_t[mlp_ok])) if mlp_ok.any() else float("nan"),
        "mean_base": float(np.mean(mlp_b[mlp_ok])) if mlp_ok.any() else float("nan"),
        "norm_ratio": float(np.mean(mlp_t[mlp_ok] / mlp_b[mlp_ok])) if mlp_ok.any() else float("nan"),
    }

    # ---- alignment: which offset reproduces the cache? ----
    alignment = [dict(offset=off, **_drift_stats(base[off], cached)) for off in offsets]
    align_df = pd.DataFrame(alignment)
    best = max(offsets, key=lambda o: (-np.inf if np.isnan(_corr_or_nan(base[o], cached))
                                       else _corr_or_nan(base[o], cached)))
    best_corr = _corr_or_nan(base[best], cached)

    # ---- drift at the offset training actually uses ----
    drift_vs_base = _drift_stats(tuned[TRAIN_OFFSET], base[TRAIN_OFFSET])
    drift_vs_cached = _drift_stats(tuned[TRAIN_OFFSET], cached)

    # ---- what is the readout actually tracking? (introspection coupling) ----
    # The head was trained against CACHED activations -- the base model's, frozen at
    # cache time. Once the adapter is active over the input text the model's own
    # neuron moves, and the two stop being the same target:
    #   corr(pred, tuned_live) > corr(pred, cached) -> COUPLED: the report follows the
    #       model's CURRENT activation, i.e. it introspects on itself rather than
    #       reproducing the value it was supervised on.
    #   corr(pred, cached) > corr(pred, tuned_live) -> the head tracks the frozen
    #       training target (or predicts it from context) rather than the live signal.
    # corr(pred, cached) is the endogenous score itself, so it doubles as a
    # cross-check against endogenous_test_metrics (same rows, same order).
    readout = {
        "corr_pred_cached": _corr_or_nan(pred, cached),
        "corr_pred_tuned_live": _corr_or_nan(pred, tuned[TRAIN_OFFSET]),
        "corr_pred_base_live": _corr_or_nan(pred, base[TRAIN_OFFSET]),
    }
    readout["coupling_gap"] = readout["corr_pred_tuned_live"] - readout["corr_pred_cached"]

    rows = [{"position_bin": "all", "pos_lo": float(pos.min()), "pos_hi": float(pos.max()),
             "n": int(len(pos)), **drift_vs_base}]
    if len(np.unique(pos)) > 1:
        edges = np.unique(np.quantile(pos, np.linspace(0, 1, n_bins + 1)))
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pos >= lo) & (pos <= hi)
            if m.sum() >= 40:
                rows.append({"position_bin": f"[{lo:.0f},{hi:.0f}]", "pos_lo": float(lo),
                             "pos_hi": float(hi), "n": int(m.sum()),
                             **_drift_stats(tuned[TRAIN_OFFSET][m], base[TRAIN_OFFSET][m])})
    by_pos = pd.DataFrame(rows)

    metrics = {
        "train_offset": TRAIN_OFFSET,
        "alignment": alignment,
        "best_alignment_offset": int(best),
        "best_alignment_corr": best_corr,
        "alignment_ok": bool(best == TRAIN_OFFSET and best_corr > 0.99),
        "drift_tuned_vs_base": drift_vs_base,
        "drift_tuned_vs_cached": drift_vs_cached,
        "readout_tracking": readout,
        "transmission_path": transmission,
        "drift_by_position": rows,
    }
    out = Path(output_dir)
    with open(out / f"activation_drift{suffix}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"token_position": pos, "cached": cached, "prediction": pred,
                  **{f"base_off{off}": base[off] for off in offsets},
                  **{f"tuned_off{off}": tuned[off] for off in offsets}}
                 ).to_csv(out / f"activation_drift{suffix}.csv", index=False)
    by_pos.to_csv(out / f"activation_drift_by_position{suffix}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].scatter(cached, base[best], s=10, alpha=0.4)
    axes[0].set_xlabel("cached activation"); axes[0].set_ylabel(f"live, adapters OFF (offset {best:+d})")
    axes[0].set_title(f"cache alignment (corr={best_corr:.4f})")
    axes[1].scatter(base[TRAIN_OFFSET], tuned[TRAIN_OFFSET], s=10, alpha=0.4)
    axes[1].set_xlabel("live, adapters OFF"); axes[1].set_ylabel("live, adapter ON")
    axes[1].set_title(f"drift (corr={drift_vs_base['corr']:.4f}, "
                      f"std ratio={drift_vs_base['std_ratio']:.3f})")
    for ax in axes:
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out / f"activation_drift{suffix}.png", dpi=150)
    plt.close()

    print("[drift] cache alignment (live activation with adapters OFF vs cached column):")
    print(align_df.to_string(index=False))
    if metrics["alignment_ok"]:
        print(f"[drift] alignment OK: offset {TRAIN_OFFSET:+d} reproduces the cache "
              f"(corr={best_corr:.4f}); the intervention hook targets the intended token.")
    else:
        print(f"[drift] *** ALIGNMENT MISMATCH *** best offset is {best:+d} (corr={best_corr:.4f}), "
              f"but training/intervention assumes {TRAIN_OFFSET:+d}. The hook is injecting into the "
              f"token at cached_position{TRAIN_OFFSET - best:+d}, i.e. NOT the token the prompt names. "
              f"Interventional metrics stay valid (the injected value is ground truth wherever it "
              f"lands) but the endogenous target and the 'reads the named token' claim do not.")
    print(f"[drift] tuned vs base @offset {TRAIN_OFFSET:+d}: corr={drift_vs_base['corr']:.4f} "
          f"std_ratio={drift_vs_base['std_ratio']:.3f} mean_shift={drift_vs_base['mean_shift']:+.4f} "
          f"rel_rmse={drift_vs_base['rel_rmse']:.3f}")
    print(f"[drift] tuned vs cached @offset {TRAIN_OFFSET:+d}: corr={drift_vs_cached['corr']:.4f} "
          f"rel_rmse={drift_vs_cached['rel_rmse']:.3f}")
    print(f"[coupling] readout vs: cached(trained target)={readout['corr_pred_cached']:.4f}  "
          f"tuned_live(own activation)={readout['corr_pred_tuned_live']:.4f}  "
          f"base_live={readout['corr_pred_base_live']:.4f}  "
          f"gap(tuned-cached)={readout['coupling_gap']:+.4f}")
    if readout["coupling_gap"] > 0.01:
        print("[coupling] => COUPLED: the report tracks the model's OWN current activation better "
              "than the frozen target it was trained on -- introspection followed the signal as "
              "the signal moved.")
    elif readout["coupling_gap"] < -0.01:
        print("[coupling] => NOT coupled: the report tracks the CACHED training target better than "
              "the live neuron, so it is reproducing supervision (or predicting from context) "
              "rather than reading the model's current state.")
    else:
        print("[coupling] => indistinguishable: live and cached activations are too similar here "
              "for the two to separate (expected when drift is small).")
    if drift_vs_base["corr"] > 0.99 and 0.95 <= drift_vs_base["std_ratio"] <= 1.05:
        print("[drift] => drift is negligible: the cached endogenous target is still what the model "
              "computes, so endogenous corr means what it did under the gated setup.")
    else:
        print("[drift] => the adapter MOVED the neuron. A high endogenous corr against the CACHED "
              "target now needs the recompute explanation ruled out: check whether corr(tuned,cached) "
              "exceeds corr(tuned,base) -- reading the live neuron cannot beat the live neuron itself.")
    print("[transmission] target neuron's MLP path, tuned vs base at the same site:")
    for site in list(TRANSMISSION_SITES) + ["product_act_x_up"]:
        st = transmission[site]
        # CV = std/|mean|: small => the value is dominated by a constant, which is how
        # a bias-free projection turns a context-dependent gate into a fixed gain
        cv = abs(st["std_tuned"] / st["mean_tuned"]) if abs(st["mean_tuned"]) > 1e-12 else float("nan")
        print(f"    {site:18s} corr={st['corr']:+.4f}  std_ratio={st['std_ratio']:.3f}  "
              f"mean {st['mean_base']:+.4f} -> {st['mean_tuned']:+.4f}  "
              f"std {st['std_base']:.4f} -> {st['std_tuned']:.4f}  CV_tuned={cv:.3f}")
    prod = transmission["product_act_x_up"]
    mlp = transmission["mlp_output_norm"]
    print(f"    {'mlp_output_norm':18s} mean {mlp['mean_base']:.4f} -> {mlp['mean_tuned']:.4f}  "
          f"ratio={mlp['norm_ratio']:.3f}")
    if np.isfinite(prod.get("abs_mean_ratio", float("nan"))) and np.isfinite(mlp["norm_ratio"]):
        print(f"[transmission] the neuron's scalar into the residual changed "
              f"{prod['abs_mean_ratio']:.2f}x while the whole MLP write changed "
              f"{mlp['norm_ratio']:.2f}x"
              + (" => this ONE neuron was singled out"
                 if prod["abs_mean_ratio"] > 2 * mlp["norm_ratio"] else
                 " => no more than the layer as a whole"))
    up = transmission.get("up_proj", {})
    gate = transmission.get("gate_proj", {})
    if up and gate:
        if up.get("std_ratio", 1) < 0.8:
            print("[transmission] => the multiplicative gate was FLATTENED: the neuron's "
                  "contribution is now less input-dependent, which is what a reliable relay "
                  "needs. Combined with an untouched gate_proj, the adapter rewired the "
                  "neuron's OUTPUT PATH without changing the value it reports.")
        elif gate.get("std_ratio", 1) < 0.8:
            print("[transmission] => gate_proj moved: the adapter changed the SIGNAL itself, "
                  "not just its transmission. Endogenous targets are then measuring a neuron "
                  "the model rewrote.")
        else:
            print("[transmission] => neither path was reshaped much at this operating point.")
    print("[drift] by position:")
    print(by_pos.to_string(index=False))
    return metrics


# ======================================================================
# Routing probe: does the injected value REACH the readout of the TRAINED
# model (fresh-probe ceiling) vs does the TRAINED HEAD read it, by position.
# ceiling high, head low -> routed but not read (optimization/head)
# ceiling low            -> not routed (routing/capacity)
# NB: this is the TRAINED model (a ceiling conditional on this run), NOT a
# pre-LoRA base-model floor.
# ======================================================================
@torch.no_grad()
def _extract_interv_readout(model, dataset, collator, config, scale, distribution="extreme"):
    """Under injection at `scale`, collect readout features, injected values,
    trained-head preds, and token_positions."""
    # dropout is a no-op under eval(); before the GatedLinear dropout fix it was a
    # no-op either way, so these forwards used to inherit train mode harmlessly.
    model.eval()
    prev_dist, prev_scale = model.intervention_distribution, model.intervention_scale
    model.intervention_distribution, model.intervention_scale = distribution, scale
    model.seed_intervention_rng(config.seed)
    dl = DataLoader(dataset, batch_size=config.training_config.eval_batch_size,
                    collate_fn=collator, shuffle=False)
    head = model.regression_head
    feats, inj, hpred, pos = [], [], [], []
    for batch in dl:
        batch = batch.to(model.device)
        # Unlike compute_loss (which forwards the whole batch), this builds the kwargs by
        # hand -- so the attention columns have to be added explicitly. Omitting them
        # would not error: forward would take the MLP branch and inject the value into
        # MLP neuron number `head` instead of the attention logit. The model asserts
        # against that too, but passing them is the fix.
        interv_kwargs = {k: batch[k] for k in ("src_token_position", "signal_src_offset")
                         if k in batch}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    lora_token_mask=batch.get("lora_token_mask"),
                    token_position=batch["token_position"], signal_layer=batch["signal_layer"],
                    signal_index=batch["signal_index"], **interv_kwargs,
                    do_intervention=True, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]
        feats.append(h.float().cpu())
        inj.append(out.intervention_values.float().cpu())
        hpred.append(head(h.to(device=head.weight.device, dtype=head.weight.dtype)).squeeze(-1).float().cpu())
        pos.append(batch["token_position"].cpu())
    model.intervention_distribution, model.intervention_scale = prev_dist, prev_scale
    return (torch.cat(feats).numpy(), torch.cat(inj).numpy(),
            torch.cat(hpred).numpy(), torch.cat(pos).numpy())


def _corr_or_nan(a, b):
    return float(pearsonr(a, b)[0]) if (np.std(a) > 1e-12 and np.std(b) > 1e-12) else float("nan")


ROUTING_ALPHAS = np.logspace(-1, 5, 13)   # wide grid so RidgeCV can regularize hard when n ~ H


def _ceiling_and_head(X, inj, hpred, seed, n_pca, split=0.7):
    """
    On ONE seeded held-out split: fit a well-regularized probe (standardize ->
    PCA -> RidgeCV) on train, and score BOTH the fresh probe (ceiling) and the
    trained head on the SAME held-out rows. Regularizing + PCA is essential when
    n_examples ~ n_features, else the ceiling overfits and comes out < head
    (which is impossible for a true ceiling).
    Returns (ceiling_corr, head_corr, best_alpha).
    """
    n = len(inj)
    if n < 20:
        return float("nan"), float("nan"), float("nan")
    perm = np.random.RandomState(seed).permutation(n)
    ntr = int(n * split)
    tr, te = perm[:ntr], perm[ntr:]
    if len(tr) < 5 or len(te) < 5:
        return float("nan"), float("nan"), float("nan")
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    if n_pca and n_pca < Xtr.shape[1]:
        pca = PCA(n_components=min(n_pca, len(tr) - 1)).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    reg = RidgeCV(alphas=ROUTING_ALPHAS).fit(Xtr, inj[tr])
    ceil = _corr_or_nan(reg.predict(Xte), inj[te])
    head = _corr_or_nan(hpred[te], inj[te])          # head on the SAME held-out rows
    return ceil, head, float(getattr(reg, "alpha_", np.nan))


def routing_probe_by_position(model, dataset, collator, config, output_dir, scale, n_pca=256, n_bins=6,
                              suffix=""):
    X, inj, hpred, pos = _extract_interv_readout(model, dataset, collator, config, scale)

    def row(bin_label, m):
        lo, hi = float(pos[m].min()), float(pos[m].max())
        c, h, a = _ceiling_and_head(X[m], inj[m], hpred[m], config.seed, n_pca)
        return {"position_bin": bin_label, "pos_lo": lo, "pos_hi": hi, "n": int(m.sum()),
                "ceiling_corr": c, "head_corr": h, "best_alpha": a}

    rows = [row("all", np.ones(len(inj), bool))]
    if len(np.unique(pos)) > 1:
        edges = np.unique(np.quantile(pos, np.linspace(0, 1, n_bins + 1)))
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pos >= lo) & (pos <= hi)
            if m.sum() >= 40:
                rows.append(row(f"[{lo:.0f},{hi:.0f}]", m))
    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / f"routing_probe_by_position{suffix}.csv", index=False)
    print(f"[joint] routing probe (inject scale={scale}, suffix={suffix or 'native'}):")
    print(df.to_string(index=False))

    binned = df[df["position_bin"] != "all"]
    if len(binned) >= 2:
        centers = (binned["pos_lo"] + binned["pos_hi"]) / 2
        plt.figure(figsize=(7, 5))
        plt.plot(centers, binned["ceiling_corr"], "o-", label="routing ceiling (fresh probe)")
        plt.plot(centers, binned["head_corr"], "s--", label="trained head")
        plt.xlabel("introspected token position"); plt.ylabel("interventional corr")
        plt.title(f"routing vs reading by position (inject scale={scale}){suffix}")
        plt.ylim(-0.05, 1.02); plt.legend(); plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"routing_probe_by_position{suffix}.png", dpi=150)
        plt.close()
    return df


def interventional_scale_sweep(model, dataset, collator, config, output_dir, scales,
                               distribution="gaussian", n_pca=256, suffix=""):
    """
    Interventional corr vs injection scale, for BOTH the trained head and a fresh
    ceiling probe (gaussian: N(mean, scale*std)).

    corr = 1 / sqrt(1 + sigma_eps^2 / (g^2 sigma_v^2)) for a linear channel with
    routing gain g and fixed context-noise eps -> corr rises with scale purely
    from SNR, no floor/nonlinearity needed. Decision:
      * CEILING also rises with scale  -> SNR/routing-gain limited (fundamental;
        natural-scale signal is swamped by context noise; needs higher gain).
      * CEILING flat & high, head rises -> signal is linearly present at natural
        scale; the head was just trained at large scale (distribution mismatch).
    """
    rows = []
    for s in scales:
        X, inj, hpred, _ = _extract_interv_readout(model, dataset, collator, config, s, distribution=distribution)
        ceil, head, alpha = _ceiling_and_head(X, inj, hpred, config.seed, n_pca)
        rows.append({"scale": s, "head_corr": head, "ceiling_corr": ceil,
                     "inj_std": float(np.std(inj)), "best_alpha": alpha})
    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / f"interventional_scale_sweep{suffix}.csv", index=False)
    print(f"[joint] interventional scale sweep ({distribution}, suffix={suffix or 'native'}):")
    print(df.to_string(index=False))
    plt.figure(figsize=(7, 5))
    plt.plot(df["scale"], df["ceiling_corr"], "o-", label="ceiling (fresh probe)")
    plt.plot(df["scale"], df["head_corr"], "s--", label="trained head")
    plt.xscale("log")
    plt.xlabel("injection scale (x natural std)"); plt.ylabel("interventional corr")
    plt.title(f"interventional corr vs injection scale ({distribution}){suffix}")
    plt.ylim(-0.05, 1.02); plt.grid(alpha=0.2, which="both"); plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"interventional_scale_sweep{suffix}.png", dpi=150)
    plt.close()
    return df


def _logit(p):
    p = np.clip(p, EMPIRICAL_EPS, 1 - EMPIRICAL_EPS)
    return np.log(p / (1 - p))


def interventional_ceiling(model, dataset, collator, config, output_dir, distribution,
                           n_pca=256):
    """Ceiling probe and head at scale 1 under `distribution`, in BOTH t and logit(t).

    Why logit(t) too: a sigmoid head is linear in log-odds, and so is the weight port's
    intervention (a shift of one attention logit). A ceiling that is ~0 in t but high in
    logit(t) would mean the value IS routed and only the probability-space probe misses
    it; ~0 in both means it is not routed to the readout at all.
    """
    X, inj, hpred, _ = _extract_interv_readout(model, dataset, collator, config, 1.0,
                                               distribution=distribution)
    assert inj.shape == hpred.shape == (X.shape[0],), f"{inj.shape} {hpred.shape} {X.shape}"
    c_t, h_t, a_t = _ceiling_and_head(X, inj, hpred, config.seed, n_pca)
    # the head's prediction is only a probability under a sigmoid head; a linear head
    # can predict <= 0, where logit is undefined, so its logit-space corr is left NaN
    sigmoid_head = type(model.regression_head).__name__ == "SigmoidRegressionHead"
    c_l, h_l, a_l = _ceiling_and_head(X, _logit(inj), _logit(hpred) if sigmoid_head else hpred,
                                      config.seed, n_pca)
    out = {"distribution": distribution, "scale": 1.0, "n": int(len(inj)),
           "ceiling_corr_t": c_t, "head_corr_t": h_t, "best_alpha_t": a_t,
           "ceiling_corr_logit": c_l, "head_corr_logit": h_l if sigmoid_head else float("nan"),
           "best_alpha_logit": a_l,
           "inj_mean": float(inj.mean()), "inj_std": float(inj.std()),
           "inj_logit_mean": float(_logit(inj).mean()), "inj_logit_std": float(_logit(inj).std())}
    with open(Path(output_dir) / f"interventional_ceiling_{distribution}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[joint] interventional ceiling ({distribution}@1): "
          f"t: ceiling={c_t:.4f} head={h_t:.4f} | logit: ceiling={c_l:.4f} head={out['head_corr_logit']:.4f}")
    return out


# Per-row columns the predictions CSVs need but the model must not see (see
# pair_metrics.attach_provenance for why they cannot ride in the batch).
PROVENANCE_COLS = ("doc_index", "dst_token_unique", "src_token_unique")


def eval_provenance(raw, preprocessed):
    """Provenance frame for an eval split, aligned row-for-row with what
    drop_invalid_positions will keep. Call BEFORE drop_invalid_positions: it reads the
    same `position_valid` flag that function filters on. None when the cache predates
    these columns."""
    cols = [c for c in PROVENANCE_COLS if c in raw.column_names]
    if not cols:
        return None
    side = raw.select_columns(cols).to_pandas()
    assert len(side) == len(preprocessed), (
        f"raw {len(side)} rows vs preprocessed {len(preprocessed)}: preprocess is 1:1")
    if "position_valid" in preprocessed.column_names:
        valid = np.asarray(preprocessed["position_valid"], dtype=bool)
        side = side[valid].reset_index(drop=True)
    return side


def run_interventional_evals(model, dataset, collator, config, output_dir, distributions,
                             n_pca=256, source_checkpoint=None, provenance=None):
    """Headline interventional test eval, once per distribution (suffix _<dist>)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if source_checkpoint is not None:
        # an eval-only pass writes next to nothing else; record whose weights these are
        with open(Path(output_dir) / "interv_eval_source.json", "w") as f:
            json.dump({"checkpoint": str(source_checkpoint)}, f, indent=2)
    for d in distributions:
        evaluate_interventional_test(model, eval_dataset=dataset, collator=collator,
                                     config=config, output_dir=output_dir,
                                     distribution=d, suffix=f"_{d}", provenance=provenance)
        if d == "empirical_logit":
            interventional_ceiling(model, dataset, collator, config, output_dir, d, n_pca=n_pca)


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(args.config.read_text())
    if not config.use_regressor:
        raise SystemExit("Joint training targets the regressor head (use_regressor must be True).")

    interv_eval_dists = [d.strip() for d in args.interv_eval_distributions.split(",") if d.strip()]
    allowed = {"gaussian", "uniform", "extreme", "prob_uniform", "empirical_logit",
               "empirical_prob"}
    bad = [d for d in interv_eval_dists if d not in allowed]
    if bad or not interv_eval_dists:
        raise ValueError(f"--interv-eval-distributions {interv_eval_dists} has {bad} "
                         f"(allowed: {sorted(allowed)})")
    if args.interv_eval_only and args.checkpoint is None:
        raise ValueError("--interv-eval-only needs --checkpoint (the run to evaluate)")

    cache_dir = config.dataset_cache_dir
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if ((output_dir / "routing_probe_by_position.csv").exists()
            and not args.overwrite_cache and not args.diag_only and not args.ood_only
            and not args.length_only and not args.neuron_gen_only
            and not args.interv_eval_only):
        print(f"Experiments ran already, skipping.. {output_dir}")
        return

    if args.prepare_only:
        # Only the config is needed: the derived caches are keyed by its content, and
        # cache_signals builds them in subprocesses. Failures are collected rather than
        # raised by the prepare functions, so turn them into an exit code here.
        failed = {}
        if args.length_eval:
            import evals.score_length_generalization as slg
            lengths = ([int(x) for x in args.eval_lengths.split(",")]
                       if args.eval_lengths else list(slg.DEFAULT_EVAL_LENGTHS))
            slg.prepare_length_caches(config, eval_lengths=lengths)
            failed.update({f"len{k}": v for k, v in slg.prepare_length_caches.errors.items()})
        if args.ood_eval:
            import evals.score_ood_generalization as sog
            corpora = ([c.strip() for c in args.ood_corpora.split(",")]
                       if args.ood_corpora else list(sog.DEFAULT_CORPORA))
            sog.prepare_ood_caches(config, corpora=corpora)
            failed.update({f"ood_{k}": v for k, v in sog.prepare_ood_caches.errors.items()})
        if failed:
            raise SystemExit(f"[joint] --prepare-only: {len(failed)} cache(s) failed: {failed}")
        print("[joint] --prepare-only: every requested eval cache is ready")
        return

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_added_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": [INTROSPECTION_SEPARATOR]})

    # ---- data ----
    train_raw = load_from_disk(str(cache_dir / "train"))
    dev_raw = load_from_disk(str(cache_dir / "dev"))
    test_raw = load_from_disk(str(cache_dir / "test"))

    # a shared cache may hold several signals (one forward pass per text serves them
    # all); config.signal_filter picks this run's neuron out of it. No-op when unset.
    train_raw = apply_signal_filter(train_raw, config.signal_filter, "train")
    dev_raw = apply_signal_filter(dev_raw, config.signal_filter, "dev")
    test_raw = apply_signal_filter(test_raw, config.signal_filter, "test")

    if args.max_train_examples is not None and args.max_train_examples < len(train_raw):
        # Contiguous prefix, NOT a shuffle: cache_signals emits every signal of a text
        # consecutively (for text: for signal: append), so a prefix holds an equal number
        # of rows per neuron by construction. Shuffling would only approximate that, and
        # an unbalanced cap would confound the per-neuron data budget this arm is setting.
        n = args.max_train_examples
        print(f"[joint] --max-train-examples: {len(train_raw)} -> {n} train rows")
        train_raw = train_raw.select(range(n))

    df = train_raw.to_pandas()
    # An attention signal is (layer, head, source offset); an MLP neuron is (layer,
    # index). Grouping attention rows on (layer, head) alone would pool ten different
    # source offsets -- whose ranges differ by an order of magnitude between the BOS
    # sink and a mid-context token -- into one range, and every intervention would then
    # be drawn from the wrong distribution. signal_range_key builds the same key that
    # IntrospectionRegressionModel.sample_intervention_values looks up.
    attention_run = bool(len(df)) and is_attention_type(df["signal_type"].iloc[0])
    if attention_run:
        assert df["src_offset"].notna().all(), (
            "attention rows without src_offset -- cache predates attention signals")
        group_cols = ["signal_layer", "signal_index", "src_offset"]
    else:
        group_cols = ["signal_layer", "signal_index"]
    # A post-softmax weight lives in (0,1), and every built-in intervention distribution
    # leaves that interval (see make_bounded_prob_intervention_fn). Its sampler works in
    # log-odds space, so the log-odds moments ride along in the same dict. They are EXTRA
    # keys: sample_intervention_values forwards stats as **kwargs and its own branches
    # index by name, so nothing that does not ask for them can see them, and no MLP, SAE
    # or attention_score run gets them at all.
    signal_type = df["signal_type"].iloc[0] if len(df) else None
    prob_signal = signal_type == "attention_weight"
    # The log-odds histogram is read ONLY by the `empirical_logit` sampler. Building it
    # unconditionally made every attention_weight run depend on logit_histogram, so a bug
    # there killed plain endogenous runs that never draw from it (7-1 tasks 45/52/54). If
    # this condition ever misses a caller, the sampler's own assert names the missing keys.
    need_logit_hist = ("empirical_logit" in interv_eval_dists or
                       config.training_config.intervention_signal_distribution == "empirical_logit")
    # Same rule for `empirical_prob`'s copy of the train weights: 300K floats per cell
    # that only that sampler reads.
    need_prob_values = ("empirical_prob" in interv_eval_dists or
                        config.training_config.intervention_signal_distribution == "empirical_prob")
    # PAIR MODE (3-1): the ranges come from the injection pool, keyed per absolute
    # (dst, src). The run's own train split cannot supply them: it holds ONE random pair
    # per document (300K rows over up to 8K pairs) and no pair with dst >= n at all,
    # while the length eval injects at every grid pair up to dst = 511.
    pair_mode = is_pair_selector(config.train_token_selector)
    if pair_mode:
        pool_dir = pair_pool_dir(config)
        if not pool_dir.exists():
            raise SystemExit(f"pair mode needs the injection pool at {pool_dir}; build it "
                             f"with `python data_prep/cache_signals.py --config {args.config} --pair-pool`")
        pool = apply_signal_filter(load_from_disk(str(pool_dir)), config.signal_filter, "pool")
        df = pool.to_pandas()
        group_cols = ["signal_layer", "signal_index", "token_position", "src_token_position"]
        print(f"[joint] pair mode: intervention ranges from {pool_dir} "
              f"({len(df)} rows, {df.groupby(group_cols).ngroups} (head, dst, src) cells)")
        pool_keys = set(df.groupby(group_cols).groups)
        # Every pair an eval reads must come from the pool, so that the supplement below
        # can never reach a test metric. Holds by construction: test pairs are
        # pair_grid(n) and pool_pairs() is the union of the grids.
        test_keys = set(test_raw.select_columns(group_cols).to_pandas().groupby(group_cols).groups)
        assert test_keys <= pool_keys, (
            f"{len(test_keys - pool_keys)} test (head, dst, src) cells are not in the pool, "
            f"e.g. {sorted(test_keys - pool_keys)[:3]}")
        if args.joint_loss_weight < 1.0:
            # The pool holds the EVAL grid only (all pairs up to m=32, a log grid above),
            # but at W < 1 every train/dev row is injected at its own random pair. 3-1
            # (W=1) never injected during training, so this never showed; at n=64/128 the
            # first off-grid pair, e.g. (37, 20), raised KeyError in
            # sample_intervention_values. Off-grid cells take their ranges from the run's
            # own train+dev rows (dev too: mid-training eval injects, and a smoke run's
            # dev pairs need not occur in its 2000 train rows). Grid cells keep the pool's.
            own = pd.concat([r.select_columns(group_cols + ["activation"]).to_pandas()
                             for r in (train_raw, dev_raw)], ignore_index=True)
            own_keys = pd.MultiIndex.from_frame(own[group_cols])
            extra = own[~own_keys.isin(list(pool_keys))]
            sizes = extra.groupby(group_cols).size()
            print(f"[joint] pair mode, W={args.joint_loss_weight}: {len(pool_keys)} pool cells "
                  f"+ {len(sizes)} off-grid train/dev cells from the run's own rows "
                  f"(min {int(sizes.min()) if len(sizes) else 0} rows/cell)")
            train_dist = config.training_config.intervention_signal_distribution
            if len(sizes) and int(sizes.min()) < 2 and train_dist != "prob_uniform":
                # one row => the cell has no spread (`std` NaN, `std_logit` 0), so a
                # moment-based sampler would inject NaN or a constant without raising.
                # prob_uniform reads no per-cell statistic.
                raise ValueError(
                    f"intervention_signal_distribution {train_dist!r} reads per-cell "
                    f"statistics, but {int((sizes < 2).sum())} off-grid cells have one row")
            df = pd.concat([df, extra], ignore_index=True)
    signal_ranges = {}
    for key, vals in df.groupby(group_cols)["activation"]:
        stats = {
            "min": float(vals.min()), "max": float(vals.max()),
            "mean": float(vals.mean()), "std": float(vals.std())}
        if prob_signal:
            assert float(vals.min()) >= 0.0 and float(vals.max()) <= 1.0, (
                f"attention_weight cell {key} has activations outside [0,1]: "
                f"[{float(vals.min())}, {float(vals.max())}] -- the cache holds "
                f"pre-softmax logits under a 'P' name (port site mismatch)")
            stats.update(logit_moments(vals.tolist()))
            if need_logit_hist:
                # extra keys for the `empirical_logit` sampler, invisible to the rest
                stats.update(logit_histogram(vals.tolist()))
            if need_prob_values:
                # extra key for the `empirical_prob` sampler, invisible to the rest
                stats.update(empirical_prob_values(vals.tolist()))
        if pair_mode:
            layer, head, d, src = key
            signal_ranges[signal_range_key(layer, head, pair=(d, src))] = stats
        else:
            signal_ranges[signal_range_key(*key)] = stats

    def preprocess(ex):
        out = preprocess_example(ex, tokenizer=tokenizer,
                                 prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
                                 max_length=config.training_config.max_seq_length,
                                 return_frozen_input_len=True, return_position_valid=True,
                                 for_prediction=True)
        out["token_position"] = ex["token_position"]
        out["signal_layer"] = ex["signal_layer"]
        out["signal_index"] = ex["signal_index"]
        out["activation"] = ex["activation"]
        # Attention needs the SOURCE token as well as the destination, and the source
        # offset to identify the signal. Emitted only for attention rows: these columns
        # are collated straight into model kwargs, and a column of Nones would reach
        # the forward as a tensor of nulls.
        if is_attention_type(ex["signal_type"]):
            out["src_token_position"] = ex["src_token_position"]
            out["signal_src_offset"] = ex["src_offset"]
        return out

    train_dataset = train_raw.map(preprocess, remove_columns=train_raw.column_names)
    dev_dataset = dev_raw.map(preprocess, remove_columns=dev_raw.column_names)
    test_dataset_pred = test_raw.map(preprocess, remove_columns=test_raw.column_names)

    # Drop rows whose introspected token is not addressable (or refuse, if there are
    # enough of them to mean something is misconfigured), then drop the flags -- these
    # datasets are collated straight into model kwargs, so they cannot stay.
    max_len = config.training_config.max_seq_length
    test_provenance = eval_provenance(test_raw, test_dataset_pred)
    train_dataset = drop_invalid_positions(train_dataset, "train", max_len)
    dev_dataset = drop_invalid_positions(dev_dataset, "dev", max_len)
    test_dataset_pred = drop_invalid_positions(test_dataset_pred, "test", max_len)

    # ---- OOD caches BEFORE the model loads ----
    # cache_signals runs as a subprocess and needs the GPU itself; once this process has
    # a model on the device, that second CUDA context fails with "CUDA-capable device(s)
    # is/are busy or unavailable". Preparing here keeps the device free for it.
    # Same GPU-ordering constraint as the OOD caches below: cache_signals is a
    # subprocess and needs the device, so this must happen before the model takes it.
    length_prepared, eval_lengths = None, None
    if args.length_eval or args.length_only:
        from evals.score_length_generalization import prepare_length_caches, DEFAULT_EVAL_LENGTHS
        eval_lengths = ([int(x) for x in args.eval_lengths.split(",")]
                        if args.eval_lengths else list(DEFAULT_EVAL_LENGTHS))
        length_prepared = prepare_length_caches(config, eval_lengths=eval_lengths,
                                                n_test=args.eval_n_test)

    ood_prepared, ood_corpora = None, None
    if args.ood_eval or args.ood_only:
        from evals.score_ood_generalization import prepare_ood_caches, DEFAULT_CORPORA
        ood_corpora = ([c.strip() for c in args.ood_corpora.split(",")]
                       if args.ood_corpora else list(DEFAULT_CORPORA))
        ood_prepared = prepare_ood_caches(config, corpora=ood_corpora)

    # ---- model (regressor + gated LoRA) ----
    # The attention implementation MUST match the one cache_signals used, or the live
    # logit and the cached target disagree at the bf16 level and the alignment check
    # cannot reach corr ~0.9999. attention_model_kwargs returns {} for neuron/SAE runs,
    # which therefore keep loading under sdpa exactly as before.
    base = AutoModelForCausalLM.from_pretrained(
        config.model_name, device_map="auto",
        torch_dtype=torch.bfloat16 if config.training_config.bf16 else torch.float32,
        **attention_model_kwargs(config.train_signals))
    resize_embeddings_for_tokenizer(base, tokenizer)
    diag_ckpt = None
    eval_only_flags = (args.diag_only or args.ood_only or args.length_only
                       or args.neuron_gen_only or args.interv_eval_only)
    _ckpts = glob.glob(f"{output_dir}/checkpoint-*")
    _latest = (Path(args.checkpoint) if args.checkpoint
               else (Path(max(_ckpts, key=os.path.getctime)) if _ckpts else None))
    # A run that finished training and then died in a downstream eval leaves a complete
    # checkpoint. Re-entering trainer.train() on it trains nothing but still replays the
    # dataloader (see _checkpoint_is_complete), so load the weights directly instead.
    training_complete = (_latest is not None and not args.overwrite_cache
                         and not eval_only_flags and _checkpoint_is_complete(_latest))
    if training_complete:
        print(f"[joint] {_latest.name} is already at the end of training "
              f"(global_step == max_steps); loading its weights and skipping "
              f"trainer.train(). Pass --overwrite-cache to retrain from scratch.")
    load_from_ckpt = eval_only_flags or training_complete
    if load_from_ckpt:
        # load a specific (possibly mid-run) checkpoint's adapter for eval WITHOUT training
        from peft import PeftModel
        diag_ckpt = _latest
        if diag_ckpt is None:
            raise SystemExit("--diag-only/--ood-only/--length-only/--neuron-gen-only "
                             "need a checkpoint "
                             "(none found; pass --checkpoint).")
        model = PeftModel.from_pretrained(base, str(diag_ckpt), autocast_adapter_dtype=False)
    else:
        peft_config = LoraConfig(
            r=config.training_config.lora_r, lora_alpha=config.training_config.lora_alpha,
            target_modules=config.training_config.lora_target_modules,
            lora_dropout=config.training_config.lora_dropout, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base, peft_config)
    resize_embeddings_for_tokenizer(model, tokenizer)
    model = IntrospectionRegressionModel(
        model, signal_ranges=signal_ranges,
        intervention_distribution=config.training_config.intervention_signal_distribution,
        intervention_extreme_scale=config.training_config.intervention_extreme_scale,
        attention_site=port_site(signal_type) if attention_run else None,
        head_activation=config.training_config.regression_head_activation)
    print(f"[joint] regression head: {type(model.regression_head).__name__} "
          f"(regression_head_activation={config.training_config.regression_head_activation})")
    regression_loss = config.training_config.regression_loss
    if regression_loss not in REGRESSION_LOSSES:
        raise ValueError(f"regression_loss {regression_loss!r} not in {REGRESSION_LOSSES}")
    if regression_loss == "norm_deviance":
        # deviance is defined on probabilities: the head must output one, and the target
        # (cached weight or injected value) must be one
        if config.training_config.regression_head_activation != "sigmoid":
            raise ValueError("regression_loss 'norm_deviance' needs regression_head_activation "
                             f"'sigmoid', got {config.training_config.regression_head_activation!r}")
        if not prob_signal:
            raise ValueError(f"regression_loss 'norm_deviance' is for attention_weight signals, "
                             f"this run is {signal_type!r}")
    print(f"[joint] regression loss: {regression_loss}")
    if "empirical_logit" in interv_eval_dists and not prob_signal:
        raise ValueError(f"empirical_logit injection is for attention_weight signals only, "
                         f"this run is {signal_type!r}")
    if need_prob_values and not prob_signal:
        # Without this an MLP run would reach sample_intervention_values' generic branch
        # and fail there with "Unknown intervention_distribution" mid-training.
        raise ValueError(f"empirical_prob injection is for attention_weight signals only, "
                         f"this run is {signal_type!r}")
    if prob_signal:
        # Attach AFTER construction: the sampler closes over the model so that it reads
        # `intervention_distribution` and `intervention_scale` live on every draw. Freeze
        # them into the closure instead and evaluate_interventional_test's forced
        # "gaussian at scale 1.0" and every row of interventional_scale_sweep silently
        # become the same distribution, with the tables filling in as if they had not.
        model.intervention_fn = make_bounded_prob_intervention_fn(model)
        print(f"[joint] attention_weight run: intervention values are drawn in log-odds "
              f"space and mapped through a sigmoid, so every draw is a valid probability "
              f"(guard band {model.intervention_fn.eps}).")
    if load_from_ckpt:
        load_regression_head(model, str(diag_ckpt))
        # no-op unless the checkpoint's frozen embeddings were stripped to save disk
        load_added_token_embeddings(model, str(diag_ckpt))
    replace_model_with_gated_version(model)
    replace_lora_with_gated_version(model)

    gate_input = config.training_config.disable_adapter_in_input
    collator_cls = LoRAGatingCollator if gate_input else UngatedCollator
    # The KL anchor is a no-op under the gate, and silently so: LoRAGatingCollator leaves
    # the mask as `lora_token_mask`, compute_loss finds no `anchor_region_mask`, and the
    # `region_mask is not None` guard drops the term. It is also VACUOUS there -- the gate
    # zeroes the LoRA delta over the input text, so tuned and base give the same
    # next-token distribution and KL is 0 by construction, not by optimization. A run like
    # that lands in a directory named _kl<budget>_ and reads, later, as a budgeted run.
    anchor_requested = (args.kl_anchor_target is not None or args.kl_anchor_weight > 0)
    if anchor_requested and gate_input:
        msg = ("KL anchor requested (--kl-anchor-target/--kl-anchor-weight) but the input "
               "gate is ON, so the anchor cannot apply: the gate already pins text "
               "behaviour to the base model, making KL 0 by construction. The run would "
               "train with NO KL term while its name still claims a budget.")
        if args.require_kl_anchor:
            raise SystemExit(f"[joint] {msg} Refusing (--require-kl-anchor).")
        print(f"[joint] *** WARNING *** {msg} Continuing unanchored.")
    print(f"[joint] LoRA input gate: "
          f"{'ON -- adapter gated to the introspection prompt' if gate_input else 'OFF -- adapter active over the input text too'} "
          f"(training_config.disable_adapter_in_input={gate_input}); lora_dropout="
          f"{config.training_config.lora_dropout}")
    pred_collator = collator_cls(tokenizer=tokenizer, model=model, padding=True,
                                 label_pad_token_id=-100, return_tensors="pt", for_prediction=True)

    if args.interv_eval_only:
        print(f"[joint] --interv-eval-only: {diag_ckpt} -> {output_dir} | {interv_eval_dists}")
        run_interventional_evals(model, test_dataset_pred, pred_collator, config, output_dir,
                                 interv_eval_dists, n_pca=args.routing_probe_pca,
                                 source_checkpoint=diag_ckpt, provenance=test_provenance)
        return

    if args.neuron_gen_only:
        # No cache prep needed: both arms read this run's own test split, unlike the
        # OOD and length evals which each build a new one.
        from evals.score_neuron_generalization import evaluate_neuron_generalization
        print(f"[joint] --neuron-gen-only: {diag_ckpt} -> {output_dir}")
        evaluate_neuron_generalization(model, config, tokenizer, pred_collator,
                                       preprocess_fn=preprocess, output_dir=output_dir,
                                       max_unseen=args.max_unseen_neurons,
                                       pool_config=args.neuron_gen_pool_config)
        return

    if args.length_only:
        # same shape as --ood-only: checkpoint loaded, collator built, Trainer skipped.
        # No preprocess_fn is passed -- it is rebuilt per eval length, because this run's
        # closes over a max_seq_length sized for the TRAINING length and would truncate
        # every longer eval document.
        from evals.score_length_generalization import evaluate_length_grid
        print(f"[joint] --length-only: {diag_ckpt} -> {output_dir} | lengths={eval_lengths}")
        evaluate_length_grid(model, config, tokenizer, pred_collator,
                             output_dir=output_dir, eval_lengths=eval_lengths,
                             n_test=args.eval_n_test, prepared=length_prepared,
                             interv_dists=interv_eval_dists)
        return

    if args.ood_only:
        # everything needed for the OOD pass exists by now (checkpoint loaded, collator
        # built); skip the Trainer entirely and write into the RUN dir so the results
        # replace the failed ones and the aggregator finds them
        from evals.score_ood_generalization import evaluate_ood
        print(f"[joint] --ood-only: {diag_ckpt} -> {output_dir} | corpora={ood_corpora}")
        evaluate_ood(model, config, tokenizer, pred_collator,
                     preprocess_fn=preprocess, output_dir=output_dir,
                     corpora=ood_corpora, prepared=ood_prepared)
        return

    # checkpoint frequency: 0 => per epoch; >0 => every save_steps (float <1 = ratio of total steps)
    save_kwargs = ({"save_strategy": "epoch"} if config.training_config.save_steps <= 0
                   else {"save_strategy": "steps", "save_steps": config.training_config.save_steps})
    epoch_kwargs = {"num_train_epochs": config.training_config.num_epochs}
    if float(config.training_config.num_epochs) < 1:
        # A fractional epoch never reaches an epoch boundary, so save_strategy="epoch"
        # would end training with NO checkpoint. Pin the length in steps (HF gives
        # max_steps precedence) and save exactly at that step, so the run ends with a
        # checkpoint-<max_steps> that _checkpoint_is_complete recognises as finished.
        steps_per_epoch = _opt_steps_per_epoch(len(train_dataset), config)
        max_steps = max(1, math.ceil(steps_per_epoch * float(config.training_config.num_epochs)))
        epoch_kwargs = {"max_steps": max_steps}
        save_kwargs = {"save_strategy": "steps", "save_steps": max_steps}
        print(f"[joint] fractional epochs ({config.training_config.num_epochs}): "
              f"max_steps={max_steps}, checkpoint saved at that step")
    training_args = TrainingArguments(
        output_dir=str(output_dir), run_name=config.experiment_name,
        learning_rate=config.training_config.learning_rate,
        warmup_ratio=config.training_config.warmup_ratio,
        lr_scheduler_type=config.training_config.lr_scheduler_type,
        per_device_train_batch_size=config.training_config.train_batch_size,
        gradient_accumulation_steps=config.training_config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.training_config.eval_batch_size,
        logging_steps=max(1, math.ceil(100 / _eff_batch(config))),
        eval_strategy="steps", eval_steps=_eval_steps(len(train_dataset), config, args.evals_per_run),
        bf16=config.training_config.bf16,
        remove_unused_columns=False, report_to="wandb", **save_kwargs, **epoch_kwargs)

    # PEFT stores the whole resized embed_tokens/lm_head (~1.0 GB, 98% of the file)
    # because the vocab grew by one; drop them as each checkpoint is written, keeping
    # the added rows so the checkpoint stays reloadable.
    callbacks = [RegressionHeadSaveCallback(model),
                 StripEmbeddingSaveCallback(n_added_tokens)]
    if config.training_config.intervention_anneal:
        # NOTE: this anneals the injection SCALE (magnitude), NOT the mix weight.
        callbacks.append(InterventionAnnealCallback(
            model,
            scale_start=config.training_config.intervention_scale_start,
            scale_end=config.training_config.intervention_scale_end,
            schedule=config.training_config.intervention_anneal_schedule,
            start_frac=config.training_config.intervention_anneal_start_frac,
            end_frac=config.training_config.intervention_anneal_end_frac,
            log_every=max(1, math.ceil(400 / _eff_batch(config)))))

    trainer = JointInterventionTrainer(
        model=model, tokenizer=tokenizer, signal_ranges=signal_ranges, args=training_args,
        train_dataset=train_dataset, eval_dataset=dev_dataset, data_collator=pred_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        preprocess_logits_for_metrics=None, experiment_config=config,
        # at w=1 there is no interventional term to train, so mid-training eval
        # reports the endogenous objective; the interventional TEST still runs
        # afterwards, where it measures context bypass rather than training progress
        do_intervention=(args.joint_loss_weight < 1.0),
        callbacks=callbacks, joint_loss_weight=args.joint_loss_weight,
        regression_loss=config.training_config.regression_loss,
        kl_anchor_target=args.kl_anchor_target, kl_anchor_weight=args.kl_anchor_weight,
        kl_anchor_positions=args.kl_anchor_positions,
        kl_anchor_coef_init=args.kl_anchor_coef_init,
        kl_anchor_coef_range=tuple(float(x) for x in args.kl_anchor_coef_range.split(",")))
    print(f"[joint] joint_loss_weight (endo)={args.joint_loss_weight}  "
          f"interv_distribution={config.training_config.intervention_signal_distribution} "
          f"scale={model.intervention_scale}")

    ckpt_dirs = glob.glob(f"{output_dir}/checkpoint-*")
    eval_out_dir = output_dir
    if args.diag_only:
        # weights were already loaded from diag_ckpt via PeftModel.from_pretrained; no training
        eval_out_dir = output_dir / f"diag_{diag_ckpt.name}"
        eval_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[joint] diag-only: loaded {diag_ckpt}; writing diagnostics to {eval_out_dir}")
    elif ckpt_dirs and not args.overwrite_cache:
        latest = str(args.checkpoint) if args.checkpoint else max(ckpt_dirs, key=os.path.getctime)
        if training_complete:
            # adapter weights already loaded above via PeftModel.from_pretrained
            print(f"[joint] using completed checkpoint {latest} as-is; going straight to evals")
        else:
            print(f"Found PARTIAL checkpoint, resuming training from {latest}")
            train_results = trainer.train(resume_from_checkpoint=latest)
        load_regression_head(model, latest)
        load_added_token_embeddings(model, latest)
    else:
        train_results = trainer.train()
        trainer.save_metrics("train", train_results.metrics)

    # ---- frontier metrics: interventional (gaussian@1 by default) AND endogenous ----
    run_interventional_evals(model, test_dataset_pred, pred_collator, config, eval_out_dir,
                             interv_eval_dists, n_pca=args.routing_probe_pca,
                             provenance=test_provenance)
    # is the cached endogenous target still what the model computes? (auto-runs
    # whenever the adapter is active over the input text, where it can go stale)
    if not args.skip_drift_eval and (not gate_input or args.force_drift_eval):
        # The MLP version reads the neuron through a forward hook on act_fn and also
        # measures up_proj/gate_proj transmission; an attention logit is not a module
        # output and has no transmission analogue, so that family gets its own
        # implementation of the same two measurements, writing the same filenames.
        if attention_run:
            # the site must match the cache, or this compares a cached probability
            # against a live logit and reports it as a position error
            measure_attention_alignment(model, test_dataset_pred, pred_collator, config,
                                        eval_out_dir, signal_type=signal_type)
        else:
            measure_activation_drift(model, test_dataset_pred, pred_collator, config,
                                     eval_out_dir)

    eval_endogenous_test(model, test_dataset_pred, pred_collator, config, eval_out_dir,
                         provenance=test_provenance)

    # diagnostic: is the injected value routed to the readout (ceiling) vs read by
    # the trained head, stratified by token position -> tells us whether earlier-
    # position failure is a routing (capacity) or reading (optimization) problem.
    routing_probe_by_position(model, test_dataset_pred, pred_collator, config, eval_out_dir,
                              scale=args.routing_probe_scale, n_pca=args.routing_probe_pca)

    # diagnostic: interventional corr vs injection scale -> reveals the numerical/
    # noise floor (corr rising with scale => natural scale is below the floor).
    # --scale-sweep "" switches it off (3-1 drops it: one scale per pair-conditional
    # pool would mean little, and it costs a full pass over the test grid per scale).
    scales = [float(x) for x in args.scale_sweep.split(",") if x.strip()]
    if scales:
        interventional_scale_sweep(model, test_dataset_pred, pred_collator, config,
                                   eval_out_dir, scales=scales, n_pca=args.routing_probe_pca)
    else:
        print("[joint] --scale-sweep is empty: interventional scale sweep skipped")

    if args.neuron_gen_eval:
        from evals.score_neuron_generalization import evaluate_neuron_generalization
        evaluate_neuron_generalization(model, config, tokenizer, pred_collator,
                                       preprocess_fn=preprocess, output_dir=eval_out_dir,
                                       max_unseen=args.max_unseen_neurons,
                                       pool_config=args.neuron_gen_pool_config)

    if args.length_eval:
        from evals.score_length_generalization import evaluate_length_grid
        evaluate_length_grid(model, config, tokenizer, pred_collator,
                             output_dir=eval_out_dir, eval_lengths=eval_lengths,
                             n_test=args.eval_n_test, prepared=length_prepared,
                             interv_dists=interv_eval_dists)

    if args.ood_eval:
        # lazy import: score_ood_generalization imports this module's eval helpers
        from evals.score_ood_generalization import evaluate_ood
        evaluate_ood(model, config, tokenizer, pred_collator,
                     preprocess_fn=preprocess, output_dir=eval_out_dir,
                     corpora=ood_corpora, prepared=ood_prepared)

    # fp32 overlay: if natural-scale corr lifts toward the large-scale plateau when
    # cast to fp32, the natural-scale failure was a bf16 numerical floor.
    if args.diag_fp32:
        print("[joint] casting model to fp32 and re-running diagnostics (_fp32) ...")
        model.float()
        if scales:
            interventional_scale_sweep(model, test_dataset_pred, pred_collator, config,
                                       eval_out_dir, scales=scales,
                                       n_pca=args.routing_probe_pca, suffix="_fp32")
        routing_probe_by_position(model, test_dataset_pred, pred_collator, config, eval_out_dir,
                                  scale=args.routing_probe_scale, n_pca=args.routing_probe_pca, suffix="_fp32")


if __name__ == "__main__":
    main()
