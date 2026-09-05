#!/usr/bin/env python3
"""
Interventional (causal) training for SAE feature DETECTION, generative readout.

Every forward we steer the SAE feature to a sampled on/off state at the target
token and set the LABEL to that steered state ("active"/"inactive"). Because the
on/off draw is random and independent of the text, context can no longer predict
the label -- the model is forced to actually READ the steered feature rather than
ride a context correlation (the bypass we saw endogenously).

Design (matches the choices made for the SAE track):
  * Generative words readout (same as train_sae_detector.py), so endogenous and
    interventional runs are directly comparable.
  * On/off sampled 50/50 per example in the collator; "on" magnitude drawn from a
    distribution with a mutable `scale` (empirical | uniform | extreme | gaussian),
    so we can start on the empirical active scale and later widen / anneal exactly
    like the neuron intervention path (reuses TrainingConfig.intervention_*).
  * Steering is the difference-of-decode clamp (SAEInterventionHook); the LoRA
    input gate stays on, so the model must fetch the steered feature via attention.

After training we run BOTH the endogenous eval (does reading now transfer to
natural data?) and the paired interventional eval, reusing train_sae_detector.
"""
from __future__ import annotations

import os
import glob
import json
import math
import argparse
import operator
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback

from src.introspector.config import ExperimentConfig
from src.introspector.models.signal_paths import resolve_sae_hook_path
from src.introspector.signals.sae import load_sae_for_spec
from src.introspector.training.collator import LoRAGatingCollator
from src.introspector.training.model import (
    StripEmbeddingSaveCallback,
    replace_model_with_gated_version, replace_lora_with_gated_version,
    SAEInterventionHook, anneal_value,
)
from src.introspector.prompts import PROMPT_REGISTRY
from train_introspector import preprocess_example, INTROSPECTION_SEPARATOR
from src.introspector.dataset_utils import apply_signal_filter
from train_sae_detector import (
    label_token_ids, auroc_eval, interventional_detection_eval, active_on_value,
    resolve_active_sae_spec, measure_sae_alignment,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--force-drift-eval", action="store_true",
                   help="Run the cache-alignment check before training (see "
                        "measure_sae_alignment). Regression test for the off-by-one/BOS "
                        "class of bug; a misaligned position means we steer a different "
                        "token than the prompt names. Want best offset +0, corr ~1.0.")
    p.add_argument("--ignore-misalignment", action="store_true",
                   help="Train even if the alignment check fails (diagnostics only).")
    return p.parse_args()


# ============================================================
# On/off + magnitude sampler (mutable scale for anneal)
# ============================================================
class OnOffSampler:
    """Draws the per-example intervention: 50/50 off(=0)/on, and the 'on' magnitude
    from `distribution` scaled by a mutable `scale`.

    Distributions (naming kept consistent with the neuron intervention path, so the
    same TrainingConfig.intervention_* knobs and scripts apply; note that 'extreme'
    at scale=1 is literally uniform over the empirical active range, which is why
    'uniform' would be the better name at publication-refactor time):
      * 'extreme'   : uniform over [amin, center + scale*(amax-amin)/2]; scale=1 ->
                      exactly the empirical active range [amin, amax], scale>1 ->
                      progressively larger-than-natural injections.
      * 'empirical' : resample an actual observed active value (x scale).
      * 'gaussian'  : N(active_mean, scale*active_std).

    IMPORTANT (differs from the neuron sampler): SAE feature activations are
    NON-NEGATIVE and an 'on' draw must stay strictly active, otherwise a value that
    lands <=0 would be labeled "active" while actually steering the feature off --
    silently corrupting the training labels at large scales. The neuron version can
    sample symmetrically around the midpoint because neuron acts are signed; here we
    widen UPWARD from `amin` (~the JumpReLU threshold) and floor every on-draw at it.
    """

    def __init__(self, active_values, distribution="empirical", scale=1.0,
                 on_prob=0.5, extreme_scale=1000.0, seed=0):
        av = np.asarray([v for v in active_values if v > 0], dtype=float)
        if len(av) == 0:
            raise SystemExit("no active (>0) activations in train cache; cannot sample 'on' values")
        self.av = av
        self.amin, self.amax = float(av.min()), float(av.max())
        self.amean, self.astd = float(av.mean()), float(av.std())
        self.distribution = distribution
        self.on_prob = on_prob
        self.extreme_scale = extreme_scale
        # start scale: extreme uses extreme_scale, else 1.0 (endogenous scale)
        self.scale = extreme_scale if distribution == "extreme" else scale
        self.rng = np.random.RandomState(seed)

    def set_scale(self, s):
        self.scale = float(s)

    def _on_value(self):
        d, s = self.distribution, self.scale
        if d == "extreme":
            # widen upward only (see class docstring); scale=1 -> [amin, amax]
            hi = (self.amin + self.amax) / 2 + s * (self.amax - self.amin) / 2
            v = self.rng.uniform(self.amin, max(hi, self.amin))
        elif d == "empirical":
            v = float(self.rng.choice(self.av)) * s
        elif d == "gaussian":
            v = self.rng.normal(self.amean, s * self.astd)
        else:
            raise ValueError(f"unknown distribution {d!r} (use extreme|empirical|gaussian)")
        # floor at the smallest observed active value so an "on" label is always
        # genuinely on (>= the JumpReLU threshold), at any scale
        return float(max(self.amin, v))

    def draw(self):
        """Returns (is_on: bool, value: float)."""
        if self.rng.random() < self.on_prob:
            return True, self._on_value()
        return False, 0.0


# ============================================================
# Collator: builds the training sequence with the SAMPLED answer word + steer info
# ============================================================
class InterventionDetectionCollator:
    """Consumes prompt-only rows (for_prediction preprocess) carrying
    frozen_input_len + token_position, samples on/off per example, appends the
    matching answer word, and emits input_ids/labels/attention_mask/lora_token_mask
    plus token_position + steer_value for the hook. Right-padded (training)."""

    def __init__(self, tokenizer, active_word, inactive_word, sampler, gate_input=True):
        self.tok = tokenizer
        self.gate_input = gate_input
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        # answer tokens AS EMITTED (leading space -> matches the readout token ids)
        self.ans_on = tokenizer(" " + active_word.strip(), add_special_tokens=False)["input_ids"]
        self.ans_off = tokenizer(" " + inactive_word.strip(), add_special_tokens=False)["input_ids"]
        self.sampler = sampler

    def __call__(self, features):
        ids_l, lab_l, mask_l, pos_l, steer_l = [], [], [], [], []
        for f in features:
            prompt = list(f["input_ids"])
            fin = int(f["frozen_input_len"])
            is_on, val = self.sampler.draw()
            ans = (self.ans_on if is_on else self.ans_off) + [self.eos_id]
            ids = prompt + ans
            lab = [-100] * len(prompt) + ans                 # supervise only the answer
            # LoRA gate: 0 on the frozen input text, 1 on introspection prompt + answer
            mask = [0] * fin + [1] * (len(ids) - fin)
            ids_l.append(ids); lab_l.append(lab); mask_l.append(mask)
            pos_l.append(int(f["token_position"])); steer_l.append(val)

        maxlen = max(len(x) for x in ids_l)
        def pad(seq, fill): return seq + [fill] * (maxlen - len(seq))
        input_ids = torch.tensor([pad(x, self.pad_id) for x in ids_l], dtype=torch.long)
        labels = torch.tensor([pad(x, -100) for x in lab_l], dtype=torch.long)
        attn = torch.tensor([[1] * len(x) + [0] * (maxlen - len(x)) for x in ids_l], dtype=torch.long)
        batch = {
            "input_ids": input_ids, "attention_mask": attn, "labels": labels,
            "token_position": torch.tensor(pos_l, dtype=torch.long),
            "steer_value": torch.tensor(steer_l, dtype=torch.float32),
        }
        # gate OFF => omit the mask entirely, so replace_model_with_gated_version sets
        # every LoRA layer's mask to None and GatedLinear reduces to stock LoRA over the
        # whole sequence (same contract as UngatedCollator).
        if self.gate_input:
            batch["lora_token_mask"] = torch.tensor([pad(x, 0) for x in mask_l],
                                                    dtype=torch.float32)
        return batch


# ============================================================
# Trainer: register the steering hook around each forward
# ============================================================
class InterventionDetectionTrainer(Trainer):
    def __init__(self, *args, bundle=None, feature_index=None, module_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bundle = bundle
        self.feature_index = feature_index
        self.module = operator.attrgetter(module_path)(self.model)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        token_position = inputs.pop("token_position")
        steer_value = inputs.pop("steer_value")
        # right-padded training: the text starts at index 0, so token_position is
        # already the absolute position (no left-pad offset).
        hook = SAEInterventionHook(self.bundle, token_position, self.feature_index, steer_value)
        handle = self.module.register_forward_hook(hook)
        try:
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        labels=inputs["labels"], lora_token_mask=inputs.get("lora_token_mask"))
        finally:
            handle.remove()
        return (out.loss, out) if return_outputs else out.loss


class SamplerAnnealCallback(TrainerCallback):
    """Anneal the sampler's magnitude scale scale_start -> scale_end over training
    (only if intervention_anneal is set), mirroring the neuron InterventionAnneal."""
    def __init__(self, sampler, tc):
        self.sampler = sampler
        self.tc = tc

    def on_step_begin(self, args, state, control, **kwargs):
        if not self.tc.intervention_anneal:
            return
        s = anneal_value(state.global_step, state.max_steps,
                         self.tc.intervention_scale_start, self.tc.intervention_scale_end,
                         self.tc.intervention_anneal_start_frac, self.tc.intervention_anneal_end_frac,
                         self.tc.intervention_anneal_schedule)
        self.sampler.set_scale(s)


# ============================================================
# main
# ============================================================
def make_pred_preprocess(config, tokenizer):
    tc = config.training_config
    def pp(ex):
        out = preprocess_example(
            ex, tokenizer=tokenizer, prompt_fn=PROMPT_REGISTRY[tc.prompt_type],
            max_length=tc.max_seq_length, return_frozen_input_len=True, for_prediction=True,
        )
        out["token_position"] = ex["token_position"]
        return out
    return pp


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(args.config.read_text())
    tc = config.training_config
    assert tc.objective == "detection", "interventional detector expects objective='detection'"

    cache_dir = config.dataset_cache_dir
    out_dir = Path(config.output_dir) / config.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "sae_intervention_train_metrics.json").exists() and not args.overwrite_cache:
        print(f"Already ran, skipping {out_dir}")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_added_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": [INTROSPECTION_SEPARATOR]})

    train_raw = load_from_disk(str(cache_dir / "train"))
    dev_raw = load_from_disk(str(cache_dir / "dev"))
    test_raw = load_from_disk(str(cache_dir / "test"))
    # a (layer, position) cell is cached once with ALL its features; keep only this
    # run's rows (no-op when signal_filter is unset)
    train_raw = apply_signal_filter(train_raw, config.signal_filter, "train")
    dev_raw = apply_signal_filter(dev_raw, config.signal_filter, "dev")
    test_raw = apply_signal_filter(test_raw, config.signal_filter, "test")

    # must honour signal_filter: steering the wrong feature would silently invalidate
    # every interventional number (see resolve_active_sae_spec)
    spec = resolve_active_sae_spec(config)

    pp = make_pred_preprocess(config, tokenizer)
    train_ds = train_raw.map(pp, remove_columns=train_raw.column_names)
    dev_ds = dev_raw.map(pp, remove_columns=dev_raw.column_names)

    # model + LoRA + gate
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16 if tc.bf16 else torch.float32, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))
    peft_config = LoraConfig(r=tc.lora_r, lora_alpha=tc.lora_alpha,
                             target_modules=tc.lora_target_modules, lora_dropout=tc.lora_dropout,
                             bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    model.resize_token_embeddings(len(tokenizer))
    replace_model_with_gated_version(model)
    replace_lora_with_gated_version(model)

    bundle = load_sae_for_spec(spec, device=str(model.device))
    module_path = resolve_sae_hook_path(model, spec.layer, site=spec.sae_site, path_to_output=False)

    sampler = OnOffSampler(
        active_values=train_raw["activation"],
        distribution=tc.intervention_signal_distribution,
        extreme_scale=tc.intervention_extreme_scale, seed=config.seed,
    )
    gate_input = tc.disable_adapter_in_input
    print(f"[interv] LoRA input gate: "
          f"{'ON -- adapter gated to the introspection prompt' if gate_input else 'OFF -- adapter active over the input text too'} "
          f"(disable_adapter_in_input={gate_input}); lora_dropout={tc.lora_dropout}")
    collator = InterventionDetectionCollator(
        tokenizer, tc.detection_active_word, tc.detection_inactive_word, sampler,
        gate_input=gate_input)

    print(f"[interv] feature={spec.layer}.{spec.index} dist={sampler.distribution} "
          f"scale0={sampler.scale:g} active[min/mean/max]="
          f"{sampler.amin:.3g}/{sampler.amean:.3g}/{sampler.amax:.3g}")

    # ---- cache-alignment regression check BEFORE training: a misaligned position
    #      means we steer a different token than the prompt names, which silently
    #      invalidates the whole run, so fail fast rather than after hours.
    from train_introspector_joint import UngatedCollator
    pred_collator_cls = LoRAGatingCollator if gate_input else UngatedCollator
    pred_collator = pred_collator_cls(tokenizer=tokenizer, model=model, padding=True,
                                      label_pad_token_id=-100, return_tensors="pt",
                                      for_prediction=True)

    # for_prediction datasets (prompt only) for the endogenous / interventional evals.
    # NOTE: no token_position column -- these go through LoRAGatingCollator, which only
    # knows input_ids/attention_mask/labels/frozen_input_len; the interventional eval
    # reads positions from test_raw directly.
    def endog_pp(ex):
        return preprocess_example(ex, tokenizer=tokenizer, prompt_fn=PROMPT_REGISTRY[tc.prompt_type],
                                  max_length=tc.max_seq_length, return_frozen_input_len=True,
                                  for_prediction=True, detection=True,
                                  detection_active_word=tc.detection_active_word,
                                  detection_inactive_word=tc.detection_inactive_word)
    dev_pred = dev_raw.map(endog_pp, remove_columns=dev_raw.column_names)
    test_pred = test_raw.map(endog_pp, remove_columns=test_raw.column_names)

    if args.force_drift_eval:
        align = measure_sae_alignment(model, spec, test_raw, test_pred, pred_collator,
                                      tokenizer, tc.eval_batch_size, out_dir)
        if not align["aligned"] and not args.ignore_misalignment:
            raise SystemExit("cache alignment FAILED (see above); pass --ignore-misalignment "
                             "to train anyway.")

    training_args = TrainingArguments(
        output_dir=str(out_dir), run_name=config.experiment_name,
        learning_rate=tc.learning_rate,
        per_device_train_batch_size=tc.train_batch_size,
        per_device_eval_batch_size=tc.eval_batch_size,
        num_train_epochs=tc.num_epochs,
        logging_steps=math.ceil(100 / tc.train_batch_size),
        eval_strategy="steps", eval_steps=math.ceil(400 // tc.train_batch_size),
        save_strategy="epoch", bf16=tc.bf16, remove_unused_columns=False, report_to="wandb",
    )
    trainer = InterventionDetectionTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=collator, bundle=bundle, feature_index=spec.index, module_path=module_path,
        callbacks=[SamplerAnnealCallback(sampler, tc),
                   StripEmbeddingSaveCallback(n_added_tokens)],
    )
    ckpts = glob.glob(f"{out_dir}/checkpoint-*")
    if ckpts and not args.overwrite_cache:
        trainer.train(resume_from_checkpoint=max(ckpts, key=os.path.getctime))
    else:
        trainer.train()

    # ---- post-hoc evals reuse train_sae_detector on natural + steered data
    #      (pred_collator / dev_pred / test_pred built before training, above) ----
    dev_endo, _, _ = auroc_eval(model, dev_pred, dev_raw["activation"], pred_collator, tokenizer, tc, tc.eval_batch_size)
    test_endo, _, _ = auroc_eval(model, test_pred, test_raw["activation"], pred_collator, tokenizer, tc, tc.eval_batch_size)
    on_value = active_on_value(train_raw)
    test_interv, _, _ = interventional_detection_eval(
        model, spec, test_raw, test_pred, pred_collator, tokenizer, tc, on_value, tc.eval_batch_size)

    metrics = {"dev_endogenous": dev_endo, "test_endogenous": test_endo,
               "test_interventional": test_interv,
               "train_distribution": sampler.distribution}
    print(json.dumps(metrics, indent=2))
    with open(out_dir / "sae_intervention_train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
