#!/usr/bin/env python3
"""
Joint SAE-feature detection trainer (gate OFF, LM-head readout).

Question: can gemma-2-2b be trained to REPORT whether one of its own Gemma Scope SAE
features is active at a named token, and is the report FAITHFUL (it causally reads the
feature) or a CONTEXT BYPASS (it guesses from the surrounding text)?

The model answers by emitting one of two words in its own tokens -- "active" /
"inactive". At evaluation the prediction is the token it actually GENERATES at the
readout position (argmax over the full vocabulary), scored as active / inactive /
neither. One greedy forward pass; no sampling.

Per optimizer step we take a weighted sum of two detection losses (two forwards):

    loss = w * endogenous_CE + (1 - w) * interventional_CE  +  kl_coef * KL_anchor

  * endogenous:      no steering; target = the feature's NATURAL firing at the token.
  * interventional:  steer the feature on/off 50/50, DECORRELATED from the text, at
                     the named token; target = the forced state. Context cannot predict
                     it, so a detector that only rides context scores chance here.
                     Skipped at w>=1 (pure endogenous), so this one file spans the
                     whole frontier and w=1 is directly comparable to a joint run.
  * KL anchor:       the gate is OFF, so the LoRA adapter runs over the input text and
                     nothing else preserves language modelling. An adaptive KL-to-base
                     penalty on the TEXT tokens holds next-token behaviour near the
                     frozen base model's. (The introspection prompt is left free -- it
                     is where the readout lives and is meant to change.)

Metrics (post-hoc, natural test distribution): balanced accuracy + precision on the
model's GENERATED token (argmax over the vocabulary), for BOTH the endogenous readout
and the interventional test -- those two metrics only. Faithfulness is read from the
interventional balanced accuracy (its labels are decorrelated from context, so ~1.0 =
causal/faithful, ~0.5 = context bypass).

Run:
  python train_sae_detector_joint.py --config <cfg> --joint-loss-weight 0.75 \
      --kl-anchor-target 0.1 --force-drift-eval
"""
from __future__ import annotations
import os, glob, json, math, operator, argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed,
)

from src.introspector.config import ExperimentConfig
from src.introspector.dataset_utils import (
    apply_signal_filter, parse_signal_filter, parse_signal_filters,
)
from src.introspector.models.signal_paths import resolve_sae_hook_path
from src.introspector.signals.sae import load_sae_for_spec
from src.introspector.training.model import (
    StripEmbeddingSaveCallback, replace_model_with_gated_version,
    replace_lora_with_gated_version, SAEInterventionHook, load_added_token_embeddings,
)
from src.introspector.prompts import get_prompt_fn
from src.introspector.sae_featgen import balance_rows_per_feature, ensure_subsample_u
from train_introspector import preprocess_example, INTROSPECTION_SEPARATOR


def is_active(activation) -> bool:
    return float(activation) > 0.0


def resolve_spec(config):
    """The ONE SAE feature this run trains on. A (layer, position) cell is cached once
    with all its features; `--signal-filter S<layer>.<index>` selects one. Resolving
    through the filter (not train_signals[0]) avoids training on one feature while
    steering another."""
    specs = [s for s in config.train_signals if getattr(s, "type", None) == "sae"]
    if not specs:
        raise SystemExit("config.train_signals contains no SAE feature")
    if config.signal_filter:
        want = parse_signal_filter(config.signal_filter)
        match = [s for s in specs if s.layer == want["signal_layer"]
                 and s.index == want["signal_index"]]
        if not match:
            raise SystemExit(f"--signal-filter {config.signal_filter!r} names no cached SAE feature")
        return match[0]
    if len(specs) > 1:
        raise SystemExit("multiple SAE features cached but no --signal-filter; ambiguous.")
    return specs[0]


def resolve_specs(config):
    """MULTI-feature counterpart of resolve_spec: every SAE feature this run trains on,
    in config order (= pool order, which the row budget and the eval follow). Uses the
    signal filter's list when given, else every cached SAE feature. All must come from
    ONE SAE: the steering hook sits on one layer and encodes with one bundle."""
    specs = [s for s in config.train_signals if getattr(s, "type", None) == "sae"]
    if not specs:
        raise SystemExit("config.train_signals contains no SAE feature")
    if config.signal_filter:
        by_key = {(s.layer, s.index): s for s in specs}
        out = []
        for w in parse_signal_filters(config.signal_filter):
            key = (w["signal_layer"], w["signal_index"])
            if key not in by_key:
                raise SystemExit(f"--signal-filter names S{key[0]}.{key[1]}, which is not cached")
            out.append(by_key[key])
        specs = out
    sae_ids = {(s.sae_release, s.sae_site, s.sae_width, s.layer) for s in specs}
    if len(sae_ids) > 1:
        raise SystemExit(f"multi-feature run spans {len(sae_ids)} SAEs {sorted(sae_ids)}; "
                         f"one hook site and one encoder are assumed")
    if len({s.index for s in specs}) != len(specs):
        raise SystemExit("duplicate feature in the trained set")
    return specs


def oversample_positives(ds, target_pos_frac, seed):
    """Duplicate active rows so positives are ~target_pos_frac of TRAIN (the natural
    ~1% base rate would otherwise swamp the endogenous term). Negatives all kept."""
    from datasets import concatenate_datasets
    acts = np.array([float(a) for a in ds["activation"]])
    pos, neg = np.where(acts > 0)[0], np.where(acts <= 0)[0]
    if len(pos) == 0 or len(neg) == 0 or not (0 < target_pos_frac < 1):
        return ds
    desired = int(round(target_pos_frac * len(neg) / (1 - target_pos_frac)))
    reps = max(1, round(desired / len(pos)))
    out = concatenate_datasets([ds] + [ds.select(pos.tolist())] * (reps - 1)).shuffle(seed=seed)
    print(f"[balance] pos {len(pos)}->x{reps}, neg {len(neg)} (n={len(out)})")
    return out


def label_token_ids(tokenizer, tc):
    """First-token ids of the emitted words (leading space: ' active' != 'active')."""
    aid = tokenizer(" " + tc.detection_active_word.strip(), add_special_tokens=False)["input_ids"][0]
    iid = tokenizer(" " + tc.detection_inactive_word.strip(), add_special_tokens=False)["input_ids"][0]
    if aid == iid:
        raise ValueError(f"active/inactive share first token id {aid}; pick distinct words.")
    return aid, iid


def active_on_value(train_raw):
    """Representative 'on' magnitude to steer to at eval: median naturally-active value."""
    acts = np.array([float(a) for a in train_raw["activation"]])
    active = acts[acts > 0]
    return float(np.median(active)) if len(active) else 1.0


def generation_metrics(pred_ids, labels, active_id, inactive_id):
    """Generation-based detection metrics. The prediction is the token the model
    actually GENERATES at the readout position -- argmax over the FULL vocabulary --
    which is `active`, `inactive`, or (if the argmax is any other token) NEITHER,
    counted as a miss. labels: 1 for active.

    Reports balanced accuracy and precision computed on that generated token. Balanced
    accuracy is the mean of the two per-class recalls, so a `neither` on a true-active
    lowers the active recall and on a true-inactive lowers the inactive recall;
    precision (positive class = active) only counts rows the model called `active`."""
    pred_ids = np.asarray(pred_ids)
    labels = np.asarray(labels, int)
    pred_active = pred_ids == active_id
    pred_inactive = pred_ids == inactive_id
    out = {"n": int(len(labels)),
           "base_rate_positive": float(labels.mean()) if len(labels) else float("nan")}
    pos, neg = labels == 1, labels == 0
    if len(labels) == 0 or not pos.any() or not neg.any():
        out.update(balanced_accuracy=float("nan"), precision=float("nan"),
                   note="single-class eval set")
        return out
    tpr = float(pred_active[pos].mean())
    tnr = float(pred_inactive[neg].mean())
    out["balanced_accuracy"] = 0.5 * (tpr + tnr)
    denom = int(pred_active.sum())
    # undefined (not 0) when the model never says `active`; NaN keeps it out of means
    out["precision"] = float((pred_active & pos).sum() / denom) if denom else float("nan")
    # 2x3 confusion counts (true active/inactive x predicted active/inactive/neither),
    # keyed like _scratch_equiv/endo_perclass.json so analysis/plot_sae_confusion.py reads both.
    pred_neither = ~(pred_active | pred_inactive)
    out.update(tpr=tpr, tnr=tnr)
    for tname, tmask in (("act", pos), ("ina", neg)):
        for pname, pmask in (("active", pred_active), ("inactive", pred_inactive),
                             ("neither", pred_neither)):
            out[f"{tname}_pred_{pname}"] = int((tmask & pmask).sum())
    return out


class OnOffSampler:
    """Per-example intervention draw: 50/50 off(=0)/on, 'on' magnitude from the
    feature's empirical active range. SAE activations are NON-NEGATIVE, so every 'on'
    draw is floored at the smallest observed active value (~the JumpReLU threshold) --
    an 'on' label is active in SAE coordinates (re-encoding the steered residual often
    reads it below threshold; see README section 6). 'extreme' at scale 1 == uniform over the
    active range; raise extreme_scale for larger-than-natural injections."""
    def __init__(self, active_values, distribution="extreme", extreme_scale=1.0, seed=0):
        av = np.asarray([v for v in active_values if v > 0], dtype=float)
        if len(av) == 0:
            raise SystemExit("no active activations in train cache; cannot sample 'on' values")
        self.av, self.distribution = av, distribution
        self.amin, self.amax = float(av.min()), float(av.max())
        self.amean, self.astd = float(av.mean()), float(av.std())
        self.scale = extreme_scale if distribution == "extreme" else 1.0
        self.rng = np.random.RandomState(seed)

    def _on_value(self):
        d, s = self.distribution, self.scale
        if d == "extreme":
            hi = (self.amin + self.amax) / 2 + s * (self.amax - self.amin) / 2
            v = self.rng.uniform(self.amin, max(hi, self.amin))
        elif d == "empirical":
            v = float(self.rng.choice(self.av)) * s
        elif d == "gaussian":
            v = self.rng.normal(self.amean, s * self.astd)
        else:
            raise ValueError(f"unknown distribution {d!r}")
        return float(max(self.amin, v))

    def draw(self):
        """(is_on, value)."""
        return (True, self._on_value()) if self.rng.random() < 0.5 else (False, 0.0)


class MultiOnOffSampler:
    """OnOffSampler per feature -- each 'on' magnitude comes from THAT feature's own
    active range -- sharing one RNG, so the draw sequence is fixed by the row order
    alone. draw(feature_index) -> (is_on, value)."""
    def __init__(self, active_by_feature, distribution="extreme", extreme_scale=1.0, seed=0):
        self.rng = np.random.RandomState(seed)
        self.samplers = {}
        for f, vals in active_by_feature.items():
            s = OnOffSampler(vals, distribution=distribution, extreme_scale=extreme_scale, seed=seed)
            s.rng = self.rng
            self.samplers[int(f)] = s

    def draw(self, feature_index):
        return self.samplers[int(feature_index)].draw()

    @classmethod
    def from_rows(cls, raw, **kw):
        sig = np.asarray(raw["signal_index"])
        act = np.asarray(raw["activation"], dtype=float)
        by_f = {int(f): act[(sig == f) & (act > 0)] for f in np.unique(sig)}
        dead = [f for f, v in by_f.items() if len(v) == 0]
        if dead:
            raise SystemExit(f"no active rows (no 'on' magnitude to steer to) for feature "
                             f"index(es) {dead[:10]}; drop them from the pool")
        return cls(by_f, **kw)


class JointTrainCollator:
    """Prompt-only rows, RIGHT-padded (readout = last real token = attn.sum(1)-1).
    Emits the natural label AND a freshly sampled on/off steer, plus an
    `anchor_region_mask` (input text=0, introspection prompt=1, pad=0) for the KL
    anchor. Gate OFF: no lora_token_mask.

    With a MultiOnOffSampler each row is steered on/off for ITS OWN feature (its
    signal_index), drawn from that feature's active range, and the batch carries
    `feature_index` for the steering hook."""
    def __init__(self, tokenizer, sampler):
        self.pad_id, self.sampler = tokenizer.pad_token_id, sampler
        self.per_feature = isinstance(sampler, MultiOnOffSampler)

    def __call__(self, features):
        ids = [list(f["input_ids"]) for f in features]
        m = max(len(x) for x in ids)
        steer, interv, region = [], [], []
        for x, f in zip(ids, features):
            is_on, value = (self.sampler.draw(f["signal_index"]) if self.per_feature
                            else self.sampler.draw())
            steer.append(value)
            interv.append(int(is_on))
            fil = int(f["frozen_input_len"])
            region.append([0] * fil + [1] * (len(x) - fil) + [0] * (m - len(x)))
        batch = {
            "input_ids": torch.tensor([x + [self.pad_id] * (m - len(x)) for x in ids]),
            "attention_mask": torch.tensor([[1] * len(x) + [0] * (m - len(x)) for x in ids]),
            "token_position": torch.tensor([int(f["token_position"]) for f in features]),
            "natural_label": torch.tensor([1 if is_active(f["activation"]) else 0 for f in features]),
            "steer_value": torch.tensor(steer, dtype=torch.float32),
            "interv_label": torch.tensor(interv, dtype=torch.long),
            "anchor_region_mask": torch.tensor(region),
        }
        if self.per_feature:
            batch["feature_index"] = torch.tensor([int(f["signal_index"]) for f in features])
        assert bool((((batch["anchor_region_mask"] == 0) & (batch["attention_mask"] == 1))
                     .any(dim=1)).all()), "a row has no input-text position for the KL anchor"
        return batch


class EvalCollator:
    """Prompt-only rows, LEFT-padded so out.logits[:, -1] is the readout at the last
    real token. Gate OFF: no lora_token_mask emitted."""
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        ids = [list(f["input_ids"]) for f in features]
        m = max(len(x) for x in ids)
        return {
            "input_ids": torch.tensor([[self.pad_id] * (m - len(x)) + x for x in ids]),
            "attention_mask": torch.tensor([[0] * (m - len(x)) + [1] * len(x) for x in ids]),
        }


def sample_text_positions(region_mask, attention_mask, k):
    """Up to k INPUT-TEXT positions per row (region==0 & attn==1), uniformly at random.
    Returns [B,k] indices and a validity mask (short rows may have < k)."""
    valid = (region_mask == 0) & (attention_mask == 1)
    scores = torch.rand(valid.shape, device=valid.device).masked_fill(~valid, -1.0)
    top = scores.topk(min(k, valid.shape[1]), dim=1)
    return top.indices, top.values >= 0.0


def kl_anchor(model, input_ids, attention_mask, tuned_logits, region_mask, k):
    """Forward KL(base || tuned) over sampled text positions, in nats/token.

    tuned_logits are the endogenous forward's logits ([B,L,V]) -- already carrying
    gradient and already through Gemma-2's final_logit_softcapping. The base reference
    is this same model with adapters disabled (for LoRA that IS the base model), whose
    logits are softcapped the SAME way -- so the KL penalises drift, not the softcap.
    (Re-running the LM head on hidden states would compare uncapped vs capped logits.)"""
    idx, ok = sample_text_positions(region_mask, attention_mask, k)
    if not bool(ok.any()):
        return None
    b = torch.arange(idx.shape[0], device=idx.device).unsqueeze(1)
    lt = tuned_logits[b, idx]
    with torch.no_grad(), model.disable_adapter():
        lb = model(input_ids=input_ids, attention_mask=attention_mask).logits[b, idx]
    lp_t, lp_b = torch.log_softmax(lt.float(), -1), torch.log_softmax(lb.float(), -1)
    kl = (lp_b.exp() * (lp_b - lp_t)).sum(-1)
    return (kl * ok.float()).sum() / ok.float().sum().clamp(min=1.0)


class JointTrainer(Trainer):
    def __init__(self, *a, bundle, feature_index, module_path, active_id, inactive_id,
                 joint_w, kl_target=None, kl_weight=0.0, kl_positions=4,
                 kl_coef_init=1.0, kl_range=(1e-4, 1e3), **k):
        super().__init__(*a, **k)
        self.bundle, self.feature_index = bundle, feature_index
        self.module = operator.attrgetter(module_path)(self.model)
        self.active_id, self.inactive_id, self.w = active_id, inactive_id, joint_w
        self.kl_target = kl_target if (kl_target is not None and kl_target > 0) else None
        self.kl_positions, self.kl_range = kl_positions, kl_range
        self.kl_coef = kl_coef_init if self.kl_target is not None else kl_weight
        self.kl_on = self.kl_coef > 0 or self.kl_target is not None

    def _readout_ce(self, logits, last, target_ids):
        b = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits[b, last].float(), target_ids)

    def _update_kl_coef(self, kl_value):
        """PPO-style multiplicative controller holding measured KL near the budget;
        1.5x dead-bands avoid oscillating on step-to-step noise."""
        if self.kl_target is None:
            return
        lo, hi = self.kl_range
        if kl_value > 1.5 * self.kl_target:
            self.kl_coef = min(self.kl_coef * 1.5, hi)
        elif kl_value < self.kl_target / 1.5:
            self.kl_coef = max(self.kl_coef / 1.5, lo)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ii, am = inputs["input_ids"], inputs["attention_mask"]
        nat, il = inputs["natural_label"], inputs["interv_label"]
        last = am.sum(1) - 1
        aid = torch.full_like(nat, self.active_id)
        iid = torch.full_like(nat, self.inactive_id)

        out_e = model(input_ids=ii, attention_mask=am)
        endo = self._readout_ce(out_e.logits, last, torch.where(nat.bool(), aid, iid))

        if self.w >= 1.0:
            loss = endo
        else:
            # feature_index=None: multi-feature run, each row steers its own feature
            fidx = inputs["feature_index"] if self.feature_index is None else self.feature_index
            hook = SAEInterventionHook(self.bundle, inputs["token_position"],
                                       fidx, inputs["steer_value"])
            handle = self.module.register_forward_hook(hook)
            try:
                out_i = model(input_ids=ii, attention_mask=am)
            finally:
                handle.remove()
            interv = self._readout_ce(out_i.logits, last, torch.where(il.bool(), aid, iid))
            loss = self.w * endo + (1.0 - self.w) * interv

        L_kl = None
        if self.kl_on:
            L_kl = kl_anchor(model, ii, am, out_e.logits,
                             inputs["anchor_region_mask"], self.kl_positions)
            if L_kl is not None:
                loss = loss + self.kl_coef * L_kl
                self._update_kl_coef(float(L_kl.detach()))

        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            log = {"L_endo": float(endo.detach())}
            if self.kl_on:
                log["L_kl"] = float(L_kl.detach()) if L_kl is not None else 0.0
                log["kl_coef"] = float(self.kl_coef)
            self.log(log)
        return (loss, out_e) if return_outputs else loss


@torch.no_grad()
def _readout_argmax(model, ds_pred, collator, batch_size, hook_factory=None,
                    module=None, positions=None):
    """Run the model over left-padded prompt-only batches and return, per row, the
    token the model GENERATES at the readout position -- argmax over the full
    vocabulary at the last token (one forward, greedy). If hook_factory is given, a
    steering hook (built per batch from `positions`) is registered for the forward."""
    model.eval()
    preds, ptr = [], 0
    for batch in DataLoader(ds_pred, batch_size=batch_size, collate_fn=collator, shuffle=False):
        ii = batch["input_ids"].to(model.device)
        am = batch["attention_mask"].to(model.device)
        n = ii.shape[0]
        handle = None
        if hook_factory is not None:
            abs_pos = torch.tensor(positions[ptr:ptr + n], device=model.device) \
                      + (am == 0).sum(dim=1)
            handle = module.register_forward_hook(hook_factory(abs_pos))
        try:
            logits = model(input_ids=ii, attention_mask=am).logits[:, -1, :]
        finally:
            if handle is not None:
                handle.remove()
        preds.append(logits.argmax(dim=-1).cpu())
        ptr += n
    return torch.cat(preds).numpy()


def endogenous_eval(model, ds_pred, activations, collator, aid, iid, batch_size):
    """Generation-based balanced accuracy + precision of the natural readout."""
    preds = _readout_argmax(model, ds_pred, collator, batch_size)
    labels = [1 if is_active(a) else 0 for a in activations]
    return generation_metrics(preds, labels, aid, iid)


def interventional_eval(model, spec, bundle, module, test_raw, ds_pred, collator,
                        aid, iid, on_value, batch_size):
    """Interventional detection on the GENERATED token. Every context is run twice --
    feature steered to `on_value` and to 0 at the SAME token -- and the two forwards
    form a 50/50 on/off set whose label is the forced state, decorrelated from the
    text. Reports balanced accuracy + precision on that set: because the label is
    decorrelated from context, a faithful (causal) detector scores ~1.0, a context
    bypass ~0.5, an inverse effect <0.5."""
    positions = np.asarray(test_raw["token_position"])

    def steer_to(value):
        """A hook_factory: given the batch's absolute positions, steer every row to `value`."""
        def factory(abs_pos):
            values = torch.full(abs_pos.shape, value, device=abs_pos.device, dtype=torch.float32)
            return SAEInterventionHook(bundle, abs_pos, spec.index, values)
        return factory

    pred_on = _readout_argmax(model, ds_pred, collator, batch_size,
                              hook_factory=steer_to(on_value), module=module, positions=positions)
    pred_off = _readout_argmax(model, ds_pred, collator, batch_size,
                               hook_factory=steer_to(0.0), module=module, positions=positions)
    preds = np.concatenate([pred_on, pred_off])
    labels = np.concatenate([np.ones_like(pred_on), np.zeros_like(pred_off)])
    return generation_metrics(preds, labels, aid, iid)


def drop_retokenization_mismatches(raw, tokenizer, split, num_proc=None):
    """Keep rows whose prompt, re-tokenized, still has the cached token at token_position.
    The prompt is f"{text} {separator}...": when a truncated text ends in a lone space
    token, the template's space merges with it ("\u2581" -> "\u2581\u2581"), so the
    prompt names, reads and steers a different token than the label describes
    (~1% of rows at the last position of a truncated text, none elsewhere)."""
    def ok(batch):
        enc = tokenizer([f"{t} {INTROSPECTION_SEPARATOR}" for t in batch["text"]])["input_ids"]
        return [p < len(ids) and tokenizer.decode(ids[p]) == tok
                for ids, p, tok in zip(enc, batch["token_position"], batch["token"])]
    kept = raw.filter(ok, batched=True, num_proc=num_proc)
    print(f"[joint] {split}: dropped {len(raw) - len(kept)}/{len(raw)} rows whose "
          f"token changes when the prompt is re-tokenized")
    return kept


class KLCoefCheckpoint(TrainerCallback):
    """Saves the adaptive KL coefficient with each checkpoint so a resumed run continues
    from the controller's state instead of restarting it at kl_coef_init."""
    def __init__(self, trainer_ref):
        self.trainer_ref = trainer_ref

    def on_save(self, args, state, control, **kw):
        d = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if d.exists():
            (d / "kl_coef.json").write_text(json.dumps({"kl_coef": self.trainer_ref[0].kl_coef}))


@torch.no_grad()
def cache_alignment(model, spec, bundle, module, test_raw, ds_pred, collator, tokenizer,
                    batch_size, out_dir, max_examples=512):
    """Gate-0 regression test: recompute the feature at the cached position with the
    adapter OFF (reproducing the model that built the cache) and compare to the cached
    activation across +/-1 token offsets. WANT best offset +0, corr ~1.0, token match
    ~1.0 -- otherwise the steering hook injects into a different token than the prompt
    names (the off-by-one class of bug). Writes sae_cache_alignment.json."""
    model.eval()
    offsets = (-1, 0, 1)
    # Up to half the sample from active rows: sparse features can have no positives in
    # the first rows at all, which makes every correlation NaN and fails the check.
    all_act = np.asarray(test_raw["activation"], float)
    pos_idx, neg_idx = np.flatnonzero(all_act > 0), np.flatnonzero(all_act <= 0)
    n_pos = min(len(pos_idx), max_examples // 2)
    sel = np.sort(np.concatenate([pos_idx[:n_pos], neg_idx[:max_examples - n_pos]])).tolist()
    positions = np.asarray(test_raw["token_position"])[sel]
    cached_tok = list(test_raw.select(sel)["token"])
    cached_act = all_act[sel]
    n = len(sel)
    live = {o: [] for o in offsets}
    tok_match, ptr = [], 0
    for batch in DataLoader(ds_pred.select(sel), batch_size=batch_size,
                            collate_fn=collator, shuffle=False):
        ii = batch["input_ids"].to(model.device)
        am = batch["attention_mask"].to(model.device)
        bs = ii.shape[0]
        abs_pos = torch.tensor(positions[ptr:ptr + bs], device=model.device) + (am == 0).sum(1)
        captured = {}

        def capture(mod, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            b = torch.arange(hidden.shape[0], device=hidden.device)
            for o in offsets:
                p = (abs_pos + o).clamp(0, hidden.shape[1] - 1)
                captured[o] = hidden[b, p].detach().float()
            return None

        handle = module.register_forward_hook(capture)
        try:
            with model.disable_adapter():
                model(input_ids=ii, attention_mask=am)
        finally:
            handle.remove()
        for o, resid in captured.items():
            live[o].append(bundle.encode(resid)[:, spec.index].float().cpu())
        for i in range(bs):
            tok_match.append(tokenizer.decode(ii[i, abs_pos[i]]) == cached_tok[ptr + i])
        ptr += bs

    cached = cached_act[:ptr]
    corr = {}
    for o in offsets:
        v = torch.cat(live[o]).numpy()
        corr[o] = float(np.corrcoef(v, cached)[0, 1]) if min(v.std(), cached.std()) > 1e-9 else float("nan")
    best = max(offsets, key=lambda o: (-np.inf if np.isnan(corr[o]) else corr[o]))
    match_rate = float(np.mean(tok_match))
    min_pos = 10
    if n_pos >= min_pos:
        aligned = best == 0 and corr[0] > 0.99 and match_rate > 0.99
    else:   # too few positives for a meaningful correlation: rely on the token check
        aligned = match_rate > 0.99
        print(f"  WARNING: only {n_pos} active test rows (< {min_pos}); "
              f"correlation not checked, token match only")
    print("\n=== SAE CACHE ALIGNMENT (adapters off) ===")
    print(f"  n={ptr}  n_active={n_pos}  token_match={match_rate:.4f}")
    for o in offsets:
        print(f"  offset {o:+d}: corr(live, cached) = {corr[o]:+.4f}"
              + ("   <-- train assumes this" if o == 0 else ""))
    print(f"  best offset = {best:+d}  ->  {'alignment OK' if aligned else 'MISALIGNED'}")
    res = {"n": ptr, "n_active": int(n_pos), "token_match": match_rate,
           "corr_by_offset": {str(k): v for k, v in corr.items()},
           "best_offset": int(best), "aligned": bool(aligned)}
    (Path(out_dir) / "sae_cache_alignment.json").write_text(json.dumps(res, indent=2))
    return res


def make_preprocess(config, tokenizer, for_train, multi=False):
    """for_train: prompt-only + frozen_input_len (for the region mask) + the raw
    token_position/activation the collator needs (+ signal_index when multi, so each
    row steers its own feature). Eval: prompt-only, detection words set so the prompt
    ends exactly before the answer token."""
    tc = config.training_config
    prompt_fn = get_prompt_fn(tc)
    def pp(ex):
        out = preprocess_example(
            ex, tokenizer=tokenizer, prompt_fn=prompt_fn,
            max_length=tc.max_seq_length, return_frozen_input_len=True, for_prediction=True,
            **({} if for_train else dict(detection=True,
                                         detection_active_word=tc.detection_active_word,
                                         detection_inactive_word=tc.detection_inactive_word)))
        if for_train:
            out["token_position"] = ex["token_position"]
            out["activation"] = ex["activation"]
            if multi:
                out["signal_index"] = ex["signal_index"]
                # never read: JointTrainCollator rebuilds the mask and has no LM labels.
                # Dropped because .map writes every column to disk -- ~2/3 of a 10M-row
                # run's preprocessing cache.
                out.pop("labels", None)
                out.pop("attention_mask", None)
        return out
    return pp


def assert_prompts_fit(ds, max_length, split):
    """preprocess_example TRUNCATES at max_length, cutting the END of the prompt -- the
    readout token -- so a too-long prompt is read out mid-sentence with no error. A
    label prompt is much longer than an index prompt, so check rather than assume."""
    # Arrow-side: ds["input_ids"] would build Python lists, ~15 GB at 10M rows
    import pyarrow.compute as pc
    longest = pc.max(pc.list_value_length(ds.data.column("input_ids"))).as_py()
    if longest >= max_length:
        raise SystemExit(f"{split}: a prompt reaches max_seq_length={max_length} tokens and "
                         f"was truncated. Raise --max-prompt-tokens in the config generator.")
    print(f"[joint] {split}: longest prompt {longest} tokens (< max_seq_length {max_length})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--joint-loss-weight", type=float, default=0.5,
                   help="w on the ENDOGENOUS term; interventional gets (1-w). "
                        "w=1 => pure endogenous (interventional forward skipped).")
    p.add_argument("--kl-anchor-target", type=float, default=None,
                   help="Adaptive KL budget (nats/token) on the text region. "
                        "None => fixed --kl-anchor-weight; 0 disables the anchor.")
    p.add_argument("--kl-anchor-weight", type=float, default=0.0)
    p.add_argument("--kl-anchor-positions", type=int, default=4)
    p.add_argument("--kl-anchor-coef-init", type=float, default=1.0)
    p.add_argument("--kl-anchor-coef-range", type=str, default="0.0001,1000")
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--force-drift-eval", action="store_true")
    p.add_argument("--ignore-misalignment", action="store_true")
    # ---- multi-feature runs (training_config.detection_total_train_rows set) ----
    p.add_argument("--feature-gen-pool-config", type=Path, default=None,
                   help="Multi-feature runs: config of the POOL cache (every pool feature over "
                        "the eval texts). The end-of-run eval scores every feature in it, "
                        "labelled trained/held-out (score_sae_feature_generalization.py). "
                        "Without it, only the run's own features on its own test split.")
    p.add_argument("--feature-gen-only", action="store_true",
                   help="Load the latest (or --checkpoint) checkpoint and run ONLY the "
                        "multi-feature eval; no training.")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--fg-interv-texts", type=int, default=500,
                   help="Natural-distribution texts per feature for the interventional eval "
                        "(each run steered on AND off).")
    p.add_argument("--fg-endo-max-pos", type=int, default=500)
    p.add_argument("--fg-endo-neg", type=int, default=1000)
    p.add_argument("--fg-distractor-features", type=int, default=20,
                   help="Seen AND held-out features (this many of each) in the steer x ask "
                        "distractor matrix. 0 disables it.")
    p.add_argument("--fg-distractor-texts", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(args.config.read_text())
    tc = config.training_config
    assert tc.objective == "detection", "joint detector expects objective='detection'"
    assert not tc.disable_adapter_in_input, "joint runs are gate-OFF; do not set disable_adapter_in_input"
    if tc.detection_total_train_rows is not None:
        # Multi-feature runs take their own path, so the single-feature recipe below --
        # which the layer x position grid runs on -- is untouched by them.
        return main_multi(args, config)
    if args.feature_gen_only or args.feature_gen_pool_config:
        raise SystemExit("--feature-gen-* need a multi-feature config "
                         "(training_config.detection_total_train_rows)")
    w = args.joint_loss_weight

    out_dir = Path(config.output_dir) / config.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "sae_joint_train_metrics.json").exists() and not args.overwrite_cache:
        print(f"Already ran, skipping {out_dir}")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": [INTROSPECTION_SEPARATOR]})

    cache = config.dataset_cache_dir
    train_raw = apply_signal_filter(load_from_disk(str(cache / "train")), config.signal_filter, "train")
    test_raw = apply_signal_filter(load_from_disk(str(cache / "test")), config.signal_filter, "test")
    train_raw = drop_retokenization_mismatches(train_raw, tokenizer, "train")
    test_raw = drop_retokenization_mismatches(test_raw, tokenizer, "test")
    spec = resolve_spec(config)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16 if tc.bf16 else torch.float32, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))
    model = get_peft_model(model, LoraConfig(
        r=tc.lora_r, lora_alpha=tc.lora_alpha, target_modules=tc.lora_target_modules,
        lora_dropout=tc.lora_dropout, bias="none", task_type="CAUSAL_LM"))
    model.resize_token_embeddings(len(tokenizer))
    replace_model_with_gated_version(model)
    replace_lora_with_gated_version(model)

    bundle = load_sae_for_spec(spec, device=str(model.device))
    bundle.sae.requires_grad_(False)   # frozen; otherwise every backward fills W_enc/W_dec .grad
    module_path = resolve_sae_hook_path(model, spec.layer, site=spec.sae_site, path_to_output=False)
    module = operator.attrgetter(module_path)(model)
    aid, iid = label_token_ids(tokenizer, tc)
    sampler = OnOffSampler(train_raw["activation"], distribution=tc.intervention_signal_distribution,
                           extreme_scale=tc.intervention_extreme_scale, seed=config.seed)
    print(f"[joint] w(endo)={w}  gate=OFF  feature={spec.layer}.{spec.index}  "
          f"kl_target={args.kl_anchor_target}  active[min/mean/max]="
          f"{sampler.amin:.3g}/{sampler.amean:.3g}/{sampler.amax:.3g}")

    train_bal = oversample_positives(train_raw, tc.detection_balance_pos_frac, config.seed)
    random.seed(config.seed)   # prompt templates are drawn with random.choice
    train_ds = train_bal.map(make_preprocess(config, tokenizer, for_train=True),
                             remove_columns=train_bal.column_names)
    random.seed(config.seed + 1)   # same eval templates for every run on this cache
    test_pred = test_raw.map(make_preprocess(config, tokenizer, for_train=False),
                             remove_columns=test_raw.column_names)
    eval_collator = EvalCollator(tokenizer)

    if args.force_drift_eval:
        align = cache_alignment(model, spec, bundle, module, test_raw, test_pred,
                                eval_collator, tokenizer, tc.eval_batch_size, out_dir)
        if not align["aligned"] and not args.ignore_misalignment:
            raise SystemExit("cache alignment FAILED; pass --ignore-misalignment to train anyway.")

    total_steps = max(1, math.ceil(len(train_ds) / tc.train_batch_size * float(tc.num_epochs)))
    warmup = {"warmup_steps": max(1, round(tc.warmup_ratio * total_steps))} \
        if tc.warmup_ratio and tc.warmup_ratio > 0 else {}
    save = {"save_strategy": "epoch"} if tc.save_steps <= 0 \
        else {"save_strategy": "steps", "save_steps": tc.save_steps}
    training_args = TrainingArguments(
        output_dir=str(out_dir), run_name=config.experiment_name,
        learning_rate=tc.learning_rate, lr_scheduler_type=tc.lr_scheduler_type,
        per_device_train_batch_size=tc.train_batch_size, num_train_epochs=tc.num_epochs,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        logging_steps=math.ceil(100 / tc.train_batch_size), eval_strategy="no",
        bf16=tc.bf16, remove_unused_columns=False, report_to="wandb", **warmup, **save)

    trainer = JointTrainer(
        model=model, args=training_args, train_dataset=train_ds,
        data_collator=JointTrainCollator(tokenizer, sampler),
        bundle=bundle, feature_index=spec.index, module_path=module_path,
        active_id=aid, inactive_id=iid, joint_w=w,
        kl_target=args.kl_anchor_target, kl_weight=args.kl_anchor_weight,
        kl_positions=args.kl_anchor_positions, kl_coef_init=args.kl_anchor_coef_init,
        kl_range=tuple(float(x) for x in args.kl_anchor_coef_range.split(",")),
        callbacks=[StripEmbeddingSaveCallback(n_added)])
    trainer.add_callback(KLCoefCheckpoint([trainer]))
    ckpts = glob.glob(f"{out_dir}/checkpoint-*")
    resume = max(ckpts, key=os.path.getctime) if ckpts and not args.overwrite_cache else None
    if resume and (Path(resume) / "kl_coef.json").exists():
        trainer.kl_coef = json.loads((Path(resume) / "kl_coef.json").read_text())["kl_coef"]
        print(f"[joint] resuming from {resume} with kl_coef={trainer.kl_coef:.4g}")
    trainer.train(resume_from_checkpoint=resume)

    test_endo = endogenous_eval(model, test_pred, test_raw["activation"], eval_collator,
                                aid, iid, tc.eval_batch_size)
    test_interv = interventional_eval(model, spec, bundle, module, test_raw, test_pred,
                                      eval_collator, aid, iid, active_on_value(train_raw),
                                      tc.eval_batch_size)
    metrics = {"joint_loss_weight": w, "gate": "off",
               "kl_anchor_target": args.kl_anchor_target,
               "kl_anchor_weight": args.kl_anchor_weight,
               "test_endogenous": test_endo, "test_interventional": test_interv}
    print(json.dumps(metrics, indent=2))
    (out_dir / "sae_joint_train_metrics.json").write_text(json.dumps(metrics, indent=2))


def build_model(config, tokenizer, checkpoint=None):
    """Gated-LoRA gemma, as main() builds it -- fresh adapter, or a trained one loaded
    frozen from `checkpoint` (with the separator's embedding row restored, since
    StripEmbeddingSaveCallback drops the embedding matrices from every checkpoint)."""
    tc = config.training_config
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16 if tc.bf16 else torch.float32, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))
    if checkpoint is None:
        model = get_peft_model(model, LoraConfig(
            r=tc.lora_r, lora_alpha=tc.lora_alpha, target_modules=tc.lora_target_modules,
            lora_dropout=tc.lora_dropout, bias="none", task_type="CAUSAL_LM"))
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(checkpoint), autocast_adapter_dtype=False)
    model.resize_token_embeddings(len(tokenizer))
    if checkpoint is not None:
        load_added_token_embeddings(model, str(checkpoint))
    replace_model_with_gated_version(model)
    replace_lora_with_gated_version(model)
    return model


def main_multi(args, config):
    """N SAE features in one run (the feature-generalization experiment).

    Same objective, gate, KL anchor and steering as the single-feature path. What
    changes: every row carries its own feature (prompt names it, steering targets it,
    'on' magnitudes come from its own active range); the train split is resampled to
    a fixed TOTAL budget split equally across features (sae_featgen.balance_rows_per_
    feature); and the eval scores every feature of a pool cache, trained or not.
    """
    from score_sae_feature_generalization import evaluate_feature_generalization
    tc = config.training_config
    w = args.joint_loss_weight
    out_dir = Path(config.output_dir) / config.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / "sae_joint_train_metrics.json"
    if done.exists() and not args.overwrite_cache and not args.feature_gen_only:
        print(f"Already ran, skipping {out_dir}")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": [INTROSPECTION_SEPARATOR]})
    specs = resolve_specs(config)
    feat_str = ",".join(f"S{s.layer}.{s.index}" for s in specs)
    print(f"[multi] {len(specs)} feature(s): {feat_str if len(specs) <= 10 else feat_str[:120] + '...'}  "
          f"prompt={tc.prompt_type}  budget={tc.detection_total_train_rows} rows")

    ckpt = None
    if args.feature_gen_only:
        ckpts = glob.glob(f"{out_dir}/checkpoint-*")
        ckpt = args.checkpoint or (max(ckpts, key=os.path.getctime) if ckpts else None)
        if ckpt is None:
            raise SystemExit(f"--feature-gen-only: no checkpoint in {out_dir}; pass --checkpoint")
        print(f"[multi] --feature-gen-only: evaluating {ckpt}")
    # Seed BEFORE building the model: resize_token_embeddings draws the separator's new
    # embedding row, and LoRA draws its A matrices, from the global torch RNG, which
    # nothing else seeds before Trainer() -- unseeded, two identical runs differ from
    # step 0 (the single-feature main() still has that property).
    set_seed(config.seed)
    model = build_model(config, tokenizer, checkpoint=ckpt)
    bundle = load_sae_for_spec(specs[0], device=str(model.device))
    bundle.sae.requires_grad_(False)
    module_path = resolve_sae_hook_path(model, specs[0].layer, site=specs[0].sae_site,
                                        path_to_output=False)
    module = operator.attrgetter(module_path)(model)
    aid, iid = label_token_ids(tokenizer, tc)
    eval_collator = EvalCollator(tokenizer)

    if not args.feature_gen_only:
        cache = config.dataset_cache_dir
        filt = config.signal_filter or feat_str
        train_raw = apply_signal_filter(load_from_disk(str(cache / "train")), filt, "train")
        # Above ~1M rows the single-process tokenization passes take hours (measured
        # ~2k rows/s), so they fan out over the job's slots. Below it they stay serial:
        # with workers, each shard draws prompt templates from its own copy of the RNG,
        # which would change the template assignment of the runs already made.
        workers = int(os.environ.get("NSLOTS", "1"))
        workers = workers if workers > 1 else None
        big = lambda ds: workers if len(ds) > 1_000_000 else None
        train_raw = drop_retokenization_mismatches(train_raw, tokenizer, "train",
                                                   num_proc=big(train_raw))
        train_raw = ensure_subsample_u(train_raw, config.seed)
        train_bal, bal_stats = balance_rows_per_feature(
            train_raw, [s.index for s in specs], tc.detection_total_train_rows,
            tc.detection_balance_pos_frac, config.seed)
        bal_stats.insert(0, "signal", [f"S{specs[0].layer}.{f}" for f in bal_stats["feature"]])
        bal_stats.to_csv(out_dir / "train_balance_by_feature.csv", index=False)
        sampler = MultiOnOffSampler.from_rows(
            train_raw, distribution=tc.intervention_signal_distribution,
            extreme_scale=tc.intervention_extreme_scale, seed=config.seed)
        print(f"[multi] w(endo)={w}  gate=OFF  kl_target={args.kl_anchor_target}  "
              f"rows={len(train_bal)}")

        random.seed(config.seed)
        train_ds = train_bal.map(make_preprocess(config, tokenizer, for_train=True, multi=True),
                                 remove_columns=train_bal.column_names, num_proc=big(train_bal))
        assert_prompts_fit(train_ds, tc.max_seq_length, "train")

        if args.force_drift_eval:
            # On the run's own TRAIN rows of its first feature: thousands of actives,
            # and it is exactly the data the run trains on.
            align_raw = train_raw.filter(lambda ex: ex["signal_index"] == specs[0].index)
            random.seed(config.seed + 2)
            align_pred = align_raw.map(make_preprocess(config, tokenizer, for_train=False),
                                       remove_columns=align_raw.column_names)
            align = cache_alignment(model, specs[0], bundle, module, align_raw, align_pred,
                                    eval_collator, tokenizer, tc.eval_batch_size, out_dir)
            if not align["aligned"] and not args.ignore_misalignment:
                raise SystemExit("cache alignment FAILED; pass --ignore-misalignment to train anyway.")

        total_steps = max(1, math.ceil(len(train_ds) / tc.train_batch_size * float(tc.num_epochs)))
        warmup = {"warmup_steps": max(1, round(tc.warmup_ratio * total_steps))} \
            if tc.warmup_ratio and tc.warmup_ratio > 0 else {}
        save = {"save_strategy": "epoch"} if tc.save_steps <= 0 \
            else {"save_strategy": "steps", "save_steps": tc.save_steps}
        training_args = TrainingArguments(
            output_dir=str(out_dir), run_name=config.experiment_name,
            learning_rate=tc.learning_rate, lr_scheduler_type=tc.lr_scheduler_type,
            per_device_train_batch_size=tc.train_batch_size, num_train_epochs=tc.num_epochs,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            logging_steps=math.ceil(100 / tc.train_batch_size), eval_strategy="no",
            bf16=tc.bf16, remove_unused_columns=False, report_to="wandb", **warmup, **save)
        trainer = JointTrainer(
            model=model, args=training_args, train_dataset=train_ds,
            data_collator=JointTrainCollator(tokenizer, sampler),
            bundle=bundle, feature_index=None, module_path=module_path,
            active_id=aid, inactive_id=iid, joint_w=w,
            kl_target=args.kl_anchor_target, kl_weight=args.kl_anchor_weight,
            kl_positions=args.kl_anchor_positions, kl_coef_init=args.kl_anchor_coef_init,
            kl_range=tuple(float(x) for x in args.kl_anchor_coef_range.split(",")),
            callbacks=[StripEmbeddingSaveCallback(n_added)])
        trainer.add_callback(KLCoefCheckpoint([trainer]))
        ckpts = glob.glob(f"{out_dir}/checkpoint-*")
        resume = max(ckpts, key=os.path.getctime) if ckpts and not args.overwrite_cache else None
        if resume and (Path(resume) / "kl_coef.json").exists():
            trainer.kl_coef = json.loads((Path(resume) / "kl_coef.json").read_text())["kl_coef"]
            print(f"[multi] resuming from {resume} with kl_coef={trainer.kl_coef:.4g}")
        trainer.train(resume_from_checkpoint=resume)

    summary = evaluate_feature_generalization(
        model, config, tokenizer, specs, bundle, module, aid, iid, out_dir,
        pool_config=args.feature_gen_pool_config, batch_size=tc.eval_batch_size,
        n_interv=args.fg_interv_texts, endo_max_pos=args.fg_endo_max_pos,
        endo_neg=args.fg_endo_neg, n_distractor=args.fg_distractor_features,
        distractor_texts=args.fg_distractor_texts)
    metrics = {"joint_loss_weight": w, "gate": "off", "n_features": len(specs),
               "prompt_type": tc.prompt_type,
               "detection_total_train_rows": tc.detection_total_train_rows,
               "kl_anchor_target": args.kl_anchor_target,
               "kl_anchor_weight": args.kl_anchor_weight,
               "feature_generalization": summary}
    print(json.dumps(metrics, indent=2, default=str))
    done.write_text(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
