import os
import pdb
import math
import bisect
import random
import operator
from typing import Any, Optional

import torch
from torch import nn

from peft import PeftModelForCausalLM
from peft.tuners.lora.layer import Linear
from transformers import TrainerCallback

from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.models.attention_logits import (
    AttentionLogitPort, AttentionWeightPort, PORT_ATTR, PORT_ATTRS,
    assert_attention_impl, is_attention_type, resolve_attention_module,
)

# the custom regression head is NOT part of the PEFT adapter, so PEFT's
# save_pretrained never serializes it. We persist/load it under this filename
# alongside each checkpoint ourselves.
REGRESSION_HEAD_FILENAME = "regression_head.pt"

# default multiplier for the "extreme" intervention sanity check (samples
# uniformly from a range EXTREME_SCALE x wider than empirical, centered on the
# empirical midpoint). Override per-experiment via TrainingConfig.intervention_extreme_scale
EXTREME_SCALE = 1000.0




class GatedLinear(Linear):

    def forward(self, x, *args, **kwargs):
        result = self.base_layer(x)
        if self.disable_adapters:
            return result
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            # peft's Linear.forward applies lora_dropout to the adapter input; this
            # hand-rolled forward used to skip it, silently training every gated run
            # at dropout=0 no matter what lora_dropout was configured to. Default
            # lora_dropout is now 0.0 so that behavior is the explicit default.
            dropout = self.lora_dropout[active_adapter]
            x_cast = dropout(x.to(lora_A.weight.dtype))
            lora_out = lora_B(lora_A(x_cast)) * self.scaling[active_adapter]
            lora_out = lora_out.to(result.dtype)

            mask = self.lora_token_mask
            if mask is not None:
                seq_len = x.shape[1]
                if mask.shape[1] != seq_len:
                    # for generation, dynamically slice the mask, so when generation uses KV-cache, set mask to 1
                    mask = mask[:, -seq_len:]

                lora_out = lora_out * mask.unsqueeze(-1).to(lora_out.dtype)

            result = result + lora_out
        return result


def replace_lora_with_gated_version(model: PeftModelForCausalLM):
    n = 0
    for module in model.modules():
        if isinstance(module, Linear):
            module.__class__ = GatedLinear
            n += 1
    print(f"patched {n} LoRA layers")

def set_lora_masks(model: PeftModelForCausalLM, mask: Optional[torch.Tensor]):
    for module in model.modules():
        if isinstance(module, Linear):
            module.lora_token_mask = mask

def replace_model_with_gated_version(model: PeftModelForCausalLM):
    # monkey patch forward to take in lora_token
    orig_forward = model.forward
    def forward_with_mask(*args, lora_token_mask=None, **kwargs):
        # store lora token mask into each lora linear
        set_lora_masks(model, lora_token_mask)
        try:
            return orig_forward(*args, **kwargs)
        except TypeError as e:
            # in these cases it complains PeftModelForCausalLM.forward() got multiple values for argument 'input_ids'
            return orig_forward(**kwargs)
        finally:
            set_lora_masks(model, None)
    model.forward = forward_with_mask

    # monkey patch generate (this wasn't in the training, so I guess eval during training is not autoregressive)
    orig_generate = model.generate
    def generate_with_mask(*args, lora_token_mask=None, **kwargs):
        # lora_token_mask = args[0].pop("lora_token_mask", None)
        set_lora_masks(model, lora_token_mask)
        try:
            return orig_generate(*args, **kwargs)
        except AttributeError as e:
            return orig_generate(**kwargs)
        finally:
            set_lora_masks(model, None)
    model.generate = generate_with_mask

# class IntrospectionRegressionModel(PeftModelForCausalLM):
#
#     def __init__(self,model):
#         # do NOT call super().__init__ because we already HAVE a fully-built peft model instance
#         self.__dict__ = model.__dict__
#         self.regression_head = nn.Linear(self.config.hidden_size,1,bias=False)
#
#     def forward(
#         self,
#         *args,
#         labels=None,
#         regression_token_positions=-1,
#         output_hidden_states=True,
#         **kwargs,
#     ):
#         outputs = super().forward(*args,output_hidden_states=True,**kwargs)
#         hidden = outputs.hidden_states[-1]
#         bs = hidden.shape[0]
#
#         batch_idx = torch.arange(bs,device=hidden.device)
#         selected_hidden = hidden[batch_idx,regression_token_positions]
#
#         pred = self.regression_head(selected_hidden).squeeze(-1)
#
#         loss = None
#         if labels is not None:
#             regression_labels = labels.to(pred.dtype)
#             try:
#                 loss = nn.functional.mse_loss(pred,regression_labels)
#             except Exception as e:
#                 print(e)
#                 pdb.set_trace()
#
#         # attach for trainer / metrics
#         # outputs.regression_predictions = pred
#         outputs.logits = pred
#         # overwrite loss
#         outputs.loss = loss
#         # pdb.set_trace()
#         return outputs

# ============================================================
# Hook helpers
# ============================================================
# Third slot of a PAIR MODE range key; a string so it can never equal a real offset.
PAIR_RANGE_TAG = "pair"


def signal_range_key(layer, index, src_offset=None, pair=None):
    """Key into `signal_ranges`. 2-tuple for MLP/SAE (unchanged), 3-tuple for attention,
    5-tuple (layer, head, "pair", dst, src) for PAIR MODE attention.

    One function so the dict that trainers BUILD and the sampler that READS it cannot
    drift apart -- a mismatch there is a KeyError at best and a wrong intervention
    distribution at worst.

    PAIR MODE keys on the ABSOLUTE (dst, src), not on the offset: the weight at offset -1
    is ~0.5 at dst=1 and ~0.01 at dst=500, so an offset-keyed range would inject values
    from the wrong distribution at nearly every position.
    """
    if pair is not None:
        assert src_offset is None, "a pair-mode key takes the pair, not an offset"
        d, s = pair
        return (int(layer), int(index), PAIR_RANGE_TAG, int(d), int(s))
    if src_offset is None:
        return (int(layer), int(index))
    return (int(layer), int(index), int(src_offset))


class _PortHandle:
    """Mimics a pytorch hook handle so ports and forward hooks share one cleanup path.

    Removal must be unconditional (the caller does it in `finally`): a port left attached
    would silently intervene on the NEXT forward, which is how an endogenous evaluation
    would end up secretly interventional.
    """

    def __init__(self, module, attr=PORT_ATTR):
        self.module = module
        self.attr = attr          # which port site to clear; defaults to the logit port

    def remove(self):
        setattr(self.module, self.attr, None)


class InterventionHook:
    # TODO extend to other types of signals
    def __init__(self, token_positions, signal_indices, intervention_values, row_indices=None):
        """
        row_indices:         [k]  WHICH rows of the batch this hook injects into
        token_positions:     [k]
        signal_indices:      [k]
        intervention_values: [k]

        k need not equal the batch size. The forward registers one hook per layer
        present in the batch and hands each hook only ITS rows' positions/indices/values,
        so a batch mixing layers gives every hook a strict subset. The hook used to index
        rows with arange(batch_size) regardless -- correct only when every row targets
        the same layer, which every run before the layer sweep happened to satisfy. The
        first multi-layer batch failed with a [32] vs [k] broadcast error at this line.

        row_indices=None keeps the old all-rows behaviour for any single-layer caller.
        """
        self.row_indices = row_indices
        self.token_positions = token_positions
        self.neuron_indices = signal_indices
        self.intervention_values = intervention_values

    def __call__(self, module, inputs, outputs):
        # outputs shape:
        # [bs, seq, hidden]
        outputs = outputs.clone()
        if self.row_indices is None:
            batch_idx = torch.arange(outputs.shape[0], device=outputs.device)
        else:
            batch_idx = self.row_indices.to(outputs.device)
        if not (batch_idx.shape[0] == self.token_positions.shape[0]
                == self.neuron_indices.shape[0] == self.intervention_values.shape[0]):
            raise ValueError(
                f"InterventionHook: rows {tuple(batch_idx.shape)}, positions "
                f"{tuple(self.token_positions.shape)}, indices {tuple(self.neuron_indices.shape)}, "
                f"values {tuple(self.intervention_values.shape)} must all have the same length "
                f"-- a multi-layer batch needs row_indices for the rows at THIS layer")

        outputs[
            batch_idx,
            self.token_positions,
            self.neuron_indices,
        ] = self.intervention_values.to(outputs.dtype)

        return outputs


class SAEInterventionHook:
    """Forward hook on a decoder LAYER (the residual site a Gemma Scope res SAE was
    trained on) that clamps ONE SAE feature to a target value at given token
    positions, via decoder-direction steering (bundle.steer_delta):

        resid[b, pos] += decode(feat[b,pos] with feature:=target) - decode(feat[b,pos])

    computed as a difference so the SAE reconstruction error is preserved and only
    feature `feature_index`'s contribution changes. Used for the interventional
    faithfulness eval: force the feature on vs off on the same context and see
    whether the introspector's report follows.

    token_positions/target_values: [bs] tensors (positions are absolute, i.e.
    already padding-adjusted). feature_index: one int for the whole batch, or a [bs]
    tensor giving each row its own feature (multi-feature runs). Handles decoder
    layers whose forward returns either a bare tensor or a (hidden, ...) tuple.
    """

    def __init__(self, bundle, token_positions, feature_index, target_values):
        self.bundle = bundle
        self.token_positions = token_positions
        self.feature_index = feature_index
        self.target_values = target_values

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output          # [bs, seq, d]
        hidden = hidden.clone()
        bs = hidden.shape[0]
        bidx = torch.arange(bs, device=hidden.device)
        resid_at = hidden[bidx, self.token_positions]        # [bs, d]
        delta = self.bundle.steer_delta(resid_at, self.feature_index, self.target_values)
        hidden[bidx, self.token_positions] = resid_at + delta.to(hidden.dtype)
        if is_tuple:
            return (hidden,) + tuple(output[1:])
        return hidden


# Guard band for signals that live in (0, 1), i.e. post-softmax attention weights.
# It was chosen LOOSER than the port's old target band (EPS, 1-EPS), which a draw on
# the edge would have hit at a random step; the port now accepts all of [0, 1] (see
# AttentionWeightPort.__call__), but the value stays so every log-odds run keeps drawing
# the same distribution it always did. `logit_moments` and the sampler share it
# so the empirical range a `uniform` draw spans is a range the sampler can actually
# reach -- with two different epsilons, uniform would clamp a third of its draws while
# reporting itself as uniform.
PROB_SIGNAL_EPS = 1e-4


def logit_moments(values, eps=PROB_SIGNAL_EPS):
    """Log-odds summary of a probability sample, for the bounded sampler below.

    `values` is an iterable of probabilities. Returns the extra `signal_ranges` keys
    (`mean_logit`, `std_logit`, `min_logit`, `max_logit`) that
    `make_bounded_prob_intervention_fn` reads. Kept next to its consumer so the two
    cannot drift apart about names.

    Clipping to [eps, 1-eps] before the log is what keeps a cell whose weight underflows
    to exactly 0 (or saturates to 1) from producing an infinite moment that would then
    poison every draw for that cell.
    """
    vals = [min(max(float(v), eps), 1.0 - eps) for v in values]
    assert vals, "logit_moments got an empty sample"
    ls = [math.log(v / (1.0 - v)) for v in vals]
    mean = sum(ls) / len(ls)
    var = sum((x - mean) ** 2 for x in ls) / max(len(ls) - 1, 1)
    return {"mean_logit": mean, "std_logit": math.sqrt(var),
            "min_logit": min(ls), "max_logit": max(ls)}


# Guard band for the `empirical_logit` EVAL sampler only. Looser than PROB_SIGNAL_EPS
# because its draws come from the cell's own histogram, not from a widened range, so a
# clamp there would only cut real mass (on L2 h26 @-27, 34% of the natural weights sit
# below 1e-4). Kept at 1e-5 when the port's target band was dropped, for the same
# reason PROB_SIGNAL_EPS was: existing empirical_logit numbers stay reproducible.
EMPIRICAL_EPS = 1e-5


def logit_histogram(values, n_bins=100, eps=EMPIRICAL_EPS):
    """Equal-width log-odds histogram of a probability sample, for `empirical_logit`.

    Returns the extra `signal_ranges` keys `logit_hist_edges` ([n_bins+1]) and
    `logit_hist_cdf` ([n_bins], last entry 1). Kept next to `logit_moments` and the
    sampler for the same reason: the key names live in one place.
    """
    # Bounds go through the SAME expression as the data. They used to be written as
    # log((1-eps)/eps), but a value clamped to 1-eps maps to log((1-eps)/(1-(1-eps))), and
    # 1-(1-eps) != eps in floating point: the clamped top value came out 4.5e-12 ABOVE the
    # bound and the assert below killed every run whose cell has a weight >= 1-eps (BOS at
    # an early destination, 7-1 tasks 45/52/54) -- before training, on a value only the
    # empirical_logit eval reads.
    def _logit(v):
        return math.log(v / (1.0 - v))
    lo_l, hi_l = _logit(eps), _logit(1.0 - eps)
    ls = [_logit(v) for v in (min(max(float(v), eps), 1.0 - eps) for v in values)]
    assert ls, "logit_histogram got an empty sample"
    lo, hi = min(ls), max(ls)
    assert lo_l <= lo <= hi <= hi_l, f"logits [{lo}, {hi}] outside the clip [{lo_l}, {hi_l}]"
    if hi - lo < 1e-9:           # a constant cell: one bin of width ~0 around the value
        lo, hi = lo - 1e-6, hi + 1e-6
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for x in ls:
        counts[min(int((x - lo) / width), n_bins - 1)] += 1
    edges = [lo + i * width for i in range(n_bins)] + [hi]
    cdf, run = [], 0
    for c in counts:
        run += c
        cdf.append(run / len(ls))
    assert len(edges) == n_bins + 1 and len(cdf) == n_bins, (
        f"edges {len(edges)} != {n_bins + 1} or cdf {len(cdf)} != {n_bins}")
    assert cdf[-1] == 1.0, f"cdf ends at {cdf[-1]}, not 1"
    return {"logit_hist_edges": edges, "logit_hist_cdf": cdf}


def empirical_prob_values(values):
    """The train split's own probabilities, for the `empirical_prob` sampler.

    Returns the extra `signal_ranges` key `train_values` (a list of floats). Kept next
    to `logit_histogram` and the sampler for the same reason: the key name lives in one
    place. The sampler draws from this list as-is, so the [0, 1] check here is the ONLY
    range guarantee its draws get -- there is no clamp behind it.
    """
    vals = [float(v) for v in values]
    assert vals, "empirical_prob_values got an empty sample"
    lo, hi = min(vals), max(vals)
    assert 0.0 <= lo and hi <= 1.0, (
        f"empirical_prob needs probabilities, got range [{lo}, {hi}] -- the cache holds "
        f"something other than post-softmax weights")
    return {"train_values": vals}


def make_bounded_prob_intervention_fn(model, eps=PROB_SIGNAL_EPS):
    """Intervention sampler for a signal that lives in (0, 1): a post-softmax weight.

    All three built-in distributions leave the domain. `gaussian` on a cell with mean
    ~1e-3 and std ~1e-4 draws negatives about half the time; `extreme` draws uniformly
    over `scale x` the empirical RANGE, which is not a probability at any scale. A
    negative target makes AttentionWeightPort's renormalization build a row with negative
    probabilities, and the softmax is already behind us, so nothing downstream raises.

    This samples in LOG-ODDS space and maps back with a sigmoid, so every draw is a valid
    probability at every scale by construction rather than by rejection. Rejection
    sampling degenerates here: at a large scale on a mean-1e-3 cell nearly every draw is
    rejected, so `extreme` would quietly collapse to ~U(eps, 1-eps) and the scale sweep
    would stop meaning anything.

    Log-odds is also the RIGHT space, not merely a safe one. Renormalizing the row to put
    t at the target cell is exactly a shift of that cell's attention logit z_j by
    logit(t) - logit(p_j) (see AttentionWeightPort), so a gaussian on logit(t) is the
    same FAMILY of intervention as the attention-logit experiment's gaussian on z_j.
    That is what makes the A and P interventional numbers comparable.

    **The closure MUST read `intervention_distribution` and `intervention_scale` off the
    model on every call, never capture them.** `sample_intervention_values` consults
    `intervention_fn` BEFORE its own distribution branches, so a closure that froze them
    would silently ignore `evaluate_interventional_test`'s "gaussian at scale 1.0" and
    every step of `interventional_scale_sweep` -- each scale row would be the same
    distribution, the tables would fill in normally, and nothing would raise.

    The log-odds branches clamp to PROB_SIGNAL_EPS (EMPIRICAL_EPS for empirical_logit);
    `empirical_prob` draws real probabilities and does not clamp. See the comment on
    PROB_SIGNAL_EPS for how these relate to the port, which accepts all of [0, 1].

    A probability is bounded, so at a large enough scale draws DO pile up on the guard
    band -- the same point mass that makes clamping in probability space wrong, only
    pushed out to where it takes an extreme scale to reach. It is not hidden: `sample`
    carries `n_drawn` / `n_clamped`, so `interventional_scale_sweep` can report the
    saturated fraction and a scale whose corr is really "the model learned the clamp"
    is visible rather than inferred.
    """
    lo, hi = math.log(eps / (1 - eps)), math.log((1 - eps) / eps)
    emp_lo, emp_hi = (math.log(EMPIRICAL_EPS / (1 - EMPIRICAL_EPS)),
                      math.log((1 - EMPIRICAL_EPS) / EMPIRICAL_EPS))

    def sample(layer, idx, **stats):
        missing = [k for k in ("mean_logit", "std_logit", "min_logit", "max_logit")
                   if k not in stats]
        assert not missing, (
            f"signal_ranges for layer {layer} index {idx} lacks {missing}; an "
            f"attention_weight run must add them with logit_moments()")
        rng = model.intervention_rng          # read live: seed_intervention_rng rebinds it
        dist = model.intervention_distribution
        scale = model.intervention_scale
        if dist == "prob_uniform":
            # Uniform in PROBABILITY space, the one branch that does not draw in log-odds.
            # The log-odds branches at a wide scale pile most draws on the guard band
            # (both ends of the clamp), so the regression targets collapse onto ~0 and ~1;
            # this spreads them over (0, 1) instead and clamps only a 2*eps sliver.
            # It has no spread to widen, so a scale other than 1 means a caller expected
            # a scale sweep to change something here -- it would not.
            assert scale == 1.0, f"prob_uniform has no scale, got intervention_scale={scale}"
            t = rng.uniform(0.0, 1.0)
            sample.n_drawn += 1
            if t < eps or t > 1 - eps:
                sample.n_clamped += 1
                t = min(max(t, eps), 1 - eps)
            return t
        if dist == "empirical_logit":
            # The cell's OWN log-odds distribution (train-split histogram, see
            # logit_histogram), for evaluation. The gaussian branch assumes the log-odds
            # are normal; on a cell like L2 h26 @-27 a third of them sit below the 1e-4
            # guard band and get clamped. Pick a bin by frequency, then uniform within it.
            assert scale == 1.0, f"empirical_logit has no scale, got intervention_scale={scale}"
            edges, cdf = stats.get("logit_hist_edges"), stats.get("logit_hist_cdf")
            assert edges is not None and cdf is not None, (
                f"signal_ranges for layer {layer} index {idx} lacks the logit histogram; "
                f"add it with logit_histogram()")
            b = min(bisect.bisect_left(cdf, rng.random()), len(cdf) - 1)
            u = rng.uniform(edges[b], edges[b + 1])
            sample.n_drawn += 1
            if u < emp_lo or u > emp_hi:
                sample.n_clamped += 1
                u = min(max(u, emp_lo), emp_hi)
            return 1.0 / (1.0 + math.exp(-u))
        if dist == "empirical_prob":
            # The cell's OWN probabilities: one train-split value, drawn uniformly
            # (bootstrap). Unlike every branch above, nothing here goes through log-odds
            # moments or a histogram, so the injected distribution IS the endogenous one.
            # No clamp: the values come from the cache, which empirical_prob_values
            # checked to lie in [0, 1], and AttentionWeightPort accepts all of [0, 1].
            # A guard band here would move real mass -- 0.7% of L14 h26 @-19's train
            # weights sit below 1e-6.
            assert scale == 1.0, f"empirical_prob has no scale, got intervention_scale={scale}"
            vals = stats.get("train_values")
            assert vals is not None, (
                f"signal_ranges for layer {layer} index {idx} lacks train_values; "
                f"add them with empirical_prob_values()")
            sample.n_drawn += 1
            return vals[rng.randrange(len(vals))]
        if dist == "gaussian":
            u = rng.gauss(stats["mean_logit"], scale * stats["std_logit"])
        elif dist == "uniform":
            u = rng.uniform(stats["min_logit"], stats["max_logit"])
        elif dist == "extreme":
            center = (stats["min_logit"] + stats["max_logit"]) / 2
            half = scale * (stats["max_logit"] - stats["min_logit"]) / 2
            u = rng.uniform(center - half, center + half)
        else:
            raise ValueError(f"Unknown intervention_distribution: {dist!r}")
        sample.n_drawn += 1
        if u < lo or u > hi:
            sample.n_clamped += 1
            u = min(max(u, lo), hi)
        return 1.0 / (1.0 + math.exp(-u))

    def reset_counts():
        sample.n_drawn = 0
        sample.n_clamped = 0

    sample.reset_counts = reset_counts
    sample.eps = eps
    reset_counts()
    return sample


class SigmoidRegressionHead(nn.Linear):
    """Logistic probe: sigmoid(W h + b), for targets that are probabilities.

    A subclass of nn.Linear so every caller that does `head(h)` or reads
    `head.weight.device/dtype` (the evals, the scale sweep, the attribution scripts)
    gets the sigmoid without being edited.

    The `_sigmoid_marker` buffer is what keeps a checkpoint honest about its head: it
    goes into state_dict, so load_regression_head's STRICT load fails when a sigmoid
    head is loaded into a plain nn.Linear or the other way round. Without it the weights
    have identical keys and shapes, and a loader that does not know the config (e.g.
    attribute_attention_weight_readout.load_everything) would silently drop the sigmoid
    and report logits as probabilities.
    """

    def __init__(self, in_features, out_features=1, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("_sigmoid_marker", torch.ones(1))

    def pre_activation(self, x):
        """z = W h + b, the log-odds before the sigmoid. A loss on probabilities takes this
        rather than torch.logit(forward(x)): fp32 sigmoid rounds to exactly 1.0 above
        z ~ 17, and the logit of that is inf (attention weights at offset -1 reach ~1)."""
        return super().forward(x)

    def forward(self, x):
        return torch.sigmoid(self.pre_activation(x))


HEAD_ACTIVATIONS = ("identity", "sigmoid")


class IntrospectionRegressionModel(PeftModelForCausalLM):

    def __init__(
        self,
        model,
        signal_ranges,
        intervention_distribution="uniform",
        intervention_extreme_scale=EXTREME_SCALE,
        intervention_fn=None,
        attention_site=None,
        head_activation="identity",
    ):
        # IMPORTANT: do NOT call super().__init__
        self.__dict__ = model.__dict__
        if head_activation not in HEAD_ACTIVATIONS:
            raise ValueError(f"head_activation {head_activation!r} not in {HEAD_ACTIVATIONS}")
        # Keep the head in fp32 for stable MSE regression, but move it onto the
        # base model's device so it does not silently sit on CPU when the base
        # model is dispatched with device_map / loaded in bf16 (which would
        # otherwise throw a device/dtype mismatch at the head matmul).
        head_cls = SigmoidRegressionHead if head_activation == "sigmoid" else nn.Linear
        self.regression_head = head_cls(self.config.hidden_size, 1, bias=True)
        _base_param = next(model.parameters())
        self.regression_head = self.regression_head.to(device=_base_param.device)
        print(
            f"[IntrospectionRegressionModel] base model dtype={_base_param.dtype}, "
            f"device={_base_param.device} | regression_head dtype="
            f"{self.regression_head.weight.dtype}, device={self.regression_head.weight.device}"
        )
        self._printed_forward_dtypes = False
        # {(layer, idx): {...}} for mlp/sae, {(layer, head, src_offset): {...}} for
        # attention (see signal_range_key).
        self.signal_ranges = signal_ranges
        key_lens = {len(k) for k in (signal_ranges or {})}
        assert len(key_lens) <= 1, f"mixed signal_ranges key shapes: {key_lens}"
        # Which intervention site this run uses. Kept so forward can REFUSE to inject an
        # attention signal's value into an MLP neuron: the two paths differ only by
        # whether src_token_position was passed, and a caller that forgot it would
        # otherwise write into `mlp.act_fn` at index `head` -- a real neuron, a plausible
        # number, and a completely wrong experiment.
        self._attention_signals = key_lens in ({3}, {5})
        # PAIR MODE: ranges are keyed per absolute (dst, src), see signal_range_key
        self._pair_ranges = key_lens == {5}
        # WHICH attention site, for runs that have one. The bool above cannot tell A from
        # P -- both key signal_ranges by 3-tuples -- and writing a probability into the
        # pre-softmax tensor (or a logit into the post-softmax one) raises nothing and
        # produces a plausible number. Defaults to "logit" on an attention run so every
        # caller written before P existed keeps its exact behaviour.
        assert attention_site in (None, "logit", "weight"), (
            f"attention_site must be None/'logit'/'weight', got {attention_site!r}")
        self._attention_site = attention_site or ("logit" if self._attention_signals else None)
        self.intervention_distribution = intervention_distribution
        self.intervention_extreme_scale = intervention_extreme_scale
        self.intervention_fn = intervention_fn  # optional custom intervention sampler
        # single annealable spread multiplier used by every distribution
        # (gaussian: std*scale, extreme: range*scale). Initialized to preserve the
        # non-annealed behavior: extreme_scale for extreme, 1.0 (endogenous) else.
        self.intervention_scale = (intervention_extreme_scale
                                   if intervention_distribution == "extreme" else 1.0)
        # dedicated RNG for intervention sampling so eval passes can be made
        # reproducible independently of training-time draws (see seed_intervention_rng)
        self.intervention_rng = random.Random()

    def seed_intervention_rng(self, seed):
        """Reset the intervention sampler to a fixed seed so a subsequent eval
        pass draws a reproducible sequence of intervention values. Call right
        before an evaluation loop."""
        self.intervention_rng = random.Random(seed)

    def set_intervention_scale(self, scale):
        """Set the current intervention spread multiplier (annealing hook)."""
        self.intervention_scale = float(scale)

    def sample_intervention_values(self, signal_layers, signal_indices, device,
                                   signal_src_offsets=None, dst_positions=None,
                                   src_positions=None):
        """`dst_positions`/`src_positions` (PAIR MODE only) are positions in the UNPADDED
        text, i.e. before forward's left-pad shift -- the coordinates the ranges are
        keyed in. Passing shifted ones would look up a different pair, or none."""
        vals = []
        # An attention signal is only identified once the source offset is included:
        # (layer, head) alone names ten different signals in a cell, whose ranges differ
        # by an order of magnitude (the BOS sink against a mid-context token). Keying
        # without it would sample every offset's intervention from one pooled range.
        n = len(signal_layers)
        if self._pair_ranges:
            assert dst_positions is not None and src_positions is not None, (
                "pair-mode signal_ranges need per-row dst/src positions")
            assert len(dst_positions) == n and len(src_positions) == n, (
                f"{len(dst_positions)} dst / {len(src_positions)} src for {n} rows")
            keys = [signal_range_key(l, i, pair=(d, s_)) for l, i, d, s_ in zip(
                signal_layers.tolist(), signal_indices.tolist(),
                dst_positions.tolist(), src_positions.tolist())]
        else:
            offsets = (signal_src_offsets.tolist() if signal_src_offsets is not None
                       else [None] * n)
            keys = [signal_range_key(l, i, off) for l, i, off in zip(
                signal_layers.tolist(), signal_indices.tolist(), offsets)]
        for key in keys:
            layer, idx = key[0], key[1]
            if key not in self.signal_ranges:
                raise KeyError(
                    f"no intervention range for {key}; signal_ranges holds "
                    f"{len(self.signal_ranges)} keys, e.g. {list(self.signal_ranges)[:3]}")
            stats = self.signal_ranges[key]
            if self.intervention_fn is not None:
                val = self.intervention_fn(layer=layer, idx=idx, **stats)
            elif self.intervention_distribution == "gaussian":
                # gaussian fit to the neuron's empirical activations; intervention_scale
                # widens the std (=1 -> exactly endogenous, the annealing target).
                val = self.intervention_rng.gauss(stats["mean"], self.intervention_scale * stats["std"])
            elif self.intervention_distribution == "uniform":
                val = self.intervention_rng.uniform(stats["min"], stats["max"])
            elif self.intervention_distribution == "extreme":
                # uniform over a range intervention_scale x the empirical range,
                # centered on the empirical midpoint (=1 -> natural range).
                center = (stats["min"] + stats["max"]) / 2
                half_width = self.intervention_scale * (stats["max"] - stats["min"]) / 2
                val = self.intervention_rng.uniform(center - half_width, center + half_width)
            else:
                raise ValueError(
                    f"Unknown intervention_distribution: {self.intervention_distribution!r}"
                )
            vals.append(val)

        return torch.tensor(vals,device=device,dtype=torch.float32)

    def forward(
        self,
        *args,
        token_position=None, # intervention inputs
        signal_layer=None,
        signal_index=None,
        src_token_position=None,  # attention signals: SOURCE token (token_position is dst)
        signal_src_offset=None,   # attention signals: part of the signal identity
        regression_token_positions=-1, # regression
        labels=None, # labels for regression are scalar and shifted tokens otherwise
        do_intervention=True,
        output_hidden_states=True,
        **kwargs,
    ):
        # avoid mutating HF internals
        kwargs = kwargs.copy()

        hooks = []
        intervention_values = None
        try:
            # REGISTER INTERVENTION HOOKS
            if (
                do_intervention
                and token_position is not None
                and signal_layer is not None
                and signal_index is not None
            ):
                bs = token_position.shape[0]
                device = token_position.device

                # token_position is relative to the UNPADDED text (text starts at
                # index 0). The regressor batches are LEFT-padded, so the text
                # actually starts at index num_left_pad. Shift token_position by the
                # leading pad count so the hook injects into the intended token
                # rather than a padding/wrong token. (Assumes left padding: with no
                # right padding, (attention_mask == 0).sum == leading pad count.)
                # PAIR MODE ranges are keyed in UNPADDED coordinates; keep those
                # before the shift below moves them.
                dst_unpadded, src_unpadded = token_position, src_token_position
                attention_mask = kwargs.get("attention_mask")
                if attention_mask is not None:
                    # The comment above only holds under LEFT padding. Assert it instead
                    # of trusting it: LoRAGatingCollator switches padding side only when
                    # for_prediction=True, and under right padding this shift would move
                    # every position by the WRONG amount without raising.
                    assert bool((attention_mask[:, -1] == 1).all()), (
                        "intervention assumes LEFT padding (no zeros at the end of "
                        "attention_mask), but some rows are right-padded")
                    num_left_pad = (attention_mask == 0).sum(dim=1).to(token_position.device)
                    token_position = token_position + num_left_pad
                    if src_token_position is not None:
                        src_token_position = src_token_position + num_left_pad.to(
                            src_token_position.device)

                # sample interventions INSIDE forward
                intervention_values = self.sample_intervention_values(
                    signal_layer, signal_index, device, signal_src_offset,
                    dst_positions=dst_unpadded, src_positions=src_unpadded)
                attention_batch = src_token_position is not None
                assert attention_batch == self._attention_signals, (
                    f"intervention site mismatch: signal_ranges says "
                    f"attention={self._attention_signals} but the batch "
                    f"{'carries' if attention_batch else 'lacks'} src_token_position. "
                    f"A caller building forward kwargs by hand must pass "
                    f"src_token_position and signal_src_offset for attention signals.")
                unique_layers = torch.unique(signal_layer)
                for layer in unique_layers.tolist():
                    mask = signal_layer == layer
                    if attention_batch:
                        # An attention logit/weight is not any module's output, so it
                        # cannot be written by a forward hook. The port is read/written
                        # inside the custom attention function; see
                        # models/attention_logits.py. WHICH port and which attribute is
                        # decided by the run's signal type, not guessed: the weight port
                        # renormalizes the row, the logit port overwrites one cell, and
                        # using the wrong one is silent.
                        site = self._attention_site
                        port_cls = AttentionWeightPort if site == "weight" else AttentionLogitPort
                        attr = PORT_ATTRS[site or "logit"]
                        port = port_cls(
                            heads=signal_index[mask],
                            dst=token_position[mask],
                            src=src_token_position[mask],
                            values=intervention_values[mask],
                        )
                        module = resolve_attention_module(self, layer)
                        assert getattr(module, attr, None) is None, (
                            f"layer {layer} already carries a {site!r} port; a previous "
                            f"forward leaked one and this batch would be intervened twice")
                        setattr(module, attr, port)
                        hooks.append(_PortHandle(module, attr))
                        continue

                    # the hook must know WHICH rows it owns, not just how many: with
                    # several layers in one batch, each hook gets a subset of the rows
                    hook = InterventionHook(
                        row_indices=mask.nonzero(as_tuple=True)[0],
                        token_positions=token_position[mask],
                        signal_indices=signal_index[mask],
                        intervention_values=intervention_values[mask],
                    )

                    module_path = resolve_mlp_signal_path(self,layer,path_to_output=False,)
                    module = operator.attrgetter(module_path)(self)
                    handle =  module.register_forward_hook(hook)
                    hooks.append(handle)
                if attention_batch:
                    assert_attention_impl(self)

                # regression target = intervention value
                if labels is not None and labels.ndim == 1:
                    labels = intervention_values

            # BASE MODEL FORWARD
            outputs = super().forward(
                *args,
                output_hidden_states=True,
                **kwargs,
            )

            # always expose the sampled intervention values (None if not
            # intervening) so downstream eval can use them as targets even when
            # no regression labels are passed
            outputs["intervention_values"] = intervention_values

            # REGRESSION MODE
            if labels is not None and labels.ndim == 1:
                hidden = outputs.hidden_states[-1]
                bs = hidden.shape[0]
                batch_idx = torch.arange(bs,device=hidden.device)
                selected_hidden = hidden[batch_idx,regression_token_positions]
                head_device = self.regression_head.weight.device
                head_dtype = self.regression_head.weight.dtype
                if not self._printed_forward_dtypes:
                    print(
                        f"[IntrospectionRegressionModel.forward] hidden dtype="
                        f"{selected_hidden.dtype}, device={selected_hidden.device} -> "
                        f"head dtype={head_dtype}, device={head_device}; "
                        f"do_intervention={do_intervention}, labels dtype={labels.dtype}"
                    )
                    self._printed_forward_dtypes = True
                # cast hidden to the head's device/dtype so a bf16/device-mapped
                # base model does not mismatch the fp32 head
                selected_hidden = selected_hidden.to(device=head_device, dtype=head_dtype)
                if isinstance(self.regression_head, SigmoidRegressionHead):
                    z = self.regression_head.pre_activation(selected_hidden).squeeze(-1)  # [B]
                    pred = torch.sigmoid(z)                                               # [B]
                else:
                    z = None
                    pred = self.regression_head(selected_hidden).squeeze(-1)              # [B]
                assert pred.shape == (bs,), f"pred {tuple(pred.shape)} != ({bs},)"
                regression_labels = labels.to(pred.dtype)
                loss = nn.functional.mse_loss(pred,regression_labels,)
                # Trainer expects logits
                outputs.logits = pred
                outputs.loss = loss
                # the head's log-odds, for losses on probabilities (None for a linear head)
                outputs["regression_pre_activation"] = z

            return outputs

        finally:
            # ALWAYS cleanup hooks
            for h in hooks:
                h.remove()


# ============================================================
# Regression head persistence
# ============================================================
class RegressionHeadSaveCallback(TrainerCallback):
    """
    PEFT's save_pretrained only serializes the adapter (+ modules_to_save)
    weights, so the custom regression head is otherwise silently dropped on
    save and re-initialized randomly on reload. This callback writes the head's
    state_dict next to every checkpoint the Trainer saves.
    """

    def __init__(self, model):
        self.model = model

    def on_save(self, args, state, control, **kwargs):
        if not hasattr(self.model, "regression_head"):
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, REGRESSION_HEAD_FILENAME)
        torch.save(self.model.regression_head.state_dict(), path)
        print(f"[RegressionHeadSaveCallback] saved regression head -> {path}")


def anneal_value(step, max_steps, start, end, start_frac=0.0, end_frac=1.0, schedule="log"):
    """
    Interpolate a scalar from `start` -> `end` over the training window
    [start_frac, end_frac] (as fractions of max_steps). Before the window it is
    `start`, after it is `end`. Default schedule is 'log' (geometric), the right
    choice when start/end span orders of magnitude; also supports 'linear'/'cosine'.
    """
    if max_steps is None or max_steps <= 0:
        return start
    frac = step / max_steps
    if frac <= start_frac:
        p = 0.0
    elif frac >= end_frac:
        p = 1.0
    else:
        p = (frac - start_frac) / max(end_frac - start_frac, 1e-8)
    if schedule == "linear":
        return start + (end - start) * p
    if schedule == "cosine":
        return end + (start - end) * 0.5 * (1 + math.cos(math.pi * p))
    # 'log' / geometric (default)
    if start <= 0 or end <= 0:
        return start + (end - start) * p
    return start * (end / start) ** p


class InterventionAnnealCallback(TrainerCallback):
    """
    Anneal the intervention distribution's spread (model.intervention_scale) from
    `scale_start` down to `scale_end` over training. Training starts on an easy,
    large injected signal and ends on the endogenous-matched distribution
    (scale ~1) the model must generalize to at test time.
    """

    def __init__(self, model, scale_start, scale_end, schedule="log",
                 start_frac=0.0, end_frac=1.0, log_every=0):
        self.model = model
        self.scale_start = scale_start
        self.scale_end = scale_end
        self.schedule = schedule
        self.start_frac = start_frac
        self.end_frac = end_frac
        self.log_every = log_every

    def _set(self, state):
        scale = anneal_value(state.global_step, state.max_steps,
                             self.scale_start, self.scale_end,
                             self.start_frac, self.end_frac, self.schedule)
        self.model.set_intervention_scale(scale)
        return scale

    def on_train_begin(self, args, state, control, **kwargs):
        scale = self._set(state)
        print(f"[anneal] start intervention_scale={scale:.4g} "
              f"({self.scale_start:g} -> {self.scale_end:g}, schedule={self.schedule}, "
              f"window=[{self.start_frac},{self.end_frac}])")

    def on_step_begin(self, args, state, control, **kwargs):
        scale = self._set(state)
        if self.log_every and state.global_step % self.log_every == 0:
            print(f"[anneal] step {state.global_step}/{state.max_steps} intervention_scale={scale:.4g}")


ADDED_TOKEN_EMBEDDINGS_FILENAME = "added_token_embeddings.pt"
EMBED_TENSOR_SUFFIXES = ("embed_tokens.weight", "lm_head.weight")


def strip_adapter_embeddings(ckpt_dir, n_added_tokens: int, verbose: bool = True):
    """
    Rewrite a saved adapter without the frozen embedding matrices.

    PEFT's save_pretrained(save_embedding_layers="auto") stores the WHOLE resized
    embed_tokens and lm_head whenever the vocab grew -- and every run here grows it by
    one for the introspection separator. At 128257 x 2048 in bf16 that is ~1.0 GB per
    checkpoint against ~21 MB of LoRA weights: 98% of the file is a frozen copy of the
    base model.

    Only the last `n_added_tokens` rows are genuinely not in the base model. They were
    random at resize time and frozen through training, but the model trained WITH those
    vectors and a fresh resize initializes them differently -- so they are preserved
    (a few KB) and restored by load_added_token_embeddings.

    n_added_tokens is passed explicitly rather than inferred from a vocab constant, so
    this works for any base model (Llama 128256, Gemma 256000, ...).
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    path = os.path.join(str(ckpt_dir), "adapter_model.safetensors")
    if not os.path.exists(path):
        return False
    before = os.path.getsize(path)
    keep, rows, had_embed = {}, {}, False
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
        for key in f.keys():
            tensor = f.get_tensor(key)
            if any(key.endswith(sfx) for sfx in EMBED_TENSOR_SUFFIXES):
                had_embed = True
                if n_added_tokens > 0 and tensor.shape[0] > n_added_tokens:
                    rows[key] = tensor[-n_added_tokens:].clone()
            else:
                keep[key] = tensor
    if not had_embed:
        return False              # already stripped, or never carried embeddings
    save_file(keep, path, metadata=meta)
    if rows:
        torch.save(rows, os.path.join(str(ckpt_dir), ADDED_TOKEN_EMBEDDINGS_FILENAME))
    if verbose:
        after = os.path.getsize(path)
        print(f"[strip_adapter_embeddings] {path}: {before/2**20:.1f}MB -> "
              f"{after/2**20:.1f}MB (kept {len(rows)} added-token row set(s))")
    return True


class StripEmbeddingSaveCallback(TrainerCallback):
    """
    Drop the frozen embedding matrices from each checkpoint as it is written.

    Cheaper and safer than cleaning up afterwards: the 1 GB per checkpoint is never
    on disk in the first place, so a sweep cannot fill the filesystem mid-run. Pass
    the number of tokens added to the tokenizer (tokenizer.add_special_tokens returns
    it); 0 disables the strip.
    """

    def __init__(self, n_added_tokens: int):
        self.n_added_tokens = n_added_tokens

    def on_save(self, args, state, control, **kwargs):
        if self.n_added_tokens is None:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        try:
            strip_adapter_embeddings(ckpt_dir, self.n_added_tokens)
        except Exception as e:      # never let disk hygiene kill a training run
            print(f"[StripEmbeddingSaveCallback] skipped ({type(e).__name__}: {e})")


def load_added_token_embeddings(model, ckpt_dir):
    """
    Restore the rows beyond the base vocabulary (the introspection separator) after a
    checkpoint has had its frozen embedding matrices stripped by
    utils/delete_checkpoints_batch.strip_embedding_tensors.

    PEFT saves the WHOLE resized embed_tokens/lm_head (~1.0 GB) because the vocab grew
    by one, even though they are frozen. Stripping them saves 97.9% of the checkpoint,
    but the separator's row is genuinely not in the base model: it was random at resize
    time, frozen through training, and a fresh resize initializes it differently. This
    puts the trained-time row back.

    No-op when the file is absent (checkpoint still carries its full embeddings).
    Returns True if rows were restored.
    """
    path = os.path.join(ckpt_dir, ADDED_TOKEN_EMBEDDINGS_FILENAME)
    if not os.path.exists(path):
        return False
    rows = torch.load(path, map_location="cpu")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    restored = []
    for key, tensor in rows.items():
        module = base.get_input_embeddings() if "embed_tokens" in key else base.get_output_embeddings()
        if module is None:
            continue
        w = module.weight
        n = tensor.shape[0]
        if w.shape[0] < n:
            raise ValueError(f"{key}: model vocab {w.shape[0]} is smaller than the {n} "
                             f"saved extra row(s); resize the base model before loading.")
        with torch.no_grad():
            w[-n:] = tensor.to(device=w.device, dtype=w.dtype)
        restored.append(key)
    print(f"[load_added_token_embeddings] restored {restored} from {path}")
    return bool(restored)


def load_regression_head(model, ckpt_dir):
    """
    Restore a regression head previously saved by RegressionHeadSaveCallback.
    Returns True if a head was found & loaded, False otherwise (leaving the
    freshly-initialized head in place).
    """
    head_path = os.path.join(ckpt_dir, REGRESSION_HEAD_FILENAME)
    if not os.path.exists(head_path):
        print(f"[load_regression_head] WARNING: no regression head at {head_path}. "
              f"Evaluating with a freshly-initialized (UNTRAINED) head. Re-train with "
              f"the updated code (e.g. --overwrite-cache) to persist it.")
        return False
    device = model.regression_head.weight.device
    state_dict = torch.load(head_path, map_location="cpu")
    model.regression_head.load_state_dict(state_dict)
    model.regression_head.to(device=device)
    print(f"[load_regression_head] loaded regression head from {head_path}")
    return True
