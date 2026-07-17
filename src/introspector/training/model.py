import os
import pdb
import random
import operator
from typing import Any, Optional

import torch
from torch import nn

from peft import PeftModelForCausalLM
from peft.tuners.lora.layer import Linear
from transformers import TrainerCallback

from src.introspector.models.signal_paths import resolve_mlp_signal_path

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
            x_cast = x.to(lora_A.weight.dtype)
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
class InterventionHook:
    # TODO extend to other types of signals
    def __init__(self, token_positions, signal_indices, intervention_values):
        """
        token_positions: [bs]
        signal_indices: [bs]
        intervention_values: [bs]
        """
        self.token_positions = token_positions
        self.neuron_indices = signal_indices
        self.intervention_values = intervention_values

    def __call__(self, module, inputs, outputs):
        # outputs shape:
        # [bs, seq, hidden]
        outputs = outputs.clone()
        bs = outputs.shape[0]
        batch_idx = torch.arange(bs, device=outputs.device)

        outputs[
            batch_idx,
            self.token_positions,
            self.neuron_indices,
        ] = self.intervention_values.to(outputs.dtype)

        return outputs

class IntrospectionRegressionModel(PeftModelForCausalLM):

    def __init__(
        self,
        model,
        signal_ranges,
        intervention_distribution="uniform",
        intervention_extreme_scale=EXTREME_SCALE,
        intervention_fn=None,
    ):
        # IMPORTANT: do NOT call super().__init__
        self.__dict__ = model.__dict__
        # Keep the head in fp32 for stable MSE regression, but move it onto the
        # base model's device so it does not silently sit on CPU when the base
        # model is dispatched with device_map / loaded in bf16 (which would
        # otherwise throw a device/dtype mismatch at the head matmul).
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=True)
        _base_param = next(model.parameters())
        self.regression_head = self.regression_head.to(device=_base_param.device)
        print(
            f"[IntrospectionRegressionModel] base model dtype={_base_param.dtype}, "
            f"device={_base_param.device} | regression_head dtype="
            f"{self.regression_head.weight.dtype}, device={self.regression_head.weight.device}"
        )
        self._printed_forward_dtypes = False
        # {(layer, idx): {"min","max","mean","std"}}
        self.signal_ranges = signal_ranges
        self.intervention_distribution = intervention_distribution
        self.intervention_extreme_scale = intervention_extreme_scale
        self.intervention_fn = intervention_fn  # optional custom intervention sampler
        # dedicated RNG for intervention sampling so eval passes can be made
        # reproducible independently of training-time draws (see seed_intervention_rng)
        self.intervention_rng = random.Random()

    def seed_intervention_rng(self, seed):
        """Reset the intervention sampler to a fixed seed so a subsequent eval
        pass draws a reproducible sequence of intervention values. Call right
        before an evaluation loop."""
        self.intervention_rng = random.Random(seed)

    def sample_intervention_values(self,signal_layers,signal_indices,device):
        vals = []
        for layer, idx in zip(signal_layers.tolist(),signal_indices.tolist()):
            stats = self.signal_ranges[(layer, idx)]
            if self.intervention_fn is not None:
                val = self.intervention_fn(layer=layer, idx=idx, **stats)
            elif self.intervention_distribution == "gaussian":
                # fit a gaussian to the neuron's empirical activations and draw from it
                val = self.intervention_rng.gauss(stats["mean"], stats["std"])
            elif self.intervention_distribution == "uniform":
                val = self.intervention_rng.uniform(stats["min"], stats["max"])
            elif self.intervention_distribution == "extreme":
                # sanity check: sample from a range intervention_extreme_scale x
                # wider than the empirical range, centered on the empirical
                # midpoint. The much larger signal should be easier to read out if
                # the model can read the neuron at all.
                center = (stats["min"] + stats["max"]) / 2
                half_width = self.intervention_extreme_scale * (stats["max"] - stats["min"]) / 2
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
                attention_mask = kwargs.get("attention_mask")
                if attention_mask is not None:
                    num_left_pad = (attention_mask == 0).sum(dim=1).to(token_position.device)
                    token_position = token_position + num_left_pad

                # sample interventions INSIDE forward
                intervention_values = self.sample_intervention_values(signal_layer,signal_index,device)
                unique_layers = torch.unique(signal_layer)
                for layer in unique_layers.tolist():
                    mask = signal_layer == layer
                    hook = InterventionHook(
                        token_positions=token_position[mask],
                        signal_indices=signal_index[mask],
                        intervention_values=intervention_values[mask],
                    )

                    module_path = resolve_mlp_signal_path(self,layer,path_to_output=False,)
                    module = operator.attrgetter(module_path)(self)
                    handle =  module.register_forward_hook(hook)
                    hooks.append(handle)

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
                pred = self.regression_head(selected_hidden).squeeze(-1)
                regression_labels = labels.to(pred.dtype)
                loss = nn.functional.mse_loss(pred,regression_labels,)
                # Trainer expects logits
                outputs.logits = pred
                outputs.loss = loss

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
