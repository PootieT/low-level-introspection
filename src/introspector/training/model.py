import pdb
import random
import operator
from typing import Any, Optional

import torch
from torch import nn

from peft import PeftModelForCausalLM
from peft.tuners.lora.layer import Linear

from src.introspector.models.signal_paths import resolve_mlp_signal_path




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
        intervention_fn=None,
    ):
        # IMPORTANT: do NOT call super().__init__
        self.__dict__ = model.__dict__
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=True)
        self.signal_ranges = signal_ranges  # {(layer, idx): (low, high)}
        self.intervention_fn = intervention_fn  # optional custom intervention sampler

    def sample_intervention_values(self,signal_layers,signal_indices,device):
        vals = []
        for layer, idx in zip(signal_layers.tolist(),signal_indices.tolist()):
            low, high = self.signal_ranges[(layer, idx)]
            if self.intervention_fn is not None:
                val = self.intervention_fn(layer=layer,idx=idx,low=low,high=high)
            else:
                val = random.uniform(low, high)
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

            # REGRESSION MODE
            if labels is not None and labels.ndim == 1:
                hidden = outputs.hidden_states[-1]
                bs = hidden.shape[0]
                batch_idx = torch.arange(bs,device=hidden.device)
                selected_hidden = hidden[batch_idx,regression_token_positions]
                pred = self.regression_head(selected_hidden).squeeze(-1)
                regression_labels = labels.to(pred.dtype)
                loss = nn.functional.mse_loss(pred,regression_labels,)
                # Trainer expects logits
                outputs.logits = pred
                outputs.loss = loss
                outputs["intervention_values"] = intervention_values

            return outputs

        finally:
            # ALWAYS cleanup hooks
            for h in hooks:
                h.remove()
