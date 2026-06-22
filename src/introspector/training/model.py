import pdb
import threading
from typing import Any, Optional

import torch
from peft.tuners.lora.layer import Linear

from peft import PeftModelForCausalLM, LoraConfig
from peft.tuners.lora import LoraLayer
from peft.tuners.lora.layer import Linear



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




