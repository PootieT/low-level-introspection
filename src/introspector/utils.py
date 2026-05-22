# copied from https://nnsight.net/tutorials/mini-papers/feucht_dual_route_induction/?h=score#code-implementation

import math
import torch
from transformers import BitsAndBytesConfig

from src.introspector.config import ExperimentConfig


def get_quantization_config(config: ExperimentConfig):
    q_config = None
    if config.load_in_8bit:
        q_config = BitsAndBytesConfig(load_in_8bit=True)
    elif config.load_in_4bit:
        q_config = BitsAndBytesConfig(load_in_4bit=True)
    return q_config


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L178
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# given a sequence and layer, get attention weights for all heads.
def get_l2_attn_weights(model, tokenized, layer, value_weighting=True, remote=False):
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    with torch.no_grad(), model.trace(tokenized, remote=remote):
        # positional_embeddings (cos, sin) each shape [bsz, seq_len, head_dim]
        position = model.model.layers[layer].self_attn.inputs[1]['position_embeddings']
        attention_mask = model.model.layers[layer].self_attn.inputs[1]['attention_mask']

        # [bsz, seq_len, model_size]
        query_states = model.model.layers[layer].self_attn.q_proj.output
        key_states = model.model.layers[layer].self_attn.k_proj.output

        bsz = query_states.shape[0]; seq_len = query_states.shape[1]
        if value_weighting:
            # [bsz, seq_len, model_size] -> [bsz, seq_len, n_heads, head_dim]
            value_states = model.model.layers[layer].self_attn.v_proj.output
            value_states = value_states.view(bsz, seq_len, n_heads, head_dim).save()

        # from modeling_llama, convert to [bsz, n_heads, seq_len, head_dim] and rotate
        query_states = query_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, position[0], position[1])

        # not needed because num_key_value_heads == num_attention_heads
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        # has to be eager implementation
        if not remote: # ndif model isn't eager implementation
          causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
          attn_weights = attn_weights + causal_mask
        attn_weights = attn_weights.save()

    if not value_weighting:
        return attn_weights.softmax(dim=-1).detach().cpu()
    else:
        # get l2 norm of each head value vector [bsz, seq_len, n_heads] -> [bsz, n_heads, seq_len]
        value_norms = torch.linalg.vector_norm(value_states, dim=-1).detach().cpu().transpose(1, 2)

        # attn_weights [bsz, n_heads, seq_len, seq_len]
        attn_weights = attn_weights.softmax(dim=-1).detach().cpu()

        # then multiply by softmax values and normalize
        effective = attn_weights * value_norms.unsqueeze(2).expand(attn_weights.shape)
        effective /= torch.sum(effective, dim=-1, keepdim=True)
        return effective