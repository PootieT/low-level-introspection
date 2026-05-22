import math
import pdb
from typing import Tuple
import regex as re

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


def compute_l2_dist(pred: np.ndarray, label: np.ndarray, tokenizer):
    # since each label maybe different number of tokens, cannot vectorize
    all_dist = []
    for i in range(len(pred)):
        mask = label[i]!=-100
        label_num = float(tokenizer.decode(label[i][mask], skip_special_tokens=True, clean_up_tokenization_spaces=False))
        try:
            pred_str = tokenizer.decode(pred[i][mask], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            pred_str_cleaned = re.sub(r'[^0-9.\-+]', '', pred_str)
            pred_num = float(pred_str_cleaned)
        except:
            continue

        l2_dist = math.sqrt((label_num - pred_num) ** 2)
        all_dist.append(l2_dist)
    mean_dist = np.mean(all_dist) if len(all_dist) > 0 else np.nan
    return mean_dist

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray], tokenizer: PreTrainedTokenizerBase):
    logits, label = eval_pred
    pred = np.argmax(logits, axis=-1)

    # compute NLL loss wrt label
    non_prompt_mask = label != -100
    loss = F.nll_loss(F.log_softmax(torch.from_numpy(logits), dim=-1)[non_prompt_mask], torch.from_numpy(label[non_prompt_mask]))

    # compute exact match token by token
    match = np.ma.masked_array(pred == label, mask=~non_prompt_mask)
    exact_match = match.all(axis=1).mean()

    # compute numerical distance to label
    l2_dist = compute_l2_dist(pred, label, tokenizer)
    return {"loss": loss, "l2_dist": l2_dist, "exact_match": exact_match}
