import math
import pdb
from typing import Tuple, List
import regex as re

import numpy as np
from scipy.stats import pearsonr, entropy
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


def get_numericals(pred: np.ndarray, label: np.ndarray, tokenizer) -> Tuple[List[float], List[float]]:
    # since each label maybe different number of tokens, cannot vectorize
    label_nums = []
    pred_nums = []
    for i in range(len(pred)):
        mask = label[i]!=-100
        label_num = float(tokenizer.decode(label[i][mask], skip_special_tokens=True, clean_up_tokenization_spaces=False))
        try:
            pred_str = tokenizer.decode(pred[i][mask], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            pred_str_cleaned = re.sub(r'[^0-9.\-+]', '', pred_str)
            pred_num = float(pred_str_cleaned)
        except:
            pred_num = np.nan
        label_nums.append(label_num)
        pred_nums.append(pred_num)
    return label_nums, pred_nums


def compute_l2_dist(label_nums: List[float], pred_nums: List[float]):
    all_dist = []
    for i in range(len(label_nums)):
        l2_dist = math.sqrt((label_nums[i] - pred_nums[i]) ** 2)
        all_dist.append(l2_dist)
    mean_dist = np.mean(all_dist) if len(all_dist) > 0 else np.nan
    return mean_dist


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray], tokenizer: PreTrainedTokenizerBase):
    logits, label = eval_pred

    # fix label shifting by 1 token
    logits = torch.from_numpy(logits)[..., :-1, :].contiguous()
    labels = torch.from_numpy(label)[..., 1:].contiguous()
    non_prompt_mask = labels != -100
    pred = np.argmax(logits, axis=-1)

    # compute NLL loss wrt label
    with torch.no_grad():
        loss = F.nll_loss(F.log_softmax(logits, dim=-1)[non_prompt_mask], labels[non_prompt_mask])

    # compute exact match token by token
    match = np.ma.masked_array(pred == labels, mask=~non_prompt_mask)
    exact_match = match.all(axis=1).mean()
    mean_char_match = match.mean(axis=1).mean()

    # get numerical labels and predictions
    label_nums, pred_nums = get_numericals(pred, labels.numpy(), tokenizer)

    # compute numerical distance to label
    l2_dist = compute_l2_dist(label_nums, pred_nums)

    # check if distributionally the predictions are lining up
    avg_diff = np.mean(pred_nums) - np.mean(label_nums)
    std_diff = np.std(pred_nums) - np.std(label_nums)

    # correlation
    corr, p_value = pearsonr(pred_nums, label_nums)
    return {
        "loss": loss,
        "l2_dist": l2_dist,
        "exact_match": exact_match,
        "mean_char_match": mean_char_match,
        "avg_diff": avg_diff,
        "std_diff": std_diff,
        "corr": corr,
        "corr_p_value": p_value,
    }
