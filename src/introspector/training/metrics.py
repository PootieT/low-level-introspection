import math
import pdb
from typing import Tuple, List
import regex as re

import numpy as np
from scipy.stats import pearsonr, spearmanr, entropy
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


def get_numericals(pred: np.ndarray, label: np.ndarray, tokenizer) -> Tuple[List[float], List[float]]:
    # since each label maybe different number of tokens, cannot vectorize
    label_nums = []
    pred_nums = []
    nan_cnt = 0
    example_nan_gen = []
    for i in range(len(pred)):
        mask = label[i]!=-100
        label_num = float(tokenizer.decode(label[i][mask], skip_special_tokens=True, clean_up_tokenization_spaces=False))
        try:
            pred_str = tokenizer.decode(pred[i][mask], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            pred_str_cleaned = re.sub(r'[^0-9.\-+]', '', pred_str)
            pred_num = float(pred_str_cleaned)
        except:
            # pred_num = np.nan
            nan_cnt += 1
            example_nan_gen.append(f"raw: {pred_str}, cleaned: {pred_str_cleaned}")
            continue  # skipping NaNs so numerical metrics don't become NaN
        label_nums.append(label_num)
        pred_nums.append(pred_num)
    print(f"Total of {nan_cnt} out of {len(pred)} samples not parsable.")
    if nan_cnt > 0:
        print("\n".join(example_nan_gen[:10]))
    return label_nums, pred_nums


def compute_l2_dist(label_nums: List[float], pred_nums: List[float]):
    all_dist = []
    for i in range(len(label_nums)):
        l2_dist = math.sqrt((label_nums[i] - pred_nums[i]) ** 2)
        all_dist.append(l2_dist)
    mean_dist = np.mean(all_dist) if len(all_dist) > 0 else np.nan
    return mean_dist


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray], tokenizer: PreTrainedTokenizerBase):
    pred, label = eval_pred
    metrics = {}
    if not (pred.ndim==label.ndim==1): # not regressor
        pred = pred[..., :-1]
        labels = label[..., 1:]
        non_prompt_mask = labels != -100
    
        # compute exact match token by token
        match = np.ma.masked_array(pred == labels, mask=~non_prompt_mask)
        exact_match = match.all(axis=1).mean()
        mean_char_match = match.mean(axis=1).mean()
        metrics.update({
            "exact_match": exact_match,
            "mean_char_match": mean_char_match,
        })

        # get numerical labels and predictions
        label_nums, pred_nums = get_numericals(pred, labels, tokenizer)
    else: # regressor
        pred_nums, label_nums = pred, label
    numerical_metrics = compute_metrics_from_numericals(pred_nums, label_nums)
    return {
        **numerical_metrics,
    }

def compute_metrics_from_numericals(pred_nums: List[float], label_nums:List[float]):
    # just in case filter out nan cases, because they corrupt the metrics
    nan_mask = np.isnan(pred_nums)
    non_nan_mask = ~nan_mask
    if sum(nan_mask) > 0:
        print(f"Removing {sum(nan_mask)} out of {len(pred_nums)} samples that are NaN.")
    pred_nums = np.array(pred_nums)[non_nan_mask]
    label_nums = np.array(label_nums)[non_nan_mask]

    # compute numerical distance to label
    l2_dist = compute_l2_dist(label_nums, pred_nums)

    # check if distributionally the predictions are lining up
    avg_diff = np.mean(pred_nums) - np.mean(label_nums)
    std_diff = np.std(pred_nums) - np.std(label_nums)

    # correlation
    try:
        corr, p_value = pearsonr(pred_nums, label_nums)
        corr, p_value = corr.item(), p_value.item()
    except ValueError as e:
        print("Value error:", e)
        corr = float("nan")
        p_value = float("nan")

    # Rank correlation ALONGSIDE Pearson, never instead of it -- an added column, so
    # every number any existing run reported is unchanged.
    #
    # Why it is needed: a post-softmax attention weight is heavily right-skewed (the BOS
    # sink sits at 0.3-0.9 while mid-context cells sit at 1e-4-1e-1). Pearson and the
    # training loss _var_norm_mse are both squared-scale, so a handful of large rows
    # decide the reported value, and a model that has learned only "is this the sink"
    # scores nearly as well as one that reads the whole range. Spearman is insensitive to
    # that, so the gap between the two IS the measurement of how much the tail is
    # carrying. Attention logits were roughly symmetric, so this never came up before.
    try:
        spearman, spearman_p = spearmanr(pred_nums, label_nums)
        spearman, spearman_p = float(spearman), float(spearman_p)
    except ValueError as e:
        print("Value error (spearman):", e)
        spearman = float("nan")
        spearman_p = float("nan")
    return {
        "l2_dist": l2_dist.item(),
        "avg_diff": avg_diff.item(),
        "std_diff": std_diff.item(),
        "corr": corr,
        "corr_p_value": p_value,
        "spearman": spearman,
        "spearman_p_value": spearman_p,
    }