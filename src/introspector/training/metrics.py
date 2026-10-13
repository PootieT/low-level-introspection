import math
import pdb
from typing import Tuple, List
import regex as re

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, entropy
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


def monotone_r2(pred_nums: np.ndarray, label_nums: np.ndarray, folds: int = 2,
                seed: int = 0) -> float:
    """Variance of the target explained by the best MONOTONE function of the prediction.

    The rank correlations say whether the ordering is right; this says how much of the
    target a monotone readout could actually reproduce, on the same scale as Pearson r^2.
    Where the two agree (r^2 ~= this), the relationship really is linear and nothing is
    being lost by measuring it linearly.

    CROSS-FIT, because isotonic regression has close to n degrees of freedom and would
    otherwise fit the noise: an in-sample isotonic R^2 is near 1 for any input and means
    nothing. Each half is fitted on the other half and scored out of sample.
    """
    from sklearn.isotonic import IsotonicRegression
    n = len(pred_nums)
    if n < 2 * folds or np.std(label_nums) < 1e-12:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    out = np.empty(n, dtype=float)
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te, assume_unique=False)
        if len(tr) < 2 or len(te) == 0:
            return float("nan")
        ir = IsotonicRegression(out_of_bounds="clip").fit(pred_nums[tr], label_nums[tr])
        out[te] = ir.predict(pred_nums[te])
    ss_tot = float(np.sum((label_nums - label_nums.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - np.sum((label_nums - out) ** 2) / ss_tot)


def rank_metrics(pred_nums: np.ndarray, label_nums: np.ndarray) -> dict:
    """Similarity measures that do NOT assume the readout is a linear function of the
    target, plus the diagnostics needed to read them.

    WHY THESE AND NOT ONLY PEARSON. Measured over this project's own saved predictions:
    on healthy runs Pearson, Spearman and the monotone R^2 agree to ~0.05, so the linear
    assumption costs nothing there. On the gate-ON arm they diverge hard -- Pearson 0.33
    against Spearman 0.72 -- and the cause is not curvature but a handful of predictions
    an order of magnitude outside the target's range (predictions spanning -3.8..5.5 for
    activations in -0.28..0.73). Trimming 1% of the prediction extremes lifts Pearson to
    0.71. So the gap is a TAIL/OUTLIER signal, and `pred_out_of_range_frac` is reported
    next to it to make that readable rather than mysterious.

      spearman     monotone, rank-based, immune to the magnitude of an outlier
      kendall_tau  tau-b, which handles TIES explicitly -- post-SiLU activations pile up
                   at the floor (~15-36% distinct values in these runs), and Spearman's
                   tie correction is cruder. Lower in magnitude than r/rho by
                   construction; use it to rank runs, not as a fraction.
      monotone_r2  see above; the only one of the three on r^2's scale
    """
    out = {}
    try:
        rho, rho_p = spearmanr(pred_nums, label_nums)
        out["spearman"] = float(rho)
        out["spearman_p_value"] = float(rho_p)
    except Exception:
        out["spearman"] = out["spearman_p_value"] = float("nan")
    try:
        tau, tau_p = kendalltau(pred_nums, label_nums, variant="b")
        out["kendall_tau"] = float(tau)
        out["kendall_p_value"] = float(tau_p)
    except Exception:
        out["kendall_tau"] = out["kendall_p_value"] = float("nan")
    try:
        out["monotone_r2"] = monotone_r2(np.asarray(pred_nums, dtype=float),
                                         np.asarray(label_nums, dtype=float))
    except Exception:
        out["monotone_r2"] = float("nan")

    # diagnostics that make the above interpretable
    a = np.asarray(label_nums, dtype=float)
    pr = np.asarray(pred_nums, dtype=float)
    lo, hi = float(a.min()), float(a.max())
    out["pred_out_of_range_frac"] = float(np.mean((pr < lo) | (pr > hi)))
    # ties in the TARGET: rank metrics degrade when most targets are indistinguishable
    out["target_unique_frac"] = float(len(np.unique(np.round(a, 6))) / max(1, len(a)))
    return out


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
    out = {
        "l2_dist": l2_dist.item(),
        "avg_diff": avg_diff.item(),
        "std_diff": std_diff.item(),
        "corr": corr,
        "corr_p_value": p_value,
    }
    # `corr` stays Pearson so every existing reader and plot is unchanged; the rank
    # measures are added alongside. Cheap at these sizes (<1 ms each at n=3000).
    #
    # Attention weights need this most: a post-softmax weight is heavily right-skewed
    # (the BOS sink sits at 0.3-0.9 while mid-context cells sit at 1e-4-1e-1). Pearson
    # and the training loss _var_norm_mse are both squared-scale, so a handful of large
    # rows decide the reported value, and a model that has learned only "is this the
    # sink" scores nearly as well as one that reads the whole range. The Pearson -
    # Spearman gap measures how much the tail is carrying.
    if len(pred_nums) >= 2:
        out.update(rank_metrics(pred_nums, label_nums))
        out["pearson_minus_spearman"] = (out["corr"] - out["spearman"]
                                         if np.isfinite(out.get("spearman", np.nan))
                                         and np.isfinite(out["corr"]) else float("nan"))
    return out
