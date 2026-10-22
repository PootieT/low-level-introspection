#!/usr/bin/env python3
"""
Prompted-baseline introspection: can the UNTRAINED model report a neuron when asked?

Same caches, signals, prompt templates and metrics as trainers/train_introspector_joint.py, with
two differences: the LM is never trained, and the readout is a closed-form ridge probe
instead of an SGD-trained linear head. Everything a linear readout of the frozen model
can do is the floor the trained introspector has to beat; the gap is what training the
LM buys.

NO ADDED VOCAB. Trained runs put the added <|introspection|> token between the text and
the prompt. Its embedding is random at resize time and frozen through training, so to a
model that never trained on it, it is noise in the middle of the input. The text and the
prompt are joined by plain text here (--separator). The intervention hook indexes the
TEXT by its cached token_position, so every row is checked to tokenize its text prefix
exactly as the text alone does (BPE can merge a final text token with what follows it);
rows that do not are dropped, and a systematic fraction is an error.

READOUT OVER k GENERATED TOKENS. The model continues the prompt greedily for k_max
tokens, and pool_k is the mean of the final-layer hidden state at [last prompt token,
gen_1 .. gen_k]. k=0 is exactly the vector the trained head reads
(hidden_states[-1][:, -1]). One k_max-token generation serves every k. pool "last" is
the hidden state at gen_k alone.

INTERVENTION DURING GENERATION. The value is injected ONCE, in the prefill forward over
the prompt; decode steps run unhooked and see it through the KV cache -- what the model
would see had the neuron really taken that value. generate() cannot do this:
IntrospectionRegressionModel.forward samples a FRESH value on every call, and
InterventionHook would index the cached text position in a length-1 decode tensor.

TWO PROBES, each scored on BOTH metrics:
  endo   fit on (h(x), a(x))                         head-only analogue of SFT (w=1)
  joint  fit on  w   * (h(x),        a(x))           the joint pipeline with LoRA lr=0:
             + (1-w) * (h(x; a<-v),  v)              same w, var-normalized, and v from
                                                     the run's own training injection
They are NOT the same fit, frozen model or not. The interventional rows are different
feature vectors (the injection changes everything downstream of it), and the two terms
reward different directions: the endogenous term credits ANY feature correlated with
a(x), context included; the interventional term only credits directions that move with
the neuron, since v is independent of the text. They coincide only for a faithful reader.

METRICS (as the trained arms report them):
  endogenous corr       probe(h(x)) vs a(x) on the test split
  interventional corr   probe(h(x; a<-v)) vs v, v ~ N(mean, std) -- gaussian, scale 1
  verbal                the number the model WRITES in its continuation, vs a(x) / v: the
                        pure prompting readout, no probe at all

OUTPUTS, under <baseline-root>/prompt_<signal>_POS<p>/ (one dir per neuron x position;
both KL regimes of the grid share it, since nothing here depends on the trained run):
  prompt_baseline_by_k.csv       one row per (prompt, probe, pool, k): the main table
  prompt_baseline_metrics.json   everything, plus probe selection diagnostics
  learning_curve.csv             (--learning-curve) k=0 metrics vs probe training rows
  verbal_report.json, generations_sample.csv
  activation_drift.*             alignment check (want offset +0, corr ~1, 'alignment OK')
  endo_probe/, joint_probe/      k=0 results under the STANDARD file names
                                 (endogenous_test_metrics_endogenous.json,
                                 interventional_test_metrics_gaussian.json,
                                 ood_generalization.csv, interventional_scale_sweep.csv), so
                                 each probe reads as one more arm to the existing aggregators

Example (one grid run; both regimes resolve to the same output dir):
    python evals/score_prompt_baseline.py \
        --config experiment_configs/mlp_joint_grid_layer_token/joint_N2.1447_POS-1_gateoff_scale10_kl0.1_w0.01_300K.json
Whole grid, sharded over an SGE array:
    python evals/score_prompt_baseline.py --group mlp_joint_grid_layer_token \
        --run-glob 'joint_*kl0.1*' --skip-existing
"""
from __future__ import annotations

import sys
import re
import json
import random
import inspect
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from src.introspector.analysis_loading import _row_seed, compute_signal_ranges
from src.introspector.dataset_utils import apply_signal_filter
from src.introspector.prompts import PROMPT_REGISTRY
from src.introspector.run_discovery import add_discovery_args, locale_tag, runs_from_args
from src.introspector.training.metrics import compute_metrics_from_numericals
from src.introspector.training.model import (
    IntrospectionRegressionModel,
    replace_lora_with_gated_version,
    replace_model_with_gated_version,
)
from trainers.train_introspector import (
    POSITION_DROP_TOLERANCE,
    drop_invalid_positions,
    preprocess_example,
)
from trainers.train_introspector_joint import (
    UngatedCollator,
    _corr_or_nan,
    interventional_scale_sweep,
    measure_activation_drift,
)

DEFAULT_ROOT = Path("outputs/runs/mlp_prompt_baseline_layer_token")
SENTINEL = "prompt_baseline_by_k.csv"
# Ridge penalties, in units of per-feature variance: features are standardized and the
# row weights sum to 1, so the normal matrix is a covariance with trace ~H.
RIDGE_LAMBDAS = np.logspace(-6, 3, 19)
LINALG_CHUNK = 16384
PROBES = ("endo", "joint")
NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)")


def parse_args():
    p = argparse.ArgumentParser()
    add_discovery_args(p)
    p.add_argument("--baseline-root", type=Path, default=DEFAULT_ROOT,
                   help="Where results go, one subdirectory per (neuron, position).")
    p.add_argument("--separator", type=str, default="\n\n",
                   help="Plain-text join between the input text and the introspection prompt "
                        "(the template already puts a space before it). Escapes like \\n are "
                        "interpreted.")
    p.add_argument("--max-train-examples", type=int, default=30000,
                   help="Probe training rows (contiguous prefix of the run's filtered train "
                        "split), of which --val-frac selects the ridge penalty.")
    p.add_argument("--max-test-examples", type=int, default=None)
    p.add_argument("--k-values", type=str, default="0,1,2,4,8,16",
                   help="Generated-token counts to pool over; one generation of max(k) "
                        "tokens serves all of them.")
    p.add_argument("--probes", type=str, default="endo,joint")
    p.add_argument("--joint-loss-weight", type=float, default=0.01,
                   help="w on the endogenous term of the joint probe (the grid's W).")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Default: the run's eval_batch_size.")
    p.add_argument("--learning-curve", type=str, default="",
                   help="Comma-separated probe training sizes; refits the k=0 probes on "
                        "prefixes of the training pool to show whether the probe is "
                        "data-limited. Sizes above the pool are skipped.")
    p.add_argument("--no-prompt-control", action="store_true",
                   help="Also fit the probes on the text alone, read at its last token: what "
                        "the residual stream gives a linear probe with nothing asked.")
    p.add_argument("--scale-sweep", type=str, default="1,3,10,30,100",
                   help="Injection scales for the k=0 scale sweep (head = joint probe, "
                        "ceiling = fresh probe per scale). Empty string disables.")
    p.add_argument("--routing-probe-pca", type=int, default=256)
    p.add_argument("--skip-alignment-check", action="store_true")
    p.add_argument("--ood-eval", action="store_true",
                   help="Score both k=0 probes on the OOD corpus ladder (evals/score_ood_generalization.py).")
    p.add_argument("--ood-corpora", type=str, default=None)
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()
    if "\\" in args.separator:            # a shell-quoted '\n\n' arrives literally
        args.separator = args.separator.encode().decode("unicode_escape")
    # k=0 is the trained head's own readout and every diagnostic is anchored on it
    args.k_values = sorted({0} | {int(k) for k in args.k_values.split(",")})
    args.probes = [s.strip() for s in args.probes.split(",") if s.strip()]
    bad = set(args.probes) - set(PROBES)
    if bad:
        p.error(f"unknown probe(s) {sorted(bad)}; choose from {PROBES}")
    args.learning_curve = [int(n) for n in args.learning_curve.split(",") if n.strip()]
    return args


# ======================================================================
# Model: the base model behind the regressor wrapper the trained runs use
# ======================================================================

def load_frozen_model(model_name, bf16, local_files_only=False):
    """
    The BASE model, wrapped exactly as a trained run is (PEFT -> IntrospectionRegression
    Model -> gated forward), so the intervention hooks and every eval function of the
    training pipeline apply unchanged.

    The adapter is a freshly initialized rank-1 LoRA on q_proj. PEFT initializes lora_B
    to zero, so its contribution is exactly 0 and the forward IS the base model's; this
    is asserted rather than assumed. The tokenizer is NOT extended: no separator token,
    no resize.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device_map = "auto" if torch.cuda.is_available() else None
    base = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, local_files_only=local_files_only,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32)
    if device_map is None and torch.backends.mps.is_available():
        base = base.to("mps")
    peft_model = get_peft_model(base, LoraConfig(
        r=1, lora_alpha=1, target_modules=["q_proj"], lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM"))
    n_b = 0
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            n_b += 1
            if param.detach().abs().max().item() != 0.0:
                raise RuntimeError(f"{name} is not zero-initialized; the wrapped model would "
                                   f"not be the base model")
    if n_b == 0:
        raise RuntimeError("no lora_B parameters found; adapter construction changed")
    model = IntrospectionRegressionModel(peft_model, signal_ranges={})
    replace_model_with_gated_version(model)
    replace_lora_with_gated_version(model)
    model.eval()
    return model, tokenizer


def _logits_kwargs(model):
    """Ask for last-position logits only, where the installed transformers supports it:
    full-vocab logits over a 128-row prefill are ~2 GB that nothing here reads."""
    params = inspect.signature(model.get_base_model().forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return {name: 1}
    return {}


def configure_run(model, config, signal_ranges):
    """Point the shared model at one run: its neuron's empirical stats and its own
    training injection distribution (what the joint probe's interventional rows use)."""
    tc = config.training_config
    model.signal_ranges = signal_ranges
    model.intervention_distribution = tc.intervention_signal_distribution
    model.intervention_extreme_scale = tc.intervention_extreme_scale
    model.intervention_scale = (tc.intervention_extreme_scale
                                if tc.intervention_signal_distribution == "extreme" else 1.0)
    if getattr(tc, "intervention_anneal", False):
        print(f"[prompt] WARNING: {config.experiment_name} annealed its injection scale; the "
              f"joint probe uses the un-annealed start distribution")


# ======================================================================
# Data
# ======================================================================

def make_preprocess(tokenizer, config, separator, no_prompt=False):
    """
    Row -> model kwargs, mirroring train_introspector_joint's preprocess, plus two
    per-row flags for whether the full prompt's tokens up to and including the
    introspected token equal the text's own:

      prefix_ok        they do (rows where they do not are dropped)
      template_merge   they do not, and the TRAINED pipeline's prompt breaks them too.
                       The template puts a space after the text; a text that ends in a
                       lone-space token merges with it ("Ġ" + " " -> "ĠĠ"), and the
                       special token cannot prevent that because it comes after the
                       space. ~1.5% of rows at POS-1, where that token is the
                       introspected one; earlier positions are unaffected.

    A mismatch that is not a template_merge is the SEPARATOR's doing, and is what
    prepare_split refuses to tolerate.

    no_prompt=True is the control: the text alone, read at its last token.
    """
    max_len = config.training_config.max_seq_length
    prompt_fn = PROMPT_REGISTRY[config.training_config.prompt_type]

    def preprocess(ex):
        text_ids = tokenizer(ex["text"], truncation=True, max_length=max_len,
                             add_special_tokens=True)["input_ids"]
        tp = int(ex["token_position"])
        if no_prompt:
            out = {"input_ids": text_ids, "attention_mask": [1] * len(text_ids),
                   "labels": [-100] * len(text_ids), "frozen_input_len": len(text_ids),
                   "position_valid": tp < len(text_ids),
                   "text_truncated": len(text_ids) >= max_len}
        else:
            # same per-row seeded template draw as analysis_loading: a row's prompt is a
            # property of the row, not of how many rows were mapped before it
            state = random.getstate()
            random.seed(_row_seed(ex))
            try:
                out = preprocess_example(
                    ex, tokenizer=tokenizer, prompt_fn=prompt_fn, max_length=max_len,
                    return_frozen_input_len=True, return_position_valid=True,
                    for_prediction=True, separator=separator)
            finally:
                random.setstate(state)
        out["prefix_ok"] = out["input_ids"][:tp + 1] == text_ids[:tp + 1]
        # what the trained prompt tokenizes the text as: the special token is split out
        # before BPE, so its text segment is exactly f"{text} "
        trained_ids = tokenizer(f"{ex['text']} ", truncation=True, max_length=max_len,
                                add_special_tokens=True)["input_ids"]
        out["template_merge"] = (not no_prompt) and trained_ids[:tp + 1] != text_ids[:tp + 1]
        # the full prompt hitting max_length means the END of the prompt was cut off
        out["prompt_truncated"] = (not no_prompt) and len(out["input_ids"]) >= max_len
        out["token_position"] = ex["token_position"]
        out["signal_layer"] = ex["signal_layer"]
        out["signal_index"] = ex["signal_index"]
        out["activation"] = ex["activation"]
        return out

    return preprocess


def prepare_split(raw, preprocess, split, max_len):
    ds = raw.map(preprocess, remove_columns=raw.column_names)
    ds = drop_invalid_positions(ds, split, max_len)
    n = len(ds)
    n_trunc = int(np.asarray(ds["prompt_truncated"], dtype=bool).sum())
    if n_trunc:
        print(f"[{split}] WARNING: {n_trunc}/{n} prompts reach max_seq_length={max_len}; "
              f"their prompt END is truncated")
    bad = ~np.asarray(ds["prefix_ok"], dtype=bool)
    inherited = bad & np.asarray(ds["template_merge"], dtype=bool)
    n_bad, n_inh = int(bad.sum()), int(inherited.sum())
    n_sep = n_bad - n_inh
    if n_sep / max(n, 1) > POSITION_DROP_TOLERANCE:
        raise ValueError(f"[{split}] {n_sep}/{n} rows tokenize their text differently once the "
                         f"separator follows it; the intervention would land on the wrong "
                         f"token. Use a separator that cannot merge with the text.")
    if n_bad:
        # dropped, not kept for parity with the trained arms: on these rows the token at
        # the cached position is not the token the cache (and the prompt) refers to
        print(f"[{split}] dropping {n_bad}/{n} rows whose introspected token re-tokenizes "
              f"once the prompt follows it ({n_inh} the template's trailing-space merge, "
              f"which the trained runs share; {n_sep} from the separator)")
        ds = ds.filter(lambda ex: ex["prefix_ok"])
    return ds.remove_columns(["prefix_ok", "template_merge", "prompt_truncated"])


# ======================================================================
# Features: final-layer readout over k greedily generated tokens
# ======================================================================

def feature_keys(k_values):
    """('mean', k) for every k, plus ('last', k) for k > 0 (at k=0 they coincide)."""
    return [("mean", k) for k in k_values] + [("last", k) for k in k_values if k > 0]


@torch.no_grad()
def extract_features(model, dataset, collator, tokenizer, batch_size, k_values, logits_kw,
                     intervene=False, distribution=None, scale=None, seed=0):
    """
    One prefill + max(k_values) greedy decode steps per batch.

    Returns {"feats": {key: fp16 [N, H]}, "target": [N] float (the true activation, or the
    injected value under intervention), "gen": [N] decoded continuations (k_max > 0)}.
    Features are kept in fp16 on the CPU (a 30k x 2048 split is 120 MB per key); they are
    accumulated in fp32 and checked finite before the cast.
    """
    k_values = sorted(set(k_values))
    k_max = max(k_values)
    want = set(k_values)
    prev = (model.intervention_distribution, model.intervention_scale)
    if intervene:
        if distribution is not None:
            model.intervention_distribution = distribution
        if scale is not None:
            model.intervention_scale = scale
        model.seed_intervention_rng(seed)
    model.eval()
    device = next(model.parameters()).device
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    feats = {key: [] for key in feature_keys(k_values)}
    targets, gens = [], []

    def keep(key, h):
        if not torch.isfinite(h).all() or h.abs().max() > 6e4:
            raise FloatingPointError(f"feature {key} is non-finite or out of fp16 range")
        feats[key].append(h.half().cpu())

    try:
        for batch in dl:
            batch = batch.to(device)
            activation = batch.pop("activation")
            batch.pop("anchor_region_mask", None)          # KL-anchor bookkeeping only
            attn = batch["attention_mask"]
            out = model(**batch, do_intervention=intervene, use_cache=k_max > 0,
                        output_hidden_states=True, **logits_kw)
            targets.append((out.intervention_values if intervene else activation).float().cpu())
            h = out.hidden_states[-1][:, -1, :].float()
            running = h.clone()
            keep(("mean", 0), h)
            if k_max == 0:
                continue
            # decode: the injection (if any) lives on only in the prefill's KV cache.
            # Positions are not passed -- the model continues from the cache length, the
            # same arange-including-pads convention the single-forward evals use.
            pkv = out.past_key_values
            tok = out.logits[:, -1, :].argmax(-1)
            gen = [tok]
            for step in range(1, k_max + 1):
                attn = torch.cat([attn, attn.new_ones(attn.shape[0], 1)], dim=1)
                out = model(input_ids=tok[:, None], attention_mask=attn, past_key_values=pkv,
                            use_cache=True, do_intervention=False, output_hidden_states=True,
                            **logits_kw)
                h = out.hidden_states[-1][:, -1, :].float()
                running += h
                if step in want:
                    keep(("mean", step), running / (step + 1))
                    keep(("last", step), h)
                pkv = out.past_key_values
                if step < k_max:
                    tok = out.logits[:, -1, :].argmax(-1)
                    gen.append(tok)
            gens.extend(tokenizer.batch_decode(torch.stack(gen, dim=1), skip_special_tokens=True))
    finally:
        model.intervention_distribution, model.intervention_scale = prev
    return {"feats": {key: torch.cat(v) for key, v in feats.items()},
            "target": torch.cat(targets), "gen": gens}


# ======================================================================
# Probes: weighted ridge, penalty selected on held-out probe-training rows
# ======================================================================

@dataclass
class RidgeProbe:
    coef: torch.Tensor          # [H] float64, in RAW feature space
    intercept: float
    lam: float

    def predict(self, X) -> np.ndarray:
        return (X.double().cpu() @ self.coef + self.intercept).numpy()


def _linalg_device():
    # MPS has no float64; the normal equations want it (eigh of a 2048^2 covariance)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _row_chunks(idx, chunk=LINALG_CHUNK):
    for i in range(0, len(idx), chunk):
        yield idx[i:i + chunk]


def _standardizer(blocks, idx, dev):
    """Per-feature mean/std over rows `idx` of every block, accumulated in chunks."""
    n, s1, s2 = 0, 0.0, 0.0
    for X, _, _ in blocks:
        for ch in _row_chunks(idx):
            x = X[ch].to(dev, torch.float64)
            n += x.shape[0]; s1 = s1 + x.sum(0); s2 = s2 + (x * x).sum(0)
    mu = s1 / n
    sd = (s2 / n - mu * mu).clamp_min(0).sqrt().clamp_min(1e-6)
    return mu, sd


def _normal_equations(blocks, idx, mu, sd, dev):
    """
    Weighted, centered normal equations over rows `idx` of every block.

    A block is (X [N,H], y [N], block_weight). Row weights are block_weight /
    (n * var(y_block)), i.e. the objective is sum_b w_b * MSE_b / var_b -- the
    variance-normalized joint loss train_introspector_joint trains the head on. Weights
    sum to 1 over a single block, so the penalty scale does not drift with n.

    Accumulated in row chunks (two passes: weighted means, then the Gram matrix), so a
    300k-row learning-curve fit costs a chunk of GPU memory rather than several copies
    of the whole design matrix.
    """
    rw = []                                   # per-block constant row weight
    R, s_z, s_y = 0.0, 0.0, 0.0
    for X, y, wb in blocks:
        yy = y[idx].double()
        r = wb / (len(idx) * float(yy.var().clamp_min(1e-12)))
        rw.append(r)
        R += r * len(idx)
        s_y += r * float(yy.sum())
        for ch in _row_chunks(idx):
            s_z = s_z + r * ((X[ch].to(dev, torch.float64) - mu) / sd).sum(0)
    zm, ym = s_z / R, s_y / R
    H = mu.shape[0]
    A = torch.zeros(H, H, dtype=torch.float64, device=dev)
    c = torch.zeros(H, dtype=torch.float64, device=dev)
    for (X, y, _), r in zip(blocks, rw):
        for ch in _row_chunks(idx):
            zc = (X[ch].to(dev, torch.float64) - mu) / sd - zm
            A += r * (zc.T @ zc)
            c += r * (zc.T @ (y[ch].to(dev, torch.float64) - ym))
    return A / R, c / R, zm, ym


def fit_probe(blocks, tr_idx, val_idx, lambdas=RIDGE_LAMBDAS):
    """Fit on tr_idx for every lambda (one eigendecomposition), select on val_idx by the
    same weighted variance-normalized loss, refit on tr+val. Returns (probe, diagnostics)."""
    dev = _linalg_device()
    tr_idx, val_idx = torch.as_tensor(tr_idx), torch.as_tensor(val_idx)
    mu, sd = _standardizer(blocks, tr_idx, dev)
    A, c, zm, ym = _normal_equations(blocks, tr_idx, mu, sd, dev)
    evals, Q = torch.linalg.eigh(A)
    qc = Q.T @ c
    val = []
    for X, y, wb in blocks:
        Pv = ((X[val_idx].to(dev, torch.float64) - mu) / sd - zm) @ Q
        val.append((Pv, y[val_idx].to(dev, torch.float64), wb,
                    float(y[tr_idx].double().var().clamp_min(1e-12))))
    curve = []
    for lam in lambdas:
        scaled = qc / (evals + lam)
        loss, corrs = 0.0, []
        for Pv, yv, wb, var in val:
            pred = Pv @ scaled + ym
            loss += wb * float(((pred - yv) ** 2).mean()) / var
            corrs.append(_corr_or_nan(pred.cpu().numpy(), yv.cpu().numpy()))
        curve.append({"lambda": float(lam), "val_loss": loss, "val_corr_by_block": corrs})
    best = min(range(len(curve)), key=lambda i: curve[i]["val_loss"])
    lam = float(lambdas[best])

    all_idx = torch.cat([tr_idx, val_idx])
    A, c, zm, ym = _normal_equations(blocks, all_idx, mu, sd, dev)
    beta = torch.linalg.solve(A + lam * torch.eye(A.shape[0], dtype=A.dtype, device=dev), c)
    # back to raw features: pred = ((x - mu)/sd - zm) . beta + ym
    coef = (beta / sd).cpu()
    intercept = float(ym - ((mu / sd + zm) * beta).sum())
    diag = {"lambda": lam, "lambda_at_edge": best in (0, len(lambdas) - 1),
            "val_loss": curve[best]["val_loss"], "val_curve": curve,
            "n_fit": int(len(all_idx))}
    return RidgeProbe(coef=coef, intercept=intercept, lam=lam), diag


def probe_blocks(name, endo, interv, key, w):
    if name == "endo":
        return [(endo["feats"][key], endo["target"], 1.0)]
    return [(endo["feats"][key], endo["target"], w),
            (interv["feats"][key], interv["target"], 1.0 - w)]


def split_indices(n, val_frac, seed):
    perm = np.random.RandomState(seed).permutation(n)
    n_val = max(20, int(round(n * val_frac)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def score_probe(probe, test_endo, test_int, key):
    pe = probe.predict(test_endo["feats"][key])
    pi = probe.predict(test_int["feats"][key])
    return (compute_metrics_from_numericals(pe, test_endo["target"].numpy()),
            compute_metrics_from_numericals(pi, test_int["target"].numpy()), pe, pi)


def fit_and_score(args, prompt_tag, keys, train_endo, train_int, test_endo, test_int,
                  tr_idx, val_idx):
    """Every (probe, feature key) cell of the main table. Returns (rows, {(probe, key):
    (probe, endo_pred, interv_pred)})."""
    rows, fitted = [], {}
    for key in keys:
        for name in args.probes:
            probe, diag = fit_probe(probe_blocks(name, train_endo, train_int, key,
                                                 args.joint_loss_weight), tr_idx, val_idx)
            me, mi, pe, pi = score_probe(probe, test_endo, test_int, key)
            fitted[(name, key)] = (probe, pe, pi)
            rows.append({"prompt": prompt_tag, "probe": name, "pool": key[0], "k": key[1],
                         "n_fit": diag["n_fit"], "lambda": diag["lambda"],
                         "lambda_at_edge": diag["lambda_at_edge"], "val_loss": diag["val_loss"],
                         "endo_corr": me["corr"], "endo_l2": me["l2_dist"],
                         "interv_corr": mi["corr"], "interv_l2": mi["l2_dist"]})
            print(f"[prompt] {prompt_tag:9s} {name:5s} {key[0]:4s} k={key[1]:<2d} "
                  f"lambda={diag['lambda']:.1e}{'*' if diag['lambda_at_edge'] else ' '} "
                  f"endo={me['corr']:+.4f} interv={mi['corr']:+.4f}")
    return rows, fitted


# ======================================================================
# Outputs
# ======================================================================

def verbal_report(test_endo, test_int):
    """The pure prompting readout: the first number in the model's own continuation."""
    def parse(gens):
        vals = []
        for g in gens:
            m = NUMBER_RE.search(g)
            try:
                vals.append(float(m.group(0)) if m else np.nan)
            except ValueError:
                vals.append(np.nan)
        return np.asarray(vals, dtype=float)

    out = {}
    for arm, d in (("endogenous", test_endo), ("interventional", test_int)):
        if not d["gen"]:
            continue
        v = parse(d["gen"])
        y = d["target"].numpy()
        ok = np.isfinite(v)
        out[arm] = {"n": int(len(v)), "parse_rate": float(ok.mean()),
                    "corr": _corr_or_nan(v[ok], y[ok]) if ok.sum() > 2 else float("nan"),
                    "n_distinct_values": int(len(np.unique(v[ok])))}
    return out


def write_standard_files(out_dir, test_ds, pe, pi, y_endo, y_int, me, mi):
    """k=0 results under the names train_introspector_joint writes, so aggregators that
    read a run dir read a probe dir the same way."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "endogenous_test_metrics_endogenous.json", "w") as f:
        json.dump(me, f, indent=2)
    with open(out_dir / "interventional_test_metrics_gaussian.json", "w") as f:
        json.dump({"distribution": "gaussian", "test": mi}, f, indent=2)
    prov = {c: list(test_ds[c]) for c in ("signal_layer", "signal_index", "token_position")}
    pd.DataFrame({"predictions": pe, "activation": y_endo, **prov}).to_csv(
        out_dir / "endogenous_test_predictions_endogenous.csv", index=False)
    pd.DataFrame({"predictions": pi, "injected": y_int, **prov}).to_csv(
        out_dir / "interventional_test_predictions_gaussian.csv", index=False)


def load_probe_into_head(model, probe):
    head = model.regression_head
    with torch.no_grad():
        head.weight.copy_(probe.coef.view(1, -1).to(device=head.weight.device,
                                                    dtype=head.weight.dtype))
        head.bias.fill_(probe.intercept)


def out_dir_for(args, spec):
    return Path(args.baseline_root) / f"prompt_{locale_tag(*spec.locales[0])}_POS{spec.position}"


# ======================================================================
# One run
# ======================================================================

def run_one(spec, model, tokenizer, args, logits_kw, ood_prepared=None):
    config = spec.config
    out_dir = out_dir_for(args, spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    tc = config.training_config
    max_len = tc.max_seq_length
    batch_size = args.batch_size or tc.eval_batch_size

    cache = config.dataset_cache_dir
    train_full = apply_signal_filter(load_from_disk(str(cache / "train")),
                                     config.signal_filter, "train")
    # stats from the WHOLE filtered train split, as the trained run computed them, so the
    # injection distributions are the run's own
    ranges = compute_signal_ranges(train_full)
    n_train = min(args.max_train_examples, len(train_full))
    train_raw = train_full.select(range(n_train))     # contiguous prefix, as the trainers cap
    test_raw = apply_signal_filter(load_from_disk(str(cache / "test")), config.signal_filter, "test")
    if args.max_test_examples is not None and args.max_test_examples < len(test_raw):
        test_raw = test_raw.select(range(args.max_test_examples))
    configure_run(model, config, ranges)

    collator = UngatedCollator(tokenizer=tokenizer, model=model, padding=True,
                               label_pad_token_id=-100, return_tensors="pt", for_prediction=True)
    preprocess = make_preprocess(tokenizer, config, args.separator)
    train_ds = prepare_split(train_raw, preprocess, "train", max_len)
    test_ds = prepare_split(test_raw, preprocess, "test", max_len)
    for i in range(2):
        print(f"[prompt] example prompt {i}: {tokenizer.decode(test_ds[i]['input_ids'])!r}")

    extract = dict(collator=collator, tokenizer=tokenizer, batch_size=batch_size,
                   k_values=args.k_values, logits_kw=logits_kw)
    print(f"[prompt] features: train {len(train_ds)} rows, test {len(test_ds)} rows, "
          f"k={args.k_values}")
    train_endo = extract_features(model, train_ds, **extract)
    # training-distribution injection (the run's own, e.g. extreme x10), seeded apart from
    # the test draw
    train_int = (extract_features(model, train_ds, **extract, intervene=True,
                                  seed=config.seed + 1)
                 if "joint" in args.probes else None)
    test_endo = extract_features(model, test_ds, **extract)
    test_int = extract_features(model, test_ds, **extract, intervene=True,
                                distribution="gaussian", scale=1.0, seed=config.seed)

    tr_idx, val_idx = split_indices(len(train_ds), args.val_frac, config.seed)
    keys = feature_keys(args.k_values)
    rows, fitted = fit_and_score(args, "templated", keys, train_endo, train_int,
                                 test_endo, test_int, tr_idx, val_idx)

    # ---- k=0 results per probe, under the standard names ----
    for name in args.probes:
        probe, pe, pi = fitted[(name, ("mean", 0))]
        me = compute_metrics_from_numericals(pe, test_endo["target"].numpy())
        mi = compute_metrics_from_numericals(pi, test_int["target"].numpy())
        write_standard_files(out_dir / f"{name}_probe", test_ds, pe, pi,
                             test_endo["target"].numpy(), test_int["target"].numpy(), me, mi)

    # ---- verbal readout (no probe) ----
    verbal = verbal_report(test_endo, test_int)
    if verbal:
        with open(out_dir / "verbal_report.json", "w") as f:
            json.dump(verbal, f, indent=2)
        n_s = min(50, len(test_ds))
        pd.DataFrame({
            "prompt_tail": [tokenizer.decode(test_ds[i]["input_ids"][-24:]) for i in range(n_s)],
            "continuation": test_endo["gen"][:n_s],
            "activation": test_endo["target"][:n_s].numpy(),
            "continuation_injected": test_int["gen"][:n_s],
            "injected": test_int["target"][:n_s].numpy(),
        }).to_csv(out_dir / "generations_sample.csv", index=False)
        print(f"[prompt] verbal readout: {verbal}")

    # ---- learning curve (k=0): is the probe data-limited? ----
    curve_rows = []
    pool = tr_idx[np.random.RandomState(config.seed + 2).permutation(len(tr_idx))]
    for n in args.learning_curve:
        if n > len(pool):
            print(f"[prompt] learning curve: skipping n={n} > pool {len(pool)}")
            continue
        sub = np.sort(pool[:n])
        for name in args.probes:
            probe, diag = fit_probe(probe_blocks(name, train_endo, train_int, ("mean", 0),
                                                 args.joint_loss_weight), sub, val_idx)
            me, mi, _, _ = score_probe(probe, test_endo, test_int, ("mean", 0))
            curve_rows.append({"probe": name, "n_train": n, "lambda": diag["lambda"],
                               "endo_corr": me["corr"], "interv_corr": mi["corr"]})
    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(out_dir / "learning_curve.csv", index=False)
        print(pd.DataFrame(curve_rows).to_string(index=False))

    # ---- control: no prompt, readout at the text's last token (k=0 only) ----
    if args.no_prompt_control:
        pre0 = make_preprocess(tokenizer, config, args.separator, no_prompt=True)
        tr0 = prepare_split(train_raw, pre0, "train/no_prompt", max_len)
        te0 = prepare_split(test_raw, pre0, "test/no_prompt", max_len)
        ex0 = dict(extract, k_values=[0])
        c_train_endo = extract_features(model, tr0, **ex0)
        c_train_int = (extract_features(model, tr0, **ex0, intervene=True, seed=config.seed + 1)
                       if "joint" in args.probes else None)
        c_test_endo = extract_features(model, te0, **ex0)
        c_test_int = extract_features(model, te0, **ex0, intervene=True,
                                      distribution="gaussian", scale=1.0, seed=config.seed)
        c_tr, c_val = split_indices(len(tr0), args.val_frac, config.seed)
        c_rows, _ = fit_and_score(args, "none", [("mean", 0)], c_train_endo, c_train_int,
                                  c_test_endo, c_test_int, c_tr, c_val)
        rows += c_rows

    # ---- diagnostics through the training pipeline's own evals (k=0) ----
    head_probe = fitted[("joint" if "joint" in args.probes else "endo", ("mean", 0))][0]
    load_probe_into_head(model, head_probe)
    if not args.skip_alignment_check:
        # the regression test for the separator: the live neuron at the cached position
        # must reproduce the cache (adapter-off and adapter-on are the same model here)
        measure_activation_drift(model, test_ds, collator, config, out_dir)
    scales = [float(s) for s in args.scale_sweep.split(",") if s.strip()]
    if scales:
        interventional_scale_sweep(model, test_ds, collator, config,
                                   out_dir / f"{'joint' if 'joint' in args.probes else 'endo'}_probe",
                                   scales=scales, n_pca=args.routing_probe_pca)
    if args.ood_eval:
        from evals.score_ood_generalization import evaluate_ood, DEFAULT_CORPORA
        corpora = ([c.strip() for c in args.ood_corpora.split(",")]
                   if args.ood_corpora else list(DEFAULT_CORPORA))
        def ood_preprocess(ex):
            # row flags are not model kwargs, and evaluate_ood collates its rows as-is
            out = preprocess(ex)
            for flag in ("prefix_ok", "template_merge", "prompt_truncated",
                         "position_valid", "text_truncated"):
                out.pop(flag)
            return out
        for name in args.probes:
            load_probe_into_head(model, fitted[(name, ("mean", 0))][0])
            evaluate_ood(model, config, tokenizer, collator, preprocess_fn=ood_preprocess,
                         output_dir=out_dir / f"{name}_probe", corpora=corpora,
                         prepared=(ood_prepared or {}).get(spec.name))

    # ---- the main table last: it is the --skip-existing sentinel ----
    by_k = pd.DataFrame(rows)
    stype, layer, index = spec.locales[0]
    meta = {"source_config": str(spec.config_path), "model": config.model_name,
            "signal_type": stype, "layer": layer, "index": index, "position": spec.position,
            "separator": args.separator, "prompt_type": tc.prompt_type,
            "n_train_rows": len(train_ds), "n_test_rows": len(test_ds),
            "k_values": args.k_values, "probes": args.probes,
            "joint_loss_weight": args.joint_loss_weight,
            "joint_interv_distribution": tc.intervention_signal_distribution,
            "joint_interv_scale": tc.intervention_extreme_scale,
            "val_frac": args.val_frac}
    with open(out_dir / "run_spec.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "prompt_baseline_metrics.json", "w") as f:
        json.dump({**meta, "table": rows, "verbal": verbal}, f, indent=2, default=float)
    by_k.to_csv(out_dir / SENTINEL, index=False)
    print(f"[prompt] wrote {out_dir}")
    return by_k


def main():
    args = parse_args()
    runs = runs_from_args(args, require_checkpoint=False)   # nothing trained is loaded
    todo = []
    for spec in runs:
        if spec.is_multi_locale or spec.position is None:
            print(f"[prompt] {spec.name}: multi-locale or random-position run, not supported; skipping")
            continue
        if args.skip_existing and (out_dir_for(args, spec) / SENTINEL).exists():
            print(f"[prompt] {spec.name}: results exist, skipping")
            continue
        todo.append(spec)
    if not todo:
        print("[prompt] nothing to do")
        return
    models = {s.config.model_name for s in todo}
    if len(models) > 1:
        raise SystemExit(f"runs span several base models {sorted(models)}; select one group")

    # OOD caches are built by a GPU subprocess, so they must exist before the model loads
    ood_prepared = None
    if args.ood_eval:
        from evals.score_ood_generalization import prepare_ood_caches, DEFAULT_CORPORA
        corpora = ([c.strip() for c in args.ood_corpora.split(",")]
                   if args.ood_corpora else list(DEFAULT_CORPORA))
        ood_prepared = {s.name: prepare_ood_caches(s.config, corpora=corpora) for s in todo}

    model, tokenizer = load_frozen_model(todo[0].config.model_name,
                                         todo[0].config.training_config.bf16,
                                         local_files_only=args.local_files_only)
    logits_kw = _logits_kwargs(model)
    for i, spec in enumerate(todo, 1):
        print(f"\n[prompt] ===== [{i}/{len(todo)}] {spec.label} ({spec.name}) =====")
        run_one(spec, model, tokenizer, args, logits_kw, ood_prepared)


if __name__ == "__main__":
    main()
