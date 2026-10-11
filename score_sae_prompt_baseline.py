#!/usr/bin/env python3
"""
Prompted baseline for SAE-feature detection: can the UNTRAINED gemma-2-2b report
whether one of its Gemma Scope features is active, when asked?

The SAE analogue of score_prompt_baseline.py (the neuron track's baseline). Same caches,
feature, token position and introspection prompts as a trained grid run
(train_sae_detector_joint.py), but the LM is never trained. Every readout makes a hard
active/inactive call, scored exactly as the trained runs are: balanced accuracy (BA) and
precision on the endogenous test split (natural ~1% base rate) and on the interventional
test (every test text steered on to the median active value and off to 0, 50/50).

READOUTS
  verbal@k   the words the model WRITES in its first k greedy tokens, matched against
             fixed paraphrase lists (src/introspector/verbal_lexicon.py): active-like,
             inactive-like (incl. negation, "not active"), or a number (0 -> inactive,
             >0 -> active). Anything else is NEITHER and counts as a miss, as in the
             trained runs. No threshold, no training: the pure prompting readout.
  logit      at the readout token (k=0): logsumexp of the next-token logits over the
             lexicon's active-like first tokens minus the inactive-like ones ("lex"),
             and plain logit(" active") - logit(" inactive") ("tok").
  probe@k    logistic regression on the final-layer hidden state, mean-pooled over
             [last prompt token, gen_1 .. gen_k] (k=0 is what a trained head reads).
             Fit on the ENDOGENOUS label only: the joint (interventional-row) probe did
             nothing for neurons and is left out.
  no-prompt  the same probe on the text alone, read at its last token (k=0). At
             position -1 that IS the introspected token's own final-layer state -- a
             "the information is there" ceiling, not a bypass control.

THRESHOLDS. logit and probe scores are continuous, so each also reports AUROC
(threshold-free), and BA/precision at the threshold that maximizes BA on a VALIDATION
slice of the TRAIN split, applied unchanged to test (endogenous and interventional
alike: the trained model has one readout, so the baseline gets one threshold). The
configs' dev split is 2000 texts (~2-20 active rows at 0.1-1% density), too few to
place a threshold; train holds 150k texts. Probe and validation rows: every active train
row up to --max-probe-pos plus --probe-neg inactive rows, split by subsample_u.

PROMPTS (--prompt). `index` is the trained grid's prompt (prompt_no_token_index: "SAE
feature 13054 at layer 10 ..."). `label` names the feature by its Neuronpedia
explanation (prompt_sae_label, as in the feature-generalization label arm), labels from
--labels-file. NO ADDED VOCAB: trained runs put a learned <|introspection|> token between
the text and the prompt; to an untrained model it is a random embedding, so plain
--separator text joins them here (as in score_prompt_baseline.py).

STEERING. Difference-of-decode at the cached position (SAEInterventionHook), applied
during the PREFILL only; generated tokens see it through the KV cache.

EXTRAS: learning curve (k=0 probe vs number of active training rows), steering-scale
sweep (k=0: endo probe, logit, and a FRESH probe fit on steered rows -- whether the
injected direction is linearly visible at the readout at all), alignment check (live
feature at the cached position vs the cache; want offset +0, corr ~1).

OUTPUTS under <baseline-root>/<prompt>/<S..>_POS<p>/ (one dir per feature x position;
the grid's W arms share it, since nothing here is trained):
  baseline_by_k.csv          one row per (readout, k, eval_type): the main table
  learning_curve.csv, scale_sweep.csv, verbal_parse_stats.csv, generations_sample.csv
  sae_baseline_alignment.json, sae_prompt_baseline.json (everything + settings)

  python score_sae_prompt_baseline.py --config experiment_configs/sae_layerpos_joint/<run>.json
  python score_sae_prompt_baseline.py --group sae_layerpos_joint --run-glob '*_w09' \
      --prompt label --labels-file scripts/sae/sweep_feature_labels.json --skip-existing
"""
from __future__ import annotations

import json
import math
import random
import argparse
import operator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.introspector.dataset_utils import apply_signal_filter
from src.introspector.models.signal_paths import resolve_sae_hook_path
from src.introspector.prompts import get_prompt_fn
from src.introspector.run_discovery import (
    add_discovery_args, load_run, locale_tag, resolve_run_action, runs_from_args, shard,
)
from src.introspector.sae_featgen import ensure_subsample_u
from src.introspector.signals.sae import load_sae_for_spec
from src.introspector.training.model import SAEInterventionHook
from src.introspector.verbal_lexicon import first_token_ids, parse_continuation
from train_introspector import preprocess_example

DEFAULT_ROOT = Path("outputs/runs/sae_prompt_baseline")
SENTINEL = "baseline_by_k.csv"
CS = np.logspace(-4, 1, 6)            # logistic-regression inverse penalties tried on val
ACTIVE, INACTIVE, NEITHER = 1, 0, -1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_discovery_args(p)
    p.add_argument("--configs", type=Path, nargs="+", default=None,
                   help="Several run configs (any groups) in place of --config/--group; "
                        "sharded over the array like a discovered group.")
    p.add_argument("--prompt", choices=["index", "label"], default="index")
    p.add_argument("--labels-file", type=Path, default=None,
                   help="{'labels': {'S<layer>.<index>': label}} (fetch_sae_feature_labels.py).")
    p.add_argument("--baseline-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--separator", default="\n\n",
                   help="Plain-text join between the text and the prompt (the template puts a "
                        "space before it). Escapes like \\n are interpreted.")
    p.add_argument("--k-values", default="0,1,2,4,8,16",
                   help="Generated-token counts; one greedy generation of max(k) serves all.")
    p.add_argument("--max-probe-pos", type=int, default=1500,
                   help="Active train rows for the probe + validation (all of them if fewer).")
    p.add_argument("--probe-neg", type=int, default=6000)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--max-test-examples", type=int, default=None,
                   help="Default: the whole test split, as the trained runs are scored.")
    p.add_argument("--learning-curve", default="25,50,100,200,500,1000",
                   help="Active-row counts for the k=0 probe learning curve ('' disables).")
    p.add_argument("--scale-sweep", default="1,3,10,30",
                   help="Steering scales (x the median active value) for the k=0 sweep ('' disables).")
    p.add_argument("--scale-sweep-texts", type=int, default=2000)
    p.add_argument("--no-prompt-control", action="store_true")
    p.add_argument("--batch-size", type=int, default=None, help="Default: the config's eval_batch_size.")
    p.add_argument("--n-samples", type=int, default=300, help="Generations written to the sample file.")
    p.add_argument("--ignore-misalignment", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()
    if "\\" in args.separator:
        args.separator = args.separator.encode().decode("unicode_escape")
    args.k_values = sorted({0} | {int(k) for k in args.k_values.split(",") if k.strip()})
    args.learning_curve = [int(n) for n in args.learning_curve.split(",") if n.strip()]
    args.scale_sweep = [float(s) for s in args.scale_sweep.split(",") if s.strip()]
    if args.prompt == "label" and not args.labels_file:
        p.error("--prompt label needs --labels-file")
    return args


# ======================================================================
# Model
# ======================================================================

def load_model(model_name, bf16, local_files_only=False):
    """The BASE model, no adapter and no added token: nothing here is trained."""
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, local_files_only=local_files_only,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if (bf16 and torch.cuda.is_available()) else torch.float32)
    model.eval()
    return model, tok


class PrefillOnly:
    """Wrap a steering hook so it fires only on the prefill forward. Decode steps carry a
    length-1 sequence whose positions do not index the text; they see the steered
    residual through the KV cache, as they would had the feature really taken the value."""
    def __init__(self, hook):
        self.hook = hook

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] == 1:
            return output
        return self.hook(module, inputs, output)


class Readout:
    """Hooks on the final norm (hidden state the LM head reads, last position, every
    forward) and on the LM head (prefill logits of the lexicon's first tokens, softcapped
    as the model does), reset per batch."""
    def __init__(self, model, lex_ids):
        base = model.model
        self.lex_ids = torch.as_tensor(lex_ids)
        self.cap = getattr(model.config, "final_logit_softcapping", None)
        self.h, self.logits = [], None
        self.handles = [base.norm.register_forward_hook(self._norm),
                        model.lm_head.register_forward_hook(self._head)]

    def reset(self):
        self.h, self.logits = [], None

    def _norm(self, module, inputs, output):
        self.h.append(output[:, -1].float())

    def _head(self, module, inputs, output):
        if self.logits is None:
            z = output[:, -1, self.lex_ids.to(output.device)].float()
            self.logits = self.cap * torch.tanh(z / self.cap) if self.cap else z

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run_readout(model, tok, readout, rows, k_values, batch_size, module=None, bundle=None,
                feat=None, values=None):
    """Prefill (+ greedy decode to max(k)) for every row.

    rows: list of dicts with input_ids and position (cached token position in the text).
    Steering (module given): feature `feat` clamped to values[i] at row i's position.
    Returns pooled hidden [n, len(k_values), d] (mean over [prompt_last, gen_1..gen_k]),
    lexicon logits [n, n_lex], and generated ids [n, max(k)].
    """
    k_max = max(k_values)
    pooled, logits, gen = [], [], []
    for b in range(0, len(rows), batch_size):
        batch = rows[b:b + batch_size]
        m = max(len(r["input_ids"]) for r in batch)
        pad = [m - len(r["input_ids"]) for r in batch]
        ii = torch.tensor([[tok.pad_token_id] * p + list(r["input_ids"]) for r, p in zip(batch, pad)],
                          device=model.device)
        am = torch.tensor([[0] * p + [1] * len(r["input_ids"]) for r, p in zip(batch, pad)],
                          device=model.device)
        handle = None
        if module is not None:
            pos = torch.tensor([r["position"] + p for r, p in zip(batch, pad)], device=model.device)
            vals = torch.as_tensor(values[b:b + len(batch)], dtype=torch.float32, device=model.device)
            handle = module.register_forward_hook(PrefillOnly(SAEInterventionHook(bundle, pos, feat, vals)))
        readout.reset()
        try:
            if k_max == 0:
                # last-position logits only: all-position logits are ~5 GB at batch 128
                model(input_ids=ii, attention_mask=am, logits_to_keep=1)
                out_ids = torch.zeros((len(batch), 0), dtype=torch.long)
            else:
                # k_max+1 new tokens: the last generated token is never fed back, so this is
                # what yields hidden states for gen_1..gen_kmax. min_new_tokens keeps every
                # batch the same length (an early EOS would otherwise stop the loop).
                g = model.generate(input_ids=ii, attention_mask=am, max_new_tokens=k_max + 1,
                                   min_new_tokens=k_max + 1, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
                out_ids = g[:, m:m + k_max].cpu()
        finally:
            if handle is not None:
                handle.remove()
        H = torch.stack(readout.h[:k_max + 1], dim=1)           # [bs, k_max+1, d]
        csum = H.cumsum(dim=1)
        pooled.append(torch.stack([csum[:, k] / (k + 1) for k in k_values], dim=1).half().cpu())
        logits.append(readout.logits.cpu())
        gen.append(out_ids)
    return torch.cat(pooled).float().numpy(), torch.cat(logits).numpy(), torch.cat(gen).numpy()


# ======================================================================
# Data
# ======================================================================

def drop_retok_mismatches(raw, tok, separator, split):
    """Keep rows whose cached token still sits at token_position once the separator
    follows the text (BPE can merge a final text token with what comes after it)."""
    def ok(batch):
        enc = tok([f"{t} {separator}" for t in batch["text"]])["input_ids"]
        return [p < len(ids) and tok.decode(ids[p]) == t
                for ids, p, t in zip(enc, batch["token_position"], batch["token"])]
    kept = raw.filter(ok, batched=True)
    dropped = len(raw) - len(kept)
    print(f"[baseline] {split}: dropped {dropped}/{len(raw)} rows whose token changes "
          f"after the separator")
    if len(raw) and dropped / len(raw) > 0.05:
        raise SystemExit(f"{split}: {dropped / len(raw):.1%} of rows misalign; check --separator")
    return kept


def build_rows(raw, tok, prompt_fn, separator, max_len, no_prompt=False):
    """Token ids + position (+ activation, u) per row. no_prompt: the text alone."""
    rows = []
    for ex in raw:
        if no_prompt:
            ids = tok(ex["text"], add_special_tokens=True)["input_ids"]
        else:
            ids = preprocess_example(ex, tokenizer=tok, prompt_fn=prompt_fn, max_length=max_len,
                                     for_prediction=True, detection=True,
                                     separator=separator)["input_ids"]
            if len(ids) >= max_len:
                raise SystemExit(f"prompt reached {max_len} tokens and was truncated")
        rows.append({"input_ids": ids, "position": int(ex["token_position"]), "token": ex["token"],
                     "activation": float(ex["activation"]), "u": float(ex["subsample_u"])})
    return rows


def probe_split(train_raw, max_pos, n_neg, val_frac):
    """Indices into train_raw: (fit, val), each class split by subsample_u order."""
    act = np.asarray(train_raw["activation"], float)
    u = np.asarray(train_raw["subsample_u"], float)
    pos = np.flatnonzero(act > 0)
    neg = np.flatnonzero(act <= 0)
    pos = pos[np.argsort(u[pos])][:max_pos]
    neg = neg[np.argsort(u[neg])][:n_neg]
    if len(pos) < 8:
        raise SystemExit(f"only {len(pos)} active train rows; too few for a probe")
    nv_p, nv_n = max(2, int(len(pos) * val_frac)), max(2, int(len(neg) * val_frac))
    val = np.concatenate([pos[:nv_p], neg[:nv_n]])
    fit = np.concatenate([pos[nv_p:], neg[nv_n:]])
    return fit, val


# ======================================================================
# Scoring
# ======================================================================

def hard_metrics(pred, y):
    """BA / precision / TPR / TNR / neither-rate of hard calls (1, 0, or -1 = neither).
    'neither' is a miss on either class, as in the trained runs' generation_metrics."""
    pred, y = np.asarray(pred), np.asarray(y, int)
    out = {"n": int(len(y)), "base_rate_positive": float(y.mean()) if len(y) else float("nan"),
           "neither_rate": float(np.mean(pred == NEITHER)) if len(y) else float("nan")}
    if not (y == 1).any() or not (y == 0).any():
        return {**out, "balanced_accuracy": float("nan"), "precision": float("nan")}
    tpr = float(np.mean(pred[y == 1] == ACTIVE))
    tnr = float(np.mean(pred[y == 0] == INACTIVE))
    called = pred == ACTIVE
    prec = float(np.mean(y[called] == 1)) if called.any() else float("nan")
    return {**out, "balanced_accuracy": 0.5 * (tpr + tnr), "precision": prec, "tpr": tpr, "tnr": tnr}


def best_ba_threshold(scores, y):
    """Threshold maximizing balanced accuracy on (scores, y): the ROC point with the
    largest (TPR + TNR) / 2."""
    fpr, tpr, thr = roc_curve(y, scores)
    i = int(np.argmax(tpr + (1 - fpr)))
    return float(thr[i])


def scored_metrics(scores, y, thr):
    out = hard_metrics(np.where(scores >= thr, ACTIVE, INACTIVE), y)
    yy = np.asarray(y, int)
    out["auroc"] = float(roc_auc_score(yy, scores)) if 0 < yy.mean() < 1 else float("nan")
    out["threshold"] = float(thr)
    return out


class Probe:
    """Standardize -> class-balanced logistic regression, C chosen by validation AUROC,
    threshold by validation BA."""
    def fit(self, Xf, yf, Xv, yv, Cs=CS):
        self.mu, self.sd = Xf.mean(0), Xf.std(0) + 1e-6
        best = (-1.0, None, None)
        for C in Cs:
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
            m.fit(self._z(Xf), yf)
            a = roc_auc_score(yv, m.decision_function(self._z(Xv)))
            if a > best[0]:
                best = (a, C, m)
        self.val_auroc, self.C, self.model = best
        self.threshold = best_ba_threshold(self.score(Xv), yv)
        return self

    def _z(self, X):
        return (X - self.mu) / self.sd

    def score(self, X):
        return self.model.decision_function(self._z(X))


def verbal_preds(gen_ids, tok, k):
    """Hard calls + parse records from the first k generated tokens of each row."""
    preds, recs = [], []
    for ids in gen_ids[:, :k]:
        text = tok.decode(ids, skip_special_tokens=True)
        cls, word = parse_continuation(text)
        preds.append({"active": ACTIVE, "inactive": INACTIVE}.get(cls, NEITHER))
        recs.append(word)
    return np.array(preds), recs


# ======================================================================
# One run
# ======================================================================

def alignment_check(model, tok, rows, spec, bundle, module, batch_size, out_dir, n=512):
    """Live feature value at the cached position (offsets -1/0/+1) vs the cached value,
    and the token there vs the cached token, on the prompted sequences this baseline
    reads. Want token match ~1, and (given >= 10 active rows) best offset +0, corr ~1."""
    act = np.array([r["activation"] for r in rows])
    pos_i, neg_i = np.flatnonzero(act > 0), np.flatnonzero(act <= 0)
    n_pos = min(len(pos_i), n // 2)
    sel = np.sort(np.concatenate([pos_i[:n_pos], neg_i[:n - n_pos]]))
    live = {o: [] for o in (-1, 0, 1)}
    match = []
    for b in range(0, len(sel), batch_size):
        batch = [rows[i] for i in sel[b:b + batch_size]]
        m = max(len(r["input_ids"]) for r in batch)
        pad = [m - len(r["input_ids"]) for r in batch]
        ii = torch.tensor([[tok.pad_token_id] * p + list(r["input_ids"]) for r, p in zip(batch, pad)],
                          device=model.device)
        am = (ii != tok.pad_token_id).long()
        for j, p in enumerate(pad):
            am[j, p:] = 1
        abs_pos = torch.tensor([r["position"] + p for r, p in zip(batch, pad)], device=model.device)
        got = {}

        def cap(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            bi = torch.arange(h.shape[0], device=h.device)
            for o in live:
                got[o] = h[bi, (abs_pos + o).clamp(0, h.shape[1] - 1)].float()

        hd = module.register_forward_hook(cap)
        try:
            with torch.no_grad():
                model(input_ids=ii, attention_mask=am)
        finally:
            hd.remove()
        for o in live:
            live[o].append(bundle.encode(got[o])[:, spec.index].float().cpu().numpy())
        match += [tok.decode(ii[j, abs_pos[j]]) == r["token"] for j, r in enumerate(batch)]
    cached = act[sel]
    corr = {}
    for o in live:
        v = np.concatenate(live[o])
        corr[o] = float(np.corrcoef(v, cached)[0, 1]) if v.std() > 1e-9 and cached.std() > 1e-9 else float("nan")
    best = max(corr, key=lambda o: -np.inf if np.isnan(corr[o]) else corr[o])
    token_match = float(np.mean(match))
    ok = token_match > 0.99 and (n_pos < 10 or (best == 0 and corr[0] > 0.99))
    if n_pos < 10:
        print(f"[baseline] alignment: only {n_pos} active rows; token match only")
    res = {"n": int(len(sel)), "n_active": int(n_pos), "token_match": token_match,
           "corr_by_offset": {str(k): v for k, v in corr.items()},
           "best_offset": int(best), "aligned": bool(ok)}
    (out_dir / "sae_baseline_alignment.json").write_text(json.dumps(res, indent=2))
    print(f"[baseline] alignment: token match {token_match:.4f}, corr by offset {corr} -> "
          f"{'OK' if ok else 'MISALIGNED'}")
    return res


def score_run(spec, model, tok, args):
    config = spec.config
    tc = config.training_config
    if args.prompt == "label":
        tc = tc.model_copy(update={"prompt_type": "prompt_sae_label",
                                   "feature_labels": json.loads(args.labels_file.read_text())["labels"]})
    prompt_fn = get_prompt_fn(tc)
    tag = locale_tag(spec.signal_type, spec.layer, spec.index)
    out_dir = args.baseline_root / args.prompt / f"{tag}_POS{spec.position}"
    out_dir.mkdir(parents=True, exist_ok=True)
    bs = args.batch_size or tc.eval_batch_size
    max_len = config.train_dataset.max_seq_length + 128
    print(f"\n[baseline] === {tag} pos {spec.position}  prompt={args.prompt}  -> {out_dir}")

    cache = config.dataset_cache_dir
    filt = config.signal_filter or tag
    train_raw = apply_signal_filter(load_from_disk(str(cache / "train")), filt, "train")
    test_raw = apply_signal_filter(load_from_disk(str(cache / "test")), filt, "test")
    train_raw = ensure_subsample_u(drop_retok_mismatches(train_raw, tok, args.separator, "train"), config.seed)
    test_raw = ensure_subsample_u(drop_retok_mismatches(test_raw, tok, args.separator, "test"), config.seed)
    if args.max_test_examples and len(test_raw) > args.max_test_examples:
        order = np.argsort(np.asarray(test_raw["subsample_u"]))[:args.max_test_examples]
        test_raw = test_raw.select(np.sort(order).tolist())
    fit_i, val_i = probe_split(train_raw, args.max_probe_pos, args.probe_neg, args.val_frac)
    probe_raw = train_raw.select(np.concatenate([fit_i, val_i]).tolist())
    n_fit = len(fit_i)

    sae_spec = next(s for s in config.train_signals if s.type == "sae" and s.index == spec.index
                    and s.layer == spec.layer)
    bundle = load_sae_for_spec(sae_spec, device=str(model.device))
    module = operator.attrgetter(resolve_sae_hook_path(model, spec.layer, site=sae_spec.sae_site,
                                                       path_to_output=False))(model)
    train_act = np.asarray(train_raw["activation"], float)
    on_value = float(np.median(train_act[train_act > 0]))

    random.seed(config.seed + 1)          # template draws, identical across prompts arms
    probe_rows = build_rows(probe_raw, tok, prompt_fn, args.separator, max_len)
    random.seed(config.seed + 2)
    test_rows = build_rows(test_raw, tok, prompt_fn, args.separator, max_len)
    y_probe = (np.array([r["activation"] for r in probe_rows]) > 0).astype(int)
    y_test = (np.array([r["activation"] for r in test_rows]) > 0).astype(int)
    yf, yv = y_probe[:n_fit], y_probe[n_fit:]

    # on the probe rows: every active train row (up to --max-probe-pos), where the test
    # split of a sparse feature can hold almost none
    align = alignment_check(model, tok, probe_rows, sae_spec, bundle, module, bs, out_dir)
    if not align["aligned"] and not args.ignore_misalignment:
        raise SystemExit("alignment check failed; pass --ignore-misalignment to continue")

    lex_ids, lex_sign, tok_pair = first_token_ids(tok)
    readout = Readout(model, lex_ids)
    K = args.k_values
    try:
        print(f"[baseline] probe rows: {len(probe_rows)} ({int(yf.sum())}+{int(yv.sum())} active)")
        Hp, Lp, Gp = run_readout(model, tok, readout, probe_rows, K, bs)
        print(f"[baseline] test endogenous: {len(test_rows)} rows ({int(y_test.sum())} active)")
        He, Le, Ge = run_readout(model, tok, readout, test_rows, K, bs)
        print(f"[baseline] test interventional: 2 x {len(test_rows)} steered rows (on={on_value:.3g})")
        n = len(test_rows)
        Hon, Lon, Gon = run_readout(model, tok, readout, test_rows, K, bs, module, bundle,
                                    spec.index, np.full(n, on_value))
        Hoff, Loff, Goff = run_readout(model, tok, readout, test_rows, K, bs, module, bundle,
                                       spec.index, np.zeros(n))
    finally:
        readout.remove()
    Hi, Li, Gi = np.concatenate([Hon, Hoff]), np.concatenate([Lon, Loff]), np.concatenate([Gon, Goff])
    y_int = np.r_[np.ones(n, int), np.zeros(n, int)]

    table, parse_stats = [], []

    def add(readout_name, k, eval_type, m, **extra):
        table.append({"readout": readout_name, "k": k, "eval_type": eval_type, **m, **extra})

    # ---- verbal@k: no threshold
    for k in [k for k in K if k > 0]:
        for et, G, y in (("endogenous", Ge, y_test), ("interventional", Gi, y_int)):
            pred, words = verbal_preds(G, tok, k)
            add("verbal", k, et, hard_metrics(pred, y))
            wc = pd.Series(words).fillna("<none>").value_counts().head(15)
            parse_stats += [{"k": k, "eval_type": et, "word": w, "count": int(c)} for w, c in wc.items()]

    # ---- logit readouts at the readout token: threshold from validation
    def lex_score(L):
        act_l, ina_l = L[:, lex_sign > 0], L[:, lex_sign < 0]
        return (np.logaddexp.reduce(act_l, axis=1) - np.logaddexp.reduce(ina_l, axis=1))

    a_i, i_i = tok_pair
    for name, f in (("logit_lex", lex_score), ("logit_tok", lambda L: L[:, a_i] - L[:, i_i])):
        thr = best_ba_threshold(f(Lp[n_fit:]), yv)
        add(name, 0, "endogenous", scored_metrics(f(Le), y_test, thr))
        add(name, 0, "interventional", scored_metrics(f(Li), y_int, thr))

    # ---- probe@k
    probes = {}
    for j, k in enumerate(K):
        pr = Probe().fit(Hp[:n_fit, j], yf, Hp[n_fit:, j], yv)
        probes[k] = pr
        extra = {"C": pr.C, "val_auroc": pr.val_auroc}
        add("probe", k, "endogenous", scored_metrics(pr.score(He[:, j]), y_test, pr.threshold), **extra)
        add("probe", k, "interventional", scored_metrics(pr.score(Hi[:, j]), y_int, pr.threshold), **extra)

    # ---- no-prompt control (k=0)
    if args.no_prompt_control:
        readout = Readout(model, lex_ids)
        try:
            pr0 = build_rows(probe_raw, tok, None, args.separator, max_len, no_prompt=True)
            te0 = build_rows(test_raw, tok, None, args.separator, max_len, no_prompt=True)
            H0p, _, _ = run_readout(model, tok, readout, pr0, [0], bs)
            H0e, _, _ = run_readout(model, tok, readout, te0, [0], bs)
            H0on, _, _ = run_readout(model, tok, readout, te0, [0], bs, module, bundle, spec.index, np.full(n, on_value))
            H0off, _, _ = run_readout(model, tok, readout, te0, [0], bs, module, bundle, spec.index, np.zeros(n))
        finally:
            readout.remove()
        pr = Probe().fit(H0p[:n_fit, 0], yf, H0p[n_fit:, 0], yv)
        extra = {"C": pr.C, "val_auroc": pr.val_auroc}
        add("noprompt_probe", 0, "endogenous", scored_metrics(pr.score(H0e[:, 0]), y_test, pr.threshold), **extra)
        add("noprompt_probe", 0, "interventional",
            scored_metrics(pr.score(np.concatenate([H0on, H0off])[:, 0]), y_int, pr.threshold), **extra)

    table = pd.DataFrame(table)
    table.insert(0, "prompt", args.prompt)
    table.insert(1, "signal", tag)
    table.insert(2, "position", spec.position)

    # ---- learning curve over active training rows (k=0 probe)
    lc = []
    pos_f, neg_f = np.flatnonzero(yf == 1), np.flatnonzero(yf == 0)
    for n_pos in [c for c in args.learning_curve if c < len(pos_f)] + [len(pos_f)]:
        n_neg = int(round(len(neg_f) * n_pos / len(pos_f)))
        idx = np.concatenate([pos_f[:n_pos], neg_f[:n_neg]])
        pr = Probe().fit(Hp[idx, 0], yf[idx], Hp[n_fit:, 0], yv)
        lc.append({"n_active": int(n_pos), "n_inactive": n_neg, "C": pr.C,
                   **{f"endo_{k}": v for k, v in scored_metrics(pr.score(He[:, 0]), y_test, pr.threshold).items()
                      if k in ("auroc", "balanced_accuracy")},
                   **{f"interv_{k}": v for k, v in scored_metrics(pr.score(Hi[:, 0]), y_int, pr.threshold).items()
                      if k in ("auroc", "balanced_accuracy")}})
    pd.DataFrame(lc).to_csv(out_dir / "learning_curve.csv", index=False)

    # ---- steering-scale sweep (k=0, prefill only)
    sweep = []
    if args.scale_sweep:
        ns = min(args.scale_sweep_texts, n)
        ts = test_rows[:ns]
        nf = min(len(probe_rows) // 2, args.scale_sweep_texts)
        # by subsample_u, not list order: probe_rows lists the active rows first
        fs = [probe_rows[i] for i in np.argsort([r["u"] for r in probe_rows])[:nf]]
        readout = Readout(model, lex_ids)
        try:
            for s in args.scale_sweep:
                v = s * on_value
                Ht = [run_readout(model, tok, readout, ts, [0], bs, module, bundle, spec.index, np.full(ns, x))
                      for x in (v, 0.0)]
                Hf = [run_readout(model, tok, readout, fs, [0], bs, module, bundle, spec.index, np.full(nf, x))
                      for x in (v, 0.0)]
                Xt = np.concatenate([Ht[0][0][:, 0], Ht[1][0][:, 0]])
                Lt = np.concatenate([Ht[0][1], Ht[1][1]])
                yt = np.r_[np.ones(ns, int), np.zeros(ns, int)]
                Xf = np.concatenate([Hf[0][0][:, 0], Hf[1][0][:, 0]])
                yfs = np.r_[np.ones(nf, int), np.zeros(nf, int)]
                half = np.random.RandomState(config.seed).permutation(len(yfs))
                cut = int(len(half) * 0.75)
                fresh = Probe().fit(Xf[half[:cut]], yfs[half[:cut]], Xf[half[cut:]], yfs[half[cut:]])
                endo = probes[0]
                row = {"scale": s, "on_value": v, "n_test": int(ns)}
                for name, sc, thr in (("probe_endo", endo.score(Xt), endo.threshold),
                                      ("probe_fresh", fresh.score(Xt), fresh.threshold),
                                      ("logit_lex", lex_score(Lt), best_ba_threshold(lex_score(Lp[n_fit:]), yv))):
                    m = scored_metrics(sc, yt, thr)
                    row[f"{name}_auroc"], row[f"{name}_ba"] = m["auroc"], m["balanced_accuracy"]
                sweep.append(row)
        finally:
            readout.remove()
        pd.DataFrame(sweep).to_csv(out_dir / "scale_sweep.csv", index=False)

    # ---- samples + summary
    samp = []
    for et, G, y in (("endogenous", Ge, y_test), ("interventional", Gi, y_int)):
        for i in range(min(args.n_samples // 2, n)):
            # interventional: alternate the steered-on copy of text i and its off copy
            j = i if et == "endogenous" or i % 2 == 0 else n + i
            samp.append({"eval_type": et, "label": int(y[j]),
                         "continuation": tok.decode(G[j], skip_special_tokens=True)})
    pd.DataFrame(samp).to_csv(out_dir / "generations_sample.csv", index=False)
    pd.DataFrame(parse_stats).to_csv(out_dir / "verbal_parse_stats.csv", index=False)
    summary = {"signal": tag, "position": spec.position, "prompt": args.prompt,
               "config": str(spec.config_path), "on_value": on_value,
               "n_probe_fit": int(n_fit), "n_probe_val": int(len(yv)),
               "n_probe_active": int(y_probe.sum()), "n_test": int(n),
               "separator": args.separator, "k_values": K, "alignment": align,
               "table": table.to_dict(orient="records"), "learning_curve": lc, "scale_sweep": sweep}
    (out_dir / "sae_prompt_baseline.json").write_text(json.dumps(summary, indent=2, default=float))
    table.to_csv(out_dir / SENTINEL, index=False)       # last: marks the run complete
    show = table[table["k"].isin([0, max(K)])][["readout", "k", "eval_type", "balanced_accuracy",
                                                 "auroc", "neither_rate"]]
    print(show.to_string(index=False))


def main():
    args = parse_args()
    if args.configs:
        runs = shard(sorted((load_run(c) for c in args.configs), key=lambda r: r.name),
                     args.task_index, args.n_tasks)
    else:
        runs = runs_from_args(args, require_checkpoint=False)
    todo = []
    for spec in runs:
        out_dir = args.baseline_root / args.prompt / f"{locale_tag(spec.signal_type, spec.layer, spec.index)}_POS{spec.position}"
        if resolve_run_action(args, out_dir / SENTINEL) == "skip":
            print(f"[baseline] {spec.name}: results exist, skipping")
            continue
        todo.append(spec)
    if not todo:
        return
    model, tok = load_model(todo[0].config.model_name, todo[0].config.training_config.bf16,
                            args.local_files_only)
    for spec in todo:
        score_run(spec, model, tok, args)


if __name__ == "__main__":
    main()
