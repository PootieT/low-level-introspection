"""
Attention-head knockout machinery shared by the ablation analyses
(score_head_ablation.py, score_head_knockout.py).

  HeadAblator     forward_pre hooks on every o_proj that overwrite selected heads'
                  slices with a per-position-bucket mean (or zero), optionally only
                  inside a token WINDOW (e.g. from the introspected token t to the
                  readout)
  build_mean_cache  the per-layer, per-bucket mean the heads are ablated to
  eval_arm_outputs  one evaluation pass (endogenous or gaussian-injected) under the
                  current ablation -> per-example predictions and targets
  successful_train_examples  the TRAIN rows the mean is taken over

Everything here mirrors the reported evaluation: same forward, the regression head on
the last token's final hidden state, and for injected arms the model's OWN
intervention path with its sampler re-seeded to config.seed, so every ablation config
sees exactly the same injected values (paired deltas).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from src.introspector.analysis_loading import absolute_positions

# token-position buckets the mean is conditioned on
BUCKETS = ("text", "prompt", "final")
PAD = -1

# token windows a knockout can be confined to (see window_mask)
WINDOWS = ("all", "t_to_last", "t")


# ======================================================================
# Arms
# ======================================================================
@dataclass
class Arm:
    name: str
    scale: float | None      # None = endogenous

    @property
    def injected(self) -> bool:
        return self.scale is not None


def parse_arms(spec: str) -> list[Arm]:
    arms = []
    for tok in (t.strip().lower() for t in spec.split(",")):
        if not tok:
            continue
        if tok in ("endogenous", "endo"):
            arms.append(Arm("endogenous", None))
        elif tok.startswith("x"):
            arms.append(Arm(tok, float(tok[1:])))
        else:
            raise SystemExit(f"Unknown arm {tok!r}; use 'endogenous' or 'xN'.")
    return arms


# ======================================================================
# Positions
# ======================================================================
def position_buckets(attention_mask: torch.Tensor, region) -> torch.Tensor:
    """[B, S] bucket index per token: 0 text, 1 prompt, 2 readout, -1 padding.

    `region` is the 0/1 text-vs-prompt mask the collators emit (lora_token_mask when
    gated, anchor_region_mask when not). The readout is the last position: batches are
    left-padded, so it is always a real token.
    """
    B, S = attention_mask.shape
    bk = torch.full((B, S), PAD, dtype=torch.long, device=attention_mask.device)
    real = attention_mask == 1
    if region is None:
        bk[real] = 0
    else:
        region = region.to(attention_mask.device)
        bk[real & (region == 0)] = 0
        bk[real & (region == 1)] = 1
    bk[:, S - 1] = 2
    return bk


def window_mask(window: str, attention_mask: torch.Tensor,
                token_position: torch.Tensor | None) -> torch.Tensor | None:
    """[B, S] bool: where a knockout applies. None = every real token.

      all        every position (full knockout, indirect effects included)
      t_to_last  the introspected token t through the readout, inclusive. Positions
                 before t cannot see the signal under causal attention, so this
                 removes every write a head makes that COULD carry it while leaving the
                 context the signal is computed from untouched.
      t          the introspected token only (local processing at the origin)
    """
    if window == "all":
        return None
    if token_position is None:
        raise ValueError(f"window {window!r} needs the batch's token_position")
    B, S = attention_mask.shape
    t = absolute_positions(token_position.to(attention_mask.device), attention_mask)
    pos = torch.arange(S, device=attention_mask.device).unsqueeze(0)       # [1, S]
    if window == "t_to_last":
        return pos >= t.unsqueeze(1)
    if window == "t":
        return pos == t.unsqueeze(1)
    raise ValueError(f"unknown window {window!r}; expected one of {WINDOWS}")


# ======================================================================
# Head knockout
# ======================================================================
class HeadAblator:
    """
    forward_pre hooks on every o_proj that overwrite the selected heads' slices of the
    concatenated head outputs -- the input o_proj mixes into the residual stream.

    Hooked on o_proj's INPUT so the head's whole write is removed, LoRA delta on
    o_proj included. With GQA the query heads' outputs are still concatenated per query
    head, so a slice is exactly one head. `window` ([B, S] bool, set per batch)
    confines the knockout to those positions; None = every real token. Inactive (no
    heads selected) the hooks return None and cost nothing.
    """

    def __init__(self, run, n_heads: int, head_dim: int, mode: str = "mean"):
        self.run, self.n_heads, self.head_dim, self.mode = run, n_heads, head_dim, mode
        self.active: dict[int, list[int]] = {}
        self.means: dict[int, torch.Tensor] = {}      # layer -> [3, n_heads*head_dim]
        self.buckets: torch.Tensor | None = None       # [B, S], set per batch
        self.window: torch.Tensor | None = None        # [B, S] bool, set per batch
        self.handles = []
        for L, layer in enumerate(run.base.model.layers):
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._hook(L)))

    def set_heads(self, heads):
        self.active = {}
        for L, h in heads:
            self.active.setdefault(int(L), []).append(int(h))

    def _hook(self, layer: int):
        def pre(_module, args):
            heads = self.active.get(layer)
            if not heads or self.buckets is None:
                return None
            x = args[0].clone()
            where = self.buckets >= 0
            if self.window is not None:
                where = where & self.window.to(where.device)
            where = where.unsqueeze(-1)
            if self.mode == "mean":
                m = self.means[layer].to(device=x.device, dtype=x.dtype)
                repl = m[self.buckets.clamp(min=0)]                     # [B, S, D]
            else:
                repl = torch.zeros_like(x)
            for h in heads:
                sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                x[:, :, sl] = torch.where(where, repl[:, :, sl], x[:, :, sl])
            return (x,) + tuple(args[1:])
        return pre

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def build_mean_cache(run, dataset, batch_size, n_layers):
    """Per-layer, per-bucket mean of the o_proj input over `dataset`, endogenously."""
    sums = {L: None for L in range(n_layers)}
    counts = {L: torch.zeros(len(BUCKETS)) for L in range(n_layers)}
    grabbed, handles = {}, []

    def cap(L):
        def pre(_m, args):
            grabbed[L] = args[0].detach().float()
            return None
        return pre

    for L, layer in enumerate(run.base.model.layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(cap(L)))
    try:
        for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=run.collator,
                                shuffle=False):
            batch = batch.to(run.model.device)
            region = batch.get("lora_token_mask", batch.get("anchor_region_mask"))
            bk = position_buckets(batch["attention_mask"], region)
            run.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                      lora_token_mask=batch.get("lora_token_mask"), do_intervention=False)
            for L in range(n_layers):
                x = grabbed[L]                                           # [B, S, D]
                if sums[L] is None:
                    sums[L] = torch.zeros(len(BUCKETS), x.shape[-1])
                for b in range(len(BUCKETS)):
                    sel = bk == b
                    if sel.any():
                        sums[L][b] += x[sel].sum(0).cpu()
                        counts[L][b] += int(sel.sum())
    finally:
        for h in handles:
            h.remove()
    return {L: sums[L] / counts[L].clamp(min=1).unsqueeze(-1) for L in range(n_layers)}


# ======================================================================
# Evaluation (mirrors the reported evals)
# ======================================================================
@torch.no_grad()
def eval_arm_outputs(run, dataset, arm: Arm, ablator: HeadAblator, batch_size: int,
                     window: str = "all") -> tuple[np.ndarray, np.ndarray]:
    """(predictions, targets) for one arm, under whatever heads are currently ablated.

    Same forward as the reported evals: the whole collated batch into the model, the
    regression head on the last token's final hidden state, and for injected arms the
    model's OWN intervention path with the sampler re-seeded to config.seed -- so the
    injected values are identical across every ablation config and to the reported run.
    Targets: the cached activation (endogenous) or the injected value.
    """
    model = run.model
    model.eval()
    prev = (model.intervention_distribution, model.intervention_scale)
    if arm.injected:
        model.intervention_distribution = "gaussian"
        model.intervention_scale = arm.scale
        model.seed_intervention_rng(run.config.seed)
    head = model.regression_head
    preds, targets = [], []
    try:
        for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=run.collator,
                                shuffle=False):
            batch = batch.to(model.device)
            act = batch.pop("activation")
            region = batch.get("lora_token_mask", batch.get("anchor_region_mask"))
            batch.pop("anchor_region_mask", None)   # bookkeeping, not a model input
            batch.pop("labels", None)               # token labels: an LM loss we do not need
            ablator.buckets = position_buckets(batch["attention_mask"], region)
            ablator.window = window_mask(window, batch["attention_mask"],
                                         batch.get("token_position"))
            out = model(**batch, do_intervention=arm.injected, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :]
            preds.append(head(h.to(device=head.weight.device, dtype=head.weight.dtype))
                         .squeeze(-1).float().cpu())
            targets.append(out.intervention_values.float().cpu() if arm.injected
                           else act.float().cpu())
    finally:
        model.intervention_distribution, model.intervention_scale = prev
        ablator.buckets = None
        ablator.window = None
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def safe_corr(p: np.ndarray, t: np.ndarray) -> float:
    if np.std(p) < 1e-12 or np.std(t) < 1e-12:
        return float("nan")
    return float(pearsonr(p, t)[0])


def eval_arm(run, dataset, arm: Arm, ablator: HeadAblator, batch_size: int,
             window: str = "all") -> float:
    """corr(prediction, target) for one arm under the current ablation."""
    return safe_corr(*eval_arm_outputs(run, dataset, arm, ablator, batch_size, window))


def eval_all(run, dataset, arms, ablator, bs, heads, window: str = "all") -> dict:
    ablator.set_heads(heads)
    try:
        return {a.name: eval_arm(run, dataset, a, ablator, bs, window) for a in arms}
    finally:
        ablator.set_heads([])


# ======================================================================
# Mean-cache examples
# ======================================================================
def successful_train_examples(run, args):
    """`args.mean_examples` TRAIN rows the model gets right endogenously.

    Train, not test: the mean is a property of the model's normal operation, and taking
    it from the evaluated rows would let each example's own output leak into the value
    it is ablated to. Falls back to the lowest-error rows (with a warning) if too few
    clear the threshold, rather than silently shrinking the cache.

    Reads args.mean_scan, args.mean_examples, args.success_threshold, args.batch_size.
    """
    from datasets import load_from_disk
    from src.introspector.dataset_utils import apply_signal_filter
    from train_introspector import drop_invalid_positions
    from trace_introspection_circuit import endogenous_predictions
    raw = apply_signal_filter(load_from_disk(str(run.config.dataset_cache_dir / "train")),
                              run.config.signal_filter, "train(mean cache)")
    if run.is_multi_locale:
        # the mean a head is ablated TO should come from the distribution being
        # evaluated -- this signal's rows -- not from a mixture over every signal the
        # run was trained on
        raw = raw.filter(lambda ex: int(ex["signal_layer"]) == run.layer
                         and int(ex["signal_index"]) == run.index)
    raw = raw.select(range(min(args.mean_scan, len(raw))))
    ds = raw.map(run.preprocess_fn, remove_columns=raw.column_names)
    ds = drop_invalid_positions(ds, "train(mean cache)",
                                run.config.training_config.max_seq_length)
    bs = args.batch_size or run.config.training_config.eval_batch_size
    preds, acts = endogenous_predictions(run.model, ds, run.collator, bs)
    err = (preds - acts).abs().numpy()
    ok = np.where(err <= args.success_threshold)[0]
    n_ok = len(ok)
    if n_ok >= args.mean_examples:
        idx = ok[:args.mean_examples]
    else:
        print(f"[ablate] WARNING: only {n_ok} of {len(err)} scanned train rows are within "
              f"{args.success_threshold}; using the {args.mean_examples} lowest-error rows "
              f"instead (raise --mean-scan to keep the threshold)")
        idx = np.argsort(err)[:args.mean_examples]
    info = {"n_scanned": int(len(err)), "n_within_threshold": int(n_ok),
            "n_used": int(len(idx)), "threshold": args.success_threshold,
            "max_error_used": float(err[idx].max()), "split": "train"}
    print(f"[ablate] mean cache: {info}")
    return ds.select(sorted(int(i) for i in idx)), info
