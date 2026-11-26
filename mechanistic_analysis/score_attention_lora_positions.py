#!/usr/bin/env python3
"""
WHERE does the signal layer's attention LoRA help the report? -- a token-resolved
follow-up to mechanistic_analysis/ablate_lora_modules.py (W7 of the rebuttal).

Zeroing the attention LoRA at the signal layer L drops the report far more than any
other block, yet that attention runs BEFORE the signal MLP in the same layer, so it
cannot read the neuron's write. Zeroing its q_proj or k_proj LoRA alone does nothing,
so the effect is in what the heads move (v, o), not where they look. This script asks
which tokens it acts at.

Per batch, layer L's attention is run twice: as finetuned, and with its LoRA zeroed
(q, k, v, o; same input). The finetuned output is kept everywhere except at one group
of positions, where the LoRA-off output is used instead. Groups (they partition the
real tokens):
  t               the introspected token, before the signal MLP reads its input
  text_before_t   earlier text tokens (BOS included)
  text_after_t    later text tokens (empty when t is the last text token)
  prompt          the question tokens, readout excluded
  readout         the last token, where the regression head reads
plus "all" (every token: the block ablation of ablate_lora_modules.py) and "all_but_t".

The gain check. The neuron writes h * W_down[:, i] with h = a * u, u = up_proj's output
for the neuron at t (and a replaced by v under injection). The attention output at t is
part of the MLP's input there, so it sets u. u can flip sign with context
(score_neuron_write_stats.py), and then h no longer tracks a. If the LoRA's job at t is
to keep u stable, the "t" condition should drop corr(v, h) and corr(pred, v) together
while corr(pred, h) holds up.

Readouts per condition (rows and injected values identical across conditions):
  corr_endogenous / corr_interv / corr_natural_under_injection   as in the head-set and
                         write-channel ablations
  corr_v_h               x1: corr(v, h), how faithfully the neuron's write carries v
  corr_interv_pred_h     x1: corr(pred, h), the report against what was actually written
  corr_endo_pred_h       endogenous: corr(pred, h)
  corr_a_live_cached     endogenous: corr(the neuron's live a at t, the cached a); below 1
                         when the edit changes the neuron's own activation
  u_mean, u_std, u_frac_positive, u_corr_none   u at t, and its correlation with the
                         unedited u row by row
  tokens_per_row         mean size of the group

Gain specificity (meta.json; --no-gain-specificity skips it): the target neuron's mean
u at t in the finetuned model, with layer L's attention LoRA removed at t, and in the
base model (every LoRA zeroed), plus where the target's finetuned/other mean-|u| ratio
ranks among all neurons of the layer -- is the boost aimed at this neuron?

OUTPUTS (per run, <run>/attn_lora_positions/)
  attn_lora_positions.csv         one row per condition
  attn_lora_positions_arrays.npz  per-example predictions, u and h per condition
  attn_lora_positions_meta.json   provenance and the gain specificity; written LAST (the
                                  --skip-existing sentinel)

Usage
-----
python mechanistic_analysis/score_attention_lora_positions.py --config <run config>
python mechanistic_analysis/score_attention_lora_positions.py --group mlp_joint_grid_layer_token \\
    --run-glob 'joint_*_kl0.1_*' --cells 2:-1,2:-28,8:-1,8:-28 --skip-existing
"""
from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from src.introspector.analysis_loading import run_per_locale
from src.introspector.head_ablation import Arm, eval_arm_with_activation, safe_corr
from src.introspector.run_discovery import (
    add_discovery_args, discover_runs, filter_cells, load_run, shard)

ANALYSIS = "attn_lora_positions"
SENTINEL = "attn_lora_positions_meta.json"
ARMS = (Arm("endogenous", None), Arm("x1", 1.0))
GROUPS = ("t", "text_before_t", "text_after_t", "prompt", "readout")
CONDITIONS = ("none", "all", *GROUPS, "all_but_t")
ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--cells", type=str, default=None,
                   help="Comma list of 'layer:position' cells kept after discovery, e.g. "
                        "'2:-1,2:-28,8:-1,8:-28'. Default: every run --run-glob matches.")
    p.add_argument("--conditions", type=str, default=",".join(CONDITIONS),
                   help="Comma list of conditions to evaluate (default: all).")
    p.add_argument("--layer-offset", type=int, default=0,
                   help="Edit the attention of layer L + offset (default 0: the signal "
                        "layer). The gain check only means something at offset 0.")
    p.add_argument("--no-gain-specificity", action="store_true",
                   help="Skip the three extra endogenous passes that compare the target's "
                        "gain u at t with every other neuron's (finetuned vs base model, "
                        "vs layer-L attention LoRA removed at t).")
    p.add_argument("--max-eval-rows", type=int, default=None,
                   help="Evaluate only the first N test rows (smoke tests).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Default: the run's own eval_batch_size.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Single-run override; a sweep writes to <run>/attn_lora_positions/.")
    add_discovery_args(p)
    return p.parse_args()


# ======================================================================
# Positions
# ======================================================================
def group_mask(name: str, buckets: torch.Tensor, t_mask: torch.Tensor) -> torch.Tensor:
    """[B, S] bool: the positions whose attention output is taken LoRA-off.

    buckets: head_ablation.position_buckets (0 text, 1 prompt, 2 readout, -1 padding);
    t_mask: [B, S] bool, exactly one True per row (head_ablation.window_mask "t")."""
    real = buckets >= 0
    if name == "none":
        return torch.zeros_like(real)
    if name == "all":
        return real
    t_mask = t_mask & real
    if name == "t":
        return t_mask
    if name == "all_but_t":
        return real & ~t_mask
    pos = torch.arange(buckets.shape[1], device=buckets.device).unsqueeze(0)
    t_idx = t_mask.float().argmax(dim=1, keepdim=True)
    text = (buckets == 0) & ~t_mask
    if name == "text_before_t":
        return text & (pos < t_idx)
    if name == "text_after_t":
        return text & (pos > t_idx)
    if name == "prompt":
        return buckets == 1
    if name == "readout":
        return buckets == 2
    raise ValueError(f"unknown condition {name!r}; choose from {CONDITIONS}")


# ======================================================================
# The edit
# ======================================================================
class AttentionLoRAPositionAblator:
    """
    A forward hook on decoder layer `attn_layer`'s self_attn that recomputes it with its
    LoRA zeroed and splices that output in at the current condition's positions, plus
    recorders on the signal layer's MLP (u = up_proj out, h = down_proj in, a = act_fn
    out, each at token t for neuron `index`).

    Same interface as head_ablation.HeadAblator for eval_arm_with_activation: it sets
    `buckets` and `window` (the token t) per batch and resets them to None; outside a
    batch every hook is a no-op.
    """

    def __init__(self, run, attn_layer: int, signal_layer: int, index: int):
        layers = run.base.model.layers
        self.attn = layers[attn_layer].self_attn
        mlp = layers[signal_layer].mlp
        self.lora_B = [lin for name in ATTN_MODULES
                       for lin in getattr(getattr(self.attn, name), "lora_B", {}).values()]
        if not self.lora_B:
            raise SystemExit(f"layer {attn_layer} attention has no LoRA to remove")
        self._zeros = [torch.zeros_like(lin.weight) for lin in self.lora_B]
        self.index = index
        self.condition = "none"
        self.buckets: torch.Tensor | None = None
        self.window: torch.Tensor | None = None
        self.reset_records()
        self.handles = [
            self.attn.register_forward_hook(self._attn_hook, with_kwargs=True),
            mlp.up_proj.register_forward_hook(
                lambda m, a, out: self._record("u", out)),
            mlp.down_proj.register_forward_pre_hook(
                lambda m, a: self._record("h", a[0])),
            mlp.act_fn.register_forward_hook(
                lambda m, a, out: self._record("a", out)),
        ]

    def reset_records(self):
        self.rec = {"u": [], "h": [], "a": [], "n": []}

    def records(self) -> dict:
        out = {k: (torch.cat(v).numpy() if v else np.array([])) for k, v in self.rec.items()}
        if not len(out["u"]) == len(out["h"]) == len(out["a"]):
            raise RuntimeError("u/h/a recorders fired a different number of times -- a "
                               "module they hook is shared or called twice")
        return out

    def _record(self, key, value):
        if self.window is None:
            return None
        w = self.window.to(value.device)
        if not bool((w.sum(dim=1) == 1).all()):
            raise RuntimeError("the token-t window must mark exactly one position per row")
        self.rec[key].append(value[..., self.index][w].float().cpu())
        return None

    def _attn_hook(self, module, args, kwargs, output):
        if self.condition == "none" or self.buckets is None or self.window is None:
            return None
        hidden = output[0]
        m = group_mask(self.condition, self.buckets.to(hidden.device),
                       self.window.to(hidden.device))
        self.rec["n"].append(m.sum(dim=1).float().cpu())
        if not bool(m.any()):
            return None
        saved = [lin.weight.data for lin in self.lora_B]
        try:
            for lin, z in zip(self.lora_B, self._zeros):
                lin.weight.data = z
            # module.forward (not __call__): this hook must not re-enter. No KV cache, so
            # the second pass cannot append this layer's keys/values twice.
            off = module.forward(*args, **{**kwargs, "past_key_values": None})
        finally:
            for lin, w in zip(self.lora_B, saved):
                lin.weight.data = w
        mixed = torch.where(m.unsqueeze(-1), off[0].to(hidden.dtype), hidden)
        return (mixed,) + tuple(output[1:])

    def remove_hooks(self):
        for h in self.handles:
            h.remove()


# ======================================================================
# Gain specificity
# ======================================================================
GAIN_MODELS = (("finetuned", "none", False), ("attn_off_t", "t", False), ("base", "none", True))


def gain_specificity(run, dataset, ablator, bs) -> dict:
    """The target neuron's gain u at t in three models -- finetuned, finetuned with layer
    L's attention LoRA removed at t, and the base model (every LoRA zeroed) -- and where
    the target's finetuned/other mean-|u| ratio ranks among all neurons of its layer
    (rank 1 = the largest boost). Endogenous passes; running sums, nothing per row."""
    from mechanistic_analysis.ablate_lora_modules import find_lora_linears, restore_lora, zero_lora
    up = run.base.model.layers[run.layer].mlp.up_proj
    sums, cur = {}, {"key": None}

    def hook(_m, _a, out):
        if ablator.window is None or cur["key"] is None:
            return None
        u = out[ablator.window.to(out.device)].double().cpu()                # [B, f]
        s = sums.setdefault(cur["key"], [0, 0.0, 0.0, 0.0])
        s[0] += len(u)
        s[1] = s[1] + u.abs().sum(0)
        s[2] = s[2] + u.sum(0)
        s[3] = s[3] + (u ** 2).sum(0)
        return None

    handle = up.register_forward_hook(hook)
    lora = list(find_lora_linears(run.model).values())
    try:
        for key, cond, zero_all in GAIN_MODELS:
            cur["key"], ablator.condition = key, cond
            saved = zero_lora(lora) if zero_all else None
            try:
                eval_arm_with_activation(run, dataset, Arm("endogenous", None), ablator, bs, "t")
            finally:
                if saved is not None:
                    restore_lora(saved)
                cur["key"], ablator.condition = None, "none"
    finally:
        handle.remove()
        ablator.reset_records()
    i, out = run.index, {}
    abs_mean = {}
    for key, (n, s_abs, s, sq) in sums.items():
        mean = (s / n).numpy()
        abs_mean[key] = (s_abs / n).numpy()
        out[f"u_{key}_mean"] = float(mean[i])
        out[f"u_{key}_std"] = float(np.sqrt(max(float(sq[i] / n) - mean[i] ** 2, 0.0)))
    for other in ("attn_off_t", "base"):
        ratio = abs_mean["finetuned"] / np.maximum(abs_mean[other], 1e-8)
        out[f"abs_u_ratio_vs_{other}"] = float(ratio[i])
        out[f"abs_u_ratio_vs_{other}_rank"] = int((ratio > ratio[i]).sum()) + 1
        out[f"abs_u_ratio_vs_{other}_median"] = float(np.median(ratio))
    out["n_neurons"] = int(len(abs_mean["finetuned"]))
    return out


# ======================================================================
# Measurement
# ======================================================================
def _stats(c, pe, ae, px, v, rec_e, rec_x, u_none) -> dict:
    u = rec_e["u"]
    return {"condition": c,
            "tokens_per_row": float(rec_e["n"].mean()) if len(rec_e["n"]) else 0.0,
            "corr_endogenous": safe_corr(pe, ae),
            "corr_interv": safe_corr(px, v),
            "corr_natural_under_injection": safe_corr(px, ae),
            "corr_v_h": safe_corr(v, rec_x["h"]),
            "corr_interv_pred_h": safe_corr(px, rec_x["h"]),
            "corr_endo_pred_h": safe_corr(pe, rec_e["h"]),
            "corr_a_live_cached": safe_corr(rec_e["a"], ae),
            "u_mean": float(u.mean()), "u_std": float(u.std()),
            "u_frac_positive": float((u > 0).mean()),
            "u_corr_none": safe_corr(u, u_none if u_none is not None else u)}


def attention_positions_run(run, spec, args, out_dir: Path):
    t0 = time.time()
    cfg = run.config
    if cfg.training_config.disable_adapter_in_input:
        raise SystemExit(f"{spec.name}: gate-on run -- the adapter is off over the text, "
                         f"so only prompt positions carry LoRA; not what this tests")
    if getattr(cfg, "mlp_site", "act_fn") != "act_fn":
        raise SystemExit(f"{spec.name}: mlp_site={cfg.mlp_site!r}; the gain check assumes "
                         f"the act_fn site (h = a * u)")
    bs = args.batch_size or cfg.training_config.eval_batch_size
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conds if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown condition(s) {unknown}; choose from {list(CONDITIONS)}")
    conds = ["none"] + [c for c in conds if c != "none"]
    attn_layer = run.layer + args.layer_offset

    rows_idx = np.arange(len(run.dataset))
    if args.max_eval_rows is not None:
        rows_idx = rows_idx[:args.max_eval_rows]
    dataset = run.dataset.select(rows_idx.tolist())
    print(f"[attn-positions] N{run.layer}.{run.index}: attention LoRA of layer {attn_layer} "
          f"removed by token group; {len(dataset)} test rows")

    ablator = AttentionLoRAPositionAblator(run, attn_layer, run.layer, run.index)
    rows, ref, arrays, u_none = [], {}, {}, None
    try:
        for c in conds:
            if c == "text_after_t" and spec.position == -1:
                print(f"[attn-positions] {c:<14} empty group (t is the last text token), skipped")
                continue
            ablator.condition = c
            out, recs = {}, {}
            try:
                for a in ARMS:
                    ablator.reset_records()
                    out[a.name] = eval_arm_with_activation(run, dataset, a, ablator, bs, "t")
                    recs[a.name] = ablator.records()
            finally:
                ablator.condition = "none"
            for name, (_, tgt, _) in out.items():
                if not np.array_equal(ref.setdefault(name, tgt), tgt):
                    raise RuntimeError(f"{name}: targets changed between conditions -- the "
                                       f"injection sampler is not reproducible")
            if c == "none":
                u_none = recs["endogenous"]["u"]
            elif c != "all" and recs["endogenous"]["n"].sum() == 0:
                print(f"[attn-positions] {c:<14} empty group, skipped")
                continue
            pe, ae, _ = out["endogenous"]
            px, v, _ = out["x1"]
            row = _stats(c, pe, ae, px, v, recs["endogenous"], recs["x1"], u_none)
            rows.append(row)
            for key, val in (("pred_endogenous", pe), ("pred_x1", px),
                             ("u", recs["endogenous"]["u"]), ("h_endogenous", recs["endogenous"]["h"]),
                             ("h_x1", recs["x1"]["h"])):
                arrays.setdefault(key, []).append(np.asarray(val, dtype=np.float32))
            print(f"[attn-positions] {c:<14} ({row['tokens_per_row']:4.1f} tok)  endo "
                  f"{row['corr_endogenous']:+.3f}  interv {row['corr_interv']:+.3f}  "
                  f"natural|inj {row['corr_natural_under_injection']:+.3f}  r(v,h) "
                  f"{row['corr_v_h']:+.3f}  r(pred,h) {row['corr_interv_pred_h']:+.3f}  "
                  f"u {row['u_mean']:+.3f}±{row['u_std']:.3f} ({row['u_frac_positive']:.0%} >0, "
                  f"r w/ none {row['u_corr_none']:+.2f})")
        gain = {}
        if not args.no_gain_specificity and args.layer_offset == 0:
            gain = gain_specificity(run, dataset, ablator, bs)
            print(f"[attn-positions] gain u at t: base {gain['u_base_mean']:+.3f}±"
                  f"{gain['u_base_std']:.3f}  finetuned {gain['u_finetuned_mean']:+.3f}±"
                  f"{gain['u_finetuned_std']:.3f}  attn LoRA off at t "
                  f"{gain['u_attn_off_t_mean']:+.3f}±{gain['u_attn_off_t_std']:.3f}; |u| "
                  f"boost vs base {gain['abs_u_ratio_vs_base']:.1f}x (rank "
                  f"{gain['abs_u_ratio_vs_base_rank']} of {gain['n_neurons']}), vs attn off at "
                  f"t {gain['abs_u_ratio_vs_attn_off_t']:.1f}x (rank "
                  f"{gain['abs_u_ratio_vs_attn_off_t_rank']})")
    finally:
        ablator.remove_hooks()

    df = pd.DataFrame(rows)
    df.insert(0, "run", spec.name)
    df.insert(1, "signal", f"N{run.layer}.{run.index}")
    df.insert(2, "signal_layer", run.layer)
    df.insert(3, "position", spec.position)
    df.insert(4, "attn_layer", attn_layer)
    df.to_csv(out_dir / "attn_lora_positions.csv", index=False)
    np.savez_compressed(out_dir / "attn_lora_positions_arrays.npz", examples=rows_idx,
                        conditions=np.array(df["condition"]), injected=ref["x1"],
                        activation=ref["endogenous"],
                        **{k: np.stack(v) for k, v in arrays.items()})
    meta = {"run": spec.name, "checkpoint": str(run.checkpoint),
            "signal": f"N{run.layer}.{run.index}", "signal_layer": run.layer,
            "position": spec.position, "attn_layer": attn_layer,
            "conditions": list(df["condition"]), "n_eval_rows": int(len(rows_idx)),
            "max_eval_rows": args.max_eval_rows, "batch_size": bs, **gain,
            "seconds": round(time.time() - t0, 1)}
    (out_dir / SENTINEL).write_text(json.dumps(meta, indent=2))
    print(f"[attn-positions] wrote {out_dir} ({meta['seconds']:.0f}s)")


def select_runs(args):
    if args.config:
        return [load_run(args.config, output_root=args.output_root)]
    runs = discover_runs(group=args.group, run_glob=args.run_glob,
                         config_root=args.config_root, output_root=args.output_root)
    return shard(filter_cells(runs, args.cells, log="[attn-positions]"),
                 args.task_index, args.n_tasks)


def main():
    args = parse_args()
    runs = select_runs(args)
    if not runs:
        raise SystemExit("No runs with checkpoints matched.")
    analysis = ANALYSIS if args.layer_offset == 0 else f"{ANALYSIS}_offset{args.layer_offset}"
    run_per_locale(args, runs, analysis, SENTINEL,
                   analyze=lambda run, spec, out_dir: attention_positions_run(run, spec, args, out_dir),
                   log="[attn-positions]")


if __name__ == "__main__":
    main()
