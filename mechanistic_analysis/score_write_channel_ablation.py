#!/usr/bin/env python3
"""
Which part of the target neuron's WRITE does the report read? -- the "private channel"
claim of Sec. 6.1, tested causally at the write site.

Fig. 36 finds that finetuning changes the target neuron's W_up row and W_down column far
more than any other neuron's, and Sec. 6.1 reads that as: the activation itself is kept,
and the finetuned W_down routes it into a new (mostly orthogonal) direction the report
reads. Weight norms only say where training changed the model. This script removes
pieces of the neuron's write at the introspected token and sees which piece the report
needs.

At the signal layer L and token t only (every other position, layer and neuron is
untouched), the neuron i writes
    h_i * W_down_eff[:, i],    h_i = a_i * u_i,   W_down_eff = W_down_base + dW_down
(a_i = the activation, replaced by v under injection; u_i = up_proj's output). Conditions:

  none                 nothing removed
  remove_native_write  subtract h_i * W_down_base[:, i]   (the base model's write direction;
                       every native downstream effect of the neuron at t goes through it)
  remove_added_write   subtract h_i * dW_down[:, i]       (the direction LoRA added: the
                       hypothesized private channel)
  base_up              u_i <- x . W_up_base[i]            (undo the up_proj edit for this
                       neuron at t; the write keeps the finetuned W_down column)
  base_write           base_up AND remove_added_write: the neuron writes exactly what the
                       base model would (same input, same a_i)
  remove_write         subtract the whole write (floor: the neuron is silent at t)

The two remove_* halves sum to remove_write exactly at the site. Readouts, as in the
head-set ablation:
  corr_endogenous               no injection: corr(prediction, natural activation a)
  corr_interv                   x1 injection: corr(prediction, injected value v)
  corr_natural_under_injection  x1 injection: corr(prediction, a), the value the injection
                                replaced (a context path around the site)
With the whole write removed, the endogenous arm measures how much of the natural report
survives without the site at all.

Gate-off runs only (a gate-on run has no adapter at t, so there is nothing "added" there),
and only the act_fn MLP site (the write decomposition above assumes h = a * u).

OUTPUTS (per run, <run>/write_channel_ablation/)
  write_channel_ablation.csv        one row per condition
  write_channel_ablation_preds.npz  per-example predictions per condition, v, a, rows
  write_channel_ablation_meta.json  provenance and the column norms; written LAST
                                    (the --skip-existing sentinel)

Usage
-----
python mechanistic_analysis/score_write_channel_ablation.py --config <run config>
python mechanistic_analysis/score_write_channel_ablation.py --group mlp_joint_grid_layer_token \\
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
from mechanistic_analysis.trace_introspection_circuit import effective_weight

ANALYSIS = "write_channel_ablation"
SENTINEL = "write_channel_ablation_meta.json"
ARMS = (Arm("endogenous", None), Arm("x1", 1.0))
# condition -> (use the base up_proj row at t, which part of the write to subtract at t)
CONDITIONS = {
    "none": (False, None),
    "remove_native_write": (False, "native"),
    "remove_added_write": (False, "added"),
    "base_up": (True, None),
    "base_write": (True, "added"),
    "remove_write": (False, "full"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--cells", type=str, default=None,
                   help="Comma list of 'layer:position' cells kept after discovery, e.g. "
                        "'2:-1,2:-28,8:-1,8:-28'. Default: every run --run-glob matches.")
    p.add_argument("--conditions", type=str, default=",".join(CONDITIONS),
                   help="Comma list of conditions to evaluate (default: all).")
    p.add_argument("--max-eval-rows", type=int, default=None,
                   help="Evaluate only the first N test rows (smoke tests).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Default: the run's own eval_batch_size.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Single-run override; a sweep writes to <run>/write_channel_ablation/.")
    add_discovery_args(p)
    return p.parse_args()


# ======================================================================
# The write-site edit
# ======================================================================
class WriteChannelAblator:
    """
    Forward hooks on the signal layer's up_proj and down_proj that edit neuron `index`'s
    write at the introspected token only.

    Same interface as head_ablation.HeadAblator as far as eval_arm_with_activation is
    concerned: it sets `buckets` and `window` per batch (window = the token t, as a
    [B, S] bool) and resets them to None afterwards; with no condition set, or outside a
    batch, the hooks return None and change nothing.

    The subtraction happens on down_proj's OUTPUT, in float, from its actual input h --
    so under injection h already carries v, and under base_up it carries the base u.
    """

    def __init__(self, run, layer: int, index: int):
        mlp = run.base.model.layers[layer].mlp
        self.up, self.down, self.index = mlp.up_proj, mlp.down_proj, index
        for name, mod in (("up_proj", self.up), ("down_proj", self.down)):
            base = getattr(mod, "base_layer", mod)
            if getattr(base, "bias", None) is not None:
                raise SystemExit(f"{name} has a bias; the write decomposition assumes none")
        w_eff = effective_weight(self.down)[:, index]                       # [d_model]
        w_base = getattr(self.down, "base_layer", self.down).weight.detach().float().cpu()[:, index]
        self.cols = {"native": w_base, "added": w_eff - w_base, "full": w_eff}
        self.up_row = (getattr(self.up, "base_layer", self.up).weight.detach().float()
                       .cpu()[index])                                       # [d_model]
        self.use_base_up, self.remove = False, None
        self.buckets = None                       # set by eval_arm_with_activation; unused
        self.window: torch.Tensor | None = None  # [B, S] bool: the token t
        self.handles = [self.up.register_forward_hook(self._up_hook),
                        self.down.register_forward_hook(self._down_hook)]

    def set_condition(self, name: str):
        self.use_base_up, self.remove = CONDITIONS[name]

    def column_stats(self) -> dict:
        nat, add = self.cols["native"], self.cols["added"]
        return {"norm_native_col": float(nat.norm()), "norm_added_col": float(add.norm()),
                "added_over_native": float(add.norm() / nat.norm().clamp_min(1e-12)),
                "cos_added_native": float(torch.nn.functional.cosine_similarity(
                    add, nat, dim=0))}

    def _up_hook(self, _module, args, out):
        if not self.use_base_up or self.window is None:
            return None
        x = args[0]
        u_base = x.float() @ self.up_row.to(x.device)                        # [B, S]
        out = out.clone()
        w = self.window.to(out.device)
        out[..., self.index] = torch.where(w, u_base.to(out.dtype), out[..., self.index])
        return out

    def _down_hook(self, _module, args, out):
        if self.remove is None or self.window is None:
            return None
        h = args[0][..., self.index].float()                                  # [B, S]
        h = h * self.window.to(h.device).float()
        delta = h.unsqueeze(-1) * self.cols[self.remove].to(out.device)       # [B, S, d]
        return (out.float() - delta).to(out.dtype)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()


# ======================================================================
# Measurement
# ======================================================================
def write_channel_run(run, spec, args, out_dir: Path):
    t0 = time.time()
    cfg = run.config
    if cfg.training_config.disable_adapter_in_input:
        raise SystemExit(f"{spec.name}: gate-on run -- the adapter is off at the "
                         f"introspected token, so there is no added write to remove")
    if getattr(cfg, "mlp_site", "act_fn") != "act_fn":
        raise SystemExit(f"{spec.name}: mlp_site={cfg.mlp_site!r}; the write decomposition "
                         f"assumes the act_fn site (h = a * u)")
    bs = args.batch_size or cfg.training_config.eval_batch_size
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conds if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown condition(s) {unknown}; choose from {list(CONDITIONS)}")
    if "none" not in conds:
        conds = ["none"] + conds

    rows_idx = np.arange(len(run.dataset))
    if args.max_eval_rows is not None:
        rows_idx = rows_idx[:args.max_eval_rows]
    dataset = run.dataset.select(rows_idx.tolist())

    ablator = WriteChannelAblator(run, run.layer, run.index)
    stats = ablator.column_stats()
    print(f"[write-channel] N{run.layer}.{run.index}: |W_down base col| "
          f"{stats['norm_native_col']:.4f}, |added| {stats['norm_added_col']:.4f} "
          f"({stats['added_over_native']:.2f}x), cos {stats['cos_added_native']:+.3f}; "
          f"{len(dataset)} test rows")
    rows, P, ref = [], {a.name: [] for a in ARMS}, {}
    try:
        for c in conds:
            ablator.set_condition(c)
            try:
                out = {a.name: eval_arm_with_activation(run, dataset, a, ablator, bs, "t")
                       for a in ARMS}
            finally:
                ablator.set_condition("none")
            for name, (_, tgt, _) in out.items():
                # same rows, same injected v in every condition
                if not np.array_equal(ref.setdefault(name, tgt), tgt):
                    raise RuntimeError(f"{name}: targets changed between conditions -- the "
                                       f"injection sampler is not reproducible")
            pe, ae, _ = out["endogenous"]
            px, v, a_nat = out["x1"]
            row = {"condition": c, "corr_endogenous": safe_corr(pe, ae),
                   "corr_interv": safe_corr(px, v),
                   "corr_natural_under_injection": safe_corr(px, a_nat),
                   "pred_std_endogenous": float(np.std(pe)), "pred_std_x1": float(np.std(px))}
            rows.append(row)
            P["endogenous"].append(pe.astype(np.float32))
            P["x1"].append(px.astype(np.float32))
            print(f"[write-channel] {c:<20} endo {row['corr_endogenous']:+.3f}  interv "
                  f"{row['corr_interv']:+.3f}  natural|inj "
                  f"{row['corr_natural_under_injection']:+.3f}")
    finally:
        ablator.remove_hooks()

    df = pd.DataFrame(rows)
    df.insert(0, "run", spec.name)
    df.insert(1, "signal", f"N{run.layer}.{run.index}")
    df.insert(2, "signal_layer", run.layer)
    df.insert(3, "position", spec.position)
    df.to_csv(out_dir / "write_channel_ablation.csv", index=False)
    np.savez_compressed(out_dir / "write_channel_ablation_preds.npz", examples=rows_idx,
                        conditions=np.array(conds), injected=ref["x1"],
                        activation=ref["endogenous"],
                        **{f"pred_{a}": np.stack(v) for a, v in P.items()})
    meta = {"run": spec.name, "checkpoint": str(run.checkpoint),
            "signal": f"N{run.layer}.{run.index}", "signal_layer": run.layer,
            "position": spec.position, "conditions": conds, "n_eval_rows": int(len(rows_idx)),
            "max_eval_rows": args.max_eval_rows, "batch_size": bs, **stats,
            "seconds": round(time.time() - t0, 1)}
    (out_dir / SENTINEL).write_text(json.dumps(meta, indent=2))
    print(f"[write-channel] wrote {out_dir} ({meta['seconds']:.0f}s)")


def select_runs(args):
    if args.config:
        return [load_run(args.config, output_root=args.output_root)]
    runs = discover_runs(group=args.group, run_glob=args.run_glob,
                         config_root=args.config_root, output_root=args.output_root)
    return shard(filter_cells(runs, args.cells, log="[write-channel]"),
                 args.task_index, args.n_tasks)


def main():
    args = parse_args()
    runs = select_runs(args)
    if not runs:
        raise SystemExit("No runs with checkpoints matched.")
    run_per_locale(args, runs, ANALYSIS, SENTINEL,
                   analyze=lambda run, spec, out_dir: write_channel_run(run, spec, args, out_dir),
                   log="[write-channel]")


if __name__ == "__main__":
    main()
