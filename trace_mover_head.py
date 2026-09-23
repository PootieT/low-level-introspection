#!/usr/bin/env python3
"""
How does the injected value get from the origin token to the readout, and who moves it?

trace_introspection_circuit.py localizes the components whose write to the FINAL
token's residual moves with the injected value, per layer and per head. That stops one
step short of a mechanism: an attention head's contribution is a sum over SOURCE
positions, and "head (M, h) carries the signal" is only a transport claim once the
share coming from the introspected token t is separated from the share coming from
everywhere else.

This script takes the top movers and decomposes them twice, exactly.

  1. BY SOURCE POSITION.  A head's contribution to the readout is
         sum_s  alpha[h, last, s] * (a . W_O^h v_s)
     with `a` the frozen-RMSNorm readout covector. Each source position's term is
     computed separately and regressed on the injected value v, giving a dose-response
     slope PER SOURCE. Reported next to the raw attention weight alpha, because the two
     answer different questions: alpha says where the head looks, the slope says where
     it picks up the signal. A head can look mostly at BOS (Llama's attention sink) and
     still take all of its v-dependence from t.

  2. BY ORIGIN COMPONENT.  Folding the head's OV circuit and the mover layer's input
     LayerNorm backwards gives a covector on the residual AT POSITION t:
         a_src = alpha[h, last, t] * (W_V^kv)^T (W_O^h)^T a * gamma_in / rms_t
     The origin residual is then decomposed into embed + per-layer attn + per-layer mlp
     writes at t, and each is regressed on v. If the story is right, `mlp_L` -- the
     signal layer's own MLP, which is where down_proj[:, i] writes -- carries it and
     nothing else does. Both decompositions are checked for exactness against the
     quantity they claim to decompose, and the residual error is printed; a
     decomposition that does not sum to its own total is not evidence of anything.

  3. INHERITED OR LEARNED ADDRESSING (the E4 question).  The same attention pattern is
     measured with the adapter ACTIVE and DISABLED. If the mover attends to t just as
     strongly with the adapter off, the "look at the token the prompt names" behaviour
     is the base model's, and training only had to supply the channel and the readout
     -- which is the natural explanation for why token position generalizes while
     neuron index and layer do not. If the pattern only exists with the adapter on, the
     addressing was learned and is a per-run cost.

     NOTE this compares the attention PATTERN, not the whole circuit: the adapter
     changes the residual stream the query and key are computed from as well as the
     q/k projections. It says whether the base model already looks there, not whether
     it would look there in the absence of training. The complementary test is
     ablate_lora_modules.py restricted to the mover layer, splitting q/k (where to
     look) from v/o (what to carry).

Usage
-----
python trace_mover_head.py --config <run config> --top-heads 3
python trace_mover_head.py --group mlp_joint_grid_layer_token \
    --run-glob 'joint_*kl0.1*' --skip-existing
"""
from __future__ import annotations

import json
import argparse
import operator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.introspector.analysis_loading import (
    load_run_for_analysis, absolute_positions, layer_output_is_tuple)
from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.run_discovery import add_discovery_args, runs_from_args, resolve_run_action
from src.introspector.training.model import InterventionHook
from trace_introspection_circuit import readout_scale, contribution, linfit_slope_r2

# Source positions are reported relative to t, plus the absolute specials (BOS is
# Llama's attention sink and takes a large share of every head's mass).
SOURCE_REL_WINDOW = 8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--n-examples", type=int, default=200)
    p.add_argument("--grid-points", type=int, default=7)
    p.add_argument("--grid-std-mult", type=float, default=3.0)
    p.add_argument("--inject-scale", type=float, default=None,
                   help="Multiply the grid's half-width by this. Default: the run's own "
                        "training scale, where the signal is well above the bf16 floor.")
    p.add_argument("--top-heads", type=int, default=3,
                   help="Movers taken into the source/origin decompositions.")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Attention probabilities are [B, heads, seq, seq] per layer, so "
                        "this is smaller than the eval default.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Single-run override; a sweep always writes to <run>/mover_trace/.")
    add_discovery_args(p)
    return p.parse_args()


# ======================================================================
# Effective (LoRA-folded) weights
# ======================================================================
def effective_weight(module) -> torch.Tensor:
    """W0 + B@A*scaling for a peft LoRA linear, W0 otherwise.

    The mover's OV circuit has to be the one the TRAINED model runs. Reading
    `module.weight` alone silently analyzes the base model's head instead -- which is
    exactly the difference this script is trying to measure in part 3.
    """
    base = getattr(module, "base_layer", module)
    W = base.weight.detach().float()
    if hasattr(module, "lora_A") and isinstance(module.lora_A, torch.nn.ModuleDict):
        for adapter, A in module.lora_A.items():
            B = module.lora_B[adapter]
            W = W + (B.weight.detach().float() @ A.weight.detach().float()
                     ) * module.scaling[adapter]
    # on CPU: every decomposition below runs on CPU captures, and a weight left on the
    # accelerator turns each einsum into a device mismatch
    return W.cpu()


# ======================================================================
# Capture
# ======================================================================
class Grab:
    """Forward hook storing one slice of a module's output (or input)."""

    def __init__(self, fn):
        self.fn = fn
        self.out = None

    def __call__(self, module, inputs, outputs):
        self.out = self.fn(inputs, outputs)
        return None


def _resid(out, is_tuple):
    return out[0] if is_tuple else out


@torch.no_grad()
def run_injected_batch(run, batch, v, mover_layers, layer_out_tuple, disable_adapter=False):
    """
    One forward with the signal injected at (L, i, t), returning everything both
    decompositions need: the readout covector's inputs, per-component writes AT t,
    the mover layers' attention probabilities and value vectors, and the prediction.

    Captures are plain pytorch forward hooks rather than nnsight: the injection is
    already a forward hook (the same InterventionHook training used), attention
    probabilities come from output_attentions, and mixing two tracing mechanisms in
    one pass is how an off-by-one in the injection site goes unnoticed.
    """
    model = run.model
    base = run.base
    device = model.device
    attn_mask = batch["attention_mask"]
    bs, seq_len = attn_mask.shape
    abs_pos = absolute_positions(batch["token_position"], attn_mask)
    rows = torch.arange(bs, device=device)
    n_layers = base.config.num_hidden_layers
    L, i = run.layer, run.index

    caps, handles = {}, []
    try:
        # component writes AT THE ORIGIN TOKEN t
        caps["embed"] = Grab(lambda inp, out: out[rows, abs_pos, :].detach().float().cpu())
        handles.append(base.model.embed_tokens.register_forward_hook(caps["embed"]))
        for l in range(n_layers):
            caps[f"attn_{l}"] = Grab(lambda inp, out: out[0][rows, abs_pos, :].detach().float().cpu())
            handles.append(base.model.layers[l].self_attn.register_forward_hook(caps[f"attn_{l}"]))
            caps[f"mlp_{l}"] = Grab(lambda inp, out: out[rows, abs_pos, :].detach().float().cpu())
            handles.append(base.model.layers[l].mlp.register_forward_hook(caps[f"mlp_{l}"]))
        # the residual entering each mover layer (for its input LayerNorm's rms at t)
        for M in mover_layers:
            caps[f"pre_{M}"] = Grab(lambda inp, out: inp[0][rows, abs_pos, :].detach().float().cpu())
            handles.append(base.model.layers[M].input_layernorm.register_forward_hook(caps[f"pre_{M}"]))
            caps[f"v_{M}"] = Grab(lambda inp, out: out.detach().float().cpu())
            handles.append(base.model.layers[M].self_attn.v_proj.register_forward_hook(caps[f"v_{M}"]))
            caps[f"o_in_{M}"] = Grab(lambda inp, out: inp[0][rows, -1, :].detach().float().cpu())
            handles.append(base.model.layers[M].self_attn.o_proj.register_forward_hook(caps[f"o_in_{M}"]))
        # pre-final-norm residual at the readout token
        caps["final_resid"] = Grab(lambda inp, out: inp[0][rows, -1, :].detach().float().cpu())
        handles.append(base.model.norm.register_forward_hook(caps["final_resid"]))

        inject = None
        if v is not None:
            vals = torch.full((bs,), float(v), device=device)
            inject = InterventionHook(
                row_indices=rows, token_positions=abs_pos,
                signal_indices=torch.full((bs,), i, device=device, dtype=torch.long),
                intervention_values=vals)
            path = resolve_mlp_signal_path(model, L, path_to_output=False)
            handles.append(operator.attrgetter(path)(model).register_forward_hook(inject))

        kwargs = dict(input_ids=batch["input_ids"], attention_mask=attn_mask,
                      lora_token_mask=batch.get("lora_token_mask"),
                      do_intervention=False, output_hidden_states=True,
                      output_attentions=True)
        if disable_adapter:
            with model.disable_adapter():
                out = model(**kwargs)
        else:
            out = model(**kwargs)
    finally:
        for h in handles:
            h.remove()

    head = model.regression_head
    h_last = out.hidden_states[-1][:, -1, :]
    pred = head(h_last.to(device=head.weight.device, dtype=head.weight.dtype)).squeeze(-1)
    cap = {k: g.out for k, g in caps.items()}
    cap["attn_probs"] = {M: out.attentions[M][:, :, -1, :].detach().float().cpu()
                         for M in mover_layers}      # [B, heads, seq] from the readout
    cap["pred"] = pred.float().cpu()
    cap["abs_pos"] = abs_pos.cpu()
    cap["seq_len"] = seq_len
    return cap


# ======================================================================
# Decompositions
# ======================================================================
def head_source_contributions(cap, M, h, a, W_O, W_V, n_heads, n_kv_heads, head_dim):
    """
    [B, seq]: this head's contribution to the readout, split by source position.

    Sums (over positions) to the head's total contribution, which the caller checks
    against the same quantity computed from o_proj's input.
    """
    group = n_heads // n_kv_heads                       # GQA: heads per kv head
    kv = h // group
    WO_h = W_O[:, h * head_dim:(h + 1) * head_dim]      # [H, hd]
    u = a @ WO_h                                        # [B, hd]
    v = cap[f"v_{M}"]                                   # [B, seq, n_kv*hd]
    v_h = v[:, :, kv * head_dim:(kv + 1) * head_dim]    # [B, seq, hd]
    per_source = torch.einsum("bd,bsd->bs", u, v_h)     # a . W_O^h v_s
    alpha = cap["attn_probs"][M][:, h, :]               # [B, seq]
    return alpha * per_source, alpha, u


def origin_covector(cap, M, h, u, alpha_t, W_V, n_heads, n_kv_heads, head_dim, gamma_in, eps):
    """
    [B, H] covector on the residual at t, such that (a_src . write) is that write's
    contribution to the head's readout contribution THROUGH SOURCE t.

    The mover's input RMSNorm is treated as frozen (its scale computed at the actual
    residual), the same approximation the final-norm decomposition already makes: it
    is exact for the decomposition into additive writes at fixed norm, and the
    exactness check downstream is what verifies it.
    """
    group = n_heads // n_kv_heads
    kv = h // group
    WV_k = W_V[kv * head_dim:(kv + 1) * head_dim, :]    # [hd, H]
    src = u @ WV_k                                      # [B, H]
    x = cap[f"pre_{M}"]                                 # [B, H] residual at t
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return alpha_t.unsqueeze(-1) * src * gamma_in.unsqueeze(0) / rms


def slopes_over_grid(values, grid):
    """Mean dose-response slope and R^2 of `values` [grid][Nexp] against the grid."""
    arr = torch.stack(values, 0).numpy()                # [grid, Nexp]
    slopes, r2s = [], []
    for e in range(arr.shape[1]):
        s, r2 = linfit_slope_r2(grid, arr[:, e])
        slopes.append(s); r2s.append(r2)
    return float(np.mean(slopes)), float(np.mean(r2s))


def trace_movers(run, args, out_dir: Path):
    cfg = run.base.config
    device = run.model.device
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    eps = cfg.rms_norm_eps
    gamma = run.base.model.norm.weight.detach().float().cpu()
    w = run.model.regression_head.weight.detach().flatten().float().cpu()
    L, i = run.layer, run.index
    layer_out_tuple = layer_output_is_tuple(run.model)

    # eager attention is what makes output_attentions return real probabilities
    prev_impl = getattr(cfg, "_attn_implementation", None)
    cfg._attn_implementation = "eager"

    dataset = run.dataset.select(range(min(args.n_examples, len(run.dataset))))
    loader = lambda: DataLoader(dataset, batch_size=args.batch_size,
                                collate_fn=run.collator, shuffle=False)

    stats = run.stats
    scale = args.inject_scale
    if scale is None:
        tc = run.config.training_config
        scale = tc.intervention_extreme_scale if (
            tc.intervention_signal_distribution == "extreme") else 1.0
    half = args.grid_std_mult * stats["std"] * scale
    grid = np.linspace(stats["mean"] - half, stats["mean"] + half, args.grid_points)
    print(f"[mover] v-grid: {args.grid_points} points in "
          f"[{grid[0]:.4f}, {grid[-1]:.4f}] (scale x{scale:g})")

    # ---- pass 1: per-head dose response, to find the movers ----
    # a head at layer M reads the residual BEFORE that layer's MLP writes, so a mover
    # must sit strictly ABOVE the signal layer; layer L's own heads cannot see it
    cand_layers = list(range(L + 1, n_layers))
    if not cand_layers:
        raise SystemExit(f"Signal layer {L} is the last layer: no attention above it "
                         f"can transport the signal to the readout position.")
    W_O = {M: effective_weight(run.base.model.layers[M].self_attn.o_proj) for M in cand_layers}
    W_V = {M: effective_weight(run.base.model.layers[M].self_attn.v_proj) for M in cand_layers}
    gammas = {M: run.base.model.layers[M].input_layernorm.weight.detach().float().cpu()
              for M in cand_layers}

    per_head = {(M, h): [] for M in cand_layers for h in range(n_heads)}
    totals = {M: [] for M in cand_layers}
    preds = []
    for gi, v in enumerate(grid):
        g_head = {k: [] for k in per_head}
        g_tot = {M: [] for M in cand_layers}
        g_pred = []
        for batch in loader():
            batch = batch.to(device)
            cap = run_injected_batch(run, batch, v, cand_layers, layer_out_tuple)
            a = readout_scale(cap["final_resid"], w, gamma, eps)
            g_pred.append(cap["pred"])
            for M in cand_layers:
                z = cap[f"o_in_{M}"]                                  # [B, heads*hd]
                tot = contribution(a, z @ W_O[M].T)
                g_tot[M].append(tot)
                for h in range(n_heads):
                    src, alpha, u = head_source_contributions(
                        cap, M, h, a, W_O[M], W_V[M], n_heads, n_kv_heads, head_dim)
                    g_head[(M, h)].append(src.sum(-1))
        for k in per_head:
            per_head[k].append(torch.cat(g_head[k]))
        for M in cand_layers:
            totals[M].append(torch.cat(g_tot[M]))
        preds.append(torch.cat(g_pred))
        print(f"[mover] pass 1: grid {gi + 1}/{len(grid)}")

    rows = []
    for (M, h), vals in per_head.items():
        slope, r2 = slopes_over_grid(vals, grid)
        rows.append({"run": run.config.experiment_name, "signal_layer": L, "signal_index": i,
                     "layer": M, "head": h, "mean_slope": slope, "mean_abs_slope": abs(slope),
                     "mean_r2": r2})
    heads_df = pd.DataFrame(rows).sort_values("mean_abs_slope", ascending=False)
    heads_df["rank"] = np.arange(1, len(heads_df) + 1)
    heads_df.to_csv(out_dir / "mover_heads.csv", index=False)

    # exactness: the per-source decomposition must reproduce the layer's own total
    err = []
    for M in cand_layers:
        by_head = sum(torch.stack(per_head[(M, h)], 0) for h in range(n_heads))
        err.append(float((by_head - torch.stack(totals[M], 0)).abs().max()))
    print(f"[sanity] max|sum_h sum_s contribution - o_proj total| = {max(err):.3e} "
          f"(should be ~0; the per-source split is exact when it is)")
    e2e = np.corrcoef(torch.stack(preds, 0).numpy().reshape(-1),
                      np.repeat(grid, len(dataset)))[0, 1]
    print(f"[sanity] corr(prediction, injected v) = {e2e:.4f}")

    top = heads_df.head(args.top_heads)[["layer", "head"]].values.tolist()
    print(f"[mover] top movers: {[(int(m), int(h)) for m, h in top]}")

    # ---- pass 2: source positions + origin components, for the movers only ----
    # The residual a mover at layer M reads is embed + every attn/mlp write BELOW M --
    # not just those up to the signal layer. Truncating at L leaves the writes of
    # layers L+1..M-1 out of the sum, which does not make the decomposition wrong so
    # much as incomplete: it stops summing to the quantity it decomposes, and the
    # exactness check below is what caught it.
    comp_names_for = lambda M: (["embed"] + [f"attn_{l}" for l in range(M)]
                                + [f"mlp_{l}" for l in range(M)])
    src_acc = {(int(M), int(h)): [] for M, h in top}   # [grid] of [Nexp, bins]
    alpha_acc = {k: [] for k in src_acc}
    comp_acc = {k: {c: [] for c in comp_names_for(k[0])} for k in src_acc}
    tot_t_acc = {k: [] for k in src_acc}
    for gi, v in enumerate(grid):
        g_src = {k: [] for k in src_acc}
        g_alpha = {k: [] for k in alpha_acc}
        g_comp = {k: {c: [] for c in comp_names_for(k[0])} for k in comp_acc}
        g_tot_t = {k: [] for k in tot_t_acc}
        for batch in loader():
            batch = batch.to(device)
            cap = run_injected_batch(run, batch, v, [int(m) for m, _ in top], layer_out_tuple)
            a = readout_scale(cap["final_resid"], w, gamma, eps)
            t = cap["abs_pos"]
            brows = torch.arange(t.shape[0])
            for M, h in top:
                M, h = int(M), int(h)
                src, alpha, u = head_source_contributions(
                    cap, M, h, a, W_O[M], W_V[M], n_heads, n_kv_heads, head_dim)
                g_src[(M, h)].append(_rel_bin(src, t))
                g_alpha[(M, h)].append(_rel_bin(alpha, t))
                g_tot_t[(M, h)].append(src[brows, t])
                a_src = origin_covector(cap, M, h, u, alpha[brows, t], W_V[M],
                                        n_heads, n_kv_heads, head_dim, gammas[M], eps)
                for c in comp_names_for(M):
                    g_comp[(M, h)][c].append(contribution(a_src, cap[c]))
        for k in src_acc:
            src_acc[k].append(torch.cat(g_src[k])); alpha_acc[k].append(torch.cat(g_alpha[k]))
            tot_t_acc[k].append(torch.cat(g_tot_t[k]))
            for c in comp_names_for(k[0]):
                comp_acc[k][c].append(torch.cat(g_comp[k][c]))
        print(f"[mover] pass 2: grid {gi + 1}/{len(grid)}")

    src_rows, comp_rows = [], []
    for (M, h) in src_acc:
        stacked = torch.stack(src_acc[(M, h)], 0)         # [grid, Nexp, bins]
        alphas = torch.stack(alpha_acc[(M, h)], 0)
        for b in range(stacked.shape[-1]):
            rel = b - SOURCE_REL_WINDOW
            col = [stacked[g, :, b] for g in range(stacked.shape[0])]
            slope, r2 = slopes_over_grid(col, grid)
            src_rows.append({"run": run.config.experiment_name, "layer": M, "head": h,
                             "rel_source": rel if b < 2 * SOURCE_REL_WINDOW + 1 else None,
                             "bin": _BIN_LABELS[b], "mean_slope": slope, "mean_r2": r2,
                             "mean_attention": float(alphas[:, :, b].mean())})
        # exactness of the origin decomposition against the source-t contribution
        tot_t = torch.stack(tot_t_acc[(M, h)], 0)
        summed = sum(torch.stack(comp_acc[(M, h)][c], 0) for c in comp_names_for(M))
        print(f"[sanity] head L{M}.h{h}: max|sum_components - contribution via t| = "
              f"{float((summed - tot_t).abs().max()):.3e} "
              f"(decomposes embed + {M} attn + {M} mlp writes at t)")
        for c in comp_names_for(M):
            slope, r2 = slopes_over_grid(comp_acc[(M, h)][c], grid)
            comp_rows.append({"run": run.config.experiment_name, "layer": M, "head": h,
                              "component": c, "mean_slope": slope, "mean_r2": r2,
                              "is_signal_layer_mlp": c == f"mlp_{L}"})
    src_df = pd.DataFrame(src_rows); src_df.to_csv(out_dir / "mover_source_positions.csv", index=False)
    comp_df = pd.DataFrame(comp_rows); comp_df.to_csv(out_dir / "mover_origin_components.csv", index=False)

    # ---- pass 3: is the addressing inherited? attention with the adapter on vs off ----
    addr_rows = []
    for disable in (False, True):
        alpha_t, alpha_bos, alpha_max_other = {k: [] for k in src_acc}, {k: [] for k in src_acc}, {k: [] for k in src_acc}
        for batch in loader():
            batch = batch.to(device)
            cap = run_injected_batch(run, batch, None, [int(m) for m, _ in top],
                                     layer_out_tuple, disable_adapter=disable)
            t = cap["abs_pos"]
            brows = torch.arange(t.shape[0])
            for M, h in src_acc:
                alpha = cap["attn_probs"][M][:, h, :]
                alpha_t[(M, h)].append(alpha[brows, t])
                first = (batch["attention_mask"] == 0).sum(dim=1).cpu()   # BOS after left pad
                alpha_bos[(M, h)].append(alpha[brows, first])
                masked = alpha.clone(); masked[brows, t] = -1
                alpha_max_other[(M, h)].append(masked.max(-1).values)
        for M, h in src_acc:
            addr_rows.append({
                "run": run.config.experiment_name, "layer": M, "head": h,
                "adapter": "off" if disable else "on",
                "attn_to_t": float(torch.cat(alpha_t[(M, h)]).mean()),
                "attn_to_bos": float(torch.cat(alpha_bos[(M, h)]).mean()),
                "max_attn_elsewhere": float(torch.cat(alpha_max_other[(M, h)]).mean())})
    addr_df = pd.DataFrame(addr_rows)
    addr_df.to_csv(out_dir / "mover_attention_base_vs_tuned.csv", index=False)
    print("\n[mover] addressing: attention from the readout token to t")
    print(addr_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    _plot(run, heads_df, src_df, comp_df, addr_df, out_dir)
    with open(out_dir / "mover_meta.json", "w") as f:
        json.dump({"run": run.config.experiment_name, "checkpoint": str(run.checkpoint),
                   "signal": f"N{L}.{i}", "top_heads": [[int(m), int(h)] for m, h in top],
                   "grid": grid.tolist(), "inject_scale": scale,
                   "per_source_max_err": max(err), "e2e_corr_pred_vs_v": float(e2e),
                   "n_examples": len(dataset)}, f, indent=2)
    if prev_impl is not None:
        cfg._attn_implementation = prev_impl
    print(f"[mover] wrote {out_dir}")


_BIN_LABELS = ([f"t{d:+d}" if d else "t" for d in range(-SOURCE_REL_WINDOW, SOURCE_REL_WINDOW + 1)]
               + ["before_window", "after_window"])


def _rel_bin(x, t):
    """[B, seq] -> [B, 2*W+3] binned on offset from t, with two catch-all bins.

    A fixed-width window plus catch-alls, because the sequences differ in length and
    the interesting structure is local to t and to the readout end; without the
    catch-alls the columns would not sum to the head's total and the exactness check
    would be meaningless.
    """
    bs, seq = x.shape
    n_bins = 2 * SOURCE_REL_WINDOW + 3
    out = torch.zeros(bs, n_bins)
    pos = torch.arange(seq).unsqueeze(0).expand(bs, -1)
    rel = pos - t.unsqueeze(1)
    idx = (rel + SOURCE_REL_WINDOW).clamp(0, 2 * SOURCE_REL_WINDOW)
    idx = torch.where(rel < -SOURCE_REL_WINDOW, torch.full_like(idx, n_bins - 2), idx)
    idx = torch.where(rel > SOURCE_REL_WINDOW, torch.full_like(idx, n_bins - 1), idx)
    out.scatter_add_(1, idx, x)
    return out


def _plot(run, heads_df, src_df, comp_df, addr_df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    top = heads_df.head(12)
    labels = [f"L{int(r.layer)}h{int(r.head)}" for r in top.itertuples()]
    axes[0].barh(labels[::-1], top["mean_slope"].values[::-1])
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("dose-response slope"); axes[0].set_title("top mover heads")

    for (M, h), g in src_df.groupby(["layer", "head"]):
        g = g[g["rel_source"].notna()].sort_values("rel_source")
        axes[1].plot(g["rel_source"], g["mean_slope"], "o-", ms=4, label=f"L{M}h{h}")
    axes[1].axvline(0, color="k", ls="--", lw=1)
    axes[1].set_xlabel("source position relative to t"); axes[1].set_ylabel("slope")
    axes[1].set_title("where the mover picks up v"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)

    # movers at different layers decompose over different component sets, so plot
    # every head against one canonical ordering rather than its own positional index
    def _ckey(c):
        if c == "embed":
            return (-1, 0)
        kind, l = c.split("_")
        return (int(l), 0 if kind == "attn" else 1)
    order = sorted(comp_df["component"].unique(), key=_ckey)
    xpos = {c: k for k, c in enumerate(order)}
    for (M, h), g in comp_df.groupby(["layer", "head"]):
        g = g.assign(x=g["component"].map(xpos)).sort_values("x")
        axes[2].plot(g["x"], g["mean_slope"], "o-", ms=4, label=f"L{M}h{h}")
    axes[2].set_xticks(range(len(order)))
    axes[2].set_xticklabels(order, rotation=90, fontsize=6)
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_ylabel("slope"); axes[2].set_title("which write at t carries it")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.25)
    fig.suptitle(f"Mover-head trace: {run.config.experiment_name}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "mover_trace.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    if not addr_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        piv = addr_df.pivot_table(index=["layer", "head"], columns="adapter", values="attn_to_t")
        piv.plot.bar(ax=ax)
        ax.set_ylabel("mean attention from readout to t")
        ax.set_title("Is the addressing inherited?\n(adapter on vs off)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "mover_addressing.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def replot_mover(spec, out_dir: Path, args) -> bool:
    """Redraw the mover-trace figures from the four cached CSVs; False if the primary
    one is absent. Optional CSVs fall back to empty frames with the right columns so the
    plot's groupbys stay valid."""
    heads_csv = out_dir / "mover_heads.csv"
    if not heads_csv.exists():
        return False

    def _rd(name, cols):
        p = out_dir / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=cols)

    _plot(spec, pd.read_csv(heads_csv),
          _rd("mover_source_positions.csv", ["layer", "head", "rel_source", "mean_slope"]),
          _rd("mover_origin_components.csv", ["layer", "head", "component", "mean_slope"]),
          _rd("mover_attention_base_vs_tuned.csv", ["layer", "head", "adapter", "attn_to_t"]),
          out_dir)
    return True


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    runs = runs_from_args(args)
    if not runs:
        raise SystemExit("No runs with checkpoints matched.")
    sweeping = len(runs) > 1 or args.group is not None
    for k, spec in enumerate(runs, 1):
        out_dir = ((args.output_dir or (spec.run_dir / "mover_trace")) if not sweeping
                   else spec.run_dir / "mover_trace")
        action = resolve_run_action(args, out_dir / "mover_heads.csv")
        if action == "skip":
            print(f"[mover] [{k}/{len(runs)}] {spec.name}: results exist, skipping")
            continue
        if action == "replot":
            if replot_mover(spec, out_dir, args):
                print(f"[mover] [{k}/{len(runs)}] {spec.name}: re-plotted from cache")
                continue
            print(f"[mover] [{k}/{len(runs)}] {spec.name}: cache incomplete, recomputing")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[mover] ===== [{k}/{len(runs)}] {spec.name} ({spec.cell}) =====")
        run = load_run_for_analysis(spec.config, checkpoint=args.checkpoint or spec.checkpoint())
        try:
            trace_movers(run, args, out_dir)
        finally:
            del run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
