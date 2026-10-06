#!/usr/bin/env python3
"""
How does the injected value get from the origin token to the readout, and who moves it?

trace_introspection_circuit.py localizes the components whose write to the FINAL
token's residual moves with the injected value, per layer and per head. That stops one
step short of a mechanism: an attention head's contribution is a sum over SOURCE
positions, and "head (M, h) carries the signal" is only a transport claim once the
share coming from the introspected token t is separated from the share coming from
everywhere else.

WHICH HEADS ARE "THE MOVERS". A head's score is NOT an attention statistic. It is the
dose-response slope of its own exact, additive contribution to the prediction:

    prediction - bias = a . x_final,        a = (w * gamma) / rms(x_final)
    contribution(head h of layer M) = sum_s alpha[h, last, s] * (a . W_O^h v_s)

with v_s the value vector at source position s (from the LoRA-FOLDED v_proj, so it is
the trained model's OV circuit) and the query fixed at the readout token. Each head's
contribution is regressed on the injected value v across the dose-response grid, and
the mean slope over examples is its score. Two consequences worth stating, because
both bit:
  * A head can attend hard at t and score ~0, if its OV does not read the private
    channel; and a head with tiny attention can score high, because the contribution
    is alpha TIMES an OV term that the injection makes enormous.
  * This is DIRECT attribution to the final residual. A head's indirect effect -- what
    it writes being re-read by later layers -- is attributed to those later components,
    which is precisely why deep heads score highly: they re-read what an early head
    already deposited. Hence the two rankings (see select_heads): `rank` over the total
    slope, and `rank_at_t` over the part routed through the origin token, which is the
    transport claim.

This script takes the selected heads and decomposes them twice, exactly.

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

EVERYTHING IS RUN AT TWO INJECTION SCALES by default (1 and the run's training scale),
because the circuit is not the same at both. On joint_N0.1377_POS-12 the top head by
total slope at x10 (L14h15) is rank 68 at x1, and the x10 transporter L1h31 -- 98.6% of
its v-dependence coming from t -- falls to rank 455 at x1. One scale cannot tell you
whether what you found is the circuit that runs endogenously.

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
    load_run_for_analysis, absolute_positions, layer_output_is_tuple, run_per_locale)
from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.run_discovery import (
    add_discovery_args, runs_from_args, resolve_run_action, locale_out_dir)
from src.introspector.regimes import (
    DEFAULT_REGIMES, DEFAULT_LOCAL_DELTA_SIGMA, parse_regimes, zero_offset_index,
    identity_check, power_warning)
from src.introspector.training.model import InterventionHook
from trace_introspection_circuit import (
    readout_scale, contribution, linfit_slope_r2, capture_live_centres, effective_weight)

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
    p.add_argument("--regimes", type=str, default=DEFAULT_REGIMES,
                   help="Injection regimes to sweep (see src/introspector/regimes.py). "
                        "'xN' is a global grid at N times the natural std, 'xtrain' the "
                        "run's own training scale, 'local' perturbs each example around "
                        "ITS OWN current value. The mechanism is scale-dependent -- the "
                        "identified mover attends to t only because the injection is "
                        "large -- so one regime cannot tell you whether it describes the "
                        "endogenous circuit. Pass one to cut the runtime.")
    p.add_argument("--local-delta-sigma", type=float, default=DEFAULT_LOCAL_DELTA_SIGMA,
                   help="Half-width of the 'local' sweep in units of the neuron's std; "
                        "override per regime with e.g. 'local@0.5'.")
    p.add_argument("--top-heads", type=int, default=3,
                   help="Movers taken into the source/origin decompositions.")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Attention probabilities are [B, heads, seq, seq] per layer, so "
                        "this is smaller than the eval default.")
    p.add_argument("--heads-from-knockout", action="store_true",
                   help="Instead of the full trace, redraw the attention-vs-v and "
                        "attention-profile figures for the top --top-heads heads PER ARM of "
                        "score_head_knockout.py (largest single-knockout drop), into "
                        "*_knockout.{csv,png} next to the mover-trace outputs. Attention "
                        "vs v is read from mover_head_dose_all.csv (run the full trace "
                        "first); only the attention profile needs the GPU.")
    p.add_argument("--knockout-dir", type=Path, default=None,
                   help="Where the head_knockout outputs are (default: "
                        "<run>/head_knockout, per signal for multi-locale runs).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Single-run override; a sweep always writes to <run>/mover_trace/.")
    add_discovery_args(p)
    return p.parse_args()


# ======================================================================
# Effective (LoRA-folded) weights
# ======================================================================
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
            # scalar for a global regime, one value per row for `local`
            vals = (v.to(device=device, dtype=torch.float32) if torch.is_tensor(v)
                    else torch.full((bs,), float(v), device=device))
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


def select_heads(heads_df, signal_layer: int, top_k: int):
    """[((layer, head), why)] for the decompositions: a union of three criteria.

    TOP BY TOTAL SLOPE is what the ranking used to be on its own, and it is not
    sufficient: it is equally large for a head that fetches v from the origin and for
    a later head that re-reads v from where the first one put it (see the rank_at_t
    note above).

    TOP BY SLOPE AT t is the transporter proper.

    FIRST LAYER ABOVE THE SIGNAL is included unconditionally because it is the only
    layer whose attention can reach v while v exists NOWHERE ELSE -- at layer L+1 the
    signal has had no opportunity to be copied anywhere, so a head there that depends
    on v must have taken it from t. It happened to land in the top 3 on the run this
    was developed against; that was luck, and the whole transport story would have been
    missed had it not.

    Duplicates collapse, keeping the first (most specific) reason.
    """
    picked = {}

    def add(row, why):
        picked.setdefault((int(row["layer"]), int(row["head"])), why)

    for _, r in heads_df.nsmallest(top_k, "rank").iterrows():
        add(r, "top total")
    for _, r in heads_df.nsmallest(max(1, top_k), "rank_at_t").iterrows():
        add(r, "top at t")
    above = heads_df[heads_df["layer"] == signal_layer + 1]
    if not above.empty:
        add(above.nsmallest(1, "rank_at_t").iloc[0], f"best in L{signal_layer + 1}")
    return list(picked.items())


def slopes_over_grid(values, grid):
    """Mean dose-response slope and R^2 of `values` [grid][Nexp] against the grid."""
    arr = torch.stack(values, 0).numpy()                # [grid, Nexp]
    slopes, r2s = [], []
    for e in range(arr.shape[1]):
        s, r2 = linfit_slope_r2(grid, arr[:, e])
        slopes.append(s); r2s.append(r2)
    return float(np.mean(slopes)), float(np.mean(r2s))


def prepare_inputs(run, args):
    """Examples, loader, injection regimes, live centres (for `local`) and the injection
    dtype -- shared by the full trace and the knockout-head attention profile, so both
    measure on the same examples under the same regimes."""
    L, i = run.layer, run.index
    dataset = run.dataset.select(range(min(args.n_examples, len(run.dataset))))
    loader = lambda: DataLoader(dataset, batch_size=args.batch_size,
                                collate_fn=run.collator, shuffle=False)

    # ---- injection regimes ----
    # SEVERAL by default: the mechanism is scale-dependent, and the regimes ask
    # different questions. The training scale (10x here) is where the dose-response is
    # cleanest and where the mover was originally identified; x1 is a chord across the
    # natural range; `local` is the tangent at each example's own operating point. The
    # 10x mover attends to t only because the injection is large (6600x the clean
    # pattern), so no single regime can say whether it describes the endogenous
    # circuit. See src/introspector/regimes.py.
    stats = run.stats
    tc = run.config.training_config
    train_scale = (tc.intervention_extreme_scale
                   if tc.intervention_signal_distribution == "extreme" else 1.0)
    regimes = parse_regimes(args.regimes, stats, args.grid_points, args.grid_std_mult,
                            delta_sigma=args.local_delta_sigma, train_scale=train_scale)
    for rg in regimes:
        print(f"[mover] regime {rg.describe(stats)}")

    # `local` centres on each example's LIVE activation, so capture it first, in the
    # same order the passes below iterate (sequential, unshuffled loader).
    centres = None
    if any(rg.is_local for rg in regimes):
        centres = capture_live_centres(run.model, dataset, run.collator,
                                       args.batch_size, L, i)
        print(f"[mover] live centres for {len(centres)} examples: "
              f"mean={centres.mean():.4f} std={centres.std():.4f} "
              f"(cached-column mean={stats['mean']:.4f} std={stats['std']:.4f})")
    inject_dtype = next(run.model.parameters()).dtype
    return dataset, loader, regimes, centres, inject_dtype, train_scale


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

    dataset, loader, regimes, centres, inject_dtype, train_scale = prepare_inputs(run, args)
    stats = run.stats

    # ---- shared setup ----
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
    comp_names_for = lambda M: (["embed"] + [f"attn_{l}" for l in range(M)]
                                + [f"mlp_{l}" for l in range(M)])

    def pass1(regime):
        """Per-head dose response over every head above the signal layer."""
        grid = regime.grid
        per_head = {(M, h): [] for M in cand_layers for h in range(n_heads)}
        per_head_t = {(M, h): [] for M in cand_layers for h in range(n_heads)}
        # attention to t for EVERY head at every grid point. Already computed inside
        # head_source_contributions and previously discarded; keeping it is what lets
        # the threshold-vs-static classification cover all heads instead of only the
        # handful that pass 2 decomposes -- and those are selected BY slope rank, which
        # is the very variable the classification is meant to be compared against.
        per_head_alpha_t = {(M, h): [] for M in cand_layers for h in range(n_heads)}
        totals = {M: [] for M in cand_layers}
        preds = []
        for gi in range(len(grid)):
            g_head = {k: [] for k in per_head}
            g_head_t = {k: [] for k in per_head}
            g_alpha_t = {k: [] for k in per_head}
            g_tot = {M: [] for M in cand_layers}
            g_pred = []
            row0 = 0
            for batch in loader():
                batch = batch.to(device)
                nrow = batch["input_ids"].shape[0]
                v = regime.values_for(
                    None if not regime.is_local else centres[row0:row0 + nrow], gi)
                if torch.is_tensor(v):
                    v = v.to(dtype=inject_dtype)
                row0 += nrow
                cap = run_injected_batch(run, batch, v, cand_layers, layer_out_tuple)
                a = readout_scale(cap["final_resid"], w, gamma, eps)
                g_pred.append(cap["pred"])
                t_pos = cap["abs_pos"]
                brows = torch.arange(t_pos.shape[0])
                for M in cand_layers:
                    z = cap[f"o_in_{M}"]
                    g_tot[M].append(contribution(a, z @ W_O[M].T))
                    for h in range(n_heads):
                        src, alpha, u = head_source_contributions(
                            cap, M, h, a, W_O[M], W_V[M], n_heads, n_kv_heads, head_dim)
                        g_head[(M, h)].append(src.sum(-1))
                        g_head_t[(M, h)].append(src[brows, t_pos])
                        g_alpha_t[(M, h)].append(alpha[brows, t_pos])
            for k in per_head:
                per_head[k].append(torch.cat(g_head[k]))
                per_head_t[k].append(torch.cat(g_head_t[k]))
                per_head_alpha_t[k].append(torch.cat(g_alpha_t[k]))
            for M in cand_layers:
                totals[M].append(torch.cat(g_tot[M]))
            preds.append(torch.cat(g_pred))
            print(f"[mover] pass 1 [{regime.name}]: grid {gi + 1}/{len(grid)}")

        rows, all_dose_rows = [], []
        # the grid point closest to "no perturbation": offset 0 for `local`, the
        # neuron's own mean for a global grid. The threshold statistic is measured
        # against it, since that is where the head sits when nothing unusual happened.
        centre_gi = int(np.argmin(np.abs(regime.grid - (0.0 if regime.is_local
                                                        else stats["mean"]))))
        for (M, h), vals in per_head.items():
            slope, r2 = slopes_over_grid(vals, regime.regressor)
            slope_t, r2_t = slopes_over_grid(per_head_t[(M, h)], regime.regressor)
            alpha_curve = np.array([float(x.mean()) for x in per_head_alpha_t[(M, h)]])
            contrib_curve = np.array([float(x.mean()) for x in per_head_t[(M, h)]])
            for gi, vv in enumerate(regime.grid):
                all_dose_rows.append({
                    "run": run.config.experiment_name, "regime": regime.name,
                    "signal_layer": L, "signal_index": i, "layer": M, "head": h,
                    "rel_depth": M - L, "v": float(vv),
                    "attn_to_t": alpha_curve[gi], "contribution_at_t": contrib_curve[gi]})
            a_centre, a_max = float(alpha_curve[centre_gi]), float(alpha_curve.max())
            rows.append({"run": run.config.experiment_name, "signal_layer": L,
                         "signal_index": i, "layer": M, "head": h,
                         "mean_slope": slope, "mean_abs_slope": abs(slope), "mean_r2": r2,
                         "mean_slope_at_t": slope_t, "mean_abs_slope_at_t": abs(slope_t),
                         "mean_r2_at_t": r2_t,
                         # can exceed 1, and legitimately: sources cancel, so the
                         # contribution routed through t may be larger than the head's
                         # net total. Read it as "is t where this head's v-dependence
                         # lives", not as a share of a positive budget.
                         "frac_slope_from_t": (abs(slope_t) / abs(slope)
                                               if abs(slope) > 1e-12 else np.nan),
                         "rel_depth": M - L,
                         # threshold statistics: how much the head's attention to t
                         # MOVES over the sweep, against where it sits unperturbed.
                         # ratio >> 1 is a detector switching on; ~1 is static
                         # attention, whose v-dependence (if any) is OV-mediated.
                         "attn_t_at_centre": a_centre, "attn_t_max": a_max,
                         "attn_t_min": float(alpha_curve.min()),
                         "attn_t_ratio": (a_max / a_centre if a_centre > 1e-9 else np.inf),
                         "attn_t_delta": a_max - a_centre})
        df = pd.DataFrame(rows).sort_values("mean_abs_slope", ascending=False)
        df["rank"] = np.arange(1, len(df) + 1)
        # A SECOND ranking, by the contribution routed through source position t.
        # Total slope does not mean "mover": it is large both for the head that FETCHES
        # v from the origin and for any later head that re-reads v from wherever the
        # first one deposited it. On joint_N0.1377_POS-12 the top head by total slope
        # (L14h15) took 0.0% of its v-dependence from t and ~99% from the prompt region,
        # while the actual transporter (L1h31) took 98.6% from t.
        df = df.sort_values("mean_abs_slope_at_t", ascending=False)
        df["rank_at_t"] = np.arange(1, len(df) + 1)
        df = df.sort_values("rank")

        err = []
        for M in cand_layers:
            by_head = sum(torch.stack(per_head[(M, h)], 0) for h in range(n_heads))
            err.append(float((by_head - torch.stack(totals[M], 0)).abs().max()))
        e2e = np.corrcoef(torch.stack(preds, 0).numpy().reshape(-1),
                          np.repeat(regime.regressor, len(dataset)))[0, 1]
        print(f"[sanity] max|sum_h sum_s contribution - o_proj total| = {max(err):.3e} "
              f"(should be ~0) | corr(prediction, injected v) = {e2e:.4f}")
        return (df, max(err), float(e2e), torch.stack(preds, 1),   # preds: [Nexp, grid]
                all_dose_rows)

    def pass2(regime, heads, why_of):
        """Source-position and origin-component decompositions for selected heads.

        `why_of` maps (layer, head) -> the criterion that selected it, and is written
        into every row: a reader of the figures should never have to guess whether a
        head is there because it was the biggest mover overall or because it was the
        best in the layer above the signal, since those mean different things.
        """
        grid = regime.grid
        src_acc = {(int(M), int(h)): [] for M, h in heads}
        alpha_acc = {k: [] for k in src_acc}
        comp_acc = {k: {c: [] for c in comp_names_for(k[0])} for k in src_acc}
        tot_t_acc = {k: [] for k in src_acc}
        for gi in range(len(grid)):
            g_src = {k: [] for k in src_acc}
            g_alpha = {k: [] for k in alpha_acc}
            g_comp = {k: {c: [] for c in comp_names_for(k[0])} for k in comp_acc}
            g_tot_t = {k: [] for k in tot_t_acc}
            row0 = 0
            for batch in loader():
                batch = batch.to(device)
                nrow = batch["input_ids"].shape[0]
                v = regime.values_for(
                    None if not regime.is_local else centres[row0:row0 + nrow], gi)
                if torch.is_tensor(v):
                    v = v.to(dtype=inject_dtype)
                row0 += nrow
                cap = run_injected_batch(run, batch, v, sorted({M for M, _ in src_acc}),
                                         layer_out_tuple)
                a = readout_scale(cap["final_resid"], w, gamma, eps)
                t = cap["abs_pos"]
                brows = torch.arange(t.shape[0])
                for (M, h) in src_acc:
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
                src_acc[k].append(torch.cat(g_src[k]))
                alpha_acc[k].append(torch.cat(g_alpha[k]))
                tot_t_acc[k].append(torch.cat(g_tot_t[k]))
                for c in comp_names_for(k[0]):
                    comp_acc[k][c].append(torch.cat(g_comp[k][c]))
            print(f"[mover] pass 2 [{regime.name}]: grid {gi + 1}/{len(grid)}")

        src_rows, comp_rows, dose_rows = [], [], []
        t_bin = SOURCE_REL_WINDOW                     # index of the "t" bin in _BIN_LABELS
        for (M, h) in src_acc:
            stacked = torch.stack(src_acc[(M, h)], 0)
            alphas = torch.stack(alpha_acc[(M, h)], 0)      # [grid, Nexp, bins]
            # attention to t AT EACH GRID POINT, not just its average over them.
            # A head that ROUTES the value should attend to t in a graded way; a head
            # that merely notices an outlier should switch on past a threshold and
            # saturate. The grid-averaged number cannot tell those apart -- four
            # layer-1 heads all averaging ~0.42 is exactly what a step function
            # sampled over a symmetric grid looks like.
            for gi, v in enumerate(grid):
                dose_rows.append({"run": run.config.experiment_name, "layer": M, "head": h,
                                  "selected_as": why_of.get((M, h), ""), "v": float(v),
                                  "attn_to_t": float(alphas[gi, :, t_bin].mean()),
                                  "contribution_at_t": float(stacked[gi, :, t_bin].mean())})
            for b in range(stacked.shape[-1]):
                rel = b - SOURCE_REL_WINDOW
                col = [stacked[g, :, b] for g in range(stacked.shape[0])]
                slope, r2 = slopes_over_grid(col, regime.regressor)
                src_rows.append({"run": run.config.experiment_name, "layer": M, "head": h,
                                 "selected_as": why_of.get((M, h), ""),
                                 "rel_source": rel if b < 2 * SOURCE_REL_WINDOW + 1 else None,
                                 "bin": _BIN_LABELS[b], "mean_slope": slope, "mean_r2": r2,
                                 "mean_attention": float(alphas[:, :, b].mean())})
            tot_t = torch.stack(tot_t_acc[(M, h)], 0)
            summed = sum(torch.stack(comp_acc[(M, h)][c], 0) for c in comp_names_for(M))
            print(f"[sanity] head L{M}.h{h}: max|sum_components - contribution via t| = "
                  f"{float((summed - tot_t).abs().max()):.3e} "
                  f"(decomposes embed + {M} attn + {M} mlp writes at t)")
            for c in comp_names_for(M):
                slope, r2 = slopes_over_grid(comp_acc[(M, h)][c], regime.regressor)
                comp_rows.append({"run": run.config.experiment_name, "layer": M, "head": h,
                                  "selected_as": why_of.get((M, h), ""),
                                  "component": c, "mean_slope": slope, "mean_r2": r2,
                                  "is_signal_layer_mlp": c == f"mlp_{L}"})
        return src_rows, comp_rows, dose_rows

    # ---- passes 1 and 2, per regime ----
    heads_frames, src_rows, comp_rows, dose_rows, sanity = [], [], [], [], {}
    all_dose_rows = []
    per_regime_heads = {}
    unperturbed_pred = None
    for rg in regimes:
        print(f"\n[mover] ===== {rg.name} =====")
        df, max_err, e2e, preds, adr = pass1(rg)
        all_dose_rows += adr
        df.insert(0, "regime", rg.name)
        heads_frames.append(df)
        per_regime_heads[rg.name] = select_heads(df, L, args.top_heads)
        sanity[rg.name] = {"kind": rg.kind, "scale": rg.scale,
                           "delta_sigma": rg.delta_sigma,
                           "grid": [float(x) for x in rg.grid],
                           "per_source_max_err": max_err, "e2e_corr_pred_vs_v": e2e}
        # `local` only: writing the neuron's own value back must leave the prediction
        # untouched, which is what certifies the centres and the row alignment.
        zi = zero_offset_index(rg)
        if zi is not None:
            if unperturbed_pred is None:
                from trace_introspection_circuit import endogenous_predictions
                unperturbed_pred, _ = endogenous_predictions(
                    run.model, dataset, run.collator, args.batch_size)
            sanity[rg.name]["delta0_identity_err"] = identity_check(
                rg, preds[:, zi].numpy(), unperturbed_pred.numpy(),
                label=run.config.experiment_name)
        power_warning(rg, float(df.nsmallest(1, "rank")["mean_r2"].iloc[0]),
                      label=run.config.experiment_name)

    # The SAME head set is decomposed in every regime, so they can be compared head by
    # head: "is the transporter at 10x still the transporter locally" is unanswerable
    # if each regime only decomposes its own winners.
    top = list({k: why for rg in regimes for k, why in per_regime_heads[rg.name]}.items())
    top_heads = [k for k, _ in top]          # ((layer, head), why) -> (layer, head)
    why_of = {k: why for k, why in top}
    print("\n[mover] heads taken into the decompositions (union over regimes):")
    for (M, h), why in top:
        bits = []
        for k, rg in enumerate(regimes):
            r = heads_frames[k]
            r = r[(r["layer"] == M) & (r["head"] == h)].iloc[0]
            bits.append(f"{rg.name}: rank={int(r['rank'])} rank@t={int(r['rank_at_t'])}")
        print(f"[mover]   L{M}h{h:<2} ({why:<16}) " + " | ".join(bits))

    for rg in regimes:
        print(f"\n[mover] ===== pass 2 [{rg.name}] =====")
        sr, cr, dr = pass2(rg, top_heads, why_of)
        for r in sr + cr + dr:
            r["regime"] = rg.name
        src_rows += sr
        comp_rows += cr
        dose_rows += dr

    heads_df = pd.concat(heads_frames, ignore_index=True)
    heads_df["selected_as"] = [why_of.get((int(L_), int(h_)), "")
                               for L_, h_ in zip(heads_df["layer"], heads_df["head"])]
    heads_df.to_csv(out_dir / "mover_heads.csv", index=False)
    src_df = pd.DataFrame(src_rows)
    src_df.to_csv(out_dir / "mover_source_positions.csv", index=False)
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out_dir / "mover_origin_components.csv", index=False)
    dose_df = pd.DataFrame(dose_rows)
    dose_df.to_csv(out_dir / "mover_attention_vs_v.csv", index=False)
    # every head above the signal layer, not just the decomposed ones: this is the
    # frame the cross-run head taxonomy is built from
    pd.DataFrame(all_dose_rows).to_csv(out_dir / "mover_head_dose_all.csv", index=False)
    _plot_attention_vs_v(run, dose_df, out_dir)

    # ---- pass 3: WHERE does each head look, in each regime ----
    #
    # The attention a mover is identified by is measured under injection, and the
    # injection is itself a large perturbation of the source token's key: on
    # joint_N0.1377_POS-12 the transporter L1h31 attends to t with weight 6.3e-05
    # clean and 0.42 under a 10x injection. A clean-only panel therefore describes a
    # different model than the rest of the script. Every regime is measured instead,
    # over a fixed set of token bins, so "where is it looking if not at t" has an
    # answer on the same figure (on that run: BOS, at 0.95).
    # AVERAGED OVER THE DOSE-RESPONSE GRID, not measured at one extreme point. The
    # attention a large injection produces SATURATES: on joint_N0.1377_POS-12 at
    # v=+2.49 (the top grid point) head L1h22 takes 100% of the readout's attention and
    # every other head's share of t collapses to ~0, while the grid-averaged figure has
    # four layer-1 heads sitting near 0.42 each. A single endpoint therefore describes
    # the endpoint, not the regime, and disagreed with pass 2's own numbers.
    prof_rows, addr_rows = measure_attention_profile(
        run, top_heads, why_of, loader, regimes, centres, inject_dtype, layer_out_tuple,
        device)
    prof_df = pd.DataFrame(prof_rows)
    prof_df.to_csv(out_dir / "mover_attention_profile.csv", index=False)

    # ---- and the same profile for EVERY head, endogenously (taxonomy input) ----
    print(f"[mover] endogenous attention profile over all {len(cand_layers) * n_heads} "
          f"heads above layer {L} ...")
    pd.DataFrame(endogenous_all_head_profile(
        run, loader, cand_layers, layer_out_tuple, device, L, i)
    ).to_csv(out_dir / "mover_attention_profile_all_heads.csv", index=False)

    # ---- writer vs mover: per-head gain onto the target direction vs the null ----
    print(f"[mover] endogenous transport gain: {len(cand_layers) * n_heads} heads x "
          f"(target + cellmates + {N_RANDOM_DIRECTIONS} random directions) ...")
    tg_rows = endogenous_transport_gain(
        run, loader, cand_layers, layer_out_tuple, device, W_O, W_V, gammas, w,
        gamma, eps, n_heads, n_kv_heads, head_dim)
    pd.DataFrame(tg_rows).to_csv(out_dir / "mover_transport_gain.csv", index=False)
    tg_sum = summarize_transport_gain(tg_rows)
    # ov_share: the analytic OV-only gain as a fraction of the MEASURED slope at t at the
    # operating point (the `local` regime). ~1 means the head transports at fixed
    # attention; ~0 means its transport is attention moving. Only meaningful where the
    # direct path is the whole path (rel_depth 1) -- flagged, not dropped, beyond that.
    local = [rg.name for rg in regimes if rg.is_local]
    if local and not tg_sum.empty:
        loc = (heads_df[heads_df["regime"] == local[0]]
               .set_index(["layer", "head"])["mean_slope_at_t"].rename("slope_at_t_local"))
        tg_sum = tg_sum.merge(loc.reset_index(), on=["layer", "head"], how="left")
        tg_sum["ov_share"] = np.where(tg_sum["slope_at_t_local"].abs() > 1e-15,
                                      tg_sum["gain_target_raw"] / tg_sum["slope_at_t_local"],
                                      np.nan)
        tg_sum["ov_share_direct_path"] = tg_sum["rel_depth"] == 1
    tg_sum.to_csv(out_dir / "mover_transport_summary.csv", index=False)
    if not tg_sum.empty:
        # NOT named `top`: that name holds the selected heads, which the meta json
        # below iterates -- shadowing it with a frame made that loop unpack column names
        tg_top = tg_sum.reindex(
            tg_sum["gain_target_raw"].abs().sort_values(ascending=False).index)
        print("[mover] top heads by endogenous gain on the target column:")
        print(tg_top.head(6)[["layer", "head", "gain_target_raw", "writer_share",
                              "z_edit_vs_random", "z_base_vs_random"]].to_string(
            index=False, float_format=lambda v: f"{v:.3g}"))
    addr_df = pd.DataFrame(addr_rows)
    addr_df.to_csv(out_dir / "mover_attention_base_vs_tuned.csv", index=False)

    # the headline contrast, stated rather than left to be spotted in the csv
    for (M, h), g in addr_df[addr_df["adapter"] == "on"].groupby(["layer", "head"]):
        clean = g[g["condition"] == "endogenous"]["attn_to_t"]
        if not len(clean) or float(clean.iloc[0]) <= 0:
            continue
        for _, r in g[g["injected"]].iterrows():
            ratio = r["attn_to_t"] / float(clean.iloc[0])
            if ratio > 10:
                print(f"[mover] NOTE L{M}h{h}: attention to t is {ratio:.0f}x larger at "
                      f"'{r['condition']}' ({r['attn_to_t']:.3f}) than endogenous "
                      f"({float(clean.iloc[0]):.2e}) -- at that scale the probe is "
                      f"creating the attention pattern, not revealing it")
    print("\n[mover] attention from the readout token (adapter on):")
    print(addr_df[addr_df["adapter"] == "on"].to_string(
        index=False, float_format=lambda v: f"{v:.5f}"))

    _plot(run, heads_df, src_df, comp_df, addr_df, prof_df, out_dir)
    with open(out_dir / "mover_meta.json", "w") as f:
        json.dump({"run": run.config.experiment_name, "checkpoint": str(run.checkpoint),
                   "signal": f"N{L}.{i}", "top_heads": [[int(M), int(h)] for M, h in top_heads],
                   "head_selection": {f"L{M}h{h}": why for (M, h), why in top},
                   "regimes": list(sanity.keys()), "training_scale": float(train_scale),
                   "sanity_per_regime": sanity,
                   "local_centre_source": "live tuned activation at (L, i, t)",
                   "n_examples": len(dataset)}, f, indent=2)
    if prev_impl is not None:
        cfg._attn_implementation = prev_impl
    print(f"[mover] wrote {out_dir}")


@torch.no_grad()
def measure_attention_profile(run, heads, why_of, loader, regimes, centres, inject_dtype,
                              layer_out_tuple, device):
    """Pass 3: where each head in `heads` looks from the readout, endogenously and in
    every regime (grid-averaged), adapter on and off. Returns (profile rows over token
    bins, per-head addressing rows). Needs eager attention (set by the caller)."""
    top_heads = heads
    conditions = [("endogenous", None)] + [(rg.name, rg) for rg in regimes]
    prof_rows, addr_rows = [], []
    for disable in (False, True):
        for cond_name, cond_regime in conditions:
            acc = {(int(M), int(h)): None for M, h in top_heads}
            n_rows_seen = 0
            n_points = 1 if cond_regime is None else len(cond_regime.grid)
            for gi in range(n_points):
                row0 = 0
                for batch in loader():
                    batch = batch.to(device)
                    nrow = batch["input_ids"].shape[0]
                    if cond_regime is None:
                        inj = None
                    else:
                        inj = cond_regime.values_for(
                            None if not cond_regime.is_local else centres[row0:row0 + nrow],
                            gi)
                        if torch.is_tensor(inj):
                            inj = inj.to(dtype=inject_dtype)
                    row0 += nrow
                    cap = run_injected_batch(run, batch, inj, sorted({M for M, _ in top_heads}),
                                             layer_out_tuple, disable_adapter=disable)
                    t = cap["abs_pos"]
                    num_left_pad = (batch["attention_mask"] == 0).sum(dim=1).cpu()
                    n_rows_seen += t.shape[0]
                    for (M, h) in acc:
                        alpha = cap["attn_probs"][M][:, h, :]
                        binned = _attention_bins(alpha, t, num_left_pad)
                        acc[(M, h)] = binned if acc[(M, h)] is None else {
                            k: acc[(M, h)][k] + v for k, v in binned.items()}
            for (M, h), sums in acc.items():
                for label, total in sums.items():
                    prof_rows.append({
                        "run": run.config.experiment_name, "layer": M, "head": h,
                        "selected_as": why_of.get((M, h), ""),
                        "adapter": "off" if disable else "on", "condition": cond_name,
                        "injected": cond_regime is not None, "token_bin": label,
                        "mean_attention": float(total) / max(1, n_rows_seen)})
                addr_rows.append({
                    "run": run.config.experiment_name, "layer": M, "head": h,
                    "selected_as": why_of.get((M, h), ""),
                    "adapter": "off" if disable else "on", "condition": cond_name,
                    "injected": cond_regime is not None,
                    "attn_to_t": float(sums["t"]) / max(1, n_rows_seen),
                    "attn_to_bos": float(sums["BOS"]) / max(1, n_rows_seen),
                    "attn_to_final": float(sums["final"]) / max(1, n_rows_seen)})
    return prof_rows, addr_rows


# Token bins the attention profile is reported over. BOS is absolute (Llama's
# attention sink); the t window is where the introspected signal lives; the final
# window is the readout end of the introspection prompt. Everything else is pooled
# into two catch-alls so the bins sum to 1 and a missing mass is visible.
ATTENTION_BINS = (["BOS"] + [f"t{d:+d}" if d else "t" for d in range(-3, 4)]
                  + [f"final{d:+d}" if d else "final" for d in range(-3, 1)]
                  + ["other_text", "other_prompt"])


N_RANDOM_DIRECTIONS = 20


@torch.no_grad()
def endogenous_transport_gain(run, loader, layers, layer_out_tuple, device, W_O, W_V,
                              gammas, w, gamma_final, eps, n_heads, n_kv_heads, head_dim,
                              n_random=N_RANDOM_DIRECTIONS):
    """
    How strongly each head carries a given DIRECTION written at t to the readout, with
    nothing injected -- the quantity that separates the writer from the mover.

    For head h at layer M and a direction d written into the residual at t, the direct
    path to the prediction at frozen attention and frozen norms is

        gain_h(d) = alpha_h(last -> t) * (a . W_O^h W_V^kv (gamma_M * d)) / rms_M(t)

    averaged over examples (a = readout covector). It is linear in d, so it is computed
    for UNIT directions and the raw gain of a column is its unit gain times its norm.

    Directions:
      target      the run's trained neuron's effective down_proj column
      target_base / target_edit
                  that column split into the base model's part and the LoRA edit -- the
                  writer is the edit, so its share of a head's gain is the writer's share
      cellmate    the other neurons of the same (layer, position) cell, same layer's
                  down_proj: untrained columns, same geometry
      random      unit vectors in the residual stream: the null for "this head reads
                  whatever direction happens to be at t"

    Reading it: a head whose unit-gain on the target sits far above its random-direction
    distribution is ALIGNED to the trained channel -- a signal-specific mover. One whose
    target gain is typical of random directions is signal-agnostic: it carries whatever
    is at t, and any selectivity for the target comes from the WRITER making that column
    large (col_norm), not from the mover. The two are exactly the writer/mover split.

    THIS IS THE OV-MEDIATED PATH ONLY. The full derivative of a head's contribution at t
    is  alpha * d(OV)/dv  +  OV * d(alpha)/dv,  and this computes the first term. An
    earlier version of this docstring claimed freezing attention was "nearly exact"
    endogenously because alpha is small there (~1e-3). That was wrong: small alpha is
    not small d(alpha)/dv RELATIVE to alpha. Measured on joint_N0.1377_POS-12 at rel_depth
    1, this gain reproduces the ORDERING of the finite-difference slope almost perfectly
    (corr 0.997 over 32 heads) but only 4-14% of its MAGNITUDE on the material heads,
    whose attention to t moves 5-7x across +/-1 sigma. So even at the operating point
    those heads transport mostly by attention moving, not through a fixed-attention OV
    path. The ratio of the two is reported as `ov_share` (see trace_movers), which is the
    threshold-free version of the transporter/grabber distinction.

    Valid as a DIRECT-path gain: at rel_depth 1 no layer sits between the write and the
    head. Deeper, the write has been read and rewritten by intermediate layers, the
    direct path is only part of what reaches the head, and agreement with the measured
    slope breaks down (corr ~0.3 at rel_depth 2, near 0 beyond) -- as it should.
    """
    L, i = run.layer, run.index
    mod = run.base.model.layers[L].mlp.down_proj
    down = effective_weight(mod)                                                # [H, I]
    base_w = getattr(mod, "base_layer", mod).weight.detach().float().cpu()
    cellmates = [int(s.index) for s in run.config.train_signals
                 if int(s.layer) == L and int(s.index) != i]
    rng = np.random.RandomState(L * 100003 + i)
    rand = torch.from_numpy(rng.randn(down.shape[0], n_random)).float()
    # The target column split into what the base model already wrote and what training
    # ADDED. The edit is near-orthogonal to the base column (cosine ~0 in the weight
    # analysis), so it barely changes the column's norm -- a norm ratio cannot see the
    # writer at all. The gain is linear in the direction, so
    #   gain(target) == gain(target_base) + gain(target_edit)
    # exactly, and the edit's share is the writer's share of this head's transport.
    dirs = [("target", "target", down[:, i]),
            ("target_base", "target_base", base_w[:, i]),
            ("target_edit", "target_edit", down[:, i] - base_w[:, i])]
    dirs += [(f"cellmate_{j}", "cellmate", down[:, j]) for j in cellmates]
    dirs += [(f"random_{k}", "random", rand[:, k]) for k in range(n_random)]
    D = torch.stack([d for _, _, d in dirs], 1)                                 # [H, K]
    norms = D.norm(dim=0)
    Du = D / norms.clamp(min=1e-12)
    group = n_heads // n_kv_heads

    sums = {M: torch.zeros(n_heads, D.shape[1]) for M in layers}
    n_seen = 0
    for batch in loader():
        batch = batch.to(device)
        cap = run_injected_batch(run, batch, None, layers, layer_out_tuple)
        a = readout_scale(cap["final_resid"], w, gamma_final, eps)              # [B, H]
        t = cap["abs_pos"]
        rows = torch.arange(t.shape[0])
        n_seen += t.shape[0]
        for M in layers:
            U = (a @ W_O[M]).view(-1, n_kv_heads, group, head_dim)               # [B,g,j,hd]
            V = (W_V[M] @ (gammas[M].unsqueeze(1) * Du)).view(n_kv_heads, head_dim, -1)
            dot = torch.einsum("bgjd,gdk->bgjk", U, V).reshape(U.shape[0], n_heads, -1)
            x = cap[f"pre_{M}"]
            rms = x.pow(2).mean(-1).add(eps).sqrt()                              # [B]
            alpha_t = cap["attn_probs"][M][rows, :, t]                           # [B, heads]
            sums[M] += (alpha_t.unsqueeze(-1) * dot / rms.view(-1, 1, 1)).sum(0)

    out = []
    for M in layers:
        g = sums[M] / max(1, n_seen)                                             # [heads, K]
        for h in range(n_heads):
            for k, (name, kind, _) in enumerate(dirs):
                out.append({"run": run.config.experiment_name, "signal_layer": L,
                            "signal_index": i, "layer": M, "head": h, "rel_depth": M - L,
                            "direction": name, "direction_kind": kind,
                            "col_norm": float(norms[k]), "gain_unit": float(g[h, k]),
                            "gain_raw": float(g[h, k] * norms[k])})
    return out


def summarize_transport_gain(rows) -> pd.DataFrame:
    """Per head: target-direction gain against the random null and the cellmates.

    writer_share        gain(target_edit) / gain(target): how much of this head's
                        transport of the trained neuron rides on the direction TRAINING
                        added rather than the one the base model already wrote.
    z_edit_vs_random    is the head's OV aligned to that trained direction, beyond what a
                        random direction of the same norm gets? High = a signal-specific
                        mover reading the private wire; ~0 = a signal-agnostic mover that
                        carries whatever is at t.
    z_base_vs_random    the same for the base model's own column.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = []
    def z(val, rnd, sd):
        return (abs(val) - rnd.mean()) / sd if sd and sd > 0 else np.nan

    for (M, h), g in df.groupby(["layer", "head"]):
        tgt = g[g["direction_kind"] == "target"].iloc[0]
        base = g[g["direction_kind"] == "target_base"].iloc[0]
        edit = g[g["direction_kind"] == "target_edit"].iloc[0]
        rnd = g[g["direction_kind"] == "random"]["gain_unit"].abs()
        cm = g[g["direction_kind"] == "cellmate"]
        sd = float(rnd.std()) if len(rnd) > 1 else np.nan
        out.append({"run": tgt["run"], "signal_layer": tgt["signal_layer"],
                    "signal_index": tgt["signal_index"], "layer": M, "head": h,
                    "rel_depth": tgt["rel_depth"],
                    "gain_target_unit": tgt["gain_unit"], "gain_target_raw": tgt["gain_raw"],
                    "gain_random_unit_abs_mean": float(rnd.mean()) if len(rnd) else np.nan,
                    "gain_base_raw": base["gain_raw"], "gain_edit_raw": edit["gain_raw"],
                    "writer_share": (edit["gain_raw"] / tgt["gain_raw"]
                                     if abs(tgt["gain_raw"]) > 1e-15 else np.nan),
                    "edit_norm": edit["col_norm"], "base_norm": base["col_norm"],
                    "z_target_vs_random": z(tgt["gain_unit"], rnd, sd),
                    "z_edit_vs_random": z(edit["gain_unit"], rnd, sd),
                    "z_base_vs_random": z(base["gain_unit"], rnd, sd),
                    "gain_cellmate_unit_abs_mean": (float(cm["gain_unit"].abs().mean())
                                                    if len(cm) else np.nan),
                    "gain_cellmate_raw_abs_mean": (float(cm["gain_raw"].abs().mean())
                                                   if len(cm) else np.nan)})
    return pd.DataFrame(out)


@torch.no_grad()
def endogenous_all_head_profile(run, loader, layers, layer_out_tuple, device,
                                signal_layer, neuron):
    """
    Where EVERY head looks with nothing injected, binned like the selected-head
    profile.

    One pass, no injection: this supplies the static-target label (BOS / final /
    prompt / text) for the head taxonomy, which the per-head alpha-vs-v curves from
    pass 1 cannot give. Only the endogenous condition is measured for all heads --
    the v-dependence is already covered head-by-head by those curves, and repeating
    the full condition grid over every head would multiply the run by the number of
    layers for no extra claim.
    """
    acc, n_seen = {}, 0
    for batch in loader():
        batch = batch.to(device)
        cap = run_injected_batch(run, batch, None, layers, layer_out_tuple)
        t = cap["abs_pos"]
        num_left_pad = (batch["attention_mask"] == 0).sum(dim=1).cpu()
        n_seen += t.shape[0]
        for M in layers:
            probs = cap["attn_probs"][M]                      # [B, heads, seq]
            for h in range(probs.shape[1]):
                binned = _attention_bins(probs[:, h, :], t, num_left_pad)
                key = (M, h)
                acc[key] = binned if key not in acc else {
                    k: acc[key][k] + v for k, v in binned.items()}
    rows = []
    for (M, h), sums in acc.items():
        for label, total in sums.items():
            rows.append({"run": run.config.experiment_name, "signal_layer": signal_layer,
                         "signal_index": neuron, "layer": M, "head": h,
                         "rel_depth": M - signal_layer, "condition": "endogenous",
                         "adapter": "on", "token_bin": label,
                         "mean_attention": float(total) / max(1, n_seen)})
    return rows


def _attention_bins(alpha, t, num_left_pad):
    """{bin label: summed attention over rows} for one batch, one head.

    Bins can overlap when the sequence is short (t+3 may BE final-3); the sum over
    bins is therefore not forced to 1 and `other_*` is computed by subtracting only
    the disjoint part. Every row contributes its full attention mass somewhere.
    """
    bs, seq = alpha.shape
    rows = torch.arange(bs)
    last = seq - 1
    out, claimed = {}, torch.zeros_like(alpha, dtype=torch.bool)

    def take(label, pos):
        valid = (pos >= num_left_pad) & (pos <= last)
        p = pos.clamp(0, last)
        vals = alpha[rows, p] * valid.float()
        out[label] = float(vals.sum())
        claimed[rows[valid], p[valid]] = True

    take("BOS", num_left_pad.clone())
    for d in range(-3, 4):
        take(f"t{d:+d}" if d else "t", t + d)
    for d in range(-3, 1):
        take(f"final{d:+d}" if d else "final", torch.full_like(t, last) + d)
    # the rest, split by region: text is everything at or before t, prompt after it
    rest = alpha.masked_fill(claimed, 0.0)
    pos_idx = torch.arange(seq).unsqueeze(0).expand(bs, -1)
    is_text = pos_idx <= t.unsqueeze(1)
    out["other_text"] = float((rest * is_text.float()).sum())
    out["other_prompt"] = float((rest * (~is_text).float()).sum())
    return out


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


def _head_label(L, h, why=""):
    """'L3h1 [top at t]' -- the selection criterion travels with the head everywhere it
    is drawn, because "top total" and "top at t" are different claims (see
    select_heads) and a reader cannot tell them apart from the head id alone."""
    why = "" if not isinstance(why, str) or not why else f" [{why}]"
    return f"L{int(L)}h{int(h)}{why}"


def _plot_attention_vs_v(run, dose_df, out_dir, suffix=""):
    """Attention to t as a function of the injected value, per head, per scale.

    This is the graded-vs-threshold test. A head that routes the value should show
    attention rising smoothly with |v|; a head that is merely detecting an out-of-
    distribution activation should switch on past a threshold and saturate, and its
    grid-AVERAGED attention is then an artefact of where the grid points happen to
    fall rather than a property of the head.
    """
    if dose_df is None or dose_df.empty:
        return
    names = list(dict.fromkeys(dose_df["regime"]))
    fig, axes = plt.subplots(1, len(names), figsize=(1 + 4.0 * len(names), 3.4),
                             squeeze=False, sharey=True)
    for ax, sc in zip(axes.flat, names):
        d = dose_df[dose_df["regime"] == sc]
        for (M, h), g in d.groupby(["layer", "head"]):
            g = g.sort_values("v")
            why = g["selected_as"].fillna("").iloc[0] if "selected_as" in g else ""
            ax.plot(g["v"], g["attn_to_t"], "o-", ms=4, label=_head_label(M, h, why))
        ax.set_xlabel("injected value (offset from centre for `local`)")
        ax.grid(alpha=0.25)
        ax.set_title(sc, fontsize=9); ax.tick_params(labelsize=8)
    axes.flat[0].set_ylabel("attention: readout -> t", fontsize=9)
    fig.suptitle("Graded routing or threshold detection?\n"
                 "attention to the introspected token vs the value injected there",
                 fontsize=9, y=1.04)
    fig.tight_layout()
    # below the panels: head labels carry their selection reason (and, for knockout
    # heads, their drops), which is too long for a legend inside an axis
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=2 if max(map(len, labels), default=0) < 60 else 1, fontsize=7,
               frameon=False)
    path = out_dir / f"mover_attention_vs_v{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[mover] wrote {path}")


def _plot_attention_profile(run, prof_df, out_dir, suffix="", heads_desc="the movers"):
    """Where each selected head looks, per regime: heads x token bins, one panel per
    condition, adapter ON. Attention is averaged over the dose-response grid, so a
    cell is what the head does across the regime rather than at one saturating endpoint.

    Log colour scale, because that is the range the quantity actually spans (1e-5 to
    1e0 on the same head) and a linear map shows one dark row and nothing else. The
    panels are the point: if the t column only lights up in the injected panels, the
    head does not natively look at the introspected token.
    """
    if prof_df.empty:
        return
    from matplotlib.colors import LogNorm
    d = prof_df[prof_df["adapter"] == "on"]
    conds = [c for c in d["condition"].unique()]
    why = {}
    if "selected_as" in d:
        why = {(int(L), int(h)): w for L, h, w in
               zip(d["layer"], d["head"], d["selected_as"].fillna(""))}
    heads = sorted({(int(L), int(h)) for L, h in zip(d["layer"], d["head"])})
    bins = [b for b in ATTENTION_BINS if b in set(d["token_bin"])]
    if not conds or not heads or not bins:
        return
    vmin = max(1e-6, float(d[d["mean_attention"] > 0]["mean_attention"].min()))
    vmax = float(d["mean_attention"].max())
    fig, axes = plt.subplots(1, len(conds), figsize=(1 + 3.6 * len(conds), 0.42 * len(heads) + 2.2),
                             squeeze=False, sharey=True)
    for ax, cond in zip(axes.flat, conds):
        sub = d[d["condition"] == cond]
        mat = np.full((len(heads), len(bins)), np.nan)
        for r in sub.itertuples():
            mat[heads.index((int(r.layer), int(r.head))), bins.index(r.token_bin)] = r.mean_attention
        im = ax.imshow(np.clip(mat, vmin, None), aspect="auto", cmap="magma",
                       norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)))
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins, rotation=90, fontsize=7)
        ax.set_yticks(range(len(heads)))
        ax.set_yticklabels([_head_label(L, h, why.get((L, h), "")) for L, h in heads],
                           fontsize=7)
        ax.set_title(cond, fontsize=9)
        if "t" in bins:
            ax.axvline(bins.index("t") - 0.5, color="cyan", lw=0.8)
            ax.axvline(bins.index("t") + 0.5, color="cyan", lw=0.8)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="mean attention from readout", pad=0.02)
    fig.suptitle(f"Where {heads_desc} look, by regime -- {run.config.experiment_name}",
                 fontsize=9, y=1.02)
    path = out_dir / f"mover_attention_profile{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[mover] wrote {path}")


def _plot(run, heads_df, src_df, comp_df, addr_df, prof_df, out_dir):
    # one three-panel figure PER INJECTION SCALE: the panels describe a circuit at a
    # particular injection magnitude, and overlaying scales would average a mechanism
    # with whatever replaces it at another scale
    scales = (list(dict.fromkeys(heads_df["regime"].dropna()))
              if "regime" in heads_df else [None])
    for sc in scales:
        sel = (lambda d: d if sc is None or "regime" not in d
               else d[d["regime"] == sc])
        _plot_scale(run, sel(heads_df), sel(src_df), sel(comp_df), sc, out_dir)
    _plot_attention_profile(run, prof_df, out_dir)

    if not addr_df.empty and "condition" in addr_df:
        # attention to t alone, with the adapter as the contrast -- the inherited-
        # addressing question, asked separately in each injection regime. Log y: the
        # regimes differ by orders of magnitude on one head, and on a linear axis the
        # endogenous bars vanish, which is the comparison the panel exists for.
        conds = list(addr_df["condition"].unique())
        fig, axes = plt.subplots(1, len(conds), figsize=(1 + 3.2 * len(conds), 3.4),
                                 squeeze=False, sharey=True)
        for ax, cond in zip(axes.flat, conds):
            sub = addr_df[addr_df["condition"] == cond]
            piv = sub.pivot_table(index=["layer", "head"], columns="adapter", values="attn_to_t")
            why = ({(int(L), int(h)): w for L, h, w in
                    zip(sub["layer"], sub["head"], sub["selected_as"].fillna(""))}
                   if "selected_as" in sub else {})
            piv.index = [_head_label(L, h, why.get((int(L), int(h)), "")) for L, h in piv.index]
            piv.plot.bar(ax=ax, width=0.78, rot=90, legend=(ax is axes.flat[0]),
                         color={"on": "tab:red", "off": "tab:grey"})
            ax.set_yscale("log"); ax.set_xlabel("")
            ax.tick_params(labelsize=7); ax.grid(axis="y", alpha=0.25, which="both")
            ax.set_title(cond, fontsize=9)
        axes.flat[0].set_ylabel("mean attention: readout -> t", fontsize=9)
        axes.flat[0].legend(fontsize=7, title="adapter", title_fontsize=7)
        fig.suptitle("Is the addressing inherited? (adapter on vs off, per regime)",
                     fontsize=9, y=1.03)
        fig.tight_layout()
        fig.savefig(out_dir / "mover_addressing.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def _plot_scale(run, heads_df, src_df, comp_df, scale, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    top = heads_df.nsmallest(12, "rank") if "rank" in heads_df else heads_df.head(12)
    sel = (top["selected_as"].fillna("") != "") if "selected_as" in top else None
    labels = [_head_label(r.layer, r.head, getattr(r, "selected_as", ""))
              for r in top.itertuples()]
    # selected heads coloured: this panel ranks by TOTAL slope, and the heads carried
    # into the other two panels are not simply its first three (see select_heads)
    colors = (["tab:red" if x else "tab:blue" for x in sel] if sel is not None
              else ["tab:blue"] * len(top))
    axes[0].barh(labels[::-1], top["mean_slope"].values[::-1], color=colors[::-1])
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].tick_params(axis="y", labelsize=7)
    axes[0].set_xlabel("dose-response slope")
    axes[0].set_title("top 12 heads by TOTAL slope\n(red = carried into the panels at right)",
                      fontsize=9)

    for (M, h), g in src_df.groupby(["layer", "head"]):
        why = g["selected_as"].fillna("").iloc[0] if "selected_as" in g else ""
        g = g[g["rel_source"].notna()].sort_values("rel_source")
        axes[1].plot(g["rel_source"], g["mean_slope"], "o-", ms=4,
                     label=_head_label(M, h, why))
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
        why = g["selected_as"].fillna("").iloc[0] if "selected_as" in g else ""
        g = g.assign(x=g["component"].map(xpos)).sort_values("x")
        axes[2].plot(g["x"], g["mean_slope"], "o-", ms=4, label=_head_label(M, h, why))
    axes[2].set_xticks(range(len(order)))
    axes[2].set_xticklabels(order, rotation=90, fontsize=6)
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_ylabel("slope"); axes[2].set_title("which write at t carries it")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.25)
    tag = "" if scale is None else f" ({scale})"
    fig.suptitle(f"Mover-head trace: {run.config.experiment_name}{tag}", y=1.02)
    fig.tight_layout()
    name = "mover_trace.png" if scale is None else f"mover_trace_{scale}.png"
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[mover] wrote {out_dir / name}")


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
          _rd("mover_source_positions.csv",
              ["layer", "head", "selected_as", "rel_source", "mean_slope"]),
          _rd("mover_origin_components.csv",
              ["layer", "head", "selected_as", "component", "mean_slope"]),
          _rd("mover_attention_base_vs_tuned.csv",
              ["layer", "head", "adapter", "condition", "injected", "attn_to_t"]),
          _rd("mover_attention_profile.csv",
              ["layer", "head", "adapter", "condition", "token_bin", "mean_attention"]),
          out_dir)
    return True


# ======================================================================
# Knockout-selected heads
# ======================================================================
KNOCKOUT_ARM_WHY = {"endogenous": "top endogenous", "x1": "top interventional"}
KNOCKOUT_SUFFIX = "_knockout"


def knockout_dir_for(run, spec, args) -> Path:
    if args.knockout_dir:
        return Path(args.knockout_dir)
    return locale_out_dir(spec.run_dir / "head_knockout", spec,
                          (run.signal_type, run.layer, run.index))


KNOCKOUT_ARM_SHORT = {"endogenous": "Endo", "x1": "Interv"}


def select_knockout_heads(ko_df: pd.DataFrame, top_k: int):
    """[((layer, head), why)]: each arm's top_k heads by single-knockout drop (rank 1 =
    largest drop; heads whose knockout helps rank last). A head in both arms' top_k is
    labelled 'top both', so the figure shows the overlap directly.

    `why` also carries the head's importance in EVERY arm -- its correlation drop when
    knocked out alone and its rank among the run's heads -- e.g.
    'top both; drop Endo 0.351 (#1), Interv 0.031 (#3)', so a head picked for one arm
    shows how much the other arm cares about it too."""
    arms_of = {}
    best = {}
    for arm, g in ko_df.groupby("arm", sort=False):
        for r in g.nsmallest(top_k, "rank").itertuples():
            k = (int(r.layer), int(r.head))
            arms_of.setdefault(k, []).append(arm)
            best[k] = min(best.get(k, np.inf), int(r.rank))
    stats = {(int(r.layer), int(r.head), r.arm): (-float(r.delta), int(r.rank))
             for r in ko_df.itertuples()}
    arm_order = list(dict.fromkeys(ko_df["arm"]))
    out = []
    for k in sorted(arms_of, key=lambda k: (best[k], k)):
        arms = arms_of[k]
        why = "top both" if len(arms) > 1 else KNOCKOUT_ARM_WHY.get(arms[0], f"top {arms[0]}")
        imp = ", ".join(f"{KNOCKOUT_ARM_SHORT.get(a, a)} {stats[(*k, a)][0]:.3f} "
                        f"(#{stats[(*k, a)][1]})" for a in arm_order if (*k, a) in stats)
        out.append((k, f"{why}; drop {imp}"))
    return out


def _knockout_dose(out_dir: Path, heads_why) -> pd.DataFrame | None:
    """Attention to t vs v for the selected heads, from the full trace's every-head
    dose table -- no forward passes needed."""
    p = out_dir / "mover_head_dose_all.csv"
    if not p.exists():
        print(f"[mover] {p.name} not found -- run the full mover trace first for the "
              f"attention-vs-v figure")
        return None
    why_of = dict(heads_why)
    d = pd.read_csv(p)
    keys = list(zip(d["layer"].astype(int), d["head"].astype(int)))
    d = d[[k in why_of for k in keys]].copy()
    d["selected_as"] = [why_of[(int(L_), int(h_))] for L_, h_ in zip(d["layer"], d["head"])]
    return d


def trace_knockout_heads(run, spec, args, out_dir: Path):
    """Attention-vs-v and attention-profile figures for the head_knockout top heads."""
    ko_csv = knockout_dir_for(run, spec, args) / "head_knockout_individual.csv"
    if not ko_csv.exists():
        raise SystemExit(f"{ko_csv} not found -- run score_head_knockout.py first")
    sel = select_knockout_heads(pd.read_csv(ko_csv), args.top_heads)
    heads, why_of = [k for k, _ in sel], dict(sel)
    print("[mover] knockout-selected heads: " +
          ", ".join(_head_label(M, h, w) for (M, h), w in sel))

    dose = _knockout_dose(out_dir, sel)
    if dose is not None:
        dose.to_csv(out_dir / f"mover_attention_vs_v{KNOCKOUT_SUFFIX}.csv", index=False)
        _plot_attention_vs_v(run, dose, out_dir, suffix=KNOCKOUT_SUFFIX)

    cfg = run.base.config
    prev_impl = getattr(cfg, "_attn_implementation", None)
    cfg._attn_implementation = "eager"      # output_attentions needs real probabilities
    try:
        _, loader, regimes, centres, inject_dtype, _ = prepare_inputs(run, args)
        prof_rows, addr_rows = measure_attention_profile(
            run, heads, why_of, loader, regimes, centres, inject_dtype,
            layer_output_is_tuple(run.model), run.model.device)
    finally:
        if prev_impl is not None:
            cfg._attn_implementation = prev_impl
    pd.DataFrame(addr_rows).to_csv(
        out_dir / f"mover_attention_base_vs_tuned{KNOCKOUT_SUFFIX}.csv", index=False)
    prof_df = pd.DataFrame(prof_rows)
    prof_df.to_csv(out_dir / f"mover_attention_profile{KNOCKOUT_SUFFIX}.csv", index=False)
    _plot_attention_profile(run, prof_df, out_dir, suffix=KNOCKOUT_SUFFIX,
                            heads_desc="the top knockout heads")


def replot_knockout_heads(spec, out_dir: Path, args) -> bool:
    prof = out_dir / f"mover_attention_profile{KNOCKOUT_SUFFIX}.csv"
    if not prof.exists():
        return False
    dose = out_dir / f"mover_attention_vs_v{KNOCKOUT_SUFFIX}.csv"
    if dose.exists():
        _plot_attention_vs_v(spec, pd.read_csv(dose), out_dir, suffix=KNOCKOUT_SUFFIX)
    _plot_attention_profile(spec, pd.read_csv(prof), out_dir, suffix=KNOCKOUT_SUFFIX,
                            heads_desc="the top knockout heads")
    return True


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    runs = runs_from_args(args)
    if not runs:
        raise SystemExit("No runs with checkpoints matched.")
    if args.heads_from_knockout:
        run_per_locale(args, runs, "mover_trace", f"mover_attention_profile{KNOCKOUT_SUFFIX}.csv",
                       analyze=lambda run, spec, out_dir: trace_knockout_heads(
                           run, spec, args, out_dir),
                       replot=replot_knockout_heads, log="[mover]")
        return
    run_per_locale(args, runs, "mover_trace", "mover_heads.csv",
                   analyze=lambda run, spec, out_dir: trace_movers(run, args, out_dir),
                   replot=replot_mover, log="[mover]")


if __name__ == "__main__":
    main()
