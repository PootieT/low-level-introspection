#!/usr/bin/env python3
"""
Trace the introspection circuit of an interventionally-trained regressor.

Two analyses (both leverage that the source is a known scalar injected at
(inject_layer MLP neuron, token_position) and the sink is a known linear readout
`w` on the last-token final residual):

  1. DIRECT ATTRIBUTION TO THE READOUT `w`.
     Decompose the head prediction into an exact additive sum of per-component
     residual writes (embed + per-layer attn + per-layer mlp) at the last token
     (final RMSNorm is frozen so the decomposition is exact). Then dose-response:
     regress each component's contribution on the injected value `v` -> the
     components whose contribution moves with `v` carry the signal. Attention is
     refined per-head to localize the cross-position "mover".

  2. DECODABILITY MAP over (layer, probe position).
     Fit a linear probe resid -> v at every layer x probe position; the map shows
     where the injected value is present and where it arrives at the readout.

     THE PROBE POSITIONS ARE THE CONTROL. "The value is decodable at the
     introspected token" is not evidence of anything on its own -- a probe with 2048
     free parameters and a few hundred rows can find structure in noise, and every
     position downstream of the injection is contaminated by construction. So the
     map is read as a CONTRAST between positions with different causal status
     (see PROBE_SPECS):
       t-1        the causal null. Strictly BEFORE the injection, so it cannot carry
                  v under causal attention. Whatever decodability appears here is
                  the probe's own floor, and every other number is only interesting
                  above it.
       t          the origin.
       t+1..t+3   downstream of the origin. NOT a null: these say whether the
                  private channel is broadcast to every later token (high) or only
                  pulled at the readout (low) -- which is exactly the question the
                  position-generalization result raises.
       last-3..last  the readout end. `last` is where the regression head reads.

Everything is config-driven (model / dataset / signal / checkpoint all come from
the experiment config) so future runs only swap the --config (+ optional CLI
overrides), and --group runs the whole sweep, writing per-run results under each
run directory for later aggregation.

Example
-------
python trace_introspection_circuit.py \
    --config experiment_configs/.../intervention_train_reg_N10.24_3K_fixedPaddingBug.json \
    --n-examples 200 --grid-points 10 --start-layer 10

python trace_introspection_circuit.py \
    --group mlp_joint_grid_layer_token --run-glob 'joint_*kl0.1*' --skip-existing
"""
from __future__ import annotations

import os
import pdb
import glob
import json
import argparse
import operator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.introspector.config import ExperimentConfig
from src.introspector.analysis_loading import (
    load_run_for_analysis, absolute_positions, layer_output_is_tuple)
from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.run_discovery import add_discovery_args, runs_from_args, resolve_run_action

from nnsight.modeling.language import LanguageModel


# ======================================================================
# Probe positions for the decodability map (analysis 2)
# ======================================================================
# (label, anchor, offset). anchor "origin" is the introspected token t; anchor "last"
# is the final token of the sequence, where the regression head reads. See the module
# docstring for what each one is FOR -- t-1 is the null every other number is read
# against, t+1..t+3 separate a broadcast channel from one that is only pulled at the
# readout, and the last-k tail shows where the value arrives.
PROBE_SPECS = [
    ("t-1", "origin", -1),
    ("t", "origin", 0),
    ("t+1", "origin", 1),
    ("t+2", "origin", 2),
    ("t+3", "origin", 3),
    ("last-3", "last", -3),
    ("last-2", "last", -2),
    ("last-1", "last", -1),
    ("last", "last", 0),
]


def probe_positions(abs_pos, attention_mask):
    """
    Absolute index of every probe position, per row, plus a validity mask.

    abs_pos: [B] the introspected token, already left-pad corrected.
    Returns (pos [B, P] long, valid [B, P] bool).

    A position is valid when it lies inside the real (non-padding) sequence. With
    LEFT padding the sequence occupies [num_left_pad, seq_len), so `t-1` on a row whose
    introspected token is the very first one falls into padding and is dropped for that
    row rather than silently probing a pad embedding.
    """
    bs, seq_len = attention_mask.shape
    num_left_pad = (attention_mask == 0).sum(dim=1).to(abs_pos.device)
    last = torch.full_like(abs_pos, seq_len - 1)
    cols = []
    for _, anchor, off in PROBE_SPECS:
        base = abs_pos if anchor == "origin" else last
        cols.append(base + off)
    pos = torch.stack(cols, dim=1)
    valid = (pos >= num_left_pad.unsqueeze(1)) & (pos <= (seq_len - 1))
    # clamp so the gather is always in range; invalid entries are masked out after
    return pos.clamp(min=0, max=seq_len - 1), valid


# ======================================================================
# Args
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Checkpoint dir override (default: latest under output_dir/experiment_name).")
    # example selection
    p.add_argument("--n-examples", type=int, default=200,
                   help="Number of endogenous-success examples to trace.")
    p.add_argument("--probe-examples", type=int, default=150,
                   help="Examples kept for the decodability captures (analysis 2). The "
                        "residual at every (layer, probe position) is held in memory for "
                        "every example x grid point, so this is the memory knob; the "
                        "attribution (analysis 1) still uses all --n-examples.")
    p.add_argument("--success-threshold", type=float, default=0.02,
                   help="Keep examples with |endogenous_pred - true_activation| <= this.")
    p.add_argument("--max-scan", type=int, default=None,
                   help="Max test examples to scan for successes (default: all).")
    # dose-response grid
    p.add_argument("--grid-points", type=int, default=10)
    p.add_argument("--grid-std-mult", type=float, default=3.0,
                   help="Grid spans mean +/- (this * std) of the neuron's endogenous distribution.")
    # scope / infra
    p.add_argument("--start-layer", type=int, default=None,
                   help="First layer for per-head attention decomposition (default: the injected layer).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Single-run override. A --group sweep always writes into each "
                        "run's own directory (<run>/circuit_trace/) so the aggregator "
                        "can find results per run.")
    add_discovery_args(p)
    return p.parse_args()


# ======================================================================
# Loading
# ======================================================================
def load_model_and_data(config: ExperimentConfig, checkpoint: Path):
    """
    (model, tokenizer, test_dataset, test_raw, collator, signal_ranges).

    A thin wrapper over the shared loader, kept because ablate_lora_modules.py imports
    it. The loading itself moved to src.introspector.analysis_loading, which fixed two
    things this copy got wrong for every sweep run: the cache path (shared caches are
    keyed by dataset_name, not experiment_name, and the run's neuron is selected with
    signal_filter) and the collator (a gate-off run analyzed with the gated collator
    has its adapter disabled over the input text -- i.e. at the origin token).
    """
    run = load_run_for_analysis(config, checkpoint=checkpoint)
    return (run.model, run.tokenizer, run.dataset, run.raw, run.collator, run.signal_ranges)


# ======================================================================
# Endogenous success filtering
# ======================================================================
@torch.no_grad()
def endogenous_predictions(model, dataset, collator, batch_size):
    """Regression prediction WITHOUT intervention, and the true activation."""
    print("forward pass without intervention to collect ground truth activations")
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    preds, acts = [], []
    head = model.regression_head
    for batch in dl:
        batch = batch.to(model.device)
        activation = batch.pop("activation")
        # readout position = last token (left-padded). Compute head(last hidden) directly.
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    lora_token_mask=batch.get("lora_token_mask"),
                    do_intervention=False, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]
        pred = head(h.to(device=head.weight.device, dtype=head.weight.dtype)).squeeze(-1)
        preds.append(pred.float().cpu())
        acts.append(activation.float().cpu())
    return torch.cat(preds), torch.cat(acts)


def select_success_indices(preds, acts, threshold, n_examples, max_scan):
    err = (preds - acts).abs()
    scan = len(err) if max_scan is None else min(max_scan, len(err))
    print(f"[success] error stats over {scan} scanned: "
          f"median={err[:scan].median():.4f} mean={err[:scan].mean():.4f} "
          f"min={err[:scan].min():.4f} max={err[:scan].max():.4f}")
    ok = (err[:scan] <= threshold).nonzero(as_tuple=True)[0]
    print(f"[success] {len(ok)} / {scan} within threshold {threshold}; taking first {n_examples}")
    if len(ok) < n_examples:
        print(f"[success] WARNING: only {len(ok)} successes; using all of them.")
    return ok[:n_examples].tolist()


# ======================================================================
# Pure analysis helpers (unit-tested)
# ======================================================================
def readout_scale(final_resid, w, gamma, eps):
    """
    Frozen-RMSNorm readout covector `a` such that a contribution of any residual
    write delta to the prediction is (a * delta).sum(-1), and the per-component
    contributions sum exactly to (prediction - bias).

    final_resid: [N, H] pre-final-norm residual at the readout token
    w, gamma: [H]   eps: scalar
    returns a: [N, H]
    """
    rms = final_resid.pow(2).mean(-1, keepdim=True).add(eps).sqrt()   # [N,1]
    return (w * gamma).unsqueeze(0) / rms                              # [N,H]


def contribution(a, write):
    """Scalar contribution of a residual write to the readout. a,write: [N,H] -> [N]."""
    return (a * write).sum(-1)


def per_head_contributions(z_cat, W_O, n_heads, head_dim, a):
    """
    z_cat: [N, n_heads*head_dim] input to o_proj (concatenated head outputs)
    W_O:   [H, n_heads*head_dim] o_proj weight (out=H, in=heads*hd; no bias in Llama)
    a:     [N, H] frozen-LN readout covector
    Returns [N, n_heads]: each head's contribution to the readout.
    Sum over heads == contribution(a, attn_output).
    """
    N = z_cat.shape[0]
    z = z_cat.view(N, n_heads, head_dim)                              # [N,h,hd]
    contribs = torch.empty(N, n_heads, dtype=a.dtype)
    for h in range(n_heads):
        Woh = W_O[:, h * head_dim:(h + 1) * head_dim]                 # [H, hd]
        write_h = z[:, h, :] @ Woh.T                                  # [N, H]
        contribs[:, h] = (a * write_h).sum(-1)
    return contribs


def linfit_slope_r2(x, y):
    """Least-squares slope and R^2 of y ~ x (1D numpy arrays)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() < 1e-12:
        return 0.0, 0.0
    b1 = np.cov(x, y, bias=True)[0, 1] / x.var()
    b0 = y.mean() - b1 * x.mean()
    yhat = b0 + b1 * x
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(b1), float(r2)


def decodability(X, y, train_frac=0.7):
    """Ridge probe X->y, split by row; return test Pearson corr. X:[n,H], y:[n]."""
    from sklearn.linear_model import Ridge
    from scipy.stats import pearsonr
    n = len(y)
    ntr = int(n * train_frac)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    if len(yte) < 3 or np.std(yte) < 1e-12:
        return float("nan")
    reg = Ridge(alpha=1.0).fit(Xtr, ytr)
    pred = reg.predict(Xte)
    if np.std(pred) < 1e-12:
        return float("nan")
    return float(pearsonr(pred, yte)[0])


# ======================================================================
# nnsight capture
# ======================================================================
# Injection-path resolvers dispatched by signal type, so tracing a different
# signal type in the future just registers its resolver here. The residual-stream
# decomposition captured below (attn/mlp/embed writes) is signal-type agnostic.
SIGNAL_PATH_RESOLVERS = {
    "mlp": resolve_mlp_signal_path,
}


def resolve_signal_injection_path(lm, signal_type: str, layer: int) -> str:
    """Path (from the nnsight LanguageModel root) to the signal source tensor we
    overwrite, dispatched by signal type. Raises for unregistered types."""
    if signal_type not in SIGNAL_PATH_RESOLVERS:
        raise NotImplementedError(
            f"No injection-path resolver registered for signal type '{signal_type}'. "
            f"Add one to SIGNAL_PATH_RESOLVERS."
        )
    return SIGNAL_PATH_RESOLVERS[signal_type](lm, layer, path_to_output=True)


def nnsight_module_handles(lm, inject_layer):
    """Resolve nnsight envoy handles from the proven resolve_mlp_signal_path prefix."""
    actfn_path = resolve_mlp_signal_path(lm, inject_layer, path_to_output=False)  # ...layers.L.mlp.act_fn
    model_base = actfn_path.split(".layers.")[0]                                  # e.g. "model.model"
    layers = operator.attrgetter(model_base + ".layers")(lm)
    embed = operator.attrgetter(model_base + ".embed_tokens")(lm)
    norm = operator.attrgetter(model_base + ".norm")(lm)
    return layers, embed, norm


@torch.no_grad()
def trace_batch(lm, layers, embed, norm, inputs, abs_pos, inject_layer, inject_path,
                neuron, v, n_layers, start_layer, probe_pos=None, capture_probes=True,
                layer_out_tuple=False):
    """
    One injected forward. Modules are accessed in COMPUTATIONAL-PATH ORDER
    (embed -> for each layer: attention block -> [inject at signal layer] -> mlp
    block -> residual -> final norm), and the injection overwrites the signal
    source in-place via `inject_path` (resolved by signal type, not a hard-coded
    module path) between the attention write and the MLP that consumes the neuron.

    Returns last-token component writes (embed, per-layer attn/mlp), per-layer
    residuals at every PROBE position (`probe_pos`: [B, P] absolute indices, from
    probe_positions()), o_proj inputs (>= start_layer), the pre-norm final residual,
    and the post-norm hidden.

    Probe residuals come back in fp16: they exist to be fed to a ridge probe over
    2048 dimensions, where fp16 is far below the noise floor, and at P=9 positions x
    16 layers x every (example, grid point) the fp32 version is the memory budget of
    the whole script. The attribution captures stay fp32 -- those are differenced and
    summed against an exactness check.

    `inputs` must contain ONLY input_ids/attention_mask/lora_token_mask so the
    model's own intervention path does not also fire.

    `layer_out_tuple` says whether a decoder layer returns `(hidden, ...)` or the bare
    tensor -- it differs between transformers 4.x and 5.x, so it is DETECTED per run
    (see analysis_loading.layer_output_is_tuple) rather than assumed here.
    """
    bs = inputs["input_ids"].shape[0]
    bidx = torch.arange(bs, device=abs_pos.device)
    attn, mlp, resid_last, oproj, probes = {}, {}, {}, {}, {}
    probe_bidx = bidx.unsqueeze(1) if probe_pos is not None else None
    with lm.trace(inputs):
        emb = embed.output[bidx, -1, :].save()
        for L in range(n_layers):
            layer = layers[L]
            # --- attention block ---
            if L >= start_layer:
                oproj[L] = layer.self_attn.o_proj.input[bidx, -1, :].save()
            attn[L] = layer.self_attn.output[0][bidx, -1, :].save()
            # --- inject the signal source in-place (resolver-derived path),
            #     after the attention write and before the MLP consumes it ---
            if L == inject_layer:
                operator.attrgetter(inject_path)(lm)[bidx, abs_pos, neuron] = v
            # --- mlp block ---
            mlp[L] = layer.mlp.output[bidx, -1, :].save()
            # --- residual after the layer ---
            resid = layer.output[0] if layer_out_tuple else layer.output
            resid_last[L] = resid[bidx, -1, :].save()
            if capture_probes and probe_pos is not None:
                # [B, P, H]: every probe position in one gather
                probes[L] = resid[probe_bidx, probe_pos, :].save()
        normed = norm.output[bidx, -1, :].save()

    val = lambda p: p.value if hasattr(p, "value") else p
    return {
        "embed": val(emb).float().cpu(),
        "attn": {L: val(t).float().cpu() for L, t in attn.items()},
        "mlp": {L: val(t).float().cpu() for L, t in mlp.items()},
        "final_resid": val(resid_last[n_layers - 1]).float().cpu(),
        "normed": val(normed).float().cpu(),
        "resid_probe": {L: val(t).half().cpu() for L, t in probes.items()},
        "oproj": {L: val(t).float().cpu() for L, t in oproj.items()},
    }


# ======================================================================
# Main
# ======================================================================
def trace_one_run(run, args, out_dir: Path):
    """The whole trace for ONE loaded run, into `out_dir`."""
    config, checkpoint = run.config, run.checkpoint
    model, tokenizer = run.model, run.tokenizer
    test_dataset, collator, signal_ranges = run.dataset, run.collator, run.signal_ranges
    inject_layer, neuron, signal_type = run.layer, run.index, run.signal_type
    start_layer = args.start_layer if args.start_layer is not None else inject_layer
    print(f"[trace] signal=N{inject_layer}.{neuron} inject_layer={inject_layer} "
          f"neuron={neuron} start_layer={start_layer} checkpoint={checkpoint}")

    device = model.device
    hf = model.get_base_model()                       # underlying LlamaForCausalLM
    n_layers = hf.config.num_hidden_layers
    n_heads = hf.config.num_attention_heads
    head_dim = getattr(hf.config, "head_dim", hf.config.hidden_size // n_heads)
    eps = hf.config.rms_norm_eps
    gamma = hf.model.norm.weight.detach().float().cpu()
    W_O = {L: hf.model.layers[L].self_attn.o_proj.weight.detach().float().cpu()
           for L in range(start_layer, n_layers)}
    w = model.regression_head.weight.detach().flatten().float().cpu()
    bias = float(model.regression_head.bias.detach().flatten()[0].cpu())

    # ---- select endogenous-success examples ----
    preds, acts = endogenous_predictions(model, test_dataset, collator, args.batch_size)
    idx = select_success_indices(preds, acts, args.success_threshold, args.n_examples, args.max_scan)
    if not idx:
        raise SystemExit("No success examples selected; loosen --success-threshold.")
    sub = test_dataset.select(idx)
    print(f"{len(idx)} endogenous examples selected ...")

    # ---- dose-response grid over the neuron's endogenous gaussian ----
    stats = signal_ranges[(inject_layer, neuron)]
    lo = stats["mean"] - args.grid_std_mult * stats["std"]
    hi = stats["mean"] + args.grid_std_mult * stats["std"]
    grid = torch.linspace(lo, hi, args.grid_points)
    print(f"[trace] v-grid: {args.grid_points} points in [{lo:.4f}, {hi:.4f}] "
          f"(neuron mean={stats['mean']:.4f} std={stats['std']:.4f})")

    # transformers 4.x decoder layers return (hidden, ...), 5.x returns the tensor;
    # every residual capture below has to index the right one
    layer_out_tuple = layer_output_is_tuple(model)
    print(f"[trace] decoder layer output is {'a tuple' if layer_out_tuple else 'a bare tensor'}")

    lm = LanguageModel(model, tokenizer=tokenizer)
    layers, embed, norm = nnsight_module_handles(lm, inject_layer)
    inject_path = resolve_signal_injection_path(lm, signal_type, inject_layer)

    # accumulators indexed [example, grid_point]
    comp_names = ["embed"] + [f"attn_{L}" for L in range(n_layers)] + [f"mlp_{L}" for L in range(n_layers)]
    contribs = {c: [] for c in comp_names}                       # each -> list over grid of [Nexp]
    head_contribs = {L: [] for L in range(start_layer, n_layers)}  # -> list over grid of [Nexp, n_heads]
    pred_check = []                                              # [grid][Nexp]
    n_probe_rows = min(args.probe_examples, len(sub))
    resid_probe = {L: [] for L in range(n_layers)}               # decodability: [Nprobe, P, H]
    probe_valid, probe_region = None, None                       # set on the first grid pass

    from torch.utils.data import DataLoader
    for gi, v in enumerate(grid):
        dl = DataLoader(sub, batch_size=args.batch_size, collate_fn=collator, shuffle=False)
        g_contribs = {c: [] for c in comp_names}
        g_head = {L: [] for L in range(start_layer, n_layers)}
        g_pred, g_probe = [], {L: [] for L in range(n_layers)}
        g_valid, g_region, row0 = [], [], 0
        for batch in dl:
            batch = batch.to(device)
            attn_mask = batch["attention_mask"]
            token_pos = batch["token_position"]
            abs_pos = absolute_positions(token_pos, attn_mask)        # left-pad offset fix
            ppos, pvalid = probe_positions(abs_pos, attn_mask)        # [b,P] each
            # probe captures are the memory budget, so only the first --probe-examples
            # rows of the (sequential, unshuffled) split contribute to analysis 2
            keep = max(0, min(ppos.shape[0], n_probe_rows - row0))
            trace_inputs = {"input_ids": batch["input_ids"],
                            "attention_mask": attn_mask,
                            "lora_token_mask": batch.get("lora_token_mask")}
            cap = trace_batch(lm, layers, embed, norm, trace_inputs, abs_pos,
                              inject_layer, inject_path, neuron, float(v), n_layers, start_layer,
                              probe_pos=ppos, capture_probes=keep > 0,
                              layer_out_tuple=layer_out_tuple)

            a = readout_scale(cap["final_resid"], w, gamma, eps)      # [b,H]
            g_contribs["embed"].append(contribution(a, cap["embed"]))
            for L in range(n_layers):
                g_contribs[f"attn_{L}"].append(contribution(a, cap["attn"][L]))
                g_contribs[f"mlp_{L}"].append(contribution(a, cap["mlp"][L]))
            for L in range(start_layer, n_layers):
                g_head[L].append(per_head_contributions(cap["oproj"][L], W_O[L], n_heads, head_dim, a))
            # head prediction (from post-norm hidden) for the decomposition sanity check
            g_pred.append((cap["normed"] * w).sum(-1) + bias)
            if keep > 0:
                for L in range(n_layers):
                    g_probe[L].append(cap["resid_probe"][L][:keep])
                if gi == 0:
                    g_valid.append(pvalid[:keep].cpu())
                    # region mask: 0 = input text, 1 = introspection prompt. The gated
                    # collator calls it lora_token_mask, the ungated one keeps the same
                    # tensor as anchor_region_mask (it is not a model input there).
                    region = batch.get("lora_token_mask", batch.get("anchor_region_mask"))
                    if region is not None:
                        g_region.append(torch.gather(region, 1, ppos)[:keep].cpu())
            row0 += ppos.shape[0]

        for c in comp_names:
            contribs[c].append(torch.cat(g_contribs[c]))
        for L in range(start_layer, n_layers):
            head_contribs[L].append(torch.cat(g_head[L]))
        pred_check.append(torch.cat(g_pred))
        for L in range(n_layers):
            if g_probe[L]:
                resid_probe[L].append(torch.cat(g_probe[L]))
        if gi == 0:
            probe_valid = torch.cat(g_valid) if g_valid else None
            probe_region = torch.cat(g_region) if g_region else None
        print(f"[trace] grid {gi + 1}/{len(grid)} (v={float(v):+.4f}) done")

    # stack -> [Nexp, grid]
    Nexp = len(idx)
    contribs = {c: torch.stack(contribs[c], 1) for c in comp_names}   # [Nexp, grid]
    pred_check = torch.stack(pred_check, 1)                           # [Nexp, grid]
    v_np = grid.numpy()

    # ---- decomposition sanity: sum of contributions + bias == prediction ----
    total = sum(contribs[c] for c in comp_names) + bias
    max_err = (total - pred_check).abs().max().item()
    print(f"[sanity] max|sum(contributions)+bias - prediction| = {max_err:.3e} (should be ~0)")

    # end-to-end validation: the traced model must still causally read the neuron,
    # i.e. its prediction tracks the injected v. Low corr here => gating/injection
    # broke in the trace (the known interventional corr for this run is ~0.99).
    from scipy.stats import pearsonr
    e2e = pearsonr(pred_check.reshape(-1).numpy(), np.tile(v_np, Nexp))[0]
    print(f"[sanity] corr(traced prediction, injected v) = {e2e:.4f} "
          f"(should be high; validates gating+injection+readout through nnsight)")

    # ======================= Analysis 1: attribution =======================
    rows = []
    for c in comp_names:
        slopes, r2s = [], []
        for e in range(Nexp):
            s, r2 = linfit_slope_r2(v_np, contribs[c][e].numpy())
            slopes.append(s); r2s.append(r2)
        rows.append({"component": c, "mean_slope": float(np.mean(slopes)),
                     "mean_abs_slope": float(np.mean(np.abs(slopes))), "mean_r2": float(np.mean(r2s))})
    attr_df = pd.DataFrame(rows)
    attr_df.to_csv(out_dir / "attribution_components.csv", index=False)

    # per-head slopes -> [layer, head]
    head_rows = []
    head_slope_mat = np.zeros((n_layers - start_layer, n_heads))
    for li, L in enumerate(range(start_layer, n_layers)):
        hc = torch.stack(head_contribs[L], 1)   # [Nexp, grid, n_heads]
        for h in range(n_heads):
            slopes = [linfit_slope_r2(v_np, hc[e, :, h].numpy())[0] for e in range(Nexp)]
            m = float(np.mean(slopes))
            head_slope_mat[li, h] = m
            head_rows.append({"layer": L, "head": h, "mean_slope": m})
    pd.DataFrame(head_rows).to_csv(out_dir / "attribution_heads.csv", index=False)

    # ======================= Analysis 2: decodability =======================
    # One probe per (layer, probe position). The row order is example-major
    # ([Nprobe, grid] flattened row-major), which is what makes decodability()'s
    # 70/30 split a split BY EXAMPLE -- the same example at a different grid point
    # must never appear on both sides, or the probe scores its own memory.
    probe_labels = [lbl for lbl, _, _ in PROBE_SPECS]
    n_probe = len(probe_labels)
    dec_rows = []
    dec_mat = np.full((n_layers, n_probe), np.nan)
    for L in range(n_layers):
        if not resid_probe[L]:
            continue
        X_all = torch.stack(resid_probe[L], 1).float()          # [Nprobe, grid, P, H]
        for pi, label in enumerate(probe_labels):
            ok = probe_valid[:, pi] if probe_valid is not None else None
            rows = torch.ones(X_all.shape[0], dtype=torch.bool) if ok is None else ok
            n_rows = int(rows.sum())
            if n_rows < 20:          # too few valid examples to fit anything
                dec_rows.append({"layer": L, "probe": label, "corr": float("nan"),
                                 "n_examples": n_rows, "frac_prompt_region": float("nan")})
                continue
            X = X_all[rows, :, pi, :].reshape(-1, X_all.shape[-1]).numpy()
            y = np.tile(v_np, n_rows)
            c = decodability(X, y)
            dec_mat[L, pi] = c
            frac = (float(probe_region[rows, pi].float().mean())
                    if probe_region is not None else float("nan"))
            dec_rows.append({"layer": L, "probe": label, "corr": c,
                             "n_examples": n_rows, "frac_prompt_region": frac})
    dec_df = pd.DataFrame(dec_rows)
    dec_df.to_csv(out_dir / "decodability.csv", index=False)
    if not dec_df.empty:
        null = dec_df[dec_df["probe"] == "t-1"]["corr"]
        print(f"[trace] decodability: max over (layer, probe) = {dec_df['corr'].max():.3f}; "
              f"causal null (t-1) max = {null.max() if len(null) else float('nan'):.3f} "
              f"-- read every other cell against that floor")

    # ============================ plots ============================
    _plot_attribution(attr_df, n_layers, start_layer, inject_layer, out_dir)
    _plot_heads(head_slope_mat, start_layer, n_layers, n_heads, out_dir)
    _plot_decodability(dec_mat, probe_labels, n_layers, inject_layer, out_dir)

    with open(out_dir / "trace_meta.json", "w") as f:
        json.dump({"run": config.experiment_name, "checkpoint": str(checkpoint),
                   "signal": f"N{inject_layer}.{neuron}", "signal_type": signal_type,
                   "inject_layer": inject_layer, "neuron": neuron,
                   "token_position": (config.test_token_selector.position
                                      if config.test_token_selector.type == "fixed" else None),
                   "start_layer": start_layer, "n_examples": Nexp,
                   "n_probe_examples": n_probe_rows, "probe_labels": probe_labels,
                   "grid": v_np.tolist(), "decomposition_max_err": max_err,
                   "e2e_corr_pred_vs_v": float(e2e)}, f, indent=2)
    print(f"[trace] wrote results to {out_dir}")


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    runs = runs_from_args(args)
    if not runs:
        raise SystemExit("No runs with checkpoints matched.")
    sweeping = len(runs) > 1 or args.group is not None
    for k, spec in enumerate(runs, 1):
        # a sweep writes into each run's own directory, so results stay next to the run
        # they describe and the aggregator can find them by globbing run dirs
        out_dir = ((args.output_dir or (spec.run_dir / "circuit_trace")) if not sweeping
                   else spec.run_dir / "circuit_trace")
        action = resolve_run_action(args, out_dir / "decodability.csv")
        if action == "skip":
            print(f"[trace] [{k}/{len(runs)}] {spec.name}: results exist, skipping")
            continue
        if action == "replot":
            if replot_circuit(spec, out_dir, args):
                print(f"[trace] [{k}/{len(runs)}] {spec.name}: re-plotted from cache")
                continue
            print(f"[trace] [{k}/{len(runs)}] {spec.name}: cache incomplete, recomputing")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[trace] ===== [{k}/{len(runs)}] {spec.name} ({spec.cell}) =====")
        run = load_run_for_analysis(spec.config, checkpoint=args.checkpoint or spec.checkpoint())
        try:
            trace_one_run(run, args, out_dir)
        finally:
            # a sweep loads one model per run; without this the previous run's weights
            # stay resident and the second checkpoint OOMs on a single GPU
            del run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _plot_attribution(attr_df, n_layers, start_layer, inject_layer, out_dir):
    order = ["embed"] + [f"attn_{L}" for L in range(n_layers)] + [f"mlp_{L}" for L in range(n_layers)]
    attr_df = attr_df.set_index("component").loc[order].reset_index()
    plt.figure(figsize=(max(8, 0.18 * len(order)), 4))
    plt.bar(attr_df["component"], attr_df["mean_slope"])
    plt.axhline(0, color="k", lw=0.5)
    plt.xticks(rotation=90, fontsize=6)
    plt.ylabel("dose-response slope (d contribution / d v)")
    plt.title("Direct attribution to readout w (per component)")
    plt.tight_layout()
    plt.savefig(out_dir / "attribution_components.png", dpi=150)
    plt.close()


def _plot_heads(mat, start_layer, n_layers, n_heads, out_dir):
    plt.figure(figsize=(max(6, 0.3 * n_heads), max(3, 0.35 * mat.shape[0])))
    lim = np.abs(mat).max() or 1.0
    sns.heatmap(mat, cmap="coolwarm", center=0, vmin=-lim, vmax=lim,
                yticklabels=list(range(start_layer, n_layers)),
                xticklabels=list(range(n_heads)),
                cbar_kws={"label": "dose-response slope"})
    plt.xlabel("head"); plt.ylabel("layer")
    plt.title("Per-head attention attribution to readout w")
    plt.tight_layout()
    plt.savefig(out_dir / "attribution_heads.png", dpi=150)
    plt.close()


def _plot_decodability(mat, probe_labels, n_layers, inject_layer, out_dir):
    plt.figure(figsize=(1.0 * len(probe_labels) + 3, max(4, 0.35 * n_layers)))
    ax = sns.heatmap(mat, cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f",
                     annot_kws={"size": 7},
                     yticklabels=list(range(n_layers)), xticklabels=probe_labels,
                     cbar_kws={"label": "probe corr (v decodability)"})
    plt.axhline(inject_layer + 0.5, color="w", ls="--", lw=1)
    # the causal null: nothing left of this line can carry v, so its column is the
    # floor the rest of the map is read against
    if "t-1" in probe_labels:
        ax.axvline(probe_labels.index("t-1") + 1, color="w", ls=":", lw=1.5)
    plt.ylabel("layer"); plt.xlabel("probe position")
    plt.title("Injected-value decodability by (layer, probe position)\n"
              "t-1 = causal null | t = origin | last = readout", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "decodability.png", dpi=150)
    plt.close()


def replot_circuit(spec, out_dir: Path, args) -> bool:
    """Redraw the three circuit-trace figures from cached CSVs + trace_meta.json; False
    if any is missing. head/decodability matrices are pivoted back out of the long CSVs;
    layer/head counts and scalars come from the CSVs and meta."""
    comp_csv = out_dir / "attribution_components.csv"
    heads_csv = out_dir / "attribution_heads.csv"
    dec_csv = out_dir / "decodability.csv"
    meta_p = out_dir / "trace_meta.json"
    if not (comp_csv.exists() and heads_csv.exists() and dec_csv.exists() and meta_p.exists()):
        return False
    meta = json.loads(meta_p.read_text())
    attr_df = pd.read_csv(comp_csv)
    head_df = pd.read_csv(heads_csv)
    dec_df = pd.read_csv(dec_csv)
    start_layer = int(meta["start_layer"])
    inject_layer = int(meta["inject_layer"])
    n_layers = int(max(head_df["layer"].max(), dec_df["layer"].max())) + 1
    n_heads = int(head_df["head"].max()) + 1
    probe_labels = meta.get("probe_labels") or list(dict.fromkeys(dec_df["probe"]))
    head_mat = (head_df.pivot(index="layer", columns="head", values="mean_slope")
                .reindex(index=range(start_layer, n_layers), columns=range(n_heads)).to_numpy())
    dec_mat = (dec_df.pivot(index="layer", columns="probe", values="corr")
               .reindex(index=range(n_layers), columns=probe_labels).to_numpy())
    _plot_attribution(attr_df, n_layers, start_layer, inject_layer, out_dir)
    _plot_heads(head_mat, start_layer, n_layers, n_heads, out_dir)
    _plot_decodability(dec_mat, probe_labels, n_layers, inject_layer, out_dir)
    return True


if __name__ == "__main__":
    main()
