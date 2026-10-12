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
import re
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
    load_run_for_analysis, absolute_positions, layer_output_is_tuple, run_per_locale)
from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.run_discovery import (
    add_discovery_args, runs_from_args, resolve_run_action, locale_out_dir, locale_tag,
    pretty_run_name)
from src.introspector.regimes import (
    DEFAULT_REGIMES, DEFAULT_LOCAL_DELTA_SIGMA, parse_regimes, zero_offset_index,
    identity_check, power_warning)

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
                   help="Global grids span mean +/- (this * std * scale) of the neuron's "
                        "endogenous distribution.")
    p.add_argument("--regimes", type=str, default=DEFAULT_REGIMES,
                   help="Injection regimes to sweep, comma separated (see "
                        "src/introspector/regimes.py). 'xN' is a global grid at N times "
                        "the natural std, 'xtrain' the run's own training scale, and "
                        "'local' perturbs each example around ITS OWN current value -- "
                        "the tangent at the operating point rather than a chord across "
                        "the whole natural range. They disagree when the response is "
                        "nonlinear, which is the point of running both.")
    p.add_argument("--local-delta-sigma", type=float, default=DEFAULT_LOCAL_DELTA_SIGMA,
                   help="Half-width of the 'local' sweep, in units of the neuron's std. "
                        "Override per regime with e.g. 'local@0.5'.")
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
    if run.is_multi_locale:
        # this tuple has no way to say which signal it is about, and its dataset holds
        # every signal's rows -- use load_run_for_analysis(...).for_locale(...) instead
        raise ValueError(
            f"{run.config.experiment_name} was trained on {len(run.locales)} signals; "
            f"load_model_and_data returns one mixed test split. Use "
            f"load_run_for_analysis(...).for_locale(locale) per signal.")
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


def effective_weight(module) -> torch.Tensor:
    """W0 + B@A*scaling for a peft LoRA linear, W0 otherwise.

    Any per-head split has to use the weights the TRAINED model runs. With LoRA on
    all-linear, `module.weight` on a peft layer returns the BASE weight only, so a
    head decomposition built from it silently drops the adapter's own o_proj edit
    (and no longer sums to the attention block's measured write).
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

    `v` is either a scalar (global regimes: every example gets the same value) or a
    [B] tensor (the `local` regime: every example gets its own value). Advanced-index
    assignment handles both identically.
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


@torch.no_grad()
def capture_live_centres(model, dataset, collator, batch_size, inject_layer, neuron):
    """
    The neuron's LIVE activation at the introspected token, per example, in dataset
    order -- the centre the `local` regime perturbs around.

    Taken from the model as it is, NOT from the cached `activation` column. Those are
    the same thing only when the adapter is gated off the input text; with the gate
    off the adapter changes the very neuron it reports on (the drift diagnostic
    measures corr ~0.87 on the pre-flight runs), and centring on the cached value
    would perturb around a point the model is not at -- which the delta=0 identity
    check would then fail.
    """
    from torch.utils.data import DataLoader
    path = resolve_mlp_signal_path(model, inject_layer, path_to_output=False)
    module = operator.attrgetter(path)(model)
    grabbed = {}

    def hook(_m, _inputs, outputs):
        grabbed["act"] = outputs.detach()

    out = []
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    for batch in dl:
        batch = batch.to(model.device)
        attn_mask = batch["attention_mask"]
        abs_pos = absolute_positions(batch["token_position"], attn_mask)
        handle = module.register_forward_hook(hook)
        try:
            model(input_ids=batch["input_ids"], attention_mask=attn_mask,
                  lora_token_mask=batch.get("lora_token_mask"), do_intervention=False)
        finally:
            handle.remove()
        rows = torch.arange(abs_pos.shape[0], device=abs_pos.device)
        out.append(grabbed["act"][rows, abs_pos, neuron].float().cpu())
    return torch.cat(out)


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
    # base + LoRA edit: the head split must use the o_proj the trained model runs
    W_O = {L: effective_weight(hf.model.layers[L].self_attn.o_proj)
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

    # ---- injection regimes ----
    stats = signal_ranges[(inject_layer, neuron)]
    tc = run.config.training_config
    train_scale = (tc.intervention_extreme_scale
                   if tc.intervention_signal_distribution == "extreme" else 1.0)
    regimes = parse_regimes(args.regimes, stats, args.grid_points, args.grid_std_mult,
                            delta_sigma=args.local_delta_sigma, train_scale=train_scale)
    for rg in regimes:
        print(f"[trace] regime {rg.describe(stats)}")

    # `local` perturbs each example around its own live activation, so those have to be
    # measured first -- and in the SAME order the trace loop will see them (the loader
    # is sequential and unshuffled, so a plain concatenation lines up row for row).
    centres = None
    if any(rg.is_local for rg in regimes):
        centres = capture_live_centres(model, sub, collator, args.batch_size,
                                       inject_layer, neuron)
        print(f"[trace] live centres for {len(centres)} examples: "
              f"mean={centres.mean():.4f} std={centres.std():.4f} "
              f"(cached-column mean={stats['mean']:.4f} std={stats['std']:.4f})")

    # transformers 4.x decoder layers return (hidden, ...), 5.x returns the tensor;
    # every residual capture below has to index the right one
    layer_out_tuple = layer_output_is_tuple(model)
    print(f"[trace] decoder layer output is {'a tuple' if layer_out_tuple else 'a bare tensor'}")

    lm = LanguageModel(model, tokenizer=tokenizer)
    layers, embed, norm = nnsight_module_handles(lm, inject_layer)
    inject_path = resolve_signal_injection_path(lm, signal_type, inject_layer)

    comp_names = ["embed"] + [f"attn_{L}" for L in range(n_layers)] + [f"mlp_{L}" for L in range(n_layers)]
    n_probe_rows = min(args.probe_examples, len(sub))
    # the injected tensor must match the model's dtype: a python float is coerced by
    # index_put, a fp32 tensor written into a bf16 activation is not. Round-tripping a
    # captured bf16 centre through fp32 and back is lossless, so delta=0 still writes
    # back exactly the value that was there.
    inject_dtype = next(model.parameters()).dtype
    from torch.utils.data import DataLoader

    def measure(regime):
      """One full sweep of `regime`: per-component/per-head contributions, the
      prediction, and the probe residuals, accumulated over its grid."""
      contribs = {c: [] for c in comp_names}                       # each -> list over grid of [Nexp]
      head_contribs = {L: [] for L in range(start_layer, n_layers)}  # -> over grid of [Nexp, n_heads]
      pred_check = []                                              # [grid][Nexp]
      resid_probe = {L: [] for L in range(n_layers)}               # decodability: [Nprobe, P, H]
      probe_valid, probe_region = None, None                       # set on the first grid pass
      for gi in range(len(regime.grid)):
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
            # scalar for a global regime; this batch's own centres + the offset for
            # `local` (rows line up because the loader is sequential and unshuffled)
            nrow = ppos.shape[0]
            v = regime.values_for(
                None if not regime.is_local else centres[row0:row0 + nrow].to(device), gi)
            if torch.is_tensor(v):
                v = v.to(dtype=inject_dtype)
            cap = trace_batch(lm, layers, embed, norm, trace_inputs, abs_pos,
                              inject_layer, inject_path, neuron, v, n_layers, start_layer,
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
        shown = float(v.mean()) if torch.is_tensor(v) else float(v)
        print(f"[trace] {regime.name}: grid {gi + 1}/{len(regime.grid)} "
              f"({'mean v' if torch.is_tensor(v) else 'v'}={shown:+.4f}) done")
      return contribs, head_contribs, pred_check, resid_probe, probe_valid, probe_region

    # ---- one measurement per regime; every output frame carries which one ----
    attr_frames, head_frames, dec_frames, meta_regimes = [], [], [], {}
    unperturbed_pred = None
    for regime in regimes:
      (contribs, head_contribs, pred_check,
       resid_probe, probe_valid, probe_region) = measure(regime)

      # stack -> [Nexp, grid]
      Nexp = len(idx)
      contribs = {c: torch.stack(contribs[c], 1) for c in comp_names}   # [Nexp, grid]
      pred_check = torch.stack(pred_check, 1)                           # [Nexp, grid]
      v_np = regime.regressor

      # ---- decomposition sanity: sum of contributions + bias == prediction ----
      total = sum(contribs[c] for c in comp_names) + bias
      max_err = (total - pred_check).abs().max().item()
      print(f"[sanity] {regime.name}: max|sum(contributions)+bias - prediction| = "
            f"{max_err:.3e} (should be ~0)")

      # per-head split sanity: heads of a layer must sum to that layer's attention write
      head_err = max(
          (torch.stack(head_contribs[L], 1).sum(-1) - contribs[f"attn_{L}"]).abs().max().item()
          for L in range(start_layer, n_layers))
      print(f"[sanity] {regime.name}: max|sum(head contributions) - attn contribution| = "
            f"{head_err:.3e} (should be ~0)")

      # end-to-end validation: the traced model must still causally read the neuron,
      # i.e. its prediction tracks the swept value.
      from scipy.stats import pearsonr
      e2e = pearsonr(pred_check.reshape(-1).numpy(), np.tile(v_np, Nexp))[0]
      print(f"[sanity] {regime.name}: corr(traced prediction, swept value) = {e2e:.4f}")

      # `local` only: re-injecting the neuron's OWN value must be a no-op, which is a
      # stronger check on the centres and the row alignment than any offset scan.
      zi = zero_offset_index(regime)
      if zi is not None:
          if unperturbed_pred is None:
              unperturbed_pred, _ = endogenous_predictions(model, sub, collator,
                                                           args.batch_size)
          identity_check(regime, pred_check[:, zi].numpy(),
                         unperturbed_pred.numpy(), label=config.experiment_name)

      # ======================= Analysis 1: attribution =======================
      rows = []
      for c in comp_names:
          slopes, r2s = [], []
          for e in range(Nexp):
              sl, r2 = linfit_slope_r2(v_np, contribs[c][e].numpy())
              slopes.append(sl); r2s.append(r2)
          rows.append({"regime": regime.name, "component": c,
                       "mean_slope": float(np.mean(slopes)),
                       "mean_abs_slope": float(np.mean(np.abs(slopes))),
                       "mean_r2": float(np.mean(r2s))})
      attr_df = pd.DataFrame(rows)
      attr_frames.append(attr_df)

      # is this regime's dose-response resolved above the numerical floor at all?
      # Checked on the component that responds most, so a flat *circuit* is not
      # mistaken for a flat *measurement*.
      power_warning(regime, float(attr_df.loc[attr_df["mean_abs_slope"].idxmax(), "mean_r2"]),
                    label=config.experiment_name)

      # per-head slopes -> [layer, head]
      head_rows = []
      head_slope_mat = np.zeros((n_layers - start_layer, n_heads))
      for li, L in enumerate(range(start_layer, n_layers)):
          hc = torch.stack(head_contribs[L], 1)   # [Nexp, grid, n_heads]
          for h in range(n_heads):
              slopes = [linfit_slope_r2(v_np, hc[e, :, h].numpy())[0] for e in range(Nexp)]
              m = float(np.mean(slopes))
              head_slope_mat[li, h] = m
              head_rows.append({"regime": regime.name, "layer": L, "head": h, "mean_slope": m})
      head_frames.append(pd.DataFrame(head_rows))

      # ======================= Analysis 2: decodability =======================
      # One probe per (layer, probe position). The row order is example-major
      # ([Nprobe, grid] flattened row-major), which is what makes decodability()'s
      # 70/30 split a split BY EXAMPLE -- the same example at a different grid point
      # must never appear on both sides, or the probe scores its own memory.
      #
      # Under `local` the target is the OFFSET, not the absolute value: the offset is
      # what was causally imposed, while the absolute value carries the example's own
      # activation and is therefore correlated with context -- a context-reading probe
      # would score on it without the signal having propagated anywhere.
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
              prows = torch.ones(X_all.shape[0], dtype=torch.bool) if ok is None else ok
              n_rows = int(prows.sum())
              if n_rows < 20:          # too few valid examples to fit anything
                  dec_rows.append({"regime": regime.name, "layer": L, "probe": label,
                                   "corr": float("nan"), "n_examples": n_rows,
                                   "frac_prompt_region": float("nan")})
                  continue
              X = X_all[prows, :, pi, :].reshape(-1, X_all.shape[-1]).numpy()
              y = np.tile(v_np, n_rows)
              c = decodability(X, y)
              dec_mat[L, pi] = c
              frac = (float(probe_region[prows, pi].float().mean())
                      if probe_region is not None else float("nan"))
              dec_rows.append({"regime": regime.name, "layer": L, "probe": label, "corr": c,
                               "n_examples": n_rows, "frac_prompt_region": frac})
      dec_df = pd.DataFrame(dec_rows)
      dec_frames.append(dec_df)
      if not dec_df.empty:
          null = dec_df[dec_df["probe"] == "t-1"]["corr"]
          print(f"[trace] {regime.name}: decodability max over (layer, probe) = "
                f"{dec_df['corr'].max():.3f}; causal null (t-1) max = "
                f"{null.max() if len(null) else float('nan'):.3f}")

      # ============================ plots (per regime) ============================
      tag = f"_{regime.name}"
      _plot_attribution(attr_df, n_layers, start_layer, inject_layer, out_dir, tag)
      _plot_heads(head_slope_mat, start_layer, n_layers, n_heads, out_dir, tag)
      _plot_decodability(dec_mat, probe_labels, n_layers, inject_layer, out_dir, tag,
                         run_name=config.experiment_name)
      meta_regimes[regime.name] = {
          "kind": regime.kind, "scale": regime.scale, "delta_sigma": regime.delta_sigma,
          "grid": [float(x) for x in regime.grid],
          "decomposition_max_err": max_err, "head_split_max_err": head_err,
          "e2e_corr_pred_vs_v": float(e2e)}

    # ---- one combined csv per analysis, regimes side by side ----
    pd.concat(attr_frames, ignore_index=True).to_csv(
        out_dir / "attribution_components.csv", index=False)
    pd.concat(head_frames, ignore_index=True).to_csv(
        out_dir / "attribution_heads.csv", index=False)
    pd.concat(dec_frames, ignore_index=True).to_csv(out_dir / "decodability.csv", index=False)

    with open(out_dir / "trace_meta.json", "w") as f:
        json.dump({"run": config.experiment_name, "checkpoint": str(checkpoint),
                   "signal": f"N{inject_layer}.{neuron}", "signal_type": signal_type,
                   "inject_layer": inject_layer, "neuron": neuron,
                   "token_position": (config.test_token_selector.position
                                      if config.test_token_selector.type == "fixed" else None),
                   "start_layer": start_layer, "n_examples": len(idx),
                   "n_probe_examples": n_probe_rows,
                   "probe_labels": [lbl for lbl, _, _ in PROBE_SPECS],
                   "regimes": meta_regimes,
                   "local_centre_source": "live tuned activation at (L, i, t)"},
                  f, indent=2)
    print(f"[trace] wrote results to {out_dir}")


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    runs = runs_from_args(args)
    if not runs:
        raise SystemExit("No runs with checkpoints matched.")
    # per run, per trained signal: a multi-locale run is traced once per signal into
    # <run>/circuit_trace/<signal>/, sharing one model load (see run_per_locale)
    run_per_locale(args, runs, "circuit_trace", "decodability.csv",
                   analyze=lambda run, spec, out_dir: trace_one_run(run, args, out_dir),
                   replot=replot_circuit, log="[trace]")
    # multi-locale runs: one side-by-side head map across the trained signals, from the
    # per-signal CSVs just written (or cached), to compare which heads carry each signal
    sweeping = len(runs) > 1 or getattr(args, "group", None) is not None
    for spec in runs:
        if spec.is_multi_locale:
            base = (spec.run_dir / "circuit_trace" if sweeping or not args.output_dir
                    else Path(args.output_dir))
            plot_heads_across_locales(spec, base)

def _plot_attribution(attr_df, n_layers, start_layer, inject_layer, out_dir, tag=""):
    order = ["embed"] + [f"attn_{L}" for L in range(n_layers)] + [f"mlp_{L}" for L in range(n_layers)]
    attr_df = attr_df.set_index("component").loc[order].reset_index()
    plt.figure(figsize=(max(8, 0.18 * len(order)), 4))
    plt.bar(attr_df["component"], attr_df["mean_slope"])
    plt.axhline(0, color="k", lw=0.5)
    plt.xticks(rotation=90, fontsize=6)
    plt.ylabel("dose-response slope (d contribution / d v)")
    plt.title(f"Direct attribution to readout w (per component){tag}")
    plt.tight_layout()
    plt.savefig(out_dir / f"attribution_components{tag}.png", dpi=150)
    plt.close()


def _plot_heads(mat, start_layer, n_layers, n_heads, out_dir, tag=""):
    plt.figure(figsize=(max(6, 0.3 * n_heads), max(3, 0.35 * mat.shape[0])))
    lim = np.abs(mat).max() or 1.0
    sns.heatmap(mat, cmap="coolwarm", center=0, vmin=-lim, vmax=lim,
                yticklabels=list(range(start_layer, n_layers)),
                xticklabels=list(range(n_heads)),
                cbar_kws={"label": "dose-response slope"})
    plt.xlabel("head"); plt.ylabel("layer")
    plt.title(f"Per-head attention attribution to readout w{tag}")
    plt.tight_layout()
    plt.savefig(out_dir / f"attribution_heads{tag}.png", dpi=150)
    plt.close()


def _regime_text(tag):
    """'_x1' -> 'Injection Scale 1\u00d7', '_local@0.5s' -> 'Local Perturbation (\u00b10.5\u03c3)'."""
    name = tag.lstrip("_")
    m = re.fullmatch(r"x([0-9.]+)", name)
    if m:
        return f"Injection Scale {m.group(1)}\u00d7"
    m = re.fullmatch(r"local@([0-9.]+)s", name)
    if m:
        return f"Local Perturbation (\u00b1{m.group(1)}\u03c3)"
    return name


def _plot_decodability(mat, probe_labels, n_layers, inject_layer, out_dir, tag="",
                       run_name=None):
    plt.figure(figsize=(1.35 * len(probe_labels) + 4, max(6, 0.55 * n_layers)))
    shown = [lbl.replace("last", "Last").replace("-", "\u2212") for lbl in probe_labels]
    ax = sns.heatmap(mat, cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f",
                     annot_kws={"size": 12},
                     yticklabels=list(range(n_layers)), xticklabels=shown,
                     cbar_kws={"label": "Probe Corr. (v Decodability)"})
    ax.figure.axes[-1].yaxis.label.set_size(16)          # colorbar label
    ax.figure.axes[-1].tick_params(labelsize=14)
    ax.tick_params(labelsize=14)
    plt.axhline(inject_layer + 0.5, color="w", ls="--", lw=1)
    # the causal null: nothing left of this line can carry v, so its column is the
    # floor the rest of the map is read against
    if "t-1" in probe_labels:
        ax.axvline(probe_labels.index("t-1") + 1, color="w", ls=":", lw=1.5)
    plt.ylabel("Layer", fontsize=17); plt.xlabel("Probe Position", fontsize=17)
    head = f"{pretty_run_name(run_name)}\n" if run_name else ""
    regime = f", {_regime_text(tag)}" if tag else ""
    plt.title(f"{head}Injected-Value Decodability{regime}\n"
              "t\u22121 = Causal Null | t = Origin | Last = Readout", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir / f"decodability{tag}.png", dpi=150, bbox_inches="tight")
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
    # one figure set per regime, as trace_one_run writes them (the CSVs are long over
    # regimes; pivoting them unfiltered raises on duplicate (layer, head) entries)
    for regime in dict.fromkeys(head_df["regime"]):
        tag = f"_{regime}"
        h = head_df[head_df["regime"] == regime]
        d = dec_df[dec_df["regime"] == regime]
        head_mat = (h.pivot(index="layer", columns="head", values="mean_slope")
                    .reindex(index=range(start_layer, n_layers), columns=range(n_heads))
                    .to_numpy())
        dec_mat = (d.pivot(index="layer", columns="probe", values="corr")
                   .reindex(index=range(n_layers), columns=probe_labels).to_numpy())
        _plot_attribution(attr_df[attr_df["regime"] == regime], n_layers, start_layer,
                          inject_layer, out_dir, tag)
        _plot_heads(head_mat, start_layer, n_layers, n_heads, out_dir, tag)
        _plot_decodability(dec_mat, probe_labels, n_layers, inject_layer, out_dir, tag,
                           run_name=spec.config.experiment_name)
    return True


def plot_heads_across_locales(spec, base: Path, top_k: int = 5):
    """
    Multi-locale runs: the per-head attribution maps of every trained signal side by
    side, one figure per regime (attribution_heads_by_locale_<regime>.png in `base`).

    Each panel is transposed relative to attribution_heads_*.png (rows = head, columns =
    layer) so a given head sits on the same row in every panel, and is trimmed to the
    layers ALL signals can reach (max over signals of start_layer -- a head below a
    signal's origin layer can't see it). Panels keep their own colour scale: slope
    magnitudes differ a lot across signal layers, and the question is which heads, not
    how much. The top-|slope| heads of each panel are boxed, and the last panel gives
    the pairwise correlation of the trimmed head maps between signals, so "the same heads
    carry every signal" is a number rather than an impression.
    """
    panels = []  # (tag, start_layer, long df)
    for loc in spec.locales:
        d = locale_out_dir(base, spec, loc)
        csv, meta_p = d / "attribution_heads.csv", d / "trace_meta.json"
        if csv.exists() and meta_p.exists():
            panels.append((locale_tag(*loc), int(json.loads(meta_p.read_text())["start_layer"]),
                           pd.read_csv(csv)))
    if len(panels) < 2:
        print(f"[trace] cross-locale head plot: {len(panels)} signal(s) traced, need 2 -- skipped")
        return
    lo = max(sl for _, sl, _ in panels)
    hi = min(int(df["layer"].max()) for _, _, df in panels)
    n_heads = int(max(df["head"].max() for _, _, df in panels)) + 1
    layers = list(range(lo, hi + 1))
    regimes = [r for r in dict.fromkeys(panels[0][2]["regime"])
               if all((df["regime"] == r).any() for _, _, df in panels)]
    for regime in regimes:
        mats = [(df[df["regime"] == regime]
                 .pivot(index="head", columns="layer", values="mean_slope")
                 .reindex(index=range(n_heads), columns=layers).to_numpy())
                for _, _, df in panels]
        tags = [t for t, _, _ in panels]
        flat = np.stack([m.ravel() for m in mats])
        corr = np.corrcoef(np.nan_to_num(flat))

        n = len(mats)
        col_w = max(1.6, 0.45 * len(layers) + 1.0)
        fig, axes = plt.subplots(
            1, n + 1, figsize=(col_w * n + 0.6 * n + 2.5, max(5, 0.22 * n_heads)),
            gridspec_kw={"width_ratios": [1] * n + [max(1.0, 0.35 * n)]})
        for k, (ax, m, tag) in enumerate(zip(axes[:n], mats, tags)):
            lim = np.nanmax(np.abs(m)) or 1.0
            sns.heatmap(m, ax=ax, cmap="coolwarm", center=0, vmin=-lim, vmax=lim,
                        xticklabels=layers, yticklabels=list(range(n_heads)) if k == 0 else False,
                        cbar_kws={"label": "slope" if k == n - 1 else None, "shrink": 0.6,
                                  "pad": 0.02})
            ax.collections[0].colorbar.ax.tick_params(labelsize=6)
            ax.tick_params(axis="y", labelsize=6)
            # box the top-k heads by |slope| within the trimmed window
            order = np.argsort(np.nan_to_num(np.abs(m)).ravel())[::-1][:top_k]
            for flat_i in order:
                h, c = divmod(int(flat_i), len(layers))
                ax.add_patch(plt.Rectangle((c, h), 1, 1, fill=False, ec="k", lw=1.2))
            ax.set_title(tag, fontsize=9)
            ax.set_xlabel("layer")
            ax.set_ylabel("head" if k == 0 else "")
        ax = axes[n]
        sns.heatmap(corr, ax=ax, cmap="viridis", vmin=-1, vmax=1, annot=True, fmt=".2f",
                    annot_kws={"size": 7}, xticklabels=tags, yticklabels=tags, square=True,
                    cbar=False)
        ax.tick_params(labelsize=7)
        ax.set_title(f"head-map corr\n(layers {lo}-{hi})", fontsize=9)
        fig.suptitle(f"{spec.name}: per-head attribution to readout, layers {lo}-{hi} "
                     f"[{regime}] -- boxes = top {top_k} |slope| per signal", fontsize=10)
        fig.tight_layout()
        out = base / f"attribution_heads_by_locale_{regime}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        off = corr[~np.eye(n, dtype=bool)]
        print(f"[trace] wrote {out.name} (layers {lo}-{hi}; head-map corr "
              f"min {off.min():.2f} / mean {off.mean():.2f})")


if __name__ == "__main__":
    main()
