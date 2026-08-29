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

  2. DECODABILITY MAP over (layer, position).
     Fit a linear probe resid -> v at every layer x {token_position, last_token};
     the correlation map shows where the injected value is present and where it
     arrives at the last token.

Everything is config-driven (model / dataset / signal / checkpoint all come from
the experiment config) so future runs only swap the --config (+ optional CLI
overrides).

Example
-------
python trace_introspection_circuit.py \
    --config experiment_configs/.../intervention_train_reg_N10.24_3K_fixedPaddingBug.json \
    --n-examples 200 --grid-points 10 --start-layer 10
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
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

from generate_experiment_config import load_spec  # noqa: F401 (kept for parity/imports)
from src.introspector.config import ExperimentConfig
from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.prompts import PROMPT_REGISTRY
from src.introspector.training.collator import LoRAGatingCollator
from src.introspector.training.model import (
    IntrospectionRegressionModel,
    load_regression_head,
    load_added_token_embeddings,
    replace_model_with_gated_version,
    replace_lora_with_gated_version,
)
from train_introspector import preprocess_example, INTROSPECTION_SEPARATOR

from nnsight.modeling.language import LanguageModel


# ======================================================================
# Args
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True,
                   help="Experiment config used to train the introspector.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Checkpoint dir override (default: latest under output_dir/experiment_name).")
    p.add_argument("--signal-idx", type=int, default=0,
                   help="Which entry of config.train_signals to trace (default: 0).")
    # example selection
    p.add_argument("--n-examples", type=int, default=200,
                   help="Number of endogenous-success examples to trace.")
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
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


# ======================================================================
# Loading (reuses the training script's building blocks)
# ======================================================================
def compute_signal_ranges(train_dataset_raw):
    df = train_dataset_raw.to_pandas()
    grouped = df.groupby(["signal_layer", "signal_index"])["activation"]
    ranges = {}
    for key, vals in grouped:
        ranges[key] = {"min": float(vals.min()), "max": float(vals.max()),
                       "mean": float(vals.mean()), "std": float(vals.std())}
    return ranges


def load_model_and_data(config: ExperimentConfig, checkpoint: Path):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [INTROSPECTION_SEPARATOR]})

    cache_dir = Path(config.cache_dir) / config.experiment_name
    train_raw = load_from_disk(str(cache_dir / "train"))
    test_raw = load_from_disk(str(cache_dir / "test"))
    signal_ranges = compute_signal_ranges(train_raw)

    base = AutoModelForCausalLM.from_pretrained(
        config.model_name, device_map="auto", local_files_only=True,
        torch_dtype=torch.bfloat16 if config.training_config.bf16 else torch.float32,
    )
    base.resize_token_embeddings(len(tokenizer))
    peft_model = PeftModel.from_pretrained(base, str(checkpoint), autocast_adapter_dtype=False)

    if not config.use_regressor:
        raise SystemExit("This tracer targets the regressor head (config.use_regressor must be True).")
    model = IntrospectionRegressionModel(
        peft_model,
        signal_ranges=signal_ranges,
        intervention_distribution=config.training_config.intervention_signal_distribution,
        intervention_extreme_scale=config.training_config.intervention_extreme_scale,
    )
    load_regression_head(model, str(checkpoint))
    # no-op unless the checkpoint's frozen embeddings were stripped to save disk
    load_added_token_embeddings(model, str(checkpoint))
    replace_model_with_gated_version(model)
    replace_lora_with_gated_version(model)
    model.eval()

    def preprocess(ex):
        out = preprocess_example(
            ex, tokenizer=tokenizer,
            prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
            max_length=config.training_config.max_seq_length,
            return_frozen_input_len=True, for_prediction=True,
        )
        out["token_position"] = ex["token_position"]
        out["signal_layer"] = ex["signal_layer"]
        out["signal_index"] = ex["signal_index"]
        out["activation"] = ex["activation"]
        return out

    test_dataset = test_raw.map(preprocess, remove_columns=test_raw.column_names)
    pred_collator = LoRAGatingCollator(
        tokenizer=tokenizer, model=model, padding=True,
        label_pad_token_id=-100, return_tensors="pt", for_prediction=True,
    )
    return model, tokenizer, test_dataset, test_raw, pred_collator, signal_ranges


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
                neuron, v, n_layers, start_layer):
    """
    One injected forward. Modules are accessed in COMPUTATIONAL-PATH ORDER
    (embed -> for each layer: attention block -> [inject at signal layer] -> mlp
    block -> residual -> final norm), and the injection overwrites the signal
    source in-place via `inject_path` (resolved by signal type, not a hard-coded
    module path) between the attention write and the MLP that consumes the neuron.

    Returns last-token component writes (embed, per-layer attn/mlp), per-layer
    residuals at (inject position, last token), o_proj inputs (>= start_layer),
    the pre-norm final residual, and the post-norm hidden.
    `inputs` must contain ONLY input_ids/attention_mask/lora_token_mask so the
    model's own intervention path does not also fire.
    """
    bs = inputs["input_ids"].shape[0]
    bidx = torch.arange(bs, device=abs_pos.device)
    attn, mlp, resid_inj, resid_last, oproj = {}, {}, {}, {}, {}
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
            # --- residual after the layer (both positions, for decodability) ---
            resid_inj[L] = layer.output[bidx, abs_pos, :].save()
            resid_last[L] = layer.output[bidx, -1, :].save()
        normed = norm.output[bidx, -1, :].save()

    val = lambda p: p.value if hasattr(p, "value") else p
    return {
        "embed": val(emb).float().cpu(),
        "attn": {L: val(t).float().cpu() for L, t in attn.items()},
        "mlp": {L: val(t).float().cpu() for L, t in mlp.items()},
        "final_resid": val(resid_last[n_layers - 1]).float().cpu(),
        "normed": val(normed).float().cpu(),
        "resid_inj": {L: val(t).float().cpu() for L, t in resid_inj.items()},
        "resid_last": {L: val(t).float().cpu() for L, t in resid_last.items()},
        "oproj": {L: val(t).float().cpu() for L, t in oproj.items()},
    }


# ======================================================================
# Main
# ======================================================================
def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(args.config.read_text())
    if args.seed is not None:
        torch.manual_seed(args.seed)

    checkpoint = args.checkpoint
    if checkpoint is None:
        ckpts = glob.glob(str(Path(config.output_dir) / config.experiment_name / "checkpoint-*"))
        if not ckpts:
            raise SystemExit("No checkpoint found; pass --checkpoint.")
        checkpoint = Path(max(ckpts, key=os.path.getctime))
    out_dir = args.output_dir or (Path(config.output_dir) / config.experiment_name / "circuit_trace")
    out_dir.mkdir(parents=True, exist_ok=True)

    signal = config.train_signals[args.signal_idx]
    inject_layer, neuron = signal.layer, signal.index
    start_layer = args.start_layer if args.start_layer is not None else inject_layer
    print(f"[trace] signal={signal} inject_layer={inject_layer} neuron={neuron} "
          f"start_layer={start_layer} checkpoint={checkpoint}")

    model, tokenizer, test_dataset, test_raw, collator, signal_ranges = load_model_and_data(config, checkpoint)
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

    lm = LanguageModel(model, tokenizer=tokenizer)
    layers, embed, norm = nnsight_module_handles(lm, inject_layer)
    inject_path = resolve_signal_injection_path(lm, signal.type, inject_layer)

    # accumulators indexed [example, grid_point]
    comp_names = ["embed"] + [f"attn_{L}" for L in range(n_layers)] + [f"mlp_{L}" for L in range(n_layers)]
    contribs = {c: [] for c in comp_names}                       # each -> list over grid of [Nexp]
    head_contribs = {L: [] for L in range(start_layer, n_layers)}  # -> list over grid of [Nexp, n_heads]
    pred_check = []                                              # [grid][Nexp]
    resid_inj = {L: [] for L in range(n_layers)}                # decodability
    resid_last = {L: [] for L in range(n_layers)}

    from torch.utils.data import DataLoader
    for gi, v in enumerate(grid):
        dl = DataLoader(sub, batch_size=args.batch_size, collate_fn=collator, shuffle=False)
        g_contribs = {c: [] for c in comp_names}
        g_head = {L: [] for L in range(start_layer, n_layers)}
        g_pred, g_ri, g_rl = [], {L: [] for L in range(n_layers)}, {L: [] for L in range(n_layers)}
        for batch in dl:
            batch = batch.to(device)
            attn_mask = batch["attention_mask"]
            token_pos = batch["token_position"]
            abs_pos = token_pos + (attn_mask == 0).sum(dim=1)         # left-pad offset fix
            trace_inputs = {"input_ids": batch["input_ids"],
                            "attention_mask": attn_mask,
                            "lora_token_mask": batch.get("lora_token_mask")}
            cap = trace_batch(lm, layers, embed, norm, trace_inputs, abs_pos,
                              inject_layer, inject_path, neuron, float(v), n_layers, start_layer)

            a = readout_scale(cap["final_resid"], w, gamma, eps)      # [b,H]
            g_contribs["embed"].append(contribution(a, cap["embed"]))
            for L in range(n_layers):
                g_contribs[f"attn_{L}"].append(contribution(a, cap["attn"][L]))
                g_contribs[f"mlp_{L}"].append(contribution(a, cap["mlp"][L]))
            for L in range(start_layer, n_layers):
                g_head[L].append(per_head_contributions(cap["oproj"][L], W_O[L], n_heads, head_dim, a))
            # head prediction (from post-norm hidden) for the decomposition sanity check
            g_pred.append((cap["normed"] * w).sum(-1) + bias)
            for L in range(n_layers):
                g_ri[L].append(cap["resid_inj"][L]); g_rl[L].append(cap["resid_last"][L])

        for c in comp_names:
            contribs[c].append(torch.cat(g_contribs[c]))
        for L in range(start_layer, n_layers):
            head_contribs[L].append(torch.cat(g_head[L]))
        pred_check.append(torch.cat(g_pred))
        for L in range(n_layers):
            resid_inj[L].append(torch.cat(g_ri[L])); resid_last[L].append(torch.cat(g_rl[L]))
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
    dec_rows = []
    dec_mat = np.zeros((n_layers, 2))  # cols: [token_position, last_token]
    for L in range(n_layers):
        Xi = torch.stack(resid_inj[L], 1).reshape(-1, resid_inj[L][0].shape[-1]).numpy()
        Xl = torch.stack(resid_last[L], 1).reshape(-1, resid_last[L][0].shape[-1]).numpy()
        y = np.tile(v_np, Nexp)  # matches reshape order [Nexp, grid] -> row-major
        ci = decodability(Xi, y); cl = decodability(Xl, y)
        dec_mat[L] = [ci, cl]
        dec_rows.append({"layer": L, "corr_token_position": ci, "corr_last_token": cl})
    pd.DataFrame(dec_rows).to_csv(out_dir / "decodability.csv", index=False)

    # ============================ plots ============================
    _plot_attribution(attr_df, n_layers, start_layer, inject_layer, out_dir)
    _plot_heads(head_slope_mat, start_layer, n_layers, n_heads, out_dir)
    _plot_decodability(dec_mat, n_layers, inject_layer, out_dir)

    with open(out_dir / "trace_meta.json", "w") as f:
        json.dump({"checkpoint": str(checkpoint), "signal": str(signal),
                   "inject_layer": inject_layer, "neuron": neuron, "start_layer": start_layer,
                   "n_examples": Nexp, "grid": v_np.tolist(),
                   "decomposition_max_err": max_err}, f, indent=2)
    print(f"[trace] wrote results to {out_dir}")


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


def _plot_decodability(mat, n_layers, inject_layer, out_dir):
    plt.figure(figsize=(4, max(4, 0.35 * n_layers)))
    sns.heatmap(mat, cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f",
                yticklabels=list(range(n_layers)),
                xticklabels=["token_position", "last_token"],
                cbar_kws={"label": "probe corr (v decodability)"})
    plt.axhline(inject_layer + 0.5, color="w", ls="--", lw=1)
    plt.ylabel("layer")
    plt.title("Injected-value decodability by (layer, position)")
    plt.tight_layout()
    plt.savefig(out_dir / "decodability.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
