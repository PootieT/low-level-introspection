#!/usr/bin/env python3
"""
Which residual-stream subspace carries each neuron's signal from where it is written
(token t) to where it is read (the last token), and is that subspace shared across
neurons or private to each?

Built for the neuron-generalization runs (one readout, N layer-4 neurons, the prompt
names which one to report), both unconstrained and KL-budgeted. Works on any run,
including single-neuron ones (the across-neuron parts then have one neuron to compare).

THE PICTURE BEING TESTED
  write-in      the neuron's value enters the residual at t (layer L) and is written
                along some direction d_k(l), layer by layer, at t. Causal attention
                makes this independent of the prompt.
  transport     attention moves it to the readout token. D_{i<-j}(l) is the change in
                the readout token's residual at layer l when neuron j is injected and
                the prompt names neuron i.
  selection     somewhere the prompt has to pick the named neuron's channel and turn it
                into the readout direction w. If transport is prompt-independent,
                D_{i<-j} ~ D_{j<-j} for i != j up to the selection layer, then collapses.

A held-out neuron (never trained) can fail at either stage:
  write-in failure        the adapter amplified only the trained neurons' writes, so
                          d_k(l) for a held-out k is small, or lies outside the trained
                          neurons' subspace
  transformation failure  d_k(l) lands in the shared subspace, but the prompt naming k
                          cannot select it, so D_{k<-k} collapses where D_{i<-i} does not

WHAT IS MEASURED (finite differences, per example, inject mu_j +/- scale*sigma_j at
(L, j, t); every direction is in units of one sigma of that neuron, so neurons with
different activation scales are comparable)

  geometry      per layer and site (t, readout): direction norms; rank-1 energy of
                the per-example directions (1 = one fixed AXIS across contexts -- the
                write-in's sign flips with up_proj(x), so axes, not signed means);
                context dimension (participation ratio of per-example directions);
                mean |cos| between trained neurons' principal directions, and between
                held-out and trained ones; |cos| of each direction with the readout
                covector (w*gamma before the final norm, w after it); the union
                participation ratio (effective number of distinct directions among the
                N trained ones, in the CSV); the same in the BASE model (adapter
                disabled), to separate directions the base model already had from ones
                the adapter created.
  cross-talk    every (prompt i, source j) pair over trained + held-out neurons, per
                layer at the readout token: norm ratio ||D_{i<-j}|| / ||D_{j<-j}|| and
                per-example |cos(D_{i<-j}(x), D_{j<-j}(x))| on the same text (signed means
                would cancel, see geometry); plus the readout itself, R[i, j] =
                change in the regression output per sigma of j with the prompt naming i.
  held-out      write-in norm adapted vs base (did the adapter amplify the write?),
                fraction of d_k(l) inside the trained neurons' write-in subspace (vs
                the random-subspace expectation dim/d_model), readout-token norm and
                readout slope.
  ablation      for each trained neuron i at a few layers: patch the residual over
                t..last back to its uninjected value INSIDE a subspace,
                    h <- h - (h - h_clean) U U^T,
                which removes exactly the injection's effect within U and nothing else.
                U = i's own subspace (top PCs of its per-example readout directions plus
                its write-in); ONE other trained neuron's own subspace at a time (same
                dimension as own, so own-vs-other is not a dimension comparison); the
                union of all other trained neurons' subspaces; and random subspaces of
                the own and union dimensions. Retained fraction = readout slope under
                the patch / clean slope, pooled over neurons as sum/sum so a neuron the
                readout barely reads cannot dominate.
                    own kills, others do not       private channels
                    others kill too                shared channel
                    random kills as much           dimension-generic, not a channel

Usage
-----
python mechanistic_analysis/trace_signal_subspace.py --config experiment_configs/mlp_neurongen/neurongen_data_n30_30k.json
python mechanistic_analysis/trace_signal_subspace.py --group mlp_neurongen --run-glob 'neurongen_kl0.1_*' --skip-existing
python mechanistic_analysis/trace_signal_subspace.py --config <cfg> --from-cache     # replot
python mechanistic_analysis/trace_signal_subspace.py --config <cfg> --scale 10        # -> signal_subspace_x10/

Outputs in <run>/signal_subspace/:
  signal_subspace_geometry.csv     per (site, layer, model): geometry summary
  signal_subspace_neurons.csv      per (neuron, site, layer, model): norms, rank-1 energy,
                                   context dim, capture into trained subspace
  signal_subspace_crosstalk.csv    per (prompt i, source j, layer): norm ratio, |cos|
  signal_subspace_readout.csv      per (prompt i, source j): readout slope R[i, j]
  signal_subspace_ablation.csv     per (neuron, layer, condition): retained fraction
  signal_subspace_dirs.npz         principal and mean directions (fp16), for later
                                   cross-run analyses
  signal_subspace_*.png            figures; signal_subspace_meta.json (sentinel)
"""
from __future__ import annotations

import sys
import json
import argparse
import operator
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from src.introspector.analysis_loading import (
    load_run_for_analysis, absolute_positions, layer_output_is_tuple)
from src.introspector.head_ablation import window_mask
from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.run_discovery import (
    add_discovery_args, runs_from_args, resolve_run_action, pretty_run_name)
from src.introspector.training.model import InterventionHook

ANALYSIS = "signal_subspace"
SENTINEL = "signal_subspace_meta.json"
SITES = ("t", "readout")
GROUPS = {("trained", "trained"): "Trained ← Trained",
          ("trained", "heldout"): "Trained Prompt ← Held-Out Source",
          ("heldout", "trained"): "Held-Out Prompt ← Trained Source",
          ("heldout", "heldout"): "Held-Out ← Held-Out"}
ABL_CONDITIONS = ("own", "single_other", "random_own", "others", "random_others")
ABL_LABELS = {"own": "Own Subspace", "single_other": "One Other Neuron's Subspace",
              "random_own": "Random (Own Dim)", "others": "All Other Neurons' Subspaces",
              "random_others": "Random (All-Others Dim)"}
FONTS = {"title": 16, "label": 15, "tick": 12, "legend": 11}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--n-texts", type=int, default=64,
                   help="Texts per neuron. Every neuron is measured on the SAME texts, so "
                        "all comparisons are paired.")
    p.add_argument("--max-trained", type=int, default=30,
                   help="Cap on trained neurons analysed (cross-talk is quadratic).")
    p.add_argument("--n-heldout", type=int, default=10,
                   help="Held-out neurons: the pool neurons this run did NOT train on "
                        "(from the group's _pool_*.json), else random layer-L indices.")
    p.add_argument("--pool-config", type=Path, default=None,
                   help="Pool config listing candidate held-out neurons (default: "
                        "_pool_L<L>_POS<pos>.json next to the run's config).")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Injection is mu +/- scale*sigma. 1 = the neuron's natural range.")
    p.add_argument("--ablate-layers", type=str, default="auto",
                   help="Comma list of layers for the subspace ablation; 'auto' = L+1, "
                        "two evenly spaced middle layers, and the second-to-last layer.")
    p.add_argument("--own-dims", type=int, default=4,
                   help="Dimensions of each neuron's own subspace (top PCs of its "
                        "per-example readout directions, plus its write-in direction).")
    p.add_argument("--max-others-dims", type=int, default=256)
    p.add_argument("--max-single-others", type=int, default=10,
                   help="Per trained neuron, how many OTHER neurons' own subspaces to "
                        "patch one at a time (same dimension as its own subspace, so the "
                        "own-vs-other comparison is not confounded by dimension).")
    p.add_argument("--no-crosstalk", action="store_true",
                   help="Diagonal only (prompt names the injected neuron): skips the "
                        "quadratic part.")
    p.add_argument("--no-ablation", action="store_true")
    p.add_argument("--bf16", action="store_true",
                   help="Keep the run's bf16 forward. Default casts the model to fp32: a "
                        "one-sigma change is ~1e-3 per residual coordinate, near bf16's "
                        "resolution at residual magnitudes, and bf16 finite differences "
                        "fail the write-in exactness check (|cos| to the W_down column "
                        "0.73-0.93 at layer L, where it is 1 by construction).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    add_discovery_args(p)
    return p.parse_args()


# ======================================================================
# Rows: the same texts, prompted to name any neuron
# ======================================================================
def source_texts(run, n_texts):
    """n_texts raw rows of the run's FIRST trained neuron. Every other neuron is
    measured on these same texts with the prompt re-pointed, so differences between
    neurons are not differences between texts."""
    L0, i0 = run.locales[0][1], run.locales[0][2]
    raw = run.raw.filter(lambda ex: int(ex["signal_layer"]) == L0
                         and int(ex["signal_index"]) == i0)
    raw = raw.select(range(min(len(raw), 4 * n_texts)))   # spare rows for invalid drops
    return [dict(r) for r in raw]


def prompted_rows(run, texts, layer, index):
    """Preprocess the texts with the prompt naming (layer, index). Returns the model
    rows and a validity flag per text (the introspected token survived max_length)."""
    out, valid = [], []
    for ex in texts:
        ex = dict(ex, signal_layer=layer, signal_index=index)
        r = run.preprocess_fn(ex)
        valid.append(bool(r.pop("position_valid", True)))
        out.append(r)
    return out, np.array(valid)


def collate(run, rows, device):
    batch = run.collator(rows)
    return {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}


def tile(batch, k):
    return {name: torch.cat([v] * k, dim=0) for name, v in batch.items()}


# ======================================================================
# Natural statistics and held-out neurons
# ======================================================================
class _Capture:
    def __init__(self, idx):
        self.idx, self.out = idx, None

    def __call__(self, module, inputs, outputs):
        self.out = outputs[:, :, self.idx].detach().float()


@torch.no_grad()
def neuron_stats(run, batches, layer, indices, device):
    """mean/std of each neuron at t over the analysis texts, in the trained model."""
    idx = torch.as_tensor(indices, device=device)
    cap = _Capture(idx)
    path = resolve_mlp_signal_path(run.model, layer, path_to_output=False)
    vals = []
    handle = operator.attrgetter(path)(run.model).register_forward_hook(cap)
    try:
        for b in batches:
            run.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                      lora_token_mask=b.get("lora_token_mask"), do_intervention=False)
            t = absolute_positions(b["token_position"], b["attention_mask"])
            vals.append(cap.out[torch.arange(len(t), device=device), t].cpu())
    finally:
        handle.remove()
    v = torch.cat(vals).numpy()
    return {k: (float(v[:, n].mean()), float(v[:, n].std())) for n, k in enumerate(indices)}


def heldout_neurons(config_path, run, trained, n, pool_config, rng):
    """Pool neurons the run did not train on (the neuron-gen eval's unseen set), else
    random indices of the signal layer."""
    L = run.layer
    if n <= 0:
        return []
    if pool_config is None:
        cands = sorted(Path(config_path).parent.glob(f"_pool_L{L}_POS*.json"))
        pool_config = cands[0] if cands else None
    if pool_config is not None and Path(pool_config).exists():
        sigs = json.loads(Path(pool_config).read_text()).get("train_signals", [])
        pool = [int(s["index"]) for s in sigs if int(s.get("layer", L)) == L]
        out = [k for k in pool if k not in set(trained)][:n]
        if out:
            print(f"[subspace] held-out neurons from {Path(pool_config).name}: {len(out)}")
            return out
    size = run.base.config.intermediate_size
    pool = np.setdiff1d(np.arange(size), trained)
    print(f"[subspace] no pool config; {n} random layer-{L} indices as held-out")
    return sorted(int(x) for x in rng.choice(pool, size=n, replace=False))


# ======================================================================
# Forward passes
# ======================================================================
class SubspacePatch:
    """Decoder-layer forward hook: h <- h - (h - h_clean) U U^T inside a token window."""

    def __init__(self, U, h_clean, mask, is_tuple):
        self.U, self.h_clean, self.mask, self.is_tuple = U, h_clean, mask, is_tuple

    def __call__(self, module, inputs, output):
        h = output[0] if self.is_tuple else output
        diff = (h - self.h_clean).float()
        proj = (diff @ self.U) @ self.U.T
        h = h - (proj * self.mask.unsqueeze(-1)).to(h.dtype)
        return (h,) + tuple(output[1:]) if self.is_tuple else h


@torch.no_grad()
def forward(run, batch, layers, inj_index=None, inj_values=None, adapter=True,
            patch=None, capture_layers=None, keep_sites=True):
    """One pass. Returns (h_t [B, nL, d], h_last [B, nL, d], pred [B], captured).

    `layers` are residual layers read from hidden_states[l + 1] (the output of decoder
    layer l; the last one is post final-norm, i.e. the space the head reads).
    `capture_layers` additionally returns the full [B, S, d] output of those decoder
    layers (the clean reference the subspace patch needs)."""
    model = run.model
    handles, captured = [], {}
    B = batch["input_ids"].shape[0]
    t = absolute_positions(batch["token_position"], batch["attention_mask"])
    try:
        if inj_index is not None:
            path = resolve_mlp_signal_path(model, run.layer, path_to_output=False)
            hook = InterventionHook(row_indices=torch.arange(B, device=t.device),
                                    token_positions=t, signal_indices=inj_index,
                                    intervention_values=inj_values)
            handles.append(operator.attrgetter(path)(model).register_forward_hook(hook))
        if patch is not None:
            l, hook = patch
            handles.append(run.base.model.layers[l].register_forward_hook(hook))
        ctx = model.disable_adapter() if not adapter else nullcontext()
        with ctx:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        lora_token_mask=batch.get("lora_token_mask"), do_intervention=False,
                        output_hidden_states=True)
    finally:
        for h in handles:
            h.remove()
    hs = out.hidden_states
    for l in capture_layers or ():
        captured[l] = hs[l + 1]
    rows = torch.arange(B, device=t.device)
    h_t = h_last = None
    if keep_sites:
        h_t = torch.stack([hs[l + 1][rows, t] for l in layers], 1).float()
        h_last = torch.stack([hs[l + 1][:, -1] for l in layers], 1).float()
    head = model.regression_head
    pred = head(hs[-1][:, -1].to(device=head.weight.device, dtype=head.weight.dtype))
    return h_t, h_last, pred.squeeze(-1).float(), captured


def inject_values(stats, sources, n, scale, device):
    """[2 * len(sources) * n] values: for each source j, n rows at mu+s*sigma then n at
    mu-s*sigma -- the layout `tile` + `split_pm` assume."""
    idx, val = [], []
    for j in sources:
        mu, sd = stats[j]
        for sign in (1.0, -1.0):
            idx += [j] * n
            val += [mu + sign * scale * sd] * n
    return (torch.tensor(idx, device=device, dtype=torch.long),
            torch.tensor(val, device=device, dtype=torch.float32))


def split_pm(x, n_src, n, scale):
    """[2*n_src*n, ...] -> per-sigma difference [n_src, n, ...]."""
    x = x.reshape(n_src, 2, n, *x.shape[1:])
    return (x[:, 0] - x[:, 1]) / (2 * scale)


# ======================================================================
# Linear algebra helpers
# ======================================================================
def unit(v, eps=1e-12):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def remove_dir(v, r):
    return v - (v @ r)[..., None] * r


def participation_ratio(X):
    """PR of the singular spectrum of the rows of X (energy, uncentered)."""
    s2 = np.linalg.svd(X, compute_uv=False) ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum()) if s2.sum() > 0 else np.nan


def principal(X):
    """Sign-invariant direction of a set of per-example vectors [n, d]: the top right
    singular vector (sign set to agree with the mean), and the fraction of energy it
    holds (1 = every example moves along one axis, whatever its sign)."""
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    if s.size == 0 or s[0] == 0:
        return np.zeros(X.shape[1], np.float32), np.nan
    v = Vt[0]
    if v @ X.mean(0) < 0:
        v = -v
    return v.astype(np.float32), float(s[0] ** 2 / (s ** 2).sum())


def mean_abs_offdiag_cos(M):
    if len(M) < 2:
        return np.nan
    C = unit(M) @ unit(M).T
    return float(np.abs(C[~np.eye(len(M), dtype=bool)]).mean())


def cross_abs_cos(A, B):
    """Mean |cos| between every row of A and every row of B."""
    if len(A) == 0 or len(B) == 0:
        return np.nan
    return float(np.abs(unit(A) @ unit(B).T).mean())


def orthobasis(X, k=None, tol=1e-6):
    """Orthonormal basis [d, r] of the row space of X, optionally top-k by energy."""
    if len(X) == 0:
        return np.zeros((X.shape[1], 0))
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    keep = s > tol * s.max()
    Vt = Vt[keep]
    if k is not None:
        Vt = Vt[:k]
    return Vt.T


def random_basis(d, k, rng):
    if k == 0:
        return np.zeros((d, 0))
    Q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return Q


# ======================================================================
# The measurement
# ======================================================================
def analyze_run(run, spec, args, out_dir: Path):
    device = run.model.device
    rng = np.random.default_rng(args.seed)
    L = run.layer
    n_layers = run.base.config.num_hidden_layers
    layers = list(range(L, n_layers))
    trained_all = [loc[2] for loc in run.locales if loc[1] == L]
    trained = trained_all[:args.max_trained]
    heldout = heldout_neurons(spec.config_path, run, trained_all, args.n_heldout,
                              args.pool_config, rng)
    neurons = trained + heldout
    role = {k: "trained" for k in trained} | {k: "heldout" for k in heldout}
    print(f"[subspace] {len(trained)} trained (of {len(trained_all)}), {len(heldout)} held-out "
          f"neurons; layers {layers[0]}..{layers[-1]}")

    # ---- rows: the same texts, prompted for every neuron; keep texts valid for all ----
    texts = source_texts(run, args.n_texts)
    rows, valid = {}, np.ones(len(texts), bool)
    for k in neurons:
        rows[k], v = prompted_rows(run, texts, L, k)
        valid &= v
    keep = np.flatnonzero(valid)[:args.n_texts]
    if len(keep) < 8:
        raise SystemExit(f"only {len(keep)} texts valid for every prompt")
    rows = {k: [r[i] for i in keep] for k, r in rows.items()}
    n = len(keep)
    # texts per forward: 2 signs x g sources x chunk texts <= batch size
    chunk = min(n, max(1, args.batch_size // 2))
    chunks = [list(range(s, min(n, s + chunk))) for s in range(0, n, chunk)]
    g_src = max(1, args.batch_size // (2 * chunk))
    batches = {k: [collate(run, [rows[k][i] for i in c], device) for c in chunks]
               for k in neurons}

    stats = neuron_stats(run, batches[neurons[0]], L, neurons, device)
    dead = [k for k in neurons if stats[k][1] < 1e-6]
    if dead:
        raise SystemExit(f"neurons with zero variance on these texts: {dead}")

    gamma = run.base.model.norm.weight.detach().float().cpu().numpy()
    w = run.model.regression_head.weight.detach().float().cpu().numpy().reshape(-1)
    # the readout covector in each layer's space: the head reads the POST-norm final
    # layer directly (w), and every earlier residual through the final RMSNorm (w*gamma)
    r_pre, r_post = unit(w * gamma), unit(w)
    r_for = lambda l: r_post if l == n_layers - 1 else r_pre
    d_model = len(gamma)

    nL = len(layers)
    K = len(neurons)
    kidx = {k: a for a, k in enumerate(neurons)}
    tr = [kidx[k] for k in trained]
    # per-example diagonal directions (prompt k, inject k), both sites, both models
    diag = {s: np.zeros((K, n, nL, d_model), np.float16) for s in SITES}
    diag_base = {s: np.zeros((K, n, nL, d_model), np.float16) for s in SITES}
    # pairs at the readout token: mean per-example norm and per-example |cos| to the
    # matched-prompt direction on the SAME text; readout slope per text
    pair_norm = np.zeros((K, K, nL), np.float64)
    pair_abscos = np.zeros((K, K, nL), np.float64)
    readout_ex = np.zeros((K, K, n), np.float32)

    # ---- phase 1: matched prompt (prompt names the injected neuron), adapted + base ----
    print(f"[subspace] {n} texts, {K} neurons: matched-prompt passes")
    for a, i in enumerate(neurons):
        for c_idx, c in enumerate(chunks):
            b, nc = batches[i][c_idx], len(c)
            idx, val = inject_values(stats, [i], nc, args.scale, device)
            for adapter, store in ((True, diag), (False, diag_base)):
                h_t, h_last, pred, _ = forward(run, tile(b, 2), layers, idx, val,
                                               adapter=adapter)
                store["t"][a, c] = split_pm(h_t, 1, nc, args.scale)[0].cpu().numpy()
                store["readout"][a, c] = split_pm(h_last, 1, nc, args.scale)[0].cpu().numpy()
                if adapter:
                    readout_ex[a, a, c] = split_pm(pred, 1, nc, args.scale)[0].cpu().numpy()
        X = diag["readout"][a].astype(np.float32)
        pair_norm[a, a] = np.linalg.norm(X, axis=-1).mean(0)
        pair_abscos[a, a] = 1.0

    # ---- phase 2: mismatched prompt (inject j, prompt names i != j) ----
    if not args.no_crosstalk and K > 1:
        print(f"[subspace] cross-talk: {K} prompts x {K - 1} other sources")
        for a, i in enumerate(neurons):
            srcs = [j for j in neurons if j != i]
            for c_idx, c in enumerate(chunks):
                b, nc = batches[i][c_idx], len(c)
                for s0 in range(0, len(srcs), g_src):
                    grp = srcs[s0:s0 + g_src]
                    idx, val = inject_values(stats, grp, nc, args.scale, device)
                    _, h_last, pred, _ = forward(run, tile(b, 2 * len(grp)), layers, idx, val)
                    D = split_pm(h_last, len(grp), nc, args.scale).cpu().numpy()
                    dp = split_pm(pred, len(grp), nc, args.scale).cpu().numpy()
                    for gi, j in enumerate(grp):
                        bj = kidx[j]
                        ref = diag["readout"][bj, c].astype(np.float32)     # [nc, nL, d]
                        nrm = np.linalg.norm(D[gi], axis=-1)
                        cos = np.abs((unit(D[gi]) * unit(ref)).sum(-1))
                        pair_norm[a, bj] += nrm.sum(0) / n
                        pair_abscos[a, bj] += cos.sum(0) / n
                        readout_ex[a, bj, c] = dp[gi]
            if (a + 1) % 5 == 0 or a + 1 == K:
                print(f"[subspace] prompts {a + 1}/{K}")
    readout = readout_ex.mean(-1)

    # ---- positive control: the matched readout must respond ----
    if len(tr) and np.abs(readout[tr, tr]).max() < 1e-6:
        print("[subspace] *** POSITIVE CONTROL FAILED: the readout does not move when the "
              "named neuron is injected. Nothing below is meaningful. ***")

    # ---- directions ----
    # The hook replaces act_fn's output, which the MLP multiplies by up_proj(x) before
    # down_proj, so the write-in is dv * up_j(x) * W_down[:, j]: its SIGN flips with
    # up_j(x) across texts. Mean directions partly cancel; every geometric quantity below
    # therefore uses the sign-invariant PRINCIPAL direction (top singular vector of the
    # per-example directions) and per-example |cos|.
    model_sites = {("adapted", s): diag[s] for s in SITES} | \
                  {("base", s): diag_base[s] for s in SITES}
    pdirs, mdirs, rank1 = {}, {}, {}
    for key, arr in model_sites.items():
        P = np.zeros((K, nL, d_model), np.float32)
        E = np.zeros((K, nL))
        for a in range(K):
            for li in range(nL):
                P[a, li], E[a, li] = principal(arr[a, :, li].astype(np.float32))
        pdirs[key], rank1[key] = P, E
        mdirs[key] = arr.astype(np.float32).mean(1)

    neuron_rows, geo_rows = [], []
    for (mdl, site), arr in model_sites.items():
        P = pdirs[(mdl, site)]
        for li, l in enumerate(layers):
            if site == "readout" and l == L:
                continue         # the readout token cannot see t at layer L: exactly 0
            Pt = P[tr, li]
            U_tr = orthobasis(Pt)
            for a, k in enumerate(neurons):
                X = arr[a, :, li].astype(np.float32)                  # [n, d]
                norms = np.linalg.norm(X, axis=1)
                if role[k] == "trained":                              # leave-one-out
                    U = orthobasis(np.delete(Pt, tr.index(a), axis=0))
                else:
                    U = U_tr
                e = float((X ** 2).sum())
                cap = float(((X @ U) ** 2).sum() / max(e, 1e-30)) if U.shape[1] else np.nan
                m = mdirs[(mdl, site)][a, li]
                neuron_rows.append(dict(
                    neuron=k, role=role[k], model=mdl, site=site, layer=l,
                    mean_norm=float(norms.mean()),
                    rank1_energy=float(rank1[(mdl, site)][a, li]),
                    coherence_signed=float(np.linalg.norm(m) / max(norms.mean(), 1e-30)),
                    context_pr=participation_ratio(X),
                    abs_cos_readout_dir=float(abs(P[a, li] @ r_for(l))),
                    capture_trained=cap, capture_random=U.shape[1] / d_model,
                    sigma=stats[k][1]))
            Pt_r = remove_dir(Pt, r_for(l))
            Ph = P[[kidx[k] for k in heldout], li]
            geo_rows.append(dict(
                model=mdl, site=site, layer=l, n_trained=len(tr),
                mean_abs_cos=mean_abs_offdiag_cos(Pt),
                mean_abs_cos_heldout_trained=cross_abs_cos(Ph, Pt),
                mean_abs_cos_r_removed=mean_abs_offdiag_cos(Pt_r),
                union_pr=participation_ratio(unit(Pt)) if len(tr) else np.nan,
                union_pr_r_removed=participation_ratio(unit(Pt_r)) if len(tr) else np.nan,
                random_abs_cos=float(np.sqrt(2 / (np.pi * d_model)))))
    neurons_df = pd.DataFrame(neuron_rows)
    geo_df = pd.DataFrame(geo_rows)
    write_in_ok = write_in_check(run, diag_base["t"][:, :, 0], neurons, stats)

    # ---- cross-talk tables ----
    xt_rows, ro_rows = [], []
    for a, i in enumerate(neurons):
        for bj, j in enumerate(neurons):
            ro_rows.append(dict(prompt=i, source=j, prompt_role=role[i], source_role=role[j],
                                readout_slope=float(readout[a, bj]),
                                readout_slope_sem=float(readout_ex[a, bj].std() / np.sqrt(n))))
            if i != j and (args.no_crosstalk or K == 1):
                continue
            for li, l in enumerate(layers):
                if l == L:
                    continue
                xt_rows.append(dict(
                    prompt=i, source=j, prompt_role=role[i], source_role=role[j], layer=l,
                    mean_norm=float(pair_norm[a, bj, li]),
                    norm_ratio=float(pair_norm[a, bj, li] / max(pair_norm[bj, bj, li], 1e-30)),
                    abs_cos_to_matched=float(pair_abscos[a, bj, li])))
    xt_df = pd.DataFrame(xt_rows)
    ro_df = pd.DataFrame(ro_rows)

    # ---- subspace ablation ----
    abl_df = pd.DataFrame()
    if not args.no_ablation:
        abl_df = ablate(run, args, layers, trained, kidx, batches, chunks, stats, diag,
                        pdirs, rng, d_model)

    out_dir.mkdir(parents=True, exist_ok=True)
    neurons_df.to_csv(out_dir / "signal_subspace_neurons.csv", index=False)
    geo_df.to_csv(out_dir / "signal_subspace_geometry.csv", index=False)
    xt_df.to_csv(out_dir / "signal_subspace_crosstalk.csv", index=False)
    ro_df.to_csv(out_dir / "signal_subspace_readout.csv", index=False)
    if len(abl_df):
        abl_df.to_csv(out_dir / "signal_subspace_ablation.csv", index=False)
    np.savez_compressed(out_dir / "signal_subspace_dirs.npz",
                        neurons=np.array(neurons), roles=np.array([role[k] for k in neurons]),
                        layers=np.array(layers), r_hat=r_pre.astype(np.float32),
                        r_hat_post=r_post.astype(np.float32),
                        **{f"principal_{m}_{s}": v.astype(np.float16)
                           for (m, s), v in pdirs.items()},
                        **{f"mean_{m}_{s}": v.astype(np.float16) for (m, s), v in mdirs.items()})
    meta = dict(run=run.config.experiment_name, checkpoint=str(run.checkpoint),
                signal_layer=L, trained=trained, n_trained_total=len(trained_all),
                heldout=heldout, n_texts=n, scale=args.scale, layers=layers,
                stats={str(k): v for k, v in stats.items()}, write_in_check=bool(write_in_ok),
                note="last layer is post final-norm (the space the head reads)")
    plot_all(out_dir, title=pretty_run_name(run.config.experiment_name))
    (out_dir / SENTINEL).write_text(json.dumps(meta, indent=2))
    print(f"[subspace] wrote {out_dir}")


def write_in_check(run, X_base_L, neurons, stats, tol=0.01):
    """Exactness check on the plumbing. In the BASE model at (layer L, token t), the
    injection changes the MLP output by dv * up_j(x) * W_down[:, j], so every per-example
    direction must be parallel to down_proj's column (|cos| = 1; the sign follows
    up_j(x)) and its norm must be sigma * |up_j(x)| * ||W_down[:, j]||, which this checks
    up to the |up_j(x)| factor via the direction alone. A failure means the finite
    differences are not resolving the injection (precision, positions, or plumbing).
    `X_base_L`: [K, n, d] base-model directions at t, layer L."""
    W = run.base.model.layers[run.layer].mlp.down_proj.weight.detach().float().cpu().numpy()
    bad = []
    for a, k in enumerate(neurons):
        X = X_base_L[a].astype(np.float32)
        nrm = np.linalg.norm(X, axis=1)
        col = unit(W[:, k])
        cos = np.abs(unit(X) @ col)
        wcos = float((cos * nrm).sum() / max(nrm.sum(), 1e-30))     # norm-weighted
        if wcos < 1 - tol:
            bad.append((k, round(wcos, 4)))
    if bad:
        print(f"[subspace] *** WRITE-IN CHECK FAILED (neuron, |cos| to W_down column): "
              f"{bad[:6]} -- directions do not resolve the injection ***")
    else:
        print(f"[subspace] write-in check passed: base-model write-in at L{run.layer} is "
              f"parallel to W_down[:, j] for all {len(neurons)} neurons")
    return not bad


def ablation_layers(spec: str, L, n_layers):
    if spec != "auto":
        return [int(x) for x in spec.split(",")]
    mids = np.linspace(L + 1, n_layers - 2, 4)[1:3].round().astype(int).tolist()
    return sorted({L + 1, *mids, n_layers - 2})


@torch.no_grad()
def ablate(run, args, layers, trained, kidx, batches, chunks, stats, diag, pdirs, rng,
           d_model):
    """Readout slope for each trained neuron with the injection's effect removed inside
    a subspace over t..last at one layer. See the module docstring for the conditions."""
    device = run.model.device
    is_tuple = layer_output_is_tuple(run.model)
    abl_layers = [l for l in ablation_layers(args.ablate_layers, run.layer,
                                             run.base.config.num_hidden_layers)
                  if l in layers]
    rows = []
    for l in abl_layers:
        li = layers.index(l)
        own = {}
        for k in trained:
            a = kidx[k]
            X = diag["readout"][a, :, li].astype(np.float32)
            # the write-in axis at t, at the scale of the readout directions, so the SVD
            # weighs it like one more example
            wt = pdirs[("adapted", "t")][a, li] * np.linalg.norm(X, axis=1).mean()
            own[k] = orthobasis(np.vstack([X, wt[None]]), k=args.own_dims)
        for k in trained:
            others = [own[j] for j in trained if j != k]
            U_oth = (orthobasis(np.hstack(others).T, k=args.max_others_dims)
                     if others else np.zeros((d_model, 0)))
            bases = {"own": own[k], "others": U_oth,
                     "random_own": random_basis(d_model, own[k].shape[1], rng),
                     "random_others": random_basis(d_model, U_oth.shape[1], rng)}
            pool = [j for j in trained if j != k]
            singles = ([int(j) for j in rng.choice(pool, size=min(args.max_single_others,
                                                                 len(pool)), replace=False)]
                       if pool else [])
            for j in singles:
                bases[("single_other", j)] = own[j]
            slopes = {c: [] for c in ("clean",) + tuple(bases)}
            for c_idx, c in enumerate(chunks):
                b = batches[k][c_idx]
                nc = len(c)
                _, _, _, cap = forward(run, b, layers, capture_layers=[l], keep_sites=False)
                h0 = tile({"h": cap[l]}, 2)["h"]
                mask = window_mask("t_to_last", tile(b, 2)["attention_mask"],
                                   tile(b, 2)["token_position"]).to(h0.dtype)
                idx, val = inject_values(stats, [k], nc, args.scale, device)
                _, _, pred, _ = forward(run, tile(b, 2), layers, idx, val, keep_sites=False)
                slopes["clean"].append(split_pm(pred, 1, nc, args.scale)[0].cpu().numpy())
                for cond, U in bases.items():
                    if U.shape[1] == 0:
                        continue
                    Ut = torch.as_tensor(U, device=device, dtype=torch.float32)
                    hook = SubspacePatch(Ut, h0, mask, is_tuple)
                    _, _, pred, _ = forward(run, tile(b, 2), layers, idx, val,
                                            patch=(l, hook), keep_sites=False)
                    slopes[cond].append(split_pm(pred, 1, nc, args.scale)[0].cpu().numpy())
            clean = np.concatenate(slopes["clean"])
            for key, U in bases.items():
                if not slopes[key]:
                    continue
                cond, other = (key if isinstance(key, tuple) else (key, -1))
                s = np.concatenate(slopes[key])
                rows.append(dict(neuron=k, layer=l, condition=cond, other=other,
                                 dim=U.shape[1],
                                 clean_slope=float(clean.mean()), slope=float(s.mean()),
                                 retained=float(s.mean() / clean.mean())
                                 if abs(clean.mean()) > 1e-12 else np.nan))
        print(f"[subspace] ablation layer {l} done")
    return pd.DataFrame(rows)


# ======================================================================
# Figures
# ======================================================================
def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=FONTS["label"])
    ax.set_ylabel(ylabel, fontsize=FONTS["label"])
    ax.tick_params(labelsize=FONTS["tick"])
    ax.grid(alpha=0.25)


def plot_selection(xt, out_dir, title):
    """Cross-talk by measurement layer: does the readout token carry neuron j's signal
    when the prompt names a different neuron, and in the same direction?"""
    off = xt[xt["prompt"] != xt["source"]]
    if off.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = dict(zip(GROUPS, plt.cm.tab10.colors))
    for key, label in GROUPS.items():
        g = off[(off["prompt_role"] == key[0]) & (off["source_role"] == key[1])]
        if g.empty:
            continue
        m = g.groupby("layer")[["norm_ratio", "abs_cos_to_matched"]].agg(["mean", "sem"])
        for ax, col in zip(axes, ("norm_ratio", "abs_cos_to_matched")):
            ax.errorbar(m.index, m[(col, "mean")], yerr=m[(col, "sem")], marker="o",
                        ms=5, capsize=2, lw=1.8, color=colors[key], label=label)
    axes[0].axhline(1, color="k", lw=0.8, ls="--")
    _style(axes[0], "Layer", "||D(i ← j)|| / ||D(j ← j)||")
    _style(axes[1], "Layer", "|cos(D(i ← j), D(j ← j))| per Text")
    axes[0].set_title("Off-Prompt Signal Strength", fontsize=FONTS["title"])
    axes[1].set_title("Off-Prompt Direction Agreement", fontsize=FONTS["title"])
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), fontsize=FONTS["legend"],
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle(title, fontsize=FONTS["title"])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_dir / "signal_subspace_selection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_readout(ro, out_dir, title):
    """R[i, j] / R[j, j]: the readout with the prompt naming i when j is injected,
    relative to the matched prompt. Diagonal = 1 by construction."""
    order = (ro[["source", "source_role"]].drop_duplicates()
             .sort_values("source_role", key=lambda s: s != "trained")["source"].tolist())
    R = ro.pivot(index="prompt", columns="source", values="readout_slope").loc[order, order]
    diag = np.diag(R.values)
    Rn = R.values / np.where(np.abs(diag) > 1e-12, diag, np.nan)[None, :]
    n_tr = int((ro.drop_duplicates("source")["source_role"] == "trained").sum())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6),
                             gridspec_kw=dict(width_ratios=[1.25, 1]))
    lim = np.nanmax(np.abs(Rn)) if np.isfinite(Rn).any() else 1
    im = axes[0].imshow(Rn, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for ax_line in (axes[0].axhline, axes[0].axvline):
        ax_line(n_tr - 0.5, color="k", lw=1)
    axes[0].set_xlabel("Injected Neuron j (Trained | Held-Out)", fontsize=FONTS["label"])
    axes[0].set_ylabel("Prompted Neuron i", fontsize=FONTS["label"])
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title("Readout Cross-Talk R[i, j] / R[j, j]", fontsize=FONTS["title"])
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    d = ro[ro["prompt"] == ro["source"]]
    for k, (rl, col) in enumerate((("trained", "tab:blue"), ("heldout", "tab:orange"))):
        v = d[d["source_role"] == rl]["readout_slope"].values
        axes[1].scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-.1, .1, len(v)),
                        v, color=col, alpha=0.7)
    axes[1].set_xticks([0, 1], ["Trained", "Held-Out"])
    axes[1].axhline(0, color="k", lw=0.8)
    _style(axes[1], "", "Readout Slope per σ (Prompt = Injected)")
    axes[1].set_title("Matched-Prompt Readout Response", fontsize=FONTS["title"])
    fig.suptitle(title, fontsize=FONTS["title"])
    fig.tight_layout()
    fig.savefig(out_dir / "signal_subspace_readout.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


SITE_TITLES = {"t": "Token t (Where the Neuron Fires)", "readout": "Readout Token (Last)"}


def plot_geometry(geo, nd, out_dir, title):
    """Per site (rows): how aligned the neurons' signal directions are (left) and how
    large the signal is (right). Directions are sign-invariant principal directions."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    for r, site in enumerate(SITES):
        ax = axes[r, 0]
        for mdl, ls in (("adapted", "-"), ("base", ":")):
            g = geo[(geo["site"] == site) & (geo["model"] == mdl)]
            ax.plot(g["layer"], g["mean_abs_cos"], ls=ls, marker="o", color="tab:blue",
                    label=f"Trained vs Trained ({mdl.capitalize()})")
            if "mean_abs_cos_heldout_trained" in g and g["mean_abs_cos_heldout_trained"].notna().any():
                ax.plot(g["layer"], g["mean_abs_cos_heldout_trained"], ls=ls, marker="s",
                        color="tab:orange", label=f"Held-Out vs Trained ({mdl.capitalize()})")
        if len(geo):
            ax.axhline(geo["random_abs_cos"].iloc[0], color="gray", lw=1, ls="--",
                       label="Random Directions")
        ax.set_ylim(0, 1.02)
        _style(ax, "Layer", "Mean |cos| of Signal Directions")
        ax.set_title(f"Direction Alignment: {SITE_TITLES[site]}", fontsize=FONTS["title"])
        ax = axes[r, 1]
        for (rl, mdl), col, ls in ((("trained", "adapted"), "tab:blue", "-"),
                                   (("heldout", "adapted"), "tab:orange", "-"),
                                   (("trained", "base"), "tab:blue", ":"),
                                   (("heldout", "base"), "tab:orange", ":")):
            g = nd[(nd["site"] == site) & (nd["role"] == rl) & (nd["model"] == mdl)]
            m = g.groupby("layer")["mean_norm"].mean()
            ax.plot(m.index, m.values, ls=ls, marker="o", ms=4, color=col,
                    label=f"{'Held-Out' if rl == 'heldout' else 'Trained'} ({mdl.capitalize()})")
        ax.set_yscale("log")
        _style(ax, "Layer", "Signal Norm ||\u0394h|| per \u03c3")
        ax.set_title(f"Signal Norm: {SITE_TITLES[site]}", fontsize=FONTS["title"])
    for c in range(2):
        h, lab = axes[0, c].get_legend_handles_labels()
        axes[1, c].legend(h, lab, fontsize=FONTS["legend"], loc="upper center",
                          bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.suptitle(title, fontsize=FONTS["title"])
    fig.tight_layout()
    fig.savefig(out_dir / "signal_subspace_geometry.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def heldout_cos_from_npz(out_dir: Path, geo: pd.DataFrame) -> pd.DataFrame:
    """Backfill Held-Out vs Trained |cos| for results written before that column."""
    f = out_dir / "signal_subspace_dirs.npz"
    if "mean_abs_cos_heldout_trained" in geo or not f.exists():
        return geo
    z = np.load(f)
    tr, layers = z["roles"] == "trained", list(z["layers"])
    vals = []
    for _, r in geo.iterrows():
        key = f"principal_{r['model']}_{r['site']}"
        if key not in z:
            vals.append(np.nan); continue
        P = z[key].astype(np.float32)[:, layers.index(int(r["layer"]))]
        vals.append(cross_abs_cos(P[~tr], P[tr]))
    return geo.assign(mean_abs_cos_heldout_trained=vals)


def plot_heldout(nd, out_dir, title):
    """Where held-out neurons fail: write-in gain at t, capture into the trained
    subspace at t, and survival to the readout token relative to the write-in."""
    a = nd[nd["model"] == "adapted"]
    b = nd[nd["model"] == "base"]
    if not (a["role"] == "heldout").any():
        return
    key = ["neuron", "site", "layer"]
    m = a.merge(b[key + ["mean_norm"]], on=key, suffixes=("", "_base"))
    m["gain"] = m["mean_norm"] / m["mean_norm_base"].clip(lower=1e-20)
    t0 = m[(m["site"] == "t") & (m["layer"] == m["layer"].min())].set_index("neuron")["mean_norm"]
    m["transport"] = m["mean_norm"] / m["neuron"].map(t0)
    panels = [("t", "gain", "Adapter Gain at t (Adapted / Base)", False),
              ("t", "capture_trained", "Fraction in Trained Write-In Subspace", False),
              ("readout", "transport", "Readout-Token Norm / Write-In Norm", True)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (site, col, ylab, logy) in zip(axes, panels):
        for rl, c in (("trained", "tab:blue"), ("heldout", "tab:orange")):
            g = m[(m["site"] == site) & (m["role"] == rl)].groupby("layer")[col]
            mu, se = g.mean(), g.sem()
            ax.errorbar(mu.index, mu.values, yerr=se.values, marker="o", ms=5, capsize=2,
                        color=c, label="Trained (Leave-One-Out)" if
                        (rl == "trained" and col == "capture_trained") else
                        ("Trained" if rl == "trained" else "Held-Out"))
        if col == "capture_trained":
            g = m[(m["site"] == site) & (m["role"] == "heldout")].groupby("layer")["capture_random"]
            ax.plot(g.mean().index, g.mean().values, color="gray", ls="--",
                    label="Random Subspace, Same Dim")
        if col == "gain":
            ax.axhline(1, color="k", lw=0.8, ls="--")
        if logy:
            ax.set_yscale("log")
        _style(ax, "Layer", ylab)
        ax.legend(fontsize=FONTS["legend"])
    axes[0].set_title("Write-In", fontsize=FONTS["title"])
    axes[1].set_title("Write-In Overlap", fontsize=FONTS["title"])
    axes[2].set_title("Transport to Readout", fontsize=FONTS["title"])
    fig.suptitle(title, fontsize=FONTS["title"])
    fig.tight_layout()
    fig.savefig(out_dir / "signal_subspace_heldout.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def pooled_retained(g):
    """Retained fraction over neurons as sum(slope) / sum(clean slope): weighted by how
    strongly each neuron is read, so a neuron whose clean slope is ~0 cannot blow the
    average up (a per-neuron ratio with a near-zero denominator is unbounded)."""
    return float(g["slope"].sum() / g["clean_slope"].sum())


def plot_ablation(abl, out_dir, title, n_boot=500):
    if abl.empty:
        return
    abl = abl.copy()
    if "other" not in abl:
        abl["other"] = -1
    # one row per (neuron, layer, condition): single_other averaged over the others first
    per = (abl.groupby(["neuron", "layer", "condition"])
           .agg(slope=("slope", "mean"), clean_slope=("clean_slope", "first"),
                dim=("dim", "median")).reset_index())
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = {"own": "tab:blue", "single_other": "tab:green", "random_own": "tab:gray",
              "others": "tab:red", "random_others": "silver"}
    for cond in ABL_CONDITIONS:
        g = per[per["condition"] == cond]
        if g.empty:
            continue
        xs, ys, lo, hi = [], [], [], []
        for l, gl in g.groupby("layer"):
            neurons = gl["neuron"].unique()
            boots = [pooled_retained(gl.set_index("neuron").loc[rng.choice(neurons, len(neurons))])
                     for _ in range(n_boot)]
            xs.append(l); ys.append(pooled_retained(gl))
            lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5))
        ys, lo, hi = map(np.array, (ys, lo, hi))
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], marker="o", capsize=3, lw=1.8,
                    color=colors[cond], ls="--" if cond.startswith("random") else "-",
                    label=f"{ABL_LABELS[cond]} (dim {int(g['dim'].median())})")
    ax.axhline(1, color="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    _style(ax, "Patched Layer (Tokens t..Last)", "Readout Slope Retained (Pooled)")
    ax.legend(fontsize=FONTS["legend"], loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=2, frameon=False)
    ax.set_title(title, fontsize=FONTS["title"])
    fig.tight_layout()
    fig.savefig(out_dir / "signal_subspace_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all(out_dir: Path, title: str):
    rd = lambda f: (pd.read_csv(out_dir / f) if (out_dir / f).exists() else pd.DataFrame())
    xt, ro = rd("signal_subspace_crosstalk.csv"), rd("signal_subspace_readout.csv")
    geo, nd = rd("signal_subspace_geometry.csv"), rd("signal_subspace_neurons.csv")
    abl = rd("signal_subspace_ablation.csv")
    if len(xt):
        plot_selection(xt, out_dir, title)
    if len(ro):
        plot_readout(ro, out_dir, title)
    if len(geo) and len(nd):
        plot_geometry(heldout_cos_from_npz(out_dir, geo), nd, out_dir, title)
        plot_heldout(nd, out_dir, title)
    plot_ablation(abl, out_dir, title)


# ======================================================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    runs = runs_from_args(args)
    if not runs:
        raise SystemExit("No runs with checkpoints matched.")
    # a non-default injection size is a separate measurement, kept beside the default
    name = ANALYSIS if args.scale == 1 else f"{ANALYSIS}_x{args.scale:g}"
    for k, spec in enumerate(runs, 1):
        out_dir = spec.run_dir / name
        action = resolve_run_action(args, out_dir / SENTINEL)
        if action == "skip":
            print(f"[subspace] {spec.name}: results exist, skipping")
            continue
        if action == "replot" and (out_dir / "signal_subspace_neurons.csv").exists():
            plot_all(out_dir, pretty_run_name(spec.name))
            print(f"[subspace] {spec.name}: re-plotted from cache")
            continue
        print(f"\n[subspace] ===== [{k}/{len(runs)}] {spec.name} =====")
        run = load_run_for_analysis(spec.config,
                                    checkpoint=args.checkpoint or spec.checkpoint())
        if not args.bf16:
            run.model.float()
        try:
            analyze_run(run, spec, args, out_dir)
        finally:
            del run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
