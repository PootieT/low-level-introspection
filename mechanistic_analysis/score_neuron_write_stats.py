#!/usr/bin/env python3
"""
Write-in statistics of the SFT grid's neurons in the base model: how readable each
neuron's value is from what it writes to the residual stream, versus from its own
input. Inputs to the per-neuron feature regression (plotters/fit_sft_feature_regression.py).

WHY
  In SFT the label is the neuron's natural activation a = silu(g), g = w_gate . x, so
  two readouts fit the training data equally well:
    causal   read the neuron's write:  a * u(x) * W_down[:, i],  u = w_up . x
    bypass   read its input: g is LINEAR in the MLP input x, and x stays in the
             residual stream through the skip connection
  Interventional faithfulness asks which one SFT learned. These features measure how
  easy each channel is to read. The injection hook replaces a, so an injected value
  is also written as a * u(x) * W_down[:, i] -- the same channel as the causal readout.

Per neuron, over every non-BOS token of the cached windows:
  A1  write-in fidelity
      r2_a_h              corr(a, a*u)^2: how well a fixed linear readout of the
                          neuron's hidden value h = a*u recovers a. u flips sign with
                          context, which breaks the monotone map from a to the write.
      u_sign_consistency  |sum_{a>0} a sign(u)| / sum_{a>0} a: 1 = u keeps one sign
                          wherever the neuron fires (weighted by firing strength).
      mean_abs_u, frac_u_positive
  A2  write size relative to the residual
      write_rel           mean over tokens of sigma_a |u| ||W_down[:, i]|| / ||r||, r =
                          the residual after layer L: the size of a one-sigma injection's
                          write relative to the stream it lands in.
      write_rel_natural   std(a*u) ||W_down[:, i]|| / mean ||r||: the neuron's natural
                          write variation, same normalization.
      wd_norm, resid_norm (the parts)
  A3  exactness of the input bypass
      r2_a_g              corr(a, g)^2: how linear SiLU is over the neuron's working
                          range. 1 = the pre-activation (linear in x) is a perfect
                          linear stand-in for a.

  Which quantity to train on: g (gate pre-activation), a = silu(g) (what the grid
  uses), h = a*u (the down_proj input; the post-down_proj write of ONE neuron is
  h * W_down[:, i], so it carries exactly h's information). For each target T:
      bypass_r2_T         held-out ridge R^2 of T from the residual stream BEFORE the
                          MLP (resid_mid: attention output added, MLP not yet) -- how
                          well the input channel predicts T without the neuron's write
      causal_ceiling_T    (E k)^2 / E k^2, where k = d(write along W_down[:, i]) / dT is
                          the per-token gain from an injected T to the residual write:
                          k = silu'(g) u for g, u for a, 1 for h. It is the corr^2 between
                          injected T and ANY fixed linear readout of the write at the
                          injected token (zero-mean injection independent of context).
                          1 = the write moves along one fixed axis with a fixed sign.

Output: <output-dir>/write_stats.csv (default
outputs/plots/mlp_endo_sweep_layer_token/write_stats/).

Usage
  python mechanistic_analysis/score_neuron_write_stats.py --dataset-dir <cache dir> [--n-texts 10000]
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_neuron_context_dependence import grid_neurons, load_windows

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--neurons", default=None, help="Default: the SFT grid's neurons.")
    p.add_argument("--n-texts", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default=None, help="Default: cuda if available, else cpu.")
    p.add_argument("--ridge", type=float, default=1e-3,
                   help="Bypass ridge penalty, as a fraction of mean diag(X'X).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "outputs/plots/mlp_endo_sweep_layer_token/write_stats")
    return p.parse_args()


class WriteCapture:
    """Per layer: the selected neurons' gate and up pre-activations (fp32, from the MLP
    input) and the residual norm after the layer."""

    def __init__(self, model, cols: dict[int, list[int]]):
        self.out, self.handles = {}, []
        for L, idx in cols.items():
            mlp = model.model.layers[L].mlp
            ix = torch.as_tensor(idx, device=mlp.gate_proj.weight.device)
            Wg = mlp.gate_proj.weight[ix].float()
            Wu = mlp.up_proj.weight[ix].float()
            self.handles.append(mlp.register_forward_pre_hook(self._mlp_hook(L, Wg, Wu)))
            norm = model.model.layers[L].post_attention_layernorm
            self.handles.append(norm.register_forward_pre_hook(self._mid_hook(L)))
            self.handles.append(model.model.layers[L].register_forward_hook(self._resid_hook(L)))

    def _mlp_hook(self, L, Wg, Wu):
        def f(_m, inputs):
            x = inputs[0].float()
            self.out[(L, "g")] = (x @ Wg.T).cpu()
            self.out[(L, "u")] = (x @ Wu.T).cpu()
        return f

    def _mid_hook(self, L):
        def f(_m, inputs):
            self.out[(L, "mid")] = inputs[0].float().cpu()
        return f

    def _resid_hook(self, L):
        def f(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            self.out[(L, "r")] = h.float().norm(dim=-1).cpu()
        return f

    def remove(self):
        for h in self.handles:
            h.remove()


def corr2(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float(((a * b).sum() / den) ** 2) if den > 0 else np.nan


class BypassRidge:
    """Held-out ridge R^2 of several targets from resid_mid, from sufficient statistics
    accumulated batch by batch (train = even windows, test = odd)."""

    def __init__(self, d, n_targets):
        z = lambda *s: np.zeros(s)
        self.XX = [z(d + 1, d + 1), z(d + 1, d + 1)]
        self.XY = [z(d + 1, n_targets), z(d + 1, n_targets)]
        self.YY = [z(n_targets), z(n_targets)]
        self.Ys = [z(n_targets), z(n_targets)]
        self.n = [0, 0]

    def add(self, X, Y, split):
        X = np.hstack([X.astype(np.float64), np.ones((len(X), 1))])
        Y = Y.astype(np.float64)
        self.XX[split] += X.T @ X
        self.XY[split] += X.T @ Y
        self.YY[split] += (Y * Y).sum(0)
        self.Ys[split] += Y.sum(0)
        self.n[split] += len(X)

    def r2(self, ridge):
        A = self.XX[0].copy()
        lam = ridge * np.diag(A)[:-1].mean()
        A[:-1, :-1] += lam * np.eye(len(A) - 1)            # intercept unpenalized
        B = np.linalg.solve(A, self.XY[0])
        XX, XY, YY, n = self.XX[1], self.XY[1], self.YY[1], self.n[1]
        sse = YY - 2 * (B * XY).sum(0) + (B * (XX @ B)).sum(0)
        sst = YY - self.Ys[1] ** 2 / n
        return 1 - sse / sst


def ceiling(k):
    return float(k.mean() ** 2 / (k * k).mean())


def write_stats(g, u, r, wd_norm):
    a = F.silu(torch.from_numpy(g)).numpy()
    h = a * u
    sa = a.std()
    fire = a > 0
    w = a[fire]
    return dict(
        r2_a_h=corr2(a, h),
        u_sign_consistency=float(abs((w * np.sign(u[fire])).sum()) / w.sum()) if fire.any() else np.nan,
        mean_abs_u=float(np.abs(u).mean()), frac_u_positive=float((u > 0).mean()),
        write_rel=float((sa * np.abs(u) * wd_norm / r).mean()),
        write_rel_natural=float(h.std() * wd_norm / r.mean()),
        wd_norm=wd_norm, resid_norm=float(r.mean()),
        r2_a_g=corr2(a, g),
        causal_ceiling_g=ceiling(dsilu(g) * u), causal_ceiling_a=ceiling(u),
        causal_ceiling_h=1.0,
        std_a=float(sa))


def dsilu(g):
    s = 1 / (1 + np.exp(-g))
    return s * (1 + g * (1 - s))


def targets(g, u):
    """[n, 3K]: g, a, h for the K neurons, in that block order."""
    a = F.silu(torch.from_numpy(g)).numpy()
    return np.hstack([g, a, a * u])


def main():
    args = parse_args()
    neurons = grid_neurons(args.neurons)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    layers = sorted({L for L, _ in neurons})
    cols = {L: sorted({i for l2, i in neurons if l2 == L}) for L in layers}
    cap = WriteCapture(model, cols)
    windows = np.array(load_windows(args.dataset_dir, args.split, args.n_texts, args.seed))
    acc = {k: [] for L in layers for k in ((L, "g"), (L, "u"), (L, "r"))}
    d_model = model.config.hidden_size
    bypass = {L: BypassRidge(d_model, 3 * len(cols[L])) for L in layers}
    split = np.arange(len(windows)) % 2                            # by window, not token
    try:
        with torch.no_grad():
            for s in range(0, len(windows), args.batch_size):
                model(input_ids=torch.as_tensor(windows[s:s + args.batch_size]).to(device))
                for k in acc:
                    acc[k].append(cap.out[k][:, 1:])               # drop BOS
                sp = split[s:s + args.batch_size]
                for L in layers:
                    K = len(cols[L])
                    X = cap.out[(L, "mid")][:, 1:].numpy()
                    T = targets(cap.out[(L, "g")][:, 1:].reshape(-1, K).numpy(),
                                cap.out[(L, "u")][:, 1:].reshape(-1, K).numpy())
                    T = T.reshape(X.shape[0], X.shape[1], -1)
                    for half in (0, 1):
                        m = sp == half
                        if m.any():
                            bypass[L].add(X[m].reshape(-1, d_model),
                                          T[m].reshape(-1, T.shape[-1]), half)
    finally:
        cap.remove()
    rows = []
    for L in layers:
        g = torch.cat(acc[(L, "g")]).reshape(-1, len(cols[L])).numpy()
        u = torch.cat(acc[(L, "u")]).reshape(-1, len(cols[L])).numpy()
        r = torch.cat(acc[(L, "r")]).reshape(-1).numpy()
        Wd = model.model.layers[L].mlp.down_proj.weight.detach().float()
        K = len(cols[L])
        br2 = bypass[L].r2(args.ridge)
        for k, i in enumerate(cols[L]):
            wd = float(Wd[:, i].norm())
            rows.append(dict(neuron=f"N{L}.{i}", layer=L, index=i,
                             **write_stats(g[:, k], u[:, k], r, wd),
                             bypass_r2_g=br2[k], bypass_r2_a=br2[K + k],
                             bypass_r2_h=br2[2 * K + k]))
    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "write_stats.csv", index=False)
    (args.output_dir / "write_stats_meta.json").write_text(json.dumps(dict(
        dataset_dir=str(args.dataset_dir), split=args.split, model=args.model,
        n_windows=len(windows), n_samples_per_neuron=int(windows.shape[0] * (windows.shape[1] - 1)),
        ridge=args.ridge, bypass_input="resid_mid (input of post_attention_layernorm)",
        bypass_split="even windows train, odd windows test"),
        indent=2))
    print(df.groupby("layer")[["r2_a_h", "u_sign_consistency", "write_rel", "r2_a_g",
                               "resid_norm"]].mean().round(3).to_string())
    print(df.groupby("layer")[["bypass_r2_g", "bypass_r2_a", "bypass_r2_h", "causal_ceiling_g",
                               "causal_ceiling_a"]].mean().round(3).to_string())
    print(f"[write] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
