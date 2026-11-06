#!/usr/bin/env python3
"""
How much preceding context does a neuron need? Scores each neuron by how well its
activation is reproduced when the model sees only the last n tokens.

For every sampled (text, position p), the base model is run twice:
  full        the whole cached window (BOS + up to 31 tokens), activation at p
  truncated   BOS + the n tokens ending at p (p itself included), activation at the
              last position, for n = 1 .. n_max

    R2_n  = corr(a_truncated_n, a_full)^2      over all (text, p) samples
    n_eff = 1 + sum_{n=1}^{n_max} (1 - R2_n)

n_eff is the weighted average "how many tokens does it take": with
dR2_n = R2_n - R2_{n-1} (R2_0 = 0) and the curve reaching 1, sum_n n * dR2_n equals
sum_{n>=0} (1 - R2_n) by summation by parts. A neuron fully determined by the current
token has R2_1 = 1 and n_eff = 1. R2 at n_max is reported separately: whatever is
still missing there is dependence on context longer than n_max, which n_eff (capped
at n_max + 1) does not resolve.

WHY THE MODEL ITSELF RATHER THAN FITTED n-GRAM TABLES. A lookup table of mean
activation per n-gram is starved beyond n=1 or 2 -- with ~10^5 windows most trigrams
occur once -- so held-out R2 would fall with n from sparsity alone and bias every
neuron toward looking local. The truncated forward has no estimation problem: it is
the neuron's own response to exactly that much context. The n=1 lookup table is
reported too (r2_token_table: held-out, each token's mean shrunk toward the global
mean) as the purest "token identity" score, fit on its own, much larger set of
windows (--table-texts, every position) because even n=1 is starved at the curve's
sample size: on 150 windows it read 0.16 at layer 0 where the BOS + 1 token forward
read 0.69. Treat it as a lower bound; rare tokens fall back to the global mean.

CAVEATS
  - Truncation also moves the token from absolute position p to position n. Llama's
    RoPE is relative, so attention between the kept tokens is unchanged; what changes
    is the distance to BOS, an attention sink. BOS is always kept for that reason.
  - corr^2 forgives an affine shift between truncated and full activations (a
    calibration a reader could learn); it does not forgive reordering.
  - Positions are restricted to p >= n_max so every n is available at every sample;
    the full context is then n_max..31 tokens.

Neurons default to the SFT layer x token grid's (the 10-per-layer sweep table), plus
--n-random random neurons per layer as a reference distribution. One forward serves
every layer, so all layers are scored in one job.

Outputs (--output-dir, default outputs/plots/mlp_endo_sweep_layer_token/context_dependence/):
  context_dependence.csv          per neuron: n_eff, r2_1, r2_nmax, r2_token_table,
                                  std, frac_positive, kurtosis, is_grid
  context_dependence_curves.csv   per (neuron, n): r2
  context_dependence_curves.png, context_dependence_by_layer.png
  context_dependence_meta.json

Usage
  python mechanistic_analysis/score_neuron_context_dependence.py --dataset-dir <cache dir> \
      [--n-texts 3000 --n-max 16 --n-random 50]
"""
from __future__ import annotations

import sys
import json
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM
from scipy.stats import kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # repo root

SFT_TABLES = [ROOT / "outputs/plots/mlp_endo_sweep_layer_token_n10/sweep_runs_both_gates.csv",
              ROOT / "outputs/plots/mlp_endo_sweep_layer_token/sweep_runs_both_gates.csv"]
CONFIG_GLOB = "experiment_configs/mlp_endo_sweep_layer_token/endo_N*_POS-1_gateoff_*.json"
NEURON_RE = re.compile(r"N(\d+)\.(\d+)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, required=True,
                   help="A cached dataset dir with train/ (only text_input_ids are used).")
    p.add_argument("--split", default="train")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--neurons", default=None,
                   help="Comma list like N0.1234,N2.24 (default: the SFT grid's neurons).")
    p.add_argument("--layers", default=None,
                   help="Restrict to these layers (comma list).")
    p.add_argument("--n-random", type=int, default=50,
                   help="Random reference neurons per layer.")
    p.add_argument("--n-texts", type=int, default=3000)
    p.add_argument("--table-texts", type=int, default=30000,
                   help="Windows for the token lookup table (full-context passes only, "
                        "every position; split into fit/test halves by window).")
    p.add_argument("--n-max", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--shrink", type=float, default=5.0,
                   help="Pseudo-count shrinking each token's mean toward the global mean "
                        "in the token lookup table.")
    p.add_argument("--fp32", action="store_true", help="Run the forward in fp32.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "outputs/plots/mlp_endo_sweep_layer_token/context_dependence")
    return p.parse_args()


def grid_neurons(spec: str | None) -> list[tuple[int, int]]:
    if spec:
        return sorted({(int(a), int(b)) for a, b in NEURON_RE.findall(spec)})
    for t in SFT_TABLES:
        if t.exists():
            d = pd.read_csv(t, usecols=["neuron"])
            out = sorted({(int(a), int(b)) for a, b in NEURON_RE.findall(" ".join(d["neuron"].unique()))})
            print(f"[ctx] grid neurons from {t.relative_to(ROOT)}: {len(out)}")
            return out
    names = " ".join(p.name for p in ROOT.glob(CONFIG_GLOB))
    out = sorted({(int(a), int(b)) for a, b in NEURON_RE.findall(names)})
    print(f"[ctx] grid neurons from configs ({CONFIG_GLOB}): {len(out)}")
    return out


def load_windows(dataset_dir: Path, split: str, n_texts: int, seed: int):
    """Unique cached token windows (the cache holds one row per text x neuron)."""
    d = load_from_disk(str(dataset_dir / split)).select_columns(["text_input_ids"])
    # Only full-length windows: a few documents are shorter than the window, and mixing
    # lengths would either need padding or cut every window to the shortest.
    full_len = max(len(x) for x in d.select(range(min(len(d), 2000)))["text_input_ids"])
    seen, out, short = set(), [], 0
    for ids in d["text_input_ids"]:
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        if len(ids) < full_len:
            short += 1
            continue
        out.append(list(ids))
        if len(out) >= n_texts:
            break
    print(f"[ctx] {len(out)} unique {full_len}-token windows from {dataset_dir.name}/{split} "
          f"({short} shorter windows skipped)")
    return out


class Capture:
    """act_fn output of several layers, only the selected neurons, at selected positions."""

    def __init__(self, model, layer_idx: dict[int, torch.Tensor]):
        self.layer_idx, self.out, self.pos, self.handles = layer_idx, {}, None, []
        for L, idx in layer_idx.items():
            mod = model.model.layers[L].mlp.act_fn
            self.handles.append(mod.register_forward_hook(self._hook(L, idx)))

    def _hook(self, L, idx):
        def f(_m, _i, out):
            sel = out[:, :, idx] if self.pos is None else out[torch.arange(out.shape[0]), self.pos][:, idx]
            self.out[L] = sel.float().cpu()
        return f

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run(model, cap, batches, device):
    outs = {L: [] for L in cap.layer_idx}
    for ids, p in batches:
        cap.pos = None if p is None else p.to(device)
        model(input_ids=ids.to(device))
        for L in outs:
            outs[L].append(cap.out[L])
    cap.pos = None
    return {L: torch.cat(v).numpy() for L, v in outs.items()}


def token_table_r2(tokens, acts, shrink, fit):
    """Held-out R2 of a per-token mean table (fit where `fit`, scored on the rest),
    each token's mean shrunk toward the global mean by `shrink` pseudo-counts so rare
    tokens do not overfit. acts: [S, K]."""
    out = np.full(acts.shape[1], np.nan)
    df = pd.DataFrame(acts[fit])
    df["tok"] = tokens[fit]
    g = df.groupby("tok")
    sums, counts = g.sum(), g.size()
    mu = acts[fit].mean(0)
    table = (sums.values + shrink * mu) / (counts.values[:, None] + shrink)
    lookup = pd.DataFrame(table, index=sums.index)
    te_tok = tokens[~fit]
    pred = lookup.reindex(te_tok).values
    pred = np.where(np.isnan(pred), mu, pred)
    y = acts[~fit]
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1 - ss_res / np.maximum(ss_tot, 1e-20)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    neurons = grid_neurons(args.neurons)
    if args.layers:
        keep = {int(x) for x in args.layers.split(",")}
        neurons = [n for n in neurons if n[0] in keep]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32 if args.fp32 else torch.bfloat16,
        device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    n_inter = model.config.intermediate_size
    layers = sorted({L for L, _ in neurons})
    grid = {L: sorted({i for l2, i in neurons if l2 == L}) for L in layers}
    rand = {L: sorted(rng.choice(np.setdiff1d(np.arange(n_inter), grid[L]),
                                 size=args.n_random, replace=False).tolist())
            for L in layers}
    cols = {L: grid[L] + rand[L] for L in layers}
    layer_idx = {L: torch.as_tensor(cols[L], device=device) for L in layers}
    print(f"[ctx] layers {layers}: {sum(len(v) for v in grid.values())} grid + "
          f"{args.n_random}/layer random neurons; n = 1..{args.n_max}")

    windows = load_windows(args.dataset_dir, args.split, args.n_texts, args.seed)
    T = min(len(w) for w in windows)
    windows = np.array([w[:T] for w in windows])               # [N, T], col 0 = BOS
    bos = int(windows[0, 0])
    if not (windows[:, 0] == bos).all():
        raise SystemExit("windows do not all start with the same BOS token")
    positions = np.arange(args.n_max, T)                        # p >= n_max
    if len(positions) == 0:
        raise SystemExit(f"windows of {T} tokens leave no position with n_max={args.n_max}")
    samples = [(t, p) for t in range(len(windows)) for p in positions]
    S = len(samples)
    print(f"[ctx] {S} samples ({len(windows)} windows x positions {positions[0]}..{positions[-1]})")

    cap = Capture(model, layer_idx)
    try:
        # ---- full context: one pass per window, read every sampled position ----
        bs = max(1, args.batch_size // 4)
        full_batches = [(torch.as_tensor(windows[s:s + bs]), None)
                        for s in range(0, len(windows), bs)]
        full_all = run(model, cap, full_batches, device)        # L -> [N, T, K]
        full = {L: v[:, positions].reshape(S, -1) for L, v in full_all.items()}
        # ---- truncated: BOS + the n tokens ending at p; all length n+1, no padding ----
        trunc = {}
        t_idx = np.array([t for t, _ in samples])
        p_idx = np.array([p for _, p in samples])
        for n in range(1, args.n_max + 1):
            seqs = np.stack([windows[t_idx, p_idx - n + 1 + k] for k in range(n)], 1)
            seqs = np.concatenate([np.full((S, 1), bos), seqs], 1)
            last = torch.full((args.batch_size,), n)
            batches = [(torch.as_tensor(seqs[s:s + args.batch_size]),
                        last[:min(args.batch_size, S - s)])
                       for s in range(0, S, args.batch_size)]
            trunc[n] = run(model, cap, batches, device)
            if n in (1, 2, 4, 8, 16) or n == args.n_max:
                print(f"[ctx] n={n} done")
        # ---- token lookup table: many more windows, every position, split by window ----
        tw = np.array([w[:T] for w in load_windows(args.dataset_dir, args.split,
                                                   args.table_texts, args.seed)])
        tb = [(torch.as_tensor(tw[s:s + bs]), None) for s in range(0, len(tw), bs)]
        tab_all = run(model, cap, tb, device)                       # L -> [M, T, K]
        tab_tokens = tw[:, 1:].reshape(-1)
        tab_fit = np.repeat(rng.random(len(tw)) < 0.5, T - 1)
        print(f"[ctx] token table: {len(tw)} windows, {tab_tokens.size} samples, "
              f"{len(np.unique(tab_tokens[tab_fit]))} distinct fit tokens")
    finally:
        cap.remove()

    # sanity: at n = p (every token since BOS) the truncated input IS the full input
    p0 = positions[0]
    if args.n_max == p0:
        m = p_idx == p0
        L0 = layers[0]
        err = np.abs(trunc[args.n_max][L0][m] - full[L0][m]).max()
        print(f"[ctx] identity check (n = p = {p0}, the whole window): max |diff| = {err:.3g}"
              + ("" if err < 0.05 * np.abs(full[L0][m]).max() else "  *** MISMATCH ***"))

    rows, curves = [], []
    for L in layers:
        a_full = full[L]
        r2 = np.stack([np.array([np.corrcoef(trunc[n][L][:, k], a_full[:, k])[0, 1] ** 2
                                 for k in range(a_full.shape[1])])
                       for n in range(1, args.n_max + 1)])          # [n_max, K]
        r2 = np.nan_to_num(r2, nan=0.0)
        n_eff = 1 + (1 - r2).sum(0)
        tt = token_table_r2(tab_tokens, tab_all[L][:, 1:].reshape(len(tab_tokens), -1),
                            args.shrink, tab_fit)
        for k, i in enumerate(cols[L]):
            a = a_full[:, k]
            rows.append(dict(neuron=f"N{L}.{i}", layer=L, index=i, is_grid=i in grid[L],
                             n_eff=n_eff[k], r2_1=r2[0, k], r2_nmax=r2[-1, k],
                             r2_token_table=tt[k], mean=a.mean(), std=a.std(),
                             frac_positive=(a > 0).mean(), kurtosis=kurtosis(a)))
            for n in range(1, args.n_max + 1):
                curves.append(dict(neuron=f"N{L}.{i}", layer=L, index=i,
                                   is_grid=i in grid[L], n=n, r2=r2[n - 1, k]))
    df, cv = pd.DataFrame(rows), pd.DataFrame(curves)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "context_dependence.csv", index=False)
    cv.to_csv(args.output_dir / "context_dependence_curves.csv", index=False)
    (args.output_dir / "context_dependence_meta.json").write_text(json.dumps(dict(
        dataset_dir=str(args.dataset_dir), split=args.split, model=args.model,
        n_windows=len(windows), window_len=T, positions=[int(positions[0]), int(positions[-1])],
        n_samples=S, n_table_windows=int(len(tw)), n_max=args.n_max, n_random=args.n_random, shrink=args.shrink,
        fp32=args.fp32), indent=2))
    plot(df, cv, args.output_dir, args.n_max)
    print(df[df.is_grid].groupby("layer")[["n_eff", "r2_1", "r2_nmax", "r2_token_table"]]
          .mean().round(3).to_string())
    print(f"[ctx] wrote {args.output_dir}")


def plot(df, cv, out_dir, n_max):
    layers = sorted(df["layer"].unique())
    cols = plt.cm.viridis(np.linspace(0, 0.95, len(layers)))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for L, c in zip(layers, cols):
        for grid, ls in ((True, "-"), (False, ":")):
            m = cv[(cv["layer"] == L) & (cv["is_grid"] == grid)].groupby("n")["r2"].mean()
            ax.plot(m.index, m.values, ls=ls, color=c, lw=2 if grid else 1.2,
                    label=f"L{L}" if grid else None)
    ax.plot([], [], "k-", label="Grid Neurons")
    ax.plot([], [], "k:", label="Random Neurons")
    ax.set_xlabel("Context Tokens Kept (n)", fontsize=15)
    ax.set_ylabel("R² (Truncated vs Full Context)", fontsize=15)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "context_dependence_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, lab in ((axes[0], "n_eff", f"Effective Context Length n_eff (max {n_max + 1})"),
                         (axes[1], "r2_token_table", "Token-Identity R² (Lookup Table)")):
        data = [df[(df["layer"] == L) & ~df["is_grid"]][col].values for L in layers]
        ax.violinplot(data, positions=range(len(layers)), showmedians=True, widths=0.8)
        for k, L in enumerate(layers):
            g = df[(df["layer"] == L) & df["is_grid"]]
            ax.scatter(np.full(len(g), k) + np.random.default_rng(k).uniform(-.08, .08, len(g)),
                       g[col], color="tab:red", s=22, zorder=3,
                       label="Grid Neurons" if k == 0 else None)
        ax.set_xticks(range(len(layers)), [f"L{L}" for L in layers])
        ax.set_ylabel(lab, fontsize=14)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("Violins: Random Neurons of the Layer", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "context_dependence_by_layer.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
