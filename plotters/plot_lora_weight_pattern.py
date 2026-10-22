#!/usr/bin/env python3
"""
Visualize the LoRA weight-change pattern across layers x modules.

For each (layer, module) that has a LoRA adapter, compute the Frobenius norm of
the effective update ||dW||_F = ||B @ A||_F, then normalize per module (column)
by that column's max across layers so values land in [0, 1] with the peak layer
= 1.0 (DIT appendix-H style). Produces a heatmap + a line plot (+ CSVs).

Notes
-----
- Per-module normalization cancels the uniform LoRA scaling (alpha/r), so the
  scaling factor does not affect the pattern and is not applied.
- ||dW||_F is a screening view of where the adapter concentrated its weight
  mass. It is NOT functional importance: a small early-layer update can still be
  amplified downstream. Use ablation/attribution for causal claims.

Usage
-----
python plotters/plot_lora_weight_pattern.py \
    --checkpoint outputs/runs/.../checkpoint-94 \
    --highlight-layer 10
"""
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
import matplotlib.pyplot as plt
import seaborn as sns

# Poster-scale fonts: large enough to read from a few feet away.
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "legend.title_fontsize": 20,
    "figure.titlesize": 26,
})

# fixed column order (Llama attn + gated MLP); unexpected modules appended after
MODULE_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LAYER_RE = re.compile(r"layers\.(\d+)\.")
MODULE_RE = re.compile(r"\.((?:q|k|v|o|gate|up|down)_proj)\.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Checkpoint dir (containing adapter_model.safetensors) or the "
                        "adapter_model.safetensors file itself.")
    p.add_argument("--output-dir", type=Path, default="outputs/plots/lora_weight_pattern",
                   help="Where to save plots/CSVs (default: the checkpoint dir).")
    p.add_argument("--highlight-layer", type=int, default=None,
                   help="Optional layer index to mark with a vertical line (e.g. the intervened layer).")
    return p.parse_args()


def resolve_safetensors(ckpt: Path) -> Path:
    if ckpt.is_dir():
        f = ckpt / "adapter_model.safetensors"
        if not f.exists():
            raise SystemExit(f"No adapter_model.safetensors found in {ckpt}")
        return f
    return ckpt


def load_lora_norms(safetensors_path: Path):
    """Return {(layer, module): ||B@A||_F} for every paired LoRA adapter."""
    A, B = {}, {}
    with safe_open(str(safetensors_path), framework="pt") as f:
        for key in f.keys():
            is_a, is_b = "lora_A" in key, "lora_B" in key
            if not (is_a or is_b):
                continue  # skips embed_tokens / lm_head full-weight tensors
            lm, mm = LAYER_RE.search(key), MODULE_RE.search(key)
            if not lm or not mm:
                continue
            k = (int(lm.group(1)), mm.group(1))
            (A if is_a else B)[k] = f.get_tensor(key).float()

    norms = {}
    for k, a in A.items():
        b = B.get(k)
        if b is None:
            continue
        dW = b @ a  # [out, in]; rank <= r so cheap
        norms[k] = torch.linalg.norm(dW).item()  # Frobenius (2D, ord=None)
    return norms


def build_matrix(norms):
    layers = sorted({l for (l, _) in norms})
    present = {m for (_, m) in norms}
    modules = [m for m in MODULE_ORDER if m in present]
    modules += sorted(present - set(modules))  # any unexpected modules last
    mat = np.full((len(layers), len(modules)), np.nan)
    for (l, m), v in norms.items():
        mat[layers.index(l), modules.index(m)] = v
    return mat, layers, modules


def normalize_per_column(mat):
    """Divide each column by its max (NaN-safe) -> [0, 1], peak layer = 1."""
    with np.errstate(invalid="ignore"):
        colmax = np.nanmax(mat, axis=0, keepdims=True)
    colmax = np.where((colmax == 0) | np.isnan(colmax), np.nan, colmax)
    return mat / colmax


def main(
    checkpoint: str,
    output_dir: str,
    highlight_layer: Optional[int] = None,
):
    output_dir = Path(output_dir)
    checkpoint = Path(checkpoint)
    sfile = resolve_safetensors(checkpoint)
    out_dir = output_dir / f'{str(sfile).split("/")[-4]}-{str(sfile).split("/")[-3]}'
    out_dir.mkdir(parents=True, exist_ok=True)

    norms = load_lora_norms(sfile)
    if not norms:
        raise SystemExit("No paired LoRA A/B tensors found in the checkpoint.")
    raw, layers, modules = build_matrix(norms)
    norm_mat = normalize_per_column(raw)

    pd.DataFrame(raw, index=layers, columns=modules).to_csv(out_dir / "lora_frobenius_raw.csv")
    pd.DataFrame(norm_mat, index=layers, columns=modules).to_csv(out_dir / "lora_frobenius_normalized.csv")

    # ---- heatmap ----
    plt.figure(figsize=(1.6 * len(modules) + 3, 0.5 * len(layers) + 3))
    hm = sns.heatmap(norm_mat, xticklabels=modules, yticklabels=layers,
                vmin=0, vmax=1, cmap="viridis", annot=True, fmt=".2f",
                annot_kws={"size": 14},
                cbar_kws={"label": "||dW||_F (per-module normalized)"})
    hm.figure.axes[-1].yaxis.label.set_size(28)  # colorbar label
    hm.figure.axes[-1].tick_params(labelsize=22)  # colorbar ticks
    hm.tick_params(axis="x", labelsize=30)
    hm.tick_params(axis="y", labelsize=30)
    plt.xlabel("module", fontsize=38); plt.ylabel("layer", fontsize=38)
    plt.title("LoRA update magnitude (per-module normalized)", fontsize=40)
    plt.tight_layout()
    plt.savefig(out_dir / "lora_frobenius_heatmap.png", dpi=150)
    plt.close()

    # ---- line plot ----
    # MLP modules get a dashed line + square marker so attn (solid/circle) vs
    # MLP (dashed/square) is separable at a glance.
    mlp_modules = {"gate_proj", "up_proj", "down_proj"}
    plt.figure(figsize=(12, 8))
    for j, m in enumerate(modules):
        is_mlp = m in mlp_modules
        plt.plot(layers, norm_mat[:, j],
                 marker="s" if is_mlp else "o", markersize=9, linewidth=2.5,
                 linestyle="--" if is_mlp else "-",
                 label=f"{m} (MLP)" if is_mlp else f"{m} (attn)")
    if highlight_layer is not None:
        plt.axvline(highlight_layer, color="k", ls="--", alpha=0.6, linewidth=2.5,
                    label=f"layer {highlight_layer}")
    plt.xlabel("layer"); plt.ylabel("||dW||_F (per-module normalized)")
    plt.title("LoRA update magnitude across layers")
    plt.ylim(0, 1.02)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "lora_frobenius_lineplot.png", dpi=150)
    plt.close()

    print(f"Saved heatmap, lineplot, and CSVs to {out_dir}")
    print(pd.DataFrame(norm_mat, index=layers, columns=modules).round(2).to_string())


if __name__ == "__main__":
    main(**vars(parse_args()))