#!/usr/bin/env python3
"""
Where does the introspection adapter put its weight, and does it target the neuron?

Runs the per-checkpoint view from plot_lora_weight_pattern.py on several conditions,
then adds the two comparative analyses that single-checkpoint heatmaps cannot give:

 1 LAYER PROFILE, ALIGNED ON THE SIGNAL LAYER.  Plotting ||dW||_F against absolute
   layer conflates "layer 10 matters" with "the layer being read matters". Re-plotting
   against (layer - signal_layer) tests the repurposing story directly: if the adapter
   works by commandeering units DOWNSTREAM of the signal, the mass should sit at
   positive relative depth, and a layer-10 signal should have less room than a layer-2
   one.

 2 TARGETED COLUMN TEST (the amplify-vs-private-wire question).  The intervention is
   written at layers.L.mlp.act_fn, so neuron n reaches the residual stream ONLY
   through down_proj[:, n]. For that column we report, against the distribution over
   all 8192 columns:
       rel_norm   ||dW[:, n]|| / ||W0[:, n]||     how much the path was changed
       percentile where that sits among all columns
       cosine     cos(dW[:, n], W0[:, n])
   cosine > 0 with a high percentile  => the existing broadcast direction was
   AMPLIFIED (the model made its own signal louder).
   cosine ~ 0 with a high percentile  => a new, near-orthogonal direction was written
   => a private relay wire, i.e. the model built new machinery rather than reading
   what was already there. Those are different scientific claims.
   up_proj/gate_proj row n are included: they set the neuron's own value, so movement
   there means the adapter changed the signal rather than its transmission.

Needs the base model for (2) (--no-base-model skips it and keeps (1)).
"""
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from safetensors import safe_open

import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_lora_weight_pattern import (  # noqa: E402
    LAYER_RE, MODULE_RE, MODULE_ORDER, load_lora_norms, build_matrix,
    normalize_per_column, resolve_safetensors, main as per_checkpoint_main,
)

ROOT = Path(__file__).parents[1]
MLP_MODULES = {"gate_proj", "up_proj", "down_proj"}


def find_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(glob.glob(str(run_dir / "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[1]))
    if not ckpts:
        raise SystemExit(f"No checkpoint under {run_dir} (pruned for disk?). The weight "
                         f"analysis needs adapter_model.safetensors.")
    return Path(ckpts[-1])


def lora_scaling(ckpt: Path) -> float:
    """alpha/r (or alpha/sqrt(r) with rslora). Irrelevant for cosines, but the
    norm RATIO against base weights is meaningless without it."""
    cfg_path = ckpt / "adapter_config.json"
    if not cfg_path.exists():
        return 1.0
    cfg = json.loads(cfg_path.read_text())
    r, alpha = cfg.get("r", 1), cfg.get("lora_alpha", 1)
    return alpha / (np.sqrt(r) if cfg.get("use_rslora") else r)


def load_delta(ckpt: Path, layer: int, module: str) -> torch.Tensor | None:
    """Effective update dW = (B @ A) * scaling for one (layer, module)."""
    sfile = resolve_safetensors(ckpt)
    A = B = None
    with safe_open(str(sfile), framework="pt") as f:
        for key in f.keys():
            lm, mm = LAYER_RE.search(key), MODULE_RE.search(key)
            if not lm or not mm or int(lm.group(1)) != layer or mm.group(1) != module:
                continue
            if "lora_A" in key:
                A = f.get_tensor(key).float()
            elif "lora_B" in key:
                B = f.get_tensor(key).float()
    if A is None or B is None:
        return None
    return (B @ A) * lora_scaling(ckpt)


def targeted_column_stats(ckpt: Path, base_model, layer: int, neuron: int) -> list[dict]:
    """Is the TARGET neuron's slice an outlier among all neurons of that layer?"""
    rows = []
    block = base_model.model.layers[layer].mlp
    for module, take_col in (("down_proj", True), ("up_proj", False), ("gate_proj", False)):
        dW = load_delta(ckpt, layer, module)
        if dW is None:
            continue
        W0 = getattr(block, module).weight.detach().float()
        # down_proj is [hidden, intermediate] -> the neuron is a COLUMN;
        # up/gate_proj are [intermediate, hidden] -> the neuron is a ROW.
        dv, w0v = (dW[:, neuron], W0[:, neuron]) if take_col else (dW[neuron, :], W0[neuron, :])
        dn = dW.norm(dim=0) if take_col else dW.norm(dim=1)
        w0n = W0.norm(dim=0) if take_col else W0.norm(dim=1)
        rel = (dn / w0n.clamp(min=1e-8))
        cos = torch.nn.functional.cosine_similarity(dv, w0v, dim=0).item()
        rows.append({
            "module": module,
            "rel_norm_target": rel[neuron].item(),
            "rel_norm_median_all": rel.median().item(),
            "percentile": float((rel < rel[neuron]).float().mean().item() * 100),
            "cosine_with_base": cos,
        })
    return rows


def profile_frame(ckpt: Path, signal_layer: int, label: str) -> pd.DataFrame:
    """Per-(layer, module) ||dW||_F, normalized within condition, plus relative depth."""
    norms = load_lora_norms(resolve_safetensors(ckpt))
    raw, layers, modules = build_matrix(norms)
    norm_mat = normalize_per_column(raw)
    rows = []
    for i, layer in enumerate(layers):
        for j, module in enumerate(modules):
            rows.append({"condition": label, "layer": layer, "module": module,
                         "signal_layer": signal_layer, "rel_depth": layer - signal_layer,
                         "frob": raw[i, j], "frob_norm": norm_mat[i, j],
                         "family": "MLP" if module in MLP_MODULES else "attn"})
    return pd.DataFrame(rows)


def plot_profiles(df, out_dir, x="rel_depth"):
    """One panel per module; a line per condition. x='rel_depth' aligns conditions on
    their own signal layer, x='layer' keeps absolute depth."""
    modules = [m for m in MODULE_ORDER if m in set(df["module"])]
    ncol = 4
    nrow = int(np.ceil(len(modules) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow), squeeze=False)
    conds = sorted(df["condition"].unique())
    palette = dict(zip(conds, sns.color_palette("tab10", len(conds))))
    for ax, module in zip(axes.flat, modules):
        sub = df[df["module"] == module]
        for cond, s in sub.groupby("condition"):
            s = s.sort_values(x)
            ax.plot(s[x], s["frob_norm"], marker="o", ms=4, label=cond, color=palette[cond])
        ax.axvline(0, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_title(module)
        ax.set_xlabel("layer - signal layer" if x == "rel_depth" else "layer")
        ax.set_ylabel("||dW||_F (normalized)")
        ax.set_ylim(0, 1.05); ax.grid(alpha=0.25)
    for ax in axes.flat[len(modules):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("LoRA update magnitude by module"
                 + (", aligned on the signal layer (0 = where the signal is read)"
                    if x == "rel_depth" else ", absolute layer"), y=1.01)
    fig.tight_layout()
    path = Path(out_dir) / f"lora_profile_by_{x}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[lora] wrote {path}")


def main(conditions, out_dir, per_checkpoint=True, use_base_model=True,
         model_name="meta-llama/Llama-3.2-1B"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base_model = None
    if use_base_model:
        from transformers import AutoModelForCausalLM
        print(f"[lora] loading {model_name} on CPU for the targeted-column comparison ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu")

    profiles, targeted = [], []
    for c in conditions:
        run_dir = ROOT / "outputs/runs" / c["group"] / c["run"]
        ckpt = find_checkpoint(run_dir)
        print(f"\n[lora] === {c['label']} ===\n[lora] {ckpt}")
        if per_checkpoint:
            per_checkpoint_main(checkpoint=str(ckpt), output_dir=str(out_dir / "per_condition"),
                                highlight_layer=c["layer"])
        profiles.append(profile_frame(ckpt, c["layer"], c["label"]))
        if base_model is not None:
            for row in targeted_column_stats(ckpt, base_model, c["layer"], c["neuron"]):
                targeted.append({"condition": c["label"], "layer": c["layer"],
                                 "neuron": c["neuron"], **row})

    df = pd.concat(profiles, ignore_index=True)
    df.to_csv(out_dir / "lora_profiles.csv", index=False)
    plot_profiles(df, out_dir, x="rel_depth")
    plot_profiles(df, out_dir, x="layer")

    if targeted:
        t = pd.DataFrame(targeted)
        t.to_csv(out_dir / "lora_targeted_columns.csv", index=False)
        print("\n[lora] targeted-column test (is the READ neuron special?):")
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("\n[lora] percentile ~50 => the target neuron's slice is unremarkable: the "
              "adapter did NOT single it out, and whatever it does is distributed.")
        print("[lora] high percentile + cosine > 0 => the existing broadcast was amplified.")
        print("[lora] high percentile + cosine ~ 0 => a new near-orthogonal direction: a "
              "private relay wire rather than louder reading.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-base-model", action="store_true",
                   help="Skip the targeted-column test (which needs base weights).")
    p.add_argument("--no-per-checkpoint", action="store_true",
                   help="Skip the per-checkpoint heatmap/lineplot from plot_lora_weight_pattern.")
    args = p.parse_args()

    GROUP = "mlp_joint_prev_token_hyper"
    # The four pre-flight cells at their operating budgets: 0.1 everywhere except the
    # hard cell, which is floor-limited there and only reads at 3.0.
    CONDITIONS = [
        dict(label="L2 POS-1 (kl0.1)",    group=GROUP, layer=2,  neuron=24, pos=-1,
             run="joint_N2.24_POS-1_gateoff_scale10_kl0.1_w0.01"),
        dict(label="L2 POS-28 (kl0.1)",   group=GROUP, layer=2,  neuron=24, pos=-28,
             run="joint_N2.24_POS-28_gateoff_scale10_kl0.1_w0.01"),
        dict(label="L10 POS-1 (kl0.1)",   group=GROUP, layer=10, neuron=24, pos=-1,
             run="joint_N10.24_POS-1_gateoff_scale10_kl0.1_w0.01"),
        dict(label="L10 POS-28 (kl3.0)",  group=GROUP, layer=10, neuron=24, pos=-28,
             run="joint_N10.24_POS-28_gateoff_scale10_kl3.0_w0.01"),
    ]
    main(CONDITIONS, out_dir=ROOT / "outputs/plots/lora_weight_pattern_compare",
         per_checkpoint=not args.no_per_checkpoint, use_base_model=not args.no_base_model)
