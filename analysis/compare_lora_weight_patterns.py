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

   RANDOM-INDEX BASELINE. A percentile is only interesting against the null it is
   being compared to, and "99.99th percentile" is a statement about one number in one
   run. --n-random untrained indices of the SAME layer in the SAME checkpoint are
   measured identically, which turns every claim into a contrast: the baseline's
   percentile is uniform by construction (~50 on average) and its cosine says whether
   near-orthogonality is special to the target or just what a rank-16 update looks
   like against an 8192-column matrix. Without it, cosine ~ 0 at the target is
   uninterpretable.

Needs the base model for (2) (--no-base-model skips it and keeps (1)).

Usage
-----
Four pre-flight conditions (the default, unchanged):
    python analysis/compare_lora_weight_patterns.py

A whole sweep, aggregated per (layer, position) cell with neurons as the error bar:
    python analysis/compare_lora_weight_patterns.py \
        --group mlp_joint_grid_layer_token --run-glob 'joint_*kl0.1*'
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
sys.path.insert(0, str(Path(__file__).parents[1]))
from plot_lora_weight_pattern import (  # noqa: E402
    LAYER_RE, MODULE_RE, MODULE_ORDER, load_lora_norms, build_matrix,
    normalize_per_column, resolve_safetensors, main as per_checkpoint_main,
)
from src.introspector.run_discovery import (  # noqa: E402
    add_discovery_args, runs_from_args, find_checkpoint as _find_latest_checkpoint,
)

ROOT = Path(__file__).parents[1]
MLP_MODULES = {"gate_proj", "up_proj", "down_proj"}
# Untrained indices of the target's own layer, measured exactly like the target.
DEFAULT_N_RANDOM = 20


def find_checkpoint(run_dir: Path) -> Path:
    ckpt = _find_latest_checkpoint(Path(run_dir), required=False)
    if ckpt is None:
        raise SystemExit(f"No checkpoint under {run_dir} (pruned for disk?). The weight "
                         f"analysis needs adapter_model.safetensors.")
    return ckpt


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


def random_baseline_indices(layer: int, neuron: int, n_indices: int, k: int) -> np.ndarray:
    """`k` indices of this layer that the run never trained on, excluding the target.

    Seeded from (layer, neuron) rather than drawn fresh, so re-running the analysis
    reproduces the same baseline and a sweep's cells are comparable to each other
    instead of each carrying its own draw noise.
    """
    rng = np.random.RandomState(layer * 100003 + neuron)
    pool = np.delete(np.arange(n_indices), neuron)
    return np.sort(rng.choice(pool, size=min(k, len(pool)), replace=False))


def targeted_column_stats(ckpt: Path, base_model, layer: int, neuron: int,
                          n_random: int = DEFAULT_N_RANDOM) -> list[dict]:
    """
    Is the TARGET neuron's slice an outlier among all neurons of that layer?

    One row per (module, index), long format: the target plus `n_random` untrained
    indices of the same layer, measured identically (`kind` distinguishes them). Long
    format because the baseline is a DISTRIBUTION -- summarizing it to a mean inside
    this function would throw away the spread the target has to stand out against.
    """
    rows = []
    block = base_model.model.layers[layer].mlp
    for module, take_col in (("down_proj", True), ("up_proj", False), ("gate_proj", False)):
        dW = load_delta(ckpt, layer, module)
        if dW is None:
            continue
        W0 = getattr(block, module).weight.detach().float()
        # down_proj is [hidden, intermediate] -> the neuron is a COLUMN;
        # up/gate_proj are [intermediate, hidden] -> the neuron is a ROW.
        dn = dW.norm(dim=0) if take_col else dW.norm(dim=1)
        w0n = W0.norm(dim=0) if take_col else W0.norm(dim=1)
        rel = (dn / w0n.clamp(min=1e-8))
        rel_median = rel.median().item()
        n_indices = rel.shape[0]

        def one(idx: int, kind: str) -> dict:
            dv, w0v = (dW[:, idx], W0[:, idx]) if take_col else (dW[idx, :], W0[idx, :])
            return {
                "module": module, "kind": kind, "signal_index_measured": int(idx),
                "rel_norm_target": rel[idx].item(),
                "rel_norm_median_all": rel_median,
                "percentile": float((rel < rel[idx]).float().mean().item() * 100),
                "cosine_with_base": torch.nn.functional.cosine_similarity(dv, w0v, dim=0).item(),
            }

        rows.append(one(neuron, "target"))
        for idx in random_baseline_indices(layer, neuron, n_indices, n_random):
            rows.append(one(int(idx), "random"))
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


# ======================================================================
# Sweep aggregation (many runs -- one line per condition is unreadable)
# ======================================================================
# Formal names for figures (the CSVs keep the module keys)
MODULE_TITLE = {"q_proj": "Attention Q Proj.", "k_proj": "Attention K Proj.",
                "v_proj": "Attention V Proj.", "o_proj": "Attention O Proj.",
                "gate_proj": "MLP Gate Proj.", "up_proj": "MLP Up Proj.",
                "down_proj": "MLP Down Proj."}
SEQ_LEN = 32      # training window: positions are shown as absolute 0-indexed indices
GROUP_TITLE = {"signal_layer": "Signal Layer", "position": "Token Position"}


def _group_label(group_col, g):
    """Legend entry for one group value: plain numbers, positions made absolute."""
    if group_col == "position" and float(g) < 0:
        g = float(g) + SEQ_LEN
    return f"{int(g)}" if float(g).is_integer() else f"{g}"


def _value_label(value):
    return r"$\|\Delta W\|_F$" + (" (Normalized)" if value == "frob_norm" else "")


def plot_profiles_grouped(df, out_dir, group_col="signal_layer", x="rel_depth",
                          value="frob_norm"):
    """Layer profiles averaged over a sweep, one line per `group_col` value, every module.

    The per-condition version draws a line per run, which at 320 runs is a solid
    block of colour. Averaging over the OTHER coordinates (neurons always; positions
    when grouping by signal layer, and vice versa) keeps the comparison that the
    grouping is about, with a +/- 1 sd band for the spread it hides. Formatted like
    plot_profiles_compact (the paper version): formal module names, no suptitle.
    """
    modules = [m for m in MODULE_ORDER if m in set(df["module"])]
    ncol, nrow = 4, int(np.ceil(len(modules) / 4))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.4 * nrow), squeeze=False)
    groups = sorted(df[group_col].dropna().unique())
    palette = dict(zip(groups, sns.color_palette("viridis", len(groups))))
    for k, (ax, module) in enumerate(zip(axes.flat, modules)):
        sub = df[df["module"] == module]
        for g in groups:
            s = (sub[sub[group_col] == g].groupby(x)[value]
                 .agg(["mean", "std", "count"]).reset_index().sort_values(x))
            ax.plot(s[x], s["mean"], marker="o", ms=4, lw=1.8, label=_group_label(group_col, g),
                    color=palette[g])
            ax.fill_between(s[x], s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0),
                            color=palette[g], alpha=0.12)
        ax.axvline(0, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_title(MODULE_TITLE.get(module, module), fontsize=18)
        ax.set_xlabel("Layer \u2212 Signal Layer" if x == "rel_depth" else "Layer", fontsize=16)
        if k % ncol == 0:                                    # left column only
            ax.set_ylabel(_value_label(value), fontsize=16)
        if value == "frob_norm":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_yscale("log")
        ax.tick_params(which="both", labelsize=13)   # log axes label minor ticks too
        ax.grid(alpha=0.25)
    # Park the legend in the first empty grid slot (7 modules -> the 8th cell is free),
    # so it never covers a panel; fall back to the first panel if the grid is full.
    empties = list(axes.flat[len(modules):])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    title = GROUP_TITLE.get(group_col, group_col.replace("_", " ").title())
    if empties and handles:
        leg_ax, *rest = empties
        leg_ax.axis("off")
        leg_ax.legend(handles, labels, loc="center", fontsize=14, title=title,
                      title_fontsize=15, frameon=False)
        for ax in rest:
            ax.set_visible(False)
    else:
        for ax in empties:
            ax.set_visible(False)
        if handles:
            axes.flat[0].legend(fontsize=13, title=title)
    fig.tight_layout()
    suffix = "" if value == "frob_norm" else "_abs"
    path = Path(out_dir) / f"lora_profile{suffix}_by_{x}_grouped_{group_col}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[lora] wrote {path}")


COMPACT_MODULES = {m: MODULE_TITLE[m] for m in ("q_proj", "down_proj")}


def plot_profiles_compact(df, out_dir, group_col="signal_layer", x="rel_depth",
                          value="frob"):
    """Paper main-text version of plot_profiles_grouped: only q_proj and down_proj on
    one row, a separate legend column, no suptitle, and formal labels throughout."""
    modules = [m for m in COMPACT_MODULES if m in set(df["module"])]
    if not modules:
        return
    groups = sorted(df[group_col].dropna().unique())
    palette = dict(zip(groups, sns.color_palette("viridis", len(groups))))
    group_name = group_col.replace("_", " ").title()
    fig, axes = plt.subplots(1, len(modules), figsize=(5.2 * len(modules), 4.6), squeeze=False)
    axes = axes[0]
    for i, (ax, module) in enumerate(zip(axes, modules)):
        sub = df[df["module"] == module]
        for g in groups:
            s = (sub[sub[group_col] == g].groupby(x)[value]
                 .agg(["mean", "std"]).reset_index().sort_values(x))
            label = f"{int(g)}" if float(g).is_integer() else f"{g}"
            ax.plot(s[x], s["mean"], marker="o", ms=4, lw=1.8, label=label, color=palette[g])
            ax.fill_between(s[x], s["mean"] - s["std"].fillna(0), s["mean"] + s["std"].fillna(0),
                            color=palette[g], alpha=0.12)
        ax.axvline(0, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_title(COMPACT_MODULES[module], fontsize=20)
        ax.set_xlabel("Layer − Signal Layer" if x == "rel_depth" else "Layer", fontsize=19)
        if i == 0:
            ax.set_ylabel(r"$\|\Delta W\|_F$" + (" (Normalized)" if value == "frob_norm" else ""),
                          fontsize=19)
        if value == "frob_norm":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_yscale("log")
        # which="both": on a log axis the in-between labels (2x10^-1, 3x10^-1, ...) are
        # MINOR ticks, and a major-only setting leaves them at the default size
        ax.tick_params(which="both", labelsize=15)
        ax.grid(alpha=0.25)
    # legend hugging the last panel, rather than in its own blank column
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                    fontsize=14, title=group_name, title_fontsize=15, frameon=False,
                    borderaxespad=0.0)
    fig.tight_layout()
    suffix = "" if value == "frob_norm" else "_abs"
    path = Path(out_dir) / f"lora_profile{suffix}_by_{x}_grouped_{group_col}_compact.png"
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[lora] wrote {path}")


def aggregate_targeted(t: pd.DataFrame, out_dir: Path):
    """
    Per (layer, position, module): the target's numbers against the random-index
    baseline drawn from the same checkpoint, with neurons as the error bar.

    `rel_norm_ratio` (target / baseline mean) is the effect size to quote: percentiles
    saturate at 100 and stop discriminating once every cell is an outlier, while the
    ratio keeps separating "10x the typical column" from "300x".
    """
    keys = ["layer", "position", "module"]
    tgt = t[t["kind"] == "target"]
    rnd = t[t["kind"] == "random"]

    # baseline per (cell, module, RUN) first, so a run with a fat baseline draw does
    # not drag the cell mean -- then over runs (= neurons) within the cell
    rnd_run = (rnd.groupby(keys + ["condition"])
                 .agg(rel_norm_rand=("rel_norm_target", "mean"),
                      cosine_rand=("cosine_with_base", "mean"),
                      abs_cosine_rand=("cosine_with_base", lambda v: v.abs().mean()))
                 .reset_index())
    tgt_run = tgt[keys + ["condition", "rel_norm_target", "percentile", "cosine_with_base",
                          "rel_norm_median_all"]].copy()
    per_run = tgt_run.merge(rnd_run, on=keys + ["condition"], how="left")
    per_run["rel_norm_ratio"] = per_run["rel_norm_target"] / per_run["rel_norm_rand"].clip(lower=1e-12)
    per_run.to_csv(out_dir / "lora_targeted_per_run.csv", index=False)

    cell = (per_run.groupby(keys)
            .agg(n_runs=("condition", "nunique"),
                 percentile_mean=("percentile", "mean"), percentile_std=("percentile", "std"),
                 rel_norm_target_mean=("rel_norm_target", "mean"),
                 rel_norm_target_std=("rel_norm_target", "std"),
                 rel_norm_rand_mean=("rel_norm_rand", "mean"),
                 rel_norm_ratio_mean=("rel_norm_ratio", "mean"),
                 rel_norm_ratio_std=("rel_norm_ratio", "std"),
                 cosine_mean=("cosine_with_base", "mean"), cosine_std=("cosine_with_base", "std"),
                 cosine_rand_mean=("cosine_rand", "mean"))
            .reset_index())
    cell.to_csv(out_dir / "lora_targeted_by_cell.csv", index=False)
    return per_run, cell


def plot_targeted_heatmaps(cell: pd.DataFrame, out_dir: Path):
    """(signal layer x token position) maps of how far the target column stands out.

    Module panels share one colour scale and one colorbar (far right), so a colour
    means the same thing across down/up/gate_proj and the panels stay comparable."""
    for value, cbar_label, title, kw in (
        ("percentile_mean", "Percentile",
         "Neuron Index Relative Norm Percentile (Across Other Neurons)",
         dict(vmin=0, vmax=100, cmap="viridis")),
        ("rel_norm_ratio_mean", "Rel. Norm Ratio",
         "Neuron Index Relative Norm Ratio (Target / Random)",
         dict(cmap="magma")),
    ):
        modules = [m for m in ("down_proj", "up_proj", "gate_proj") if m in set(cell["module"])]
        if not modules or cell["position"].isna().all():
            continue
        kw = dict(kw)
        if "vmin" not in kw:                       # shared scale across the module panels
            vals = cell.loc[cell["module"].isin(modules), value].to_numpy(dtype=float)
            kw["vmin"], kw["vmax"] = float(np.nanmin(vals)), float(np.nanmax(vals))

        fig, axes = plt.subplots(1, len(modules), figsize=(5.2 * len(modules) + 1.4, 5.2),
                                 squeeze=False, sharey=True, constrained_layout=True)
        for i, (ax, module) in enumerate(zip(axes.flat, modules)):
            piv = (cell[cell["module"] == module]
                   .pivot(index="layer", columns="position", values=value).sort_index())
            annot = piv.round(1) if value == "percentile_mean" else piv.round(0)
            sns.heatmap(piv, ax=ax, annot=annot, fmt="g", annot_kws={"size": 10},
                        cbar=False, yticklabels=(i == 0), **kw)
            ax.set_title(module, fontsize=14)
            ax.set_xlabel("Token Position", fontsize=13)
            ax.set_ylabel("Signal Layer" if i == 0 else "", fontsize=13)
            ax.tick_params(axis="y", rotation=0, labelsize=11)   # horizontal -> no overlap
            ax.tick_params(axis="x", labelsize=11)
        # one shared colorbar to the right of the last panel
        cbar = fig.colorbar(axes.flat[len(modules) - 1].collections[0],
                            ax=axes.ravel().tolist(), fraction=0.046, pad=0.02)
        cbar.set_label(cbar_label, fontsize=13)
        cbar.ax.tick_params(labelsize=11)
        fig.suptitle(title, fontsize=16)
        path = out_dir / f"lora_targeted_{value}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"[lora] wrote {path}")


def plot_target_vs_random(t: pd.DataFrame, out_dir: Path):
    """Pooled target-vs-random distributions: the claim the percentile is shorthand for.

    rel_norm on a log axis (the target is orders of magnitude out, not a few percent);
    cosine on a linear one, where the question is whether ~0 is special -- if the random
    columns are also ~0, near-orthogonality says nothing about the target and only the
    MAGNITUDE is evidence of a private wire.
    """
    modules = [m for m in ("down_proj", "up_proj", "gate_proj") if m in set(t["module"])]
    if not modules:
        return
    # one title per ROW (what is measured) and per COLUMN (which module), instead of
    # repeating "<module>: <quantity>" on every panel
    fig = plt.figure(figsize=(5.6 * len(modules), 10), layout="constrained")
    rows = fig.subfigures(2, 1, hspace=0.06)
    kinds = (("target", "tab:red", "Target Neuron"), ("random", "tab:grey", "Random Neurons"))
    specs = (("Update Size", "Relative Update Norm (log$_{10}$)", "rel_norm_target", True),
             ("Update vs. Base Direction", "Cosine Similarity to Base Weight", "cosine_with_base", False))
    for r, (row_title, xlabel, col, log) in enumerate(specs):
        rows[r].suptitle(row_title, fontsize=22)
        axes = rows[r].subplots(1, len(modules))
        for j, (ax, module) in enumerate(zip(np.atleast_1d(axes), modules)):
            sub = t[t["module"] == module]
            for kind, color, label in kinds[::-1]:           # random underneath
                v = sub.loc[sub["kind"] == kind, col]
                if log:
                    v = np.log10(v[v > 0])
                if len(v):
                    ax.hist(v, bins=40, alpha=0.6, color=color, label=label, density=True)
            if not log:
                ax.axvline(0, color="k", ls="--", lw=1)
            if r == 0:
                ax.set_title(MODULE_TITLE.get(module, module), fontsize=20)
            ax.set_xlabel(xlabel, fontsize=18)
            ax.set_ylabel("Density" if j == 0 else "", fontsize=18)
            ax.tick_params(labelsize=15)
            ax.grid(alpha=0.2)
            if r == 0 and j == len(modules) - 1:           # one legend for the figure
                h, l = ax.get_legend_handles_labels()
                ax.legend(h[::-1], l[::-1], fontsize=15, loc="upper left")
    path = out_dir / "lora_target_vs_random.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[lora] wrote {path}")


def plot_profiles(df, out_dir, x="rel_depth", value="frob_norm"):
    """One panel per module; a line per condition.

    x='rel_depth' aligns conditions on their own signal layer, x='layer' keeps absolute
    depth. value='frob_norm' normalizes each condition by ITS OWN max, so shapes are
    comparable but LEVELS ARE NOT -- conditions trained at different KL budgets have
    very different absolute update sizes. Use value='frob' (log scale) for any claim
    of the form "condition A moved more weight than condition B"."""
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
            ax.plot(s[x], s[value], marker="o", ms=4, label=cond, color=palette[cond])
        ax.axvline(0, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_title(module)
        ax.set_xlabel("Layer - Signal Layer" if x == "rel_depth" else "Layer")
        if value == "frob_norm":
            ax.set_ylabel("||dW||_F (normalized)")
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylabel("||dW||_F (absolute)")
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
    # Legend in the first empty grid slot (see plot_profiles_grouped), not over a panel.
    empties = list(axes.flat[len(modules):])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if empties and handles:
        leg_ax, *rest = empties
        leg_ax.axis("off")
        leg_ax.legend(handles, labels, loc="center", fontsize=12, title="Condition",
                      frameon=False)
        for ax in rest:
            ax.set_visible(False)
    else:
        for ax in empties:
            ax.set_visible(False)
        if handles:
            axes.flat[0].legend(fontsize=8)
    scale = "per-condition normalized" if value == "frob_norm" else "ABSOLUTE (log)"
    fig.suptitle(f"LoRA update magnitude by module [{scale}]"
                 + (", aligned on the signal layer (0 = where the signal is read)"
                    if x == "rel_depth" else ", absolute layer"), y=1.01)
    fig.tight_layout()
    suffix = "" if value == "frob_norm" else "_abs"
    path = Path(out_dir) / f"lora_profile{suffix}_by_{x}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[lora] wrote {path}")


def conditions_from_runs(runs) -> list[dict]:
    """RunSpecs -> the condition dicts this script has always taken.

    One condition per TRAINED SIGNAL: a multi-locale run (neuron / layer generalization
    sweeps) edited one down_proj column per signal, and each is its own targeted-column
    test. Reading only the first would report one of N edits as the run's.
    """
    out = []
    for r in runs:
        for stype, layer, index in (r.locales or [(r.signal_type, r.layer, r.index)]):
            label = r.name if not r.is_multi_locale else f"{r.name} [N{layer}.{index}]"
            out.append(dict(label=label, group=r.group, run=r.name, layer=layer,
                            neuron=index, pos=r.position, ckpt=r.checkpoint(required=False)))
    return out


def plot_profiles_from_df(df, out_dir, group_profiles=None):
    """Draw the profile figures from an already-built profiles frame. Shared by main()
    and the --from-cache re-plot path so both stay in lockstep. Returns group_profiles."""
    # A line per condition is the point at 4 conditions and unreadable at 320, so a
    # handful gets per-condition panels and a sweep gets the grouped version instead.
    if group_profiles is None:
        group_profiles = df["condition"].nunique() > 12
    if group_profiles:
        for by in ("signal_layer", "position"):
            if df[by].nunique() > 1:
                plot_profiles_grouped(df, out_dir, group_col=by, x="rel_depth")
                plot_profiles_grouped(df, out_dir, group_col=by, x="rel_depth", value="frob")
                if by == "signal_layer":
                    # paper main-text version of the absolute profile (q_proj + down_proj)
                    plot_profiles_compact(df, out_dir, group_col=by, x="rel_depth", value="frob")
    else:
        for x in ("rel_depth", "layer"):
            plot_profiles(df, out_dir, x=x)
            # absolute (log) version too: required for any "A moved more weight than B"
            # claim, since the normalized panels hide that different KL budgets rescale all.
            plot_profiles(df, out_dir, x=x, value="frob")
    return group_profiles


def plot_targeted_from_t(t, out_dir):
    """Aggregate the long-format targeted table and draw its figures. Shared by main()
    and --from-cache. Returns (per_run, cell)."""
    per_run, cell = aggregate_targeted(t, out_dir)
    plot_target_vs_random(t, out_dir)
    if cell["position"].notna().any() and len(cell) > len(cell["module"].unique()):
        plot_targeted_heatmaps(cell, out_dir)
    return per_run, cell


def replot_from_cache(out_dir, group_profiles=None):
    """Re-draw every figure from the CSVs a previous run cached in out_dir, without
    loading any checkpoint or the base model. Use after tweaking plot styling."""
    out_dir = Path(out_dir)
    prof_csv = out_dir / "lora_profiles.csv"
    if not prof_csv.exists():
        raise SystemExit(f"No cached CSVs in {out_dir} (missing {prof_csv.name}). "
                         f"Run once without --from-cache to generate them.")
    plot_profiles_from_df(pd.read_csv(prof_csv), out_dir, group_profiles)

    t_csv = out_dir / "lora_targeted_columns.csv"
    if t_csv.exists():
        t = pd.read_csv(t_csv)
        if "kind" in t.columns:
            plot_targeted_from_t(t, out_dir)
        else:
            # cached before the random-index baseline existed: no target/random split
            print(f"[lora] {t_csv.name} in {out_dir} predates the random baseline (no 'kind' "
                  "column); skipping targeted-column plots -- rerun without --from-cache")
    else:
        print(f"[lora] no {t_csv.name} in {out_dir}; skipping targeted-column plots "
              "(the first run used --no-base-model).")
    print(f"[lora] re-plotted from cache -> {out_dir}")


def main(conditions, out_dir, per_checkpoint=True, use_base_model=True,
         model_name="meta-llama/Llama-3.2-1B", n_random=DEFAULT_N_RANDOM,
         group_profiles=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base_model = None
    if use_base_model:
        from transformers import AutoModelForCausalLM
        print(f"[lora] loading {model_name} on CPU for the targeted-column comparison ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu")

    profiles, targeted = [], []
    for k, c in enumerate(conditions, 1):
        ckpt = c.get("ckpt") or find_checkpoint(ROOT / "outputs/runs" / c["group"] / c["run"])
        print(f"\n[lora] === [{k}/{len(conditions)}] {c['label']} ===\n[lora] {ckpt}")
        if per_checkpoint:
            per_checkpoint_main(checkpoint=str(ckpt), output_dir=str(out_dir / "per_condition"),
                                highlight_layer=c["layer"])
        profiles.append(profile_frame(ckpt, c["layer"], c["label"])
                        .assign(position=c.get("pos"), neuron=c["neuron"]))
        if base_model is not None:
            for row in targeted_column_stats(ckpt, base_model, c["layer"], c["neuron"],
                                             n_random=n_random):
                targeted.append({"condition": c["label"], "layer": c["layer"],
                                 "neuron": c["neuron"], "position": c.get("pos"), **row})

    df = pd.concat(profiles, ignore_index=True)
    df.to_csv(out_dir / "lora_profiles.csv", index=False)
    plot_profiles_from_df(df, out_dir, group_profiles)

    tot = (df.groupby("condition")["frob"].sum().sort_values(ascending=False)
             .rename("total_||dW||_F").reset_index())
    print("\n[lora] TOTAL update magnitude per condition (absolute, not normalized):")
    print(tot.head(20).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    if targeted:
        t = pd.DataFrame(targeted)
        t.to_csv(out_dir / "lora_targeted_columns.csv", index=False)
        per_run, cell = plot_targeted_from_t(t, out_dir)

        print("\n[lora] targeted-column test (is the READ neuron special?), per run:")
        cols = ["condition", "layer", "position", "module", "rel_norm_target",
                "rel_norm_rand", "rel_norm_ratio", "percentile", "cosine_with_base", "cosine_rand"]
        show = per_run[cols]
        print((show if len(show) <= 40 else show.head(40)).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))
        if len(show) > 40:
            print(f"[lora] ... {len(show) - 40} more rows in lora_targeted_per_run.csv")
            print("\n[lora] pooled over the sweep (mean over runs):")
            print(per_run.groupby("module")[
                ["rel_norm_target", "rel_norm_rand", "rel_norm_ratio", "percentile",
                 "cosine_with_base", "cosine_rand"]].mean().to_string(
                float_format=lambda v: f"{v:.4f}"))
        print("\n[lora] percentile ~50 => the target neuron's slice is unremarkable: the "
              "adapter did NOT single it out, and whatever it does is distributed.")
        print("[lora] high percentile + cosine > 0 => the existing broadcast was amplified.")
        print("[lora] high percentile + cosine ~ 0 => a new near-orthogonal direction: a "
              "private relay wire rather than louder reading.")
        print("[lora] READ THE BASELINE COLUMNS: rel_norm_ratio is the effect size once "
              "percentiles saturate, and cosine_rand says whether ~0 is special at all.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-base-model", action="store_true",
                   help="Skip the targeted-column test (which needs base weights).")
    p.add_argument("--no-per-checkpoint", action="store_true",
                   help="Skip the per-checkpoint heatmap/lineplot from plot_lora_weight_pattern.")
    # p.add_argument("--from-cache", action="store_true",
    #                help="Re-draw every figure from a previous run's cached CSVs in --out-dir "
    #                     "(or the group's default plot dir) -- no checkpoints or base model "
    #                     "loaded. For iterating on plot styling.")
    p.add_argument("--n-random", type=int, default=DEFAULT_N_RANDOM,
                   help="Untrained indices of the signal layer measured alongside the target "
                        "as the baseline distribution (0 disables).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: outputs/plots/lora_weight_pattern_compare, or "
                        "outputs/plots/<group>/lora_weights when sweeping a group.")
    add_discovery_args(p)
    args = p.parse_args()

    GROUP = "mlp_joint_prev_token_hyper"
    if args.from_cache:
        # out_dir is resolved the same way as a real run, so --from-cache pairs with the
        # same --group / --out-dir the original run used.
        if args.out_dir:
            cache_dir = args.out_dir
        elif args.group:
            cache_dir = ROOT / "outputs/plots" / args.group / "lora_weights"
        else:
            cache_dir = ROOT / "outputs/plots/lora_weight_pattern_compare"
        replot_from_cache(cache_dir)
        raise SystemExit(0)

    if args.group or args.config:
        conditions = conditions_from_runs(runs_from_args(args))
        if not conditions:
            raise SystemExit("No runs with checkpoints matched.")
        out_dir = args.out_dir or (ROOT / "outputs/plots" /
                                   (args.group or conditions[0]["group"]) / "lora_weights")
        # per-checkpoint heatmaps are one figure per run: opt in for a sweep
        per_checkpoint = args.no_per_checkpoint is False and len(conditions) <= 12
    else:
        # The four pre-flight cells at their operating budgets: 0.1 everywhere except the
        # hard cell, which is floor-limited there and only reads at 3.0.
        conditions = [
            dict(label="L2 POS-1 (kl0.1)",    group=GROUP, layer=2,  neuron=24, pos=-1,
                 run="joint_N2.24_POS-1_gateoff_scale10_kl0.1_w0.01"),
            dict(label="L2 POS-28 (kl0.1)",   group=GROUP, layer=2,  neuron=24, pos=-28,
                 run="joint_N2.24_POS-28_gateoff_scale10_kl0.1_w0.01"),
            dict(label="L10 POS-1 (kl0.1)",   group=GROUP, layer=10, neuron=24, pos=-1,
                 run="joint_N10.24_POS-1_gateoff_scale10_kl0.1_w0.01"),
            dict(label="L10 POS-28 (kl3.0)",  group=GROUP, layer=10, neuron=24, pos=-28,
                 run="joint_N10.24_POS-28_gateoff_scale10_kl3.0_w0.01"),
        ]
        out_dir = args.out_dir or (ROOT / "outputs/plots/lora_weight_pattern_compare")
        per_checkpoint = not args.no_per_checkpoint

    main(conditions, out_dir=out_dir, per_checkpoint=per_checkpoint,
         use_base_model=not args.no_base_model, n_random=args.n_random)
