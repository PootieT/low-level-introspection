#!/usr/bin/env python3
"""
Per-module LoRA ablation of an interventionally-trained regressor.

For a trained checkpoint, zero each LoRA module (per (layer, module)) one at a
time and measure the change in:
  - interventional@gaussian corr  (scale=1, the causal-reading metric)
  - endogenous corr               (predict the true activation, no intervention)

Reading each cell as Delta = corr(ablated) - corr(baseline):
  * Delta_interventional << 0  -> the module is part of the READING circuit
    (removing it breaks the model's ability to read the injected value).
  * Delta_endogenous     >> 0  -> the module was SUPPRESSING context that helps
    endogenous prediction (a "blocker").

Baseline (nothing ablated) and all-LoRA-off rows are included as references. Each pass
also records corr(prediction under injection, natural activation): the report following
the value the injection replaced (a context path around the site).

GRANULARITY. `module` (default) zeroes one (layer, module) at a time; `block` zeroes a
layer's whole attention LoRA (q,k,v,o) or whole MLP LoRA (gate,up,down) at once -- the
units Fig. 7's norm profile is drawn in, and the causal counterpart of "the update norm
is largest at the signal layer's MLP and the next layer's attention"; `both` does both.
WHICH INJECTION REGIME THE DELTA IS MEASURED IN matters more than it looks. This
used to score the interventional arm at gaussian scale=1 always. For a run trained
with a 10x injection that is the wrong operating point: its own scale sweep puts the
head at ~0.19 there against ~0.98 at scale 10, so an ablation delta is a difference
between two numbers that are mostly noise, and a module that matters can read as
doing nothing. The default is now the regime the run was TRAINED in (--interv-scale /
--interv-distribution to override), where the baseline is near ceiling and a drop is
unambiguous.

Config-driven (model/dataset/signal/checkpoint from --config) so future runs
just swap the config, and --group ablates a whole sweep.

Example
-------
python mechanistic_analysis/ablate_lora_modules.py \
    --config experiment_configs/.../<run>.json \
    --n-examples 512 --layers 7,8,9,10,11,12
# rebuttal: whole test split, x1 gaussian, modules and blocks, 4 cells of the joint grid
python mechanistic_analysis/ablate_lora_modules.py --group mlp_joint_grid_layer_token \
    --run-glob 'joint_*_kl0.1_*' --cells 2:-1,2:-28,8:-1,8:-28 --no-success-filter \
    --n-examples 100000 --interv-distribution gaussian --interv-scale 1 \
    --granularity both --analysis-name lora_ablation_x1 --skip-existing
"""
from __future__ import annotations

import os
import sys
import re
import glob
import json
import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from src.introspector.config import ExperimentConfig
from src.introspector.analysis_loading import load_run_for_analysis, run_per_locale
from src.introspector.run_discovery import (
    add_discovery_args, discover_runs, filter_cells, load_run, resolve_run_action, shard)
from mechanistic_analysis.trace_introspection_circuit import endogenous_predictions, select_success_indices

MODULE_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BLOCKS = {"attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
          "mlp": ("gate_proj", "up_proj", "down_proj")}
LAYER_RE = re.compile(r"layers\.(\d+)\.")
MODULE_RE = re.compile(r"\.((?:q|k|v|o|gate|up|down)_proj)$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Checkpoint dir override (default: latest under output_dir/experiment_name).")
    p.add_argument("--no-success-filter", action="store_true", help="Disable success sample filtering.")
    p.add_argument("--n-examples", type=int, default=200,
                   help="Test examples used per ablation eval (subset for speed).")
    p.add_argument("--success-threshold", type=float, default=0.02,
                   help="Keep examples with |endogenous_pred - true_activation| <= this.")
    p.add_argument("--max-scan", type=int, default=None,
                   help="Max test examples to scan for successes (default: all).")
    p.add_argument("--layers", type=str, default=None,
                   help="Comma-separated layer indices to ablate (default: all).")
    p.add_argument("--modules", type=str, default=None,
                   help="Comma-separated module names to ablate (default: all present).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Eval batch size (default: the run's own eval_batch_size).")
    p.add_argument("--interv-distribution", type=str, default="auto",
                   help="Injection distribution for the interventional arm. 'auto' (default) "
                        "uses the one the run was TRAINED with -- see the module docstring "
                        "for why the old fixed gaussian@1 hid real effects.")
    p.add_argument("--interv-scale", type=float, default=None,
                   help="Injection scale for the interventional arm (default: the run's own).")
    p.add_argument("--granularity", choices=("module", "block", "both"), default="module",
                   help="Zero one (layer, module) at a time, a layer's whole attention / MLP "
                        "LoRA at once, or both (see the module docstring).")
    p.add_argument("--cells", type=str, default=None,
                   help="Comma list of 'layer:position' cells kept after discovery, e.g. "
                        "'2:-1,8:-28'. Default: every run --run-glob matches.")
    p.add_argument("--analysis-name", type=str, default="lora_ablation",
                   help="Output subfolder <run>/<name>/. Give non-default settings their own "
                        "name, or --skip-existing will keep results computed under others.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Single-run override; a sweep always writes to <run>/<analysis-name>/.")
    add_discovery_args(p)
    return p.parse_args()


def find_lora_linears(model):
    """Return {(layer, module): peft_lora_linear} for every gated LoRA linear."""
    out = {}
    for name, mod in model.named_modules():
        if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
            continue
        if not isinstance(mod.lora_B, torch.nn.ModuleDict) or len(mod.lora_B) == 0:
            continue
        lm, mm = LAYER_RE.search(name), MODULE_RE.search(name)
        if lm and mm:
            out[(int(lm.group(1)), mm.group(1))] = mod
    return out


def zero_lora(modules):
    """Zero lora_B for the given peft linears; return state to restore."""
    saved = []
    for mod in modules:
        for adapter, lin in mod.lora_B.items():
            saved.append((lin, lin.weight.data.clone()))
            lin.weight.data.zero_()
    return saved


def restore_lora(saved):
    for lin, w in saved:
        lin.weight.data.copy_(w)


@torch.no_grad()
def predict_and_target(model, dataset, collator, batch_size, do_intervention):
    """
    Regression-head prediction at the last token, its target and the row's natural
    (cached) activation, over the dataset.
    do_intervention=True: inject the signal (via the model's own forward hook,
    reproducible under the current distribution/scale/rng) -> target = injected
    value. do_intervention=False: no injection -> target = true activation.
    Only input_ids/attention_mask/lora_token_mask (+ signal columns when
    intervening) are passed, so the endogenous pass fires no intervention.
    """
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    head = model.regression_head
    preds, targets, acts = [], [], []
    for batch in dl:
        batch = batch.to(model.device)
        activation = batch.pop("activation")
        kwargs = dict(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                      lora_token_mask=batch.get("lora_token_mask"),
                      do_intervention=do_intervention, output_hidden_states=True)
        if do_intervention:
            kwargs.update(token_position=batch["token_position"],
                          signal_layer=batch["signal_layer"], signal_index=batch["signal_index"])
        out = model(**kwargs)
        h = out.hidden_states[-1][:, -1, :]
        pred = head(h.to(device=head.weight.device, dtype=head.weight.dtype)).squeeze(-1)
        preds.append(pred.float().cpu())
        targets.append(out.intervention_values.float().cpu() if do_intervention
                       else activation.float().cpu())
        acts.append(activation.float().cpu())
    return torch.cat(preds), torch.cat(targets), torch.cat(acts)


def _corr(pred, target):
    p, t = pred.numpy(), target.numpy()
    if np.std(p) < 1e-12 or np.std(t) < 1e-12:
        return float("nan")
    return float(pearsonr(p, t)[0])


def resolve_injection_regime(config, distribution="auto", scale=None):
    """(distribution, scale) the interventional arm is scored at -- the run's own
    training regime unless overridden. See the module docstring."""
    tc = config.training_config
    if distribution == "auto":
        distribution = tc.intervention_signal_distribution
    if scale is None:
        scale = tc.intervention_extreme_scale if distribution == "extreme" else 1.0
    return distribution, float(scale)


def eval_corrs(model, dataset, collator, config, distribution="gaussian", scale=1.0,
               batch_size=None):
    """(interventional corr at the given regime, endogenous corr, corr of the injected-run
    prediction with the natural activation) for the current state."""
    bs = batch_size or config.training_config.eval_batch_size
    # reproducibly seeded so every ablated variant sees the SAME injected values --
    # otherwise the delta would include the sampler's noise
    prev_dist, prev_scale = model.intervention_distribution, model.intervention_scale
    model.intervention_distribution, model.intervention_scale = distribution, scale
    model.seed_intervention_rng(config.seed)
    p_iv, y_iv, a_iv = predict_and_target(model, dataset, collator, bs, do_intervention=True)
    model.intervention_distribution, model.intervention_scale = prev_dist, prev_scale
    # endogenous
    p_en, y_en, _ = predict_and_target(model, dataset, collator, bs, do_intervention=False)
    return _corr(p_iv, y_iv), _corr(p_en, y_en), _corr(p_iv, a_iv)


def main():
    args = parse_args()
    if args.config:
        runs = [load_run(args.config, output_root=args.output_root)]
    else:
        runs = discover_runs(group=args.group, run_glob=args.run_glob,
                             config_root=args.config_root, output_root=args.output_root)
        runs = shard(filter_cells(runs, args.cells, log="[ablate]"),
                     args.task_index, args.n_tasks)
    if not runs:
        raise SystemExit("No runs with checkpoints matched.")
    run_per_locale(args, runs, args.analysis_name, "lora_ablation.csv",
                   analyze=lambda run, spec, out_dir: ablate_one_run(run, spec, args, out_dir),
                   replot=replot_ablation, log="[ablate]")

def ablate_one_run(run, spec, args, out_dir: Path):
    config, checkpoint = run.config, run.checkpoint
    model, test_dataset, collator = run.model, run.dataset, run.collator
    distribution, scale = resolve_injection_regime(
        config, args.interv_distribution, args.interv_scale)
    print(f"[ablate] interventional arm scored at distribution={distribution} scale={scale:g} "
          f"(the run's own training regime unless overridden)")
    if args.no_success_filter:
        dataset = test_dataset.select(range(min(args.n_examples, len(test_dataset))))
    else:
        # ---- select endogenous-success examples ----
        preds, acts = endogenous_predictions(model, test_dataset, collator,
                                             args.batch_size or config.training_config.eval_batch_size)
        idx = select_success_indices(preds, acts, args.success_threshold, args.n_examples, args.max_scan)
        if not idx:
            raise SystemExit("No success examples selected; loosen --success-threshold.")
        dataset = test_dataset.select(idx)
    print(f"{len(dataset)} endogenous examples selected ...")

    lora = find_lora_linears(model)
    if not lora:
        raise SystemExit("No LoRA linears found on the model.")
    layers_all = sorted({l for (l, _) in lora})
    modules_all = [m for m in MODULE_ORDER if m in {mm for (_, mm) in lora}]
    layer_filter = [int(x) for x in args.layers.split(",")] if args.layers else layers_all
    module_filter = args.modules.split(",") if args.modules else modules_all
    print(f"[ablate] checkpoint={checkpoint}")
    print(f"[ablate] layers={layer_filter} modules={module_filter} n={len(dataset)}")

    # ---- references ----
    base_iv, base_en, base_na = eval_corrs(model, dataset, collator, config, distribution, scale, args.batch_size)
    print(f"[ablate] BASELINE          interv={base_iv:.4f}  endo={base_en:.4f}  "
          f"natural|inj={base_na:.4f}")
    saved = zero_lora(list(lora.values()))
    off_iv, off_en, off_na = eval_corrs(model, dataset, collator, config, distribution, scale, args.batch_size)
    restore_lora(saved)
    print(f"[ablate] ALL-LoRA-OFF      interv={off_iv:.4f}  endo={off_en:.4f}  "
          f"natural|inj={off_na:.4f}")

    def row(L, m, granularity, iv, en, na):
        return {"layer": L, "module": m, "granularity": granularity, "interv": iv, "endo": en,
                "natural_inj": na, "d_interv": iv - base_iv, "d_endo": en - base_en,
                "d_natural_inj": na - base_na}

    rows = [row("baseline", "-", "reference", base_iv, base_en, base_na),
            row("all_off", "-", "reference", off_iv, off_en, off_na)]
    # ---- ablation units: single (layer, module) and/or a layer's attention / MLP block ----
    units = []
    for L in layer_filter:
        if args.granularity in ("module", "both"):
            units += [(L, m, "module", [lora[(L, m)]]) for m in module_filter if (L, m) in lora]
        if args.granularity in ("block", "both"):
            for b, mods in BLOCKS.items():
                members = [lora[(L, m)] for m in mods if (L, m) in lora]
                if members:
                    units.append((L, b, "block", members))
    d_iv_mat = np.full((len(layer_filter), len(module_filter)), np.nan)
    d_en_mat = np.full((len(layer_filter), len(module_filter)), np.nan)
    for L, m, granularity, mods in units:
        saved = zero_lora(mods)
        iv, en, na = eval_corrs(model, dataset, collator, config, distribution, scale, args.batch_size)
        restore_lora(saved)
        if granularity == "module":
            li, mi = layer_filter.index(L), module_filter.index(m)
            d_iv_mat[li, mi] = iv - base_iv
            d_en_mat[li, mi] = en - base_en
        rows.append(row(L, m, granularity, iv, en, na))
        print(f"[ablate] L{L:<2} {m:<9} interv={iv:.4f} (d={iv-base_iv:+.4f})  "
              f"endo={en:.4f} (d={en-base_en:+.4f})  natural|inj={na:.4f}")

    abl_df = pd.DataFrame(rows)
    abl_df.insert(0, "run", config.experiment_name)
    abl_df.insert(1, "signal_layer", run.layer)
    abl_df.insert(2, "signal_index", run.index)
    abl_df.insert(3, "position", getattr(spec, "position", None))
    abl_df.to_csv(out_dir / "lora_ablation.csv", index=False)
    if args.granularity != "block":          # the heatmaps are per (layer, module)
        _plot_delta(d_iv_mat, layer_filter, module_filter, base_iv,
                    "Δ interventional corr when module ablated",
                    out_dir / "ablation_delta_interventional.png")
        _plot_delta(d_en_mat, layer_filter, module_filter, base_en,
                    "Δ endogenous corr when module ablated",
                    out_dir / "ablation_delta_endogenous.png")
    with open(out_dir / "ablation_meta.json", "w") as f:
        json.dump({"run": config.experiment_name, "checkpoint": str(checkpoint),
                   "signal": f"N{run.layer}.{run.index}",
                   "interv_distribution": distribution, "interv_scale": scale,
                   "baseline_interv": base_iv,
                   "baseline_endo": base_en, "all_off_interv": off_iv,
                   "all_off_endo": off_en, "n_examples": len(dataset),
                   "success_filter": not args.no_success_filter,
                   "granularity": args.granularity}, f, indent=2)
    print(f"[ablate] wrote results to {out_dir}")


def _plot_delta(mat, layers, modules, baseline, title, path):
    lim = np.nanmax(np.abs(mat)) or 1.0
    plt.figure(figsize=(1.1 * len(modules) + 2, 0.34 * len(layers) + 2))
    sns.heatmap(mat, xticklabels=modules, yticklabels=layers, cmap="coolwarm",
                center=0, vmin=-lim, vmax=lim, annot=True, fmt=".2f", annot_kws={"size": 7},
                cbar_kws={"label": "Δ corr vs baseline"})
    plt.xlabel("module"); plt.ylabel("layer")
    plt.title(f"{title}\n(baseline={baseline:.3f})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def replot_ablation(spec, out_dir: Path, args) -> bool:
    """Redraw both ablation heatmaps from lora_ablation.csv; False if it is absent.
    The per-(layer, module) delta matrices are pivoted back out of the long CSV."""
    csv = out_dir / "lora_ablation.csv"
    if not csv.exists():
        return False
    df = pd.read_csv(csv)
    per = df[~df["layer"].astype(str).isin(["baseline", "all_off"])].copy()
    if "granularity" in per:
        per = per[per["granularity"] == "module"].copy()
    if per.empty:                            # block-only run: no per-module heatmap
        return True
    per["layer"] = per["layer"].astype(int)
    layers = sorted(per["layer"].unique())
    modules = [m for m in MODULE_ORDER if m in set(per["module"])]

    def _mat(col):
        return (per.pivot_table(index="layer", columns="module", values=col)
                .reindex(index=layers, columns=modules).to_numpy())

    base = df[df["layer"].astype(str) == "baseline"].iloc[0]
    _plot_delta(_mat("d_interv"), layers, modules, float(base["interv"]),
                "Δ interventional corr when module ablated",
                out_dir / "ablation_delta_interventional.png")
    _plot_delta(_mat("d_endo"), layers, modules, float(base["endo"]),
                "Δ endogenous corr when module ablated",
                out_dir / "ablation_delta_endogenous.png")
    return True


if __name__ == "__main__":
    main()
