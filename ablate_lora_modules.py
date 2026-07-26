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

Baseline (nothing ablated) and all-LoRA-off rows are included as references.
Config-driven (model/dataset/signal/checkpoint from --config) so future runs
just swap the config.

Example
-------
python ablate_lora_modules.py \
    --config experiment_configs/.../<run>.json \
    --n-examples 512 --layers 7,8,9,10,11,12
"""
from __future__ import annotations

import os
import re
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader

from src.introspector.config import ExperimentConfig
from trace_introspection_circuit import load_model_and_data, endogenous_predictions, select_success_indices

MODULE_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LAYER_RE = re.compile(r"layers\.(\d+)\.")
MODULE_RE = re.compile(r"\.((?:q|k|v|o|gate|up|down)_proj)$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Checkpoint dir override (default: latest under output_dir/experiment_name).")
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
    p.add_argument("--output-dir", type=Path, default=None)
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
    Regression-head prediction at the last token and its target, over the dataset.
    do_intervention=True: inject the signal (via the model's own forward hook,
    reproducible under the current distribution/scale/rng) -> target = injected
    value. do_intervention=False: no injection -> target = true activation.
    Only input_ids/attention_mask/lora_token_mask (+ signal columns when
    intervening) are passed, so the endogenous pass fires no intervention.
    """
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    head = model.regression_head
    preds, targets = [], []
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
    return torch.cat(preds), torch.cat(targets)


def _corr(pred, target):
    p, t = pred.numpy(), target.numpy()
    if np.std(p) < 1e-12 or np.std(t) < 1e-12:
        return float("nan")
    return float(pearsonr(p, t)[0])


def eval_corrs(model, dataset, collator, config):
    """(interventional@gaussian scale=1 corr, endogenous corr) for current model state."""
    bs = config.training_config.eval_batch_size
    # interventional (endogenous-matched gaussian, scale=1), reproducibly seeded
    prev_dist, prev_scale = model.intervention_distribution, model.intervention_scale
    model.intervention_distribution, model.intervention_scale = "gaussian", 1.0
    model.seed_intervention_rng(config.seed)
    p_iv, y_iv = predict_and_target(model, dataset, collator, bs, do_intervention=True)
    model.intervention_distribution, model.intervention_scale = prev_dist, prev_scale
    # endogenous
    p_en, y_en = predict_and_target(model, dataset, collator, bs, do_intervention=False)
    return _corr(p_iv, y_iv), _corr(p_en, y_en)


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(args.config.read_text())

    checkpoint = args.checkpoint
    if checkpoint is None:
        ckpts = glob.glob(str(Path(config.output_dir) / config.experiment_name / "checkpoint-*"))
        if not ckpts:
            raise SystemExit("No checkpoint found; pass --checkpoint.")
        checkpoint = Path(max(ckpts, key=os.path.getctime))
    out_dir = args.output_dir or (Path(config.output_dir) / config.experiment_name / "lora_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, test_dataset, test_raw, collator, signal_ranges = load_model_and_data(config, checkpoint)
    # dataset = test_dataset.select(range(min(args.n_examples, len(test_dataset))))
    # ---- select endogenous-success examples ----
    preds, acts = endogenous_predictions(model, test_dataset, collator, args.batch_size)
    idx = select_success_indices(preds, acts, args.success_threshold, args.n_examples, args.max_scan)
    if not idx:
        raise SystemExit("No success examples selected; loosen --success-threshold.")
    dataset = test_dataset.select(idx)
    print(f"{len(idx)} endogenous examples selected ...")

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
    base_iv, base_en = eval_corrs(model, dataset, collator, config)
    print(f"[ablate] BASELINE          interv={base_iv:.4f}  endo={base_en:.4f}")
    saved = zero_lora(list(lora.values()))
    off_iv, off_en = eval_corrs(model, dataset, collator, config)
    restore_lora(saved)
    print(f"[ablate] ALL-LoRA-OFF      interv={off_iv:.4f}  endo={off_en:.4f}")

    # ---- per-(layer, module) ablation ----
    rows = [{"layer": "baseline", "module": "-", "interv": base_iv, "endo": base_en,
             "d_interv": 0.0, "d_endo": 0.0},
            {"layer": "all_off", "module": "-", "interv": off_iv, "endo": off_en,
             "d_interv": off_iv - base_iv, "d_endo": off_en - base_en}]
    d_iv_mat = np.full((len(layer_filter), len(module_filter)), np.nan)
    d_en_mat = np.full((len(layer_filter), len(module_filter)), np.nan)
    for li, L in enumerate(layer_filter):
        for mi, m in enumerate(module_filter):
            mod = lora.get((L, m))
            if mod is None:
                continue
            saved = zero_lora([mod])
            iv, en = eval_corrs(model, dataset, collator, config)
            restore_lora(saved)
            d_iv_mat[li, mi] = iv - base_iv
            d_en_mat[li, mi] = en - base_en
            rows.append({"layer": L, "module": m, "interv": iv, "endo": en,
                         "d_interv": iv - base_iv, "d_endo": en - base_en})
            print(f"[ablate] L{L:<2} {m:<9} interv={iv:.4f} (d={iv-base_iv:+.4f})  "
                  f"endo={en:.4f} (d={en-base_en:+.4f})")

    pd.DataFrame(rows).to_csv(out_dir / "lora_ablation.csv", index=False)
    _plot_delta(d_iv_mat, layer_filter, module_filter, base_iv,
                "Δ interventional corr when module ablated", out_dir / "ablation_delta_interventional.png")
    _plot_delta(d_en_mat, layer_filter, module_filter, base_en,
                "Δ endogenous corr when module ablated", out_dir / "ablation_delta_endogenous.png")
    with open(out_dir / "ablation_meta.json", "w") as f:
        json.dump({"checkpoint": str(checkpoint), "baseline_interv": base_iv,
                   "baseline_endo": base_en, "all_off_interv": off_iv,
                   "all_off_endo": off_en, "n_examples": len(dataset)}, f, indent=2)
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


if __name__ == "__main__":
    main()
