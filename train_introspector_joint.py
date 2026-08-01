#!/usr/bin/env python3
"""
Joint endogenous + interventional introspection training.

Per batch we run TWO forwards on the same examples and step on a weighted sum:

    L = w * L_endo + (1 - w) * L_interv

  - L_endo:   no injection, target = the true activation      (natural-scale target)
  - L_interv: inject the signal, target = the injected value  (large learnable scale)

Both losses are VARIANCE-NORMALIZED (MSE / var(target) ~= 1 - R^2) so their
magnitudes are comparable despite the ~10x scale difference, and `w` genuinely
controls the balance rather than the accident of scale.

Rationale (see design notes):
  * The faithful absolute-value reader is the UNIQUE joint optimum: a bypass head
    fails L_interv (context is uncorrelated with the injected value, so it adds
    noise), a perturbation-only head fails L_endo (no perturbation endogenously).
    So L_interv is the anti-bypass regularizer -- keep `w` (endo weight) < 1 and
    CONSTANT (do NOT anneal the mix toward endogenous; that removes the pressure).
  * Interventional injection must stay at a LEARNABLE scale (>=10x, the natural
    scale is at the bf16 floor), configured via the usual distribution/scale; the
    endogenous term supplies the natural-scale target.
  * Two forwards per batch => both gradients enter every step (no tug-of-war).

Sweep `--joint-loss-weight` across runs and read the (endogenous, interventional)
FRONTIER: a weight with BOTH high => a faithful reader exists; a pure trade-off
=> it is not representable in this single-forward regime.
"""
from __future__ import annotations

import os
import glob
import json
import math
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.introspector.config import ExperimentConfig
from src.introspector.prompts import PROMPT_REGISTRY
from src.introspector.training.collator import LoRAGatingCollator
from src.introspector.training.metrics import compute_metrics, compute_metrics_from_numericals
from src.introspector.training.model import (
    IntrospectionRegressionModel,
    RegressionHeadSaveCallback,
    InterventionAnnealCallback,
    load_regression_head,
    replace_model_with_gated_version,
    replace_lora_with_gated_version,
)
from train_introspector import preprocess_example, INTROSPECTION_SEPARATOR
from train_introspector_with_intervention import (
    InterventionTrainer,
    evaluate_interventional_test,
    get_argmax_from_logits,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--joint-loss-weight", type=float, default=0.5,
                   help="Weight w on the ENDOGENOUS term; interventional gets (1-w). "
                        "Keep constant; sweep across runs to map the frontier.")
    p.add_argument("--routing-probe-scale", type=float, default=10.0,
                   help="Injection scale for the routing-probe diagnostic (use a LARGE learnable "
                        "scale so 'does it reach the readout' is detectable above the bf16 floor).")
    p.add_argument("--routing-probe-pca", type=int, default=256,
                   help="PCA components for the routing-probe ceiling (regularize when n ~ n_features; "
                        "0 disables PCA).")
    p.add_argument("--scale-sweep", type=str, default="1,3,10,30,100",
                   help="Comma-separated injection scales for the interventional scale sweep.")
    p.add_argument("--diag-fp32", action="store_true",
                   help="Also re-run the diagnostics (scale sweep + routing probe) with the model cast "
                        "to fp32, to test whether natural-scale failure is a bf16 numerical floor. "
                        "Cheap: resume the checkpoint (skips training) and pass this flag.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Specific checkpoint dir to load (default: latest under output_dir).")
    p.add_argument("--diag-only", action="store_true",
                   help="Load --checkpoint's weights and run evals/diagnostics WITHOUT training. Use to "
                        "eval a MID-RUN checkpoint (plain resume would continue training instead). "
                        "Outputs go to output_dir/diag_<checkpoint>/ so they don't clobber the final run.")
    return p.parse_args()


def _var_norm_mse(pred, target):
    """Variance-normalized MSE = MSE / var(target) (~ 1 - R^2, scale-invariant)."""
    pred = pred.float()
    target = target.float()
    var = torch.clamp(target.var(), min=1e-8)
    return nn.functional.mse_loss(pred, target) / var


class JointInterventionTrainer(InterventionTrainer):
    """Two-forward joint loss: w*endogenous + (1-w)*interventional (both var-normed)."""

    def __init__(self, *args, joint_loss_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_loss_weight = joint_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = inputs.copy()

        # --- endogenous forward: no injection, target = true activation ---
        endo_inputs = inputs.copy()
        endo_inputs["labels"] = inputs["activation"]
        endo_inputs["do_intervention"] = False
        endo_out = model(**endo_inputs)
        L_endo = _var_norm_mse(endo_out.logits, inputs["activation"])

        # --- interventional forward: inject, target = injected value ---
        interv_inputs = inputs.copy()
        interv_inputs["labels"] = inputs["activation"]   # forward overwrites with injected value
        interv_inputs["do_intervention"] = True
        interv_out = model(**interv_inputs)
        L_interv = _var_norm_mse(interv_out.logits, interv_out.intervention_values)

        w = self.joint_loss_weight
        loss = w * L_endo + (1.0 - w) * L_interv
        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            self.log({"L_endo": float(L_endo.detach()), "L_interv": float(L_interv.detach())})
        return (loss, interv_out) if return_outputs else loss


@torch.no_grad()
def eval_endogenous_test(model, dataset, collator, config, output_dir, suffix="_endogenous"):
    """
    Endogenous eval (no injection): full metrics (corr, l2_dist, avg/std diff, ...)
    matching the interventional eval, plus a predicted-vs-actual scatter.
    """
    dl = DataLoader(dataset, batch_size=config.training_config.eval_batch_size,
                    collate_fn=collator, shuffle=False)
    head = model.regression_head
    preds, acts = [], []
    for batch in dl:
        batch = batch.to(model.device)
        activation = batch["activation"]
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    lora_token_mask=batch.get("lora_token_mask"),
                    do_intervention=False, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]
        pred = head(h.to(device=head.weight.device, dtype=head.weight.dtype)).squeeze(-1)
        preds.append(pred.float().cpu()); acts.append(activation.float().cpu())
    pred, act = torch.cat(preds).numpy(), torch.cat(acts).numpy()

    metrics = compute_metrics_from_numericals(pred, act)   # corr, l2_dist, avg_diff, std_diff, ...
    with open(Path(output_dir) / f"endogenous_test_metrics{suffix}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"predictions": pred, "activation": act}).to_csv(
        Path(output_dir) / f"endogenous_test_predictions{suffix}.csv", index=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(act, pred, s=12, alpha=0.5)
    lim = [float(min(act.min(), pred.min())), float(max(act.max(), pred.max()))]
    plt.plot(lim, lim, "k--", lw=1, alpha=0.6)   # y = x reference
    plt.xlabel("true activation"); plt.ylabel("prediction")
    plt.title(f"endogenous: pred vs actual "
              f"(corr={metrics['corr']:.3f}, l2={metrics['l2_dist']:.3f})")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"endogenous_test_scatter{suffix}.png", dpi=150)
    plt.close()
    print(f"[joint] endogenous test: corr={metrics['corr']:.4f} l2={metrics['l2_dist']:.4f} "
          f"avg_diff={metrics['avg_diff']:.4f} std_diff={metrics['std_diff']:.4f}")
    return metrics


# ======================================================================
# Routing probe: does the injected value REACH the readout of the TRAINED
# model (fresh-probe ceiling) vs does the TRAINED HEAD read it, by position.
# ceiling high, head low -> routed but not read (optimization/head)
# ceiling low            -> not routed (routing/capacity)
# NB: this is the TRAINED model (a ceiling conditional on this run), NOT a
# pre-LoRA base-model floor.
# ======================================================================
@torch.no_grad()
def _extract_interv_readout(model, dataset, collator, config, scale, distribution="extreme"):
    """Under injection at `scale`, collect readout features, injected values,
    trained-head preds, and token_positions."""
    prev_dist, prev_scale = model.intervention_distribution, model.intervention_scale
    model.intervention_distribution, model.intervention_scale = distribution, scale
    model.seed_intervention_rng(config.seed)
    dl = DataLoader(dataset, batch_size=config.training_config.eval_batch_size,
                    collate_fn=collator, shuffle=False)
    head = model.regression_head
    feats, inj, hpred, pos = [], [], [], []
    for batch in dl:
        batch = batch.to(model.device)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    lora_token_mask=batch.get("lora_token_mask"),
                    token_position=batch["token_position"], signal_layer=batch["signal_layer"],
                    signal_index=batch["signal_index"], do_intervention=True, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]
        feats.append(h.float().cpu())
        inj.append(out.intervention_values.float().cpu())
        hpred.append(head(h.to(device=head.weight.device, dtype=head.weight.dtype)).squeeze(-1).float().cpu())
        pos.append(batch["token_position"].cpu())
    model.intervention_distribution, model.intervention_scale = prev_dist, prev_scale
    return (torch.cat(feats).numpy(), torch.cat(inj).numpy(),
            torch.cat(hpred).numpy(), torch.cat(pos).numpy())


def _corr_or_nan(a, b):
    return float(pearsonr(a, b)[0]) if (np.std(a) > 1e-12 and np.std(b) > 1e-12) else float("nan")


ROUTING_ALPHAS = np.logspace(-1, 5, 13)   # wide grid so RidgeCV can regularize hard when n ~ H


def _ceiling_and_head(X, inj, hpred, seed, n_pca, split=0.7):
    """
    On ONE seeded held-out split: fit a well-regularized probe (standardize ->
    PCA -> RidgeCV) on train, and score BOTH the fresh probe (ceiling) and the
    trained head on the SAME held-out rows. Regularizing + PCA is essential when
    n_examples ~ n_features, else the ceiling overfits and comes out < head
    (which is impossible for a true ceiling).
    Returns (ceiling_corr, head_corr, best_alpha).
    """
    n = len(inj)
    if n < 20:
        return float("nan"), float("nan"), float("nan")
    perm = np.random.RandomState(seed).permutation(n)
    ntr = int(n * split)
    tr, te = perm[:ntr], perm[ntr:]
    if len(tr) < 5 or len(te) < 5:
        return float("nan"), float("nan"), float("nan")
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    if n_pca and n_pca < Xtr.shape[1]:
        pca = PCA(n_components=min(n_pca, len(tr) - 1)).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    reg = RidgeCV(alphas=ROUTING_ALPHAS).fit(Xtr, inj[tr])
    ceil = _corr_or_nan(reg.predict(Xte), inj[te])
    head = _corr_or_nan(hpred[te], inj[te])          # head on the SAME held-out rows
    return ceil, head, float(getattr(reg, "alpha_", np.nan))


def routing_probe_by_position(model, dataset, collator, config, output_dir, scale, n_pca=256, n_bins=6,
                              suffix=""):
    X, inj, hpred, pos = _extract_interv_readout(model, dataset, collator, config, scale)

    def row(bin_label, m):
        lo, hi = float(pos[m].min()), float(pos[m].max())
        c, h, a = _ceiling_and_head(X[m], inj[m], hpred[m], config.seed, n_pca)
        return {"position_bin": bin_label, "pos_lo": lo, "pos_hi": hi, "n": int(m.sum()),
                "ceiling_corr": c, "head_corr": h, "best_alpha": a}

    rows = [row("all", np.ones(len(inj), bool))]
    if len(np.unique(pos)) > 1:
        edges = np.unique(np.quantile(pos, np.linspace(0, 1, n_bins + 1)))
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pos >= lo) & (pos <= hi)
            if m.sum() >= 40:
                rows.append(row(f"[{lo:.0f},{hi:.0f}]", m))
    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / f"routing_probe_by_position{suffix}.csv", index=False)
    print(f"[joint] routing probe (inject scale={scale}, suffix={suffix or 'native'}):")
    print(df.to_string(index=False))

    binned = df[df["position_bin"] != "all"]
    if len(binned) >= 2:
        centers = (binned["pos_lo"] + binned["pos_hi"]) / 2
        plt.figure(figsize=(7, 5))
        plt.plot(centers, binned["ceiling_corr"], "o-", label="routing ceiling (fresh probe)")
        plt.plot(centers, binned["head_corr"], "s--", label="trained head")
        plt.xlabel("introspected token position"); plt.ylabel("interventional corr")
        plt.title(f"routing vs reading by position (inject scale={scale}){suffix}")
        plt.ylim(-0.05, 1.02); plt.legend(); plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"routing_probe_by_position{suffix}.png", dpi=150)
        plt.close()
    return df


def interventional_scale_sweep(model, dataset, collator, config, output_dir, scales,
                               distribution="gaussian", n_pca=256, suffix=""):
    """
    Interventional corr vs injection scale, for BOTH the trained head and a fresh
    ceiling probe (gaussian: N(mean, scale*std)).

    corr = 1 / sqrt(1 + sigma_eps^2 / (g^2 sigma_v^2)) for a linear channel with
    routing gain g and fixed context-noise eps -> corr rises with scale purely
    from SNR, no floor/nonlinearity needed. Decision:
      * CEILING also rises with scale  -> SNR/routing-gain limited (fundamental;
        natural-scale signal is swamped by context noise; needs higher gain).
      * CEILING flat & high, head rises -> signal is linearly present at natural
        scale; the head was just trained at large scale (distribution mismatch).
    """
    rows = []
    for s in scales:
        X, inj, hpred, _ = _extract_interv_readout(model, dataset, collator, config, s, distribution=distribution)
        ceil, head, alpha = _ceiling_and_head(X, inj, hpred, config.seed, n_pca)
        rows.append({"scale": s, "head_corr": head, "ceiling_corr": ceil,
                     "inj_std": float(np.std(inj)), "best_alpha": alpha})
    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / f"interventional_scale_sweep{suffix}.csv", index=False)
    print(f"[joint] interventional scale sweep ({distribution}, suffix={suffix or 'native'}):")
    print(df.to_string(index=False))
    plt.figure(figsize=(7, 5))
    plt.plot(df["scale"], df["ceiling_corr"], "o-", label="ceiling (fresh probe)")
    plt.plot(df["scale"], df["head_corr"], "s--", label="trained head")
    plt.xscale("log")
    plt.xlabel("injection scale (x natural std)"); plt.ylabel("interventional corr")
    plt.title(f"interventional corr vs injection scale ({distribution}){suffix}")
    plt.ylim(-0.05, 1.02); plt.grid(alpha=0.2, which="both"); plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"interventional_scale_sweep{suffix}.png", dpi=150)
    plt.close()
    return df


def main():
    args = parse_args()
    config = ExperimentConfig.model_validate_json(args.config.read_text())
    if not config.use_regressor:
        raise SystemExit("Joint training targets the regressor head (use_regressor must be True).")

    cache_dir = config.dataset_cache_dir
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "routing_probe_by_position.csv").exists() and not args.overwrite_cache and not args.diag_only:
        print(f"Experiments ran already, skipping.. {output_dir}")
        return

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [INTROSPECTION_SEPARATOR]})

    # ---- data ----
    train_raw = load_from_disk(str(cache_dir / "train"))
    dev_raw = load_from_disk(str(cache_dir / "dev"))
    test_raw = load_from_disk(str(cache_dir / "test"))

    df = train_raw.to_pandas()
    signal_ranges = {}
    for key, vals in df.groupby(["signal_layer", "signal_index"])["activation"]:
        signal_ranges[key] = {"min": float(vals.min()), "max": float(vals.max()),
                              "mean": float(vals.mean()), "std": float(vals.std())}

    def preprocess(ex):
        out = preprocess_example(ex, tokenizer=tokenizer,
                                 prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
                                 max_length=config.training_config.max_seq_length,
                                 return_frozen_input_len=True, for_prediction=True)
        out["token_position"] = ex["token_position"]
        out["signal_layer"] = ex["signal_layer"]
        out["signal_index"] = ex["signal_index"]
        out["activation"] = ex["activation"]
        return out

    train_dataset = train_raw.map(preprocess, remove_columns=train_raw.column_names)
    dev_dataset = dev_raw.map(preprocess, remove_columns=dev_raw.column_names)
    test_dataset_pred = test_raw.map(preprocess, remove_columns=test_raw.column_names)

    # ---- model (regressor + gated LoRA) ----
    base = AutoModelForCausalLM.from_pretrained(
        config.model_name, device_map="auto",
        torch_dtype=torch.bfloat16 if config.training_config.bf16 else torch.float32)
    base.resize_token_embeddings(len(tokenizer))
    diag_ckpt = None
    if args.diag_only:
        # load a specific (possibly mid-run) checkpoint's adapter for eval WITHOUT training
        from peft import PeftModel
        ckpts = glob.glob(f"{output_dir}/checkpoint-*")
        diag_ckpt = Path(args.checkpoint) if args.checkpoint else (
            Path(max(ckpts, key=os.path.getctime)) if ckpts else None)
        if diag_ckpt is None:
            raise SystemExit("--diag-only needs a checkpoint (none found; pass --checkpoint).")
        model = PeftModel.from_pretrained(base, str(diag_ckpt), autocast_adapter_dtype=False)
    else:
        peft_config = LoraConfig(
            r=config.training_config.lora_r, lora_alpha=config.training_config.lora_alpha,
            target_modules=config.training_config.lora_target_modules,
            lora_dropout=config.training_config.lora_dropout, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base, peft_config)
    model.resize_token_embeddings(len(tokenizer))
    model = IntrospectionRegressionModel(
        model, signal_ranges=signal_ranges,
        intervention_distribution=config.training_config.intervention_signal_distribution,
        intervention_extreme_scale=config.training_config.intervention_extreme_scale)
    if args.diag_only:
        load_regression_head(model, str(diag_ckpt))
    replace_model_with_gated_version(model)
    replace_lora_with_gated_version(model)

    pred_collator = LoRAGatingCollator(tokenizer=tokenizer, model=model, padding=True,
                                       label_pad_token_id=-100, return_tensors="pt", for_prediction=True)

    # checkpoint frequency: 0 => per epoch; >0 => every save_steps (float <1 = ratio of total steps)
    save_kwargs = ({"save_strategy": "epoch"} if config.training_config.save_steps <= 0
                   else {"save_strategy": "steps", "save_steps": config.training_config.save_steps})
    training_args = TrainingArguments(
        output_dir=str(output_dir), run_name=config.experiment_name,
        learning_rate=config.training_config.learning_rate,
        warmup_ratio=config.training_config.warmup_ratio,
        lr_scheduler_type=config.training_config.lr_scheduler_type,
        per_device_train_batch_size=config.training_config.train_batch_size,
        per_device_eval_batch_size=config.training_config.eval_batch_size,
        num_train_epochs=config.training_config.num_epochs,
        logging_steps=math.ceil(100 / config.training_config.train_batch_size),
        eval_strategy="steps", eval_steps=math.ceil(400 // config.training_config.train_batch_size),
        bf16=config.training_config.bf16,
        remove_unused_columns=False, report_to="wandb", **save_kwargs)

    callbacks = [RegressionHeadSaveCallback(model)]
    if config.training_config.intervention_anneal:
        # NOTE: this anneals the injection SCALE (magnitude), NOT the mix weight.
        callbacks.append(InterventionAnnealCallback(
            model,
            scale_start=config.training_config.intervention_scale_start,
            scale_end=config.training_config.intervention_scale_end,
            schedule=config.training_config.intervention_anneal_schedule,
            start_frac=config.training_config.intervention_anneal_start_frac,
            end_frac=config.training_config.intervention_anneal_end_frac,
            log_every=math.ceil(400 / config.training_config.train_batch_size)))

    trainer = JointInterventionTrainer(
        model=model, tokenizer=tokenizer, signal_ranges=signal_ranges, args=training_args,
        train_dataset=train_dataset, eval_dataset=dev_dataset, data_collator=pred_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        preprocess_logits_for_metrics=None, experiment_config=config,
        do_intervention=True, callbacks=callbacks, joint_loss_weight=args.joint_loss_weight)
    print(f"[joint] joint_loss_weight (endo)={args.joint_loss_weight}  "
          f"interv_distribution={config.training_config.intervention_signal_distribution} "
          f"scale={model.intervention_scale}")

    ckpt_dirs = glob.glob(f"{output_dir}/checkpoint-*")
    eval_out_dir = output_dir
    if args.diag_only:
        # weights were already loaded from diag_ckpt via PeftModel.from_pretrained; no training
        eval_out_dir = output_dir / f"diag_{diag_ckpt.name}"
        eval_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[joint] diag-only: loaded {diag_ckpt}; writing diagnostics to {eval_out_dir}")
    elif ckpt_dirs and not args.overwrite_cache:
        latest = str(args.checkpoint) if args.checkpoint else max(ckpt_dirs, key=os.path.getctime)
        print(f"Found checkpoint, resuming/loading from {latest}")
        train_results = trainer.train(resume_from_checkpoint=latest)
        load_regression_head(model, latest)
    else:
        train_results = trainer.train()
        trainer.save_metrics("train", train_results.metrics)

    # ---- frontier metrics: interventional@gaussian(scale=1) AND endogenous ----
    evaluate_interventional_test(model, eval_dataset=test_dataset_pred, collator=pred_collator,
                                 config=config, output_dir=eval_out_dir,
                                 distribution="gaussian", suffix="_gaussian")
    eval_endogenous_test(model, test_dataset_pred, pred_collator, config, eval_out_dir)

    # diagnostic: is the injected value routed to the readout (ceiling) vs read by
    # the trained head, stratified by token position -> tells us whether earlier-
    # position failure is a routing (capacity) or reading (optimization) problem.
    routing_probe_by_position(model, test_dataset_pred, pred_collator, config, eval_out_dir,
                              scale=args.routing_probe_scale, n_pca=args.routing_probe_pca)

    # diagnostic: interventional corr vs injection scale -> reveals the numerical/
    # noise floor (corr rising with scale => natural scale is below the floor).
    scales = [float(x) for x in args.scale_sweep.split(",")]
    interventional_scale_sweep(model, test_dataset_pred, pred_collator, config, eval_out_dir,
                               scales=scales, n_pca=args.routing_probe_pca)

    # fp32 overlay: if natural-scale corr lifts toward the large-scale plateau when
    # cast to fp32, the natural-scale failure was a bf16 numerical floor.
    if args.diag_fp32:
        print("[joint] casting model to fp32 and re-running diagnostics (_fp32) ...")
        model.float()
        interventional_scale_sweep(model, test_dataset_pred, pred_collator, config, eval_out_dir,
                                   scales=scales, n_pca=args.routing_probe_pca, suffix="_fp32")
        routing_probe_by_position(model, test_dataset_pred, pred_collator, config, eval_out_dir,
                                  scale=args.routing_probe_scale, n_pca=args.routing_probe_pca, suffix="_fp32")


if __name__ == "__main__":
    main()
