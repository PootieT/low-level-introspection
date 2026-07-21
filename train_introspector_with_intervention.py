#!/usr/bin/env python3

"""
Train an introspector model WITH causal intervention training.

Key idea:
-----------
During forward pass, we intervene on the target neuron activation
and replace it with a random value sampled from the neuron's empirical range.

The model is then trained to predict THIS INTERVENED VALUE.

This prevents causal bypass where the model predicts activations
from contextual correlations instead of actually reading the neuron.

Architecture:
--------------
- frozen base model
- LoRA adapters
- token-wise LoRA gating
- native torch forward hooks for intervention
- normal teacher-forced LM training

"""

from __future__ import annotations

import os
import pdb
import glob
import argparse
import json
import math
import random
import operator
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_from_disk, Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.introspector.utils import float_to_str_target
from src.introspector.config import ExperimentConfig
from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.prompts import PROMPT_REGISTRY
from src.introspector.training.collator import LoRAGatingCollator
from src.introspector.training.metrics import (
    compute_metrics,
    compute_metrics_from_numericals,
    get_numericals,
)
from src.introspector.training.model import (
    replace_model_with_gated_version,
    replace_lora_with_gated_version,
    IntrospectionRegressionModel,
    RegressionHeadSaveCallback,
    InterventionAnnealCallback,
    load_regression_head,
)
from train_introspector import (
    preprocess_example,
    INTROSPECTION_SEPARATOR,
    evaluate_test_split
)

# ============================================================
# Utilities
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="Whether to overwrite cached experiment files if there exist one")
    parser.add_argument("--no-intervention", action="store_true",
                        help="Disable causal intervention. For the regressor this trains/evaluates against "
                             "the ORIGINAL activation instead of a random intervened value, which is a "
                             "sanity check that the regression head can read a scalar out of the hidden "
                             "states at all.")
    parser.add_argument("--refit-head", action="store_true",
                        help="Post-hoc diagnostic: freeze the LoRA/base model and solve for the OPTIMAL "
                             "linear regression head (closed-form ridge/OLS) that predicts the true "
                             "activation from the frozen representation. Answers 'what is the best a "
                             "linear head could do on the normal activation range given this model'.")
    parser.add_argument("--refit-estimator", type=str, default="ridge", choices=["ridge", "pls"],
                        help="Linear fitter for --refit-head: 'ridge' (L2, dense direction, hyperparam=lambda) "
                             "or 'pls' (partial least squares, low-rank subspace, hyperparam=n_components).")
    parser.add_argument("--refit-lambda-grid", type=str, default="0.01,0.1,1,10,100,1000,10000",
                        help="Comma-separated ridge lambdas to sweep (--refit-estimator ridge). The best is "
                             "selected on the dev split (by correlation); the final head is fit on "
                             "train+dev with that lambda and evaluated on test.")
    parser.add_argument("--refit-pls-components", type=str, default="1,2,4,8,16,32,64",
                        help="Comma-separated n_components to sweep (--refit-estimator pls).")


    return parser.parse_args()


def get_argmax_from_logits(logits, labels):
    return torch.argmax(logits, dim=-1)


# ============================================================
# Post-hoc optimal regression head (closed-form ridge / OLS)
# ============================================================

@torch.no_grad()
def _extract_readout_features(model, dataset, collator, config, readout_pos=-1, do_intervention=False):
    """
    Run the frozen model and collect, per example:
      - the last-layer hidden state at the readout position (the vector the
        regression head reads), and
      - the target: the true activation when do_intervention=False, or the
        (sampled) injected value when do_intervention=True. Under intervention
        the hidden state also reflects the injection.
    Returns (X [N, H], y [N]) as float32 CPU tensors.
    """
    dl = DataLoader(
        dataset,
        batch_size=config.training_config.eval_batch_size,
        collate_fn=collator,
        shuffle=False,
    )
    feats, targets = [], []
    model.eval()
    for batch in dl:
        batch = batch.to(model.device)
        activation = batch.pop("activation")
        outputs = model(**batch, do_intervention=do_intervention, output_hidden_states=True)
        h = outputs.hidden_states[-1][:, readout_pos, :].float().cpu()
        feats.append(h)
        if do_intervention:
            targets.append(outputs.intervention_values.float().cpu())
        else:
            targets.append(activation.float().cpu())
    return torch.cat(feats), torch.cat(targets)


# ---- linear probes: any fitter returns a (coef, intercept) linear map ----
@dataclass
class LinearProbe:
    coef: torch.Tensor   # [H]
    intercept: float

    def predict(self, X):  # X: [N, H] -> [N]
        return X @ self.coef.to(X.dtype) + self.intercept


def fit_ridge(X, y, lam):
    """Closed-form ridge (L2) probe."""
    Xb = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
    A = Xb.T @ Xb + lam * torch.eye(Xb.shape[1])
    w = torch.linalg.solve(A, Xb.T @ y)
    return LinearProbe(coef=w[:-1].float(), intercept=float(w[-1]))


def fit_pls(X, y, n_components):
    """
    Partial least squares (single response) via sklearn's PLSRegression, which
    finds n_components low-rank latent directions of maximum covariance with y.
    PLS predict is affine, so we recover the exact effective (coef, intercept)
    linear map by probing with 0 and the identity (convention-proof across
    sklearn versions).
    """
    from sklearn.cross_decomposition import PLSRegression

    hdim = X.shape[1]
    Xnp = X.double().numpy()
    ynp = y.double().numpy()
    pls = PLSRegression(n_components=int(n_components), scale=True).fit(Xnp, ynp)
    intercept = float(pls.predict(np.zeros((1, hdim))).ravel()[0])
    coef = pls.predict(np.eye(hdim)).ravel() - intercept
    return LinearProbe(coef=torch.tensor(coef, dtype=torch.float32), intercept=intercept)


FITTERS = {"ridge": fit_ridge, "pls": fit_pls}


def _corr_score(metrics):
    """Selection score = dev correlation, treating nan (e.g. degenerate constant
    predictions) as worst-possible."""
    c = metrics["corr"]
    return -float("inf") if (c is None or np.isnan(c)) else c


def evaluate_interventional_test(
    model: IntrospectionRegressionModel,
    eval_dataset: Dataset,
    collator: LoRAGatingCollator,
    config: ExperimentConfig,
    output_dir: str,
    distribution: str = "gaussian",
    suffix: str = "_gaussian",
):
    """
    Evaluate the CURRENT (trained) regression head on the test split under
    intervention drawn from `distribution`.

    With distribution="gaussian" the injected values match the neuron's endogenous
    N(mean, std) -- i.e. the same range the model sees naturally. Because the
    injected value is independent of context, contextual bypass scores exactly 0
    here by construction: any correlation is genuine causal reading. This is the
    metric to select on.
    """
    prev_dist, prev_scale = model.intervention_distribution, model.intervention_scale
    model.intervention_distribution = distribution
    model.intervention_scale = 1.0   # endogenous-matched, regardless of the annealed training scale
    model.seed_intervention_rng(config.seed)
    X, y_inj = _extract_readout_features(model, eval_dataset, collator, config, do_intervention=True)
    model.intervention_distribution, model.intervention_scale = prev_dist, prev_scale

    head = model.regression_head
    with torch.no_grad():
        pred = head(X.to(device=head.weight.device, dtype=head.weight.dtype)).squeeze(-1).float().cpu()
    metrics = compute_metrics_from_numericals(pred.numpy(), y_inj.numpy())
    print(f"[interventional test eval | {distribution} injection] "
          f"corr={metrics['corr']:.4f}  l2={metrics['l2_dist']:.4f}")

    with open(str(Path(output_dir) / f"interventional_test_metrics{suffix}.json"), "w") as f:
        json.dump({"distribution": distribution, "test": metrics}, f, indent=2)
    df = pd.DataFrame({"predictions": pred.numpy(), "injected": y_inj.numpy()})
    df.to_csv(str(Path(output_dir) / f"interventional_test_predictions{suffix}.csv"), index=False)
    plt.figure()
    sns.scatterplot(data=df, x="injected", y="predictions")
    plt.xlabel("injected value"); plt.ylabel("prediction")
    plt.savefig(str(Path(output_dir) / f"interventional_test_scatter{suffix}.png"))
    plt.close()
    return metrics


def refit_regression_head(
    model: IntrospectionRegressionModel,
    fit_dataset: Dataset,
    dev_dataset: Dataset,
    eval_dataset: Dataset,
    collator: LoRAGatingCollator,
    config: ExperimentConfig,
    output_dir: str,
    estimator: str,
    hyperparam_grid,
):
    """
    Freeze everything and fit the optimal LINEAR readout (estimator = ridge or
    pls) from the frozen representation -> true activation. The hyperparameter
    (ridge lambda / pls n_components) is selected on dev by correlation; the
    final probe is fit on train+dev and:
      1. evaluated ENDOGENOUSLY on test (predict true activation, no intervention)
      2. evaluated INTERVENTIONALLY on test (inject gaussian-distributed values
         into the neuron and check whether this SAME endogenously-fit direction
         tracks the INJECTED value -> causal-transfer / faithfulness).
    The selected probe is loaded into model.regression_head.
    """
    fit_fn = FITTERS[estimator]

    # snapshot the CURRENT (intervention-trained, "causal") head direction before
    # the refit probe overwrites it, so we can compare the two readout directions
    causal_coef = model.regression_head.weight.detach().flatten().float().cpu().clone()

    # extract frozen endogenous features once per split (no intervention)
    Xtr, ytr = _extract_readout_features(model, fit_dataset, collator, config)
    Xdev, ydev = _extract_readout_features(model, dev_dataset, collator, config)
    Xte, yte = _extract_readout_features(model, eval_dataset, collator, config)

    # ---- hyperparam sweep: fit on train, select on dev ----
    print(f"[refit_regression_head] estimator={estimator}; sweep (fit train, select dev corr):")
    dev_sweep = []
    best_hp = hyperparam_grid[0]
    best_score = -float("inf")
    for hp in hyperparam_grid:
        probe = fit_fn(Xtr, ytr, hp)
        dev_metrics = compute_metrics_from_numericals(probe.predict(Xdev).numpy(), ydev.numpy())
        dev_sweep.append({"hyperparam": hp, "dev_corr": dev_metrics["corr"], "dev_l2": dev_metrics["l2_dist"]})
        print(f"  hp={hp:<10g} dev_corr={dev_metrics['corr']:.4f}  dev_l2={dev_metrics['l2_dist']:.4f}")
        score = _corr_score(dev_metrics)
        if score > best_score:
            best_score, best_hp = score, hp
    print(f"[refit_regression_head] best hyperparam={best_hp:g} (dev_corr={best_score:.4f})")

    # ---- final fit on train+dev with best hyperparam ----
    Xfit = torch.cat([Xtr, Xdev], dim=0)
    yfit = torch.cat([ytr, ydev], dim=0)
    probe = fit_fn(Xfit, yfit, best_hp)

    # load the selected probe into the model head (keep head dtype/device)
    head = model.regression_head
    with torch.no_grad():
        head.weight.copy_(probe.coef.view(1, -1).to(head.weight.dtype).to(head.weight.device))
        head.bias.copy_(torch.tensor([probe.intercept]).to(head.bias.dtype).to(head.bias.device))

    # ---- (1) endogenous test eval ----
    endo_metrics = compute_metrics_from_numericals(probe.predict(Xte).numpy(), yte.numpy())
    print(f"  test ENDOGENOUS   corr={endo_metrics['corr']:.4f}  l2={endo_metrics['l2_dist']:.4f}")

    # ---- (2) interventional (causal-transfer) test eval, gaussian injection ----
    prev_dist, prev_scale = model.intervention_distribution, model.intervention_scale
    model.intervention_distribution = "gaussian"
    model.intervention_scale = 1.0   # endogenous-matched, regardless of the annealed training scale
    model.seed_intervention_rng(config.seed)
    Xiv, yiv = _extract_readout_features(model, eval_dataset, collator, config, do_intervention=True)
    model.intervention_distribution, model.intervention_scale = prev_dist, prev_scale
    interv_metrics = compute_metrics_from_numericals(probe.predict(Xiv).numpy(), yiv.numpy())
    print(f"  test INTERVENTION corr={interv_metrics['corr']:.4f}  l2={interv_metrics['l2_dist']:.4f}  (gaussian)")

    # ---- (a) perturbation diagnostic: how much does the injection actually move
    # the readout? Xiv and Xte are per-example aligned (same eval_dataset, no
    # shuffle), so (Xiv - Xte) is exactly the injection's effect on the readout.
    # If this is ~0, the readout position is blind to the neuron and interventional
    # corr ~0 is a readout-location problem, NOT proof of contextual bypass.
    dX = (Xiv - Xte).norm(dim=1)          # [N] per-example ||ΔX||
    base = Xte.norm(dim=1).clamp_min(1e-8)
    rel = dX / base
    perturbation = {
        "delta_readout_norm_mean": float(dX.mean()),
        "delta_readout_norm_median": float(dX.median()),
        "readout_norm_mean": float(Xte.norm(dim=1).mean()),
        "relative_delta_mean": float(rel.mean()),
        "relative_delta_median": float(rel.median()),
    }
    print(f"  intervention Δreadout: |ΔX|={perturbation['delta_readout_norm_mean']:.4g} "
          f"(relative {perturbation['relative_delta_mean']:.3g} of "
          f"|X|={perturbation['readout_norm_mean']:.4g})")

    # ---- direction comparison: is the causal (intervention-trained) readout the
    # same direction as the endogenous (bypass-prone) refit? ~0 => two separate
    # mechanisms; ~1 => the same direction serves both.
    cos_causal_refit = float(nn.functional.cosine_similarity(
        causal_coef.unsqueeze(0), probe.coef.unsqueeze(0)).item())
    print(f"  cosine(causal head, refit head) = {cos_causal_refit:+.4f}")

    out = {
        "estimator": estimator,
        "hyperparam_grid": list(hyperparam_grid),
        "best_hyperparam": best_hp,
        "dev_sweep": dev_sweep,
        "endogenous_test": endo_metrics,
        "interventional_test": interv_metrics,
        "readout_perturbation": perturbation,
        "cosine_causal_vs_refit": cos_causal_refit,
    }
    with open(str(Path(output_dir) / "refit_head_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    pd.DataFrame({"predictions": probe.predict(Xte).numpy(), "activation": yte.numpy()}).to_csv(
        str(Path(output_dir) / "refit_head_predictions_endogenous.csv"), index=False)
    pd.DataFrame({"predictions": probe.predict(Xiv).numpy(), "injected": yiv.numpy()}).to_csv(
        str(Path(output_dir) / "refit_head_predictions_interventional.csv"), index=False)
    plt.figure()
    sns.scatterplot(x=yte.numpy(), y=probe.predict(Xte).numpy())
    plt.xlabel("true activation"); plt.ylabel("prediction")
    plt.savefig(str(Path(output_dir) / "refit_head_endogenous_scatter.png"))
    plt.close()
    plt.figure()
    sns.scatterplot(x=yiv.numpy(), y=probe.predict(Xiv).numpy())
    plt.xlabel("injected value"); plt.ylabel("prediction")
    plt.savefig(str(Path(output_dir) / "refit_head_interventional_scatter.png"))
    plt.close()
    return out



# ============================================================
# Custom Trainer
# ============================================================

class InterventionTrainer(Trainer):

    def __init__(
        self,
        *args,
        signal_ranges,
        tokenizer: AutoTokenizer,
        experiment_config: ExperimentConfig,
        do_intervention: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.signal_ranges = signal_ranges
        self.tokenizer = tokenizer
        self.experiment_config = experiment_config
        self.do_intervention = do_intervention

    def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            **kwargs,
    ):
        # clone to avoid HF mutation weirdness
        inputs = inputs.copy()
        # ====================================================
        # REGRESSION MODE
        # ====================================================
        if self.experiment_config.use_regressor:
            # labels should be original activations. When intervention is on,
            # model.forward overwrites them with the sampled intervention value
            # internally; when off, they stay as the true activation target.
            inputs["labels"] = inputs["activation"]

        # ====================================================
        # TEXT GENERATION MODE
        # ====================================================
        else:
            # standard LM labels
            pass

        if not self.do_intervention:
            inputs["do_intervention"] = False

        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        # for the text-generation model there is no regression head /
        # intervention_values, so use the default HF prediction path
        if not self.experiment_config.use_regressor:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

        # make evaluation/predict, use EXACT SAME LOGIC as compute_loss
        inputs = inputs.copy()
        inputs["labels"] = inputs["activation"]
        if not self.do_intervention:
            inputs["do_intervention"] = False

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs.loss.detach()
        preds = outputs.logits.detach().float()
        if self.do_intervention:
            # IMPORTANT: compare against the intervened values, NOT original activations
            labels = outputs.intervention_values.detach().float()
        else:
            # sanity-check mode: target is the true, un-intervened activation
            labels = inputs["activation"].detach().float()
        return loss, preds, labels

# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    config = ExperimentConfig.model_validate_json(args.config.read_text())

    cache_dir = Path(config.cache_dir) / config.experiment_name
    train_path = cache_dir / "train"
    dev_path = cache_dir / "dev"
    test_path = cache_dir / "test"
    output_dir = Path(config.output_dir) / f"{config.experiment_name}"

    if (output_dir / "refit_head_predictions_scatter.png").exists() and not args.overwrite_cache:
        print(f"Experiments ran already, skipping.. {output_dir}")
        return

    # update tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {"additional_special_tokens": [INTROSPECTION_SEPARATOR]}
    tokenizer.add_special_tokens(special_tokens)

    # ========================================================
    # Load datasets
    # ========================================================

    train_dataset_raw = load_from_disk(str(train_path))
    dev_dataset_raw = load_from_disk(str(dev_path))
    test_dataset_raw = load_from_disk(str(test_path))

    # ========================================================
    # Compute signal ranges
    # ========================================================
    df = train_dataset_raw.to_pandas() # TODO might be computationally expensive, could load cache stats from data
    grouped = df.groupby(["signal_layer", "signal_index"])["activation"]
    signal_ranges = {}
    for key, vals in grouped:
        signal_ranges[key] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }

    # ========================================================
    # Preprocess datasets
    # ========================================================
    def preprocess(ex, for_prediction=False, keep_signal_specs=True):
        out = preprocess_example(
            ex,
            tokenizer=tokenizer,
            prompt_fn=PROMPT_REGISTRY[config.training_config.prompt_type],
            max_length=config.training_config.max_seq_length,
            return_frozen_input_len=True,
            for_prediction=for_prediction,
        )
        if keep_signal_specs:
            out["token_position"] = ex["token_position"]
            out["signal_layer"] = ex["signal_layer"]
            out["signal_index"] = ex["signal_index"]
            if config.use_regressor:
                out["activation"] = ex["activation"]
        return out
    
    training_preprocess_func = partial(preprocess, for_prediction=True) if config.use_regressor else preprocess
    train_dataset = train_dataset_raw.map(
        training_preprocess_func,
        remove_columns=train_dataset_raw.column_names,
    )
    dev_dataset = dev_dataset_raw.map(
        training_preprocess_func,
        remove_columns=dev_dataset_raw.column_names,
    )
    test_dataset = test_dataset_raw.map(
        partial(preprocess, keep_signal_specs=True),
        remove_columns=test_dataset_raw.column_names,
    )
    test_dataset_pred = test_dataset_raw.map(
        partial(preprocess, for_prediction=True, keep_signal_specs=True),
        remove_columns=test_dataset_raw.column_names,
    )
    
    # Load model add peft
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.training_config.bf16 else torch.float32,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=config.training_config.lora_r,
        lora_alpha=config.training_config.lora_alpha,
        target_modules=config.training_config.lora_target_modules,
        lora_dropout=config.training_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.resize_token_embeddings(len(tokenizer))

    if config.use_regressor:
        model = IntrospectionRegressionModel(
            model,
            signal_ranges=signal_ranges,
            intervention_distribution=config.training_config.intervention_signal_distribution,
            intervention_extreme_scale=config.training_config.intervention_extreme_scale,
        )
        # model.__class__ = IntrospectionRegressionModel
        # model.regression_head = nn.Linear(model.config.hidden_size,1,bias=False,)

    # Token-wise LoRA gating
    replace_model_with_gated_version(model)
    replace_lora_with_gated_version(model)


    collator = LoRAGatingCollator(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    pred_collator = LoRAGatingCollator(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
        for_prediction=True
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=config.experiment_name,
        learning_rate=config.training_config.learning_rate,
        per_device_train_batch_size=config.training_config.train_batch_size,
        per_device_eval_batch_size=config.training_config.eval_batch_size,
        num_train_epochs=config.training_config.num_epochs,
        logging_steps=math.ceil(100/config.training_config.train_batch_size),
        eval_strategy="steps",
        eval_steps=math.ceil(400//config.training_config.train_batch_size),
        save_strategy="epoch",
        bf16=config.training_config.bf16,
        remove_unused_columns=False,
        report_to="wandb",
    )

    callbacks = None
    if config.use_regressor:
        # persist the custom regression head next to every checkpoint (PEFT won't)
        callbacks = [RegressionHeadSaveCallback(model)]
        if config.training_config.intervention_anneal:
            callbacks.append(InterventionAnnealCallback(
                model,
                scale_start=config.training_config.intervention_scale_start,
                scale_end=config.training_config.intervention_scale_end,
                schedule=config.training_config.intervention_anneal_schedule,
                start_frac=config.training_config.intervention_anneal_start_frac,
                end_frac=config.training_config.intervention_anneal_end_frac,
                log_every=math.ceil(400 / config.training_config.train_batch_size),
            ))

    trainer = InterventionTrainer(
        model=model,
        tokenizer=tokenizer,
        signal_ranges=signal_ranges,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=pred_collator if config.use_regressor else collator,
        compute_metrics=partial(compute_metrics,tokenizer=tokenizer),
        preprocess_logits_for_metrics=None if config.use_regressor else get_argmax_from_logits,
        experiment_config=config,
        do_intervention=not args.no_intervention,
        callbacks=callbacks,
    )

    ckpt_dirs = glob.glob(f"{output_dir}/checkpoint-*")
    if ckpt_dirs and not args.overwrite_cache:
        print(f"Found Trained model, skipping training and loading directly ..")
        latest_checkpoint_path = max(ckpt_dirs, key=os.path.getctime)
        # resume loads the adapter/embeddings; the regression head is NOT in the
        # PEFT checkpoint, so restore it explicitly afterwards
        train_results = trainer.train(resume_from_checkpoint=latest_checkpoint_path)
        if config.use_regressor:
            load_regression_head(model, latest_checkpoint_path)
    else:
        train_results = trainer.train()
        trainer.save_metrics( "train", train_results.metrics)
    # seed the intervention sampler so eval draws are reproducible across reloads
    if config.use_regressor:
        model.seed_intervention_rng(config.seed)
    dev_results = trainer.evaluate()
    trainer.save_metrics("dev", dev_results,)

    # Eval with default trainer setup (if train with intervention then test with intervention)
    if not (output_dir / "test_predictions.json").exists() or args.overwrite_cache:
        if config.use_regressor:
            model.seed_intervention_rng(config.seed)
        evaluate_test_split(
            config,
            model,
            pred_collator,
            test_dataset,
            test_dataset_pred,
            test_dataset_raw,
            tokenizer,
            train_dataset_raw,
            trainer
        )

    # alternative eval setting (flip the intervention flag)
    trainer.do_intervention = args.no_intervention
    if not (output_dir / "test_predictions_alt_intervention.json").exists() or args.overwrite_cache:
        if config.use_regressor:
            model.seed_intervention_rng(config.seed)
        evaluate_test_split(
            config,
            model,
            pred_collator,
            test_dataset,
            test_dataset_pred,
            test_dataset_raw,
            tokenizer,
            train_dataset_raw,
            trainer,
            suffix="_alt_intervention"
    )

    # Interventional test eval of the TRAINED head with the endogenous-matched
    # (gaussian, k=1) injection distribution. Contextual bypass scores 0 here by
    # construction, so this is the non-gameable causal-reading metric. Must run
    # BEFORE --refit-head, which overwrites the trained head.
    if config.use_regressor:
        evaluate_interventional_test(
            model,
            eval_dataset=test_dataset_pred,
            collator=pred_collator,
            config=config,
            output_dir=output_dir,
            distribution="gaussian",
            suffix="_gaussian",
        )

    # Post-hoc diagnostic: solve for the optimal linear head on the normal
    # activation range given the frozen representation (mutates the head, so run
    # it LAST, after the standard evals).
    if args.refit_head and config.use_regressor:
        if args.refit_estimator == "pls":
            hyperparam_grid = [int(x) for x in args.refit_pls_components.split(",")]
        else:
            hyperparam_grid = [float(x) for x in args.refit_lambda_grid.split(",")]
        refit_regression_head(
            model,
            fit_dataset=train_dataset,       # for_prediction=True, has "activation"
            dev_dataset=dev_dataset,         # for_prediction=True, has "activation"
            eval_dataset=test_dataset_pred,  # for_prediction=True, has "activation"
            collator=pred_collator,
            config=config,
            output_dir=output_dir,
            estimator=args.refit_estimator,
            hyperparam_grid=hyperparam_grid,
        )

if __name__ == "__main__":
    main()

