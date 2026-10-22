#!/usr/bin/env python3
"""
Generate BLINDED steered-vs-unsteered text samples for SAE features, for a human /
agent judge to evaluate.

This is the steerability gate (feature-selection criterion 3), but the judging is
done OUT OF BAND -- NOT by any API call from this script. The script only produces
artifacts:

  * for each candidate feature, take neutral prompts and generate continuations
    with the feature steered OFF (natural) and steered ON at a few strengths;
  * write a blinded judging file (`*.judge.md`): per prompt the continuations are
    shuffled and labeled Option A/B/C..., the feature's concept (Neuronpedia
    description) is shown, but WHICH option was steered is hidden;
  * write the answer key separately (`*.key.json`) so the judge cannot peek;
  * write the raw records (`*.samples.jsonl`) for reference.

A feature QUALIFIES if the judge, blind, consistently identifies the steered
continuations (especially the stronger ones) as the ones expressing the concept,
and the unsteered baseline as not. The passing features are the ones we train on.

Steering is difference-of-decode (SAEBundle.steer_delta) applied at EVERY position
each forward, so the concept persists through generation.

Run on the cluster (needs torch + sae_lens + the 2b model):

  python data_prep/generate_sae_steering_samples.py \
      --annotated data/sae_stats/gemma2-2b_res9_16k.annotated.parquet \
      --model google/gemma-2-2b --layer 9 --width 16k --site res \
      --n-prompts 8 --strengths 4 8 --max-new-tokens 40 \
      --max-features 20 --out-dir data/sae_steering_samples
"""
from __future__ import annotations

import sys
import argparse
import json
import operator
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from src.introspector.models.signal_paths import resolve_sae_hook_path
from src.introspector.signals.sae import load_sae


class SAESteerAllPositionsHook:
    """Clamp ONE SAE feature to `target_value` at EVERY position each forward, via
    difference-of-decode (bundle.steer_delta). Persists the concept through
    generation (the prompt pass clamps all prompt tokens; each decode step clamps the
    one new token under the KV cache). Handles decoder outputs that are a bare tensor
    or a (hidden, ...) tuple, like training/model.py:SAEInterventionHook."""

    def __init__(self, bundle, feature_index, target_value):
        self.bundle = bundle
        self.feature_index = feature_index
        self.target_value = target_value

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = (output[0] if is_tuple else output).clone()
        delta = self.bundle.steer_delta(hidden, self.feature_index, self.target_value)
        hidden = hidden + delta.to(hidden.dtype)
        return (hidden,) + tuple(output[1:]) if is_tuple else hidden


def _on_magnitude(row):
    """PEAK 'on' activation to scale steering strengths by. Steering to a small
    multiple of the feature's *mean* active value (~4-5) barely perturbs a residual of
    norm ~127 and produces no visible concept, so we anchor to the feature's PEAK
    activation instead -- Neuronpedia's maxActApprox, else our own active_max. Doses
    are then multiples of that peak (e.g. 5x, 10x, 20x = clamp well above natural)."""
    for key in ("np_max_act", "active_max", "active_mean"):
        v = row.get(key)
        if v is not None and np.isfinite(v) and float(v) > 0:
            return float(v)
    return 1.0


@torch.no_grad()
def generate_continuations(model, module, bundle, tokenizer, batch, feature_index,
                           target, max_new_tokens):
    """Greedy continuations for a batch of (left-padded) prompts. If target is None,
    no steering (baseline); else clamp the feature to `target` at all positions."""
    handle = None
    if target is not None:
        hook = SAESteerAllPositionsHook(bundle, feature_index, target)
        handle = module.register_forward_hook(hook)
    try:
        out = model.generate(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
    finally:
        if handle is not None:
            handle.remove()
    gen = out[:, batch["input_ids"].shape[1]:]
    return [tokenizer.decode(g, skip_special_tokens=True) for g in gen]


def build_prompt_pool(args):
    """A POOL of short neutral lead-ins. Each feature later draws its own random
    subset (seeded per feature) so the deterministic unsteered baseline text is NOT
    shared across features -- otherwise a judge can spot the baseline by recognizing
    the same string repeated in every feature block (the blinding leak)."""
    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    texts = []
    for ex in ds:
        t = (ex.get("text") or "").strip()
        if len(t.split()) >= args.prompt_words:
            texts.append(" ".join(t.split()[:args.prompt_words]))
        if len(texts) >= args.prompt_pool:
            break
    return texts


def tokenize_prompts(tokenizer, texts, args, device):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=args.prompt_words + 4, padding_side="left")
    return {k: v.to(device) for k, v in enc.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotated", type=Path, required=True,
                   help="Enriched parquet from data_prep/annotate_sae_features_neuronpedia.py "
                        "(must carry explanation + feature columns).")
    p.add_argument("--model", default="google/gemma-2-2b",
                   help="MUST be the base model the SAE was trained on (not -it).")
    p.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--width", default="16k")
    p.add_argument("--site", default="res", choices=["res", "mlp", "attn"])
    p.add_argument("--dataset", default="allenai/c4")
    p.add_argument("--dataset-config", default="en")
    p.add_argument("--split", default="validation")
    p.add_argument("--n-prompts", type=int, default=8,
                   help="Prompts sampled PER FEATURE from the pool (varies per feature "
                        "to avoid a shared, recognizable baseline).")
    p.add_argument("--prompt-pool", type=int, default=64,
                   help="Size of the shared prompt pool each feature draws its subset from.")
    p.add_argument("--prompt-words", type=int, default=8,
                   help="Words of neutral lead-in per prompt.")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--strengths", type=float, nargs="+", default=[5, 10, 20],
                   help="Steering multiples of the feature's PEAK activation to sample.")
    p.add_argument("--features", type=int, nargs="+", default=None,
                   help="Explicit feature indices to sample (overrides --max-features "
                        "ranking). Use for targeted calibration on known-semantic features.")
    p.add_argument("--max-features", type=int, default=20,
                   help="Cap number of explained features sampled (when --features unset).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("data/sae_steering_samples"))
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[steer-gen] loading base model {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()

    print(f"[steer-gen] loading SAE {args.sae_release} layer={args.layer} width={args.width}")
    bundle = load_sae(release=args.sae_release, site=args.site, width=args.width,
                      layer=args.layer, device=device, dtype=torch.float32)
    module_path = resolve_sae_hook_path(model, args.layer, site=args.site, path_to_output=False)
    module = operator.attrgetter(module_path)(model)
    print(f"[steer-gen] steering module = {module_path}")

    df = pd.read_parquet(args.annotated)
    if "explanation" not in df.columns:
        raise SystemExit(f"{args.annotated} has no 'explanation' column; run "
                         f"data_prep/annotate_sae_features_neuronpedia.py (Step 1.5) first.")
    cand = df[df.get("has_explanation", False) == True].copy()
    if args.features:
        cand = cand[cand["feature"].isin(args.features)]
        missing = set(args.features) - set(cand["feature"].tolist())
        if missing:
            print(f"[steer-gen] WARNING: requested features not explained/selected: {sorted(missing)}")
    else:
        if "proj_pos_at_fixed_pos" in cand.columns:
            cand = cand.sort_values("proj_pos_at_fixed_pos", ascending=False)
        cand = cand.head(args.max_features)
    print(f"[steer-gen] sampling {len(cand)} explained features")

    pool = build_prompt_pool(args)
    print(f"[steer-gen] prompt pool: {len(pool)} lead-ins; each feature draws {args.n_prompts}")

    stem = f"{args.model.split('/')[-1]}_res{args.layer}_{args.width}"
    samples_path = args.out_dir / f"{stem}.samples.jsonl"
    judge_path = args.out_dir / f"{stem}.judge.md"
    key_path = args.out_dir / f"{stem}.key.json"

    key = {}
    judge_md = ["# SAE steering — blind judging\n",
                "For each feature you are given its CONCEPT and, per prompt, several\n"
                "unlabeled continuation options. Mark which option(s) express the\n"
                "concept. Do NOT assume any ordering. Prompts differ per feature and the\n"
                "steered/unsteered mapping is withheld (separate key) so judging stays\n"
                "blind -- judge each option on the concept alone, not by comparing across\n"
                "features.\n"]

    with open(samples_path, "w") as sf:
        for _, row in cand.iterrows():
            feat = int(row["feature"])
            concept = str(row.get("explanation"))
            on_mag = _on_magnitude(row.to_dict())
            rng = random.Random(args.seed + feat)

            feat_prompts = rng.sample(pool, min(args.n_prompts, len(pool)))
            batch = tokenize_prompts(tokenizer, feat_prompts, args, device)

            conditions = [("baseline", None)] + [
                (f"steer_x{s:g}", float(s) * on_mag) for s in args.strengths]
            texts_by_cond = {name: generate_continuations(
                model, module, bundle, tokenizer, batch, feat, target, args.max_new_tokens)
                for name, target in conditions}

            key[str(feat)] = {"concept": concept, "on_magnitude": on_mag, "prompts": []}
            judge_md.append(f"\n## Feature S{args.layer}.{feat}\n")
            judge_md.append(f"**Concept:** {concept}\n")

            for pi, prompt in enumerate(feat_prompts):
                options = [(name, texts_by_cond[name][pi]) for name, _ in conditions]
                rng.shuffle(options)
                labels = [chr(ord("A") + i) for i in range(len(options))]
                mapping = {lab: name for lab, (name, _) in zip(labels, options)}
                key[str(feat)]["prompts"].append({"prompt": prompt, "mapping": mapping})

                judge_md.append(f"\n**Prompt {pi+1}:** …{prompt}\n")
                for lab, (_, text) in zip(labels, options):
                    judge_md.append(f"- **{lab}.** {text.strip()!r}")

                sf.write(json.dumps({
                    "feature": feat, "concept": concept, "prompt": prompt,
                    "on_magnitude": on_mag,
                    "continuations": {name: texts_by_cond[name][pi] for name, _ in conditions},
                    "blind_mapping": mapping,
                }) + "\n")

            print(f"  feat {feat:>6}  concept={concept[:50]!r}")

    judge_path.write_text("\n".join(judge_md))
    key_path.write_text(json.dumps(key, indent=2))
    print(f"\n[steer-gen] wrote:\n  {judge_path}  (blind — give this to the judge)\n"
          f"  {key_path}      (answer key — withhold from the judge)\n"
          f"  {samples_path}  (raw records)")
    print("[steer-gen] next: an agent/human reads the .judge.md blind, picks the "
          "concept-expressing option(s) per prompt; features whose STEERED options are "
          "consistently chosen (and baseline not) qualify.")


if __name__ == "__main__":
    main()
