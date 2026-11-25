#!/usr/bin/env python3
"""
Cross-model configs: model B trained to predict model A's neuron from the text alone.

Rebuttal (privileged access): an endogenous correlation of ~0.9 could come from the text
rather than from reading the model's own state. The test, as in Li et al. (2025): train
ANOTHER model on exactly the same rows -- same texts, same cached activations of model
A, same test split -- and compare it with A's own (self) introspector. The self runs
already exist (each model's SFT grid); this script writes the off-diagonal configs.

A cross config is the SOURCE run's config (its cache, signal, token selector, splits,
seed -- so the rows are identical and self vs cross is paired row by row) with the
PREDICTOR's model and training_config (its own tuned LR / batch / LoRA, as in its own
SFT grid), plus:
  signal_model_name   the source model. Marks the run as cross-model: the trainer then
                      trains SFT only, evaluates endogenously only, never injects and
                      never builds a cache, and cache_signals refuses the config.
  max_seq_length      raised to --max-seq-length (default 128). The source's text is
                      re-tokenized by the predictor and can come out longer (Qwen3 on
                      Llama's 31-token windows: up to 72 text tokens, 96 with the
                      prompt), and truncation cuts the PROMPT, so the readout would sit
                      mid-prompt. Raising max_length only adds padding.

Usage
-----
python data_prep/make_cross_model_configs.py \\
    --predictor-config experiment_configs/mlp_endo_sweep_layer_token_qwen3_06b_hyper_tuned/<any run>.json \\
    --predictor-key qwen06b --source-key llama1b \\
    --source-configs experiment_configs/mlp_endo_sweep_layer_token/endo_N2.1447_POS-1_gateoff_w1.0_300K.json ...
"""
from __future__ import annotations

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from src.introspector.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUP = "mlp_cross_model_sft"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-configs", type=Path, nargs="+", required=True,
                   help="The source model's own (self) run configs, one per signal.")
    p.add_argument("--predictor-config", type=Path, required=True,
                   help="Any config of the predictor's own SFT grid: supplies model_name "
                        "and training_config (LR, batch, LoRA, gate).")
    p.add_argument("--source-key", required=True, help="Short model key in run names, e.g. llama1b.")
    p.add_argument("--predictor-key", required=True, help="Short model key, e.g. qwen06b.")
    p.add_argument("--group", default=DEFAULT_GROUP)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--config-root", type=Path, default=ROOT / "experiment_configs")
    p.add_argument("--output-root", default="outputs/runs",
                   help="Runs go to <output-root>/<group>/<experiment_name>.")
    return p.parse_args()


def cross_name(source_name: str, source_key: str, predictor_key: str) -> str:
    """'endo_N2.1447_POS-1_gateoff_w1.0_300K' -> 'cross_qwen06b_from_llama1b_N2.1447_POS-1_
    gateoff_w1.0_300K' (the source's signal/position/gate/weight tags carried over)."""
    stem = source_name.split("_", 1)[1] if source_name.startswith("endo_") else source_name
    return f"cross_{predictor_key}_from_{source_key}_{stem}"


def make_cross_config(source: ExperimentConfig, predictor: ExperimentConfig, *,
                      name: str, output_dir: str, max_seq_length: int) -> ExperimentConfig:
    if source.model_name == predictor.model_name:
        raise SystemExit(f"{name}: source and predictor are both {source.model_name}; the "
                         f"self run already exists in that model's own grid")
    if not source.dataset_name:
        raise SystemExit(f"{source.experiment_name}: no dataset_name, so the source cache "
                         f"cannot be located from a different model's config")
    if getattr(source, "signal_model_name", None):
        raise SystemExit(f"{source.experiment_name} is itself a cross-model config")
    s_tc, p_tc = source.training_config, predictor.training_config
    if s_tc.prompt_type != p_tc.prompt_type:
        raise SystemExit(f"{name}: prompt_type differs ({s_tc.prompt_type} vs "
                         f"{p_tc.prompt_type}); self and cross would see different prompts")
    if s_tc.disable_adapter_in_input != p_tc.disable_adapter_in_input:
        raise SystemExit(f"{name}: LoRA input gate differs between the source run and the "
                         f"predictor's grid; self and cross must share it")
    tc = p_tc.model_copy(update={"max_seq_length": max(max_seq_length, p_tc.max_seq_length)})
    return source.model_copy(update={
        "experiment_name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": predictor.model_name,
        "load_in_4bit": predictor.load_in_4bit,
        "load_in_8bit": predictor.load_in_8bit,
        "signal_model_name": source.model_name,
        "training_config": tc,
        "output_dir": output_dir,
    })


def main():
    args = parse_args()
    predictor = ExperimentConfig.model_validate_json(args.predictor_config.read_text())
    out_dir = args.config_root / args.group
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.source_configs:
        source = ExperimentConfig.model_validate_json(Path(path).read_text())
        name = cross_name(source.experiment_name, args.source_key, args.predictor_key)
        cfg = make_cross_config(source, predictor, name=name,
                                output_dir=f"{args.output_root}/{args.group}",
                                max_seq_length=args.max_seq_length)
        dst = out_dir / f"{name}.json"
        dst.write_text(json.dumps(json.loads(cfg.model_dump_json()), indent=2))
        print(f"[cross] {dst}  ({cfg.model_name} <- {cfg.signal_model_name}, "
              f"cache {cfg.dataset_name}, signal {cfg.signal_filter})")


if __name__ == "__main__":
    main()
