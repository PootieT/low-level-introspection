#!/usr/bin/env python3
"""
Seed-replicate configs (rebuttal: "no seeds"): copies of EXISTING run configs that differ
only in experiment name, output dir and TrainingConfig.train_seed.

Copying rather than regenerating is the point. A regenerated config would hash to a new
cache name (see data_prep/pin_dataset_cache.py) and could drift in any default that changed
since the original grid ran; the SFT and joint arms also differ in more than the loss
weight (the SFT grid used its own 10-neuron cache, a cosine schedule and 3% warm-up). A
copy keeps every field -- dataset_name, signal_filter, LR schedule, ExperimentConfig.seed
(which fixes the evaluation) -- byte-for-byte, so the seed replicates train on the same
cache and are scored on the same injected values as the run they replicate. The original
run stays where it is and counts as one more (unseeded) replicate.

Names: <original experiment_name>_seed<S>, written to experiment_configs/<dest-group>/ with
output_dir outputs/runs/<dest-group>. The SCC script picks the loss weight / KL budget from
the name prefix (joint_ / endo_), exactly as the original grid scripts passed them.

Usage
-----
python data_prep/make_seed_configs.py \\
    --source mlp_joint_grid_layer_token:'joint_*_gateoff_scale10_kl0.1_w0.01_300K' \\
    --source mlp_endo_sweep_layer_token:'endo_*_gateoff_w1.0_300K' \\
    --cells 2:-1,2:-28,8:-1,8:-28 \\
    --neurons 2.1447,2.1886,2.2566,2.5949,2.6998,8.3852,8.4144,8.4956,8.5777,8.6370 \\
    --seeds 1,2,3,4
"""
from __future__ import annotations

import sys
import json
import argparse
import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.introspector.config import ExperimentConfig  # noqa: E402
from src.introspector.run_discovery import resolve_locale, resolve_position  # noqa: E402

DEST_GROUP = "mlp_seed_variance_layer_token"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", required=True,
                   help="'<group>:<config-name glob>' of runs to replicate; repeatable.")
    p.add_argument("--cells", required=True,
                   help="Comma list of 'layer:position', positions as in the run names.")
    p.add_argument("--neurons", required=True, help="Comma list of 'layer.index'.")
    p.add_argument("--seeds", default="1,2,3,4", help="Comma list of train seeds.")
    p.add_argument("--dest-group", default=DEST_GROUP)
    p.add_argument("--config-root", type=Path, default=ROOT / "experiment_configs")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_pairs(spec: str, sep: str) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in (t.strip().split(sep) for t in spec.split(",") if t.strip())}


def seed_copy(raw: dict, seed: int, dest_group: str) -> dict:
    """The original config with only name, output dir and train_seed changed."""
    out = json.loads(json.dumps(raw))                     # deep copy
    out["experiment_name"] = f"{raw['experiment_name']}_seed{seed}"
    out["output_dir"] = f"outputs/runs/{dest_group}"
    out.setdefault("training_config", {})["train_seed"] = int(seed)
    return out


def main():
    args = parse_args()
    cells, neurons = parse_pairs(args.cells, ":"), parse_pairs(args.neurons, ".")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    dest = args.config_root / args.dest_group
    dest.mkdir(parents=True, exist_ok=True)
    picked, written, kept = [], 0, 0
    for spec in args.source:
        group, glob = spec.split(":", 1)
        for path in sorted((args.config_root / group).glob("*.json")):
            if path.name.startswith("_") or not fnmatch.fnmatch(path.stem, glob):
                continue
            raw = json.loads(path.read_text())
            cfg = ExperimentConfig.model_validate(raw)
            if cfg.training_config.train_seed is not None:
                raise SystemExit(f"{path}: already a seed replicate")
            _, layer, index = resolve_locale(cfg)
            position, _ = resolve_position(cfg)
            if (layer, position) in cells and (layer, index) in neurons:
                picked.append((path, raw))
    for path, raw in picked:
        for s in seeds:
            new = seed_copy(raw, s, args.dest_group)
            ExperimentConfig.model_validate(new)              # still loads
            out = dest / f"{new['experiment_name']}.json"
            if out.exists() and not args.overwrite:
                kept += 1
                continue
            out.write_text(json.dumps(new, indent=2) + "\n")
            written += 1
    by_group = {}
    for path, _ in picked:
        by_group[path.parent.name] = by_group.get(path.parent.name, 0) + 1
    print(f"[seeds] {len(picked)} source runs {by_group} x {len(seeds)} seeds -> {dest}: "
          f"{written} written, {kept} already there")
    if not picked:
        raise SystemExit("no source config matched --cells / --neurons")


if __name__ == "__main__":
    main()
