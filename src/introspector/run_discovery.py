"""
Find the runs of a sweep, and shard them across array tasks.

WHY THIS EXISTS
---------------
The circuit analyses are all "run the same measurement on every checkpoint of a
sweep and aggregate per cell". Each of them previously needed its own hard-coded
list of conditions (see the CONDITIONS literal that compare_lora_weight_patterns.py
used to carry), which is fine for four pre-flight runs and hopeless for the 312-run
(layer x position x neuron) grid.

A sweep is discovered from its EXPERIMENT CONFIGS, not from its output directories:
the config is what names the signal, the token selector and the cache, so a run whose
checkpoint is missing is a reportable gap rather than an invisible one. Configs whose
name starts with "_" are the sweep's shared cell/pool caches, not runs, and are
skipped.

LOCALE. Every analysis needs the (layer, index, position) the run was trained on.
`signal_filter` is the authority when it is set -- a grid run trains ONE neuron out of
a shared 5-neuron cell cache, so config.train_signals lists all five and only the
filter says which one this run is. Falls back to train_signals when there is exactly
one.

SHARDING is 1-based to match SGE's $SGE_TASK_ID, and contiguous rather than strided so
a partial array leaves a contiguous prefix of the sweep done.

NOT analysis/run_catalog.py, which is the other half of the same story and should be
used instead wherever it fits. That module locates runs by their OUTPUT directories and
reads the metrics they already produced (corr, drift, KL) into one cross-sweep frame,
with pandas as its only dependency, so it works on a laptop holding nothing but synced
CSVs. This one locates runs by their CONFIGS and hands back a checkpoint to measure on,
which needs torch. Rule of thumb: reading finished results -> run_catalog; computing new
ones -> here. An aggregator over the per-run files these analyses write
(<run>/circuit_trace/, /injection_specificity/, ...) belongs on the run_catalog side:
they land under outputs/runs/<group>/<run>/, exactly where read_metrics already looks.
"""
from __future__ import annotations

import os
import glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import ExperimentConfig
from .dataset_utils import parse_signal_filter

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "experiment_configs"
# Shared cell/pool caches a sweep builds once and every run draws from. They are
# configs, but they are not runs and have no checkpoint.
SHARED_CONFIG_PREFIX = "_"


@dataclass
class RunSpec:
    """One trained run, with everything an analysis needs to locate it."""
    name: str                       # experiment_name
    group: str                      # experiment group (config subdirectory)
    config_path: Path
    config: ExperimentConfig
    run_dir: Path
    signal_type: str
    layer: int
    index: int
    position: Optional[int]         # fixed token selector; None for a random range
    position_range: Optional[tuple[int, int]] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        pos = (f"pos{self.position}" if self.position is not None
               else f"pos[{self.position_range[0]},{self.position_range[1]}]"
               if self.position_range else "pos?")
        return f"N{self.layer}.{self.index} {pos}"

    @property
    def cell(self) -> str:
        """The (layer, position) cell this run belongs to -- the aggregation key.
        Neurons vary WITHIN a cell, which is what gives every cell its error bar."""
        pos = self.position if self.position is not None else self.position_range
        return f"L{self.layer}/pos{pos}"

    def checkpoint(self, required: bool = True) -> Optional[Path]:
        return find_checkpoint(self.run_dir, required=required)

    def analysis_dir(self, name: str) -> Path:
        d = self.run_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def find_checkpoint(run_dir: Path, required: bool = True) -> Optional[Path]:
    """Latest checkpoint-N under run_dir, by STEP NUMBER (not mtime: a re-run of the
    diagnostics touches an older checkpoint's directory and would win on mtime)."""
    ckpts = [Path(p) for p in glob.glob(str(Path(run_dir) / "checkpoint-*")) if Path(p).is_dir()]
    if not ckpts:
        if required:
            raise SystemExit(f"No checkpoint under {run_dir}")
        return None
    def step(p: Path) -> int:
        try:
            return int(p.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return -1
    return max(ckpts, key=step)


def resolve_locale(config: ExperimentConfig) -> tuple[str, int, int]:
    """(signal_type, layer, index) this run reports on.

    signal_filter wins: a grid run's train_signals is the whole shared CELL (5
    neurons) and only the filter says which one this run trained. Refuses a
    multi-signal run rather than picking one, since every analysis downstream
    assumes a single locale.
    """
    if config.signal_filter:
        if "," in config.signal_filter:
            raise ValueError(
                f"{config.experiment_name}: signal_filter {config.signal_filter!r} names "
                f"several signals; the single-locale analyses do not apply to it.")
        f = parse_signal_filter(config.signal_filter)
        return f["signal_type"], f["signal_layer"], f["signal_index"]
    if len(config.train_signals) == 1:
        s = config.train_signals[0]
        return s.type, s.layer, s.index
    raise ValueError(
        f"{config.experiment_name}: {len(config.train_signals)} train signals and no "
        f"signal_filter; cannot resolve a single locale.")


def resolve_position(config: ExperimentConfig) -> tuple[Optional[int], Optional[tuple[int, int]]]:
    """(fixed position, range) from the TEST token selector -- the positions the
    analyses actually evaluate at. Exactly one of the two is not None."""
    sel = config.test_token_selector
    if sel.type == "fixed":
        return int(sel.position), None
    if sel.type == "random_range":
        lo, hi = sorted((int(sel.start), int(sel.stop)))
        return None, (lo, hi)
    raise ValueError(f"Unsupported token selector {sel.type!r}")


def load_run(config_path: Path, output_root: Optional[Path] = None) -> RunSpec:
    """Build a RunSpec from one experiment config."""
    config_path = Path(config_path)
    config = ExperimentConfig.model_validate_json(config_path.read_text())
    stype, layer, index = resolve_locale(config)
    pos, pos_range = resolve_position(config)
    # config.output_dir is the GROUP directory the run was written under; honour an
    # override so results synced to a different root can be analyzed in place.
    base = Path(output_root) if output_root else Path(config.output_dir)
    if not base.is_absolute():
        base = REPO_ROOT / base
    return RunSpec(
        name=config.experiment_name,
        group=config_path.parent.name,
        config_path=config_path,
        config=config,
        run_dir=base / config.experiment_name,
        signal_type=stype, layer=layer, index=index,
        position=pos, position_range=pos_range,
    )


def discover_runs(
    group: Optional[str] = None,
    run_glob: str = "*",
    configs: Optional[list[Path]] = None,
    config_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    require_checkpoint: bool = True,
    verbose: bool = True,
) -> list[RunSpec]:
    """
    Every run of `group` whose config name matches `run_glob`, sorted by name.

    require_checkpoint drops (and reports) runs whose checkpoint is absent, so a
    sweep with a few failed tasks still analyzes. Configs starting with "_" are the
    sweep's shared caches and are never runs.
    """
    if configs is None:
        if group is None:
            raise ValueError("Pass either `group` or an explicit `configs` list.")
        root = Path(config_root) if config_root else CONFIG_ROOT
        if not root.is_absolute():
            root = REPO_ROOT / root
        paths = sorted((root / group).glob(f"{run_glob}.json"))
        if not paths:
            raise SystemExit(f"No configs matching {run_glob}.json under {root / group}")
    else:
        paths = [Path(p) for p in configs]

    runs, skipped_shared, bad, missing = [], [], [], []
    for p in paths:
        if p.name.startswith(SHARED_CONFIG_PREFIX):
            skipped_shared.append(p.name)
            continue
        try:
            run = load_run(p, output_root=output_root)
        except Exception as e:                      # a malformed config is a gap, not a crash
            bad.append((p.name, f"{type(e).__name__}: {e}"))
            continue
        if require_checkpoint and run.checkpoint(required=False) is None:
            missing.append(run.name)
            continue
        runs.append(run)

    if verbose:
        print(f"[runs] {len(runs)} run(s) from {paths[0].parent if paths else '?'} "
              f"matching {run_glob!r}")
        if skipped_shared:
            print(f"[runs] skipped {len(skipped_shared)} shared cache config(s) "
                  f"(leading '{SHARED_CONFIG_PREFIX}')")
        if missing:
            print(f"[runs] SKIPPED {len(missing)} run(s) with no checkpoint, e.g. {missing[:3]}")
        for name, err in bad:
            print(f"[runs] SKIPPED unreadable config {name}: {err}")
    return runs


def shard(items: list, task_index: Optional[int] = None, n_tasks: Optional[int] = None) -> list:
    """
    This array task's contiguous slice of `items`. task_index is 1-BASED (SGE's
    $SGE_TASK_ID); both default to the environment, and to "everything" when unset.

    Contiguous, not strided: a half-finished array then leaves a contiguous prefix
    of the sweep done, which is easier to reason about than every third run.
    """
    if task_index is None:
        env = os.environ.get("SGE_TASK_ID")
        task_index = int(env) if env and env.isdigit() else None
    if n_tasks is None:
        env = os.environ.get("SGE_TASK_LAST")
        n_tasks = int(env) if env and env.isdigit() else None
    if not task_index or not n_tasks or n_tasks <= 1:
        return items
    if not 1 <= task_index <= n_tasks:
        raise SystemExit(f"task index {task_index} outside 1..{n_tasks}")
    per = (len(items) + n_tasks - 1) // n_tasks
    lo = (task_index - 1) * per
    out = items[lo:lo + per]
    print(f"[runs] task {task_index}/{n_tasks}: items {lo}..{lo + len(out) - 1} of {len(items)}")
    return out


def add_discovery_args(parser):
    """The selection/sharding flags every circuit analysis shares."""
    parser.add_argument("--config", type=Path, default=None,
                        help="Analyze ONE run by its experiment config (skips discovery).")
    parser.add_argument("--group", type=str, default=None,
                        help="Experiment group (subdirectory of experiment_configs/) to sweep.")
    parser.add_argument("--run-glob", type=str, default="*",
                        help="Glob over config names within the group, without '.json' "
                             "(e.g. 'joint_*kl0.1*' for one KL arm).")
    parser.add_argument("--config-root", type=Path, default=None,
                        help="Override experiment_configs/ (e.g. a synced copy).")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Override each config's output_dir, for runs synced elsewhere.")
    parser.add_argument("--task-index", type=int, default=None,
                        help="1-based array task index (default: $SGE_TASK_ID).")
    parser.add_argument("--n-tasks", type=int, default=None,
                        help="Total array tasks (default: $SGE_TASK_LAST).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs whose result file already exists.")
    return parser


def runs_from_args(args, require_checkpoint: bool = True) -> list[RunSpec]:
    """Resolve --config / --group into the list of runs this invocation handles."""
    if args.config:
        runs = [load_run(args.config, output_root=getattr(args, "output_root", None))]
    else:
        runs = discover_runs(group=args.group, run_glob=args.run_glob,
                             config_root=getattr(args, "config_root", None),
                             output_root=getattr(args, "output_root", None),
                             require_checkpoint=require_checkpoint)
    return shard(runs, getattr(args, "task_index", None), getattr(args, "n_tasks", None))
