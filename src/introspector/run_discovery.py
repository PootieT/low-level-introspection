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

NOT plotters/run_catalog.py, which is the other half of the same story and should be
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
import re
import glob
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from .config import ExperimentConfig
from .dataset_utils import parse_signal_filter, parse_signal_filters

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
    # EVERY signal the run was trained on, as (type, layer, index). A single-locale run
    # has one, and `layer`/`index` above are it. A multi-locale run (the neuron and layer
    # generalization sweeps train one readout on several signals) has several, and
    # `layer`/`index` are only its FIRST -- analyses iterate `locales` instead, one pass
    # per trained signal, and must never read `layer`/`index` as "the" locale.
    locales: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def is_multi_locale(self) -> bool:
        return len(self.locales) > 1

    def for_locale(self, locale) -> "RunSpec":
        """This run re-pointed at one of its trained signals. `locales` is kept whole, so
        the view still knows it belongs to a multi-locale run (which decides where its
        results are written)."""
        stype, layer, index = locale
        return replace(self, signal_type=stype, layer=layer, index=index)

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
        if self.is_multi_locale:
            return f"multi[{len(self.locales)}]/pos{pos}"
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


def resolve_locales(config: ExperimentConfig) -> list[tuple[str, int, int]]:
    """Every (signal_type, layer, index) this run was trained on, in config order.

    signal_filter wins when set: a grid run's train_signals is the whole shared CELL (5
    neurons) and only the filter says which one this run trained -- and a comma list
    there is how a multi-neuron run trains a subset of a shared cache. Otherwise the
    run trained on all of train_signals.
    """
    if config.signal_filter:
        return [(f["signal_type"], f["signal_layer"], f["signal_index"])
                for f in parse_signal_filters(config.signal_filter)]
    return [(s.type, s.layer, s.index) for s in config.train_signals]


def resolve_locale(config: ExperimentConfig) -> tuple[str, int, int]:
    """The ONE (signal_type, layer, index) of a single-locale run.

    Refuses a multi-locale run rather than silently picking its first signal: a caller
    that needs exactly one locale and gets several has a bug, and reporting the first
    signal as "the" locale would hide it. Multi-locale-aware code uses
    resolve_locales and iterates.
    """
    locs = resolve_locales(config)
    if len(locs) == 1:
        return locs[0]
    raise ValueError(
        f"{config.experiment_name}: {len(locs)} trained signals; this caller needs a "
        f"single locale. Iterate resolve_locales() instead.")


def locale_tag(signal_type: str, layer: int, index: int) -> str:
    """'N4.520' -- the per-locale subdirectory name for a multi-locale run."""
    code = {"mlp": "N", "attention_score": "A", "residual_stream": "R", "sae": "S"}
    return f"{code.get(signal_type, 'X')}{layer}.{index}"


def locale_out_dir(base: Path, spec: "RunSpec", locale) -> Path:
    """Where one locale's results go: `base` itself for a single-locale run (so every
    existing grid result stays where it is), `base/<tag>` for a multi-locale run."""
    return Path(base) / locale_tag(*locale) if spec.is_multi_locale else Path(base)


def select_locales(spec: "RunSpec", selector: Optional[str]) -> list[tuple[str, int, int]]:
    """Restrict a run's locales: None/'all' = every one, an int K = the first K (config
    order), or a comma list of signal specs ('N4.520,N4.1216') = exactly those."""
    locs = spec.locales
    if not selector or selector == "all":
        return locs
    if selector.isdigit():
        return locs[:int(selector)]
    wanted = {(f["signal_type"], f["signal_layer"], f["signal_index"])
              for f in parse_signal_filters(selector)}
    missing = wanted - set(locs)
    if missing:
        print(f"[runs] {spec.name}: --locales names {len(missing)} signal(s) the run was "
              f"not trained on, ignored: {sorted(missing)[:5]}")
    return [l for l in locs if l in wanted]


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
    locales = resolve_locales(config)
    if not locales:
        raise ValueError(f"{config.experiment_name}: no trained signals")
    stype, layer, index = locales[0]
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
        position=pos, position_range=pos_range, locales=locales,
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
    parser.add_argument("--locales", type=str, default=None,
                        help="For MULTI-locale runs (several trained signals), which to "
                             "analyze: 'all' (default), an integer K for the first K, or a "
                             "comma list of signal specs like 'N4.520,N4.1216'. Each one "
                             "is analyzed separately into <run>/<analysis>/<signal>/.")
    parser.add_argument("--from-cache", action="store_true",
                        help="For each run: if its cached results exist, skip the slow "
                             "recompute and just regenerate the plots from those cached "
                             "CSVs/arrays; otherwise compute it normally. For iterating on "
                             "plot styling without re-running the whole sweep.")
    return parser


def resolve_run_action(args, sentinel: Path) -> str:
    """Per-run dispatch shared by every analysis loop: returns 'replot', 'skip', or
    'compute' from --from-cache / --skip-existing and whether `sentinel` (a file that
    only exists once the run finished) is present.

    --from-cache wins: a cached run is re-plotted, an uncached one is computed (resume).
    The caller's replot function still validates the cache and may fall back to compute."""
    exists = Path(sentinel).exists()
    if getattr(args, "from_cache", False):
        return "replot" if exists else "compute"
    if getattr(args, "skip_existing", False) and exists:
        return "skip"
    return "compute"


def plan_locales(spec: RunSpec, base_dir: Path, sentinel: str, args, replot=None,
                 log: str = "[run]") -> list[tuple[tuple, Path]]:
    """[(locale, out_dir), ...] still to COMPUTE for this run, after --locales,
    --skip-existing and --from-cache have had their say, locale by locale.

    Decided per locale rather than per run, so a multi-locale run that died halfway
    resumes at the first missing locale instead of recomputing (or skipping) all of
    them. An empty list means the model need not be loaded at all.
    """
    todo = []
    for loc in select_locales(spec, getattr(args, "locales", None)):
        out_dir = locale_out_dir(base_dir, spec, loc)
        label = spec.name + (f" [{locale_tag(*loc)}]" if spec.is_multi_locale else "")
        action = resolve_run_action(args, out_dir / sentinel)
        if action == "skip":
            print(f"{log} {label}: results exist, skipping")
            continue
        if action == "replot" and replot is not None:
            if replot(spec.for_locale(loc), out_dir, args):
                print(f"{log} {label}: re-plotted from cache")
                continue
            print(f"{log} {label}: cache incomplete, recomputing")
        todo.append((loc, out_dir))
    return todo


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


def pretty_run_name(name: str, seq_len: int = 32) -> str:
    """Figure title for a run directory name, e.g.
    'joint_N12.2721_POS-28_gateoff_scale10_kl0.1_w0.01_300K'
        -> 'Joint Loss (KL=0.1), Single-locale N12.2721, t=4'.
    Token positions are shown as 0-indexed absolute indices in the seq_len window
    (POS-1 -> t=31), as in the layer x position heatmaps. Names in an unrecognised
    scheme are returned unchanged rather than mislabelled."""
    neuron = re.search(r"N(\d+)\.(\d+)", name)
    pos = re.search(r"POS(-?\d+)", name)
    kl = re.search(r"_kl([0-9.]+|none|inf)", name)
    w = re.search(r"_w([0-9.]+)", name)
    gate = re.search(r"gate(on|off)", name)
    w_val = float(w.group(1)) if w else None
    if name.startswith("joint") and w_val != 1.0:
        k = kl.group(1) if kl else None
        arm = f"Joint Loss (KL={'None' if k in (None, 'none', 'inf') else k})"
    elif name.startswith(("endo", "sft")) or w_val == 1.0:       # endogenous loss only
        arm = "SFT" + (f" (Gate-{gate.group(1).capitalize()})" if gate else "")
    elif name.startswith("prompt"):
        arm = "Prompt Baseline"
    else:
        return name
    parts = [arm]
    if neuron:
        parts.append(f"Single-locale N{neuron.group(1)}.{neuron.group(2)}")
    if pos:
        p_ = int(pos.group(1))
        parts.append(f"t={p_ + seq_len if p_ < 0 else p_}")
    return ", ".join(parts)
