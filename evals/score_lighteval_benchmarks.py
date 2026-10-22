#!/usr/bin/env python3
"""
Behavioural check: does the introspection adapter change what the model can DO?

Runs lighteval twice over the SAME tasks and the same few-shot seeds -- once with
the adapter active, once inside `disable_adapter()` -- and reports both, plus the
delta. Companion to evals/score_lm_degradation.py, which measures the same question at
the distribution level (KL per token). Read them together: KL is the sensitive
instrument, benchmarks are the "nothing grossly broke" evidence a reader expects.

LOADING. lighteval's AdapterModelConfig merges the adapter and WRITES the merged
weights to disk (`{adapter}-adapter-applied`) before evaluating. We do not use it:
TransformersModel.from_model() takes an already-instantiated model, so a PeftModel
goes straight in -- one model in memory, nothing written, nothing merged.

TASK CHOICE. At 1B, sensitivity varies enormously. Tasks where the base model sits
near chance (MMLU, TriviaQA) cannot detect degradation -- a real regression will not
move them. The defaults below favour headroom: ARC-Easy, WinoGrande, HellaSwag,
PIQA, and LAMBADA (a next-token task, the closest benchmark analogue to LM damage).
Keep the near-chance ones only if the paper wants capability coverage, and read them
as reporting rather than detection.

READ THE DELTA, NOT THE SCORES. Two runs at 61% can disagree on 15% of items. With
--save-details lighteval writes per-sample outputs for both conditions; a paired
per-example comparison over those is far more powerful than differencing aggregates.

ONLY MEANINGFUL FOR GATE-OFF CHECKPOINTS -- see evals/score_lm_degradation.py.

Example
-------
python evals/score_lighteval_benchmarks.py --config experiment_configs/<group>/<run>.json --max-samples 200
"""
from __future__ import annotations

import os
import sys
import glob
import shutil
from datetime import datetime
import json
import argparse
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from src.introspector.config import ExperimentConfig
from evals.score_lm_degradation import load_adapter_model

# Requested by BARE task name; the suite prefix is resolved against the installed
# lighteval's own registry (see resolve_tasks). Hardcoding "leaderboard|winogrande"
# breaks whenever a release reshuffles suites, which is exactly what happened.
DEFAULT_TASK_NAMES = [
    "arc:easy",     # reasoning, well above chance at 1B
    "winogrande",   # commonsense
    "hellaswag",    # commonsense
    # "piqa",         # commonsense / physical  (dataset loader bug)
    "lambada",      # next-token: the closest benchmark analogue to LM damage
]
# deterministic tie-break when a task appears under several suites
SUITE_PRIORITY = ["leaderboard", "lighteval", "harness", "helm", "original", "bigbench"]


def _import_lighteval():
    """Centralized so an API change breaks in ONE place with an actionable message."""
    try:
        from lighteval.logging.evaluation_tracker import EvaluationTracker
        from lighteval.models.transformers.transformers_model import (
            TransformersModel, TransformersModelConfig,
        )
        from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
    except ImportError as e:
        raise SystemExit(
            f"lighteval import failed ({e}).\n"
            f"  pip install lighteval\n"
            f"If it IS installed, the API moved: this script needs EvaluationTracker, "
            f"TransformersModel(.from_model)/TransformersModelConfig, and "
            f"Pipeline/PipelineParameters/ParallelismManager. Check `lighteval --version` and "
            f"adjust _import_lighteval + run_condition, which are the only lighteval-aware parts.")
    return (EvaluationTracker, TransformersModel, TransformersModelConfig,
            ParallelismManager, Pipeline, PipelineParameters)


def _registry():
    """Instantiate a task Registry across lighteval versions with different signatures."""
    from lighteval.tasks.registry import Registry
    for kwargs in ({"tasks": None}, {}, {"cache_dir": None}):
        try:
            return Registry(**kwargs)
        except TypeError:
            continue
    raise SystemExit("Could not instantiate lighteval's Registry; run `lighteval tasks list` "
                     "and pass fully-qualified ids via --tasks instead.")


def _task_table() -> dict:
    """{key: LightevalTaskConfig} from the installed lighteval. Key SHAPE varies by
    version: some releases use 'suite|task', others bare 'task'."""
    reg = _registry()
    for attr in ("_task_registry", "task_registry", "TASK_REGISTRY"):
        table = getattr(reg, attr, None)
        if isinstance(table, dict) and table:
            return table
    raise SystemExit("Could not read the task table from lighteval's Registry; run "
                     "`lighteval tasks list` and pass fully-qualified ids via --tasks.")


def available_tasks() -> list[str]:
    return sorted(_task_table().keys())


def task_metrics(key: str) -> list[str]:
    """Metric names a task will be scored with. Worth printing BEFORE a run: the same
    dataset exists under several suites with very different configs, and a generative
    metric (em/exact_match) on a multiple-choice benchmark scores 0 for every model --
    which looks like 'no difference between conditions' rather than like a broken
    setup."""
    cfg = _task_table().get(key)
    for attr in ("metrics", "metric"):
        metrics = getattr(cfg, attr, None)
        if metrics:
            out = []
            for m in (metrics if isinstance(metrics, (list, tuple, set)) else [metrics]):
                name = getattr(m, "metric_name", None) or getattr(m, "name", None) or str(m)
                out.extend(name if isinstance(name, (list, tuple)) else [str(name)])
            return out
    return []


# metrics that cannot score a multiple-choice task sensibly
GENERATIVE_METRICS = {"em", "exact_match", "quasi_exact_match", "f1", "rouge"}


def _index_tasks(keys: list[str]) -> dict[str, list[str | None]]:
    """{task_name: [suite or None, ...]} -- tolerant of both key shapes, since
    assuming 'suite|task' silently indexes everything under the empty string when the
    keys are bare names (which is how 'arc:easy' came back as 'similar: none' while
    sitting in plain sight in --list-tasks)."""
    by_task: dict[str, list[str | None]] = {}
    for key in keys:
        if "|" in key:
            suite, _, task = key.partition("|")
        else:
            suite, task = None, key
        by_task.setdefault(task, []).append(suite)
    return by_task


def _candidate_strings(suite: str | None, task: str, few_shot: int):
    """Task-string layouts to try, most specific first. Which one a release accepts
    varies, and the trailing few-shot fields can end up part of the lookup key."""
    base = f"{suite}|{task}" if suite else task
    return [("base|few_shot|truncate", f"{base}|{few_shot}|0"),
            ("base|few_shot", f"{base}|{few_shot}"),
            ("base", base)]


def _registry_accepts(task_string: str) -> bool:
    """Registry.__init__ eagerly expands and validates its `tasks` argument (that is
    where the 'Cannot find task' error comes from), so constructing one is a real
    format check."""
    from lighteval.tasks.registry import Registry
    try:
        Registry(tasks=task_string)
        return True
    except Exception:
        return False


def _probe_format(suite, task, few_shot) -> str:
    """Probe once for the layout THIS install accepts; the answer is the same for
    every task."""
    for fmt, candidate in _candidate_strings(suite, task, few_shot):
        if _registry_accepts(candidate):
            print(f"[lighteval] task string format for this install: {fmt}")
            if fmt == "base" and few_shot:
                print(f"[lighteval] WARNING: this format has no few-shot field, so --few-shot "
                      f"{few_shot} is NOT applied. Set it in the task config or accept 0-shot.")
            return fmt
    raise SystemExit(
        f"No task-string layout was accepted for {task!r}. Tried: "
        f"{[c for _, c in _candidate_strings(suite, task, few_shot)]}. Run "
        f"`lighteval tasks list` and pass a working id verbatim via --tasks.")


def resolve_tasks(names: list[str], few_shot: int) -> str:
    """
    Bare names -> task strings the installed lighteval accepts.

    Exact match first, then prefix (so 'lambada' finds 'lambada:openai'), with
    SUITE_PRIORITY breaking ties when a task lives under several suites. The string
    layout is probed rather than assumed, and the whole set is validated before any
    model runs.
    """
    keys = available_tasks()
    by_task = _index_tasks(keys)

    resolved, fmt = [], None
    for name in names:
        matched, suites = name, by_task.get(name)
        if not suites:                       # prefix match: 'lambada' -> 'lambada:openai'
            cands = sorted(t for t in by_task if t.startswith(f"{name}:"))
            if cands:
                matched, suites = cands[0], by_task[cands[0]]
        if not suites:
            stem = name.split(":")[0]
            near = sorted(t for t in by_task if stem in t)[:10]
            raise SystemExit(
                f"Task {name!r} not found in this lighteval install.\n"
                f"  similar: {near or 'none'}\n"
                f"  example keys: {keys[:5]}\n"
                f"  list them all: python evals/{Path(__file__).name} --list-tasks")
        suite = min(suites, key=lambda s: (SUITE_PRIORITY.index(s) if s in SUITE_PRIORITY
                                           else len(SUITE_PRIORITY), s or ""))
        if fmt is None:
            fmt = _probe_format(suite, matched, few_shot)
        resolved.append(dict(_candidate_strings(suite, matched, few_shot))[fmt])
        key = f"{suite}|{matched}" if suite else matched
        mets = task_metrics(key)
        print(f"[lighteval] {name:12s} -> {resolved[-1]}   metrics={mets or 'unknown'}"
              + (f"   (suites: {sorted(str(x) for x in suites)})" if len(suites) > 1 else ""))
        if mets and set(m.lower() for m in mets) <= GENERATIVE_METRICS:
            print(f"[lighteval]   WARNING: {name} resolves to a variant scored ONLY by "
                  f"{mets} -- a generative metric, which scores ~0 for every model on a "
                  f"multiple-choice task. Pick another variant (see --list-tasks).")

    joined = ",".join(resolved)
    # a task can exist yet fail to expand (missing dataset, gated repo); finding that
    # out here beats finding it out after the base-model pass has already run
    if not _registry_accepts(joined):
        bad = [t for t in resolved if not _registry_accepts(t)]
        raise SystemExit(f"lighteval rejected these resolved tasks: {bad or joined}\n"
                         f"Inspect with `lighteval tasks inspect <task>`, or pass working ids "
                         f"verbatim via --tasks.")
    return joined


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=False,
                   help="Experiment config (not needed with --list-tasks).")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Adapter checkpoint (default: latest under the run's output dir).")
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated. Bare names (winogrande, arc:easy) are resolved against "
                        "the installed lighteval registry; anything containing '|' is passed "
                        "through verbatim. Default: " + ",".join(DEFAULT_TASK_NAMES))
    p.add_argument("--few-shot", type=int, default=0)
    p.add_argument("--list-tasks", action="store_true",
                   help="Print every task the installed lighteval knows about, and exit.")
    p.add_argument("--max-samples", type=int, default=200,
                   help="Examples per task. Small values iterate fast but the noise floor is "
                        "several points -- fine for 'nothing broke', useless for '2 points worse'.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--conditions", type=str, default="adapter,base",
                   help="Which passes to run; 'base' is the same checkpoint with adapters disabled.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if benchmarks_raw.json already exists. Worth having: a "
                        "benchmark pass is far more expensive than the KL scorer, and sweep "
                        "scripts re-invoke it on every resubmission.")
    p.add_argument("--force", action="store_true", help="Run even on a gated checkpoint.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Response-cache root. Default: a fresh per-invocation directory under "
                        "the run's lighteval/_cache, which prevents both cross-condition replay "
                        "and stale reuse after retraining. Set this only to deliberately reuse.")
    p.add_argument("--keep-cache", action="store_true",
                   help="Keep the response cache after the run (default: delete it).")
    return p.parse_args()


def run_condition(peft_model, config, args, condition: str, out_dir: Path,
                  cache_root: Path) -> dict:
    """One full pass over the task list. condition='base' runs the identical harness
    with the adapter disabled, so few-shot examples and ordering are shared and the
    two conditions differ only by the adapter."""
    (EvaluationTracker, TransformersModel, TransformersModelConfig,
     ParallelismManager, Pipeline, PipelineParameters) = _import_lighteval()

    # ISOLATE the response cache PER CONDITION AND PER INVOCATION.
    #
    # lighteval caches responses under {cache_dir}/{model_name}/{hash(model_config)}.
    # Both conditions share the model id and, with an identical config, the hash too --
    # so the second condition silently REPLAYS the first's responses. Not theoretical:
    # it produced byte-identical arc/winogrande logprobs across adapter and base (the
    # "base" numbers were really the adapter's, which is why they looked like chance)
    # while lambada, on an uncached path, differed by 9.8 nats.
    #
    # Setting _cache = None does NOT work: Pipeline.__init__ calls
    # self.model._cache._init_registry(...) unconditionally. Instead give each
    # condition its own directory -- which also perturbs the hash, since cache_dir is
    # part of the hashed config. The per-invocation stamp additionally prevents a stale
    # cache from replaying a PREVIOUS checkpoint's answers: the hash covers the config,
    # not the adapter weights, so same-path reruns after retraining would otherwise hit.
    cache_dir = Path(cache_root) / condition
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_config = TransformersModelConfig(
        model_name=config.model_name,
        batch_size=args.batch_size,
        dtype="bfloat16" if config.training_config.bf16 else "float32",
        cache_dir=str(cache_dir),
    )
    lm = TransformersModel.from_model(model=peft_model, config=model_config)
    print(f"[lighteval] response cache isolated at {cache_dir}")
    tracker = EvaluationTracker(output_dir=str(out_dir / condition), save_details=True)
    params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=args.max_samples,
        num_fewshot_seeds=1,          # same few-shot draw in both conditions
    )
    pipeline = Pipeline(tasks=args.tasks, pipeline_parameters=params,
                        evaluation_tracker=tracker, model=lm)
    pipeline.evaluate()
    pipeline.show_results()
    pipeline.save_and_push_results()
    return pipeline.get_results()


def flatten_results(results: dict, condition: str) -> list[dict]:
    """lighteval returns {'results': {task: {metric: value, ...}}, ...} -> long rows."""
    rows = []
    for task, metrics in (results or {}).get("results", {}).items():
        for metric, value in (metrics or {}).items():
            if isinstance(value, (int, float)) and not metric.endswith("_stderr"):
                rows.append({"condition": condition, "task": task, "metric": metric,
                             "value": float(value)})
    return rows


def verify_conditions_differ(out_dir: Path):
    """Acceptance test for the run: per-example scores MUST differ between conditions.
    Identical scores mean one condition replayed the other (see the cache note in
    run_condition), and every delta in the table is then meaningless."""
    try:
        from plotters.compare_lighteval_details import compare_task, _detail_files
    except Exception as e:                      # plotters module absent / pyarrow missing
        print(f"[lighteval] (skipping per-example verification: {e})")
        return
    run_dir = out_dir.parent if out_dir.name == "lighteval" else out_dir
    base, adapter = _detail_files(run_dir, "base"), _detail_files(run_dir, "adapter")
    shared = sorted(set(base) & set(adapter))
    if not shared:
        return
    bad = []
    for task in shared:
        r = compare_task(base[task], adapter[task], task)
        frac = r.get("identical_scores_frac", r.get("identical_text_frac"))
        if frac is not None and frac > 0.99:
            bad.append((task, frac))
    if bad:
        print("\n[lighteval] *** CONDITIONS ARE NOT DISTINCT ***")
        for task, frac in bad:
            print(f"    {task}: {frac:.1%} of examples scored IDENTICALLY in both conditions")
        print("    One condition replayed the other's responses. Treat every delta above as "
              "invalid, and check that the response cache is disabled.")
    else:
        print("\n[lighteval] verified: per-example scores differ between conditions")


def main():
    args = parse_args()
    if args.list_tasks:
        print("\n".join(available_tasks()))
        return
    if args.config is None:
        raise SystemExit("--config is required (except with --list-tasks).")
    config = ExperimentConfig.model_validate_json(args.config.read_text())
    run_dir = Path(config.output_dir) / config.experiment_name
    out_dir = args.output_dir or (run_dir / "lighteval")
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "benchmarks_raw.json").exists() and not args.overwrite:
        print(f"[lighteval] already evaluated: {out_dir}/benchmarks_raw.json (--overwrite to redo)")
        return

    if config.training_config.disable_adapter_in_input and not args.force:
        raise SystemExit(
            "This is a GATED checkpoint: the adapter was masked off over ordinary text during "
            "training, so its benchmark behaviour IS the base model's by construction. Nothing "
            "to measure here; pass --force for the counterfactual anyway.")

    ckpts = glob.glob(f"{run_dir}/checkpoint-*")
    ckpt = Path(args.checkpoint) if args.checkpoint else (
        Path(max(ckpts, key=os.path.getctime)) if ckpts else None)
    if ckpt is None:
        raise SystemExit(f"No checkpoint found under {run_dir}; pass --checkpoint.")
    if args.tasks and "|" in args.tasks:
        tasks = args.tasks                      # fully qualified, pass through
    else:
        names = [n.strip() for n in args.tasks.split(",")] if args.tasks else DEFAULT_TASK_NAMES
        tasks = resolve_tasks(names, args.few_shot)
    args.tasks = tasks
    print(f"[lighteval] checkpoint={ckpt}\n[lighteval] tasks={tasks}")

    # NOT merged and NOT written to disk -- see the module docstring. Vocab is resized
    # to load (PEFT saved the resized embeddings) then trimmed back, so lighteval's own
    # base tokenizer matches the model and no phantom token can be generated.
    model, _ = load_adapter_model(config, ckpt, trim_vocab=True)

    # fresh cache root per invocation; conditions get subdirectories of it
    cache_root = Path(args.cache_dir) if args.cache_dir else (
        out_dir / "_cache" / datetime.now().strftime("%Y%m%dT%H%M%S"))
    rows, raw = [], {}
    for condition in [c.strip() for c in args.conditions.split(",")]:
        print(f"\n[lighteval] ===== condition: {condition} =====")
        if condition == "base":
            with model.disable_adapter():
                results = run_condition(model, config, args, condition, out_dir, cache_root)
        else:
            results = run_condition(model, config, args, condition, out_dir, cache_root)
        raw[condition] = results
        rows.extend(flatten_results(results, condition))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("lighteval returned no parseable metrics; check the task ids.")
    df.to_csv(out_dir / "benchmarks_long.csv", index=False)

    wide = df.pivot_table(index=["task", "metric"], columns="condition", values="value").reset_index()
    if {"adapter", "base"} <= set(wide.columns):
        wide["delta"] = wide["adapter"] - wide["base"]
    wide.to_csv(out_dir / "benchmarks_comparison.csv", index=False)
    print("\n[lighteval] adapter vs base (same tasks, same few-shot seed):")
    print(wide.to_string(index=False))
    returned = {t for t in df["task"].unique() if t != "all"}
    requested = {t.split("|")[0] if "|" in t else t for t in args.tasks.split(",")}
    missing = {r for r in requested if not any(r.split("|")[0] in t for t in returned)}
    if missing:
        print(f"\n[lighteval] WARNING: requested but ABSENT from results: {sorted(missing)}. "
              f"lighteval dropped them silently (dataset download failed, or gated repo). "
              f"Inspect with `lighteval tasks inspect <task>`.")

    if "delta" in wide:
        # A metric identical across conditions while another moves by orders of
        # magnitude is not evidence of preserved behaviour -- it means that metric is
        # not sensitive to the model here. Compare the base column against published
        # numbers for this model before trusting any of it.
        flat = wide[wide["delta"].abs() < 1e-12]
        moved = wide[wide["delta"].abs() > 1e-6]
        if len(flat) and len(moved):
            print(f"\n[lighteval] WARNING: {len(flat)} metric(s) are EXACTLY identical across "
                  f"conditions while {len(moved)} moved. Identical scores usually mean the metric "
                  f"is not model-sensitive in this configuration (wrong task variant, degenerate "
                  f"scoring), not that behaviour is preserved:\n"
                  f"{flat[['task', 'metric', 'base']].to_string(index=False)}")
            print("[lighteval] Sanity-check the BASE column against published numbers for this "
                  "model. A base score at chance means the harness, not the adapter, is broken.")

        worst = wide.loc[wide["delta"].idxmin()]
        print(f"\n[lighteval] largest drop: {worst['task']} / {worst['metric']} "
              f"{worst['base']:.4f} -> {worst['adapter']:.4f} ({worst['delta']:+.4f})")
        print("[lighteval] NOTE: aggregate deltas near zero do NOT mean per-example agreement. "
              "Per-sample outputs for both conditions are under "
              f"{out_dir}/<condition>/details -- compare them pairwise before concluding "
              "the behaviour is unchanged.")
    verify_conditions_differ(out_dir)
    with open(out_dir / "benchmarks_raw.json", "w") as f:
        json.dump({"checkpoint": str(ckpt), "tasks": args.tasks,
                   "max_samples": args.max_samples, "results": raw}, f, indent=2, default=str)
    print(f"[lighteval] wrote {out_dir}/benchmarks_{{long,comparison}}.csv + benchmarks_raw.json")
    if not args.keep_cache and args.cache_dir is None and cache_root.exists():
        shutil.rmtree(cache_root, ignore_errors=True)


if __name__ == "__main__":
    main()
