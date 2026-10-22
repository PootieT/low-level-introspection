#!/usr/bin/env python3
"""
Rewrite references to the pre-reorganization layout (before commit b409457) in a branch.

Commit b409457 moved every root-level script into a purpose folder and renamed analysis/
to plotters/ (see MIGRATION.md). Git carries your edits to the MOVED files across a merge
on its own; what it cannot fix is everything that NAMES them -- `python train_...py` in
your shell scripts, `from train_introspector import ...` in your Python, "see
analysis/x.py" in your docs. This rewrites those, in place, from the repo root:

    python utils/migrate_old_paths.py            # dry run: what would change
    python utils/migrate_old_paths.py --apply    # rewrite

Idempotent: a rewritten path is preceded by its folder, so a second run matches nothing.
It does NOT move files. A .py you added at the repo root is listed at the end; move it
into the folder for its purpose and add the repo-root line from MIGRATION.md.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# old root-level module -> the folder it lives in now
MOVED = {
    **dict.fromkeys(["train_introspector", "train_introspector_joint",
                     "train_introspector_with_intervention", "train_sae_detector",
                     "train_sae_detector_intervention", "train_sae_detector_joint"], "trainers"),
    **dict.fromkeys(["cache_signals", "generate_experiment_config", "select_layer_gen_index",
                     "select_sae_features", "build_sae_featgen_pool", "fetch_sae_feature_labels",
                     "annotate_sae_features_neuronpedia", "generate_sae_steering_samples"],
                    "data_prep"),
    **dict.fromkeys(["score_length_generalization", "score_lighteval_benchmarks",
                     "score_lm_degradation", "score_neuron_generalization",
                     "score_ood_generalization", "score_prompt_baseline",
                     "score_sae_prompt_baseline", "score_sae_feature_generalization"], "evals"),
    **dict.fromkeys(["score_gradient_attribution", "score_head_ablation", "score_head_knockout",
                     "score_injection_specificity", "score_neuron_bypass_ceiling",
                     "score_neuron_redundancy", "trace_introspection_circuit", "trace_mover_head",
                     "ablate_lora_modules", "attribute_attention_weight_readout",
                     "probe_attention_weight_transmission", "scan_attention_weight_spread"],
                    "mechanistic_analysis"),
    **dict.fromkeys(["validate_attention_logits", "validate_attention_weights",
                     "validate_sae_hook", "probe_cache_alignment"], "tests"),
}
# removed outright -- references to these need a decision, not a rewrite
DELETED = {
    "intervene_introspector": "every trainer runs its own interventional eval; re-score a "
                              "checkpoint with trainers/train_introspector_joint.py --interv-eval-only",
    "score_rank_metrics": "the default eval writes spearman / kendall_tau / monotone_r2",
    "relocate_wrong_model_runs": "removed; recover with `git show 849be90:utils/relocate_wrong_model_runs.py`",
}
SKIP = ("outputs/", "data/", "wandb/", "experiment_configs/")
# these name the old layout on purpose (the map, its docs, its test fixtures)
KEEPS_OLD_PATHS = {"MIGRATION.md", "utils/migrate_old_paths.py", "tests/test_migrate_old_paths.py"}

_alt = "|".join(map(re.escape, MOVED))
PATH_RE = re.compile(rf"(?<![\w/.-])(\./)?({_alt})\.py(?!\w)")
# the old analysis/ held only .py files; scripts/*/analysis/ (shell drivers) kept its name
ANALYSIS_RE = re.compile(r"(?<![\w/.-])analysis/(?=\w+\.py(?!\w))")
FROM_RE = re.compile(rf"^(\s*)from ({_alt}) import ", re.M)
IMPORT_AS_RE = re.compile(rf"^(\s*)import ({_alt}) as (\w+)", re.M)
IMPORT_RE = re.compile(rf"^(\s*)import ({_alt})[ \t]*(#.*)?$", re.M)
ANALYSIS_IMPORT_RE = re.compile(r"^(\s*)(from|import) analysis\.", re.M)
DELETED_RE = re.compile(rf"(?<![\w/.-])({'|'.join(DELETED)})(\.py)?(?!\w)")


def rewrite(text: str, is_python: bool) -> str:
    text = PATH_RE.sub(lambda m: f"{m[1] or ''}{MOVED[m[2]]}/{m[2]}.py", text)
    text = ANALYSIS_RE.sub("plotters/", text)
    if is_python:
        text = FROM_RE.sub(lambda m: f"{m[1]}from {MOVED[m[2]]}.{m[2]} import ", text)
        text = IMPORT_AS_RE.sub(lambda m: f"{m[1]}import {MOVED[m[2]]}.{m[2]} as {m[3]}", text)
        text = IMPORT_RE.sub(lambda m: f"{m[1]}import {MOVED[m[2]]}.{m[2]} as {m[2]}"
                             + (f"  {m[3]}" if m[3] else ""), text)
        text = ANALYSIS_IMPORT_RE.sub(r"\1\2 plotters.", text)
    return text


def candidate_files(root: Path) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=root, capture_output=True, text=True, check=True).stdout.split()
    return [root / p for p in out if p.endswith((".py", ".sh", ".md", ".txt"))
            and not p.startswith(SKIP) and (root / p).is_file()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="rewrite files (default: dry run)")
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    args = ap.parse_args()

    changed, deleted_refs = 0, []
    for f in candidate_files(args.root):
        rel = f.relative_to(args.root)
        if rel.as_posix() in KEEPS_OLD_PATHS:
            continue
        text = f.read_text(errors="surrogateescape")
        new = rewrite(text, f.suffix == ".py")
        for i, line in enumerate(new.splitlines(), 1):
            for m in DELETED_RE.finditer(line):
                deleted_refs.append(f"{rel}:{i}: {m[1]} -- {DELETED[m[1]]}")
        if new == text:
            continue
        changed += 1
        n = sum(a != b for a, b in zip(text.splitlines(), new.splitlines()))
        print(f"{'rewrote' if args.apply else 'would rewrite'} {rel} ({n} line(s))")
        if args.apply:
            f.write_text(new, errors="surrogateescape")

    print(f"\n{changed} file(s) {'rewritten' if args.apply else 'to rewrite'}"
          + ("" if args.apply or not changed else "; rerun with --apply"))
    if deleted_refs:
        print(f"\n{len(deleted_refs)} reference(s) to removed scripts, fix by hand:")
        print("\n".join(f"    {r}" for r in deleted_refs))
    stray = sorted(p.name for p in args.root.glob("*.py"))
    if stray:
        print(f"\n.py files at the repo root (move into a purpose folder, see MIGRATION.md): "
              f"{stray}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
