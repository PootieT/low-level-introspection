"""
Bulk-delete checkpoint artifacts, optionally KEEPING a few runs per grid cell.

Sweeps produce one checkpoint per (cell x neuron), and the adapter weights are the
bulk of it -- but the weight-space analyses (plotters/compare_lora_weight_patterns.py)
need adapter_model.safetensors for at least one neuron per cell. `keep_per_cell`
keeps that many and deletes the rest.

Which neuron survives is deterministic: the LOWEST neuron index in the cell. Since
resolution.py seeds neuron selection per (seed, layer), a layer uses the same neuron
set at every position, so the same neuron is the survivor across positions -- which is
what makes cross-position weight comparisons possible afterwards.

Anything whose run name cannot be parsed is KEPT and reported. A deletion tool should
only remove what it understands.
"""
import os
import re
import glob
import shutil
from collections import defaultdict

from tqdm import tqdm

# endo_N10.24_POS-28_gateon_..., joint_N2.24_POS-1_gateoff_..._w0.01, etc.
RUN_RE = re.compile(r"N(?P<layer>\d+)\.(?P<index>\d+)")
POS_RE = re.compile(r"POS(?P<pos>-?\d+)")
GATE_RE = re.compile(r"gate(?P<gate>on|off)")
W_RE = re.compile(r"_w(?P<w>[0-9.]+)")
KL_RE = re.compile(r"_kl(?P<kl>[0-9.]+)")


def _run_name(path: str):
    """The run-directory component of a path, i.e. the first ancestor that looks like
    a run name. Robust to how deep the matched file sits under it."""
    for part in reversed(os.path.normpath(path).split(os.sep)):
        if RUN_RE.search(part):
            return part
    return None


def cell_of(run_name: str):
    """Grid cell = everything that identifies the experiment EXCEPT which neuron.
    Returns (cell_key, neuron_index) or None when the name cannot be parsed."""
    n = RUN_RE.search(run_name)
    if not n:
        return None
    p, g, w, kl = (POS_RE.search(run_name), GATE_RE.search(run_name),
                   W_RE.search(run_name), KL_RE.search(run_name))
    cell = (int(n["layer"]),
            int(p["pos"]) if p else None,
            g["gate"] if g else None,
            w["w"] if w else None,
            kl["kl"] if kl else None)
    return cell, int(n["index"])


def _size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024


def plan_deletion(paths, keep_per_cell=0):
    """Split matched paths into (to_delete, kept). keep_per_cell=0 deletes everything
    matched, which is the original behaviour."""
    if keep_per_cell <= 0:
        return list(paths), []

    by_cell, unparsed = defaultdict(list), []
    for p in paths:
        run = _run_name(p)
        parsed = cell_of(run) if run else None
        if parsed is None:
            unparsed.append(p)
            continue
        cell, neuron = parsed
        by_cell[cell].append((neuron, p))

    to_delete, kept = [], list(unparsed)
    for cell, entries in by_cell.items():
        # deterministic survivors: lowest neuron indices in the cell
        entries.sort(key=lambda t: (t[0], t[1]))
        survivors = {p for _, p in entries[:keep_per_cell]}
        for _, p in entries:
            (kept if p in survivors else to_delete).append(p)

    if unparsed:
        print(f"  [keep] {len(unparsed)} path(s) had unparseable run names and are KEPT: "
              f"{[os.path.basename(_run_name(p) or p) for p in unparsed[:3]]}")
    print(f"  [keep] {len(by_cell)} cell(s); keeping up to {keep_per_cell} neuron(s) each "
          f"-> {len(kept)} kept, {len(to_delete)} to delete")
    return to_delete, kept


def delete_files_matching_pattern(exp_pattern: str, keep_per_cell: int = 0,
                                  dry_run: bool = False, assume_yes: bool = False):
    files = sorted(glob.glob(exp_pattern))
    print(f"\n{exp_pattern}\nFound {len(files)} files")
    if not files:
        print("No files found")
        return
    to_delete, kept = plan_deletion(files, keep_per_cell)
    freed = sum(_size(f) for f in to_delete)
    print(f"  would free {_human(freed)} "
          f"(keeping {_human(sum(_size(f) for f in kept))} across {len(kept)} path(s))")
    if kept and keep_per_cell:
        print(f"  keeping e.g.: {[_run_name(p) for p in kept[:3]]}")
    if dry_run:
        print("  DRY RUN -- nothing deleted")
        return
    if not to_delete:
        return
    if not assume_yes:
        res = input(f"Sure you want to delete {len(to_delete)} path(s)? (Y/N)")
        if not res.lower().startswith("y"):
            print("  aborted")
            return
    for f in tqdm(to_delete):
        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f)


# ---------------------------------------------------------------------------
# Frozen embedding tensors
# ---------------------------------------------------------------------------
# PEFT's save_pretrained(save_embedding_layers="auto") stores embed_tokens and lm_head
# whenever the vocab was resized -- and every run here resizes by one for the
# introspection separator. At vocab 128257 x 2048 in bf16 that is ~1.0 GB per
# checkpoint against ~21 MB of actual LoRA weights: 97.9% of the file is a frozen copy
# of the base model.
#
# The single row that is NOT in the base model is the separator's embedding. It was
# random-initialized at resize time and frozen, so it never changed -- but the model
# trained WITH that specific vector, and a reload re-initializes it differently
# (transformers mean-resizing). Stripping the matrices while preserving that one row
# (4 KB) keeps checkpoints reloadable; stripping it outright does not.
EMBED_KEYS = ("embed_tokens.weight", "lm_head.weight")
ADDED_ROWS_FILE = "added_token_embeddings.safetensors"


def strip_embedding_tensors(exp_pattern: str, n_added_tokens: int = 1,
                            dry_run: bool = True, assume_yes: bool = False):
    """Post-hoc version of what StripEmbeddingSaveCallback now does at save time, for
    checkpoints written before that existed. Delegates to the same implementation.

    n_added_tokens is how many tokens the run added to the tokenizer (1 for the
    introspection separator); those rows are preserved so the checkpoint stays
    reloadable, and load_added_token_embeddings puts them back.
    """
    import sys
    sys.path.insert(0, os.getcwd())
    from src.introspector.training.model import strip_adapter_embeddings

    files = sorted(glob.glob(exp_pattern))
    print(f"\n{exp_pattern}\nFound {len(files)} adapter file(s)")
    if not files:
        return
    before = sum(_size(f) for f in files)
    print(f"  {_human(before)} total; keeping the last {n_added_tokens} row(s) per file")
    if dry_run:
        print(f"  DRY RUN -- nothing rewritten (expect ~{_human(before * 0.02)} to remain)")
        return
    if not assume_yes:
        res = input(f"Rewrite {len(files)} adapter file(s)? (Y/N)")
        if not res.lower().startswith("y"):
            print("  aborted")
            return
    n = 0
    for f in tqdm(files):
        n += bool(strip_adapter_embeddings(os.path.dirname(f), n_added_tokens, verbose=False))
    after = sum(_size(f) for f in files)
    print(f"  rewrote {n} file(s): {_human(before)} -> {_human(after)} "
          f"(freed {_human(before - after)})")


if __name__ == "__main__":
    GROUP = "outputs/runs/mlp_joint_weight_sweep"
    # Adapter weights: keep ONE neuron per (layer, position, gate) cell so the
    # weight-pattern analysis still has something to read. Set DRY_RUN=False to act.
    DRY_RUN = False
    delete_files_matching_pattern(f"{GROUP}/*/checkpoint-*/adapter_model.safetensors", keep_per_cell=1, dry_run=DRY_RUN)
    # Optimizer state and the tokenizer copy are never needed again -> delete all.
    delete_files_matching_pattern(f"{GROUP}/*/checkpoint-*/optimizer.pt", dry_run=DRY_RUN)
    delete_files_matching_pattern(f"{GROUP}/*/checkpoint-*/tokenizer.json", dry_run=DRY_RUN)

    # Frozen embeddings: ~1.0 GB of each checkpoint, 97.9% of the file. Keeps the
    # separator's row so the checkpoint stays reloadable for introspection evals.
    # strip_embedding_tensors(f"{GROUP}/*/checkpoint-*/adapter_model.safetensors",
    #                         keep_added_rows=True, dry_run=DRY_RUN)

    # other sweeps:
    # delete_files_matching_pattern("outputs/runs/mlp_joint_weight_sweep/*/checkpoint-*/optimizer.pt", dry_run=DRY_RUN)
    # delete_files_matching_pattern("outputs/runs/single_sae_detection_sweep_layer_pos/*/checkpoint-*/adapter_model.safetensors", keep_per_cell=1, dry_run=DRY_RUN)
