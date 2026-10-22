#!/usr/bin/env python3
"""
Find and relocate/delete run dirs that were produced by the WRONG model inside a
sweep group -- e.g. Llama-3.2-3B runs accidentally dumped into the 1B group
`mlp_endo_sweep_layer_token` on 9/15, when the correct 1B runs are from 9/1-9/2.

It NEVER deletes anything by default. Without --apply it only scans and writes a
manifest CSV you can eyeball. Classification is by MODEL IDENTITY, not date:

  1. lighteval path  .../results/meta-llama/Llama-3.2-<1B|3B>/   (definitive)
  2. total_flos in all_results.json  (1B ~9.2e16, 3B ~2.4e17 -> bimodal)
  3. neither present -> model UNKNOWN; the run is only flagged if its mtime is on
     or after --cutoff-date, and even then it is left for review unless
     --include-uncertain is passed.

Only runs whose detected model == --wrong-model are acted on (plus UNKNOWN+recent
ones if --include-uncertain). Runs detected as --keep-model are never touched.

Typical use on the cluster
---------------------------
# 1) DRY RUN: just classify and write outputs/plots/relocate_manifest.csv
python utils/relocate_wrong_model_runs.py \
    --runs-root /projectnb/llamagrp/<you>/low-level-introspection/outputs/runs \
    --source mlp_endo_sweep_layer_token \
    --target mlp_endo_sweep_layer_token_llama323b

# 2) MERGE the 3B runs into the 3B group (skips names already present there):
python utils/relocate_wrong_model_runs.py ... --apply --action move

# 3) Or DELETE them (defaults to moving to a _trash_ dir, not hard rm):
python utils/relocate_wrong_model_runs.py ... --apply --action delete
"""
import os
import csv
import json
import time
import shutil
import argparse
from glob import glob
from pathlib import Path
from datetime import datetime, timezone

LIGHTEVAL_GLOB = "lighteval/**/results/meta-llama/*"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path,
                   default=Path(__file__).parents[1] / "outputs" / "runs",
                   help="Dir that holds the sweep groups (default: repo outputs/runs).")
    p.add_argument("--source", required=True,
                   help="Group to clean, e.g. mlp_endo_sweep_layer_token")
    p.add_argument("--target", default=None,
                   help="Group to move wrong-model runs INTO (required for --action move/copy).")
    p.add_argument("--keep-model", default="1B", help="Model the source SHOULD contain.")
    p.add_argument("--wrong-model", default="3B", help="Model to pull out of the source.")

    p.add_argument("--flos-threshold", type=float, default=None,
                   help="total_flos above this => wrong(3B). Default: auto-calibrated from "
                        "lighteval-confirmed runs, else 1.5e17.")
    p.add_argument("--cutoff-date", default="2026-09-10",
                   help="YYYY-MM-DD. Runs modified on/after this are date-'suspect'.")
    p.add_argument("--select-by", choices=["both", "model", "date"], default="both",
                   help="both (default): flos primary, date fallback, flag disagreements as "
                        "REVIEW. model: flos/lighteval only. date: purely mtime>=cutoff.")
    p.add_argument("--min-age-min", type=float, default=30.0,
                   help="Skip acting on runs modified within this many minutes (active jobs).")

    p.add_argument("--apply", action="store_true", help="Actually move/copy/delete. Off = dry run.")
    p.add_argument("--action", choices=["report", "move", "copy", "delete"], default="report")
    p.add_argument("--on-conflict", choices=["skip", "overwrite", "suffix"], default="skip",
                   help="If the destination run name already exists (default: skip & report).")
    p.add_argument("--include-uncertain", action="store_true",
                   help="Also act on UNKNOWN-model runs whose mtime >= cutoff-date.")
    p.add_argument("--hard-delete", action="store_true",
                   help="For --action delete: rm -rf instead of moving to the trash dir.")
    p.add_argument("--trash-dir", type=Path, default=None,
                   help="Where 'delete' relocates runs (default: <runs-root>/_trash_wrong_model).")
    p.add_argument("--manifest", type=Path, default=None,
                   help="CSV path for the report (default: <runs-root>/../plots/relocate_manifest.csv).")
    return p.parse_args()


def norm_model(name: str) -> str:
    """Map a raw model string to '1B'/'3B'/... for comparison."""
    n = name.lower()
    if "3b" in n:
        return "3B"
    if "1b" in n:
        return "1B"
    return name


def lighteval_models(run_dir: Path) -> set:
    """Model names recorded in any lighteval results path under this run."""
    out = set()
    for pth in glob(str(run_dir / LIGHTEVAL_GLOB), recursive=True):
        p = Path(pth)
        if p.is_dir():
            out.add(norm_model(p.name))
    return out


def read_total_flos(run_dir: Path):
    for fname in ("all_results.json", "train_results.json"):
        f = run_dir / fname
        if f.exists():
            try:
                v = json.loads(f.read_text()).get("total_flos")
                if v is not None:
                    return float(v)
            except (ValueError, OSError):
                pass
    return None


def run_mtime(run_dir: Path) -> float:
    """Newest mtime among result files (fallback: dir mtime). Result files are
    written at the end of a run, so this dates the run, not a later re-touch."""
    candidates = [run_dir]
    for name in ("all_results.json", "train_results.json", "test_results.json",
                 "interventional_test_metrics_gaussian.json",
                 "endogenous_test_metrics_endogenous.json",
                 "intervention_results_metrics.json"):
        f = run_dir / name
        if f.exists():
            candidates.append(f)
    return max(c.stat().st_mtime for c in candidates)


def calibrate_flos_threshold(records, keep_model, wrong_model, override):
    """Where to split total_flos into keep(1B) vs wrong(3B).

    Endo runs have no lighteval, so we can't anchor on confirmed model names. Instead
    we find the natural bimodal gap in the flos values: 1B clusters ~9e16, a 3B
    contaminant sits ~2.6x higher, so the largest RELATIVE jump between sorted values
    is the 1B/3B boundary. If there is no big jump (all one model), we don't split."""
    if override is not None:
        return override, "cli"
    import statistics as st
    # if lighteval happens to be present on some runs, prefer it
    keep = [r["total_flos"] for r in records
            if r["lighteval_model"] == keep_model and r["total_flos"]]
    wrong = [r["total_flos"] for r in records
             if r["lighteval_model"] == wrong_model and r["total_flos"]]
    if keep and wrong:
        return (st.median(keep) * st.median(wrong)) ** 0.5, "geo-midpoint of confirmed runs"

    flos = sorted(r["total_flos"] for r in records if r["total_flos"])
    if len(flos) >= 2:
        ratio, i = max((flos[j + 1] / flos[j], j)
                       for j in range(len(flos) - 1) if flos[j] > 0)
        if ratio >= 1.5:                       # a real gap (1B vs 3B ~2.6x), not jitter
            return (flos[i] * flos[i + 1]) ** 0.5, f"bimodal gap split ({ratio:.1f}x jump)"
    if keep:
        return st.median(keep) * 1.8, "1.8x confirmed-keep median"
    if wrong:
        return st.median(wrong) * 0.6, "0.6x confirmed-wrong median"
    if flos:
        return st.median(flos) * 1.5, "1.5x overall median (no clear bimodal gap)"
    return 1.5e17, "default (no flos to calibrate)"


def classify(rec, flos_threshold, keep_model, wrong_model):
    """Return (model, confidence) from model signals only (lighteval/flos).
    model may be keep/wrong/other/None(unknown)."""
    lem = rec["lighteval_model"]
    if lem and not lem.startswith("MIXED"):
        return lem, "high(lighteval)"
    if lem:
        return lem, "review(mixed-lighteval)"
    flos = rec["total_flos"]
    if flos is not None:
        model = wrong_model if flos > flos_threshold else keep_model
        ratio = flos / flos_threshold          # far from split => high; near => medium
        conf = "high(flos)" if (ratio > 1.4 or ratio < 0.7) else "medium(flos-near-split)"
        return model, conf
    return None, "unknown(no-model-signal)"


def decide(model, date_suspect, keep_model, wrong_model, select_by, include_uncertain):
    """Combine the model verdict with the date cross-check into an action."""
    if select_by == "date":                    # trust the clock alone
        return "ACT" if date_suspect else "KEEP"

    if model == wrong_model:
        if select_by == "both" and not date_suspect:
            return "REVIEW"                    # flos says wrong-model but the run is OLD
        return "ACT"
    if model == keep_model:
        return "KEEP"                          # a recent keep-model run is still legit
    if model is None:                          # no flos/lighteval -> lean on date
        if date_suspect:
            return "ACT" if include_uncertain else "REVIEW"
        return "KEEP"
    return "REVIEW"                            # some other / mixed model


def unique_dest(dst: Path) -> Path:
    i = 1
    base = dst
    while dst.exists():
        dst = base.with_name(f"{base.name}__dup{i}")
        i += 1
    return dst


def main():
    a = parse_args()
    src_group = a.runs_root / a.source
    if not src_group.is_dir():
        raise SystemExit(f"source group not found: {src_group}")
    cutoff_ts = datetime.strptime(a.cutoff_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp()
    now = time.time()

    run_dirs = sorted(d for d in src_group.iterdir() if d.is_dir())
    records = []
    for d in run_dirs:
        lem = sorted(lighteval_models(d))
        records.append({
            "run": d.name, "path": d,
            "lighteval_model": None if not lem
                               else lem[0] if len(lem) == 1
                               else "MIXED:" + ",".join(lem),
            "total_flos": read_total_flos(d),
            "mtime": run_mtime(d),
        })

    flos_threshold, how = calibrate_flos_threshold(
        records, a.keep_model, a.wrong_model, a.flos_threshold)
    print(f"[reloc] flos threshold = {flos_threshold:.3e}  ({how})")
    print(f"[reloc] cutoff date    = {a.cutoff_date}  (UNKNOWN runs newer than this are suspect)")

    for r in records:
        model, conf = classify(r, flos_threshold, a.keep_model, a.wrong_model)
        r["model"], r["confidence"] = model, conf
        r["date_suspect"] = r["mtime"] >= cutoff_ts
        r["decision"] = decide(model, r["date_suspect"], a.keep_model, a.wrong_model,
                               a.select_by, a.include_uncertain)

    # ---- act ----
    if a.action in ("move", "copy") and not a.target:
        raise SystemExit(f"--action {a.action} needs --target")
    target_group = (a.runs_root / a.target) if a.target else None
    trash_dir = a.trash_dir or (a.runs_root / "_trash_wrong_model" / a.source)

    for r in records:
        r["action_taken"], r["dest"] = "", ""
        if r["decision"] != "ACT":
            continue
        age_min = (now - r["mtime"]) / 60.0
        if age_min < a.min_age_min:
            r["action_taken"] = f"SKIP(too-recent {age_min:.0f}min)"
            continue
        if a.action == "report" or not a.apply:
            r["action_taken"] = "WOULD-" + (a.action if a.action != "report" else "act")
            if target_group is not None:
                r["dest"] = str(target_group / r["run"])
            continue

        src = r["path"]
        if a.action in ("move", "copy"):
            dst = target_group / r["run"]
            target_group.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                if a.on_conflict == "skip":
                    r["action_taken"] = "CONFLICT-skip"
                    r["dest"] = str(dst)
                    continue
                if a.on_conflict == "overwrite":
                    shutil.rmtree(dst)
                elif a.on_conflict == "suffix":
                    dst = unique_dest(dst)
            if a.action == "move":
                shutil.move(str(src), str(dst))
            else:
                shutil.copytree(src, dst)
            r["action_taken"], r["dest"] = a.action.upper(), str(dst)
        elif a.action == "delete":
            if a.hard_delete:
                shutil.rmtree(src)
                r["action_taken"] = "HARD-DELETE"
            else:
                dst = trash_dir / r["run"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    dst = unique_dest(dst)
                shutil.move(str(src), str(dst))
                r["action_taken"], r["dest"] = "TRASH", str(dst)

    # ---- report ----
    manifest = a.manifest or (a.runs_root.parent / "plots" / "relocate_manifest.csv")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    cols = ["run", "model", "confidence", "decision", "total_flos",
            "lighteval_model", "mtime_iso", "date_suspect", "action_taken", "dest"]
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({
                "run": r["run"], "model": r["model"], "confidence": r["confidence"],
                "decision": r["decision"], "total_flos": r["total_flos"],
                "lighteval_model": r["lighteval_model"],
                "mtime_iso": datetime.fromtimestamp(r["mtime"]).isoformat(timespec="minutes"),
                "date_suspect": r["date_suspect"],
                "action_taken": r["action_taken"], "dest": r["dest"],
            })

    from collections import Counter
    dec = Counter(r["decision"] for r in records)
    act = Counter(r["action_taken"] for r in records if r["action_taken"])
    print(f"\n[reloc] {len(records)} runs in {a.source}")
    print(f"[reloc] decisions: {dict(dec)}")
    if act:
        print(f"[reloc] actions:   {dict(act)}")
    if not a.apply:
        print("[reloc] DRY RUN -- nothing changed. Re-run with --apply to act.")
    print(f"[reloc] wrote manifest -> {manifest}")

    review = [r["run"] for r in records if r["decision"] == "REVIEW"]
    if review:
        print(f"[reloc] {len(review)} run(s) need manual REVIEW (unknown/mixed model): "
              f"{review[:8]}{' ...' if len(review) > 8 else ''}")


if __name__ == "__main__":
    main()
