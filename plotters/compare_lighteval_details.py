#!/usr/bin/env python3
"""
Paired per-example comparison of two lighteval conditions (adapter vs base).

Aggregate scores hide two very different situations. If ARC accuracy is identical in
both conditions while lambada perplexity moves by four orders of magnitude, either:

  RANK PRESERVATION  the model's per-choice log-probabilities changed but their
                     ORDER did not. Accuracy reads only the argmax, so it cannot see
                     the damage. A real (and reportable) finding.
  NO EFFECT APPLIED  the two conditions produced byte-identical scores, i.e. the
                     accuracy path never saw the adapter. A bug; the numbers mean
                     nothing.

Byte-identical per-choice scores separate them, and nothing else does.

Run after evals/score_lighteval_benchmarks.py, pointed at a run directory:
    python plotters/compare_lighteval_details.py            # uses RUN_DIR below
    python plotters/compare_lighteval_details.py --columns  # dump parquet schema
"""
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]

# lighteval writes three nested struct columns: doc / metric / model_response.
# The per-choice log-probabilities we need are model_response["logprobs"], the gold
# index is doc["gold_index"], and the scored metric is whatever key metric holds.
SCORE_KEYS = ["logprobs", "unconditioned_logprobs", "logits"]


def _detail_files(run_dir: Path, condition: str) -> dict[str, Path]:
    """{task_name: parquet} for one condition, from the LATEST evaluation pass.

    A run evaluated twice has two timestamped detail dirs per condition. Unsorted glob
    order picked one arbitrarily (and differently per machine), and the early pass of the
    prev-token hyper sweep is broken: its "base" details are the ADAPTER's (identical
    scores), so pairing it gave 0% changed answers. Paths sort by their timestamp
    directory, so iterating sorted lets the latest pass overwrite earlier ones."""
    out = {}
    for f in sorted(glob.glob(str(run_dir / "lighteval" / condition / "details" / "**" / "*.parquet"),
                              recursive=True)):
        name = Path(f).name
        task = name.split("details_")[-1].rsplit("_", 1)[0]
        out[task] = Path(f)
    return out


def _as_array(v):
    """Normalize a per-choice score cell to a 1-D float array."""
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(v, dtype=float).ravel()
        except (TypeError, ValueError):
            return None
        return arr if arr.size else None
    return None


def _extract(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the nested detail schema to one row per example."""
    rows = []
    for _, r in df.iterrows():
        doc = r.get("doc") or {}
        resp = r.get("model_response") or {}
        met = r.get("metric") or {}
        scores = next((_as_array(resp.get(k)) for k in SCORE_KEYS
                       if _as_array(resp.get(k)) is not None), None)
        rows.append({
            "id": doc.get("id"),
            "gold": doc.get("gold_index"),
            "scores": scores,
            # generative tasks have no choice scores; keep the emitted text instead.
            # `or` on a numpy array raises, so length-check explicitly.
            "text": (lambda t: str(t[0]) if t is not None and len(t) else None)(resp.get("text")),
            **{f"metric_{k}": v for k, v in (met or {}).items()},
        })
    return pd.DataFrame(rows)


def compare_task(base_path: Path, adapter_path: Path, task: str) -> dict:
    b, a = _extract(pd.read_parquet(base_path)), _extract(pd.read_parquet(adapter_path))
    # align on example id where available, else by position
    if b["id"].notna().all() and a["id"].notna().all():
        merged = b.merge(a, on="id", suffixes=("_base", "_adapter"))
    else:
        n = min(len(b), len(a))
        merged = pd.concat([b.iloc[:n].add_suffix("_base").reset_index(drop=True),
                            a.iloc[:n].add_suffix("_adapter").reset_index(drop=True)], axis=1)
    row = {"task": task, "n": len(merged)}

    metric_cols = [c for c in b.columns if c.startswith("metric_")]
    for mc in metric_cols:
        row[f"{mc}_base"] = float(np.nanmean(pd.to_numeric(merged[f"{mc}_base"], errors="coerce")))
        row[f"{mc}_adapter"] = float(np.nanmean(pd.to_numeric(merged[f"{mc}_adapter"], errors="coerce")))

    same_scores, same_argmax, deltas, n_cmp = 0, 0, [], 0
    for bs, as_ in zip(merged["scores_base"], merged["scores_adapter"]):
        if bs is None or as_ is None or np.shape(bs) != np.shape(as_):
            continue
        n_cmp += 1
        if np.array_equal(bs, as_):
            same_scores += 1
        if int(np.argmax(bs)) == int(np.argmax(as_)):
            same_argmax += 1
        deltas.append(float(np.mean(np.abs(np.asarray(bs) - np.asarray(as_)))))
    if not n_cmp:
        row["note"] = "no per-choice scores (generative task)"
        if "text_base" in merged:
            same_text = (merged["text_base"].astype(str) == merged["text_adapter"].astype(str)).mean()
            row["identical_text_frac"] = float(same_text)
        return row

    row["n_compared"] = n_cmp
    row["identical_scores_frac"] = same_scores / n_cmp
    row["same_argmax_frac"] = same_argmax / n_cmp
    row["mean_abs_logprob_delta"] = float(np.mean(deltas))

    # per-example metric flips: far more sensitive than the aggregate. Two runs can
    # both score 0.60 while disagreeing on 15% of items.
    for mc in metric_cols:
        b_vals = pd.to_numeric(merged[f"{mc}_base"], errors="coerce")
        a_vals = pd.to_numeric(merged[f"{mc}_adapter"], errors="coerce")
        ok = b_vals.notna() & a_vals.notna()
        if ok.any():
            row[f"{mc}_changed_frac"] = float((b_vals[ok] != a_vals[ok]).mean())
    return row


def summarize_run(run_dir: Path, cache=True) -> dict:
    """Per-example comparison for one run, aggregated over tasks. Cached to
    benchmarks_per_example.json so repeated plotting does not re-read every parquet."""
    import json
    out_path = Path(run_dir) / "lighteval" / "benchmarks_per_example.json"
    base, adapter = _detail_files(Path(run_dir), "base"), _detail_files(Path(run_dir), "adapter")
    shared = sorted(set(base) & set(adapter))
    # the cache records which detail files it was computed from; a cache written before
    # the latest-pass fix (or from different files) does not match and is recomputed
    sources = {t: [str(base[t].parent.name), str(adapter[t].parent.name)] for t in shared}
    if cache and out_path.exists():
        cached = json.loads(out_path.read_text())
        if cached.get("sources") == sources:
            return cached
    if not shared:
        return {}
    rows = [compare_task(base[t], adapter[t], t) for t in shared]
    df = pd.DataFrame(rows)
    changed = [c for c in df.columns if c.endswith("_changed_frac")]
    out = {"tasks": len(df), "sources": sources}
    if changed:
        # mean over tasks of "fraction of examples whose score changed"
        out["example_changed_frac"] = float(np.nanmean(df[changed].to_numpy(dtype=float)))
        for c in changed:
            out[c] = float(np.nanmean(df[c]))
    if "same_argmax_frac" in df:
        out["same_argmax_frac"] = float(np.nanmean(df["same_argmax_frac"]))
    if cache and out:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
    return out


def main(run_dir: Path, dump_columns=False):
    base, adapter = _detail_files(run_dir, "base"), _detail_files(run_dir, "adapter")
    shared = sorted(set(base) & set(adapter))
    if not shared:
        raise SystemExit(f"No matching detail files under {run_dir}/lighteval/"
                         f"{{base,adapter}}/details (base={list(base)}, adapter={list(adapter)})")
    if dump_columns:
        df = pd.read_parquet(base[shared[0]])
        print(f"columns in {shared[0]}: {list(df.columns)}")
        print(df.head(2).to_string())
        return

    rows = [compare_task(base[t], adapter[t], t) for t in shared]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if "identical_scores_frac" in df:
        for _, r in df.dropna(subset=["identical_scores_frac"]).iterrows():
            if r["identical_scores_frac"] > 0.99:
                print(f"\n[{r['task']}] BUG: {r['identical_scores_frac']:.1%} of examples have "
                      f"BYTE-IDENTICAL per-choice scores. The adapter never reached this "
                      f"scoring path -- these numbers describe one model evaluated twice.")
            elif r["same_argmax_frac"] > 0.99:
                print(f"\n[{r['task']}] RANK PRESERVATION: scores differ (mean |delta| = "
                      f"{r['mean_abs_logprob_delta']:.3f} nats) but the argmax is unchanged on "
                      f"{r['same_argmax_frac']:.1%} of examples. Accuracy is blind to this "
                      f"damage; it is not evidence that behaviour is preserved.")
            else:
                print(f"\n[{r['task']}] predictions genuinely differ on "
                      f"{1 - r['same_argmax_frac']:.1%} of examples "
                      f"(mean |delta| = {r['mean_abs_logprob_delta']:.3f} nats).")


if __name__ == "__main__":
    RUN_DIR = ROOT / "outputs/runs/mlp_joint_prev_token_hyper/joint_N10.24_POS-1_gateoff_scale10_kl0.01_w0.01"
    main(RUN_DIR, dump_columns="--columns" in sys.argv)
