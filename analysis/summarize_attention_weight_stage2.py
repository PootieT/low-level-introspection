#!/usr/bin/env python3
"""
Every number Stage 2 produced, in two tables and four figures.

Point it at the PARENT dir of the Stage 2 sweep:
    outputs/runs/attn_weight_endo_picked/

plot_attention_srctoken_sweep.py draws the headline readout figure (arm x eval). This
script covers everything else the runs already wrote and nothing else reads: the
interventional scale sweep, the OOD ladder, the LM cost, the Pearson/Spearman gap, and
the per-cell aggregate with error bars.

WHY A NEW FILE AND NOT A WIDENING. compare_sft_vs_joint.py owns the OOD ladder but joins
arms on KEY = ["layer","position","neuron"] against a hardcoded MLP grid
(compare_sft_vs_joint.py:56-60); an attention cell needs head and src_offset in that key,
so porting it costs more than writing this. The run-name pattern is NOT duplicated,
though -- RUN_RE and ARM_ORDER are imported from plot_attention_srctoken_sweep, following
compare_lora_weight_patterns.py:47-51. A third copy of that regex is how the two arms got
averaged together in the first place.

Outputs (under outputs/plots/<sweep>/):
    stage2_master.csv      one row per run, every metric joined
    stage2_by_cell.csv     mean +- SE over the heads in each (cell, arm). The paper table
    stage2_validity.csv    the alignment / low-variance checks, one row per run
    scale_sweep_grid.png   head vs fresh-probe ceiling corr against injection scale
    inj_std_saturation.png the sigmoid saturation evidence (see the note below)
    ood_ladder.png         endogenous corr per OOD corpus
    pearson_vs_spearman.png  how much of the correlation the right tail carries

A CELL IS (layer, head, src_offset) BUT A ROW IS A RUN. Each cell was trained at three
heads -- the hand-picked one plus two fill heads -- and the spread across those three is
the error bar on everything in stage2_by_cell.csv. Which head was hand-picked is read
from the `_cell_l<L>_h<H>_src<OFF>.json` config filenames, not inferred: the fill heads
happen to be 0 and 1 today, and "the head that is not 0 or 1" breaks silently the day a
pick lands on one of them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The sibling owns the attention run-name pattern and the arm ordering; import rather
# than restate. (compare_lora_weight_patterns.py:47-51 sets this precedent.)
sys.path.insert(0, str(Path(__file__).parent))
from plot_attention_srctoken_sweep import (  # noqa: E402
    RUN_RE, ARM_ORDER, ARM_TITLES, flag_low_variance,
)

CELL_CONFIG_RE = re.compile(r"^_cell_l(?P<layer>\d+)_h(?P<head>\d+)_src(?P<src>-?\d+)$")

# measure_attention_alignment scans a window of offsets around the cached one and reports
# which matched best. Agreement at the WRONG offset and mere numerical disagreement at the
# right one are different failures, and train_introspector_joint.py:813 folds them into
# one boolean: `best == TRAIN_OFFSET and best_corr > 0.99`. A run sitting at 0.98956 with
# the offset correct then reads exactly like an off-by-one, which is the bug class that
# produced scripts_old_with_off_by_one_bug/. Split here.
TRAIN_OFFSET = 0
ALIGN_CORR_OK = 0.99

OOD_CORPORA = ("shuffled", "wikipedia", "math", "code", "c4_zh", "c4_bn")


def _cell_key(cell: str):
    """Sort "L7@-27" by (offset, layer) numerically. Sorting the string puts L11 before
    L2 and the layer axis then reads out of order in every faceted figure."""
    layer, src = cell.lstrip("L").split("@")
    return (int(src), int(layer))


def _cells(df: pd.DataFrame) -> list:
    return sorted(df["cell"].unique(), key=_cell_key)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", help="parent directory holding the run directories")
    ap.add_argument("--configs-dir", default=None,
                    help="experiment_configs/<group> (default: derived from sweep name). "
                         "Supplies the train size and the hand-picked head per cell.")
    ap.add_argument("--data-dir", default="data",
                    help="cache root, for dataset_signal_stats.json (target mean/std)")
    ap.add_argument("--out", default=None,
                    help="output dir (default outputs/plots/<sweep name>)")
    return ap.parse_args()


def picked_heads(configs_dir: Path) -> dict:
    """{(layer, src_offset): head} from the per-cell config filenames.

    These files are written once per cell by the Stage 2 script before any arm runs, so
    they are the authoritative record of which head was hand-picked from the Stage 1 scan.
    """
    picked = {}
    for p in sorted(configs_dir.glob("_cell_*.json")):
        m = CELL_CONFIG_RE.match(p.stem)
        if not m:
            print(f"[sum] ignoring unparsable cell config {p.name}")
            continue
        picked[(int(m["layer"]), int(m["src"]))] = int(m["head"])
    return picked


def signal_stats(data_dir: Path, train_count: int) -> dict:
    """{(layer, index, src_offset): stats} for caches of exactly `train_count` rows.

    The count is the disambiguator, and it is not optional: the same (layer, index,
    offset) is cached again at 64 rows by the port validators and at 3000 by the pilot,
    and a cell's mean/std differ enough between 3000 and 60000 rows to move the
    low-variance flag. Keyed on the train split, which is what the intervention sampler's
    moments come from too (train_introspector_joint.py:1341-1352).
    """
    stats = {}
    for p in sorted(data_dir.glob("*/*/dataset_signal_stats.json")):
        try:
            payload = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"[sum] skipping unreadable {p}: {e}")
            continue
        for row in payload.get("train", []):
            if row.get("signal_type") != "attention_weight":
                continue
            if int(row.get("count", -1)) != train_count:
                continue
            key = (int(row["signal_layer"]), int(row["signal_index"]),
                   int(row["src_offset"]))
            stats[key] = row
    return stats


def _read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _scale_sweep(run: Path) -> dict:
    """head vs ceiling corr at the natural scale and at the widest one."""
    p = run / "interventional_scale_sweep.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p).sort_values("scale")
    lo, hi = d.iloc[0], d.iloc[-1]
    return {
        "sweep_scale_min": float(lo["scale"]), "sweep_scale_max": float(hi["scale"]),
        "head_corr_at1": float(lo["head_corr"]), "ceil_corr_at1": float(lo["ceiling_corr"]),
        "head_corr_atmax": float(hi["head_corr"]), "ceil_corr_atmax": float(hi["ceiling_corr"]),
        "inj_std_at1": float(lo["inj_std"]), "inj_std_atmax": float(hi["inj_std"]),
        # RidgeCV pinned at the top of its grid means the fresh probe regularized as hard
        # as it could, i.e. it found nothing -- corroboration for a near-zero ceiling.
        "best_alpha_at1": float(lo["best_alpha"]),
    }


def _lm_cost(run: Path) -> dict:
    """KL(base||tuned) and capability, per eval length.

    Absent on most runs BY DESIGN, not by failure: the Stage 2 script scores two heads on
    the unanchored arm (where the achieved KL is the result and needs an error bar) and
    one on the anchored arm (where it is only a check that the budget bound), and refuses
    gate-on runs outright because the gate makes the measurement 0 by construction.
    """
    payload = _read_json(run / "lm_degradation.json")
    if not payload:
        return {}
    out = {"lm_negligible": payload.get("negligible")}
    for row in payload.get("lengths", []):
        L = int(row["length"])
        out[f"kl_at{L}"] = row.get("kl_mean")
        out[f"top1_at{L}"] = row.get("top1_agreement")
        if row.get("ppl_base"):
            out[f"ppl_ratio_at{L}"] = row["ppl_tuned"] / row["ppl_base"]
        if "kl_mean_beyond_trainlen" in row:
            out[f"kl_beyond_at{L}"] = row["kl_mean_beyond_trainlen"]
            out[f"kl_within_at{L}"] = row["kl_mean_within_trainlen"]
    return out


def _ood(run: Path) -> dict:
    p = run / "ood_generalization.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    out = {}
    for _, r in d.iterrows():
        out[f"ood_endo_{r['corpus']}"] = float(r["endogenous_corr"])
        out[f"ood_interv_{r['corpus']}"] = float(r["interventional_corr"])
    return out


def _alignment(run: Path) -> dict:
    d = _read_json(run / "activation_drift.json")
    if not d:
        return {}
    out = {
        "align_best_offset": d.get("best_alignment_offset"),
        "align_best_corr": d.get("best_alignment_corr"),
        "alignment_ok_raw": d.get("alignment_ok"),
    }
    if isinstance(d.get("drift_tuned_vs_cached"), dict):
        out["drift_vs_cached_corr"] = d["drift_tuned_vs_cached"].get("corr")
    if isinstance(d.get("readout_tracking"), dict):
        out["coupling_gap"] = d["readout_tracking"].get("coupling_gap")
    return out


def collect(sweep_dir: Path, configs_dir: Path, data_dir: Path) -> pd.DataFrame:
    picked = picked_heads(configs_dir)
    print(f"[sum] hand-picked heads from {configs_dir}: "
          + ", ".join(f"L{l}@{o}->h{h}" for (l, o), h in sorted(picked.items())))

    stats_by_count = {}
    rows, skipped = [], []
    for run in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        m = RUN_RE.search(run.name)
        if not m:
            skipped.append((run.name, "name does not parse"))
            continue
        endo = _read_json(run / "endogenous_test_metrics_endogenous.json")
        if endo is None:
            skipped.append((run.name, "no endogenous metrics"))
            continue

        cfg = _read_json(configs_dir / f"{run.name}.json")
        if cfg is None:
            skipped.append((run.name, f"no config under {configs_dir}"))
            continue
        train_count = int(cfg["train_dataset"]["max_samples"])
        if train_count not in stats_by_count:
            stats_by_count[train_count] = signal_stats(data_dir, train_count)

        gate = m.group("gate") or "off"
        kl = m.group("kl") or "none"
        layer, head, src = int(m["layer"]), int(m["head"]), int(m["src"])
        row = {
            "run": run.name,
            "cell": f"L{layer}@{src}",
            "signal": f"{m['code']}{layer}.{head}@{src}",
            "layer": layer, "head": head, "src_offset": src,
            "gate": gate, "kl": kl, "arm": f"gate{gate}/{kl}",
            "train_samples": train_count,
            "is_picked_head": picked.get((layer, src)) == head,
        }
        # endogenous metrics are flat; .get() throughout because spearman was added to
        # metrics.py after most of outputs/ was written
        for k in ("corr", "spearman", "l2_dist", "avg_diff", "std_diff"):
            row[k] = endo.get(k)

        interv = _read_json(run / "interventional_test_metrics_gaussian.json")
        if interv is not None:
            assert "test" in interv, (
                f"{run}/interventional_test_metrics_gaussian.json nests its scalars under "
                f"'test'; got keys {sorted(interv)}")
            for k in ("corr", "spearman", "l2_dist", "std_diff"):
                row[f"interv_{k}"] = interv["test"].get(k)

        st = stats_by_count[train_count].get((layer, head, src))
        if st is not None:
            row["target_mean"] = st["mean"]
            row["target_std"] = st["std"]
            row["target_cv"] = st["std"] / st["mean"] if st["mean"] else float("nan")
            row["target_max"] = st["max"]
        low, reason = flag_low_variance(m.group("code"), row.get("target_std"),
                                        row.get("target_mean"))
        row["low_variance"], row["low_variance_reason"] = low, reason

        row.update(_scale_sweep(run))
        row.update(_lm_cost(run))
        row.update(_ood(run))
        row.update(_alignment(run))
        rows.append(row)

    if skipped:
        print(f"[sum] skipped {len(skipped)}: {skipped[:5]}")
    if not rows:
        raise SystemExit(f"no parsable runs under {sweep_dir}")
    return pd.DataFrame(rows)


def by_cell(df: pd.DataFrame) -> pd.DataFrame:
    """mean +- SE over the heads trained in each (cell, arm).

    sem() over n=3 is what the third head is for; it is also why n is carried in the
    table. Three is not many, and a reader comparing cells has to be able to see that.
    """
    metrics = [c for c in ("corr", "spearman", "interv_corr", "interv_spearman",
                           "head_corr_at1", "ceil_corr_at1", "kl_at32", "top1_at32")
               if c in df.columns]
    g = df.groupby(["cell", "layer", "src_offset", "arm"], as_index=False)
    out = g.agg(n=("run", "size"),
                n_scored=(metrics[0], "count"),
                **{f"{c}_{s}": (c, s) for c in metrics for s in ("mean", "sem")})
    return out.sort_values(["src_offset", "layer", "arm"])


def validity(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Split the alignment verdict, then say plainly what is and is not interpretable."""
    v = df[["run", "cell", "arm", "is_picked_head", "align_best_offset",
            "align_best_corr", "alignment_ok_raw", "low_variance",
            "low_variance_reason", "target_mean", "target_std"]].copy()
    off = v["align_best_offset"]
    corr = v["align_best_corr"]
    v["align_verdict"] = np.select(
        [off.isna(),
         off != TRAIN_OFFSET,
         corr < ALIGN_CORR_OK],
        ["not measured",
         "HARD FAIL: cached signal sits at the wrong token (off-by-one)",
         f"warn: right token, corr {ALIGN_CORR_OK} not reached"],
        default="ok")
    v.to_csv(out / "stage2_validity.csv", index=False)

    hard = v[v["align_verdict"].str.startswith("HARD FAIL")]
    warn = v[v["align_verdict"].str.startswith("warn")]
    dead = v[v["low_variance"].fillna(False)]
    print("\n[sum] === validity ===")
    print(f"  alignment measured on {int(off.notna().sum())}/{len(v)} runs "
          f"(it only runs with the gate off, or with --force-drift-eval)")
    if len(hard):
        print(f"  *** {len(hard)} HARD FAIL (wrong token): {sorted(hard['run'])}")
    else:
        print("  0 hard failures: every measured run agreed at the cached token")
    print(f"  {len(warn)} runs matched the right token but under corr {ALIGN_CORR_OK} "
          f"-- numerical, not positional. alignment_ok in the raw json calls these "
          f"failures too, which is why this table exists.")
    if len(warn):
        print("    " + ", ".join(f"{r.run.split('_POS')[0]}={r.align_best_corr:.4f}"
                                 for r in warn.head(6).itertuples()))
    print(f"  {len(dead)} runs flagged low variance / dead cell "
          f"({', '.join(sorted(set(dead['low_variance_reason']))) or '-'}); their "
          f"correlations are reported but should not be read as introspection")
    return v


def fig_scale_sweep(df: pd.DataFrame, sweep_dir: Path, out: Path):
    """Per cell: trained head vs a freshly fit probe, against injection scale.

    The ceiling is the question that matters here. If the fresh probe cannot recover the
    injected value either, the injection never reached the readout (routing), and no
    amount of training the head would have helped (reading).
    """
    cells = _cells(df)
    arms = [a for a in ARM_ORDER if a in set(df["arm"])]
    ncol = 3
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.3 * nrow),
                             squeeze=False, sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, cell in zip(axes.ravel(), cells):
        for i, arm in enumerate(arms):
            runs = df[(df["cell"] == cell) & (df["arm"] == arm) & df["is_picked_head"]]
            for _, r in runs.iterrows():
                p = sweep_dir / r["run"] / "interventional_scale_sweep.csv"
                if not p.exists():
                    continue
                d = pd.read_csv(p).sort_values("scale")
                ax.plot(d["scale"], d["head_corr"], "-o", ms=4, color=cmap(i),
                        label=f"{arm} head")
                ax.plot(d["scale"], d["ceiling_corr"], "--s", ms=4, color=cmap(i),
                        alpha=0.55, label=f"{arm} ceiling")
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(f"{cell}  (picked head)")
        ax.set_xlabel("injection scale (x natural log-odds std)")
        ax.grid(alpha=0.25, which="both")
    axes[0][0].set_ylabel("interventional corr")
    axes[0][0].legend(fontsize=7, ncol=2)
    for ax in axes.ravel()[len(cells):]:
        ax.axis("off")
    fig.suptitle("does the injected value reach the readout at all?\n"
                 "solid = trained head, dashed = fresh ridge probe (ceiling). "
                 "At W=1.0 no interventional loss was trained, so this is the null baseline.")
    fig.tight_layout()
    fig.savefig(out / "scale_sweep_grid.png", dpi=150)
    plt.close(fig)


def fig_inj_std(df: pd.DataFrame, sweep_dir: Path, out: Path):
    """Saturation evidence, such as it is.

    For a logit signal inj_std is linear in scale. For a post-softmax weight the sampler
    draws in log odds and maps through a sigmoid, so past some scale the draws pile up
    against 0 and 1 and inj_std stops growing. That bend is the ONLY saturation signal in
    these files: make_bounded_prob_intervention_fn counts clamped draws (n_drawn /
    n_clamped) but interventional_scale_sweep never reads them, so "corr rose at scale 16"
    cannot be separated from "the model learned the clamp" without it.

    Each curve is divided by its OWN value at the smallest scale, so the reference is one
    diagonal instead of one dotted line per cell. Absolute inj_std spans 0.008 to 0.17
    across these cells, so an unnormalized plot anchors the reference to whichever cell
    happens to be drawn first and every other cell then sits above it -- which reads as
    the opposite of saturation.

    READ THE BEND, NOT THE SIDE OF THE DIAGONAL. A cell whose mean weight is ~1e-3 sits
    at log odds ~-7, far out on the flat part of the sigmoid, so widening the log-odds
    std first moves draws INTO the responsive region and the probability spread grows
    faster than the scale -- the far-offset cells start well above the diagonal. What
    every cell does share is that the slope falls towards zero. That flattening is the
    saturation; being below the diagonal is only what it looks like for cells that
    started near the middle of the sigmoid already.
    """
    cells = _cells(df)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.get_cmap("viridis")
    ref = None
    for i, cell in enumerate(cells):
        runs = df[(df["cell"] == cell) & df["is_picked_head"]]
        if not len(runs):
            continue
        p = sweep_dir / runs.iloc[0]["run"] / "interventional_scale_sweep.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p).sort_values("scale")
        base = float(d["inj_std"].iloc[0])
        if base <= 0:
            continue
        ax.plot(d["scale"] / float(d["scale"].iloc[0]), d["inj_std"] / base, "-o", ms=5,
                color=cmap(i / max(len(cells) - 1, 1)),
                label=f"{cell}  (std@1 = {base:.3g})")
        if ref is None:
            ref = d
    if ref is not None:
        x = ref["scale"] / float(ref["scale"].iloc[0])
        ax.plot(x, x, "k:", lw=1.8, label="slope 1 (spread tracks the scale)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("injection scale (x the smallest swept scale)")
    ax.set_ylabel("std of the injected values, relative to its own value there")
    ax.set_title("sigmoid saturation: every cell's curve FLATTENS\n"
                 "(read the bend, not the side of the diagonal -- a cell at p~1e-3 starts "
                 "on the flat\npart of the sigmoid and first grows faster than the scale. "
                 "Indirect either way:\nthe sampler's clamp counters are not written to "
                 "the CSV)", fontsize=10)
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "inj_std_saturation.png", dpi=150)
    plt.close(fig)


def fig_ood(df: pd.DataFrame, out: Path):
    """Endogenous corr per corpus, one line per arm, one panel per cell."""
    corpora = [c for c in OOD_CORPORA if f"ood_endo_{c}" in df.columns]
    if not corpora:
        print("[sum] no ood_generalization.csv anywhere; skipping the OOD ladder")
        return
    cells = _cells(df)
    arms = [a for a in ARM_ORDER if a in set(df["arm"])]
    ncol = 3
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.3 * nrow),
                             squeeze=False, sharey=True)
    cmap = plt.get_cmap("tab10")
    # in-distribution first, so the drop off the left edge is the thing being read
    xs = ["c4 (test)"] + list(corpora)
    for ax, cell in zip(axes.ravel(), cells):
        for i, arm in enumerate(arms):
            sub = df[(df["cell"] == cell) & (df["arm"] == arm)]
            if not len(sub):
                continue
            means = [sub["corr"].mean()] + [sub[f"ood_endo_{c}"].mean() for c in corpora]
            sems = [sub["corr"].sem()] + [sub[f"ood_endo_{c}"].sem() for c in corpora]
            ax.errorbar(range(len(xs)), means, yerr=sems, marker="o", ms=4, capsize=3,
                        color=cmap(i), label=arm)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(cell)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("endogenous corr")
    axes[0][0].legend(fontsize=8)
    for ax in axes.ravel()[len(cells):]:
        ax.axis("off")
    fig.suptitle("does the readout survive off-distribution text?  "
                 "(bars = +-SEM across the heads in the cell)")
    fig.tight_layout()
    fig.savefig(out / "ood_ladder.png", dpi=150)
    plt.close(fig)


def fig_pearson_spearman(df: pd.DataFrame, out: Path):
    """Pearson next to Spearman, per (cell, arm).

    A post-softmax weight is heavily right-skewed: an attention sink sits at 0.3-0.9 while
    mid-context cells sit at 1e-4-1e-1. Pearson and the training loss are both
    squared-scale, so a model that has only learned "is this the sink" scores nearly as
    well as one reading the whole range. The gap to the rank correlation is how much of
    the score the tail is carrying.
    """
    if "spearman" not in df.columns or df["spearman"].isna().all():
        print("[sum] no spearman column (older runs); skipping the skew figure")
        return
    arms = [a for a in ARM_ORDER if a in set(df["arm"])]
    g = df.groupby(["cell", "arm"], as_index=False)[["corr", "spearman"]].mean()
    cells = _cells(g)
    # width per CELL GROUP, not per bar: scaling by cells x arms x 1.6 gave a 29-inch
    # figure whose bars were 30 px tall in a 4000 px image
    fig, ax = plt.subplots(figsize=(max(8.0, 2.4 * len(cells)), 5.5))
    width = 0.8 / (2 * len(arms))
    cmap = plt.get_cmap("tab10")
    for i, arm in enumerate(arms):
        sub = g[g["arm"] == arm].set_index("cell").reindex(cells)
        base = np.arange(len(cells)) + i * 2 * width - 0.4
        ax.bar(base, sub["corr"], width, color=cmap(i), label=f"{arm} Pearson")
        ax.bar(base + width, sub["spearman"], width, color=cmap(i), alpha=0.5,
               hatch="//", label=f"{arm} Spearman")
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels(cells)
    ax.set_ylabel("endogenous corr (mean over heads)")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    # Both directions occur in these runs and mean opposite things, so say both.
    ax.set_title("Pearson vs Spearman: how much of the score is the right tail?\n"
                 "Pearson > Spearman: the score leans on a few large weights.  "
                 "Spearman > Pearson: the order\nis right but the large values are "
                 "mispredicted.  Equal: the readout is scale-faithful.", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=7, ncol=len(arms))
    fig.tight_layout()
    fig.savefig(out / "pearson_vs_spearman.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    configs_dir = (Path(args.configs_dir) if args.configs_dir
                   else Path("experiment_configs") / sweep_dir.name)
    out = Path(args.out) if args.out else Path("outputs/plots") / sweep_dir.name
    out.mkdir(parents=True, exist_ok=True)

    df = collect(sweep_dir, configs_dir, Path(args.data_dir))
    df.to_csv(out / "stage2_master.csv", index=False)
    print(f"[sum] {len(df)} runs -> {out/'stage2_master.csv'} ({len(df.columns)} columns)")

    cells = by_cell(df)
    cells.to_csv(out / "stage2_by_cell.csv", index=False)
    print(f"[sum] {len(cells)} (cell, arm) groups -> {out/'stage2_by_cell.csv'}")

    validity(df, out)

    n_picked = int(df["is_picked_head"].sum())
    print(f"\n[sum] {n_picked} runs are on a hand-picked head "
          f"({len(df) - n_picked} are the fill heads that give the error bar)")
    print(f"[sum] LM cost scored on {int(df.get('kl_at32', pd.Series(dtype=float)).notna().sum())}"
          f"/{len(df)} runs (by design -- see _lm_cost)")

    print("\n[sum] === endogenous corr, mean +- SE over heads ===")
    piv = df.pivot_table(index=["src_offset", "layer"], columns="arm", values="corr",
                         aggfunc="mean")
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))

    fig_scale_sweep(df, sweep_dir, out)
    fig_inj_std(df, sweep_dir, out)
    fig_ood(df, out)
    fig_pearson_spearman(df, out)
    print(f"\n[sum] wrote {len(list(out.glob('*.png')))} figures and "
          f"{len(list(out.glob('*.csv')))} tables under {out}")


if __name__ == "__main__":
    main()
