#!/usr/bin/env python3
"""
SFT + KL baseline (rebuttal): in-domain endogenous and interventional Pearson at the four
rebuttal locales, for every training arm.

    SFT              w=1.0, no KL        mlp_endo_sweep_layer_token
    SFT + KL         w=1.0, KL 0.1       mlp_sft_kl_grid_layer_token
    Joint (no KL)    w=0.01, unanchored  mlp_joint_grid_layer_token
    Joint + KL       w=0.01, KL 0.1      mlp_joint_grid_layer_token   (ours)

SFT + KL vs Joint + KL differ only in w, so the pair isolates the interventional loss at
equal behavioural cost. (Plain SFT also used a cosine schedule with warm-up; the other
three use linear, no warm-up.)

LOCALES  L2 and L8 x POS-1 / POS-28 (t=31 / t=4), 5 neurons each.
PAIRING  per locale, only neurons present in EVERY arm that has any run there, so a
         neuron missing from one arm cannot move another arm's mean. n is printed; an
         arm with no runs at a locale shows "--" and does not shrink the others.
CELLS    mean ± std over neurons; the bottom row pools the paired runs of the locales
         EVERY arm has (all 4 once the SFT + KL runs finish).
KL       achieved next-token KL on 32-token C4 windows (lm_degradation.json, length 32),
         mean over the runs that were scored (one or two neurons per cell, by arm).

Reads run directories (outputs/runs/<group>/<run>/), so run it where they live (SCC).

Outputs (outputs/plots/sft_vs_joint/sft_kl_baseline/)
  sft_kl_baseline_table.md   markdown tables for the rebuttal
  sft_kl_baseline_runs.csv   one row per (arm, run) used

Usage
  python plotters/table_sft_kl_baseline.py
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotters.compare_sft_vs_joint import load_group   # noqa: E402

ARMS = {   # label -> (group, load_group filters)
    "SFT": ("mlp_endo_sweep_layer_token", dict(gate="off", w=1.0)),
    "SFT + KL": ("mlp_sft_kl_grid_layer_token", dict(gate="off", w=1.0, kl="0.1")),
    "Joint (no KL)": ("mlp_joint_grid_layer_token", dict(gate="off", w=0.01, kl="none")),
    "Joint + KL (Ours)": ("mlp_joint_grid_layer_token", dict(gate="off", w=0.01, kl="0.1")),
}
LOCALES = [(2, -1), (2, -28), (8, -1), (8, -28)]
SEQ_LEN = 32
METRICS = {"endogenous_corr": "Endo.", "interventional_corr": "Interv."}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "outputs/plots/sft_vs_joint/sft_kl_baseline")
    return p.parse_args()


def locale_label(layer, pos):
    return f"L{layer}, t={pos + SEQ_LEN if pos < 0 else pos}"


def achieved_kl(group, run):
    f = ROOT / "outputs/runs" / group / run / "lm_degradation.json"
    if not f.exists():
        return np.nan
    for row in json.loads(f.read_text()).get("lengths", []):
        if row.get("length") == SEQ_LEN:
            return row.get("kl_mean", np.nan)
    return np.nan


def load_arms():
    frames = []
    for label, (group, filters) in ARMS.items():
        try:
            d = load_group(group, **filters)
        except SystemExit as e:
            print(f"[sftkl] {label}: no runs ({e})")
            continue
        d = d[[(l, p) in LOCALES for l, p in zip(d["layer"], d["position"])]].copy()
        d["arm"] = label
        d["kl_achieved"] = [achieved_kl(group, r) for r in d["run"]]
        print(f"[sftkl] {label}: {len(d)} runs at the rebuttal locales ({group})")
        frames.append(d)
    if not frames:
        raise SystemExit("no runs found for any arm")
    return pd.concat(frames, ignore_index=True)


def paired_runs(df):
    """Per locale, the neurons every arm present at that locale has."""
    keep = []
    for (layer, pos), g in df.groupby(["layer", "position"]):
        common = set.intersection(*(set(a["neuron"]) for _, a in g.groupby("arm")))
        dropped = sorted(set(g["neuron"]) - common)
        if dropped:
            print(f"[sftkl] {locale_label(layer, pos)}: dropping {dropped} (missing from an arm)")
        keep.append(g[g["neuron"].isin(common)])
    return pd.concat(keep, ignore_index=True)


def fmt(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "--"
    return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v.mean():.2f}"


def markdown(df):
    arms = [a for a in ARMS if a in set(df["arm"])]
    missing = [a for a in ARMS if a not in arms]
    head = "| Locale | n | " + " | ".join(f"{a} {m}" for a in arms for m in METRICS.values()) + " |"
    out = ["In-domain (C4 test, 3K rows) Pearson r, mean ± std over neurons; "
           "n = neurons per locale shared by every arm.", "", head,
           "|---|---|" + "---|" * (len(arms) * len(METRICS))]
    rows = [((l, p), df[(df["layer"] == l) & (df["position"] == p)]) for l, p in LOCALES]
    # pool only locales every arm has, so a partially finished arm is not averaged over
    # fewer (and different) locales than the arms beside it
    full = [(l, p) for (l, p), g in rows if set(g["arm"]) == set(arms)]
    pooled = df[[(l, p) in full for l, p in zip(df["layer"], df["position"])]]
    rows.append((f"All {len(full)} locales" if len(full) == len(LOCALES)
                 else f"Pooled ({len(full)} of {len(LOCALES)} locales with every arm)", pooled))
    for key, g in rows:
        lab = key if isinstance(key, str) else locale_label(*key)
        n = (g.groupby(["layer", "position"])["neuron"].nunique().sum() if len(g) else 0)
        cells = [fmt(g.loc[g["arm"] == a, m]) for a in arms for m in METRICS]
        out.append(f"| {lab} | {n} | " + " | ".join(cells) + " |")
    if missing:
        out += ["", f"No runs yet: {', '.join(missing)}."]
    kl = df.groupby("arm")["kl_achieved"].agg(["mean", "count"]).reindex(arms)
    out += ["", "Achieved KL (next-token, 32-token C4 windows), mean over the runs scored:", "",
            "| Arm | KL | runs scored |", "|---|---|---|"]
    for a, r in kl.iterrows():
        out.append(f"| {a} | {r['mean']:.3f} | {int(r['count'])} |" if r["count"]
                   else f"| {a} | -- | 0 |")
    return "\n".join(out) + "\n"


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = paired_runs(load_arms())
    cols = ["arm", "group", "run", "layer", "position", "neuron", *METRICS, "kl_achieved"]
    df[cols].sort_values(["layer", "position", "neuron", "arm"]).to_csv(
        args.out_dir / "sft_kl_baseline_runs.csv", index=False)
    md = markdown(df)
    (args.out_dir / "sft_kl_baseline_table.md").write_text(md)
    print("\n" + md)
    print(f"[sftkl] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
