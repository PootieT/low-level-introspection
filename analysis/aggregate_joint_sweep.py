#!/usr/bin/env python3
"""
Aggregate a joint multi-position sweep into a summary table + overlay plots.

For each run dir (parsed for POS / scale / r / lr / w from its name) it pulls the
diagnostics written by train_introspector_joint.py:
  - interventional_test_metrics_gaussian.json : the honest interv@gaussian(scale=1) corr
  - endogenous_test_metrics_endogenous.json   : endogenous corr (NB: bypass-prone)
  - interventional_scale_sweep.csv            : head/ceiling corr vs injection scale
  - routing_probe_by_position.csv             : head/ceiling corr per token-position bin

Emits:
  - summary.csv (one row per run, all params + key metrics)
  - scale_sweep_overlay.png  (head corr vs injection scale, one line per run)
  - by_position_overlay.png  (head corr vs token position, one line per run)

The metric that matters is interv@gaussian(scale=1) and the scale-sweep curve's
value AT scale=1; endogenous is reported but is bypass-prone, don't rank on it.
head_corr is used as primary since ceiling ~= head (the head is fit; the limit is
routing gain upstream of the readout).

Example
-------
python analysis/aggregate_joint_sweep.py \
    --runs-glob 'outputs/runs/intervention_train_regressor_joint_prev_token_hyper/*scale30*'
"""
import re
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POS_RE = re.compile(r"POS(.+?)_scale")
SCALE_RE = re.compile(r"scale([0-9]+)")
R_RE = re.compile(r"_r([0-9]+)")
LR_RE = re.compile(r"_lr([^_]+)")
W_RE = re.compile(r"_w([^_]+)")
STEP_RE = re.compile(r"checkpoint-([0-9]+)")   # per-checkpoint diag subdirs (convergence mode)


def parse_run(name):
    def grab(rx):
        m = rx.search(name)
        return m.group(1) if m else None
    sm = STEP_RE.search(name)
    return {"pos": grab(POS_RE), "scale": grab(SCALE_RE), "r": grab(R_RE),
            "lr": grab(LR_RE) or "default", "w": grab(W_RE),
            "step": int(sm.group(1)) if sm else None}


def _json_corr(path, nested_key=None):
    if not path.exists():
        return None
    d = json.load(open(path))
    if nested_key and nested_key in d:
        d = d[nested_key]
    return d.get("corr")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-glob", required=True, nargs="+",
                   help="One or more globs matching run dirs (pass multiple, space-separated and each "
                        "quoted separately: --runs-glob '.../*_1M' '.../POS-28_scale*').")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    p.add_argument("--out-name", default="joint_sweep")
    p.add_argument("--label-fields", default="scale,r,lr",
                   help="Comma-sep params to build each run's legend label from.")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_fields = args.label_fields.split(",")

    rows, sweeps, positions = [], {}, {}
    run_dirs = sorted({d for pat in args.runs_glob for d in glob.glob(pat)})
    for rd in run_dirs:
        rdp = Path(rd)
        if not rdp.is_dir():
            continue
        params = parse_run(rdp.name)
        label = (f"step{params['step']}" if params.get("step") is not None
                 else "_".join(f"{k}{params[k]}" for k in label_fields if params.get(k)))

        row = dict(params)
        row["run"] = rdp.name
        row["label"] = label
        row["interv_gaussian"] = _json_corr(rdp / "interventional_test_metrics_gaussian.json", "test")
        row["endogenous"] = _json_corr(rdp / "endogenous_test_metrics_endogenous.json")

        # scale sweep
        ss_path = rdp / "interventional_scale_sweep.csv"
        if ss_path.exists():
            ss = pd.read_csv(ss_path)
            sweeps[label] = ss
            s1 = ss[np.isclose(ss["scale"], 1.0)]
            row["head@1"] = float(s1["head_corr"].iloc[0]) if len(s1) else None
            row["ceil@1"] = float(s1["ceiling_corr"].iloc[0]) if len(s1) else None
            row["head@maxscale"] = float(ss.sort_values("scale")["head_corr"].iloc[-1])

        # routing by position (bins only)
        rp_path = rdp / "routing_probe_by_position.csv"
        if rp_path.exists():
            rp = pd.read_csv(rp_path)
            binned = rp[rp["position_bin"] != "all"].copy()
            if len(binned):
                binned["pos_center"] = (binned["pos_lo"] + binned["pos_hi"]) / 2
                positions[label] = binned
                row["pos_head_mean"] = float(binned["head_corr"].mean())
                row["pos_head_min"] = float(binned["head_corr"].min())
                row["pos_ceil_mean"] = float(binned["ceiling_corr"].mean())
        rows.append(row)

    if not rows:
        raise SystemExit(f"No run dirs matched {args.runs_glob!r}")
    df = pd.DataFrame(rows)
    checkpoint_mode = df["step"].notna().any()   # dirs are per-checkpoint diag subdirs
    df = df.sort_values(["step"] if checkpoint_mode else ["scale", "r", "lr"], na_position="last")
    cols = (["step"] if checkpoint_mode else ["label", "pos", "scale", "r", "lr"]) + \
           ["interv_gaussian", "endogenous", "head@1", "ceil@1", "head@maxscale",
            "pos_head_mean", "pos_head_min"]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(args.output_dir / f"{args.out_name}_summary.csv", index=False)
    print(df[cols].to_string(index=False))

    # ---- scale-sweep overlay (head corr vs injection scale) ----
    if sweeps:
        plt.figure(figsize=(8, 5.5))
        for label, ss in sweeps.items():
            ss = ss.sort_values("scale")
            plt.plot(ss["scale"], ss["head_corr"], "o-", label=label, markersize=4)
        plt.xscale("log")
        plt.axhline(0, color="k", lw=0.5, alpha=0.3)
        plt.xlabel("injection scale (x natural std)"); plt.ylabel("interventional head corr")
        plt.title("Interventional corr vs injection scale (per run)")
        plt.ylim(-0.05, 1.02); plt.grid(alpha=0.2, which="both")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(args.output_dir / f"{args.out_name}_scale_sweep_overlay.png", dpi=150)
        plt.close()

    # ---- by-position overlay (head corr vs token position) ----
    if positions:
        plt.figure(figsize=(8, 5.5))
        for label, b in positions.items():
            b = b.sort_values("pos_center")
            plt.plot(b["pos_center"], b["head_corr"], "o-", label=label, markersize=4)
        plt.axhline(0, color="k", lw=0.5, alpha=0.3)
        plt.xlabel("introspected token position"); plt.ylabel("interventional head corr")
        plt.title("Interventional corr vs token position (per run)")
        plt.ylim(-0.05, 1.02); plt.grid(alpha=0.2)
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(args.output_dir / f"{args.out_name}_by_position_overlay.png", dpi=150)
        plt.close()

    # ---- convergence (metric vs training step), when dirs are per-checkpoint diags ----
    if checkpoint_mode:
        cd = df.dropna(subset=["step"]).sort_values("step")
        plt.figure(figsize=(8, 5.5))
        for metric in ["head@maxscale", "pos_head_mean", "ceil@1", "interv_gaussian", "endogenous"]:
            if metric in cd.columns and cd[metric].notna().any():
                plt.plot(cd["step"], cd[metric], "o-", label=metric, markersize=4)
        plt.axhline(0, color="k", lw=0.5, alpha=0.3)
        plt.xlabel("training step (checkpoint)"); plt.ylabel("corr")
        plt.title("Metrics vs training step (convergence)")
        plt.ylim(-0.05, 1.02); plt.grid(alpha=0.2); plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(args.output_dir / f"{args.out_name}_convergence.png", dpi=150)
        plt.close()

    print(f"[aggregate] wrote summary + plots for {len(df)} dirs to {args.output_dir} "
          f"({'convergence' if checkpoint_mode else 'sweep'} mode)")


if __name__ == "__main__":
    main()
