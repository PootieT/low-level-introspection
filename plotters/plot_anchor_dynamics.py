#!/usr/bin/env python3
"""
KL-anchor control dynamics over training, from each run's trainer_state.json.

The end-of-run measured KL says whether a budget was met; this says HOW, and it
separates two very different failures:

  coef RAILED AT THE MAX and KL still above target -> the anchor could not hold the
      budget at any strength it was allowed. The run is anchor-dominated, and its low
      interventional score reflects the constraint, not the site's difficulty.
  coef AT THE FLOOR and KL at/below target -> the anchor stopped binding; the model
      simply did not want more drift. The frontier's loose end is then "unconstrained
      in practice", which changes how that point should be described.

Also plots L_endo / L_interv so the objective terms can be sanity-checked (at W=1 the
interventional term is absent by construction, at W=0 the endogenous one is).
"""
import os
import re
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
KL_RE = re.compile(r"_kl(?P<kl>[0-9.]+)")
NEURON_RE = re.compile(r"N(?P<layer>\d+)\.(?P<index>\d+)")
POS_RE = re.compile(r"POS(?P<pos>-?\d+)")


def load_histories(exp_group: str, pattern="*") -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/outputs/runs/{exp_group}/{pattern}/checkpoint-*/trainer_state.json")):
        run = Path(f).parents[1].name
        n, p, kl = NEURON_RE.search(run), POS_RE.search(run), KL_RE.search(run)
        st = json.loads(Path(f).read_text())
        for e in st.get("log_history", []):
            if "L_kl" not in e and "L_endo" not in e:
                continue
            rows.append({
                "run": run,
                "cell": (f"N{n['layer']}.{n['index']} POS{p['pos']}" if n and p else run),
                "kl_target": float(kl["kl"]) if kl else np.inf,
                "step": e.get("step"),
                **{k: e[k] for k in ("L_endo", "L_interv", "L_kl", "kl_coef", "loss") if k in e},
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No trainer_state.json with loss terms under outputs/runs/{exp_group}")
    return df


def plot_dynamics(df, out_dir, smooth=50):
    cells = sorted(df["cell"].unique())
    fig, axes = plt.subplots(2, len(cells), figsize=(6 * len(cells), 9), squeeze=False)
    budgets = sorted(b for b in df["kl_target"].unique() if np.isfinite(b))
    cmap = plt.get_cmap("viridis")
    colors = {b: cmap(i / max(1, len(budgets) - 1)) for i, b in enumerate(budgets)}
    for j, cell in enumerate(cells):
        sub_cell = df[df["cell"] == cell]
        for b in sorted(sub_cell["kl_target"].unique()):
            if not np.isfinite(b):
                continue
            s = sub_cell[sub_cell["kl_target"] == b].sort_values("step")
            kl = s["L_kl"].rolling(smooth, min_periods=1).mean()
            axes[0][j].plot(s["step"], kl, color=colors[b], label=f"{b:g}")
            axes[0][j].axhline(b, color=colors[b], ls=":", lw=1, alpha=0.7)
            if "kl_coef" in s:
                axes[1][j].plot(s["step"], s["kl_coef"], color=colors[b], label=f"{b:g}")
        axes[0][j].set_title(cell)
        axes[0][j].set_ylabel(f"measured KL (rolling {smooth})")
        axes[1][j].set_ylabel("anchor coefficient")
        for ax in (axes[0][j], axes[1][j]):
            ax.set_yscale("log"); ax.set_xlabel("step"); ax.grid(alpha=0.25, which="both")
        axes[0][j].legend(title="budget", fontsize=9, ncol=2)
    fig.suptitle("KL anchor: measured KL (dotted = its budget) and the adaptive coefficient", y=1.0)
    fig.tight_layout()
    path = Path(out_dir) / "anchor_dynamics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[anchor] wrote {path}")


def report(df):
    print("\n[anchor] end-of-training state (last 100 logged steps):")
    rows = []
    for (cell, b), s in df.groupby(["cell", "kl_target"]):
        if not np.isfinite(b):
            continue
        tail = s.sort_values("step").tail(100)
        coef = tail["kl_coef"] if "kl_coef" in tail else pd.Series(dtype=float)
        row = {"cell": cell, "budget": b,
               "kl_final": tail["L_kl"].mean() if "L_kl" in tail else np.nan,
               "coef_final": coef.mean() if len(coef) else np.nan,
               "coef_min": s["kl_coef"].min() if "kl_coef" in s else np.nan,
               "coef_max": s["kl_coef"].max() if "kl_coef" in s else np.nan}
        row["ratio_kl_to_budget"] = row["kl_final"] / b if b else np.nan
        # which regime is this run in?
        row["regime"] = ("RAILED (anchor maxed, budget missed)"
                         if row["ratio_kl_to_budget"] > 1.5 and
                         np.isclose(row["coef_final"], row["coef_max"], rtol=0.05)
                         else "floor (anchor not binding)"
                         if np.isclose(row["coef_final"], row["coef_min"], rtol=0.05)
                         else "tracking")
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["cell", "budget"])
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return out


if __name__ == "__main__":
    EXP_GROUP = "mlp_joint_prev_token_hyper"
    OUT_DIR = f"{ROOT}/outputs/plots/{EXP_GROUP}"
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_histories(EXP_GROUP, pattern="*gateoff*kl*")
    summary = report(df)
    summary.to_csv(f"{OUT_DIR}/anchor_dynamics_summary.csv", index=False)
    plot_dynamics(df, OUT_DIR)
