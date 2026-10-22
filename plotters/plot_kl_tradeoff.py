#!/usr/bin/env python3
"""
The cost of introspection: what does introspective readout cost the language model?

Reads a KL-anchor sweep (runs named ..._kl<target>_...) and plots introspection
performance against measured LM degradation. Sources, all written by the training and
scoring pipeline:

  interventional_test_metrics_gaussian.json   causal reading (the headline metric)
  endogenous_test_metrics_endogenous.json     endogenous readout
  lm_degradation.json                         KL(base||tuned), top-1 agreement, ppl,
                                              per eval length (evals/score_lm_degradation.py)
  activation_drift.json                       neuron drift + readout coupling

WHY THE AXES ARE THIS WAY ROUND. Putting "performance and degradation" both on y
against the KL *target* hides the interesting failure: the controller does not always
reach its budget. Measured KL is the degradation, so it belongs on x -- then the plot
IS the frontier, and each point's distance from the y-axis is what that run actually
cost. The target survives as the point label, plus its own controller-tracking panel.

Panels
  1 frontier      measured KL (log x) vs interventional & endogenous corr
  2 controller    target vs achieved KL -- did the anchor hold its budget?
  3 capability    measured KL vs top-1 agreement and perplexity ratio
  4 length leak   achieved KL per eval length -- the anchor is applied at the TRAINING
                  length, so this shows whether longer contexts drift more
"""
import os
import re
import sys
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "legend.title_fontsize": 13, "figure.titlesize": 18,
})

ROOT = Path(__file__).parents[1]
KL_RE = re.compile(r"_kl(?P<kl>[0-9.]+)")
POS_RE = re.compile(r"POS(?P<pos>-?\d+)")
GATE_RE = re.compile(r"gate(?P<gate>on|off)")
W_RE = re.compile(r"_w(?P<w>[0-9.]+)")
# Every signal family, not just MLP neurons: N = MLP neuron, A = attention logit,
# P = attention weight, S = SAE feature. The attention families add `@<src offset>`, and
# that offset is part of the cell identity -- two offsets in the same head are two
# different signals. Widened rather than left MLP-only because this regex is OPTIONAL
# below (the row keeps `signal=None` on a miss) rather than a filter, so an attention
# sweep did not get skipped here, it got silently merged: every run collapsed to the one
# cell label "None POS-1" and the frontier averaged 6 cells x 3 arms into a single curve.
SIGNAL_RE = re.compile(r"(?P<code>[NAPS])(?P<layer>\d+)\.(?P<index>\d+)"
                       r"(?:@(?P<src>-?\d+))?")
TRAIN_LEN = 32          # the length the anchor is applied at during training
SEQ_LEN = 32            # training window; titles show 0-indexed absolute positions (-1 -> 31)


def _cell_title(cell):
    """'N10.24 POS-1' -> 'N10.24 t=31': token position as an absolute 0-indexed index,
    matching the layer x position heatmaps."""
    return POS_RE.sub(lambda m: f"t={int(m['pos']) + SEQ_LEN if int(m['pos']) < 0 else m['pos']}",
                      cell).replace(" t=", ", t=")


def load_kl_sweep(exp_group: str, gate="off", position=None, exclude=("lrcos",)) -> pd.DataFrame:
    """One row per run. Runs without a _kl tag are the UNANCHORED control and are kept
    with kl_target = inf, since they anchor the right-hand end of the frontier."""
    rows, skipped = [], []
    group_dir = Path(f"{ROOT}/outputs/runs/{exp_group}")
    if not group_dir.is_dir():
        raise SystemExit(f"No run group at {group_dir}. Pass --exp-group with one of: "
                         + ", ".join(sorted(p.name for p in (ROOT / "outputs/runs").iterdir()
                                            if p.is_dir())))
    for d in sorted(group_dir.iterdir()):
        if not d.is_dir() or any(x in d.name for x in exclude):
            continue
        g, p = GATE_RE.search(d.name), POS_RE.search(d.name)
        if gate and (not g or g["gate"] != gate):
            continue
        if position is not None and (not p or int(p["pos"]) != position):
            continue

        def load(fname, key=None):
            path = d / fname
            if not path.exists():
                return None
            payload = json.loads(path.read_text())
            return payload[key] if key else payload

        interv = load("interventional_test_metrics_gaussian.json", "test")
        endo = load("endogenous_test_metrics_endogenous.json")
        lm = load("lm_degradation.json")
        drift = load("activation_drift.json")
        if interv is None or endo is None:
            skipped.append((d.name, "missing introspection metrics"))
            continue

        m_kl, m_n, m_w = KL_RE.search(d.name), SIGNAL_RE.search(d.name), W_RE.search(d.name)
        signal = None
        if m_n:
            signal = f"{m_n['code']}{m_n['layer']}.{m_n['index']}"
            if m_n["src"] is not None:
                signal += f"@{m_n['src']}"
        row = {
            "run": d.name,
            "kl_target": float(m_kl["kl"]) if m_kl else np.inf,   # inf == unanchored
            "anchored": bool(m_kl),
            "position": int(p["pos"]) if p else None,
            "gate": g["gate"] if g else None,
            "signal": signal,
            "neuron": signal,          # back-compat: the old column name, same value
            "w": float(m_w["w"]) if m_w else None,
            "interventional_corr": interv.get("corr"),
            "endogenous_corr": endo.get("corr"),
        }
        if drift:
            row["drift_corr"] = drift.get("drift_tuned_vs_base", {}).get("corr")
            row["coupling_gap"] = (drift.get("readout_tracking") or {}).get("coupling_gap")
        row.update(_load_benchmarks(d))
        if lm:
            for entry in lm.get("lengths", []):
                L = entry["length"]
                row[f"kl@{L}"] = entry.get("kl_mean")
                row[f"top1@{L}"] = entry.get("top1_agreement")
                row[f"ppl_tuned@{L}"] = entry.get("ppl_tuned")
                row[f"ppl_base@{L}"] = entry.get("ppl_base")
                if entry.get("ppl_base"):
                    row[f"ppl_ratio@{L}"] = entry["ppl_tuned"] / entry["ppl_base"]
        else:
            # Not necessarily an error. The attention-weight Stage 2 script scores the
            # unanchored arm on two heads (its measured KL is the result, so it needs an
            # error bar) and the anchored arm on one, so most gate-off runs legitimately
            # have no lm_degradation.json.
            skipped.append((d.name, "no lm_degradation.json -- run evals/score_lm_degradation.py"))
        rows.append(row)

    if skipped:
        print(f"[kl] {len(skipped)} run(s) incomplete: {skipped[:6]}")
    df = pd.DataFrame(rows).sort_values("kl_target").reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"No runs matched under outputs/runs/{exp_group} (gate={gate}).")
    if df["signal"].isna().all():
        raise SystemExit(
            f"No run name under outputs/runs/{exp_group} carries a signal tag "
            f"(N/A/P/S<layer>.<index>). Every row would collapse into one cell and the "
            f"frontier would average unrelated signals into a single curve.")
    df["cell"] = df["signal"].astype(str) + " POS" + df["position"].astype(str)
    if "kl@32" in df:
        no_kl = sorted(df[df["kl@32"].isna()]["cell"].unique())
        if no_kl:
            print(f"[kl] cells with NO measured KL (evals/score_lm_degradation.py not run): {no_kl}\n"
                  f"[kl]   -> those cells can only be plotted against the BUDGET, which is the "
                  f"knob, not the cost. Run the scorer on them before quoting a frontier.")
    return df


def _load_benchmarks(run_dir: Path) -> dict:
    """lighteval results for one run: aggregate deltas from benchmarks_comparison.csv,
    plus the per-example flip rate (aggregate deltas can be ~0 while a sizeable
    fraction of individual questions changed answer)."""
    out = {}
    comp = run_dir / "lighteval" / "benchmarks_comparison.csv"
    if comp.exists():
        b = pd.read_csv(comp)
        for _, r in b.iterrows():
            task = str(r["task"]).replace(":", "").replace("|", "")
            key = f"bench_{task}_{r['metric']}"
            out[f"{key}_adapter"] = r.get("adapter")
            out[f"{key}_base"] = r.get("base")
            out[f"{key}_delta"] = r.get("delta")
        agg = b[(b["task"] == "all") & (b["metric"] == "acc")]
        if len(agg):
            out["bench_acc_base"] = float(agg["base"].iloc[0])
            out["bench_acc_adapter"] = float(agg["adapter"].iloc[0])
            out["bench_acc_delta"] = float(agg["delta"].iloc[0])
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from compare_lighteval_details import summarize_run
        per_ex = summarize_run(run_dir)
        if per_ex.get("example_changed_frac") is not None:
            out["bench_example_changed_frac"] = per_ex["example_changed_frac"]
    except Exception:
        pass          # pyarrow missing or details absent: aggregate deltas still fine
    return out


def plot_benchmarks(df, out_dir, length=TRAIN_LEN):
    """Downstream capability vs measured LM degradation.

    Two views, because they disagree: the aggregate accuracy delta is what a paper
    table reports, while the per-example flip rate is what actually changed. A run can
    show a 0.005 accuracy drop while answering 4.5% of questions differently."""
    x = f"kl@{length}"
    have_agg = "bench_acc_delta" in df and df["bench_acc_delta"].notna().any()
    have_ex = "bench_example_changed_frac" in df and df["bench_example_changed_frac"].notna().any()
    if not (have_agg or have_ex):
        print("[kl] no lighteval results found; skipping benchmark panel")
        return
    panels = [("bench_acc_delta", "Benchmark Accuracy", "Accuracy \u0394 (Adapter \u2212 Base)", True),
              ("bench_example_changed_frac", "Per-Question Agreement",
               "Fraction of Answers Changed", False)]
    panels = [p for p in panels if p[0] in df and df[p[0]].notna().any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 6), squeeze=False)
    cells = sorted(df.dropna(subset=[x])["cell"].unique())
    colors = dict(zip(cells, plt.get_cmap("tab10").colors))      # same colour in both panels
    for ax, (col, title, ylabel, zero_line) in zip(axes.flat, panels):
        for cell, sub in df.dropna(subset=[x, col]).groupby("cell"):
            sub = sub.sort_values(x)
            ax.plot(sub[x], sub[col], marker="o", ms=9, lw=1.8, color=colors[cell],
                    label=_cell_title(cell))
        if zero_line:
            ax.axhline(0, color="grey", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=28)
        ax.set_xlabel(f"Measured KL @ L={length} (nats/token)", fontsize=24)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.tick_params(which="both", labelsize=20)
        ax.grid(alpha=0.25, which="both")
    # one legend, lower-left of the first panel: empty there (low KL = ~0 accuracy change)
    axes.flat[0].legend(fontsize=16, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    path = Path(out_dir) / "kl_benchmarks.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[kl] wrote {path}")


def plot_joint_pareto(df, out_dir):
    """The joint frontier: endogenous vs interventional, one point per KL budget.

    Top-right = a faithful reader (both readouts high). A run that is high on
    endogenous but low on interventional is predicting from context, not reading --
    which is exactly what the tight-budget runs do."""
    cells = sorted(df["cell"].unique())
    ncol = min(2, len(cells))
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow), squeeze=False)
    finite = df[np.isfinite(df["kl_target"])]
    norm = plt.Normalize(np.log10(finite["kl_target"].min()), np.log10(finite["kl_target"].max()))
    for ax, cell in zip(axes.flat, cells):
        sub = df[df["cell"] == cell].sort_values("kl_target")
        anch = sub[sub["anchored"]]
        sc = ax.scatter(anch["interventional_corr"], anch["endogenous_corr"],
                        c=np.log10(anch["kl_target"]), cmap="viridis", norm=norm, s=90, zorder=3)
        ax.plot(anch["interventional_corr"], anch["endogenous_corr"], "-", alpha=0.4, color="grey")
        for _, r in anch.iterrows():
            ax.annotate(_target_label(r["kl_target"]),
                        (r["interventional_corr"], r["endogenous_corr"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=9)
        un = sub[~sub["anchored"]]
        if len(un):
            ax.scatter(un["interventional_corr"], un["endogenous_corr"], marker="X",
                       s=200, color="crimson", zorder=4, label="unanchored")
            ax.legend(fontsize=10)
        ax.set_title(cell)
        ax.set_xlabel("interventional corr (causal reading)")
        ax.set_ylabel("endogenous corr")
        ax.set_xlim(-0.1, 1.05); ax.set_ylim(-0.1, 1.05)
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(cells):]:
        ax.set_visible(False)
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="log10 KL budget", shrink=0.8)
    fig.suptitle("joint frontier: endogenous vs interventional (labels = KL budget)", y=1.01)
    path = Path(out_dir) / "kl_joint_pareto.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[kl] wrote {path}")


def _label_points(ax, x, y, labels):
    for xi, yi, li in zip(x, y, labels):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(li, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=10)


def _target_label(t):
    return "none" if not np.isfinite(t) else f"{t:g}"


def _facet(df, x, out_path, xlabel, logx=True):
    """One panel per (neuron, position) cell; interventional and endogenous on y."""
    cells = sorted(df["cell"].unique())
    ncol = min(2, len(cells))
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4.8 * nrow), squeeze=False)
    for ax, cell in zip(axes.flat, cells):
        sub = df[df["cell"] == cell].dropna(subset=[x]).sort_values(x)
        if sub.empty:
            ax.set_visible(False)
            continue
        for col, label, marker in (("interventional_corr", "Interventional", "o"),
                                   ("endogenous_corr", "Endogenous", "s")):
            ax.plot(sub[x], sub[col], marker=marker, label=label)
        unanch = sub[~sub["anchored"]]
        if len(unanch) and np.isfinite(unanch[x]).any():
            ax.scatter(unanch[x], unanch["interventional_corr"], marker="X", s=200,
                       color="crimson", zorder=5, label="Unanchored")
        if logx:
            ax.set_xscale("log")
        ax.set_title(_cell_title(cell))
        ax.set_xlabel(xlabel); ax.set_ylabel("Correlation")
        ax.set_ylim(-0.1, 1.05); ax.axhline(0, color="grey", ls="--", lw=1)
        ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=10)
    for ax in axes.flat[len(cells):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[kl] wrote {out_path}")


def plot_frontier(df, out_dir, length=TRAIN_LEN):
    """The money plot: what you get vs what it actually cost (measured KL)."""
    x = f"kl@{length}"
    if x not in df or df[x].notna().sum() == 0:
        print(f"[kl] no measured {x} anywhere; skipping the measured-cost frontier")
        return
    _facet(df[df[x].notna()], x, Path(out_dir) / f"kl_frontier_L{length}.png",
           f"Measured KL(Base || Tuned) @ L={length} (nats/token)")


def plot_frontier_by_budget(df, out_dir):
    """Same curves against the BUDGET. Always available (no scorer required), but the
    budget is the knob, not the cost -- the controller does not always reach it, so
    read this for reachability across cells and the measured version for cost."""
    sub = df[df["anchored"]]
    if sub.empty:
        return
    _facet(sub, "kl_target", Path(out_dir) / "kl_frontier_by_budget.png",
           "KL Budget (Target, nats/token)")


def plot_controller(df, out_dir, length=TRAIN_LEN):
    """Did the adaptive coefficient actually hold its budget? Points below y=x mean
    the model never wanted that much drift and the anchor stopped binding."""
    kl_col = f"kl@{length}"
    sub = df[df["anchored"]].dropna(subset=[kl_col])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    lims = [min(sub["kl_target"].min(), sub[kl_col].min()) * 0.6,
            max(sub["kl_target"].max(), sub[kl_col].max()) * 1.6]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.6, label="achieved = target")
    ax.plot(sub["kl_target"], sub[kl_col], "o-", label=f"achieved @L={length}")
    other = [c for c in df.columns if c.startswith("kl@") and c != kl_col]
    for col in sorted(other):
        ax.plot(sub["kl_target"], sub[col], "s--", alpha=0.7, label=f"achieved @L={col.split('@')[1]}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("KL budget (target, nats/token)")
    ax.set_ylabel("measured KL (nats/token)")
    ax.set_title("anchor controller: budget vs achieved")
    ax.grid(alpha=0.25, which="both"); ax.legend()
    fig.tight_layout()
    path = Path(out_dir) / "kl_controller_tracking.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[kl] wrote {path}")


def plot_capability(df, out_dir, length=TRAIN_LEN):
    """Degradation in units people read directly: agreement with the base model's
    next token, and the perplexity multiple."""
    kl_col, top1, ppl = f"kl@{length}", f"top1@{length}", f"ppl_ratio@{length}"
    if top1 not in df:
        return
    sub = df.dropna(subset=[kl_col, top1]).sort_values(kl_col)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(sub[kl_col], sub[top1], "o-")
    axes[0].set_ylabel("top-1 agreement with base"); axes[0].set_ylim(0, 1.02)
    axes[1].plot(sub[kl_col], sub[ppl], "o-", color="darkorange")
    axes[1].axhline(1.0, color="grey", ls="--", lw=1)
    axes[1].set_ylabel("perplexity ratio (tuned / base)"); axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(f"measured KL @L={length} (nats/token)")
        ax.grid(alpha=0.25, which="both")
        _label_points(ax, sub[kl_col], sub[ax.get_ylabel() and top1 if ax is axes[0] else ppl],
                      [_target_label(t) for t in sub["kl_target"]])
    fig.suptitle("what the degradation looks like in behavioural units")
    fig.tight_layout()
    path = Path(out_dir) / f"kl_capability_L{length}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[kl] wrote {path}")


def plot_length_leak(df, out_dir):
    """The anchor is applied at the TRAINING length only. If longer contexts drift
    more, protection does not generalize past what training saw."""
    kl_cols = sorted((c for c in df.columns if c.startswith("kl@")),
                     key=lambda c: int(c.split("@")[1]))
    if len(kl_cols) < 2:
        return
    sub = df[df["anchored"]].dropna(subset=kl_cols)
    if sub.empty:
        return
    long = sub.melt(id_vars=["kl_target"], value_vars=kl_cols,
                    var_name="eval_length", value_name="kl")
    long["eval_length"] = long["eval_length"].str.split("@").str[1].astype(int)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.lineplot(data=long, x="kl_target", y="kl", hue="eval_length", marker="o",
                 palette="viridis", ax=ax)
    ax.plot(sorted(sub["kl_target"]), sorted(sub["kl_target"]), "k--", lw=1, alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("KL budget (target)"); ax.set_ylabel("measured KL (nats/token)")
    ax.set_title(f"anchor trained at L={TRAIN_LEN}: does it hold at longer contexts?")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    path = Path(out_dir) / "kl_length_leak.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[kl] wrote {path}")


def report(df, length=TRAIN_LEN):
    """Terminal summary: the knee of the frontier is the number to quote."""
    kl_col, top1 = f"kl@{length}", f"top1@{length}"
    cols = [c for c in ["kl_target", "interventional_corr", "endogenous_corr", kl_col,
                        f"kl@512", top1, f"ppl_ratio@{length}", "drift_corr", "coupling_gap"]
            if c in df.columns]
    print("\n[kl] sweep summary:")
    for cell, sub in df.groupby("cell"):
        print(f"\n--- {cell} ---")
        print(sub[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    # per cell: the cheapest budget that keeps most of the unanchored reading
    for cell, sub in df.groupby("cell"):
        anchored = sub[sub["anchored"]].dropna(subset=["interventional_corr"])
        unanchored = sub[~sub["anchored"]].dropna(subset=["interventional_corr"])
        if anchored.empty or unanchored.empty:
            continue
        best_unanchored = unanchored["interventional_corr"].max()
        within = anchored[anchored["interventional_corr"] >= 0.95 * best_unanchored]
        # rank by measured KL when it exists, else by budget -- saying so either way
        rank_col, basis = (kl_col, "measured KL") if (kl_col in within and
                                                      within[kl_col].notna().any()) else \
                          ("kl_target", "BUDGET (no measured KL; run evals/score_lm_degradation.py)")
        within = within.dropna(subset=[rank_col])
        if within.empty:
            print(f"\n[kl] {cell}: no budget reached 95% of the unanchored "
                  f"interventional corr ({best_unanchored:.3f}) -- this cell needs a "
                  f"looser budget than any tested.")
            continue
        knee = within.loc[within[rank_col].idxmin()]
        print(f"\n[kl] {cell}: cheapest run within 5% of unanchored "
              f"({best_unanchored:.3f}) by {basis}: budget "
              f"{_target_label(knee['kl_target'])}, interventional "
              f"{knee['interventional_corr']:.3f}"
              + (f", measured KL {knee[kl_col]:.4f}" if kl_col in knee and
                 pd.notna(knee.get(kl_col)) else ""))
    if False:
        if len(within):
            knee = within.loc[within[kl_col].idxmin()]
            print(f"\n[kl] cheapest run within 5% of the unanchored interventional corr "
                  f"({best_unanchored:.3f}): budget {_target_label(knee['kl_target'])}, "
                  f"interventional {knee['interventional_corr']:.3f}, "
                  f"measured KL {knee[kl_col]:.4f} nats/token"
                  + (f", top-1 {knee[top1]:.3f}" if top1 in knee else ""))


def parse_args():
    """Defaults reproduce the pre-CLI behaviour exactly, so the neuron figures keep
    regenerating from a bare `python plotters/plot_kl_tradeoff.py`."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-group", default="mlp_joint_prev_token_hyper",
                    help="run group under outputs/runs/")
    # the anchor only applies where the adapter touches the text, so gate on has nothing
    # to measure: KL there is 0 by construction, not by optimization
    ap.add_argument("--gate", default="off", choices=("on", "off"))
    ap.add_argument("--position", type=int, default=None,
                    help="restrict to one destination position (default: all cells)")
    ap.add_argument("--out-dir", default=None,
                    help="default outputs/plots/<exp-group>")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    EXP_GROUP = args.exp_group
    GATE = args.gate
    POSITION = args.position
    OUT_DIR = args.out_dir or f"{ROOT}/outputs/plots/{EXP_GROUP}"
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_kl_sweep(EXP_GROUP, gate=GATE, position=POSITION)
    df.to_csv(f"{OUT_DIR}/kl_sweep.csv", index=False)
    report(df)
    plot_frontier_by_budget(df, OUT_DIR)     # every cell
    plot_frontier(df, OUT_DIR)               # cells with measured KL only
    plot_joint_pareto(df, OUT_DIR)
    plot_benchmarks(df, OUT_DIR)
    measured = df[df.get(f"kl@{TRAIN_LEN}").notna()] if f"kl@{TRAIN_LEN}" in df else df.iloc[0:0]
    for cell in sorted(measured["cell"].unique()):
        sub = measured[measured["cell"] == cell]
        tag = cell.replace(" ", "_").replace(".", "-")
        cell_dir = Path(OUT_DIR) / tag
        cell_dir.mkdir(parents=True, exist_ok=True)
        plot_controller(sub, cell_dir)
        plot_capability(sub, cell_dir)
        plot_length_leak(sub, cell_dir)
