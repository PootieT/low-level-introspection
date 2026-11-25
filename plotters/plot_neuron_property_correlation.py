#!/usr/bin/env python3
"""
Do easy-to-state properties of a neuron predict how well it can be introspected?

Rebuttal check for "heads and SAE features were selected by variance / firing frequency,
which may favour signals that are easy to intervene on or report". The neuron targets
were drawn at random, so across them we can measure how much such properties actually
move performance -- for SFT, for the joint loss, and for the joint-minus-SFT gain (the
paper's comparative claim, which a selection effect shared by both arms would not bias
unless the gain itself depends on the property).

OUTCOMES  Pearson corr on the 3K C4 test split, endogenous and interventional (gaussian
          injection at the neuron's own std), gate off:
            SFT     outputs/plots/mlp_endo_sweep_layer_token/rank_metrics.csv  (w=1.0)
            Joint   outputs/plots/mlp_joint_grid_layer_token/rank_metrics.csv  (KL=0.1)
            Gain    Joint - SFT, paired on (layer, neuron, position)
FEATURES  base model, from the existing per-neuron scorers (via
          plotters/fit_sft_feature_regression.load_features):
            log std, firing rate (a > 0), log kurtosis   score_neuron_activation_stats.py
            token-identity R^2                            score_neuron_context_dependence.py
            redundancy R^2 (from same-layer neurons)      score_neuron_redundancy.py
          plus layer itself as a reference row (pooled only).

UNIT      the neuron (5 per layer x 8 layers = 40), each outcome averaged over its 8
          token positions: the same neuron at 8 positions is not 8 independent points.
SCOPES    pooled     Pearson across all 40 neurons
          within     layer means removed from feature and outcome (layer fixed effects),
                     dof = n - 2 - (n_layers - 1). This is the scope that matches the
                     selection being questioned, which ranked heads / features WITHIN a
                     layer (and offset).
CIs       Fisher z, 95%.

Outputs (outputs/plots/sft_vs_joint/neuron_property_correlation/)
  neuron_property_correlation.csv        long table: feature x model x eval x scope
  neuron_property_correlation_data.csv   one row per neuron
  neuron_property_correlation.md         rebuttal-ready markdown tables
  neuron_property_correlation.png        r with 95% CI, pooled and within-layer

Usage
  python plotters/plot_neuron_property_correlation.py
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, t as tdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plotters.fit_sft_feature_regression import load_features   # noqa: E402

PLOTS = ROOT / "outputs/plots"
SOURCES = {   # model -> (rank_metrics table, run-name filter)
    "SFT": (PLOTS / "mlp_endo_sweep_layer_token/rank_metrics.csv", "_gateoff_w1.0_"),
    "Joint": (PLOTS / "mlp_joint_grid_layer_token/rank_metrics.csv", "_gateoff_scale10_kl0.1_"),
}
EVALS = {"endogenous": "Endogenous", "gaussian": "Interventional"}
MODELS = ["SFT", "Joint", "Gain"]
MODEL_LABELS = {"SFT": "SFT", "Joint": "Joint Loss (Ours)", "Gain": "Gain (Joint − SFT)"}
FEATURES = {   # name -> (label, transform of the feature frame)
    "log_std":        ("Activation Std (log)", lambda d: np.log(d["std"])),
    "frac_positive":  ("Firing Rate (a > 0)", lambda d: d["frac_positive"]),
    "log_kurtosis":   ("Kurtosis (log)", lambda d: np.log(d["kurtosis"] + 3)),
    "r2_token_table": ("Token-Identity R²", lambda d: d["r2_token_table"]),
    "redundancy_r2":  ("Redundancy R²", lambda d: d["R2"]),
    "layer":          ("Layer (Reference)", lambda d: d["layer"].astype(float)),
}
# categorical slots 1-3 of the reference palette (validated all-pairs, light mode)
COLORS = {"SFT": "#2a78d6", "Joint": "#eb6834", "Gain": "#1baf7a"}
MARKERS = {"SFT": "o", "Joint": "s", "Gain": "D"}
INK, MUTED = "#0b0b0b", "#52514e"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path,
                   default=PLOTS / "sft_vs_joint/neuron_property_correlation")
    return p.parse_args()


# ======================================================================
# Data
# ======================================================================
def load_outcomes(model):
    path, pat = SOURCES[model]
    r = pd.read_csv(path)
    r = r[r["run"].str.contains(pat, regex=False) & r["arm"].isin(list(EVALS))]
    if r.duplicated(["run", "arm"]).any():
        raise SystemExit(f"{path}: duplicate (run, arm) rows for {pat}")
    w = r.pivot_table(index=["layer", "index", "position"], columns="arm",
                      values="pearson").reset_index()
    print(f"[prop] {model}: {len(w)} runs, {w[['layer', 'index']].drop_duplicates().shape[0]} "
          f"neurons from {path.relative_to(ROOT)}")
    return w


def per_neuron_table():
    sft, joint = load_outcomes("SFT"), load_outcomes("Joint")
    keys = ["layer", "index", "position"]
    paired = joint.merge(sft, on=keys, suffixes=("_Joint", "_SFT"), validate="1:1")
    if len(paired) != len(joint) or len(paired) != len(sft):
        raise SystemExit(f"SFT ({len(sft)}) and Joint ({len(joint)}) runs do not pair "
                         f"one-to-one ({len(paired)} paired)")
    for e in EVALS:
        paired[f"{e}_Gain"] = paired[f"{e}_Joint"] - paired[f"{e}_SFT"]
    cols = [f"{e}_{m}" for m in MODELS for e in EVALS]
    neu = paired.groupby(["layer", "index"])[cols].mean().reset_index()
    neu["n_positions"] = paired.groupby(["layer", "index"]).size().values
    feats = load_features()
    X = pd.DataFrame({f: fn(feats) for f, (_, fn) in FEATURES.items() if f != "layer"})
    X[["layer", "index"]] = feats[["layer", "index"]]
    neu = neu.merge(X, on=["layer", "index"], how="left", validate="1:1")
    neu["layer_f"] = neu["layer"].astype(float)
    missing = neu[[f for f in FEATURES if f != "layer"]].isna().any(axis=1)
    if missing.any():
        raise SystemExit(f"features missing for {missing.sum()} grid neurons")
    return neu


# ======================================================================
# Statistics
# ======================================================================
def corr_stats(x, y, n_absorbed=0):
    """Pearson r, Fisher-z 95% CI and two-sided p. `n_absorbed` = parameters already
    spent on fixed effects (layer means), which reduce the effective sample size."""
    r = float(np.corrcoef(x, y)[0, 1])
    n_eff = len(x) - n_absorbed
    dof = n_eff - 2
    tstat = r * np.sqrt(dof / max(1 - r ** 2, 1e-12))
    p = float(2 * tdist.sf(abs(tstat), dof))
    z, se = np.arctanh(np.clip(r, -0.999999, 0.999999)), 1 / np.sqrt(n_eff - 3)
    lo, hi = np.tanh(z + np.array([-1, 1]) * norm.ppf(0.975) * se)
    return dict(r=r, ci_lo=float(lo), ci_hi=float(hi), p=p, n=len(x), dof=dof)


def correlations(neu):
    layers = neu["layer"]
    n_absorbed = layers.nunique() - 1
    rows = []
    for f in FEATURES:
        x = neu["layer_f" if f == "layer" else f]
        for m in MODELS:
            for e in EVALS:
                y = neu[f"{e}_{m}"]
                rows.append(dict(feature=f, model=m, eval=e, scope="pooled",
                                 **corr_stats(x.values, y.values)))
                if f == "layer":
                    continue
                xd = x - x.groupby(layers).transform("mean")
                yd = y - y.groupby(layers).transform("mean")
                rows.append(dict(feature=f, model=m, eval=e, scope="within",
                                 **corr_stats(xd.values, yd.values, n_absorbed)))
    return pd.DataFrame(rows)


# ======================================================================
# Outputs
# ======================================================================
def markdown(stats, neu):
    def cell(s):
        star = "*" if s.p < 0.05 else ""
        return f"{s.r:+.2f}{star}"

    out = [f"Pearson r between base-model neuron properties and introspection performance "
           f"(n = {len(neu)} randomly drawn neurons, 5 per layer x 8 layers, each averaged "
           f"over 8 token positions; * p < 0.05).", ""]
    for scope, title in (("pooled", "Pooled across layers"),
                         ("within", "Within layer (layer means removed)")):
        out += [f"**{title}**", "",
                "| Property | SFT Endo. | SFT Interv. | Ours Endo. | Ours Interv. "
                "| Gain Endo. | Gain Interv. |",
                "|---|---|---|---|---|---|---|"]
        s = stats[stats["scope"] == scope].set_index(["feature", "model", "eval"])
        for f, (lab, _) in FEATURES.items():
            if (f, "SFT", "endogenous") not in s.index:
                continue
            out.append("| " + lab + " | " + " | ".join(
                cell(s.loc[(f, m, e)]) for m in MODELS for e in EVALS) + " |")
        out.append("")
    return "\n".join(out)


def plot(stats, out_png):
    feats = [f for f in FEATURES]
    scopes = [("pooled", "Pooled Across Layers"), ("within", "Within Layer")]
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[len(feats), len(feats) - 1]))
    offs = dict(zip(MODELS, (0.22, 0.0, -0.22)))
    for i, (scope, stitle) in enumerate(scopes):
        fl = [f for f in feats if not (scope == "within" and f == "layer")]
        ypos = dict(zip(fl, np.arange(len(fl))[::-1]))
        for j, (e, etitle) in enumerate(EVALS.items()):
            ax = axes[i, j]
            s = stats[(stats["scope"] == scope) & (stats["eval"] == e)]
            for m in MODELS:
                for f in fl:
                    row = s[(s["feature"] == f) & (s["model"] == m)].iloc[0]
                    y = ypos[f] + offs[m]
                    ax.plot([row.ci_lo, row.ci_hi], [y, y], color=COLORS[m], lw=2,
                            solid_capstyle="round")
                    ax.scatter([row.r], [y], marker=MARKERS[m], s=64, zorder=3,
                               linewidths=1.8, edgecolors=COLORS[m],
                               facecolors=COLORS[m] if row.p < 0.05 else "white")
            ax.axvline(0, color=MUTED, lw=0.8)
            for y in list(ypos.values())[:-1]:
                ax.axhline(y - 0.5, color="#e6e5e1", lw=0.8, zorder=0)
            ax.set_xlim(-1, 1)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_ylim(-0.6, len(fl) - 0.4)
            ax.set_yticks(list(ypos.values()), [FEATURES[f][0] for f in fl])
            if j == 1:
                ax.tick_params(labelleft=False)
            ax.set_title(f"{etitle} Corr. — {stitle}", fontsize=12, color=INK)
            ax.grid(axis="x", alpha=0.25)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Pearson r with Property (95% CI)")
    handles = [plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], lw=2, ms=8,
                          label=MODEL_LABELS[m]) for m in MODELS]
    handles.append(plt.Line2D([], [], color=MUTED, marker="o", mfc="white", ls="", ms=8,
                              label="Hollow = p ≥ 0.05"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    neu = per_neuron_table()
    stats = correlations(neu)
    neu.to_csv(args.out_dir / "neuron_property_correlation_data.csv", index=False)
    stats.to_csv(args.out_dir / "neuron_property_correlation.csv", index=False)
    md = markdown(stats, neu)
    (args.out_dir / "neuron_property_correlation.md").write_text(md)
    plot(stats, args.out_dir / "neuron_property_correlation.png")
    print(md)
    print(f"[prop] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
