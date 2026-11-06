#!/usr/bin/env python3
"""
Which properties of a neuron predict how well SFT introspection on it works?

One regression per outcome (SFT interventional corr, SFT endogenous corr; gate off,
each neuron averaged over its 8 token positions) on every per-neuron feature built so
far, with layer fixed effects.

FEATURES (all from the base model; see FEATURES below for the transforms)
  activation   log std, mean/std, firing rate (a > 0), strong-firing rate (a above 25%
               of the neuron's own 99th percentile), log firing magnitude (mean of
               a > 0), log kurtosis
                 mechanistic_analysis/score_neuron_activation_stats.py
  redundancy   log10(1 - R2) of the neuron from its same-layer siblings
                 mechanistic_analysis/score_neuron_redundancy.py
  context      n_eff, R2 with the current token only, token-identity table R2
                 mechanistic_analysis/score_neuron_context_dependence.py

MODEL
  outcome ~ layer (fixed effects) + features
  Features are z-scored by their WITHIN-LAYER standard deviation, so a coefficient is
  the change in the outcome per 1 within-layer SD of that feature, holding layer and
  every other feature fixed. Layer enters as dummies, never as a feature to rank: it
  is the confound, and every coefficient here is a within-layer effect.

WHAT IS REPORTED
  coefficients   beta, 95% bootstrap CI (neurons resampled within layer), OLS p, VIF
                 (within-layer). With ~10 correlated features on 80 neurons a single
                 coefficient can be unstable: read VIF next to it. A large VIF means
                 the feature shares its signal with others and its coefficient is
                 only the part they do not explain.
  cv_r2          out-of-sample R2 (repeated 5-fold, folds stratified by layer):
                 layer only / features only / layer + features. Whether the features
                 predict anything beyond layer is layer+features minus layer only.
  unique         the drop in CV R2 when one feature (or one feature group) is removed
                 from the full model: what it adds that no other feature carries.
                 Robust to collinearity in the way coefficients are not -- but two
                 interchangeable features each show ~0 unique contribution.

Outputs: outputs/plots/mlp_endo_sweep_layer_token/feature_regression/
  feature_regression_coefficients.csv, _cv_r2.csv, _unique.csv, _data.csv
  feature_regression_coefficients.png, _cv_r2.png, feature_correlations.png

Usage
  python plotters/fit_sft_feature_regression.py [--gate off] [--n-boot 2000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as tdist

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "outputs/plots/mlp_endo_sweep_layer_token"
SFT_TABLES = [ROOT / "outputs/plots/mlp_endo_sweep_layer_token_n10/sweep_runs_both_gates.csv",
              ROOT / "outputs/plots/mlp_endo_sweep_layer_token/sweep_runs_both_gates.csv"]
OUTCOMES = [("interventional", "SFT Interventional Corr"), ("endogenous", "SFT Endogenous Corr")]
# name -> (label, group, transform of the merged frame)
FEATURES = {
    "log_std":        ("log Std", "activation", lambda d: np.log(d["std"])),
    "mean_over_std":  ("Mean / Std", "activation", lambda d: d["mean"] / d["std"]),
    "frac_positive":  ("Firing Rate (a > 0)", "activation", lambda d: d["frac_positive"]),
    "frac_strong":    ("Strong-Firing Rate", "activation", lambda d: d["frac_strong"]),
    "log_mean_pos":   ("log Firing Magnitude", "activation", lambda d: np.log(d["mean_pos"])),
    "log_kurtosis":   ("log Kurtosis", "activation", lambda d: np.log(d["kurtosis"] + 3)),
    "log_unexplained": ("log10(1 − Redundancy R²)", "redundancy",
                        lambda d: np.log10(np.clip(1 - d["R2"], 1e-6, None))),
    "n_eff":          ("Effective Context Length", "context", lambda d: d["n_eff"]),
    "r2_1":           ("R² Current Token Only", "context", lambda d: d["r2_1"]),
    "r2_token_table": ("Token-Identity R²", "context", lambda d: d["r2_token_table"]),
}
GROUP_COLORS = {"activation": "tab:blue", "redundancy": "tab:orange", "context": "tab:green"}
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate", default="off", choices=["off", "on"])
    p.add_argument("--features", default=None,
                   help=f"Comma subset of {list(FEATURES)} (default: all).")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-repeats", type=int, default=50, help="CV repeats.")
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=GRID / "feature_regression")
    return p.parse_args()


# ======================================================================
# Data
# ======================================================================
def load_sft(gate):
    path = next(p for p in SFT_TABLES if p.exists())
    d = pd.read_csv(path)
    d = d[(d["metric"] == "corr") & (d["gate"] == gate)]
    out = d.pivot_table(index=["neuron", "layer", "index"], columns="eval_type",
                        values="value", aggfunc="mean").reset_index()
    print(f"[reg] {path.name}: {len(out)} neurons, gate {gate}")
    return out


def load_features():
    stats = pd.read_csv(GRID / "activation_stats/activation_stats.csv")
    ctx = pd.read_csv(GRID / "context_dependence/context_dependence.csv")
    ctx = ctx[ctx["is_grid"]][["layer", "index", "n_eff", "r2_1", "r2_token_table"]]
    red = []
    for f in sorted((GRID / "redundancy").glob("L*/neuron_redundancy.csv")):
        r = pd.read_csv(f).rename(columns={"neuron": "index"})
        r["layer"] = int(f.parent.name[1:])
        red.append(r[["layer", "index", "R2"]])
    red = pd.concat(red)
    m = stats.merge(ctx, on=["layer", "index"], how="left").merge(red, on=["layer", "index"],
                                                                  how="left")
    return m


def design(m, names):
    X = pd.DataFrame({n: FEATURES[n][2](m) for n in names}, index=m.index)
    return X


# ======================================================================
# Model
# ======================================================================
def layer_dummies(layers):
    u = sorted(set(layers))
    return np.column_stack([(layers == l).astype(float) for l in u])   # full set, no intercept


def within_z(X, layers):
    """Z-score each feature by its within-layer SD (layer means removed for the SD only;
    the layer dummies absorb the means in the fit)."""
    dm = X - X.groupby(layers).transform("mean")
    return (X - X.mean()) / dm.std(ddof=1)


def ols(y, A):
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = len(y) - np.linalg.matrix_rank(A)
    s2 = resid @ resid / max(dof, 1)
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, se, dof


def fit_coefficients(y, Z, layers, n_boot, rng):
    names = list(Z.columns)
    D = layer_dummies(layers)
    A = np.column_stack([D, Z.values])
    beta, se, dof = ols(y, A)
    k = D.shape[1]
    b, s = beta[k:], se[k:]
    p = 2 * tdist.sf(np.abs(b / np.where(s > 0, s, np.nan)), dof)
    # bootstrap: resample neurons within layer, refit
    idx_by_layer = [np.flatnonzero(layers == L) for L in sorted(set(layers))]
    boots = np.empty((n_boot, len(names)))
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by_layer])
        bb, *_ = np.linalg.lstsq(A[idx], y[idx], rcond=None)
        boots[i] = bb[k:]
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    dm = Z - Z.groupby(layers).transform("mean")
    vif = vifs(dm.values)
    return pd.DataFrame(dict(feature=names, beta=b, se=s, p=p, ci_lo=lo, ci_hi=hi, vif=vif))


def vifs(X):
    out = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(len(X)), others])
        beta, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        r = X[:, j] - A @ beta
        r2 = 1 - r.var() / X[:, j].var()
        out.append(1 / max(1 - r2, 1e-12))
    return out


def stratified_folds(layers, k, rng):
    fold = np.empty(len(layers), int)
    for L in sorted(set(layers)):
        ix = np.flatnonzero(layers == L)
        fold[ix] = rng.permutation(np.arange(len(ix)) % k)
    return fold


def cv_r2(y, blocks, layers, k, repeats, seed):
    """Out-of-sample R2 of OLS on the concatenated blocks, pooled over folds, averaged
    over repeats. Returns (mean, sd over repeats)."""
    A = np.column_stack([np.ones(len(y))] + [b for b in blocks if b.shape[1]])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(repeats):
        fold = stratified_folds(layers, k, rng)
        pred = np.empty(len(y))
        for f in range(k):
            tr, te = fold != f, fold == f
            beta, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
            pred[te] = A[te] @ beta
        vals.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(vals)), float(np.std(vals))


# ======================================================================
# Plots
# ======================================================================
def plot_coefficients(coef, out_png):
    fig, axes = plt.subplots(1, len(OUTCOMES), figsize=(7 * len(OUTCOMES), 0.55 * coef["feature"].nunique() + 2.2),
                             sharey=True)
    order = list(FEATURES)
    feats = [f for f in order if f in set(coef["feature"])]
    ypos = np.arange(len(feats))[::-1]
    for ax, (oc, lab) in zip(axes, OUTCOMES):
        c = coef[coef["outcome"] == oc].set_index("feature").loc[feats]
        for y, (f, r) in zip(ypos, c.iterrows()):
            col = GROUP_COLORS[FEATURES[f][1]]
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=2.2)
            ax.scatter([r.beta], [y], s=70, color=col if r.p < 0.05 else "white",
                       edgecolors=col, linewidths=2, zorder=3)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(lab)
        ax.set_xlabel("β per Within-Layer SD (95% Bootstrap CI)")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(ypos, [f"{FEATURES[f][0]}  (VIF {coef[coef.feature == f]['vif'].iloc[0]:.1f})"
                              for f in feats])
    handles = [plt.Line2D([], [], color=c, lw=2.2, marker="o", label=g.capitalize())
               for g, c in GROUP_COLORS.items()]
    handles += [plt.Line2D([], [], color="gray", marker="o", ls="", ms=8, label="p < 0.05"),
                plt.Line2D([], [], color="gray", marker="o", ls="", ms=8, mfc="white",
                           label="p ≥ 0.05")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Feature Coefficients, Layer Fixed Effects", fontsize=16)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cv(cv, uniq, out_png):
    fig, axes = plt.subplots(2, len(OUTCOMES), figsize=(7.5 * len(OUTCOMES), 10.5),
                             gridspec_kw=dict(height_ratios=[1, 2]))
    for j, (oc, lab) in enumerate(OUTCOMES):
        ax = axes[0, j]
        c = cv[cv["outcome"] == oc]
        ax.bar([m.replace(" + ", "\n+ ") for m in c["model"]], c["cv_r2"], yerr=c["cv_r2_sd"],
               color=["gray", "tab:purple", "k"], capsize=4)
        for x, (v, sd) in enumerate(zip(c["cv_r2"], c["cv_r2_sd"])):
            ax.text(x, max(v, 0) + sd + 0.015, f"{v:.2f}", ha="center", fontsize=12)
        ax.set_ylim(min(0, c["cv_r2"].min() - 0.05), (c["cv_r2"] + c["cv_r2_sd"]).max() + 0.08)
        ax.set_ylabel("Out-of-Sample R²")
        ax.set_title(lab)
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(axis="y", alpha=0.25)
        ax = axes[1, j]
        u = uniq[uniq["outcome"] == oc].sort_values("delta_cv_r2")
        colors = [GROUP_COLORS.get(FEATURES[f][1], "k") if f in FEATURES
                  else GROUP_COLORS.get(f.replace("group: ", ""), "k") for f in u["feature"]]
        labels = [FEATURES[f][0] if f in FEATURES else f.replace("group: ", "All ").title()
                  for f in u["feature"]]
        ax.barh(labels, u["delta_cv_r2"], color=colors,
                hatch=["//" if f.startswith("group") else "" for f in u["feature"]])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Drop in CV R² When Removed (Unique Contribution)")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlations(Z, layers, out_png):
    dm = Z - Z.groupby(layers).transform("mean")
    C = dm.corr().values
    labels = [FEATURES[f][0] for f in Z.columns]
    fig, ax = plt.subplots(figsize=(0.75 * len(labels) + 3, 0.75 * len(labels) + 2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(C)):
        for j in range(len(C)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(C[i, j]) > 0.6 else "k")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Within-Layer Feature Correlations")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    names = args.features.split(",") if args.features else list(FEATURES)
    m = load_sft(args.gate).merge(load_features(), on=["layer", "index"], how="left",
                                  suffixes=("", "_feat"))
    X = design(m, names)
    ok = X.notna().all(axis=1) & m[["interventional", "endogenous"]].notna().all(axis=1)
    if (~ok).any():
        print(f"[reg] dropping {(~ok).sum()} neurons with missing features: "
              f"{m.loc[~ok, 'neuron'].tolist()[:10]}")
    m, X = m[ok].reset_index(drop=True), X[ok].reset_index(drop=True)
    layers = m["layer"].values
    Z = within_z(X, layers)
    D = layer_dummies(layers)[:, 1:]                       # intercept added in cv_r2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([m, X.add_prefix("x_")], axis=1).to_csv(
        args.out_dir / "feature_regression_data.csv", index=False)
    print(f"[reg] {len(m)} neurons, {len(names)} features, {len(set(layers))} layers")

    coefs, cvs, uniq = [], [], []
    for oc, _ in OUTCOMES:
        y = m[oc].values
        c = fit_coefficients(y, Z, layers, args.n_boot, rng)
        coefs.append(c.assign(outcome=oc))
        cv = lambda blocks: cv_r2(y, blocks, layers, args.k_folds, args.n_repeats, args.seed)
        for model, blocks in (("Layer Only", [D]), ("Features Only", [Z.values]),
                              ("Layer + Features", [D, Z.values])):
            r, s = cv(blocks)
            cvs.append(dict(outcome=oc, model=model, cv_r2=r, cv_r2_sd=s))
        full, _ = cv([D, Z.values])
        for f in names:
            r, _ = cv([D, Z.drop(columns=f).values])
            uniq.append(dict(outcome=oc, feature=f, delta_cv_r2=full - r))
        for g in sorted({FEATURES[f][1] for f in names}):
            drop = [f for f in names if FEATURES[f][1] == g]
            if len(drop) > 1:
                r, _ = cv([D, Z.drop(columns=drop).values])
                uniq.append(dict(outcome=oc, feature=f"group: {g}", delta_cv_r2=full - r))
    coef, cvdf, uq = pd.concat(coefs), pd.DataFrame(cvs), pd.DataFrame(uniq)
    coef.to_csv(args.out_dir / "feature_regression_coefficients.csv", index=False)
    cvdf.to_csv(args.out_dir / "feature_regression_cv_r2.csv", index=False)
    uq.to_csv(args.out_dir / "feature_regression_unique.csv", index=False)
    plot_coefficients(coef, args.out_dir / "feature_regression_coefficients.png")
    plot_cv(cvdf, uq, args.out_dir / "feature_regression_cv_r2.png")
    plot_correlations(Z, layers, args.out_dir / "feature_correlations.png")
    fmt = lambda v: f"{v:.3g}"
    print("\n" + coef[["outcome", "feature", "beta", "ci_lo", "ci_hi", "p", "vif"]]
          .to_string(index=False, float_format=fmt))
    print("\n" + cvdf.to_string(index=False, float_format=fmt))
    print("\n" + uq.sort_values(["outcome", "delta_cv_r2"], ascending=[True, False])
          .to_string(index=False, float_format=fmt))
    print(f"[reg] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
