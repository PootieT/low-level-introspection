"""
Investigate whether introspecting on neurons that are less redundant results in smaller gap of between interventional
eval vs. non interventional eval (if the redundancy is the cause of weird gap)

Findings:
"""
import os
import pdb
import json
import glob
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr, t as tdist


def partial_corr(x, y, Z):
    """
    Partial correlation of x and y controlling for covariates Z, via
    residualization: regress x and y on [1, Z], correlate the residuals.
    Returns (r_partial, p_value, dof). p uses dof = n - 2 - #covariates.
    """
    x = np.asarray(x, float); y = np.asarray(y, float); Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    n = len(x)
    A = np.column_stack([np.ones(n), Z])

    def resid(v):
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ beta

    rx, ry = resid(x), resid(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan"), float("nan"), n - 2 - Z.shape[1]
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - 2 - Z.shape[1]
    if dof <= 0:
        return r, float("nan"), dof
    tstat = r * np.sqrt(dof / max(1 - r * r, 1e-12))
    return r, float(2 * tdist.sf(abs(tstat), dof)), dof


def parse_dir(exp_dir):
    neuron = exp_dir.split("_")[3]
    layer, index = neuron.replace("N","").split(".")
    layer, index = int(layer), int(index)
    bin = int(exp_dir.split("_")[4].replace("Bin",""))
    return {"neuron": neuron, "layer": layer, "neuron_index": index, "bin": bin}


REDUNDANCY_CSV = "outputs/circuit/intervention_train_regressor_extreme/intervention_train_reg_extreme10_300K/candidate_neurons.csv"

def analyze_correlation(exp_group_path: str, redundancy_csv: str=REDUNDANCY_CSV):
    exp_group_name = exp_group_path.split("/")[-1]
    all_res = []
    for p in tqdm(glob.glob(os.path.join(exp_group_path, "*/interventional_test_metrics_gaussian.json"))):
        with open(p) as f:
            res = json.load(f)["test"]
        exp_name = p.split("/")[-2]
        res = {f"test_intervention_{k}": v for k, v in res.items()}
        res.update(parse_dir(exp_name))

        # load dataset
        alt_eval_path = f"{Path(p).parent}/test_alt_intervention_results.json"
        with open(alt_eval_path) as f:
            stats = json.load(f)
        eval_res = {f"test_{k}": v for k, v in stats.items()}
        res.update(eval_res)
        all_res.append(res)
    df = pd.DataFrame.from_records(all_res)

    # long form: one row per (experiment, eval_type) so we can hue on eval_type
    long_df = df.melt(
        id_vars=["bin", "neuron", "layer", "neuron_index"],
        value_vars=["test_corr", "test_intervention_corr"],
        var_name="eval_type", value_name="corr",
    )
    long_df["eval_type"] = long_df["eval_type"].map(
        {"test_corr": "non_intervention", "test_intervention_corr": "intervention"})

    # 1) combined boxplot: bin vs corr, hue = eval type
    plt.figure()
    sns.boxplot(x="bin", y="corr", hue="eval_type", data=long_df)
    plt.title("test correlation vs neuron redundancy bin")
    plt.tight_layout()
    plt.savefig("outputs/plots/box_test_correlation_vs_neuron_redundancy_bin.png")
    plt.close()

    # 2) scatterplot vs actual redundancy R2, hue = eval type (+ robust stats)
    red = pd.read_csv(redundancy_csv).rename(columns={"neuron": "neuron_index"})
    covars = [c for c in ["std", "firing_rate"] if c in red.columns]
    merged = long_df.merge(red[["neuron_index", "R2"] + covars], on="neuron_index", how="left")

    eval_types = ["non_intervention", "intervention"]
    palette = dict(zip(eval_types, sns.color_palette(n_colors=len(eval_types))))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x="R2", y="corr", hue="eval_type", hue_order=eval_types,
                    palette=palette, data=merged, ax=ax)

    # per-eval-type stats: OLS (leverage-sensitive) + Spearman (rank, robust) +
    # partial correlation controlling std/firing (confound-adjusted)
    stat_lines, stat_rows = [], []
    for et in eval_types:
        sub = merged[merged["eval_type"] == et].dropna(subset=["R2", "corr"] + covars)
        if len(sub) < 3:
            continue
        lr = linregress(sub["R2"], sub["corr"])
        rho, rho_p = spearmanr(sub["R2"], sub["corr"])
        if covars:
            pr, pr_p, dof = partial_corr(sub["R2"], sub["corr"], sub[covars].values)
        else:
            pr, pr_p = float("nan"), float("nan")
        xs = np.linspace(sub["R2"].min(), sub["R2"].max(), 100)
        ax.plot(xs, lr.intercept + lr.slope * xs, color=palette[et], lw=2)
        stat_lines.append(
            f"{et} (n={len(sub)}):\n"
            f"  OLS   slope={lr.slope:+.3f} r={lr.rvalue:+.3f} p={lr.pvalue:.2g}\n"
            f"  Spear rho={rho:+.3f} p={rho_p:.2g}\n"
            f"  partial(|{','.join(covars)}) r={pr:+.3f} p={pr_p:.2g}")
        stat_rows.append({"eval_type": et, "n": len(sub), "ols_slope": lr.slope,
                          "ols_r": lr.rvalue, "ols_p": lr.pvalue, "spearman_rho": rho,
                          "spearman_p": rho_p, "partial_r": pr, "partial_p": pr_p,
                          "controlled_for": ",".join(covars)})
    ax.text(0.02, 0.02, "\n".join(stat_lines), transform=ax.transAxes, va="bottom",
            fontsize=7, family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set_xlabel("same-layer redundancy R2 (context-predictability)")
    ax.set_ylabel("test correlation")
    ax.set_title("test correlation vs neuron redundancy R2")
    fig.tight_layout()
    fig.savefig("outputs/plots/scatter_test_correlation_vs_neuron_redundancy_R2.png")
    plt.close(fig)

    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv("outputs/plots/redundancy_correlation_stats.csv", index=False)
    print(stats_df.to_string(index=False))

    # 3) confound check: same scatter, points colored by each covariate.
    # If the R2 trend is really a std/firing trend, the color gradient will run
    # along R2 and the partial correlation above will be ~0.
    if covars:
        fig2, axes = plt.subplots(len(covars), len(eval_types),
                                  figsize=(5.5 * len(eval_types), 4 * len(covars)), squeeze=False)
        for i, cov in enumerate(covars):
            for j, et in enumerate(eval_types):
                sub = merged[merged["eval_type"] == et].dropna(subset=["R2", "corr", cov])
                axc = axes[i][j]
                sc = axc.scatter(sub["R2"], sub["corr"], c=sub[cov], cmap="viridis", s=30)
                fig2.colorbar(sc, ax=axc, label=cov)
                axc.set_title(f"{et}")
                axc.set_xlabel("redundancy R2"); axc.set_ylabel("corr")
        fig2.suptitle("corr vs R2 colored by covariate (confound check)")
        fig2.tight_layout()
        fig2.savefig("outputs/plots/scatter_covariate_colored.png")
        plt.close(fig2)
    return


if __name__ == "__main__":
    analyze_correlation("outputs/runs/intervention_train_regressor_redundancy")