import os
import re
import json
import glob
import pdb
from pathlib import Path

import numpy as np

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Poster-scale fonts: large enough to read from a few feet away.
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "legend.title_fontsize": 20,
    "figure.titlesize": 28,
})

ROOT=Path(__file__).parents[1]


def _metric_words(metric: str) -> str:
    """Readable colorbar name for a sweep metric key, e.g. 'test_corr' ->
    'Endogenous Correlation', 'intervention_l2_dist' -> 'Interventional L2'."""
    quantity = "Correlation" if metric.endswith("corr") else "L2"
    if metric.startswith("intervention"):
        kind = "Interventional"
    elif metric.startswith("test_pred"):
        kind = "Predicted Endogenous"
    else:                                    # test_ / endogenous readout
        kind = "Endogenous"
    return f"{kind} {quantity}"

def plot_token_overlap_vs_test_metrics(exp_pattern:str, output_dir:str="single_neuron"):
    data_dirs = sorted(glob.glob(f"{ROOT}/data/{exp_pattern}*"))
    out_dirs = sorted(glob.glob(f"{ROOT}/outputs/runs/{exp_pattern}*"))
    assert len(data_dirs) == len(out_dirs)
    df = []
    for data_dir, out_dir in zip(data_dirs, out_dirs):
        with open(f"{data_dir}/token_overlaps.json", encoding="utf-8") as f:
            overlaps = json.load(f)
        with open(f"{out_dir}/test_seen_token_results.json", encoding="utf-8") as f:
            seen_res = json.load(f)
        with open(f"{out_dir}/test_unseen_token_results.json", encoding="utf-8") as f:
            unseen_res = json.load(f)
        res = {
            "test_token_overlaps": overlaps["train_test_overlaps"]["unique_overlap"],
            "test_token_overlaps_normalized": overlaps["train_test_overlaps"]["unique_overlap"]/overlaps["train_test_overlaps"]["unique_eval"],
        }

        res_seen = res.copy()
        for metric, metric_val in seen_res.items():
            res_seen[metric] = metric_val
            res_seen["token_seen_in_train"] = True
        df.append(res_seen)

        res_unseen = res.copy()
        for metric, metric_val in unseen_res.items():
            res_unseen[metric] = metric_val
            res_unseen["token_seen_in_train"] = False
        df.append(res_unseen)

    df = pd.DataFrame(df)
    plot_dir = f"{ROOT}/outputs/plots/single_neuron"
    os.makedirs(plot_dir, exist_ok=True)

    for metric in tqdm(unseen_res.keys()):
        sns.lineplot(data=df, x="test_token_overlaps", y=metric, hue="token_seen_in_train")
        plt.savefig(f"{plot_dir}/token_overlap_vs_test_{metric}.png")
        plt.close()


def plot_sweep_eval_metrics(exp_name: str, output_dir:str="single_neuron"):
    eval_dirs = sorted(glob.glob(f"{ROOT}/outputs/runs/{exp_name}/*"))
    rows = []
    for eval_dir in tqdm(eval_dirs):
        try:
            with open(f"{eval_dir}/test_pred_results.json", encoding="utf-8") as f:
                test_pred_res = json.load(f)
            with open(f"{eval_dir}/test_results.json", encoding="utf-8") as f:
                test_res = json.load(f)
            with open(f"{eval_dir}/intervention_results_metrics.json", encoding="utf-8") as f:
                intervention_res = json.load(f)

            test_results = {f"test_{k}": v for k, v in test_res.items()}
            test_pred_results = {f"test_pred_{k}": v for k, v in test_pred_res.items()}
            intervention_results = {f"intervention_{k}": v for k, v in intervention_res[list(intervention_res.keys())[0]].items()}
            exp_dir_splits = eval_dir.split("/")[-1].split("_")
            rows.append({
                "sample_index": int(exp_dir_splits[1].replace("sample","")) if "sample" in exp_dir_splits[1] else int(exp_dir_splits[2].replace("sample","")),
                "layer": int(exp_dir_splits[0].replace("l","")),
                "token_pos": int(exp_dir_splits[2].replace("pos","")) if "pos" in exp_dir_splits[2] else int(exp_dir_splits[1].replace("pos","")),
                **test_results,
                **test_pred_results,
                **intervention_results,
            })
        except Exception as e:
            continue

    # TODO could annotate with mean+-std,
    df = pd.DataFrame(rows)
    metrics = ["test_corr", "test_l2_dist", "test_pred_corr", "test_pred_l2_dist", "intervention_corr", "intervention_l2_dist"]
    for metric in metrics:
        df[metric] = df[metric] * 100
    # shared color scale for all L2 metrics so the interventional and endogenous L2
    # heatmaps use one colorbar -> cell colors are directly comparable across settings.
    l2_metrics = [m for m in metrics if "l2_dist" in m]
    l2_all = pd.concat([df[m] for m in l2_metrics]) if l2_metrics else pd.Series(dtype=float)
    l2_vmin, l2_vmax = (float(l2_all.min()), float(l2_all.max())) if len(l2_all.dropna()) else (0, 100)
    for metric in tqdm(metrics):
        is_l2 = "l2_dist" in metric
        vmin, vmax = (l2_vmin, l2_vmax) if is_l2 else (0, 100)
        pivot_df = df.pivot_table(index="layer", columns="token_pos", values=metric, aggfunc="mean")
        pivot_df_std = df.pivot_table(index="layer", columns="token_pos", values=metric, aggfunc="std")

        annot_df = pd.DataFrame()
        for col in pivot_df.columns:
            annot_df[col] = pivot_df[col].map(lambda x: f"{x:.1f}") + "±" + \
                            pivot_df_std[col].map(lambda x: f"{x:.1f}")

        plt.figure(figsize=[16,16])
        hm = sns.heatmap(
            pivot_df,
            annot=annot_df,  # Pass your custom annotation DataFrame here
            fmt="",
            vmin=vmin, vmax=vmax,
            annot_kws={"size": 14},
            cbar_kws={"label": f"{_metric_words(metric)}\n(Mean ± Std over Neurons)"},
        )
        hm.figure.axes[-1].yaxis.label.set_size(28)  # colorbar label
        hm.figure.axes[-1].tick_params(labelsize=22)  # colorbar ticks
        hm.tick_params(axis="x", labelsize=30)
        hm.tick_params(axis="y", labelsize=30)
        plt.xlabel("Token Position", fontsize=38); plt.ylabel("Layer", fontsize=38)
        plt.tight_layout()
        plot_dir = f"{ROOT}/outputs/plots/single_neuron"
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/sweep_eval_metrics_{metric}.png", bbox_inches="tight")
        plt.close()


def correlate_sweep_metrics_with_data_stats(exp_name: str):
    eval_dirs = sorted(glob.glob(f"{ROOT}/outputs/runs/{exp_name}/*"))
    rows = []
    for eval_dir in tqdm(eval_dirs):
        try:
            run_name = Path(eval_dir).name
            with open(f"{eval_dir}/test_pred_results.json", encoding="utf-8") as f:
                test_res = json.load(f)
            with open(f"{eval_dir}/test_results.json", encoding="utf-8") as f:
                test_pred_res = json.load(f)
            with open(f"{eval_dir}/intervention_results_metrics.json", encoding="utf-8") as f:
                intervention_res = json.load(f)
            with open(f"{ROOT}/data/{exp_name}/{run_name}/token_overlaps.json", encoding="utf-8") as f:
                overlaps = json.load(f)
            with open(f"{ROOT}/data/{exp_name}/{run_name}/dataset_signal_stats.json", encoding="utf-8") as f:
                signal_stats = json.load(f)
            test_results = {f"test_{k}": v for k, v in test_res.items()}
            test_pred_results = {f"test_pred_{k}": v for k, v in test_pred_res.items()}
            intervention_results = {f"intervention_{k}": v for k, v in intervention_res[list(intervention_res.keys())[0]].items()}
            rows.append({
                "sample_index": int(eval_dir.split("/")[-1].split("_")[1].replace("sample", "")),
                "layer": int(eval_dir.split("/")[-1].split("_")[0].replace("l", "")),
                "token_pos": int(eval_dir.split("/")[-1].split("_")[2].replace("pos", "")),
                "test_token_overlaps": overlaps["train_test_overlaps"]["unique_overlap"],
                "dev_token_overlaps": overlaps["train_dev_overlaps"]["unique_overlap"],
                "signal_index": signal_stats["train"][0]["signal_index"],
                "train_signal_mean": signal_stats["train"][0]["mean"],
                "train_signal_std": signal_stats["train"][0]["std"],
                "dev_signal_mean": signal_stats["dev"][0]["mean"],
                "dev_signal_std": signal_stats["dev"][0]["std"],
                **test_results,
                **test_pred_results,
                **intervention_results,
            })
        except Exception as e:
            continue

    df = pd.DataFrame(rows)
    for metric in ["test_corr", "test_l2_dist", "test_pred_corr", "test_pred_l2_dist", "intervention_corr", "intervention_l2_dist"]:
        for sig in ["layer", "token_pos", "sample_index", "dev_token_overlaps", "train_signal_mean", "train_signal_std", "dev_signal_mean", "dev_signal_std"]:
            print(f"Metric {metric} vs. attribute {sig} correlation: {df[metric].corr(df[sig])}")

def plot_sweep_eval_metrics_intervention_train(exp_name: str, output_dir:str="single_neuron_intervention_train"):
    """ Plot interventional train results (no joint loss yet) """
    eval_dirs = sorted(glob.glob(f"{ROOT}/outputs/runs/{exp_name}/*"))
    rows = []
    fail_cnt = 0
    for eval_dir in tqdm(eval_dirs):
        try:
            with open(f"{eval_dir}/test_alt_intervention_results.json", encoding="utf-8") as f:
                test_res = json.load(f)
            with open(f"{eval_dir}/interventional_test_metrics_gaussian.json", encoding="utf-8") as f:
                intervention_res = json.load(f)

            test_results = {f"test_{k}": v for k, v in test_res.items()}
            intervention_results = {f"intervention_{k}": v for k, v in intervention_res["test"].items()}
            exp_dir_splits = eval_dir.split("/")[-1].split("_")
            rows.append({
                "sample_index": int(exp_dir_splits[2].replace("sample","")),
                "layer": int(exp_dir_splits[0].replace("l","")),
                "token_pos": int(exp_dir_splits[1].replace("pos","")),
                **test_results,
                **intervention_results,
            })
        except Exception as e:
            print(f"failed at {eval_dir}: {e}")
            fail_cnt += 1
            continue

    df = pd.DataFrame(rows)
    metrics = ["test_corr", "test_l2_dist", "intervention_corr", "intervention_l2_dist"]
    for metric in metrics:
        df[metric] = df[metric] * 100
    # shared color scale for all L2 metrics so the interventional and endogenous L2
    # heatmaps use one colorbar -> cell colors are directly comparable across settings.
    l2_metrics = [m for m in metrics if "l2_dist" in m]
    l2_all = pd.concat([df[m] for m in l2_metrics]) if l2_metrics else pd.Series(dtype=float)
    l2_vmin, l2_vmax = (float(l2_all.min()), float(l2_all.max())) if len(l2_all.dropna()) else (0, 100)
    for metric in tqdm(metrics):
        is_l2 = "l2_dist" in metric
        vmin, vmax = (l2_vmin, l2_vmax) if is_l2 else (0, 100)
        pivot_df = df.pivot_table(index="layer", columns="token_pos", values=metric, aggfunc="mean")
        pivot_df_std = df.pivot_table(index="layer", columns="token_pos", values=metric, aggfunc="std")

        annot_df = pd.DataFrame()
        for col in pivot_df.columns:
            annot_df[col] = pivot_df[col].map(lambda x: f"{x:.1f}") + "±" + \
                            pivot_df_std[col].map(lambda x: f"{x:.1f}")

        plt.figure(figsize=[16,16])
        hm = sns.heatmap(
            pivot_df,
            annot=annot_df,  # Pass your custom annotation DataFrame here
            fmt="",
            vmin=vmin, vmax=vmax,
            annot_kws={"size": 14},
            cbar_kws={"label": f"{_metric_words(metric)}\n(Mean ± Std over Neurons)"},
        )
        hm.figure.axes[-1].yaxis.label.set_size(28)  # colorbar label
        hm.figure.axes[-1].tick_params(labelsize=22)  # colorbar ticks
        hm.tick_params(axis="x", labelsize=30)
        hm.tick_params(axis="y", labelsize=30)
        plt.xlabel("Token Position", fontsize=38); plt.ylabel("Layer", fontsize=38)
        plt.tight_layout()
        plot_dir = f"{ROOT}/outputs/plots/{output_dir}"
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/sweep_eval_metrics_{metric}.png", bbox_inches="tight")
        plt.close()



# ============================================================
# Endogenous (W=1.0) layer x token sweep
#   scripts/neuron/train_single_mlp_introspector_endogenous_sweep_layer_token.sh
#   run dirs: endo_N<layer>.<idx>_POS<pos>_gate<on|off>_r16_lr5e-5_w1.0_300K
#
# Each (layer, position) cell holds 10 neurons, so every point is a mean over 10
# runs and the spread across them is NEURON HETEROGENEITY, not seed noise. The
# neuron set is fixed per layer across all positions (resolution.py seeds per
# (seed, layer)), so a neuron can be followed along the position axis -- which is
# what hue="neuron" is for.
#
# Two evals per run:
#   endogenous    -- predict the true activation (what the model was trained on)
#   interventional -- inject a value and see whether the report follows. With no
#                    interventional loss (W=1.0) this measures CONTEXT BYPASS: how
#                    much of the endogenous score is causal reading rather than
#                    prediction from context.
# ============================================================

# Independent searches rather than one whole-name pattern, so both naming variants
# parse: the full form endo_N14.1668_POS-28_gateon_r16_lr5e-5_w1.0_300K and the
# shortened endo_N12.7950_POS-1_gateon_w1.0_300K (r/lr dropped). Fields absent from a
# name come back None instead of failing the match.
NEURON_RE = re.compile(r"N(?P<layer>\d+)\.(?P<index>\d+)")
POS_RE = re.compile(r"POS(?P<pos>-?\d+)")
GATE_RE = re.compile(r"gate(?P<gate>on|off)")
R_RE = re.compile(r"_r(?P<r>\d+)_")
LR_RE = re.compile(r"_lr(?P<lr>[0-9.e+-]+?)_")
W_RE = re.compile(r"_w(?P<w>[0-9.]+)")

# (file, key path into the json, label). Both files are written by
# train_introspector_joint.py at the end of a run.
ENDO_EVALS = [
    ("endogenous_test_metrics_endogenous.json", None, "endogenous"),
    ("interventional_test_metrics_gaussian.json", "test", "interventional"),
]
ENDO_METRICS = ("corr", "l2_dist")
# metrics whose natural range is [0, 1] -> fixed colour/axis scale
RANGE01 = {"corr"}


def load_endogenous_sweep(exp_group: str, metrics=ENDO_METRICS, gate="on",
                          lora_r=None, lr=None, w=None) -> pd.DataFrame:
    """
    Long-form table: one row per (run, eval, metric).

    `gate` selects one arm -- the group dir also holds gateoff experiments, and
    averaging the two arms into a cell would silently mix two different models.
    Pass gate=None to keep both (and then facet on the column).

    `lora_r` / `lr` / `w` are optional filters for when a group dir holds several
    configurations of the same cell. Runs missing a metrics file are skipped and
    counted, so unfinished cells appear in the coverage report rather than quietly
    shifting a mean.
    """
    run_dirs = sorted(Path(f"{ROOT}/outputs/runs/{exp_group}").glob("endo_*"))
    rows, skipped, excluded = [], [], 0
    for d in tqdm(run_dirs, desc=exp_group):
        n, p = NEURON_RE.search(d.name), POS_RE.search(d.name)
        if n is None or p is None:
            skipped.append((d.name, "unparsable name"))
            continue
        g, m_lr, m_w = GATE_RE.search(d.name), LR_RE.search(d.name), W_RE.search(d.name)
        m_r = R_RE.search(d.name)
        base = {
            "run": d.name,
            "neuron": f"N{n['layer']}.{n['index']}",
            "layer": int(n["layer"]), "index": int(n["index"]),
            "position": int(p["pos"]),
            "gate": g["gate"] if g else None,
            "lora_r": int(m_r["r"]) if m_r else None,
            "lr": m_lr["lr"] if m_lr else None,
            "w": float(m_w["w"]) if m_w else None,
        }
        if ((gate is not None and base["gate"] != gate)
                or (lora_r is not None and base["lora_r"] != lora_r)
                or (lr is not None and base["lr"] != lr)
                or (w is not None and base["w"] != w)):
            excluded += 1
            continue
        for fname, key, eval_name in ENDO_EVALS:
            fpath = d / fname
            if not fpath.exists():
                skipped.append((d.name, f"missing {fname}"))
                continue
            payload = json.loads(fpath.read_text())
            payload = payload[key] if key else payload
            for metric in metrics:
                if payload.get(metric) is not None:
                    # column is eval_type, NOT eval: a column named "eval" shadows
                    # DataFrame.eval, so df.eval == "endogenous" silently returns
                    # all-False for anyone filtering the exported CSV interactively
                    rows.append({**base, "eval_type": eval_name, "metric": metric,
                                 "value": float(payload[metric])})
    if excluded:
        print(f"[endo] excluded {excluded} run(s) not matching the requested "
              f"gate={gate} lora_r={lora_r} lr={lr} w={w}")
    if skipped:
        print(f"[endo] skipped {len(skipped)} run/file(s); first few: {skipped[:5]}")
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No metrics parsed under {ROOT}/outputs/runs/{exp_group} "
                         f"(gate={gate} lora_r={lora_r} lr={lr} w={w})")

    # What configurations actually survived? A cell built from two different configs
    # is not a cell, and with two naming schemes in one directory that is easy to do
    # by accident.
    inventory = (df.groupby(["gate", "lora_r", "lr", "w"], dropna=False)["run"]
                 .nunique().reset_index(name="n_runs"))
    print("[endo] configurations kept:")
    print(inventory.to_string(index=False))
    if len(inventory) > 1:
        print("[endo] WARNING: more than one configuration is present. If these are not meant "
              "to be pooled, pass lora_r=/lr=/w= to pick one.")

    # Same neuron+position twice means one cell is averaging two runs -- typically the
    # short-named and fully-specified copies of the same experiment.
    keys = ["neuron", "position", "eval_type", "metric"]
    dup = df[df.duplicated(keys, keep=False)]
    if len(dup):
        pairs = (dup.groupby(keys)["run"].apply(lambda r: sorted(set(r))).reset_index(name="runs"))
        print(f"[endo] WARNING: {len(pairs)} duplicated (neuron, position, metric) cell(s); "
              f"keeping the first run name alphabetically. Example: {pairs['runs'].iloc[0]}")
        df = df.sort_values("run").drop_duplicates(keys, keep="first").reset_index(drop=True)
    df["panel"] = df["eval_type"] + " " + df["metric"]
    n_nan = int(df["value"].isna().sum())
    if n_nan:
        # a dead / near-constant neuron gives a nan correlation; drop so it does not
        # silently poison a cell mean, but say how many
        print(f"[endo] dropping {n_nan} nan metric values (dead or constant neurons?)")
        df = df.dropna(subset=["value"])
    return df


def summarize_endogenous_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """mean / standard error / n per (eval, metric, layer, position)."""
    g = df.groupby(["eval_type", "metric", "layer", "position"])["value"]
    out = g.agg(mean="mean", std="std", n="count").reset_index()
    out["se"] = out["std"] / np.sqrt(out["n"].clip(lower=1))
    return out


def report_endogenous_coverage(df: pd.DataFrame, expected_neurons: int = 10):
    """Which cells are short of their full complement of neurons (crashed runs,
    dead neurons)? A cell averaging 3 neurons is not comparable to one averaging 10."""
    cov = (df[df["metric"] == "corr"]
           .groupby(["eval_type", "layer", "position"])["neuron"].nunique().reset_index(name="n_neurons"))
    short = cov[cov["n_neurons"] < expected_neurons]
    print(f"[endo] cells: {len(cov)} | complete: {(cov['n_neurons'] == expected_neurons).sum()}")
    if len(short):
        print(f"[endo] INCOMPLETE cells (<{expected_neurons} neurons):")
        print(short.to_string(index=False))
    return cov


def _fmt2(df: pd.DataFrame) -> pd.DataFrame:
    """Cell text with exactly 2 decimals ('0.50', not '0.5'); NaN -> ''; '-0.00' -> '0.00'."""
    def one(v):
        if pd.isna(v):
            return ""
        s = f"{v:.2f}"
        return "0.00" if s == "-0.00" else s
    return df.apply(lambda col: col.map(one))


def draw_layer_position_heatmap(mean_df: pd.DataFrame, spread_df: pd.DataFrame, title: str,
                                path, vmin=None, vmax=None, cmap="viridis", center=None):
    """The ONE style for every layer x token-position heatmap -- the SFT sweep maps
    here and the SFT-vs-joint arm/delta maps in compare_sft_vs_joint.py -- so panels
    from both scripts sit side by side in a paper figure. Cells read 'mean\\n±spread';
    the metric goes in the title (not on the colorbar)."""
    annot = _fmt2(mean_df) + "\n±" + _fmt2(spread_df.reindex_like(mean_df).fillna(0))
    plt.figure(figsize=(16, 12))
    hm = sns.heatmap(mean_df, annot=annot, fmt="", vmin=vmin, vmax=vmax, center=center,
                     cmap=cmap, annot_kws={"size": 24})
    hm.figure.axes[-1].tick_params(labelsize=28)      # colorbar ticks
    hm.tick_params(axis="x", labelsize=32)
    hm.tick_params(axis="y", labelsize=32)
    plt.xlabel("Token Position", fontsize=40)
    plt.ylabel("Layer", fontsize=40)
    plt.title(title, fontsize=38, pad=20)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_endogenous_heatmaps(df: pd.DataFrame, out_dir: str):
    """One heatmap per (eval, metric): layer x position, annotated mean±SE over the
    cell's neurons."""
    os.makedirs(out_dir, exist_ok=True)
    summary = summarize_endogenous_sweep(df)
    summary.to_csv(f"{out_dir}/sweep_summary.csv", index=False)
    # One colour scale per metric, computed across BOTH eval types (endogenous +
    # interventional), so e.g. the two l2_dist heatmaps share a colorbar and are
    # comparable side by side. corr-like metrics keep the fixed [0, 1] scale.
    scale = {}
    for metric, msub in summary.groupby("metric"):
        if metric in RANGE01:
            scale[metric] = (0, 1)
        else:
            vals = msub["mean"].dropna()
            scale[metric] = (float(vals.min()), float(vals.max())) if len(vals) else (None, None)
    quantity = {"corr": "Correlation", "l2_dist": "L2"}
    for (eval_name, metric), sub in summary.groupby(["eval_type", "metric"]):
        mean_df = sub.pivot(index="layer", columns="position", values="mean")
        se_df = sub.pivot(index="layer", columns="position", values="se")
        vmin, vmax = scale[metric]
        # these heatmaps annotate ± SE (se_df), so the title says SE, not Std
        title = (f"Neuron {eval_name.capitalize()} {quantity.get(metric, metric)}\n"
                 f"(Mean ± SE over Neurons)")
        path = f"{out_dir}/heatmap_{eval_name}_{metric}.png"
        draw_layer_position_heatmap(mean_df, se_df, title, path, vmin=vmin, vmax=vmax)
        print(f"[endo] wrote {path}")


def plot_endogenous_lines(df: pd.DataFrame, out_dir: str, hue: str = "layer"):
    """Position on x, metric on y, one line per `hue`.

    hue="layer"  -> mean over the cell's 10 neurons, shaded band = standard error
                    across neurons (heterogeneity, not seed noise).
    hue="neuron" -> every neuron as its own line; no aggregation, so the spaghetti
                    shows whether the layer mean hides a split population. Only
                    readable one layer at a time, so this writes one file per layer.
    """
    os.makedirs(out_dir, exist_ok=True)
    panels = sorted(df["panel"].unique())

    def _draw(data, title, path, hue_col, errorbar):
        levels = sorted(data[hue_col].unique())
        data = data.copy()
        data[hue_col] = pd.Categorical(data[hue_col], categories=levels, ordered=True)
        palette = dict(zip(levels, sns.color_palette("viridis", len(levels))))
        g = sns.relplot(
            data=data, x="position", y="value", hue=hue_col, palette=palette,
            col="panel", col_order=panels, col_wrap=min(2, len(panels)),
            kind="line", marker="o", errorbar=errorbar,
            facet_kws={"sharey": False}, height=4.5, aspect=1.4, legend="brief",
        )
        g.set_titles("{col_name}")
        g.set_axis_labels("Token Position", "")
        if g.legend is not None:
            g.legend.set_title(hue_col.capitalize())
        for ax in g.axes.flat:
            if any(m in ax.get_title() for m in RANGE01):
                ax.set_ylim(-0.05, 1.02)
                ax.axhline(0, color="grey", ls="--", lw=1)
            ax.grid(alpha=0.2)
        g.figure.suptitle(title, y=1.03)
        g.tight_layout()
        g.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(g.figure)
        print(f"[endo] wrote {path}")

    if hue == "layer":
        _draw(df, "endogenous sweep: metric vs token position (mean±SE over 10 neurons)",
              f"{out_dir}/lines_by_layer.png", "layer", "se")
    elif hue == "neuron":
        for layer, sub in df.groupby("layer"):
            _draw(sub, f"layer {layer}: per-neuron lines (no aggregation)",
                  f"{out_dir}/lines_by_neuron_l{layer}.png", "neuron", None)
    else:
        raise ValueError(f"hue must be 'layer' or 'neuron', got {hue!r}")


def plot_gate_comparison(df_by_gate: dict, out_dir: str, expected_neurons: int = 10):
    """Gate ON vs OFF side by side, same axes, one line per layer.

    This is the panel the gate claim rests on: with the gate ON no trainable parameter
    touches the computation at the introspected token, so the model cannot read the
    neuron and interventional corr sits at 0 however good the endogenous score looks --
    the bypass signature. With it OFF the same objective produces causal reading.
    Shared axes make that a visual comparison rather than a cross-table one."""
    gates = [g for g in ("on", "off") if g in df_by_gate]
    if len(gates) < 2:
        print(f"[endo] only gate={gates} present; skipping the side-by-side comparison")
        return
    os.makedirs(out_dir, exist_ok=True)
    panels = sorted(set().union(*(set(df_by_gate[g]["panel"]) for g in gates)))
    for panel in panels:
        fig, axes = plt.subplots(1, len(gates), figsize=(7 * len(gates), 5.2),
                                 squeeze=False, sharey=True)
        layers = sorted(set().union(*(set(df_by_gate[g]["layer"]) for g in gates)))
        palette = dict(zip(layers, sns.color_palette("viridis", len(layers))))
        for ax, gate in zip(axes.flat, gates):
            sub = df_by_gate[gate]
            sub = sub[sub["panel"] == panel]
            if sub.empty:
                ax.set_visible(False)
                continue
            sns.lineplot(data=sub, x="position", y="value", hue="layer", palette=palette,
                         marker="o", errorbar="se", ax=ax, legend=(ax is axes.flat[-1]))
            ax.set_title(f"LoRA Gate-{gate.capitalize()}")
            ax.set_xlabel("Token Position")
            ax.set_ylabel(panel.title())
            if "corr" in panel:
                ax.set_ylim(-0.1, 1.05)
                ax.axhline(0, color="grey", ls="--", lw=1)
            ax.grid(alpha=0.25)
        # Park the legend outside the right-hand panel. seaborn's default "best" placement
        # sits inside the axes and lands on the data whenever the curves span the plot,
        # which on these panels is always. bbox_inches="tight" at save time keeps it in.
        last = axes.flat[-1]
        if last.get_visible() and last.get_legend() is not None:
            sns.move_legend(last, "center left", bbox_to_anchor=(1.02, 0.5),
                            title="Layer", frameon=False)
        fig.tight_layout()
        path = Path(out_dir) / f"gate_comparison_{panel.replace(' ', '_')}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[endo] wrote {path}")


def run_endogenous_sweep_plots(exp_group: str, out_root: str, gates=("on", "off"),
                               expected_neurons: int = 10):
    """Plot each gate arm into its OWN subdirectory (gate_on/, gate_off/), then the
    side-by-side comparison at the root. A missing arm is reported and skipped rather
    than aborting the other one."""
    loaded = {}
    for gate in gates:
        out_dir = f"{out_root}/gate_{gate}"
        print(f"\n[endo] ===== gate={gate} -> {out_dir} =====")
        try:
            df = load_endogenous_sweep(exp_group, gate=gate)
        except SystemExit as e:
            print(f"[endo] skipping gate={gate}: {e}")
            continue
        os.makedirs(out_dir, exist_ok=True)
        report_endogenous_coverage(df, expected_neurons=expected_neurons)
        df.to_csv(f"{out_dir}/sweep_runs.csv", index=False)      # per-run long table
        plot_endogenous_heatmaps(df, out_dir)
        plot_endogenous_lines(df, out_dir, hue="layer")
        plot_endogenous_lines(df, out_dir, hue="neuron")
        loaded[gate] = df
    if loaded:
        pd.concat([d.assign(gate_arm=g) for g, d in loaded.items()], ignore_index=True) \
          .to_csv(f"{out_root}/sweep_runs_both_gates.csv", index=False)
    plot_gate_comparison(loaded, out_root, expected_neurons=expected_neurons)
    return loaded


if __name__ == "__main__":
    # ---- older sweeps ---------------------------------------------------
    # plot_token_overlap_vs_test_metrics("single_mlp_l10_n24_train")
    # plot_sweep_eval_metrics("single_mlp_sweep")
    # correlate_sweep_metrics_with_data_stats("single_mlp_sweep")
    # plot_sweep_eval_metrics_intervention_train("intervention_train_regressor_sweep_layer_token_old_fixed_scale10")


    # ---- endogenous (W=1.0) layer x token sweep -------------------------
    EXP_GROUP = "mlp_endo_sweep_layer_token_llama323b" # "mlp_endo_sweep_layer_token"
    OUT_DIR = f"{ROOT}/outputs/plots/{EXP_GROUP}"

    os.makedirs(OUT_DIR, exist_ok=True)
    # both arms: each into its own subdir, plus the gate on/off comparison at the root
    run_endogenous_sweep_plots(EXP_GROUP, OUT_DIR, gates=("on", "off"))

