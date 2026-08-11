import os
import json
import glob
import pdb
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).parents[1]

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
    for metric in tqdm(["test_corr", "test_l2_dist", "test_pred_corr", "test_pred_l2_dist", "intervention_corr", "intervention_l2_dist"]):
        df[metric] = df[metric] * 100
        pivot_df = df.pivot_table(index="layer", columns="token_pos", values=metric, aggfunc="mean")
        pivot_df_std = df.pivot_table(index="layer", columns="token_pos", values=metric, aggfunc="std")

        annot_df = pd.DataFrame()
        for col in pivot_df.columns:
            annot_df[col] = pivot_df[col].map(lambda x: f"{x:.1f}") + "±" + \
                            pivot_df_std[col].map(lambda x: f"{x:.1f}")

        plt.figure(figsize=[14,14])
        sns.heatmap(
            pivot_df,
            annot=annot_df,  # Pass your custom annotation DataFrame here
            fmt="",
            vmin=0, vmax=100
        )
        plt.title(f"{metric}(X100) across layers and tokens")
        plt.tight_layout()
        plot_dir = f"{ROOT}/outputs/plots/single_neuron"
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/sweep_eval_metrics_{metric}.png")
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
    for metric in tqdm(["test_corr", "test_l2_dist", "intervention_corr", "intervention_l2_dist"]):
        df[metric] = df[metric] * 100
        pivot_df = df.pivot_table(index="layer", columns="token_pos", values=metric, aggfunc="mean")
        pivot_df_std = df.pivot_table(index="layer", columns="token_pos", values=metric, aggfunc="std")

        annot_df = pd.DataFrame()
        for col in pivot_df.columns:
            annot_df[col] = pivot_df[col].map(lambda x: f"{x:.1f}") + "±" + \
                            pivot_df_std[col].map(lambda x: f"{x:.1f}")

        plt.figure(figsize=[14,14])
        sns.heatmap(
            pivot_df,
            annot=annot_df,  # Pass your custom annotation DataFrame here
            fmt="",
            vmin=0, vmax=100
        )
        plt.title(f"{metric}(X100) across layers and tokens")
        plt.tight_layout()
        plot_dir = f"{ROOT}/outputs/plots/{output_dir}"
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/sweep_eval_metrics_{metric}.png")
        plt.close()


if __name__ == "__main__":
    # plot_token_overlap_vs_test_metrics("single_mlp_l10_n24_train")
    # plot_sweep_eval_metrics("single_mlp_sweep")
    # correlate_sweep_metrics_with_data_stats("single_mlp_sweep")
    plot_sweep_eval_metrics_intervention_train("intervention_train_regressor_sweep_layer_token_old_fixed_scale10")

