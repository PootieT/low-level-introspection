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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_dir(exp_dir):
    neuron = exp_dir.split("_")[3]
    layer, index = neuron.replace("N","").split(".")
    layer, index = int(layer), int(index)
    bin = int(exp_dir.split("_")[4].replace("Bin",""))
    return {"neuron": neuron, "layer": layer, "neuron_index": index, "bin": bin}


REDUNDANCY_CSV = "outputs/circuit/intervention_train_regressor_extreme/intervention_train_reg_extreme10_300K/candidate_neurons.csv"


def analyze_correlation(exp_group_path: str, redundancy_csv: str = REDUNDANCY_CSV):
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

    # 2) scatterplot vs actual redundancy R2, hue = eval type
    red = pd.read_csv(redundancy_csv).rename(columns={"neuron": "neuron_index"})
    merged = long_df.merge(red[["neuron_index", "R2"]], on="neuron_index", how="left")
    plt.figure()
    sns.scatterplot(x="R2", y="corr", hue="eval_type", data=merged)
    plt.xlabel("same-layer redundancy R2"); plt.ylabel("test correlation")
    plt.title("test correlation vs neuron redundancy R2")
    plt.tight_layout()
    plt.savefig("outputs/plots/scatter_test_correlation_vs_neuron_redundancy_R2.png")
    plt.close()
    return


if __name__ == "__main__":
    analyze_correlation("outputs/runs/intervention_train_regressor_redundancy")