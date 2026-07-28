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


def analyze_correlation(exp_group_path: str):
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
    pdb.set_trace()
    sns.boxplot(x="bin", y="test_corr", data=df)
    plt.savefig("outputs/plots/box_non_intervention_test_correlation_vs_neuron_redundancy_bin.png")
    plt.close()
    sns.boxplot(x="bin", y="test_intervention_corr", data=df)
    plt.savefig("outputs/plots/box_intervention_test_correlation_vs_neuron_redundancy_bin.png")
    plt.close()
    return

if __name__ == "__main__":
    analyze_correlation("outputs/runs/intervention_train_regressor_redundancy")