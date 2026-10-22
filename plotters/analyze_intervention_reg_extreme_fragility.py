"""
Intervention training with regressor extreme range is quite fragile

Investigate if training signal stats has anything to do with success

Findings:
- not really
- only thing significant is position of token, first token much easier
  to train.
- probably need annealing where begining of training we use large scale
  and aneal to smaller range, because large range also yield bad test
  corr even if it has high train corr.
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

def parse_exp_dir_extreme(exp_dir):
    layer = int(exp_dir.split("_")[0].replace("l", ""))
    pos = int(exp_dir.split("_")[1].replace("pos", ""))
    sample = int(exp_dir.split("_")[2].replace("sample", ""))
    return {"layer": layer, "pos": pos, "sample": sample}

def parse_exp_dir_anneal_hyper(exp_dir):
    neuron = exp_dir.split("_")[3]
    dist = exp_dir.split("_")[4]
    scale = int(exp_dir.split("_")[5].replace("scale", "").split("-")[0])
    epoch = float(exp_dir.split("_")[6].replace("epoch", "").split("-")[1])
    return {"neuron": neuron, "dist": dist, "scale_start": scale, "epoch_end": epoch}

def analyze_reg_extreme_hyper(exp_group_path: str):
    exp_group_name = exp_group_path.split("/")[-1]

    # load test_pred results
    all_res = []
    for p in tqdm(glob.glob(os.path.join(exp_group_path, "*/test_results.json"))):
        with open(p) as f:
            res = json.load(f)
        exp_name = p.split("/")[-2]
        res.update(parse_exp_dir_extreme(exp_name))

        # load dataset
        data_stat_path = f"data/{exp_group_name}/{exp_name}/dataset_signal_stats.json"
        with open(data_stat_path) as f:
            stats = json.load(f)
        train_stats = {f"train_{k}": v for k, v in stats["train"][0].items()}
        res.update(train_stats)
        all_res.append(res)

    df = pd.DataFrame.from_records(all_res)
    print(df[["corr", "layer", "pos", "sample", "train_mean", "train_std", "train_50%", "train_min", "train_max"]].corr())
    #                corr     layer       pos    sample  train_mean  train_std  train_50%  train_min  train_max
    # corr        1.000000  0.020668  0.503358  0.001058    0.031783   0.102548   0.014620  -0.024085   0.082430
    sns.lineplot(x="pos", y="corr", data=df)
    plt.savefig("plots/intervention_train_extreme_reg_corr_vs_pos.png")
    return


def analyze_anneal_hyper(exp_group_path: str):
    exp_group_name = exp_group_path.split("/")[-1]
    all_res = []
    for p in tqdm(glob.glob(os.path.join(exp_group_path, "*/interventional_test_metrics_gaussian.json"))):
        with open(p) as f:
            res = json.load(f)["test"]
        exp_name = p.split("/")[-2]
        res = {f"test_intervention_{k}": v for k, v in res.items()}
        res.update(parse_exp_dir_anneal_hyper(exp_name))

        # load dataset
        alt_eval_path = f"{Path(p).parent}/test_alt_intervention_results.json"
        with open(alt_eval_path) as f:
            stats = json.load(f)
        eval_res = {f"test_{k}": v for k, v in stats.items()}
        res.update(eval_res)
        all_res.append(res)
    df = pd.DataFrame.from_records(all_res)
    pdb.set_trace()
    return

if __name__ == "__main__":
    analyze_reg_extreme_hyper("outputs/runs/intervention_train_regressor_sweep_layer_token")
    # analyze_anneal_hyper("outputs/runs/intervention_train_regressor_anneal_hyper")