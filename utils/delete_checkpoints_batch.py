import os
import pdb
import glob
import shutil

from tqdm import tqdm

def delete_files_matching_pattern(exp_pattern:str):
    files = glob.glob(exp_pattern)
    print(f"Found {len(files)} files")
    if len(files) == 0:
        print("No files found")
        return
    res = input("Sure you want to delete? (Y/N)")
    if res.lower().startswith("y"):
        for f in tqdm(files):
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f)

if __name__== "__main__":
    # delete_files_matching_pattern("outputs/runs/single_sae_detection_sweep_layer_pos/*/checkpoint-*/adapter_model.safetensors")
    delete_files_matching_pattern("outputs/runs/mlp_endo_sweep_layer_token/*/checkpoint-*/adapter_model.safetensors")
    delete_files_matching_pattern("outputs/runs/mlp_endo_sweep_layer_token/*/checkpoint-*/optimizer.pt")
    delete_files_matching_pattern("outputs/runs/mlp_endo_sweep_layer_token/*/checkpoint-*/tokenizer.json")


