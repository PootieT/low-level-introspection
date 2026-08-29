import os
import pdb
import glob
import json
from pathlib import Path
import shutil
import argparse

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument("--expected-files", type=str, required=True,
                        help="comma separated list of files needed before deleting checkpoints, "
                             "if not all of them are present, checkpoints will not be deleted (experiment incomplete).")
    return parser.parse_args()

def delete_files_matching_pattern(exp_pattern:str):
    files = glob.glob(exp_pattern)
    print(f"Found {len(files)} files")
    for f in tqdm(files):
        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f)
    print(f"Deleted {len(files)} files")

def main():
    args = parse_args()
    config = json.loads(args.config.read_text())
    files = args.expected_files.split(',')
    exp_dir = Path(config['output_dir'])/ config['experiment_name']
    for f in files:
        if not os.path.exists(exp_dir / f):
            print(f"File {f} does not exist, skipping checkpoint deletion")
            exit(1)
    print(f"all expected files are present in {exp_dir}, deleting checkpoint files")
    delete_files_matching_pattern(f"{exp_dir}/checkpoint-*/adapter_model.safetensors")
    delete_files_matching_pattern(f"{exp_dir}/checkpoint-*/optimizer.pt")

if __name__== "__main__":
    main()


