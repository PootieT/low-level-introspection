"""data_prep/: every script loads the way scripts/ runs it, a generated experiment config is
one the trainers can read back, and the layer-generalization index picker ranks by the
weakest layer.
"""
import subprocess
import sys

import numpy as np
import pytest
from datasets import Dataset

from script_smoke import REPO_ROOT, run_smoke, script_env, scripts_in, smoke_failure

DATA_PREP = scripts_in("data_prep")
MODEL = "meta-llama/Llama-3.2-1B"


@pytest.fixture(scope="module")
def smoke():
    return run_smoke(DATA_PREP)


@pytest.mark.parametrize("path", DATA_PREP, ids=lambda p: p.name)
def test_script_loads(smoke, path):
    proc = smoke[path]
    assert proc.returncode == 0, smoke_failure(proc)


def test_generated_config_round_trips(tmp_path):
    """The docstring's example call, written to tmp and read back the way trainers/ does."""
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(MODEL, local_files_only=True)
    except Exception:
        pytest.skip(f"{MODEL} config not in the local HF cache")
    from src.introspector.config import ExperimentConfig

    out = tmp_path / "mlp_l10_n200.json"
    proc = subprocess.run(
        [sys.executable, "data_prep/generate_experiment_config.py",
         "--experiment-name", "mlp_l10_n200", "--model", MODEL,
         "--train-dataset", "allenai/c4", "--train-dataset-split", "train[:1000]",
         "--test-dataset", "allenai/c4", "--test-dataset-split", "test[:1000]",
         "--train-signals", "N10.1", "--test-signals", "N10.1", "--precision", "3",
         "--train-token-selector", "-1", "--test-token-selector", "-1",
         "--output-dir", str(tmp_path / "runs"), "--output", str(out)],
        cwd=REPO_ROOT, env=script_env(), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    cfg = ExperimentConfig.model_validate_json(out.read_text())
    assert (cfg.experiment_name, cfg.model_name) == ("mlp_l10_n200", MODEL)
    assert cfg.dataset_name                              # derived, so caches are shared
    assert str(cfg.dataset_cache_dir).endswith(cfg.dataset_name)


def test_layer_gen_index_prefers_the_index_with_no_dead_layer():
    from data_prep.select_layer_gen_index import rank_candidates

    rng = np.random.default_rng(0)
    rows = []
    for layer in range(4):
        for index, std in ((7, 1.0), (8, 0.5)):
            if index == 7 and layer == 2:
                std = 0.0                                # strong elsewhere, dead at layer 2
            for a in rng.normal(0, std, 50) if std else np.zeros(50):
                rows.append({"signal_layer": layer, "signal_index": index, "activation": a})
    rank, per = rank_candidates(Dataset.from_list(rows), min_std=1e-3)
    assert list(rank["index"]) == [8, 7]
    assert rank.set_index("index").loc[7, "n_dead"] == 1
    assert rank.set_index("index").loc[7, "weakest_layer"] == 2
    assert len(per) == 8
