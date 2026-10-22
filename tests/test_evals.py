"""evals/: every eval script loads the way scripts/ runs it, the default eval metrics carry
the rank measures (Spearman, Kendall tau, monotone R^2) next to Pearson, and the
length / LM-degradation summaries split at the training window correctly.

The prompted baselines have their own files: test_prompt_baseline_probe.py and
test_sae_prompt_baseline.py.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from script_smoke import run_smoke, scripts_in, smoke_failure

EVALS = scripts_in("evals")


@pytest.fixture(scope="module")
def smoke():
    return run_smoke(EVALS)


@pytest.mark.parametrize("path", EVALS, ids=lambda p: p.name)
def test_script_loads(smoke, path):
    proc = smoke[path]
    assert proc.returncode == 0, smoke_failure(proc)


def test_default_metrics_include_rank_measures():
    """Every eval goes through compute_metrics_from_numericals, so this is what lands in
    each *_test_metrics*.json."""
    from src.introspector.training.metrics import compute_metrics_from_numericals

    rng = np.random.default_rng(0)
    act = rng.normal(size=1000)
    m = compute_metrics_from_numericals(act + 0.2 * rng.normal(size=1000), act)
    for k in ("corr", "spearman", "kendall_tau", "monotone_r2", "pearson_minus_spearman",
              "pred_out_of_range_frac", "target_unique_frac"):
        assert np.isfinite(m[k]), k
    assert m["spearman"] > 0.9 and m["monotone_r2"] > 0.8 and abs(m["pearson_minus_spearman"]) < 0.05

    # A handful of predictions far outside the target's range: Pearson collapses, the
    # rank measure does not, and the out-of-range fraction says why.
    pred = act.copy()
    pred[:10] = 1e3
    m = compute_metrics_from_numericals(pred, act)
    assert m["spearman"] > 0.95 and m["corr"] < 0.5
    assert m["pred_out_of_range_frac"] == pytest.approx(0.01)


def test_length_bins_respect_the_training_window():
    from evals.score_length_generalization import DISTANCE_BINS, _bin_by_distance

    rng = np.random.default_rng(0)
    d = np.repeat(np.arange(0, max(DISTANCE_BINS) + 1), 20)   # no bin too thin for a corr
    act = rng.normal(size=len(d))
    df = pd.DataFrame({"distance_from_end": d, "activation": act,
                       "predictions": act + 0.1 * rng.normal(size=len(d))})
    out = _bin_by_distance(df, "predictions", "activation", train_window=DISTANCE_BINS[1])
    assert out["n"].sum() == len(d)
    first, second = out.iloc[0], out.iloc[1]
    assert bool(first["in_train_range"]) and bool(second["in_train_range"])
    assert not out.iloc[2:]["in_train_range"].any()
    assert not out["straddles_window"].any()     # the window sits on a bin edge
    assert (out["corr"] > 0.9).all()
    assert _bin_by_distance(df.drop(columns="distance_from_end"), "predictions",
                            "activation", 32).empty


def test_lm_degradation_summary_splits_at_train_length():
    from evals.score_lm_degradation import summarize

    L = 64
    kl = torch.cat([torch.full((31,), 0.01), torch.full((L - 1 - 31,), 0.5)])
    per_pos = {"kl": kl, "nll_on": torch.full((L - 1,), 2.0),
               "nll_off": torch.full((L - 1,), 2.0), "top1_same": torch.ones(L - 1),
               "n_seq": 8}
    row = summarize(per_pos, L, train_len=32)
    assert row["kl_mean_within_trainlen"] == pytest.approx(0.01)
    assert row["kl_mean_beyond_trainlen"] == pytest.approx(0.5)
    assert row["nll_delta"] == 0 and row["ppl_base"] == pytest.approx(np.exp(2.0))
    assert "kl_mean_within_trainlen" not in summarize(per_pos, 32, train_len=32)
