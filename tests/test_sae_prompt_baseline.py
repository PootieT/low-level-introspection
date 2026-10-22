"""Unit tests for the SAE prompted baseline (evals/score_sae_prompt_baseline.py)."""
import numpy as np
import pytest
import torch

from src.introspector.verbal_lexicon import parse_continuation
from evals.score_sae_prompt_baseline import (
    ACTIVE, INACTIVE, NEITHER, PrefillOnly, best_ba_threshold, hard_metrics, scored_metrics,
)


@pytest.mark.parametrize("text,cls", [
    # continuations the untrained gemma-2-2b actually wrote
    (" not supported.\n\nI'm not", None),
    (" not a valid token.\nThe SAE", None),
    (" 1.\nThe woman who died", "active"),
    (" 100% and the other", "active"),
    (" 1000000", "active"),
    (" 0.00000", "inactive"),
    (" the \"7\" bit set.\n", None),
    (" a bit of a misnomer.", None),
    (" enabled", "active"),
    # negation of a state word flips it; anything else after a negator is neither
    (" not active", "inactive"),
    (" not firing at all", "inactive"),
    (" isn't active here", "inactive"),
    (" not inactive", "active"),
    (" Active.", "active"),
    (" inactive", "inactive"),
    (" no", "inactive"),
    ("", None),
    (" \n\n", None),
])
def test_parse_continuation(text, cls):
    assert parse_continuation(text)[0] == cls


def test_neither_is_a_miss_on_both_classes():
    y = np.array([1, 1, 0, 0])
    m = hard_metrics(np.array([ACTIVE, NEITHER, INACTIVE, NEITHER]), y)
    assert m["tpr"] == 0.5 and m["tnr"] == 0.5 and m["balanced_accuracy"] == 0.5
    assert m["neither_rate"] == 0.5 and m["precision"] == 1.0


def test_best_ba_threshold_separates_and_transfers():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(50, int), np.zeros(950, int)]
    s = np.r_[rng.normal(2, 0.5, 50), rng.normal(0, 0.5, 950)]
    thr = best_ba_threshold(s, y)
    assert 0.3 < thr < 1.8
    m = scored_metrics(s, y, thr)
    assert m["balanced_accuracy"] > 0.95 and m["auroc"] > 0.99


def test_prefill_only_leaves_decode_steps_alone():
    calls = []

    def hook(module, inputs, output):
        calls.append(output.shape)
        return output + 1

    h = PrefillOnly(hook)
    prefill, decode = torch.zeros(2, 7, 4), torch.zeros(2, 1, 4)
    assert torch.equal(h(None, None, prefill), prefill + 1)
    assert h(None, None, decode) is decode
    tup = (torch.zeros(2, 1, 4), "cache")
    assert h(None, None, tup) is tup
    assert calls == [prefill.shape]
