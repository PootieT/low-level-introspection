"""plotters/: every script loads the way it is run, and the shared readers parse the run
directories the sweeps write.

The fixtures are synthetic run trees in the layout trainers/train_introspector_joint.py writes
(one directory per run, name = the cell, metrics JSONs inside), so these run without a
synced outputs/ tree.
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from script_smoke import REPO_ROOT, run_smoke, script_env, scripts_in, smoke_failure

PLOTTERS = scripts_in("plotters")


@pytest.fixture(scope="module")
def smoke():
    return run_smoke(PLOTTERS)


@pytest.mark.parametrize("path", PLOTTERS, ids=lambda p: p.name)
def test_script_loads(smoke, path):
    proc = smoke[path]
    assert proc.returncode == 0, smoke_failure(proc)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_run_catalog_parses_grid_runs(tmp_path, monkeypatch):
    from plotters import run_catalog

    group = tmp_path / "mlp_joint_grid_layer_token"
    kept = group / "joint_N12.2721_POS-28_gateoff_scale10_kl0.1_w0.01_300K"
    _write(kept / "endogenous_test_metrics_endogenous.json", {"corr": 0.8, "corr_macro": 0.7})
    _write(kept / "interventional_test_metrics_gaussian.json", {"test": {"corr": 0.6}})
    _write(kept / "lm_degradation.json", {"negligible": True, "lengths": [
        {"length": 64, "kl_mean": 0.01, "kl_p95": 0.05, "top1_agreement": 0.99}]})
    _write(group / "joint_N2.1447_POS-1_gateon_scale10_w0.01_300K" / "x.json", {})
    (group / "not_a_run").mkdir()
    monkeypatch.setattr(run_catalog, "RUNS", tmp_path)

    df = run_catalog.grid_runs("mlp_joint_grid_layer_token", gate="off")
    assert list(df["run"]) == [kept.name]          # the gate-on run and the stray dir drop
    row = df.iloc[0]
    assert (row["layer"], row["index"], row["position"]) == (12, 2721, -28)
    assert row["kl_tag"] == "0.1" and row["w"] == 0.01
    assert row["endogenous_corr"] == 0.8 and row["endogenous_corr_macro"] == 0.7
    assert row["interventional_corr"] == 0.6       # read from under "test"
    assert row["kl@64"] == 0.01 and bool(row["has_kl"]) and not row["has_drift"]

    assert run_catalog.grid_runs("no_such_group").empty


def _attention_sweep(root, layers=(0, 1), offsets=(-1, -3, -7), gates=("on", "off")):
    rng = np.random.default_rng(0)
    for layer in layers:
        for src in offsets:
            for gate in gates:
                run = root / f"endo_A{layer}.5@{src}_POS-1_gate{gate}_w1.0_300K"
                _write(run / "endogenous_test_metrics_endogenous.json",
                       {"corr": float(rng.uniform(0.3, 0.9)), "l2_dist": 0.1})
                _write(run / "interventional_test_metrics_gaussian.json",
                       {"distribution": "gaussian", "test": {"corr": 0.5}})
                pd.DataFrame({"predictions": rng.normal(size=50),
                              "activation": rng.normal(size=50)}).to_csv(
                    run / "endogenous_test_predictions_endogenous.csv", index=False)
    (root / "unrelated_dir").mkdir()


def test_attention_collect_flattens_nested_interventional(tmp_path):
    from plotters.plot_attention_srctoken_sweep import collect

    _attention_sweep(tmp_path)
    df = collect(tmp_path)
    assert len(df) == 12
    assert set(df["arm"]) == {"gateon/none", "gateoff/none"}   # no _kl tag -> "none"
    assert (df["interventional_corr"] == 0.5).all()
    assert "test" not in df.columns
    assert df["target_std"].gt(0).all()


def test_attention_sweep_plotter_end_to_end(tmp_path):
    """The full script on a synthetic sweep: summary table plus figures, exit 0."""
    sweep, out = tmp_path / "attn_sweep", tmp_path / "plots"
    _attention_sweep(sweep)
    proc = subprocess.run([sys.executable, "plotters/plot_attention_srctoken_sweep.py",
                           str(sweep), "--out", str(out)], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=300,
                          env=script_env())
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert len(pd.read_csv(out / "summary.csv")) == 12
    assert list(out.glob("*.png")) or list(out.glob("*.pdf"))


def test_injection_regime_tables_bold_only_significant_joint_cells(tmp_path):
    """plotters/plot_injection_regimes.py: Holm adjustment, the paired joint-vs-SFT test
    over neurons, and that only significant JOINT cells come out bold."""
    from plotters.plot_injection_regimes import holm, paired_tests, summarize, write_markdown

    adj = holm([0.01, 0.04, np.nan, 0.03])              # m = 3: NaN does not count
    assert np.allclose(adj[[0, 3, 1]], [0.03, 0.06, 0.06]) and np.isnan(adj[2])

    corr = {("joint", "endogenous"): [0.97, 0.99, 0.98, 0.96, 0.99],
            ("sft", "endogenous"): [0.98, 0.97, 0.99, 0.98, 0.97],   # no systematic gap
            ("joint", "x1"): [0.95, 0.97, 0.93, 0.96, 0.94],
            ("sft", "x1"): [0.20, 0.25, 0.15, 0.22, 0.18],          # large, consistent gap
            ("joint", "local@1s"): [0.0] * 5, ("sft", "local@1s"): [0.0] * 5}  # untestable
    df = pd.DataFrame([{"method": m, "arm": a, "layer": 2, "t": 31, "index": i, "corr": r}
                       for (m, a), rs in corr.items() for i, r in enumerate(rs)])
    tests = paired_tests(df).set_index("arm")
    assert tests.loc["x1", "significant"] and not tests.loc["endogenous", "significant"]
    assert np.isnan(tests.loc["local@1s", "p"]) and not tests.loc["local@1s", "significant"]
    assert tests.attrs["n_tests"] == 2

    md = write_markdown(summarize(df), tmp_path, paired_tests(df))
    rows = {line.split("|")[1].strip(): line for line in md.splitlines() if line.startswith("| ")}
    assert rows["Gaussian 1×"].count("**") == 2                  # joint bold, SFT not
    assert rows["Gaussian 1×"].split("|")[2].strip().startswith("**0.95")
    assert "**" not in rows["Endogenous (no injection)"] and "**" not in rows["Local ±1σ"]
    assert "Holm-corrected over its 2 comparisons" in md


def test_injection_regime_pooled_table_pairs_runs_across_locales(tmp_path):
    """The pooled comparison pairs each (layer, neuron, token) run across methods over
    EVERY locale, and its table comes first with its own bolding."""
    from plotters.plot_injection_regimes import (
        POOLED, paired_tests, pooled_title, summarize, write_markdown)

    rng = np.random.default_rng(0)
    rows = []
    for layer in (2, 8):
        for t in (4, 31):
            for i in range(3):           # neuron i is shared by both token positions
                for m, a, r in (("joint", "x1", 0.95), ("sft", "x1", 0.4),
                                ("joint", "endogenous", 0.98), ("sft", "endogenous", 0.98)):
                    rows.append({"method": m, "arm": a, "layer": layer, "t": t,
                                 "index": i, "corr": r + rng.normal(0, 0.02)})
    df = pd.DataFrame(rows)
    pt = paired_tests(df, by=POOLED).set_index("arm")
    assert (pt["n_pairs"] == 12).all()                         # 2 layers x 2 tokens x 3
    assert pt.loc["x1", "significant"] and not pt.loc["endogenous", "significant"]
    title = pooled_title(df)
    assert "All 4 locales pooled" in title and "12 runs per method = 6 neurons × 2 token" in title

    md = write_markdown(summarize(df), tmp_path, paired_tests(df),
                        pooled=summarize(df, by=POOLED), pooled_tests=pt.reset_index(),
                        pooled_title=title)
    first = md.split("\n\n")[0]
    assert first == title                                      # pooled table leads
    pooled_block = md.split("**Layer")[0]
    assert "| Gaussian 1× | **" in pooled_block
    assert "Pooled table, pairs = the same (layer, neuron, token) run" in md
