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


def test_cross_model_grid_pairs_columns_and_marks_significant_cross_cells(tmp_path):
    """plotters/plot_cross_model_prediction.py: self cells come from each model's own SFT
    grid for exactly the cross runs' signals, every column is paired, and only cross cells
    that differ significantly from their column's self run are bold."""
    from plotters import plot_cross_model_prediction as pc

    models = ["llama1b", "qwen06b"]
    sigs = {"llama1b": [(2, i, -1) for i in range(5)], "qwen06b": [(4, i, -28) for i in range(5)]}
    rng = np.random.default_rng(0)
    for src, ss in sigs.items():
        for k, (L, i, pos) in enumerate(ss):
            base = 0.90 + 0.01 * k
            # llama1b's neurons: self clearly better; qwen06b's: no gap
            runs = {src: base, [m for m in models if m != src][0]:
                    base - (0.30 if src == "llama1b" else 0.0) + 0.01 * rng.normal()}
            for pred, r in runs.items():
                d = (tmp_path / pc.MODELS[src][1] / f"endo_N{L}.{i}_POS{pos}_gateoff_w1.0_300K"
                     if pred == src else tmp_path / pc.CROSS_GROUP /
                     f"cross_{pred}_from_{src}_N{L}.{i}_POS{pos}_gateoff_w1.0_300K")
                _write(d / "endogenous_test_metrics_endogenous.json", {"corr": r, "spearman": r})
    # a self run for a signal no cross run covers must not enter the grid
    _write(tmp_path / pc.MODELS["llama1b"][1] / "endo_N8.9_POS-1_gateoff_w1.0_300K"
           / "endogenous_test_metrics_endogenous.json", {"corr": 0.1, "spearman": 0.1})

    df = pc.complete_columns(pc.collect(tmp_path, models), models)
    assert len(df) == 20 and set(df.groupby("source").size()) == {10}
    tests = pc.paired_tests(df).set_index(["source", "predictor", "metric"])
    assert tests.loc[("llama1b", "qwen06b", "pearson"), "significant"]
    assert not tests.loc[("qwen06b", "llama1b", "pearson"), "significant"]
    md = pc.write_markdown(pc.summarize(df), models, tmp_path, pc.paired_tests(df))
    qwen_row = next(l for l in md.splitlines() if l.startswith("| Qwen3-0.6B"))
    llama_row = next(l for l in md.splitlines() if l.startswith("| Llama-3.2-1B"))
    assert qwen_row.split("|")[2].strip().startswith("**")       # cross on llama: bold
    assert "(self)" in llama_row.split("|")[2] and "**" not in llama_row
    pc.plot(pc.summarize(df), models, tmp_path, pc.paired_tests(df))
    assert (tmp_path / "cross_model_heatmap.png").exists()


def test_fig3_significance_summary_route_matches_raw_tests():
    """plotters/stats_sft_vs_joint.py: the attention panel is tested from printed
    (mean, SE) and pooled across cells by the law of total variance, so that route must
    give exactly the paired t-test the raw neuron/SAE units get."""
    from scipy.stats import ttest_1samp
    from plotters.stats_sft_vs_joint import (
        t_test, summarize_units, pool_cells, holm, attention_cells, ATTN_HEADS)

    rng = np.random.default_rng(0)
    units = [rng.normal(rng.normal(), 0.3, 3) for _ in range(5)]
    cells = pd.DataFrame([{"n": 3, "mean": u.mean(), "sd": u.std(ddof=1)} for u in units])
    flat = np.concatenate(units)
    ref = ttest_1samp(flat, 0)
    assert np.allclose(t_test(*pool_cells(cells)), (ref.statistic, ref.pvalue))
    assert np.allclose(t_test(*summarize_units(units[0])), tuple(ttest_1samp(units[0], 0)))

    assert t_test(3, 0.0, 0.0) == (0.0, 1.0)                     # all-zero deltas: no effect
    assert all(np.isnan(t_test(3, 0.01, 0.0)))                    # constant shift: untestable
    adj = holm([0.01, 0.04, np.nan, 0.03])
    assert np.allclose(adj[[0, 3, 1]], [0.03, 0.06, 0.06]) and np.isnan(adj[2])

    nom, con = attention_cells(False), attention_cells(True)
    assert len(nom) == 2 * 4 * 5 and (nom["n"] == ATTN_HEADS).all()
    # the conservative copy never has a larger |mean| or a smaller sd than the nominal one
    assert (con["mean"].abs() <= nom["mean"].abs() + 1e-12).all() and (con["sd"] > nom["sd"]).all()


def test_seed_variance_separates_seed_from_neuron_spread(tmp_path):
    """plotters/table_seed_variance.py: pairs joint and SFT replicate by replicate, and
    splits the delta's spread into across-seed (within neuron) and across-neuron."""
    from plotters.table_seed_variance import load_runs, summarize

    seed_g, joint_g, sft_g = (tmp_path / g for g in ("seedgrp", "jg", "sg"))
    # delta = 0.5 + neuron offset (0, 0.2) + seed jitter (+-0.01): seed SD << neuron SD
    for k, idx in enumerate((11, 22)):
        for rep, jitter in (("", 0.0), ("_seed1", 0.01), ("_seed2", -0.01)):
            for arm, base, grp in (("joint", 0.9, joint_g), ("endo", 0.4, sft_g)):
                name = (f"joint_N2.{idx}_POS-1_gateoff_scale10_kl0.1_w0.01_300K" if arm == "joint"
                        else f"endo_N2.{idx}_POS-1_gateoff_w1.0_300K") + rep
                d = (seed_g if rep else grp) / name
                d.mkdir(parents=True)
                v = base + (0.2 * k + jitter if arm == "joint" else 0.0)
                (d / "endogenous_test_metrics_endogenous.json").write_text(json.dumps({"corr": v}))
                (d / "interventional_test_metrics_gaussian.json").write_text(
                    json.dumps({"test": {"corr": v}}))
    df = load_runs(tmp_path, "seedgrp", {"joint": "jg", "endo": "sg"}, True)
    assert set(df["replicate"]) == {"original", "seed1", "seed2"} and len(df) == 2 * 2 * 3 * 2
    s = summarize(df).set_index("metric").loc["interventional"]
    assert s["n_neurons"] == 2 and s["n_replicates"] == 3 and s["n_deltas"] == 6
    assert np.isclose(s["delta_mean"], 0.6)
    assert np.isclose(s["delta_seed_sd"], 0.01) and np.isclose(s["delta_neuron_sd"], 0.2 / np.sqrt(2))
    assert s["delta_positive"] == 6 and np.isclose(s["sft_seed_sd"], 0.0)
