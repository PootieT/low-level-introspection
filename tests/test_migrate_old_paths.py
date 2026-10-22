"""utils/migrate_old_paths.py: rewrites a pre-reorganization branch's references to the new
layout, leaves the current tree alone, and knows where every moved script actually is.
"""
import pytest

from script_smoke import REPO_ROOT

from utils.migrate_old_paths import DELETED, KEEPS_OLD_PATHS, MOVED, candidate_files, rewrite


def test_every_moved_script_exists_where_the_map_says():
    missing = [f"{d}/{s}.py" for s, d in MOVED.items() if not (REPO_ROOT / d / f"{s}.py").is_file()]
    assert not missing, missing
    assert not [s for s in DELETED if list(REPO_ROOT.rglob(f"{s}.py"))]


def test_rewrites_old_style_references():
    sh = ("python train_introspector_joint.py --config x.json\n"
          "INDEX=$(python select_layer_gen_index.py --config c)\n"
          "# then: python analysis/plot_weight_sweep.py ./cache_signals.py\n"
          "python scripts/attention/analysis/foo.sh mechanistic_analysis/x.py\n"
          "| `analysis/1_attn_logit_analysis.sh` | shell drivers keep their analysis/ name\n")
    assert rewrite(sh, is_python=False) == (
        "python trainers/train_introspector_joint.py --config x.json\n"
        "INDEX=$(python data_prep/select_layer_gen_index.py --config c)\n"
        "# then: python plotters/plot_weight_sweep.py ./data_prep/cache_signals.py\n"
        "python scripts/attention/analysis/foo.sh mechanistic_analysis/x.py\n"
        "| `analysis/1_attn_logit_analysis.sh` | shell drivers keep their analysis/ name\n")
    py = ("from train_introspector import preprocess_example\n"
          "    from score_ood_generalization import evaluate_ood\n"
          "import score_prompt_baseline as spb\n"
          "import cache_signals  # legacy\n"
          "from analysis.run_catalog import ROOT\n"
          "from train_introspector_joint_extra import x\n")
    assert rewrite(py, is_python=True) == (
        "from trainers.train_introspector import preprocess_example\n"
        "    from evals.score_ood_generalization import evaluate_ood\n"
        "import evals.score_prompt_baseline as spb\n"
        "import data_prep.cache_signals as cache_signals  # legacy\n"
        "from plotters.run_catalog import ROOT\n"
        "from train_introspector_joint_extra import x\n")


@pytest.mark.parametrize("is_python", [True, False])
def test_rewrite_is_idempotent(is_python):
    old = "from cache_signals import x\npython trainers/train_sae_detector.py analysis/a.py\n"
    once = rewrite(old, is_python)
    assert rewrite(once, is_python) == once


def test_current_tree_needs_no_rewrite():
    """main is already migrated; any hit here is a reference written in the old style."""
    hits = [str(f.relative_to(REPO_ROOT)) for f in candidate_files(REPO_ROOT)
            if f.relative_to(REPO_ROOT).as_posix() not in KEEPS_OLD_PATHS
            and rewrite(t := f.read_text(errors="surrogateescape"), f.suffix == ".py") != t]
    assert not hits, hits
