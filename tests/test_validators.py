"""tests/validate_*.py: the model-backed sanity checks (attention-logit and attention-weight
ports, the SAE hook site) are scripts, not pytest tests -- they need the real model and
SAE weights. Here they only have to load and parse their arguments.
"""
import pytest

from script_smoke import REPO_ROOT, run_smoke, smoke_failure

VALIDATORS = sorted((REPO_ROOT / "tests").glob("validate_*.py"))


@pytest.fixture(scope="module")
def smoke():
    return run_smoke(VALIDATORS)


@pytest.mark.parametrize("path", VALIDATORS, ids=lambda p: p.name)
def test_script_loads(smoke, path):
    proc = smoke[path]
    assert proc.returncode == 0, smoke_failure(proc)
