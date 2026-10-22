"""The model-backed sanity checks in tests/ (attention-logit and attention-weight ports, the
SAE hook site, cache alignment) are scripts, not pytest tests -- they need the real model
and SAE weights. Here they only have to load and parse their arguments.
"""
import pytest

from script_smoke import REPO_ROOT, run_smoke, smoke_failure

# every script in tests/ that pytest itself does not run
VALIDATORS = sorted(p for p in (REPO_ROOT / "tests").glob("*.py")
                    if not p.name.startswith("test_")
                    and p.name not in ("conftest.py", "script_smoke.py"))


@pytest.fixture(scope="module")
def smoke():
    return run_smoke(VALIDATORS)


@pytest.mark.parametrize("path", VALIDATORS, ids=lambda p: p.name)
def test_script_loads(smoke, path):
    proc = smoke[path]
    assert proc.returncode == 0, smoke_failure(proc)
