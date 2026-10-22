"""Shared pytest setup.

Puts the repo root on sys.path so the tests import `src`, `plotters`, `trainers`, ... the
same way under `pytest` and `python -m pytest`. Run from the repo root:

    python -m pytest tests/
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
