"""Load every script in a folder the way scripts/ and the READMEs run it.

Each script gets its own interpreter, started from the repo root as `python <folder>/<x>.py`,
so `sys.path[0]` is the script's own folder and NOT the repo root. That is what makes a
missing sys.path shim or a stale bare-name import (`from train_introspector import ...`
after the file moved) fail here instead of an hour into a cluster job. In-process imports
would hide both: pytest already has the repo root on sys.path.

What "load" means depends on the file:
  cli     a `__main__` guard plus argparse   ->  `python <path> --help` must exit 0
  script  a `__main__` guard, no argparse    ->  run the module body, not main()
  module  no `__main__` guard (a library)    ->  `import <folder>.<stem>` from the repo root

The folders' scripts are launched in parallel: each one imports torch or matplotlib, and
one at a time a folder takes minutes.
"""
import ast
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# `python -c` puts '' (the cwd) at sys.path[0]; `python <path>` puts the script's folder
# there. Swap it so the module body sees exactly what a real run sees.
_LOAD_AS_SCRIPT = (
    "import os, runpy, sys; p = os.path.abspath(sys.argv[1]); "
    "sys.path[0] = os.path.dirname(p); runpy.run_path(p, run_name='__smoke__')"
)


def scripts_in(folder: str) -> list[Path]:
    return sorted((REPO_ROOT / folder).glob("*.py"))


def _has_main_guard(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            names = [n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)]
            consts = [n.value for n in ast.walk(node.test) if isinstance(n, ast.Constant)]
            if "__name__" in names and "__main__" in consts:
                return True
    return False


def classify(path: Path) -> str:
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    if not _has_main_guard(tree):
        return "module"
    return "cli" if ("ArgumentParser" in src or "parse_args" in src) else "script"


def smoke_command(path: Path) -> list[str]:
    rel = path.relative_to(REPO_ROOT)
    kind = classify(path)
    if kind == "cli":
        return [sys.executable, str(rel), "--help"]
    if kind == "script":
        return [sys.executable, "-c", _LOAD_AS_SCRIPT, str(rel)]
    dotted = ".".join(rel.with_suffix("").parts)
    return [sys.executable, "-c", f"import importlib; importlib.import_module({dotted!r})"]


def script_env() -> dict:
    env = dict(os.environ)
    # A PYTHONPATH pointing at the repo root would make every missing shim pass.
    env.pop("PYTHONPATH", None)
    env.update(MPLBACKEND="Agg", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1",
               WANDB_MODE="disabled")
    return env


def run_smoke(paths: list[Path], workers: int | None = None,
              timeout: int = 300) -> dict[Path, subprocess.CompletedProcess]:
    env = script_env()

    def one(path):
        return subprocess.run(smoke_command(path), cwd=REPO_ROOT, env=env, timeout=timeout,
                              capture_output=True, text=True)

    workers = workers or min(8, os.cpu_count() or 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(paths, pool.map(one, paths)))


def smoke_failure(proc: subprocess.CompletedProcess) -> str:
    tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
    return f"`{' '.join(proc.args[1:])}` exited {proc.returncode}:\n{tail}"
