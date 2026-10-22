"""Repo-wide checks that keep the folder layout honest after files move.

Static on purpose: most cross-folder imports in this repo are LAZY (inside the function
that needs them, e.g. the OOD and length-generalization evals that trainers/ pulls in
after training), so `--help` never executes them and a stale one only fails at the end of
a cluster run. Here every import in every file is resolved against the file it names,
and every `python <path>.py` that a shell script, README or docstring tells someone to
run must point at a file that exists.
"""
import ast
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

from script_smoke import classify

REPO_ROOT = Path(__file__).resolve().parents[1]
# Kept for the record, not maintained: their commands name scripts that no longer exist.
ARCHIVES = ("scripts_old_with_off_by_one_bug", "outputs_old_with_off_by_one_bug", "outputs")
# The ways a script in a folder puts the repo root on sys.path: the path itself, a ROOT
# name bound to it, or the cwd (everything is run from the repo root).
ROOT_EXPR = r"Path\(__file__\)(?:\.resolve\(\))?\.parents\[1\]"
SHIM_RE = re.compile(rf"sys\.path\.insert\(\s*0\s*,\s*(?:str\(\s*{ROOT_EXPR}\s*\)|os\.getcwd\(\))")
ROOT_NAME_RE = re.compile(rf"^(\w+)\s*=\s*{ROOT_EXPR}\s*$", re.M)
SIBLING_SHIM_RE = re.compile(r"sys\.path\.insert\(\s*0\s*,\s*str\(\s*Path\(__file__\)"
                             r"(?:\.resolve\(\))?\.parent\s*\)\s*\)")


def _listed(*patterns) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard",
                          "--", *patterns], cwd=REPO_ROOT, capture_output=True, text=True,
                         check=True).stdout.split()
    return sorted(REPO_ROOT / p for p in out
                  if not p.startswith(ARCHIVES) and (REPO_ROOT / p).is_file())


@lru_cache(maxsize=None)
def code_files() -> tuple[Path, ...]:
    return tuple(_listed("*.py"))


@lru_cache(maxsize=None)
def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _target_names(t) -> set[str]:
    if isinstance(t, ast.Name):
        return {t.id}
    if isinstance(t, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(e) for e in t.elts))
    return set()


@lru_cache(maxsize=None)
def bindings(path: Path):
    """Names bound at module level (including inside top-level if/try), or None when the
    module star-imports and anything could be there."""
    names = set()

    def visit(stmts):
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(s.name)
            elif isinstance(s, ast.Assign):
                for t in s.targets:
                    names.update(_target_names(t))
            elif isinstance(s, (ast.AnnAssign, ast.AugAssign)):
                names.update(_target_names(s.target))
            elif isinstance(s, ast.Import):
                names.update(a.asname or a.name.split(".")[0] for a in s.names)
            elif isinstance(s, ast.ImportFrom):
                if any(a.name == "*" for a in s.names):
                    return False
                names.update(a.asname or a.name for a in s.names)
            elif isinstance(s, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                blocks = [getattr(s, "body", []), getattr(s, "orelse", []),
                          getattr(s, "finalbody", [])]
                blocks += [h.body for h in getattr(s, "handlers", [])]
                if not all(visit(b) is not False for b in blocks):
                    return False
        return True

    return names if visit(_tree(path).body) is not False else None


def _resolve(base: Path, dotted: str):
    p = base.joinpath(*dotted.split(".")) if dotted else base
    if p.with_suffix(".py").is_file():
        return p.with_suffix(".py")
    if (p / "__init__.py").is_file():
        return p / "__init__.py"
    if p.is_dir():
        return p                                   # namespace package
    return None


def _missing_names(target: Path, names) -> list[str]:
    """Imported names that `target` (a module file or package dir) does not provide."""
    pkg_dir = target if target.is_dir() else (target.parent if target.name == "__init__.py"
                                              else None)
    bound = None if target.is_dir() else bindings(target)
    missing = []
    for n in names:
        if n == "*":
            continue
        if bound is None and not target.is_dir():
            continue                               # star import: cannot tell
        if bound is not None and n in bound:
            continue
        if pkg_dir is not None and _resolve(pkg_dir, n) is not None:
            continue                               # a submodule
        missing.append(n)
    return missing


@lru_cache(maxsize=None)
def _layout():
    files = code_files()
    root_modules = {f.stem for f in files if f.parent == REPO_ROOT}
    root_folders = {f.relative_to(REPO_ROOT).parts[0] for f in files if f.parent != REPO_ROOT}
    # where each script-folder module lives, to name the right fix for a stale bare import
    homes = {}
    for f in files:
        rel = f.relative_to(REPO_ROOT)
        if rel.parts[0] != "src":
            homes.setdefault(f.stem, rel)
    return root_modules, root_folders, homes


def _reaches_root(path: Path) -> bool:
    """Is the repo root on sys.path when this file's imports run? True at the root, under
    src/ and tests/ (only ever imported from code that has it), for library modules with
    no __main__ guard (same), and for scripts that insert it themselves."""
    rel = path.relative_to(REPO_ROOT)
    if len(rel.parts) == 1 or rel.parts[0] in ("src", "tests"):
        return True
    if classify(path) == "module":
        return True
    src = path.read_text()
    return bool(SHIM_RE.search(src)) or any(
        re.search(rf"sys\.path\.insert\(\s*0\s*,\s*str\(\s*{name}\s*\)\s*\)", src)
        for name in ROOT_NAME_RE.findall(src))


def import_problems(path: Path) -> list[str]:
    root_modules, root_folders, homes = _layout()
    rel = path.relative_to(REPO_ROOT)
    siblings = {p.stem for p in path.parent.glob("*.py")} if path.parent != REPO_ROOT else set()
    problems = []
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            mods = [(a.name, []) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:                         # relative import inside a package
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                target = _resolve(base, node.module or "")
                where = f"{rel}:{node.lineno}"
                if target is None:
                    problems.append(f"{where}: relative import {'.' * node.level}"
                                    f"{node.module or ''} does not resolve")
                else:
                    miss = _missing_names(target, [a.name for a in node.names])
                    if miss:
                        problems.append(f"{where}: {miss} not defined in "
                                        f"{target.relative_to(REPO_ROOT)}")
                continue
            mods = [(node.module, [a.name for a in node.names])]
        else:
            continue
        for mod, names in mods:
            head = mod.split(".")[0]
            where = f"{rel}:{node.lineno}"
            if head in root_folders or head in root_modules:
                if not _reaches_root(path):
                    problems.append(f"{where}: imports `{mod}` but the repo root is not on "
                                    f"sys.path when this script runs (add the parents[1] shim)")
                    continue
                target = _resolve(REPO_ROOT, mod)
            elif head in siblings:
                runs_from_folder = (rel.parts[0] == "tests"
                                    or SIBLING_SHIM_RE.search(path.read_text())
                                    or classify(path) != "module")
                if not runs_from_folder:
                    problems.append(f"{where}: bare sibling import `{mod}` in a module other "
                                    f"code imports; write `{rel.parts[0]}.{mod}`")
                    continue
                target = _resolve(path.parent, mod)
            elif head in homes:
                problems.append(f"{where}: bare import `{mod}`, but {homes[head]} is not on "
                                f"this file's sys.path; write "
                                f"`{'.'.join(homes[head].with_suffix('').parts)}`")
                continue
            else:
                continue                           # stdlib / third party
            if target is None:
                problems.append(f"{where}: `{mod}` does not resolve to a file")
                continue
            miss = _missing_names(target, names)
            if miss:
                problems.append(f"{where}: {miss} not defined in {target.relative_to(REPO_ROOT)}")
    return problems


def test_no_scripts_at_repo_root():
    """Every script lives in the folder for its purpose (README.md, "Repo Structure")."""
    stray = [f.name for f in code_files() if f.parent == REPO_ROOT]
    assert not stray, f"move these into a purpose folder: {stray}"


@pytest.mark.parametrize("path", code_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_imports_resolve(path):
    problems = import_problems(path)
    assert not problems, "\n".join(problems)


# `python [-u] <path>.py` and `python -m <module>`, as written in shell scripts, READMEs,
# usage docstrings and "next, run ..." comments.
CMD_RE = re.compile(r"(?:^|[\s(`\"'=#])python3?\s+(?:-u\s+)?(?:(-m)\s+)?([A-Za-z0-9_./-]+)")


def documented_commands():
    out = []
    for f in _listed("*.sh", "*.md", "*.py"):
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            for m in CMD_RE.finditer(line):
                is_module, target = bool(m.group(1)), m.group(2)
                if is_module:
                    if target.split(".")[0] in _layout()[1]:
                        out.append((f, i, target, True))
                elif target.endswith(".py") and not target.startswith(("/", "..")):
                    out.append((f, i, target, False))
    return out


# Folders whose commands are run from inside the folder itself (media/teaser_video);
# everything else is run from the repo root.
SELF_CONTAINED = ("media",)


def test_documented_commands_point_at_real_files():
    bad = []
    for f, i, target, is_module in documented_commands():
        base = (f.parent if f.relative_to(REPO_ROOT).parts[0] in SELF_CONTAINED
                else REPO_ROOT)
        ok = (_resolve(base, target) is not None if is_module
              else (base / target).is_file())
        if not ok:
            bad.append(f"{f.relative_to(REPO_ROOT)}:{i}: python {'-m ' if is_module else ''}"
                       f"{target}")
    assert not bad, f"{len(bad)} command(s) name a missing file:\n" + "\n".join(bad)
