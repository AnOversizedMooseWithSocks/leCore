"""Guards against the two ways a repo REORGANISATION silently breaks the build.

Both of these bit us when the flat layout became the `holographic/` package:

  1. Bare `pytest` (what CI runs) does NOT put the repo root on sys.path -- it only inserts each test file's first
     parent without an __init__.py, which is `tests/`. So `import holographic` and `import app` died at collection
     with ModuleNotFoundError, while `python -m pytest` (which DOES add the cwd) kept working locally. The cure is
     `pythonpath = ["."]` in pyproject.toml, and this test pins it there.

  2. An import can name a module that no longer exists at that location and still "work" by accident (the old file is
     lying around, or the root happens to be on sys.path) until the day it doesn't. tools/audit_imports.py resolves
     every import in the repo against what's actually on disk; this test fails if any of them dangle.
"""
import os
import sys
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))


def test_pyproject_puts_repo_root_on_the_path():
    """Without this, bare `pytest` cannot import `holographic` or the root modules -- every test errors at collection."""
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as fh:
        cfg = tomllib.load(fh)
    ini = cfg.get("tool", {}).get("pytest", {}).get("ini_options", {})
    assert "." in ini.get("pythonpath", []), (
        "pyproject.toml [tool.pytest.ini_options] must set pythonpath = ['.'] so bare `pytest` can import the engine"
    )


def test_the_engine_is_importable_the_way_ci_imports_it():
    """The two import styles the tests actually use: the package, and the root-level modules."""
    import holographic                      # the package
    import app                              # a root-level module
    assert holographic is not None and app is not None


def test_no_broken_imports_anywhere():
    """Every import of ours resolves to a file that exists on disk (catches a half-finished move)."""
    from audit_imports import audit
    broken, _flat = audit(REPO)
    assert not broken, "imports that resolve to nothing on disk:\n" + "\n".join(
        "  %s:%d imports %r" % (rel, line, name) for rel, line, name, _hint in broken[:20])
