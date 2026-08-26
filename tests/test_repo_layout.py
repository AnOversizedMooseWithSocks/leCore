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


# ---------------------------------------------------------------------------------------------------------
# LINE ENDINGS. .gitattributes declares `* text=auto`, i.e. every text file is stored LF-normalised. A file
# that ships with CRLF fights that declaration: git normalises it on commit, so the working tree and the
# index disagree and EVERY LINE of the file shows as changed while the rendered diff looks empty. That is a
# real failure mode here -- a whole delivery once read as ~137 modified files with nothing visible in them --
# and it is invisible to every other audit because the CONTENT is identical.
# ---------------------------------------------------------------------------------------------------------

_TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".json", ".txt", ".sh", ".cfg", ".toml", ".in", ".bat"}
_TEXT_NAMES = {".gitignore", ".gitattributes", "VERSION", "LICENSE", "MANIFEST.in", "requirements.txt"}


def _repo_root():
    import pathlib
    return pathlib.Path(__file__).resolve().parent.parent


def test_no_text_file_ships_crlf():
    """Every tracked TEXT file is LF-only, matching `* text=auto`.

    Fails LOUDLY with the offenders named, because the symptom otherwise reaches a human as 'lots of files
    changed but the diff is empty' -- which reads like a tooling bug rather than a line-ending one."""
    bad = []
    for p in _repo_root().rglob("*"):
        if not p.is_file() or "__pycache__" in str(p) or "/.git/" in str(p):
            continue
        if p.suffix.lower() not in _TEXT_SUFFIXES and p.name not in _TEXT_NAMES:
            continue
        data = p.read_bytes()
        if b"\r\n" in data:
            bad.append("%s (%d CRLF)" % (p.relative_to(_repo_root()), data.count(b"\r\n")))
    assert not bad, ("text files with CRLF endings -- they fight `* text=auto` and show as whole-file "
                     "diffs with no visible change:\n  " + "\n  ".join(sorted(bad)[:40]))


def test_gitattributes_declares_lf_normalisation():
    """The policy the test above enforces must actually be declared, or a fresh clone on Windows reintroduces
    CRLF and the guard becomes a lie about a setting nobody set."""
    ga = _repo_root() / ".gitattributes"
    assert ga.is_file(), ".gitattributes is missing: nothing declares the line-ending policy"
    assert "text=auto" in ga.read_text(), ".gitattributes no longer declares text=auto"
