"""**Lean 4 and the GPU backends are OPTIONAL, installable, and never required.**

Three claims, and each is worth a different test:

  1. THE ENGINE RUNS WITH NEITHER. Not "we avoided importing them at the top" --
     actually runs, with the imports HARD-BLOCKED at the hook so a lazy import
     inside a function fails too. A module-scope scan cannot prove this; a
     deferred import that every real call path hits is a dependency wearing a
     disguise.
  2. NOTHING IMPORTS THEM AT MODULE SCOPE, which is what keeps `import lecore`
     fast and failure-free on a bare machine.
  3. EACH HAS ONE COMMAND THAT INSTALLS IT, and the engine can tell you what
     that command is. An optional dependency you cannot install on request is
     just a missing feature.
"""

import ast
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Everything the house rules call opt-in. numpy is NOT here: it is the core.
OPTIONAL = ("cupy", "numba", "torch", "scipy", "sklearn", "pyfftw",
            "matplotlib", "faiss", "sympy", "wgpu")


def test_no_optional_dependency_is_imported_at_module_scope():
    """A module-scope import of an optional package makes it mandatory."""
    offenders = []
    for p in (ROOT / "holographic").rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        try:
            tree = ast.parse(p.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in tree.body:                        # TOP LEVEL ONLY
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in OPTIONAL:
                    offenders.append("%s:%d imports %s" % (p.name, node.lineno, n))
    assert not offenders, "\n".join(sorted(offenders))


def test_the_engine_works_with_every_optional_dependency_blocked():
    """**The real test: block them at the import hook and use the engine anyway.**

    Run in a SUBPROCESS with a guard installed before `import lecore`, so a
    deferred import inside a faculty raises too. This is the difference between
    "we were careful about imports" and "it runs on a bare machine" -- and only
    the second is a contract."""
    snippet = r'''
import builtins, sys
BLOCK = {"cupy","numba","torch","scipy","sklearn","pyfftw","matplotlib","faiss","sympy"}
_real = builtins.__import__
def _guard(name, *a, **k):
    if name.split(".")[0] in BLOCK:
        raise ImportError("blocked for the optional-dependency test: %s" % name)
    return _real(name, *a, **k)
builtins.__import__ = _guard

import numpy as np
import lecore
m = lecore.UnifiedMind(dim=128, seed=0)
assert m.find_capability("rotate a mesh")
assert len(m.levers()) == 6
o = m.ouroboros(dim=64)
v = np.random.default_rng(0).standard_normal(64); v /= np.linalg.norm(v)
o["write"](0, v)
cos = float(o["read"](0) @ v)
assert cos > 0.99, cos
# LEAN SOURCE WITHOUT LEAN: emitting needs no binary, which is the whole
# reason the emitter and the verifier are separable.
r = m.lean_export(["q", ["a"]],
                  [{"head": ["p", ["a"]]},
                   {"head": ["q", ["a"]], "body": [["p", ["a"]]]}],
                  check=False)
assert r["ok"] and len(str(r["lean"])) > 50, r
print("OK", round(cos, 4), len(str(r["lean"])))
'''
    out = subprocess.run([sys.executable, "-c", snippet], cwd=str(ROOT),
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "OK" in out.stdout, out.stdout[-500:]


@pytest.mark.parametrize("script", ["tools/install_lean.py",
                                    "tools/install_gpu.py"])
def test_each_optional_backend_has_an_installer_that_does_nothing_by_default(script):
    """**One command installs it, and running that command bare installs nothing.**

    An installer whose DEFAULT action mutates the environment is a trap in a
    script that people run to find out what it would do."""
    p = ROOT / script
    assert p.exists(), script
    out = subprocess.run([sys.executable, str(p), "--help"], cwd=str(ROOT),
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    # argparse shows the docstring's FIRST LINE, and install_lean's says
    # "Opt-in ... NEVER a dependency" while install_gpu's says "Opt-in GPU
    # backend installer". Match the shared idea rather than one spelling:
    # every optional installer must say it is a choice, somewhere in --help.
    text = (out.stdout + out.stderr).lower()
    assert ("opt-in" in text or "optional" in text), out.stdout[:400]
    assert "--remove" in text, "an installer with no way back is not optional"


def test_the_mind_can_report_what_is_optional_and_how_to_get_it():
    """The engine itself answers "do I need Lean or a GPU", with the commands.

    Discoverability is the point: a capability an agent cannot find does not
    exist, and "what would make this faster" is the question an agent actually
    asks."""
    import lecore

    m = lecore.UnifiedMind(dim=64, seed=0)
    rep = m.optional_backends()
    assert set(rep) >= {"lean", "gpu", "core_requires"}
    assert rep["core_requires"] == ["numpy", "python stdlib"]
    for k in ("lean", "gpu"):
        assert rep[k]["install"].startswith("python3 tools/install_"), rep[k]
        assert len(rep[k]["buys"]) > 30, rep[k]
    for q in ("do I need lean or a gpu", "how do I turn on gpu acceleration",
              "what optional things can I install"):
        assert "optional_backends" in str(m.find_capability(q)[:3]), q
