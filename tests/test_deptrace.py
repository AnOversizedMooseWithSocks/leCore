"""Backlog D2: import-footprint tracing, and the constitutional gate it makes possible.

WHY THESE TESTS: the backlog asked for a lazy-accessor SWEEP across ~500 modules so tooling could see the real
dep set. Measurement refuted the premise (only 3 guarded external imports exist; the balloon is deferred
self-imports, which already never run at import time). What shipped instead is the measuring instrument -- so
these tests pin the CLASSIFIER (everything rests on it) and the engine-wide numbers that carry the refutation.
"""

import os
import tempfile

import pytest

import lecore
from holographic.io_and_interop.holographic_deptrace import (
    footprint_report, import_edges, trace)


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


def _write(td, text):
    p = os.path.join(td, "probe.py")
    with open(p, "w") as fh:
        fh.write(text)
    return p


def test_classifier_labels_each_position_exactly():
    src = ("import os\n"
           "try:\n"
           "    import numba\n"
           "except ImportError:\n"
           "    numba = None\n"
           "from json import dumps\n"
           "def f():\n"
           "    import cupy\n"
           "class C:\n"
           "    def m(self):\n"
           "        from sympy import Symbol\n")
    with tempfile.TemporaryDirectory() as td:
        got = {(e["name"], e["kind"]) for e in import_edges(_write(td, src))}
    assert got == {("os", "hard"), ("json", "hard"), ("numba", "guarded"),
                   ("cupy", "deferred"), ("sympy", "deferred")}


def test_deferred_beats_guarded():
    """A try INSIDE a function is still deferred: it does not run at import, which is the question being asked."""
    src = "def f():\n    try:\n        import pyfftw\n    except ImportError:\n        pass\n"
    with tempfile.TemporaryDirectory() as td:
        got = {(e["name"], e["kind"]) for e in import_edges(_write(td, src))}
    assert got == {("pyfftw", "deferred")}


def test_syntax_error_is_not_our_problem():
    with tempfile.TemporaryDirectory() as td:
        assert import_edges(_write(td, "def (:\n")) == []


def test_core_import_closure_is_numpy_only(mind):
    """THE CONSTITUTION AS A GATE: hard constraint #1 says NumPy/Flask/stdlib/hashlib only in core. This fails by
    name the day anything top-level-imports torch/scipy/sklearn anywhere in lecore's import-time closure."""
    core = mind.import_footprint("lecore")
    assert core["required_external"] == ["numpy"], core["required_external"]
    for banned in ("torch", "scipy", "sklearn", "numba", "cupy", "sympy"):
        assert banned not in core["required_external"]


def test_naive_tracing_balloons_and_the_engine_is_already_lazy(mind):
    """The refutation, pinned: the balloon is real and large, and it is NOT caused by try/except accelerators --
    guarded edges are a rounding error next to deferred ones."""
    core = mind.import_footprint("lecore")
    assert core["required"] < 60, core["required"]           # the honest footprint is small
    assert core["naive"] > 400, core["naive"]               # a blind tracer drags in essentially everything
    assert core["ratio"] > 5.0, core["ratio"]
    kinds = core["edges_by_kind"]
    assert kinds["deferred"] > 10 * kinds["guarded"], kinds  # deferred dominates: already lazy, by a landslide


def test_follow_selects_which_edges_are_crossed(mind):
    """`follow` is the knob the whole design rests on -- hard-only must be a strict subset of follow-everything."""
    hard = set(mind.trace_imports("lecore", follow=("hard",))["modules"])
    everything = set(mind.trace_imports("lecore", follow=("hard", "guarded", "deferred"))["modules"])
    assert hard < everything
    assert "lecore" in hard


def test_flat_and_packaged_entry_names_agree(mind):
    """The flat/packaged duality (backlog D1) must not change the answer -- flatcompat resolves both at runtime,
    so a tracer that disagreed with the runtime would be lying."""
    a = mind.trace_imports("holographic.io_and_interop.holographic_ccrun")
    b = mind.trace_imports("holographic_ccrun")
    assert a["modules"] == b["modules"]


def test_ccrun_footprint_is_small_and_honest(mind):
    """A leaf module's required closure is itself plus what it truly needs -- ccrun really does hard-import the
    emitter, and really does not need Zig."""
    r = mind.import_footprint("holographic.io_and_interop.holographic_ccrun")
    assert "holographic.io_and_interop.holographic_emit" in r["required_modules"]
    assert r["required_external"] == ["numpy"], r["required_external"]


def test_unresolved_is_reported_not_raised(mind):
    """audit_imports owns broken imports; deptrace reports and moves on rather than dying on someone else's bug."""
    t = trace("lecore", root=".", follow=("hard", "guarded", "deferred"))
    assert isinstance(t["unresolved"], list)
    assert t["unresolved"] == []                            # and today the tree is clean
