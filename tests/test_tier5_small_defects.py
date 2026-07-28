"""Regression traps for the small Tier 5 defects (work plan items 5.4, 5.5, 5.6).

Three unrelated repairs that share one shape: a condition fully knowable on one side of a boundary became
an unhelpful answer on the other. 5.4 crashed 100 lines below the surface that accepted the bad input; 5.5
returned a confident wrong atom name rather than refusing; 5.6 left a correctness guarantee looking like a
lucky implementation detail.
"""
import numpy as np
import pathlib

import pytest

import lecore
from holographic.misc.holographic_determinism import argmax_tiebreak


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


# --------------------------------------------------------------------------------------
# 5.4 -- climb_ladder input guard.
# --------------------------------------------------------------------------------------

def test_climb_ladder_names_the_expected_type(mind):
    """It used to die with 'unsupported operand type(s) for -: str and str' 100 lines deep in a private
    helper -- a message naming neither the argument nor the expectation. This faculty is reachable from
    /invoke, where free-form data is the DEFAULT case, so the guard belongs at the boundary."""
    with pytest.raises(TypeError) as exc:
        mind.climb_ladder(["hello", "world", "hello", "world"])
    message = str(exc.value)
    assert "list[list[int]]" in message, "the guard does not state the expected type"
    assert "climb_ladder" in message, "the guard does not name the function"


def test_climb_ladder_rejects_empty_and_non_sequence(mind):
    for bad in ([], "a string", 42, None):
        with pytest.raises(TypeError):
            mind.climb_ladder(bad)


def test_valid_input_is_unaffected(mind):
    # The guard must not narrow what already worked.
    assert mind.climb_ladder([[1, 2, 1, 2], [1, 2, 1, 2]])


# --------------------------------------------------------------------------------------
# 5.5 -- the NaN edge, pinned as a CONTRACT rather than fixed.
# --------------------------------------------------------------------------------------

def test_the_nan_behaviour_is_what_the_isa_now_documents():
    """DELIBERATELY PINNING THE CURRENT BEHAVIOUR, not fixing it. Guarding inside argmax_tiebreak would flip
    an existing decision path, and this engine's rule is that existing decisions never flip. If this test
    ever fails, someone changed the ISA-level behaviour and docs/ISA.md must be updated with it."""
    assert argmax_tiebreak([0.1, float("nan"), 0.9]) == 1, \
        "argmax_tiebreak changed; update the non-finite section of docs/ISA.md"


def test_the_isa_documents_the_non_finite_contract():
    # The whole point of 5.5: the edge existed and the contract did not mention it.
    isa = pathlib.Path("docs/ISA.md").read_text()
    assert "NaN IS NOT GUARDED AT THE ISA LEVEL" in isa
    assert "GUARD AT THE BOUNDARY, NOT IN THE ISA" in isa


def test_the_boundary_guards_the_isa_points_to_actually_exist():
    # A contract that names two guards is only worth having if the guards are real.
    from holographic.agents_and_reasoning.holographic_declare import finite_score
    assert finite_score(1.0) and not finite_score(float("nan"))
    from holographic.agents_and_reasoning.holographic_agentloop import _finite
    assert _finite({"a": 1}) and not _finite({"a": float("inf")})


# --------------------------------------------------------------------------------------
# 5.6 -- the citation.
# --------------------------------------------------------------------------------------

def test_distribute_exact_cites_its_published_algorithm():
    import inspect
    from holographic.scene_and_pipeline import holographic_distribute as mod
    src = inspect.getsource(mod.distribute_exact)
    assert "superaccumulator" in src.lower(), "the citation left the WHY-comment"


def test_the_partition_invariance_guarantee_is_covered_elsewhere():
    """RULE 0 APPLIES TO TESTS TOO, and this is the note-to-self.

    I wrote a fresh partition-invariance check to back up the citation, and got the harness wrong TWICE --
    first putting the float non-determinism inside the worker (`sum(chunk)`), then again inside a scatter
    accumulator. Both times the drift was mine, not the code's. The guarantee is about the COMBINATION of
    worker outputs; it cannot rescue arithmetic the worker already rounded.

    tests/test_holographic_exact_and_swept.py already covers this correctly, and its module docstring states
    the contract in one line: the worker returns CONTRIBUTIONS, not a sum. Probing for existing coverage
    first would have saved both attempts -- so this test asserts the coverage exists rather than duplicating
    it badly."""
    import pathlib as _pathlib
    src = _pathlib.Path("tests/test_holographic_exact_and_swept.py").read_text()
    assert "def test_distribute_exact_is_bit_identical_under_any_bucketing" in src, \
        "the real partition-invariance test moved; find it before deleting this pointer"
    assert "CONTRIBUTIONS, not a sum" in src
