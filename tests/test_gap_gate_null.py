"""Regression traps for the null-referenced synthesis gate (work plan item 2.3).

The item was "give permutation_null its first real client". The client found a defect on its first run,
and that defect -- A NULL SEEDED LIKE ITS DATA REPLAYS THE DATA -- is the most valuable thing here, so it
is pinned hardest. Everything else guards the gate itself.
"""
import numpy as np
import pytest

import lecore
from holographic.misc.holographic_voidsynth import chain_signature, gap_gate_null, synthesize_for_goal


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=256, seed=0)


@pytest.fixture(scope="module")
def library():
    rng = np.random.default_rng(0)          # deliberately seed 0 -- the colliding configuration
    lib = rng.standard_normal((10, 256))
    return lib / np.linalg.norm(lib, axis=1, keepdims=True)


def test_a_null_seeded_like_its_data_replays_the_data():
    # THE DEFECT, REPRODUCED. This is why gap_gate_null derives its null seed from the library instead of
    # taking the caller's. If this ever stops holding, numpy's stream changed and the derivation's
    # rationale should be re-read -- but the derivation is still correct, because it removes the
    # possibility rather than avoiding one instance of it.
    rng = np.random.default_rng(0)
    lib = rng.standard_normal((10, 256))
    lib /= np.linalg.norm(lib, axis=1, keepdims=True)
    naive = np.random.default_rng(0)        # a null seeded the same way the data was
    v = naive.standard_normal(256)
    v /= np.linalg.norm(v)
    assert abs(float(v @ lib[0]) - 1.0) < 1e-9, \
        "the collision is gone; re-read gap_gate_null's seed-derivation rationale"


def test_gap_gate_null_does_not_collide_with_the_library_rng(library):
    # The fix, checked through the real path: a real goal must now STAND OUT. Before the derivation, 9 of
    # 32 "random goals" scored a perfect 1.000 because they WERE library atoms, and the null duly reported
    # that a real goal did not stand out (p=0.333).
    goal = chain_signature(library[[2, 5, 7]])
    out = gap_gate_null(library, goal, n_null=32)
    assert out["collapsed"], "a real goal failed its own null: p=%.3f null_mean=%.3f" % (
        out["p"], out["null_mean"])
    assert out["null_mean"] < 0.5, "the null mean is contaminated (%.3f)" % out["null_mean"]


def test_a_random_goal_does_not_clear_the_null(library):
    # The other direction. A null that only ever says yes is decoration.
    rng = np.random.default_rng(7)
    junk = rng.standard_normal(256)
    junk /= np.linalg.norm(junk)
    out = gap_gate_null(library, junk, n_null=32)
    assert not out["collapsed"]
    assert not out["threshold_is_meaningful"]


def test_the_bare_threshold_is_actually_separating_on_random_libraries():
    # The measurement the bare 0.85 constant hides: what DO random goals score? False-accept rate must be
    # 0 across dims and library sizes, and coherence must rise with library size / fall with dimension --
    # which is precisely why one constant cannot be right for every library.
    for dim, size in ((128, 10), (256, 20), (512, 10)):
        rng = np.random.default_rng(1)
        lib = rng.standard_normal((size, dim))
        lib /= np.linalg.norm(lib, axis=1, keepdims=True)
        scores = []
        for _ in range(20):
            v = rng.standard_normal(dim)
            v /= np.linalg.norm(v)
            scores.append(synthesize_for_goal(lib, v, max_length=4, threshold=0.85)["coherence"])
        assert max(scores) < 0.85, "a random goal cleared the bar at D=%d, L=%d" % (dim, size)


def test_gap_gate_null_is_deterministic(library):
    goal = chain_signature(library[[1, 4]])
    a, b = gap_gate_null(library, goal, n_null=16), gap_gate_null(library, goal, n_null=16)
    assert a["p"] == b["p"] and a["null_seed"] == b["null_seed"]


def test_the_derived_seed_varies_with_the_library(library):
    other = library[::-1].copy()
    goal = chain_signature(library[[1, 4]])
    assert gap_gate_null(library, goal, n_null=8)["null_seed"] != \
        gap_gate_null(other, goal, n_null=8)["null_seed"]


def test_permutation_null_now_has_a_real_client():
    # The item's own success condition: the unifier goes 0 clients -> 1, and the client is a live code
    # path rather than a selftest.
    import inspect
    from holographic.misc import holographic_voidsynth as vs
    assert "permutation_null" in inspect.getsource(vs.gap_gate_null)


def test_the_ladder_can_use_the_null_gate_and_defaults_off(mind, library):
    from holographic.agents_and_reasoning.holographic_declare import Ladder
    goal = chain_signature(library[[2, 5, 7]])
    args = {"library": library, "goal_sig": goal}
    assert Ladder(mind).null_check is False, "the null check must default OFF (additive)"
    checked = Ladder(mind, null_check=True, n_null=16).resolve("x", args=args)
    for rung in checked.descent:
        if rung.index == 3:
            assert rung.why, "rung 3 must state a reason either way"


def test_the_gate_null_is_discoverable(mind):
    for query in ("is my threshold meaningful", "null reference a coherence gate",
                  "check a synthesis threshold against chance"):
        assert "Null-reference" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces the gate null" % query
