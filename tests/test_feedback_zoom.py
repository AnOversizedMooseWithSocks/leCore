"""Regression traps for the demo-scene operator (sweep 133): feedback, and the deep zoom it accelerates.

Three claims, and all three are the kind that rot quietly. The acceleration is real (so it must be
measured against the same faculty, not an estimate). The error it trades for that speed is BOUNDED (so
it must be checked at depth, not on frame two). And the float64 wall is where the module says it is (so
it must be bracketed from both sides, because a floor that is merely conservative would also "pass").
"""
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

TARGET = (-0.743643887037151, 0.13182590420533)


@pytest.fixture(scope="module")
def mind():
    import lecore
    return lecore.UnifiedMind(dim=256, seed=0)


def test_feedback_step_is_deterministic_and_does_not_mutate(mind):
    """A demo that renders differently twice is a broken demo, and a step that edits its input in place
    makes every trail depend on call order."""
    buf = np.random.default_rng(0).random((24, 32))
    before = buf.copy()
    a = mind.feedback_step(buf, zoom=1.05, decay=0.9)
    b = mind.feedback_step(buf, zoom=1.05, decay=0.9)
    assert np.array_equal(a, b) and np.array_equal(buf, before)
    assert a.shape == buf.shape


def test_decay_is_the_control_parameter_and_one_is_critical(mind):
    """Pinned on BOTH sides. A classifier that only ever answered 'decaying' would pass a one-sided
    test, which is how a dynamical claim turns into a decorative one."""
    buf = np.random.default_rng(1).random((24, 32))
    assert mind.feedback_fixed_point(buf, steps=200, zoom=1.0, decay=0.9)["verdict"] == "converged"
    assert mind.feedback_fixed_point(buf, steps=400, zoom=1.0, decay=1.6)["verdict"] == "diverged"
    steady = mind.feedback_fixed_point(buf, steps=24, zoom=1.0, decay=1.0)
    assert abs(steady["ratio"] - 1.0) < 1e-9, steady
    # the measured energy ratio must BE the decay, or 'ratio' is a number with no meaning
    r = mind.feedback_fixed_point(buf, steps=12, zoom=1.0, decay=0.95, tol=0.0)
    assert abs(r["ratio"] - 0.95) < 0.02, r


def test_the_float64_wall_is_bracketed_not_merely_asserted(mind):
    """A floor that is too conservative also 'passes' a one-sided check. Distinct AT the floor and
    colliding a decade BELOW it is what makes 13.8 decades a measurement."""
    w = mind.zoom_floor(TARGET, 320)
    assert 13.0 < w["decades"] < 14.5, w
    assert w["distinct_at_floor"] == 320, w
    assert w["distinct_below_floor"] < 320, w
    assert w["verified"] is True


def test_the_wall_depends_on_where_you_look(mind):
    """The part a single constant would hide: centred on the origin the largest coordinate shrinks with
    the span, so there is no eps wall at all. Quoting 13.8 decades for every target would be wrong."""
    seahorse = mind.zoom_floor(TARGET, 320)["decades"]
    assert mind.zoom_floor((0.0, 0.0), 320)["decades"] > 100.0
    assert mind.zoom_floor((1e-3, 0.0), 320)["decades"] > seahorse


def test_the_zoom_stops_at_the_wall_instead_of_rendering_noise(mind):
    """The abstention, at the arithmetic layer. Past the floor there is no structure left to resolve --
    only the float -- and continuing would be showing a viewer rounding error as detail."""
    r = mind.deep_zoom(centre=TARGET, span0=1e-12, rate=0.1, frames=8, width=64, height=36,
                       max_iter=16, band=4)
    assert r["stopped"].startswith("precision floor"), r["stopped"]
    assert r["frames_rendered"] < 8


def test_the_acceleration_is_real_and_its_error_is_bounded(mind):
    """Measured against the SAME faculty at band=1 over the SAME span range -- escape cost grows with
    depth, so a baseline run over a shallower prefix would understate the speedup (it did: 6.1x
    against a true 9.6x). Small resolution here to stay inside the test budget; the 60 fps figure is
    quoted at 320x180 and checked in the application's own selftest."""
    fast = mind.deep_zoom(centre=TARGET, frames=8, width=160, height=90, max_iter=48, band=8)
    slow = mind.deep_zoom(centre=TARGET, frames=8, width=160, height=90, max_iter=48, band=1)
    assert fast["ms_per_frame"] < slow["ms_per_frame"], (fast["ms_per_frame"], slow["ms_per_frame"])
    acc = mind.deep_zoom(centre=TARGET, frames=8, width=160, height=90, max_iter=48, band=8,
                         verify=True)
    assert acc["mean_abs_error"] < 0.05 and acc["max_abs_error"] < 0.10, acc


def test_band_one_is_the_exact_path(mind):
    """band=1 recomputes every row every frame, so it must have no reuse error at all. If this drifts,
    the band arithmetic is wrong and every other error number here is meaningless."""
    r = mind.deep_zoom(centre=TARGET, frames=5, width=96, height=54, max_iter=32, band=1, verify=True)
    assert r["mean_abs_error"] == 0.0, r["mean_abs_error"]


def test_the_effect_ships_as_a_runnable_application(mind):
    """A demo-scene effect that is not in the library is a demo nobody can run."""
    names = [a["name"] for a in mind.apps()]
    assert "infinite_zoom" in names
    entry = next(a for a in mind.apps() if a["name"] == "infinite_zoom")
    assert entry["domain"] == "demoscene" and entry["artefact"]


# ---------------------------------------------------------------------------------------------
# THE SIDEWAYS DIRECTION (sweep 134): one operator, two costumes -- and the claim is a NUMBER.
# ---------------------------------------------------------------------------------------------

def test_the_critical_decay_is_the_same_constant_in_both_costumes(mind):
    """THE AS-ABOVE-SO-BELOW RESULT, made empirical instead of aesthetic. If the field and sequence
    paths are one operator, something must be the same number in both. It is the critical decay, and
    it is exactly 1.0 -- because a cyclic permute is orthogonal exactly as an integer pixel roll is,
    so `decay * T` has every eigenvalue on a circle of radius `decay`."""
    vec = np.random.default_rng(3).random(256)
    frame = np.random.default_rng(4).random((48, 64))
    seq = mind.feedback_fixed_point(vec, steps=16, zoom=1.0, rotate=3, decay=0.95, tol=0.0)
    field = mind.feedback_fixed_point(frame, steps=16, zoom=1.0, rotate=0.0, decay=0.95, tol=0.0)
    assert abs(seq["ratio"] - 0.95) < 1e-12, seq
    assert abs(field["ratio"] - 0.95) < 1e-12, field
    # both sides of the critical value, in the SEQUENCE costume
    assert mind.feedback_fixed_point(vec, steps=200, zoom=1.0, rotate=3, decay=0.9)["verdict"] == "converged"
    assert mind.feedback_fixed_point(vec, steps=400, zoom=1.0, rotate=3, decay=1.6)["verdict"] == "diverged"


def test_permutation_not_rank_is_what_buys_the_exactness(mind):
    """THE ACTUAL FINDING, and it killed two tidier hypotheses. The constant is not a property of
    sequences: a rounded ROTATION misses it in the field costume while the identity and a roll hit it
    exactly, in either rank. What separates them is whether the transform is a PERMUTATION."""
    vec = np.random.default_rng(5).random(256)
    frame = np.random.default_rng(6).random((48, 64))
    assert mind.is_permutation(vec, zoom=1.0, rotate=3)["permutation"] is True
    assert mind.is_permutation(frame, zoom=1.0, rotate=0.0)["permutation"] is True
    rot = mind.is_permutation(frame, zoom=1.0, rotate=0.15)
    assert rot["permutation"] is False
    assert 0 < rot["sampled_once"] < rot["cells"], rot          # many-to-one, and it says how badly
    # a resample breaks it in the SEQUENCE costume too -- same cause, both ranks
    assert mind.is_permutation(vec, zoom=1.03, rotate=0)["permutation"] is False


def test_the_field_costume_did_not_move(mind):
    """The additive rule. Adding a rank-1 path must not move a byte of the rank-2/3 result, and the
    2-D and 3-D shapes and determinism are the contract everything else in this file rests on."""
    frame = np.random.default_rng(7).random((32, 40))
    a = mind.feedback_step(frame, zoom=1.02, rotate=0.15, decay=0.9)
    b = mind.feedback_step(frame, zoom=1.02, rotate=0.15, decay=0.9)
    assert np.array_equal(np.asarray(a), np.asarray(b)) and np.asarray(a).shape == frame.shape
    colour = np.stack([frame, frame * 0.5, frame * 0.25], axis=-1)
    assert np.asarray(mind.feedback_step(colour, zoom=1.05, decay=0.8)).shape == colour.shape
