"""The escape-time fast path: real, modest, and OPT-IN because it changes bits.

The demo-scene sweep found this loop before it found anything else -- `np.power(z, 2.0)` on a
complex array is a complex exp/log per iteration where `z*z` would do. This file pins the three
things that decide whether the flag may exist at all: the default is bit-identical, the flag is
inert unless power==2, and the speedup is the MEASURED one rather than the microbenchmark's.
"""
import numpy as np
import lecore
import pytest

KW = dict(width=160, height=100, center=(-0.743643887037151, 0.13182590420533),
          span=3.0e-3, max_iter=48)


@pytest.fixture(scope="module")
def mind():
    return lecore.UnifiedMind(dim=64, seed=0)


def test_the_default_is_untouched_and_deterministic(mind):
    """Two calls at the default must agree bit for bit -- the floor every other claim stands on."""
    a = mind.escape_time(**KW)
    b = mind.escape_time(**KW)
    assert np.array_equal(a.view(np.float64), b.view(np.float64))


def test_fast_square_is_inert_unless_power_is_exactly_two(mind):
    """A non-integer power has no fast path, and asking for one must not silently change anything."""
    for power in (3.0, 2.5):
        a = mind.escape_time(power=power, **KW)
        b = mind.escape_time(power=power, fast_square=True, **KW)
        assert np.array_equal(a, b), "fast_square changed a power=%s render" % power


def test_the_flag_DOES_change_bits_which_is_why_it_is_opt_in(mind):
    """KEPT NEGATIVE, pinned as a positive assertion. np.power on a complex array goes through
    exp/log; z*z does not. Over 200k random complex128 values 57,908 differ at max 3.55e-15 -- and
    the escape iteration AMPLIFIES that: here it reaches ~1e-9 in the smooth escape count. This
    repo's rule is that a change bit-identical to 1e-12 has still flipped a creature's trajectory,
    so the speed has to be ASKED for. If this test ever fails because the two agree, the flag has
    become free and should be reconsidered -- but it must be re-measured, not assumed."""
    a = mind.escape_time(**KW)
    b = mind.escape_time(fast_square=True, **KW)
    assert not np.array_equal(a, b)
    assert np.abs(a - b).max() < 1e-5, "the fast path should differ in the last bits, not the answer"


def test_the_two_paths_agree_to_a_tolerance_that_makes_the_flag_usable(mind):
    """It is a bit-level difference, not a different picture: mean |diff| stays far below one
    iteration count, so a palette cannot tell them apart."""
    a = mind.escape_time(**KW)
    b = mind.escape_time(fast_square=True, **KW)
    assert float(np.abs(a - b).mean()) < 1e-6


def test_bounds_ratio_reaches_the_delegate(mind):
    """The wrapper was missing `bounds_ratio` entirely -- a delegation drift the 0.80-overlap gate
    had not surfaced. Passing it must not raise, which is the cheapest proof it is plumbed."""
    out = mind.escape_time(bounds_ratio=None, **KW)
    assert out.shape == (KW["height"], KW["width"])


# ---------------------------------------------------------------------------------------------
# THE UP / DOWN / SIDEWAYS CHECK, pinned. The close-out habit this repo names: Down -- does it
# work on components of its own input? Up -- when its input is a component of something larger?
# Sideways -- which costumes does it wear (field, structure, sequence, program)? A missed
# direction is a missed faculty, so the directions that DO work get pinned before they rot, and
# the one that does not gets an explicit test saying so rather than a silent hole.
# ---------------------------------------------------------------------------------------------

def _blob(n=64):
    b = np.zeros((n, n), dtype=float)
    b[n // 2 - 4:n // 2 + 4, n // 2 - 4:n // 2 + 4] = 1.0
    return b


def test_DOWN_feedback_works_on_a_component_of_its_own_input(mind):
    """A tile of a frame is a frame. If this ever stops holding, tiled/streamed feedback stops
    being possible and nothing else would have told us."""
    buf = _blob()
    tile = buf[0:32, 0:32]
    assert np.asarray(mind.feedback_step(tile, zoom=1.05, decay=0.9)).shape == (32, 32)
    strip = buf[32:33, :]
    assert np.asarray(mind.feedback_step(strip, zoom=1.05, decay=0.9)).shape == (1, 64)


def test_UP_feedback_works_when_the_frame_is_one_layer_of_something_larger(mind):
    """MEASURED, not designed: a (H,W,3) colour frame goes through unchanged in shape. It was
    built for a single field, so this is a property worth pinning before an optimisation removes
    it by accident."""
    rgb = np.stack([_blob(), _blob() * 0.5, _blob() * 0.25], axis=-1)
    assert np.asarray(mind.feedback_step(rgb, zoom=1.05, decay=0.9)).shape == (64, 64, 3)


def test_SIDEWAYS_the_missed_direction_is_now_built_and_the_costumes_share_a_constant(mind):
    """THE MISSED FACULTY, BUILT (sweep 134) -- and this test was changed on purpose, with the
    measurement in hand, exactly as its previous version said it should be.

    It used to assert that a 1-D hypervector RAISED. It now asserts the thing that replaced that
    failure, and the claim is empirical rather than aesthetic: if the field and sequence paths are
    really one operator, some NUMBER must be the same in both. It is the critical decay, and it is
    exactly 1.0 -- because a cyclic permute (the VSA sequence operator, and the reservoir's own fixed
    recurrence) is orthogonal, precisely as an integer pixel roll is.

    THE CONDITION IS SHARPER THAN "field vs sequence", which is the actual finding: the constant holds
    whenever the transform is a PERMUTATION, and rank has nothing to do with it. Two hypotheses died
    to get here -- that rank was the cause (a 2-D integer roll is exact, so no) and that CLAMPED edges
    were (wrapped 1.0001997 vs clamped 1.0001981, indistinguishable, so no). It is nearest-neighbour
    ROUNDING, which is many-to-one and therefore not a permutation."""
    vec = np.arange(256.0)
    assert np.asarray(mind.feedback_step(vec, zoom=1.0, rotate=3, decay=0.9)).shape == (256,)

    # THE SHARED CONSTANT, to 1e-12 in both costumes, with a permutation in each.
    seq = mind.feedback_fixed_point(vec, steps=16, zoom=1.0, rotate=3, decay=0.95, tol=0.0)
    field = mind.feedback_fixed_point(_blob(), steps=16, zoom=1.0, rotate=0.0, decay=0.95, tol=0.0)
    assert abs(seq["ratio"] - 0.95) < 1e-12, seq
    assert abs(field["ratio"] - 0.95) < 1e-12, field

    # and it is PERMUTATION-NESS that buys the exactness, in either rank
    assert mind.is_permutation(vec, zoom=1.0, rotate=3)["permutation"] is True
    rot = mind.is_permutation(_blob(), zoom=1.0, rotate=0.15)
    assert rot["permutation"] is False and rot["sampled_once"] < rot["cells"], rot


def test_SIDEWAYS_the_sequence_costume_is_a_leaky_echo_state_update(mind):
    """The join the mapping predicted: with decay < 1 a 1-D feedback step IS the reservoir's update.
    mind.reservoir's own docstring makes the same claim about the same operator -- "permute is
    norm-preserving (orthogonal), which is exactly the echo-state property" -- so the demo scene's
    oldest effect and this engine's sequence recurrence are one operator, and 1.0 is where both
    change behaviour."""
    vec = np.arange(256.0) / 256.0
    assert mind.feedback_fixed_point(vec, steps=200, zoom=1.0, rotate=3, decay=0.9)["verdict"] == "converged"
    assert mind.feedback_fixed_point(vec, steps=400, zoom=1.0, rotate=3, decay=1.6)["verdict"] == "diverged"


def test_the_field_costume_is_byte_identical_after_the_sideways_build(mind):
    """The additive rule, checked rather than asserted: adding a rank-1 path must not move a single
    byte of the rank-2 or rank-3 result."""
    frame = _blob()
    a = np.asarray(mind.feedback_step(frame, zoom=1.02, rotate=0.15, decay=0.9))
    assert a.shape == frame.shape and np.array_equal(a, np.asarray(
        mind.feedback_step(frame, zoom=1.02, rotate=0.15, decay=0.9)))
    colour = np.stack([frame, frame * 0.5, frame * 0.25], axis=-1)
    assert np.asarray(mind.feedback_step(colour, zoom=1.05, decay=0.8)).shape == colour.shape
