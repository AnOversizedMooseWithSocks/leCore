"""Tests for holographic_opponent.py -- the leOS-compatible opponent-channel decomposition (A vs B)."""
import numpy as np
from holographic.rendering.holographic_opponent import opponent_channels, decompose, classify, blend, agree, divergence


def test_identical_sources_redundant_no_purple():
    rng = np.random.default_rng(1)
    v = rng.standard_normal(256)
    ch = opponent_channels(v, v)
    assert ch["divergence_score"] < 1e-6 and ch["cosine_similarity"] > 0.999
    assert ch["channel_magnitudes"]["purple"] < 1e-6           # nothing emergent when they agree
    assert classify(ch)["type"] == "redundant"


def test_orthogonal_sources_purple_is_a_plus_b():
    a = np.zeros(128); a[0] = 1.0
    b = np.zeros(128); b[1] = 1.0
    ch = opponent_channels(a, b)
    assert np.allclose(ch["purple"], a + b)                    # leOS: purple = a_exclusive + b_exclusive = a + b here
    assert ch["channel_magnitudes"]["purple"] > 1.0
    assert ch["divergence_score"] > 1.5                        # ~pi/2


def test_purple_identity_always_holds():
    # the whole point of the leOS design: purple == a_exclusive + b_exclusive (it does NOT cancel to zero)
    rng = np.random.default_rng(2)
    for _ in range(20):
        a = rng.standard_normal(200); b = rng.standard_normal(200)
        ch = opponent_channels(a, b)
        assert np.allclose(ch["purple"], ch["a_exclusive"] + ch["b_exclusive"])


def test_agreement_is_sign_match_times_min_magnitude():
    p = np.array([2.0, 3.0, -1.0, 0.0])
    q = np.array([1.0, 5.0,  4.0, 2.0])
    ch = opponent_channels(p, q)
    # sign(p)*sign(q)*min(|p|,|q|): [1*1*1, 1*1*3, (-1)*1*1, 0*1*0] = [1, 3, -1, 0]
    assert np.allclose(ch["agreement"], [1.0, 3.0, -1.0, 0.0])


def test_exclusive_is_orthogonal_to_the_other_direction():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(64); b = rng.standard_normal(64)
    ch = opponent_channels(a, b)
    # a_exclusive is A minus its projection onto B -> orthogonal to B
    assert abs(float(np.dot(ch["a_exclusive"], b))) < 1e-9
    assert abs(float(np.dot(ch["b_exclusive"], a))) < 1e-9


def test_classify_types():
    rng = np.random.default_rng(4)
    base = rng.standard_normal(128); base /= np.linalg.norm(base)
    assert classify(opponent_channels(base, base * 1.01))["type"] == "redundant"
    a = np.zeros(128); a[0] = 1.0
    assert classify(opponent_channels(a, -a))["type"] == "contradictory"


def test_blend_is_unit_and_between():
    rng = np.random.default_rng(5)
    a = rng.standard_normal(256); b = rng.standard_normal(256)
    bl = blend(a, b, ratio=0.7)
    assert abs(np.linalg.norm(bl) - 1.0) < 1e-6


def test_agree_and_divergence_helpers():
    v = np.arange(1, 33, dtype=float)
    a = np.zeros(32); a[0] = 1.0
    b = np.zeros(32); b[1] = 1.0
    assert agree(v, v) and not agree(a, b)
    assert divergence(v, v) < 1e-6
    assert divergence(a, b) > 1.5


def test_decompose_alias_matches():
    rng = np.random.default_rng(6)
    a = rng.standard_normal(48); b = rng.standard_normal(48)
    assert np.allclose(decompose(a, b)["purple"], opponent_channels(a, b)["purple"])


def test_zero_vector_is_empty_not_crash():
    ch = opponent_channels(np.zeros(16), np.ones(16))
    assert ch["divergence_score"] == 0.0 and ch["channel_magnitudes"]["purple"] == 0.0
