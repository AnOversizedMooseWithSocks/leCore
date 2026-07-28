"""Regression traps for decision-safe quantization (work plan item 1.4, Gate B).

The headline is a MECHANISM, not a number: flip rate is governed by MARGIN, not by corpus size or bit
width. Well-separated queries survive 2-bit quantization; queries midway between two documents are unsafe
at 8. Both halves are pinned, plus the confound that makes a uint8-vs-float8 verdict impossible on the
shipped index -- because the confounded numbers looked like a clean win and would have been easy to ship.
"""
import numpy as np
import pytest

import lecore
from holographic.misc.holographic_ratedistortion import (crowded_subset, decision_flip_rate,
                                                        quantize_float8, quantize_uniform)


@pytest.fixture(scope="module")
def index():
    """The real shipped routing index, dequantized -- the artifact the plan's Gate B is about."""
    import pathlib
    p = pathlib.Path("lecore_data/routing/index_128d.npz")
    if not p.is_file():
        pytest.skip("no routing index shipped")
    d = np.load(p, allow_pickle=True)
    q = d["q"].astype(np.float64)
    lo, hi = d["lo"].astype(np.float64), d["hi"].astype(np.float64)
    return lo + (hi - lo) * (q / 255.0)


def _rows(V, n, seed=0):
    return V[np.random.default_rng(seed).choice(V.shape[0], size=n, replace=False)]


def test_normal_queries_are_decision_safe_on_the_shipped_index(index):
    # The plan's Gate B, run on the real artifact. Even heavily noised queries must not flip.
    scale = np.abs(index).std()
    rng = np.random.default_rng(1)
    noisy = _rows(index, 200, seed=1) + 0.6 * scale * rng.standard_normal((200, index.shape[1]))
    r = decision_flip_rate(index, noisy, bits=8, mode="uniform")
    assert r["flip_rate"] == 0.0
    assert r["margin_median"] > 0.4


def test_ambiguous_queries_collapse_the_margin(index):
    # THE MECHANISM. Midpoints between two documents are ambiguous by construction; their margins must
    # collapse relative to ordinary queries. This is why flip rate is not predictable from N and bits.
    amb = 0.5 * (_rows(index, 200, seed=2) + _rows(index, 200, seed=3))
    normal = _rows(index, 200, seed=4)
    r_amb = decision_flip_rate(index, amb, bits=8, mode="uniform")
    r_norm = decision_flip_rate(index, normal, bits=8, mode="uniform")
    assert r_amb["margin_median"] < 0.2 * r_norm["margin_median"]
    assert r_amb["flip_rate"] > r_norm["flip_rate"]


def test_well_separated_queries_survive_aggressive_quantization(index):
    # The surprising half, and the useful one: normal queries hold at 2 BITS. If this ever regresses, the
    # "margin governs, not bit width" claim is wrong and the docstring must change.
    normal = _rows(index, 150, seed=5)
    for bits in (8, 4, 2):
        assert decision_flip_rate(index, normal, bits=bits, mode="uniform")["flip_rate"] == 0.0


def test_flip_rate_is_monotone_in_coarseness(index):
    # Sanity on the instrument itself: a coarser code cannot be safer. If it is, the probe is broken.
    amb = 0.5 * (_rows(index, 200, seed=6) + _rows(index, 200, seed=7))
    fine = decision_flip_rate(index, amb, bits=8, mode="uniform")["flip_rate"]
    coarse = decision_flip_rate(index, amb, bits=2, mode="uniform")["flip_rate"]
    assert coarse >= fine


def test_the_uint8_float8_comparison_is_confounded_on_this_index(index):
    # THE CONFOUND, PINNED. The shipped index is ALREADY uint8, so uniform re-quantization is a no-op while
    # float8 genuinely re-quantizes. Any verdict taken here measures the source grid, not the quantizers --
    # and the confounded numbers looked like a clean 7.5x win for uint8. Asserted so the comparison cannot
    # be quietly revived on this artifact; it needs a float32 corpus.
    requantized, _ = quantize_uniform(index, bits=8)
    assert np.abs(requantized - index).max() < 1e-9, "the index is no longer on the uint8 grid"
    assert np.abs(quantize_float8(index) - index).max() > 1e-6, "float8 became a no-op too"


def test_float8_preserves_sign_and_flushes_subnormals():
    x = np.array([0.0, 1.0, -1.0, 0.03, -0.02])
    f = quantize_float8(x)
    assert f[0] == 0.0
    assert np.all(np.sign(f[1:]) == np.sign(x[1:]))
    # 4 exponent bits => smallest normal magnitude 2^-7; below that it disappears, which matters for
    # decision safety because a small component does not lose precision, it vanishes.
    assert quantize_float8(np.array([-1e-9]))[0] == 0.0


def test_crowded_subset_is_deterministic_and_sized(index):
    a, b = crowded_subset(index, 40), crowded_subset(index, 40)
    assert a.shape == (40, index.shape[1]) and np.array_equal(a, b)


def test_guards():
    V = np.random.default_rng(0).standard_normal((20, 8))
    with pytest.raises(ValueError):
        decision_flip_rate(V, np.zeros((3, 5)))
    with pytest.raises(ValueError):
        decision_flip_rate(V, V[:2], mode="not-a-mode")


def test_wired_and_discoverable(index):
    mind = lecore.UnifiedMind(dim=128, seed=0)
    r = mind.decision_flip_rate(index, _rows(index, 40, seed=8), bits=8)
    assert set(("flip_rate", "margin_median", "margin_p05", "margin_min")) <= set(r)
    assert mind.crowded_subset(index, 20).shape[0] == 20
    for query in ("does quantization change the answer", "is this index decision safe",
                  "how few bits can i use for retrieval"):
        assert "Decision-safe" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces decision-safe quantization" % query
