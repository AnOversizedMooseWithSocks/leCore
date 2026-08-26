"""G3 -- the exact, partition-invariant prefix sum (scan): the GPU workhorse, made reproducible.

THE DATA MATTERS HERE MORE THAN ANYWHERE. A scan of uniform [0,1) values hides the whole problem: float addition is
nearly exact on well-scaled positive data, and every blocking agrees to 1e-12. The failure only shows on data with
DYNAMIC RANGE or CANCELLATION, so all three live in this suite:

    data                       seq vs 4-block   4-block vs 7-block
    uniform [0, 1)                  2.05e-12            1.14e-12
    16 orders of magnitude          6.26e-07            3.87e-07
    [1e16, 1.0, -1e16] repeated     0.00e+00            **9.20e+01**

THE HONEST FRAMING. 92.0 sounds catastrophic; the amplitude there is 1e16, so it is 9.2e-15 RELATIVE. Both blockings
are accurate; they simply disagree. Absolute disagreement is what matters when a prefix sum is a ledger, a counter,
or a checkpoint hash.

AND THE KEPT NEGATIVE, stated first in the docstring and pinned below: `scan_exact` is **not more accurate** than
`np.cumsum`. It is more **reproducible**. A sequential cumsum wins on precision every time; it just cannot run on
eight blocks and give the same bits.
"""

import math

import numpy as np
import pytest

from holographic.scene_and_pipeline.holographic_distribute import (
    exact_scale, scan_exact, scan_exact_blocked)


N = 4096


def uniform():
    """Well-scaled positive data. The case where float scanning is already fine -- included so the suite cannot
    claim a win that only exists on pathological input."""
    return np.random.default_rng(0).random(N)


def wide():
    """16 orders of magnitude: the realistic farm/physics case. Float non-associativity bites on DYNAMIC RANGE."""
    rng = np.random.default_rng(1)
    return rng.normal(size=N) * 10.0 ** rng.integers(-8, 8, size=N)


def cancellation():
    """[1e16, 1, -1e16] repeated: the 1.0 survives only if it is added before the -1e16 cancels the 1e16."""
    return np.array([1e16, 1.0, -1e16] * (N // 3) + [1.0])


ALL = (("uniform", uniform), ("wide", wide), ("cancellation", cancellation))


def _blocked_float(x, k):
    """The standard parallel scan: per-block cumsum, then add the exclusive prefix of the block sums."""
    blocks = np.array_split(x, k)
    local = [np.cumsum(b) for b in blocks]
    sums = np.array([b.sum() for b in blocks])
    carry = np.concatenate([[0.0], np.cumsum(sums)[:-1]])
    return np.concatenate([l + c for l, c in zip(local, carry)])


def _fsum_prefix(x):
    """Ground truth: an exactly-rounded running sum. Slow, and the only honest reference."""
    return np.array([math.fsum(x[:i + 1]) for i in range(len(x))])


# ---------------------------------------------------------------------------------------------------------
# the premise: a BLOCKED float scan disagrees with itself
# ---------------------------------------------------------------------------------------------------------

def test_the_data_actually_exposes_the_problem():
    # Guard against a suite that passes because its input is too easy.
    assert np.abs(wide()).max() / max(np.abs(wide()).min(), 1e-300) > 1e10   # genuinely wide
    c = cancellation()
    assert np.abs(np.cumsum(c)).max() > 1e15                                  # the amplitude is enormous
    assert abs(math.fsum(c) - float(len(c) // 3) - 1.0) < 1e-6                # ... and the truth is small


@pytest.mark.parametrize("name,gen", ALL)
def test_a_blocked_float_scan_depends_on_the_block_count(name, gen):
    x = gen()
    assert not np.array_equal(_blocked_float(x, 4), _blocked_float(x, 7))


def test_the_cancellation_case_disagrees_by_ninety_two_absolute_but_not_relatively():
    x = cancellation()
    diff = float(np.abs(_blocked_float(x, 4) - _blocked_float(x, 7)).max())
    assert diff > 50.0                                    # measured 92.0
    amp = float(np.abs(np.cumsum(x)).max())
    assert diff / amp < 1e-13                             # ... which is 9.2e-15 RELATIVE. Say both numbers.


# ---------------------------------------------------------------------------------------------------------
# the fix: an int64 scan is blocking-invariant
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,gen", ALL)
def test_the_exact_scan_is_bit_identical_for_every_block_count(name, gen):
    x = gen()
    base = scan_exact(x)
    for k in (1, 2, 4, 7, 13, 256, len(x)):
        assert np.array_equal(scan_exact_blocked(x, k), base), (name, k)


def test_the_exact_scan_agrees_with_its_own_sequential_form():
    x = wide()
    assert np.array_equal(scan_exact_blocked(x, 1), scan_exact(x))


def test_the_last_element_of_the_scan_is_the_exact_total():
    for _name, gen in ALL:
        x = gen()
        assert scan_exact(x)[-1] == pytest.approx(np.sum(np.rint(x * exact_scale(float(np.abs(x).max()), x.size))
                                                         ).astype(np.float64)
                                                  / exact_scale(float(np.abs(x).max()), x.size))


def test_scan_edge_cases():
    assert scan_exact(np.array([])).shape == (0,)
    assert np.array_equal(scan_exact(np.zeros(5)), np.zeros(5))
    assert np.array_equal(scan_exact_blocked(np.zeros(5), 3), np.zeros(5))
    assert scan_exact(np.array([2.0]))[0] == pytest.approx(2.0)
    # a 2-D input is raveled, as np.cumsum's default does
    assert scan_exact(np.ones((2, 3))).shape == (6,)


def test_scan_exact_blocked_handles_more_blocks_than_elements():
    x = np.arange(1.0, 4.0)
    assert np.array_equal(scan_exact_blocked(x, 99), scan_exact(x))


# ---------------------------------------------------------------------------------------------------------
# THE KEPT NEGATIVE: reproducible, not accurate
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_the_exact_scan_is_less_accurate_than_a_sequential_cumsum():
    # The claim this suite must NOT make. np.cumsum is sequential and very accurate; quantization costs precision.
    for name, gen in (("uniform", uniform), ("wide", wide)):
        x = gen()
        truth = _fsum_prefix(x)
        amp = max(float(np.abs(truth).max()), 1.0)
        e_exact = float(np.abs(scan_exact(x) - truth).max()) / amp
        e_np = float(np.abs(np.cumsum(x) - truth).max()) / amp
        assert e_np < e_exact, (name, e_np, e_exact)      # the sequential float scan WINS on precision
        assert e_exact < 1e-9                              # ... and the exact scan is still perfectly usable


def test_what_the_exact_scan_actually_buys_is_bit_identity_under_blocking():
    x = wide()
    # the float scan is more accurate per-blocking, and disagrees across blockings
    assert not np.array_equal(_blocked_float(x, 4), _blocked_float(x, 7))
    # the exact scan is slightly less accurate, and agrees exactly
    assert np.array_equal(scan_exact_blocked(x, 4), scan_exact_blocked(x, 7))


def test_more_bits_buys_back_precision():
    x = wide()
    truth = _fsum_prefix(x)
    amp = float(np.abs(truth).max())
    coarse = float(np.abs(scan_exact(x, bits=20) - truth).max()) / amp
    fine = float(np.abs(scan_exact(x, bits=40) - truth).max()) / amp
    assert fine < coarse                                   # the quantization is the whole error term
    # ... and both are still blocking-invariant, which is the property that does not degrade
    assert np.array_equal(scan_exact_blocked(x, 7, bits=20), scan_exact(x, bits=20))


def test_values_below_the_scale_resolution_round_to_zero():
    # The stated trade: a bounded dynamic range. A 1e-30 next to a 1e16 vanishes, and that is not a bug.
    x = np.array([1e16, 1e-30, 1e-30])
    out = scan_exact(x)
    assert out[1] == out[0] == out[2]                       # the tiny terms contributed nothing


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    x = cancellation()
    assert np.array_equal(m.scan_exact_blocked(x, 7), m.scan_exact(x))
    assert np.array_equal(m.scan_exact_blocked(x, 4), m.scan_exact_blocked(x, 7))
    assert "Partition-invariant" in str(m.find_capability("prefix sum of an array")[:3])


def test_the_scan_reuses_the_same_monoid_as_the_reduce():
    # G3 does not grow a second opinion about exact arithmetic: it calls `exact_scale`, the same global-scale rule
    # `reduce_sum_exact_partitioned` uses (max and len are partition-invariant, so every block derives it alone).
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)
    x = wide()
    total_from_scan = m.scan_exact(x)[-1]
    total_from_reduce = m.reduce_sum_exact_partitioned([[v] for v in x])
    assert abs(float(total_from_scan) - float(np.atleast_1d(total_from_reduce)[0])) < 1e-6 * max(abs(x).max(), 1.0)
