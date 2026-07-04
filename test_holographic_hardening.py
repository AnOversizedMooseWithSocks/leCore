"""Tests for holographic_hardening (R5: retry, redundant-compute voting, canaries, straggler backups)."""
import numpy as np
from holographic_hardening import (agree, retrying, NoConsensus, CanaryFailed, HardenedCoordinator,
                                   run_with_backups, _sum_bucket, _FlakyBackend)
from holographic_coordinator import InProcessBackend
from holographic_distribute import reduce_sum, reduce_min


def test_agree_majority():
    assert agree([5.0, 5.0, 7.0]) == 5.0
    assert agree([5.0, 5.0000000001, 5.0], tol=1e-6) == 5.0
    assert agree(["red", "red", "blue"]) == "red"


def test_agree_arrays():
    a = np.array([1.0, 2.0, 3.0])
    assert np.allclose(agree([a, a.copy(), a + 10.0]), a)


def test_agree_no_consensus():
    for bad in ([1.0, 2.0, 3.0], [1.0, 2.0], []):
        try:
            agree(bad); assert False
        except NoConsensus:
            pass


def test_retrying_succeeds_after_failures():
    state = {"n": 0}
    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("nope")
        return 42
    assert retrying(flaky, attempts=3, backoff=0.001) == 42


def test_retrying_gives_up():
    def always_fail():
        raise ValueError("down")
    try:
        retrying(always_fail, attempts=2, backoff=0.001); assert False
    except ValueError:
        pass


def test_hardened_retry_reissues():
    flaky = _FlakyBackend(fail_times=2)                        # fails the first 2 submits, succeeds on the 3rd
    hc = HardenedCoordinator(flaky, redundancy=1, attempts=3, backoff=0.001)
    assert hc.run([[1, 2, 3]], _sum_bucket, reduce=reduce_sum) == 6.0


def test_redundant_voting_accepts_agreement():
    hc = HardenedCoordinator(InProcessBackend(), redundancy=3, attempts=1)
    assert hc.run([[1, 2, 3], [4, 5]], _sum_bucket, reduce=reduce_sum) == 15.0


def test_canary_rejects_untrusted_worker():
    liar = _FlakyBackend(wrong_for=([1, 2, 3], 999.0))        # returns a wrong answer for the canary bucket
    hc = HardenedCoordinator(liar, redundancy=1, attempts=1)
    try:
        hc.run([[9]], _sum_bucket, canaries=[([1, 2, 3], 6.0)]); assert False
    except CanaryFailed:
        pass


def test_canary_passes_honest_worker():
    hc = HardenedCoordinator(InProcessBackend(), redundancy=1, attempts=1)
    got = hc.run([[10]], _sum_bucket, canaries=[([1, 2, 3], 6.0)], reduce=reduce_sum)
    assert got == 10.0                                         # canary passed, real work proceeded


def test_straggler_backups_stay_correct():
    got = run_with_backups(InProcessBackend(), [[1, 2], [3, 4], [5]], _sum_bucket, reduce=reduce_sum, grace=0.0)
    assert got == 15.0


def test_min_reduce_with_hardening():
    def _min_b(b, c):
        return float(np.min(b))
    hc = HardenedCoordinator(InProcessBackend(), redundancy=2)
    assert hc.run([[3.0, 1.0], [2.0, 0.5]], _min_b, reduce=reduce_min) == 0.5
