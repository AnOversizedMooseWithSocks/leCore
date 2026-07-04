"""holographic_hardening.py -- R5: fault tolerance + verification for the distributed coordinator.

WHY
---
A trusted local pool needs none of this. The moment work runs on machines you do not fully control -- a flaky LAN
node, or a public SETI@home-style contributor -- three things become mandatory (the BOINC / SETI@home discipline):

  1. RETRY + REASSIGN  -- a node can drop a task or die. Retry with backoff; a reissue naturally lands on whatever
     node the backend picks next, so a dead node's buckets get reassigned.
  2. REDUNDANT COMPUTE + VOTING -- a node can faithfully return a plausible WRONG answer, which the repair layer
     (cleanup/fountain/verify) does NOT catch (it repairs corruption, not a confident lie). So run each bucket on
     several INDEPENDENT workers and accept only the answer they AGREE on. Never trust a single public node.
  3. CANARY BUCKETS -- spot-check with a bucket whose answer you already know; a node that gets the canary wrong is
     untrusted, and the whole run is rejected before its results are used.

This module adds those on top of the existing Coordinator backends (local pool / network farm / command), reusing the
same submit(worker, bucket, handle) -> future interface. Straggler/backup execution is here too (reissue a slow task
to an idle worker and take whichever finishes first) -- correctness-preserving; its wall-time benefit shows on a
multi-core / multi-node setup.

KEPT NEGATIVES (loud)
  * Voting defends against a node returning a DIFFERENT wrong answer. It cannot catch every node computing the SAME
    wrong answer (a shared bug) -- that is why canaries (known answers) complement voting.
  * REDUNDANCY costs compute: redundancy=3 triples the work. Use it for UNTRUSTED nodes; a trusted pool runs
    redundancy=1 (no voting) and just uses retry.
  * Straggler backups help wall-time only when there are IDLE workers and more than one core/node; on a single core
    they just add work. Off by default.
"""
import time

import numpy as np

from holographic_distribute import reduce_sum


class NoConsensus(Exception):
    """Redundant workers did not agree on a bucket's result -- the coordinator abstains rather than guess."""


class CanaryFailed(Exception):
    """A known-answer canary bucket came back wrong -- a worker/node is untrusted, so the run is rejected."""


# ============================================================================================================
# Voting -- the heart of untrusted-node defense.
# ============================================================================================================
def _close(a, b, tol):
    """Approximate equality across the result types workers return: arrays (allclose), numbers (within tol), else =="""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return a.shape == b.shape and np.allclose(a, b, atol=tol, rtol=0.0)
    if isinstance(a, (int, float, np.floating, np.integer)) and isinstance(b, (int, float, np.floating, np.integer)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def agree(results, tol=1e-9, quorum=None):
    """Return the value a MAJORITY of `results` agree on (within `tol` for numbers/arrays), or raise NoConsensus.
    Groups the results by approximate equality, takes the largest group, and requires it to reach `quorum` (default:
    a strict majority, floor(n/2)+1). This is the accept-only-on-agreement rule: one node cannot force a result."""
    if not results:
        raise NoConsensus("no results to vote on")
    if quorum is None:
        quorum = len(results) // 2 + 1                     # strict majority by default
    groups = []                                            # each: [representative_value, count]
    for r in results:
        for g in groups:
            if _close(r, g[0], tol):
                g[1] += 1
                break
        else:
            groups.append([r, 1])
    rep, count = max(groups, key=lambda g: g[1])
    if count < quorum:
        raise NoConsensus("no majority agreement among %d results (best %d/%d, need %d)"
                          % (len(results), count, len(results), quorum))
    return rep


# ============================================================================================================
# Retry with backoff (a reissue reassigns to whatever node the backend picks next).
# ============================================================================================================
def retrying(make_and_wait, attempts=3, backoff=0.1):
    """Call make_and_wait() (submit + wait for a result); on any exception wait backoff*2^k and try again, up to
    `attempts`. `make_and_wait` re-submits each time, so a retry naturally lands on a fresh worker/node."""
    last = None
    for k in range(attempts):
        try:
            return make_and_wait()
        except Exception as e:                             # a dropped task / dead node -> back off and reissue
            last = e
            if k < attempts - 1:
                time.sleep(backoff * (2 ** k))
    raise last


# ============================================================================================================
# The hardened coordinator -- retry + redundant voting + canaries, over any backend.
# ============================================================================================================
class HardenedCoordinator:
    """Like Coordinator, but every bucket is run with RETRY, optionally REDUNDANTLY with VOTING, and the run can be
    gated on CANARY buckets first. Use redundancy>1 + canaries for untrusted nodes; redundancy=1 for a trusted pool
    (then it is just retry)."""

    def __init__(self, backend, redundancy=1, attempts=3, backoff=0.1, tol=1e-9, quorum=None):
        self.backend = backend
        self.redundancy = max(1, int(redundancy))
        self.attempts = max(1, int(attempts))
        self.backoff = backoff
        self.tol = tol
        self.quorum = quorum

    def run(self, buckets, worker, cache=None, reduce=reduce_sum, canaries=None):
        """Publish the cache once, (optionally) verify canaries, resolve every bucket with retry + voting, and reduce
        the agreed parts with the monoid reducer. Releases the cache even on error."""
        handle = self.backend.publish_cache(cache)
        try:
            if canaries:
                self._check_canaries(worker, handle, canaries)     # preflight before trusting any real result
            parts = [self._resolve_bucket(worker, b, handle) for b in buckets]
        finally:
            self.backend.release_cache(handle)
        return reduce(parts)

    def close(self):
        self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- internals ------------------------------------------------------------------------------------------
    def _resolve_bucket(self, worker, bucket, handle):
        """Run one bucket `redundancy` times (independent workers), each with retry, then VOTE. redundancy==1 skips
        the vote (a trusted pool)."""
        results = [self._one_result(worker, bucket, handle) for _ in range(self.redundancy)]
        if self.redundancy == 1:
            return results[0]
        return agree(results, tol=self.tol, quorum=self.quorum)

    def _one_result(self, worker, bucket, handle):
        """One result for a bucket, retried on failure. Each attempt re-submits, so a retry reassigns the work."""
        return retrying(lambda: self.backend.submit(worker, bucket, handle).result(),
                        attempts=self.attempts, backoff=self.backoff)

    def _check_canaries(self, worker, handle, canaries):
        """Run each known-answer canary; if any comes back wrong, reject the whole run (a node is untrusted)."""
        for bucket, expected in canaries:
            got = self._one_result(worker, bucket, handle)
            if not _close(got, expected, self.tol):
                raise CanaryFailed("canary bucket returned %r, expected %r -- worker/node untrusted" % (got, expected))


# ============================================================================================================
# Straggler / backup execution (speculative) -- reissue a slow task, take whichever finishes first.
# ============================================================================================================
def run_with_backups(backend, buckets, worker, cache=None, reduce=reduce_sum, grace=0.5, poll=0.02):
    """Submit every bucket, then reissue any task still unfinished after `grace` seconds as a BACKUP; take whichever
    copy finishes first. Correctness-preserving (the buckets are deterministic monoid work, so either copy is fine).
    KEPT NEGATIVE: this helps WALL-TIME only with idle workers on more than one core/node; on a single core it just
    adds work -- so it is opt-in, not the default path."""
    handle = backend.publish_cache(cache)
    try:
        primary = [backend.submit(worker, b, handle) for b in buckets]
        deadline = time.perf_counter() + grace
        backups = [None] * len(buckets)
        parts = [None] * len(buckets)
        pending = set(range(len(buckets)))
        while pending:
            for i in list(pending):
                f = primary[i]
                b = backups[i]
                if f.done():
                    parts[i] = f.result()
                    pending.discard(i)
                elif b is not None and b.done():
                    parts[i] = b.result()
                    pending.discard(i)
                elif b is None and time.perf_counter() > deadline:
                    backups[i] = backend.submit(worker, buckets[i], handle)   # speculative backup for a straggler
            if pending:
                time.sleep(poll)
        return reduce(parts)
    finally:
        backend.release_cache(handle)


# ---- module-level workers for the self-test (top-level so a process backend could pickle them) -------------
def _sum_bucket(bucket, cache):
    return float(np.sum(bucket))


class _FlakyBackend:
    """A tiny in-process backend that FAILS a worker's first `fail_times` calls (to exercise retry) and can return a
    WRONG answer for a chosen bucket on a chosen call (to exercise voting). Deterministic, for tests only."""

    def __init__(self, fail_times=0, wrong_for=None):
        self.calls = 0
        self.fail_times = fail_times
        self.wrong_for = wrong_for                          # (bucket_tuple, wrong_value) or None

    def publish_cache(self, cache):
        return ("direct", cache)

    def submit(self, worker, bucket, handle):
        self.calls += 1
        idx = self.calls
        wrong = self.wrong_for

        class _F:
            def result(_self):
                if idx <= self.fail_times:
                    raise RuntimeError("simulated node failure #%d" % idx)
                if wrong is not None and tuple(bucket) == tuple(wrong[0]):
                    return wrong[1]
                return worker(bucket, handle[1])
        return _F()

    def release_cache(self, handle):
        pass

    def close(self):
        pass


def _selftest():
    # (1) voting: a clear majority wins; a tie / no-majority abstains
    assert agree([5.0, 5.0, 7.0]) == 5.0
    assert agree([5.0, 5.0000000001, 5.0], tol=1e-6) == 5.0
    try:
        agree([1.0, 2.0, 3.0]); assert False                 # three different answers -> no majority
    except NoConsensus:
        pass

    # (2) retry: a backend that fails the first 2 attempts still returns on the 3rd
    from holographic_coordinator import InProcessBackend
    flaky = _FlakyBackend(fail_times=2)
    hc = HardenedCoordinator(flaky, redundancy=1, attempts=3, backoff=0.001)
    assert hc.run([[1, 2, 3]], _sum_bucket, reduce=reduce_sum) == 6.0

    # (3) redundant voting: 3 honest copies agree -> accepted
    good = HardenedCoordinator(InProcessBackend(), redundancy=3, attempts=1)
    assert good.run([[1, 2, 3], [4, 5]], _sum_bucket, reduce=reduce_sum) == 15.0

    # (4) canary catches a bad worker: expected 6 for [1,2,3], but this backend returns a wrong value for it
    liar = _FlakyBackend(wrong_for=([1, 2, 3], 999.0))
    guard = HardenedCoordinator(liar, redundancy=1, attempts=1)
    try:
        guard.run([[9]], _sum_bucket, canaries=[([1, 2, 3], 6.0)]); assert False
    except CanaryFailed:
        pass

    # (5) straggler backups still produce the correct reduction (correctness, not speed, is what we assert)
    got = run_with_backups(InProcessBackend(), [[1, 2], [3, 4], [5]], _sum_bucket, reduce=reduce_sum, grace=0.0)
    assert got == 15.0

    print("OK: holographic_hardening self-test passed (majority voting + abstain, retry-with-backoff, redundant "
          "accept-on-agreement, canary rejects an untrusted worker, straggler backups stay correct -- R5)")


if __name__ == "__main__":
    _selftest()
