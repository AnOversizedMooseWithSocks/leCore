"""Guards tools/shard_tests.py -- the CI test splitter. If the partition ever drops a file, double-assigns one, or
goes non-deterministic, some tests silently stop running in the weekly full-suite matrix. That is the worst kind of
CI failure (a green build that tested less than it claimed), so the invariants are pinned here and the workflow also
runs --selfcheck before every shard."""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "tools"))


def test_shards_exactly_cover_the_suite():
    import shard_tests
    for k in (2, 4, 7):                                   # the workflow uses 4; oddball counts must hold too
        shards, loads = shard_tests.partition(k)
        universe = set(shard_tests.test_files())
        seen = []
        for s in shards:
            seen.extend(s)
        assert len(seen) == len(set(seen)), "a file landed in two shards (k=%d)" % k
        assert set(seen) == universe, "shards do not exactly cover the test files (k=%d)" % k
        assert all(l > 0 for l in loads), "an empty shard wastes a whole CI job (k=%d)" % k


def test_shards_are_deterministic_and_balanced():
    import shard_tests
    a, loads = shard_tests.partition(4)
    b, _ = shard_tests.partition(4)
    assert [sorted(s) for s in a] == [sorted(s) for s in b], "partition must be identical across calls"
    # Balance: greedy largest-first keeps the spread tiny; a blowup means the weight proxy broke. Loose bound on
    # purpose -- this trips on a real regression (one shard 2x another), not on suite growth.
    assert max(loads) <= 1.5 * min(loads), ("shard load spread blew up", loads)


def test_this_file_is_in_some_shard():
    """The self-referential smoke check: the splitter must see the file that tests it."""
    import shard_tests
    shards, _ = shard_tests.partition(4)
    me = os.path.abspath(__file__)
    assert any(me in s for s in shards)
