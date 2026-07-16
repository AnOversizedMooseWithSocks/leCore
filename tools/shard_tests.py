#!/usr/bin/env python3
"""shard_tests.py -- split the test suite into N balanced, deterministic shards for CI.

WHY THIS EXISTS. GitHub's `timeout-minutes` budget applies PER JOB, and the true full suite (slow tests selected,
watchdog lifted via --run-slow) does not fit in one 20-minute job -- which is why the weekly/tag "full" run kept
tripping the timeout, and why the 15 s per-test watchdog was doing double duty as a runtime cap. Sharding fixes the
budget the right way: a matrix of K jobs each runs 1/K of the suite with its OWN 20-minute budget, in parallel, and
`--run-slow` can finally mean what it says. tools/select_tests.py answers "which tests does THIS CHANGE touch?";
this answers "give me slice i of K of EVERYTHING" -- same output convention (whitespace-separated paths on stdout)
so the workflow feeds either straight to pytest.

DETERMINISTIC BY CONSTRUCTION (same discipline as the engine): files are sorted, weights are integers derived from
the file text alone, and the greedy largest-first bin-packing breaks ties by shard index -- so shard i of K is the
same set on every machine, every run, no state file to go stale.

THE WEIGHT IS A PROXY, kept honest: wall time isn't knowable statically, so a file weighs (its number of `def
test_` functions) + SLOW_WEIGHT x (its number of `pytest.mark.slow` marks). Slow-marked tests are the ones that
dominate a full run (that is the marker's definition), so they are weighted as ~20 ordinary tests. If a shard still
runs long, the fix is raising --num-shards in ci.yml, not tuning this file.

Run:
    python3 tools/shard_tests.py --shard 0 --num-shards 4      # stdout: the files in shard 0
    python3 tools/shard_tests.py --report --num-shards 4       # balance table, human-readable
    python3 tools/shard_tests.py --selfcheck --num-shards 4    # exact-cover / disjoint / determinism asserts
"""
import argparse
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOW_WEIGHT = 20     # one slow-marked test costs about this many ordinary tests of wall time (proxy, see above)


def test_files():
    """Every test file, sorted -- the deterministic universe the shards partition."""
    return sorted(glob.glob(os.path.join(REPO, "tests", "test_*.py")))


def weight(path):
    """Integer cost proxy for one file: ordinary tests count 1, slow-marked tests count SLOW_WEIGHT."""
    text = open(path, encoding="utf-8", errors="ignore").read()
    n_tests = len(re.findall(r"^def test_|^    def test_", text, flags=re.M))
    n_slow = text.count("pytest.mark.slow")
    return max(n_tests, 1) + SLOW_WEIGHT * n_slow


def partition(num_shards):
    """Greedy largest-first bin packing into `num_shards` bins; ties break by shard index (deterministic).
    Returns (shards, loads): shards is a list of file lists, loads the weight totals."""
    files = [(weight(p), p) for p in test_files()]
    files.sort(key=lambda wp: (-wp[0], wp[1]))            # heaviest first; path as the deterministic tiebreak
    shards = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for w, p in files:
        i = min(range(num_shards), key=lambda j: (loads[j], j))
        shards[i].append(p)
        loads[i] += w
    return shards, loads


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=None, help="which shard to print (0-based)")
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--report", action="store_true", help="print the balance table instead of a file list")
    ap.add_argument("--selfcheck", action="store_true", help="assert exact cover, disjointness, determinism")
    args = ap.parse_args()

    shards, loads = partition(args.num_shards)

    if args.selfcheck:
        universe = set(test_files())
        seen = set()
        for s in shards:
            for p in s:
                assert p not in seen, "file assigned to two shards: %s" % p
                seen.add(p)
        assert seen == universe, "shards do not exactly cover the test files"
        again, _ = partition(args.num_shards)
        assert [sorted(s) for s in shards] == [sorted(s) for s in again], "partition is not deterministic"
        spread = (max(loads) - min(loads)) / max(sum(loads) / len(loads), 1)
        print("OK: %d files -> %d shards; exact cover, disjoint, deterministic; load spread %.0f%% of mean"
              % (len(universe), args.num_shards, 100 * spread))
        return 0

    if args.report:
        print("shard  files  weight")
        for i, (s, l) in enumerate(zip(shards, loads)):
            print("%5d  %5d  %6d" % (i, len(s), l))
        print("total  %5d  %6d" % (sum(len(s) for s in shards), sum(loads)))
        return 0

    if args.shard is None or not (0 <= args.shard < args.num_shards):
        ap.error("--shard must be in [0, %d)" % args.num_shards)
    print(" ".join(os.path.relpath(p, REPO) for p in shards[args.shard]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
