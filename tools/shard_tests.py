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
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOW_WEIGHT = 20     # one slow-marked test costs about this many ordinary tests of wall time (proxy, see above)


def test_files():
    """Every test file, sorted -- the deterministic universe the shards partition."""
    return sorted(glob.glob(os.path.join(REPO, "tests", "test_*.py")))


#: MEASURED seconds per test file, when available. Written by
#: `python3 tools/shard_tests.py --measure` (which runs the suite with
#: --durations and records the totals) and committed alongside the sharder.
#: MEASUREMENT BEATS THE PROXY, and this project already knows it: the
#: count-based weight below treats a file with three 15-second tests as
#: WEIGHT 3 and a file with thirty fast ones as WEIGHT 30 -- so shard 3 drew
#: test_holographic_shader (17.4s + 15.0s in two tests), test_holographic_market
#: (15.0s + 15.0s + 13.4s) and test_holographic_scene (15.0s), SIX FILES COSTING
#: FOUR AND A HALF MINUTES, and timed out at 93% while the other three shards
#: finished. The bins were balanced by the WRONG UNIT.
DURATIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "test_durations.json")


def _measured():
    """{filename: seconds} if a measurement has been recorded, else {}."""
    try:
        with open(DURATIONS_FILE, encoding="utf-8") as f:
            return {str(k): float(v) for k, v in json.load(f).items()}
    except Exception:
        return {}


def weight(path, measured=None):
    """Integer cost for one file: MEASURED seconds when known, else the proxy.

    The proxy (ordinary tests count 1, slow-marked count SLOW_WEIGHT) stays as
    the fallback for files nobody has timed yet -- a new test must not vanish
    from the schedule just because it is unmeasured, and it must not silently
    weigh nothing either."""
    measured = _measured() if measured is None else measured
    secs = measured.get(os.path.basename(path))
    if secs is not None:
        # centiseconds, so the greedy packer stays integer and deterministic
        return max(1, int(round(secs * 100)))
    text = open(path, encoding="utf-8", errors="ignore").read()
    n_tests = len(re.findall(r"^def test_|^    def test_", text, flags=re.M))
    n_slow = text.count("pytest.mark.slow")
    # SAME UNIT AS THE MEASURED BRANCH, or the two cannot be packed together:
    # an unmeasured file would weigh 30 against a measured file's 4,000 and
    # every unmeasured file would land in one bin. 0.25 s per ordinary test is
    # the observed median; the slow multiplier is unchanged.
    return max(1, int(round(25 * (max(n_tests, 1) + SLOW_WEIGHT * n_slow))))


def partition(num_shards):
    """Greedy largest-first bin packing into `num_shards` bins; ties break by shard index (deterministic).
    Returns (shards, loads): shards is a list of file lists, loads the weight totals."""
    _m = _measured()
    files = [(weight(p, _m), p) for p in test_files()]
    # KEPT NEGATIVE -- SPLITTING OVERSIZED FILES INTO PER-TEST SELECTORS.
    # Written and REMOVED after measuring: the packer's spread across 4 bins is
    # 1.00x, i.e. PERFECTLY EVEN, so no file was ever the imbalance. The 19m51s
    # cancellation was a BIN COUNT problem -- ~2,200 s of local suite over 4
    # shards is 550 s each and CI runs ~3x slower with --run-slow, so 27
    # minutes. Ten bins fixes it arithmetically.
    # The splitter also broke the --selfcheck contract (it compares against a
    # universe of FILES, and node selectors are not files), which is the check
    # doing its job: a partition that no longer partitions files is a different
    # thing wearing the same name. RAISING --num-shards IS THE SANCTIONED FIX
    # and it needed no change here at all.
    files.sort(key=lambda wp: (-wp[0], wp[1]))            # heaviest first; path as the deterministic tiebreak
    shards = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for w, p in files:
        i = min(range(num_shards), key=lambda j: (loads[j], j))
        shards[i].append(p)
        loads[i] += w
    return shards, loads


def measure(files, run_slow=True, timeout=None):
    """RUN the given test files with --durations=0 and RECORD per-file wall seconds
    into DURATIONS_FILE, merging with whatever is already there (sweep 116).

    The flag was documented for a whole checkpoint arc and NEVER IMPLEMENTED -- the
    durations file was a one-time hand transcription of a CI log covering 51 of 680
    files, so 629 files rode the 0.25 s/test proxy and shard 3 of ten blew a
    20-minute job at 91% while the packer believed every bin was ~226 s. Balanced by
    the wrong unit, for the second time, because the measuring instrument did not
    exist. Chunk with --shard/--num-shards or --only-missing so a measurement fits
    a local wall clock; pytest prints the durations table only at exit, so a killed
    run records nothing (measure smaller chunks rather than longer ones)."""
    import subprocess, sys as _sys
    if not files:
        return 0
    cmd = [_sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
           "--durations=0", "--durations-min=0"] + (["--run-slow"] if run_slow else []) + list(files)
    env = dict(os.environ, PYTHONHASHSEED="0")
    if run_slow:
        env["LECORE_RUN_SLOW"] = "1"
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env).stdout
    except subprocess.TimeoutExpired as e:
        out = (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""))
        print("measure: chunk timed out -- nothing recorded for it; use smaller chunks")
    per_file = {}
    # lines look like:  "12.34s call     tests/test_x.py::test_y[param]"  (setup/teardown rows too)
    for line in out.splitlines():
        m_ = re.match(r"^\s*([0-9.]+)s\s+(call|setup|teardown)\s+(\S+?\.py)::", line)
        if m_:
            per_file[os.path.basename(m_.group(3))] = per_file.get(os.path.basename(m_.group(3)), 0.0) + float(m_.group(1))
    if not per_file:
        print("measure: no durations parsed (did pytest reach its summary?)")
        return 1
    merged = _measured()
    merged.update({k: round(v, 3) for k, v in per_file.items()})
    with open(DURATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(merged.items())), f, indent=1, sort_keys=True)
        f.write("\n")
    print("measure: recorded %d file(s) (%.0f s); durations file now covers %d" %
          (len(per_file), sum(per_file.values()), len(merged)))
    return 0


def merge_logs(paths):
    """CI PATH (sweep 116): each full-suite shard runs pytest with --durations=0 and
    saves its stdout; this merges every log's per-file totals into DURATIONS_FILE.
    The shards already run every file every time -- the measurement was always
    being made and never recorded. Measuring where the suite actually runs also
    removes the ci-factor guess: recorded seconds ARE CI seconds."""
    per_file = {}
    for lp in paths:
        try:
            text = open(lp, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for line in text.splitlines():
            m_ = re.match(r"^\s*([0-9.]+)s\s+(call|setup|teardown)\s+(\S+?\.py)::", line)
            if m_:
                k = os.path.basename(m_.group(3))
                per_file[k] = per_file.get(k, 0.0) + float(m_.group(1))
    if not per_file:
        print("merge-logs: no durations found in %d log(s)" % len(paths))
        return 1
    merged = _measured()
    merged.update({k: round(v, 3) for k, v in per_file.items()})
    with open(DURATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(merged.items())), f, indent=1, sort_keys=True)
        f.write("\n")
    print("merge-logs: %d file(s) from %d log(s); durations file now covers %d (%.0f s recorded)"
          % (len(per_file), len(paths), len(merged), sum(per_file.values())))
    return 0


def suggest_shards(budget_s, ci_factor):
    """The smallest shard count whose heaviest predicted bin fits the budget."""
    for n in range(1, 200):
        _sh, loads = partition(n)
        if max(loads) / 100.0 * ci_factor <= budget_s:
            return n, max(loads) / 100.0
    return 200, max(partition(200)[1]) / 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=None, help="which shard to print (0-based)")
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--report", action="store_true", help="print the balance table instead of a file list")
    ap.add_argument("--selfcheck", action="store_true", help="assert exact cover, disjointness, determinism")
    ap.add_argument("--measure", action="store_true",
                    help="run the selected files (all, or --shard i of --num-shards, or --only-missing) with "
                         "--durations and record per-file seconds into tools/test_durations.json")
    ap.add_argument("--only-missing", action="store_true", help="with --measure: only files without a record")
    ap.add_argument("--no-slow", action="store_true", help="with --measure: leave slow tests deselected")
    ap.add_argument("--chunk-timeout", type=int, default=None, help="with --measure: seconds per pytest run")
    ap.add_argument("--budget", type=float, default=None,
                    help="seconds a shard may take on CI; with --selfcheck the check FAILS if the heaviest "
                         "predicted bin x --ci-factor exceeds it (catch an overfull shard in the 12 s sanity "
                         "step, not after 20 minutes); with --suggest prints the smallest shard count that fits")
    ap.add_argument("--ci-factor", type=float, default=3.0, help="CI runner slowdown vs the measuring machine")
    ap.add_argument("--min-coverage", type=float, default=0.9,
                    help="with --selfcheck: fraction of test files that must carry a MEASURED duration; below "
                         "it the packer is balancing by proxy and the check fails with the remedy")
    ap.add_argument("--suggest", action="store_true", help="print the smallest --num-shards that fits --budget")
    ap.add_argument("--merge-logs", nargs="+", default=None,
                    help="merge per-file durations out of saved pytest --durations=0 logs (the CI path)")
    args = ap.parse_args()
    if args.merge_logs:
        raise SystemExit(merge_logs(args.merge_logs))
    if args.measure:
        files = test_files()
        if args.shard is not None:
            files = partition(args.num_shards)[0][args.shard]
        if args.only_missing:
            _m = _measured()
            files = [p for p in files if os.path.basename(p) not in _m]
        raise SystemExit(measure(files, run_slow=not args.no_slow, timeout=args.chunk_timeout))
    if args.suggest:
        if args.budget is None:
            raise SystemExit("--suggest needs --budget SECONDS")
        n, mx = suggest_shards(args.budget, args.ci_factor)
        print("smallest shard count fitting %.0f s at ci-factor %.1f: %d (heaviest bin %.0f s local)"
              % (args.budget, args.ci_factor, n, mx))
        raise SystemExit(0)

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
        # THE TWO GUARDS THAT WERE MISSING (sweep 116): (1) an exact, disjoint,
        # deterministic partition can still be OVERFULL -- so the heaviest bin is
        # checked against the CI budget here, where failing costs 12 seconds instead
        # of a 20-minute cancellation; (2) a partition balanced by the PROXY is not
        # balanced at all -- so measurement coverage is enforced, with the remedy.
        _m = _measured()
        covered = sum(1 for p in universe if os.path.basename(p) in _m) / max(len(universe), 1)
        if covered < args.min_coverage:
            print("FAIL: only %.0f%% of test files carry a measured duration (need %.0f%%) -- the packer is "
                  "balancing by proxy. Remedy: python tools/shard_tests.py --measure --only-missing "
                  "(chunk with --shard/--num-shards; commit tools/test_durations.json)"
                  % (100 * covered, 100 * args.min_coverage))
            return 1
        if args.budget is not None:
            heaviest = max(loads) / 100.0
            if heaviest * args.ci_factor > args.budget:
                n, _mx = suggest_shards(args.budget, args.ci_factor)
                print("FAIL: heaviest shard predicts %.0f s local x %.1f = %.0f s on CI > budget %.0f s. "
                      "Remedy: --num-shards %d (and the matrix must match)"
                      % (heaviest, args.ci_factor, heaviest * args.ci_factor, args.budget, n))
                return 1
        print("OK: %d files -> %d shards; exact cover, disjoint, deterministic; load spread %.0f%% of mean; "
              "measured coverage %.0f%%; heaviest bin %.0f s local"
              % (len(universe), args.num_shards, 100 * spread, 100 * covered, max(loads) / 100.0))
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
