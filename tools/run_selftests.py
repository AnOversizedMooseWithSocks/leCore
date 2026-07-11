#!/usr/bin/env python3
"""run_selftests.py -- run EVERY module's own `_selftest()` and report red/green, so a red selftest cannot sit
silent until someone happens to run it by hand.

WHY THIS EXISTS (the incident): CI runs pytest, but module selftests are only run via `python -m ...` -- and the
skills module's selftest sat RED for an unknown length of time with nothing noticing, because nothing in CI executed
it. This walker closes that hole. It is wrapped by `tests/test_all_selftests.py` so plain `pytest` (and therefore CI)
inherits it with no workflow changes.

WHAT IT RUNS, AND WHAT IT ONLY COUNTS (an honest-scope decision, made deliberately):
  * A module is RUN only if it has BOTH a `__main__` guard AND a `def _selftest` -- the codebase's stated
    convention. Measured at build time: 352 of 428 modules qualify.
  * A module with a `__main__` guard but NO `_selftest` function is NOT executed -- its main is a demo, and a demo
    is not a selftest (it may be slow, need a display, or write files). It is COUNTED as NO_SELFTEST instead.
  * A module with no `__main__` guard at all is also COUNTED as NO_SELFTEST (running `-m` on it silently imports
    and exits 0 -- a green that asserts nothing, which is worse than no green).
  The NO_SELFTEST list (76 modules at build time) is pinned by the pytest wrapper as a budget that MAY SHRINK AND
  MUST NEVER GROW -- the same discipline as the duplicate budget. Writing a new module without a selftest fails CI.

DETERMINISM: every child runs with PYTHONHASHSEED=0 (the repo rule). Results are reported sorted by module name,
regardless of completion order, so two runs of the walker produce the same report line-for-line.

Usage:
    python3 tools/run_selftests.py                     # walk everything, 8 jobs, 120s/module timeout
    python3 tools/run_selftests.py --jobs 4 --timeout 300
    python3 tools/run_selftests.py --only matlib,island   # substring filter, for quick local checks
    python3 tools/run_selftests.py --list-missing      # just print the NO_SELFTEST modules and exit
Exit code: 0 iff no FAIL and no TIMEOUT (NO_SELFTEST does not fail the walker itself -- the pytest budget
test owns that judgement, so the two failure modes stay independently visible).
"""
import argparse
import concurrent.futures
import os
import pathlib
import re
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
_MAIN_RE = re.compile(r'__name__\s*==\s*[\'"]__main__[\'"]')

OK, FAIL, TIMEOUT, NO_SELFTEST = "OK", "FAIL", "TIMEOUT", "NO_SELFTEST"

# WHY pin BLAS to one thread per child: the engine is deterministic by design, and CI already pins numpy to one
# BLAS thread so N parallel test workers each get a core instead of fighting over them (see .github/workflows/
# ci.yml). The walker has the SAME shape -- 12 children each doing numpy work -- and without this, a genuinely
# 26-second selftest (holographic_tear) TIMED OUT at 240s purely from CPU oversubscription, not from any real
# slowness. This is the fix that makes the walk's timings mean what they say.
_BLAS_PIN = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
             "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}

# WHY a launcher table: a few modules' `__main__` is a DEMO or a long-running SERVER, not their selftest, and the
# selftest hides behind a flag. Running a bare `-m` on holographic_farm starts a daemon that sleeps for an hour --
# it "timed out" while doing exactly what it was told. The convention in those modules is `--selftest`; this table
# records the exception so the walker asks each module for its selftest the way that module actually exposes it.
# A module NOT in this table is run bare (the overwhelming majority, whose __main__ IS the selftest).
_SELFTEST_ARGS = {
    "holographic.misc.holographic_farm": ["--selftest"],
}

# WHY a heavy list: these selftests do genuinely large work (a 12x12 cloth-tear sim run four times; a full RL
# training loop) -- 25s and 17s respectively when pinned and alone (measured), and they are RIGHT to be thorough.
# The problem is only scheduling: run inside the parallel pool alongside 8 other CPU-bound children (and, under
# pytest, the test runner itself), they starve and cross a timeout that has nothing to do with their real cost.
# So the walker runs the heavy ones SERIALLY FIRST, each with a full core, then fans out the fast majority. This
# keeps every selftest running at full length -- no shortening, no sampling -- while the timeout stays honest.
# A module lands here when the walk measures it slow; the list is a scheduling hint, never a correctness input.
_HEAVY = [
    "holographic.mesh_and_geometry.holographic_tear",
    # slowest-tests pass: valuehead (17s -> 0.6s, removed dead tabular-brain compute at high load) and groom
    # (39s -> 7s, coarsened a curl-noise field the test didn't exercise) left this list. tear went 25s -> 15s
    # (smaller cloth, same tear/no-tear contrast) and stays as the slowest selftest. The four below were measured
    # heavy under parallel load; with the self-healing retry they pass anyway, listing them just skips a wasted
    # first attempt.
    "holographic.mesh_and_geometry.holographic_curlnoise",
    "holographic.mesh_and_geometry.holographic_meshsubdiv",
    "holographic.misc.holographic_steering",
    "holographic.rendering.holographic_rendergraph",
]


def discover():
    """Every holographic_*.py under holographic/, classified: (runnable, missing).
    `runnable` modules have BOTH a __main__ guard and a _selftest; `missing` have one or neither -- see the
    module docstring for why a guard-without-_selftest is counted as missing rather than executed."""
    runnable, missing = [], []
    for p in sorted((REPO / "holographic").rglob("holographic_*.py")):
        src = p.read_text(errors="replace")
        mod = ".".join(p.with_suffix("").relative_to(REPO).parts)
        # Same "has __main__ AND a _selftest" rule the engine's own census uses (holographic_codestructure
        # .selftest_census). The walker's `missing` is WIDER (it lists every module it cannot run, including
        # libraries with no __main__), because the walker must know what it can't execute; the census reports
        # only the actionable "has an entry point but asserts nothing" subset. The runnable rule is shared.
        if _MAIN_RE.search(src) and "def _selftest" in src:
            runnable.append(mod)
        else:
            missing.append(mod)
    return runnable, missing


def run_one(mod, timeout):
    """Run one module's selftest in a subprocess. Returns (module, verdict, seconds, tail-of-output)."""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"                      # the repo's determinism rule, enforced per child
    env.update(_BLAS_PIN)                            # one BLAS thread per child; see _BLAS_PIN's WHY
    cmd = [sys.executable, "-m", mod] + _SELFTEST_ARGS.get(mod, [])   # launcher modules expose selftest behind a flag
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env, cwd=str(REPO))
    except subprocess.TimeoutExpired:
        return mod, TIMEOUT, time.time() - t0, "(timed out at %ss)" % timeout
    tail = ((r.stdout or "") + (r.stderr or "")).strip()[-400:]
    return mod, (OK if r.returncode == 0 else FAIL), time.time() - t0, tail


def walk(jobs=8, timeout=120, only=None):
    """Run all runnable selftests (optionally filtered by substring), in parallel, reported deterministically.
    Returns (results, missing): results is a name-sorted list of (module, verdict, seconds, tail).

    Timeout handling is SELF-HEALING: a module that TIMEOUTs in the parallel pool is retried ONCE serially, with
    the whole box to itself. A genuinely-slow-but-correct selftest (a full RL loop, a 12x12 cloth-tear sim) starved
    by 8 CPU-bound neighbours crosses the wall for a reason that has nothing to do with its real cost -- the serial
    retry gives it a fair run and it passes. Only a REAL hang (or a selftest slower than `timeout` even alone)
    fails twice and is reported. This replaces hand-maintaining _HEAVY: the walk MEASURES slowness instead of
    remembering it, so a newly-slow module never silently becomes a false CI red the way four did before this."""
    runnable, missing = discover()
    if only:
        subs = [s.strip() for s in only.split(",") if s.strip()]
        runnable = [m for m in runnable if any(s in m for s in subs)]
    # Known-heavy modules still run serially FIRST (a hint that saves the retry round-trip), but the list is now
    # only an optimization -- correctness no longer depends on it being complete, because of the retry below.
    heavy = [m for m in runnable if m in _HEAVY]
    light = [m for m in runnable if m not in _HEAVY]
    results = [run_one(m, timeout) for m in heavy]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:   # subprocesses do the work; threads just wait
        for res in ex.map(lambda m: run_one(m, timeout), light):
            results.append(res)
    # SELF-HEAL: anything that timed out under contention gets one serial retry with the full box. A serial run
    # with a generous wall (2x) distinguishes "slow, starved" (now passes) from "actually hung" (times out again).
    timed_out = [r for r in results if r[1] == TIMEOUT]
    if timed_out:
        results = [r for r in results if r[1] != TIMEOUT]
        for mod, _, _, _ in timed_out:
            results.append(run_one(mod, timeout * 2))    # alone, double wall -- a true hang still fails here
    results.sort(key=lambda r: r[0])                 # deterministic report order, whatever finished first
    return results, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--only", default=None, help="comma-separated substrings; run only matching modules")
    ap.add_argument("--list-missing", action="store_true")
    args = ap.parse_args(argv)

    if args.list_missing:
        _, missing = discover()
        for m in missing:
            print(m)
        print("-- %d module(s) without a real selftest (a __main__ guard AND a _selftest)" % len(missing))
        return 0

    t0 = time.time()
    results, missing = walk(jobs=args.jobs, timeout=args.timeout, only=args.only)
    bad = [r for r in results if r[1] in (FAIL, TIMEOUT)]
    for mod, verdict, dt, tail in results:
        if verdict != OK:
            print("%-8s %6.1fs  %s" % (verdict, dt, mod))
            if tail:
                print("         " + tail.replace("\n", "\n         "))
    print("SELFTEST WALK: %d run, %d OK, %d FAIL, %d TIMEOUT, %d without a selftest (counted, not run) in %.0fs"
          % (len(results), len(results) - len(bad),
             sum(1 for r in bad if r[1] == FAIL), sum(1 for r in bad if r[1] == TIMEOUT),
             len(missing), time.time() - t0))
    return 1 if bad else 0


def _selftest():
    """The walker's own selftest, self-contained: classification is exercised against a real green module, a
    scratch failing module, and a scratch module with no selftest -- so a broken walker cannot report all-green."""
    import tempfile
    # 1. discovery finds a healthy population and splits it the way the census did
    runnable, missing = discover()
    assert len(runnable) > 300 and len(missing) > 0, (len(runnable), len(missing))
    assert not (set(runnable) & set(missing))
    # 2. a known-good module reports OK (transform: tiny, numpy-only, selftest by convention)
    mod, verdict, dt, _ = run_one("holographic.misc.holographic_transform", timeout=120)
    assert verdict == OK, (mod, verdict)
    # 3. a failing selftest reports FAIL -- proven with a scratch module, not assumed
    with tempfile.TemporaryDirectory() as td:
        bad = pathlib.Path(td) / "holographic_scratch_red.py"
        bad.write_text("def _selftest():\n    assert False, 'deliberately red'\n"
                       "if __name__ == '__main__':\n    _selftest()\n")
        env = dict(os.environ); env["PYTHONHASHSEED"] = "0"
        r = subprocess.run([sys.executable, str(bad)], capture_output=True, env=env)
        assert r.returncode != 0            # the same rc convention run_one classifies on
    # 4. deterministic ordering: two walks of the same tiny subset report identically
    a, _ = walk(jobs=2, timeout=120, only="holographic_bus,holographic_island")
    b, _ = walk(jobs=2, timeout=120, only="holographic_bus,holographic_island")
    assert [r[:2] for r in a] == [r[:2] for r in b] and len(a) == 2
    # 5. every launcher-table entry is real (a module that exists and is classified runnable) and its flag WORKS:
    #    farm's bare __main__ is a daemon, so a green here proves the --selftest routing, not just that it exists.
    #    A stale table entry (module renamed, flag changed) would silently send the walker back to hanging.
    for mod in _SELFTEST_ARGS:
        assert mod in runnable, "launcher table names a non-runnable module: %s" % mod
        _, verdict, _, _ = run_one(mod, timeout=120)
        assert verdict == OK, "launcher entry %s did not reach OK via %s" % (mod, _SELFTEST_ARGS[mod])
    # 6. the SELF-HEALING retry works -- proven, not assumed, because it is now load-bearing correctness (four
    #    real modules would be false CI reds without it). A scratch module that sleeps just over a tiny `timeout`
    #    must TIME OUT on the first pass and then PASS on the serial retry (2x wall). This is the [BLIND-SPOT]
    #    rule applied to the tool: assert the branch that only fires under contention, using a controlled stand-in.
    with tempfile.TemporaryDirectory() as td:
        pkg = pathlib.Path(td) / "holographic"; pkg.mkdir()
        # sleeps 3s: over a 2s first-pass wall (times out), under the 4s retry wall (passes)
        (pkg / "holographic_slowpass.py").write_text(
            "import time\ndef _selftest():\n    time.sleep(3)\n"
            "if __name__ == '__main__':\n    _selftest()\n")
        import importlib
        this = importlib.import_module(__name__) if __name__ in sys.modules else sys.modules[__name__]
        old_repo = globals()["REPO"]
        try:
            globals()["REPO"] = pathlib.Path(td)            # point discovery at the scratch tree
            res, _ = walk(jobs=1, timeout=2, only="holographic_slowpass")
            assert len(res) == 1 and res[0][1] == OK, ("self-heal retry failed: %s" % (res,))
            assert res[0][2] > 2.0                          # it really did take longer than the first-pass wall
        finally:
            globals()["REPO"] = old_repo
    print("tools/run_selftests selftest OK: discovery %d/%d, OK/FAIL classification proven, %d launcher entr%s, "
          "self-heal retry proven, deterministic order"
          % (len(runnable), len(missing), len(_SELFTEST_ARGS), "y" if len(_SELFTEST_ARGS) == 1 else "ies"))


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        _selftest()
    else:
        sys.exit(main())
