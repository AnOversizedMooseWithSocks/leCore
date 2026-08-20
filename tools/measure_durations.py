#!/usr/bin/env python3
"""measure_durations.py -- record REAL per-file test wall time for the sharder.

WHY THIS EXISTS. shard_tests.py balances by a STATIC PROXY: a file weighs its
number of `def test_` functions plus 20 per `pytest.mark.slow`. That proxy
cannot see wall time, and the gap showed up exactly where it hurts -- CI's
`full-suite (1)` job ran 19m51s and was CANCELLED at 91% against a 20-minute
budget while the other three shards finished comfortably. THE SUITE WAS NOT TOO
BIG; ONE SHARD WAS TOO HEAVY.

The sharder ALREADY blends measured seconds when tools/test_durations.json has
an entry -- it held TEN FILES out of 672, and not test_integration.py, which is
the single heaviest thing in the tree at ~280 s. So the one file most able to
blow a shard was the one the balancer was blindest to.

HOW: run pytest with --durations=0, which prints per-TEST timings, and SUM THEM
PER FILE. Setup and teardown count, because a fixture that costs 8 seconds costs
the shard 8 seconds regardless of which phase the report calls it.

    python3 tools/measure_durations.py                 # whole suite (slow)
    python3 tools/measure_durations.py --shard 0 --num-shards 24
    python3 tools/measure_durations.py --merge         # keep existing entries

MEASURED NUMBERS GO STALE, AND THAT IS FINE -- the blend degrades to the proxy
for anything unmeasured, so a partial file is strictly better than none. Re-run
it when a shard starts creeping toward the budget.
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "test_durations.json")

#: "12.34s call     tests/test_foo.py::test_bar" -- the shape pytest prints for
#: --durations. The phase word varies (call/setup/teardown) and all three are
#: real time the shard has to spend.
_LINE = re.compile(r"^\s*([\d.]+)s\s+\w+\s+(tests/[\w./]+\.py)::")


def collect(argv):
    """Run pytest and return {basename: total seconds}."""
    cmd = [sys.executable, "-m", "pytest"] + list(argv) + [
        "-q", "--durations=0", "--durations-min=0.01"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    totals = collections.defaultdict(float)
    for line in (proc.stdout or "").splitlines():
        m = _LINE.match(line)
        if m:
            totals[os.path.basename(m.group(2))] += float(m.group(1))
    return totals, proc.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--merge", action="store_true",
                    help="keep existing entries not measured this run")
    ap.add_argument("--min-seconds", type=float, default=1.0,
                    help="do not record files below this (the proxy is fine "
                         "for them, and noise is worse than absence)")
    a = ap.parse_args(argv)

    if a.shard is not None and a.num_shards:
        sel = subprocess.run(
            [sys.executable, os.path.join(HERE, "shard_tests.py"),
             "--shard", str(a.shard), "--num-shards", str(a.num_shards)],
            cwd=ROOT, capture_output=True, text=True).stdout.split()
    else:
        sel = ["tests/"]

    totals, rc = collect(sel)
    print("measured %d file(s), pytest rc=%d" % (len(totals), rc))

    out = {}
    if a.merge and os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            out = json.load(f)
    for k, v in totals.items():
        if v >= a.min_seconds:
            # KEEP THE LARGER NUMBER when merging. A file measured in one shard
            # can come out lower than the same file measured elsewhere (parallel
            # workers, deselected slow tests, a warm cache), and the two branches
            # that merged here disagreed 6x on one file. UNDER-ESTIMATING IS WHAT
            # BLOWS A 20-MINUTE BUDGET; over-estimating only packs conservatively,
            # so the max is the safe direction and --merge now says so.
            out[k] = max(round(v, 1), float(out.get(k, 0.0))) if a.merge \
                else round(v, 1)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(out.items())), f, indent=1)
        f.write("\n")
    top = sorted(out.items(), key=lambda kv: -kv[1])[:8]
    print("wrote %s (%d entries). Heaviest:" % (OUT, len(out)))
    for k, v in top:
        print("  %7.1fs  %s" % (v, k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
