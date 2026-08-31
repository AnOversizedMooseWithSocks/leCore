"""The memory mountain: leCore measures its own cache hierarchy, and the tiers predict the
benchmarks.

The classic instrument (Bryant & O'Hallaron's "memory mountain"), kept deliberately simple:
streaming bandwidth of a dot product as the working set sweeps from cache-resident to
RAM-resident. What it bought on the box that ran the fast-arbiter benchmarks: peak 89 GB/s
at ~512 KB (L2-resident), a clear knee through 1-4 MB, and a ~26 GB/s floor from 4 MB out
(L3 and RAM indistinguishable on a virtualized host -- reported as ONE floor, honestly,
rather than inventing a boundary the data does not show). PREDICTION, the point of the
instrument: bytes_touched / floor_bandwidth reproduced the measured Index numbers -- exact
f64 predicted 9.1 vs 10.4 ms measured, f32 4.5 vs 5.1, screens-f32 1.6 vs 1.9 -- the whole
fast-path table is the mountain wearing different working sets.

KEPT NEGATIVE (the instrument's own blind spot, named not hidden): the LEFT flank ascends
(20 -> 89 GB/s) because a Python-dispatched BLAS call is OVERHEAD-bound below ~256 KB --
this probe measures dispatch there, not L1. A Python-level instrument resolves the L2 / L3 /
RAM regimes and CANNOT see L1; anyone quoting the small-size numbers as cache bandwidth is
reading the instrument, not the machine.
"""
import time

import numpy as np


def measure_memory_mountain(sizes=None, repeats=3, target_seconds=0.06):
    """Streaming-bandwidth curve: [(working_set_bytes, GB_per_s_median, lo, hi), ...].
    Deterministic protocol (fixed sizes, median of `repeats`), honest spread reported --
    bandwidth is a physical measurement, so the variance travels with the number."""
    if sizes is None:
        sizes = [32e3, 64e3, 128e3, 256e3, 512e3, 1e6, 2e6, 4e6, 8e6,
                 16e6, 32e6, 64e6, 128e6]
    out = []
    for nbytes in sizes:
        n = max(1024, int(nbytes // 16))                 # two f64 operands
        x = np.ones(n)
        y = np.ones(n)
        reps = max(3, int(target_seconds / max(2 * n * 8 / 2.0e10, 1e-6)))
        runs = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(reps):
                x @ y
            runs.append(2 * n * 8 * reps / (time.perf_counter() - t0) / 1e9)
        out.append((float(nbytes), float(np.median(runs)), float(min(runs)), float(max(runs))))
    return out


def detect_tiers(curve, overhead_floor_bytes=256e3):
    """Read the regimes off the curve: peak tier (fastest cache the dispatch can see), the
    knee (largest relative bandwidth drop after the peak), and the floor (median of the last
    three points = the streaming tier every big matvec lives in). Sizes below
    `overhead_floor_bytes` are EXCLUDED from the peak search -- that flank is dispatch
    overhead, the pinned blind spot."""
    pts = [(s, b) for s, b, _, _ in curve if s >= overhead_floor_bytes]
    peak_size, peak_bw = max(pts, key=lambda p: p[1])
    after = [(s, b) for s, b in pts if s >= peak_size]
    knee = None
    worst = 1.0
    for (s0, b0), (s1, b1) in zip(after, after[1:]):
        if b1 / b0 < worst:
            worst = b1 / b0
            knee = (s1, b1)
    floor_bw = float(np.median([b for _, b in pts[-3:]]))
    return {"peak_bytes": peak_size, "peak_gbs": peak_bw,
            "knee_bytes": knee[0] if knee else None, "knee_gbs": knee[1] if knee else None,
            "floor_gbs": floor_bw,
            "note": "floor = L3/RAM merged when the drop past the knee is gradual; a "
                    "virtualized host often shows no separate RAM cliff -- one floor is the "
                    "honest reading"}


def predict_streaming_ms(nbytes_touched, tiers):
    """The instrument's payoff: predicted wall-clock (ms) for a streaming pass over
    `nbytes_touched`, from the measured floor bandwidth. The fast-arbiter table validated
    this to ~15% (exact f64 9.1 predicted / 10.4 measured; f32 4.5/5.1; screens 1.6/1.9)."""
    return float(nbytes_touched) / (tiers["floor_gbs"] * 1e9) * 1e3


def _selftest():
    curve = measure_memory_mountain(sizes=[64e3, 256e3, 512e3, 1e6, 4e6, 16e6, 64e6],
                                    repeats=2, target_seconds=0.03)
    tiers = detect_tiers(curve)
    # planted truths of any real memory hierarchy, asserted not assumed:
    assert tiers["peak_gbs"] > tiers["floor_gbs"] * 1.5, \
        "a machine whose cache is not faster than its RAM is a broken instrument, not a machine"
    assert tiers["peak_bytes"] <= 4e6, "peak must sit in a cache-sized working set"
    # the prediction must be self-consistent: predicted time for the largest measured size,
    # from the floor, within 60% of the measured time implied by its own bandwidth (loose
    # band -- this is a physical measurement on a shared box, and the pin must not flake)
    s_big, b_big, _, _ = curve[-1]
    pred = predict_streaming_ms(s_big, tiers)
    meas = s_big / (b_big * 1e9) * 1e3
    assert 0.4 < pred / meas < 2.5, (pred, meas)
    # V16 pin: the floor must predict a REAL matvec, not merely its own curve (measured
    # ratio 1.02 when this pin was written; the band is wide because the box is shared)
    A = np.ones((1500, 1500))
    q = np.ones(1500)
    A @ q
    t0 = time.perf_counter()
    for _ in range(8):
        A @ q
    meas_ms = (time.perf_counter() - t0) / 8 * 1e3
    pred_ms = predict_streaming_ms(A.nbytes, tiers)
    assert 0.3 < pred_ms / meas_ms < 3.0, (pred_ms, meas_ms)
    print("OK: holographic_memorymountain self-test passed (peak %.0f GB/s @ %.0f KB, floor "
          "%.0f GB/s; prediction self-consistent; dispatch flank excluded by design)"
          % (tiers["peak_gbs"], tiers["peak_bytes"] / 1e3, tiers["floor_gbs"]))


if __name__ == "__main__":
    _selftest()
