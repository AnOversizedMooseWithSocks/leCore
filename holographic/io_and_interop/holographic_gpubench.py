"""M1-1 -- the host<->device crossover benchmark (holographic_gpubench).

WHY THIS EXISTS AS CODE RATHER THAN AS INSTRUCTIONS
---------------------------------------------------
Exactly one number blocks the rest of the compute backlog: WHERE DOES A DEVICE START WINNING. Everything
else is wired and waiting on it — `cleanup_batch(backend='wgsl')` and `wgsl_bind_batch` exist,
`should_offload` gates them, `place_work` composes the decision and `resource_policy` caps it — but
`should_offload`'s thresholds are ARITHMETIC FROM PCIe BANDWIDTH, not measurements, and they are marked
provisional everywhere they surface.

Nobody has been able to take that measurement because no environment this engine has run in has had a real
GPU. So the useful thing to build is not another kernel: it is the thing that turns "please benchmark it"
into one call, for whoever eventually has hardware.

THE TRAP THIS HARNESS EXISTS TO AVOID
-------------------------------------
GPU calls are ASYNCHRONOUS. Timing a dispatch without forcing completion measures KERNEL LAUNCH, not kernel
execution, and produces numbers that look spectacular and are wrong — a classic and very easy mistake. Every
device timing here READS THE RESULT BACK before stopping the clock, which forces completion through the
wgpu buffer-mapping path. That is slower than an ideal `device.poll()` would be, and it is honest: it
measures the round trip a real caller actually pays, transfer included, which is the number
`should_offload` needs.

WHAT IT DELIBERATELY DOES NOT DO
  * It does not pick a winner for you. It reports the crossover and the ratios; changing a default is a
    separate, deliberate act.
  * It does not run on a software adapter and pretend the answer means something. `llvmpipe` and `WARP` are
    CPU adapters — a timing there is NumPy against a CPU driver emulating a GPU — so the report FLAGS
    itself as meaningless when the adapter type is not a real GPU, rather than emitting a plausible table.
"""

import time

import numpy as np


def _ms(fn, repeats=5):
    """Best-of-N wall time in ms. BEST, not mean: we want the machine's capability, and a slow sample is
    always contamination (a scheduler hiccup, a driver warm-up) rather than signal."""
    fn()
    best = float("inf")
    for _ in range(int(repeats)):
        start = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - start) * 1e3)
    return best


def crossover(kind="cleanup", dims=(512, 1024), counts=(256, 1024, 4096), batches=(1, 8, 64, 256),
              repeats=5, seed=0):
    """Sweep CPU against the device and report where the device starts winning.

    `kind` is 'cleanup' (codebook similarity + argmax) or 'bind' (batched circular convolution) -- the two
    VSA kernels with a device path today.

    Returns {adapter, trustworthy, rows, crossover, note}. `rows` is one dict per configuration with the CPU
    time, the device time, the ratio and the byte count. `crossover` is the smallest `n_bytes` at which the
    device won, or None if it never did -- WHICH IS A RESULT, not a failure, and should be published as one.

    THE ANSWER FEEDS should_offload: replace MIN_BYTES_PROVISIONAL with the measured crossover and
    MIN_INTENSITY_PROVISIONAL with the arithmetic intensity at that point."""
    from holographic.io_and_interop.holographic_wgpurun import device_info

    # VALIDATE THE ARGUMENT BEFORE THE SWEEP, NOT INSIDE IT. The per-row `except` below exists for an
    # unsupported SHAPE -- a non-power-of-two dim that `bind` refuses, say -- so one bad cell does not lose
    # the whole table. It must not also swallow a MISTYPED ARGUMENT: `kind='cleanupp'` would otherwise
    # produce a full table of error rows and a cheerful "crossover: never", which reads like a finding and
    # is actually a typo. A caught test caught exactly that.
    if kind not in ("cleanup", "bind"):
        raise ValueError("kind must be 'cleanup' or 'bind', got %r" % (kind,))

    info = device_info()
    trustworthy = bool(info.get("available")) and str(info.get("type", "")).upper() != "CPU"
    if not info.get("available"):
        return {"adapter": info, "trustworthy": False, "rows": [], "crossover": None,
                "note": "no compute adapter; nothing was measured"}

    rows = []
    rng = np.random.default_rng(int(seed))
    for dim in dims:
        for count in counts:
            book = rng.standard_normal((count, dim)).astype(np.float32)
            book /= np.linalg.norm(book, axis=1, keepdims=True)
            for batch in batches:
                queries = rng.standard_normal((batch, dim)).astype(np.float32)
                try:
                    cpu_ms, gpu_ms = _time_pair(kind, book, queries, dim, batch, repeats)
                except Exception as exc:                # an unsupported shape is a skipped row, not a crash
                    rows.append({"dim": dim, "count": count, "batch": batch, "error": str(exc)[:80]})
                    continue
                n_bytes = int(book.nbytes + queries.nbytes)
                rows.append({"dim": dim, "count": count, "batch": batch, "n_bytes": n_bytes,
                             "cpu_ms": cpu_ms, "gpu_ms": gpu_ms,
                             "ratio": (cpu_ms / gpu_ms) if gpu_ms else float("inf")})

    winners = [r for r in rows if r.get("ratio", 0) > 1.0]
    point = min((r["n_bytes"] for r in winners), default=None)
    return {
        "adapter": info, "trustworthy": trustworthy, "rows": rows, "crossover": point,
        "note": ("MEANINGLESS ON THIS ADAPTER: %s is a CPU adapter, so these timings are NumPy against a "
                 "CPU driver emulating a GPU. Run on real hardware." % info.get("device")
                 if not trustworthy else
                 "measured on %s; feed `crossover` into should_offload's MIN_BYTES_PROVISIONAL"
                 % info.get("device")),
    }


def _time_pair(kind, book, queries, dim, batch, repeats):
    """CPU and device timings for one configuration. EVERY DEVICE CALL READS ITS RESULT BACK, which forces
    completion -- see the module docstring on why timing an async dispatch without that is the classic way
    to produce a spectacular wrong number."""
    from holographic.io_and_interop.holographic_wgpurun import bind_batch_kernel, matmul_kernel

    if kind == "cleanup":
        cpu = _ms(lambda: np.argmax(queries @ book.T, axis=1), repeats)
        gpu = _ms(lambda: matmul_kernel(book, queries), repeats)
        return cpu, gpu
    if kind == "bind":
        other = queries[: max(1, batch)]
        left = book[: other.shape[0]]
        cpu = _ms(lambda: np.fft.irfft(np.fft.rfft(left, axis=1) * np.fft.rfft(other, axis=1),
                                       n=dim, axis=1), repeats)
        gpu = _ms(lambda: bind_batch_kernel(left, other), repeats)
        return cpu, gpu
    raise ValueError("kind must be 'cleanup' or 'bind', got %r" % (kind,))


def crossover_report(result):
    """The sweep as a readable table, with the trustworthiness banner FIRST so nobody quotes a software-
    adapter run as a hardware result.

    NAMED crossover_report, not `report`: holographic_measure.report(name, stats, floor) already owns that
    name for a different job (formatting a mean +/- CI stats dict). The name-collision budget MAY SHRINK AND
    MUST NEVER GROW, so the newer arrival takes the qualified name rather than spending budget on a homonym."""
    lines = [result["note"], ""]
    lines.append("%6s %7s %7s %12s %11s %11s %8s"
                 % ("dim", "count", "batch", "bytes", "cpu ms", "gpu ms", "ratio"))
    lines.append("-" * 70)
    for row in result["rows"]:
        if "error" in row:
            lines.append("%6d %7d %7d   %s" % (row["dim"], row["count"], row["batch"], row["error"]))
            continue
        lines.append("%6d %7d %7d %12d %11.4f %11.4f %7.2fx"
                     % (row["dim"], row["count"], row["batch"], row["n_bytes"],
                        row["cpu_ms"], row["gpu_ms"], row["ratio"]))
    lines.append("")
    lines.append("crossover: %s" % ("never -- the device did not win at any tested size (a RESULT, publish it)"
                                    if result["crossover"] is None else "%d bytes" % result["crossover"]))
    return "\n".join(lines)


def _selftest():
    out = crossover(kind="cleanup", dims=(64,), counts=(32,), batches=(1, 4), repeats=1)
    assert set(out) == {"adapter", "trustworthy", "rows", "crossover", "note"}

    if out["adapter"].get("available"):
        assert out["rows"], "an available adapter produced no rows"
        for row in out["rows"]:
            if "error" not in row:
                assert row["cpu_ms"] > 0 and row["gpu_ms"] > 0

        # THE HONESTY GUARD: a software adapter must say so, loudly, in the note that prints first.
        if str(out["adapter"].get("type", "")).upper() == "CPU":
            assert out["trustworthy"] is False
            assert "MEANINGLESS" in out["note"]

    text = report(out)
    assert "crossover:" in text and out["note"].split(":")[0] in text

    try:
        crossover(kind="nonsense", dims=(64,), counts=(32,), batches=(1,), repeats=1)
    except ValueError:
        pass                                    # reached only when an adapter exists to get that far

    print("holographic_gpubench: selftest passed (adapter honesty, rows, report)")


if __name__ == "__main__":
    _selftest()
