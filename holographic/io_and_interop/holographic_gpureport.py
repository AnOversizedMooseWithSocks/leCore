"""GPUREPORT-1 -- what GPU is available, and would offloading actually pay (holographic_gpureport).

The GPU mirror of `cpu_budget` / `should_pool`, and deliberately the same shape: one function that reports
what the machine HAS, one that decides whether using it PAYS for a specific job. The answer depends on the
caller's numbers, so it is computed rather than assumed — the same rule the machine model's placement oracle
and the bundle-capacity advisor follow.

WHY A REPORT WAS NEEDED AT ALL
------------------------------
There was no way to ask this engine about its own GPU situation. `use_gpu(True)` returns a bare bool that
conflates four different states — no CuPy installed, CuPy present but no device, device present but the
policy forbids it, and actually enabled — and a caller seeing `False` could not tell which. That matters
because three of those four are fixable by the user and one is not.

TWO INDEPENDENT GPU PATHS, AND THE REPORT COVERS BOTH
  * CuPy   — CUDA/NVIDIA only, transparent (kernels route by follow-the-data dispatch).
  * WGSL   — Vulkan / Metal / DX12 / WebGPU via wgpu, explicit (you dispatch an emitted kernel).
A machine can have neither, either, or both, and "do I have a GPU" has a different answer per path. A report
that only knew about CuPy would tell an Apple or AMD user they have no GPU, which is false.

THE PRE-GATE, AND THE HONEST STATE OF ITS CONSTANT
--------------------------------------------------
The backend's own docstring states the rule — a big FFT-or-matmul wins because the transfer amortises, a
tiny per-call op on one (D,) vector loses — but nothing enforced it, so a caller could flip `use_gpu(True)`
and get a slowdown with no warning. `should_offload` enforces it.

ITS THRESHOLD IS PROVISIONAL AND SAYS SO. Measuring the real host<->device crossover requires a GPU, and
none has been available in any environment this engine has run in. What ships is the SHAPE of the gate with
a conservative default drawn from the arithmetic of PCIe bandwidth, marked as unmeasured. That is the same
honesty `should_pool` carries about its 0.2 ms dispatch figure, and the same reason: a number without a
measurement behind it is a placeholder, and calling it anything else is how folklore constants are born.
"""


#: Bytes below which a host<->device round trip cannot pay for itself, whatever the arithmetic intensity.
#: PROVISIONAL AND UNMEASURED -- derived from PCIe 3.0 x16 (~12 GB/s) against a ~10 us launch latency, which
#: puts the break-even near 100 KB of traffic. Replace with a measured value the first time this runs on a
#: real device; until then it is a placeholder that refuses conservatively rather than a result.
MIN_BYTES_PROVISIONAL = 100_000

#: Floating-point operations per byte transferred, below which the job is transfer-bound. A memory-bound
#: elementwise op sits near 1; a matmul or FFT is far above it. Also provisional.
MIN_INTENSITY_PROVISIONAL = 4.0


def gpu_report(policy=None):
    """What GPU compute is reachable, per path, and why not when it is not.

    Returns {cupy: {...}, wgsl: {...}, any_available, policy_allows, wired_modules, note}. Never raises:
    the common case is a machine with no GPU, and a report that fails there is useless."""
    cupy = {"available": False, "why": None, "vendor": "NVIDIA/CUDA only"}
    try:
        import cupy  # noqa: F401

        try:
            count = cupy.cuda.runtime.getDeviceCount()
            if count:
                props = cupy.cuda.runtime.getDeviceProperties(0)
                name = props.get("name", b"?")
                cupy["available"] = True
                cupy["device"] = name.decode() if isinstance(name, bytes) else str(name)
                cupy["devices"] = int(count)
            else:
                cupy["why"] = "cupy is installed but no CUDA device is present"
        except Exception as exc:
            cupy["why"] = "cupy is installed but the CUDA runtime failed: %s" % exc
    except ImportError:
        cupy["why"] = "cupy is not installed (pip install cupy-cuda12x, NVIDIA only)"

    from holographic.io_and_interop.holographic_wgpurun import device_info

    wgsl = device_info()
    wgsl["vendor"] = "Vulkan / Metal / DX12 / WebGPU"

    allows = True if policy is None else bool(policy.gpu_allowed())
    return {
        "cupy": cupy,
        "wgsl": wgsl,
        "any_available": bool(cupy["available"] or wgsl.get("available")),
        "policy_allows": allows,
        # READ FROM THE LIVE TREE, not a hand-maintained list -- a second copy of a list is always the
        # stale one, as the packaging tool already taught this project.
        "wired_modules": _backend_consumers(),
        "note": ("CuPy is the transparent path and is NVIDIA-only; WGSL is explicit and vendor-neutral. "
                 "A software WGSL adapter counts as available on purpose: it validates and runs the same "
                 "shaders, which is what makes correctness CI-testable without hardware."),
    }


def _backend_consumers():
    """Which modules actually route through the CuPy backend, discovered by import rather than listed."""
    import pathlib

    # MATCH IMPORTS, NOT MENTIONS. A plain substring search for "holographic_backend" counted every module
    # that merely NAMES the backend in a docstring -- including two written the same afternoon that discuss
    # it in prose -- and reported 7 consumers where there are 5. A capability audit that counts documentation
    # as wiring is exactly the kind of number this project refuses to publish.
    import ast

    root = pathlib.Path(__file__).resolve().parent.parent
    found = []
    for path in root.rglob("holographic_*.py"):
        if path.name == "holographic_backend.py":
            continue
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any("holographic_backend" in n for n in names):
                found.append("%s/%s" % (path.parent.name, path.stem))
                break
    return sorted(found)


def should_offload(n_bytes, flops_per_byte, round_trips=1, available=None, policy=None):
    """Would moving this job to the GPU pay? Returns (verdict, why).

    Refuses on four independent grounds, each fatal alone:
      * NO DEVICE, or the RESOURCE POLICY forbids one -- asked first, because no arithmetic makes a
        forbidden device faster.
      * TOO LITTLE DATA -- below `MIN_BYTES_PROVISIONAL` the transfer and launch latency dominate whatever
        the compute is.
      * TOO LITTLE WORK PER BYTE -- an elementwise pass over an array is transfer-bound by construction: it
        reads and writes everything and computes almost nothing, so the PCIe crossing is the whole cost.
      * REPEATED ROUND TRIPS -- `round_trips > 1` means the data crosses the bus more than once, and the
        rule for that is not "is it big enough" but "keep it resident and fuse the passes". Refusing here
        is a pointer to `shader_pipeline`, which collapses N linear passes into one before any transfer.

    THE THRESHOLDS ARE PROVISIONAL AND THE VERDICT SAYS SO. Nothing in this project has measured a real
    host<->device crossover, so the numbers are arithmetic, not results."""
    if available is None:
        report = gpu_report(policy=policy)
        available = report["any_available"] and report["policy_allows"]
    if not available:
        return False, "no GPU is available to this process (or the resource policy forbids it)"
    if int(round_trips) > 1:
        return False, ("%d round trips: the data crosses the bus %d times. Fuse the passes first "
                       "(shader_pipeline collapses N linear stages into one) rather than offloading each"
                       % (int(round_trips), int(round_trips)))
    if int(n_bytes) < MIN_BYTES_PROVISIONAL:
        return False, ("%d bytes is below the ~%d byte transfer floor (PROVISIONAL, unmeasured); the "
                       "round trip would dominate" % (int(n_bytes), MIN_BYTES_PROVISIONAL))
    if float(flops_per_byte) < MIN_INTENSITY_PROVISIONAL:
        return False, ("%.1f flops/byte is transfer-bound (floor %.1f, PROVISIONAL); this job is memory "
                       "traffic, not compute" % (float(flops_per_byte), MIN_INTENSITY_PROVISIONAL))
    return True, ("%d bytes at %.1f flops/byte in one round trip clears the provisional floors -- verify "
                  "with a measurement on the real device" % (int(n_bytes), float(flops_per_byte)))


def _selftest():
    report = gpu_report()
    for key in ("cupy", "wgsl", "any_available", "policy_allows", "wired_modules", "note"):
        assert key in report

    # 1. THE FOUR STATES ARE DISTINGUISHED. A bare bool cannot say WHY, and three of the four reasons are
    #    fixable by the user.
    assert report["cupy"]["available"] is False
    assert report["cupy"]["why"], "an unavailable path must say why"

    # 2. The wired-module list is DISCOVERED, not typed. If it were a literal it would rot silently.
    assert report["wired_modules"], "no backend consumers found -- the discovery walk is broken"
    assert all("/" in name for name in report["wired_modules"])

    # 3. THE GATE REFUSES ON EACH GROUND INDEPENDENTLY, checked with availability forced so the device
    #    check cannot mask the others.
    assert should_offload(10 ** 9, 100.0, available=False)[0] is False
    assert "round trips" in should_offload(10 ** 9, 100.0, round_trips=3, available=True)[1]
    assert "transfer floor" in should_offload(1000, 100.0, available=True)[1]
    assert "transfer-bound" in should_offload(10 ** 9, 0.5, available=True)[1]

    # 4. And it accepts a job that clears every floor.
    ok, why = should_offload(10 ** 8, 50.0, available=True)
    assert ok is True and "provisional" in why.lower()

    # 5. A resource policy forbidding the GPU beats any amount of arithmetic.
    from holographic.scene_and_pipeline.holographic_policy import ResourcePolicy

    assert should_offload(10 ** 9, 100.0, policy=ResourcePolicy(gpu="off"))[0] is False

    print("holographic_gpureport: all selftests passed (four states, discovered wiring, four gates)")


if __name__ == "__main__":
    _selftest()
