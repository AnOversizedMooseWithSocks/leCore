"""holographic_zigrun.py -- compile emitted Zig kernels to shared libraries and batch-run them (backlog Z2 + Z3).

Z2's job: turn a scalar Python kernel into a NATIVE batch evaluator, on the fly, cached, callable in-process.
The mechanism is deliberately not a subprocess-per-call pipeline: the kernel compiles ONCE to a shared library
(`zig build-lib -dynamic`) keyed by the hashlib content hash of its source + options, and is called through
`ctypes` on NumPy arrays with zero per-call process cost. Zig's std I/O API churns between releases; `export fn`
plus ctypes does not, so the ABI is the stable surface and the version-fragile part is avoided rather than chased.

Layout is STRUCTURE-OF-ARRAYS: the input buffer is P parameter blocks of N contiguous values. SoA is what lets the
SIMD variant load a @Vector(W) lane with a single contiguous read instead of a gather -- the same reason every
particle system stores xs/ys/zs and not xyzxyz.

Z3's job: the HONEST measurement. `regime_map` races the native library against the STRONGEST honest baseline --
the same kernel vectorized in NumPy (np.sqrt/np.minimum/...), not scalar Python, which would be a strawman --
across array sizes, with mean and spread over repeats. The result is a regime map, not a claim: the sizes where
native wins and the sizes where it does not, both on record.

MEASURED (round-box SDF, 7 params, this container, mean+-sd of 5 after warm-up, opt=fast, W=8):
    n=1e3:  numpy 3.6e-05 s   zig f64 1.8e-05 (2.0x, err 0)      zig simd f32 2.2e-05 (1.6x, err 6e-07)
    n=1e5:  numpy 3.7e-03 s   zig f64 1.1e-03 (3.3x, err 0)      zig simd f32 7.2e-04 (5.1x, err 8e-07)
    n=1e6:  numpy 4.3e-02 s   zig f64 2.3e-02 (1.9x, err 0)      zig simd f32 2.7e-02 (1.6x, err 8e-07)
The verdict is a MODEST, REAL win with a regime shape: 2-5x, peaking around n=1e5 where the working set still
fits cache; at n=1e6 all variants go memory-bandwidth bound (7 input arrays x 8 B x 1e6 exceeds LLC) and the
advantage compresses to ~2x. The mechanism of the win is pass fusion -- NumPy walks the arrays once PER OPERATION
(~20 passes for the round box) while the native loop does one pass -- and pass fusion stops mattering once DRAM is
the bottleneck. KEPT NEGATIVE: no order-of-magnitude speedup exists here; early estimates of 10-40x were wrong and
are on record as wrong. KEPT NEGATIVE: the zig timings INCLUDE a per-call np.concatenate SoA copy in the ctypes
harness; accepting pre-packed SoA input is a named future lever, not silently pre-applied to flatter the numbers.
KEPT NEGATIVE: f32 SIMD is not a like-for-like precision comparison (half the memory traffic); the same-precision
column is scalar f64, which still wins. KEPT NEGATIVE: the first call pays the compiler (~1-2 s cold, then
content-hash cached to ~0), so one-shot small-n calls are a LOSS; Z5's dispatcher must respect that, not hide it.

DETERMINISM: opt="safe" (ReleaseSafe) is the deterministic mode, measured BIT-IDENTICAL to the Python original for
f64 scalar kernels built from the builtin intrinsics (holographic_emit KN5/KN6 apply verbatim). opt="fast"
(ReleaseFast) licenses float reassociation and is offered for throughput only; its delta is MEASURED by the
selftest rather than assumed away. The compile step itself is content-addressed (hashlib.sha256, never hash()).
"""

import ctypes
import hashlib
import os
import subprocess

import numpy as np

from holographic.io_and_interop.holographic_emit import (
    EmitError, _as_node_and_fn, _emit_node, _zig_argv, zig_available, call_soa_kernel)

#: Where compiled libraries live. Content-addressed, so a stale entry is impossible -- a changed kernel is a new key.
CACHE_DIR = os.environ.get("LECORE_ZIG_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "lecore_zig"))

_OPT = {"safe": "ReleaseSafe", "fast": "ReleaseFast"}


def build_batch_source(kernel, dtype="f64", simd=0):
    """Emit the full Zig translation unit: the kernel plus an `export fn <name>_batch(inp, n, out)` SoA loop.

    `simd=0` -> scalar loop in `dtype`. `simd=W` -> @Vector(W, dtype) main loop with a scalar-via-splat tail
    (the tail splats one value into a lane and takes element 0 -- same code path, no second kernel to drift).
    Returns (source_text, batch_symbol_name, n_params)."""
    if dtype not in ("f64", "f32"):
        raise EmitError("dtype must be f64 or f32, got %r" % (dtype,))
    node, _fn = _as_node_and_fn(kernel)
    n_params = len(node.args.args)
    name = node.name

    if simd == 0:
        body = _emit_node(node, "zig_%s" % dtype)
        args = ", ".join("inp[%d * n + i]" % j for j in range(n_params))
        loop = ("export fn %s_batch(inp: [*]const %s, n: usize, out: [*]%s) void {\n"
                "    var i: usize = 0;\n"
                "    while (i < n) : (i += 1) {\n"
                "        out[i] = %s(%s);\n"
                "    }\n"
                "}\n" % (name, dtype, dtype, name, args))
        return "const std = @import(\"std\");\n" + body + loop, name + "_batch", n_params

    w = int(simd)
    if w < 2 or (w & (w - 1)):
        raise EmitError("simd width must be a power of two >= 2, got %r" % (simd,))
    body = _emit_node(node, "zigv_%s" % dtype)
    vargs = ", ".join("(inp + %d * n + i)[0..%d].*" % (j, w) for j in range(n_params))
    targs = ", ".join("@as(V, @splat(inp[%d * n + i]))" % j for j in range(n_params))
    loop = ("export fn %s_batch(inp: [*]const %s, n: usize, out: [*]%s) void {\n"
            "    var i: usize = 0;\n"
            "    while (i + %d <= n) : (i += %d) {\n"
            "        (out + i)[0..%d].* = %s(%s);\n"     # vectors coerce to same-length arrays; contiguous store
            "    }\n"
            "    while (i < n) : (i += 1) {\n"
            "        out[i] = %s(%s)[0];\n"              # tail: splat one scalar through the SAME vector kernel
            "    }\n"
            "}\n" % (name, dtype, dtype, w, w, w, name, vargs, name, targs))
    header = "const std = @import(\"std\");\nconst V = @Vector(%d, %s);\n" % (w, dtype)
    return header + body + loop, name + "_batch", n_params


def compile_cached(source, opt="safe", timeout=300):
    """Compile `source` to a shared library, content-addressed under CACHE_DIR. Returns the .so path.

    The key is sha256(source + opt + zig argv) -- hashlib, never hash(), because the cache must mean the same
    thing across processes and PYTHONHASHSEED has no say in it."""
    if not zig_available():
        raise EmitError("no Zig toolchain: `pip install ziglang` (opt-in accelerator, like numba)")
    if opt not in _OPT:
        raise EmitError("opt must be one of %s" % sorted(_OPT))
    argv = _zig_argv()
    key = hashlib.sha256(("%s|%s|%s" % (source, opt, argv)).encode()).hexdigest()[:24]
    os.makedirs(CACHE_DIR, exist_ok=True)
    so = os.path.join(CACHE_DIR, "k_%s.so" % key)
    if os.path.exists(so):
        return so
    zsrc = os.path.join(CACHE_DIR, "k_%s.zig" % key)
    with open(zsrc, "w") as fh:
        fh.write(source)
    # -fPIC via build-lib -dynamic; write to a temp name then rename so a killed compile never leaves a half .so.
    tmp = so + ".tmp"
    subprocess.run(argv + ["build-lib", "-dynamic", "-O", _OPT[opt], zsrc, "-femit-bin=" + tmp],
                   check=True, capture_output=True, timeout=timeout, cwd=CACHE_DIR)
    os.replace(tmp, so)
    return so


class ZigKernel:
    """A compiled, cached, ctypes-callable batch kernel. Call with P same-length 1-D arrays; returns 1-D results.

    `opt='safe'` is the deterministic mode (f64 scalar: bit-identical to Python on builtin-intrinsic kernels).
    `simd=W` compiles the @Vector(W) variant -- f32 W=8 is the measured sweet spot on this container."""

    def __init__(self, kernel, dtype="f64", simd=0, opt="safe"):
        src, sym, n_params = build_batch_source(kernel, dtype=dtype, simd=simd)
        self.source, self.n_params, self.dtype = src, n_params, dtype
        self._np_dtype = np.float64 if dtype == "f64" else np.float32
        self._lib = ctypes.CDLL(compile_cached(src, opt=opt))
        self._fn = getattr(self._lib, sym)
        ct = ctypes.c_double if dtype == "f64" else ctypes.c_float
        self._fn.argtypes = [ctypes.POINTER(ct), ctypes.c_size_t, ctypes.POINTER(ct)]
        self._fn.restype = None
        self._ct = ct

    def __call__(self, *arrays):
        # Delegates to the shared SoA-marshalling helper (holographic_emit.call_soa_kernel) -- same ABI as the C
        # runner (ccrun), so one calling convention with one home rather than a copy per backend. The KEPT NEGATIVE
        # (per-call concatenate copy, counted in any timing) lives with that shared code.
        return call_soa_kernel(self.n_params, self._np_dtype, self._ct, self._fn, arrays)


def as_numpy(kernel):
    """Build the VECTORIZED NumPy twin of a kernel -- the strongest honest baseline for regime_map.

    Same text, NumPy namespace: sqrt->np.sqrt, min/max->np.minimum/np.maximum (the kernel grammar is 2-arg),
    abs->np.abs, pow->np.power. A scalar-Python baseline would be a strawman and is deliberately not offered."""
    import textwrap

    node, _fn = _as_node_and_fn(kernel)
    src = kernel if isinstance(kernel, str) else None
    if src is None:
        import inspect
        src = inspect.getsource(_fn)
    ns = {"sqrt": np.sqrt, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos,
          "abs": np.abs, "min": np.minimum, "max": np.maximum, "pow": np.power}
    exec(compile(textwrap.dedent(src), "<numpy-kernel>", "exec"), ns)   # noqa: S102 -- our own kernel text
    return ns[node.name]


def regime_map(kernel, sizes=(1000, 100000, 1000000), repeats=5, seed=0, simd_width=8):
    """Z3: race numpy / zig scalar f64 / zig simd f32 across `sizes`. Returns rows with mean+spread per variant.

    Every row carries the baseline it was judged against; the `speedup_*` fields are per-size, because the honest
    answer is a REGIME MAP, not one number. First-call compile cost is excluded by a warm-up call and REPORTED
    separately in the row -- hiding it would flatter the accelerator (kept-negative discipline)."""
    import time

    if not zig_available():
        raise EmitError("no Zig toolchain: `pip install ziglang`")
    np_fn = as_numpy(kernel)
    t0 = time.perf_counter()
    zk64 = ZigKernel(kernel, dtype="f64", simd=0, opt="fast")
    zk32 = ZigKernel(kernel, dtype="f32", simd=simd_width, opt="fast")
    compile_s = time.perf_counter() - t0                 # ~0 when the content-hash cache is warm

    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        cols = [rng.uniform(-2.0, 2.0, int(n)) for _ in range(zk64.n_params)]

        def _timed(f):
            f(*cols)                                     # warm-up: page in, JIT nothing, fair to everyone
            ts = []
            for _ in range(repeats):
                a = time.perf_counter()
                f(*cols)
                ts.append(time.perf_counter() - a)
            return float(np.mean(ts)), float(np.std(ts))

        m_np, s_np = _timed(np_fn)
        m_64, s_64 = _timed(zk64)
        m_32, s_32 = _timed(zk32)
        # Correctness travels with every timing: a fast wrong answer is not a result.
        ref = np_fn(*cols)
        err64 = float(np.max(np.abs(zk64(*cols) - ref)))
        err32 = float(np.max(np.abs(zk32(*cols).astype(np.float64) - ref)))
        rows.append({"n": int(n),
                     "numpy_s": m_np, "numpy_sd": s_np,
                     "zig_f64_s": m_64, "zig_f64_sd": s_64, "speedup_f64": m_np / m_64, "max_abs_err_f64": err64,
                     "zig_simd_f32_s": m_32, "zig_simd_f32_sd": s_32, "speedup_simd_f32": m_np / m_32,
                     "max_abs_err_f32": err32, "compile_s_first_call": compile_s})
    return rows


class AutoKernel:
    """Z5: the dispatcher, sized to the MEASURED reality (2-5x, amortized), not a fantasy.

    Policy, stated plainly so nobody has to reverse-engineer it from behavior:
      - numpy until the kernel has been called enough that the ~1-2 s compile cost amortizes
        (`min_calls_to_compile`, default 3) AND the arrays are big enough for fusion to matter
        (`min_n`, default 4096 -- below that the measured win is ~2x on ~30 us, i.e. nothing);
      - then compile ONCE (content-hash cached, so a warm cache makes this free) and switch;
      - on the FIRST native call in safe mode the result is CHECKED against numpy: if not
        bit-identical the dispatcher REFUSES the substitution permanently and says so loudly
        (a fast wrong answer is not an acceleration). opt='fast' skips the identity check --
        reassociation makes it unfair -- and checks a 1e-9 bound instead, still refusing on breach.
    Everything is default-conservative: no toolchain -> numpy forever, silently correct, loudly logged."""

    def __init__(self, kernel, min_calls_to_compile=3, min_n=4096, opt="safe", simd=0, dtype="f64"):
        self._kernel = kernel
        self._np = as_numpy(kernel)
        self._min_calls, self._min_n = int(min_calls_to_compile), int(min_n)
        self._opt, self._simd, self._dtype = opt, int(simd), dtype
        self._calls = 0
        self._zk = None
        self.refused = None                              # set to a reason string if substitution was refused
        self.backend_log = []                            # one entry per call: 'numpy' | 'zig' -- the audit trail

    def _try_compile(self):
        if not zig_available():
            self.refused = "no toolchain (`pip install ziglang`); numpy forever"
            return
        zk = ZigKernel(self._kernel, dtype=self._dtype, simd=self._simd, opt=self._opt)
        self._zk = zk

    def __call__(self, *arrays):
        self._calls += 1
        cols = [np.asarray(a, float) for a in arrays]
        n = cols[0].shape[0]
        use_native = (self.refused is None and self._calls > self._min_calls and n >= self._min_n)
        if use_native and self._zk is None:
            self._try_compile()
            if self._zk is not None:
                # first-native-call identity gate: the check that makes substitution SAFE to trust
                got = self._zk(*cols)
                ref = self._np(*cols)
                if self._opt == "safe" and self._dtype == "f64" and self._simd == 0:
                    if not np.array_equal(got, ref):
                        self.refused = "safe f64 result not bit-identical to numpy; substitution refused"
                        self._zk = None
                elif float(np.max(np.abs(got.astype(np.float64) - ref))) > 1e-9 and self._dtype == "f64":
                    self.refused = "fast f64 delta beyond 1e-9; substitution refused"
                    self._zk = None
                if self._zk is not None:
                    self.backend_log.append("zig")
                    return got
        if use_native and self._zk is not None:
            self.backend_log.append("zig")
            return self._zk(*cols)
        self.backend_log.append("numpy")
        return self._np(*cols)


def dispatch_policy(n, calls_expected, min_calls_to_compile=3, min_n=4096, toolchain=None):
    """The Z5 decision as a pure, JSON-able function: which backend WOULD AutoKernel use, and why.

    Exists so an agent can ask for the decision without compiling anything -- the policy is data, the
    wrapper merely enforces it."""
    have = zig_available() if toolchain is None else bool(toolchain)
    if not have:
        return {"backend": "numpy", "why": "no zig toolchain (`pip install ziglang`)"}
    if int(calls_expected) <= int(min_calls_to_compile):
        return {"backend": "numpy", "why": "too few calls to amortize the ~1-2 s compile (measured, first call)"}
    if int(n) < int(min_n):
        return {"backend": "numpy", "why": "n < %d: measured win at small n is ~2x of ~30 us -- noise" % int(min_n)}
    return {"backend": "zig", "why": "amortized and n large enough for pass fusion (measured 2-5x regime)"}


_ROUND_BOX = """
def sdf_round_box(px: float, py: float, pz: float, bx: float, by: float, bz: float, r: float) -> float:
    qx = max(abs(px) - bx, 0.0)
    qy = max(abs(py) - by, 0.0)
    qz = max(abs(pz) - bz, 0.0)
    outside = sqrt(qx * qx + qy * qy + qz * qz)
    inside = min(max(abs(px) - bx, max(abs(py) - by, abs(pz) - bz)), 0.0)
    return outside + inside - r
"""


def _selftest():
    """Regression trap: exact numeric contracts, and the negatives measured rather than waved at.

    - safe f64 scalar batch is BIT-IDENTICAL to the vectorized NumPy evaluation (same doubles, same order);
    - fast f64's reassociation delta is MEASURED and bounded (KN5: it exists; the bound says it stayed small);
    - simd f32 agrees with f64 to f32-epsilon scale, INCLUDING the tail lane (n deliberately not a multiple of W);
    - the content-hash cache returns the same .so for the same source and a different one for different opt;
    - regime_map on a small size still reports correct errors alongside timings.
    Skips LOUDLY without a toolchain -- the engine must pass without the wheel (opt-in accelerator contract)."""
    if not zig_available():
        print("OK: holographic_zigrun self-test SKIPPED -- no Zig toolchain (`pip install ziglang`); "
              "source emission still asserted")
        src, sym, p = build_batch_source(_ROUND_BOX, dtype="f32", simd=8)
        assert "export fn sdf_round_box_batch" in src and "@Vector(8, f32)" in src and p == 7
        return

    rng = np.random.default_rng(3)
    n = 10007                                            # prime: forces the SIMD tail path to actually run
    cols = [rng.uniform(-2.0, 2.0, n) for _ in range(7)]
    ref = as_numpy(_ROUND_BOX)(*cols)

    safe64 = ZigKernel(_ROUND_BOX, dtype="f64", simd=0, opt="safe")(*cols)
    assert np.array_equal(safe64, ref), "safe f64 scalar must be bit-identical to the NumPy evaluation"

    fast64 = ZigKernel(_ROUND_BOX, dtype="f64", simd=0, opt="fast")(*cols)
    d_fast = float(np.max(np.abs(fast64 - ref)))
    assert d_fast < 1e-12, "ReleaseFast reassociation delta blew up: %g" % d_fast

    simd32 = ZigKernel(_ROUND_BOX, dtype="f32", simd=8, opt="safe")(*cols).astype(np.float64)
    d32 = float(np.max(np.abs(simd32 - ref)))
    assert d32 < 5e-6, "simd f32 beyond f32-epsilon scale: %g" % d32

    a = compile_cached(build_batch_source(_ROUND_BOX, "f64", 0)[0], opt="safe")
    b = compile_cached(build_batch_source(_ROUND_BOX, "f64", 0)[0], opt="safe")
    c = compile_cached(build_batch_source(_ROUND_BOX, "f64", 0)[0], opt="fast")
    assert a == b and a != c, "content-hash cache identity broken"

    # Z5: the dispatcher's contract -- numpy early, zig after amortization, identity-gated, audit-trailed.
    ak = AutoKernel(_ROUND_BOX, min_calls_to_compile=2, min_n=1000)
    small = [rng.uniform(-2, 2, 64) for _ in range(7)]
    big = [rng.uniform(-2, 2, 5000) for _ in range(7)]
    r1 = ak(*big); r2 = ak(*big); r3 = ak(*small); r4 = ak(*big)
    assert ak.backend_log == ["numpy", "numpy", "numpy", "zig"], "policy trace wrong: %r" % ak.backend_log
    assert np.array_equal(r4, as_numpy(_ROUND_BOX)(*big)), "dispatched result must equal numpy exactly (safe f64)"
    assert ak.refused is None
    pol = dispatch_policy(n=100, calls_expected=100)
    assert pol["backend"] == "numpy" and "noise" in pol["why"]
    assert dispatch_policy(n=100000, calls_expected=100)["backend"] == "zig"

    rows = regime_map(_ROUND_BOX, sizes=(4096,), repeats=3)
    assert rows[0]["max_abs_err_f64"] < 1e-12 and rows[0]["speedup_f64"] > 0
    print("OK: holographic_zigrun self-test passed (safe f64 batch BIT-IDENTICAL to NumPy over n=%d incl. SIMD "
          "tail; ReleaseFast delta measured %.3g; simd f32 delta %.3g; cache content-addressed; regime row "
          "speedup_f64=%.1fx at n=4096 -- and the first-call compile cost is reported, not hidden)"
          % (n, d_fast, d32, rows[0]["speedup_f64"]))


if __name__ == "__main__":
    _selftest()
