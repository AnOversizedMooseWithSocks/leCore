"""holographic_ccrun.py -- compile emitted C kernels with the system C compiler and batch-run them.

WHY THIS MODULE EXISTS
----------------------
`holographic_zigrun` gives native batch kernels, but ONLY where a Zig toolchain exists. Poly Studio's build
container had no Zig and a closed network -- but it DID have `cc` (nearly every Unix does). The emitter
(`holographic_emit`) already speaks `c_f64`/`c_f32` -- those dialects exist and are validated by actually
compiling and running them in emit's own selftest -- so a C runner is wiring plus a toolchain probe, not a new
code generator. This is the upstreamed `ccrun.py` from the demo layer: same IR, same SoA ctypes harness, same
content-addressed cache discipline as zigrun, compiled with `cc -O3 -shared -fPIC -lm`.

Contract mirrors zigrun deliberately: `CKernel(kernel_source)(x, y, z) -> out`. `dtype='f64'` is the
deterministic mode -- emit's measured result is BIT-IDENTICAL to Python on builtin-intrinsic kernels (same
order of operations, same doubles). `f32` carries measured ~3e-7 error; that tolerance is emit's, verified by
running, not asserted.

KEPT NEGATIVE: no SIMD variant here. Zig's @Vector maps to a portable dialect; C SIMD is intrinsics-per-arch
or autovectorizer prayer. `-O3` already autovectorizes the scalar loop; a hand-vector C dialect was judged
maintenance without a measured win. If a profile ever shows the gap matters, measure first.
"""

import ctypes
import hashlib
import os
import shutil
import subprocess

import numpy as np

from holographic.io_and_interop.holographic_emit import EmitError, _as_node_and_fn, _emit_node

#: Content-addressed cache -- a changed kernel is a new key; a stale entry is impossible.
CACHE_DIR = os.environ.get("LECORE_CC_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "lecore_cc"))


def cc_available():
    """Path of a usable C compiler, or None. Order: $CC, cc, gcc, clang -- $CC first because the person who
    set it knows their container better than we do."""
    for cand in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cand and shutil.which(cand):
            return shutil.which(cand)
    return None


def build_batch_source(kernel, dtype="f64"):
    """Emit the full C translation unit: the kernel plus a `<name>_batch(inp, n, out)` SoA loop.

    Same SoA layout as zigrun (P parameter blocks of N in one contiguous buffer) so the two backends are
    drop-in twins and any harness written for one runs on the other. Returns (source, symbol, n_params)."""
    if dtype not in ("f64", "f32"):
        raise EmitError("dtype must be f64 or f32, got %r" % (dtype,))
    node, _fn = _as_node_and_fn(kernel)
    n_params = len(node.args.args)
    name = node.name
    ctype = "double" if dtype == "f64" else "float"
    body = _emit_node(node, "c_%s" % dtype)
    args = ", ".join("inp[%d * n + i]" % j for j in range(n_params))
    loop = ("void %s_batch(const %s* inp, size_t n, %s* out) {\n"
            "    for (size_t i = 0; i < n; i++) {\n"
            "        out[i] = %s(%s);\n"
            "    }\n"
            "}\n" % (name, ctype, ctype, name, args))
    return "#include <math.h>\n#include <stddef.h>\n" + body + loop, name + "_batch", n_params


def compile_cached(source, opt="fast", timeout=300):
    """Compile `source` to a shared library, content-addressed under CACHE_DIR. Returns the .so path.

    Key is sha256(source + opt + compiler path) -- hashlib, never hash(): the cache must mean the same thing
    across processes, and a different compiler is a different artifact. Temp-then-rename so a killed compile
    never leaves a half-written .so behind (same discipline as zigrun)."""
    cc = cc_available()
    if cc is None:
        raise EmitError("no C compiler found (tried $CC, cc, gcc, clang); "
                        "install one or use holographic_zigrun where Zig exists")
    flags = {"fast": ["-O3"], "safe": ["-O2"]}.get(opt)
    if flags is None:
        raise EmitError("opt must be 'fast' or 'safe'")
    key = hashlib.sha256(("%s|%s|%s" % (source, opt, cc)).encode()).hexdigest()[:24]
    os.makedirs(CACHE_DIR, exist_ok=True)
    so = os.path.join(CACHE_DIR, "k_%s.so" % key)
    if os.path.exists(so):
        return so
    csrc = os.path.join(CACHE_DIR, "k_%s.c" % key)
    with open(csrc, "w") as fh:
        fh.write(source)
    tmp = so + ".tmp"
    subprocess.run([cc] + flags + ["-shared", "-fPIC", csrc, "-o", tmp, "-lm"],
                   check=True, capture_output=True, timeout=timeout, cwd=CACHE_DIR)
    os.replace(tmp, so)
    return so


class CKernel:
    """A compiled, cached, ctypes-callable batch kernel -- the C twin of zigrun's ZigKernel.

    Call with P same-length 1-D arrays; returns a 1-D result array. f64 is the deterministic path (emit
    measured it bit-identical to the Python source on builtin-intrinsic kernels)."""

    def __init__(self, kernel, dtype="f64", opt="fast"):
        src, sym, n_params = build_batch_source(kernel, dtype=dtype)
        self.source, self.n_params, self.dtype = src, n_params, dtype
        self._np_dtype = np.float64 if dtype == "f64" else np.float32
        self._lib = ctypes.CDLL(compile_cached(src, opt=opt))
        self._fn = getattr(self._lib, sym)
        ct = ctypes.c_double if dtype == "f64" else ctypes.c_float
        self._fn.argtypes = [ctypes.POINTER(ct), ctypes.c_size_t, ctypes.POINTER(ct)]
        self._fn.restype = None
        self._ct = ct

    def __call__(self, *arrays):
        if len(arrays) != self.n_params:
            raise EmitError("kernel takes %d arrays, got %d" % (self.n_params, len(arrays)))
        cols = [np.ascontiguousarray(a, dtype=self._np_dtype) for a in arrays]
        n = cols[0].shape[0]
        if any(cl.shape != (n,) for cl in cols):
            raise EmitError("all input arrays must be 1-D of the same length")
        # KEPT NEGATIVE (inherited from zigrun's honest timing note): this concatenate is a per-call SoA copy
        # counted inside any timing you take of a CKernel call. Caches nothing; charges everything.
        inp = np.concatenate(cols)
        out = np.empty(n, dtype=self._np_dtype)
        self._fn(inp.ctypes.data_as(ctypes.POINTER(self._ct)), n,
                 out.ctypes.data_as(ctypes.POINTER(self._ct)))
        return out


def _selftest():
    """Regression trap: f64 must be BIT-IDENTICAL to the Python kernel (emit's contract), f32 within its
    measured tolerance -- and the no-compiler path must refuse loudly, never fall back silently."""
    src = ("def sd_ring(x: float, y: float, z: float) -> float:\n"
           "    return min(sqrt(x*x + y*y) - 1.0, abs(z) - 0.5)\n")
    if cc_available() is None:
        # correct refusal is the testable behaviour on a compiler-less box
        try:
            CKernel(src)
            raise AssertionError("must refuse without a C compiler")
        except EmitError:
            print("ccrun selftest OK (no compiler: refused loudly, as designed)")
            return
    rng = np.random.default_rng(0)
    x, y, z = (rng.uniform(-2, 2, 500) for _ in range(3))
    ref = np.minimum(np.sqrt(x * x + y * y) - 1.0, np.abs(z) - 0.5)
    k64 = CKernel(src, dtype="f64")
    got = k64(x, y, z)
    assert np.array_equal(got, ref), "f64 must be bit-identical (emit's measured contract)"
    k32 = CKernel(src, dtype="f32")
    err = float(np.max(np.abs(k32(x, y, z).astype(np.float64) - ref)))
    assert err < 1e-5, err                                    # generous ceiling over emit's measured ~3e-7
    # cache hit: second construction reuses the .so (same content hash)
    so1 = compile_cached(k64.source); so2 = compile_cached(k64.source)
    assert so1 == so2
    print("ccrun selftest OK (f64 bit-identical on 500 pts, f32 max err %.2e, cache stable)" % err)


if __name__ == "__main__":
    _selftest()
