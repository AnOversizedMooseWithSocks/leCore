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
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from holographic.io_and_interop.holographic_emit import EmitError, _as_node_and_fn, _emit_node, call_soa_kernel

#: Content-addressed cache -- a changed kernel is a new key; a stale entry is impossible.
CACHE_DIR = os.environ.get("LECORE_CC_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "lecore_cc"))


def cc_available():
    """Path of a usable C compiler, or None. Order: $CC, cc, gcc, clang -- $CC first because the person who
    set it knows their container better than we do."""
    for cand in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cand and shutil.which(cand):
            return shutil.which(cand)
    return None


def compiler_flags(opt):
    """Deterministic supported flags for one cache/precision policy."""
    flags = {"fast": ["-O3", "-ffp-contract=off"],
             "safe": ["-O2", "-ffp-contract=off"]}.get(opt)
    if flags is None:
        raise EmitError("opt must be 'fast' or 'safe'")
    return flags


def compiler_identity(cc=None):
    """JSON-able compiler/target identity bound into native cache keys."""
    cc = cc or cc_available()
    if cc is None:
        raise EmitError("no C compiler found")

    def capture(*args):
        try:
            completed = subprocess.run(
                [cc, *args], capture_output=True, text=True, check=False,
                timeout=30)
            return (completed.stdout + completed.stderr).strip()[:4000]
        except (OSError, subprocess.SubprocessError) as exc:
            # Identity probing must not turn an otherwise usable compiler into
            # a hard failure.  The failure text still enters the cache key and
            # evidence, so it cannot alias a successful probe.
            return "%s: %s" % (type(exc).__name__, exc)

    resolved = os.path.realpath(cc)
    try:
        stat = os.stat(resolved)
        file_identity = {"bytes": int(stat.st_size),
                         "mtime_ns": int(stat.st_mtime_ns),
                         "inode": int(stat.st_ino)}
    except OSError:
        file_identity = None
    return {
        "path": resolved,
        "file_identity": file_identity,
        "version": capture("--version"),
        "target": capture("-dumpmachine"),
        "host_system": platform.system(),
        "host_machine": platform.machine(),
        "pointer_bits": ctypes.sizeof(ctypes.c_void_p) * 8,
        "byteorder": sys.byteorder,
    }


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


def compile_cached_details(source, opt="fast", timeout=300):
    """Compile source and return its cache/toolchain provenance.

    The key binds source, flags, compiler version/binary identity, target and
    host ABI.  Private temporary paths plus atomic publication make concurrent
    first use safe; a reader can observe either no library or a complete one,
    never a partially linked file.
    """
    cc = cc_available()
    if cc is None:
        raise EmitError("no C compiler found (tried $CC, cc, gcc, clang); "
                        "install one or use holographic_zigrun where Zig exists")
    # C permits contraction at these optimization levels even without
    # ``-ffast-math``.  Disable it so the accelerator cannot silently change
    # the registered operation order; throughput work must pass parity first.
    flags = compiler_flags(opt)
    identity = compiler_identity(cc)
    full_key = hashlib.sha256(
        json.dumps({"source": source, "opt": opt, "flags": flags,
                    "compiler": identity}, sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()
    key = full_key[:24]
    os.makedirs(CACHE_DIR, exist_ok=True)
    so = os.path.join(CACHE_DIR, "k_%s.so" % key)
    if os.path.exists(so):
        return _cache_details(so, full_key, True, identity, flags, opt)

    # Atomic replace protects readers from partial output, but it does not stop
    # two cold compilers from publishing different byte-level builds (Mach-O
    # UUIDs and embedded temporary paths can differ).  An atomic directory is a
    # portable cross-process owner token.  Waiters reuse the one published
    # artifact, making its reported digest stable as well as its contents valid.
    lock = so + ".lock"
    deadline = time.monotonic() + float(timeout) + 60.0
    owner = False
    while not owner:
        try:
            os.mkdir(lock)
            owner = True
        except FileExistsError:
            if os.path.exists(so):
                return _cache_details(so, full_key, True, identity, flags, opt)
            # Recover a lock orphaned by a killed compiler only after longer
            # than the compiler's own timeout.  Removal races are harmless.
            try:
                if time.time() - os.stat(lock).st_mtime > float(timeout) + 30.0:
                    os.rmdir(lock)
                    continue
            except (FileNotFoundError, OSError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for C cache key %s" % key)
            time.sleep(0.05)

    try:
        # The file may have appeared after our first check but before we won a
        # stale lock race.  Never rebuild an already complete cache entry.
        if os.path.exists(so):
            return _cache_details(so, full_key, True, identity, flags, opt)
        descriptor, csrc = tempfile.mkstemp(
            prefix="k_%s." % key, suffix=".c", dir=CACHE_DIR)
        os.close(descriptor)
        tmp = csrc[:-2] + ".so.tmp"
        try:
            with open(csrc, "w") as fh:
                fh.write(source)
            subprocess.run(
                [cc] + flags + ["-shared", "-fPIC", csrc, "-o", tmp, "-lm"],
                check=True, capture_output=True, timeout=timeout, cwd=CACHE_DIR)
            os.replace(tmp, so)
        finally:
            for path in (csrc, tmp):
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        return _cache_details(so, full_key, False, identity, flags, opt)
    finally:
        if owner:
            try:
                os.rmdir(lock)
            except FileNotFoundError:
                pass


def _cache_details(path, full_key, cache_hit, identity, flags, opt):
    return {"path": path, "cache_key_sha256": full_key,
            "cache_hit": bool(cache_hit), "compiler": identity,
            "flags": list(flags), "opt": opt,
            "library_sha256": _sha256_file(path)}


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compile_cached(source, opt="fast", timeout=300):
    """Backward-compatible path-only wrapper over ``compile_cached_details``."""
    return compile_cached_details(source, opt=opt, timeout=timeout)["path"]


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
        # Delegates to the shared SoA-marshalling helper (holographic_emit.call_soa_kernel) -- the C runner and the
        # Zig runner emit the same `void k(const T* in, long n, T* out)` ABI, so the Python-side marshalling is one
        # convention with one home, not copied per backend. The KEPT NEGATIVE (per-call concatenate copy counted in
        # any timing) now lives with that shared code.
        return call_soa_kernel(self.n_params, self._np_dtype, self._ct, self._fn, arrays)


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
