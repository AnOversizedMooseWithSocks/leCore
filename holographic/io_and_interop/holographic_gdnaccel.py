"""Optional native accelerator for Qwen Gated-DeltaNet's sequential scan.

Only the state recurrence lives here.  The projection, causal convolution,
normalisation and output projection stay in :mod:`holographic_gdnruntime`, so
this is a narrow substitution rather than a second model implementation.

The default remains NumPy.  Request the native path with
``gdn_recurrence_backend="c"`` in the runtime config or
``LECORE_GDN_BACKEND=c``.  The C compiler is optional: source is compiled once
through ``holographic_ccrun``'s content-addressed cache and called over a
contiguous ctypes ABI.  The first call for each dtype/shape/state regime is
compared with NumPy.  A missing compiler, compile error, or parity failure
permanently falls back to NumPy for that runtime.

The parity gate is intentionally on the *actual model inputs*, including a
separate gate for resumed (non-zero) recurrent state.  A synthetic self-test
alone would not protect the carried-state path used by chunked evaluation.
"""

import ctypes

import numpy as np


def gdn_recurrence_numpy(q, k, v, beta, g, initial=None):
    """Reference recurrence, factored out verbatim from ``GDNRuntime._gdn``.

    Shapes are ``q,k=(T,H,K)``, ``v,out=(T,H,V)``, ``beta,g=(T,H)`` and
    ``initial,state=(H,K,V)``.  The returned state is independent of ``initial``.
    """
    q = np.asarray(q)
    k = np.asarray(k)
    v = np.asarray(v)
    beta = np.asarray(beta)
    g = np.asarray(g)
    nt, nh, nk = q.shape
    nv = v.shape[-1]
    state = (np.zeros((nh, nk, nv), dtype=q.dtype) if initial is None else
             np.array(initial, dtype=q.dtype, copy=True))
    out = np.zeros((nt, nh, nv), dtype=q.dtype)
    for t in range(nt):
        state = state * np.exp(g[t])[:, None, None]
        kv = np.einsum("hkv,hk->hv", state, k[t])
        delta = (v[t] - kv) * beta[t][:, None]
        state = state + k[t][:, :, None] * delta[:, None, :]
        out[t] = np.einsum("hkv,hk->hv", state, q[t])
    return out, state


_C_SOURCE = r"""
#include <math.h>
#include <stddef.h>
#include <string.h>

#define DEFINE_SCAN(T, SUFFIX, EXPFN)                                             \
void lecore_gdn_scan_##SUFFIX(                                                    \
    const T *q, const T *k, const T *v, const T *beta, const T *g,               \
    const T *initial, T *scratch, size_t nt, size_t nh, size_t nk, size_t nv,     \
    T *state, T *out) {                                                           \
    const size_t state_n = nh * nk * nv;                                          \
    memcpy(state, initial, state_n * sizeof(T));                                  \
    for (size_t t = 0; t < nt; ++t) {                                            \
        for (size_t h = 0; h < nh; ++h) {                                        \
            const T decay = EXPFN(g[t * nh + h]);                                \
            const size_t hs = h * nk * nv;                                       \
            T *kv = scratch + h * nv;                                             \
            T *out_h = out + (t * nh + h) * nv;                                  \
            memset(kv, 0, nv * sizeof(T));                                       \
            memset(out_h, 0, nv * sizeof(T));                                    \
            /* Fuse decay and S@k into one contiguous state pass. */              \
            for (size_t i = 0; i < nk; ++i) {                                    \
                T *row = state + hs + i * nv;                                    \
                const T ki = k[(t * nh + h) * nk + i];                           \
                for (size_t j = 0; j < nv; ++j) {                                \
                    const T value = row[j] * decay;                              \
                    row[j] = value;                                              \
                    kv[j] += value * ki;                                         \
                }                                                                 \
            }                                                                     \
            for (size_t j = 0; j < nv; ++j)                                     \
                kv[j] = (v[(t * nh + h) * nv + j] - kv[j]) *                    \
                        beta[t * nh + h];                                        \
            /* Fuse the rank-one update and q read into the second pass. */       \
            for (size_t i = 0; i < nk; ++i) {                                    \
                T *row = state + hs + i * nv;                                    \
                const T ki = k[(t * nh + h) * nk + i];                           \
                const T qi = q[(t * nh + h) * nk + i];                           \
                for (size_t j = 0; j < nv; ++j) {                                \
                    const T value = row[j] + ki * kv[j];                         \
                    row[j] = value;                                              \
                    out_h[j] += value * qi;                                      \
                }                                                                 \
            }                                                                     \
        }                                                                         \
    }                                                                             \
}

DEFINE_SCAN(float, f32, expf)
DEFINE_SCAN(double, f64, exp)
"""


class _CScan:
    """Compiled scan with a zero-copy ABI for already-contiguous arrays."""

    def __init__(self, dtype):
        # Reuse the project's compiler probe, deterministic flags and
        # source+compiler content-addressed cache.  Import is lazy so importing
        # the NumPy-only runtime never probes a toolchain or writes a cache.
        from holographic.io_and_interop.holographic_ccrun import compile_cached_details

        self.dtype = np.dtype(dtype)
        if self.dtype == np.dtype(np.float32):
            suffix, ctype = "f32", ctypes.c_float
        elif self.dtype == np.dtype(np.float64):
            suffix, ctype = "f64", ctypes.c_double
        else:
            raise TypeError("native GDN scan supports float32/float64, got %s" % self.dtype)
        self._ctype = ctype
        cache = compile_cached_details(_C_SOURCE, opt="safe")
        library_path = cache["path"]
        self._lib = ctypes.CDLL(library_path)
        self.provenance = {
            "dtype": self.dtype.name,
            "compiler": cache["compiler"],
            "flags": cache["flags"],
            "optimization_policy": cache["opt"],
            "cache_key_sha256": cache["cache_key_sha256"],
            "cache_hit_on_load": cache["cache_hit"],
            "library_sha256": cache["library_sha256"],
        }
        self._fn = getattr(self._lib, "lecore_gdn_scan_" + suffix)
        ptr = ctypes.POINTER(ctype)
        self._fn.argtypes = ([ptr] * 7 + [ctypes.c_size_t] * 4 + [ptr, ptr])
        self._fn.restype = None

    def __call__(self, q, k, v, beta, g, initial):
        arrays = [np.ascontiguousarray(a, dtype=self.dtype)
                  for a in (q, k, v, beta, g, initial)]
        q, k, v, beta, g, initial = arrays
        if q.ndim != 3:
            raise ValueError("q must have shape (tokens, heads, key_dim)")
        nt, nh, nk = q.shape
        if k.shape != (nt, nh, nk):
            raise ValueError("k shape %r does not match q %r" % (k.shape, q.shape))
        if v.ndim != 3 or v.shape[:2] != (nt, nh):
            raise ValueError("v shape %r does not match q tokens/heads %r" %
                             (v.shape, (nt, nh)))
        nv = v.shape[-1]
        if beta.shape != (nt, nh) or g.shape != (nt, nh):
            raise ValueError("beta/g must both have shape %r" % ((nt, nh),))
        if initial.shape != (nh, nk, nv):
            raise ValueError("initial state shape %r != %r" %
                             (initial.shape, (nh, nk, nv)))
        state = np.empty((nh, nk, nv), dtype=self.dtype)
        out = np.empty((nt, nh, nv), dtype=self.dtype)
        scratch = np.empty((nh, nv), dtype=self.dtype)
        ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(self._ctype))
        self._fn(ptr(q), ptr(k), ptr(v), ptr(beta), ptr(g), ptr(initial), ptr(scratch),
                 nt, nh, nk, nv, ptr(state), ptr(out))
        return out, state


class GDNRecurrence:
    """Conservative per-runtime dispatcher for the recurrence accelerator."""

    def __init__(self, requested="numpy"):
        requested = str(requested or "numpy").strip().lower()
        if requested not in ("numpy", "c"):
            raise ValueError("gdn_recurrence_backend must be numpy or c")
        self.requested = requested
        self.active = "numpy"
        self.refused = None
        self._kernels = {}
        self._validated = set()
        self._checks = []
        self._native_calls = 0
        self._native_tokens = 0
        self._native_attempts = 0
        self._native_attempt_tokens = 0
        self._numpy_calls = 0
        self._numpy_tokens = 0
        self._direct_numpy_calls = 0
        self._direct_numpy_tokens = 0
        self._fallback_calls = 0
        self._fallback_tokens = 0

    @staticmethod
    def _initial(q, v, initial):
        if initial is not None:
            return np.ascontiguousarray(initial, dtype=q.dtype)
        return np.zeros((q.shape[1], q.shape[2], v.shape[2]), dtype=q.dtype)

    def _native(self, dtype):
        key = np.dtype(dtype).str
        if key not in self._kernels:
            self._kernels[key] = _CScan(dtype)
        return self._kernels[key]

    def __call__(self, q, k, v, beta, g, initial=None):
        token_count = int(np.asarray(q).shape[0])
        if self.requested == "numpy":
            self._numpy_calls += 1
            self._numpy_tokens += token_count
            self._direct_numpy_calls += 1
            self._direct_numpy_tokens += token_count
            return gdn_recurrence_numpy(q, k, v, beta, g, initial)
        if self.refused is not None:
            self._numpy_calls += 1
            self._numpy_tokens += token_count
            self._fallback_calls += 1
            self._fallback_tokens += token_count
            return gdn_recurrence_numpy(q, k, v, beta, g, initial)

        q = np.ascontiguousarray(q)
        k = np.ascontiguousarray(k, dtype=q.dtype)
        v = np.ascontiguousarray(v, dtype=q.dtype)
        beta = np.ascontiguousarray(beta, dtype=q.dtype)
        g = np.ascontiguousarray(g, dtype=q.dtype)
        initial_c = self._initial(q, v, initial)
        resumed = initial is not None
        signature = (q.dtype.str, q.shape[1:], v.shape[-1], bool(resumed))
        try:
            self._native_attempts += 1
            self._native_attempt_tokens += int(q.shape[0])
            got = self._native(q.dtype)(q, k, v, beta, g, initial_c)
        except Exception as exc:
            self.refused = "%s: %s" % (type(exc).__name__, exc)
            self.active = "numpy"
            self._numpy_calls += 1
            self._numpy_tokens += int(q.shape[0])
            self._fallback_calls += 1
            self._fallback_tokens += int(q.shape[0])
            return gdn_recurrence_numpy(q, k, v, beta, g, initial_c)

        if signature not in self._validated:
            ref = gdn_recurrence_numpy(q, k, v, beta, g, initial_c)
            abs_err = max(float(np.max(np.abs(got[0] - ref[0]), initial=0.0)),
                          float(np.max(np.abs(got[1] - ref[1]), initial=0.0)))
            scale = max(float(np.max(np.abs(ref[0]), initial=0.0)),
                        float(np.max(np.abs(ref[1]), initial=0.0)), 1e-30)
            rel_err = abs_err / scale
            if q.dtype == np.float32:
                ok = (np.allclose(got[0], ref[0], rtol=5e-5, atol=5e-6) and
                      np.allclose(got[1], ref[1], rtol=5e-5, atol=5e-6))
            else:
                ok = (np.allclose(got[0], ref[0], rtol=2e-12, atol=2e-13) and
                      np.allclose(got[1], ref[1], rtol=2e-12, atol=2e-13))
            check = {"dtype": q.dtype.name, "heads": int(q.shape[1]),
                     "key_dim": int(q.shape[2]), "value_dim": int(v.shape[2]),
                     "resumed": bool(resumed), "max_abs_error": abs_err,
                     "max_relative_error": rel_err, "passed": bool(ok)}
            self._checks.append(check)
            if not ok:
                self.refused = "first-call parity gate failed: %r" % check
                self.active = "numpy"
                self._numpy_calls += 1
                self._numpy_tokens += int(q.shape[0])
                self._fallback_calls += 1
                self._fallback_tokens += int(q.shape[0])
                return ref
            self._validated.add(signature)
        self.active = "c"
        self._native_calls += 1
        self._native_tokens += int(q.shape[0])
        return got

    def report(self):
        libraries = [kernel.provenance for _dtype, kernel in
                     sorted(self._kernels.items())]
        passed_states = {bool(row["resumed"]) for row in self._checks
                         if row.get("passed")}
        return {"scope": "full_sequence_gdn_recurrence",
                "requested": self.requested, "active": self.active,
                "refused": self.refused, "validated_regimes": list(self._checks),
                "fresh_and_resumed_parity_complete":
                    passed_states == {False, True},
                "native_attempts": self._native_attempts,
                "native_attempt_tokens": self._native_attempt_tokens,
                "native_calls": self._native_calls,
                "native_tokens": self._native_tokens,
                "numpy_calls": self._numpy_calls,
                "numpy_tokens": self._numpy_tokens,
                "direct_numpy_calls": self._direct_numpy_calls,
                "direct_numpy_tokens": self._direct_numpy_tokens,
                "fallback_calls": self._fallback_calls,
                "fallback_tokens": self._fallback_tokens,
                "native_libraries": libraries}


def _selftest():
    """Exercise both dispatcher regimes and carried-state parity."""
    rng = np.random.default_rng(7)
    q = rng.standard_normal((9, 3, 5)).astype(np.float32)
    k = rng.standard_normal(q.shape).astype(np.float32)
    k /= np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-12)
    v = rng.standard_normal((9, 3, 4)).astype(np.float32)
    beta = rng.random((9, 3), dtype=np.float32)
    g = -rng.random((9, 3), dtype=np.float32)
    initial = rng.standard_normal((3, 5, 4)).astype(np.float32)

    direct = GDNRecurrence("numpy")
    assert all(np.array_equal(a, b) for a, b in zip(
        direct(q, k, v, beta, g, initial),
        gdn_recurrence_numpy(q, k, v, beta, g, initial)))

    from holographic.io_and_interop.holographic_ccrun import cc_available
    if cc_available() is None:
        print("gdnaccel selftest OK (NumPy); native scan SKIPPED (no C compiler)")
        return
    native = GDNRecurrence("c")
    fresh = native(q[:4], k[:4], v[:4], beta[:4], g[:4])
    carried = native(q[4:], k[4:], v[4:], beta[4:], g[4:], fresh[1])
    ref_fresh = gdn_recurrence_numpy(q[:4], k[:4], v[:4], beta[:4], g[:4])
    ref_carried = gdn_recurrence_numpy(
        q[4:], k[4:], v[4:], beta[4:], g[4:], ref_fresh[1])
    for got, expected in zip(fresh + carried, ref_fresh + ref_carried):
        assert np.allclose(got, expected, rtol=5e-5, atol=5e-6)
    report = native.report()
    assert report["fresh_and_resumed_parity_complete"]
    assert report["native_calls"] == 2 and report["fallback_calls"] == 0
    print("gdnaccel selftest OK (NumPy + native fresh/resumed parity)")


if __name__ == "__main__":
    _selftest()
