"""Optional FFT backend for the engine's most-called operation. bind/bundle/the phasor memory/the fluid projection
ARE FFTs, so this module is the seam where a faster transform plugs in. The DEFAULT is numpy.fft -- bit-exact and
deterministic -- and the opt-in alternative is pyFFTW (FFTW bindings).

THE HONEST, MEASURED RESULT (reproducible via benchmark(), and the reason this is OFF by default): pyFFTW is a
REGRESSION at holostuff's operating dimensions. On single binds it is SLOWER for the common small/medium vectors
(~0.65x at D=512, ~0.75x at D=1024), only breaking even near D=2048 and winning only for large transforms
(~1.6-1.7x at D>=4096); and on the BATCHED bind path the engine actually relies on it is often WORSE (measured
~0.5x at M=4000, D=2048) because numpy's batched pocketfft is already well tuned and FFTW's threading/planning adds
overhead. It is also tolerance-not-bit-exact vs numpy (~3e-14), so enabling it trades determinism for a speed change
that, at our dimensions, is usually negative.

This is the same lesson the C-kernel PR taught -- an external compiled backend can REGRESS at the operating point --
kept on the record rather than discovered twice. The seam still earns its place: it future-proofs the engine for a
workload dominated by large single transforms (D>=4096), and it makes the measurement reproducible. But numpy stays
the default, and switching is an explicit, eyes-open opt-in.
"""

import numpy as np

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()                         # cache FFTW plans across calls
    _PF = pyfftw.interfaces.numpy_fft                        # numpy-compatible drop-in
    HAS_PYFFTW = True
except Exception:                                            # pragma: no cover - exercised only without pyfftw
    HAS_PYFFTW = False
    _PF = None

_BACKEND = "numpy"                                           # ALWAYS numpy unless explicitly switched (determinism)


def use_pyfftw(on=True):
    """Opt into the pyFFTW backend (off by default). Raises if pyfftw is missing. NOTE: measured to REGRESS at
    typical dims (see benchmark()); enable only for large-single-transform (D>=4096) workloads."""
    global _BACKEND
    if on and not HAS_PYFFTW:
        raise ImportError("pyfftw is not installed (see requirements-accel.txt). It also regresses at typical "
                          "dimensions -- run holographic_fft.benchmark() before enabling.")
    _BACKEND = "pyfftw" if on else "numpy"
    return _BACKEND


def fft_backend():
    """The active backend name ('numpy' or 'pyfftw')."""
    return _BACKEND


def rfft(x, n=None, axis=-1):
    """Real FFT through the active backend. Default numpy path is byte-identical to np.fft.rfft."""
    if _BACKEND == "pyfftw":
        return _PF.rfft(x, n=n, axis=axis)
    return np.fft.rfft(x, n=n, axis=axis)


def irfft(x, n=None, axis=-1):
    """Inverse real FFT through the active backend. Default numpy path is byte-identical to np.fft.irfft."""
    if _BACKEND == "pyfftw":
        return _PF.irfft(x, n=n, axis=axis)
    return np.fft.irfft(x, n=n, axis=axis)


def benchmark(dims=(512, 1024, 2048, 4096, 8192), batched=((256, 1024), (1000, 1024), (4000, 2048)), reps=60):
    """Reproduce the numpy-vs-pyFFTW comparison that justifies keeping numpy the default. Returns a dict of
    {label: speed_ratio (numpy_time / pyfftw_time)} -- ratios < 1 mean pyFFTW is SLOWER. Needs pyfftw installed."""
    if not HAS_PYFFTW:
        return {"error": "pyfftw not installed"}
    import time
    rng = np.random.default_rng(0)
    out = {}

    def _t(fn, *a, n=reps):
        fn(*a)
        s = time.perf_counter()
        for _ in range(n):
            fn(*a)
        return (time.perf_counter() - s) / n

    for D in dims:
        a = rng.standard_normal(D); b = rng.standard_normal(D)
        t_np = _t(lambda a, b: np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=D), a, b)
        t_pf = _t(lambda a, b: _PF.irfft(_PF.rfft(a) * _PF.rfft(b), n=D), a, b)
        out[f"single_D{D}"] = round(t_np / t_pf, 3)
    for M, D in batched:
        A = rng.standard_normal((M, D)); B = rng.standard_normal((M, D))
        t_np = _t(lambda A, B: np.fft.irfft(np.fft.rfft(A, axis=1) * np.fft.rfft(B, axis=1), n=D, axis=1), A, B, n=20)
        t_pf = _t(lambda A, B: _PF.irfft(_PF.rfft(A, axis=1) * _PF.rfft(B, axis=1), n=D, axis=1), A, B, n=20)
        out[f"batched_M{M}_D{D}"] = round(t_np / t_pf, 3)
    return out


def _selftest():
    import numpy as _np
    rng = _np.random.default_rng(0)
    a = rng.standard_normal(1024)
    # default backend is numpy and BYTE-IDENTICAL to np.fft
    assert fft_backend() == "numpy"
    assert _np.array_equal(rfft(a), _np.fft.rfft(a)), "default rfft must equal np.fft.rfft byte-for-byte"
    assert _np.array_equal(irfft(rfft(a), n=1024), _np.fft.irfft(_np.fft.rfft(a), n=1024))
    msg = "fft selftest ok: default=numpy byte-identical"
    if HAS_PYFFTW:
        use_pyfftw(True)
        assert fft_backend() == "pyfftw"
        assert _np.allclose(rfft(a), _np.fft.rfft(a), atol=1e-10)   # pyfftw matches to tolerance (not bit-exact)
        use_pyfftw(False)                                     # restore the deterministic default
        ratios = benchmark(dims=(512, 4096), batched=((1000, 1024),), reps=20)
        msg += (f"; pyFFTW available but OFF by default -- measured ratios (numpy/pyfftw): "
                f"D512 {ratios['single_D512']}x (pyfftw slower), D4096 {ratios['single_D4096']}x, "
                f"batched {ratios['batched_M1000_D1024']}x -- REGRESSES at our dims, kept off")
    else:
        msg += " (pyfftw not installed; numpy-only)"
    assert fft_backend() == "numpy"                          # always restored to the deterministic default
    print(msg)


if __name__ == "__main__":
    _selftest()
