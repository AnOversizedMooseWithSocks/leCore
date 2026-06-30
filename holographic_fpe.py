"""Fractional Power Encoding / Vector Function Architecture, N-dimensional (BLD-7).

WHAT THIS IS, AND WHAT WAS ALREADY HERE
---------------------------------------
Fractional Power Encoding (FPE) encodes a continuous value by raising a fixed random base vector to a real
power -- exp(i * x * theta) per component -- so that a SHIFT of the value becomes a BINDING of the codes, and
the similarity between two codes is a kernel you DESIGN by choosing the phase distribution (Bochner's theorem:
the kernel is that distribution's characteristic function).

holostuff's 1-D `ScalarEncoder` is ALREADY exactly this: `encode(x) = irfft(exp(i*scale*x*phases))` is "raise
the base to power x", its `kernel_at` is the Bochner kernel (sinc for uniform phases, RBF for Gaussian), and
because the engine's `bind` is circular convolution -- which MULTIPLIES the spectra and so ADDS the phases --
`bind(encode(x), encode(s)) == encode(x+s)` to numerical exactness. So the backlog premise that the RBF encoder
is "a locality-preserving approximation" is wrong: 1-D FPE, with shift-as-bind and a designed kernel, has been
in the box all along. (Verified live in test_holographic_fpe.py.)

THE GENUINE ADDITION (this module)
----------------------------------
What was NOT here is the step up from a scalar to a VECTOR domain and to whole FUNCTIONS:

  * N-dimensional encoding -- a point p in R^n is encoded by binding one per-axis 1-D FPE per coordinate.
    A shift along ANY axis is still a single binding, and the similarity is the PRODUCT of the per-axis
    kernels (a product of Gaussians = an n-D RBF). This is the spatial substrate the resonator / scene-
    factoring literature (Frady, Kymn, Olshausen, Sommer) builds on.

  * Compute on functions -- a function f: R^n -> R is represented as a weighted superposition of encoded
    points, f = sum_i w_i encode(p_i). Querying f at q reads sum_i w_i kernel(q, p_i): a holographic
    kernel-density / function evaluation. And the whole function translates by ONE binding --
    bind(f, encode(delta)) = sum_i w_i encode(p_i + delta) -- which generalises the rigid-shift-is-a-bind
    trick the motion compensator already uses, from a single image to an arbitrary function.

KEPT NEGATIVES (measured, in the selftest)
------------------------------------------
  * The standing capacity cliff applies: a function is a bundle, and a bundle of too many atoms drowns each
    one in the others' cross-talk -- query separation (placed point vs empty point) decays as the atom count
    grows. The selftest measures where it falls off; it is finite, like every bundle in the engine.
  * Where a scalar suffices, the n-D machinery buys nothing: 1-D FPE IS the ScalarEncoder, so reach for this
    only when the domain is genuinely multi-dimensional or you need the function algebra.

Only NumPy, the engine's bind/cosine, and the existing ScalarEncoder -- no new dependency, nothing learned.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from holographic_ai import bind, cosine
from holographic_encoders import ScalarEncoder

_BUNDLE_ENCODE_BATCH_ROWS = 512
_FPE_PARALLEL_MIN_ROWS = 1024


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _fpe_parallel_workers(task_count, item_count=0, workers=None, min_items=_FPE_PARALLEL_MIN_ROWS):
    if task_count <= 1:
        return 1
    if workers is not None:
        return max(1, min(int(workers), task_count))
    requested = _env_int("HOLOSTUFF_FPE_THREADS", 0)
    if requested > 0:
        return min(requested, task_count)
    if requested < 0 or item_count < min_items:
        return 1
    return max(1, min(os.cpu_count() or 1, task_count))


class VectorFunctionEncoder:
    """N-dimensional FPE: encode a continuous point in R^n, and represent / query / shift whole functions.

    One per-axis `ScalarEncoder` provides the verified 1-D FPE per coordinate; `encode` binds them. Because
    each axis already has shift-as-bind and a Bochner kernel, the n-D code inherits both: a shift in any axis
    is a binding, and the kernel is the product of the per-axis kernels.
    """

    def __init__(self, n_dims, dim=1024, bounds=None, kernel="rbf", bandwidth=3.0, seed=0):
        # bounds[k] = (lo, hi) sets axis k's working range (values that far apart come out ~orthogonal). RBF
        # phases are the sane default here: a function is a BUNDLE, and the RBF kernel is non-negative and
        # monotone, so the bundle reads as a proper kernel-density estimate rather than oscillating negative.
        if n_dims < 1:
            raise ValueError("n_dims must be >= 1")
        self.n_dims = int(n_dims)
        self.dim = int(dim)
        if bounds is None:
            bounds = [(0.0, 1.0)] * n_dims
        if len(bounds) != n_dims:
            raise ValueError("bounds must have one (lo, hi) pair per dimension")
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        # `bandwidth` may be a SCALAR (isotropic -- the same falloff on every axis, the original behaviour) or a
        # PER-AXIS list/array (ANISOTROPIC / steering, RT-IV1): a small bandwidth makes an axis SMOOTH (wide,
        # slow-falloff kernel), a large bandwidth makes it SHARP. A diagonal anisotropic kernel like this is the
        # bounded form of Milanfar's steering kernel -- n bandwidths, not a per-point covariance that overfits.
        if np.isscalar(bandwidth):
            self.bandwidth = [float(bandwidth)] * self.n_dims
        else:
            self.bandwidth = [float(b) for b in bandwidth]
            if len(self.bandwidth) != self.n_dims:
                raise ValueError(f"per-axis bandwidth must have {self.n_dims} entries, got {len(self.bandwidth)}")
        # One independent base per axis (distinct seeds) so the axes are orthogonal sub-codes; binding them
        # keeps each coordinate separately recoverable and makes the kernel factor across axes.
        self.axes = [
            ScalarEncoder(self.dim, lo=lo, hi=hi, seed=seed * 97 + k + 1, kernel=kernel, bandwidth=self.bandwidth[k])
            for k, (lo, hi) in enumerate(self.bounds)
        ]
        self._half_len = self.dim // 2 + 1
        self._axis_half_phases = [ax.phases[:self._half_len] for ax in self.axes]
        self._rfft_weights = np.ones(self._half_len)
        if self._half_len > 1:
            self._rfft_weights[1:] = 2.0
            if self.dim % 2 == 0:
                self._rfft_weights[-1] = 1.0

    def encode(self, point):
        """The n-D FPE vector for `point` = (x_0, ..., x_{n-1}): bind the per-axis FPE encodings.

        bind is circular convolution = spectrum multiply = phase add, so binding the axes simply stacks their
        independent fractional powers -- and a shift in any axis is therefore another binding (see `shift`)."""
        point = np.atleast_1d(np.asarray(point, float))
        if point.shape[0] != self.n_dims:
            raise ValueError(f"point must have {self.n_dims} coordinates")
        v = self.axes[0].encode(point[0])
        for k in range(1, self.n_dims):
            v = bind(v, self.axes[k].encode(point[k]))
        return v

    def _coerce_points(self, points):
        pts = np.asarray(points, float)
        if self.n_dims == 1:
            if pts.ndim == 0:
                pts = pts.reshape(1, 1)
            elif pts.ndim == 1:
                pts = pts.reshape(-1, 1)
        else:
            pts = np.atleast_2d(pts)
        if pts.ndim != 2 or pts.shape[1] != self.n_dims:
            raise ValueError(f"points must have shape (count, {self.n_dims})")
        return pts

    def _point_spectra(self, pts):
        spectrum = np.ones((pts.shape[0], self._half_len), dtype=np.complex128)
        for k, ax in enumerate(self.axes):
            values = pts[:, k]
            warp_x = getattr(ax, "_warp_x", None)
            if warp_x is not None:
                values = np.interp(values, warp_x, ax._warp_u)
            spectrum *= np.exp(1j * ax.scale * values[:, None] * self._axis_half_phases[k][None, :])
        return spectrum

    def encode_many(self, points):
        """Vectorised n-D FPE encoding for a row stack of points.

        This is algebraically the same as calling encode() for each row: it
        multiplies the per-axis FPE spectra directly, which is exactly what the
        bind loop would do after FFTing each axis code.
        """
        spectrum = self._point_spectra(self._coerce_points(points))
        out = np.fft.irfft(spectrum, n=self.dim, axis=1)
        norms = np.linalg.norm(out, axis=1)
        nz = norms > 0
        out[nz] /= norms[nz, None]
        return np.ascontiguousarray(out)

    def kernel_at(self, delta):
        """The similarity this encoder realises between two points `delta` apart: the PRODUCT of the per-axis
        Bochner kernels. For RBF axes that is a product of Gaussians -- the n-D squared-exponential kernel --
        so you can ASSERT the n-D kernel from the 1-D ones rather than eyeball it (checked in the selftest)."""
        delta = np.atleast_1d(np.asarray(delta, float))
        k = 1.0
        for ax, d in zip(self.axes, delta):
            k *= ax.kernel_at(float(d))
        return float(k)

    def bundle(self, points, weights=None, chunk_size=_BUNDLE_ENCODE_BATCH_ROWS, workers=None):
        """Represent a function f: R^n -> R as a weighted superposition of encoded points,
        f = sum_i w_i encode(p_i). With RBF axes, querying f is a holographic kernel-density estimate."""
        pts = self._coerce_points(points)
        if len(pts) == 0:
            raise ValueError("need at least one point")
        if weights is None:
            weights_arr = None
        else:
            weights_arr = np.asarray(weights, float).ravel()
            if weights_arr.shape[0] != len(pts):
                raise ValueError("weights must match the number of points")
        step = max(1, int(chunk_size))
        spans = [(start, min(start + step, len(pts))) for start in range(0, len(pts), step)]

        def chunk_spectrum(span):
            start, end = span
            chunk = self._point_spectra(pts[start:end])
            chunk_weights = None if weights_arr is None else weights_arr[start:end]
            if chunk_weights is None:
                return chunk.sum(axis=0)
            return chunk_weights @ chunk

        worker_count = _fpe_parallel_workers(len(spans), len(pts), workers)
        if worker_count == 1:
            parts = [chunk_spectrum(span) for span in spans]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                parts = list(executor.map(chunk_spectrum, spans))
        spectrum = np.zeros(self._half_len, dtype=np.complex128)
        for part in parts:
            spectrum += part
        return np.fft.irfft(spectrum, n=self.dim)

    def query(self, function, point):
        """Evaluate the represented function at `point`: cosine(function, encode(point)) reads
        sum_i w_i kernel(point, p_i), up to the bundle's norm -- the function's value, holographically."""
        return float(cosine(function, self.encode(point)))

    def query_many(self, function, points, chunk_size=_BUNDLE_ENCODE_BATCH_ROWS, workers=None):
        """Evaluate a represented function at many points in batched FPE blocks.

        This is the read-side twin of ``bundle``: encode a row stack once, then
        use one matrix-vector multiply per chunk instead of a Python loop of
        ``query(function, point)`` calls. It preserves query() semantics exactly
        up to batched-FFT roundoff.
        """
        pts = self._coerce_points(points)
        fn = np.asarray(function, float)
        fnorm = np.linalg.norm(fn)
        if fnorm == 0:
            return np.zeros(pts.shape[0], dtype=float)
        fn_spectrum = np.conj(np.fft.rfft(fn)) * self._rfft_weights
        step = max(1, int(chunk_size))
        spans = [(start, min(start + step, pts.shape[0])) for start in range(0, pts.shape[0], step)]

        def chunk_query(span):
            start, end = span
            spectra = self._point_spectra(pts[start:end])
            return np.real(spectra @ fn_spectrum) / (self.dim * fnorm)

        worker_count = _fpe_parallel_workers(len(spans), pts.shape[0], workers)
        out = np.empty(pts.shape[0], dtype=float)
        if worker_count == 1:
            parts = [chunk_query(span) for span in spans]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                parts = list(executor.map(chunk_query, spans))
        for (start, end), values in zip(spans, parts):
            out[start:end] = values
        return out

    def shift(self, function, delta):
        """Translate the WHOLE function by `delta` with a single binding:
        bind(f, encode(delta)) = sum_i w_i encode(p_i + delta). Shift-as-bind, lifted from a point to a
        function -- the same rigid-shift-is-a-bind identity the motion compensator uses, generalised."""
        return bind(function, self.encode(delta))


# ---------------------------------------------------------------------------

def _selftest():
    rng = np.random.default_rng(0)

    # (1) n-D shift-as-bind is exact: bind(encode(p), encode(d)) == encode(p+d) in direction (cosine 1).
    enc = VectorFunctionEncoder(n_dims=2, dim=1024, bounds=[(0, 10), (0, 10)], kernel="rbf", bandwidth=3.0, seed=1)
    sb = []
    for _ in range(20):
        p = rng.uniform(0, 5, 2)
        d = rng.uniform(0, 3, 2)
        sb.append(cosine(bind(enc.encode(p), enc.encode(d)), enc.encode(p + d)))
    assert min(sb) > 0.999, f"n-D shift-as-bind not exact: {min(sb)}"

    # (2) the n-D kernel is the PRODUCT of the per-axis Bochner kernels (assert, don't eyeball).
    p = np.array([3.0, 4.0])
    for q in ([3.5, 4.0], [3.0, 5.0], [4.0, 5.0]):
        q = np.array(q)
        meas = cosine(enc.encode(p), enc.encode(q))
        pred = enc.kernel_at(q - p)
        assert abs(meas - pred) < 0.02, f"product kernel mismatch at {q}: {meas} vs {pred}"

    # (3) a function reads HIGH at its placed points and LOW at an empty spot.
    pts = [(2.0, 2.0), (7.0, 3.0), (4.0, 6.0)]
    wts = [1.0, 0.6, 0.8]
    f = enc.bundle(pts, wts)
    at_points = [enc.query(f, p) for p in pts]
    at_empty = enc.query(f, (9.5, 9.5))
    assert min(at_points) > 3 * at_empty, f"function does not localise: points {at_points} vs empty {at_empty}"

    # (4) shift-as-bind on a SINGLE atom is exact; on a function the peak MOVES to point+delta.
    single = enc.encode((2.0, 2.0))
    moved = enc.shift(single, (1.5, 1.0))
    assert cosine(moved, enc.encode((3.5, 3.0))) > 0.999, "single-atom function shift not exact"
    fs = enc.shift(f, (1.0, 1.0))                       # whole function moves by (1,1)
    assert enc.query(fs, (3.0, 3.0)) > enc.query(fs, (2.0, 2.0)), "function peak did not move under shift"

    # (5) KEPT NEGATIVE: the capacity cliff -- query separation (placed vs empty) decays as atoms pile up.
    seps = {}
    for K in (2, 8, 32, 128):
        ps = rng.uniform(0, 10, (K, 2))
        g = enc.bundle(ps)
        placed = np.mean([enc.query(g, ps[i]) for i in range(min(K, 12))])
        empty = np.mean([enc.query(g, rng.uniform(0, 10, 2)) for _ in range(12)])
        seps[K] = placed - empty
    assert seps[2] > seps[128], f"capacity cliff not monotone: {seps}"      # more atoms -> less separation

    # (6) determinism: same seed -> identical codes.
    e2 = VectorFunctionEncoder(n_dims=2, dim=1024, bounds=[(0, 10), (0, 10)], kernel="rbf", bandwidth=3.0, seed=1)
    assert np.allclose(enc.encode((3.0, 4.0)), e2.encode((3.0, 4.0))), "not deterministic"

    print("holographic_fpe selftest OK:")
    print(f"  n-D shift-as-bind cosine    min {min(sb):.5f}  (exact)")
    print(f"  product-kernel check        within 0.02 of per-axis product")
    print(f"  function localises          points {[round(v,2) for v in at_points]} vs empty {at_empty:.2f}")
    print(f"  capacity cliff (sep vs K)   " + "  ".join(f"K={k}:{seps[k]:+.2f}" for k in seps))


if __name__ == "__main__":
    _selftest()
