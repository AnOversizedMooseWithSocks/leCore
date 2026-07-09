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
import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, cosine
from holographic.io_and_interop.holographic_encoders import ScalarEncoder


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

    def kernel_at(self, delta):
        """The similarity this encoder realises between two points `delta` apart: the PRODUCT of the per-axis
        Bochner kernels. For RBF axes that is a product of Gaussians -- the n-D squared-exponential kernel --
        so you can ASSERT the n-D kernel from the 1-D ones rather than eyeball it (checked in the selftest).

        IN EXPECTATION. The realized inner product <encode(p), encode(q)> equals this product plus an O(1/sqrt(D))
        deviation -- deterministic, but pseudo-random in the gap. Measured mean |error| 0.0306 / 0.0111 / 0.0061 /
        0.0028 at D = 1,024 / 4,096 / 16,384 / 65,536, tracking 1/sqrt(D). That floor is where this representation's
        crosstalk budget lives (see the H5 block below `shift`), and it is why a bundled function's error falls with
        D rather than with restraint about how many points you bundle."""
        delta = np.atleast_1d(np.asarray(delta, float))
        k = 1.0
        for ax, d in zip(self.axes, delta):
            k *= ax.kernel_at(float(d))
        return float(k)

    def bundle(self, points, weights=None):
        """Represent a function f: R^n -> R as a weighted superposition of encoded points,
        f = sum_i w_i encode(p_i). With RBF axes, querying f is a holographic kernel-density estimate."""
        points = list(points)
        if not points:
            raise ValueError("need at least one point")
        if weights is None:
            weights = [1.0] * len(points)
        f = None
        for w, p in zip(weights, points):
            term = float(w) * self.encode(p)
            f = term if f is None else f + term
        return f

    def query(self, function, point):
        """Evaluate the represented function at `point`: cosine(function, encode(point)) reads
        sum_i w_i kernel(point, p_i), up to the bundle's norm -- the function's value, holographically."""
        return float(cosine(function, self.encode(point)))

    def shift(self, function, delta):
        """Translate the WHOLE function by `delta` with a single binding:
        bind(f, encode(delta)) = sum_i w_i encode(p_i + delta). Shift-as-bind, lifted from a point to a
        function -- the same rigid-shift-is-a-bind identity the motion compensator uses, generalised."""
        return bind(function, self.encode(delta))

    # ==============================================================================================================
    # H5 -- THE N-D NYQUIST, AND WHERE THE CROSSTALK BUDGET ACTUALLY LIVES.
    #
    # The 1-D bake learned (holographic_shader, H3) that the phasor bandwidth B must exceed the signal's maximum
    # angular frequency, and that below it the code does not gently blur -- it returns a confident, smooth-looking,
    # WRONG answer, and raises nothing. That lesson never reached this class, whose `bandwidth` defaults to 3.0 on
    # every axis regardless of the data.
    #
    # MEASURED (a separable 2-D sine at 2 cycles/unit per axis -> per-axis w_max = 12.57; 40x40 samples, D=8192;
    # scale-free RMS against the truth, so 1.0 means "carries no information at all"):
    #
    #       B          B / w_max        scale-free RMS
    #       1.00          0.08              0.9924
    #       3.00          0.24              1.0019      <-- THE LIBRARY DEFAULT. Worse than predicting the mean.
    #       6.28          0.50              0.4744
    #      12.57          1.00              0.1233
    #      18.85          1.50              0.0796      <-- the sweet spot, exactly as in 1-D
    #      31.42          2.50              0.1685      <-- past it, capacity buys frequencies that aren't there
    #
    # The 1-D law transfers unchanged: B ~ 1.5 * w_max per axis. `for_grid` chooses it from the data.
    #
    # THE CROSSTALK BUDGET THE BACKLOG ASKED US TO MEASURE -- and it is not where it was expected. The backlog looked
    # for a budget on the NUMBER OF BUNDLED POINTS. There isn't one: a bundled function is only ever summed, never
    # unbound, so it sits on the good side of the same line the H2 gather sits on. The budget is in the KERNEL.
    # `kernel_at` says the n-D similarity is the PRODUCT of the per-axis kernels. That is true only in expectation;
    # the realized kernel deviates, and the deviation is a clean 1/sqrt(D):
    #
    #       D              1,024    4,096   16,384   65,536
    #       mean |error|   0.0306   0.0111   0.0061   0.0028
    #       1/sqrt(D)      0.0313   0.0156   0.0078   0.0039
    #
    # So every kernel evaluation carries an O(1/sqrt(D)) error, deterministic but pseudo-random in the gap. Summed
    # over N points it accumulates like sqrt(N)/sqrt(D) against a signal that also grows like sqrt(N) -- which is
    # why the measured error is FLAT IN N. Adding samples never costs you anything (2-D sine at 2 cycles/axis,
    # B = 1.5 * w_max):
    #
    #                N=100    N=400   N=1600   N=3600
    #     D= 2,048   0.2734   0.1976   0.1911   0.1890
    #     D= 8,192   0.1875   0.0848   0.0796   0.0779
    #     D=32,768   0.1309   0.0469   0.0452   0.0448
    #
    # SUPERSEDED, and left here because getting it wrong is the instructive part: this table also seemed to say
    # "and the error falls with D, so spend dimensions." It does not, in general. Those columns fall with D because
    # that particular signal, at that margin, happened to be VARIANCE-limited. Read the re-measurement below before
    # believing any row of the table above about D.
    #
    # KEPT NEGATIVE, and it is the one to read before trusting a number: at the recommended margin the normalized
    # n-D readout is a SHAPE estimator, not a calibrated one. The RBF kernel is a Gaussian smoother, so it attenuates
    # the signal's own top frequency, and the crosstalk floor attenuates it further.
    #
    # RE-MEASURED (2-D sine sin(2pi x)cos(2pi y), 40x40 grid, 200 random query points, scale-free RMS / std, and the
    # amplitude gain against the truth, where 1.0 would be calibrated). The earlier table here reported a single
    # condition and concluded "the error falls with D". It does not -- not at the default margin:
    #
    #       margin      D=4,096          D=16,384         D=65,536        gain @ D=65,536
    #         1.5      RMS 0.1179       RMS 0.1174       RMS 0.1191            0.66
    #         2.5      RMS 0.0715       RMS 0.0566       RMS 0.0315            0.84
    #         4.0      RMS 0.1256       RMS 0.0777       RMS 0.0320            0.91
    #         6.0      RMS 0.1972       RMS 0.1546       RMS 0.0568            0.88
    #
    # BANDWIDTH IS A BIAS-VARIANCE DIAL AND `dim` IS THE VARIANCE BUDGET. At margin 1.5 the error is pure BIAS -- a
    # kernel too smooth to hold the function -- and sixteen times the dimension changes nothing (0.1179 -> 0.1191).
    # Raise the margin and the bias falls, but the narrower kernel has fewer effective neighbours and so more
    # crosstalk, which is exactly what D pays for: margin 4.0 and 6.0 are WORSE than 1.5 at D=4,096 and much better
    # at D=65,536. The knee is around margin 2.5. Raise margin and dim TOGETHER, or leave both alone and read shape.
    # Do not read amplitudes off the default.
    #
    # AND THE CAUSAL VARIABLE IS B, NOT `margin`. The two tables above disagree about D only because margin is a
    # RATIO -- B = margin * w_max -- so "margin 1.5" means B = 18.8 on a 2-cycle sine and B = 9.4 on a 1-cycle one.
    # Hold B fixed and the confound disappears (40x40 grid, bandwidth supplied, not probed):
    #
    #        B      signal      D=2,048   D=8,192   D=32,768     does D pay?
    #       9.4    1-cycle       0.1315    0.1117    0.1104          no
    #      18.8    1-cycle       0.1220    0.0777    0.0431         YES
    #      18.8    2-cycle       0.1801    0.0905    0.0547         YES
    #      28.3    2-cycle       0.3240    0.1451    0.0419         YES
    #
    # Two different signals at the SAME B behave the same; one signal at two different B's does not. Both tables
    # above were right about their own B, and neither said what it was. STATE B, NOT MARGIN.
    #
    # THE DIAGNOSTIC, and it costs one extra bake: DOUBLE D. If the error drops, you are variance-limited -- keep
    # spending dimension. If it does not move, you are BIAS-limited, and no amount of dimension will help; raise the
    # margin instead. You cannot buy your way out of a bad bandwidth with dimension.
    # ==============================================================================================================
    @classmethod
    def for_grid(cls, grids, values, dim=1024, margin=1.5, seed=0, kernel="rbf"):
        """Build an encoder whose per-axis bandwidths are probed FROM THE DATA -- the n-D `bake_1d(bandwidth=None)`.

        `grids`  -- one 1-D array of UNIFORMLY spaced coordinates per axis.
        `values` -- the sampled function, shaped (len(grids[0]), len(grids[1]), ...).

        Sets B_k = margin * w_max(axis k). WHY THIS EXISTS: the class default of 3.0 measures at 1.0019 scale-free
        RMS on a 2-D sine -- literally no information, silently. See the block above, including why `margin` trades
        amplitude fidelity against shape fidelity."""
        w = axis_bandwidths(grids, values)
        bounds = [(float(g[0]), float(g[-1])) for g in grids]
        return cls(len(grids), dim=dim, bounds=bounds, kernel=kernel, seed=seed,
                   bandwidth=[max(1.0, float(margin) * wk) for wk in w])

    def bundle_normalized(self, points, weights):
        """Bundle the function AND the constant 1 over the same points, so a query can divide out the sample density.

        Returns (function, density). `query_normalized(f, d, p)` is then a kernel AVERAGE (Nadaraya-Watson) rather
        than a kernel SUM, which is what makes the readout an estimate of f(p) instead of f(p) times however densely
        you happened to sample. Same lesson as `holographic_shader.fetch(normalize=True)`, one dimension up."""
        pts = [np.atleast_1d(np.asarray(p, float)) for p in points]
        w = np.asarray(list(weights), float)
        if w.size != len(pts):
            raise ValueError("bundle_normalized: %d points but %d weights" % (len(pts), w.size))
        f = np.zeros(self.dim)
        d = np.zeros(self.dim)
        for wi, p in zip(w, pts):
            z = self.encode(p)
            f += float(wi) * z
            d += z
        return f, d

    def query_normalized(self, function, density, point):
        """Evaluate the bundled function at `point` as a kernel AVERAGE: <f, Z(p)> / <density, Z(p)>.

        Unlike `query` (a cosine, hence scale-free but not calibrated) this returns f(point) directly, with no
        fitted constant, and stays correct when the samples are unevenly spaced."""
        z = self.encode(point)
        den = float(np.dot(density, z))
        if den == 0.0:
            raise ValueError("query_normalized: zero density at this point -- no samples are near it")
        return float(np.dot(function, z)) / den


def axis_bandwidths(grids, values, energy=0.995):
    """The maximum angular frequency along each axis of a gridded function -- what an n-D bake must resolve.

    Pools the POWER SPECTRUM over all 1-D slices along an axis, then applies the energy criterion once. Taking the
    per-slice maximum instead looks equivalent and is not: a slice through a node of a separable function carries
    almost no signal, so its 99.5%-energy cut lands on whatever noise is there. MEASURED on a 2-D sine (true w_max
    12.57) with 1e-6 of added noise -- an amount no one would notice -- the per-slice max reports **248.22** while
    the pooled spectrum reports 12.41. A twentyfold over-estimate of the bandwidth spends the code's capacity on
    frequencies that do not exist.

    Two honest limits. (1) Requires UNIFORMLY spaced coordinates: the probe is an FFT, and on scattered samples it
    reads the spacing jitter as high-frequency content. (2) On a coarse grid the signal's frequency rarely lands on
    an FFT bin, so leakage pushes the estimate up a bin -- measured 18.38 at 40 samples, converging to 12.41 / 12.49
    at 81 / 161. It errs high, which costs a little capacity and never returns garbage. That is the right direction
    to be wrong in."""
    V = np.asarray(values, float)
    out = []
    for k, g in enumerate(grids):
        g = np.asarray(g, float)
        span = float(g[-1] - g[0])
        moved = np.moveaxis(V, k, 0).reshape(V.shape[k], -1)         # every 1-D slice along axis k, as columns
        if moved.shape[0] < 4 or span <= 0:
            out.append(0.0)
            continue
        centred = moved - moved.mean(axis=0, keepdims=True)
        power = (np.abs(np.fft.rfft(centred, axis=0)) ** 2).sum(axis=1)   # pool every slice's power, then cut once
        if power.sum() <= 0:
            out.append(0.0)
            continue
        freqs = np.fft.rfftfreq(moved.shape[0], d=span / (moved.shape[0] - 1))
        keep = np.searchsorted(np.cumsum(power) / power.sum(), energy)
        out.append(2.0 * np.pi * float(freqs[min(keep, freqs.size - 1)]))
    return out


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
