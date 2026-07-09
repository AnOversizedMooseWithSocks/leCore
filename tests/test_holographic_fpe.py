"""BLD-7: N-dimensional Fractional Power Encoding + the compute-on-functions algebra (holographic_fpe.py),
and the finding that 1-D FPE was already the ScalarEncoder."""
import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, cosine
from holographic.io_and_interop.holographic_encoders import ScalarEncoder
from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder, _selftest


def test_selftest_passes():
    _selftest()


def test_scalar_encoder_is_already_fpe_shift_as_bind_and_kernel():
    # The finding: holostuff's 1-D ScalarEncoder IS a fractional power encoder -- encode(x) = "base^x" -- so it
    # already has shift-as-bind (bind multiplies the spectra = adds the positions) and a Bochner kernel.
    enc = ScalarEncoder(dim=1024, lo=0.0, hi=10.0, seed=1, kernel="rbf", bandwidth=3.0)
    rng = np.random.default_rng(0)
    for _ in range(15):
        x = rng.uniform(0, 5)
        s = rng.uniform(0, 4)
        assert cosine(bind(enc.encode(x), enc.encode(s)), enc.encode(x + s)) > 0.999
    # and the measured similarity matches the encoder's own analytic kernel
    for dx in (0.2, 0.5, 1.0):
        assert abs(cosine(enc.encode(2.0), enc.encode(2.0 + dx)) - enc.kernel_at(dx)) < 0.02


def test_nd_shift_is_a_binding():
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 10), (0, 10)], seed=1)
    rng = np.random.default_rng(1)
    for _ in range(15):
        p = rng.uniform(0, 5, 2)
        d = rng.uniform(0, 3, 2)
        assert cosine(bind(enc.encode(p), enc.encode(d)), enc.encode(p + d)) > 0.999


def test_nd_kernel_is_product_of_axis_kernels():
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 10), (0, 10)], seed=1)
    p = np.array([3.0, 4.0])
    for q in ([3.5, 4.0], [3.0, 5.0], [4.0, 5.0]):
        q = np.array(q)
        assert abs(cosine(enc.encode(p), enc.encode(q)) - enc.kernel_at(q - p)) < 0.02


def test_function_localises_at_its_points():
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 10), (0, 10)], seed=2)
    pts = [(2.0, 2.0), (7.0, 3.0), (4.0, 6.0)]
    f = enc.bundle(pts, [1.0, 0.6, 0.8])
    assert min(enc.query(f, p) for p in pts) > 3 * enc.query(f, (9.5, 9.5))


def test_function_translates_under_one_binding():
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 10), (0, 10)], seed=2)
    # a single atom shifts exactly...
    moved = enc.shift(enc.encode((2.0, 2.0)), (1.5, 1.0))
    assert cosine(moved, enc.encode((3.5, 3.0))) > 0.999
    # ...and a whole function's peak moves with it
    f = enc.bundle([(2.0, 2.0), (7.0, 3.0)], [1.0, 0.7])
    fs = enc.shift(f, (1.0, 1.0))
    assert enc.query(fs, (3.0, 3.0)) > enc.query(fs, (2.0, 2.0))


def test_capacity_cliff_is_a_kept_negative():
    # A function is a bundle; query separation (placed vs empty) decays as atoms pile up. The cliff is real.
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 10), (0, 10)], seed=3)
    rng = np.random.default_rng(7)

    def sep(K):
        ps = rng.uniform(0, 10, (K, 2))
        g = enc.bundle(ps)
        placed = np.mean([enc.query(g, ps[i]) for i in range(min(K, 12))])
        empty = np.mean([enc.query(g, rng.uniform(0, 10, 2)) for _ in range(12)])
        return placed - empty

    assert sep(2) > sep(128)


def test_function_shape_reconstruction_does_not_cap_so_overlap_add_chunking_is_a_no_op():
    """The COMPLEMENT of the detection cliff above, and the kept negative for chunking-transfer item S1.
    The detection separation (is THIS point placed?) decays with K, but the SHAPE of the reconstructed
    function -- the kernel-smoothed signal an overlap-add scheme would try to rebuild -- is preserved at
    any domain length: FPE codes are shift-invariant powers of one base, so the inner-product readout is the
    EXACT kernel sum (the finite-dim error is a deterministic sidelobe, not √N noise). So a long-domain
    function reconstructs near-perfectly from a SINGLE bundle, and overlap-add chunking (which was the
    'elegant rhyme' in the sweep) only adds boundary-incomplete error -- measured corr ~0 for both hard-cut
    and Hann overlap-add. Conclusion: chunking helps DECODE-VIA-CLEANUP, not LINEAR-FUNCTIONAL EVALUATION."""
    dim, D, bw = 1024, 400, 2.0
    enc = VectorFunctionEncoder(1, dim=dim, bounds=[(0, D)], kernel="rbf", bandwidth=bw, seed=0)
    xs = np.arange(D, dtype=float)
    ys = np.sin(0.6 * xs) + 0.5 * np.sin(0.17 * xs + 1.0) + 0.3 * np.cos(0.4 * xs)
    E = np.stack([enc.encode(x) for x in xs])
    qs = np.linspace(4, D - 4, 200)
    EQ = np.stack([enc.encode(q) for q in qs])
    ideal = np.array([ys @ np.array([enc.kernel_at(q - x) for x in xs]) for q in qs])  # noise-free kernel sum
    rec = EQ @ (ys[:, None] * E).sum(0)                                                # single bundle, raw readout
    a, b = rec - rec.mean(), ideal - ideal.mean()
    corr = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert corr > 0.99            # near-perfect SHAPE at N=400 -> no capacity problem for chunking to solve


def test_deterministic():
    a = VectorFunctionEncoder(2, dim=512, bounds=[(0, 5), (0, 5)], seed=9)
    b = VectorFunctionEncoder(2, dim=512, bounds=[(0, 5), (0, 5)], seed=9)
    assert np.allclose(a.encode((1.0, 2.0)), b.encode((1.0, 2.0)))


# ======================================================================================================
# H5 -- the n-D Nyquist, and where the crosstalk budget actually lives.
# ======================================================================================================
def _g2(P):
    return np.sin(2 * np.pi * 2.0 * P[..., 0]) * np.cos(2 * np.pi * 2.0 * P[..., 1])


def _grid(n=40):
    ax = np.linspace(0.0, 1.0, n)
    P = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1)
    return ax, P, _g2(P)


def _scale_free(got, truth):
    """1.0 means 'carries no information at all' -- the same reading as a random vector."""
    got = np.asarray(got, float)
    c = float(np.dot(got, truth) / np.dot(got, got)) if np.dot(got, got) > 0 else 0.0
    return float(np.sqrt(np.mean((c * got - truth) ** 2)) / np.std(truth))


def test_the_library_default_bandwidth_carries_no_information_and_probing_fixes_it():
    """The class default of 3.0 is not "a bit blurry". Measured scale-free RMS 1.0015 -- worse than the mean."""
    from holographic.rendering.holographic_shader import bake_nd, fetch_nd
    ax, P, V = _grid()
    qs = np.random.default_rng(0).uniform(0.15, 0.85, (120, 2))
    truth = _g2(qs)

    bad = VectorFunctionEncoder(2, dim=8192, bounds=[(0, 1), (0, 1)], bandwidth=3.0, seed=0)
    F, D = bad.bundle_normalized(P.reshape(-1, 2), V.reshape(-1))
    assert _scale_free([bad.query_normalized(F, D, q) for q in qs], truth) > 0.9

    good = bake_nd([ax, ax], V, dim=8192, seed=0)          # bandwidth probed from the data
    assert _scale_free(fetch_nd(good, qs), truth) < 0.20   # measured 0.101


def test_axis_bandwidths_pools_slices_because_a_near_zero_slice_reads_pure_noise():
    """A slice through a node of a separable function carries no signal, so its 99.5%-energy cut lands on noise.
    Measured with 1e-6 of added noise: per-slice max 248.22, pooled 12.41, true 12.57."""
    from holographic.rendering.holographic_shader import bandwidth_probe
    from holographic.sampling_and_signal.holographic_fpe import axis_bandwidths
    ax = np.linspace(0.0, 1.0, 81)
    P = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1)
    V = _g2(P) + 1e-6 * np.random.default_rng(0).standard_normal(P.shape[:2])

    pooled = axis_bandwidths([ax, ax], V)
    sl = np.moveaxis(V, 0, 0).reshape(V.shape[0], -1)
    per_slice_max = max(bandwidth_probe(ax, sl[:, j]) for j in range(sl.shape[1]))
    assert per_slice_max > 100.0                            # measured 248.22
    assert abs(pooled[0] - 2 * np.pi * 2.0) < 0.15 * 2 * np.pi * 2.0    # measured 12.41 vs 12.57

    # on a coarse grid the probe errs HIGH (spectral leakage), which is the safe direction
    coarse = axis_bandwidths(*[[np.linspace(0, 1, 40)] * 2, _grid(40)[2]])
    assert coarse[0] >= 2 * np.pi * 2.0 * 0.95


def test_the_nd_kernel_is_the_product_only_in_expectation_with_a_one_over_sqrt_d_floor():
    """`kernel_at` promises the product of the per-axis kernels. The realized inner product deviates by ~1/sqrt(D),
    and THAT is this representation's crosstalk budget -- not the number of points you bundle."""
    rng = np.random.default_rng(0)
    devs = {}
    for D in (1024, 16384):
        enc = VectorFunctionEncoder(2, dim=D, bounds=[(0, 1), (0, 1)], bandwidth=[18.0, 18.0], seed=0)
        e = [abs(float(np.dot(enc.encode(p), enc.encode(q))) - enc.kernel_at(p - q))
             for p, q in ((rng.uniform(0, 1, 2), rng.uniform(0, 1, 2)) for _ in range(50))]
        devs[D] = float(np.mean(e))
    assert devs[1024] < 0.06 and devs[16384] < 0.015       # measured 0.031 and 0.006
    assert devs[1024] / devs[16384] > 2.5                  # 16x the dimension -> ~4x less deviation


def test_bundling_more_points_never_hurts_and_bandwidth_is_a_bias_variance_dial():
    """Two separate claims, and only one of them is "spend dimensions".

    (a) N: a bundled function is only ever SUMMED, never unbound, so there is no capacity wall -- the same side of
        the line the H2 gather sits on. More samples never hurt.
    (b) D: dimension is the VARIANCE budget, not a cure-all. At a margin too small for the signal the error is pure
        BIAS -- a kernel too smooth to hold the function -- and sixteen times the dimension changes nothing. That
        is the diagnostic: double D, and if nothing moves, raise the margin instead."""
    from holographic.rendering.holographic_shader import bake_nd, fetch_nd
    qs = np.random.default_rng(0).uniform(0.15, 0.85, (120, 2))
    truth = _g2(qs)

    # (a) more bundled points never hurt
    errs_n = [_scale_free(fetch_nd(bake_nd(*[[np.linspace(0, 1, n)] * 2, _grid(n)[2]], dim=8192, seed=0), qs), truth)
              for n in (10, 40)]
    assert errs_n[1] < errs_n[0] / 2.0, errs_n             # measured 0.708 -> 0.104

    # (b) BIAS-limited: a 1-cycle sine at the default margin does not improve with 16x the dimension
    ax = np.linspace(0.0, 1.0, 40)
    P = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1)
    g1 = np.sin(2 * np.pi * P[..., 0]) * np.cos(2 * np.pi * P[..., 1])
    q1 = np.random.default_rng(0).uniform(0.05, 0.95, (200, 2))
    t1 = np.sin(2 * np.pi * q1[:, 0]) * np.cos(2 * np.pi * q1[:, 1])

    def sf1(got):
        c = float(np.dot(got, t1) / np.dot(got, got))
        return float(np.sqrt(np.mean((c * got - t1) ** 2)) / np.std(t1))

    lo = sf1(fetch_nd(bake_nd([ax, ax], g1, dim=4096, margin=1.5), q1))
    hi = sf1(fetch_nd(bake_nd([ax, ax], g1, dim=65536, margin=1.5), q1))
    assert hi > 0.85 * lo, (lo, hi)                        # measured 0.1179 -> 0.1191: a bias FLOOR

    # ...and the fix is the margin, not the dimension
    knee = sf1(fetch_nd(bake_nd([ax, ax], g1, dim=65536, margin=2.5), q1))
    assert knee < lo / 2.0, (lo, knee)                     # measured 0.0315


def test_nd_readout_is_a_shape_estimator_not_a_calibrated_one():
    """KEPT NEGATIVE, pinned. The RBF kernel is a Gaussian smoother and the crosstalk floor attenuates further, so
    at the default margin the amplitude gain is ~0.6. A wider margin plus more dimensions buys amplitude back."""
    from holographic.rendering.holographic_shader import bake_nd, fetch_nd
    ax, _, V = _grid(40)
    qs = np.random.default_rng(0).uniform(0.15, 0.85, (300, 2))
    truth = _g2(qs)
    gain = lambda got: float(np.dot(got, truth) / np.dot(truth, truth))

    g_default = gain(fetch_nd(bake_nd([ax, ax], V, dim=8192, seed=0), qs))
    g_wide = gain(fetch_nd(bake_nd([ax, ax], V, dim=32768, seed=0, margin=4.0), qs))
    assert g_default < 0.75, g_default                      # measured 0.580 -- do NOT read amplitudes off this
    assert g_wide > 0.85, g_wide                            # measured 0.943
    assert _scale_free(fetch_nd(bake_nd([ax, ax], V, dim=8192, seed=0), qs), truth) < 0.20   # ...shape is fine
