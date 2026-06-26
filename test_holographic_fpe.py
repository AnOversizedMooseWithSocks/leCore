"""BLD-7: N-dimensional Fractional Power Encoding + the compute-on-functions algebra (holographic_fpe.py),
and the finding that 1-D FPE was already the ScalarEncoder."""
import numpy as np

from holographic_ai import bind, cosine
from holographic_encoders import ScalarEncoder
from holographic_fpe import VectorFunctionEncoder, _selftest


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


def test_deterministic():
    a = VectorFunctionEncoder(2, dim=512, bounds=[(0, 5), (0, 5)], seed=9)
    b = VectorFunctionEncoder(2, dim=512, bounds=[(0, 5), (0, 5)], seed=9)
    assert np.allclose(a.encode((1.0, 2.0)), b.encode((1.0, 2.0)))
