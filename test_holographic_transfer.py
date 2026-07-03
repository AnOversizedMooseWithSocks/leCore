"""Shared kernel scatter/gather: the ONE bundle/readout unifying fields (bilinear), MPM (B-spline), splat (Gaussian)."""
import numpy as np
from holographic_transfer import scatter, gather


def test_scatter_gather_adjoint():
    rng = np.random.default_rng(0)
    for kern in ("bilinear", "bspline"):
        pts = rng.uniform(3, 13, (40, 2)); v = rng.standard_normal(40); f = rng.standard_normal((16, 16))
        lhs = float(np.sum(scatter(pts, v, (16, 16), kernel=kern) * f))
        rhs = float(np.sum(v * gather(f, pts, kernel=kern)))
        assert abs(lhs - rhs) < 1e-9


def test_partition_of_unity_preserves_total():
    rng = np.random.default_rng(1)
    for kern in ("bilinear", "bspline"):
        pts = rng.uniform(3, 13, (50, 2))
        assert abs(scatter(pts, np.ones(50), (16, 16), kernel=kern).sum() - 50.0) < 1e-9


def test_reproduces_fields_bilinear():
    from holographic_fields import scatter_to_field, sample_field
    rng = np.random.default_rng(2)
    pos = rng.uniform(0, 20, (30, 2)); vals = rng.standard_normal(30)
    assert np.allclose(scatter(pos[:, ::-1], vals, (20, 20), kernel="bilinear", periodic=True),
                       scatter_to_field((20, 20), pos, vals), atol=1e-12)
    fld = rng.standard_normal((20, 20))
    assert np.allclose(gather(fld, pos[:, ::-1], kernel="bilinear", periodic=True),
                       sample_field(fld, pos), atol=1e-12)


def test_reproduces_mpm_p2g():
    from holographic_mpm import MPMSnow
    m = MPMSnow(grid=32, seed=0).seed_block(cx=16, cy=16, w=8, h=8, n=200)
    assert np.allclose(scatter(m.x * m.inv_dx, m.m, (32, 32), kernel="bspline"),
                       m.p2g_mass_grid(), atol=1e-10)


def test_vector_values_and_constant_readout():
    rng = np.random.default_rng(3)
    pts = rng.uniform(3, 13, (60, 2)); mom = rng.standard_normal((60, 2))
    assert scatter(pts, mom, (16, 16), kernel="bspline").shape == (16, 16, 2)
    const = np.ones((16, 16, 2)) * np.array([2.0, -1.0])
    assert np.allclose(gather(const, pts, kernel="bspline"), [2.0, -1.0], atol=1e-9)


def test_deterministic():
    rng = np.random.default_rng(4); pts = rng.uniform(3, 13, (20, 2)); v = rng.standard_normal(20)
    assert np.array_equal(scatter(pts, v, (16, 16)), scatter(pts, v, (16, 16)))
