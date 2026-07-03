"""Physics backlog #8B (rung 4, the last item): snow via MLS-MPM, with P2G/G2P verified as bundle/readout."""
import numpy as np
from holographic_mpm import MPMSnow, _bundle_mass_grid, _bspline


def test_p2g_is_a_bundle():
    """The holographic identity: P2G scatter == an independent bundle (superposition) of kernel splats."""
    m = MPMSnow(grid=48, seed=0).seed_block(cx=24, cy=30, w=10, h=10, n=300)
    assert np.allclose(m.p2g_mass_grid(), _bundle_mass_grid(m), atol=1e-9)


def test_bspline_partition_of_unity():
    """The B-spline weights sum to 1 -- a normalized bundle -- so P2G preserves total mass."""
    for fx in [0.5, 0.7, 1.0, 1.3, 1.49]:
        assert abs(sum(_bspline(fx)) - 1.0) < 1e-12


def test_mass_conserved():
    m = MPMSnow(grid=48, seed=0).seed_block(cx=24, cy=24, w=8, h=8, n=200)
    assert abs(m.p2g_mass_grid().sum() - m.total_mass()) < 1e-9


def test_transfer_conserves_momentum():
    """P2G -> G2P round-trip (gravity off) conserves total momentum -- the bundle->readout fidelity property."""
    m = MPMSnow(grid=48, gravity=0.0, seed=1).seed_block(cx=24, cy=24, w=8, h=8, n=200)
    m.v[:] = np.array([0.4, -0.2])
    p0 = m.total_momentum().sum(0)
    m.step(dt=1e-3)
    assert np.allclose(p0, m.total_momentum().sum(0), atol=2e-2)


def test_snow_falls_and_compresses():
    snow = MPMSnow(grid=48, gravity=9.81, seed=2).seed_block(cx=24, cy=12, w=10, h=8, n=400)
    y0 = snow.center_of_mass()[1]; top0 = snow.x[:, 1].max()
    extent0 = snow.x[:, 1].max() - snow.x[:, 1].min()
    snow.run(dt=2e-3, steps=800)
    assert snow.center_of_mass()[1] < y0 - 1.5              # fell
    assert snow.x[:, 1].max() < top0 - 3.0                 # settled down
    assert (snow.x[:, 1].max() - snow.x[:, 1].min()) < 0.7 * extent0   # compressed plastically
    assert abs(snow.total_mass() - 400.0) < 1e-9           # mass conserved
    assert np.isfinite(snow.x).all()


def test_deterministic():
    a = MPMSnow(grid=48, seed=5).seed_block(24, 30, 8, 8, 100).run(2e-3, 50).x
    b = MPMSnow(grid=48, seed=5).seed_block(24, 30, 8, 8, 100).run(2e-3, 50).x
    assert np.array_equal(a, b)
