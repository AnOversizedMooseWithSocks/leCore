"""Physics backlog #7: diffusion-limited branching -- ice dendrites AND lightning from one dielectric-breakdown engine."""
import numpy as np
from holographic.misc.holographic_dendrite import DielectricBreakdown, ice_dendrite, lightning, _relax_potential


def test_laplace_potential():
    cluster = np.zeros((41, 41), bool); cluster[20, 20] = True
    source = np.zeros((41, 41), bool)
    source[0, :] = source[-1, :] = source[:, 0] = source[:, -1] = True
    phi = _relax_potential(cluster, source, iters=200)
    assert phi[20, 20] == 0.0 and phi[0, 0] == 1.0
    assert 0.0 < phi[10, 20] < 1.0
    assert phi[5, 20] > phi[15, 20]                           # closer to the border -> higher potential


def test_ice_dendrite_is_sparse_fractal():
    ice = ice_dendrite(shape=(81, 81), eta=1.0, steps=250, seed=0)
    n = int(ice.cluster.sum())
    assert n >= 200
    fd = ice.fractal_dimension()
    assert 1.0 < fd < 1.9                                     # branching, not a line and not a filled disk
    ys, xs = np.where(ice.cluster)
    bbox = max(ys.max() - ys.min(), xs.max() - xs.min())
    assert n > 2 * bbox                                       # more than a single line -> it branches


def test_lightning_grows_downward():
    bolt = lightning(shape=(81, 81), eta=3.0, steps=100, seed=1)
    ys, _ = np.where(bolt.cluster)
    assert ys.max() > 40                                      # reached well below the top cloud


def test_same_engine_different_seed_and_source():
    # ice: point seed + full border source (radial); lightning: line seed + bottom source (downward)
    ice = ice_dendrite(shape=(61, 61), steps=100, seed=0)
    bolt = lightning(shape=(61, 61), steps=100, seed=0)
    assert ice.cluster[30, 30]                                # ice seeded at the centre
    assert bolt.cluster[0, :].any()                          # lightning seeded along the top


def test_eta_tunes_shape():
    # higher eta reaches deeper per cell (stringier) than low eta (bushier)
    thin = lightning(shape=(81, 81), eta=4.0, steps=100, seed=2)
    bushy = lightning(shape=(81, 81), eta=0.4, steps=100, seed=2)
    assert np.where(thin.cluster)[0].max() >= np.where(bushy.cluster)[0].max() - 2


def test_deterministic():
    a = ice_dendrite(shape=(61, 61), steps=80, seed=7).cluster
    b = ice_dendrite(shape=(61, 61), steps=80, seed=7).cluster
    assert np.array_equal(a, b)
