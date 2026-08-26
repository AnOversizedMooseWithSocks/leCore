"""Cellular/crystalline (M2): correct Voronoi, crack metric, facet albedo, bit-exact lattice."""
import numpy as np
from holographic.simulation_and_physics.holographic_cellular import VoronoiCells, cell_albedo, crack_mask, lattice


def test_voronoi_assignment_matches_brute_force():
    cells = VoronoiCells(n_seeds=24, seed=0)
    P = np.random.default_rng(1).uniform(-1.5, 1.5, (600, 3))
    got = cells.ids(P)
    true = np.argmin(np.linalg.norm(P[:, None, :] - cells.seeds[None, :, :], axis=2), axis=1)
    assert np.array_equal(got, true)
    assert len(np.unique(got)) >= 2


def test_crack_metric_peaks_on_boundaries():
    cells = VoronoiCells(n_seeds=24, seed=0)
    mid = (cells.seeds[0] + cells.seeds[1]) * 0.5                    # on the bisector -> an edge
    assert cells.edge_distance([mid])[0] < 0.05
    assert cells.edge_distance([cells.seeds[0]])[0] > cells.edge_distance([mid])[0]
    m = crack_mask(cells, crack_width=0.05)
    assert 0.0 <= float(m([cells.seeds[0]])[0]) <= 1.0


def test_facet_albedo_valid_and_deterministic():
    cells = VoronoiCells(n_seeds=30, seed=1)
    alb = cell_albedo(cells)
    P = np.random.default_rng(2).uniform(-1.5, 1.5, (500, 3))
    a = alb(P)
    assert a.shape == (500, 3) and a.min() >= 0 and a.max() <= 1 and np.array_equal(a, alb(P))


def test_lattice_tiles_bit_exactly():
    motif = lambda L: np.linalg.norm(L, axis=1)
    lat = lattice(motif, period=0.5)
    Q = np.random.default_rng(3).uniform(-1, 1, (400, 3))
    assert np.allclose(lat(Q), lat(Q + np.array([0.5, 0.0, 0.0])))
    assert np.allclose(lat(Q), lat(Q + np.array([0.0, 0.5, 0.5])))
