"""Regression traps for F4 (stable neo-Hookean tets + muscle fibers, hand-derived gradients).

Two pins carry the module: the analytic stress against fd_gradient, and the INVERSION
behaviour that motivated choosing Smith/De Goes/Kim 2018 over the source document's
classical log-J neo-Hookean.
"""
import numpy as np
import pytest

from holographic.simulation_and_physics import holographic_fem as F

REF = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
TET = np.array([[0, 1, 2, 3]])


def test_rest_state_is_stress_free():
    """A planted truth with a KNOWN answer: an undeformed body exerts no force. This caught a
    real derivation error -- using alpha = 1 + mu/lam (the form quoted in many summaries)
    instead of the rest-stable alpha = 1 + 3mu/(4lam) left a residual stress measured at
    4.17e-2, i.e. a body that shrinks the moment you press play."""
    _, g = F.neohookean_energy_and_grad(REF, TET, mu=1.0, lam=10.0)
    assert np.abs(g).max() < 1e-9
    for mu, lam in ((2.0, 5.0), (0.5, 50.0)):
        _, g2 = F.neohookean_energy_and_grad(REF, TET, mu=mu, lam=lam)
        assert np.abs(g2).max() < 1e-9, "rest stress at mu=%s lam=%s" % (mu, lam)


def test_analytic_stress_matches_finite_difference():
    from holographic.misc.holographic_optimize import fd_gradient
    pts = REF + np.random.default_rng(20260816).normal(scale=0.25, size=(4, 3))
    f = lambda flat: F.neohookean_energy_and_grad(flat.reshape(-1, 3), TET, 1.0, 10.0,
                                                  rest=REF)[0]
    num = fd_gradient(f, pts.ravel().copy(), eps=1e-6).reshape(-1, 3)
    _, ana = F.neohookean_energy_and_grad(pts, TET, 1.0, 10.0, rest=REF)
    assert np.abs(num - ana).max() < 1e-4


def test_inverted_element_stays_finite():
    """THE REASON FOR THE STABLE MODEL. The classical neo-Hookean's log(J) is undefined for
    J <= 0, so one inverted tet returns NaN and kills the whole solve. Morphogenesis meshes
    are generated, not authored, and they do invert."""
    from holographic.misc.holographic_optimize import fd_gradient
    inv = REF.copy()
    inv[3, 2] = -1.0
    e, g = F.neohookean_energy_and_grad(inv, TET, 1.0, 10.0, rest=REF)
    assert np.isfinite(e) and np.all(np.isfinite(g)) and e > 0
    f = lambda flat: F.neohookean_energy_and_grad(flat.reshape(-1, 3), TET, 1.0, 10.0,
                                                  rest=REF)[0]
    num = fd_gradient(f, inv.ravel().copy(), eps=1e-6).reshape(-1, 3)
    assert np.abs(num - g).max() < 1e-4     # correct where it matters most


def test_muscle_contracts_and_relaxes():
    from holographic.misc.holographic_optimize import fd_gradient
    pts = REF + np.random.default_rng(2).normal(scale=0.2, size=(4, 3))
    fib, l0 = np.array([[0, 1]]), np.array([1.0])
    f = lambda flat: F.muscle_energy_and_grad(flat.reshape(-1, 3), fib, l0, np.array([0.5]))[0]
    num = fd_gradient(f, pts.ravel().copy(), eps=1e-6).reshape(-1, 3)
    _, ana = F.muscle_energy_and_grad(pts, fib, l0, np.array([0.5]))
    assert np.abs(num - ana).max() < 1e-5
    # BEHAVIOUR not sign convention: a descent step must SHORTEN an activated fiber
    _, g = F.muscle_energy_and_grad(REF, fib, l0, np.array([0.5]))
    moved = REF - 0.01 * g
    assert np.linalg.norm(moved[0] - moved[1]) < np.linalg.norm(REF[0] - REF[1])
    _, gr = F.muscle_energy_and_grad(REF, fib, l0, np.array([1.0]))
    assert np.abs(gr).max() < 1e-12       # relaxed fiber at rest length does nothing


def test_tet_orientation_is_consistent():
    """F4's rest_quality report is what caught the tetrahedraliser emitting MIXED WINDING
    (33 of 70 tets with negative rest volume). Harmless for a symmetric energy, wrong for
    every consumer that reads a signed volume. Pinned here so it cannot regress."""
    from holographic.simulation_and_physics.holographic_morphogen import grow_aggregate
    from holographic.mesh_and_geometry.holographic_tetmesh import tetrahedralize
    agg = grow_aggregate(n_cells=30, seed=0, steps=60)
    mesh = tetrahedralize(agg["positions"], agg["radii"])
    q = F.rest_quality(agg["positions"], mesh["tets"])
    assert q["inverted"] == 0 and q["degenerate"] == 0 and q["min_vol"] > 0


def test_end_to_end_solve_descends():
    from lecore import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    a = m.morphogenesis_grow(n_cells=30, seed=0, steps=60)
    mesh = m.tetrahedralize(a["positions"], a["radii"])
    fib, rl = m.fem_select_fibers(a["positions"], mesh["tets"], axis=0, fraction=0.2)
    assert len(fib) > 0
    out = m.fem_simulate(a["positions"], mesh["tets"], steps=60, fibers=fib,
                         rest_lengths=rl, activation=0.7, pinned=[0])
    assert np.all(np.isfinite(out["positions"]))
    assert out["history"][-1] <= out["history"][0] + 1e-9
    assert np.allclose(out["positions"][0], a["positions"][0])   # pinned vertex stayed put
