"""Regression traps for holographic_wods -- Walk on Decomposed Subdomains.

Two claims written for this module BEFORE it was measured were REFUTED by the measurement: the paper's
low-variance headline does not reproduce on this simplified operator, and the accuracy advantage is not
monotone. Both refutations are pinned here, because a negative that is not tested quietly comes back as
folklore -- and the honest scope of this module is exactly the difference between what the paper claims
and what this code earns.
"""
import numpy as np

import lecore
from holographic.misc.holographic_wos import walk_on_spheres
from holographic.simulation_and_physics.holographic_wods import (estimate_local_operators,
                                                                 interface_grid,
                                                                 measure_vs_pure_wos, solve_decomposed)


def _exact(p):
    """u(x,y) = x^2 - y^2 -- harmonic, so it is the exact solution of Laplace with its own boundary data."""
    p = np.atleast_2d(np.asarray(p, float))
    return p[:, 0] ** 2 - p[:, 1] ** 2


def _dist(p):
    p = np.atleast_2d(np.asarray(p, float))
    return np.minimum(np.minimum(p[:, 0], 1 - p[:, 0]), np.minimum(p[:, 1], 1 - p[:, 1]))


def test_it_actually_solves_the_pde():
    pts = interface_grid(6, 6)
    u = solve_decomposed(pts, _exact, walks=400, seed=0)
    assert float(np.mean(np.abs(u - _exact(pts)))) < 0.05


def test_the_coupling_operator_is_sub_stochastic():
    # Rows must be non-negative and sum to at most 1; the deficit is escape-to-boundary mass. A row above 1
    # means walks were double counted, which would silently inflate the whole solution.
    stats = {}
    pts = interface_grid(5, 5)
    solve_decomposed(pts, _exact, walks=200, seed=1, stats=stats)
    P = stats["P"]
    assert np.all(P >= 0.0)
    assert np.all(P.sum(axis=1) <= 1.0 + 1e-12)


def test_escape_fraction_is_reported_and_small():
    # The honesty channel: a high escape fraction means the operator rows are unreliable and the solve is
    # extrapolating. It is returned rather than swallowed, so it gets asserted rather than ignored.
    _, _, escaped = estimate_local_operators(interface_grid(5, 5), _exact, walks=200, seed=2)
    assert 0.0 <= escaped < 0.05


def test_solve_is_deterministic_at_a_fixed_seed():
    pts = interface_grid(5, 5)
    assert np.array_equal(solve_decomposed(pts, _exact, walks=120, seed=3),
                          solve_decomposed(pts, _exact, walks=120, seed=3))


def test_interface_grid_excludes_boundary_and_is_ordered():
    pts = interface_grid(4, 4)
    assert pts.shape == (9, 2)                     # (4-1)^2 interior nodes
    assert np.all((pts > 0.0) & (pts < 1.0))       # boundary nodes are data, not unknowns
    assert np.array_equal(pts, interface_grid(4, 4))


def test_wods_is_more_accurate_than_pure_wos_at_a_tight_budget():
    # THE CLAIM THIS CODE EARNS: roughly half the error at 32 walks. If it stops holding, re-measure --
    # this is the only reason to prefer WoDS over the shipped pointwise solver.
    m = measure_vs_pure_wos(nx=5, ny=5, walks=32, seeds=6)
    assert m["wods_err"] < m["wos_err"]


def test_the_advantage_is_not_monotone_kept_negative():
    # REFUTATION 1, PINNED. WoDS is biased by the interface resolution; unbiased WoS keeps converging and
    # OVERTAKES at a generous budget. The advantage is a low-budget effect, not a free win.
    tight = measure_vs_pure_wos(nx=5, ny=5, walks=32, seeds=6)
    rich = measure_vs_pure_wos(nx=5, ny=5, walks=256, seeds=6)
    tight_edge = tight["wos_err"] - tight["wods_err"]
    rich_edge = rich["wos_err"] - rich["wods_err"]
    assert rich_edge < tight_edge, "the WoDS edge no longer shrinks with budget -- the bias story changed"


def test_the_papers_low_variance_headline_does_not_reproduce_here():
    # REFUTATION 2, PINNED. The paper reports low variance for a fuller method that estimates proper
    # subdomain solution operators; this estimates the discrete harmonic measure, which is the same IDEA
    # and a weaker INSTRUMENT. Measured, pure WoS often has the SMALLER across-seed spread. If this ever
    # flips, the module docstring's scope paragraph is wrong and must be rewritten, not the test.
    m = measure_vs_pure_wos(nx=8, ny=8, walks=64, seeds=6)
    assert m["wos_sd"] <= m["wods_sd"] * 1.5, \
        "WoDS became the clearly lower-variance method (%.5f vs %.5f) -- rewrite the scope claim" % (
            m["wods_sd"], m["wos_sd"])


def test_bias_does_not_vanish_with_more_walks_at_a_fixed_interface():
    # The defining property of a discretisation bias: samples do not remove it, resolution does.
    coarse = interface_grid(3, 3)
    few = float(np.mean(np.abs(solve_decomposed(coarse, _exact, walks=200, seed=5) - _exact(coarse))))
    many = float(np.mean(np.abs(solve_decomposed(coarse, _exact, walks=1600, seed=5) - _exact(coarse))))
    assert abs(many - few) < 0.05


# --------------------------------------------------------------------------------------
# CROSS-FACULTY
# --------------------------------------------------------------------------------------

def test_wods_agrees_with_the_shipped_pointwise_solver():
    # Both must be solving the SAME problem. If the two disagree beyond their error bars, one of them has
    # the wrong domain or the wrong boundary data -- something neither module's selftest can see alone.
    pts = interface_grid(5, 5)
    u = solve_decomposed(pts, _exact, walks=400, seed=0)
    v, _ = walk_on_spheres(pts, _dist, _exact, n_walks=400, seed=0)
    assert float(np.mean(np.abs(u - np.asarray(v).ravel()))) < 0.08


def test_wods_is_wired_and_discoverable():
    mind = lecore.UnifiedMind(dim=128, seed=0)
    pts = mind.wods_interface_grid(5, 5)
    u = mind.wods_solve(pts, _exact, walks=300, seed=0)
    assert float(np.mean(np.abs(u - _exact(pts)))) < 0.06
    for query in ("split a domain into pieces and solve each one",
                  "combine local solvers into one global sparse system",
                  "estimate a local solution operator by random walks"):
        assert "Decomposed Subdomains" in str(mind.find_capability(query)[:3]), \
            "%r no longer surfaces WoDS" % query
