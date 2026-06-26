"""Tests for B6: the Tero flow-conductance maze solver -- optimal, deterministic, and matching the
true shortest path on braided mazes; the principled-physics counterpart to the stochastic ant."""

from holographic_creature import GridWorld
from holographic_flow import tero_solve, solve_maze_flow


def test_tero_finds_optimal_on_braided_mazes():
    for fs in (3, 7, 11):
        w = GridWorld(16, 16, maze=True, fixed_seed=fs, braid=1.0)
        path, info = solve_maze_flow(w)
        assert info["reached"]
        assert info["extracted_len"] == info["optimal"]    # flow collapses onto the shortest tube


def test_tero_is_deterministic():
    w = GridWorld(16, 16, maze=True, fixed_seed=7, braid=1.0)
    p1, _ = solve_maze_flow(w)
    p2, _ = solve_maze_flow(w)
    assert p1 == p2                                          # no randomness: identical every run


def test_tero_picks_the_short_route_through_a_loop():
    # a braided diamond: 0-1-2 (short, 2 edges) vs the detour 0-3-4-2 (3 edges)
    nbr = {0: [1, 3], 1: [0, 2], 2: [1, 4], 3: [0, 4], 4: [3, 2]}
    path = tero_solve(nbr, 0, 2, steps=120)
    assert path == [0, 1, 2]


def test_disconnected_graph_returns_none():
    nbr = {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}
    assert tero_solve(nbr, "A", "D", steps=30) is None


def test_missing_endpoint_returns_none():
    nbr = {0: [1], 1: [0]}
    assert tero_solve(nbr, 0, 9, steps=10) is None          # goal not in the graph


def test_solve_maze_flow_reports_optimum_and_determinism_flag():
    w = GridWorld(16, 16, maze=True, fixed_seed=15, braid=1.0)
    path, info = solve_maze_flow(w)
    assert info["deterministic"] is True
    assert info["extracted_len"] == info["optimal"] and info["cells"] > 0


# --- the converged flux as a Hodge-decomposable flow (above/below sweep) ----
import numpy as np
from holographic_flow import tero_flux
from holographic_spectral import boundary_matrices, hodge_decomposition, betti_numbers


def _grid_nbr(R, C):
    nbr = {}
    for r in range(R):
        for c in range(C):
            nbr[(r, c)] = [(r + dr, c + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                           if 0 <= r + dr < R and 0 <= c + dc < C]
    return nbr


def test_tero_flux_gradient_divergence_is_the_injected_current():
    # the gradient part of the flux is the NET transport: its divergence is +/-I0 at source/goal, 0 elsewhere.
    n, edges, flux = tero_flux(_grid_nbr(4, 4), (0, 0), (3, 3))
    d1, _ = boundary_matrices(n, edges, None)
    grad, curl, harm = hodge_decomposition(n, edges, flux, None)
    div = d1 @ grad
    assert abs(abs(div).max() - 1.0) < 1e-6                  # I0 = 1.0 injected
    assert np.sum(np.abs(div) > 1e-6) == 2                   # exactly the source and the goal


def test_tero_flux_harmonic_dimension_equals_b1():
    # circulation lives in the harmonic subspace, whose dimension is the graph's loop count B1.
    n, edges, flux = tero_flux(_grid_nbr(4, 4), (0, 0), (3, 3))
    _, b1 = betti_numbers(n, edges, None)
    assert b1 == len(edges) - n + 1                          # E - V + 1 for a connected graph
    d1, _ = boundary_matrices(n, edges, None)
    L1 = d1.T @ d1                                           # no triangles -> L1 = d1^T d1
    assert int(np.sum(np.linalg.eigvalsh(L1) < 1e-9)) == b1  # harmonic subspace dim == B1


def test_tero_flux_no_curl_on_a_graph():
    # a maze graph has no filled triangles, so the curl part is identically zero.
    n, edges, flux = tero_flux(_grid_nbr(3, 4), (0, 0), (2, 3))
    _, curl, _ = hodge_decomposition(n, edges, flux, None)
    assert np.linalg.norm(curl) < 1e-12


def test_tree_flow_has_zero_circulation():
    # kept-negative-shaped fact: on a tree (no loops, B1=0) the route is forced, so harmonic flux is exactly 0.
    tree = {0: [1], 1: [0, 2, 3], 2: [1], 3: [1, 4], 4: [3]}
    n, edges, flux = tero_flux(tree, 0, 4, steps=120)
    _, b1 = betti_numbers(n, edges, None)
    assert b1 == 0
    _, _, harm = hodge_decomposition(n, edges, flux, None)
    assert np.linalg.norm(harm) < 1e-9


def test_tero_flux_disconnected_returns_none():
    nbr = {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}
    assert tero_flux(nbr, "A", "D", steps=30) is None
