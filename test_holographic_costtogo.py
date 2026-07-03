"""Cost-to-go value field: one goal-rooted solve, optimal descent routing from anywhere, on any cost manifold."""
import numpy as np
from holographic_ndfield import field_weighted_graph, least_cost_path, cost_to_go, route_from, value_grid, path_cost


def _field(seed=0, shape=(16, 16)):
    rng = np.random.default_rng(seed)
    cost = rng.random(shape); cost[shape[0] // 2, :] += 4.0        # an expensive band
    nbr, ec = field_weighted_graph(shape, cost)
    return nbr, ec, cost, shape


def test_descent_routes_are_optimal():
    nbr, ec, cost, shape = _field()
    goal = (shape[0] - 1, shape[1] - 1)
    V, nxt = cost_to_go(nbr, ec, goal)
    for s in [(0, 0), (0, shape[1] - 1), (shape[0] - 1, 0), (3, 7)]:
        r = route_from(nxt, s, goal)
        d = least_cost_path(nbr, ec, s, goal)
        assert r is not None and abs(path_cost(r, cost, shape) - path_cost(d, cost, shape)) < 1e-9  # same optimum


def test_one_solve_serves_every_start():
    nbr, ec, cost, shape = _field(seed=1)
    goal = (0, 0)
    V, nxt = cost_to_go(nbr, ec, goal)
    # every reachable cell has a value and a route to the goal
    reached = [route_from(nxt, c, goal) for c in list(nbr.keys())[:40]]
    assert all(r is not None for r in reached)
    assert V[goal] == 0.0 and all(V[c] >= 0 for c in V)


def test_value_grid_and_potential():
    nbr, ec, cost, shape = _field(seed=2)
    goal = (shape[0] - 1, shape[1] - 1)
    V, nxt = cost_to_go(nbr, ec, goal)
    g = value_grid(V, shape)
    assert g[goal] == 0.0 and g.shape == shape                    # goal is the potential minimum
    assert np.isfinite(g).all()                                   # a connected grid: every cell reached


def test_same_faculty_on_non_3d_manifold():
    # the identical solver routes over an abstract 2D cost manifold (not a 3D scene)
    rng = np.random.default_rng(3); man = rng.random((20, 20)); man[:, :3] += 3.0
    nbr, ec = field_weighted_graph((20, 20), man)
    goal = (19, 19); V, nxt = cost_to_go(nbr, ec, goal)
    assert route_from(nxt, (0, 0), goal) is not None              # one manifold solve routes any query state


def test_disconnected_returns_none():
    # a fully walled-off start returns None cleanly
    shape = (8, 8)
    blocked = {(r, 3) for r in range(8)} | {(3, c) for c in range(8)}   # seal off a corner
    nbr, ec = field_weighted_graph(shape, np.zeros(shape), blocked=blocked)
    V, nxt = cost_to_go(nbr, ec, (7, 7))
    assert route_from(nxt, (0, 0), (7, 7)) is None
