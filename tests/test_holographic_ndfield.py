"""Tests for holographic_ndfield: dimension-agnostic grid solve + sparse-probe-interpolate reconstruction."""
import numpy as np
from holographic.misc.holographic_ndfield import grid_graph, solve_grid_maze, sparse_reconstruct, _nadaraya_watson


def test_grid_graph_neighbours_scale_with_dimension():
    g2 = grid_graph((5, 5)); g3 = grid_graph((5, 5, 5))
    assert len(g2[(2, 2)]) == 4                                 # interior cell: 2*D neighbours
    assert len(g3[(2, 2, 2)]) == 6
    assert len(g2[(0, 0)]) == 2                                 # corner: D neighbours


def test_same_solver_solves_2d_3d_4d():
    assert solve_grid_maze((8, 8), set(), (0, 0), (7, 7)) is not None
    assert solve_grid_maze((5, 5, 5), set(), (0, 0, 0), (4, 4, 4)) is not None
    assert solve_grid_maze((3, 3, 3, 3), set(), (0, 0, 0, 0), (2, 2, 2, 2)) is not None


def test_3d_maze_respects_walls():
    shape = (5, 5, 5)
    blocked = {(2, y, z) for y in range(4) for z in range(5)}   # wall with a gap at y=4
    path = solve_grid_maze(shape, blocked, (0, 0, 0), (4, 4, 4))
    assert path[0] == (0, 0, 0) and path[-1] == (4, 4, 4)
    assert not any(c in blocked for c in path)                  # never through a wall
    assert all(sum(abs(a - b) for a, b in zip(p, q)) == 1 for p, q in zip(path, path[1:]))


def test_adaptive_reconstruction_beats_uniform():
    def oracle(P):
        return np.sin(2.1 * P[:, 0]) * np.cos(1.6 * P[:, 1]) + 0.4 * np.sin(2.4 * P[:, 2])
    lo = np.zeros(3); hi = np.full(3, 3.0); bw = 0.12 * 3.0
    test = lo + (hi - lo) * np.random.default_rng(7).random((400, 3)); truth = oracle(test)
    u = lo + (hi - lo) * np.random.default_rng(2).random((160, 3))
    err_uniform = np.abs(_nadaraya_watson(test, u, oracle(u), bw) - truth).mean()
    pts, vals, recon = sparse_reconstruct(oracle, lo, hi, n_seed=80, n_refine=80, bandwidth=bw, seed=0)
    err_adaptive = np.abs(recon(test) - truth).mean()
    assert err_adaptive < err_uniform                          # refine-where-uncertain wins at equal budget


def test_navigate_field_routes_around_cost():
    import numpy as np
    from holographic.misc.holographic_ndfield import navigate_field, path_cost, straight_line_cells
    shape = (12, 12, 12); lo = np.zeros(3); hi = np.full(3, 3.0)
    def blob(P):                                               # a dense obstacle in the middle
        return 6.0 * np.exp(-(((P[:, 0] - 1.5) ** 2 + (P[:, 1] - 1.5) ** 2 + (P[:, 2] - 1.5) ** 2)) / 0.4)
    straight = straight_line_cells((0, 0, 0), (11, 11, 11))
    routed = navigate_field(blob, shape, (0, 0, 0), (11, 11, 11), lo=lo, hi=hi)
    cs = path_cost(straight, blob, shape, lo=lo, hi=hi); cr = path_cost(routed, blob, shape, lo=lo, hi=hi)
    assert cr < cs * 0.5                                       # the navigated route crosses far less of the field
    assert routed[0] == (0, 0, 0) and routed[-1] == (11, 11, 11)


def test_navigate_field_array_cost_and_blocked():
    import numpy as np
    from holographic.misc.holographic_ndfield import navigate_field
    shape = (6, 6); cost = np.zeros(shape)
    blocked = {(3, y) for y in range(5)}                       # a wall with a gap at y=5
    path = navigate_field(cost, shape, (0, 0), (5, 5), blocked=blocked)
    assert path is not None and not any(c in blocked for c in path)


def test_navigate_scene_stays_out_of_geometry():
    """navigate_scene routes through free space -- never inside an object (the scene's SDF is the cost field)."""
    import numpy as np
    from holographic.simulation_and_physics.holographic_semantic import parse_description, realize_scene, _scene_setup
    from holographic.misc.holographic_ndfield import navigate_scene
    objs = parse_description("a red box beside a blue box")["objects"]; rs = realize_scene(objs)
    un = _scene_setup(None, False, "clear", "bright", (0.75, 0.9, 0.85), rs=rs)["union"]
    lo = np.array([-3, -1.5, -3.0]); hi = np.array([3, 1.5, 3.0])
    path = navigate_scene(lambda P: un.eval(P), lo, hi, (20, 10, 20), (-2.5, 0, -2.5), (2.5, 0, 2.5), clearance=0.3)
    assert path is not None and len(path) > 2
    assert float(un.eval(np.array(path)).min()) >= 0.0            # never inside geometry


def test_path_is_composable_hypervector():
    """A navigated path round-trips through ONE hypervector -- composable VSA data, not just a Python list."""
    from holographic.misc.holographic_ndfield import encode_path, decode_path_step
    path = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1)]
    vec, sm, keys = encode_path(path, dim=2048, seed=0)
    ok = sum(decode_path_step(vec, sm, keys, i) == keys[i] for i in range(len(keys)))
    assert ok == len(keys)                                        # every waypoint decodes back exactly


def test_navigate_raw_data_field():
    """The navigator runs on a raw-data cost field (a 2D occupancy surface), not just spatial scenes."""
    import numpy as np
    from holographic.misc.holographic_ndfield import navigate_field, encode_path, decode_path_step
    rng = np.random.default_rng(0)                                # a stand-in occupancy surface with a cheap corridor
    occ = rng.random((16, 16)) * 5.0; occ[:, 7:9] = 0.0           # a low-cost channel down the middle
    route = navigate_field(occ, (16, 16), (0, 0), (15, 15))
    assert route[0] == (0, 0) and route[-1] == (15, 15)
    vec, sm, keys = encode_path(route)                            # and the raw-data route is composable too
    assert decode_path_step(vec, sm, keys, 0) == keys[0]
