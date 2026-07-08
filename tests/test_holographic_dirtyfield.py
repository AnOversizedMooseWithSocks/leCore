"""Dirty-flag deltas for a nav/physics cost field: move one collider, re-evaluate only its footprint."""
import numpy as np
from holographic.misc.holographic_dirtyfield import DirtyField


def _blob(P, c, R=4.0, s=8.0):
    d = np.linalg.norm(P - c, axis=1)
    return s * np.exp(-(d ** 2) / (2 * (R / 2) ** 2))


def test_delta_matches_full_rebuild_bit_for_bit():
    shape = (40, 40); lo = np.zeros(2); hi = np.full(2, 40.0)
    df = DirtyField(shape, lo, hi)
    df.place("a", lambda P, c: _blob(P, c), (10.0, 20.0), 8.0)
    df.place("b", lambda P, c: _blob(P, c), (30.0, 10.0), 8.0)
    df.move("a", (25.0, 28.0))
    ref = DirtyField(shape, lo, hi)
    ref.place("a", lambda P, c: _blob(P, c), (25.0, 28.0), 8.0)
    ref.place("b", lambda P, c: _blob(P, c), (30.0, 10.0), 8.0)
    assert np.allclose(df.cost_grid(), ref.cost_grid(), atol=1e-9)   # bit-identical to a full rebuild


def test_delta_is_cheaper_than_full_and_grid_independent():
    small = DirtyField((30, 30), np.zeros(2), np.full(2, 30.0))
    big = DirtyField((90, 90), np.zeros(2), np.full(2, 90.0))
    for df, G in [(small, 30), (big, 90)]:
        df.place("a", lambda P, c: _blob(P, c), (G * 0.3, G * 0.5), 8.0)
        df.place("b", lambda P, c: _blob(P, c), (G * 0.7, G * 0.3), 8.0)
        df.evals = 0; df.move("a", (G * 0.6, G * 0.6)); df._de = df.evals
        df.evals = 0; df.full_rebuild(); df._fe = df.evals
    assert small._de < small._fe and big._de < big._fe             # delta beats full at both sizes
    assert abs(big._de - small._de) < 0.3 * small._de              # delta ~constant as the grid grows
    assert big._fe > small._fe * 5                                 # full rebuild scales with area


def test_updated_field_reroutes():
    from holographic.misc.holographic_ndfield import navigate_field, path_cost, straight_line_cells
    shape = (30, 30); lo = np.zeros(2); hi = np.full(2, 30.0)
    df = DirtyField(shape, lo, hi)
    df.place("obs", lambda P, c: _blob(P, c, R=6.0, s=12.0), (15.0, 15.0), 12.0)
    at_old_before = float(df.cost_grid()[15, 15])
    df.move("obs", (5.0, 25.0))                                    # move the obstacle
    at_old_after = float(df.cost_grid()[15, 15]); at_new_after = float(df.cost_grid()[5, 25])
    assert at_old_after < at_old_before - 1.0                      # cost fell where the obstacle LEFT
    assert at_new_after > 1.0                                      # cost rose where it ARRIVED (delta moved it)
    route = navigate_field(df.cost_grid(), shape, (0, 0), (29, 29), lo=lo, hi=hi)
    c_line = path_cost(straight_line_cells((0, 0), (29, 29)), df.cost_grid(), shape)
    assert path_cost(route, df.cost_grid(), shape) <= c_line       # the route on the updated field is (still) good
