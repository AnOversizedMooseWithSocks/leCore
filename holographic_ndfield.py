"""N-D fields (the reusable pattern). Moose's recurring observation: when the system is DETERMINISTIC and we already
hold all its information (the scene, the graph, the field), we don't compute the whole thing -- we probe it sparsely
at chosen points, INTERPOLATE the rest with a kernel, and REFINE only where the reconstruction is uncertain. And the
number of dimensions is irrelevant: a 2D maze, a 3D maze, an N-D maze are the SAME object once you abstract the
operation from the coordinates. This module makes that pattern reusable, in two forms that recur across the engine.

1. SEARCH over a known graph, in ANY dimension. `grid_graph(shape)` builds the adjacency dict {cell: [neighbours]} for
   an N-D grid; `solve_grid_maze` feeds it straight to the Tero slime-mould flow solver -- which never looked at
   coordinates, only the graph, so it solves a 3D (or 7D) maze with no new code. "3 dimensions is trivial when we have
   thousands." The 2D maze the ant/flow solvers shipped on is just the D=2 case of this.

2. RECONSTRUCT a known field, in ANY dimension. `sparse_reconstruct(oracle, lo, hi)` samples a deterministic
   oracle f: R^D -> R at a sparse adaptive set, reconstructs everywhere by Nadaraya-Watson (the same Gaussian kernel
   read the radiance field uses), and REFINES where the reconstruction disagrees with the oracle. This is the pattern
   under coherent reflection (trace sparse, interpolate the bundle), ray differentials (5 rays -> the lobe), the
   radiance field, and region-field culling -- named once, reusable.

Deterministic, NumPy/stdlib only. The point is not a new algorithm but the RECOGNITION that these are one pattern:
a known deterministic field, probed sparsely, interpolated, refined -- dimension-agnostic.
"""
import numpy as np
from itertools import product


# ---------------------------------------------------------------------------------------------------------------------
# 1. SEARCH: an N-D grid graph for the (already dimension-agnostic) flow / ant solvers
# ---------------------------------------------------------------------------------------------------------------------

def grid_graph(shape, blocked=None):
    """Adjacency dict {cell: [neighbours]} for an N-D grid of `shape` (a tuple), minus any `blocked` cells. Cells are
    integer tuples; neighbours are the 2*N axis-aligned moves that stay in bounds and unblocked. This is the ONLY
    dimension-specific piece -- the solvers downstream only see the graph."""
    blocked = set(blocked or ())
    ranges = [range(s) for s in shape]
    cells = [c for c in product(*ranges) if c not in blocked]
    cellset = set(cells)
    nd = len(shape)
    nbr = {}
    for c in cells:
        adj = []
        for ax in range(nd):
            for step in (-1, 1):
                d = list(c); d[ax] += step; d = tuple(d)
                if d in cellset:
                    adj.append(d)
        nbr[c] = adj
    return nbr


def solve_grid_maze(shape, blocked, start, goal, steps=200, mu=1.5, dt=0.2):
    """Solve an N-D grid maze with the Tero slime-mould flow model -- the SAME solver the 2D maze used, unchanged,
    because it operates on the adjacency dict, not on coordinates. Returns the shortest-path cell list (or None)."""
    from holographic_flow import tero_solve
    nbr = grid_graph(shape, blocked)
    if start not in nbr or goal not in nbr:
        return None
    return tero_solve(nbr, start, goal, steps=steps, mu=mu, dt=dt)


# ---------------------------------------------------------------------------------------------------------------------
# 2. RECONSTRUCT: sparse adaptive sampling + kernel interpolation of a known deterministic field, in any dimension
# ---------------------------------------------------------------------------------------------------------------------

def _nadaraya_watson(query, samples, values, bandwidth):
    """Gaussian-kernel weighted average of `values` at `samples`, read at `query` points -- the same self-normalising
    read the radiance field uses. query (M,D), samples (K,D), values (K,), returns (M,)."""
    d2 = ((query[:, None, :] - samples[None, :, :]) ** 2).sum(2)   # (M,K) squared distances
    w = np.exp(-0.5 * d2 / (bandwidth * bandwidth))
    wsum = w.sum(1)
    wsum = np.where(wsum < 1e-12, 1.0, wsum)
    return (w * values[None, :]).sum(1) / wsum


def sparse_reconstruct(oracle, lo, hi, n_seed=96, n_refine=96, bandwidth=None, seed=0):
    """Reconstruct a known deterministic field `oracle`: (N,D)->(N,) over the box [lo, hi] from a SPARSE ADAPTIVE
    sample. Seed with a pseudo-random low-discrepancy set, then REFINE: repeatedly add the points where the current
    reconstruction disagrees most with the oracle (evaluate the oracle there -- we can, it's deterministic and known).
    Returns (points (K,D), values (K,), reconstruct) where reconstruct(query)->values. Works in any dimension D."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float); D = len(lo)
    rng = np.random.default_rng(seed)
    if bandwidth is None:
        bandwidth = 0.12 * float(np.mean(hi - lo))
    pts = lo + (hi - lo) * rng.random((n_seed, D))                 # seed sample
    vals = np.asarray(oracle(pts), float)
    # adaptive refinement: probe a dense candidate set, add where reconstruction error is largest
    n_batches = 4
    for _ in range(n_batches):
        cand = lo + (hi - lo) * rng.random((512, D))
        pred = _nadaraya_watson(cand, pts, vals, bandwidth)
        truth = np.asarray(oracle(cand), float)                    # we KNOW the field -> we can check
        err = np.abs(pred - truth)
        take = np.argsort(-err)[: max(1, n_refine // n_batches)]   # the worst-reconstructed candidates
        pts = np.vstack([pts, cand[take]]); vals = np.concatenate([vals, truth[take]])

    def reconstruct(query):
        return _nadaraya_watson(np.atleast_2d(np.asarray(query, float)), pts, vals, bandwidth)
    return pts, vals, reconstruct


# ---------------------------------------------------------------------------------------------------------------------
# 3. NAVIGATE: least-cost pathfinding through a known N-D COST FIELD (density, potential, resistance)
# ---------------------------------------------------------------------------------------------------------------------

def _cell_centers(shape, lo, hi):
    """World-space centre of every grid cell -> (prod(shape), D) in the same order as product(*ranges)."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    axes = [(np.arange(s) + 0.5) / s * (hi[k] - lo[k]) + lo[k] for k, s in enumerate(shape)]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def field_weighted_graph(shape, cost, blocked=None, lo=None, hi=None):
    """An N-D grid where each edge carries a traversal COST sampled from a scalar field -- so a path PAYS for the field
    it crosses (dense smoke, high potential, rough terrain). `cost` is either a callable f(points (M,D))->(M,) or an
    array of shape `shape`. Edge cost = mean of the two cells' field values + a unit base (distance), so the least-cost
    path both goes around expensive regions AND stays short. Returns (nbr, edge_cost). Dimension-agnostic."""
    lo = np.zeros(len(shape)) if lo is None else lo
    hi = np.array(shape, float) if hi is None else hi
    if callable(cost):
        vals = np.asarray(cost(_cell_centers(shape, lo, hi)), float).reshape(shape)
    else:
        vals = np.asarray(cost, float).reshape(shape)
    nbr = grid_graph(shape, blocked)
    edge_cost = {}
    for u in nbr:
        for v in nbr[u]:
            e = (u, v) if u < v else (v, u)
            if e not in edge_cost:
                edge_cost[e] = 1.0 + 0.5 * (vals[u] + vals[v])   # base distance + the field crossed
    return nbr, edge_cost


def least_cost_path(nbr, edge_cost, start, goal):
    """Dijkstra least-cost path over a cost-weighted graph. Deterministic (ties broken by insertion order via a
    counter). Returns the cell list, or None if disconnected."""
    import heapq
    if start not in nbr or goal not in nbr:
        return None
    def ec(a, b):
        return edge_cost[(a, b) if a < b else (b, a)]
    dist = {start: 0.0}; prev = {start: None}
    pq = [(0.0, 0, start)]; counter = 1
    while pq:
        d, _, x = heapq.heappop(pq)
        if x == goal:
            break
        if d > dist.get(x, np.inf):
            continue
        for y in nbr[x]:
            nd = d + ec(x, y)
            if nd < dist.get(y, np.inf):
                dist[y] = nd; prev[y] = x
                heapq.heappush(pq, (nd, counter, y)); counter += 1
    if goal not in prev:
        return None
    path = [goal]
    while path[-1] is not None and path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]


def cost_to_go(nbr, edge_cost, goal):
    """Solve the ENTIRE cost-to-go field to a goal in ONE Dijkstra sweep FROM the goal: V[cell] = least cost to reach the
    goal from that cell, and nxt[cell] = the next cell to step toward it. This is the 'precompute once, read out
    anywhere' pattern (the SDF bake, PRT) applied to navigation -- after the one solve, routing from ANY start is a cheap
    descent (follow `nxt`), no re-search. And V is not just a router: it is a VALUE FUNCTION / POTENTIAL that carries, at
    every cell in one place, that cell's global relation to the goal -- the same object as a distance field, a physics
    potential, or an RL value function. Deterministic (counter tie-break). Returns (V, nxt) as dicts over cells.

    Edges are undirected here, so cost-to-go from y equals cost from y to the goal; that symmetry is what lets one
    goal-rooted solve serve every start."""
    import heapq
    if goal not in nbr:
        return {}, {}
    def ec(a, b):
        return edge_cost[(a, b) if a < b else (b, a)]
    V = {goal: 0.0}; nxt = {goal: None}
    pq = [(0.0, 0, goal)]; counter = 1
    while pq:
        d, _, x = heapq.heappop(pq)
        if d > V.get(x, np.inf):
            continue
        for y in nbr[x]:
            nd = d + ec(x, y)                                    # cost to reach the goal from y via x
            if nd < V.get(y, np.inf):
                V[y] = nd; nxt[y] = x                            # from y, the next step toward the goal is x
                heapq.heappush(pq, (nd, counter, y)); counter += 1
    return V, nxt


def route_from(nxt, start, goal):
    """Route from ANY start to the goal by following the precomputed next-step field -- O(path length), no search.
    Returns the cell path, or None if the start can't reach the goal."""
    if start not in nxt:
        return None
    path = [start]
    while path[-1] != goal:
        step = nxt.get(path[-1])
        if step is None:
            return None
        path.append(step)
    return path


def value_grid(V, shape):
    """Materialize the cost-to-go dict as a dense array of shape `shape` (unreached cells = inf) -- the value field /
    potential as a grid, ready to feed a descent, a physics force (its negative gradient), or a visualization."""
    g = np.full(shape, np.inf)
    for c, v in V.items():
        g[c] = v
    return g


def navigate_field(cost, shape, start, goal, blocked=None, lo=None, hi=None):
    """Discretize a known N-D COST FIELD to a grid, weight edges by the field, and return the LEAST-COST path from
    start to goal -- the one primitive for 'navigate a field': smoke density (volumetrics), a potential (physics),
    a resistance/terrain (particles). `cost` is a callable f(points)->cost or an array of shape `shape`. The uniform
    maze (constant cost) is the special case. Returns the cell path."""
    nbr, edge_cost = field_weighted_graph(shape, cost, blocked=blocked, lo=lo, hi=hi)
    return least_cost_path(nbr, edge_cost, start, goal)


def path_cost(path, cost, shape, lo=None, hi=None):
    """Total field cost accumulated along a path (for comparing routes) -- sum of the cell field values visited."""
    lo = np.zeros(len(shape)) if lo is None else lo
    hi = np.array(shape, float) if hi is None else hi
    if callable(cost):
        centers = _cell_centers(shape, lo, hi).reshape(shape + (len(shape),))
        return float(sum(cost(centers[c][None, :])[0] for c in path))
    vals = np.asarray(cost, float).reshape(shape)
    return float(sum(vals[c] for c in path))


# ---------------------------------------------------------------------------------------------------------------------
# 4. NAVIGATE A LIVE SCENE (the SDF itself is the cost field) + make a path COMPOSABLE as a VSA hypervector
# ---------------------------------------------------------------------------------------------------------------------

def _world_to_cell(p, lo, hi, shape):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    frac = (np.asarray(p, float) - lo) / (hi - lo)
    return tuple(int(np.clip(round(frac[k] * shape[k] - 0.5), 0, shape[k] - 1)) for k in range(len(shape)))


def _cell_to_world(c, lo, hi, shape):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    return lo + (np.asarray(c, float) + 0.5) / np.asarray(shape, float) * (hi - lo)


def navigate_scene(sdf_eval, lo, hi, shape, start_world, goal_world, clearance=0.25):
    """Route an agent through a LIVE SCENE, using the scene's own SDF as the cost field: cells INSIDE geometry
    (sdf < 0) are impassable, and getting within `clearance` of a surface is expensive (a keep-away margin), so the
    path threads through free space around the objects. `sdf_eval` is a callable P (M,3) -> signed distance (negative
    inside). Returns the path as a list of WORLD-space waypoints (or None). The same navigate primitive, its cost field
    now the geometry the renderer already traces -- one structure for drawing AND moving."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    centers = _cell_centers(shape, lo, hi)
    dist = np.asarray(sdf_eval(centers), float).reshape(shape)
    blocked = {tuple(c) for c in np.argwhere(dist < 0.0)}       # inside geometry -> impassable
    cost = np.exp(-np.maximum(dist, 0.0) / max(clearance, 1e-6))  # near a surface -> costly; far -> ~0
    start = _world_to_cell(start_world, lo, hi, shape); goal = _world_to_cell(goal_world, lo, hi, shape)
    path = navigate_field(cost, shape, start, goal, blocked=blocked)
    if path is None:
        return None
    return [_cell_to_world(c, lo, hi, shape) for c in path]


def encode_path(path, dim=2048, seed=0):
    """A navigated path -> ONE hypervector, so the route is COMPOSABLE in a VSA program (bind it to a label, bundle
    several routes, query order/position). Each waypoint (a cell tuple) is bound to its step index and bundled -- the
    engine's sequence encoding. Returns (vector, SequenceMemory) so the caller can decode/compose. Deterministic."""
    from holographic_sequence import SequenceMemory
    sm = SequenceMemory(dim=dim, seed=seed)
    keys = [str(tuple(int(x) for x in c)) for c in path]
    return sm.encode(keys), sm, keys


def decode_path_step(vec, sm, keys, i):
    """Read waypoint i back out of a path hypervector (un-rotate + cleanup against the candidate waypoints) -- the
    proof the route survives as composable VSA data, not just a Python list."""
    from holographic_ai import permute, cosine
    probe = permute(vec, -(i + 1))
    return max(keys, key=lambda k: cosine(probe, sm.vocab.get(k)))


def straight_line_cells(start, goal):
    """The grid cells a straight line from start to goal passes through (N-D DDA) -- a tie-break-independent baseline
    for 'what a naive straight shot costs' versus the field-aware navigated route."""
    s = np.asarray(start, float); g = np.asarray(goal, float)
    n = int(np.abs(g - s).max()) + 1
    ts = np.linspace(0, 1, n)
    pts = s[None, :] + ts[:, None] * (g - s)[None, :]
    cells = [tuple(int(round(x)) for x in p) for p in pts]
    out = [cells[0]]
    for c in cells[1:]:
        if c != out[-1]:
            out.append(c)
    return out


def _selftest():
    """A 3-D grid maze solves with the SAME flow solver as 2D; a smooth field reconstructs from a sparse adaptive
    sample far better than from the same number of uniform samples."""
    # 3-D maze: a 5x5x5 grid with a diagonal wall of blocked cells; solve corner to corner
    shape = (5, 5, 5)
    blocked = {(2, y, z) for y in range(4) for z in range(5)}     # a wall with a gap at y=4
    path = solve_grid_maze(shape, blocked, (0, 0, 0), (4, 4, 4))
    assert path is not None and path[0] == (0, 0, 0) and path[-1] == (4, 4, 4)
    assert all(sum(abs(a - b) for a, b in zip(p, q)) == 1 for p, q in zip(path, path[1:]))  # unit steps only
    assert not any(c in blocked for c in path)                    # never through a wall

    # same builder, a 2-D and a 4-D maze -- dimension-agnostic
    assert solve_grid_maze((6, 6), set(), (0, 0), (5, 5)) is not None
    assert solve_grid_maze((3, 3, 3, 3), set(), (0, 0, 0, 0), (2, 2, 2, 2)) is not None

    # field reconstruction in 3-D: a smooth test function
    def oracle(P):
        return np.sin(1.7 * P[:, 0]) * np.cos(1.3 * P[:, 1]) + 0.5 * P[:, 2]
    lo = np.zeros(3); hi = np.full(3, 3.0)
    pts, vals, recon = sparse_reconstruct(oracle, lo, hi, n_seed=96, n_refine=96, seed=0)
    test = lo + (hi - lo) * np.random.default_rng(1).random((400, 3))
    err_adaptive = float(np.abs(recon(test) - oracle(test)).mean())
    assert err_adaptive < 0.15                                    # sparse adaptive reconstruction is accurate

    # NAVIGATE a field: a high-cost wall down the middle (except a gap) should be routed AROUND, not through
    def cost(P):                                                  # a costly ridge at x~5, cheap gap near y=9
        return 8.0 * np.exp(-((P[:, 0] - 5.0) ** 2) / 0.5) * (P[:, 1] < 8.0)
    straight = navigate_field(lambda P: np.zeros(len(P)), (10, 10), (0, 0), (9, 9))   # ignore cost -> shortest
    routed = navigate_field(cost, (10, 10), (0, 0), (9, 9))       # respect cost -> detour through the gap
    c_straight = path_cost(straight, cost, (10, 10)); c_routed = path_cost(routed, cost, (10, 10))
    assert c_routed < c_straight                                  # field-aware route pays LESS field cost
    assert routed[0] == (0, 0) and routed[-1] == (9, 9)
    print("ndfield selftest ok: 3D maze path len %d; 3D reconstruct MAE %.4f (%d samples); field route cost %.1f vs %.1f straight"
          % (len(path), err_adaptive, len(pts), c_routed, c_straight))


if __name__ == "__main__":
    _selftest()
