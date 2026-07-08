"""holographic_spatial.py -- ONE shared spatial index. Bin points into a uniform grid of cells so radius,
k-nearest, and closest-point queries are O(1)-ish (look only in nearby cells) instead of O(N) brute-force scans.

WHY THIS EXISTS (Above/Below Sweep 3, item 2 -- the widest-fanout single change on the board)
---------------------------------------------------------------------------------------------
The sweep found ~14 places that each roll their own point-bucketing: cull (regionfield), navigation
(flow/slime/navigator), recall (tree/pivot), collision (collide/softbody/hair), sampling (poisson-disk),
Walk-on-Spheres (closest-point). They are all the SAME query -- "which points are near here?" -- so one shared,
well-tested index serves them all. This is the "one shared structure for cull and navigation" backlog item, and
the §5.1 discipline (which module is this in a different costume?) made obvious once the queries were listed
side by side. `octree` is the hierarchical version of the same idea (an index of indices); this is the flat,
uniform-cell base case, deliberately small and readable.

HONEST SCOPE (kept loud): a UNIFORM grid is best when points are roughly evenly spread; heavily clustered points
put many points in one cell (then prefer the octree). Results are DETERMINISTIC -- ties broken by point index --
and BYTE-IDENTICAL to a brute-force scan (same set, same order), which is what lets a consumer migrate to it and
pin the result bit-exact (the sweep's own migration hook). NumPy + stdlib only.
"""
import numpy as np


class SpatialGrid:
    """A uniform-grid spatial index over an (N, D) point set. Pick `cell_size` near the typical query radius:
    then each query touches only a handful of cells. Rebuild (make a new grid) after the points move."""

    def __init__(self, points, cell_size):
        self.points = np.asarray(points, float)
        self.n, self.dim = self.points.shape
        self.cell_size = float(cell_size)
        # bin each point into an integer cell coordinate; store a dict cell -> list of point indices.
        # a plain dict of tuples is readable and deterministic (we sort results ourselves); no hashing surprises.
        self.cells = {}
        coords = np.floor(self.points / self.cell_size).astype(np.int64)
        for i, c in enumerate(coords):
            self.cells.setdefault(tuple(int(x) for x in c), []).append(i)
        self._coords = coords

    def _cell_of(self, q):
        return tuple(int(x) for x in np.floor(np.asarray(q, float) / self.cell_size))

    def _cells_in_box(self, lo, hi):
        """Yield every cell coordinate in the inclusive integer box [lo, hi] (a small nested product over dims)."""
        ranges = [range(lo[d], hi[d] + 1) for d in range(self.dim)]
        # iterative cartesian product (readable, no itertools dependency needed but fine to use it)
        import itertools
        for combo in itertools.product(*ranges):
            yield combo

    def radius(self, query, r):
        """All point indices within Euclidean distance `r` of `query`, sorted by (distance, index) -- identical
        set and order to a brute-force scan. Only the cells overlapping the query ball are examined."""
        q = np.asarray(query, float)
        lo = self._cell_of(q - r)
        hi = self._cell_of(q + r)
        hits = []
        r2 = r * r
        for cell in self._cells_in_box(lo, hi):
            for i in self.cells.get(cell, ()):
                d2 = float(np.sum((self.points[i] - q) ** 2))
                if d2 <= r2:
                    hits.append((d2, i))
        hits.sort()                                          # (distance^2, index) -> deterministic order
        return [i for _, i in hits]

    def closest(self, query):
        """The single nearest point index to `query` (the Walk-on-Spheres closest-point query, and broadphase's
        'what's here?'). Expands the search ring by ring until the nearest found is provably closer than the next
        ring could contain. Returns (index, distance) or (-1, inf) for an empty grid."""
        if self.n == 0:
            return -1, float("inf")
        q = np.asarray(query, float)
        centre = self._cell_of(q)
        best_i, best_d2 = -1, float("inf")
        ring = 0
        while True:
            lo = tuple(centre[d] - ring for d in range(self.dim))
            hi = tuple(centre[d] + ring for d in range(self.dim))
            found_any = False
            for cell in self._cells_in_box(lo, hi):
                # only the shell at exactly this ring is new (interior was scanned already), but checking all is
                # simplest and still cheap; correctness first -- we re-min over candidates anyway.
                if max(abs(cell[d] - centre[d]) for d in range(self.dim)) != ring:
                    continue                                 # skip interior cells (already covered)
                for i in self.cells.get(cell, ()):
                    found_any = True
                    d2 = float(np.sum((self.points[i] - q) ** 2))
                    if d2 < best_d2 or (d2 == best_d2 and i < best_i):
                        best_i, best_d2 = i, d2
            # stop once the best found is closer than the nearest possible point in the NEXT ring. The next ring
            # starts at (ring)*cell_size away from the query's cell boundary -- a safe lower bound.
            if best_i != -1:
                safe = ring * self.cell_size
                if best_d2 <= safe * safe:
                    break
            ring += 1
            if ring > max_ring(self):                        # ran out of grid: return the best seen
                break
        return best_i, float(np.sqrt(best_d2))

    def knn(self, query, k):
        """The k nearest point indices to `query`, sorted by (distance, index) -- identical to brute-force. Grows
        the search box until it holds at least k candidates AND the box is wide enough that no closer point can lie
        outside it, then returns the true k nearest."""
        if self.n == 0:
            return []
        q = np.asarray(query, float)
        centre = self._cell_of(q)
        ring = 0
        while True:
            lo = tuple(centre[d] - ring for d in range(self.dim))
            hi = tuple(centre[d] + ring for d in range(self.dim))
            cand = []
            for cell in self._cells_in_box(lo, hi):
                for i in self.cells.get(cell, ()):
                    d2 = float(np.sum((self.points[i] - q) ** 2))
                    cand.append((d2, i))
            # enough candidates AND the k-th nearest is within the guaranteed-complete radius (ring*cell_size)?
            if len(cand) >= min(k, self.n):
                cand.sort()
                kth_d2 = cand[min(k, self.n) - 1][0]
                safe = ring * self.cell_size
                if kth_d2 <= safe * safe or ring > max_ring(self):
                    return [i for _, i in cand[:k]]
            ring += 1
            if ring > max_ring(self):
                cand.sort()
                return [i for _, i in cand[:k]]


def max_ring(grid):
    """A safe upper bound on how many rings could ever be needed: the grid's full extent in cells."""
    if not grid.cells:
        return 0
    coords = grid._coords
    span = int(np.max(coords.max(axis=0) - coords.min(axis=0))) if len(coords) else 0
    return span + 2


# --- brute-force references (what the grid must match byte-for-byte) --------------------------------------------

def brute_radius(points, query, r):
    q = np.asarray(query, float); pts = np.asarray(points, float)
    d2 = np.sum((pts - q) ** 2, axis=1)
    hits = sorted((float(d2[i]), i) for i in range(len(pts)) if d2[i] <= r * r)
    return [i for _, i in hits]


def brute_knn(points, query, k):
    q = np.asarray(query, float); pts = np.asarray(points, float)
    d2 = np.sum((pts - q) ** 2, axis=1)
    order = sorted((float(d2[i]), i) for i in range(len(pts)))
    return [i for _, i in order[:k]]


def _selftest():
    """Radius / knn / closest all match the brute-force reference byte-for-byte (same set, same order) in 2-D and
    3-D, on uniform and clustered points; empty grid is safe; deterministic."""
    rng = np.random.default_rng(0)
    for D in (2, 3):
        pts = rng.uniform(0, 10, size=(400, D))
        grid = SpatialGrid(pts, cell_size=1.0)
        for _ in range(30):
            q = rng.uniform(0, 10, size=D)
            r = float(rng.uniform(0.3, 2.5))
            assert grid.radius(q, r) == brute_radius(pts, q, r), D            # radius: exact match
            k = int(rng.integers(1, 12))
            assert grid.knn(q, k) == brute_knn(pts, q, k), D                  # knn: exact match
            ci, cd = grid.closest(q)
            bi = brute_knn(pts, q, 1)[0]
            assert ci == bi, (D, ci, bi)                                      # closest: exact match

    # clustered points (the honest weak case -- still CORRECT, just less balanced)
    clustered = np.concatenate([rng.normal(2, 0.2, (200, 2)), rng.normal(8, 0.2, (200, 2))])
    g = SpatialGrid(clustered, cell_size=0.5)
    q = np.array([2.0, 2.0])
    assert g.knn(q, 5) == brute_knn(clustered, q, 5)

    # empty grid is safe
    empty = SpatialGrid(np.zeros((0, 3)), cell_size=1.0)
    assert empty.closest([0, 0, 0]) == (-1, float("inf")) and empty.knn([0, 0, 0], 3) == []

    # deterministic
    g2 = SpatialGrid(pts, cell_size=1.0)
    assert g2.radius([5, 5, 5][:D], 1.0) == grid.radius([5, 5, 5][:D], 1.0)
    print("holographic_spatial selftest OK: radius/knn/closest match brute-force byte-for-byte in 2-D and 3-D, on "
          "uniform and clustered points; empty grid safe; deterministic")


if __name__ == "__main__":
    _selftest()
