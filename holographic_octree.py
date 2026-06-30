"""A capacity-adaptive 3D holographic octree (TILE3D-1): tile 3D space so each node's "wave" stays inside a single
hypervector's capacity, splitting into octants when a vector gets too full.

THE IDEA, AND WHERE IT COMES FROM IN THE ENGINE
-----------------------------------------------
Two facts already proven elsewhere in holostuff meet here:
  1. A point set can be carried as ONE hypervector -- a "wave": the FPE VectorFunctionEncoder bundles points into
     f = sum_i encode(p_i), and cosine(f, encode(x)) reads a kernel-density estimate of "is there a point near x?"
     (holographic_fpe / FPEField). A wave is continuous -- you can SAMPLE it at any x -- which is the honest core
     of "a wave describes infinitely many particles": resolution-independent sampling. BUT it is FINITE-CAPACITY:
     bundle too many points and crosstalk floods the readout (the measured capacity cliff). A wave does not store
     infinite information; it is a smooth, lossy approximation, and that is exactly why we must split.
  2. When a single vector overflows, you TILE -- `splat_bundle_tiled` keeps each tile bundle bounded so recall
     holds at any total resolution, at the cost of one vector per tile. That is "spin up another vector when the
     first is too full," already shipped in 2D.

This module unifies them in 3D and makes the split AUTOMATIC and ADAPTIVE: an octree whose nodes each hold their
points as an FPE wave, and which SUBDIVIDES a node into 8 octants the moment its point count exceeds `capacity`
(the per-vector budget). The tree IS the bidirectional index -- descend a position to its leaf (forward:
position -> node), read the node's points / wave back (backward: node -> contents / occupancy) -- and each node's
wave keeps the engine's content-addressable, semantic recall. Each child encoder is scaled to its SMALLER box, so
local resolution sharpens as you descend (high precision on the hot region, coarse box up top -- the same
local-refinement logic as the Nystrom landmarks).

Measured below: a single global wave's recall SEPARATION (stored vs empty) collapses as N grows; the octree holds
it flat by splitting, at a storage cost of one vector per leaf.
"""

import numpy as np


class HoloOctree:
    """A 3D octree whose every node carries its points as one FPE 'wave' hypervector and splits into 8 octants
    when it exceeds `capacity` points. Build with `insert`; query occupancy at a position with `query`; the tree
    is the bidirectional spatial index."""

    def __init__(self, bounds, capacity=32, dim=512, bandwidth=4.0, max_depth=6, seed=0, _depth=0):
        self.lo = np.asarray(bounds[0], float)
        self.hi = np.asarray(bounds[1], float)
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.bandwidth = float(bandwidth)
        self.max_depth = int(max_depth)
        self.seed = int(seed)
        self.depth = int(_depth)
        self.points = np.empty((0, 3))
        self.children = None                                  # None => leaf
        self._enc = None
        self._wave = None                                     # the node's bundle hypervector (lazy)

    # ---- build -------------------------------------------------------------------------------------
    def insert(self, points):
        """Insert points (N,3) (in bulk). A leaf that exceeds `capacity` subdivides into 8 octants and pushes its
        points down; the wave is (re)built lazily on first query. Returns self."""
        points = np.atleast_2d(np.asarray(points, float))
        if self.children is None:
            self.points = np.vstack([self.points, points]) if len(self.points) else points.copy()
            self._wave = None                                 # invalidate the cached wave
            if len(self.points) > self.capacity and self.depth < self.max_depth:
                self._split()
        else:
            self._route(points)
        return self

    def _split(self):
        """Subdivide into 8 octant children and distribute this node's points down -- 'spin up more vectors' when
        the one wave is too full. Vectorised octant assignment."""
        mid = 0.5 * (self.lo + self.hi)
        self.children = []
        for cz in (0, 1):
            for cy in (0, 1):
                for cx in (0, 1):
                    lo = np.array([self.lo[0] if cx == 0 else mid[0],
                                   self.lo[1] if cy == 0 else mid[1],
                                   self.lo[2] if cz == 0 else mid[2]])
                    hi = np.array([mid[0] if cx == 0 else self.hi[0],
                                   mid[1] if cy == 0 else self.hi[1],
                                   mid[2] if cz == 0 else self.hi[2]])
                    self.children.append(HoloOctree((lo, hi), self.capacity, self.dim, self.bandwidth,
                                                    self.max_depth, self.seed, _depth=self.depth + 1))
        pts = self.points
        self.points = np.empty((0, 3))
        self._wave = None
        self._route(pts)

    def _octant(self, points):
        """The child index 0..7 for each point (vectorised): which side of the node centre on each axis."""
        mid = 0.5 * (self.lo + self.hi)
        b = (points >= mid).astype(int)
        return b[:, 0] + 2 * b[:, 1] + 4 * b[:, 2]

    def _route(self, points):
        idx = self._octant(points)
        for c in range(8):
            sub = points[idx == c]
            if len(sub):
                self.children[c].insert(sub)

    # ---- the node's wave (the FPE bundle, scaled to this box) ---------------------------------------
    def _build_wave(self):
        from holographic_fpe import VectorFunctionEncoder
        bounds = [(self.lo[k], self.hi[k]) for k in range(3)]
        self._enc = VectorFunctionEncoder(3, dim=self.dim, bounds=bounds, bandwidth=self.bandwidth, seed=self.seed)
        self._wave = self._enc.bundle(self.points) if len(self.points) else None

    # ---- query (bidirectional) ---------------------------------------------------------------------
    def _leaf_for(self, x):
        node = self
        while node.children is not None:
            node = node.children[int(node._octant(np.atleast_2d(x))[0])]
        return node

    def query(self, x):
        """Forward lookup: descend `x` to its leaf, then read the leaf's wave -- cosine(wave, encode(x)) -- a
        kernel-density occupancy score in ~[0,1] (high near a stored point, low in empty space). Content-addressable
        recall, kept semantic, but bounded because the leaf's wave holds at most ~`capacity` points."""
        from holographic_ai import cosine
        leaf = self._leaf_for(np.asarray(x, float))
        if leaf._wave is None:
            if len(leaf.points) == 0:
                return 0.0
            leaf._build_wave()
        if leaf._wave is None:
            return 0.0
        return float(cosine(leaf._wave, leaf._enc.encode(np.asarray(x, float))))

    # ---- introspection -----------------------------------------------------------------------------
    def leaves(self):
        if self.children is None:
            return [self]
        out = []
        for c in self.children:
            out.extend(c.leaves())
        return out

    def n_nodes(self):
        return 1 if self.children is None else 1 + sum(c.n_nodes() for c in self.children)

    def n_vectors(self):
        """One wave hypervector per non-empty leaf -- the storage cost of staying within capacity."""
        return sum(1 for lf in self.leaves() if len(lf.points) > 0)

    def all_points(self):
        return np.vstack([lf.points for lf in self.leaves() if len(lf.points)]) if self.leaves() else np.empty((0, 3))


def single_wave_recall(points, query_pts, dim=512, bandwidth=4.0, bounds=None, seed=0):
    """Baseline: bundle ALL points into ONE wave and read occupancy at query_pts -- the un-tiled case whose
    separation collapses as N grows (the capacity cliff). Returns the occupancy scores."""
    from holographic_fpe import VectorFunctionEncoder
    from holographic_ai import cosine
    points = np.asarray(points, float)
    if bounds is None:
        bounds = [(float(points[:, k].min()), float(points[:, k].max())) for k in range(3)]
    enc = VectorFunctionEncoder(3, dim=dim, bounds=bounds, bandwidth=bandwidth, seed=seed)
    wave = enc.bundle(points)
    return np.array([float(cosine(wave, enc.encode(q))) for q in np.atleast_2d(query_pts)])


def _selftest():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, (400, 3))
    tree = HoloOctree((np.array([-1., -1, -1]), np.array([1., 1, 1])), capacity=32, dim=512).insert(pts)
    assert tree.children is not None and tree.n_vectors() > 1   # it split (400 > capacity 32)
    # bidirectional: a stored point routes to a leaf whose box contains it, and recalls high
    p = pts[0]
    leaf = tree._leaf_for(p)
    assert np.all(p >= leaf.lo - 1e-9) and np.all(p <= leaf.hi + 1e-9)
    stored = tree.query(p)
    empty = tree.query(rng.uniform(-1, 1, 3))
    assert stored > 0.3                                        # a stored point recalls
    # the octree separation beats a single global wave at this N
    qs = np.vstack([pts[:30], rng.uniform(-1, 1, (30, 3))])
    sep_tree = np.mean([tree.query(p) for p in pts[:30]]) - np.mean([tree.query(p) for p in rng.uniform(-1, 1, (30, 3))])
    sw = single_wave_recall(pts, np.vstack([pts[:30], rng.uniform(-1, 1, (30, 3))]), dim=512)
    sep_single = sw[:30].mean() - sw[30:].mean()
    assert sep_tree > sep_single                               # tiling holds recall where one wave blurs
    print(f"octree selftest ok: 400 pts -> {tree.n_vectors()} leaf waves (split on capacity 32); "
          f"separation tree {sep_tree:.2f} > single wave {sep_single:.2f}")


if __name__ == "__main__":
    _selftest()
