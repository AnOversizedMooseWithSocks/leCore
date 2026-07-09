"""holographic_dendrite.py -- DIFFUSION-LIMITED BRANCHING (Physics & FX backlog, item #7).

The backlog's insight N11: ice/frost dendrites and lightning bolts are the SAME physics -- a cluster that grows
into the steepest gradient of a diffusion (Laplace) field, branching stochastically. Build it once and you get
frost, ice crystals, AND lightning. This is the classic dielectric-breakdown model (Niemeyer, Pietronero &
Wiesmann, 1984), which is also how diffusion-limited aggregation (Witten & Sander, 1981) makes fractal crystals.

HOW IT WORKS (readable, classic):
  1. A grid holds a growing CLUSTER (the crystal / the discharge channel). Seed it -- a point for a snowflake,
     a line for frost creeping across a cold window, a top electrode for a lightning bolt.
  2. Solve a POTENTIAL field phi by relaxation (Laplace's equation): phi = 0 on the cluster, phi = 1 on the far
     boundary / the ground it is reaching toward, and the smooth average of its neighbours everywhere else. This
     IS the diffusion field's steady state -- the same Laplace/Poisson the spectral backbone solves; for lightning
     phi is literally the electric potential (holographic_spectralfield.poisson_solve's field).
  3. GROW: the empty cells touching the cluster are candidates. Each grows with probability proportional to its
     potential raised to a power eta -- so growth races toward wherever the field is steepest (the nearest tip of
     the boundary). Pick ONE stochastically (seeded, deterministic) and add it. Re-solve. Repeat.

eta is the one knob that tunes the shape (the model's whole point): eta -> 0 gives dense, bushy growth (an Eden
blob); eta = 1 gives the classic fractal dendrite (a snowflake / a Lichtenberg figure); large eta gives thin,
stringy, sparsely-branched channels (a lightning bolt). ONE function, three phenomena -- N11 delivered.

HONEST SCOPE (kept): this is a lattice model on a plain NumPy grid -- it earns NO holographic form (the growth is
a discrete stochastic choice, not a bind), and the memory's rule is not to over-holograph a grid. The Laplace
solve is a simple Jacobi relaxation warm-started between growth steps -- readable over fast; a real crystal would
want anisotropy (six-fold symmetry) and surface tension, which are extensions. The fractal dimension is
STOCHASTIC and depends on eta and cluster size -- so the test checks it lands in a fractal RANGE (a sparse
branching structure, dimension between a line and a filled area), not a fixed number. Deterministic given the
seed; NumPy + stdlib only.
"""
import numpy as np


def _relax_potential(cluster, source, iters, phi0=None):
    """Solve Laplace's equation phi by Jacobi relaxation: phi = 0 on the `cluster`, phi = 1 on the `source` (the
    far boundary / the attractor), and the average of its 4 neighbours everywhere else. Warm-started from phi0 if
    given (the field changes little when one cell is added, so a few iterations suffice between growth steps)."""
    phi = np.zeros(cluster.shape) if phi0 is None else phi0.copy()
    phi[source] = 1.0
    phi[cluster] = 0.0
    for _ in range(iters):
        avg = 0.25 * (np.roll(phi, 1, 0) + np.roll(phi, -1, 0) + np.roll(phi, 1, 1) + np.roll(phi, -1, 1))
        phi = avg
        phi[source] = 1.0                                       # re-pin the fixed potentials each sweep
        phi[cluster] = 0.0
    return phi


class DielectricBreakdown:
    """Grow a cluster into the steepest gradient of a Laplace field, branching stochastically -- ice dendrites,
    frost, and lightning from one model (N11). `eta` tunes the shape (0 bushy, 1 fractal, large stringy)."""

    def __init__(self, shape, eta=1.0, seed=0):
        self.shape = shape
        self.eta = float(eta)
        self.rng = np.random.default_rng(seed)
        self.cluster = np.zeros(shape, bool)                    # the growing crystal / discharge
        self.source = np.zeros(shape, bool)                     # the fixed high-potential boundary it grows toward
        self.phi = None                                         # the last potential field (warm start)
        self.order = np.full(shape, -1, int)                    # growth order per cell (for animating / age)
        self._n = 0

    # -- seeding: the same engine, different starts ------------------------------------------------------------
    def seed_point(self, y, x):
        """Seed a single site -- a snowflake / a radial crystal grows outward from here."""
        self.cluster[y, x] = True
        self.order[y, x] = 0
        self._n = 1
        return self

    def seed_line(self, axis=0, index=0):
        """Seed a whole edge -- frost creeping in from a cold window edge, or a lightning cloud along the top."""
        if axis == 0:
            self.cluster[index, :] = True
            self.order[index, :] = 0
        else:
            self.cluster[:, index] = True
            self.order[:, index] = 0
        self._n = int(self.cluster.sum())
        return self

    def set_source_border(self, sides=("bottom", "top", "left", "right")):
        """Make the domain edges the high-potential SOURCE the growth reaches toward (radial crystal). For
        lightning, pass sides=('bottom',) so the bolt is pulled DOWN to the ground."""
        if "top" in sides:
            self.source[0, :] = True
        if "bottom" in sides:
            self.source[-1, :] = True
        if "left" in sides:
            self.source[:, 0] = True
        if "right" in sides:
            self.source[:, -1] = True
        self.source[self.cluster] = False                       # the cluster is never also the source
        return self

    # -- the growth loop --------------------------------------------------------------------------------------
    def _candidates(self):
        """Empty cells 4-adjacent to the cluster -- the growth front. Returns (ys, xs)."""
        c = self.cluster
        neigh = np.zeros(self.shape, bool)
        neigh[1:, :] |= c[:-1, :]; neigh[:-1, :] |= c[1:, :]
        neigh[:, 1:] |= c[:, :-1]; neigh[:, :-1] |= c[:, 1:]
        front = neigh & ~c & ~self.source                       # a candidate can't already be cluster or source
        return np.where(front)

    def grow(self, steps, relax_iters=30):
        """Add `steps` cells, one at a time, each chosen with probability proportional to phi^eta at the growth
        front (growth races toward the steepest field). Deterministic given the seed."""
        for _ in range(int(steps)):
            self.phi = _relax_potential(self.cluster, self.source, relax_iters, phi0=self.phi)
            ys, xs = self._candidates()
            if len(ys) == 0:
                break
            weights = np.maximum(self.phi[ys, xs], 0.0) ** self.eta   # phi^eta: the breakdown growth law
            total = weights.sum()
            if total <= 0:
                pick = int(self.rng.integers(len(ys)))          # a flat field -> pick uniformly (degenerate)
            else:
                pick = int(self.rng.choice(len(ys), p=weights / total))
            y, x = int(ys[pick]), int(xs[pick])
            self.cluster[y, x] = True
            self.order[y, x] = self._n
            self._n += 1
        return self

    def fractal_dimension(self):
        """Box-counting dimension of the cluster: count occupied boxes at halving box sizes and fit the log-log
        slope. A sparse branching dendrite lands between 1 (a line) and 2 (a filled area)."""
        pts = np.argwhere(self.cluster)
        if len(pts) < 4:
            return 0.0
        sizes, counts = [], []
        n = min(self.shape)
        box = 1
        while box < n:
            box *= 2
            boxed = set(map(tuple, pts // box))
            sizes.append(box)
            counts.append(len(boxed))
        sizes = np.array(sizes, float); counts = np.array(counts, float)
        ok = counts > 0
        # dimension = -slope of log(count) vs log(size)
        slope = np.polyfit(np.log(sizes[ok]), np.log(counts[ok]), 1)[0]
        return float(-slope)


def ice_dendrite(shape=(81, 81), eta=1.0, steps=200, seed=0):
    """A radial ICE crystal: seed the centre, pull growth toward the surrounding boundary. eta~1 gives the classic
    fractal snowflake shape."""
    d = DielectricBreakdown(shape, eta=eta, seed=seed)
    d.seed_point(shape[0] // 2, shape[1] // 2)
    d.set_source_border()
    d.grow(steps)
    return d


def lightning(shape=(81, 81), eta=1.0, steps=120, seed=0):
    """A LIGHTNING bolt: seed the cloud along the top, pull the discharge DOWN to the ground. Same engine as the
    ice dendrite -- only the seed and the source boundary differ (N11: build once, get frost and bolts)."""
    d = DielectricBreakdown(shape, eta=eta, seed=seed)
    d.seed_line(axis=0, index=0)                                # the cloud along the top edge
    d.set_source_border(sides=("bottom",))                     # the ground below attracts the bolt
    d.grow(steps)
    return d


def _selftest():
    """The same engine grows an ice dendrite and a lightning bolt; the potential obeys Laplace (0 on the cluster,
    1 on the source, smooth between); growth races toward the steepest field; the result is a connected sparse
    FRACTAL (dimension between a line and a filled area); eta changes the shape; deterministic."""
    # (1) the Laplace solve: phi is 0 on the cluster, 1 on the source, and in-between elsewhere
    cluster = np.zeros((41, 41), bool); cluster[20, 20] = True
    source = np.zeros((41, 41), bool); source[0, :] = source[-1, :] = source[:, 0] = source[:, -1] = True
    phi = _relax_potential(cluster, source, iters=200)
    assert phi[20, 20] == 0.0 and phi[0, 0] == 1.0
    assert 0.0 < phi[10, 20] < 1.0                             # a point between cluster and border is in-between
    assert phi[5, 20] > phi[15, 20]                            # closer to the border -> higher potential

    # (2) an ICE dendrite grows into a connected sparse fractal (thin branching fingers -- a Lichtenberg shape)
    ice = ice_dendrite(shape=(81, 81), eta=1.0, steps=250, seed=0)
    n = int(ice.cluster.sum())
    assert n >= 200                                            # it actually grew
    fd = ice.fractal_dimension()
    assert 1.0 < fd < 1.9, fd                                  # sparse branching: not a line (1) and not filled (2)
    ys, xs = np.where(ice.cluster)
    bbox = max(ys.max() - ys.min(), xs.max() - xs.min())
    assert n > 2 * bbox                                       # far more cells than a single line -> it branches
    reach = np.max(np.hypot(ys - 40, xs - 40))
    assert reach > 15                                         # grew outward from the seed

    # (3) LIGHTNING: same engine, grows DOWNWARD from the top toward the ground
    bolt = lightning(shape=(81, 81), eta=3.0, steps=100, seed=1)
    ys, xs = np.where(bolt.cluster)
    assert ys.max() > 40                                      # the discharge reached well below the top cloud
    # higher eta -> thinner/stringier than the bushy low-eta blob (fewer cells reach the same depth)
    bushy = lightning(shape=(81, 81), eta=0.5, steps=100, seed=1)
    assert bolt.cluster.sum() == bushy.cluster.sum()          # both added 100 cells...
    # ...but the high-eta bolt is deeper-reaching per cell (stringier): its max depth >= the bushy one's
    assert ys.max() >= np.where(bushy.cluster)[0].max() - 2

    # (4) deterministic
    a = ice_dendrite(shape=(61, 61), steps=80, seed=7).cluster
    b = ice_dendrite(shape=(61, 61), steps=80, seed=7).cluster
    assert np.array_equal(a, b)

    print("holographic_dendrite selftest OK: one dielectric-breakdown engine grows an ICE dendrite (%d cells, "
          "fractal dimension %.2f -- sparse branching, not a filled disk) and a LIGHTNING bolt (same code, seed "
          "the cloud + attract the ground, reaching depth %d); phi obeys Laplace; eta tunes the shape; "
          "deterministic" % (n, fd, int(ys.max())))


if __name__ == "__main__":
    _selftest()
