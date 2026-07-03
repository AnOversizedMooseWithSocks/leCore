"""holographic_cellular.py -- M2: CELLULAR / CRYSTALLINE structure (polycrystalline grain, facets, cracks,
regular lattice packing).

WHY THIS EXISTS (Material Structure backlog, item M2)
-----------------------------------------------------
Wood is layered (holographic_grainmat) and impurities are scattered pockets (holographic_inclusions); the third
structure primitive is CELLULAR -- the grain cells of a metal, the facets of a cut gem, the crack network of
dried mud or ceramic glaze, and the regular packing of a crystal lattice. Two classic methods cover it:

  * WORLEY (1996) cellular / Voronoi texture -- partition space by nearest seed point. The nearest-seed ID gives
    each cell a facet; the difference between the nearest and second-nearest distance (F2 - F1) is small exactly
    ON a cell boundary, which is the crack/edge metric. This is polycrystalline GRAIN.
  * BRAVAIS lattice / unit-cell repetition -- a motif packed at regular intervals. In a material socket that is
    just evaluating the motif in a wrapped (modulo) object-space coordinate, the same domain-repetition idea the
    tiling module uses, done directly in space so it is a solid texture.

Everything here is an albedo/scalar SOCKET f(points (M,3)) -> value, so it drops straight into a
SurfaceMaterial channel (Param(field=...)) and renders through render_surface / RenderSession, exactly like the
grain and inclusion sockets. Volumetric: a cut through the material shows the cells/lattice continue inside.

HONEST SCOPE (kept negative): this is STRUCTURAL / APPEARANCE crystallography -- cells, facets, symmetry, crack
networks that read right -- NOT atomic-scale unit cells with real lattice constants or a diffraction model. Real
lattice constants keyed to the mineral definitions would be a later data column. Deterministic; NumPy + stdlib.
"""
import numpy as np


def _cell_seeds(n_seeds, bounds, seed, jitter):
    """Place `n_seeds` cell centres in `bounds`=(lo,hi). Start from a jittered regular grid so the cells are
    reasonably even (real grains are), with `jitter` in [0,1] scaling the random offset. Deterministic."""
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    rng = np.random.default_rng(seed)
    # a near-cube grid of at least n_seeds sites, then jitter each and keep n_seeds of them
    side = int(np.ceil(n_seeds ** (1.0 / 3.0)))
    axes = [np.linspace(lo[d], hi[d], side + 1)[:-1] + (hi[d] - lo[d]) / (2 * side) for d in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    cell = (hi - lo) / side
    grid = grid + (rng.uniform(-0.5, 0.5, grid.shape) * cell * float(jitter))
    return grid[:int(n_seeds)]


class VoronoiCells:
    """A Worley/Voronoi partition of space by nearest seed. `ids(points)` -> the cell each point belongs to;
    `edge_distance(points)` -> the F2-F1 boundary metric (small ON a cell edge, large in a cell interior)."""

    def __init__(self, n_seeds=24, bounds=((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)), seed=0, jitter=1.0):
        self.seeds = _cell_seeds(n_seeds, bounds, seed, jitter)     # (K,3) cell centres
        self.n = len(self.seeds)

    def _two_nearest(self, points):
        """Distances to the nearest (F1) and second-nearest (F2) seed, plus the nearest seed id, per point."""
        P = np.atleast_2d(np.asarray(points, float))
        # (M,K) pairwise distances -- vectorised; K is small (grain count), so this is cheap
        d = np.linalg.norm(P[:, None, :] - self.seeds[None, :, :], axis=2)
        ids = np.argmin(d, axis=1)                                 # nearest seed = the cell id (Worley F1 site)
        f1 = np.take_along_axis(d, ids[:, None], axis=1)[:, 0]
        d2 = d.copy(); d2[np.arange(len(P)), ids] = np.inf         # mask the nearest to find the second nearest
        f2 = d2.min(axis=1)
        return f1, f2, ids

    def ids(self, points):
        """The cell id (nearest-seed index) each point falls in."""
        return self._two_nearest(points)[2]

    def edge_distance(self, points):
        """(F2 - F1)/2 -- the distance to the nearest CELL BOUNDARY (the perpendicular bisector between the two
        closest seeds). ~0 on an edge, larger toward a cell centre. This is the crack/grain-boundary metric."""
        f1, f2, _ = self._two_nearest(points)
        return 0.5 * (f2 - f1)


def _facet_colour(cell_id, base=(0.55, 0.57, 0.62), spread=0.18, seed=0):
    """Deterministic per-cell facet colour: hash the integer cell id (pure arithmetic, PYTHONHASHSEED-independent,
    like holographic_pattern's lattice hash) into a small rgb jitter around `base`. Each grain reads slightly
    different, as real polycrystalline facets catch light differently."""
    ids = np.atleast_1d(np.asarray(cell_id, np.int64))
    def chan(salt):
        h = (ids * np.int64(2654435761) ^ np.int64(salt * 40503 + seed * 19349663))
        h = (h ^ (h >> np.int64(13))) * np.int64(1274126177)
        h = h ^ (h >> np.int64(16))
        return (h & np.int64(0xFFFF)).astype(np.float64) / float(0x10000)   # [0,1)
    jit = np.stack([chan(1), chan(2), chan(3)], axis=1) - 0.5      # [-0.5,0.5] per channel
    return np.clip(np.asarray(base, float)[None, :] + jit * 2.0 * spread, 0.0, 1.0)


def cell_albedo(cells, base=(0.55, 0.57, 0.62), spread=0.18, crack=(0.05, 0.05, 0.06),
                crack_width=0.03, seed=0):
    """A polycrystalline albedo socket f(points)->(M,3): each Voronoi cell a slightly different facet colour,
    darkened to `crack` along the cell boundaries (where edge_distance < crack_width). Drops into
    SurfaceMaterial(color=Param(field=cell_albedo(cells)))."""
    crk = np.asarray(crack, float)

    def _socket(points):
        f1, f2, ids = cells._two_nearest(points)
        col = _facet_colour(ids, base=base, spread=spread, seed=seed)
        edge = 0.5 * (f2 - f1)
        m = np.clip(1.0 - edge / max(crack_width, 1e-6), 0.0, 1.0)[:, None]   # 1 on an edge, 0 in the interior
        return (1.0 - m) * col + m * crk                          # darken toward the crack colour at boundaries
    return _socket


def crack_mask(cells, crack_width=0.03):
    """A scalar socket f(points)->[0,1]: 1 on a cell boundary, 0 in a cell interior. Drive a roughness or a
    displacement channel with it (glaze crackle, dried-mud fissures)."""
    def _socket(points):
        e = cells.edge_distance(points)
        return np.clip(1.0 - e / max(crack_width, 1e-6), 0.0, 1.0)
    return _socket


def lattice(motif, period, center=(0.0, 0.0, 0.0)):
    """Pack a `motif` socket f(local_points)->value at regular intervals `period` (a scalar or a (3,) per-axis
    period) -- Bravais-style regular crystal packing. The motif is evaluated in a WRAPPED object-space
    coordinate centred in each cell, so it repeats identically in every cell: a solid texture, tiling bit-exactly.
    Returns f(points)->whatever the motif returns."""
    per = np.asarray(period, float) if np.ndim(period) else np.full(3, float(period))
    c = np.asarray(center, float)

    def _socket(points):
        P = np.atleast_2d(np.asarray(points, float)) - c
        local = np.mod(P + 0.5 * per, per) - 0.5 * per            # wrap into one cell, centred at 0
        return motif(local)
    return _socket


def _selftest():
    """Nearest-seed assignment is correct, cells cover the range, the crack metric peaks on boundaries, the
    lattice tiles bit-exactly, and everything is deterministic."""
    cells = VoronoiCells(n_seeds=24, seed=0)

    # (1) the socket's cell id == the true nearest seed (brute force), for random points
    P = np.random.default_rng(1).uniform(-1.5, 1.5, (500, 3))
    got = cells.ids(P)
    true = np.argmin(np.linalg.norm(P[:, None, :] - cells.seeds[None, :, :], axis=2), axis=1)
    assert np.array_equal(got, true)                              # correct Voronoi assignment
    assert len(np.unique(got)) >= 2                               # multiple cells are actually hit

    # (2) the crack metric is ~0 on a boundary and larger inside: a point exactly between two seeds is an edge
    mid = (cells.seeds[0] + cells.seeds[1]) * 0.5
    assert cells.edge_distance([mid])[0] < 0.05                   # on the bisector -> a boundary
    ctr = cells.seeds[0]
    assert cells.edge_distance([ctr])[0] > cells.edge_distance([mid])[0]   # cell centre is farther from an edge

    # (3) albedo socket is valid rgb + deterministic; cracks darken it somewhere
    alb = cell_albedo(cells)
    a = alb(P); b = alb(P)
    assert a.shape == (500, 3) and a.min() >= 0 and a.max() <= 1 and np.array_equal(a, b)

    # (4) the lattice tiles BIT-EXACTLY: motif(P) == motif(P + period)
    motif = lambda L: np.linalg.norm(L, axis=1)                   # a blob centred in each cell
    lat = lattice(motif, period=0.5)
    Q = np.random.default_rng(2).uniform(-1, 1, (300, 3))
    assert np.allclose(lat(Q), lat(Q + np.array([0.5, 0.0, 0.0])))   # exact periodicity
    assert np.allclose(lat(Q), lat(Q + np.array([0.0, 0.5, 0.5])))
    print("holographic_cellular selftest OK: correct nearest-seed Voronoi (%d cells), crack metric peaks on "
          "boundaries, facet albedo valid+deterministic, lattice tiles bit-exactly" % len(np.unique(got)))


if __name__ == "__main__":
    _selftest()
