"""Holographic radiance field (RAD): the scene's RADIANCE carried over all space as hypervectors (RENDER = QUERY).

THE PROGRESSION
---------------
holographic_fpefield carries the GEOMETRY (an SDF) as one hypervector. holographic_volint carries the volumetric
DENSITY and integrates it along a ray in closed form. This module carries the third thing a render needs -- the
RADIANCE (the colour leaving each point) -- as a field over all of space, so that the whole render becomes a QUERY of
hypervectors rather than a march through geometry. Geometry, density, radiance: all FPE fields, all defined at every
point including empty space.

THE REPRESENTATION (Nadaraya-Watson kernel regression, holographically)
-----------------------------------------------------------------------
Bake the radiance: at a set of points p_i (surface samples from a render) with colours c_i, build four FPE bundles --
one per colour channel weighted by the channel value, and one "coverage" field weighted by 1:

    F_r = sum_i c_i^r encode(p_i)   F_g, F_b likewise   F_w = sum_i encode(p_i)

Querying any point x reads kernel-weighted sums (the FPE Bochner/RBF kernel): <F_r, encode(x)> ~ sum_i c_i^r k(x,p_i)
and <F_w, encode(x)> ~ sum_i k(x,p_i). Their RATIO is the kernel-weighted average colour at x --

    radiance(x) = <F_r, encode(x)> / <F_w, encode(x)>            (Nadaraya-Watson estimator)

-- and because the ratio divides out the bundle's norm and FFT constant, NO calibration is needed (unlike the density
integral). Where coverage <F_w, encode(x)> ~ 0 there are no samples nearby: that point is EMPTY SPACE, known directly
from the field, not discovered by a ray. The query is vectorised in the Fourier domain (one matmul, chunked) using the
same Theta = scale*phases basis as the density integral, so reconstructing a whole frame is a couple of matmuls.

WHAT THIS IS AND ISN'T (kept negatives, loud)
---------------------------------------------
  * The field STORES radiance; it does not COMPUTE the lighting. A solver (the path tracer / shaded render) bakes the
    colours first; the field then makes free-viewpoint QUERY fast -- exactly how light fields, precomputed radiance
    transfer, and NeRF-style baking work. "Real-time" is the query/playback, not the bake.
  * Radiance baked from rendered pixels is VIEW-DEPENDENT data captured from one view: it reproduces diffuse shading
    and the highlights AS SEEN, but it is a view-independent (diffuse-ish) approximation. Full view dependence is the
    5-D plenoptic field -- add two direction axes to the encoder and bake more samples (the documented next step).
  * It is an RBF kernel regression, so it is smooth: hard radiance edges band-limit (denser samples / smaller
    bandwidth trade that off, the usual KDE bias/variance).

Basis: Frady/Kleyko/Sommer VFA; Komer/Eliasmith SSP; Nadaraya-Watson kernel regression; Levoy & Hanrahan light fields.
NumPy/stdlib only, deterministic.
"""
import numpy as np


class HolographicRadianceField:
    """Scene radiance as FPE hypervectors. `query(points)` returns (rgb, coverage): the kernel-weighted colour at each
    point, and the coverage (near 0 = empty space, known from the field). Built from baked surface samples."""

    def __init__(self, encoder, points, rgb, chunk=4096):
        self.enc = encoder
        P = np.atleast_2d(np.asarray(points, float))
        rgb = np.atleast_2d(np.asarray(rgb, float))
        self.Theta = np.stack([ax.scale * ax.phases for ax in encoder.axes], axis=0)     # (n_dims, dim)
        # Build the four NW bundles spectrally as a CHUNKED MATMUL -- no per-point bundle loop, no ifft, bounded
        # memory. specs[:, c] = sum_i rgb_i^c * exp(i p_i.Theta) ; the 4th channel (weight 1) is the coverage field.
        # (The FFT/norm constants drop out in the query's num/den ratio, so we never need them -- see query.)
        rgb4 = np.concatenate([rgb, np.ones((len(P), 1))], axis=1)                        # (N, 4)
        self.specs = np.zeros((self.Theta.shape[1], 4), dtype=complex)                    # (dim, 4)
        for s in range(0, len(P), chunk):
            e = min(len(P), s + chunk)
            cs = np.exp(1j * (P[s:e] @ self.Theta))                                       # (c, dim)
            self.specs += cs.T @ rgb4[s:e]                                                # accumulate (dim, 4)

    def query(self, points, chunk=4096, eps=1e-9):
        """Vectorised radiance query: rgb[r] = (num_r, num_g, num_b)/coverage via one matmul per chunk. Returns
        (rgb (R,3) clipped to [0,1], coverage (R,)). coverage ~ 0 marks empty space (no nearby samples)."""
        P = np.atleast_2d(np.asarray(points, float)); R = len(P)
        rgb = np.zeros((R, 3)); cov = np.zeros(R)
        for s in range(0, R, chunk):
            e = min(R, s + chunk)
            # code spectrum for these points: exp(-i * P.Theta) ; query_c = Re( M @ F_spec_c )
            M = np.exp(-1j * (P[s:e] @ self.Theta))                  # (c, dim) complex
            vals = np.real(M @ self.specs)                           # (c, 4): num_r, num_g, num_b, coverage
            den = vals[:, 3]
            safe = np.where(np.abs(den) < eps, 1.0, den)
            rgb[s:e] = vals[:, :3] / safe[:, None]
            cov[s:e] = den
        return np.clip(rgb, 0.0, 1.0), cov


def reconstruct_view(rad_field, hit_points, hit_mask, width, height, background=(0.0, 0.0, 0.0), cov_floor=None):
    """Render a frame purely by QUERYING the radiance field at each pixel's surface hit point. `hit_points` (H*W,3)
    are the 3-D points the camera rays hit (from a depth buffer: eye + depth*dir); `hit_mask` (H*W,) marks which rays
    hit a surface (the rest are background / empty space). Returns (H,W,3). This is "render = query the field"."""
    hit_points = np.atleast_2d(np.asarray(hit_points, float))
    rgb, cov = rad_field.query(hit_points)
    out = np.tile(np.asarray(background, float), (len(hit_points), 1))
    m = np.asarray(hit_mask, bool).reshape(-1)
    if cov_floor is not None:                                        # also drop near-empty queries to background
        m = m & (cov > cov_floor)
    out[m] = rgb[m]
    return np.clip(out.reshape(height, width, 3), 0.0, 1.0)


class TiledRadianceField:
    """Radiance over space TILED into a deterministic grid of bricks -- the engine's answer to the single-vector
    capacity wall (the same move HoloOctree makes for occupancy). Space is partitioned into grid^3 cells; each
    OCCUPIED cell carries its own small HolographicRadianceField built from the samples in that cell plus a one-cell
    halo (so queries near a border still see samples on both sides). Only occupied cells are stored (a sparse dict --
    no giant dense structure is ever built), and a query routes each point to its cell deterministically and reads
    just that brick. Capacity becomes (per-brick capacity) x (number of bricks): refine the grid and the wall moves.

    Because the bricks are independent, a change to one region rebuilds ONLY the affected cells -- an O(change) delta,
    not a global rebuild. `rebuild_cells(points, rgb, cells)` does exactly that."""

    def __init__(self, bounds, grid=8, dim=1024, bandwidth=20.0, halo=1, seed=0):
        from holographic_fpe import VectorFunctionEncoder
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.lo = np.array([b[0] for b in self.bounds]); self.hi = np.array([b[1] for b in self.bounds])
        self.grid = int(grid)
        self.cell = np.where((self.hi - self.lo) > 0, (self.hi - self.lo) / self.grid, 1.0)
        self.dim = int(dim); self.bandwidth = float(bandwidth); self.halo = int(halo); self.seed = int(seed)
        self.cells = {}                                        # sparse: only occupied cells -> HolographicRadianceField
        # ONE shared encoder for all bricks (a sharp kernel so points inside a cell stay distinct); the capacity win
        # comes from each brick bundling only its OWN (few) samples, not from per-cell encoders.
        self.enc = VectorFunctionEncoder(len(self.lo), dim=self.dim, bounds=self.bounds,
                                         kernel="rbf", bandwidth=self.bandwidth, seed=self.seed)

    def _cell_of(self, P):
        """Deterministic point(s) -> integer cell coords, clamped to the grid (the 'mark a location in space' index)."""
        return np.clip(((np.atleast_2d(P) - self.lo) / self.cell).astype(int), 0, self.grid - 1)

    def bake(self, points, rgb):
        """Build a per-cell radiance field for every occupied cell, each from that cell's samples plus a halo."""
        P = np.atleast_2d(np.asarray(points, float)); rgb = np.atleast_2d(np.asarray(rgb, float))
        ci = self._cell_of(P)
        for c in set(map(tuple, ci)):
            inhalo = np.all(np.abs(ci - np.array(c)) <= self.halo, axis=1)   # this cell + halo ring of samples
            self.cells[c] = HolographicRadianceField(self.enc, P[inhalo], rgb[inhalo])
        return self

    def rebuild_cells(self, points, rgb, cells):
        """Delta update: rebuild only the named cells from the (new) samples -- O(changed region), not a global redo."""
        P = np.atleast_2d(np.asarray(points, float)); rgb = np.atleast_2d(np.asarray(rgb, float))
        ci = self._cell_of(P)
        for c in cells:
            inhalo = np.all(np.abs(ci - np.array(c)) <= self.halo, axis=1)
            if inhalo.any():
                self.cells[tuple(c)] = HolographicRadianceField(self.enc, P[inhalo], rgb[inhalo])
            else:
                self.cells.pop(tuple(c), None)
        return self

    def query(self, points, chunk=4096):
        """Route each point to its cell and read that brick. Points in empty cells return coverage 0 (known empty)."""
        P = np.atleast_2d(np.asarray(points, float)); R = len(P)
        ci = self._cell_of(P)
        rgb = np.zeros((R, 3)); cov = np.zeros(R)
        # group query points by cell so each brick is queried once, vectorised
        lin = ci[:, 0] * self.grid * self.grid + ci[:, 1] * self.grid + ci[:, 2]
        for c in np.unique(lin):
            cc = (int(c) // (self.grid * self.grid), (int(c) // self.grid) % self.grid, int(c) % self.grid)
            field = self.cells.get(cc)
            if field is None:
                continue
            m = lin == c
            rgb[m], cov[m] = field.query(P[m], chunk=chunk)
        return np.clip(rgb, 0.0, 1.0), cov

    def n_bricks(self):
        return len(self.cells)


def _selftest():
    """Baked radiance must be reconstructable by query, and empty space must read ~0 coverage."""
    from holographic_fpe import VectorFunctionEncoder
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, (60, 3))
    cols = np.clip(0.5 + 0.5 * np.sin(pts * 3.0), 0, 1)             # a smooth colour function over space
    enc = VectorFunctionEncoder(3, dim=1024, bounds=[(-1.5, 1.5)] * 3, kernel="rbf", bandwidth=18.0, seed=0)
    rad = HolographicRadianceField(enc, pts, cols)
    rgb, cov = rad.query(pts)
    err = np.abs(rgb - cols).mean()
    assert err < 0.09, err                                          # the baked colours come back (sharp kernel)
    # empty space far from every sample -> ~0 coverage (known empty)
    _, cov_far = rad.query(np.array([[5.0, 5.0, 5.0]]))
    assert abs(cov_far[0]) < abs(cov).mean() * 0.2
    print("radiance selftest ok: bake->query mean abs err=%.4f ; empty coverage=%.4f vs occupied mean=%.4f"
          % (err, float(cov_far[0]), float(np.abs(cov).mean())))


if __name__ == "__main__":
    _selftest()
