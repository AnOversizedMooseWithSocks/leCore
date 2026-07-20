"""Terrain (G4): a holographic fBm heightfield, liftable to a displaced-grid mesh or a heightfield SDF.

WHY THIS MODULE EXISTS
----------------------
Terrain is the canonical use of the noise keystone: a 2-D fractal heightfield, exposed as geometry.
This module is mostly a COMPOSITION of G1 (FractalNoise gives the height) and the displacement idea
(lift a flat grid by that height) -- which is exactly why it is cheap to build once those exist.

WHAT IT GIVES
-------------
  * `Terrain` wraps a 2-D FractalNoise: `height(xy)` is one fBm query; `heightmap(res)` samples a grid.
  * `terrain_to_mesh` lifts a flat res x res grid into a triangulated surface with z = height(x,y) and
    UVs set to the normalized grid -- ready to texture with a G2 material and shade with G3 normals.
  * `terrain_to_sdf` builds a HolographicField from the heightfield sign function so the terrain can be
    marched at any LOD and unioned/edited with the rest of the field algebra.

The terrain is a FIELD, so level-of-detail is just re-sampling at the resolution you need, and it
composes with materials, displacement, and unions in-space.

HONEST SCOPE (kept negatives)
-----------------------------
  * NO EROSION. This is pure fBm -- it has the right roughness statistics but not the drainage networks,
    ridges, or talus of hydraulic/thermal erosion. Erosion is an iterative simulation, a separate item;
    it is deliberately NOT folded in here.
  * The heightfield SDF uses sdf(x,y,z) = z - height(x,y): sign-correct and unit-gradient in z (so it
    marches cleanly), but NOT the true Euclidean distance to the surface where the terrain is steep --
    the standard heightfield-SDF approximation, kept on the record.
  * Band-limited (it inherits G1's smooth-spectrum nature); the finest detail is set by the top octave.
"""

import numpy as np

from holographic.sampling_and_signal.holographic_noise import FractalNoise
from holographic.mesh_and_geometry.holographic_mesh import Mesh


class Terrain:
    """A 2-D fractal heightfield over `bounds` = [(x0,x1),(y0,y1)], backed by a FractalNoise."""

    def __init__(self, bounds=None, octaves=5, lacunarity=2.0, gain=0.5,
                 base_bandwidth=2.0, dim=1024, seed=0):
        if bounds is None:
            bounds = [(0.0, 1.0), (0.0, 1.0)]
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.fbm = FractalNoise(2, dim=dim, bounds=self.bounds, octaves=octaves,
                                lacunarity=lacunarity, gain=gain, base_bandwidth=base_bandwidth, seed=seed)

    def height(self, xy):
        """The terrain height at a single (x, y)."""
        return float(self.fbm.query(xy))

    def heightmap(self, res):
        """A res x res array of heights over `bounds` (for measuring or rasterizing)."""
        return self.fbm.sample_grid(res)


def terrain_to_mesh(terrain, res, z_scale=1.0):
    """Lift the heightfield into a triangulated res x res grid mesh (z = z_scale * height(x, y)).

    Vertices are laid out row-major; each grid cell becomes two triangles. UVs are the normalized grid
    coordinates so a G2 material maps straight on. Normals are computed for the lifted surface.
    """
    (x0, x1), (y0, y1) = terrain.bounds
    xs = np.linspace(x0, x1, res)
    ys = np.linspace(y0, y1, res)
    verts = []
    uvs = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            verts.append([x, y, z_scale * terrain.height([x, y])])
            uvs.append([i / (res - 1), j / (res - 1)])
    faces = []
    for i in range(res - 1):
        for j in range(res - 1):
            a = i * res + j
            b = (i + 1) * res + j
            c = (i + 1) * res + (j + 1)
            d = i * res + (j + 1)
            faces.append((a, b, c))         # two triangles per cell
            faces.append((a, c, d))
    mesh = Mesh(np.array(verts), faces, uvs=np.array(uvs))
    mesh.vertex_normals(store=True)
    return mesh


def terrain_to_sdf(terrain, z_bounds, res=10, dim=2048, bandwidth=10.0, seed=0):
    """Build a HolographicField for the terrain via the heightfield sign function sdf = z - height(x,y).

    Samples that sdf on a res^3 lattice over (terrain bounds, z_bounds) and bundles it. Marchable with
    the field's own `surface()`; unionable/editable with the rest of the field algebra.
    """
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
    from holographic.sampling_and_signal.holographic_fpefield import HolographicField
    (x0, x1), (y0, y1) = terrain.bounds
    z0, z1 = z_bounds
    xs = np.linspace(x0, x1, res); ys = np.linspace(y0, y1, res); zs = np.linspace(z0, z1, res)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    H = np.array([terrain.height([p[0], p[1]]) for p in P])
    sdf = P[:, 2] - H                                  # z - height: + above the terrain, - below
    enc = VectorFunctionEncoder(3, dim=dim, bounds=[(x0, x1), (y0, y1), (z0, z1)],
                                bandwidth=bandwidth, seed=seed)
    return HolographicField(enc, P, sdf)


# ---------------------------------------------------------------------------

def _selftest():
    from holographic.misc.holographic_fractal import box_counting_dimension

    # (1) DETERMINISM: same seed -> identical heightmap.
    t = Terrain(bounds=[(0, 4), (0, 4)], octaves=4, gain=0.5, base_bandwidth=2.0, dim=512, seed=7)
    hm1 = t.heightmap(20)
    t2 = Terrain(bounds=[(0, 4), (0, 4)], octaves=4, gain=0.5, base_bandwidth=2.0, dim=512, seed=7)
    assert np.allclose(hm1, t2.heightmap(20)), "terrain not deterministic for a fixed seed"

    # (2) ROUGHNESS tracks persistence. Measure normalized gradient roughness of the heightmap (robust)
    #     AND the box-counting dimension of a height transect (connects to the shipped measurer).
    def rough(gain):
        tt = Terrain(bounds=[(0, 4), (0, 4)], octaves=4, gain=gain, base_bandwidth=2.5, dim=512, seed=3)
        hm = tt.heightmap(40)
        g = np.abs(np.diff(hm, axis=0)).mean() + np.abs(np.diff(hm, axis=1)).mean()
        return g / (hm.std() + 1e-9)
    r_lo, r_hi = rough(0.30), rough(0.85)
    assert r_hi > r_lo, f"higher persistence should be rougher terrain: {r_hi:.3f} !> {r_lo:.3f}"

    tt = Terrain(bounds=[(0, 4), (0, 4)], octaves=4, gain=0.8, base_bandwidth=2.5, dim=512, seed=3)
    xs = np.linspace(0, 4, 200)
    transect = np.stack([xs, np.array([tt.height([x, 2.0]) for x in xs])], axis=1)
    bcd = box_counting_dimension(transect)
    # A BAND-LIMITED fBm transect is a rectifiable curve, so its box-counting dimension reads ~1 (NOT a
    # true 1<D<2 fractal) -- exactly the smooth-spectrum scope kept from G1. We assert it lands in the
    # curve regime, which confirms the shipped measurer applies and the kept negative holds.
    assert 0.8 <= bcd <= 1.6, f"transect should read as a near-curve (band-limited): {bcd:.3f}"

    # (3) terrain_to_mesh: res^2 vertices, z matches height exactly at grid points, faces well-formed.
    mesh = terrain_to_mesh(t, 12)
    assert mesh.n_vertices == 12 * 12, f"expected 144 vertices, got {mesh.n_vertices}"
    assert len(mesh.faces) == 2 * 11 * 11, f"expected {2*11*11} triangles, got {len(mesh.faces)}"
    # spot-check a vertex height equals the terrain height there
    v = mesh.vertices[5 * 12 + 5]
    assert abs(v[2] - t.height([v[0], v[1]])) < 1e-9, "mesh vertex z does not match terrain height"

    # (4) terrain_to_sdf: the field sign flips across the terrain surface (negative below, positive above).
    fld = terrain_to_sdf(t, z_bounds=(-2, 2), res=8, dim=1024, bandwidth=8.0, seed=1)
    xy = [2.0, 2.0]; h = t.height(xy)
    below = float(fld.value([[xy[0], xy[1], h - 1.0]])[0])
    above = float(fld.value([[xy[0], xy[1], h + 1.0]])[0])
    assert below < 0 < above, f"heightfield SDF sign wrong: below={below:.3f} above={above:.3f}"

    print("holographic_terrain selftest passed:",
          f"rough(0.30)={r_lo:.3f} rough(0.85)={r_hi:.3f} transect_dim={bcd:.3f} "
          f"sdf below/above={below:.2f}/{above:.2f}")


if __name__ == "__main__":
    _selftest()


def erode(height, droplets=2000, steps=30, inertia=0.05, capacity=1.0, deposition=0.3,
          erosion=0.3, evaporation=0.02, min_slope=0.01, radius=2, seed=0):
    """HYDRAULIC EROSION of a height grid: droplet simulation that carves drainage channels and softens peaks.

    Each seeded droplet walks downhill (gradient descent with `inertia` momentum so channels meander rather
    than zigzag), picks up sediment proportional to speed*water up to `capacity`, deposits when over capacity
    or on uphill motion, and evaporates. Erosion is brushed over a `radius` neighbourhood so channels have
    width instead of single-cell scratches (the classic droplet model, e.g. Beyer 2015 / common in procedural
    terrain literature). Purely additive: takes and returns a plain (H, W) float array; the fBm `Terrain`
    class is untouched. Deterministic under `seed` (seeded default_rng; no Python hash anywhere).

    HEIGHT UNITS: scale-invariant. The grid is normalised to [0,1] internally, eroded, and rescaled to the
    caller's units, so the result is independent of the input's height scale -- a normalised height grid and the
    same terrain in metres erode identically (previously capacity=4.0 caused a deposition runaway on any
    off-unit-scale grid: a 1.0 peak rose to 11.77, a 20-unit terrain to 74,306). `capacity` defaults to 1.0
    (stable and still carving); 4.0 grew peaks even on the module's own fBm terrain, which is why the default
    changed. Lower capacity = gentler carving, higher = more aggressive; above ~2.0 risks the old feedback.

    Returns the eroded copy. Conservation note: sediment leaving the grid edge with a dying droplet is lost --
    total material is NOT exactly conserved, matching real drainage out of the tile.
    """
    # SCALE INVARIANCE (fixes the reported runaway). The droplet dynamics are NOT height-unit-agnostic: `speed`
    # updates via dh*gravity and `cap` scales with dh, so a 20-unit terrain gets 20x the sediment per step and the
    # deposition feedback explodes (client measured a 1.0 peak -> 11.77, a 20-unit terrain -> 74,306). The physics
    # only behaves in the unit-height regime the parameters were tuned for. So normalise the grid to [0,1], run
    # there, and rescale back -- ANY input then behaves like the well-tested unit case. Additive: a grid already
    # in [0,1] is unchanged to floating point (span==1, offset==0), so no existing unit-scale result moves; only
    # the previously-corrupt off-unit-scale results change, which is the fix, not a regression.
    H_in = np.asarray(height, float)
    _lo = float(H_in.min())
    _span = float(H_in.max()) - _lo
    if _span < 1e-12:
        return H_in.astype(float).copy()                      # flat grid: nothing to erode, and 1/span would blow up
    H = (np.array(height, float, copy=True) - _lo) / _span     # work in [0,1], the regime the params were tuned for
    h, w = H.shape
    rng = np.random.default_rng(seed)

    # precompute the erosion brush: gaussian-ish weights over the radius disc, normalised
    ys, xs = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    dist = np.sqrt(xs * xs + ys * ys)
    brush = np.maximum(0.0, 1.0 - dist / max(radius, 1e-9))
    brush /= brush.sum()

    def grad(px, py):
        # bilinear height + gradient at a continuous position (cell-local finite differences)
        xi, yi = int(px), int(py)
        fx, fy = px - xi, py - yi
        h00, h10 = H[yi, xi], H[yi, xi + 1]
        h01, h11 = H[yi + 1, xi], H[yi + 1, xi + 1]
        gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
        gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
        hh = h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) + h01 * (1 - fx) * fy + h11 * fx * fy
        return hh, gx, gy

    starts = rng.uniform([1.0, 1.0], [w - 2.0, h - 2.0], size=(droplets, 2))
    for (px, py) in starts:
        dx = dy = 0.0
        speed, water, sediment = 1.0, 1.0, 0.0
        for _ in range(steps):
            h0, gx, gy = grad(px, py)
            # momentum blend: pure gradient at inertia=0, straight-line at inertia=1
            dx = dx * inertia - gx * (1 - inertia)
            dy = dy * inertia - gy * (1 - inertia)
            n = np.hypot(dx, dy)
            if n < 1e-12:
                break
            dx, dy = dx / n, dy / n
            nx, ny = px + dx, py + dy
            if not (1.0 <= nx < w - 2 and 1.0 <= ny < h - 2):
                break                                        # droplet leaves the tile; its sediment leaves too
            h1, _, _ = grad(nx, ny)
            dh = h1 - h0
            cap = max(-dh, min_slope) * speed * water * capacity
            if sediment > cap or dh > 0:
                # deposit: over capacity, or moving uphill (fill the pit it just climbed out of)
                amt = min(dh, sediment) if dh > 0 else (sediment - cap) * deposition
                sediment -= amt
                H[int(py), int(px)] += amt
            else:
                # erode, brushed over the disc so channels get width
                amt = min((cap - sediment) * erosion, -dh)
                cy, cx = int(py), int(px)
                y0, y1 = cy - radius, cy + radius + 1
                x0, x1 = cx - radius, cx + radius + 1
                if 0 <= y0 and y1 <= h and 0 <= x0 and x1 <= w:
                    H[y0:y1, x0:x1] -= amt * brush
                else:
                    H[cy, cx] -= amt
                sediment += amt
            speed = float(np.sqrt(max(speed * speed + dh * -9.81 * 0.1, 0.0)))
            water *= (1.0 - evaporation)
            px, py = nx, ny
    return H * _span + _lo                                     # rescale back to the caller's height units


def _selftest_erode():
    """Pin: deterministic, peaks lowered, real material moved, no NaN, additive (input untouched)."""
    t = Terrain(seed=3, octaves=4)
    H0 = t.heightmap(48)
    H1 = erode(H0, droplets=400, steps=25, seed=0)
    H2 = erode(H0, droplets=400, steps=25, seed=0)
    assert np.array_equal(H1, H2), "erosion must be deterministic under a seed"
    assert np.array_equal(H0, t.heightmap(48)), "input grid must not be mutated (additive contract)"
    assert np.isfinite(H1).all()
    assert H1.max() <= H0.max() + 1e-12, "peaks must not GROW under erosion"

    # THE RUNAWAY REGRESSION TRAP (client-reported, and the old selftest missed it because 400 droplets on a
    # gentle fBm never triggered the deposition feedback). Two conditions that USED to explode with capacity=4.0:
    # a clean unit-height gaussian at 3000 droplets, and the SAME shape at 20x. Both must stay bounded now that
    # the default is 1.0 and the run is height-normalised. Peaks GREW to 2.79 and 1537 respectively before the fix.
    import numpy as _np
    _n = 48
    _xx, _yy = _np.meshgrid(_np.linspace(-3, 3, _n), _np.linspace(-3, 3, _n))
    _peak = _np.exp(-(_xx * _xx + _yy * _yy) / 2.0)                      # max ~1.0
    for _scale in (1.0, 20.0):
        _e = erode(_peak * _scale, droplets=3000, seed=0)
        assert _e.max() <= _peak.max() * _scale + 1e-6, (
            "erode RUNAWAY at scale %g: peak %g grew to %g -- the height-scale feedback is back"
            % (_scale, _peak.max() * _scale, _e.max()))
    # SCALE INVARIANCE, pinned: eroding H and eroding 20*H must agree after rescaling (same normalised run).
    _a = erode(_peak, droplets=800, seed=1)
    _b = erode(_peak * 20.0, droplets=800, seed=1) / 20.0
    assert _np.allclose(_a, _b, atol=1e-9), "erode must be height-scale invariant (normalise-run-rescale)"
    moved = float(np.abs(H1 - H0).sum())
    assert moved > 1e-3, "erosion moved no material -- simulation is dead"
    # a different seed carves different channels (the droplets, not the terrain, are the randomness)
    H3 = erode(H0, droplets=400, steps=25, seed=1)
    assert not np.array_equal(H1, H3)
    print("terrain erode selftest OK (deterministic, peaks<=, |moved|=%.4f)" % moved)


if __name__ == "__main__":
    _selftest_erode()
