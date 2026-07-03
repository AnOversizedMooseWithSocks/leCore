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

from holographic_noise import FractalNoise
from holographic_mesh import Mesh


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

    def heights(self, xy_points):
        """Terrain heights at many (x, y) points via batched fBm reads."""
        return self.fbm.query_many(xy_points)

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
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
    heights = terrain.heightmap(res).ravel()
    verts = np.column_stack([xy[:, 0], xy[:, 1], z_scale * heights])
    ui, vj = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res), indexing="ij")
    uvs = np.stack([ui.ravel(), vj.ravel()], axis=1)
    faces = []
    for i in range(res - 1):
        for j in range(res - 1):
            a = i * res + j
            b = (i + 1) * res + j
            c = (i + 1) * res + (j + 1)
            d = i * res + (j + 1)
            faces.append((a, b, c))         # two triangles per cell
            faces.append((a, c, d))
    mesh = Mesh(verts, faces, uvs=uvs)
    mesh.vertex_normals(store=True)
    return mesh


def terrain_to_sdf(terrain, z_bounds, res=10, dim=2048, bandwidth=10.0, seed=0):
    """Build a HolographicField for the terrain via the heightfield sign function sdf = z - height(x,y).

    Samples that sdf on a res^3 lattice over (terrain bounds, z_bounds) and bundles it. Marchable with
    the field's own `surface()`; unionable/editable with the rest of the field algebra.
    """
    from holographic_fpe import VectorFunctionEncoder
    from holographic_fpefield import HolographicField
    (x0, x1), (y0, y1) = terrain.bounds
    z0, z1 = z_bounds
    xs = np.linspace(x0, x1, res); ys = np.linspace(y0, y1, res); zs = np.linspace(z0, z1, res)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    H = np.repeat(terrain.heightmap(res).ravel(), res)
    sdf = P[:, 2] - H                                  # z - height: + above the terrain, - below
    enc = VectorFunctionEncoder(3, dim=dim, bounds=[(x0, x1), (y0, y1), (z0, z1)],
                                bandwidth=bandwidth, seed=seed)
    return HolographicField(enc, P, sdf)


# ---------------------------------------------------------------------------

def _selftest():
    from holographic_fractal import box_counting_dimension

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
