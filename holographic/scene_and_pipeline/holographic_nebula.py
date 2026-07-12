"""holographic_nebula.py -- a NEBULA: turbulent volumetric gas/dust you can render (leCore scene_and_pipeline).

WHY THIS EXISTS
---------------
The star cluster (star_cluster) gives the stars; a nebula gives the GAS and DUST they are born from and light up.
This builds a deterministic 3-D density volume with nebula-like structure -- wispy filaments and dark voids --
that plugs straight into the engine's volumetric renderer (render_volume). It is the emission/dark-nebula
counterpart to the cluster, and closes the 'simulate a region of a galaxy' picture: gas + stars together.

It ASSEMBLES BY DELEGATION -- it invents no noise it can borrow:
  * the turbulence is the engine's own FractalNoise sampled on a 3-D lattice (holographic_noise) -- the same fBm
    the rest of the engine uses, not a private copy;
  * rendering is render_volume (the field format here is exactly what it marches);
  * cavities are carved by STAR positions (a cluster's stars blow bubbles and ionise the gas) -- the tie to
    star_cluster, so a cluster and its nebula share structure instead of being unrelated pictures.

STRUCTURE (honest, procedural, not a hydro sim): fBm -> ridged transform (filaments) -> threshold/contrast
(dark voids between bright sheets) -> optional spherical cavities where stars sit. This is an ARTIST'S nebula,
a legible density field, not a solution of the equations of a real HII region -- stated so it is not mistaken
for one. A true fluid nebula would advect this with the Stam solver (fluid_step_3d); declared, not implied.

DIRECTIONS (up/down/sideways)
  DOWN  -- nebula_column projects the volume to a 2-D column-density image (a cheap look without ray-marching).
  UP    -- many nebulae + a cluster tile into a galaxy region; the cavities bind it to star_cluster above.
  SIDEWAYS
    field    -- the density volume is the native costume (feeds render_volume). structure -- the {volume, res,
    seed} record. sequence -- animating the seed / advecting with a fluid step walks it in time (declared).

Determinism: FractalNoise is deterministic; all shaping is pure numpy. Same seed -> identical volume.
"""

import numpy as np


def nebula_volume(res=48, seed=0, level=0.5, gain=3.0, ridged=True, star_positions=None,
                  cavity_radius=0.16, dim=256, octaves=5):
    """Build a 3-D nebula density volume (res, res, res) in [0,1]. `level` sets how much is empty void, `gain` the
    contrast of the filaments. With ridged=True the fBm is folded into thin high-density SHEETS (the wispy look);
    pass `star_positions` (list of (x,y,z) in [0,1]) to carve spherical CAVITIES of radius `cavity_radius` where
    stars blow bubbles in the gas -- the tie to star_cluster. Reuses the engine's FractalNoise. Feeds render_volume."""
    from holographic.sampling_and_signal.holographic_noise import FractalNoise
    res = int(res)
    fn = FractalNoise(n_dims=3, dim=int(dim), octaves=int(octaves), seed=int(seed))
    base = np.asarray(fn.sample_grid(res), float)
    b = (base - base.min()) / (base.max() - base.min() + 1e-12)        # -> [0,1]
    if ridged:
        b = 1.0 - np.abs(2.0 * b - 1.0)                                # ridged noise -> thin filaments/sheets
    d = np.clip((b - level) * gain, 0.0, 1.0)                          # threshold + contrast -> bright sheets, dark voids
    if star_positions is not None and len(star_positions):
        # normalized voxel coordinates in [0,1]
        ax = (np.arange(res) + 0.5) / res
        gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
        for s in star_positions:
            sx, sy, sz = (list(s) + [0.5, 0.5, 0.5])[:3]               # accept 2-D positions -> mid-plane
            r = np.sqrt((gx - sx) ** 2 + (gy - sy) ** 2 + (gz - sz) ** 2)
            d = d * np.clip(r / max(cavity_radius, 1e-6), 0.0, 1.0)    # 0 at the star, full at the cavity edge
    return d


def nebula_column(volume, axis=2):
    """Column density: sum the volume along one axis -> a 2-D image (what you'd see looking through the cloud). The
    cheap look at a nebula without a full ray-march; also a quick way to check structure in a test."""
    return np.sum(np.asarray(volume, float), axis=int(axis))


def nebula_field_fn(volume, bounds=None):
    """Wrap a nebula volume as the CALLABLE the engine's volume renderer wants: points (N,3) -> density (N,), by
    TRILINEAR interpolation of the voxel grid over `bounds` (default 0..res per axis). This is the exact `field`
    argument render_volume / holographic_render.volume_render marches -- so a nebula drops straight into the ray-
    marcher. The renderer owns the camera; this owns the density lookup. Delegates the drawing, reimplements none."""
    vol = np.asarray(volume, float)
    n0, n1, n2 = vol.shape[:3]
    # bounds match render_volume's convention: (lo_xyz, hi_xyz), two 3-vectors -- so the field callable and the
    # marcher agree on world space. Default spans one voxel per unit (lo=0, hi=n-1) so world coord == voxel index.
    if bounds is None:
        bounds = ((0.0, 0.0, 0.0), (float(n0 - 1), float(n1 - 1), float(n2 - 1)))
    lo = np.asarray(bounds[0], float); hi = np.asarray(bounds[1], float)

    def field(points):
        p = np.asarray(points, float).reshape(-1, 3)
        # world -> voxel coordinates (lo -> node 0, hi -> node n-1)
        gx = (p[:, 0] - lo[0]) / (hi[0] - lo[0]) * (n0 - 1)
        gy = (p[:, 1] - lo[1]) / (hi[1] - lo[1]) * (n1 - 1)
        gz = (p[:, 2] - lo[2]) / (hi[2] - lo[2]) * (n2 - 1)
        ix = np.clip(np.floor(gx).astype(int), 0, n0 - 1); iy = np.clip(np.floor(gy).astype(int), 0, n1 - 1); iz = np.clip(np.floor(gz).astype(int), 0, n2 - 1)
        jx = np.clip(ix + 1, 0, n0 - 1); jy = np.clip(iy + 1, 0, n1 - 1); jz = np.clip(iz + 1, 0, n2 - 1)
        fx = np.clip(gx - ix, 0, 1); fy = np.clip(gy - iy, 0, 1); fz = np.clip(gz - iz, 0, 1)
        # trilinear blend of the eight corner voxels
        c000 = vol[ix, iy, iz]; c100 = vol[jx, iy, iz]; c010 = vol[ix, jy, iz]; c001 = vol[ix, iy, jz]
        c110 = vol[jx, jy, iz]; c101 = vol[jx, iy, jz]; c011 = vol[ix, jy, jz]; c111 = vol[jx, jy, jz]
        c00 = c000 * (1 - fx) + c100 * fx; c01 = c001 * (1 - fx) + c101 * fx
        c10 = c010 * (1 - fx) + c110 * fx; c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy; c1 = c01 * (1 - fy) + c11 * fy
        # density must be >= 0 for the marcher; the volume already is, but clip defensively
        return np.maximum(c0 * (1 - fz) + c1 * fz, 0.0)
    return field


def _selftest():
    """Regression trap: the volume is deterministic, actually structured (bright sheets AND dark voids), stars carve
    real cavities, and the column projection is a sane 2-D image."""
    v = nebula_volume(res=32, seed=0)
    assert v.shape == (32, 32, 32) and v.min() >= 0.0 and v.max() <= 1.0, "volume shape/range wrong"
    assert np.array_equal(v, nebula_volume(res=32, seed=0)), "nebula must be deterministic"
    # structure: there are near-empty voids and bright filaments, and real variance (not a flat fog)
    assert np.mean(v < 0.02) > 0.2, "no dark voids -- not nebula-like"
    assert v.max() > 0.5 and np.var(v) > 1e-3, "no bright structure -- flat field"

    # a star at the centre carves a cavity: central density drops vs no star
    core = (slice(12, 20), slice(12, 20), slice(12, 20))
    v_nostar = nebula_volume(res=32, seed=0)
    v_star = nebula_volume(res=32, seed=0, star_positions=[(0.5, 0.5, 0.5)], cavity_radius=0.25)
    assert np.mean(v_star[core]) < np.mean(v_nostar[core]), "star did not carve a cavity in the gas"
    # a 2-D star position is accepted (placed at mid-plane)
    assert nebula_volume(res=16, seed=1, star_positions=[(0.3, 0.7)]).shape == (16, 16, 16)

    # column projection is a sane image
    col = nebula_column(v)
    assert col.shape == (32, 32) and np.all(col >= 0.0), "column projection wrong"

    # ridged vs smooth are different fields (the filament transform actually does something)
    assert not np.array_equal(nebula_volume(res=16, seed=2, ridged=True), nebula_volume(res=16, seed=2, ridged=False))

    # render bridge: the field callable matches the voxel grid at grid points (trilinear is exact on nodes)
    fn = nebula_field_fn(v, bounds=((0.0, 0.0, 0.0), (31.0, 31.0, 31.0)))
    pts = np.array([[10.0, 12.0, 5.0], [3.0, 20.0, 28.0]])
    got = fn(pts)
    assert np.allclose(got, [v[10, 12, 5], v[3, 20, 28]], atol=1e-6), "field callable disagrees with the grid"
    assert np.all(got >= 0.0), "density fed to the marcher must be non-negative"

    print("holographic_nebula selftest OK  |  deterministic 3-D volume; dark voids + bright filaments (var %.3f); "
          "stars carve cavities; column projection sane  |  artist's nebula, NOT a hydro sim (fluid advection declared)"
          % float(np.var(v)))


if __name__ == "__main__":
    _selftest()
