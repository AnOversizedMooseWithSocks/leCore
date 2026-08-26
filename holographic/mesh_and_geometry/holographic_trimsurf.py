"""holographic_trimsurf.py -- TRIMMED SURFACES (K3): a surface plus boundary loops in parameter space, the way Rhino
represents the majority of real geometry (a trimmed NURBS face, not a whole rectangular patch).

WHY THIS IS THE RHINO-CRITICAL PIECE
------------------------------------
A raw NURBS surface is a full rectangular (u,v) patch. Real geometry is that patch with pieces cut away: a hole, a
clipped edge, the region bounded by where another surface crossed it (the SSI curve from K2). A trimmed surface is
therefore (surface, trim loops in parameter space): an OUTER loop the face lives inside, and zero or more HOLE loops
cut out of it. Everything a modeler shows or exports for a trimmed face is "evaluate the surface, but only where
(u,v) is inside the trim region."

WHAT IT PROVIDES
----------------
  * TrimmedSurface(surf_uv, outer, holes) -- surf_uv(u,v)->(x,y,z); outer/holes are (n,2) loops in (u,v).
  * is_inside(u,v) -- robust even-odd point-in-region test (inside outer, outside every hole), decided by the EXACT
    orient2d crossing count so a query exactly on a trim edge is classified deterministically, not by luck.
  * tessellate(nu,nv) -- sample the surface on a (u,v) grid and keep only the quads whose center is inside the trim
    region -> a mesh (vertices, faces) that respects the trim. The honest simple tessellation: precision is the grid;
    a boundary-fitted triangulation is a later refinement (kept negative, below).
  * trim_loop_from_curve(surf_uv, curve3d, ...) -- project a 3-D trimming curve (e.g. a K2 SSI polyline) to parameter
    space by nearest-(u,v) lookup, giving the loop to trim with. This is the bridge K2 -> K3.

KEPT NEGATIVE (loud)
--------------------
tessellate() is a GRID keep-inside, so the trimmed boundary is stair-stepped at grid resolution -- fine for display
and point queries, NOT a boundary-fitted mesh. A boundary-conforming trimmed triangulation (constrained Delaunay in
parameter space along the trim loops) is the declared refinement; this module does not claim it.

Deterministic; NumPy + stdlib only.
"""
import numpy as np

from holographic.mesh_and_geometry.holographic_geomkernel import orient2d, DEFAULT_TOL


def _point_in_loop(u, v, loop):
    """Even-odd point-in-polygon for (u,v) against a closed loop (n,2), using a ray to +u and counting crossings via
    the EXACT orient2d so a vertex-grazing ray is decided deterministically. Returns True if strictly inside."""
    loop = np.asarray(loop, float)
    n = len(loop)
    inside = False
    j = n - 1
    for i in range(n):
        yi = loop[i][1]; yj = loop[j][1]
        # does the horizontal ray at height v cross edge (j->i)?
        if (yi > v) != (yj > v):
            # x of the crossing, decided robustly: is the query left of the edge at height v?
            # orient2d(loop[j], loop[i], (u,v)) sign tells the side; combined with edge direction gives the crossing.
            s = orient2d(loop[j], loop[i], (u, v))
            if s != 0 and (s > 0) == (yj > yi):
                inside = not inside
        j = i
    return inside


class TrimmedSurface:
    """A surface `surf_uv(u,v)->(x,y,z)` restricted to a trim region: inside `outer` and outside every hole in
    `holes` (each an (n,2) closed loop in parameter space). u,v default to [0,1]x[0,1]."""

    def __init__(self, surf_uv, outer, holes=None, u_range=(0.0, 1.0), v_range=(0.0, 1.0)):
        self.surf_uv = surf_uv
        self.outer = np.asarray(outer, float)
        self.holes = [np.asarray(h, float) for h in (holes or [])]
        self.u_range = tuple(u_range)
        self.v_range = tuple(v_range)

    def is_inside(self, u, v):
        """Is parameter (u,v) inside the trimmed region (inside the outer loop and outside all holes)?"""
        if not _point_in_loop(u, v, self.outer):
            return False
        for h in self.holes:
            if _point_in_loop(u, v, h):
                return False
        return True

    def eval(self, uv):
        """Evaluate the underlying surface at (m,2) parameter points -> (m,3)."""
        uv = np.asarray(uv, float)
        return np.array([self.surf_uv(float(p[0]), float(p[1])) for p in uv])

    def tessellate(self, nu=48, nv=48):
        """A trim-respecting mesh: sample the surface on an (nu,nv) grid and keep quads whose CENTER is inside the
        trim region. Returns (vertices (V,3), faces list of quads). Stair-stepped at grid resolution (kept negative)."""
        us = np.linspace(self.u_range[0], self.u_range[1], nu)
        vs = np.linspace(self.v_range[0], self.v_range[1], nv)
        # grid of surface points
        grid = np.empty((nu, nv, 3), float)
        for iu in range(nu):
            for iv in range(nv):
                grid[iu, iv] = self.surf_uv(us[iu], vs[iv])
        verts = grid.reshape(-1, 3)
        def vid(iu, iv):
            return iu * nv + iv
        faces = []
        for iu in range(nu - 1):
            for iv in range(nv - 1):
                uc = 0.5 * (us[iu] + us[iu + 1]); vc = 0.5 * (vs[iv] + vs[iv + 1])
                if self.is_inside(uc, vc):
                    faces.append((vid(iu, iv), vid(iu + 1, iv), vid(iu + 1, iv + 1), vid(iu, iv + 1)))
        return verts, faces

    def area_fraction(self, res=64):
        """Fraction of the (u,v) domain that survives the trim -- a cheap measure the tessellation should match, and
        a handy check that a trim loop encloses what you think it does."""
        us = np.linspace(self.u_range[0], self.u_range[1], res)
        vs = np.linspace(self.v_range[0], self.v_range[1], res)
        inside = 0
        for u in us:
            for v in vs:
                if self.is_inside(u, v):
                    inside += 1
        return inside / (res * res)


def trim_loop_from_curve(surf_uv, curve3d, res=40):
    """Project a 3-D trimming curve to parameter space by nearest-(u,v) lookup on an (res,res) sampling of the
    surface -- the bridge from a K2 SSI polyline to a K3 trim loop. Returns an (n,2) loop in (u,v). Coarse by design
    (nearest grid node); refine `res` for accuracy, or Newton-polish per point in a caller that needs sub-cell (a
    declared refinement, not done here)."""
    us = np.linspace(0.0, 1.0, res); vs = np.linspace(0.0, 1.0, res)
    U, V = np.meshgrid(us, vs, indexing="ij")
    samples = np.array([surf_uv(float(u), float(v)) for u, v in zip(U.ravel(), V.ravel())])
    uv_grid = np.stack([U.ravel(), V.ravel()], axis=-1)
    out = []
    for p in np.asarray(curve3d, float):
        d = np.linalg.norm(samples - p, axis=1)
        out.append(uv_grid[int(np.argmin(d))])
    return np.asarray(out)


def _selftest():
    tol = DEFAULT_TOL

    # a flat unit surface in the z=0 plane: surf_uv(u,v) = (u, v, 0), u,v in [0,1]
    def flat(u, v):
        return np.array([u, v, 0.0])

    # --- point-in-region on a simple square hole ---
    outer = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    hole = np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]])
    ts = TrimmedSurface(flat, outer, [hole])
    assert ts.is_inside(0.1, 0.1) and ts.is_inside(0.9, 0.9)   # inside outer, outside hole
    assert not ts.is_inside(0.5, 0.5)                          # inside the hole -> trimmed away
    assert not ts.is_inside(1.5, 0.5)                          # outside the outer loop

    # --- area fraction: outer area 1 minus hole area 0.04 -> ~0.96 ---
    af = ts.area_fraction(res=100)
    assert abs(af - 0.96) < 0.02, af

    # --- a circular trim: keep only inside a disc of radius 0.3 centred at (0.5,0.5) ---
    th = np.linspace(0, 2 * np.pi, 64)
    disc = np.c_[0.5 + 0.3 * np.cos(th), 0.5 + 0.3 * np.sin(th)]
    tc = TrimmedSurface(flat, disc)
    assert tc.is_inside(0.5, 0.5) and not tc.is_inside(0.5, 0.95)
    assert abs(tc.area_fraction(res=120) - np.pi * 0.3 ** 2) < 0.01   # disc area pi r^2

    # --- tessellation respects the trim: no kept quad's center is inside the hole ---
    verts, faces = ts.tessellate(nu=40, nv=40)
    assert len(faces) > 0
    us = np.linspace(0, 1, 40); vs = np.linspace(0, 1, 40)
    for (a, b, c, d) in faces:
        # recover the quad's (u,v) center from vertex ids and assert it's in-region
        iu = a // 40; iv = a % 40
        uc = 0.5 * (us[iu] + us[iu + 1]); vc = 0.5 * (vs[iv] + vs[iv + 1])
        assert ts.is_inside(uc, vc)
    # the kept-face fraction tracks the area fraction
    frac = len(faces) / (39 * 39)
    assert abs(frac - 0.96) < 0.03, frac

    # --- bridge from a 3-D curve: project a circle in the z=0 plane back to (u,v) ---
    curve3d = np.c_[0.5 + 0.3 * np.cos(th), 0.5 + 0.3 * np.sin(th), np.zeros_like(th)]
    loop_uv = trim_loop_from_curve(flat, curve3d, res=50)
    assert loop_uv.shape[1] == 2 and np.max(np.abs(loop_uv[:, 0] - curve3d[:, 0])) < 0.05

    # --- determinism ---
    v1, f1 = ts.tessellate(20, 20); v2, f2 = ts.tessellate(20, 20)
    assert np.array_equal(v1, v2) and f1 == f2

    print("holographic_trimsurf selftest OK: point-in-trim is exact (inside outer & outside holes, robust orient2d "
          "crossing count); area fraction matches analytic (square-with-hole 0.96, disc pi*r^2); tessellate keeps "
          "only in-region quads and its face fraction tracks the area; a 3-D curve projects to a (u,v) trim loop "
          "(the K2->K3 bridge); deterministic. Kept negative: grid keep-inside stair-steps the trim boundary; a "
          "boundary-fitted triangulation is a declared refinement.")


if __name__ == "__main__":
    _selftest()
