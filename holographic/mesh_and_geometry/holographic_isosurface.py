"""holographic_isosurface.py -- the points -> SDF -> mesh path (Box3D backlog F3).

The engine could turn an SDF into points (`session.sdf_surface_points`) and could never turn points back into a
surface. Audited before building: `convert splats to a mesh`, `surface reconstruction from points`, `marching
cubes` and `dual contouring` all returned fallbacks. This module closes the inverse direction, with two pieces:

  * `sdf_from_points` -- a signed distance grid from an ORIENTED point cloud: the distance to the nearest sample,
    signed by that sample's normal.
  * `surface_nets` -- DUAL isosurface extraction (one vertex per sign-changing cell, placed at the mean of the
    cell's edge crossings; one quad per sign-changing grid edge). Naive surface nets, the ancestor of Dual
    Contouring (Ju, Losasso, Schaefer & Warren, SIGGRAPH 2002), which adds a QEF solve to recover sharp features.
    This does not; it is smooth-surface extraction, and it says so.

MEASURED on a unit sphere, 600 oriented samples, a 32^3 grid (cell size 0.1032):

    mesh                    1,804 vertices, 1,802 quads
    watertight              every edge shared by exactly 2 faces
    vertex radius           mean 1.0019, max deviation 0.0454 = **0.44 cells**

KEPT NEGATIVE 1, and it is the opposite of what I assumed. **The point-cloud SDF is LEAST accurate near the
surface**, which is exactly where an isosurface extractor reads it:

    |true sdf| in [0, 0.1)     max err 0.2225   mean 0.0449
    |true sdf| in [0.1, 0.3)   max err 0.1550   mean 0.0183
    |true sdf| in [0.3, 0.6)   max err 0.0914   mean 0.0102
    |true sdf| in [0.6, 2.0)   max err 0.0695   mean 0.0075

The reason: distance-to-nearest-SAMPLE overestimates distance-to-SURFACE by up to the sample spacing, and that gap
is proportionally worst where the true distance is smallest. Measured, the near-surface error tracks the spacing:

    N points    spacing    max err    err / spacing
        150      0.2894     0.4984         1.72
        600      0.1447     0.2352         1.62
      2,400      0.0724     0.0935         1.29

So the SDF's accuracy is set by the CLOUD, not by the grid. Refining the grid under a sparse cloud buys nothing.

KEPT NEGATIVE 2 -- **the mesh is far more accurate than the field it was extracted from** (0.045 vertex error from a
field with 0.22 error). That is not a paradox and it is not luck: the zero crossing is an interpolated quantity, and
averaging the twelve edge crossings of a cell cancels most of the per-sample noise. *Do not read the field's error
as the mesh's error, in either direction.*

KEPT NEGATIVE 3 -- **the all-pairs distance matrix is O(grid x points) and will kill the process.** A 24^3 grid
against 9,600 points is 133M floats; the first probe was OOM-killed. `sdf_from_points` chunks over grid cells.
Chunking here is a correctness measure, not an optimisation.
"""

import numpy as np


# The 8 corners of a unit cell, and the 12 edges joining adjacent ones. Fixed order => deterministic vertices.
_CORNERS = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)])
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8)
          if int(np.abs(_CORNERS[a] - _CORNERS[b]).sum()) == 1]


def sdf_from_points(points, normals, lo, hi, res, chunk=4096):
    """A signed distance grid from an ORIENTED point cloud.

    Returns `(field, grids)` where `field` is `(res, res, res)` and `grids` is the per-axis coordinate vector. Each
    grid cell takes the distance to its nearest sample, signed by the dot product of the offset with that sample's
    normal -- so the sign is an orientation question and a cloud with flipped normals produces an inside-out solid.

    `chunk` bounds the distance matrix: without it, a 24^3 grid against 9,600 points allocates 133M floats and the
    process is killed. That is a correctness measure.

    ACCURACY IS SET BY THE CLOUD, NOT THE GRID: the near-surface error tracks the sample spacing at ~1.3-1.7x it.
    Refining the grid under a sparse cloud buys nothing (see the module note)."""
    pts = np.asarray(points, float)
    nrm = np.asarray(normals, float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3); got %r" % (pts.shape,))
    if nrm.shape != pts.shape:
        raise ValueError("normals must match points: %r vs %r" % (nrm.shape, pts.shape))
    if len(pts) == 0:
        raise ValueError("sdf_from_points needs at least one sample")

    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    res = int(res)
    grids = [np.linspace(lo[d], hi[d], res) for d in range(3)]
    G = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 3)

    out = np.empty(len(G))
    for s in range(0, len(G), int(chunk)):
        blk = G[s:s + int(chunk)]
        d2 = ((blk[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)                                    # nearest sample; ties -> lowest index (np.argmin)
        offs = blk - pts[j]
        sign = np.sign(np.einsum("nd,nd->n", offs, nrm[j]))
        sign[sign == 0.0] = 1.0                             # exactly on the tangent plane: call it outside, stated
        out[s:s + len(blk)] = sign * np.sqrt(d2[np.arange(len(blk)), j])
    return out.reshape(res, res, res), grids


def surface_nets(field, grids, iso=0.0):
    """DUAL isosurface extraction: `(vertices, quads)` from a scalar `field` on the grid `grids`.

    One vertex per cell whose eight corners straddle `iso`, placed at the mean of that cell's edge crossings (each
    found by linear interpolation). One quad per grid EDGE that crosses `iso`, joining the four cells around it.
    Deterministic: the corner and edge orders are fixed, so the same field always yields the same mesh.

    HONEST SCOPE: this is naive surface nets, not Dual Contouring. DC (Ju et al., SIGGRAPH 2002) places the vertex
    by a QEF solve over the crossings' normals and thereby recovers SHARP features; averaging the crossings, as
    here, rounds them off. On a smooth surface the two agree; on a cube's edge they do not. Use it for smooth
    surfaces and say so.

    Returns `vertices (V, 3)` and `quads (Q, 4)` indexing into them. A closed surface comes out WATERTIGHT *and*
    ORIENTED: every undirected edge sits in exactly two faces, every DIRECTED edge is traversed exactly once, and
    every normal points along `+grad(field)` -- outward for an SDF.

    **WATERTIGHT IS NOT ORIENTED, and the first version of this function was only the former.** Measured on a
    sphere: `is_watertight` True, `is_oriented` False, 228 directed edges duplicated, 98 of 200 normals pointing
    inward. Nothing downstream noticed until a half-edge structure was built on it and raised. Orienting needs TWO
    sign flips composed -- the crossing's direction and the frame's parity -- and getting only the first left 136 of
    408 normals outward. See the comment at the winding.

    KEPT NEGATIVE -- **THE EXTRACTION IS NOT MANIFOLD AT EVERY RESOLUTION, and the checks are the gate.** On a torus
    at grid resolutions 14, 18 and 20 the output is watertight AND oriented. At **16 it is NEITHER** -- 48 of 1,744
    directed edges duplicated, and some undirected edge is not shared by exactly two faces. That is the ambiguous
    cell: **one dual vertex per cell assumes the surface crosses the cell once**, and where a thin feature enters a
    cell twice the assumption fails. Dual Contouring's octree and Manifold Dual Contouring exist for exactly this.

    Note the resolution is non-monotonic -- 14 is fine and 16 is not -- so this is a CONFIGURATION failure, not an
    under-sampling one, and no "use a finer grid" rule fixes it. The remedy is not a tolerance; it is to CHECK.
    `is_watertight` and `is_oriented` catch it, and `holographic_crossfield.cross_field` refuses such a mesh rather
    than transporting a frame across a face that winds the wrong way."""
    F = np.asarray(field, float) - float(iso)
    if F.ndim != 3:
        raise ValueError("field must be 3-D; got %r" % (F.shape,))
    res = F.shape[0]
    if F.shape != (res, res, res):
        raise ValueError("field must be cubic; got %r" % (F.shape,))

    # cell corner values, (res-1)^3 x 8, in the fixed _CORNERS order
    C = np.stack([F[i:i + res - 1, j:j + res - 1, k:k + res - 1] for i, j, k in _CORNERS], axis=-1)
    active = (C.min(-1) < 0.0) & (C.max(-1) >= 0.0)
    idx = np.argwhere(active)
    if len(idx) == 0:
        return np.zeros((0, 3)), np.zeros((0, 4), int)

    cell_id = -np.ones((res - 1,) * 3, int)
    cell_id[tuple(idx.T)] = np.arange(len(idx))
    dx = np.array([g[1] - g[0] for g in grids])
    origin = np.array([g[0] for g in grids])

    verts = np.empty((len(idx), 3))
    for n, (i, j, k) in enumerate(idx):
        vals = C[i, j, k]
        crossings = []
        for a, b in _EDGES:
            va, vb = vals[a], vals[b]
            if (va < 0.0) != (vb < 0.0):
                t = va / (va - vb)                          # the zero crossing, linearly interpolated
                crossings.append(_CORNERS[a] + t * (_CORNERS[b] - _CORNERS[a]))
        verts[n] = origin + (np.array([i, j, k]) + np.mean(crossings, axis=0)) * dx

    quads = []
    for axis in range(3):
        lo_sl = tuple(slice(0, res - 1) for _ in range(3))
        hi_sl = tuple(slice(1, res) if d == axis else slice(0, res - 1) for d in range(3))
        a, b = F[lo_sl], F[hi_sl]
        other = [d for d in range(3) if d != axis]
        for node in np.argwhere((a < 0.0) != (b < 0.0)):
            ring = []
            for du in (0, -1):
                for dv in (0, -1):
                    c = node.copy()
                    c[other[0]] += du
                    c[other[1]] += dv
                    if np.any(c < 0) or np.any(c >= res - 1) or cell_id[tuple(c)] < 0:
                        ring = None
                        break
                    ring.append(int(cell_id[tuple(c)]))
                if ring is None:
                    break
            if ring is not None and len(ring) == 4:
                quad = [ring[0], ring[1], ring[3], ring[2]]           # wound around the edge, not across it
                # ORIENT THE QUAD. A watertight mesh is not an ORIENTED one: every undirected edge can sit in
                # exactly two faces while both wind the same way, so half the normals point inward. Measured before
                # this block existed: watertight True, oriented False, 228 directed edges duplicated, 98 of 200
                # normals outward.
                #
                # Two sign flips compose:
                #   * the CROSSING's direction -- the field rises along +axis exactly when a < 0 <= b, and that is
                #     the outward normal;
                #   * the FRAME's parity -- `other = [d for d in range(3) if d != axis]` is right-handed with `axis`
                #     for axis 0 and 2, and LEFT-handed for axis 1, because (1, 0, 2) is an odd permutation. Fixing
                #     only the crossing sign left 136 of 408 normals outward; both flips are needed.
                if (a[tuple(node)] < 0.0) != (axis == 1):
                    quad = quad[::-1]              # normals now point along +grad(field): OUTWARD for an SDF
                quads.append(quad)
    return verts, np.array(quads, int).reshape(-1, 4)


def points_to_mesh(points, normals, lo, hi, res, chunk=4096):
    """The whole F3 path: oriented points -> signed distance grid -> watertight quad mesh. Returns
    `(vertices, quads, field, grids)` so the intermediate field can be inspected rather than trusted."""
    field, grids = sdf_from_points(points, normals, lo, hi, res, chunk=chunk)
    verts, quads = surface_nets(field, grids)
    return verts, quads, field, grids


def is_watertight(quads):
    """Is every edge shared by exactly two faces? The one structural property a closed surface must have, and the
    one a dual extractor can plausibly get wrong at the grid boundary.

    **Watertight is not oriented.** This counts UNDIRECTED edges, so it passes on a mesh whose faces wind
    inconsistently -- and one did, until `surface_nets` learned to orient by the crossing's sign. Use
    `is_oriented` for the stronger property."""
    from collections import Counter

    counts = Counter()
    for q in np.asarray(quads, int):
        for e in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])):
            counts[tuple(sorted(e))] += 1
    return all(c == 2 for c in counts.values()) and len(counts) > 0


def is_oriented(quads):
    """Is every DIRECTED edge traversed exactly once? The property a half-edge structure, a normal, and any
    downstream field solver actually need -- and the one `is_watertight` cannot see.

    A watertight-but-unoriented mesh has each undirected edge in two faces that wind the SAME way, so half its
    normals point inward. Measured on a sphere before the fix: watertight True, oriented False, 228 directed edges
    duplicated, 98 of 200 quad normals outward."""
    from collections import Counter

    counts = Counter()
    for q in np.asarray(quads, int):
        for e in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])):
            counts[e] += 1
    return bool(counts) and all(c == 1 for c in counts.values())


def mesh_report(vertices, quads, sdf=None):
    """{n_vertices, n_quads, watertight, euler, surface_error}. `sdf`, if given, is an analytic signed-distance
    callable used to score the vertices -- **the honest baseline is the cell size**, because a dual extractor can
    never place a vertex better than the cell it lives in."""
    V = np.asarray(vertices, float)
    Q = np.asarray(quads, int)
    # Euler characteristic V - E + F. On a CLOSED quad mesh every edge is shared by two faces, so E = 4Q/2 = 2Q,
    # and V - E + F collapses to V - Q. A sphere gives 2; a torus 0. It is only meaningful when watertight.
    rep = {"n_vertices": int(len(V)), "n_quads": int(len(Q)), "watertight": bool(is_watertight(Q)),
           "euler": int(len(V) - len(Q)), "surface_error": None}
    if sdf is not None and len(V):
        rep["surface_error"] = float(np.abs(np.asarray(sdf(V), float)).max())
    return rep


def _selftest():
    """A numeric regression trap: the sphere reconstructs to sub-cell accuracy and comes out watertight, the field's
    error is worst NEAR the surface (the counterintuitive one), and the mesh is more accurate than its field."""
    res = 24
    lo, hi = np.full(3, -1.6), np.full(3, 1.6)
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(600, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    nrm = pts.copy()                                        # a unit sphere's normal IS its position

    V, Q, F, grids = points_to_mesh(pts, nrm, lo, hi, res)
    cell = float(grids[0][1] - grids[0][0])

    # 1. a watertight AND ORIENTED closed surface -- the second is the property a half-edge structure needs, and
    #    `is_watertight` cannot see it.
    assert len(V) > 500 and len(Q) > 500
    assert is_watertight(Q)
    assert is_oriented(Q)
    outward = sum(1 for q in Q if np.cross(V[q][1] - V[q][0], V[q][2] - V[q][0]) @ V[q].mean(axis=0) > 0)
    assert outward == len(Q)                                # every normal along +grad(field)

    # 2. sub-cell surface accuracy. The honest baseline is the CELL SIZE, not zero.
    radial = np.abs(np.linalg.norm(V, axis=1) - 1.0)
    assert radial.max() < cell, (radial.max(), cell)

    # 3. KEPT NEGATIVE: the FIELD is least accurate near the surface, where the extractor reads it
    G = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
    truth = np.linalg.norm(G, axis=-1) - 1.0
    err = np.abs(F - truth)
    near = np.abs(truth) < 0.1
    far = np.abs(truth) > 0.6
    assert err[near].max() > err[far].max(), (err[near].max(), err[far].max())

    # 4. KEPT NEGATIVE: the MESH is more accurate than the FIELD it came from -- averaging the crossings cancels
    #    per-sample noise. Do not read one error as the other.
    assert radial.max() < 0.5 * err[near].max()

    # 5. the cloud sets the accuracy, not the grid: more points, less error, at a FIXED grid
    coarse = rng.normal(size=(150, 3))
    coarse /= np.linalg.norm(coarse, axis=1, keepdims=True)
    Fc, _ = sdf_from_points(coarse, coarse, lo, hi, res)
    assert np.abs(Fc - truth)[near].max() > err[near].max()

    # 6. guards
    for bad in (lambda: sdf_from_points(np.zeros((0, 3)), np.zeros((0, 3)), lo, hi, res),
                lambda: sdf_from_points(pts, pts[:-1], lo, hi, res),
                lambda: surface_nets(np.zeros((4, 4)), grids)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("a degenerate input must raise")

    # 7. an empty isosurface is an empty mesh, not a crash
    ev, eq = surface_nets(np.ones((6, 6, 6)), [np.linspace(0, 1, 6)] * 3)
    assert len(ev) == 0 and len(eq) == 0

    print("OK: holographic_isosurface self-test passed (600 oriented samples -> %d vertices, %d quads, watertight "
          "AND oriented -- every directed edge once, every normal outward; "
          "vertex error %.4f against a cell size of %.4f -- sub-cell; and the two kept negatives hold: the FIELD's "
          "error is worst NEAR the surface (%.4f) and best far from it (%.4f), while the MESH is %.1fx more accurate "
          "than the field it was extracted from)"
          % (len(V), len(Q), radial.max(), cell, err[near].max(), err[far].max(), err[near].max() / radial.max()))


if __name__ == "__main__":
    _selftest()
