"""The remaining classic mesh tools (ANIM-3): mirror and merge-by-distance (weld).

WHY THIS MODULE EXISTS
----------------------
The mesh kernel already ships the editing verbs a modeler reaches for -- extrude/inset (meshverbs),
bevel/bridge/loop_cut (meshverbs2), flip/split/collapse edge (eulerops), laplacian_smooth, loop_subdivide,
qem/cluster decimate. Two everyday tools were missing: MIRROR (reflect across a plane and weld the seam -- how
half a symmetric model is built) and MERGE-BY-DISTANCE / WELD (collapse coincident vertices into one -- the
cleanup every import and every mirror needs). Both are here, vectorised for triangle meshes (no per-vertex
Python loop; the face remap is array ops over the (T,3) face table).
"""

import numpy as np


def merge_by_distance(mesh, tol=1e-5):
    """Weld vertices closer than `tol` into one. Vertices are grouped by snapping to a `tol` grid; each group
    becomes one vertex at the group's mean; faces are remapped and any face that collapsed to < 3 distinct
    vertices is dropped. Vectorised for triangle meshes (the (T,3) face table is remapped and degenerate-filtered
    as array ops). The cleanup after a mirror / import / boolean."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = mesh.vertices
    key = np.round(V / tol).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)      # old vertex -> merged group id
    inv = np.asarray(inv).ravel()
    nnew = int(inv.max()) + 1
    counts = np.bincount(inv, minlength=nnew).astype(float)
    Vnew = np.zeros((nnew, 3))
    np.add.at(Vnew, inv, V)                                   # group sum (scatter)
    Vnew /= counts[:, None]                                   # -> group mean

    faces = mesh.faces
    if faces and all(len(f) == 3 for f in faces):
        F = inv[np.asarray(faces, dtype=int)]                 # remap all triangles at once
        good = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])   # drop degenerates
        Fnew = [tuple(int(x) for x in row) for row in F[good]]
    else:
        Fnew = []
        for f in faces:                                       # polygon fallback: remap + drop repeats
            seq = []
            for vi in f:
                m = int(inv[vi])
                if not seq or seq[-1] != m:
                    seq.append(m)
            if len(seq) >= 2 and seq[0] == seq[-1]:
                seq.pop()
            if len(seq) >= 3:
                Fnew.append(tuple(seq))
    return Mesh(Vnew, Fnew)


def mirror(mesh, axis=0, plane=0.0, weld=True, tol=1e-5):
    """Mirror a mesh across the `axis`=const `plane`: append a reflected copy with reversed winding (a reflection
    flips orientation, so the normals stay consistent), then optionally WELD the seam vertices that land on the
    plane. The standard way to model a symmetric object from one half. Vectorised."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = mesh.vertices
    Vm = V.copy()
    Vm[:, axis] = 2.0 * plane - Vm[:, axis]                   # reflect across the plane
    Vall = np.vstack([V, Vm])
    off = len(V)
    faces = mesh.faces
    Fm = [tuple(reversed([int(vi) + off for vi in f])) for f in faces]   # reversed winding for the reflection
    out = Mesh(Vall, list(faces) + Fm)
    if weld:
        out = merge_by_distance(out, tol=tol)                 # fuse the coincident seam vertices
    return out


def symmetrize(mesh, axis=0, plane=0.0, side=+1, tol=1e-5):
    """SYMMETRIZE a (possibly asymmetric) mesh across the `axis`=const `plane`: KEEP the half on `side` (+1 = the
    positive side of the plane, -1 = the negative), then MIRROR that half back across the plane and weld the seam,
    producing a bilaterally-symmetric mesh. The 'symmetrize' a modeler runs to clean up a sculpt that drifted off
    axis -- unlike `mirror` (which doubles the WHOLE mesh), this first discards the far side, so it fixes asymmetry
    instead of preserving it.

    A face is kept iff its CENTROID is on `side` of the plane (the robust, Blender-style rule -- no cutting of
    straddling faces, which would need re-triangulation). Vertices within `tol` of the plane are SNAPPED exactly onto
    it first, so the seam welds cleanly after the mirror. Returns a new `Mesh`. Composes the existing mirror + weld
    (merge_by_distance) primitives -- symmetrize is 'keep one side, then mirror', not a new reflection kernel.

    KEPT NEGATIVE: a face that lies IN the mirror direction (its centroid on the plane, e.g. the side faces of an
    axis-aligned box mirrored across x=0) is kept and then mirrored, producing a coplanar duplicate -- so a box goes
    6 -> 10 faces. The result is still SYMMETRIC (the contract), just not minimal; a proper fix would drop faces whose
    supporting plane contains the mirror normal, deferred. For the intended use (an off-axis sculpt with faces that
    genuinely pick a side) this does not arise."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = mesh.vertices.copy()
    s = float(np.sign(side)) or 1.0
    # snap near-plane vertices exactly onto the plane, so the mirrored copy's seam vertices coincide and weld.
    near = np.abs(V[:, axis] - plane) <= tol
    V[near, axis] = plane
    kept_faces = []
    for f in mesh.faces:
        idx = list(f)
        centroid_side = s * (V[idx, axis].mean() - plane)   # which side the face's centroid is on
        # KEEP a face whose centroid is on the keep side OR on the plane (a straddling face belongs to both halves;
        # keeping it and mirroring it produces the symmetric pair). Only faces whose centroid is clearly on the FAR
        # side are discarded -- that is the asymmetry being cut away. (No cutting of straddling faces, Blender-style.)
        if centroid_side >= -tol:
            kept_faces.append(tuple(idx))
    half = Mesh(V, kept_faces)
    # mirror the kept half back across the plane and weld the on-plane seam -> the symmetric whole.
    return mirror(half, axis=axis, plane=plane, weld=True, tol=tol)


def _selftest():
    from holographic.mesh_and_geometry.holographic_mesh import Mesh, box
    # weld: duplicate every vertex of a box, then merge_by_distance should recover the original count
    b = box()
    dup = Mesh(np.vstack([b.vertices, b.vertices]),
               [tuple(f) for f in b.faces])                   # faces still index the first copy
    w = merge_by_distance(dup, tol=1e-5)
    assert w.n_vertices == b.n_vertices, (w.n_vertices, b.n_vertices)
    # mirror a half-grid across x=0 -> symmetric, and the seam welds
    from holographic.mesh_and_geometry.holographic_mesh import grid
    g = grid(4, 4)
    g.vertices[:, 0] = np.abs(g.vertices[:, 0])               # fold to +x half (a crude half)
    m = mirror(g, axis=0, plane=0.0, weld=True)
    assert m.n_vertices < g.n_vertices * 2                    # the seam welded (fewer than a naive double)
    assert np.allclose(m.vertices[:, 0].min(), -m.vertices[:, 0].max(), atol=1e-6)   # symmetric about x=0

    # symmetrize an ASYMMETRIC mesh -> a bilaterally symmetric result (unlike mirror, which doubles the whole mesh).
    g2 = grid(6, 6)
    Va = g2.vertices.copy()
    Va[Va[:, 0] > 0.0, 2] += 0.5                              # bump the +x half up in z -> asymmetric
    asym = Mesh(Va, [tuple(f) for f in g2.faces])
    sym = symmetrize(asym, axis=0, plane=0.0, side=+1)
    Vs = sym.vertices
    mirrored = Vs.copy(); mirrored[:, 0] = -mirrored[:, 0]    # every mirrored vertex must match an original one
    ok_sym = all(np.linalg.norm(Vs - mv, axis=1).min() < 1e-4 for mv in mirrored)
    assert ok_sym, "symmetrize must produce a mesh symmetric across the plane"
    assert sym.n_faces == asym.n_faces, "keeping +x half (half the faces) then mirroring restores the face count"

    # SOLIDIFY an open sheet -> a CLOSED, MANIFOLD watertight slab (this operator had no test, and shipped a
    # non-manifold bridge: the rim quads traversed the boundary edge the SAME way as the outer wall, so a directed
    # edge appeared twice. The bridge must traverse it the OTHER way. Pin closed+manifold+zero-boundary so it stays.)
    sheet = grid(4, 4)
    slab = solidify(sheet, thickness=0.2)
    assert slab.is_closed() and slab.is_manifold(), "a solidified open sheet must be a closed manifold slab"
    assert slab.euler_characteristic() == 2, "a thickened disk is a topological sphere (chi 2)"
    from collections import defaultdict as _dd
    _ec = _dd(int)
    for _f in slab.faces:
        for _k in range(len(_f)):
            _ec[tuple(sorted((_f[_k], _f[(_k + 1) % len(_f)])))] += 1
    assert not any(_n == 1 for _n in _ec.values()), "no boundary edges -- the slab is watertight"
    assert abs(np.ptp(slab.vertices, axis=0)[2] - 0.2) < 1e-9, "the thickness is applied along the surface normal"

    print(f"meshtools selftest ok: weld {dup.n_vertices}->{w.n_vertices} verts; "
          f"mirror is symmetric and welds the seam ({g.n_vertices*2} naive -> {m.n_vertices}); "
          f"symmetrize turns an asymmetric grid into a bilaterally-symmetric mesh ({sym.n_faces} faces); "
          f"solidify thickens an open sheet into a closed manifold slab ({slab.n_faces} faces, chi 2, watertight)")


def _boundary_edges(faces):
    """Return the boundary edges (those used by exactly one triangle) as a list of (a, b) with the face's
    winding order preserved -- needed so a solidify bridge keeps consistent orientation. Triangle meshes."""
    from collections import defaultdict
    count = defaultdict(int)
    oriented = {}
    for f in faces:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            key = (a, b) if a < b else (b, a)
            count[key] += 1
            oriented[key] = (a, b)                            # remember one oriented use
    return [oriented[k] for k, c in count.items() if c == 1]


def solidify(mesh, thickness, flip=False):
    """Give a surface thickness (the 'shell' / 'solidify' modifier): offset a copy of the mesh inward along the
    vertex normals by `thickness`, add it with reversed winding (so its normals face out of the inner wall), and
    BRIDGE the boundary edges of an open mesh with quads so the result is a closed solid. A closed input becomes
    a hollow double wall; an open input (a disk, a curved sheet) becomes a watertight thick slab. `flip` offsets
    outward instead. Vertex offset is vectorised; the boundary bridge loops over boundary edges only (a 1-D loop,
    not over all vertices)."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = mesh.vertices
    N = mesh.vertex_normals()
    s = -1.0 if not flip else 1.0
    Vinner = V + s * thickness * N
    Vall = np.vstack([V, Vinner])
    off = len(V)
    faces = [tuple(f) for f in mesh.faces]
    inner = [tuple(reversed([vi + off for vi in f])) for f in faces]   # reversed winding for the back wall
    bridge = []
    for a, b in _boundary_edges(faces):                      # close the open rim with two triangles per edge
        # _boundary_edges returns (a,b) in the direction the OUTER wall traverses it, so the bridge must traverse it
        # the OTHER way (b->a) or the directed edge (a,b) would appear twice -> non-manifold. Winding: b, a, a+off,
        # b+off makes a quad whose outer edge is (b,a), matching the reversed inner wall and closing the shell.
        bridge.append((b, a, a + off))
        bridge.append((b, a + off, b + off))
    return Mesh(Vall, faces + inner + bridge)


if __name__ == "__main__":
    _selftest()
