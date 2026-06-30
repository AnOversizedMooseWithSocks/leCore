"""Face-type control for projected meshes: triangles -> quads -> n-gons (FWD/poly).

WHY THIS MODULE EXISTS
----------------------
The engine projects a field to a mesh by marching tetrahedra, which emits TRIANGLES only. But a modeler
expects to choose the face standard: a quad-dominant mesh (the norm for subdivision / sculpting / clean edge
flow), or n-gons on flat regions (a flat wall should be ONE face, not 200 triangles). The Mesh container
already stores polygons of any size (a quad stays a quad through OBJ); what was missing is the operation that
GROUPS a triangle soup into quads or n-gons on the way out. That is this module: pure, deterministic polygon
merging over an existing triangle Mesh.

WHAT IT PROVIDES
  * `triangles_to_quads(mesh, planarity)` -- greedily pair adjacent triangles into PLANAR, CONVEX quads (the
    most-coplanar pairs first, each triangle used once); leftover triangles stay triangles. Quad-DOMINANT, the
    Blender "Tris to Quads" behaviour.
  * `merge_coplanar(mesh, normal_tol)` -- region-grow connected coplanar faces and emit each flat region whose
    boundary is a single simple loop as ONE n-gon; non-flat regions keep their triangles. Flat faces collapse
    to single polygons.
  * `face_type_counts(mesh)` -- {3: n_tris, 4: n_quads, 5+: n_ngons}, the report a modeler reads.

DETERMINISM
  Both operations sort their candidate merges by a fixed key (quality, then the vertex-index tuple as a
  tie-break) and grow regions in face order, so the output face list is bit-for-bit reproducible. They change
  only the FACE grouping; vertices (and their stable marching keys) are untouched -- the per-vertex identity
  the stable-projection contract relies on still holds. KEPT HONEST: the face grouping itself is NOT stable
  across edits (a flat region's n-gon boundary moves when the region is edited); faces are a derived view,
  vertices are the stable identity.
"""

import numpy as np


def _face_normals(V, faces):
    """A unit normal per face (Newell over its corners -- robust for slightly non-planar polygons)."""
    out = np.zeros((len(faces), 3))
    for fi, f in enumerate(faces):
        n = np.zeros(3)
        m = len(f)
        for k in range(m):
            cur = V[f[k]]; nxt = V[f[(k + 1) % m]]
            n[0] += (cur[1] - nxt[1]) * (cur[2] + nxt[2])
            n[1] += (cur[2] - nxt[2]) * (cur[0] + nxt[0])
            n[2] += (cur[0] - nxt[0]) * (cur[1] + nxt[1])
        nn = np.linalg.norm(n)
        out[fi] = n / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0])
    return out


def _is_convex_quad(V, quad):
    """True iff the 4 vertices form a convex simple polygon, tested in the quad's own average-normal plane via
    consistent-sign cross products of consecutive edges."""
    P = V[list(quad)]
    nrm = np.cross(P[1] - P[0], P[2] - P[0])
    nn = np.linalg.norm(nrm)
    if nn < 1e-12:
        return False
    nrm = nrm / nn
    signs = []
    for k in range(4):
        a = P[(k + 1) % 4] - P[k]
        b = P[(k + 2) % 4] - P[(k + 1) % 4]
        signs.append(np.dot(np.cross(a, b), nrm))
    signs = np.array(signs)
    return bool(np.all(signs > 1e-12) or np.all(signs < -1e-12))


def triangles_to_quads(mesh, planarity=0.90):
    """Merge adjacent triangle pairs into PLANAR, CONVEX quads -- quad-dominant output. `planarity` is the
    minimum dot of the two triangle normals to allow a merge (1.0 = perfectly flat; 0.90 ~ within ~26 deg).
    Greedy by coplanarity (best first), each triangle used at most once; leftovers stay triangles. Returns a new
    Mesh whose faces are quads + residual triangles, on the SAME vertices (their stable keys are unchanged)."""
    from holographic_mesh import Mesh
    V = mesh.vertices
    faces = [tuple(f) for f in mesh.faces]
    tri = [fi for fi, f in enumerate(faces) if len(f) == 3]
    fn = _face_normals(V, faces)

    edge2f = {}                                            # undirected edge -> [triangle face indices]
    for fi in tri:
        f = faces[fi]
        for k in range(3):
            e = (min(f[k], f[(k + 1) % 3]), max(f[k], f[(k + 1) % 3]))
            edge2f.setdefault(e, []).append(fi)

    cands = []                                             # (-coplanarity, edge, f1, f2) -- best (most planar) first
    for e, fl in edge2f.items():
        if len(fl) == 2:
            f1, f2 = fl
            score = float(np.dot(fn[f1], fn[f2]))
            if score >= planarity:
                cands.append((-score, e, f1, f2))
    cands.sort(key=lambda c: (c[0], c[1], c[2], c[3]))     # deterministic: quality then index tie-break

    used = set()
    quads = []
    for _, e, f1, f2 in cands:
        if f1 in used or f2 in used:
            continue
        t1 = faces[f1]
        # find the shared edge (x -> y) in t1's winding order; z = t1's third vertex; d = t2's opposite vertex
        x = y = None
        for k in range(3):
            if (min(t1[k], t1[(k + 1) % 3]), max(t1[k], t1[(k + 1) % 3])) == e:
                x, y = t1[k], t1[(k + 1) % 3]
        z = [v for v in t1 if v not in (x, y)][0]
        d = [v for v in faces[f2] if v not in e][0]
        quad = (z, x, d, y)                                # boundary z->x->d->y, consistent with t1's winding
        if _is_convex_quad(V, quad):
            quads.append(quad)
            used.add(f1); used.add(f2)

    leftovers = [faces[fi] for fi in range(len(faces)) if fi not in used]
    return Mesh(V, quads + leftovers)


def merge_coplanar(mesh, normal_tol=0.999):
    """Merge connected COPLANAR faces into n-gons. Region-grow (in face order, deterministic) over faces whose
    normal stays within `normal_tol` (dot) of the seed normal; emit each region whose boundary is a single
    simple loop as one n-gon, ordered around that loop. Regions that aren't flat, or whose boundary isn't a
    clean loop (holes / pinches), keep their original faces. Flat surfaces collapse to single polygons; curved
    ones are left alone. Same vertices, so stable keys are preserved."""
    from holographic_mesh import Mesh
    V = mesh.vertices
    faces = [tuple(f) for f in mesh.faces]
    fn = _face_normals(V, faces)

    # face adjacency across shared edges
    edge2f = {}
    for fi, f in enumerate(faces):
        m = len(f)
        for k in range(m):
            e = (min(f[k], f[(k + 1) % m]), max(f[k], f[(k + 1) % m]))
            edge2f.setdefault(e, []).append(fi)

    seen = [False] * len(faces)
    out_faces = []
    for seed in range(len(faces)):
        if seen[seed]:
            continue
        seed_n = fn[seed]
        comp = []
        stack = [seed]
        seen[seed] = True
        while stack:                                       # grow the flat region
            fi = stack.pop()
            comp.append(fi)
            f = faces[fi]
            m = len(f)
            for k in range(m):
                e = (min(f[k], f[(k + 1) % m]), max(f[k], f[(k + 1) % m]))
                for nb in edge2f.get(e, []):
                    if not seen[nb] and float(np.dot(fn[nb], seed_n)) >= normal_tol:
                        seen[nb] = True
                        stack.append(nb)
        if len(comp) == 1:
            out_faces.append(faces[comp[0]])
            continue
        ngon = _region_boundary_loop(faces, comp, edge2f)
        out_faces.append(ngon if ngon is not None else None)
        if ngon is None:                                   # couldn't make a clean loop: keep the region's faces
            out_faces.pop()
            out_faces.extend(faces[fi] for fi in comp)
    return Mesh(V, out_faces)


def _region_boundary_loop(faces, comp, edge2f):
    """The single boundary loop of a connected face region as an ordered vertex list, or None if the boundary
    isn't one simple cycle. Boundary edges are those with exactly one incident face INSIDE the region; we walk
    them into a loop and require every boundary vertex to have degree 2 (no pinch/hole)."""
    compset = set(comp)
    # directed boundary edges, oriented as they appear in each region face (so the loop inherits the winding)
    nxt = {}
    bedges = []
    for fi in comp:
        f = faces[fi]
        m = len(f)
        for k in range(m):
            a, b = f[k], f[(k + 1) % m]
            e = (min(a, b), max(a, b))
            inside = [g for g in edge2f.get(e, []) if g in compset]
            if len(inside) == 1:                           # boundary edge of the region
                bedges.append((a, b))
    if not bedges:
        return None
    for a, b in bedges:
        if a in nxt:                                       # a vertex leaving twice -> not a simple loop
            return None
        nxt[a] = b
    if len(nxt) != len(bedges):
        return None
    # walk the cycle
    start = bedges[0][0]
    loop = [start]
    cur = nxt[start]
    while cur != start:
        if cur not in nxt or len(loop) > len(nxt) + 1:
            return None
        loop.append(cur)
        cur = nxt[cur]
    if len(loop) != len(nxt):                              # must visit every boundary vertex exactly once
        return None
    return tuple(loop)


def face_type_counts(mesh):
    """{3: triangles, 4: quads, 5: n-gons(>=5)} -- the face-standard summary a modeler reads."""
    c = {3: 0, 4: 0, 5: 0}
    for f in mesh.faces:
        c[3 if len(f) == 3 else 4 if len(f) == 4 else 5] += 1
    return c


def _selftest():
    from holographic_mesh import Mesh, box
    m = box()                                              # 6 quads
    assert face_type_counts(m)[4] == 6
    tri = Mesh(m.vertices, [tuple(t) for t in m.triangulate()])   # 12 triangles
    assert face_type_counts(tri)[3] == 12
    q = triangles_to_quads(tri)                            # back to (mostly) quads -- box faces are flat
    cc = face_type_counts(q)
    assert cc[4] == 6 and cc[3] == 0, cc                   # every face re-paired into its quad
    ng = merge_coplanar(tri)                               # each flat box face -> one quad/ngon
    assert face_type_counts(ng)[5] + face_type_counts(ng)[4] == 6, face_type_counts(ng)
    print(f"meshpoly selftest ok: box tris {face_type_counts(tri)} -> quads {cc} -> coplanar {face_type_counts(ng)}")


if __name__ == "__main__":
    _selftest()
