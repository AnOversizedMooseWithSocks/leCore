"""Modeler verbs (FWD-7, core three): extrude, inset, dissolve-vertex -- the operations a person reaches for in a
modeller, built on the explicit mesh kernel.

WHY THIS MODULE EXISTS
----------------------
Tier 2's lead item. The shipped primitive Euler operators (holographic_eulerops: flip_edge, split_edge,
collapse_edge, split_face) are the ATOMIC, invariant-preserving moves; the modeller VERBS here are the
human-facing operations built ON those ideas. The backlog's thesis is that the verbs DECOMPOSE into Euler
operators, and that is the honest frame -- with one caveat made explicit below.

WHAT EACH VERB IS (and its relation to the Euler-operator algebra)
  * extrude_face -- lift a face along its normal and wall it in. In the full Euler-operator algebra this is a
    loop of MEV (make-edge-vertex: extrude each corner outward) followed by MEF (make-edge-face: close the side
    walls). HONEST CAVEAT: the shipped primitive set does NOT include MEV (it adds a vertex but only by
    SPLITTING an existing edge, not by extruding a fresh one), so extrude is implemented here as a direct, readable
    face-list construction in the SAME style as the primitives (find the patch, rewrite the faces) rather than as
    a literal call sequence we cannot make from the four we shipped. The decomposition is the conceptual model;
    the construction is the honest implementation.
  * inset_face -- shrink a face toward its centroid, ringing it with new faces. The same MEV+MEF shape as extrude
    but IN-PLANE (no normal displacement) -- a ring of new geometry around a smaller central face.
  * dissolve_vertex -- remove a vertex and its incident "umbrella", then retriangulate the hole. This is the
    Euler KEV (kill-edge-vertex) verb. The shipped `collapse_edge` is the DECIMATION cousin (it removes a vertex
    by merging it onto a neighbour); this dissolve instead KEEPS the surrounding ring fixed and fills the hole,
    which is what a modeller's "dissolve" does.

ALL THREE PRESERVE THE EULER CHARACTERISTIC (chi) and keep a closed mesh closed and manifold -- the measurement
bar for "the verb produced a valid mesh", checked exactly in the self-test. Each also has an EXACT geometric
signature (the extruded cap moves by exactly `distance` along the normal; the inset face's area is exactly
(1-ratio)^2 of the original), measured rather than asserted.

DESIGN CHOICE (readable over clever, matching holographic_eulerops)
  Each verb returns a NEW Mesh and rewrites the FACE LIST directly: find the affected corners, drop the old
  face(s), append the new ones with windings chosen so every directed edge still appears exactly once (the
  manifold condition the kernel enforces). Side walls are TRIANGULATED so the output stays pure-triangle and safe
  for the tri-assuming faculties (cotangent weights, curvature) downstream.

DETERMINISM (per ISA.md)
  No randomness: new vertices are deterministic functions of the input positions, and faces are appended in a
  fixed order. Same mesh + same arguments -> byte-identical result (asserted).

KEPT NEGATIVES / SCOPE (loud)
  * This module ships the CORE three verbs. bevel, bridge, and loop-cut are the FWD-7 remainder, deliberately
    deferred: bevel and bridge need vertex DUPLICATION with an offset/correspondence (fiddlier and easy to get
    subtly wrong), and a general loop-cut needs robust loop tracing on an arbitrary triangle mesh. Shipping three
    correct, measured verbs beats shipping six shaky ones -- the engine's discipline.
  * extrude is NOT a literal composition of the four shipped primitives (it needs MEV, which we did not ship);
    see the caveat above. The decomposition is the model, the direct construction is the implementation.
  * dissolve_vertex fan-triangulates the hole from one ring vertex; for a wildly non-convex link this is valid
    TOPOLOGICALLY (it covers the polygon and stays manifold) but not a quality remesh -- a curvature-aware fill
    would be better, and is out of scope here.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh


def _face_normal(V, face):
    """Unit normal of a (possibly non-planar) face by Newell's method -- for a triangle this reduces to the edge
    cross product, but Newell is robust if the face is a slightly bent polygon."""
    n = np.zeros(3)
    m = len(face)
    for k in range(m):
        cur = V[face[k]]
        nxt = V[face[(k + 1) % m]]
        n[0] += (cur[1] - nxt[1]) * (cur[2] + nxt[2])
        n[1] += (cur[2] - nxt[2]) * (cur[0] + nxt[0])
        n[2] += (cur[0] - nxt[0]) * (cur[1] + nxt[1])
    ln = float(np.linalg.norm(n))
    return n / ln if ln > 1e-12 else n


def _ring_walls(face, new_idx):
    """The triangulated side/ring walls connecting an original face's corners to a parallel set of new corners.

    For each original directed edge (a -> b) of the face, the wall is the quad (a, b, b_new, a_new) split into two
    triangles. The windings are chosen so the wall SUPPLIES the directed edge a -> b -- exactly the edge that was
    freed when the original face was removed -- restoring the manifold balance (every directed edge once). Shared
    by extrude (new corners lifted) and inset (new corners pulled inward): the only difference is WHERE the new
    corners sit, not how they are wired."""
    m = len(face)
    walls = []
    for k in range(m):
        a = face[k]
        b = face[(k + 1) % m]
        a_new = new_idx[k]
        b_new = new_idx[(k + 1) % m]
        walls.append((a, b, b_new))        # supplies a -> b (rebalances the freed edge)
        walls.append((a, b_new, a_new))    # the quad's other half; the diagonal balances internally
    return walls


def extrude_face(mesh, face_index, distance):
    """EXTRUDE: lift face `face_index` along its outward normal by `distance` and connect it back with side walls.
    The lifted face becomes the cap (same winding); the walls keep the mesh a closed manifold with chi unchanged.
    The cap's centroid moves by exactly `distance` along the face normal. Returns a new Mesh."""
    V = mesh.vertices
    face = mesh.faces[face_index]
    m = len(face)
    nrm = _face_normal(V, face)
    base = mesh.n_vertices
    lifted = np.array([V[face[k]] + distance * nrm for k in range(m)])
    verts = np.vstack([V, lifted])
    new_idx = [base + k for k in range(m)]
    faces = [f for i, f in enumerate(mesh.faces) if i != face_index]   # drop the original face
    faces.append(tuple(new_idx))                                       # the cap (same winding)
    faces.extend(_ring_walls(face, new_idx))                           # the side walls
    return Mesh(verts, faces)


def inset_face(mesh, face_index, ratio):
    """INSET: shrink face `face_index` toward its centroid by `ratio` (0 = unchanged, 1 = collapsed to a point),
    ringing the original with new faces around a smaller central face. In-plane (no displacement), so the central
    face stays coplanar with the original; its area is exactly (1-ratio)^2 of the original. Returns a new Mesh."""
    V = mesh.vertices
    face = mesh.faces[face_index]
    m = len(face)
    centroid = np.mean([V[face[k]] for k in range(m)], axis=0)
    base = mesh.n_vertices
    inset = np.array([V[face[k]] + ratio * (centroid - V[face[k]]) for k in range(m)])   # toward the centroid
    verts = np.vstack([V, inset])
    new_idx = [base + k for k in range(m)]
    faces = [f for i, f in enumerate(mesh.faces) if i != face_index]
    faces.append(tuple(new_idx))                                       # the central (inset) face
    faces.extend(_ring_walls(face, new_idx))                           # the surrounding ring
    return Mesh(verts, faces)


def _order_link_loop(directed_edges):
    """Chain directed edges (a -> b) into the cyclic order of the link loop. The umbrella of a manifold vertex is
    consistently oriented, so the opposite edges (one per incident triangle) form a single directed cycle; follow
    the successor map from any start until it returns."""
    succ = {a: b for (a, b) in directed_edges}
    start = directed_edges[0][0]
    loop = [start]
    cur = succ.get(start)
    while cur is not None and cur != start:
        loop.append(cur)
        cur = succ.get(cur)
    return loop


def dissolve_vertex(mesh, vertex):
    """DISSOLVE a vertex (Euler KEV): remove `vertex` and its incident faces, then fan-triangulate the resulting
    hole, leaving the surrounding ring fixed. Preserves chi and keeps a closed mesh closed and manifold. Distinct
    from `collapse_edge` (which removes a vertex by merging it onto a neighbour -- the decimation cousin). Returns
    a new Mesh. Requires an interior manifold vertex whose link is a single loop."""
    incident = [i for i, f in enumerate(mesh.faces) if vertex in f]
    # the link: for each incident triangle (vertex, a, b) the OPPOSITE directed edge a -> b
    ring = []
    for i in incident:
        f = mesh.faces[i]
        m = len(f)
        for k in range(m):
            a = f[k]
            b = f[(k + 1) % m]
            if a != vertex and b != vertex:
                ring.append((a, b))
    loop = _order_link_loop(ring)
    if len(loop) < 3:
        raise ValueError(f"vertex {vertex} has a degenerate link; cannot dissolve")
    # fan-triangulate the link polygon from its first vertex (supplies each freed boundary edge once)
    fill = [(loop[0], loop[k], loop[k + 1]) for k in range(1, len(loop) - 1)]
    incident_set = set(incident)
    faces = [f for i, f in enumerate(mesh.faces) if i not in incident_set]
    faces.extend(fill)
    # drop the vertex and reindex everything above it
    keep = [i for i in range(mesh.n_vertices) if i != vertex]
    remap = {old: i for i, old in enumerate(keep)}
    new_verts = mesh.vertices[keep]
    new_faces = [tuple(remap[v] for v in f) for f in faces]
    return Mesh(new_verts, new_faces)


# =====================================================================================================
# Self-test -- each verb preserves chi + manifold + closed, with its exact geometric signature.
# =====================================================================================================
def _selftest():
    from holographic.mesh_and_geometry.holographic_meshsmooth import _icosphere

    sphere = _icosphere(2)                              # closed genus-0: chi = 2, manifold
    chi0 = sphere.euler_characteristic()
    assert chi0 == 2 and sphere.is_closed() and sphere.is_manifold()

    # --- EXTRUDE: chi preserved, still a closed manifold, cap moved exactly `distance` along the normal ---
    face = sphere.faces[0]
    nrm = _face_normal(sphere.vertices, face)
    cap_before = np.mean([sphere.vertices[v] for v in face], axis=0)
    ex = extrude_face(sphere, 0, distance=0.3)
    assert ex.euler_characteristic() == chi0, "extrude must preserve chi"
    assert ex.is_closed() and ex.is_manifold(), "extrude must keep the mesh a closed manifold"
    cap_after = ex.vertices[ex.n_vertices - 3:].mean(axis=0)   # the 3 lifted (cap) vertices
    moved = cap_after - cap_before
    assert abs(float(np.dot(moved, nrm)) - 0.3) < 1e-9, "cap moves exactly `distance` along the normal"
    assert float(np.linalg.norm(moved - np.dot(moved, nrm) * nrm)) < 1e-9, "...and only along the normal"

    # --- INSET: chi preserved, closed manifold, central-face area exactly (1-ratio)^2 of the original ---
    def tri_area(verts, f):
        a, b, c = verts[f[0]], verts[f[1]], verts[f[2]]
        return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
    area_before = tri_area(sphere.vertices, sphere.faces[0])
    ins = inset_face(sphere, 0, ratio=0.4)
    assert ins.euler_characteristic() == chi0 and ins.is_closed() and ins.is_manifold()
    central = ins.faces[sphere.n_faces - 1]            # the central face was appended right after dropping #0
    area_after = tri_area(ins.vertices, central)
    assert abs(area_after - (1 - 0.4) ** 2 * area_before) < 1e-9, "inset area = (1-ratio)^2 * original"

    # --- DISSOLVE a vertex: chi preserved, closed manifold, one fewer vertex ---
    diss = dissolve_vertex(sphere, vertex=5)
    assert diss.euler_characteristic() == chi0, "dissolve must preserve chi"
    assert diss.is_closed() and diss.is_manifold(), "dissolve must keep the mesh a closed manifold"
    assert diss.n_vertices == sphere.n_vertices - 1, "dissolve removes exactly one vertex"

    # --- determinism ---
    assert np.array_equal(extrude_face(sphere, 0, 0.3).vertices, extrude_face(sphere, 0, 0.3).vertices)

    print(f"holographic_meshverbs selftest: ok (extrude/inset/dissolve all preserve chi={chi0}, closed, manifold; "
          f"extrude cap moves exactly 0.300 along the normal; inset area = (1-ratio)^2 exactly; dissolve removes "
          f"1 vertex ({sphere.n_vertices} -> {diss.n_vertices}); deterministic)")


if __name__ == "__main__":
    _selftest()
