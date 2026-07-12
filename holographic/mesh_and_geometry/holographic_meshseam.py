"""Seam cutting / atlas (ARCH-4): open a closed surface along a seam so it can be unwrapped -- a REAL FWD-3 seam.

WHY THIS MODULE EXISTS
----------------------
FWD-3 (UV unwrapping) shipped with a kept negative: a CLOSED surface has no boundary and cannot be flattened to a
disk without a CUT, and FWD-3's only opener was `puncture` -- remove one vertex, leaving a tiny hole, which
unwraps badly (everything crams toward the puncture's antipode). ARCH-4 pays that back with the real thing: cut
the surface along a SEAM (a path of edges) by DUPLICATING the seam's interior vertices, opening it into a disk
that unwraps far more faithfully. This is the vertex-duplication that FWD-3 deliberately deferred, and the first
piece of an atlas (a surface covered by chart(s) with seams between them).

THE TOPOLOGY (why a single arc opens a sphere into a disk)
  Cutting a closed genus-0 surface (chi = 2) along a simple ARC of k edges duplicates its k-1 INTERIOR vertices
  (the two endpoints stay single -- they are the pinch points where the cut ends) and splits each of the k seam
  edges into two boundary edges. So dV = +(k-1), dE = +k, dF = 0, and dchi = (k-1) - k = -1: chi 2 -> 1, a DISK.
  The self-test confirms exactly this (chi = 1, one boundary loop, still a manifold).

THE SUBTLE PART: A CONSISTENT CUT
  A seam arc does NOT separate the surface (it is still one piece after cutting), so you cannot 2-colour faces
  into "left"/"right" globally. The sides are LOCAL. The fix is to ORIENT the seam (v0 -> v1 -> ... -> vk) and, at
  every interior vertex, duplicate the fan of faces on the side that matches the path direction -- the fan
  containing the face that carries the directed edge (v_i -> v_{i+1}). Because that side is defined by the single
  path orientation, the duplicated side is consistent all along the seam, so the two lips of the cut line up and
  the result is a manifold. (Getting this wrong gives a non-manifold mess -- it is the whole reason FWD-3 deferred it.)

WHAT IT PROVIDES
  * cut_seam(mesh, seam) -- cut a mesh open along `seam` (an ordered list of vertex indices forming an edge path),
    duplicating interior seam vertices on a consistent side. Returns a new (open) Mesh.
  * shortest_seam(mesh, a, b) -- a seam path between two vertices: the shortest edge path (Dijkstra), e.g. a
    meridian from a pole to its antipode.

THE MEASUREMENT BAR (checked exactly in the self-test)
  * cut_seam(sphere, meridian) gives a DISK: chi = 1, NOT closed, still a manifold, with exactly ONE boundary
    loop, and V grown by (interior seam vertices).
  * THE ROBUST PAYBACK (always true): the cut is NON-DESTRUCTIVE -- it preserves EVERY face, where the puncture
    DELETES the vertex's incident faces (losing geometry). A real seam keeps the whole surface.
  * THE DISTORTION PAYBACK (with a good seam): a well-chosen seam (pole-to-equator) UV-unwraps with LOWER
    distortion than the crude puncture -- measured.

DETERMINISM (per ISA.md)
  Dijkstra ties break on vertex index; the cut visits seam vertices and faces in fixed order; duplicates are
  appended deterministically. Same mesh + same seam -> byte-identical result (asserted).

KEPT NEGATIVES (loud)
  * SEAM CHOICE MATTERS, measured: a full pole-to-pole meridian opens a valid disk but unwraps WORSE than the
    puncture (it makes a long thin lune), while a pole-to-equator cut beats it. One cut never makes a sphere
    unwrap WELL -- Gauss forbids it -- so a good atlas uses several cuts / multiple charts (the rest of ARCH-4,
    deferred). The win here is "non-destructive, and beats the puncture with a sensible seam", not "distortion-free".
  * The seam must be a simple edge path (no self-crossings); a branching cut graph or a seam through a non-
    manifold vertex is out of scope.
"""

import heapq

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh


def shortest_seam(mesh, a, b):
    """The shortest edge path from vertex `a` to vertex `b` (Dijkstra on the mesh edge graph with Euclidean
    weights), returned as an ordered list of vertex indices -- a seam to cut along (e.g. a meridian, pole to
    antipode). Ties break on vertex index for determinism."""
    V = mesh.vertices
    adj = {v: [] for v in range(mesh.n_vertices)}
    for (lo, hi) in mesh.edges():
        d = float(np.linalg.norm(V[lo] - V[hi]))
        adj[lo].append((hi, d))
        adj[hi].append((lo, d))
    dist = {a: 0.0}
    prev = {}
    pq = [(0.0, int(a))]
    while pq:
        d, u = heapq.heappop(pq)
        if u == b:
            break
        if d > dist.get(u, np.inf):
            continue
        for (v, w) in adj[u]:
            nd = d + w
            if nd < dist.get(v, np.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    # reconstruct
    path = [int(b)]
    while path[-1] != a:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def _face_with_directed_edge(faces, u, w):
    """The index of the face that contains the directed edge u -> w (u immediately followed by w in winding
    order). None if no face does (a boundary edge in that direction)."""
    for fi, f in enumerate(faces):
        n = len(f)
        for k in range(n):
            if f[k] == u and f[(k + 1) % n] == w:
                return fi
    return None


def _components_of_incident_faces(faces, incident, vi, seam_neighbours):
    """Connected components of vi's incident faces, where two faces are adjacent if they share an edge (vi, x)
    that is NOT a seam edge. For an interior manifold vertex with two seam edges, this yields the two fans."""
    # edge (vi, x) -> faces using it
    edge_faces = {}
    for fi in incident:
        f = faces[fi]
        n = len(f)
        for k in range(n):
            if f[k] == vi:
                for x in (f[(k - 1) % n], f[(k + 1) % n]):
                    edge_faces.setdefault(x, []).append(fi)
    adj = {fi: set() for fi in incident}
    for x, fs in edge_faces.items():
        if x in seam_neighbours:
            continue                                       # do NOT connect across a seam edge (it is a cut)
        fs = list(dict.fromkeys(fs))                       # unique, order-stable
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                adj[fs[i]].add(fs[j])
                adj[fs[j]].add(fs[i])
    # flood fill
    comps = []
    seen = set()
    for start in incident:
        if start in seen:
            continue
        stack = [start]
        comp = set()
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            comp.add(c)
            stack.extend(adj[c] - seen)
        comps.append(comp)
    return comps


def cut_seam(mesh, seam):
    """Cut `mesh` open along `seam` (an ordered list of vertex indices forming an edge path) by duplicating the
    seam's INTERIOR vertices on a consistent side, opening the surface into a disk. Returns a new Mesh."""
    faces = [tuple(f) for f in mesh.faces]
    verts = list(mesh.vertices)
    # incident faces per vertex
    incident = {v: [] for v in range(mesh.n_vertices)}
    for fi, f in enumerate(faces):
        for v in set(f):
            incident[v].append(fi)

    reassign = {}                                          # (face index, original vertex) -> duplicate vertex
    for i in range(1, len(seam) - 1):                      # interior seam vertices only
        vi = seam[i]
        prev_v, next_v = seam[i - 1], seam[i + 1]
        comps = _components_of_incident_faces(faces, incident[vi], vi, {prev_v, next_v})
        if len(comps) < 2:
            continue                                       # not actually split here (degenerate) -- skip safely
        # the side to duplicate: the fan containing the face carrying the directed path edge (vi -> next_v)
        f_out = _face_with_directed_edge(faces, vi, next_v)
        dup_fan = next((c for c in comps if f_out in c), comps[0])
        dup_idx = len(verts)
        verts.append(mesh.vertices[vi].copy())
        for fi in dup_fan:
            reassign[(fi, vi)] = dup_idx

    # rebuild faces with the per-corner reassignment
    new_faces = []
    for fi, f in enumerate(faces):
        new_faces.append(tuple(reassign.get((fi, v), v) for v in f))
    return Mesh(np.array(verts), new_faces)


def _boundary_loop_count(mesh):
    """How many boundary loops the mesh has (a boundary edge is in exactly one face). A disk has exactly one."""
    edge_count = {}
    for f in mesh.faces:
        n = len(f)
        for k in range(n):
            e = frozenset((f[k], f[(k + 1) % n]))
            edge_count[e] = edge_count.get(e, 0) + 1
    boundary = [tuple(e) for e, c in edge_count.items() if c == 1]
    if not boundary:
        return 0
    # walk the boundary edges into loops
    adj = {}
    for (a, b) in boundary:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen_e = set()
    loops = 0
    for (a, b) in boundary:
        if frozenset((a, b)) in seen_e:
            continue
        loops += 1
        # traverse this loop
        prev, cur = a, b
        seen_e.add(frozenset((a, b)))
        while True:
            nxts = [x for x in adj[cur] if x != prev and frozenset((cur, x)) not in seen_e]
            if not nxts:
                break
            nxt = nxts[0]
            seen_e.add(frozenset((cur, nxt)))
            prev, cur = cur, nxt
            if cur == a:
                break
    return loops


def auto_seam(mesh, threshold_deg=40.0, method="crease"):
    """AUTO-MARK SEAMS: choose which edges to cut for UV unwrapping, WITHOUT the caller naming a path. Returns a
    sorted list of (lo,hi) seam edges -- the 'marked seams' (the red edges a modeler sees), ready to feed a cut.
    `mesh_cut_seam`/`shortest_seam` cut a GIVEN seam; this SELECTS one.

    method='crease' (default): seam along the SHARP edges (interior edges whose dihedral angle exceeds
    `threshold_deg`) -- the standard heuristic, since an artist cuts where the surface already folds, so the cut is
    hidden and each side unwraps with little distortion. Composes holographic_meshcurvature.detect_creases (reuse,
    don't reimplement the dihedral test). On a cube every 90-deg edge is marked; on a smooth sphere none are (the
    kept negative: a smooth closed surface has no creases, so crease-seaming leaves it closed -- use a shortest_seam
    meridian there instead; auto_seam reports the empty set honestly rather than inventing a cut)."""
    if method != "crease":
        raise ValueError("auto_seam: only method='crease' is implemented; a smooth surface needs shortest_seam")
    from holographic.mesh_and_geometry.holographic_meshcurvature import detect_creases
    return detect_creases(mesh, threshold_deg=threshold_deg)   # the sharp edges ARE the seam marks


# =====================================================================================================
# Self-test -- a meridian cut opens a sphere into a disk, and beats the crude puncture on unwrap distortion.
# =====================================================================================================
def _selftest():
    from holographic.mesh_and_geometry.holographic_meshsmooth import _icosphere
    from holographic.mesh_and_geometry.holographic_meshuv import uv_unwrap, uv_distortion, puncture

    sphere = _icosphere(3)
    assert sphere.is_closed() and sphere.euler_characteristic() == 2
    north = int(np.argmax(sphere.vertices[:, 2]))
    south = int(np.argmin(sphere.vertices[:, 2]))
    meridian = shortest_seam(sphere, north, south)         # a full pole-to-pole meridian
    assert len(meridian) >= 3 and meridian[0] == north and meridian[-1] == south

    # --- the cut opens the sphere into a DISK: chi=1, open, manifold, ONE boundary loop, GEOMETRY PRESERVED ---
    disk = cut_seam(sphere, meridian)
    assert disk.is_manifold(), "the cut must stay a manifold (the consistent-side requirement)"
    assert not disk.is_closed(), "the cut surface has a boundary"
    assert disk.euler_characteristic() == 1, f"a meridian-cut sphere is a disk (chi=1), got {disk.euler_characteristic()}"
    assert _boundary_loop_count(disk) == 1, "a disk has exactly one boundary loop"
    assert disk.n_vertices == sphere.n_vertices + (len(meridian) - 2), "interior seam vertices duplicated"

    # --- THE ROBUST PAYBACK (always true): the cut PRESERVES all geometry; the puncture DELETES faces ---
    punct = puncture(sphere, vertex=0)
    assert disk.n_faces == sphere.n_faces, "cut_seam preserves every face (non-destructive)"
    assert punct.n_faces < sphere.n_faces, "...whereas puncture deletes the vertex's incident faces"

    # --- THE DISTORTION PAYBACK: a WELL-CHOSEN seam unwraps with LOWER distortion than the crude puncture ---
    equator = int(np.argmin(np.abs(sphere.vertices[:, 2])))
    good = cut_seam(sphere, shortest_seam(sphere, north, equator))   # a pole-to-equator seam
    good_dist = uv_distortion(good, uv_unwrap(good))
    punct_dist = uv_distortion(punct, uv_unwrap(punct))
    assert good_dist < punct_dist, f"a well-chosen seam beats the puncture on unwrap: {good_dist:.3f} vs {punct_dist:.3f}"

    # --- KEPT NEGATIVE (loud): seam choice MATTERS -- a full pole-to-pole meridian does NOT beat the puncture ---
    full_dist = uv_distortion(disk, uv_unwrap(disk))
    assert full_dist > punct_dist, "the full meridian is a worse cut than the half-meridian (seam choice matters)"

    # --- determinism ---
    assert np.array_equal(cut_seam(sphere, meridian).vertices, cut_seam(sphere, meridian).vertices)

    # --- AUTO_SEAM: crease-based seam marking. A cube's 12 sharp edges get marked; a smooth sphere gets NONE ---
    from holographic.mesh_and_geometry.holographic_mesh import box
    cube = Mesh(box(2.0, 2.0, 2.0).vertices.copy(), [tuple(t) for t in box(2.0, 2.0, 2.0).triangulate()])
    cube_seams = auto_seam(cube, threshold_deg=40.0)
    assert len(cube_seams) >= 12, "a cube's 12 sharp (90-deg) edges must be auto-marked as seams"
    assert auto_seam(cube) == auto_seam(cube), "auto_seam is deterministic"
    sphere_seams = auto_seam(sphere, threshold_deg=40.0)      # the smooth sphere has no creases
    assert sphere_seams == [], "KEPT NEGATIVE: a smooth closed surface has no creases -- auto_seam reports empty, " \
                              "not an invented cut (use a shortest_seam meridian there)"

    print(f"holographic_meshseam selftest: ok (meridian cut opens the sphere into a DISK: chi=1, one boundary "
          f"loop, manifold, ALL {disk.n_faces} faces preserved (V {sphere.n_vertices}->{disk.n_vertices}, +{len(meridian) - 2} dup); "
          f"PAYBACK -- cut preserves geometry where puncture deletes {sphere.n_faces - punct.n_faces} faces, and a "
          f"well-chosen seam unwraps at {good_dist:.3f} < puncture {punct_dist:.3f}; KEPT NEGATIVE -- a full meridian "
          f"({full_dist:.3f}) is a worse cut, seam choice matters; auto_seam marks {len(cube_seams)} crease edges on a "
          f"cube, empty on a smooth sphere; deterministic)")


if __name__ == "__main__":
    _selftest()
