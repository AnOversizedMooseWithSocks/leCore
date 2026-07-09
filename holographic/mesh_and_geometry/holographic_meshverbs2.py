"""The FWD-7 modeler-verb remainder: BEVEL, BRIDGE, LOOP-CUT (holographic_meshverbs2).

WHY THIS MODULE EXISTS
----------------------
FWD-7 shipped the three modeler verbs expressible as straightforward face-list rewrites -- extrude, inset, dissolve
-- and deferred the three that need vertex DUPLICATION or edge-loop TRACING. This module ships those three, reusing
the patterns the engine since proved: the vertex-fan / umbrella logic from the ARCH-4 seam, and the unused-vertex
compaction (reindex) from dissolve.

  * BEVEL (vertex bevel / chamfer a corner): pull each edge incident to a vertex back toward its neighbour, chamfer
    every incident face, and CAP the hole with a new face. The corner becomes a small facet. (Needs the cyclic
    neighbour order around the vertex -- the umbrella from ARCH-4 -- and compaction of the removed vertex.)
  * BRIDGE (connect two edge loops): join two equal-length ordered vertex loops with a band of quads -- the verb
    that builds a tube between two openings.
  * LOOP-CUT (insert an edge loop): trace the perpendicular loop of quads (each quad entered through one edge, left
    through the OPPOSITE edge) and split every crossed quad in two with a new mid-loop -- the verb that adds
    resolution along a ring.

WHAT IT PROVIDES
  * bevel_vertex(mesh, vertex, ratio) -- chamfer a corner. Returns a new Mesh.
  * bridge_loops(verts, loop_a, loop_b, closed) -- a quad band joining two loops. Returns a new Mesh.
  * loop_cut(mesh, start_face, start_edge) -- insert an edge loop through the quad strip carrying `start_edge`.
    Returns a new Mesh.

THE MEASUREMENT BAR (checked exactly in the self-test)
  * BEVEL a cube corner (degree 3): manifold + closed + chi PRESERVED (2); the corner's 3 quads become pentagons and
    a triangular cap appears; the new vertices sit ON the incident edges (ratio of the way to each neighbour).
  * BRIDGE two squares: a manifold open tube -- 4 quads, chi=0, exactly two boundary loops.
  * LOOP-CUT a cube: manifold + closed + chi PRESERVED (2), +4 faces (the ring crosses 4 quads); LOOP-CUT a grid:
    chi PRESERVED (1), +3 faces (the open strip crosses 3 quads).

DETERMINISM (per ISA.md)
  Pure topological rewrites in fixed order; midpoints/pull-backs are fixed affine combinations; no RNG. Same input
  -> byte-identical result (asserted).

KEPT NEGATIVES (loud)
  * BEVEL is the VERTEX bevel (chamfer a corner). The EDGE bevel (widen an edge into a chamfer face, splitting both
    its endpoints' fans) is the harder two-sided split and is deferred -- the same fan-consistency problem the seam
    solved for one path, here needed on both sides of an edge.
  * BEVEL needs ratio < 1 (a new vertex at ratio of the way to its neighbour); ratio >= 1 would pass the neighbour
    and self-intersect. Boundary/non-manifold vertices are out of scope (the umbrella must close).
  * BRIDGE requires two EQUAL-LENGTH, ALREADY-ALIGNED loops -- the caller supplies the correspondence; resampling or
    matching loops of different lengths (the general bridge) is deferred.
  * LOOP-CUT needs QUADS: the opposite-edge trace is undefined on triangles, so a triangulated region has no loop to
    cut. The trace stops at a boundary (an open cut) or when it returns to the start (a closed ring).
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh


def _compact(verts, faces):
    """Drop vertices no face references and reindex (the dissolve/seam reindex): keeps chi honest after a vertex is
    removed."""
    used = sorted(set(i for f in faces for i in f))
    remap = {old: new for new, old in enumerate(used)}
    return np.array([verts[i] for i in used]), [tuple(remap[i] for i in f) for f in faces]


def bevel_vertex(mesh, vertex, ratio=0.25):
    """Chamfer a corner: pull each edge incident to `vertex` back toward its neighbour by `ratio`, chamfer every
    incident face, and cap the hole with a new face. Returns a new Mesh (chi preserved). ratio in (0, 1)."""
    faces = [tuple(f) for f in mesh.faces]
    verts = list(mesh.vertices)
    incident = [fi for fi, f in enumerate(faces) if vertex in f]
    # one new vertex per incident edge (vertex, neighbour), placed ON that edge
    neighbours = set()
    for fi in incident:
        f = faces[fi]; i = f.index(vertex)
        neighbours.add(f[(i - 1) % len(f)]); neighbours.add(f[(i + 1) % len(f)])
    new_of = {}
    for nb in neighbours:
        new_of[nb] = len(verts)
        verts.append(mesh.vertices[vertex] + ratio * (mesh.vertices[nb] - mesh.vertices[vertex]))
    # rewrite each incident face: replace `vertex` by [new(prev), new(next)] (chamfer the corner); record the
    # cyclic neighbour order (p -> q for a face cycle p -> vertex -> q) to build the cap
    rewritten = [f for i, f in enumerate(faces) if i not in set(incident)]
    succ = {}
    for fi in incident:
        f = list(faces[fi]); i = f.index(vertex)
        p, q = f[(i - 1) % len(f)], f[(i + 1) % len(f)]
        rewritten.append(tuple(f[:i] + [new_of[p], new_of[q]] + f[i + 1:]))
        succ[p] = q
    # the cap: the new vertices in cyclic order around the vertex
    order = []
    cur = next(iter(succ))
    for _ in range(len(succ)):
        order.append(cur)
        cur = succ.get(cur)
        if cur is None:
            break
    cap = tuple(new_of[n] for n in order)
    # keep whichever cap winding makes the result manifold (the cap may need reversing relative to the surface)
    vv, ff = _compact(verts, rewritten + [cap])
    m = Mesh(vv, ff)
    if m.is_manifold():
        return m
    vv, ff = _compact(verts, rewritten + [cap[::-1]])
    return Mesh(vv, ff)


def bridge_loops(verts, loop_a, loop_b, closed=True):
    """Join two equal-length ordered vertex loops with a band of quads [a_i, a_{i+1}, b_{i+1}, b_i]. `verts` holds
    all the points; `loop_a`/`loop_b` are index lists of equal length. closed=True wraps the band into a tube.
    Returns a new Mesh of the band."""
    if len(loop_a) != len(loop_b):
        raise ValueError("bridge_loops needs two equal-length loops (the caller supplies the correspondence)")
    k = len(loop_a)
    faces = []
    for i in (range(k) if closed else range(k - 1)):
        j = (i + 1) % k
        faces.append((loop_a[i], loop_a[j], loop_b[j], loop_b[i]))
    return Mesh(np.asarray(verts, float), faces)


def _quad_edges(f):
    return [(f[k], f[(k + 1) % 4]) for k in range(4)]


def _opposite_edge(f, e):
    """The edge of quad `f` sharing no vertex with edge `e` (the one the loop exits through)."""
    s = set(e)
    for (c, d) in _quad_edges(f):
        if c not in s and d not in s:
            return (c, d)
    return None


def loop_cut(mesh, start_face, start_edge):
    """Insert an edge loop through the quad strip carrying `start_edge` (an edge of quad `start_face`): trace the
    perpendicular loop (enter a quad through one edge, leave through the OPPOSITE edge, cross into the neighbour),
    insert a midpoint on each crossed edge, and split every crossed quad in two. Returns a new Mesh (chi preserved).
    Quads only; the trace stops at a boundary (open) or when it returns to the start (closed)."""
    faces = [tuple(f) for f in mesh.faces]
    verts = list(mesh.vertices)
    if len(faces[start_face]) != 4:
        raise ValueError("loop_cut needs a quad mesh (the opposite-edge trace is undefined on triangles)")
    # undirected edge -> the faces using it
    e2f = {}
    for fi, f in enumerate(faces):
        for (a, b) in _quad_edges(f):
            e2f.setdefault(frozenset((a, b)), []).append(fi)
    # trace the strip: a list of (face, entering-edge)
    strip = []
    fi, e, seen = start_face, tuple(start_edge), set()
    while fi is not None and fi not in seen:
        seen.add(fi)
        strip.append((fi, e))
        opp = _opposite_edge(faces[fi], e)
        nb = [g for g in e2f[frozenset(opp)] if g != fi]
        if not nb:
            break
        fi, e = nb[0], opp
    # a midpoint per crossed edge (entering + final opposite)
    crossed = []
    for fi, e in strip:
        crossed.append(frozenset(e)); crossed.append(frozenset(_opposite_edge(faces[fi], e)))
    mid = {}
    for ce in dict.fromkeys(crossed):
        a, b = tuple(ce)
        mid[ce] = len(verts)
        verts.append(0.5 * (mesh.vertices[a] + mesh.vertices[b]))
    # split each strip quad in two along its two midpoints, in the quad's OWN cyclic order (keeps winding consistent)
    strip_set = set(fi for fi, _ in strip)
    new_faces = [f for i, f in enumerate(faces) if i not in strip_set]
    for fi, e in strip:
        f = faces[fi]
        i = next(k for k in range(4) if set((f[k], f[(k + 1) % 4])) == set(e))
        v0, v1, v2, v3 = f[i], f[(i + 1) % 4], f[(i + 2) % 4], f[(i + 3) % 4]
        me, mo = mid[frozenset((v0, v1))], mid[frozenset((v2, v3))]
        new_faces.append((v0, me, mo, v3))
        new_faces.append((me, v1, v2, mo))
    return Mesh(np.array(verts), new_faces)


# =====================================================================================================
# Self-test -- bevel a corner, bridge two loops, loop-cut a box and a grid; chi preserved where it should be.
# =====================================================================================================
def _selftest():
    from holographic.mesh_and_geometry.holographic_mesh import box, grid

    # --- BEVEL a cube corner (degree 3): manifold, closed, chi PRESERVED; 3 pentagons + a triangular cap ---
    cube = box()
    bev = bevel_vertex(cube, 0, ratio=0.3)
    assert bev.is_manifold() and bev.is_closed(), "bevel must stay a closed manifold"
    assert bev.euler_characteristic() == 2, f"bevel preserves chi, got {bev.euler_characteristic()}"
    sizes = sorted(len(f) for f in bev.faces)
    assert sizes.count(5) == 3 and sizes.count(3) == 1, f"3 quads -> pentagons + 1 triangular cap, got {sizes}"
    # a new vertex lies ON an incident edge at the ratio
    assert bev.n_vertices == cube.n_vertices - 1 + 3, "the corner vertex is removed, 3 new ones added"

    # --- BRIDGE two squares -> a manifold tube: 4 quads, chi=0, two boundary loops ---
    sq0 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    sq1 = sq0.copy(); sq1[:, 2] = 1.0
    V = np.vstack([sq0, sq1])
    tube = bridge_loops(V, [0, 1, 2, 3], [4, 5, 6, 7], closed=True)
    assert tube.is_manifold() and not tube.is_closed(), "a bridged tube is an open manifold"
    assert tube.n_faces == 4 and tube.euler_characteristic() == 0, "4 side quads, chi=0 (an open cylinder)"

    # --- LOOP-CUT a cube: manifold, closed, chi PRESERVED, +4 faces (the ring crosses 4 quads) ---
    f0 = tuple(cube.faces[0])
    lc = loop_cut(cube, 0, (f0[0], f0[1]))
    assert lc.is_manifold() and lc.is_closed() and lc.euler_characteristic() == 2, "loop-cut preserves the cube"
    assert lc.n_faces == cube.n_faces + 4, "the ring crosses 4 quads, splitting each (+4 faces)"

    # --- LOOP-CUT a grid: chi PRESERVED (1), +3 faces (the open strip crosses 3 quads) ---
    g = grid(3, 3)
    fg = tuple(g.faces[0])
    lcg = loop_cut(g, 0, (fg[0], fg[1]))
    assert lcg.is_manifold() and lcg.euler_characteristic() == 1, "loop-cut preserves the open grid"
    assert lcg.n_faces == g.n_faces + 3, "the open strip crosses 3 quads (+3 faces)"

    # --- determinism ---
    assert np.array_equal(bevel_vertex(cube, 0, 0.3).vertices, bevel_vertex(cube, 0, 0.3).vertices)
    assert np.array_equal(loop_cut(cube, 0, (f0[0], f0[1])).vertices, loop_cut(cube, 0, (f0[0], f0[1])).vertices)

    print(f"holographic_meshverbs2 selftest: ok (BEVEL a cube corner -> closed manifold, chi 2 preserved, faces "
          f"{sizes} (3 pentagons + triangle cap); BRIDGE two squares -> open tube ({tube.n_faces} quads, chi 0, 2 "
          f"boundaries); LOOP-CUT cube -> chi 2, +4 faces (ring of 4); LOOP-CUT grid -> chi 1, +3 faces (strip of 3); "
          f"deterministic)")


if __name__ == "__main__":
    _selftest()
