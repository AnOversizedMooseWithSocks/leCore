"""Local Euler edit operators on the explicit mesh (FWD-7): the invariant-preserving rewrites a modeler runs.

WHY THIS MODULE EXISTS
----------------------
holographic_mesh.py (FWD-1) ships the explicit-geometry SUBSTRATE -- a `Mesh` with half-edge adjacency and
the Euler well-formedness invariants -- but it is effectively read-only: it can hold and measure a mesh, not
EDIT one. Every higher modeling operation a DCC tool exposes (subdivide, bevel, decimate / LOD, remesh,
knife) decomposes into a small set of LOCAL connectivity rewrites known since Baumgart and Mantyla as the
*Euler operators*: each adds or removes a bounded patch of vertices / edges / faces while keeping the surface
a valid manifold and bookkeeping the Euler characteristic exactly. This module is that edit layer -- the
foundation the kernel docstring named ("the foundation Euler edit operators (FWD-7) will mutate") -- built
native-first on the shipped half-edge kernel so the explicit-geometry stack can finally CHANGE a mesh, not
just describe one.

WHY THESE FOUR (and the make/kill pairing)
  Euler operators come in inverse make/kill PAIRS; a correct pair gives an exact round-trip, which is the
  cleanest correctness witness there is -- do-then-undo must return bit-identical connectivity. The four here
  are the workhorses every remesher / decimator is built from:
    * flip_edge     -- rotate the shared edge of two triangles. chi / V / E / F all unchanged: the simplest
                       rewrite, so the strongest first test that adjacency surgery stays manifold. The
                       Delaunay-remeshing primitive. (Its own inverse: flipping the new edge restores the old.)
    * split_edge    -- insert a vertex on an edge, splitting the incident triangle(s). V+1, chi unchanged.
                       The refinement primitive (subdivision, adaptive detail).
    * collapse_edge -- the INVERSE of split_edge: merge an edge's two endpoints into one. V-1, chi unchanged.
                       The decimation / LOD primitive -- and the one with a real precondition (the LINK
                       CONDITION) without which it would tear the manifold. Guarded and documented below.
    * split_face    -- cut a polygon face with a diagonal between two of its corners. E+1, F+1, chi unchanged.
                       The one operator that works on n-gons, not just triangles (MEF, "make edge-face").

  Higher operations (subdivision, bevel, full decimation) are SEQUENCES of these; they are later items. This
  module ships the verified primitives they will stand on, on the project's own "prove the foundation first"
  rule -- the same discipline that put the mesh kernel before the bridge.

DESIGN: FACE-LIST REWRITE, NOT IN-PLACE HALF-EDGE SURGERY
  Each operator returns a NEW `Mesh` (immutable style). It uses the half-edge adjacency to FIND the local
  patch (which faces share this edge, what are the opposite apexes), then emits a rewritten face list and
  lets the new `Mesh` rebuild its own half-edge table. This is deliberately the readable choice over mutating
  the parallel half-edge arrays in place: the combinatorics stay legible (you can read off the new faces),
  and there is no cache to invalidate and no twin pointer to fix up by hand. The cost is a half-edge rebuild
  per edit -- acceptable, and consistent with the kernel's already-recorded "NumPy is the wrong tool for
  per-element mesh loops" negative.

DETERMINISM (per ISA.md)
  A new vertex is always APPENDED (its index is the old vertex count); faces are rewritten in face order;
  vertex removal reindexes by one fixed rule (drop the removed index, decrement every higher index). So every
  operator is a pure deterministic function of (mesh, selection) -- same in, byte-identical out (asserted in
  the self-test). Every output is the EXACT (combinatorial) class: no float comparison ever chooses
  connectivity here.

KEPT NEGATIVES (loud)
  * collapse_edge is NOT always legal. Collapsing an edge whose two endpoints share a neighbour OTHER than the
    two triangle apexes of that edge would weld the surface onto itself and produce a non-manifold result.
    This is the classic LINK CONDITION; the operator CHECKS it and refuses (returns None) rather than
    silently emitting a broken mesh. Not every edge is collapsible -- that is a true property of meshes, not a
    shortcoming of the code, and a caller (a decimator) must handle the refusal.
  * flip_edge / split_edge / collapse_edge assume TRIANGLE faces on the touched faces (the operators are only
    well defined there) and raise otherwise. split_face is the n-gon operator. Triangulate first if needed.
  * The same per-element Python-loop bound as the kernel: fine for interactive single edits, not for millions
    of operations per second. A compiled core remains the eventual need for heavy remeshing.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh


# =====================================================================================================
# Small adjacency helpers (read the local patch off the face list / half-edge structure).
# =====================================================================================================
def _face_with_directed_edge(faces, a, b):
    """Index of the face that traverses the DIRECTED edge a->b (a then b, consecutively and cyclically), or
    None. On a consistently-oriented manifold a directed edge belongs to at most one face, so this resolves
    the two sides of an undirected edge unambiguously."""
    for fi, f in enumerate(faces):
        n = len(f)
        for k in range(n):
            if f[k] == a and f[(k + 1) % n] == b:
                return fi
    return None


def _third(tri, a, b):
    """The remaining vertex of a triangle once a and b are removed (its apex across edge {a,b})."""
    rest = [v for v in tri if v != a and v != b]
    if len(rest) != 1:
        raise ValueError(f"{tri} is not a triangle containing exactly the edge {{{a},{b}}}")
    return rest[0]


# =====================================================================================================
# flip_edge -- rotate the shared edge of two triangles (chi / V / E / F invariant).
# =====================================================================================================
def flip_edge(mesh, a, b):
    """Rotate the edge shared by the two triangles on undirected edge {a,b}: remove edge a-b and add edge c-d,
    where c and d are the opposite apexes. Requires {a,b} to be an INTERIOR edge of two TRIANGLE faces.
    Returns a new `Mesh`. V, E and F are all unchanged, so chi is unchanged -- the purest connectivity
    rewrite. Flipping the resulting edge c-d restores the original connectivity (the operator's inverse).
    Refuses (raises) if the new diagonal c-d is already an edge, since that would create a non-manifold edge.
    """
    faces = list(mesh.faces)
    f1 = _face_with_directed_edge(faces, a, b)        # the triangle that goes a -> b
    f2 = _face_with_directed_edge(faces, b, a)        # the triangle that goes b -> a
    if f1 is None or f2 is None:
        raise ValueError(f"edge {{{a},{b}}} is not an interior edge of two faces (cannot flip)")
    if len(faces[f1]) != 3 or len(faces[f2]) != 3:
        raise ValueError("flip_edge requires both incident faces to be triangles")
    c = _third(faces[f1], a, b)                        # apex of the a->b triangle
    d = _third(faces[f2], a, b)                        # apex of the b->a triangle
    if c == d:
        raise ValueError("degenerate flip: the two faces share an apex")
    # PRECONDITION (kept loud): the new diagonal c-d must not ALREADY be an edge, or it would end up shared by
    # three faces -- a non-manifold result. Flipping into an existing edge is illegal; refuse it explicitly.
    if (min(c, d), max(c, d)) in set(mesh.edges()):
        raise ValueError(f"flip_edge would duplicate existing edge {{{c},{d}}} (illegal: non-manifold)")
    # The union is a quad whose boundary runs a -> d -> b -> c -> a; the new diagonal is c-d. Splitting that
    # quad on c-d gives these two consistently-wound triangles (verified manifold in the self-test).
    out = [f for fi, f in enumerate(faces) if fi != f1 and fi != f2]
    out.append((a, d, c))
    out.append((d, b, c))
    return Mesh(mesh.vertices.copy(), out)


# =====================================================================================================
# split_edge -- insert a midpoint vertex, splitting the incident triangle(s) (V+1, chi invariant).
# =====================================================================================================
def _split_triangle_on_edge(tri, a, b, m):
    """Split triangle `tri` (which contains edge {a,b}) into two triangles meeting at the new vertex m,
    PRESERVING winding. If the triangle traverses a->b it becomes (a,m,c)+(m,b,c); if b->a, (b,m,c)+(m,a,c)
    -- i.e. m is spliced into whichever direction the original edge ran, so the outer boundary is unchanged."""
    c = _third(tri, a, b)
    seq = list(tri)
    nxt = seq[(seq.index(a) + 1) % 3]
    if nxt == b:                                      # the triangle runs ... a -> b ...
        return [(a, m, c), (m, b, c)]
    return [(b, m, c), (m, a, c)]                     # the triangle runs ... b -> a ...


def split_edge(mesh, a, b):
    """Insert a new vertex at the MIDPOINT of edge {a,b} and split each incident TRIANGLE into two. Works for
    an interior edge (2 triangles -> 4) or a boundary edge (1 triangle -> 2). Returns (new_mesh, m) where m is
    the new vertex index (always the last vertex, so the index is deterministic). V+1 and chi is unchanged.
    Exact inverse: collapse_edge(new_mesh, keep=a, remove=m) restores the original mesh.
    """
    faces = list(mesh.faces)
    incident = [fi for fi, f in enumerate(faces) if (a in f and b in f)]
    if not incident:
        raise ValueError(f"edge {{{a},{b}}} bounds no face")
    for fi in incident:
        if len(faces[fi]) != 3:
            raise ValueError("split_edge requires the incident faces to be triangles")

    m = mesh.n_vertices                               # appended index -> deterministic
    midpoint = (mesh.vertices[a] + mesh.vertices[b]) / 2.0
    new_verts = np.vstack([mesh.vertices, midpoint])

    out = []
    for fi, f in enumerate(faces):
        if fi in incident:
            out.extend(_split_triangle_on_edge(f, a, b, m))
        else:
            out.append(f)
    return Mesh(new_verts, out), m


# =====================================================================================================
# collapse_edge -- merge an edge's endpoints (V-1, chi invariant) -- the decimation primitive, GUARDED.
# =====================================================================================================
def collapse_edge(mesh, keep, remove):
    """Merge vertex `remove` into vertex `keep`: the edge {keep, remove} contracts to the single vertex
    `keep` (which stays at its position). The decimation / LOD primitive and the exact inverse of split_edge.
    V-1, chi unchanged.

    Returns the new `Mesh`, or None if the collapse would break the manifold -- the LINK CONDITION: `keep` and
    `remove` may share neighbours ONLY at the apexes of the faces on edge {keep, remove}. Any other common
    neighbour means contracting the edge would fold the surface onto itself along a new non-manifold edge, so
    the operator REFUSES (returns None) rather than emit a broken mesh. (This is a true property of the mesh,
    not a limitation of the code; a decimator must handle the refusal -- the kept negative made operational.)
    """
    a, b = keep, remove
    faces = list(mesh.faces)
    if not any(a in f and b in f for f in faces):
        raise ValueError(f"{{{a},{b}}} is not an edge of the mesh")

    # --- the link condition: shared neighbours must be exactly the apexes of the shared faces ---
    shared = set(mesh.vertex_neighbours(a)) & set(mesh.vertex_neighbours(b))
    apexes = set()
    for f in faces:
        if a in f and b in f:
            apexes |= {v for v in f if v != a and v != b}
    if shared != apexes:
        return None                                   # unsafe -> refuse (see docstring)

    # --- rewrite: repoint every `remove` to `keep`; drop faces that collapse to a degenerate sliver ---
    welded = []
    for f in faces:
        g = tuple(a if v == b else v for v in f)
        if len(set(g)) < len(g):                      # `keep` now appears twice -> the collapsed sliver, drop
            continue
        welded.append(g)

    # --- remove vertex `remove` and shift every higher index down by one (one fixed deterministic rule) ---
    kept_indices = [i for i in range(mesh.n_vertices) if i != b]
    remap = {old: new for new, old in enumerate(kept_indices)}
    out = [tuple(remap[v] for v in f) for f in welded]
    new_verts = mesh.vertices[kept_indices]
    return Mesh(new_verts, out)


# =====================================================================================================
# split_face -- cut an n-gon with a diagonal (E+1, F+1, chi invariant) -- the only n-gon operator (MEF).
# =====================================================================================================
def split_face(mesh, f_index, i, j):
    """Cut polygon face `f_index` with a diagonal between its i-th and j-th CORNERS (positions WITHIN the
    face, not vertex ids), producing two faces that share that diagonal. The corners must be non-adjacent
    (a diagonal, not an existing edge). E+1, F+1, chi unchanged. This is MEF -- the one operator that works on
    arbitrary polygons, not just triangles. Returns a new `Mesh`.
    """
    faces = list(mesh.faces)
    f = faces[f_index]
    n = len(f)
    i, j = sorted((i % n, j % n))
    if i == j or (j - i) < 2 or (i == 0 and j == n - 1):
        raise ValueError("split_face needs two NON-adjacent corners of the face (a diagonal)")
    side_a = tuple(f[i:j + 1])                         # corners i..j inclusive
    side_b = tuple(f[j:] + f[:i + 1])                  # the wrap-around side, sharing corners i and j
    out = [nf for k, nf in enumerate(faces) if k != f_index]
    out.append(side_a)
    out.append(side_b)
    return Mesh(mesh.vertices.copy(), out)


# =====================================================================================================
# Self-test -- asserts the invariants and the make/kill round-trips; prints a one-line summary.
# =====================================================================================================
def _selftest():
    from holographic.mesh_and_geometry.holographic_mesh import box
    from collections import Counter

    def canon(mesh):
        """Canonical connectivity: rotate each face to start at its lowest vertex, then sort the faces. Lets
        us compare 'same mesh' independent of face order and rotation -- the right notion for a round-trip."""
        out = []
        for f in mesh.faces:
            k = f.index(min(f))
            out.append(tuple(f[k:] + f[:k]))
        return tuple(sorted(out))

    # a closed TRIANGLE mesh to edit: triangulate a cube (12 triangles, chi = 2)
    cube = box(2.0, 2.0, 2.0)
    tm = Mesh(cube.vertices.copy(), [tuple(t) for t in cube.triangulate()])
    chi0 = tm.euler_characteristic()
    assert chi0 == 2 and tm.is_closed() and tm.is_manifold()

    # find an interior edge (one present in both directions across two triangles)
    dirset = Counter()
    for f in tm.faces:
        for k in range(3):
            dirset[(f[k], f[(k + 1) % 3])] += 1
    # find an interior edge whose flip is LEGAL: shared by two triangles whose apexes are not already an edge
    # (on a triangulated cube the quad diagonals qualify; a cube edge would flip into an existing edge)
    edgeset = set(tm.edges())
    a = b = c = d = None
    for (x, y) in dirset:
        if (y, x) not in dirset:
            continue
        fa = _face_with_directed_edge(tm.faces, x, y)
        fb = _face_with_directed_edge(tm.faces, y, x)
        if len(tm.faces[fa]) != 3 or len(tm.faces[fb]) != 3:
            continue
        cc, dd = _third(tm.faces[fa], x, y), _third(tm.faces[fb], x, y)
        if (min(cc, dd), max(cc, dd)) not in edgeset:
            a, b, c, d = x, y, cc, dd
            break
    assert a is not None, "expected at least one flippable interior edge on a triangulated cube"

    # --- flip_edge: V/E/F (hence chi) invariant, result manifold, and flip-the-new-edge restores it ---
    flipped = flip_edge(tm, a, b)
    assert flipped.is_manifold() and flipped.is_closed()
    assert flipped.euler_characteristic() == chi0
    assert (flipped.n_vertices, flipped.n_faces) == (tm.n_vertices, tm.n_faces)
    assert canon(flip_edge(flipped, c, d)) == canon(tm), "flip then flip-back must restore connectivity"

    # --- split_edge then collapse_edge: an exact inverse on connectivity (the make/kill round-trip) ---
    split, m = split_edge(tm, a, b)
    assert split.n_vertices == tm.n_vertices + 1
    assert split.euler_characteristic() == chi0       # chi preserved across the refinement
    assert split.is_manifold() and split.is_closed()
    back = collapse_edge(split, keep=a, remove=m)
    assert back is not None, "collapsing the freshly-split midpoint must be legal"
    assert back.n_vertices == tm.n_vertices
    assert canon(back) == canon(tm), "split then collapse must restore the original mesh exactly"

    # --- collapse_edge link condition: a triangular bipyramid has a non-collapsible equatorial edge ---
    bp_v = np.array([[0, 0, 1], [1, 0, 0], [-0.5, 0.87, 0], [-0.5, -0.87, 0], [0, 0, -1]], dtype=float)
    bp_f = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (4, 2, 1), (4, 3, 2), (4, 1, 3)]
    bp = Mesh(bp_v, bp_f)
    assert bp.euler_characteristic() == 2 and bp.is_closed() and bp.is_manifold()
    # edge {1,2}: endpoints 1 and 2 also share neighbour 3, which is NOT an apex of {1,2} -> must refuse
    assert collapse_edge(bp, keep=1, remove=2) is None, "link-condition-violating collapse must be refused"
    # edge {0,1}: a legal collapse (apex vertex onto the equator) -> yields a tetrahedron, chi still 2
    safe = collapse_edge(bp, keep=0, remove=1)
    assert safe is not None and safe.n_vertices == 4 and safe.euler_characteristic() == 2
    assert safe.is_manifold() and safe.is_closed()

    # --- split_face on an n-gon (a cube quad): E+1, F+1, chi unchanged, still a closed manifold ---
    quad = box(2.0, 2.0, 2.0)                          # 6 quads
    sf = split_face(quad, 0, 0, 2)                     # diagonal of the first quad -> two triangles
    assert sf.n_faces == quad.n_faces + 1
    assert sf.euler_characteristic() == 2 and sf.is_manifold() and sf.is_closed()

    # --- determinism: every operator is a pure function of (mesh, selection) -- byte-identical out ---
    s1, _ = split_edge(tm, a, b)
    s2, _ = split_edge(tm, a, b)
    assert np.array_equal(s1.vertices, s2.vertices) and s1.faces == s2.faces, "split_edge must be deterministic"
    assert flip_edge(tm, a, b).faces == flip_edge(tm, a, b).faces, "flip_edge must be deterministic"

    print("holographic_eulerops selftest: ok (flip chi/V/E/F-invariant + flip-back restores; "
          "split_edge/collapse_edge exact make-kill round-trip; collapse link-condition refuses the "
          "bipyramid equator; split_face n-gon E+1/F+1 chi=2; all operators deterministic)")


if __name__ == "__main__":
    _selftest()
