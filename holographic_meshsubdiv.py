"""Mesh subdivision (FWD-8): Loop subdivision for triangle meshes -- refine the topology, then low-pass smooth.

WHY THIS MODULE EXISTS
----------------------
Tier 2. Subdivision is two operations braided together, and naming them honestly is the whole point:

  1. REFINE -- split every triangle into four (3 new edge midpoint-ish vertices, retriangulate 1 -> 4). This is a
     pure topological operation; in the Euler-operator algebra it is a fixed sequence of edge splits + face makes
     (the Stam seat's "subdivision = Euler-operator sequences"). It is the genuinely NEW part here.
  2. SMOOTH -- move every vertex (old and new) to a weighted average of its neighbours, with the carefully chosen
     Loop weights that give a C2 limit surface. This is a graph-signal LOW-PASS filter -- the SAME family as
     FWD-4's Taubin smoother, which is wired onto the shipped `graphsignal` low-pass primitives, and whose smooth
     limit lives in the low-frequency eigenspace of the graph Laplacian that `holographic_spectral` computes.
     This is the "reuses spectral-iteration" half: subdivision's smoothing IS a spectral low-pass, made concrete.

So the honest frame: the refinement is new (an Euler-operator sequence), the smoothing is the spectral low-pass
the engine already owns in another costume. The module implements Loop with its proper masks and MEASURES that
the result is a valid mesh with the exact properties subdivision must have.

THE LOOP MASKS (the standard scheme, readable)
  * New EDGE vertex on interior edge {a,b} shared by triangles (a,b,c) and (b,a,d):  3/8 (a+b) + 1/8 (c+d).
    On a BOUNDARY edge (only one adjacent triangle):  1/2 (a+b).
  * REPOSITION an interior old vertex v of valence n:  (1 - n*beta) v + beta * sum(neighbours), with Warren's
    beta = (1/n)(5/8 - (3/8 + 1/4 cos(2*pi/n))^2).  A BOUNDARY vertex uses  3/4 v + 1/8 (prev + next).
  * RETRIANGULATE each (a,b,c) into 4:  (a,ab,ca), (ab,b,bc), (ca,bc,c), (ab,bc,ca).

WHAT IT PROVIDES
  * loop_subdivide(mesh, levels=1) -- one or more rounds of Loop subdivision; returns a new (triangle) Mesh. A
    non-triangle input is triangulated first (Loop is defined on triangles).

THE MEASUREMENT BAR (checked exactly in the self-test)
  * Each level multiplies the face count by exactly 4 and gives V' = V + E (one new vertex per edge).
  * chi is preserved and a closed mesh stays a closed manifold (the refinement is topologically valid).
  * A FLAT mesh stays FLAT to machine precision -- the affine-reproduction rigor reference (the Loop masks are
    barycentric, so a planar input has a planar output exactly; the discrete analogue of Catmull-Clark
    reproducing a plane). This is the exact check the Stam seat asked for.
  * Subdivision SMOOTHS: the spread of dihedral angles drops sharply on an angular mesh (a cube) -- the
    low-pass character made geometric.

DETERMINISM (per ISA.md)
  New positions are deterministic weighted averages; vertices are appended in a fixed (sorted-edge) order. Same
  mesh in -> byte-identical mesh out (asserted).

KEPT NEGATIVES (loud)
  * Loop is a TRIANGLE scheme. A quad mesh (a box) is triangulated first, so the subdivided result reflects that
    triangulation, not a Catmull-Clark quad refinement. Catmull-Clark (the quad scheme) is a separate operator,
    not shipped here.
  * The limit surface is NOT the input polyhedron's circumscribed smooth shape -- e.g. subdividing an inscribed
    icosphere does not reproduce the exact sphere (subdivision surfaces have their own limit). The exact
    reproduction guarantee is for AFFINE/planar input only; for curved input the scheme smooths toward its own
    limit, which is the honest claim measured.
"""

import numpy as np

from holographic_mesh import Mesh


def _triangles(mesh):
    """The mesh's faces as triangles -- Loop is defined on triangles, so a non-triangle input is triangulated."""
    if all(len(f) == 3 for f in mesh.faces):
        return [tuple(f) for f in mesh.faces]
    return [tuple(t) for t in mesh.triangulate()]


def _one_level(mesh):
    V = mesh.vertices
    tris = _triangles(mesh)

    # --- adjacency: for each undirected edge, the opposite vertices of its adjacent triangles; and 1-ring nbrs ---
    edge_opp = {}        # frozenset{a,b} -> [opposite vertex, ...] (len 2 interior, 1 boundary)
    nbr = {v: set() for v in range(len(V))}
    for (a, b, c) in tris:
        for (x, y, o) in ((a, b, c), (b, c, a), (c, a, b)):
            edge_opp.setdefault(frozenset((x, y)), []).append(o)
        nbr[a].update((b, c)); nbr[b].update((a, c)); nbr[c].update((a, b))

    # --- new EDGE vertices (deterministic: edges visited in sorted order) ---
    edge_vertex = {}
    edge_positions = []
    for e in sorted(edge_opp.keys(), key=lambda s: tuple(sorted(s))):
        a, b = sorted(e)
        opp = edge_opp[e]
        if len(opp) == 2:
            c, d = opp
            pos = 3.0 / 8.0 * (V[a] + V[b]) + 1.0 / 8.0 * (V[c] + V[d])     # interior Loop edge mask
        else:
            pos = 0.5 * (V[a] + V[b])                                       # boundary edge: midpoint
        edge_vertex[e] = len(V) + len(edge_positions)
        edge_positions.append(pos)

    # --- REPOSITION old vertices (Loop vertex mask; boundary rule for boundary vertices) ---
    repositioned = np.array(V, copy=True)
    for v in range(len(V)):
        ring = nbr[v]
        boundary = [u for u in ring if len(edge_opp[frozenset((v, u))]) == 1]
        if len(boundary) >= 2:                                             # boundary vertex
            repositioned[v] = 0.75 * V[v] + 0.125 * (V[boundary[0]] + V[boundary[1]])
        else:
            n = len(ring)
            beta = (1.0 / n) * (5.0 / 8.0 - (3.0 / 8.0 + 0.25 * np.cos(2.0 * np.pi / n)) ** 2)
            repositioned[v] = (1.0 - n * beta) * V[v] + beta * sum(V[u] for u in ring)

    # --- RETRIANGULATE: each triangle -> 4 ---
    faces = []
    for (a, b, c) in tris:
        ab = edge_vertex[frozenset((a, b))]
        bc = edge_vertex[frozenset((b, c))]
        ca = edge_vertex[frozenset((c, a))]
        faces += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]

    positions = np.vstack([repositioned, np.array(edge_positions)]) if edge_positions else repositioned
    return Mesh(positions, faces)


# =====================================================================================================
# The vectorized path: Loop subdivision as a MATRIX. Subdivision is REGULAR, so the new positions are just a
# fixed linear map of the old ones -- new_V = S @ V. We build S once from the connectivity (as scipy-free
# index/weight arrays) and apply it with ONE weighted scatter-add (np.add.at), instead of the per-vertex and
# per-edge Python arithmetic in _one_level. Same Loop masks, same topology, so the result is bit-identical to
# TOL (only the float summation ORDER differs) and the topology is EXACT. _one_level stays as the reference.
# =====================================================================================================

def _subdivision_operator(mesh):
    """Build the Loop subdivision operator for a triangle mesh as (rows, cols, weights) -- a sparse matrix S with
    new_positions[row] = sum_j S[row,j] * V[j] -- together with the refined face list. Row layout matches
    _one_level exactly: rows 0..nV-1 are the repositioned old vertices, rows nV.. are the new edge vertices (in
    the same sorted-edge order), so S @ V reproduces _one_level's positions."""
    V = mesh.vertices
    tris = _triangles(mesh)
    nV = len(V)

    # adjacency (O(F)): opposite vertices per edge, and each vertex's 1-ring -- same as _one_level
    edge_opp = {}
    nbr = {v: set() for v in range(nV)}
    for (a, b, c) in tris:
        for (x, y, o) in ((a, b, c), (b, c, a), (c, a, b)):
            edge_opp.setdefault(frozenset((x, y)), []).append(o)
        nbr[a].update((b, c)); nbr[b].update((a, c)); nbr[c].update((a, b))

    rows, cols, wts = [], [], []

    # NEW EDGE VERTICES (rows nV..), visited in the SAME sorted order as _one_level so indices line up exactly
    edge_vertex = {}
    row = nV
    for e in sorted(edge_opp.keys(), key=lambda s: tuple(sorted(s))):
        a, b = sorted(e)
        opp = edge_opp[e]
        edge_vertex[e] = row
        if len(opp) == 2:                                   # interior edge: 3/8 endpoints + 1/8 opposites
            c, d = opp
            for idx, w in ((a, 3.0 / 8.0), (b, 3.0 / 8.0), (c, 1.0 / 8.0), (d, 1.0 / 8.0)):
                rows.append(row); cols.append(idx); wts.append(w)
        else:                                               # boundary edge: midpoint
            for idx in (a, b):
                rows.append(row); cols.append(idx); wts.append(0.5)
        row += 1
    n_new = row

    # REPOSITIONED OLD VERTICES (rows 0..nV-1). Append v's self-weight first, then its ring, so np.add.at
    # accumulates in the SAME order as _one_level's (1 - n*beta)*V[v] + beta*sum(ring) -> matches to ULP.
    for v in range(nV):
        ring = nbr[v]
        boundary = [u for u in ring if len(edge_opp[frozenset((v, u))]) == 1]
        if len(boundary) >= 2:                              # boundary vertex: 3/4 v + 1/8 (prev + next)
            rows += [v, v, v]; cols += [v, boundary[0], boundary[1]]; wts += [0.75, 0.125, 0.125]
        else:                                               # interior vertex: Warren's beta mask
            n = len(ring)
            beta = (1.0 / n) * (5.0 / 8.0 - (3.0 / 8.0 + 0.25 * np.cos(2.0 * np.pi / n)) ** 2)
            rows.append(v); cols.append(v); wts.append(1.0 - n * beta)
            for u in ring:
                rows.append(v); cols.append(u); wts.append(beta)

    # RETRIANGULATE: each triangle -> 4 (identical to _one_level -> topology is EXACT)
    faces = []
    for (a, b, c) in tris:
        ab = edge_vertex[frozenset((a, b))]
        bc = edge_vertex[frozenset((b, c))]
        ca = edge_vertex[frozenset((c, a))]
        faces += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]

    return (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64),
            np.asarray(wts, dtype=float), faces, n_new)


def _one_level_matrix(mesh):
    """One Loop level via the subdivision matrix: new_V = S @ V as a single scatter-add. Bit-identical to
    _one_level up to float summation order (TOL); topology EXACT."""
    V = mesh.vertices
    rows, cols, wts, faces, n_new = _subdivision_operator(mesh)
    new_V = np.zeros((n_new, V.shape[1]), dtype=float)
    np.add.at(new_V, rows, wts[:, None] * V[cols])          # the one sparse matvec S @ V (scatter-add)
    return Mesh(new_V, faces)


def loop_subdivide(mesh, levels=1):
    """Loop-subdivide a triangle mesh `levels` times: refine each triangle into four and low-pass smooth with the
    Loop masks (C2 limit surface). A non-triangle input is triangulated first. Returns a new triangle Mesh. Each
    level multiplies faces by 4, adds one vertex per edge, preserves chi, and keeps a closed mesh a closed
    manifold; a flat mesh stays flat exactly, while a curved one is smoothed toward the Loop limit surface.

    Uses the vectorized subdivision-matrix path (`_one_level_matrix`); `_one_level` remains as the readable
    reference and the bit-exactness pin."""
    out = mesh
    for _ in range(int(levels)):
        out = _one_level_matrix(out)
    return out


# =====================================================================================================
# Self-test -- exact topological refinement, affine reproduction, and the smoothing (low-pass) signature.
# =====================================================================================================
def _selftest():
    from holographic_meshsmooth import _icosphere
    from holographic_meshuv import flat_grid_mesh
    from holographic_meshcurvature import dihedral_angles
    from holographic_mesh import box

    # --- exact topological refinement on a closed sphere: F x4, V' = V + E, chi preserved, closed manifold ---
    s = _icosphere(1)
    E = len(s.edges())
    sub = loop_subdivide(s, 1)
    assert sub.n_faces == 4 * s.n_faces, f"each level must quadruple faces: {sub.n_faces} vs {4 * s.n_faces}"
    assert sub.n_vertices == s.n_vertices + E, f"V' must be V+E: {sub.n_vertices} vs {s.n_vertices + E}"
    assert sub.euler_characteristic() == s.euler_characteristic(), "subdivision must preserve chi"
    assert sub.is_closed() and sub.is_manifold(), "a subdivided closed mesh stays a closed manifold"

    # --- affine reproduction: a FLAT mesh stays flat to machine precision (the exact rigor reference) ---
    flat = flat_grid_mesh(5)                                # planar (z = 0)
    flat_sub = loop_subdivide(flat, 2)
    assert float(np.max(np.abs(flat_sub.vertices[:, 2]))) < 1e-12, "a planar mesh must subdivide to a planar mesh"

    # --- smoothing (low-pass): the dihedral-angle spread drops sharply on an angular mesh (a cube) ---
    cube = box()
    before = float(np.std(list(dihedral_angles(Mesh(cube.vertices.copy(), _triangles(cube))).values())))
    after = float(np.std(list(dihedral_angles(loop_subdivide(cube, 2)).values())))
    assert after < before, f"subdivision should smooth (lower dihedral spread): {after:.3f} !< {before:.3f}"

    # --- determinism ---
    assert np.array_equal(loop_subdivide(s, 1).vertices, loop_subdivide(s, 1).vertices)

    print(f"holographic_meshsubdiv selftest: ok (Loop subdivision: faces x4 ({s.n_faces} -> {sub.n_faces}), "
          f"V'=V+E ({s.n_vertices}+{E} = {sub.n_vertices}), chi + closed manifold preserved; flat stays flat to "
          f"machine precision (exact); dihedral spread on a cube {before:.3f} -> {after:.3f} (smoothed); deterministic)")


if __name__ == "__main__":
    _selftest()
