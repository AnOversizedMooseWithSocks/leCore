"""The explicit polygon mesh kernel (FWD-1): the substrate every explicit-geometry operator mutates.

WHY THIS MODULE EXISTS
----------------------
holostuff is mature on the *implicit / native* geometry axis -- SDF fields (`holographic_field`), Gaussian
splats (`holographic_splat`), scene-graph algebra (`holographic_scene`/`typed`/`recipe`), procedural
generation. But a live audit confirms it cannot represent or operate on an EXPLICIT polygon mesh at all:
no half-edge, no faces, no triangles, no marching cubes, nothing that a Blender-class modeler edits and
nothing a three.js front end can be handed. This module is that missing foundation -- the gate the forward
3-D backlog puts everything else behind. It is deliberately a SEPARATE concern from the VSA kernel: a mesh
is integer connectivity over float positions, not a hypervector, so it does not pretend to be one. The
bridges to the native VSA reps (mesh<->SDF<->splat, a mesh AS a StructureRecipe) are a later item (FWD-11 /
ARCH); this module is the honest explicit substrate they will bridge to.

WHAT IT PROVIDES
  * `Mesh` -- vertices (positions) + faces (ordered vertex-index polygons, triangles OR quads OR n-gons),
    with optional per-vertex attributes (normals, uvs, colours). The thing the modeler holds.
  * A HALF-EDGE adjacency structure built deterministically from the faces -- the standard data structure
    that makes "the faces around this vertex / the neighbour across this edge" an O(1) walk rather than an
    O(F) scan. This is the foundation Euler edit operators (FWD-7) will mutate; it ships here so adjacency
    has a real home, not as a throwaway.
  * Euler invariants: V - E + F = chi, and chi = 2 - 2g for a closed orientable surface -- the manifold
    well-formedness check (`euler_characteristic`, `genus`, `is_closed`, `is_manifold`).
  * `vertex_normals` (Newell's method, area-weighted), `triangulate` (fan), and `to_buffers` -- the flat,
    indexed buffers (positions / normals / uvs / triangle indices) that glTF and three.js consume. The
    boundary to the front end (FWD-2) reads exactly these.
  * Minimal OBJ read/write -- a known-mesh round-trip ("OBJ in/out", the FWD-1 bar) that preserves face
    topology exactly (a quad stays a quad), unlike a buffer round-trip which is triangle-soup.
  * Primitives to test against: `box`, `tetrahedron`, `grid` (a subdivided plane, for scaling tests).

ISA / DETERMINISM CONFORMANCE (per ISA.md, executable in holographic_determinism.py)
  Connectivity is COMBINATORIAL: vertex and face indices are exact integers with no float drift, so every
  integer buffer this module emits (face lists, the half-edge tables, the triangle index buffer) is
  bit-for-bit reproducible run to run -- the EXACT class. The half-edge build walks faces in face order and
  corners in corner order, and the twin lookup is a deterministic dict, so adjacency enumeration is itself
  reproducible. The only TOL (continuous) outputs are vertex NORMALS, and normals feed no argmax-style
  decision here, so the contract's one concern -- a reduction order leaking through to flip a decision (the
  bind_batch lesson) -- does not arise: there is no decision downstream of the float work. (If a future edit
  operator ever lets a float comparison choose WHICH element to split, that comparison must be pinned then;
  this module has no such path.)

KEPT NEGATIVE (loud, because it shapes the whole downstream plan)
  NumPy is a poor fit for the tight PER-ELEMENT loops a mesh kernel lives on. The half-edge build and the
  Euler-operator-style local rewrites are inherently scalar Python loops (one vertex, one corner at a time);
  they are correct and fine for the test meshes and modest assets here, but they will NOT scale to
  interactive editing of million-polygon meshes -- that needs a compiled core. The "NumPy-only" rule is an
  ENGINE rule (for the VSA substrate, where vectorisation is natural); it is not an interactive-mesh-core
  rule. This module honours NumPy-only as shipped, and flags plainly that the interactive future may not be
  able to. `_selftest` prints the measured half-edge build rate as evidence of the Python-loop bound.
"""

import numpy as np


# =====================================================================================================
# The mesh container.
# =====================================================================================================
class Mesh:
    """An explicit polygon mesh: positions + faces, with optional per-vertex attributes.

    vertices : (V, 3) float array -- one position per vertex.
    faces    : list of tuples of vertex indices -- each an ordered polygon (length 3 = triangle, 4 = quad,
               n = n-gon). Orientation is the winding order; a consistently-oriented closed mesh has every
               directed edge appearing exactly once (the manifold condition this kernel checks).
    normals  : (V, 3) or None -- per-vertex shading normals (computed by `vertex_normals` if absent).
    uvs      : (V, 2) or None -- per-vertex texture coordinates.
    colours  : (V, 4) or None -- per-vertex RGBA.

    The half-edge adjacency is built lazily on first request (`half_edges()`), so constructing a mesh is
    cheap; the combinatorial structure is paid for only when adjacency is actually needed.
    """

    def __init__(self, vertices, faces, normals=None, uvs=None, colours=None):
        self.vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
        # Store faces as plain tuples of python ints -- exact, hashable (for the twin lookup), and the form
        # the OBJ/buffer code expects. Reject a degenerate face (< 3 corners) early.
        self.faces = []
        for f in faces:
            f = tuple(int(i) for i in f)
            if len(f) < 3:
                raise ValueError(f"a face needs at least 3 vertices, got {f}")
            self.faces.append(f)
        self.normals = None if normals is None else np.asarray(normals, float).reshape(-1, 3)
        self.uvs = None if uvs is None else np.asarray(uvs, float).reshape(-1, 2)
        self.colours = None if colours is None else np.asarray(colours, float).reshape(-1, 4)
        self._he = None        # cached half-edge structure

    # ----- basic counts ------------------------------------------------------------------------------
    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_faces(self):
        return len(self.faces)

    def edges(self):
        """The set of UNDIRECTED edges as frozensets {vi, vj}. Deterministic (sorted) order on return."""
        es = set()
        for f in self.faces:
            n = len(f)
            for k in range(n):
                es.add(frozenset((f[k], f[(k + 1) % n])))
        # return as a sorted list of (lo, hi) tuples so the order is reproducible
        return [tuple(sorted(e)) for e in sorted(es, key=lambda e: tuple(sorted(e)))]

    @property
    def n_edges(self):
        return len(self.edges())

    # ----- half-edge adjacency -----------------------------------------------------------------------
    def half_edges(self):
        """Build (and cache) the half-edge table. Returns a dict of parallel arrays:

            origin[h]  -- the vertex each half-edge points OUT of
            face[h]    -- the face each half-edge borders
            nxt[h]     -- the next half-edge going around that face (origin->...->origin cycle)
            twin[h]    -- the opposite half-edge (other face) for the same undirected edge, or -1 at a boundary

        Built in face order, corner order: half-edge h = the directed edge (f[k] -> f[k+1]). The twin of a
        directed edge (a -> b) is the half-edge keyed (b -> a). If a directed edge (a -> b) appears twice the
        mesh is non-manifold / inconsistently oriented -- flagged, because every operator below assumes a
        clean manifold. Pure Python loops: this is the per-element cost the module docstring keeps as a
        negative.
        """
        if self._he is not None:
            return self._he
        origin, face, nxt, twin = [], [], [], []
        directed = {}                      # (a, b) -> half-edge index, for twin matching
        for fi, f in enumerate(self.faces):
            n = len(f)
            base = len(origin)             # index of this face's first half-edge
            for k in range(n):
                a, b = f[k], f[(k + 1) % n]
                h = len(origin)
                origin.append(a)
                face.append(fi)
                nxt.append(base + (k + 1) % n)     # cyclic within the face
                twin.append(-1)                    # filled below
                if (a, b) in directed:
                    raise ValueError(
                        f"non-manifold or inconsistently-oriented mesh: directed edge {(a, b)} appears twice")
                directed[(a, b)] = h
        # second pass: hook up twins now that every directed edge is registered
        for (a, b), h in directed.items():
            t = directed.get((b, a), -1)
            twin[h] = t
        self._he = {
            "origin": np.asarray(origin, dtype=np.int64),
            "face": np.asarray(face, dtype=np.int64),
            "nxt": np.asarray(nxt, dtype=np.int64),
            "twin": np.asarray(twin, dtype=np.int64),
        }
        return self._he

    def vertex_faces(self, v):
        """The faces incident to vertex `v`, as a sorted list of face indices (deterministic)."""
        he = self.half_edges()
        return sorted({int(he["face"][h]) for h in range(len(he["origin"])) if he["origin"][h] == v})

    def vertex_neighbours(self, v):
        """The 1-ring of vertex `v`: vertices sharing an edge with it, as a sorted list (deterministic)."""
        nb = set()
        for f in self.faces:
            if v in f:
                n = len(f)
                i = f.index(v)
                nb.add(f[(i + 1) % n])
                nb.add(f[(i - 1) % n])
        nb.discard(v)
        return sorted(nb)

    # ----- well-formedness (Euler) -------------------------------------------------------------------
    def euler_characteristic(self):
        """chi = V - E + F. For a closed orientable surface chi = 2 - 2g (g = genus): 2 for a sphere/cube,
        0 for a torus. An exact integer invariant -- the combinatorial well-formedness signature."""
        return self.n_vertices - self.n_edges + self.n_faces

    def is_closed(self):
        """True iff the mesh has no boundary: every half-edge has a twin (the surface fully wraps)."""
        he = self.half_edges()
        return bool(np.all(he["twin"] >= 0))

    def is_manifold(self):
        """True iff the half-edge structure built without raising (no directed edge appeared twice) AND
        -- the boundary-aware part -- every undirected edge is shared by at most two faces. Building the
        half-edge table already enforces the orientation condition; here we just confirm it built."""
        try:
            self.half_edges()
            return True
        except ValueError:
            return False

    def genus(self):
        """The genus g of a CLOSED orientable mesh from chi = 2 - 2g. Returns None (undefined) for an open
        mesh, since the formula needs a closed surface."""
        if not self.is_closed():
            return None
        return (2 - self.euler_characteristic()) // 2

    # ----- normals -----------------------------------------------------------------------------------
    def vertex_normals(self, store=True):
        """Per-vertex shading normals by Newell's method: each face contributes a normal whose magnitude is
        proportional to the face area (so big faces weigh more), accumulated at each of the face's vertices,
        then normalised per vertex. Newell's method is robust for non-planar polygons (a slightly bent quad),
        which is why it is preferred over a single cross product. Deterministic: faces are summed in face
        order. A continuous (TOL) value that feeds no decision. VECTORIZED for all-triangle meshes (the common
        case -- every marched/subdivided mesh): Newell's cross-terms over all faces at once, then a scatter-add to
        vertices in face order (bit-identical to the loop); falls back to the per-face loop for polygon meshes."""
        V = self.vertices
        acc = np.zeros((len(V), 3), dtype=float)
        faces = self.faces
        if faces and all(len(f) == 3 for f in faces):
            F = np.asarray(faces, dtype=int)                  # (nf, 3)
            P = V[F]                                          # (nf, 3, 3): per-face vertex positions
            cur = P
            nxt = P[:, [1, 2, 0], :]                          # the next vertex around each triangle
            cx, cy, cz = cur[:, :, 0], cur[:, :, 1], cur[:, :, 2]
            nxx, nyy, nzz = nxt[:, :, 0], nxt[:, :, 1], nxt[:, :, 2]
            fnx = np.sum((cy - nyy) * (cz + nzz), axis=1)      # Newell's normal, summed over the 3 edges
            fny = np.sum((cz - nzz) * (cx + nxx), axis=1)
            fnz = np.sum((cx - nxx) * (cy + nyy), axis=1)
            fn = np.stack([fnx, fny, fnz], axis=1)            # (nf, 3)
            np.add.at(acc, F.ravel(), np.repeat(fn, 3, axis=0))   # face-order accumulation (matches the loop)
        else:
            for f in faces:
                n = len(f)
                nx = ny = nz = 0.0
                for k in range(n):
                    cur = V[f[k]]; nxt = V[f[(k + 1) % n]]
                    nx += (cur[1] - nxt[1]) * (cur[2] + nxt[2])
                    ny += (cur[2] - nxt[2]) * (cur[0] + nxt[0])
                    nz += (cur[0] - nxt[0]) * (cur[1] + nxt[1])
                fn = np.array([nx, ny, nz])
                for vi in f:
                    acc[vi] += fn
        norms = np.linalg.norm(acc, axis=1, keepdims=True)
        out = np.where(norms > 1e-12, acc / np.where(norms > 1e-12, norms, 1.0), np.array([0.0, 0.0, 1.0]))
        if store:
            self.normals = out
        return out

    # ----- triangulation & flat buffers --------------------------------------------------------------
    def triangulate(self):
        """Fan-triangulate every polygon: a face [v0, v1, ..., v_{n-1}] becomes triangles
        (v0,v1,v2), (v0,v2,v3), ... Works for convex polygons (a box's quads are convex). Returns a
        (T, 3) int array. Deterministic. NOTE: fan triangulation is correct for convex faces only -- a
        concave n-gon needs ear-clipping (a later item); the primitives here are all convex."""
        tris = []
        for f in self.faces:
            for k in range(1, len(f) - 1):
                tris.append((f[0], f[k], f[k + 1]))
        return np.asarray(tris, dtype=np.int64).reshape(-1, 3)

    def to_buffers(self):
        """The flat, INDEXED buffers a glTF / three.js renderer consumes:

            position : (V, 3) float32 -- always present
            normal   : (V, 3) float32 -- the stored normals, or freshly computed if absent
            uv       : (V, 2) float32 -- only if the mesh has uvs
            colour   : (V, 4) float32 -- only if the mesh has colours
            indices  : (T*3,) int -- the triangle index buffer (flattened)

        float32 because that is what glTF stores; the integer index buffer is exact. This is the data FWD-2
        serialises; keeping it as a plain dict here keeps the mesh kernel independent of the file format."""
        normals = self.normals if self.normals is not None else self.vertex_normals(store=False)
        buf = {
            "position": self.vertices.astype(np.float32),
            "normal": normals.astype(np.float32),
            "indices": self.triangulate().reshape(-1).astype(np.int64),
        }
        if self.uvs is not None:
            buf["uv"] = self.uvs.astype(np.float32)
        if self.colours is not None:
            buf["colour"] = self.colours.astype(np.float32)
        return buf

    @staticmethod
    def from_buffers(position, indices, normal=None, uv=None, colour=None):
        """Reconstruct a (triangle) Mesh from flat buffers -- the inverse of `to_buffers`. The faces come
        back as TRIANGLES (the buffer form is triangle soup), so a quad mesh round-tripped through buffers
        returns triangulated; positions and the triangle set are exact. (Quad topology survives only the OBJ
        round-trip, which keeps faces verbatim.)"""
        position = np.asarray(position, float).reshape(-1, 3)
        idx = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        faces = [tuple(int(i) for i in tri) for tri in idx]
        return Mesh(position, faces, normals=normal, uvs=uv, colours=colour)

    # ----- OBJ I/O (the known-mesh round-trip) -------------------------------------------------------
    def to_obj(self):
        """Serialise to a Wavefront OBJ string: `v x y z` lines then `f i j k ...` lines (1-indexed, OBJ's
        convention). Faces are written VERBATIM, so a quad stays a quad -- the topology-preserving round-trip
        (unlike buffers). UVs/normals are omitted here to keep it minimal and unambiguous on read-back."""
        lines = ["# holostuff mesh"]
        for x, y, z in self.vertices:
            lines.append(f"v {x:.9g} {y:.9g} {z:.9g}")
        for f in self.faces:
            lines.append("f " + " ".join(str(i + 1) for i in f))   # OBJ is 1-indexed
        return "\n".join(lines) + "\n"

    @staticmethod
    def from_obj(text):
        """Parse a minimal Wavefront OBJ (v / f lines). Handles `f a b c`, `f a b c d`, and the
        `f a/vt/vn` slash form (takes the vertex index before the first slash). 1-indexed in the file,
        converted to 0-indexed here. Other statements (vt, vn, g, o, ...) are ignored."""
        verts, faces = [], []
        for raw in text.splitlines():
            s = raw.strip()
            if s.startswith("v "):
                parts = s.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif s.startswith("f "):
                idx = []
                for tok in s.split()[1:]:
                    idx.append(int(tok.split("/")[0]) - 1)        # vertex index, drop /vt/vn, to 0-indexed
                faces.append(tuple(idx))
        return Mesh(verts, faces)


# =====================================================================================================
# Primitives -- the meshes the tests and the vertical slice exercise.
# =====================================================================================================
def box(width=1.0, height=1.0, depth=1.0, center=(0.0, 0.0, 0.0)):
    """An axis-aligned box as a QUAD mesh: 8 vertices, 6 quad faces, consistently oriented (outward CCW).
    The canonical first mesh -- V=8, E=12, F=6, chi=2 (a genus-0 closed surface). Quads (not pre-triangulated)
    so the half-edge / Euler machinery is exercised on n-gons, not just triangles."""
    cx, cy, cz = center
    hx, hy, hz = width / 2.0, height / 2.0, depth / 2.0
    # 8 corners
    v = np.array([
        [cx - hx, cy - hy, cz - hz],   # 0
        [cx + hx, cy - hy, cz - hz],   # 1
        [cx + hx, cy + hy, cz - hz],   # 2
        [cx - hx, cy + hy, cz - hz],   # 3
        [cx - hx, cy - hy, cz + hz],   # 4
        [cx + hx, cy - hy, cz + hz],   # 5
        [cx + hx, cy + hy, cz + hz],   # 6
        [cx - hx, cy + hy, cz + hz],   # 7
    ], dtype=float)
    # 6 quads, each wound counter-clockwise as seen from OUTSIDE (so face normals point outward)
    faces = [
        (0, 3, 2, 1),   # -z (back)
        (4, 5, 6, 7),   # +z (front)
        (0, 1, 5, 4),   # -y (bottom)
        (2, 3, 7, 6),   # +y (top)
        (0, 4, 7, 3),   # -x (left)
        (1, 2, 6, 5),   # +x (right)
    ]
    return Mesh(v, faces)


def tetrahedron(scale=1.0, center=(0.0, 0.0, 0.0)):
    """A regular tetrahedron as 4 triangles: V=4, E=6, F=4, chi=2. The smallest closed manifold -- a good
    second test that the Euler machinery is not box-specific."""
    cx, cy, cz = center
    s = scale
    v = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
    ], dtype=float) * (s / np.sqrt(3.0)) + np.array([cx, cy, cz])
    # outward-oriented faces
    faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
    return Mesh(v, faces)


def grid(nx=4, ny=4, width=1.0, height=1.0, center=(0.0, 0.0, 0.0)):
    """A flat subdivided plane in the z=0 plane: an (nx by ny) grid of quads. OPEN (has a boundary), so
    chi = 1 and `is_closed()` is False. Used to test boundary handling and to give the per-element build a
    bigger mesh to time (the scaling/negative evidence)."""
    cx, cy, cz = center
    xs = np.linspace(cx - width / 2.0, cx + width / 2.0, nx + 1)
    ys = np.linspace(cy - height / 2.0, cy + height / 2.0, ny + 1)
    verts = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            verts.append([xs[i], ys[j], cz])
    def vid(i, j):
        return j * (nx + 1) + i
    faces = []
    for j in range(ny):
        for i in range(nx):
            faces.append((vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)))
    return Mesh(verts, faces)


# =====================================================================================================
# Self-test -- asserts the measured claims; prints a one-line summary (the house convention).
# =====================================================================================================
def _selftest():
    import time

    # --- the canonical cube: counts and Euler invariant ---
    m = box(2.0, 2.0, 2.0)
    assert m.n_vertices == 8 and m.n_faces == 6, (m.n_vertices, m.n_faces)
    assert m.n_edges == 12, m.n_edges
    assert m.euler_characteristic() == 2, m.euler_characteristic()      # genus-0 closed surface
    assert m.is_closed(), "a box has no boundary"
    assert m.is_manifold(), "a box is manifold"
    assert m.genus() == 0, m.genus()

    # --- half-edge consistency: every twin is reciprocal, and next-cycles return to their start ---
    he = m.half_edges()
    H = len(he["origin"])
    assert H == 24, H                                                   # 6 quads * 4 corners
    for h in range(H):
        t = he["twin"][h]
        assert t >= 0 and he["twin"][t] == h, f"twin not reciprocal at {h}"
    # walking `next` around a face returns to the start in exactly (face length) steps
    for h in range(H):
        cur, steps = h, 0
        while True:
            cur = int(he["nxt"][cur]); steps += 1
            if cur == h:
                break
            assert steps < 8, "next-cycle did not close"
    assert steps == 4, steps                                            # quads

    # --- normals point outward on a centred box (each vertex normal roughly along its position) ---
    nrm = m.vertex_normals()
    pos_dir = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    dots = np.sum(nrm * pos_dir, axis=1)
    assert np.all(dots > 0.5), f"box vertex normals should point outward, min dot {dots.min():.2f}"

    # --- buffers round-trip: positions exact, triangle set exact ---
    buf = m.to_buffers()
    assert buf["indices"].size == 6 * 2 * 3, buf["indices"].size        # 6 quads -> 12 tris -> 36 indices
    m2 = Mesh.from_buffers(buf["position"], buf["indices"], normal=buf["normal"])
    assert np.allclose(m2.vertices, m.vertices), "positions must survive the buffer round-trip exactly"
    assert m2.triangulate().shape == m.triangulate().shape

    # --- OBJ round-trip: topology preserved (quads stay quads) ---
    m3 = Mesh.from_obj(m.to_obj())
    assert m3.n_vertices == 8 and m3.n_faces == 6, (m3.n_vertices, m3.n_faces)
    assert all(len(f) == 4 for f in m3.faces), "OBJ round-trip must keep quads as quads"
    assert np.allclose(np.sort(m3.vertices, axis=0), np.sort(m.vertices, axis=0))

    # --- tetrahedron: a different closed manifold, same invariant ---
    t = tetrahedron()
    assert t.n_vertices == 4 and t.n_faces == 4 and t.n_edges == 6
    assert t.euler_characteristic() == 2 and t.is_closed() and t.genus() == 0

    # --- open grid: boundary, chi = 1, genus undefined ---
    g = grid(4, 4)
    assert not g.is_closed(), "a flat grid has a boundary"
    assert g.euler_characteristic() == 1, g.euler_characteristic()
    assert g.genus() is None

    # --- determinism: building the same mesh twice yields byte-identical integer buffers ---
    b1 = box(2, 2, 2).to_buffers()["indices"]
    b2 = box(2, 2, 2).to_buffers()["indices"]
    assert np.array_equal(b1, b2), "index buffers must be bit-reproducible (ISA EXACT class)"

    # --- the kept negative, measured: the half-edge build is Python-loop bound ---
    big = grid(60, 60)                                                  # 3600 quads, 14400 half-edges
    big._he = None
    t0 = time.perf_counter()
    big.half_edges()
    dt = time.perf_counter() - t0
    rate = len(big.half_edges()["origin"]) / dt if dt > 0 else float("inf")

    print(f"holographic_mesh selftest: ok (cube V8/E12/F6 chi=2 g=0; 24 half-edges reciprocal; "
          f"normals outward; OBJ+buffer round-trips; half-edge build ~{rate:,.0f} he/s "
          f"-- Python-loop bound, the kept negative)")


if __name__ == "__main__":
    _selftest()
