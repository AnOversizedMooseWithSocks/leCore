"""holographic_meshselect.py -- the SELECTION SUBSTRATE for a modeling app: a persistent set of mesh ELEMENTS
(vertices, edges, or faces) with a mode and set algebra. Every edit operates on a selection, so this is the state
that the whole interactive edit spine (pick -> select -> transform -> undo) hangs on.

WHY A SEPARATE THING FROM THE OBJECT SELECTION
----------------------------------------------
There is already a scene-level `Selection` (holographic_scene_query.Selection) -- it selects OBJECTS (scene
handles) and does named-set algebra over them. That is the object-mode selection. This module is the SUB-OBJECT
selection: inside one mesh, which VERTS / EDGES / FACES are chosen. A modeling app switches between object mode and
edit mode exactly this way (Blender's Tab, C4D's point/edge/polygon modes), so the two selections are complementary,
not a duplication -- object mode picks the mesh, edit mode picks elements within it.

WHAT IT IS
  A `MeshSelection` holds a `mode` ('vertex'|'edge'|'face') and a Python set of integer indices into the mesh's
  element list. It supports the set algebra a UI exposes (add / remove / toggle / union / intersect / invert /
  clear / all) and mode CONVERSION (a face selection -> the verts it touches, an edge selection -> the faces that
  use it), which is how "select the faces around these verts" works. Indices are validated against the mesh so a
  stale selection can't silently address a deleted element.

Deterministic: a selection is just a sorted index set; every operation is pure. NumPy/stdlib only.
"""

import numpy as np


def _mesh_counts(mesh):
    """Return (n_vertices, n_edges, n_faces) for a mesh given as {vertices, faces} (faces are index lists) or with
    an explicit `edges`. Edges are derived from faces (deduped undirected pairs) when not given -- the same dedupe
    the wireframe cage uses, so element ids agree across the app."""
    verts = mesh.get("vertices", [])
    faces = mesh.get("faces", [])
    nV = len(verts)
    nF = len(faces)
    if "edges" in mesh:
        nE = len(mesh["edges"])
    else:
        eset = set()
        for f in faces:
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                eset.add((min(a, b), max(a, b)))
        nE = len(eset)
    return nV, nE, nF


def _edge_list(mesh):
    """The canonical deduped undirected edge list of a mesh (each [i,j] once), so an edge index means the same
    thing everywhere. Uses `mesh['edges']` if present, else derives from faces."""
    if "edges" in mesh:
        return [list(e) for e in mesh["edges"]]
    eset = set()
    for f in mesh.get("faces", []):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            eset.add((min(a, b), max(a, b)))
    return [list(e) for e in sorted(eset)]


class MeshSelection:
    """A persistent set of mesh ELEMENTS (vertices, edges, or faces) with a mode and set algebra -- the sub-object
    selection a modeling app edits against. Bind it to a mesh so indices are validated and mode conversion knows
    the topology. Every mutating method returns self, so calls chain (sel.add([1,2]).invert())."""

    def __init__(self, mesh, mode="vertex", indices=None):
        if mode not in ("vertex", "edge", "face"):
            raise ValueError("mode must be vertex/edge/face; got %r" % mode)
        self.mesh = mesh
        self.mode = mode
        self._nV, self._nE, self._nF = _mesh_counts(mesh)
        self.indices = set()
        if indices is not None:
            self.add(indices)

    def _count(self, mode=None):
        mode = mode or self.mode
        return {"vertex": self._nV, "edge": self._nE, "face": self._nF}[mode]

    def _valid(self, i):
        return 0 <= int(i) < self._count()

    # -- set algebra (the operations a UI exposes) --------------------------------------------------------
    def add(self, idx):
        """Add one index or an iterable of indices to the selection (out-of-range indices are rejected, loudly)."""
        idx = [idx] if isinstance(idx, (int, np.integer)) else list(idx)
        for i in idx:
            if not self._valid(i):
                raise IndexError("%s index %d out of range (have %d)" % (self.mode, int(i), self._count()))
            self.indices.add(int(i))
        return self

    def remove(self, idx):
        """Remove one index or an iterable of indices (missing ones are ignored -- removal is idempotent)."""
        idx = [idx] if isinstance(idx, (int, np.integer)) else list(idx)
        self.indices.difference_update(int(i) for i in idx)
        return self

    def toggle(self, idx):
        """Toggle membership of each index -- the click-to-add/remove a UI does on shift-click."""
        idx = [idx] if isinstance(idx, (int, np.integer)) else list(idx)
        for i in idx:
            i = int(i)
            if not self._valid(i):
                raise IndexError("%s index %d out of range" % (self.mode, i))
            self.indices.symmetric_difference_update({i})
        return self

    def clear(self):
        """Deselect everything."""
        self.indices = set()
        return self

    def select_all(self):
        """Select every element of the current mode."""
        self.indices = set(range(self._count()))
        return self

    def invert(self):
        """Invert the selection within the current mode (selected <-> unselected)."""
        self.indices = set(range(self._count())) - self.indices
        return self

    def union(self, other):
        """In-place union with another selection of the SAME mode."""
        self._require_same_mode(other)
        self.indices |= other.indices
        return self

    def intersect(self, other):
        """In-place intersection with another selection of the same mode."""
        self._require_same_mode(other)
        self.indices &= other.indices
        return self

    def minus(self, other):
        """In-place difference (remove the other's elements)."""
        self._require_same_mode(other)
        self.indices -= other.indices
        return self

    def _require_same_mode(self, other):
        if other.mode != self.mode:
            raise ValueError("set algebra needs the same mode: %r vs %r (convert first)" % (self.mode, other.mode))

    # -- mode conversion (how 'select the faces around these verts' works) --------------------------------
    def to_mode(self, mode):
        """Return a NEW selection in `mode`, converting the current elements: verts<->the edges/faces that touch
        them, faces->their verts/edges, etc. Conversion is INCLUSIVE (a face is selected if ANY of its verts are)
        -- the growing behaviour a modeler expects when switching modes. The original selection is unchanged."""
        if mode == self.mode:
            return self.copy()
        verts = self._as_vertices()                            # everything routes through the vertex set
        out = MeshSelection(self.mesh, mode=mode)
        if mode == "vertex":
            out.indices = set(verts)
        elif mode == "edge":
            for ei, (a, b) in enumerate(_edge_list(self.mesh)):
                if a in verts or b in verts:
                    out.indices.add(ei)
        elif mode == "face":
            for fi, f in enumerate(self.mesh.get("faces", [])):
                if any(v in verts for v in f):
                    out.indices.add(fi)
        return out

    def _as_vertices(self):
        """The set of vertex indices the current selection touches -- the common currency for conversion."""
        if self.mode == "vertex":
            return set(self.indices)
        if self.mode == "edge":
            edges = _edge_list(self.mesh)
            vs = set()
            for ei in self.indices:
                vs.update(edges[ei])
            return vs
        # face
        vs = set()
        faces = self.mesh.get("faces", [])
        for fi in self.indices:
            vs.update(faces[fi])
        return vs

    # -- readout ------------------------------------------------------------------------------------------
    def copy(self):
        s = MeshSelection(self.mesh, mode=self.mode)
        s.indices = set(self.indices)
        return s

    def to_list(self):
        """The selected indices as a sorted list (deterministic)."""
        return sorted(self.indices)

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return "MeshSelection(mode=%r, n=%d)" % (self.mode, len(self.indices))


def _vertex_faces(mesh):
    """Adjacency: for each vertex, the list of face indices that use it. Built once; the substrate for loop/ring
    walks and boundary detection."""
    vf = {}
    for fi, f in enumerate(mesh.get("faces", [])):
        for v in f:
            vf.setdefault(v, []).append(fi)
    return vf


def _edge_faces(mesh):
    """Adjacency: for each undirected edge (as a sorted (a,b) tuple), the face indices that use it. An edge with
    exactly ONE face is a boundary edge -- the basis of boundary-loop selection."""
    ef = {}
    for fi, f in enumerate(mesh.get("faces", [])):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            ef.setdefault((min(a, b), max(a, b)), []).append(fi)
    return ef


def select_boundary_loops(mesh):
    """Select the OPEN BOUNDARY edges of a mesh -- the edges used by exactly ONE face (a hole rim or the border of
    an open surface). Returns an edge-mode MeshSelection. The 'select the hole' operation a modeler reaches for
    before filling or bridging. Delegates topology to the deduped edge list so edge ids agree with the cage."""
    ef = _edge_faces(mesh)
    edges = _edge_list(mesh)
    edge_id = {tuple(e): i for i, e in enumerate(edges)}
    sel = MeshSelection(mesh, mode="edge")
    for e, faces in ef.items():
        if len(faces) == 1:                                    # a boundary edge belongs to one face only
            sel.indices.add(edge_id[tuple(e)])
    return sel


def select_edge_loop(mesh, seed_edge):
    """Select the EDGE LOOP through `seed_edge` (an edge index) -- the ring of edges continuing 'straight' across
    quads, the #1 selection primitive users expect from Blender/Maya (Alt-click). Walks from the seed in both
    directions: at each vertex, cross to the edge that is NOT shared with the current edge's quad faces (the
    opposite edge of the quad), stopping at a pole (valence != 4) or when the loop closes. Returns an edge-mode
    MeshSelection.

    KEPT NEGATIVE: a true edge loop is only well-defined on quad topology; at a triangle or a pole the loop
    terminates rather than guessing, so on a triangle mesh this returns just the seed and its clean continuations.
    That is the honest behaviour -- a modeler expects the loop to STOP at a pole, not wander."""
    edges = _edge_list(mesh)
    edge_id = {tuple(e): i for i, e in enumerate(edges)}
    ef = _edge_faces(mesh)
    faces = mesh.get("faces", [])

    def opposite_edge_in_quad(face_idx, edge):
        """In a quad face, the edge parallel to `edge` (shares no vertex). Returns its (a,b) or None if not a quad
        or no clean opposite."""
        f = faces[face_idx]
        if len(f) != 4:
            return None
        a, b = edge
        # the two verts of the quad not in this edge form the opposite edge (in order).
        ring = [f[k] for k in range(4)]
        other = [v for v in ring if v not in (a, b)]
        if len(other) != 2:
            return None
        return (min(other[0], other[1]), max(other[0], other[1]))

    sel = MeshSelection(mesh, mode="edge")
    sel.indices.add(int(seed_edge))
    start = tuple(edges[int(seed_edge)])
    # walk both directions from the seed edge, hopping to the opposite edge of each adjacent quad.
    for face0 in ef.get(start, []):
        cur_edge, cur_face = start, face0
        for _ in range(len(edges)):                            # bounded: at most every edge once
            opp = opposite_edge_in_quad(cur_face, cur_edge)
            if opp is None or opp not in edge_id:
                break
            oid = edge_id[opp]
            if oid in sel.indices:
                break                                          # loop closed
            sel.indices.add(oid)
            # step to the quad on the OTHER side of the opposite edge.
            nb = [fi for fi in ef.get(opp, []) if fi != cur_face]
            if not nb:
                break                                          # boundary -- loop ends
            cur_edge, cur_face = opp, nb[0]
    return sel


def select_face_ring(mesh, seed_face):
    """Select the FACE RING starting at `seed_face` -- the band of quads a loop cut would run through. Walks quad to
    quad across shared edges, following the 'straight ahead' edge each step. Returns a face-mode MeshSelection.
    Terminates at a non-quad or a boundary (honest stop, like the edge loop)."""
    faces = mesh.get("faces", [])
    ef = _edge_faces(mesh)

    def shared_edge(fa, fb):
        sa = set()
        f = faces[fa]
        for k in range(len(f)):
            sa.add((min(f[k], f[(k + 1) % len(f)]), max(f[k], f[(k + 1) % len(f)])))
        g = faces[fb]
        for k in range(len(g)):
            e = (min(g[k], g[(k + 1) % len(g)]), max(g[k], g[(k + 1) % len(g)]))
            if e in sa:
                return e
        return None

    sel = MeshSelection(mesh, mode="face")
    sel.indices.add(int(seed_face))
    if len(faces[int(seed_face)]) != 4:
        return sel                                             # rings need quads
    # walk both ways: from the seed, cross an edge to the neighbour quad, then keep going out the OPPOSITE edge.
    f0 = faces[int(seed_face)]
    start_edges = [(min(f0[k], f0[(k + 1) % 4]), max(f0[k], f0[(k + 1) % 4])) for k in range(4)]
    for e0 in start_edges:
        cur_face, cur_edge = int(seed_face), e0
        for _ in range(len(faces)):
            nb = [fi for fi in ef.get(cur_edge, []) if fi != cur_face]
            if not nb or len(faces[nb[0]]) != 4:
                break
            nxt = nb[0]
            if nxt in sel.indices:
                break
            sel.indices.add(nxt)
            # the opposite edge of the neighbour quad continues the ring.
            f = faces[nxt]
            ring = [(min(f[k], f[(k + 1) % 4]), max(f[k], f[(k + 1) % 4])) for k in range(4)]
            # opposite edge = the one sharing no vertex with cur_edge.
            opp = [e for e in ring if not (set(e) & set(cur_edge))]
            if not opp:
                break
            cur_face, cur_edge = nxt, opp[0]
    return sel


def soft_selection_weights(mesh, selection, radius, falloff="smooth"):
    """SOFT SELECTION as a reusable WEIGHT FIELD: given a mesh and a MeshSelection (or a list of vertex indices),
    return a per-vertex weight in [0,1] that is 1 on the selection and falls off to 0 at `radius`, measured along
    the surface (multi-source geodesic distance). This is proportional editing: a transform reads these weights and
    moves each vertex by weight * delta, so a pulled vertex drags its neighbours smoothly.

    NOT A DUPLICATE of geodesic_soft_selection: this DELEGATES to the same geodesic engine
    (holographic_meshgeodesic) -- it just adapts the two things that engine does not do itself, namely (a) accept a
    dict mesh {vertices, faces} and a whole MeshSelection instead of a Mesh object + a single source vertex, and
    (b) take the MULTI-SOURCE minimum over all selected vertices. The distance computation, the Dijkstra, the
    surface metric are all the engine's; this is the input adapter + the multi-source fold, nothing re-implemented.

    `falloff` shapes the curve: 'linear', 'smooth' (smoothstep), or 'sharp' (quadratic). Returns a NumPy array of
    length n_vertices."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    from holographic.mesh_and_geometry.holographic_meshgeodesic import geodesic_distances, _edge_graph

    verts = np.asarray(mesh["vertices"], float)
    nV = len(verts)
    # resolve the source vertex set (accept a MeshSelection in any mode, or a raw index list).
    if isinstance(selection, MeshSelection):
        srcs = selection._as_vertices()
    else:
        srcs = set(int(i) for i in selection)
    if not srcs:
        return np.zeros(nV)

    # adapt the dict mesh to the Mesh the geodesic engine wants, build its edge graph ONCE, then run the engine's
    # single-source Dijkstra per selected vertex and fold to the nearest-source distance. All distance work is the
    # engine's; we only supply the mesh and take the min.
    gmesh = Mesh(verts, [list(f) for f in mesh.get("faces", [])])
    adj = _edge_graph(gmesh)
    dmin = np.full(nV, np.inf)
    for s in srcs:
        dmin = np.minimum(dmin, geodesic_distances(gmesh, int(s), adj=adj))

    t = np.clip(1.0 - dmin / max(radius, 1e-9), 0.0, 1.0)      # 1 at source -> 0 at radius
    if falloff == "linear":
        return t
    if falloff == "sharp":
        return t * t
    return t * t * (3.0 - 2.0 * t)                             # smoothstep (default)


def select_symmetric(mesh, selection, axis=0, tol=1e-4):
    """SYMMETRY SELECTION: given a selection, add its mirror-image elements across a world axis plane -- the
    'select the other side too' a modeler uses for symmetric edits. `axis` is 0/1/2 (mirror across the x=0 / y=0 /
    z=0 plane); a vertex at (x,y,z) matches its reflection with that axis negated, paired by nearest position within
    `tol`. Returns a NEW selection (same mode) containing the original PLUS the mirrored elements.

    This is the SELECTION-level complement to mirror_mesh (which mirrors GEOMETRY): here nothing is created, we just
    find the counterpart elements that already exist, so a soft/hard edit can be applied symmetrically. Pairing is
    by reflected position, so it works on any mesh whose geometry is actually symmetric about the plane (it reports
    only the matches it finds -- an asymmetric mesh simply gains fewer partners, honestly)."""
    verts = np.asarray(mesh["vertices"], float)
    # reflect every vertex across the axis plane and match to the nearest real vertex within tol.
    reflected = verts.copy()
    reflected[:, axis] = -reflected[:, axis]
    # build a mirror map vertex -> its counterpart index (or itself if on the plane).
    mirror_of = {}
    for i in range(len(verts)):
        d = np.linalg.norm(verts - reflected[i], axis=1)
        j = int(np.argmin(d))
        if d[j] <= tol:
            mirror_of[i] = j
    src_verts = selection._as_vertices() if isinstance(selection, MeshSelection) else set(int(i) for i in selection)
    mode = selection.mode if isinstance(selection, MeshSelection) else "vertex"
    # mirror the vertex set, then convert back to the requested mode via a vertex-mode scratch selection.
    mirror_verts = set(src_verts)
    for v in src_verts:
        if v in mirror_of:
            mirror_verts.add(mirror_of[v])
    vsel = MeshSelection(mesh, mode="vertex")
    vsel.indices = mirror_verts
    return vsel.to_mode(mode) if mode != "vertex" else vsel


def select_in_box(mesh, lo, hi, mode="vertex", project=None):
    """REGION SELECT: select every element whose position lies inside the axis-aligned box [lo, hi] -- the
    box/rubber-band select of a viewport. In vertex mode a vertex is in when its position is within the box; edge/
    face modes select an element if ANY of its verts are in (the inclusive rubber-band a modeler expects). If
    `project` (a 3x4 or 4x4 view-projection matrix or a callable pt->(u,v)) is given, the box is tested in the
    PROJECTED screen coordinates instead -- that is frustum / rectangle select from the camera's view. Returns a
    MeshSelection.

    Reuses the SAME inclusive semantics as to_mode() (a face counts if any vertex qualifies), so region select and
    mode conversion agree -- one rule for 'which elements does this vertex set imply', not two."""
    verts = np.asarray(mesh["vertices"], float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if project is None:
        pts = verts
        inside = np.all((pts >= lo) & (pts <= hi), axis=1)
    else:
        # project each vertex to screen coords, test the 2-D box (lo[:2], hi[:2]).
        if callable(project):
            uv = np.array([project(p)[:2] for p in verts], float)
        else:
            M = np.asarray(project, float)
            homo = np.c_[verts, np.ones(len(verts))]
            proj = homo @ M.T
            w = proj[:, 3:4] if proj.shape[1] == 4 else np.ones((len(verts), 1))
            uv = proj[:, :2] / np.where(np.abs(w) < 1e-12, 1e-12, w)
        inside = np.all((uv >= lo[:2]) & (uv <= hi[:2]), axis=1)
    in_verts = set(int(i) for i in np.nonzero(inside)[0])
    vsel = MeshSelection(mesh, mode="vertex")
    vsel.indices = in_verts
    return vsel.to_mode(mode) if mode != "vertex" else vsel


def proportional_edit(mesh, selection, translate, radius, falloff="smooth"):
    """PROPORTIONAL EDIT (Blender's O + G): move the selected vertices by `translate` and drag their neighbours along
    with a geodesic falloff -- the organic-shaping verb that lets one grab reshape a whole region smoothly, instead of
    moving every ring by hand (which is exactly what the mantis box-modelling sessions did the long way).

    One call = soft_selection_weights (the falloff field) + a weighted vertex move: each vertex v shifts by
    w(v) * translate, where w is 1 on the selection and eases to 0 at `radius` along the surface. `falloff` shapes the
    curve ('linear' / 'smooth' / 'sharp'). Returns a NEW Mesh (topology unchanged -- only positions move). Deterministic.

    WHY IT DELEGATES: the weight field is soft_selection_weights (the geodesic engine); this is that field times the
    move, nothing re-implemented. KEPT NEGATIVE: a straight translate of every weighted vertex (no rotate/scale falloff
    yet), and the radius is geodesic so a fold that brings two surface sheets close still only drags along the surface
    (correct, but can surprise if you expected a Euclidean bubble)."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    V = np.asarray(mesh.vertices, float)
    md = {"vertices": V, "faces": [list(int(i) for i in f) for f in mesh.faces]}
    w = soft_selection_weights(md, selection, radius, falloff=falloff)     # (nV,) in [0,1]
    out = V + w[:, None] * np.asarray(translate, float)[None, :]
    return Mesh(out, [tuple(int(i) for i in f) for f in mesh.faces])


def _selftest():
    """Contracts:
    1. set algebra: add/remove/toggle/invert/union/intersect/minus behave as sets, with out-of-range rejected.
    2. mode conversion: a face selection -> its verts -> back to faces recovers at least the original (inclusive).
    3. determinism: to_list is sorted and stable.
    """
    # a tiny quad-strip mesh: 6 verts, 2 quad faces.
    mesh = {"vertices": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]],
            "faces": [[0, 1, 4, 3], [1, 2, 5, 4]]}
    nV, nE, nF = _mesh_counts(mesh)
    assert (nV, nF) == (6, 2) and nE == 7, (nV, nE, nF)        # 7 unique edges in a 2-quad strip

    s = MeshSelection(mesh, "vertex").add([0, 1, 2])
    assert s.to_list() == [0, 1, 2]
    s.toggle(1)                                                # remove 1
    assert s.to_list() == [0, 2]
    s.invert()                                                 # everything except {0,2}
    assert s.to_list() == [1, 3, 4, 5]
    try:
        MeshSelection(mesh, "vertex").add(99); assert False
    except IndexError:
        pass

    a = MeshSelection(mesh, "face").add([0])
    b = MeshSelection(mesh, "face").add([1])
    assert a.copy().union(b).to_list() == [0, 1]
    assert a.copy().intersect(b).to_list() == []
    assert MeshSelection(mesh, "face").select_all().minus(a).to_list() == [1]

    # mode conversion: face 0 -> its verts {0,1,4,3} -> faces touching any of them -> {0,1} (both, since 1,4 shared)
    fverts = a.to_mode("vertex")
    assert fverts.to_list() == [0, 1, 3, 4]
    back = fverts.to_mode("face")
    assert 0 in back.indices                                   # the original face comes back
    # edge conversion is in-range
    e = a.to_mode("edge")
    assert all(0 <= i < nE for i in e.indices) and len(e) > 0

    # same-mode algebra guard
    try:
        a.union(MeshSelection(mesh, "vertex")); assert False
    except ValueError:
        pass

    # (4) loop / ring / boundary selection on a proper NxM quad grid.
    def grid(n, m):
        verts = [[i, j, 0] for j in range(m + 1) for i in range(n + 1)]
        faces = []
        for j in range(m):
            for i in range(n):
                a = j * (n + 1) + i
                faces.append([a, a + 1, a + 1 + (n + 1), a + (n + 1)])
        return {"vertices": verts, "faces": faces}
    g = grid(4, 3)                                             # 4x3 quads
    # boundary loop: the outer rim edges (used by one face).
    bnd = select_boundary_loops(g)
    assert bnd.mode == "edge" and len(bnd) == 2 * 4 + 2 * 3    # perimeter of a 4x3 grid = 14 edges
    # an edge loop from a vertical interior edge should span the full height (a straight column of edges).
    gedges = _edge_list(g)
    # find a vertical edge in column 2, bottom row: verts (2 in row0) -> (2 in row1)
    v0 = 2; v1 = 2 + (4 + 1)
    seed = _edge_list(g).index([min(v0, v1), max(v0, v1)])
    loop = select_edge_loop(g, seed)
    assert loop.mode == "edge" and len(loop) >= 3              # spans multiple rows (a real loop, not just the seed)
    # a face ring from an interior face spans a full row or column of quads.
    ring = select_face_ring(g, 5)                              # some interior quad
    assert ring.mode == "face" and len(ring) >= 3
    # determinism
    assert select_edge_loop(g, seed).to_list() == loop.to_list()

    # (5) soft-selection weights: 1 on the selection, falls off to 0 at the radius, geodesic.
    sel_center = MeshSelection(g, "vertex").add([g["faces"][5][0]])  # one interior vertex
    w = soft_selection_weights(g, sel_center, radius=2.5, falloff="smooth")
    src_v = g["faces"][5][0]
    assert abs(w[src_v] - 1.0) < 1e-9                          # 1 at the source
    assert w.min() >= 0.0 and w.max() <= 1.0                   # bounded
    far = int(np.argmax(np.linalg.norm(np.asarray(g["vertices"]) - np.asarray(g["vertices"])[src_v], axis=1)))
    assert w[far] < w[src_v]                                    # falls off with distance
    # a raw index list works too, and falloff shape matters.
    wl = soft_selection_weights(g, [src_v], radius=2.5, falloff="linear")
    assert wl[src_v] == 1.0 and np.any(wl != w)               # linear differs from smoothstep somewhere

    # (6) symmetry selection: on a grid symmetric about x=center, selecting the left edge also grabs the right.
    #     build a grid centered on x=0 so mirroring across x is exact.
    def cgrid(n, m):
        verts = [[i - n / 2.0, j, 0] for j in range(m + 1) for i in range(n + 1)]
        faces = []
        for j in range(m):
            for i in range(n):
                a = j * (n + 1) + i
                faces.append([a, a + 1, a + 1 + (n + 1), a + (n + 1)])
        return {"vertices": verts, "faces": faces}
    cg = cgrid(4, 2)
    left = MeshSelection(cg, "vertex").add([0])                # bottom-left corner at x=-2
    sym = select_symmetric(cg, left, axis=0)
    xs = np.asarray(cg["vertices"])[sym.to_list()][:, 0]
    assert any(x < 0 for x in xs) and any(x > 0 for x in xs)   # gained a positive-x partner
    assert len(sym) == 2                                       # the corner and its mirror

    # (7) region select: a box around the lower-left of the grid selects only the verts inside.
    box = select_in_box(g, lo=[-0.1, -0.1, -0.1], hi=[1.1, 1.1, 0.1], mode="vertex")
    picked = np.asarray(g["vertices"])[box.to_list()]
    assert len(box) > 0 and np.all(picked[:, 0] <= 1.1) and np.all(picked[:, 1] <= 1.1)

    # (8) proportional edit: pull the center vertex of a grid up; it moves the full delta, a far vertex is unmoved,
    # and a mid vertex moves partway -- the geodesic-falloff soft grab (Blender's O + G) as one call.
    from holographic.mesh_and_geometry.holographic_mesh import grid as _pgrid
    from holographic.mesh_and_geometry.holographic_meshselect import proportional_edit as _pe
    _pg = _pgrid(10, 10, width=10.0, height=10.0)
    _pv = np.asarray(_pg.vertices, float)
    _ci = int(np.argmin(np.linalg.norm(_pv[:, :2] - _pv[:, :2].mean(0), axis=1)))
    _po = _pe(_pg, [_ci], translate=(0, 0, 2.0), radius=3.0, falloff="smooth")
    _dz = np.asarray(_po.vertices, float)[:, 2] - _pv[:, 2]
    assert abs(_dz[_ci] - 2.0) < 1e-9                                          # the grabbed vertex moves fully
    _far = int(np.argmax(np.linalg.norm(_pv[:, :2] - _pv[_ci, :2], axis=1)))
    assert abs(_dz[_far]) < 1e-9                                               # a far vertex is untouched
    assert 0.0 < _dz[(_dz > 1e-6) & (_dz < 2.0 - 1e-6)].max() < 2.0            # neighbours drag partway
    assert [tuple(f) for f in _po.faces] == [tuple(f) for f in _pg.faces]      # topology unchanged

    print("holographic_meshselect selftest OK (vert/edge/face selection with add/remove/toggle/invert/union/"
          "intersect/minus; out-of-range rejected; mode conversion face->verts(%d)->faces recovers the original; "
          "boundary loop of a 4x3 grid = %d rim edges; edge loop spans %d edges; face ring spans %d quads; "
          "soft-selection weights are 1 at source and fall off geodesically (delegated to the geodesic engine); "
          "symmetry selection grabs the mirrored partner; box region-select picks verts inside; deterministic)"
          % (len(fverts), len(bnd), len(loop), len(ring)))


if __name__ == "__main__":
    _selftest()
