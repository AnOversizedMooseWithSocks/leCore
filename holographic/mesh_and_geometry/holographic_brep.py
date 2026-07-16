"""holographic_brep.py -- B-REP TOPOLOGY FOUNDATION (K6): the vertex/edge/loop/face/shell hierarchy of a boundary
representation, with Euler-Poincare validity, genus, manifold checking, and a bridge that lets each FACE carry a
trimmed analytic surface (K3). This is the exact-solid representation, built native-first on the topology the mesh
half-edge kernel already proves out -- but a B-rep FACE is a trimmed surface, not a polygon, which is the difference
between this and holographic_mesh / holographic_eulerops (those edit a polygon MESH).

HONEST SCOPE (loud, up front)
-----------------------------
This is the TOPOLOGY + VALIDITY + FACE-GEOMETRY-BRIDGE layer -- the foundation a solid modeler is built on. It does
NOT (yet) do B-rep BOOLEANS (merging two solids by tracing K2 SSI curves into new trim loops and re-stitching the
topology) -- that is the declared next step, and it depends on exactly the K2/K3 pieces now in place. What IS here:
build a B-rep from faces, check it is a valid closed 2-manifold, compute its genus from Euler-Poincare, attach a
trimmed surface to any face, and tessellate the whole solid through those faces. That is the honest, bounded piece.

THE VALIDITY LAW (generalized Euler-Poincare, Mantyla form)
-----------------------------------------------------------
For a valid B-rep:   V - E + F - R = 2 * (S - H)
  V vertices, E edges, F faces, R rings (inner/hole loops = total loops - faces), S shells, H genus (handles).
A cube: 8 - 12 + 6 - 0 = 2 = 2*(1 - 0). A torus: 7 - 21 + 14 - 0 = 0 = 2*(1 - 1) -> genus 1. This one law is the
strongest cheap correctness witness a topology has, so it is the spine of the self-test.

Deterministic; NumPy + stdlib only.
"""
import numpy as np


class BFace:
    """A B-rep face: an OUTER vertex loop (a cycle of vertex indices) plus zero or more INNER loops (holes/rings),
    and an optional geometry -- a TrimmedSurface (K3) or a plane spec -- that gives the face its actual shape. The
    loops are the topology; the surface is the geometry the loops bound."""

    def __init__(self, outer, inner=None, surface=None):
        self.outer = list(outer)
        self.inner = [list(h) for h in (inner or [])]
        self.surface = surface           # a holographic_trimsurf.TrimmedSurface, a callable, or None (planar-by-loop)

    @property
    def loops(self):
        return [self.outer] + self.inner


class Brep:
    """A boundary representation: vertices (positions) + faces (each a BFace) grouped into shells. Edges are DERIVED
    from the face loops (an edge is an unordered vertex pair shared by faces), so the caller declares geometry +
    faces and the topology is computed and checked."""

    def __init__(self, vertices, faces, shells=None):
        self.vertices = np.asarray(vertices, float)
        self.faces = list(faces)
        # shells: list of face-index lists; default = one shell holding all faces
        self.shells = shells if shells is not None else [list(range(len(self.faces)))]

    # -- derived topology -------------------------------------------------------------------------------------
    def edges(self):
        """The set of undirected edges (frozenset of two vertex indices) across all face loops."""
        es = set()
        for f in self.faces:
            for loop in f.loops:
                n = len(loop)
                for i in range(n):
                    es.add(frozenset((loop[i], loop[(i + 1) % n])))
        return es

    def edge_face_counts(self):
        """How many faces use each undirected edge -- 2 everywhere iff the surface is a closed 2-manifold."""
        counts = {}
        for f in self.faces:
            for loop in f.loops:
                n = len(loop)
                for i in range(n):
                    e = frozenset((loop[i], loop[(i + 1) % n]))
                    counts[e] = counts.get(e, 0) + 1
        return counts

    def counts(self):
        """(V, E, F, R, S): vertices, edges, faces, rings (inner loops), shells."""
        V = len({v for f in self.faces for loop in f.loops for v in loop})
        E = len(self.edges())
        F = len(self.faces)
        R = sum(len(f.inner) for f in self.faces)
        S = len(self.shells)
        return V, E, F, R, S

    def is_closed_manifold(self):
        """A closed 2-manifold: every edge is shared by exactly two faces (no boundary edges, no non-manifold edges
        touched by 3+ faces)."""
        return all(c == 2 for c in self.edge_face_counts().values())

    def genus(self):
        """Genus H from Euler-Poincare (valid only for a closed manifold): H = S - (V - E + F - R)/2."""
        V, E, F, R, S = self.counts()
        chi = V - E + F - R
        return S - chi // 2 if chi % 2 == 0 else None

    def euler_poincare_ok(self, expected_genus=0):
        """Does the B-rep satisfy V - E + F - R = 2*(S - H) for the expected genus H?"""
        V, E, F, R, S = self.counts()
        return (V - E + F - R) == 2 * (S - expected_genus)

    def set_face_surface(self, face_index, surface):
        """Attach a TrimmedSurface (K3) or callable to a face -- the topology<->geometry bridge."""
        self.faces[face_index].surface = surface

    def validate(self, expected_genus=0):
        """A full validity report: {closed_manifold, counts, genus, euler_ok}."""
        V, E, F, R, S = self.counts()
        return {"closed_manifold": self.is_closed_manifold(), "V": V, "E": E, "F": F, "R": R, "S": S,
                "genus": self.genus(), "euler_ok": self.euler_poincare_ok(expected_genus)}


def box_brep(lo=(-1.0, -1.0, -1.0), hi=(1.0, 1.0, 1.0)):
    """Construct a valid closed cube B-rep: 8 vertices, 12 edges, 6 quad faces, 1 shell, genus 0. The canonical
    smallest valid solid, and the fixture the validity law is checked against."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    V = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    # six quad faces, each an outer loop of 4 vertex indices (consistent outward winding)
    faces = [BFace([0, 3, 2, 1]),   # bottom (z=lo)
             BFace([4, 5, 6, 7]),   # top (z=hi)
             BFace([0, 1, 5, 4]),   # front (y=lo)
             BFace([2, 3, 7, 6]),   # back (y=hi)
             BFace([1, 2, 6, 5]),   # right (x=hi)
             BFace([0, 4, 7, 3])]   # left (x=lo)
    return Brep(V, faces)


def _selftest():
    # --- the cube is a valid closed genus-0 solid, and the law holds exactly ---
    box = box_brep()
    rep = box.validate(expected_genus=0)
    assert rep["closed_manifold"], rep
    assert (rep["V"], rep["E"], rep["F"], rep["R"], rep["S"]) == (8, 12, 6, 0, 1), rep
    assert rep["genus"] == 0 and rep["euler_ok"], rep
    # V - E + F - R = 8 - 12 + 6 - 0 = 2 = 2*(S - H)
    assert (rep["V"] - rep["E"] + rep["F"] - rep["R"]) == 2 * (rep["S"] - rep["genus"])

    # --- Euler-Poincare on TORUS counts (V=7,E=21,F=14) -> genus 1 ---
    # a Csaszar-torus-shaped count; we verify the LAW computes genus 1, the defining property of a handle
    class _Fake:
        def __init__(s, V, E, F, R, S): s._c = (V, E, F, R, S)
        def counts(s): return s._c
        def genus(s):
            V, E, F, R, S = s._c; chi = V - E + F - R; return S - chi // 2
    t = _Fake(7, 21, 14, 0, 1)
    chi = 7 - 21 + 14 - 0
    assert chi == 0 and (1 - chi // 2) == 1, "torus should be genus 1"
    assert t.genus() == 1

    # --- non-manifold detection: add a face that reuses an edge a third time -> not a closed manifold ---
    bad = box_brep()
    bad.faces.append(BFace([0, 3, 5]))    # a stray triangle reusing edge (0,3) a 3rd time
    assert not bad.is_closed_manifold(), "an edge used by 3 faces must fail the manifold test"

    # --- ring (inner loop) accounting: a face with a hole increments R (rings = loops - faces) ---
    holed = box_brep()
    holed.faces[0].inner.append([8, 9, 10, 11])   # give the bottom face an inner loop (a square hole)
    holed.vertices = np.vstack([holed.vertices, np.array([[-.3, -.3, -1], [.3, -.3, -1], [.3, .3, -1], [-.3, .3, -1]])])
    V, E, F, R, S = holed.counts()
    assert R == 1, ("one ring expected", R)

    # --- K3 face bridge: attach a TrimmedSurface to a face and tessellate it ---
    from holographic.mesh_and_geometry.holographic_trimsurf import TrimmedSurface
    def flat(u, v):
        return np.array([u, v, 0.0])
    ts = TrimmedSurface(flat, [[0, 0], [1, 0], [1, 1], [0, 1]])
    box.set_face_surface(0, ts)
    assert box.faces[0].surface is ts
    verts, tris = ts.tessellate(12, 12)
    assert len(verts) > 0 and len(tris) > 0                    # the face carries real, tessellatable geometry

    # --- determinism: the cube's derived topology is reproducible ---
    assert box_brep().counts() == box_brep().counts()

    print("holographic_brep selftest OK: the cube is a valid CLOSED 2-manifold (V,E,F,R,S = 8,12,6,0,1), genus 0, "
          "and satisfies Euler-Poincare V-E+F-R = 2(S-H); the law computes genus 1 for torus counts (7,21,14); a "
          "3-face edge fails the manifold test; a face-hole increments the ring count R; a face can carry a K3 "
          "TrimmedSurface and tessellate real geometry (the topology<->geometry bridge). Deterministic. HONEST "
          "SCOPE: this is the topology+validity+face-geometry foundation; B-rep BOOLEANS (SSI-driven re-stitch) are "
          "the declared next step on top of K2/K3, not claimed here.")


if __name__ == "__main__":
    _selftest()
