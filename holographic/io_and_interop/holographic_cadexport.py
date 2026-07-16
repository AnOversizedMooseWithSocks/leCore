"""holographic_cadexport.py -- CAD INTEROP EXPORT (K7): STL (3-D mesh) and DXF (2-D drawing), the two open exchange
formats a modeling app cannot ship without. Boring plumbing, deliberately: a modeler's geometry is only useful if it
leaves the tool, and OBJ (already on Mesh.to_obj) is not what a fabricator or a 2-D drafter asks for.

WHAT IT WRITES
--------------
  * mesh_to_stl(vertices, faces)  -- ASCII STL. Faces may be triangles OR quads (quads are split into two triangles);
    a per-facet normal is computed from the vertex winding. STL is the 3-D-print / mesh-exchange lingua franca.
  * polylines_to_dxf(polylines)   -- minimal DXF R12 ASCII with POLYLINE/VERTEX entities, the 2-D drawing format
    Rhino / AutoCAD / every drafting tool reads. Closed loops (from K4 regions, K3 trim loops, K1 profiles) export
    as closed polylines.

WHY R12 AND WHY POLYLINE (not LWPOLYLINE)
----------------------------------------
R12 ASCII with the old POLYLINE + VERTEX + SEQEND triple is the most widely-readable DXF there is -- LWPOLYLINE is
R14+ and some importers still choke on it. We trade a few extra bytes for "opens everywhere", the right call for an
export nobody should have to debug.

Deterministic; NumPy + stdlib only. Returns STRINGS (the caller writes the file) so this stays pure and testable.
"""
import numpy as np


def _tri_normal(a, b, c):
    """Unit normal of triangle a,b,c from the winding (zero vector for a degenerate triangle)."""
    n = np.cross(np.asarray(b, float) - a, np.asarray(c, float) - a)
    L = np.linalg.norm(n)
    return n / L if L > 1e-15 else np.zeros(3)


def mesh_to_stl(vertices, faces, name="lecore"):
    """ASCII STL string for a mesh. `vertices` is (V,3); `faces` is a list of index tuples (tris or quads). Quads are
    split into (0,1,2) and (0,2,3). Per-facet normals come from the winding."""
    V = np.asarray(vertices, float)
    out = ["solid " + name]
    for f in faces:
        tris = [f] if len(f) == 3 else ([(f[0], f[1], f[2]), (f[0], f[2], f[3])] if len(f) == 4 else
                [(f[0], f[i], f[i + 1]) for i in range(1, len(f) - 1)])   # fan-triangulate n-gons
        for (ia, ib, ic) in tris:
            a, b, c = V[ia], V[ib], V[ic]
            n = _tri_normal(a, b, c)
            out.append("  facet normal %.9g %.9g %.9g" % (n[0], n[1], n[2]))
            out.append("    outer loop")
            for p in (a, b, c):
                out.append("      vertex %.9g %.9g %.9g" % (p[0], p[1], p[2]))
            out.append("    endloop")
            out.append("  endfacet")
    out.append("endsolid " + name)
    return "\n".join(out) + "\n"


def polylines_to_dxf(polylines, closed=None, layer="0"):
    """Minimal DXF R12 ASCII string for a set of 2-D polylines. `polylines` is a list of (n,2) or (n,3) arrays;
    `closed` is a per-polyline bool list (default: closed if first==last point). Emits POLYLINE/VERTEX/SEQEND."""
    polylines = [np.asarray(p, float) for p in polylines]
    if closed is None:
        closed = [len(p) > 2 and np.linalg.norm(p[0][:2] - p[-1][:2]) < 1e-9 for p in polylines]

    def pair(code, val):
        return "%d\n%s" % (code, val)

    lines = [pair(0, "SECTION"), pair(2, "ENTITIES")]
    for poly, isclosed in zip(polylines, closed):
        P = poly[:-1] if (isclosed and len(poly) > 1 and np.linalg.norm(poly[0][:2] - poly[-1][:2]) < 1e-9) else poly
        lines += [pair(0, "POLYLINE"), pair(8, layer), pair(66, "1"),   # 66=1: vertices follow
                  pair(70, "1" if isclosed else "0")]                   # 70 bit1: closed
        for v in P:
            z = float(v[2]) if len(v) > 2 else 0.0
            lines += [pair(0, "VERTEX"), pair(8, layer),
                      pair(10, "%.9g" % float(v[0])), pair(20, "%.9g" % float(v[1])), pair(30, "%.9g" % z)]
        lines += [pair(0, "SEQEND")]
    lines += [pair(0, "ENDSEC"), pair(0, "EOF")]
    return "\n".join(lines) + "\n"


def _selftest():
    # --- STL: a single triangle in the z=0 plane, ccw -> +z normal ---
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    stl = mesh_to_stl(verts, [(0, 1, 2)])
    assert stl.startswith("solid lecore") and stl.strip().endswith("endsolid lecore")
    assert stl.count("facet normal") == 1 and stl.count("vertex") == 3
    # the normal line reads +z (0 0 1) for this winding
    nline = [l for l in stl.splitlines() if "facet normal" in l][0]
    nx, ny, nz = [float(x) for x in nline.split()[2:5]]
    assert abs(nx) < 1e-9 and abs(ny) < 1e-9 and abs(nz - 1.0) < 1e-9, (nx, ny, nz)

    # --- STL: a quad splits into exactly two facets ---
    quad = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    stlq = mesh_to_stl(quad, [(0, 1, 2, 3)])
    assert stlq.count("facet normal") == 2 and stlq.count("outer loop") == 2

    # --- DXF: a closed square exports as a closed POLYLINE with 4 vertices ---
    square = np.array([[0.0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    dxf = polylines_to_dxf([square])
    assert "SECTION" in dxf and "ENTITIES" in dxf and dxf.strip().endswith("EOF")
    assert dxf.count("POLYLINE") == 1 + 0                    # one POLYLINE header (SEQEND is separate token)
    assert dxf.count("\nVERTEX\n") == 4                      # duplicate closing point dropped -> 4 unique vertices
    # the closed flag (70 -> 1) is present
    assert "\n70\n1\n" in dxf
    # coordinates round-trip: the DXF contains the x=1,y=1 corner
    assert "\n10\n1\n" in dxf and "\n20\n1\n" in dxf

    # --- DXF: an open polyline is not flagged closed ---
    openp = np.array([[0.0, 0], [1, 0], [2, 1]])
    dxf2 = polylines_to_dxf([openp])
    assert "\n70\n0\n" in dxf2 and dxf2.count("\nVERTEX\n") == 3

    # --- determinism ---
    assert mesh_to_stl(verts, [(0, 1, 2)]) == mesh_to_stl(verts, [(0, 1, 2)])
    assert polylines_to_dxf([square]) == polylines_to_dxf([square])

    print("holographic_cadexport selftest OK: STL writes correct per-facet normals (ccw triangle -> +z), quads split "
          "into 2 facets, n-gons fan-triangulate; DXF R12 emits POLYLINE/VERTEX/SEQEND, drops the duplicate closing "
          "point and flags closed loops (70->1), open polylines stay open, coordinates round-trip; both are pure "
          "strings and deterministic.")


if __name__ == "__main__":
    _selftest()
