"""holographic_metrology.py -- MEASUREMENT with UNITS, from geometry (modeling-app backlog: measurement + units).

(The name is "metrology" -- the science of measurement -- because holographic_measure.py is already the statistical
variance harness; this is the geometric measuring tool, a different thing.)

A measuring tool must hand back a DIMENSIONED quantity, and it must read the ACTUAL geometry -- never a lossy VSA
readback (the backlog's honest-measurement note). So every function here walks the mesh vertices/faces directly and
wraps the result in a holographic_quantities.Quantity, which carries the unit AND the dimension: a length is
12.4 [m], an area is [m^2], a volume is [m^3]. Two payoffs come for free from that Quantity type:

  * CONVERSION is a single multiply -- q.to("ft"), q.to("L") -- with no room to fumble the factor;
  * the dimensional algebra REFUSES nonsense -- adding a length to an area is a grammar error, raised loudly, so a
    measurement bug can't silently produce a meaningless number.

Honest scope (kept): areas and lengths need no assumptions -- they are exact for the mesh as given. VOLUME uses the
divergence theorem, so it assumes a CLOSED, consistently-wound surface; on an open or inconsistently-wound mesh it
is meaningless (flagged here, not hidden). Angles are DIMENSIONLESS (radians) -- returned as plain floats with a
degrees() helper, since a radian is a ratio, not a physical dimension. NumPy + stdlib only; deterministic.
"""
import numpy as np

from holographic.misc.holographic_quantities import Quantity


def _tris(mesh):
    """Triangulate each (possibly polygonal) face by a simple fan, yielding (i0, i1, i2) vertex-index triples.
    A triangle face yields itself; an n-gon yields n-2 triangles sharing vertex 0."""
    for f in mesh.faces:
        for k in range(1, len(f) - 1):
            yield (f[0], f[k], f[k + 1])


def surface_area(mesh):
    """Total surface area = the sum of triangle areas, each |edge1 x edge2| / 2. Winding-independent (a cross-
    product magnitude is always positive), so it needs no orientation assumption. Returns a [m^2] Quantity."""
    V = mesh.vertices
    total = 0.0
    for (a, b, c) in _tris(mesh):
        total += 0.5 * np.linalg.norm(np.cross(V[b] - V[a], V[c] - V[a]))
    return Quantity(total, "m2", source="surface_area")


def volume(mesh):
    """Enclosed volume via the divergence theorem: (1/6) * |sum over triangles of v0 . (v1 x v2)|. Each triangle
    with the origin forms a signed tetrahedron; a closed, consistently-wound surface sums them to the enclosed
    volume. ASSUMES a closed, consistently-wound mesh -- otherwise the number is meaningless. Returns [m^3]."""
    V = mesh.vertices
    six_v = 0.0
    for (a, b, c) in _tris(mesh):
        six_v += float(np.dot(V[a], np.cross(V[b], V[c])))
    return Quantity(abs(six_v) / 6.0, "m3", source="volume")


class BBox:
    """An axis-aligned bounding box. .min/.max/.center/.size are plain 3-vectors (metres); .extent(axis) and
    .diagonal are length Quantities for the property panel."""

    def __init__(self, lo, hi):
        self.min = np.asarray(lo, float)
        self.max = np.asarray(hi, float)

    @property
    def size(self):
        return self.max - self.min                          # the per-axis extents, a plain 3-vector (metres)

    @property
    def center(self):
        return 0.5 * (self.min + self.max)

    def extent(self, axis):
        return Quantity(float(self.size[axis]), "m", source="bbox_extent")

    @property
    def diagonal(self):
        return Quantity(float(np.linalg.norm(self.size)), "m", source="bbox_diagonal")


def bounding_box(mesh):
    """The axis-aligned bounding box of the mesh vertices."""
    V = mesh.vertices
    return BBox(V.min(axis=0), V.max(axis=0))


def centroid(mesh):
    """The AREA-WEIGHTED surface centroid (mean triangle centroid weighted by triangle area) -- a 3-point in
    metres. Area-weighting is the honest centroid of the surface, not just the average of the vertices."""
    V = mesh.vertices
    num = np.zeros(3)
    den = 0.0
    for (a, b, c) in _tris(mesh):
        area = 0.5 * np.linalg.norm(np.cross(V[b] - V[a], V[c] - V[a]))
        num += area * (V[a] + V[b] + V[c]) / 3.0
        den += area
    return num / den if den > 0 else V.mean(axis=0)


def distance(p, q):
    """The straight-line length between two points. Returns a [m] Quantity."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    return Quantity(float(np.linalg.norm(q - p)), "m", source="distance")


def edge_length(mesh, i, j):
    """The length of the edge between vertices i and j. Returns [m]."""
    return distance(mesh.vertices[i], mesh.vertices[j])


# ---- angles (dimensionless: radians) -------------------------------------------------------------------------

def angle_between(u, v):
    """The angle in RADIANS between two vectors (dimensionless). Returns a plain float."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom < 1e-15:
        return 0.0
    return float(np.arccos(np.clip(np.dot(u, v) / denom, -1.0, 1.0)))


def angle_at(a, b, c):
    """The interior angle (radians) at the middle point b of the corner a-b-c."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    return angle_between(a - b, c - b)


def dihedral_angle(mesh, face_a, face_b):
    """The angle (radians) between two faces -- the angle between their normals. Typically the two faces share an
    edge (a crease). Face indices into mesh.faces."""
    V = mesh.vertices

    def face_normal(fi):
        f = mesh.faces[fi]
        n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
        return n / (np.linalg.norm(n) + 1e-15)

    return angle_between(face_normal(face_a), face_normal(face_b))


def degrees(radians):
    """Radians -> degrees (a convenience for a UI that shows angles in degrees)."""
    return float(np.degrees(radians))


def _unit_cube():
    """A closed, consistently-wound unit cube [0,1]^3 (8 verts, 12 triangles) -- the reference shape for the
    selftest: surface area 6, volume 1, bbox size (1,1,1), diagonal sqrt(3)."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    verts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
             (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    faces = [(0, 1, 2), (0, 2, 3),          # bottom  z=0
             (4, 6, 5), (4, 7, 6),          # top     z=1
             (0, 5, 1), (0, 4, 5),          # front   y=0
             (1, 6, 2), (1, 5, 6),          # right   x=1
             (2, 7, 3), (2, 6, 7),          # back    y=1
             (3, 4, 0), (3, 7, 4)]          # left    x=0
    return Mesh(verts, faces)


def _selftest():
    """Measurements read the geometry and come back dimensioned: a unit cube has area 6 m^2, volume 1 m^3, bbox
    size (1,1,1) and diagonal sqrt(3); units convert by one multiply; the dimensional algebra refuses length+area;
    angles are right; deterministic."""
    cube = _unit_cube()

    # (1) area and volume of the unit cube, as dimensioned quantities
    A = surface_area(cube)
    Vq = volume(cube)
    assert abs(A.to("m2") - 6.0) < 1e-9, A.to("m2")
    assert abs(Vq.to("m3") - 1.0) < 1e-9, Vq.to("m3")

    # (2) CONVERSION is one multiply: 1 m^3 = 1000 litres; a 1 m edge = ~3.28 ft
    assert abs(Vq.to("L") - 1000.0) < 1e-6
    d = edge_length(cube, 0, 1)                             # the unit edge, 1 m
    assert abs(d.to("m") - 1.0) < 1e-9 and abs(d.to("ft") - 3.280839895) < 1e-6

    # (3) the DIMENSIONAL ALGEBRA: adding two areas is fine; adding a length to an area is a grammar error
    assert abs((A + A).to("m2") - 12.0) < 1e-9
    try:
        _ = A + d                                          # [m^2] + [m] -- must refuse
        raise AssertionError("length + area should have raised")
    except ValueError:
        pass
    try:
        _ = Vq.to("m2")                                    # expressing a volume as an area -- must refuse
        raise AssertionError("volume.to('m2') should have raised")
    except ValueError:
        pass

    # (4) bounding box + centroid
    bb = bounding_box(cube)
    assert np.allclose(bb.size, [1, 1, 1]) and abs(bb.diagonal.to("m") - np.sqrt(3)) < 1e-9
    assert np.allclose(bb.center, [0.5, 0.5, 0.5])
    assert np.allclose(centroid(cube), [0.5, 0.5, 0.5], atol=1e-9)   # symmetric cube -> centre

    # (5) angles: a right angle is 90 degrees; the dihedral between two perpendicular cube faces is 90 degrees
    assert abs(degrees(angle_at((1, 0, 0), (0, 0, 0), (0, 1, 0))) - 90.0) < 1e-9
    assert abs(degrees(dihedral_angle(cube, 0, 4)) - 90.0) < 1e-6   # bottom vs front face

    # (6) deterministic
    assert surface_area(cube).to("m2") == surface_area(cube).to("m2")

    print("holographic_metrology selftest OK: a unit cube measures area %.1f m^2, volume %.1f m^3, bbox diagonal "
          "%.4f m (sqrt 3); a 1 m edge is %.4f ft and 1 m^3 is %.0f L by one multiply; the dimensional algebra "
          "refuses length + area and volume-as-area (grammar errors); a cube corner is a 90-degree right angle; "
          "measured straight from the geometry" % (A.to("m2"), Vq.to("m3"), bb.diagonal.to("m"),
                                                    d.to("ft"), Vq.to("L")))


if __name__ == "__main__":
    _selftest()
