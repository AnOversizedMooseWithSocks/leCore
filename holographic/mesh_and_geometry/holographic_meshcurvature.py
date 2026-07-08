"""Mesh curvature & feature detection (FWD-6): mean/Gaussian curvature and sharp-edge (crease) detection.

WHY THIS MODULE EXISTS
----------------------
Tier 1, item four of the forward backlog -- and the one with the most rigorous reference of the lot, because
discrete differential geometry hands us EXACT identities to check against. Three measurements, each reusing
machinery already in the codebase or grounded in a hard invariant:

  * MEAN curvature is the discrete Laplace-Beltrami of the vertex POSITIONS. The cotangent-weighted Laplacian
    operator K(x_i) = (1/2A_i) sum_j (cot a + cot b)(x_i - x_j) equals 2 H_i n_i (Meyer et al. 2003) -- so the
    magnitude of that operator IS twice the mean curvature. This reuses the EXACT cotangent edge weights FWD-4
    computes (`holographic_meshsmooth.cotangent_edge_weights`); curvature and smoothing are the same operator,
    one applied, one measured. On a unit sphere H = 1/R = 1 everywhere -- the reference.

  * GAUSSIAN curvature is the ANGLE DEFECT: K_i = (2*pi - sum of incident triangle angles at i) / A_i. And it
    carries the strongest check in the module: by discrete Gauss-Bonnet, the SUM of angle defects over a closed
    mesh equals 2*pi*chi EXACTLY -- a topological invariant. So the curvature estimate is validated against the
    Euler characteristic the mesh kernel (FWD-1) already computes: total defect must be 2*pi*chi to floating
    point. On a unit sphere K = 1/R^2 = 1 everywhere.

  * CREASES (sharp edges) are where the DIHEDRAL angle -- the angle between the two faces meeting at an edge --
    is large. A cube's 12 edges are 90-degree creases; a smooth sphere has none. This feeds crease-aware
    smoothing (FWD-4), adaptive subdivision (FWD-8), and shading-normal splitting.

THE [MILANFAR] STRUCTURE-TENSOR / STEERING CONNECTION
  Curvature is the surface's local SHAPE -- the directions and rates it bends -- which is exactly the local
  anisotropic metric a steering kernel encodes (the structure-tensor idea applied to geometry rather than image
  gradients). The crease set and the curvature field are what an adaptive operator STEERS by: subdivide where
  |K| is high, smooth along creases not across them, split shading normals at sharp edges. This module ships
  the scalar curvatures and the crease set (the measurable core); the full anisotropic-tensor steering of
  downstream operators is the consumer, not this module.

WHAT IT PROVIDES
  * angle_defects(mesh) / gaussian_curvature(mesh)  -- angle defect, and per-area Gaussian curvature.
  * mean_curvature(mesh)                            -- |H| via the cotangent Laplacian of positions.
  * vertex_areas(mesh)                              -- barycentric vertex areas (the per-vertex normaliser).
  * dihedral_angles(mesh) / detect_creases(mesh, threshold_deg) -- sharp-edge detection.
  * gauss_bonnet_defect(mesh)                       -- total angle defect minus 2*pi*chi (the exact check; ~0).

DETERMINISM (per ISA.md): every accumulation is a fixed-order sum over faces/edges; all outputs are pure
functions of the mesh. Curvatures are continuous (TOL) and feed no argmax decision here; crease detection is a
threshold on a continuous value -- a caller choosing creases by a float comparison must pin the threshold,
which `detect_creases` makes an explicit argument.

KEPT NEGATIVES (loud)
  * Per-vertex curvature is NOISY on coarse/irregular meshes. The MEAN over a closed surface is accurate (and
    the Gauss-Bonnet total is exact), but individual vertex values have high variance when the 1-ring is small
    or irregular -- the estimate needs a reasonably regular neighbourhood. `_selftest` measures and reports the
    per-vertex spread as evidence, and `curvature_confidence` gives a per-vertex regularity score so a caller
    can down-weight unreliable vertices rather than trust them blindly.
  * The exact Gauss-Bonnet check is for CLOSED meshes; an open mesh has a boundary (geodesic-curvature) term,
    so its total defect is not 2*pi*chi and the check is skipped there.
  * Angle defect and the cotangent operator assume TRIANGLE faces; n-gons are triangulated for the computation.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh
from holographic.mesh_and_geometry.holographic_meshsmooth import cotangent_edge_weights   # FWD-4: the raw cotangent edge weights, reused


# =====================================================================================================
# Per-vertex areas (the normaliser that turns a summed quantity into a per-unit-area curvature).
# =====================================================================================================
def vertex_areas(mesh):
    """Barycentric vertex areas: A_i = (1/3) * sum of incident triangle areas. The local surface area each
    vertex 'owns'. Simple and robust; the Voronoi/mixed area (Meyer) is more accurate pointwise but barycentric
    is sufficient for the curvature normalisation and keeps the code readable."""
    V = mesh.vertices
    A = np.zeros(mesh.n_vertices)
    for (i, j, k) in mesh.triangulate():
        area = 0.5 * float(np.linalg.norm(np.cross(V[j] - V[i], V[k] - V[i])))
        A[i] += area / 3.0
        A[j] += area / 3.0
        A[k] += area / 3.0
    return A


# =====================================================================================================
# Gaussian curvature via the angle defect -- and the EXACT Gauss-Bonnet check.
# =====================================================================================================
def angle_defects(mesh):
    """Per-vertex angle defect: 2*pi - (sum of incident triangle angles at the vertex). For a closed mesh the
    SUM of these equals 2*pi*chi (discrete Gauss-Bonnet) -- the exact topological check `gauss_bonnet_defect`
    verifies. Returns an array of length V. (Boundary vertices of an open mesh carry an uncorrected term.)"""
    V = mesh.vertices
    defect = np.full(mesh.n_vertices, 2.0 * np.pi)
    for (i, j, k) in mesh.triangulate():
        for (c, a, b) in [(i, j, k), (j, k, i), (k, i, j)]:
            u = V[a] - V[c]
            w = V[b] - V[c]
            cu = float(np.linalg.norm(u))
            cw = float(np.linalg.norm(w))
            if cu > 1e-12 and cw > 1e-12:
                ang = float(np.arccos(np.clip(np.dot(u, w) / (cu * cw), -1.0, 1.0)))
                defect[c] -= ang
    return defect


def gaussian_curvature(mesh):
    """Per-vertex Gaussian curvature K_i = angle_defect_i / A_i (the area-normalised angle defect). On a unit
    sphere K = 1/R^2 = 1 everywhere. Returns an array of length V."""
    return angle_defects(mesh) / np.maximum(vertex_areas(mesh), 1e-12)


def gauss_bonnet_defect(mesh):
    """The EXACT check: (sum of angle defects) - 2*pi*chi. Should be ~0 (floating point) for a CLOSED mesh, by
    discrete Gauss-Bonnet -- validating the curvature estimate against the Euler characteristic FWD-1 computes.
    Returns a float; meaningful only when `mesh.is_closed()`."""
    total = float(angle_defects(mesh).sum())
    return total - 2.0 * np.pi * mesh.euler_characteristic()


# =====================================================================================================
# Mean curvature via the cotangent Laplacian of positions (reusing FWD-4's cotangent weights).
# =====================================================================================================
def mean_curvature(mesh):
    """Per-vertex mean curvature |H| via the discrete mean-curvature-normal operator
        K(x_i) = (1/A_i) * sum_j w_ij (x_i - x_j) = 2 H_i n_i,
    where w_ij = (cot a + cot b)/2 are the cotangent edge weights (reused from FWD-4) and A_i the vertex area.
    |H_i| = |K(x_i)| / 2. On a unit sphere H = 1/R = 1 everywhere. Returns an array of length V."""
    V = mesh.vertices
    w = cotangent_edge_weights(mesh)
    A = vertex_areas(mesh)
    Kvec = np.zeros((mesh.n_vertices, 3))
    for (lo, hi), wij in w.items():
        Kvec[lo] += wij * (V[lo] - V[hi])
        Kvec[hi] += wij * (V[hi] - V[lo])
    Kvec = Kvec / np.maximum(A[:, None], 1e-12)
    return np.linalg.norm(Kvec, axis=1) / 2.0


# =====================================================================================================
# Creases: sharp edges, via the dihedral angle between adjacent faces.
# =====================================================================================================
def _newell_normal(verts, face):
    """A robust face normal by Newell's method (works for non-planar polygons), unit-length."""
    n = np.zeros(3)
    m = len(face)
    for k in range(m):
        cur = verts[face[k]]
        nxt = verts[face[(k + 1) % m]]
        n[0] += (cur[1] - nxt[1]) * (cur[2] + nxt[2])
        n[1] += (cur[2] - nxt[2]) * (cur[0] + nxt[0])
        n[2] += (cur[0] - nxt[0]) * (cur[1] + nxt[1])
    nn = float(np.linalg.norm(n))
    return n / nn if nn > 1e-12 else n


def dihedral_angles(mesh):
    """Per-interior-edge dihedral angle: the angle between the two faces meeting at the edge (computed from
    their normals). 0 = coplanar/flat, pi/2 = perpendicular faces (a cube edge). Returns {(lo,hi): radians} for
    every edge shared by exactly two faces (boundary edges are omitted)."""
    V = mesh.vertices
    normals = [_newell_normal(V, f) for f in mesh.faces]
    edge_faces = {}
    for fi, f in enumerate(mesh.faces):
        m = len(f)
        for k in range(m):
            e = (min(f[k], f[(k + 1) % m]), max(f[k], f[(k + 1) % m]))
            edge_faces.setdefault(e, []).append(fi)
    out = {}
    for e, fs in edge_faces.items():
        if len(fs) == 2:
            d = float(np.clip(np.dot(normals[fs[0]], normals[fs[1]]), -1.0, 1.0))
            out[e] = float(np.arccos(d))
    return out


def detect_creases(mesh, threshold_deg=30.0):
    """The sharp edges: every interior edge whose dihedral angle exceeds `threshold_deg`. On a cube the 12
    cube edges (90 deg) are creases and any flat triangulation diagonals (0 deg) are not; a smooth sphere has
    none. Returns a sorted list of (lo,hi) edges. The threshold is explicit (the float comparison that chooses
    creases is the caller's to pin)."""
    thr = np.radians(threshold_deg)
    return sorted(e for e, ang in dihedral_angles(mesh).items() if ang > thr)


# =====================================================================================================
# Confidence: a per-vertex regularity score, so a caller can down-weight noisy curvature (the kept negative).
# =====================================================================================================
def curvature_confidence(mesh):
    """A per-vertex confidence in [0,1] for the curvature estimate, from the REGULARITY of the 1-ring: the more
    uniform the incident triangle areas, the more trustworthy the discrete curvature. Computed as
    1 - normalised spread of incident triangle areas (coefficient of variation, clipped). Low on a sliver-heavy
    or sparse neighbourhood -- the kept negative made actionable rather than just stated."""
    V = mesh.vertices
    incident_areas = [[] for _ in range(mesh.n_vertices)]
    for (i, j, k) in mesh.triangulate():
        area = 0.5 * float(np.linalg.norm(np.cross(V[j] - V[i], V[k] - V[i])))
        for v in (i, j, k):
            incident_areas[v].append(area)
    conf = np.zeros(mesh.n_vertices)
    for v in range(mesh.n_vertices):
        a = np.asarray(incident_areas[v])
        if a.size >= 2 and a.mean() > 1e-12:
            cv = a.std() / a.mean()                    # coefficient of variation of incident triangle areas
            conf[v] = float(np.clip(1.0 - cv, 0.0, 1.0))
        elif a.size == 1:
            conf[v] = 0.5                              # a single incident triangle: weak but not zero
    return conf


# =====================================================================================================
# Self-test -- the EXACT references (Gauss-Bonnet, unit-sphere curvature) and the cube crease set.
# =====================================================================================================
def _selftest():
    from holographic.mesh_and_geometry.holographic_meshsmooth import _icosphere
    from holographic.mesh_and_geometry.holographic_mesh import box

    sphere = _icosphere(3)                              # unit sphere: K = H = 1 everywhere (the reference)
    assert sphere.is_closed() and sphere.is_manifold()

    # --- EXACT: discrete Gauss-Bonnet -- total angle defect equals 2*pi*chi to floating point ---
    gb = gauss_bonnet_defect(sphere)
    assert abs(gb) < 1e-6, f"Gauss-Bonnet defect should be ~0, got {gb}"
    # the sphere has chi=2, so the total defect is 4*pi
    assert abs(float(angle_defects(sphere).sum()) - 4.0 * np.pi) < 1e-6

    # --- unit-sphere curvature: mean Gaussian and mean |H| both near 1 (discrete, so within a band) ---
    K = gaussian_curvature(sphere)
    H = mean_curvature(sphere)
    assert 0.8 < float(K.mean()) < 1.25, float(K.mean())
    assert 0.8 < float(H.mean()) < 1.25, float(H.mean())

    # --- the kept negative, measured: per-vertex curvature is NOISY even where the mean is right ---
    spread = float(H.std() / max(H.mean(), 1e-12))     # coefficient of variation of per-vertex mean curvature
    assert spread > 0.02, "expected real per-vertex variance on a coarse mesh (the kept negative)"
    conf = curvature_confidence(sphere)
    assert conf.shape == (sphere.n_vertices,) and np.all((conf >= 0) & (conf <= 1))

    # --- creases: a cube's 12 edges are 90-degree creases; a smooth sphere has none ---
    cube = box(2.0, 2.0, 2.0)                           # quad cube: 12 edges, each shared by perpendicular faces
    creases = detect_creases(cube, threshold_deg=30.0)
    assert len(creases) == 12, f"a cube has 12 sharp edges, detected {len(creases)}"
    # every cube dihedral is ~90 degrees
    ang = dihedral_angles(cube)
    assert all(abs(a - np.pi / 2) < 1e-6 for a in ang.values())
    # a triangulated cube: the 6 flat diagonals are NOT creases, so still exactly 12
    tcube = Mesh(cube.vertices.copy(), [tuple(t) for t in cube.triangulate()])
    assert len(detect_creases(tcube, threshold_deg=30.0)) == 12
    # the smooth sphere has no sharp creases at a 30-degree threshold
    assert len(detect_creases(sphere, threshold_deg=30.0)) == 0

    # --- determinism ---
    assert np.array_equal(mean_curvature(sphere), mean_curvature(sphere))
    assert detect_creases(cube) == detect_creases(cube)

    print(f"holographic_meshcurvature selftest: ok (Gauss-Bonnet EXACT: total defect = 2*pi*chi to {abs(gb):.1e}; "
          f"unit sphere mean K={float(K.mean()):.3f}, mean |H|={float(H.mean()):.3f} (~1); cube = 12 creases at "
          f"90deg, sphere = 0; per-vertex CoV={spread:.2f} -- the kept noise negative; deterministic)")


if __name__ == "__main__":
    _selftest()
