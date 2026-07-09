"""UV unwrapping (FWD-3): the shipped manifold chart (Isomap = MDS of geodesic distances) on MESH edges.

WHY THIS MODULE EXISTS
----------------------
The LAST Tier-1 item -- and the payoff of FWD-5. The backlog's sharpest irony lands here: UV unwrapping, the
"least-holostuff" item on the original DCC list, turns out to be a near-direct reuse of shipped, tested
faculties. `chart.manifold_chart` already flattens a curved manifold to 2-D with minimal distortion -- Isomap,
which is classical MDS of the GEODESIC distance matrix -- and `chart.classical_mds` embeds any distance matrix
whatsoever. So UV unwrapping is: take the mesh's OWN along-surface distances (FWD-5's `geodesic_matrix`, computed
on EXPLICIT mesh edges, not a k-NN graph) and hand them to the shipped `classical_mds`. The 2-D embedding that
comes back IS the UV chart -- a flattening that preserves surface distances as well as a plane can. The machinery
is shipped; the substitution is "mesh geodesics in place of k-NN geodesics."

WHY THIS IS THE RIGHT FLATTENING
  Classical MDS finds the 2-D coordinates whose pairwise Euclidean distances best match the given distance matrix
  (least-squares, via the top-2 eigenvectors of the double-centred distances). Feed it surface (geodesic)
  distances and you get the planar layout that best preserves how far apart points are ON THE SURFACE -- exactly
  what a UV map wants. On a DEVELOPABLE surface (flat, or a cylinder/cone cut open) the surface is isometric to
  the plane, so the unwrap is near-distortion-free; on a CURVED surface it is not (Gauss's Theorema Egregium:
  Gaussian curvature is an isometry invariant, so a curved patch CANNOT be flattened without stretch), and the
  distortion is the irreducible price the metric reports honestly.

WHAT IT PROVIDES
  * uv_unwrap(mesh, method) -- (V,2) UV coordinates packed into ~[0,1]^2, via classical MDS (Isomap) of the
    mesh geodesic matrix. `method='spectral'` routes to the shipped Laplacian-eigenmaps chart instead.
  * uv_distortion(mesh, uv) -- the per-edge STRETCH spread: 0 = isometric, growing with curvature. The measure
    the LSCM-distortion bar compares against (lower is flatter).
  * hemisphere_cap(subdiv) -- an open CURVED disk (upper half of an icosphere): the Gauss test surface.
  * puncture(mesh, vertex) -- remove a vertex + its incident faces, turning a CLOSED genus-0 mesh into a DISK
    (chi 2 -> 1) so it can be unwrapped at all. A crude seam; a real seam (a cut PATH chosen by curvature/genus)
    is the ARCH-4 atlas piece, shared with the concept-manifold chart.

DETERMINISM (per ISA.md)
  `geodesic_matrix` is deterministic (FWD-5), and `classical_mds` pins each embedding axis's sign via the one
  determinism contract (`fix_eigvec_signs`), so the UV is a pure, reproducible function of the mesh (asserted).

KEPT NEGATIVES (loud)
  * Parameterization assumes a DISK-topology chart. A closed surface (a sphere) has no boundary and cannot be
    flattened to a disk without a CUT; unwrapping it directly produces large distortion. `puncture` opens it
    crudely (one vertex removed), and the self-test MEASURES that even a punctured sphere distorts far more than
    a developable patch -- the seam-need made concrete, not asserted. A good seam (a cut path placed by the
    `topology`/genus faculty) is the ARCH-4 item, deliberately deferred.
  * HIGH curvature -> unavoidable distortion (Gauss). The unwrap minimises it but cannot remove it; the metric
    reports the irreducible amount.
  * The geodesic INPUT is the edge-graph approximation (FWD-5's kept negative), so the unwrap inherits a few
    percent of geodesic error on top of the parameterisation distortion. It is also sensitive to connectivity
    ANISOTROPY: a fan triangulation (all diagonals one way) biases the edge-graph geodesic and the distortion
    GROWS with resolution; an isotropic mesh unwraps cleanly and the distortion SHRINKS toward isometric as it
    refines (measured -- see `flat_grid_mesh`). Use isotropic meshes; flag anisotropic ones.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh
from holographic.mesh_and_geometry.holographic_meshgeodesic import geodesic_matrix
from holographic.misc.holographic_chart import classical_mds, laplacian_eigenmaps   # shipped embedders, reused verbatim


def uv_unwrap(mesh, method="isomap"):
    """Flatten a (disk-topology) mesh to 2-D UV by classical MDS of its geodesic distance matrix -- Isomap on
    explicit mesh edges. Returns (V,2) UV packed into ~[0,1]^2 with a single uniform scale (so the aspect ratio
    is preserved).

    `method`:
      'isomap'   -- classical MDS of the GEODESIC matrix (the recommended geodesic-preserving chart; WINS on
                    curved surfaces where a linear projection folds).
      'planar'   -- linear best-fit-plane (PCA) projection. EXACTLY isometric on a developable (flat) surface,
                    and the right choice there -- but it folds a curved one. The honest counterpart to isomap:
                    measured, the planar projection beats isomap on a flat patch (0.00 vs 0.05, the edge-graph
                    geodesic's small error) and isomap beats it on a curved cap (0.23 vs 0.46). Pick by curvature.
      'spectral' -- the shipped Laplacian-eigenmaps chart (local structure, not a faithful metric)."""
    if method == "planar":
        X = mesh.vertices - mesh.vertices.mean(axis=0)
        Vt = np.linalg.svd(X, full_matrices=False)[2]         # principal axes; top-2 span the best-fit plane
        uv = X @ Vt[:2].T
    elif method == "spectral":
        uv = laplacian_eigenmaps(mesh.vertices, dim=2, k=6)   # note: this path still uses the position k-NN
    else:
        uv = classical_mds(geodesic_matrix(mesh), dim=2)      # the recommended geodesic-preserving chart
    lo = uv.min(axis=0)
    span = float((uv.max(axis=0) - lo).max())
    if span < 1e-12:
        span = 1.0
    return (uv - lo) / span                                   # uniform scale -> ~[0,1]^2, aspect preserved


def stable_uv(mesh, bounds=None, mode="triplanar", axis=2):
    """UVs that are a deterministic function of WORLD POSITION, so they DON'T move under local edits -- the
    stable counterpart to uv_unwrap. The global unwraps (isomap/planar-PCA/spectral) solve an MDS / eigenmap
    over the WHOLE mesh, so a local edit shifts every UV, and the solution carries a sign/rotation ambiguity
    (the chart can flip on re-run). A position-projection UV has neither problem: a vertex at a given position
    always gets the same UV, whatever was edited elsewhere. The trade-off is honest -- this is stable texturing,
    not a single seam-cut chart; for a faithful low-distortion unwrap use uv_unwrap and accept that it re-solves.

    Normalised by `bounds` (the FIXED field domain, (min_corner, max_corner)) rather than the mesh's current
    extent, so the UV scale is itself invariant to edits that change the bounding box. Falls back to the mesh's
    own extent if bounds is None.

    mode='planar'    -- drop `axis` (default z); UV = the other two normalised coords. Exact on a flat-ish face,
                        folds on a curved surface.
    mode='triplanar' -- each vertex projects onto the axis-plane its NORMAL most faces (so curves don't fold);
                        still purely position+normal-determined, hence stable. Returns (V,2) in ~[0,1]^2."""
    V = np.asarray(mesh.vertices, float)
    if bounds is None:
        lo = V.min(axis=0); hi = V.max(axis=0)
    else:
        lo = np.asarray(bounds[0], float); hi = np.asarray(bounds[1], float)
    span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    Vn = (V - lo) / span                                       # normalised to ~[0,1]^3 by the FIXED domain
    if mode == "planar":
        others = [k for k in range(3) if k != int(axis)]
        return Vn[:, others].copy()
    # triplanar: pick the projection plane per vertex by its dominant normal component
    N = mesh.vertex_normals(store=False)
    dom = np.argmax(np.abs(N), axis=1)                         # 0/1/2 = the axis the normal most aligns with
    uv = np.zeros((len(V), 2))
    for ax in range(3):
        m = dom == ax
        if not np.any(m):
            continue
        others = [k for k in range(3) if k != ax]
        uv[m] = Vn[m][:, others]
    return uv


def uv_distortion(mesh, uv):
    """Per-edge STRETCH distortion: the spread of the ratio (UV edge length / 3-D edge length), normalised by the
    median ratio and measured as the standard deviation of its log. 0 = isometric (every edge scaled equally, a
    developable surface); it grows with Gaussian curvature. This is the scale-invariant flatness measure the LSCM
    -distortion bar compares against -- lower is a better (flatter) parameterisation."""
    V = mesh.vertices
    ratios = []
    for (lo, hi) in mesh.edges():
        d3 = float(np.linalg.norm(V[lo] - V[hi]))
        duv = float(np.linalg.norm(uv[lo] - uv[hi]))
        if d3 > 1e-12 and duv > 1e-12:
            ratios.append(duv / d3)
    ratios = np.asarray(ratios)
    if ratios.size == 0:
        return 0.0
    med = float(np.median(ratios))
    return float(np.std(np.log(ratios / med)))                # log-ratio spread; 0 = perfectly isometric


def flat_grid_mesh(n=9, width=2.0, height=1.4):
    """A FLAT (developable) triangulated patch -- the near-isometric reference surface. The diagonals ALTERNATE
    direction per quad (a checkerboard), which keeps the edge graph isotropic.

    WHY ALTERNATING (a measured insight): a naive 'fan' triangulation puts every diagonal the SAME way, which
    biases the edge-graph geodesic along that direction -- and because the bias is systematic, the unwrap
    distortion GROWS with resolution instead of shrinking (measured: 0.14 -> 0.17 from 5x5 to 15x15). The
    isotropic alternating mesh behaves correctly: distortion SHRINKS toward isometric as the mesh refines
    (0.063 -> 0.044). The edge-graph geodesic is sensitive to connectivity anisotropy; an isotropic mesh unwraps
    cleanly, an anisotropic one carries a directional bias. That is a real limitation of the edge-graph
    approximation, kept on record rather than hidden by picking whichever triangulation passes."""
    xs = np.linspace(0.0, width, n)
    ys = np.linspace(0.0, height, n)
    V = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)
    at = lambda i, j: j * n + i
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)
            if (i + j) % 2 == 0:
                F += [(a, b, c), (a, c, d)]                # diagonal a-c
            else:
                F += [(a, b, d), (b, c, d)]                # diagonal b-d (alternating -> isotropic)
    return Mesh(V, F)


def hemisphere_cap(subdiv=3):
    """An open CURVED disk: the upper hemisphere of a unit icosphere (every face whose vertices have z >= 0), with
    a boundary at the equator. Curved, so by Gauss it cannot be flattened without distortion -- the test surface
    that shows the unavoidable-distortion negative against a developable reference."""
    from holographic.mesh_and_geometry.holographic_meshsmooth import _icosphere
    s = _icosphere(subdiv)
    keep = [f for f in s.faces if all(s.vertices[v][2] >= -1e-9 for v in f)]
    used = sorted({v for f in keep for v in f})
    remap = {old: i for i, old in enumerate(used)}
    return Mesh(s.vertices[used], [tuple(remap[v] for v in f) for f in keep])


def puncture(mesh, vertex=0):
    """Remove `vertex` and its incident faces, turning a CLOSED genus-0 mesh into a DISK (chi 2 -> 1) whose
    boundary is the removed vertex's 1-ring. The crude seam that makes a closed surface unwrappable at all -- a
    real seam (a cut path) is the ARCH-4 piece. Returns a new Mesh."""
    keep = [f for f in mesh.faces if vertex not in f]
    used = sorted({v for f in keep for v in f})
    remap = {old: i for i, old in enumerate(used)}
    return Mesh(mesh.vertices[used], [tuple(remap[v] for v in f) for f in keep])


# =====================================================================================================
# Self-test -- developable (low distortion) vs curved (Gauss distortion) vs closed-needs-a-seam.
# =====================================================================================================
def _selftest():
    # --- a FLAT (developable) isotropic patch: the unwrap is near-isometric (the reference) ---
    flat = flat_grid_mesh(9)                                   # alternating diagonals -> isotropic edge graph
    uv_flat = uv_unwrap(flat)
    dist_flat = uv_distortion(flat, uv_flat)
    assert dist_flat < 0.07, f"a flat isotropic patch should unwrap nearly isometrically, got {dist_flat:.3f}"
    # the UV is non-degenerate: vertices map to distinct points
    assert len({(round(float(x), 6), round(float(y), 6)) for x, y in uv_flat}) == flat.n_vertices

    # --- a CURVED open cap: distorts MORE than the flat patch (Gauss -- unavoidable) ---
    cap = hemisphere_cap(3)
    assert not cap.is_closed(), "the hemisphere cap is an open disk"
    dist_cap = uv_distortion(cap, uv_unwrap(cap))
    assert dist_cap > dist_flat + 0.05, "a curved surface must distort clearly more than a developable one"

    # --- a CLOSED sphere needs a SEAM: puncture it to a disk, and it STILL distorts far more than the cap ---
    from holographic.mesh_and_geometry.holographic_meshsmooth import _icosphere
    sphere = _icosphere(3)
    assert sphere.is_closed()
    disk = puncture(sphere, vertex=0)
    assert disk.euler_characteristic() == 1 and not disk.is_closed(), "puncturing gives a disk (chi=1, open)"
    dist_punct = uv_distortion(disk, uv_unwrap(disk))
    assert dist_punct > dist_cap, "flattening (most of) a closed sphere distorts more than a hemisphere cap"

    # --- determinism: the UV is a pure function of the mesh ---
    assert np.array_equal(uv_unwrap(flat), uv_unwrap(flat))

    print(f"holographic_meshuv selftest: ok (flat isotropic patch unwraps near-isometric, stretch spread "
          f"{dist_flat:.3f}; curved hemisphere cap {dist_cap:.3f} (Gauss -- unavoidable); punctured sphere "
          f"{dist_punct:.3f} (closed needs a real seam -- the kept negative); UV non-degenerate + deterministic)")


if __name__ == "__main__":
    _selftest()
