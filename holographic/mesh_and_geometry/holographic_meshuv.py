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
      'spectral' -- the shipped Laplacian-eigenmaps chart (local structure, not a faithful metric).
      'lscm'     -- least-squares CONFORMAL map (Levy et al. 2002): the angle-preserving chart, exact on a
                    developable surface (1.000000) and best on angle everywhere. It pays in AREA -- 0.4420 spread
                    on a hemisphere cap against isomap's 0.2957. Compare charts on the functional they optimise;
                    `uv_report` prints all three metrics so nobody has to choose one blind."""
    if method == "lscm":
        uv = lscm(mesh)
    elif method == "planar":
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


def _triangle_frame(p):
    """A triangle's own 2-D coordinates: origin at p0, x along e1, y in the plane. The frame every per-face
    Jacobian is measured in."""
    e1, e2 = p[1] - p[0], p[2] - p[0]
    n = np.cross(e1, e2)
    A = float(np.linalg.norm(n))
    if A < 1e-14:
        return None, 0.0
    ex = e1 / np.linalg.norm(e1)
    ey = np.cross(n / A, ex)
    return np.array([[e1 @ ex, e2 @ ex], [e1 @ ey, e2 @ ey]]), A


def lscm(mesh, pins=None):
    """LEAST-SQUARES CONFORMAL MAP (Levy, Petitjean, Ray & Maillot, SIGGRAPH 2002) -- the angle-preserving unwrap.

    Per triangle, conformality is the Cauchy-Riemann condition, which in the triangle's own frame is the single
    complex equation `sum_j W_j z_j = 0` with `W_j` the opposite edge as a complex number and `z_j = u_j + i v_j`.
    Stack one row per face, weight by 1/sqrt(area) so large triangles do not dominate, PIN two vertices to fix the
    map's remaining similarity freedom, and solve the (real-embedded) least-squares system. One linear solve, no
    iteration, no autodiff.

    `pins` is `[(vertex, u, v), (vertex, u, v)]`; by default the two FARTHEST-APART vertices are pinned to (0,0)
    and (1,0), chosen deterministically (lowest index first on a tie) because the pin choice changes the map.

    MEASURED against the other charts (mean quasi-conformal ratio sigma1/sigma2; 1.0 is conformal):

        surface            lscm      isomap    planar
        flat patch      1.00000     1.10866   1.00000     <- exact on a developable surface
        hemisphere cap  1.08570     1.87790   3.11662

    KEPT NEGATIVE 1 -- **LSCM buys angles with AREA.** On the same cap its area-distortion spread is 0.4420
    against isomap's 0.2957. That is not a defect; it is the definition of a conformal map. Compare charts on the
    metric they optimise, or you will conclude the wrong thing.

    KEPT NEGATIVE 2 -- **LSCM does not guarantee a fold-free map**, and neither does anything else here. Stretch the
    cap's z by 6 and LSCM flips 72 of 256 triangles. Levy et al. say as much: the pins matter, and a free-boundary
    conformal map on a high-curvature patch can invert. Use `uv_angle_distortion`'s `flipped` count."""
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    nV, nF = len(V), len(F)
    if nF == 0 or nV < 3:
        raise ValueError("lscm needs at least one triangle")

    if pins is None:
        d2 = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)   # ties -> lowest flat index: deterministic
        pins = [(int(min(i, j)), 0.0, 0.0), (int(max(i, j)), 1.0, 0.0)]
    if len(pins) != 2 or pins[0][0] == pins[1][0]:
        raise ValueError("lscm needs exactly two DISTINCT pinned vertices")

    pin_idx = [int(p[0]) for p in pins]
    free = [k for k in range(nV) if k not in pin_idx]
    M = np.zeros((nF, nV), complex)
    for t, (a, b, c) in enumerate(F):
        D, A = _triangle_frame(V[[a, b, c]])
        if D is None:
            continue                                            # a degenerate face contributes nothing
        loc = np.array([[0.0, 0.0], [D[0, 0], D[1, 0]], [D[0, 1], D[1, 1]]])
        W = [complex(loc[2, 0] - loc[1, 0], loc[2, 1] - loc[1, 1]),
             complex(loc[0, 0] - loc[2, 0], loc[0, 1] - loc[2, 1]),
             complex(loc[1, 0] - loc[0, 0], loc[1, 1] - loc[0, 1])]
        for k, vi in enumerate((a, b, c)):
            M[t, vi] += W[k] / np.sqrt(A)                       # area weighting: no triangle dominates

    Mf, Mp = M[:, free], M[:, pin_idx]
    zp = np.array([complex(p[1], p[2]) for p in pins])
    rhs = -Mp @ zp
    # real embedding of the complex least-squares problem: [Re -Im; Im Re] [u; v] = [Re rhs; Im rhs]
    A2 = np.vstack([np.hstack([Mf.real, -Mf.imag]), np.hstack([Mf.imag, Mf.real])])
    b2 = np.concatenate([rhs.real, rhs.imag])
    sol = np.linalg.lstsq(A2, b2, rcond=None)[0]

    nf = len(free)
    uv = np.zeros((nV, 2))
    uv[free, 0] = sol[:nf]
    uv[free, 1] = sol[nf:]
    for (vi, u, v) in pins:
        uv[int(vi)] = (u, v)
    return uv


def uv_angle_distortion(mesh, uv):
    """The metric LSCM actually optimises: per-face quasi-conformal ratio `sigma1 / sigma2` of the 3-D -> UV
    Jacobian. Returns `{mean, median, max, flipped, n_faces}`. **1.0 is conformal.**

    Report the MEDIAN. The mean is UNBOUNDED: a single near-degenerate triangle sends sigma2 -> 0 and the ratio to
    infinity. Measured on a cap stretched 6x in z, LSCM's mean is 398.0 and its median 4.8, and reading the mean as
    "the map is 398x distorted" is wrong.

    And report `flipped`. **Neither `uv_distortion` (stretch) nor the mean ratio can see a FOLD.** On a cap
    stretched 6x in z, `isomap` has a *better* mean ratio than LSCM (2.573 vs 398.038) while folding half its map.
    Every scalar summary above says it is fine. A chart with folds is not a chart.

    A FOLD IS NOT A MIRROR. `flipped` counts faces whose orientation disagrees with the MAJORITY, not faces with
    `det J < 0`. A chart may be globally mirrored -- classical MDS and a PCA plane routinely return one -- and then
    EVERY face has a negative determinant while nothing is folded at all. My first version counted the sign, and
    reported `planar` as flipping 256 of 256 faces on a hemisphere cap. It had folded none of them."""
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    uv = np.asarray(uv, float)
    ratios, signs = [], []
    for a, b, c in F:
        D, _A = _triangle_frame(V[[a, b, c]])
        if D is None or abs(np.linalg.det(D)) < 1e-14:
            continue
        Q = uv[[a, b, c]]
        E = np.array([Q[1] - Q[0], Q[2] - Q[0]]).T
        J = E @ np.linalg.inv(D)
        signs.append(1 if np.linalg.det(J) >= 0.0 else -1)
        s = np.linalg.svd(J, compute_uv=False)
        if s[1] > 1e-14:
            ratios.append(float(s[0] / s[1]))
    if not ratios:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "flipped": 0, "n_faces": int(len(F))}
    sg = np.asarray(signs)
    flipped = int(min(int((sg > 0).sum()), int((sg < 0).sum())))   # the MINORITY orientation: the folds
    r = np.asarray(ratios)
    return {"mean": float(r.mean()), "median": float(np.median(r)), "max": float(r.max()),
            "flipped": flipped, "n_faces": int(len(F))}


def uv_area_distortion(mesh, uv):
    """The log-spread of (UV face area / 3-D face area), normalised by the median. 0 = area-preserving.

    This is the price a conformal map pays: on a hemisphere cap LSCM scores 0.4420 here against isomap's 0.2957,
    while beating it 1.086 to 1.878 on angle. **Different charts optimise different functionals; compare them on
    the one they optimise, and report the other.**"""
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    uv = np.asarray(uv, float)
    ratios = []
    for a, b, c in F:
        p = V[[a, b, c]]
        A3 = 0.5 * float(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])))
        q = uv[[a, b, c]]
        e1, e2 = q[1] - q[0], q[2] - q[0]
        A2 = 0.5 * abs(float(e1[0] * e2[1] - e1[1] * e2[0]))
        if A3 > 1e-12 and A2 > 1e-12:
            ratios.append(A2 / A3)
    if not ratios:
        return 0.0
    r = np.asarray(ratios)
    return float(np.std(np.log(r / np.median(r))))


def uv_report(mesh, methods=("lscm", "isomap", "planar")):
    """Every chart on every metric, so nobody compares them on the wrong one. Returns
    `{method: {angle, area, stretch}}` with `angle` the full quasi-conformal dict including `flipped`."""
    out = {}
    for meth in methods:
        uv = lscm(mesh) if meth == "lscm" else uv_unwrap(mesh, method=meth)
        out[meth] = {"angle": uv_angle_distortion(mesh, uv),
                     "area": uv_area_distortion(mesh, uv),
                     "stretch": uv_distortion(mesh, uv)}
    return out


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


def _mesh_components(mesh):
    """Split `mesh` into its connected components (per FACE adjacency via shared vertices), returning a list of
    (vertex-index array, remapped face list) for each. The 'find the UV islands' step: a mesh made of separate
    pieces (or one cut open along seams) has one island per component. Deterministic (components ordered by their
    smallest vertex id)."""
    faces = [tuple(f) for f in mesh.faces]
    # union-find over vertices joined by any shared face.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    for f in faces:
        for v in f[1:]:
            union(f[0], v)
    groups = {}
    for f in faces:
        r = find(f[0])
        groups.setdefault(r, []).append(f)
    out = []
    for r in sorted(groups):
        gf = groups[r]
        used = sorted({v for f in gf for v in f})
        remap = {old: i for i, old in enumerate(used)}
        out.append((np.asarray(used, int), [tuple(remap[v] for v in f) for f in gf]))
    return out


def pack_uv_islands(mesh, method="lscm", margin=0.02):
    """SMART-UV-style island packing: unwrap each connected component (UV island) of `mesh` SEPARATELY, then lay the
    islands out in a non-overlapping grid inside the unit UV square [0,1]^2. Returns a (n_vertices, 2) UV array, one
    row per original vertex, so islands never overlap -- the 'pack islands' step LSCM/uv_unwrap alone skip (they solve
    every component in one frame, so disconnected pieces land on top of each other).

    method: 'lscm' (conformal, per island) or 'isomap' (uv_unwrap's geodesic MDS). Each island is unwrapped, shifted
    to the origin, uniformly scaled to fit its cell (aspect preserved -- no UV stretch), and placed with a `margin`
    gutter. The layout is a near-square grid of ceil(sqrt(k)) columns for k islands (deterministic). Composes the
    existing per-chart unwrap (lscm / uv_unwrap) + connected-components split; it does NOT re-solve the unwrap."""
    comps = _mesh_components(mesh)
    k = len(comps)
    uv = np.zeros((mesh.n_vertices, 2), float)
    if k == 0:
        return uv
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    cell_w = 1.0 / cols
    cell_h = 1.0 / rows
    for idx, (used, faces) in enumerate(comps):
        island = Mesh(mesh.vertices[used], faces)
        if method == "isomap":
            iuv = np.asarray(uv_unwrap(island))
        else:
            iuv = np.asarray(lscm(island))
        # normalise the island to [0,1] preserving aspect (uniform scale by the larger extent), then fit its cell.
        lo = iuv.min(axis=0)
        span = np.ptp(iuv, axis=0)
        scale = float(max(span[0], span[1]))
        if scale < 1e-12:
            scale = 1.0                                          # a degenerate (collapsed) island -> avoid /0
        norm = (iuv - lo) / scale                               # in [0,1] for the larger axis, <=1 for the other
        col = idx % cols
        row = idx // cols
        ox = col * cell_w + margin * cell_w
        oy = row * cell_h + margin * cell_h
        avail_w = cell_w * (1.0 - 2.0 * margin)
        avail_h = cell_h * (1.0 - 2.0 * margin)
        fit = min(avail_w, avail_h)                             # uniform fit -> no anisotropic UV stretch
        placed = norm * fit + np.array([ox, oy])
        uv[used] = placed
    return uv


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

    # --- PACK_UV_ISLANDS: a 2-component mesh packs into NON-OVERLAPPING cells of the unit square ---
    two_v = np.vstack([flat.vertices, flat.vertices + np.array([10.0, 0, 0])])   # two separate copies
    nf = len(flat.vertices)
    two = Mesh(two_v, [tuple(f) for f in flat.faces] +
              [tuple(v + nf for v in f) for f in flat.faces])
    puv = pack_uv_islands(two)
    assert puv.shape == (two.n_vertices, 2)
    assert puv.min() >= -1e-9 and puv.max() <= 1 + 1e-9, "packed UVs must lie in the unit square"
    a, b = puv[:nf], puv[nf:]                                    # the two islands' UV bboxes must be disjoint
    ax0, ax1, ay0, ay1 = a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max()
    bx0, bx1, by0, by1 = b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max()
    disjoint = (ax1 <= bx0 + 1e-9) or (bx1 <= ax0 + 1e-9) or (ay1 <= by0 + 1e-9) or (by1 <= ay0 + 1e-9)
    assert disjoint, "packed islands must not overlap in UV space"
    assert np.array_equal(pack_uv_islands(two), pack_uv_islands(two)), "packing is deterministic"

    print(f"holographic_meshuv selftest: ok (flat isotropic patch unwraps near-isometric, stretch spread "
          f"{dist_flat:.3f}; curved hemisphere cap {dist_cap:.3f} (Gauss -- unavoidable); punctured sphere "
          f"{dist_punct:.3f} (closed needs a real seam -- the kept negative); pack_uv_islands lays 2 components into "
          f"disjoint cells of the unit square; UV non-degenerate + deterministic)")


if __name__ == "__main__":
    _selftest()
