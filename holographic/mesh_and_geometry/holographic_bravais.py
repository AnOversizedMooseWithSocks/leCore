"""Bravais crystal lattices + faceted habit (organics backlog C-1/C-3): the 14 translation lattices as point sets.

WHY THIS MODULE EXISTS
----------------------
The organics audit found NO crystal lattice anywhere in the tree: `crystal_material` is a Voronoi COLOUR
socket, `domain_repeat` is an SDF modulo, `fpe_lattice_resonator` is unrelated FHRR factoring. This is the
one genuinely blank slate on the crystals/grass/trees/plants/creatures list, so it earns a module.

A common premise correction, kept loud: there are 7 crystal SYSTEMS but 14 BRAVAIS LATTICES -- the systems
crossed with the centring types (P primitive, I body-centred, F face-centred, C base-centred). We build the
14; the 7 fall out as a grouping. Everything downstream is REUSE: the lattice is just a deterministic point
set, and the engine already eats point sets four ways (metaball_mesh -> atomic blobs, sweep_tube -> bonds,
realize_scatter -> instanced unit cells, scatter_to_field -> density).

THE MATH (why a 3x3 basis is the whole lattice)
    A Bravais lattice is {i*a1 + j*a2 + k*a3} for integer i,j,k, plus a MOTIF of fractional offsets inside
    the cell for the centring: I adds (1/2,1/2,1/2), F adds the three face centres, C adds (1/2,1/2,0).
    The basis vectors come from the cell parameters (a,b,c, alpha,beta,gamma) by the standard
    crystallographic construction: a1 along x, a2 in the xy-plane at gamma to a1, a3 fixed by beta, alpha.

FACETED HABIT (C-3, by reuse)
    A crystal's outward FORM is the intersection of half-spaces whose normals are lattice-plane (Miller
    index) directions in the RECIPROCAL basis. Intersection-of-half-spaces is exactly what the shipped SDF
    algebra does (max of planes), so `crystal_habit` is a thin composer, not a new convex-hull path.

KEPT NEGATIVES (loud)
  * TRANSLATION lattices only -- no space groups (there are 230), no point-symmetry operators, no
    multi-atom chemical bases beyond the centring motif. Scoped so; say it before someone asks.
  * `neighbor_pairs` is O(N^2) by design -- honest and exact for the display-sized lattices this feeds
    (hundreds to a few thousand sites). A cell-hash pass is the documented upgrade IF measurement ever
    shows it pays; do not pre-optimize.
  * HCP is NOT one of the 14 (it is hexagonal-P with a TWO-ATOM basis); provided as the explicit
    convenience `hcp_points` rather than pretending it is a Bravais type.
"""

import hashlib

import numpy as np

#: The 7 crystal systems -> the cell-parameter CONSTRAINTS that define them, and their legal centrings.
#: (system: (constraint description, allowed centrings)) -- the grouping that makes 7 out of the 14.
SYSTEMS = {
    "cubic":        ("a=b=c, alpha=beta=gamma=90", ("P", "I", "F")),
    "tetragonal":   ("a=b!=c, alpha=beta=gamma=90", ("P", "I")),
    "orthorhombic": ("a!=b!=c, alpha=beta=gamma=90", ("P", "I", "F", "C")),
    "hexagonal":    ("a=b!=c, alpha=beta=90, gamma=120", ("P",)),
    "trigonal":     ("a=b=c, alpha=beta=gamma!=90 (rhombohedral)", ("P",)),
    "monoclinic":   ("a!=b!=c, alpha=gamma=90!=beta", ("P", "C")),
    "triclinic":    ("a!=b!=c, alpha!=beta!=gamma", ("P",)),
}

#: Centring motifs: fractional offsets added to every lattice point. P is just the origin.
_MOTIFS = {
    "P": [(0.0, 0.0, 0.0)],
    "I": [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],                                      # body centre
    "F": [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)],   # face centres
    "C": [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0)],                                      # base centre (ab face)
}


def lattice_basis(system, a=1.0, b=None, c=None, alpha=90.0, beta=90.0, gamma=90.0, centring="P"):
    """Build the (3,3) row-vector basis + centring motif for one of the 14 Bravais lattices.

    `system` is one of the 7 crystal systems (see SYSTEMS); the system FILLS IN its own constraints
    (cubic forces b=c=a and right angles, hexagonal forces gamma=120, ...), so the caller states only
    what the system leaves free. Returns (basis, motif): basis rows are a1,a2,a3 in Cartesian space,
    motif is the (m,3) array of fractional centring offsets. Raises on a centring the system forbids --
    that combination is not one of the 14, and silently accepting it would emit a lattice that does
    not exist in nature.
    """
    system = str(system).lower()
    if system not in SYSTEMS:
        raise ValueError("unknown crystal system %r; one of %s" % (system, sorted(SYSTEMS)))
    centring = str(centring).upper()
    if centring not in SYSTEMS[system][1]:
        raise ValueError("centring %r is not legal for %s (legal: %s) -- there are exactly 14 Bravais "
                         "lattices, and this is not one of them" % (centring, system, SYSTEMS[system][1]))

    b = a if b is None else b
    c = a if c is None else c
    # Each system OVERWRITES the parameters its definition constrains -- the constraint is the system.
    if system == "cubic":
        b = c = a; alpha = beta = gamma = 90.0
    elif system == "tetragonal":
        b = a; alpha = beta = gamma = 90.0
    elif system == "orthorhombic":
        alpha = beta = gamma = 90.0
    elif system == "hexagonal":
        b = a; alpha = beta = 90.0; gamma = 120.0
    elif system == "trigonal":
        b = c = a; beta = gamma = alpha                      # rhombohedral: one angle, three times
    elif system == "monoclinic":
        alpha = gamma = 90.0

    al, be, ga = np.radians([alpha, beta, gamma])
    # Standard crystallographic Cartesian construction: a1 || x, a2 in xy at gamma, a3 from beta/alpha.
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / max(np.sin(ga), 1e-12)
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0:
        raise ValueError("cell angles are geometrically impossible (a3 has no real z-component)")
    a3 = np.array([cx, cy, np.sqrt(cz2)])
    return np.stack([a1, a2, a3]), np.asarray(_MOTIFS[centring], float)


def lattice_points(basis, motif, extent=2):
    """All lattice sites i*a1+j*a2+k*a3 + basis@offset for i,j,k in [-extent, extent]: an (N,3) array.

    Deterministic and vectorized (one einsum, no Python loop over sites). `extent` counts unit cells
    each side of the origin, so N = (2*extent+1)^3 * len(motif). This point set is the CONTRACT the
    rest of the engine consumes: metaball_mesh(centers=...), sweep_tube along neighbor_pairs,
    realize_scatter for instanced cells.
    """
    basis = np.asarray(basis, float)
    motif = np.atleast_2d(np.asarray(motif, float))
    r = np.arange(-int(extent), int(extent) + 1)
    ijk = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3).astype(float)
    frac = (ijk[:, None, :] + motif[None, :, :]).reshape(-1, 3)   # every cell x every motif offset
    return frac @ basis


def neighbor_pairs(points, tol=1e-6):
    """Index pairs (i,j), i<j, at the MINIMUM inter-site distance (within tol) -- the nearest-neighbour
    bonds of the lattice, ready for sweep_tube (ball-and-stick). O(N^2) on purpose: exact and honest at
    display sizes; see the module's kept negative before optimizing."""
    P = np.asarray(points, float)
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    d[np.arange(len(P)), np.arange(len(P))] = np.inf
    dmin = d.min()
    ii, jj = np.where(d <= dmin + tol)
    return [(int(i), int(j)) for i, j in zip(ii, jj) if i < j], float(dmin)


def packing_fraction(system, centring, radius=None):
    """The fraction of space filled by touching spheres on this lattice -- the classic textbook numbers,
    used by the selftest as the module's EXACT numeric contract (FCC 0.7405, BCC 0.6802, SC 0.5236)."""
    basis, motif = lattice_basis(system, a=1.0, centring=centring)
    pts = lattice_points(basis, motif, extent=2)
    _, dmin = neighbor_pairs(pts)
    r = dmin / 2.0 if radius is None else float(radius)     # touching spheres: r = half the NN distance
    cell_vol = abs(np.linalg.det(basis))
    n_per_cell = len(motif)
    return n_per_cell * (4.0 / 3.0) * np.pi * r ** 3 / cell_vol


def hcp_points(a=1.0, extent=2):
    """Hexagonal CLOSE-PACKED sites: hexagonal-P lattice with the TWO-ATOM basis (0,0,0), (2/3,1/3,1/2)
    and the ideal c/a = sqrt(8/3). Kept negative made explicit: HCP is NOT one of the 14 Bravais
    lattices (it has a basis), which is why it is a named convenience instead of a 15th entry."""
    c_over_a = np.sqrt(8.0 / 3.0)
    basis, _ = lattice_basis("hexagonal", a=a, c=a * c_over_a)
    motif = np.array([[0.0, 0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 0.5]])
    return lattice_points(basis, motif, extent=extent), basis


def reciprocal_basis(basis):
    """The reciprocal lattice basis b_i (rows), defined by a_i . b_j = 2*pi*delta_ij. Miller-index plane
    normals live HERE, not in the direct basis -- using direct-basis normals gives wrong facets for any
    non-orthogonal cell, which is exactly the bug this helper exists to prevent."""
    A = np.asarray(basis, float)
    return 2.0 * np.pi * np.linalg.inv(A).T


def form_faces(system, hkl):
    """Expand a Miller FORM {hkl} into its symmetry-equivalent faces for `system`.

    WHY THIS EXISTS: in crystallography the braces mean the FORM -- every face equivalent under the
    lattice's point group -- so {100} in the cubic system is SIX faces (a cube) and {111} is EIGHT
    (an octahedron). `crystal_habit` takes explicit faces and applies each with both signs, which is
    documented and correct for what it does, but it means asking for {100} and expecting a cube
    yields a SLAB: measured, occupancy 0.418 of the sample box where a cube is 0.072, and the solid
    is not invariant under a 90-degree turn (pointwise field difference up to 0.97). Nothing warned.

    Cubic takes all PERMUTATIONS of the indices and all sign combinations (the m-3m point group's
    action on a direction). Tetragonal and hexagonal permute only the two equivalent axes, so a
    c-axis face stays distinct from an a-axis one -- the whole reason those systems are not cubic.
    Lower symmetries return the face itself; `crystal_habit` still adds the centrosymmetric pair.
    """
    import itertools
    h = tuple(int(round(float(x))) for x in hkl)
    s = str(system).lower()
    if s == "cubic":
        out = {tuple(p) for p in itertools.permutations(h)}
    elif s in ("tetragonal", "hexagonal", "trigonal", "rhombohedral"):
        out = {(h[0], h[1], h[2]), (h[1], h[0], h[2])}
    elif s == "orthorhombic":
        out = {h}
    else:
        out = {h}
    signed = set()
    for f in out:
        for sg in itertools.product((1, -1), repeat=3):
            signed.add(tuple(int(x * y) for x, y in zip(f, sg)))
    # Drop the antipode of each face: crystal_habit applies both signs itself, so keeping both here
    # would double every plane -- harmless for the max() but confusing in a face count.
    keep = []
    for f in sorted(signed):
        if tuple(-x for x in f) in keep:
            continue
        keep.append(f)
    return [f for f in keep if any(f)]


def crystal_habit(system, miller_faces, sizes, a=1.0, b=None, c=None,
                  alpha=90.0, beta=90.0, gamma=90.0, centring="P", form=False):
    """C-3: a faceted crystal FORM as an SDF -- the intersection of half-spaces whose normals are the
    given Miller-index directions (in the reciprocal basis) at the given centre distances.

    `miller_faces` is a list of (h,k,l); `sizes` a matching list (or scalar) of face distances. Each
    face is applied with BOTH signs (+hkl and -hkl), the centrosymmetric habit. Returns a plain callable
    sdf(P)->distances -- max over the half-space planes, i.e. the exact convex-intersection SDF the
    shipped algebra uses -- so it drops straight into mesh_from_sdf / the raymarcher. Reuse, not a new
    convex-hull path. `sizes` scale by growth progress t to scrub habit growth (G-1)."""
    basis, _ = lattice_basis(system, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma, centring=centring)
    B = reciprocal_basis(basis)
    hkl = np.atleast_2d(np.asarray(miller_faces, float))
    sz = np.broadcast_to(np.asarray(sizes, float), (len(hkl),)).astype(float)
    if form:
        # Expand each given face into its symmetry-equivalent FORM, carrying that face's size.
        _f, _s = [], []
        for _face, _size in zip(hkl, sz):
            for _e in form_faces(system, _face):
                _f.append(_e)
                _s.append(float(_size))
        hkl = np.asarray(_f, float)
        sz = np.asarray(_s, float)
    N = hkl @ B                                              # plane normals in Cartesian space
    N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    N = np.concatenate([N, -N]); D = np.concatenate([sz, sz])   # centrosymmetric: both signs of each face

    def sdf(P):
        """Distance to the faceted habit: max over half-space plane distances (convex intersection)."""
        P = np.atleast_2d(np.asarray(P, float))
        return (P @ N.T - D[None, :]).max(axis=1)

    # BOTH SDF INTERFACES. The docstring says this "drops straight into mesh_from_sdf / the
    # raymarcher", and it does -- but the PATH TRACER calls `sdf.eval(P)` (it needs to march rays
    # through the interior to find the exit face for refraction), and a plain function has no
    # `.eval`. A habit could be meshed and sphere-traced but NOT path-traced, so no crystal could be
    # rendered with the refractive gem materials it exists to wear. One attribute closes it; the
    # function stays a function, so nothing that already calls it changes.
    sdf.eval = sdf
    return sdf


def nucleation_order(points, seed_point=None, seed=0):
    """The order in which lattice sites ACCRETE from a nucleus: indices sorted by distance from
    `seed_point` (default: the site nearest the origin), ties broken deterministically by a hashlib
    hash of the site index -- never by float luck, per the engine's determinism rule. This is the
    scrub axis for crystal growth (G-1): revealing a PREFIX of this order is the physically honest
    picture of a crystal growing outward from a nucleus."""
    P = np.asarray(points, float)
    if seed_point is None:
        seed_point = P[np.argmin(np.linalg.norm(P, axis=1))]
    d = np.linalg.norm(P - np.asarray(seed_point, float), axis=1)
    tie = np.array([int.from_bytes(hashlib.sha256(b"%d:%d" % (seed, i)).digest()[:4], "little")
                    for i in range(len(P))], float)
    return np.lexsort((tie, np.round(d, 9)))                 # distance first, hash breaks exact ties


def _selftest_forms():
    """C-3b: a Miller FORM must be the whole form, and the habit must have the lattice's symmetry."""
    import numpy as _np
    # {100} in the cubic system is SIX faces (a cube), not one face and its antipode (a slab).
    assert len(form_faces("cubic", (1, 0, 0))) == 3, form_faces("cubic", (1, 0, 0))
    assert len(form_faces("cubic", (1, 1, 1))) == 4, form_faces("cubic", (1, 1, 1))

    rng = _np.random.default_rng(1)
    P = rng.uniform(-1.2, 1.2, size=(20000, 3))
    box = 2.4 ** 3
    slab = crystal_habit("cubic", ((1, 0, 0),), 0.5)
    cube = crystal_habit("cubic", ((1, 0, 0),), 0.5, form=True)
    octa = crystal_habit("cubic", ((1, 1, 1),), 0.5, form=True)

    # WITHOUT form expansion {100} is a SLAB -- documented behaviour, and the trap this guards:
    # asking for a cube and getting a slab, silently. Kept as an assertion so the difference is
    # visible rather than surprising.
    v_slab = float((_np.asarray(slab(P), float) < 0).mean()) * box
    v_cube = float((_np.asarray(cube(P), float) < 0).mean()) * box
    v_octa = float((_np.asarray(octa(P), float) < 0).mean()) * box
    assert v_slab > 4.0, "one face pair is a slab, volume %.3f" % v_slab
    assert abs(v_cube - 1.0) < 0.06, "cube of half-size 0.5 has volume 1, got %.4f" % v_cube
    # Octahedron |x|+|y|+|z| <= 0.5*sqrt(3): volume (4/3)a^3 = 0.866.
    assert abs(v_octa - 0.866) < 0.06, "octahedron volume should be 0.866, got %.4f" % v_octa

    # SYMMETRY: a cubic form must be invariant under a 90-degree turn, POINTWISE. Comparing
    # occupancy of two different random samples would hide a real asymmetry inside sampling noise --
    # it did, at 1.7%, until this was written pointwise.
    R90 = _np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    for name, f in (("cube", cube), ("octahedron", octa)):
        a = _np.asarray(f(P), float).ravel()
        b = _np.asarray(f(P @ R90.T), float).ravel()
        assert float(_np.abs(a - b).max()) < 1e-9, \
            "%s must be invariant under a cubic 90-degree turn, max diff %.2e" % (
                name, _np.abs(a - b).max())
    print("bravais forms OK: {100} slab %.2f -> cube %.4f, {111} octahedron %.4f, 90deg symmetric"
          % (v_slab, v_cube, v_octa))


def _selftest():
    """The EXACT numeric contract, not a smoke test: textbook packing fractions to 1e-3, HCP ideal c/a
    to 1e-12, reciprocal-basis identity to 1e-10, habit SDF sign correctness, nucleation determinism."""
    # 1) Packing fractions -- the classic numbers, the strongest possible correctness anchor.
    assert abs(packing_fraction("cubic", "F") - 0.74048) < 1e-3, "FCC packing"
    assert abs(packing_fraction("cubic", "I") - 0.68017) < 1e-3, "BCC packing"
    assert abs(packing_fraction("cubic", "P") - 0.52360) < 1e-3, "SC packing"

    # 2) FCC nearest-neighbour count: every interior site touches 12 others.
    basis, motif = lattice_basis("cubic", a=1.0, centring="F")
    pts = lattice_points(basis, motif, extent=2)
    pairs, dmin = neighbor_pairs(pts)
    assert abs(dmin - np.sqrt(0.5)) < 1e-9, "FCC NN distance a/sqrt(2)"
    centre = int(np.argmin(np.linalg.norm(pts, axis=1)))
    deg = sum(1 for i, j in pairs if centre in (i, j))
    assert deg == 12, "FCC coordination number is 12, got %d" % deg

    # 3) All 14 build; the 15th (illegal centring) refuses.
    n_built = 0
    for sysname, (_, cents) in SYSTEMS.items():
        for cn in cents:
            bb, mm = lattice_basis(sysname, a=1.0, b=1.3, c=1.7, alpha=75.0, beta=80.0, gamma=85.0, centring=cn)
            assert np.linalg.det(bb) > 0
            n_built += 1
    assert n_built == 14, "there are exactly 14 Bravais lattices, built %d" % n_built
    try:
        lattice_basis("cubic", centring="C"); raise AssertionError("cubic-C must refuse")
    except ValueError:
        pass

    # 4) HCP ideal ratio, exact.
    hp, hb = hcp_points(a=1.0, extent=1)
    assert abs(np.linalg.norm(hb[2]) / np.linalg.norm(hb[0]) - np.sqrt(8.0 / 3.0)) < 1e-12, "HCP c/a"

    # 5) Reciprocal basis identity a_i . b_j = 2 pi delta_ij, to 1e-10, on a fully triclinic cell.
    tb, _ = lattice_basis("triclinic", a=1.0, b=1.3, c=1.7, alpha=75.0, beta=80.0, gamma=85.0)
    G = tb @ reciprocal_basis(tb).T
    assert np.abs(G - 2 * np.pi * np.eye(3)).max() < 1e-10, "reciprocal identity"

    # 6) Habit SDF: origin inside (negative), far point outside (positive); octahedron {111} has 8 faces.
    sdf = crystal_habit("cubic", [(1, 1, 1)], 0.5)
    assert sdf(np.zeros((1, 3)))[0] < 0 and sdf(np.array([[5.0, 0, 0]]))[0] > 0, "habit sign"

    # 7) Nucleation order: deterministic across calls, starts at the nucleus, distance-monotone.
    o1 = nucleation_order(pts, seed=3); o2 = nucleation_order(pts, seed=3)
    assert np.array_equal(o1, o2), "nucleation order must be a pure function"
    dseq = np.linalg.norm(pts[o1] - pts[o1[0]], axis=1)
    assert np.all(np.diff(np.round(dseq, 9)) >= 0), "accretion must be distance-monotone"

    print("holographic_bravais selftest OK: 14 lattices, FCC 0.7405 / BCC 0.6802 / SC 0.5236, "
          "coordination 12, HCP c/a sqrt(8/3), reciprocal identity 1e-10")


if __name__ == "__main__":
    _selftest()
    _selftest_forms()
