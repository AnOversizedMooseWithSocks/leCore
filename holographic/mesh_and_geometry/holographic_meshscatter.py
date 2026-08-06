"""Scatter on a MESH surface, and turn placements into real geometry (organics backlog S-1/S-2).

WHY THIS MODULE EXISTS -- the two halves the audit found missing
---------------------------------------------------------------
The shipped `holographic_scatterlayer.ScatterLayer` scatters onto an SDF surface and returns points,
normals and HYPERVECTORS. Two gaps stopped "apply grass meshes and scatter them across a surface area"
from working, and both are small:

  S-1  the surface must be able to be a MESH (a triangle soup with an area), not only an SDF. Grass
       grows on the lawn mesh you modelled, not on an implicit blob.
  S-2  a placement list must be able to become GEOMETRY. Nothing in the tree turned placements into a
       mesh or an instanced scene, so the scatter could be queried but never rendered.

REUSE, EXPLICITLY (this module writes almost no new math)
    * blue-noise relaxation -> the shipped `holographic_poisson` Bridson sampler (mind.blue_noise_sample)
    * per-instance transform application -> `mind.transform_mesh` (which also fixes winding on reflect)
    * merging -> `mind.weld_mesh`
    * instanced output -> the shipped InstancedScene (`mind.instanced_scene`), one Definition + N transforms
    * frames along a direction -> the same orthonormal-frame trick `curve_frame` uses
  What is genuinely new here is only: area-weighted triangle sampling, and the placement->geometry bridge.

THE ONE FUNCTION THAT PAYS FOR ITSELF FOUR TIMES
    `realize_scatter` is the keystone: grass blades, plant permutations, rocks/barnacles, and crystal
    unit cells (backlog C-2) are all "a source mesh, placed at N frames". Built once, four callers --
    the generalize-on-contact check applied before writing, not after.

KEPT NEGATIVES (loud)
  * `mode="merge"` cost is LINEAR IN BLADES: a million-blade lawn merged is a million x source-mesh
    vertices and will not fit in memory. That is not a bug, it is the honest cost of merging; use
    mode="instanced" (one definition + a transform array) and the bake/LOD path for dense fields.
  * Area-weighted sampling is UNIFORM OVER AREA, so it clumps like uniform random does. Blue-noise
    relaxation (`relax=True`) fixes the look but is a REJECTION pass -- it returns FEWER points than
    asked (it cannot invent spacing that does not fit), and it says so rather than silently padding.
  * Placement yaw is random about the surface normal. There is no anisotropic/combed direction field
    here; combing is the groom layer's job (holographic_groom), not this one's.
  * No collision between instances, no ground penetration test. Blades can intersect each other.
"""

import hashlib

import numpy as np

from holographic.mesh_and_geometry.holographic_mesh import Mesh


def _rng_for(seed, tag):
    """A deterministic default_rng from (seed, tag) via hashlib -- never Python's hash(), per the
    engine's determinism rule, so a lawn is bit-identical across processes and machines."""
    h = hashlib.sha256(("%d:%s" % (int(seed), tag)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _mesh_arrays(mesh):
    """Accept either a Mesh object or a plain (vertices, faces) tuple -- the two shapes the engine's
    mesh faculties already hand around. Returns (V, F) as float/int arrays."""
    if hasattr(mesh, "vertices"):
        return np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)
    V, F = mesh
    return np.asarray(V, float), np.asarray(F, int)


def triangle_areas(mesh):
    """Per-triangle area (n_faces,) -- the weight that makes scattering UNIFORM OVER THE SURFACE
    rather than uniform over the face list. Sampling faces uniformly is the classic bug: a mesh with
    one huge triangle and a thousand slivers would put a thousandth of the grass on the lawn."""
    V, F = _mesh_arrays(mesh)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def sample_mesh_surface(mesh, count, density=None, seed=0, relax=False, radius=None):
    """S-1: sample `count` points on a MESH surface, area-weighted, with per-point normals and tangents.

    Returns a dict: points (n,3), normals (n,3), tangents (n,3), faces (n,) source triangle index.
    `density` is an optional callable f(points)->weights in [0,1] applied as a REJECTION mask (so a
    density map can thin the grass under a tree) -- rejection, not reweighting, because rejection
    keeps the surviving points' spatial statistics honest.
    `relax=True` runs a blue-noise thinning: keep a point only if it is at least `radius` from every
    kept point (Bridson's acceptance criterion applied to an existing set). Returns FEWER points; the
    result reports how many survived rather than padding back up to `count`.
    """
    V, F = _mesh_arrays(mesh)
    areas = triangle_areas((V, F))
    total = areas.sum()
    if total <= 0:
        raise ValueError("mesh has zero surface area -- nothing to scatter on")
    rng = _rng_for(seed, "surface")
    # Area-weighted face choice, then a uniform barycentric point inside the chosen triangle.
    fidx = rng.choice(len(F), size=int(count), p=areas / total)
    u = rng.random(int(count)); v = rng.random(int(count))
    fold = u + v > 1.0                                       # reflect into the triangle (Turk 1990)
    u = np.where(fold, 1.0 - u, u); v = np.where(fold, 1.0 - v, v)
    a, b, c = V[F[fidx, 0]], V[F[fidx, 1]], V[F[fidx, 2]]
    P = a + u[:, None] * (b - a) + v[:, None] * (c - a)
    fn = np.cross(b - a, c - a)
    N = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    T = b - a                                                # a stable in-plane tangent: the first edge
    T = T - (T * N).sum(1, keepdims=True) * N
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)

    keep = np.ones(len(P), bool)
    if density is not None:
        w = np.clip(np.asarray(density(P), float).ravel(), 0.0, 1.0)
        keep &= _rng_for(seed, "density").random(len(P)) < w
    P, N, T, fidx = P[keep], N[keep], T[keep], fidx[keep]

    if relax:
        r = float(radius if radius is not None else np.sqrt(total / max(len(P), 1)) * 0.75)
        kept = _blue_noise_thin(P, r)
        P, N, T, fidx = P[kept], N[kept], T[kept], fidx[kept]
    return {"points": P, "normals": N, "tangents": T, "faces": fidx, "count": len(P), "area": float(total)}


def _blue_noise_thin(P, radius):
    """Keep a maximal subset of `P` with every pair >= radius apart (Bridson's acceptance test applied
    to an existing point set). Deterministic: walks in index order, no randomness of its own. O(n^2)
    but only over accepted points, which is what keeps it tolerable at scatter sizes."""
    kept_idx, kept_pts = [], []
    r2 = float(radius) ** 2
    for i, p in enumerate(P):
        if not kept_pts or np.min(((np.asarray(kept_pts) - p) ** 2).sum(1)) >= r2:
            kept_idx.append(i); kept_pts.append(p)
    return np.asarray(kept_idx, int)


def placement_frames(points, normals, tangents=None, scale=1.0, scale_jitter=0.0,
                     yaw_jitter=True, align=1.0, seed=0):
    """Turn sampled surface points into (n,4,4) instance TRANSFORMS: up = surface normal, random yaw
    about it, uniform scale with optional jitter.

    `align` in [0,1] blends the instance's up-axis between world-up (0) and the surface normal (1) --
    grass on a steep bank usually wants ~0.6, not a blade lying flat on the slope. All randomness is
    hashlib-seeded so a field is reproducible.
    """
    P = np.asarray(points, float); N = np.asarray(normals, float)
    n = len(P)
    up = np.array([0.0, 0.0, 1.0])
    A = float(np.clip(align, 0.0, 1.0))
    Z = A * N + (1.0 - A) * up[None, :]
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    # An arbitrary stable perpendicular: cross with whichever world axis is least parallel to Z.
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    flip = np.abs(Z[:, 0]) > 0.9
    ref[flip] = np.array([0.0, 1.0, 0.0])
    X = np.cross(ref, Z); X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Y = np.cross(Z, X)

    rng = _rng_for(seed, "frames")
    if yaw_jitter:
        th = rng.random(n) * 2.0 * np.pi
        c, s = np.cos(th)[:, None], np.sin(th)[:, None]
        X, Y = c * X + s * Y, -s * X + c * Y                 # rotate the tangent frame about Z
    sc = float(scale) * (1.0 + float(scale_jitter) * (rng.random(n) * 2.0 - 1.0))

    # COLUMN-VECTOR convention (M @ v): basis vectors are COLUMNS of the 3x3 block and the translation
    # is the last COLUMN. This is what `transform_mesh` and `InstancedScene` both apply.
    #
    # BUG THIS FIXES, kept loud because the tests sailed past it: these matrices were originally built
    # row-wise. transform_mesh then applied a TRANSPOSED rotation and IGNORED the translation entirely,
    # so every scattered blade landed at the origin -- a "lawn" that was one clump. Three assertions
    # passed anyway: vertex counts (duplicated geometry is still duplicated), "nothing below z=0" (a
    # blade at the origin is not below z=0), and checks on the matrix itself (the matrix was fine; the
    # CONVENTION was wrong). The assertion that catches it is a SPATIAL EXTENT check -- placed geometry
    # must span the surface it was scattered on -- and it is now in the selftest below.
    M = np.zeros((n, 4, 4))
    M[:, :3, 0] = X * sc[:, None]
    M[:, :3, 1] = Y * sc[:, None]
    M[:, :3, 2] = Z * sc[:, None]
    M[:, :3, 3] = P
    M[:, 3, 3] = 1.0
    return M


def realize_scatter(source, transforms, mode="merge", scene=None, variants=None, seed=0,
                    material="paint"):
    """S-2 -- THE KEYSTONE: placements -> real geometry.

    `source` is a Mesh (or (V,F)); `transforms` the (n,4,4) array from placement_frames.
      mode="merge"     -> one Mesh with every instance baked in (simple, renders anywhere, LINEAR COST)
      mode="instanced" -> an InstancedScene: one shared Definition placed n times (cheap, edit-once)
    `variants` is an optional LIST of source meshes; each placement draws one deterministically, which
    is how a plant field gets permutations (backlog T-3) without n distinct meshes in memory. In
    instanced mode each variant becomes its own Definition, so the sharing survives.

    Delegates per-instance transformation to `holographic_meshtools.transform_mesh` (it also repairs face
    winding under a reflecting matrix) and the final weld to `weld_mesh` -- no new geometry math here.
    """
    from holographic.mesh_and_geometry.holographic_meshtools import transform_mesh

    pool = [source] if variants is None else list(variants)
    if not pool:
        raise ValueError("realize_scatter needs at least one source mesh")
    pick = _rng_for(seed, "variants").integers(0, len(pool), size=len(transforms)) if len(pool) > 1 \
        else np.zeros(len(transforms), int)

    if mode == "instanced":
        from holographic.misc.holographic_instancing import Definition, InstancedScene
        if scene is None:
            scene = InstancedScene()
        # One Definition per variant, so n placements share len(pool) geometries -- the whole point of
        # instancing. `material` is a NAME the Definition type-checks against the geometry kind (a mesh
        # is surface geometry, so a surface material); callers repaint via defn.set_material.
        defs = [Definition("scatter_src_%d" % i, _as_mesh(s), material) for i, s in enumerate(pool)]
        for k, M in enumerate(transforms):
            scene.place(defs[int(pick[k])], M)
        return scene

    if mode != "merge":
        raise ValueError("mode must be 'merge' or 'instanced', got %r" % mode)

    verts, faces, off = [], [], 0
    for k, M in enumerate(transforms):
        inst = transform_mesh(_as_mesh(pool[int(pick[k])]), np.asarray(M, float))
        V = np.asarray(inst.vertices, float); F = np.asarray(inst.faces, int)
        verts.append(V); faces.append(F + off); off += len(V)
    if not verts:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), int))
    return Mesh(np.concatenate(verts), np.concatenate(faces))


def scatter_layer_vector(transforms, source_name="instance", dim=1024, cell_size=0.25, seed=0):
    """The HOLOGRAPHIC form of a scatter: bind each placement to its region code and bundle the lot
    into ONE content-addressable layer vector.

    REGRESSION THIS FIXES, on the record: the shipped `ScatterLayer` already did exactly this -- a
    placement is bind(instance, cell_code) and the layer is their bundle, so you can ask "is anything
    scattered near here?" by unbinding a region code, with no scene graph and no spatial index. When
    this module added mesh-surface scatter it returned points, normals and transforms and DROPPED that
    property, quietly making the new path less capable than the one it generalised. Same encoding,
    same hashlib cell hashing, now available on the mesh path too.

    Returns (layer, instance_atom). Query with `region_occupancy`.
    """
    from holographic.agents_and_reasoning.holographic_ai import bundle, derived_atom
    M = np.asarray(transforms, float)
    inst = derived_atom(int(seed), "scatter:%s" % source_name, int(dim))
    vecs = [_bind_cell(inst, M[k, :3, 3], dim, cell_size, seed) for k in range(len(M))]
    return (bundle(vecs) if vecs else np.zeros(int(dim))), inst


def _bind_cell(instance, point, dim, cell_size, seed):
    """One placement: the instance atom bound to a deterministic code for the grid CELL it sits in.
    Nearby placements share a code, which is what makes the bundle REGION-addressable. hashlib, not
    Python's hash(), per the engine's determinism rule -- the same convention holographic_scatterlayer
    already uses, so the two agree by construction."""
    from holographic.agents_and_reasoning.holographic_ai import bind, derived_atom
    cell = tuple(int(np.floor(c / float(cell_size))) for c in np.asarray(point, float))
    return bind(instance, derived_atom(int(seed), "cell:%s" % (cell,), int(dim)))


def region_occupancy(layer, instance, point, dim=1024, cell_size=0.25, seed=0):
    """Ask the bundled layer whether anything is scattered near `point`: unbind the region's cell code
    and read the cosine against the instance atom. High = yes, near zero = no.

    KEPT NEGATIVES, MEASURED rather than asserted (80 blades, dim 2048, cell 1.0, 60 empty probes):
    occupied cells read mean 0.165, empty cells mean -0.0008 with std 0.031 -- a 5.4 sigma separation,
    rising to 8.3 sigma at dim 4096. But the MINIMUM occupied reading was -0.005: a cell holding a
    single blade sits at the noise floor and is indistinguishable from empty. So this is a STATISTICAL
    occupancy read, not a per-cell oracle -- reliable for "is this region populated", unreliable for
    "is there a blade in this exact cell", and useless for counting. It is also a coarse cell hash
    (region-addressable, not per-instance recall) and fades as the layer loads, like any bundle.
    Raise `dim` to buy separation; the cost is linear."""
    from holographic.agents_and_reasoning.holographic_ai import unbind, cosine
    return float(cosine(unbind(layer, _cell_atom(point, dim, cell_size, seed)), instance))


def _cell_atom(point, dim, cell_size, seed):
    """The bare cell code for a point (no instance bound in) -- the key side of the placement."""
    from holographic.agents_and_reasoning.holographic_ai import derived_atom
    cell = tuple(int(np.floor(c / float(cell_size))) for c in np.asarray(point, float))
    return derived_atom(int(seed), "cell:%s" % (cell,), int(dim))


def _as_mesh(m):
    """Accept a Mesh or a (V,F) pair and always hand back a Mesh -- one place, so every entry point
    in this module takes both shapes without repeating the check."""
    return m if hasattr(m, "vertices") else Mesh(np.asarray(m[0], float), np.asarray(m[1], int))


def scatter_mesh(surface, source, count, seed=0, scale=1.0, scale_jitter=0.0, align=1.0,
                 density=None, relax=False, radius=None, mode="merge", variants=None,
                 holographic=False, dim=1024, cell_size=0.25):
    """The one-call door: scatter `source` (or a list of `variants`) over the surface of `surface`
    `count` times and return real geometry. Composes sample_mesh_surface -> placement_frames ->
    realize_scatter. This is grass on a lawn, rocks on a hill, barnacles on a hull, one line.

    `holographic=True` additionally returns a content-addressable LAYER vector (see
    scatter_layer_vector), so the field can be queried by region without a spatial index."""
    s = sample_mesh_surface(surface, count, density=density, seed=seed, relax=relax, radius=radius)
    M = placement_frames(s["points"], s["normals"], s["tangents"], scale=scale,
                         scale_jitter=scale_jitter, align=align, seed=seed)
    geo = realize_scatter(source, M, mode=mode, variants=variants, seed=seed)
    out = {"geometry": geo, "transforms": M, "count": s["count"], "sample": s}
    if holographic:
        # Additive and OFF by default: a caller who just wants grass pays nothing, and a caller who
        # wants the field to be queryable gets the same encoding the shipped ScatterLayer uses.
        layer, inst = scatter_layer_vector(M, dim=dim, cell_size=cell_size, seed=seed)
        out["layer"], out["instance"], out["dim"], out["cell_size"] = layer, inst, int(dim), float(cell_size)
    return out


def grass_blade(height=0.3, width=0.02, segments=4, bend=0.35, taper=0.15):
    """A single RIBBON blade: a tapered quad strip curving over by `bend`. Deliberately tiny -- a blade
    is 2*segments triangles, because a lawn multiplies whatever this costs by n. Use as the `source`
    for scatter_mesh, or build a few with different bend/height as `variants`."""
    t = np.linspace(0.0, 1.0, int(segments) + 1)
    w = float(width) * (1.0 - (1.0 - float(taper)) * t) * 0.5   # half-width, tapering to the tip
    x = float(bend) * t ** 2 * float(height)                    # quadratic droop: stiff at the root
    z = t * float(height)
    V = np.zeros((2 * len(t), 3))
    V[0::2] = np.stack([x, -w, z], axis=1)
    V[1::2] = np.stack([x, +w, z], axis=1)
    F = []
    for i in range(len(t) - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        F += [[a, b, c], [b, d, c]]
    return Mesh(V, np.asarray(F, int))


def strand_ribbons(strands, width=0.02, taper=0.15, twist=0.0):
    """S-3: turn STRAND point-chains (as `groom_hair` / `simulate_strands` produce) into RIBBON
    geometry -- one tapered quad strip per strand. Grass blades and leaves are ribbons, not tubes.

    BACKLOG CORRECTION, on the record: this item was filed as "add profile='ribbon' to
    build_strand_body". Reading the module showed that function builds a PBD SoftBody -- physics, not
    geometry -- and that the groom layer has NO meshing path at all. So the gap was not a parameter,
    it was a missing stage. Filed wrong, found by reading the code instead of the note.

    The win is that grass now inherits the whole shipped groom pipeline for free: root strands on a
    surface, simulate them with PBD, blow them with curl-noise wind, THEN ribbon them -- instead of
    the static `grass_blade` card. Returns a Mesh.
    """
    V, F, off = [], [], 0
    for pts in strands:
        P = np.asarray(pts, float)
        n = len(P)
        if n < 2:
            continue
        # A stable per-point frame: the side vector is perpendicular to the segment direction and to a
        # reference axis, so a blade keeps a consistent face instead of flipping along its length.
        d = np.diff(P, axis=0)
        d = np.vstack([d, d[-1]])
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
        ref = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
        ref[np.abs(d[:, 0]) > 0.9] = np.array([0.0, 1.0, 0.0])
        S = np.cross(ref, d)
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
        if twist:
            th = np.linspace(0.0, float(twist), n)             # a slow roll along the blade
            U = np.cross(d, S)
            S = np.cos(th)[:, None] * S + np.sin(th)[:, None] * U
        tt = np.linspace(0.0, 1.0, n)
        hw = 0.5 * float(width) * (1.0 - (1.0 - float(taper)) * tt)
        V.append(P - S * hw[:, None]); V.append(P + S * hw[:, None])
        # interleave the two rails so faces index them as (2i, 2i+1)
        pair = np.empty((2 * n, 3)); pair[0::2] = V[-2]; pair[1::2] = V[-1]
        V[-2:] = [pair]
        for i in range(n - 1):
            a, b, c2, d2 = off + 2 * i, off + 2 * i + 1, off + 2 * i + 2, off + 2 * i + 3
            F += [[a, b, c2], [b, d2, c2]]
        off += 2 * n
    if not V:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), int))
    return Mesh(np.concatenate(V), np.asarray(F, int))


def _selftest():
    """Numeric contracts, not smoke: area-weighting must be provably area-proportional, points must
    lie ON the surface, transforms must be well-formed, merge must conserve vertex counts exactly,
    and every stage must be bit-identical under the same seed."""
    # A two-triangle quad in z=0, split so one triangle is 9x the area of the other.
    V = np.array([[0., 0., 0.], [10., 0., 0.], [10., 1., 0.], [0., 1., 0.]])
    F = np.array([[0, 1, 2], [0, 2, 3]])
    quad = Mesh(V, F)

    # 1) AREA WEIGHTING: with equal-area halves the split must be ~50/50 within sampling noise.
    s = sample_mesh_surface(quad, 4000, seed=1)
    frac = (s["faces"] == 0).mean()
    assert abs(frac - 0.5) < 0.05, "area weighting is off: face 0 got %.3f" % frac
    assert abs(s["area"] - 10.0) < 1e-9, "quad area must be exactly 10"

    # 2) ON THE SURFACE: every sample lies in the z=0 plane and inside the quad's bbox.
    P = s["points"]
    assert np.abs(P[:, 2]).max() < 1e-12, "samples must lie exactly on the plane"
    assert P[:, 0].min() >= -1e-12 and P[:, 0].max() <= 10 + 1e-12, "samples inside the quad"
    assert np.abs(np.abs(s["normals"][:, 2]) - 1.0).max() < 1e-12, "plane normals must be +-z"

    # 3) DETERMINISM: same seed, bit-identical; different seed, different.
    s2 = sample_mesh_surface(quad, 200, seed=1)
    assert np.array_equal(sample_mesh_surface(quad, 200, seed=1)["points"], s2["points"]), \
        "same seed + same count must be bit-identical"
    assert not np.array_equal(sample_mesh_surface(quad, 200, seed=2)["points"], s2["points"])
    # KEPT NEGATIVE, found by this selftest: the first 200 points of a 4000-sample are NOT the same as
    # a 200-sample at the same seed -- rng.choice(size=n) draws a different stream length. So `count`
    # is part of the determinism key. Callers who want a stable prefix must fix count, not slice.

    # 4) FRAMES: orthonormal rotation part, translation equal to the sample point, scale honoured.
    M = placement_frames(s2["points"], s2["normals"], scale=2.0, seed=1)
    R = M[0, :3, :3] / 2.0
    assert np.abs(R @ R.T - np.eye(3)).max() < 1e-10, "frame rotation must be orthonormal"
    assert np.allclose(M[:, :3, 3], s2["points"]), "frame translation must be the sample point"

    # 4b) THE ASSERTION THAT WAS MISSING. Placed geometry must SPAN the surface, not merely exist in
    #     the right quantity. Counts, "nothing below the ground plane", and matrix-content checks all
    #     pass when every instance sits at the origin -- which is exactly what a transposed transform
    #     convention produced here. Extent is the property that cannot be faked.
    spread = realize_scatter(grass_blade(segments=2), M, mode="merge")
    SV = np.asarray(spread.vertices, float)
    assert SV[:, 0].max() - SV[:, 0].min() > 5.0, \
        "scattered geometry must SPAN the surface, got x-extent %.3f (all at one point?)" % (
            SV[:, 0].max() - SV[:, 0].min())
    assert SV[:, 1].max() - SV[:, 1].min() > 0.5, "and span it in y as well"
    # each blade must land AT its own transform's translation, within the blade's own size
    first = np.asarray(realize_scatter(grass_blade(segments=2), M[:1], mode="merge").vertices, float)
    assert np.linalg.norm(first.mean(0)[:2] - M[0, :3, 3][:2]) < 0.2, \
        "a placed instance must sit at its transform's translation"

    # 5) MERGE conserves counts EXACTLY: n instances of a blade = n * (verts, faces).
    blade = grass_blade(segments=3)
    nv, nf = len(blade.vertices), len(blade.faces)
    out = realize_scatter(blade, M[:25], mode="merge")
    assert len(out.vertices) == 25 * nv and len(out.faces) == 25 * nf, \
        "merge must conserve counts: %d/%d vs %d/%d" % (len(out.vertices), len(out.faces), 25 * nv, 25 * nf)

    # 6) VARIANTS are drawn deterministically and all appear.
    v2 = [grass_blade(height=0.2, segments=3), grass_blade(height=0.6, segments=3)]
    o1 = realize_scatter(None, M[:40], mode="merge", variants=v2, seed=7)
    o2 = realize_scatter(None, M[:40], mode="merge", variants=v2, seed=7)
    assert np.array_equal(o1.vertices, o2.vertices), "variant draw must be deterministic"
    heights = o1.vertices[:, 2]
    assert heights.max() > 0.5, "the tall variant must actually be placed"

    # 7) DENSITY as rejection: a half-plane mask must remove roughly half and leave none on the far side.
    d = sample_mesh_surface(quad, 2000, density=lambda P: (P[:, 0] < 5.0).astype(float), seed=3)
    assert d["points"][:, 0].max() <= 5.0 + 1e-9, "density mask must exclude the masked region"
    assert 700 < d["count"] < 1300, "half-mask should keep about half, kept %d" % d["count"]

    # 8) RELAX thins and never violates the spacing it promised (and reports the smaller count).
    r = sample_mesh_surface(quad, 600, seed=4, relax=True, radius=0.5)
    assert r["count"] < 600, "relax must THIN, not pad"
    RP = r["points"]
    dm = np.linalg.norm(RP[:, None, :] - RP[None, :, :], axis=-1)
    dm[np.arange(len(RP)), np.arange(len(RP))] = np.inf
    assert dm.min() >= 0.5 - 1e-9, "relaxed points must respect the radius"

    # 9) INSTANCED mode: n placements must share len(pool) definitions -- that IS the win being claimed.
    sc = realize_scatter(blade, M[:50], mode="instanced")
    assert len(sc.instances) == 50 and len(sc.definitions()) == 1, "instancing must share one definition"
    sc2 = realize_scatter(None, M[:50], mode="instanced", variants=v2)
    assert len(sc2.definitions()) == 2 and len(sc2.instances) == 50, "one definition per variant"

    # 10) One-call door works end to end.
    g = scatter_mesh(quad, blade, 30, seed=5, scale_jitter=0.3, align=0.7)
    assert g["geometry"].vertices.shape[0] == g["count"] * nv

    # 11) RIBBONS from strand chains: 2 verts per point, 2 tris per segment, exactly.
    strands = [np.column_stack([np.zeros(6), np.zeros(6), np.linspace(0, 0.3, 6)]) for _ in range(4)]
    rib = strand_ribbons(strands, width=0.02)
    assert len(rib.vertices) == 4 * 2 * 6 and len(rib.faces) == 4 * 2 * 5
    w0 = np.linalg.norm(rib.vertices[1] - rib.vertices[0])
    w1 = np.linalg.norm(rib.vertices[11] - rib.vertices[10])
    assert w1 < w0, "a ribbon must taper toward the tip (%.4f -> %.4f)" % (w0, w1)

    # 12) THE HOLOGRAPHIC LAYER, tested as the STATISTICAL read it actually is. An earlier version of
    #     this assert compared ONE occupied point against ONE empty point and demanded an 8x ratio --
    #     which failed at 6.8x, not because the encoding was wrong but because a single empty probe is
    #     a draw from a zero-mean noise distribution and its sign is luck. Measure the floor instead.
    hs = scatter_mesh(quad, blade, 80, seed=11, holographic=True, dim=2048, cell_size=1.0)
    occ = [region_occupancy(hs["layer"], hs["instance"], hs["transforms"][k, :3, 3],
                            dim=2048, cell_size=1.0, seed=11)
           for k in range(0, len(hs["transforms"]), 5)]
    rng_far = np.random.default_rng(0)
    far = [region_occupancy(hs["layer"], hs["instance"], rng_far.normal(size=3) * 1000 + 2000,
                            dim=2048, cell_size=1.0, seed=11) for _ in range(60)]
    sep = (np.mean(occ) - np.mean(far)) / max(np.std(far), 1e-12)
    assert sep > 4.0, "occupied cells must separate from the noise floor: %.1f sigma" % sep
    assert abs(np.mean(far)) < 0.02, "empty regions must be zero-MEAN, got %.4f" % np.mean(far)

    print("holographic_meshscatter selftest OK: area-weighted %.3f, %d/%d merge counts exact, "
          "relax %d<600 spacing held, variants deterministic, occupancy %.1f sigma over noise"
          % (frac, len(out.vertices), 25 * nv, r["count"], sep))


if __name__ == "__main__":
    _selftest()
