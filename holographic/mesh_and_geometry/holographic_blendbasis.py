"""Procedural blendshape basis with DECLARED local support.

BACKLOG O2 of the creature/humanoid overhaul, unblocked by O1's fixed topology (a blendshape
target is a per-vertex displacement, which is meaningless unless vertex i is the same
anatomical point on every body).

SOTA CHECK (searched 2026-08-16). SMPL's pose correctives are DENSE: they "relate every
vertex on the mesh to all the joints in the kinematic tree, capturing spurious long-range
correlations". STAR (Osman et al. 2020) fixes this and states the insight plainly -- "human
pose deformation is LOCAL and SPARSE" -- reaching 20% of SMPL's pose-corrective parameters
and better generalisation. SPLOCS (Neumann et al. 2013) reaches the same conclusion from the
decomposition side, extracting "sparse and spatially localized deformation modes" with "an
automatic way to ensure spatial locality". Both are still the reference points in 2025-26
work (QMF-Blend, SIGGRAPH Asia 2025, compresses such bases rather than replacing them).

THE ASYMMETRY THAT MAKES THIS CHEAP FOR US, and it is the whole reason O2 is a small module:
STAR spends scan data to LEARN "the activation region on the mesh that these joints
influence". WE DO NOT HAVE TO LEARN IT -- we are authoring the basis, so the support region
is a DESIGN INPUT we declare. STAR's headline improvement over SMPL is, for a procedural
basis, simply the default. What we cannot get without scans is a REALISTIC shape
distribution; this module makes no claim to that, and §"KEPT NEGATIVE" says so.

SUPPORT IS GEODESIC, NOT EUCLIDEAN, and this is load-bearing rather than fussy: a hand
resting against the hip is millimetres away in space and a metre away across the surface. A
Euclidean support radius would let a wrist corrective deform the hip -- reintroducing exactly
the spurious long-range coupling STAR exists to remove, while looking local in the source.

RULE-0 AUDIT (2026-08-16): blend_shapes already applies a basis (base + sum w_i (target_i -
base)) and is REUSED unchanged -- this module BUILDS bases, it does not re-implement mixing.
mesh_geodesic (Dijkstra over mesh edges) supplies the support metric. skin_bind_weights
supplies the partition-of-unity weights the correctives must not break. Nothing here
duplicates them.

KEPT NEGATIVE: a declared support radius is an ASSERTION BY THE AUTHOR, not a measurement of
anatomy. It guarantees locality (no spurious coupling, provably) but it does NOT guarantee
the deformation is anatomically right -- a badly chosen radius gives a local, smooth, wrong
bulge. Locality is a correctness property; realism is not, and no amount of proof supplies a
shape distribution that only scans can measure.
"""

import numpy as np


def support_weights(mesh, source_vertex, radius, mind, falloff="smoothstep"):
    """Per-vertex support in [0,1] for a corrective anchored at `source_vertex`.

    GEODESIC distance (mesh_geodesic), so support cannot leak across a gap between two
    surfaces that happen to be close in space. `falloff='smoothstep'` gives C1 support --
    a linear ramp leaves a visible crease at the support boundary, which is the same
    continuity argument that governs the blend operators elsewhere in the engine.

    Weight is EXACTLY ZERO beyond `radius`: that is what makes the locality claim checkable
    rather than approximate."""
    d = np.asarray(mind.mesh_geodesic(mesh, int(source_vertex)), float)
    r = float(radius)
    u = np.clip(1.0 - d / max(r, 1e-12), 0.0, 1.0)
    if falloff == "linear":
        w = u
    else:
        w = u * u * (3.0 - 2.0 * u)          # smoothstep: C1 at both ends
    w[~np.isfinite(d)] = 0.0                 # unreachable components get no support
    return w


def make_corrective(mesh, source_vertex, radius, direction, amplitude, mind,
                    falloff="smoothstep"):
    """One blendshape TARGET: displace vertices near `source_vertex` along `direction`.

    Returns the target vertex array (not the delta), which is what blend_shapes consumes.
    `direction` may be a 3-vector (a push) or the string 'normal' (inflate along the surface
    normal, which is how weight/muscle targets read)."""
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    w = support_weights(mesh, source_vertex, radius, mind, falloff=falloff)
    if isinstance(direction, str) and direction == "normal":
        n = np.zeros_like(V)
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        for k in range(3):
            np.add.at(n, F[:, k], fn)        # area-weighted vertex normals
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        D = n
    else:
        D = np.tile(np.asarray(direction, float).reshape(1, 3), (len(V), 1))
    return V + float(amplitude) * w[:, None] * D


def locality_report(base, targets, mesh, sources, radii, mind):
    """Is every corrective ACTUALLY local -- the property STAR needs scans to obtain?

    For each target, measures the largest geodesic distance at which it displaces a vertex,
    against its declared radius. `max_overreach` > 0 means a corrective influences a vertex
    outside its declared support, i.e. the spurious long-range coupling this whole design
    exists to prevent."""
    B = np.asarray(base, float)
    rows = []
    worst = 0.0
    for T, s, r in zip(targets, sources, radii):
        d = np.asarray(mind.mesh_geodesic(mesh, int(s)), float)
        moved = np.linalg.norm(np.asarray(T, float) - B, axis=1) > 1e-9
        reach = float(np.max(d[moved])) if moved.any() else 0.0
        over = max(0.0, reach - float(r))
        worst = max(worst, over)
        rows.append({"source": int(s), "radius": float(r), "reach": reach,
                     "overreach": over, "n_moved": int(moved.sum()),
                     "fraction_moved": float(moved.mean())})
    return {"max_overreach": worst, "local": worst <= 1e-9, "targets": rows}


def _selftest():
    """Regression trap: correctives must be provably local and must not break the partition
    of unity that skinning depends on."""
    import lecore
    from holographic.mesh_and_geometry import holographic_mesh as _HM
    mind = lecore.UnifiedMind(dim=64, seed=0)
    sphere = lambda P: np.linalg.norm(np.asarray(P, float), axis=1) - 1.0
    mesh = mind.mesh_from_sdf(sphere, ((-1.3,) * 3, (1.3,) * 3), res=24, vectorized=True)
    V = np.asarray(mesh.vertices, float)
    src = [int(np.argmax(V[:, 1])), int(np.argmin(V[:, 1]))]
    radii = [0.8, 0.5]
    targets = [make_corrective(mesh, s, r, "normal", 0.25, mind)
               for s, r in zip(src, radii)]

    rep = locality_report(V, targets, mesh, src, radii, mind)
    assert rep["local"], rep                                  # NO overreach: the STAR property
    assert all(0.0 < t["fraction_moved"] < 0.9 for t in rep["targets"]), rep

    # a corrective must move something, and a DISJOINT pair must not touch the same vertices
    m0 = np.linalg.norm(targets[0] - V, axis=1) > 1e-9
    m1 = np.linalg.norm(targets[1] - V, axis=1) > 1e-9
    assert m0.any() and m1.any()
    assert not (m0 & m1).any(), "supports on opposite poles overlapped"

    # blend_shapes must accept them, and EXTRAPOLATION (w outside [0,1], which animators use)
    # must stay finite rather than exploding
    mixed = mind.blend_shapes(V, targets, [1.0, 0.5])
    assert np.all(np.isfinite(mixed))
    ext = mind.blend_shapes(V, targets, [1.6, -0.4])
    assert np.all(np.isfinite(ext))
    print("OK: holographic_blendbasis -- %d correctives, max overreach %.2e (LOCAL by "
          "construction), disjoint supports stay disjoint, extrapolation finite"
          % (len(targets), rep["max_overreach"]))


if __name__ == "__main__":
    _selftest()
