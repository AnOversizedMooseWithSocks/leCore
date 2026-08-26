"""Primitive-set fitting: approximate an arbitrary shape with a small UNION of SDF primitives (holographic_primfit).

WHY THIS MODULE EXISTS
----------------------
fit_shape handles two families: self-similar 3-D structure (fold_fractal) and plant/branching point-sets (affine IFS).
Neither represents a HARD-SURFACE or NON-FRACTAL ORGANIC shape -- a 'creature', a mechanical part, a blobby form. The
honest model for those is a small UNION of SDF PRIMITIVES (a sphere set / sphere tree): cover the shape with a handful
of spheres, unioned into one SDF. That SDF is exact (spheres emit cleanly), so it raymarches, meshes, AND emits a
Shadertoy shader. This is the third fitter fit_shape flagged.

THE METHOD (NumPy-only, deterministic -- no sklearn)
  Cluster the target surface points into K groups with a deterministic clustering (farthest-point SEEDING so the seeds
  are reproducible, then Lloyd/k-means iterations), fit one sphere per cluster (centre = cluster centroid, radius =
  mean point distance), and UNION them (min of the sphere SDFs). More spheres = closer fit; `auto_k` grows K until the
  fit stops improving materially (the elbow), so a simple shape gets few spheres and a complex one gets more.

WHAT IT PROVIDES
  * fit_primitives(target, k, ...) -- fit K spheres to a (M,3) point cloud, return {sdf, spheres, quality, baseline, k}.
  * The returned `sdf` is a real SDF: .eval / raymarch / sdf_to_mesh / sdf_shader all work on it.

KEPT NEGATIVES (loud)
  * SPHERES ONLY (for now): a box/capsule-augmented fit is a scoped extension. A sphere set approximates blobby and
    rounded shapes well and blocky/flat shapes coarsely (a cube becomes a cluster of spheres) -- measured, not hidden.
  * It approximates the SURFACE the points sample; it does not recover a minimal or canonical CSG tree (that is a
    harder search). `quality` is the mean surface residual vs a single-bounding-sphere baseline -- 'how much better
    than one sphere', an honest floor, not a claim of optimality.
  * The clustering is deterministic (fixed farthest-point seed) but the K-means objective is non-convex -- a different
    seed could find a different local optimum; we pin ONE deterministic result, not the global best.
"""

import numpy as np


def _cluster(points, k, iters=12, seed=0):
    """Deterministic k-means: farthest-point SEEDING (reproducible) then Lloyd iterations. Returns (labels, centres).
    Farthest-point seeding spreads the initial centres out (better than random for shape coverage) AND is
    deterministic given the first point, so the whole fit is reproducible."""
    points = np.asarray(points, float)
    n = len(points)
    k = min(k, n)
    # farthest-point seeding: start at the point nearest the centroid, then repeatedly add the farthest-from-any-seed.
    c0 = int(np.argmin(np.linalg.norm(points - points.mean(0), axis=1)))
    seed_idx = [c0]
    d2 = np.linalg.norm(points - points[c0], axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(d2))
        seed_idx.append(nxt)
        d2 = np.minimum(d2, np.linalg.norm(points - points[nxt], axis=1))
    centres = points[seed_idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
        new_labels = d.argmin(1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break                                             # converged
        labels = new_labels
        for j in range(k):
            m = labels == j
            if np.any(m):
                centres[j] = points[m].mean(0)
    return labels, centres


def _sphere_union(spheres):
    """Build a union SDF from a list of (centre, radius) spheres: union(translate(sphere(r), c) ...). Returns an SDF
    with .eval / .to_glsl / raymarch, all exact (a sphere is an exact distance)."""
    from holographic.mesh_and_geometry.holographic_sdf import sphere, SDF
    nodes = [SDF("translate", tuple(float(x) for x in c), (sphere(float(r)),)) for c, r in spheres]
    node = nodes[0]
    for nxt in nodes[1:]:
        node = SDF("union", (), (node, nxt))
    return node


def _residual(spheres, target):
    """Mean |distance| from the target points to the sphere-union surface (0 = the surface passes through them)."""
    return float(np.mean(np.abs(_sphere_union(spheres).eval(target))))


# ---------------------------------------------------------------------------------------------------------
# Mixed-primitive fitting: per cluster, fit a sphere, an ORIENTED box, and a capsule, and keep the best. Boxes
# and capsules capture the blocky / elongated parts a sphere can only approximate coarsely. All exact -> all emit.
# ---------------------------------------------------------------------------------------------------------

def _mat_to_axis_angle(R):
    """Convert a rotation matrix to (axis, angle) for the SDF `rotate` op. Guards the two singular cases (angle ~0 and
    angle ~pi) so a degenerate cluster never produces a NaN rotation."""
    ang = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if ang < 1e-6:
        return np.array([0.0, 0.0, 1.0]), 0.0
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(ax)
    if n < 1e-6:                                              # angle ~pi: axis from the largest diagonal term
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        e = np.zeros(3); e[i] = 1.0
        return e, ang
    return ax / n, ang


def _principal_axes(pts):
    """The (centre, principal-axes-matrix, projected-coords) of a point cluster via the covariance eigenbasis. The
    axes are columns, sorted by descending variance, made right-handed (so `rotate` gets a proper rotation)."""
    c = pts.mean(0)
    cov = np.cov((pts - c).T) if len(pts) > 3 else np.eye(3)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    if np.linalg.det(vecs) < 0:                              # ensure a right-handed (proper) rotation
        vecs[:, 2] *= -1
    return c, vecs, vals[order]


def _fit_sphere(pts):
    from holographic.mesh_and_geometry.holographic_sdf import sphere, SDF
    c = pts.mean(0)
    r = max(float(np.linalg.norm(pts - c, axis=1).mean()), 1e-4)
    return ("sphere", (tuple(float(x) for x in c), float(r))), SDF("translate", tuple(float(x) for x in c), (sphere(r),))


def _fit_box(pts):
    """An ORIENTED bounding box: principal-axis frame, half-extents from the max projection on each axis. The SDF
    `rotate` evaluates ch(P @ Rm), i.e. it rotates the query INTO the child frame, so we pass the axis-angle of the
    principal-axis matrix directly (validated: +angle is the convention that aligns the box to the cluster)."""
    from holographic.mesh_and_geometry.holographic_sdf import box, SDF
    c, vecs, _ = _principal_axes(pts)
    proj = (pts - c) @ vecs
    half = np.maximum(np.abs(proj).max(0), 1e-4)
    ax, ang = _mat_to_axis_angle(vecs)
    node = SDF("translate", tuple(float(x) for x in c),
               (SDF("rotate", (float(ax[0]), float(ax[1]), float(ax[2]), float(ang)),
                    (box(float(half[0]), float(half[1]), float(half[2])),)),))
    return ("box", (tuple(float(x) for x in c), tuple(float(x) for x in half),
                    (tuple(float(x) for x in ax), float(ang)))), node


def _fit_capsule(pts):
    """A CAPSULE along the cluster's dominant axis: a segment (half-height from the projection extent minus the radius)
    inflated by the mean perpendicular distance. The natural primitive for a limb / rounded rod. Rotates the Y-axis
    capsule to the dominant axis."""
    from holographic.mesh_and_geometry.holographic_sdf import capsule, SDF
    c, vecs, _ = _principal_axes(pts)
    dom = vecs[:, 0]
    proj = (pts - c) @ dom
    perp = (pts - c) - np.outer(proj, dom)
    r = max(float(np.linalg.norm(perp, axis=1).mean()), 1e-4)
    h = max(float(np.abs(proj).max() - r), 1e-3)
    # rotation Y -> dom
    y = np.array([0.0, 1.0, 0.0])
    axis = np.cross(y, dom); s = np.linalg.norm(axis); cth = float(np.dot(y, dom))
    if s < 1e-8:
        ax, ang = np.array([1.0, 0.0, 0.0]), (0.0 if cth > 0 else np.pi)
    else:
        ax, ang = axis / s, float(np.arctan2(s, cth))
    node = SDF("translate", tuple(float(x) for x in c),
               (SDF("rotate", (float(ax[0]), float(ax[1]), float(ax[2]), float(ang)), (capsule(h, r),)),))
    return ("capsule", (tuple(float(x) for x in c), float(h), float(r),
                        (tuple(float(x) for x in ax), float(ang)))), node


_FITTERS = {"sphere": _fit_sphere, "box": _fit_box, "capsule": _fit_capsule}


def _fit_best_primitive(pts, kinds):
    """Fit each allowed primitive `kind` to the cluster and keep the one with the lowest residual on the cluster's own
    points. Returns (record, node, residual). A box/capsule wins on a blocky/elongated cluster; a sphere on a round
    one -- the primitive is CHOSEN by fit, not assumed."""
    best = None
    for kind in kinds:
        rec, node = _FITTERS[kind](pts)
        res = float(np.mean(np.abs(node.eval(pts))))
        if best is None or res < best[2]:
            best = (rec, node, res)
    return best


def _mixed_union(nodes):
    """Union a list of primitive SDF nodes (min of children)."""
    from holographic.mesh_and_geometry.holographic_sdf import SDF
    node = nodes[0]
    for nxt in nodes[1:]:
        node = SDF("union", (), (node, nxt))
    return node


def fit_primitives(target, k=6, auto_k=False, k_max=16, tol=0.05, iters=12, seed=0,
                   primitives=("sphere", "box", "capsule")):
    """Approximate a (M,3) point cloud `target` with a UNION of `k` PRIMITIVES, choosing the best-fitting primitive per
    cluster. Clusters the points deterministically, then for each cluster fits every allowed `primitives` kind and
    keeps the one with the lowest residual: a SPHERE for round parts, an ORIENTED BOX for blocky parts, a CAPSULE for
    elongated/rounded limbs. All are EXACT SDFs, so the union raymarches / sdf_to_mesh's / to_shadertoy's.

    Returns {sdf, parts, quality, baseline, residual, k, kinds}: `parts` is the [(kind, params)] list; `kinds` counts
    how many of each type were chosen; `quality` is single-bounding-sphere_residual / fit_residual (>1 = better than
    one sphere); `spheres` is kept for back-compat (the sphere parts only). `primitives` restricts the palette (e.g.
    ('sphere',) reproduces the old sphere-only behaviour exactly).

    auto_k=True grows K to the elbow (a simple shape gets few parts, a complex one more; the residual is non-monotonic
    in K, so a patience window is used). KEPT NEGATIVE: an oriented box/capsule is fit from the cluster's covariance,
    so a cluster spanning TWO oriented parts is fit by ONE loose box -- raise K; and this approximates the surface, it
    does not recover a minimal or canonical CSG tree."""
    target = np.asarray(target, float)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("fit_primitives expects a (M,3) point cloud")
    kinds_allowed = tuple(primitives)

    # single-bounding-sphere baseline (the honest floor: 'is the multi-primitive fit actually better than one sphere?')
    bc = target.mean(0)
    br = float(np.linalg.norm(target - bc, axis=1).mean())
    baseline = _residual([(bc, br)], target)

    def fit_k(kk):
        labels, centres = _cluster(target, kk, iters=iters, seed=seed)
        records, nodes = [], []
        for j in range(len(centres)):
            pts = target[labels == j]
            if len(pts) == 0:
                continue
            rec, node, _ = _fit_best_primitive(pts, kinds_allowed)
            records.append(rec)
            nodes.append(node)
        residual = float(np.mean(np.abs(_mixed_union(nodes).eval(target))))
        return records, nodes, residual

    if auto_k:
        best = fit_k(1)
        best_res = best[2]
        since_improve = 0
        for kk in range(2, k_max + 1):
            cand = fit_k(kk)
            if cand[2] < best_res * (1.0 - tol):             # a real improvement over the best so far
                best, best_res = cand, cand[2]
                since_improve = 0
            else:
                since_improve += 1
                if since_improve >= 3:                       # patience: 3 Ks without a real gain -> stop (elbow)
                    break
        records, nodes, residual = best
    else:
        records, nodes, residual = fit_k(k)

    kind_counts = {}
    for kind, _ in records:
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    spheres_only = [params for kind, params in records if kind == "sphere"]

    return {
        "sdf": _mixed_union(nodes),
        "parts": records,
        "spheres": spheres_only,                             # back-compat: the sphere parts (centre, radius)
        "kinds": kind_counts,
        "quality": baseline / max(residual, 1e-9),           # >1 = better than a single bounding sphere
        "baseline": baseline,
        "residual": residual,
        "k": len(records),
    }


def _selftest():
    rng = np.random.default_rng(0)

    def sphere_surface(c, r, n):
        d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        return np.asarray(c) + r * d

    # (1) a two-blob 'peanut' fits MUCH better with 2 primitives than 1, and each cluster picks a SPHERE (round).
    peanut = np.vstack([sphere_surface([-0.5, 0, 0], 0.6, 300), sphere_surface([0.5, 0, 0], 0.5, 300)])
    r2 = fit_primitives(peanut, k=2)
    assert r2["k"] == 2 and r2["quality"] > 2.0, "two primitives fit the peanut much better than one (%.1fx)" % r2["quality"]
    assert r2["kinds"].get("sphere", 0) == 2, "round blobs are fit by SPHERES, got %s" % r2["kinds"]
    assert "mainImage" in _sphere_union_shader(r2["sdf"]), "the union SDF emits a Shadertoy shader"

    # (2) determinism.
    assert fit_primitives(peanut, k=2)["parts"] == r2["parts"], "fit_primitives is deterministic"

    # (3) an ELONGATED oriented block is fit by an ELONGATED primitive (box OR capsule -- chosen by fit, not label),
    #     and much closer than a sphere. A thin box and a capsule are near-identical, so either is a correct choice.
    rng2 = np.random.default_rng(1)
    th = np.radians(30.0); Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    face = []
    for _ in range(500):
        f = rng2.integers(0, 3); s = rng2.choice([-1, 1]); p = rng2.uniform(-1, 1, 3) * np.array([0.5, 0.1, 0.1])
        p[f] = s * [0.5, 0.1, 0.1][f]; face.append(p)
    box_pts = (np.array(face) @ Rz.T) + np.array([0.7, 0.3, 0.0])
    rb = fit_primitives(box_pts, k=1)
    assert rb["kinds"].get("box", 0) + rb["kinds"].get("capsule", 0) == 1, ("an elongated block is fit by a box or "
                                                                            "capsule, not a sphere, got %s" % rb["kinds"])
    rb_sphere = fit_primitives(box_pts, k=1, primitives=("sphere",))
    assert rb["residual"] < rb_sphere["residual"] * 0.6, ("an elongated primitive fits a block far better than a sphere "
                                                          "(%.3f vs sphere %.3f)" % (rb["residual"], rb_sphere["residual"]))
    # a FLAT SLAB (two large extents, one thin) is where a BOX clearly beats a capsule -- box wins here.
    slab = []
    for _ in range(500):
        f = rng2.integers(0, 3); s = rng2.choice([-1, 1]); p = rng2.uniform(-1, 1, 3) * np.array([0.5, 0.5, 0.06])
        p[f] = s * [0.5, 0.5, 0.06][f]; slab.append(p)
    slab = np.array(slab) + np.array([0.0, 0.0, 1.0])
    rb_box = fit_primitives(slab, k=1)
    assert rb_box["kinds"].get("box", 0) == 1, "a flat slab is fit by a BOX (a capsule can't be flat), got %s" % rb_box["kinds"]

    # (4) an elongated ROUNDED LIMB is fit by an elongated primitive, and closer than a sphere.
    d = np.array([1.0, 1.0, 0.3]); d /= np.linalg.norm(d)
    hh, rr, ctr = 0.6, 0.18, np.array([0.5, 0.2, 0.0])
    limb = []
    for _ in range(500):
        t = rng2.uniform(-hh, hh); off = rng2.normal(size=3); off -= off.dot(d) * d; off /= np.linalg.norm(off)
        limb.append(ctr + t * d + rr * off)
    limb = np.array(limb)
    rc = fit_primitives(limb, k=1)
    assert rc["kinds"].get("capsule", 0) + rc["kinds"].get("box", 0) == 1, "a limb is fit by a capsule/box, got %s" % rc["kinds"]
    rc_sphere = fit_primitives(limb, k=1, primitives=("sphere",))
    assert rc["residual"] < rc_sphere["residual"] * 0.6, "an elongated primitive fits a limb better than a sphere"

    # (5) auto_k adapts; and a mixed-primitive fit of a blocky+round creature beats sphere-only.
    one_blob = sphere_surface([0, 0, 0], 0.7, 400)
    assert fit_primitives(one_blob, auto_k=True, k_max=8)["k"] <= fit_primitives(peanut, auto_k=True, k_max=8)["k"]
    creature = np.vstack([sphere_surface([0, 0, 0], 0.6, 300), box_pts, limb])
    mixed = fit_primitives(creature, k=3)
    sphere_only = fit_primitives(creature, k=3, primitives=("sphere",))
    assert mixed["residual"] <= sphere_only["residual"], ("mixed primitives fit a blocky+round shape at least as well "
                                                          "as spheres alone (mixed %.3f vs sphere %.3f)"
                                                          % (mixed["residual"], sphere_only["residual"]))
    # back-compat: primitives=('sphere',) reproduces the sphere-only behaviour (all parts are spheres).
    assert all(kind == "sphere" for kind, _ in sphere_only["parts"])

    print("holographic_primfit selftest: ok (round blobs -> spheres, a flat slab -> a BOX, an elongated block/limb -> "
          "a box/capsule (%.2fx tighter than a sphere); the union SDF emits a Shadertoy shader; mixed primitives beat "
          "sphere-only on a blocky+round creature (%.3f <= %.3f); auto_k adapts; deterministic; primitives=('sphere',) "
          "reproduces the old behaviour)"
          % (rb_sphere["residual"] / max(rb["residual"], 1e-9), mixed["residual"], sphere_only["residual"]))


def _sphere_union_shader(sdf):
    """Emit a Shadertoy shader for a fitted sphere-union SDF (used by the selftest to prove it round-trips to code)."""
    from holographic.mesh_and_geometry.holographic_sdf import _emit_shader
    return _emit_shader(sdf)


if __name__ == "__main__":
    _selftest()
