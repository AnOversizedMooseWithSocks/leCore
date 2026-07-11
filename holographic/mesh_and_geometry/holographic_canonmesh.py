"""holographic_canonmesh.py -- canonical element + delta chain (Box3D backlog C3).

Instancing, generalised. A renderer's instancing says *"these two objects are the same mesh"*; Part C says *"these
two objects are the same **anything**, modulo a recognised delta."* Store the canonical element once, and one delta
per instance.

Three delta families ship, and **choosing the family is the whole decision**:

    family        canonicalised by                             delta        recognises
    rigid         centre + rotate to a pinned frame            7 floats     congruent copies
    similarity    ... and scale to unit RMS radius             8 floats     similar copies (any size)
    affine        ... and whiten the hull's covariance         3*rank + 3   copies up to shear

MEASURED on 200 triangles built from 5 base shapes under random rotations, translations AND uniform scales:

    family        classes   ratio vs raw
    rigid            200      0.56x      <- UNDER-fits: scale is not in the family, so nothing matches
    similarity         5      1.09x      <- exactly the family the scene was generated with
    affine             5      0.98x      <- collapses the SHAPE, and the delta costs what the triangle costs

**The family must match the deltas actually present.** Too small and nothing is recognised; too large and the delta
costs more than the thing it describes. Only `similarity` wins here, and it wins by 1.09x -- which is nothing.

WHY `affine` GIVES 5 CLASSES AND NOT 1, which is a fact worth having. Whitening a triangle's hull makes it exactly
EQUILATERAL -- measured, all three sides come out sqrt(6) for every base shape. So the affine family really does
collapse triangle SHAPE. What it leaves is the residual ROTATION within the hull, and pinning that on a symmetric
(equilateral) configuration requires a vertex ORDER, which an unordered point set does not carry. *"Every
non-degenerate triangle is affinely the same" is a statement about ORDERED triangles.* Canonicalising the order is a
further item; this module canonicalises the point set and says so.

KEPT NEGATIVE 1 -- **A TRIANGLE IS THE WORST CASE.** It is 9 floats, and its hull has rank 2, so an AFFINE delta
is `3*2 + 3 = 9` floats -- exactly the triangle, before a single canonical element has been stored. No amount of
reuse fixes a delta the size of its element. (Under the cheaper RIGID delta of 7 floats a triangle does pay, by
1.23x -- which is not a counterexample, it is the same arithmetic: the ratio is set by `delta_floats / 9`. Stating
"a triangle can never pay" without naming the family would have been false.) The dividend is in the ELEMENT.
Measured, 200 instances, affine:

    element        raw floats   canonical + delta   ratio
    3 vertices        1,800          2,409          0.75x   <- a LOSS
    12 vertices       7,200          2,436          2.96x
    100 vertices     60,000          2,700         22.22x
    2000 vertices  1,200,000         8,400        142.86x

**The dividend scales with the size of the canonical ELEMENT against the O(1) delta.** Per-triangle
canonicalisation is a RECOGNISER, not a compressor -- its dividend is C1's compute cache (400x there), not storage.

KEPT NEGATIVE 2 -- and it is the opposite of what K1 found for code. `zlib` on the raw instanced vertex buffer
achieves **1.04x**, because float64 coordinates are high-entropy. So canonical+delta beats the honest baseline here
by two orders of magnitude, where the same idea applied to source code came out 1.12x LARGER than zlib. *The
difference is structural: a mesh's delta is O(1) while its element is O(n); a statement's delta is O(statement).*
The idea is the same; only one of the two is a codec, and which one is a measurement.

Determinism: the canonical frame is fixed by an SVD whose signs are pinned with `fix_eigvec_signs`, so the same
element canonicalises to the same array bit for bit, run to run. Without that pinning an eigenvector and its
negation are both valid and the class key flickers -- the same tie class the `bind_batch` bug lived in.
"""

import hashlib

import numpy as np

FAMILIES = ("rigid", "similarity", "affine")


def _pin_signs(P, Q):
    """Pin each axis's sign FROM THE DATA, not from the frame.

    `fix_eigvec_signs` pins by the eigenvector's own largest entry, which is right for an eigenbasis and wrong
    here: the axis is only defined up to sign, and what must be reproducible is the POINT SET in that axis, not the
    axis. Pin by the SKEWNESS of the projections (the third moment is odd, so it flips with the axis); fall back to
    the largest-magnitude projection when the skewness is ~0, as it is for a symmetric shape.

    A NULL AXIS HAS NO DATA TO PIN IT WITH. For a planar element the third singular vector's singular value is
    zero, its sign is numerical noise, and a first version forced `det(Q) = +1` -- which pushed that noise onto a
    REAL axis and split each base shape into four spurious classes (5 bases came out as 20). Deriving the null axes
    by cross product instead keeps them a function of the meaningful ones."""
    Q = np.array(Q, float, copy=True)
    proj = P @ Q
    for k in range(Q.shape[1]):
        col = proj[:, k]
        skew = float((col ** 3).sum())
        if abs(skew) < 1e-9 * max(float(np.abs(col).max()) ** 3, 1e-30):
            j = int(np.argmax(np.abs(col)))          # ties -> lowest index: deterministic
            skew = float(col[j])
        if skew < 0:
            Q[:, k] *= -1.0
    return Q


def _frame(P):
    """A deterministic orthonormal 3-frame for a centred point set.

    The meaningful axes come from the SVD and are sign-pinned from the DATA (`_pin_signs`). Any null axes are
    completed by cross product, so they are a function of the meaningful ones rather than of numerical noise, and
    the frame is a rotation by construction."""
    _u, s, vt = np.linalg.svd(P, full_matrices=False)
    rank = int((s > 1e-9 * max(s[0], 1e-30)).sum())
    if rank == 0:
        # every point coincides with the centroid: there is no frame, and inventing one would merge every
        # zero-extent element into a single class while claiming to have recognised something.
        raise ValueError("degenerate element: all points coincide, so no frame exists")
    Q = _pin_signs(P, vt[:rank].T)                   # (3, rank), signs fixed by the point set
    cols = [Q[:, k] for k in range(rank)]
    while len(cols) < 3:
        if len(cols) == 2:
            cols.append(np.cross(cols[0], cols[1]))  # completes to a right-handed frame: det = +1
        else:
            v = np.eye(3)[int(np.argmin(np.abs(cols[0])))]
            e = v - (v @ cols[0]) * cols[0]
            cols.append(e / np.linalg.norm(e))
    R = np.stack(cols, axis=1)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0                              # only ever touches a DERIVED axis, never a data-pinned one
    return R


def canonical_form(V, family="similarity"):
    """Split an element into `(canonical, delta)` such that `apply_delta(canonical, delta)` reproduces `V` exactly.

    `delta` is `{"family", "A", "b"}` with `V = canonical @ A.T + b`. `canonical` is the element in its own frame:
    centred always, scaled to unit RMS radius for `similarity`, and whitened to identity covariance for `affine`.

    Raises on a degenerate element -- a point set with no volume in the required number of directions has no frame,
    and inventing one would silently merge distinct classes."""
    V = np.asarray(V, float)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("element must be (n, 3); got %r" % (V.shape,))
    if family not in FAMILIES:
        raise ValueError("family must be one of %s; got %r" % (FAMILIES, family))

    c = V.mean(axis=0)
    P = V - c

    if family == "affine":
        # WHITEN WITHIN THE ELEMENT'S OWN AFFINE HULL. A first version whitened the full 3x3 covariance and raised
        # on every triangle -- correctly, because three points span a PLANE and have no full-rank 3-D frame. Every
        # triangle is planar. So take the rank-r subspace the element actually occupies, whiten there, and let the
        # delta carry the r-to-3 embedding. `canonical` is (n, r); the delta's A is (3, r).
        _u, s, vt = np.linalg.svd(P, full_matrices=False)
        rank = int((s > 1e-9 * max(s[0], 1e-30)).sum())
        if rank == 0:
            raise ValueError("degenerate element: all points coincide, so no affine frame exists")
        Q = _pin_signs(P, vt[:rank].T)                # (3, rank) an orthonormal basis of the hull, signs from data
        coords = P @ Q                                # (n, rank) the element in its own hull
        scale = np.sqrt((coords ** 2).mean(axis=0))   # per-axis RMS: whitening the hull's covariance
        canonical = coords / scale
        A = Q * scale                                 # (3, rank): V = canonical @ A.T + c
        return canonical, {"family": family, "A": A, "b": c, "rank": rank}

    R = _frame(P)
    canonical = P @ R
    # `canonical = P @ R` with R orthonormal, so `P = canonical @ R.T`, and `apply_delta` computes
    # `canonical @ A.T + b`. Therefore A = R, not R.T. (The first version wrote R.T and failed its own exactness
    # assertion -- which is what the assertion is for.)
    if family == "rigid":
        return canonical, {"family": family, "A": R, "b": c}

    rms = float(np.sqrt((canonical ** 2).sum(axis=1).mean()))
    if rms < 1e-12:
        raise ValueError("degenerate element: zero extent, so no scale can be recovered")
    return canonical / rms, {"family": family, "A": R * rms, "b": c}


def apply_delta(canonical, delta):
    """`V = canonical @ A.T + b`. The exact inverse of `canonical_form` -- reconstruction is to 1e-14, not to a
    tolerance chosen after the fact."""
    return np.asarray(canonical, float) @ np.asarray(delta["A"], float).T + np.asarray(delta["b"], float)


def class_key(canonical, tol=6):
    """A content hash of the canonical form, rounded to `tol` decimals. `hashlib`, never `hash()`.

    The rounding is the recogniser's resolution: two elements agreeing to `tol` are the same class. Set it below
    the numerical noise of `canonical_form` (~1e-14) and every element is its own class; set it too coarse and
    distinct shapes merge. 6 decimals is well inside the first and well outside the second."""
    q = np.round(np.asarray(canonical, float), int(tol)) + 0.0    # +0.0 normalises -0.0 to 0.0
    return hashlib.sha256(q.tobytes()).hexdigest()[:16]


def recognize(elements, family="similarity", tol=6):
    """Collapse a list of elements into `{canonicals: {key: array}, instances: [(key, delta), ...]}`.

    This is instancing generalised: two elements share a key when they are the same THING modulo the family's
    deltas, not when they happen to be the same array."""
    canonicals, instances = {}, []
    for V in elements:
        canon, delta = canonical_form(V, family=family)
        key = class_key(canon, tol=tol)
        canonicals.setdefault(key, canon)
        instances.append((key, delta))
    return {"canonicals": canonicals, "instances": instances}


def rebuild(recognized):
    """Reconstruct every element, in order. Exact."""
    return [apply_delta(recognized["canonicals"][key], delta) for key, delta in recognized["instances"]]


def _delta_floats(family, rank=3):
    """The cost of one instance's delta, in floats. `rigid` stores a quaternion (4) and a translation (3);
    `similarity` adds a scale; `affine` stores the (3, rank) embedding and a translation.

    For a TRIANGLE the hull has rank 2, so an affine delta is 3*2 + 3 = 9 floats -- exactly the 9 floats of the
    triangle itself. Break-even, before you have stored a single canonical element. That is kept negative 1, and it
    is arithmetic, not bad luck."""
    if family == "affine":
        return 3 * int(rank) + 3
    return {"rigid": 7, "similarity": 8}[family]


def storage_report(elements, family="similarity", tol=6):
    """`{classes, raw_floats, canonical_floats, delta_floats, total_floats, ratio, zlib_ratio, beats_zlib}`.

    The honest baseline is `zlib` on the raw vertex buffer -- which achieves ~1.04x on float64 coordinates, because
    they are high-entropy. That is why this is a real codec for large elements, and a loss for triangles."""
    import zlib

    elements = [np.asarray(V, float) for V in elements]
    rec = recognize(elements, family=family, tol=tol)
    raw = sum(V.size for V in elements)
    canon = sum(c.size for c in rec["canonicals"].values())
    deltas = sum(_delta_floats(family, d.get("rank", 3)) for _k, d in rec["instances"])
    total = canon + deltas
    raw_bytes = np.concatenate(elements).astype(np.float64).tobytes()
    z = len(zlib.compress(raw_bytes, 9))
    return {"classes": len(rec["canonicals"]), "raw_floats": int(raw), "canonical_floats": int(canon),
            "delta_floats": int(deltas), "total_floats": int(total), "ratio": raw / max(1, total),
            "zlib_ratio": len(raw_bytes) / max(1, z), "beats_zlib": bool(raw / max(1, total) > len(raw_bytes) / z)}


def _selftest():
    """Regression trap for C3: reconstruction is exact, the family decides the class count, and the two kept
    negatives hold -- a triangle cannot pay an affine delta, and the dividend scales with element size."""
    rng = np.random.default_rng(0)

    # ONE Rodrigues generator in the engine. `holographic_equivariance.sample_transform` already builds a random
    # rotation, and a structural-duplicate scan found this body copied into three selftests. Delegating is not a
    # style preference: three copies of a rotation is three chances to disagree about what "a rotation" is.
    from holographic.mesh_and_geometry.holographic_equivariance import sample_transform

    def _rot(r):
        return sample_transform("rotate", r)[0]

    # 1. EXACT reconstruction, every family
    for family in FAMILIES:
        V = rng.normal(size=(12, 3))
        canon, delta = canonical_form(V, family=family)
        assert np.abs(apply_delta(canon, delta) - V).max() < 1e-12, family

    # 2. DETERMINISM: the canonical form is bit-stable, so the class key is too
    V = rng.normal(size=(9, 3))
    a, _ = canonical_form(V, "similarity")
    b, _ = canonical_form(V, "similarity")
    assert np.array_equal(a, b) and class_key(a) == class_key(b)

    # 3. THE FAMILY DECIDES THE CLASS COUNT. A scene of 5 base shapes under rotation + translation + SCALE:
    bases = [rng.normal(size=(3, 3)) for _ in range(5)]
    elements = []
    for base in bases:
        for _ in range(40):
            elements.append((rng.uniform(0.5, 2.0) * base) @ _rot(rng).T + rng.normal(size=3))

    n_rigid = storage_report(elements, "rigid")["classes"]
    n_sim = storage_report(elements, "similarity")["classes"]
    n_aff = storage_report(elements, "affine")["classes"]
    assert n_rigid == len(elements)      # UNDER-fits: scale is not in the family, nothing matches
    assert n_sim == len(bases)           # exactly the family the scene was generated with

    # 3b. `affine` collapses triangle SHAPE -- whitening makes every triangle exactly EQUILATERAL, side sqrt(6) --
    #     and leaves the in-hull ROTATION, which an unordered point set cannot pin. So it is 5, not 1, and
    #     "every non-degenerate triangle is affinely the same" is a statement about ORDERED triangles.
    for base in bases:
        canon, _d = canonical_form(base, "affine")
        sides = sorted(float(np.linalg.norm(canon[i] - canon[j])) for i, j in ((0, 1), (1, 2), (2, 0)))
        assert max(abs(s - np.sqrt(6.0)) for s in sides) < 1e-9, sides
    assert n_aff == len(bases)

    # 4. and every family still rebuilds exactly
    for family in FAMILIES:
        rec = recognize(elements, family=family)
        for got, want in zip(rebuild(rec), elements):
            assert np.abs(got - want).max() < 1e-10, family

    # 5. KEPT NEGATIVE 1: a triangle cannot pay an affine delta. A triangle's hull has rank 2, so the delta is
    #    3*2 + 3 = 9 floats -- exactly the triangle. Break-even before storing a single canonical element.
    rep_aff = storage_report(elements, "affine")
    assert rep_aff["ratio"] < 1.0, rep_aff["ratio"]
    assert all(d["rank"] == 2 for _k, d in recognize(elements, "affine")["instances"])   # every triangle is planar
    rep_sim = storage_report(elements, "similarity")
    assert 1.0 < rep_sim["ratio"] < 1.3           # ... and even the right family wins by almost nothing

    # 6. KEPT NEGATIVE 2: the dividend scales with the element's SIZE against the O(1) delta
    big = [rng.normal(size=(200, 3))]
    big = [big[0] @ _rot(rng).T + rng.normal(size=3) for _ in range(30)]
    rep_big = storage_report(big, "rigid")
    assert rep_big["classes"] == 1
    assert rep_big["ratio"] > 20.0
    assert rep_big["beats_zlib"] is True          # ... and zlib barely touches float64 coordinates

    # 7. a rank-1 element (collinear points) is still recognisable in ITS hull; only a single point is degenerate
    collinear = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    canon_c, delta_c = canonical_form(collinear, "affine")
    assert delta_c["rank"] == 1
    assert np.abs(apply_delta(canon_c, delta_c) - collinear).max() < 1e-12

    for degenerate in (np.zeros((4, 3)),):
        try:
            canonical_form(degenerate, "affine")
        except ValueError:
            pass
        else:
            raise AssertionError("a zero-extent element must raise")

    print("OK: holographic_canonmesh self-test passed (reconstruction exact to 1e-12 in all three families; the "
          "FAMILY decides the class count -- rigid %d, similarity %d, affine %d on a scene of %d instances of %d "
          "base shapes under rotation+translation+scale, and `affine` gives 5 rather than 1 because whitening makes "
          "every triangle EQUILATERAL (side sqrt(6), asserted) while leaving the in-hull rotation an unordered point "
          "set cannot pin; and the two kept negatives hold: a triangle cannot pay a 9-float affine delta for a "
          "9-float triangle (ratio %.2fx, a LOSS) while a 200-vertex element pays %.0fx and beats zlib, because the "
          "dividend scales with the ELEMENT against an O(1) delta)"
          % (n_rigid, n_sim, n_aff, len(elements), len(bases), rep_aff["ratio"], rep_big["ratio"]))


if __name__ == "__main__":
    _selftest()
