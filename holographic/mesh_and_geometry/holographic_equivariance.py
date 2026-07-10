"""holographic_equivariance.py -- the equivariance table (Box3D backlog C2), MEASURED rather than declared.

Part C's proposal: every triangle is a delta of THE canonical triangle, a computation runs on the canonical once,
and its RESULT is transformed through the delta. Three mechanisms make that legal, and the whole cache policy is
knowing which one applies to which (operator, transform) pair:

    INVARIANT     the delta drops out entirely       -- op(T x) == op(x)             -> omit T from the cache key
    EQUIVARIANT   the delta becomes a transform of the output   -- op(T x) == law(T, read_set(x))
    ADJOINT       the delta moves to the other operand          -- op(T x, y) == op(x, T* y)
    RECOMPUTE     no law exists                                 -- the honest entry

This module does not assert that table. It MEASURES it: `classify` runs the operator on transformed inputs across
seeds and returns the strongest verdict that holds to a stated tolerance.

THE MEASURED TABLE (affine transforms x triangle operators, 1e-9 tolerance, 12 trials each):

    operator   translate    rotate      uniform     nonuniform   shear       reflect
    area       invariant    invariant   equivariant equivariant  equivariant invariant
    centroid   equivariant  equivariant equivariant equivariant  equivariant equivariant
    normal     invariant    equivariant invariant   equivariant  equivariant equivariant
    inertia    invariant    equivariant equivariant equivariant  equivariant equivariant
    max_x      equivariant  RECOMPUTE   equivariant equivariant  RECOMPUTE   invariant

`max_x` (the largest vertex x-coordinate) is here so the `recompute` branch is exercised by something REAL rather
than by an omission. It is equivariant under anything that acts on x alone, and it has no law under rotation --
because *which vertex attained the maximum* is information the scalar threw away. The read-set that would rescue it
is all three vertices, i.e. the input. **That is what `recompute` means.**

THE FINDING, and it cost me two wrong cells. My first pass reported `area` under shear and non-uniform scale as
RECOMPUTE, and `normal` under reflection as RECOMPUTE. **Both were my missing law, not a missing law.** From
`A e1 x A e2 = det(A) * A^-T (e1 x e2)`:

    area(A x)   = |det A| * ||A^-T n|| * area(x)          -- verified to 1.8e-15 for EVERY affine
    normal(A x) = sign(det A) * normalize(A^-T n)         -- verified to 6.7e-16; a reflection flips the winding

**`RECOMPUTE` must mean NO LAW EXISTS, not "I did not write one down."** A table that says recompute where a law
exists is a cache that never fires, and it looks exactly like a table that is merely honest.

AND THE READ-SET IS THE POINT. `area`'s law reads the NORMAL, not just the area. So the cache key for area under a
non-rigid delta must carry the normal's class too. Part C says "filtered to the deltas the computation reads"; the
measurement says the filter is over the quantities the LAW reads, and every non-rigid law here reads the normal.

THE ADJOINT, and where the famous version of it is wrong. Part C's claim C is *"shade a rotated triangle by
unrotating the light"*: `shade(R x, L) == shade(x, R^-1 L)`. Measured, that holds for a ROTATION (3.9e-16) and
FAILS for everything else -- including a plain UNIFORM SCALE, by 0.38, because the normal is renormalised and the
scale does not cancel. The law that holds for every affine is

    shade(A x, L) == max(0, n . (A^-1 L) / ||A^-T n||)

... which, again, reads the normal. `shade` is ADJOINT under orthogonal transforms and adjoint-with-a-correction
otherwise -- and the correction is exactly the factor the naive statement drops.
"""

import numpy as np


# ---- the operators. Each is a pure function of a (3, 3) triangle (rows = vertices). ---------------------------

def area(tri):
    """Triangle area. Its transformation law reads the NORMAL as well as the area."""
    e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
    return 0.5 * float(np.linalg.norm(np.cross(e1, e2)))


def centroid(tri):
    """The vertex mean. The only operator here whose law reads nothing but its own value."""
    return np.asarray(tri, float).mean(axis=0)


def normal(tri):
    """The unit face normal. Orientation-sensitive: a reflection flips it."""
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    return n / np.linalg.norm(n)


def inertia(tri):
    """The second moment of the vertices about their centroid: a (3, 3) tensor, so it transforms as one."""
    d = np.asarray(tri, float) - centroid(tri)
    return d.T @ d / 3.0


def shade(tri, light):
    """Lambert shading against a FIXED world light -- a world-coupled quantity, which is why it is the adjoint case."""
    return max(0.0, float(normal(tri) @ np.asarray(light, float)))


# ---- the transformation laws. `read_set` names what each law needs from the canonical result. -----------------

def _law_area(A, _b, rs):
    return abs(np.linalg.det(A)) * float(np.linalg.norm(np.linalg.inv(A).T @ rs["normal"])) * rs["area"]


def _law_centroid(A, b, rs):
    return A @ rs["centroid"] + b


def _law_normal(A, _b, rs):
    v = np.linalg.inv(A).T @ rs["normal"]
    return float(np.sign(np.linalg.det(A))) * v / np.linalg.norm(v)


def _law_inertia(A, _b, rs):
    return A @ rs["inertia"] @ A.T


def max_x(tri):
    """The largest vertex x-coordinate. A GENUINE `recompute` case, and it is here so the classifier's negative
    branch is exercised by something real rather than by an omission.

    It is invariant under a z-reflection (which never touches x) and equivariant under a translation (add b_x). It
    has NO law under rotation: `max_i (A x_i)_x` depends on WHICH vertex attained the maximum before, and that is
    information the scalar `max_x` threw away. The read-set that would rescue it is all three vertices -- i.e. the
    input -- which is what `recompute` means."""
    return float(np.asarray(tri, float)[:, 0].max())


def _law_max_x(A, b, rs):
    """The only law available from `max_x` alone: valid when A leaves x alone, wrong otherwise. Written honestly
    so the classifier can FAIL it, which is the point."""
    return float(A[0, 0]) * rs["max_x"] + float(b[0])


OPERATORS = {
    "area": {"fn": area, "law": _law_area, "read_set": ("area", "normal")},
    "centroid": {"fn": centroid, "law": _law_centroid, "read_set": ("centroid",)},
    "normal": {"fn": normal, "law": _law_normal, "read_set": ("normal",)},
    "inertia": {"fn": inertia, "law": _law_inertia, "read_set": ("inertia",)},
    "max_x": {"fn": max_x, "law": _law_max_x, "read_set": ("max_x",)},
}


# ---- the transforms ------------------------------------------------------------------------------------------

def _rotation(rng):
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    th = rng.uniform(0.2, 2.0)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def sample_transform(kind, rng):
    """A random member of the named affine family, as `(A, b)`. Deterministic given `rng`."""
    if kind == "translate":
        return np.eye(3), rng.normal(size=3)
    if kind == "rotate":
        return _rotation(rng), np.zeros(3)
    if kind == "uniform_scale":
        return float(rng.uniform(0.4, 2.0)) * np.eye(3), np.zeros(3)
    if kind == "nonuniform_scale":
        return np.diag(rng.uniform(0.4, 2.0, 3)), np.zeros(3)
    if kind == "shear":
        A = np.eye(3)
        A[0, 1] = float(rng.uniform(0.3, 1.0))
        return A, np.zeros(3)
    if kind == "reflect":
        return np.diag([1.0, 1.0, -1.0]), np.zeros(3)
    raise ValueError("unknown transform family %r; try %s" % (kind, sorted(TRANSFORMS)))


TRANSFORMS = ("translate", "rotate", "uniform_scale", "nonuniform_scale", "shear", "reflect")


def apply_affine(A, b, tri):
    """`x -> A x + b`, applied to each row of a (3, 3) triangle."""
    return np.asarray(tri, float) @ np.asarray(A, float).T + np.asarray(b, float)


# ---- the classifier ------------------------------------------------------------------------------------------

def _close(x, y, tol):
    return bool(np.abs(np.asarray(x, float) - np.asarray(y, float)).max() <= tol)


def classify(op_name, transform, trials=12, tol=1e-9, seed=0):
    """MEASURE the strongest verdict that holds for `op_name` under the `transform` family.

    Returns one of `"invariant"`, `"equivariant"`, `"recompute"`. Invariance is checked first because it is the
    strongest claim (the delta leaves the cache key entirely); equivariance is checked against the operator's
    REGISTERED law, evaluated on the operator's read-set of the canonical result.

    A `recompute` verdict is a statement that the registered law FAILED -- which may mean no law exists, or that
    the registered one is wrong. Those look identical from here, and the second is the one that bit this module:
    two cells read `recompute` until the laws were derived properly. When you see `recompute`, derive before you
    believe."""
    if op_name not in OPERATORS:
        raise ValueError("unknown operator %r; try %s" % (op_name, sorted(OPERATORS)))
    spec = OPERATORS[op_name]
    rng = np.random.default_rng(seed)

    invariant = True
    equivariant = True
    for _ in range(int(trials)):
        tri = rng.normal(size=(3, 3))
        A, b = sample_transform(transform, rng)
        out0 = spec["fn"](tri)
        out1 = spec["fn"](apply_affine(A, b, tri))
        if not _close(out0, out1, tol):
            invariant = False
        rs = {k: OPERATORS[k]["fn"](tri) for k in spec["read_set"]}
        if not _close(out1, spec["law"](A, b, rs), tol):
            equivariant = False
        if not (invariant or equivariant):
            break
    if invariant:
        return "invariant"
    return "equivariant" if equivariant else "recompute"


def equivariance_table(trials=12, tol=1e-9, seed=0):
    """The whole policy, measured: `{operator: {transform: verdict}}`. Regenerate it on your own box; a table that
    cannot re-measure itself is a rumour."""
    return {op: {t: classify(op, t, trials=trials, tol=tol, seed=seed) for t in TRANSFORMS} for op in OPERATORS}


def cache_policy(op_name, transform, trials=12, tol=1e-9, seed=0):
    """The cache-key consequence of the verdict: `{verdict, key_includes_delta, read_set, note}`.

    * `invariant`   -> the delta is omitted from the key. Every member of the family collapses to one entry.
    * `equivariant` -> the delta stays out of the COMPUTE key (one compute per canonical class) but the law needs
      the operator's `read_set`, so those quantities must be cached alongside it.
    * `recompute`   -> the delta is part of the key; there is nothing to reuse."""
    verdict = classify(op_name, transform, trials=trials, tol=tol, seed=seed)
    rs = OPERATORS[op_name]["read_set"]
    note = {"invariant": "the delta drops out of the key entirely",
            "equivariant": "one compute per canonical class; the law reads %s" % (", ".join(rs),),
            "recompute": "no law held; the delta is part of the key"}[verdict]
    return {"verdict": verdict, "key_includes_delta": verdict == "recompute",
            "read_set": list(rs), "note": note}


def shade_adjoint(tri, light, A):
    """The ADJOINT move done correctly for ANY affine: push the delta onto the LIGHT.

        shade(A x, L) == max(0, n . (A^-1 L) / ||A^-T n||)

    Part C states it as *"shade a rotated triangle by unrotating the light"*, `shade(A x, L) == shade(x, A^-1 L)`.
    Measured, that is exact for a ROTATION (3.9e-16) and wrong for everything else -- including a plain uniform
    scale, by 0.38 -- because the normal is renormalised and the scale does not cancel. The factor above is what
    the naive statement drops. It reads the normal, like every other non-rigid law here."""
    n = normal(tri)
    A = np.asarray(A, float)
    val = float(n @ (np.linalg.inv(A) @ np.asarray(light, float))) / float(np.linalg.norm(np.linalg.inv(A).T @ n))
    return max(0.0, val)


def _selftest():
    """Regression trap: the table's every cell, the two cells that were WRONG until the laws were derived, and the
    adjoint statement that is true only for rotations."""
    table = equivariance_table()

    # 1. the strongest claims: what genuinely does not move
    assert table["area"]["translate"] == "invariant"
    assert table["area"]["rotate"] == "invariant"
    assert table["normal"]["translate"] == "invariant"
    assert table["normal"]["uniform_scale"] == "invariant"
    assert table["inertia"]["translate"] == "invariant"
    assert table["centroid"]["translate"] == "equivariant"    # a translation MOVES the centroid; it is not invariant

    # 2. Of the four GEOMETRIC operators, not one cell is `recompute`. Both that were, were my missing law.
    for op in ("area", "centroid", "normal", "inertia"):
        for t, verdict in table[op].items():
            assert verdict != "recompute", (op, t)

    # ... and `max_x` is the genuine recompute case, present so the negative branch is exercised by something real.
    assert table["max_x"]["translate"] == "equivariant"        # x -> x + b_x: the law holds
    assert table["max_x"]["reflect"] == "invariant"            # a z-reflection never touches x
    assert table["max_x"]["rotate"] == "recompute"             # WHICH vertex won is information max_x threw away
    assert table["max_x"]["shear"] == "recompute"

    # 3. the two laws that had to be derived, checked directly at machine precision
    rng = np.random.default_rng(1)
    for kind in TRANSFORMS:
        for _ in range(8):
            tri = rng.normal(size=(3, 3))
            A, b = sample_transform(kind, rng)
            t2 = apply_affine(A, b, tri)
            rs = {"area": area(tri), "normal": normal(tri)}
            assert abs(_law_area(A, b, rs) - area(t2)) < 1e-12
            assert np.abs(_law_normal(A, b, {"normal": normal(tri)}) - normal(t2)).max() < 1e-12

    # 4. THE ADJOINT: the famous statement is true for rotations and false for a uniform scale
    rng = np.random.default_rng(2)
    worst_rot, worst_scale = 0.0, 0.0
    for _ in range(12):
        tri = rng.normal(size=(3, 3))
        L = rng.normal(size=3)
        L /= np.linalg.norm(L)
        R, _ = sample_transform("rotate", rng)
        S = 1.7 * np.eye(3)
        worst_rot = max(worst_rot, abs(shade(apply_affine(R, 0, tri), L) - shade(tri, np.linalg.inv(R) @ L)))
        worst_scale = max(worst_scale, abs(shade(apply_affine(S, 0, tri), L) - shade(tri, np.linalg.inv(S) @ L)))
        # ... while the corrected adjoint holds for BOTH
        for A in (R, S):
            assert abs(shade(apply_affine(A, 0, tri), L) - shade_adjoint(tri, L, A)) < 1e-12
    assert worst_rot < 1e-12
    assert worst_scale > 0.1, worst_scale                      # measured 0.38: the naive statement is wrong here

    # 5. the cache policy follows from the verdict
    assert cache_policy("area", "rotate")["key_includes_delta"] is False
    assert set(cache_policy("area", "shear")["read_set"]) == {"area", "normal"}
    assert cache_policy("centroid", "translate")["verdict"] == "equivariant"

    # 6. the classifier refuses what it does not know
    for bad in (lambda: classify("nonsense", "rotate"), lambda: sample_transform("wobble", np.random.default_rng(0))):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown name must raise")

    n_cells = sum(len(row) for row in table.values())
    n_recompute = sum(1 for row in table.values() for v in row.values() if v == "recompute")
    print("OK: holographic_equivariance self-test passed (the table is MEASURED -- %d cells, %d of them `recompute`, "
          "and all %d belong to `max_x`, the operator that genuinely has no law. Not one of the 24 GEOMETRIC cells "
          "is recompute; the two that were became `equivariant` once the laws were derived: area(Ax) = |det A| * "
          "||A^-T n|| * area(x) and normal(Ax) = sign(det A) * normalize(A^-T n), both exact to 1e-12 for every "
          "affine family. And Part C's 'unrotate the light' is exact for a rotation (%.1e) but wrong by %.2f for a "
          "uniform scale -- the corrected adjoint reads the normal, as every non-rigid law here does)"
          % (n_cells, n_recompute, n_recompute, worst_rot, worst_scale))


if __name__ == "__main__":
    _selftest()
