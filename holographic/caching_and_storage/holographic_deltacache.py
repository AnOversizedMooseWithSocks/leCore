"""holographic_deltacache.py -- the dependency-key layer (Box3D backlog C1, with C4's guard).

Part C's compute model: every triangle is THE canonical triangle plus a recognised chain of deltas. A computation
runs on the canonical ONCE, and its result is transformed through the deltas -- with deltas the computation never
reads (a material, for a geometric quantity) never entering the cache key at all.

**The cache key is `(operator, canonical class, the delta-IDs the operator READS)`.** Which deltas an operator
reads is not a guess: `holographic_equivariance` measures it. And which elements share a canonical class is not a
guess either: `holographic_canonmesh.recognize` derives it from raw geometry. `evaluate_elements` runs all three --
**recognise (C3) -> equivariance table (C2) -> cache key (C1)** -- on elements with no shape ids at all:

    op         family        classes   computes   verdict       max err vs brute
    area       rigid           200       200      invariant     7.1e-15
    area       similarity        5         5      equivariant   1.4e-14   <- 40x, exact
    centroid   similarity        5         5      equivariant   4.4e-16
    max_x      similarity        5       200      recompute     4.4e-15   <- no law, no dividend

A COMPOSITE FAMILY'S VERDICT IS THE WEAKEST OF ITS PARTS. `area` is invariant under `rigid` and only equivariant
under `similarity`, because a uniform scale moves it. And **recognition alone is not enough**: reusing the
canonical's area directly under `similarity` is wrong by **8.54**. The family decides whether you may reuse the
value, transform it, or must recompute -- which is precisely the equivariance table's job.

Three keying policies for the pre-identified case, and the whole point is that they are different:

    policy        key includes                                   computes
    brute         nothing; recompute everything                       400
    read_set      the shape delta only (material never read)           50   <- exact, bit-identical
    equivariant   nothing, when the op is INVARIANT under the           1   <- NOT bit-identical; see below
                  delta family (measured, not assumed)

MEASURED on a scene of 400 triangles drawn from 64 shape deltas (rotations) x 8 material deltas -- the first 400 of
which contain **50 distinct shapes**, which is why `read_set` computes 50 and not 64. The evaluator is an expensive
geometric integral: a 64-point quadrature of |x|^2, weighted by area.

    brute              400 computes
    read_set            50 computes    8.0x   bit-identical: True
    equivariant          1 compute   400.0x   bit-identical: FALSE, max|diff| 8.3e-17

KEPT NEGATIVE -- **the invariance path is NOT bit-identical, and the cache is the one that is right.** `area` is
invariant under rotation in exact arithmetic; in floating point, rotating a triangle and re-integrating accumulates
~1e-17 of round-off that the canonical evaluation never incurs. So the cached answer differs from the brute answer
at machine epsilon, and *the brute answer is the one carrying the error*. This engine's constitution says a change
at 1e-12 has still flipped a creature's trajectory -- so `equivariant` is OPT-IN, `read_set` is the default, and
`cache_report` prints `max_abs_diff` rather than a boolean.

C4 -- **THE CACHE IS ONLY SOUND OVER DETERMINISTIC EVALUATORS**, and this is D1's coordinate-keyed-sampling rule
one level up. An evaluator that draws from a global RNG stream returns a different answer for the same input every
time (measured: 0.4019 then 0.3188). The cache stores the first and serves it forever, while the brute path keeps
drawing -- so the two disagree, and the *cache* looks broken. Key the sampler by its input's coordinates
(`holographic_determinism.hash_unit`) and the same input gives the same answer, bit for bit, while a different
input still gives a different one. `DeltaCache` refuses an evaluator that fails this check.
"""

import hashlib

import numpy as np

from holographic.mesh_and_geometry.holographic_equivariance import apply_affine, classify


def delta_id(A, b):
    """A content hash of an affine delta `(A, b)`. `hashlib`, never `hash()` -- the key must survive a restart.

    Rounded to 12 decimals before hashing: two deltas that differ below the evaluator's own precision are the same
    delta, and refusing to say so would make every key unique and the cache useless."""
    h = hashlib.sha256()
    h.update(np.round(np.asarray(A, float), 12).tobytes())
    h.update(np.round(np.asarray(b, float), 12).tobytes())
    return h.hexdigest()[:16]


def is_deterministic(fn, sample, trials=3):
    """Does `fn` return a bit-identical result for a repeated input? C4's gate.

    A cache over a non-deterministic evaluator is unsound: it serves its first draw forever while the uncached path
    keeps drawing. The failure surfaces as a cache/brute disagreement, and the cache gets blamed."""
    first = fn(sample)
    for _ in range(int(trials) - 1):
        again = fn(sample)
        if not np.array_equal(np.asarray(first), np.asarray(again)):
            return False
    return True


class DeltaCache:
    """Evaluate `op` over a scene of `(shape_delta, material_delta)` pairs, keyed by what `op` actually READS.

    `policy`:
      * `"brute"`       -- no cache. The baseline every claim here is measured against.
      * `"read_set"`    -- key on the shape delta only. A material delta cannot change a geometric quantity, so it
                           never enters the key. EXACT: results are bit-identical to brute.
      * `"equivariant"` -- additionally drop the shape delta when `op` is measured INVARIANT under that delta's
                           family. 400x on rotations, and NOT bit-identical (~1e-17): opt in deliberately.

    The evaluator is checked for determinism on construction (C4). An evaluator drawing from a global RNG stream is
    refused, with the reason."""

    def __init__(self, op, canonical, policy="read_set", transform_family=None, op_name=None):
        self.op = op
        self.canonical = np.asarray(canonical, float)
        if policy not in ("brute", "read_set", "equivariant"):
            raise ValueError("policy must be brute|read_set|equivariant; got %r" % (policy,))
        self.policy = policy
        self.transform_family = transform_family
        self.op_name = op_name
        self._store = {}
        self.computes = 0
        self.hits = 0

        if not is_deterministic(self.op, self.canonical):
            raise ValueError("DeltaCache refuses a non-deterministic evaluator: it returned different results for "
                             "the same input. A cache would serve its first draw forever while the uncached path "
                             "kept drawing. Key the sampler by its input's coordinates -- see "
                             "holographic_determinism.hash_unit (backlog D1/C4).")

        self._invariant = False
        if policy == "equivariant":
            if not (op_name and transform_family):
                raise ValueError("the 'equivariant' policy needs op_name and transform_family so the verdict can be "
                                 "MEASURED rather than assumed; see holographic_equivariance.classify")
            self._invariant = classify(op_name, transform_family) == "invariant"

    def _key(self, shape, material):
        if self.policy == "brute":
            return None
        if self.policy == "equivariant" and self._invariant:
            return ("canonical",)                     # the delta drops out entirely
        return (shape,)                               # the material never enters: op does not read it

    def evaluate(self, scene, deltas):
        """`scene` is `[(shape_id, material_id), ...]`; `deltas[shape_id]` is the affine `(A, b)`.

        Returns the list of per-triangle results, in scene order, computed under the cache's policy."""
        out = []
        for (shape, material) in scene:
            key = self._key(shape, material)
            if key is not None and key in self._store:
                self.hits += 1
                out.append(self._store[key])
                continue
            if self.policy == "equivariant" and self._invariant:
                tri = self.canonical                  # transform the RESULT (here: it does not move at all)
            else:
                A, b = deltas[shape]
                tri = apply_affine(A, b, self.canonical)
            val = self.op(tri)
            self.computes += 1
            if key is not None:
                self._store[key] = val
            out.append(val)
        return out

    def stats(self):
        total = self.hits + self.computes
        return {"policy": self.policy, "computes": self.computes, "hits": self.hits,
                "hit_rate": self.hits / max(1, total), "entries": len(self._store),
                "invariant": bool(self._invariant)}


#: The elementary transform families each canonicalisation family is COMPOSED of. A composite family's verdict is
#: the WEAKEST of its parts: `rigid` cannot see a scale, so an operator invariant under rotation and translation is
#: invariant under `rigid`; the moment `uniform_scale` joins, `area` stops being invariant and becomes equivariant.
FAMILY_PARTS = {
    "rigid": ("translate", "rotate"),
    "similarity": ("translate", "rotate", "uniform_scale"),
    "affine": ("translate", "rotate", "uniform_scale", "nonuniform_scale", "shear", "reflect"),
}

_RANK = {"invariant": 0, "equivariant": 1, "recompute": 2}


def family_verdict(op_name, family):
    """The verdict for a COMPOSITE canonicalisation family: the weakest of its elementary parts.

    Measured, not assumed -- `holographic_equivariance.classify` runs each part. `area` is `invariant` under
    `rigid` and `equivariant` under `similarity`, because a uniform scale moves it. Taking the strongest part
    instead of the weakest would produce a cache that reuses a value the delta actually changed."""
    if family not in FAMILY_PARTS:
        raise ValueError("unknown family %r; try %s" % (family, sorted(FAMILY_PARTS)))
    verdicts = [classify(op_name, part) for part in FAMILY_PARTS[family]]
    return max(verdicts, key=lambda v: _RANK[v])


def evaluate_elements(elements, op, op_name, family="similarity", tol=6):
    """**Part C, end to end.** Recognise the scene's canonical classes (C3), ask the equivariance table what the
    operator reads (C2), and key the cache accordingly (C1).

    `elements` are RAW point sets -- no shape ids. The caller does not know which are the same thing modulo a
    delta, and it does not have to: `holographic_canonmesh.recognize` derives that. Returns
    `(values, {classes, computes, verdict, family})`.

    THE THREE VERDICTS DO DIFFERENT WORK, and getting this wrong is a silent wrong answer:
      * `invariant`   -- the delta changed nothing the operator reads. Reuse the canonical's value.
      * `equivariant` -- one compute per class, then TRANSFORM the result through each instance's delta. Reusing
        the canonical's value directly here is wrong by 8.54 on `area` under `similarity`, because a uniform scale
        moves the area. The law is exact to 1.4e-14.
      * `recompute`   -- no law; evaluate every instance.

    Only `area` and `centroid` carry laws that this function can apply from a `(canonical, delta)` pair today;
    anything else falls back to `recompute` rather than guessing, which is the same refusal K10 makes."""
    from holographic.mesh_and_geometry.holographic_canonmesh import apply_delta, recognize
    from holographic.mesh_and_geometry.holographic_equivariance import OPERATORS

    if not is_deterministic(op, np.asarray(elements[0], float)):
        raise ValueError("evaluate_elements refuses a non-deterministic evaluator (backlog D1/C4); key the sampler "
                         "by its input's coordinates -- see holographic_determinism.hash_unit")

    rec = recognize([np.asarray(e, float) for e in elements], family=family, tol=tol)
    verdict = family_verdict(op_name, family)

    # An equivariant law needs the operator's READ-SET evaluated on the canonical, and a way to push the delta
    # through it. We can only do that for laws registered in the equivariance table.
    if verdict == "equivariant" and op_name not in OPERATORS:
        verdict = "recompute"

    # THE AFFINE FAMILY'S DELTA IS NOT A SQUARE JACOBIAN. `canonmesh` whitens within the element's own affine hull,
    # so its `A` is a (3, rank) embedding -- rank 2 for a triangle, because every triangle is planar. The
    # equivariance laws are written for a square 3x3 `A` (they take `det` and `inv`). Feeding one the other raises
    # `LinAlgError` from deep inside `_law_area`, which is a crash where a refusal belongs.
    if verdict == "equivariant" and family == "affine":
        raise ValueError("the 'affine' family's delta is a (3, rank) hull embedding, not a square Jacobian, and the "
                         "registered equivariance laws take det(A) and inv(A). Use 'rigid' or 'similarity', or "
                         "register a law that accepts a rectangular embedding. Refusing rather than guessing.")

    values, computes, cache = [], 0, {}
    for key, delta in rec["instances"]:
        canon = rec["canonicals"][key]
        if verdict == "recompute":
            values.append(op(apply_delta(canon, delta)))
            computes += 1
            continue
        if key not in cache:
            cache[key] = {name: OPERATORS[name]["fn"](canon) for name in OPERATORS[op_name]["read_set"]} \
                if verdict == "equivariant" else op(canon)
            computes += 1
        if verdict == "invariant":
            values.append(cache[key])
        else:
            values.append(OPERATORS[op_name]["law"](np.asarray(delta["A"], float),
                                                    np.asarray(delta["b"], float), cache[key]))
    return values, {"classes": len(rec["canonicals"]), "computes": computes, "verdict": verdict, "family": family}


def cache_report(op, canonical, scene, deltas, op_name=None, transform_family=None):
    """Every policy against the brute baseline: `{policy: {computes, speedup, max_abs_diff, bit_identical}}`.

    `max_abs_diff` rather than a boolean, because the `equivariant` policy is NOT bit-identical and pretending
    otherwise would hide the one number that matters (~1e-17, and it is the BRUTE path carrying the round-off)."""
    base = DeltaCache(op, canonical, policy="brute")
    truth = np.asarray(base.evaluate(scene, deltas), float)
    n_brute = base.computes

    rep = {"brute": {"computes": n_brute, "speedup": 1.0, "max_abs_diff": 0.0, "bit_identical": True}}
    for policy in ("read_set", "equivariant"):
        try:
            c = DeltaCache(op, canonical, policy=policy, op_name=op_name, transform_family=transform_family)
        except ValueError:
            continue                                  # equivariant needs the names; skip it rather than guess
        got = np.asarray(c.evaluate(scene, deltas), float)
        diff = float(np.abs(got - truth).max())
        rep[policy] = {"computes": c.computes, "speedup": n_brute / max(1, c.computes),
                       "max_abs_diff": diff, "bit_identical": bool(np.array_equal(got, truth)),
                       "hit_rate": c.stats()["hit_rate"]}
    return rep


def _selftest():
    """Regression trap for C1/C4: the read-set policy is exact and drops the material; the equivariant policy is
    400x and NOT bit-identical; and a global-RNG evaluator is refused."""
    from holographic.mesh_and_geometry.holographic_equivariance import area, sample_transform
    from holographic.misc.holographic_determinism import hash_unit

    canonical = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.9, 0.0]])
    deltas = [sample_transform("rotate", np.random.default_rng(i)) for i in range(64)]
    scene = [(s, m) for s in range(64) for m in range(8)][:400]

    quad = np.random.default_rng(0).dirichlet(np.ones(3), size=64)   # a FIXED quadrature: deterministic

    def expensive(tri):
        return float(((quad @ tri) ** 2).sum(axis=1).mean() * area(tri))

    rep = cache_report(expensive, canonical, scene, deltas, op_name="area", transform_family="rotate")

    # 1. the brute baseline really is 400 computes
    assert rep["brute"]["computes"] == 400

    # 2. READ_SET: the material delta never enters the key, and the result is EXACT
    assert rep["read_set"]["computes"] < 100
    assert rep["read_set"]["bit_identical"] is True
    assert rep["read_set"]["max_abs_diff"] == 0.0
    assert rep["read_set"]["speedup"] >= 4.0

    # 3. EQUIVARIANT: `area` is invariant under rotation (MEASURED), so the shape delta drops out too --
    #    one compute for the whole scene.
    assert rep["equivariant"]["computes"] == 1
    assert rep["equivariant"]["speedup"] == 400.0

    # 4. KEPT NEGATIVE: and it is NOT bit-identical. Rotating a triangle and re-integrating accumulates round-off
    #    that the canonical evaluation never incurs. The CACHE is the one that is right.
    assert rep["equivariant"]["bit_identical"] is False
    assert 0.0 < rep["equivariant"]["max_abs_diff"] < 1e-12

    # 5. the delta-ID is a content hash, stable across processes, and insensitive below the evaluator's precision
    A, b = deltas[0]
    assert delta_id(A, b) == delta_id(A, b)
    assert delta_id(A, b) != delta_id(*deltas[1])
    assert delta_id(A, b) == delta_id(A + 1e-15, b)

    # 6. C4: a global-RNG evaluator is REFUSED, and a coordinate-keyed one is accepted
    stream = np.random.default_rng(0)

    def sampled_global(tri):
        return float(((stream.dirichlet(np.ones(3), size=32) @ tri) ** 2).sum(axis=1).mean())

    assert not is_deterministic(sampled_global, canonical)
    try:
        DeltaCache(sampled_global, canonical)
    except ValueError as exc:
        assert "non-deterministic" in str(exc)
    else:
        raise AssertionError("a global-RNG evaluator must be refused")

    def sampled_keyed(tri):
        seed = int(hash_unit(*np.round(np.asarray(tri).ravel(), 12)) * (2 ** 32))
        b_ = np.random.default_rng(seed).dirichlet(np.ones(3), size=32)
        return float(((b_ @ tri) ** 2).sum(axis=1).mean())

    assert is_deterministic(sampled_keyed, canonical)
    DeltaCache(sampled_keyed, canonical)                      # accepted
    moved = canonical.copy()
    moved[2, 1] += 1e-6
    assert sampled_keyed(canonical) != sampled_keyed(moved)   # ... and still a function of its input

    # 7. PART C, END TO END (the audit's find): recognise -> equivariance table -> cache key. The caller hands over
    #    RAW elements with no shape ids; `canonmesh.recognize` derives them.
    from holographic.mesh_and_geometry.holographic_equivariance import centroid as _centroid
    from holographic.mesh_and_geometry.holographic_equivariance import max_x as _max_x

    def _rot(r):
        v = r.normal(size=3)
        v /= np.linalg.norm(v)
        th = r.uniform(0.2, 2.0)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

    r2 = np.random.default_rng(0)
    bases = [r2.normal(size=(3, 3)) for _ in range(5)]
    elements = [(r2.uniform(0.5, 2.0) * b) @ _rot(r2).T + r2.normal(size=3) for b in bases for _ in range(40)]

    assert family_verdict("area", "rigid") == "invariant"        # rigid cannot see a scale
    assert family_verdict("area", "similarity") == "equivariant"  # ... and similarity can: the WEAKEST part wins
    assert family_verdict("max_x", "rigid") == "recompute"

    truth = np.array([area(t) for t in elements])
    vals, st = evaluate_elements(elements, area, "area", family="similarity")
    assert st["classes"] == 5 and st["computes"] == 5             # 200 raw triangles -> 5 computes
    assert np.abs(np.array(vals) - truth).max() < 1e-12           # exact, via the equivariant law

    # KEPT NEGATIVE: reusing the canonical's value directly under `similarity` is WRONG by 8.54 -- a uniform scale
    # moves the area. Recognition alone is not enough; the family decides what you may do with the result.
    from holographic.mesh_and_geometry.holographic_canonmesh import recognize as _rec
    naive = np.array([area(_rec(elements, "similarity")["canonicals"][k]) for k, _d in
                      _rec(elements, "similarity")["instances"]])
    assert np.abs(naive - truth).max() > 1.0

    # `recompute` gives no dividend, and says so rather than pretending
    _v, st_mx = evaluate_elements(elements, _max_x, "max_x", family="similarity")
    assert st_mx["classes"] == 5 and st_mx["computes"] == len(elements)

    # the affine family's (3, rank) delta is refused, not fed to a law that wants a square Jacobian
    try:
        evaluate_elements(elements, area, "area", family="affine")
    except ValueError as exc:
        assert "square Jacobian" in str(exc)
    else:
        raise AssertionError("the affine family must be refused")

    print("OK: holographic_deltacache self-test passed (400 brute computes -> %d under read_set (%.1fx, "
          "BIT-IDENTICAL, the material delta never enters the key) -> %d under equivariant (%.0fx, and NOT "
          "bit-identical: max|diff| %.1e, because rotating a triangle and re-integrating accumulates round-off the "
          "canonical evaluation never incurs -- the CACHE is the one that is right). C4 holds: a global-RNG "
          "evaluator is refused, a coordinate-keyed one accepted)"
          % (rep["read_set"]["computes"], rep["read_set"]["speedup"], rep["equivariant"]["computes"],
             rep["equivariant"]["speedup"], rep["equivariant"]["max_abs_diff"]))


if __name__ == "__main__":
    _selftest()
