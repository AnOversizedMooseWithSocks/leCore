"""StructureRecipe validator + edit operators (ARCH-1): the recipe equivalent of the mesh Euler operators.

WHY THIS MODULE EXISTS
----------------------
The §ARCH program turns the 3-D DCC concepts INWARD on the engine's own structures. FWD-7 gave the mesh its local,
invariant-preserving editors (the Euler operators: flip/split/collapse, each preserving the manifold and chi).
ARCH-1 is the exact mirror for the StructureRecipe (holographic_recipe) -- the one build-graph that program, tree,
and scene all reduce to (B7). A recipe needs the same two things a mesh kernel needs: a way to CHECK a structure
is well-formed, and a set of LOCAL EDITS that rewrite it while preserving its meaning.

THE KEY PARALLEL
  A mesh Euler operator preserves a topological invariant (chi, the manifold property). A recipe edit operator
  preserves the REALIZED VECTOR -- and it does so for the same reason a flip preserves chi: it is a local rewrite
  that is an IDENTITY of the underlying algebra. The engine's bind is circular convolution (commutative), and
  bundle/superpose are sums (commutative), so:
    * commute_bind  -- bind(a,b) = bind(b,a)         <-> flip_edge (its own inverse, preserves the invariant)
    * reorder_members -- bundle([a,b,c]) = bundle(any order)  <-> a parameterised flip (invertible by the inverse perm)
  These are vector-preserving to FFT precision. A CONTENT edit (changing WHICH atom a leaf is) is the recipe
  analogue of moving a vertex -- it preserves the STRUCTURE (validity) while changing the result predictably, and
  is invertible by substituting back:
    * substitute_atom  <-> a vertex-position edit (topology fixed, geometry changes, reversible)

WHAT IT PROVIDES
  * validate(recipe) -> (ok, problems) -- the well-formedness checker: every op references only EARLIER existing
    results (a DAG, no forward/dangling/out-of-range refs), raw indices and repeat templates are in range. This is
    the recipe's is_manifold().
  * commute_bind(recipe, handle) -- swap the two arguments of a bind. Vector-preserving, ITS OWN INVERSE.
  * reorder_members(recipe, handle, perm) -- permute a bundle/superpose's members. Vector-preserving, invertible
    by the inverse permutation.
  * substitute_atom(recipe, handle, new_name) -- rename an atom leaf. Validity-preserving; the result changes
    predictably; invertible by renaming back.
  Each returns a NEW recipe (the originals are untouched, exactly as the mesh operators returned new meshes).

THE MEASUREMENT BAR (checked exactly in the self-test)
  * validate accepts a well-formed recipe and REJECTS a corrupted one (a forward/out-of-range reference).
  * commute_bind and reorder_members leave the realized vector BIT-EXACT (to FFT precision) and the recipe valid;
    commute_bind applied twice is the identity; reorder undone by the inverse perm is the identity.
  * substitute_atom CHANGES the realized vector and is reversed exactly by substituting the original name back.

DETERMINISM (per ISA.md)
  Pure structural rewrites; recipes replay bit-exact from their seed (the recipe module's contract). No RNG.

KEPT NEGATIVES (loud)
  * These are the VECTOR-PRESERVING / structure-preserving edits (the recipe's Euler-operator core). Edits that
    REMOVE or RESIZE ops (flatten a nested superpose, splice out dead results) require re-indexing every downstream
    handle -- the recipe analogue of the mesh face-list reindex in collapse/dissolve -- and are deferred here; the
    in-place edits below are correct and complete on their own, the same way flip_edge is.
  * commute_bind / reorder are bit-exact only up to FFT/float round-off (~1e-12), not literally identical bits --
    the same honest caveat the bind_batch vectorization carries (an algebraic identity, FP-equal not bit-equal).
"""

import copy

import numpy as np

from holographic.misc.holographic_recipe import StructureRecipe


def _clone(recipe):
    """A deep, independent copy of a recipe (so an edit never mutates the original)."""
    r = StructureRecipe(recipe.dim, recipe.seed)
    r._ops = copy.deepcopy(recipe._ops)
    r._raws = [x.copy() for x in recipe._raws]
    r._outputs = list(recipe._outputs)
    r._n_results = recipe._n_results
    return r


def _op_index_for_handle(recipe, handle):
    """The position in `_ops` of the op that PRODUCES result `handle` (handles are absolute result indices, and a
    `repeat` op produces several, so this is not just `handle`)."""
    count = 0
    for p, op in enumerate(recipe._ops):
        n = op[1] if op[0] == "repeat" else 1
        if count <= handle < count + n:
            return p
        count += n
    raise IndexError(f"handle {handle} is out of range")


def _template_problems(template, op_pos):
    """Validate a repeat template's INTERNAL references (each local op refers to an earlier local result)."""
    problems = []
    seen = 0
    for q, t in enumerate(template):
        kind = t[0]
        refs = []
        if kind == "bind":
            refs = [t[1], t[2]]
        elif kind == "bundle":
            refs = list(t[1])
        elif kind in ("permute", "normalize"):
            refs = [t[1]]
        elif kind == "atom":
            pass
        else:
            problems.append(f"op {op_pos} template[{q}]: unknown kind {kind!r}")
        for r in refs:
            if not isinstance(r, int) or r < 0 or r >= seen:
                problems.append(f"op {op_pos} template[{q}]: reference {r!r} not an earlier local result (< {seen})")
        seen += 1
    return problems


def validate(recipe):
    """Check a recipe is WELL-FORMED -- the recipe's is_manifold(). Every op must reference only EARLIER, existing
    results (the build-graph is a DAG: no forward, dangling, or out-of-range references); raw indices and repeat
    templates must be in range. Returns (ok, problems) where `problems` is a list of human-readable strings (empty
    iff ok)."""
    problems = []
    count = 0                                              # results that exist BEFORE the current op
    for p, op in enumerate(recipe._ops):
        kind = op[0]
        refs = []
        if kind == "bind":
            refs = [op[1], op[2]]
        elif kind in ("bundle", "superpose"):
            refs = list(op[1])
        elif kind in ("permute", "normalize"):
            refs = [op[1]]
        elif kind == "raw":
            if not (0 <= op[1] < len(recipe._raws)):
                problems.append(f"op {p}: raw index {op[1]} out of range (have {len(recipe._raws)})")
        elif kind == "atom":
            pass
        elif kind == "repeat":
            problems.extend(_template_problems(op[2], p))
        else:
            problems.append(f"op {p}: unknown kind {kind!r}")
        for r in refs:
            if not isinstance(r, (int, np.integer)) or r < 0:
                problems.append(f"op {p}: bad reference {r!r}")
            elif r >= count:
                problems.append(f"op {p}: reference {r} points to a not-yet-produced result (>= {count}) "
                                f"-- forward/dangling reference")
        count += op[1] if kind == "repeat" else 1
    return (len(problems) == 0, problems)


def commute_bind(recipe, handle):
    """Swap the two arguments of the bind producing `handle`: bind(a,b) -> bind(b,a). Because the engine's bind is
    circular convolution (commutative), the realized vector is unchanged (to FFT precision). ITS OWN INVERSE --
    the recipe analogue of mesh flip_edge. Returns a new recipe."""
    r = _clone(recipe)
    p = _op_index_for_handle(r, handle)
    op = r._ops[p]
    if op[0] != "bind":
        raise ValueError(f"handle {handle} is not a bind (it is {op[0]!r})")
    r._ops[p] = ("bind", op[2], op[1])
    return r


def reorder_members(recipe, handle, perm):
    """Permute the members of the bundle/superpose producing `handle` by `perm` (a permutation of its member
    positions). Because bundle/superpose are sums (commutative), the realized vector is unchanged (to FFT
    precision). Invertible by the inverse permutation. Returns a new recipe."""
    r = _clone(recipe)
    p = _op_index_for_handle(r, handle)
    op = r._ops[p]
    if op[0] not in ("bundle", "superpose"):
        raise ValueError(f"handle {handle} is not a bundle/superpose (it is {op[0]!r})")
    members = list(op[1])
    if sorted(perm) != list(range(len(members))):
        raise ValueError(f"perm must be a permutation of 0..{len(members) - 1}")
    r._ops[p] = (op[0], [members[i] for i in perm])
    return r


def substitute_atom(recipe, handle, new_name):
    """Rename the atom leaf producing `handle` to `new_name` -- the recipe analogue of moving a vertex: it keeps
    the STRUCTURE valid while changing the realized vector predictably (a different atom), and is reversed exactly
    by substituting the original name back. Returns a new recipe."""
    r = _clone(recipe)
    p = _op_index_for_handle(r, handle)
    op = r._ops[p]
    if op[0] != "atom":
        raise ValueError(f"handle {handle} is not an atom (it is {op[0]!r})")
    r._ops[p] = ("atom", str(new_name), op[2])
    return r


# =====================================================================================================
# Self-test -- validate accepts/rejects; the structure edits preserve the vector + invert; content edit reverses.
# =====================================================================================================
def _selftest():
    r = StructureRecipe(dim=512, seed=0)
    a = r.atom("a"); b = r.atom("b"); c = r.atom("c")
    ab = r.bind(a, b)                                       # handle 3
    bun = r.bundle([a, b, c])                               # handle 4
    r.mark_output(ab); r.mark_output(bun)
    base = [v.copy() for v in r.outputs()]

    # --- validate accepts a well-formed recipe, REJECTS a corrupted one (forward/out-of-range reference) ---
    ok, problems = validate(r)
    assert ok, problems
    bad = _clone(r)
    bad._ops[3] = ("bind", 3, 99)                          # references handle 99 (and itself) -- forward/dangling
    bad_ok, bad_problems = validate(bad)
    assert not bad_ok and bad_problems, "validate must reject a forward/dangling reference"

    # --- commute_bind: vector unchanged (FFT precision), its own inverse, still valid ---
    flipped = commute_bind(r, ab)
    assert validate(flipped)[0]
    assert np.allclose(flipped.outputs()[0], base[0], atol=1e-12), "commute_bind preserves the realized vector"
    twice = commute_bind(flipped, ab)
    assert np.allclose(twice.outputs()[0], base[0], atol=1e-12) and twice._ops[3] == r._ops[3], "own inverse"

    # --- reorder_members: vector unchanged, invertible by the inverse perm, still valid ---
    perm = [2, 0, 1]
    reordered = reorder_members(r, bun, perm)
    assert validate(reordered)[0]
    assert np.allclose(reordered.outputs()[1], base[1], atol=1e-12), "reorder preserves the realized vector (sum is commutative)"
    inv = [perm.index(i) for i in range(len(perm))]
    assert np.allclose(reorder_members(reordered, bun, inv).outputs()[1], base[1], atol=1e-12)

    # --- substitute_atom: the result CHANGES, and substituting back restores it exactly ---
    swapped = substitute_atom(r, a, "z")
    assert validate(swapped)[0]
    assert not np.allclose(swapped.outputs()[0], base[0], atol=1e-6), "a different atom must change the result"
    restored = substitute_atom(swapped, a, "a")
    assert np.allclose(restored.outputs()[0], base[0], atol=1e-12), "substituting the original name back restores it"

    # --- determinism ---
    assert np.array_equal(commute_bind(r, ab).outputs()[0], commute_bind(r, ab).outputs()[0])

    print("holographic_recipeops selftest: ok (the recipe's Euler operators: validate accepts well-formed + "
          "REJECTS a dangling reference; commute_bind preserves the vector and is its own inverse; reorder_members "
          "preserves the vector and inverts by the inverse perm; substitute_atom changes the result and reverses "
          "exactly; deterministic)")


if __name__ == "__main__":
    _selftest()
