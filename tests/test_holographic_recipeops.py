"""Tests for the StructureRecipe validator + edit operators (ARCH-1): the recipe equivalent of the mesh Euler
operators. validate accepts/rejects; the structure edits (commute_bind, reorder_members) preserve the realized
vector and invert; the content edit (substitute_atom) changes the result and reverses; edits don't mutate the
original."""

import numpy as np

from holographic.misc.holographic_recipe import StructureRecipe
from holographic.misc.holographic_recipeops import validate, commute_bind, reorder_members, substitute_atom, _clone, _op_index_for_handle


def _recipe():
    r = StructureRecipe(dim=512, seed=0)
    a = r.atom("a"); b = r.atom("b"); c = r.atom("c")
    ab = r.bind(a, b)
    bun = r.bundle([a, b, c])
    r.mark_output(ab); r.mark_output(bun)
    return r, ab, bun


# ---- validate ----------------------------------------------------------------------------------
def test_validate_accepts_well_formed():
    r, _, _ = _recipe()
    ok, problems = validate(r)
    assert ok and problems == []


def test_validate_rejects_forward_reference():
    r, _, _ = _recipe()
    bad = _clone(r)
    bad._ops[3] = ("bind", 3, 99)                          # references a not-yet-produced / nonexistent result
    ok, problems = validate(bad)
    assert not ok and len(problems) >= 1


def test_validate_rejects_out_of_range_raw():
    r = StructureRecipe(dim=128, seed=0)
    r.atom("a")
    bad = _clone(r)
    bad._ops.append(("raw", 5))                            # raw index 5 with no raw payloads
    bad._n_results += 1
    assert not validate(bad)[0]


# ---- commute_bind (the flip_edge analogue) -----------------------------------------------------
def test_commute_bind_preserves_the_realized_vector():
    r, ab, _ = _recipe()
    base = r.outputs()[0]
    assert np.allclose(commute_bind(r, ab).outputs()[0], base, atol=1e-12)


def test_commute_bind_is_its_own_inverse():
    r, ab, _ = _recipe()
    once = commute_bind(r, ab)
    twice = commute_bind(once, ab)
    assert twice._ops[3] == r._ops[3]                      # the op is literally restored
    assert np.allclose(twice.outputs()[0], r.outputs()[0], atol=1e-12)


def test_commute_bind_on_non_bind_raises():
    r, _, bun = _recipe()
    try:
        commute_bind(r, bun)                               # bun is a bundle, not a bind
        assert False, "should have raised"
    except ValueError:
        pass


# ---- reorder_members -----------------------------------------------------------------------------
def test_reorder_members_preserves_the_realized_vector():
    r, _, bun = _recipe()
    base = r.outputs()[1]
    assert np.allclose(reorder_members(r, bun, [2, 0, 1]).outputs()[1], base, atol=1e-12)


def test_reorder_members_inverts_by_the_inverse_perm():
    r, _, bun = _recipe()
    base = r.outputs()[1]
    perm = [1, 2, 0]
    reordered = reorder_members(r, bun, perm)
    inv = [perm.index(i) for i in range(len(perm))]
    assert np.allclose(reorder_members(reordered, bun, inv).outputs()[1], base, atol=1e-12)


def test_reorder_members_rejects_a_non_permutation():
    r, _, bun = _recipe()
    try:
        reorder_members(r, bun, [0, 0, 1])                 # not a permutation
        assert False
    except ValueError:
        pass


# ---- substitute_atom (the vertex-move analogue) ------------------------------------------------
def test_substitute_atom_changes_the_result():
    r, ab, _ = _recipe()
    base = r.outputs()[0]
    assert not np.allclose(substitute_atom(r, 0, "z").outputs()[0], base, atol=1e-6)


def test_substitute_atom_reverses_exactly():
    r, ab, _ = _recipe()
    base = r.outputs()[0]
    swapped = substitute_atom(r, 0, "z")
    assert np.allclose(substitute_atom(swapped, 0, "a").outputs()[0], base, atol=1e-12)


# ---- general guarantees ------------------------------------------------------------------------
def test_edits_keep_the_recipe_valid():
    r, ab, bun = _recipe()
    for edited in (commute_bind(r, ab), reorder_members(r, bun, [2, 1, 0]), substitute_atom(r, 0, "q")):
        assert validate(edited)[0]


def test_edits_do_not_mutate_the_original():
    r, ab, bun = _recipe()
    before = [op if op[0] != "bundle" else ("bundle", list(op[1])) for op in r._ops]
    commute_bind(r, ab); reorder_members(r, bun, [2, 0, 1]); substitute_atom(r, 0, "z")
    after = [op if op[0] != "bundle" else ("bundle", list(op[1])) for op in r._ops]
    assert before == after                                 # the original recipe is untouched


def test_edits_are_deterministic():
    r, ab, _ = _recipe()
    assert np.array_equal(commute_bind(r, ab).outputs()[0], commute_bind(r, ab).outputs()[0])
