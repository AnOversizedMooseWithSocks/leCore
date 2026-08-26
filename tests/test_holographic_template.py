"""Tests for parameterized recipe templates (ISA-6): a template instantiated with different arguments produces
distinct, BIT-EXACT structures, the starter library exists, and the fresh-atom discipline prevents capture."""

import numpy as np

from holographic.simulation_and_physics.holographic_template import STARTER_LIBRARY, RecipeTemplate, _UnhygienicTemplate, _pair, TMPL_NS
from holographic.agents_and_reasoning.holographic_ai import unbind, cosine, derived_atom

DIM, SEED = 1024, 0


def test_instantiation_is_bit_exact_and_deterministic():
    pair = STARTER_LIBRARY["pair"]
    v1 = pair.build_vector(DIM, SEED, x="a")
    v2 = pair.build_vector(DIM, SEED, x="a")
    assert np.array_equal(v1, v2)                            # the recipe replays bit-for-bit


def test_distinct_arguments_give_distinct_structures():
    pair = STARTER_LIBRARY["pair"]
    va = pair.build_vector(DIM, SEED, x="a")
    vb = pair.build_vector(DIM, SEED, x="b")
    assert cosine(va, vb) < 0.5                              # pair(a) and pair(b) are different structures


def test_single_binding_template_recovers_exactly():
    # pair = bind(unitary role, value); unbinding the role recovers the value exactly
    pair = STARTER_LIBRARY["pair"]
    v = pair.build_vector(DIM, SEED, x="a")
    role = pair.role_atom(DIM, SEED, "role")
    assert cosine(unbind(v, role), derived_atom(SEED, "a", DIM)) > 0.99


def test_record_fields_are_separable():
    # a two-field record; each field is recoverable by unbinding its role (approximate -- a 2-item bundle --
    # but cleanly separable: the right value wins by a wide margin over the wrong one)
    rec = STARTER_LIBRARY["record"]
    v = rec.build_vector(DIM, SEED, key="name", val="moose")
    VAL = rec.role_atom(DIM, SEED, "VAL")
    got = unbind(v, VAL)
    assert cosine(got, derived_atom(SEED, "moose", DIM)) > cosine(got, derived_atom(SEED, "name", DIM)) + 0.3


def test_ordered_pair_is_order_sensitive():
    op = STARTER_LIBRARY["ordered_pair"]
    ab = op.build_vector(DIM, SEED, a="x", b="y")
    ba = op.build_vector(DIM, SEED, a="y", b="x")
    assert cosine(ab, ba) < 0.5                              # position is encoded -> (x,y) != (y,x)


def test_starter_library_exists():
    assert {"pair", "record", "ordered_pair"} <= set(STARTER_LIBRARY)
    for t in STARTER_LIBRARY.values():
        assert isinstance(t, RecipeTemplate) and t.params


def test_hygiene_prevents_capture_kept_negative():
    # THE KEPT NEGATIVE: a template-internal atom of the same NAME and kind as a caller atom would be the SAME
    # vector (capture). The discipline namespaces internal atoms under TMPL_NS, so no caller bare name collides.
    pair = STARTER_LIBRARY["pair"]
    caller = derived_atom(SEED, "role", DIM, unitary=True)
    # hygienic: the internal "role" atom lives in the reserved namespace -> disjoint from the caller's "role"
    assert cosine(pair.role_atom(DIM, SEED, "role"), caller) < 0.1
    # unhygienic (tests only): bare internal name == caller name -> capture (the same vector)
    bad = _UnhygienicTemplate("pair", ["x"], _pair)
    assert cosine(bad.role_atom(DIM, SEED, "role"), caller) > 0.99


def test_namespace_prefix_is_reserved():
    # the discipline is just a reserved naming convention -- the prefix is on the internal atom name
    pair = STARTER_LIBRARY["pair"]
    a_namespaced = pair.role_atom(DIM, SEED, "role")
    a_bare = derived_atom(SEED, TMPL_NS + "pair:role", DIM, unitary=True)
    assert np.array_equal(a_namespaced, a_bare)              # internal atom is exactly the namespaced derived atom
