"""Tests for the structure-description language (ISA-7): an S-expression surface that lowers to the recipe IR.
Round-trips, lowers correctly and bit-exact, its template forms agree with ISA-6, and it is scoped to one
domain (structure description) -- unknown forms are an error, not silently a general language."""

import numpy as np
import pytest

from holographic.misc.holographic_lang import parse, unparse, compile_spec, realize_spec
from holographic.simulation_and_physics.holographic_template import STARTER_LIBRARY
from holographic.agents_and_reasoning.holographic_ai import bind, bundle, permute, cosine, unbind, derived_atom

DIM, SEED = 1024, 0


def test_surface_round_trips():
    for spec in ["a", "(bind a b)", "(bundle (pair a) (record name moose))", "(permute a 1)"]:
        assert unparse(parse(spec)) == spec                  # parse o unparse is the identity on the surface


def test_base_forms_lower_correctly():
    # (bind a b) realizes EXACTLY to bind(atom a, atom b); same for bundle and permute
    a, b, c = (derived_atom(SEED, n, DIM) for n in ("a", "b", "c"))
    assert np.array_equal(realize_spec("(bind a b)", DIM, SEED), bind(a, b))
    assert np.array_equal(realize_spec("(bundle a b c)", DIM, SEED), bundle([a, b, c]))
    assert np.array_equal(realize_spec("(permute a 2)", DIM, SEED), permute(a, 2))


def test_realization_is_bit_exact_and_deterministic():
    spec = "(bundle (record k v) (pair x))"
    assert np.array_equal(realize_spec(spec, DIM, SEED), realize_spec(spec, DIM, SEED))


def test_template_forms_agree_with_isa6():
    # the language's (record ...) form is exactly the ISA-6 record template instantiated directly
    lang = realize_spec("(record name moose)", DIM, SEED)
    tmpl = STARTER_LIBRARY["record"].build_vector(DIM, SEED, key="name", val="moose")
    assert np.array_equal(lang, tmpl)


def test_nested_composition_compiles_and_is_meaningful():
    # a nested spec compiles to one recipe; a single-binding pair recovers its value via the role
    v = realize_spec("(pair alpha)", DIM, SEED)
    role = STARTER_LIBRARY["pair"].role_atom(DIM, SEED, "role")
    assert cosine(unbind(v, role), derived_atom(SEED, "alpha", DIM)) > 0.99


def test_compile_returns_a_replayable_recipe():
    r = compile_spec("(bind a b)", DIM, SEED)
    # the recipe replays bit-exact to the realized vector
    assert np.array_equal(r.get(r._outputs[-1]), realize_spec("(bind a b)", DIM, SEED))


def test_scope_boundary_unknown_form_is_an_error():
    # ISA-7 is scoped to structure description -- it is NOT a general language. An unknown head is an error,
    # not a no-op; bad arity is an error. (This is the kept negative: do not over-scope into a general language.)
    with pytest.raises(ValueError):
        realize_spec("(while a b)", DIM, SEED)               # no control flow
    with pytest.raises(ValueError):
        realize_spec("(bind a)", DIM, SEED)                  # wrong arity
    with pytest.raises(ValueError):
        realize_spec("(record name)", DIM, SEED)             # template arity enforced


def test_parser_rejects_malformed_input():
    for bad in ["(bind a b", "bind a b)", "()"]:
        with pytest.raises(ValueError):
            realize_spec(bad, DIM, SEED)
