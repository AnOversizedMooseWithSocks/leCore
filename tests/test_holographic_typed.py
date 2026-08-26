"""Tests for the B7 keystone: program / expression-tree / nested-scene all reduce to ONE
StructureRecipe, bit-exactly, and UnifiedMind speaks that one type directly."""

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import cosine
from holographic.agents_and_reasoning.holographic_machine import HoloMachine
from holographic.misc.holographic_unified import UnifiedMind
from holographic.misc.holographic_recipe import StructureRecipe
from holographic.misc.holographic_typed import program_to_recipe, encode_tree, tree_to_recipe, nested_scene_to_recipe, max_abs_diff, op_kinds

ALLOWED = {"atom", "raw", "bind", "bundle", "permute", "superpose", "normalize"}
SCENE = {"g1": [{"colour": "red", "shape": "circle", "texture": "smooth"},
                {"colour": "green", "shape": "triangle", "texture": "busy"}],
         "g2": [{"colour": "cyan", "shape": "line", "texture": "vertical"}]}


def test_program_reduces_to_recipe_bit_exact():
    m = HoloMachine(dim=2048, seed=7)
    prog = [("LOAD", "a"), ("BIND", "b"), ("BUNDLE", "c"), ("PERMUTE", "d"), ("HALT", "a")]
    direct = m.assemble(prog)
    r = program_to_recipe(m, prog)
    via = r.get(r._outputs[0])
    assert cosine(direct, via) > 0.999999 and max_abs_diff(direct, via) == 0.0   # fully name-addressable
    assert op_kinds(r) <= ALLOWED


def test_expression_tree_recipe_matches_direct_kernel():
    tree = ("eml", ("mul", "x", "two"), ("ln", "y"))            # depth-3 expression
    d = encode_tree(2048, 5, tree)
    r = tree_to_recipe(2048, 5, tree)
    via = r.get(r._outputs[0])
    assert cosine(d, via) > 0.999999 and max_abs_diff(d, via) == 0.0
    assert op_kinds(r) <= ALLOWED


def test_nested_scene_reduces_to_recipe_bit_exact():
    mind = UnifiedMind(dim=1024, seed=3)
    direct = mind.compose_nested(SCENE)
    r = nested_scene_to_recipe(mind, SCENE)
    via = r.get(r._outputs[0])
    assert cosine(direct, via) > 0.999999 and max_abs_diff(direct, via) < 1e-9   # raw leaves -> ~float64
    assert op_kinds(r) <= ALLOWED
    assert "superpose" in op_kinds(r) and "raw" in op_kinds(r)                    # rng leaves ride as raw


def test_all_three_types_share_one_small_alphabet():
    m = HoloMachine(dim=1024, seed=7)
    rp = program_to_recipe(m, [("LOAD", "a"), ("HALT", "a")])
    rt = tree_to_recipe(1024, 5, ("op", "x", "y"))
    rs = nested_scene_to_recipe(UnifiedMind(dim=1024, seed=3), SCENE)
    union = op_kinds(rp) | op_kinds(rt) | op_kinds(rs)
    assert union <= ALLOWED and len(union) <= 6                                   # one structure, one alphabet


def test_call_is_out_of_scope():
    m = HoloMachine(dim=512, seed=7)
    try:
        program_to_recipe(m, [("CALL", "f")])                                     # runtime, not structure
        assert False, "CALL should be rejected"
    except ValueError:
        pass


def test_unified_mind_speaks_the_typed_structure():
    mind = UnifiedMind(dim=1024, seed=3)
    r = mind.typed_structure()
    assert isinstance(r, StructureRecipe) and r.dim == mind.dim and r.seed == mind.seed
    a = r.atom("x"); b = r.atom("y"); r.mark_output(r.bind(a, b))
    assert mind.realize(r).shape == (mind.dim,)
    # tree_structure and nested_scene_structure realize to the same vectors as the source ops
    rt = mind.tree_structure(("eml", "x", ("ln", "y")))
    assert cosine(mind.realize(rt), encode_tree(mind.dim, mind.seed, ("eml", "x", ("ln", "y")))) > 0.999999
    rs = mind.nested_scene_structure(SCENE)
    assert cosine(mind.realize(rs), mind.compose_nested(SCENE)) > 0.999999


def test_superpose_constructed_roundtrips_bit_exact():
    r = StructureRecipe(256, 1)
    a = r.atom("a"); b = r.atom("b"); r.mark_output(r.superpose([a, b]))
    rt = StructureRecipe.from_dict(r.to_dict())
    assert max_abs_diff(r.get(r._outputs[0]), rt.get(rt._outputs[0])) == 0.0      # all-atom: truly bit-exact
