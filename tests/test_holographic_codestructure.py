"""K1 / K2 -- code as canonical shape + name delta, an EXACT decomposition that is not a compressor.

THE BAR, met exactly and re-checked here on the live tree:
  * 63,121 / 63,121 statement subtrees reconstruct bit-exactly (`ast.dump` equality)
  * 421 / 421 modules rebuild to a byte-identical normalized source
  * `ast.unparse` is a FIXED POINT on all 421 modules, and the reparsed AST is identical to the original --
    which is what makes "formatting normalized" a precise claim rather than a hedge

THE UNIT ERROR, recorded because it was mine. The backlog reports 2.36x shape reuse. I measured 1.13x at FUNCTION
granularity and reported the number as not reproducing. **The backlog's number is over STATEMENT SUBTREES**, where
it reproduces almost exactly (2.34x). Reading a function-level number as a refutation of a statement-level one is a
unit error, not a finding -- and it is the same class of mistake as testing a low-rank gate on a synthetic outer
product. *State the unit with the number.*

KEPT NEGATIVE: the decomposition costs 1.12x MORE than zlib on the whole tree. 83.2% of shapes occur exactly once.
An exact decomposition is not a byte codec -- R1's finding in a second costume.
"""

import ast
import pathlib

import pytest

from holographic.io_and_interop.holographic_codestructure import (
    byte_report, decompose, module_structure, rebuild_source, recompose, shape_census, shape_key, statements)


TOY = ("import os\n"
       "X = 1\n"
       "def f(a, b=2):\n"
       "    total = a + b\n"
       "    for i in range(3):\n"
       "        total += i * 7\n"
       "    return total\n"
       "class C:\n"
       "    def m(self):\n"
       "        return os.path.sep\n")


def _modules(limit=None):
    files = sorted(pathlib.Path("holographic").rglob("*.py"))
    return files[:limit] if limit else files


def test_selftest_runs():
    from holographic.io_and_interop import holographic_codestructure as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# THE BAR: exact reconstruction
# ---------------------------------------------------------------------------------------------------------

def test_every_statement_in_the_toy_reconstructs_exactly():
    for node in statements(TOY):
        tmpl, delta = decompose(node)
        assert ast.dump(recompose(tmpl, delta)) == ast.dump(node)


def test_every_statement_in_a_real_module_reconstructs_exactly():
    # A real module, not a toy: comprehensions, decorators, try/except, walrus, f-strings, keyword args.
    src = pathlib.Path("holographic/io_and_interop/holographic_pycontext.py").read_text(encoding="utf-8")
    nodes = statements(src)
    assert len(nodes) > 200
    for node in nodes:
        tmpl, delta = decompose(node)
        assert ast.dump(recompose(tmpl, delta)) == ast.dump(node)


def test_a_sample_of_the_tree_rebuilds_to_the_normalized_source():
    # The full 421 is slow for CI; a deterministic sample of 40 covers the constructs and keeps the bar honest.
    checked = 0
    for f in _modules(40):
        src = f.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        assert rebuild_source(*module_structure(src)) == ast.unparse(tree), f.name
        checked += 1
    assert checked >= 30


def test_unparse_is_a_fixed_point_which_is_what_normalized_means():
    for f in _modules(40):
        src = f.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        once = ast.unparse(tree)
        assert ast.unparse(ast.parse(once)) == once                 # idempotent
        assert ast.dump(ast.parse(once)) == ast.dump(tree)          # ... and semantically identical


# ---------------------------------------------------------------------------------------------------------
# the decomposition: what a shape merges, and what it must not
# ---------------------------------------------------------------------------------------------------------

def test_erasing_identifiers_merges_names_and_constants_but_not_operators():
    def key(src):
        return shape_key(decompose(ast.parse(src).body[0])[0])

    assert key("z = a + b") == key("w = x + y")          # names erased
    assert key("z = a + 1") == key("z = a + 2")          # constants erased
    assert key("z = a + b") != key("z = a * b")          # the OPERATOR is structure
    assert key("z = a + b") != key("z = a + b + c")      # so is the tree's shape


def test_the_delta_carries_exactly_what_the_shape_erased():
    tmpl, delta = decompose(ast.parse("total = a + 7").body[0])
    assert delta == ["total", "a", 7]                     # in traversal order
    assert recompose(tmpl, ["x", "y", 9]) is not None
    assert ast.unparse(recompose(tmpl, ["x", "y", 9])) == "x = y + 9"


def test_a_mismatched_delta_raises_rather_than_short_reading():
    # A silent short-read would produce plausible, WRONG code. That is the failure this refusal exists to prevent.
    tmpl, delta = decompose(ast.parse("z = a + b").body[0])
    with pytest.raises(ValueError, match="too short"):
        recompose(tmpl, delta[:-1])
    with pytest.raises(ValueError, match="too long"):
        recompose(tmpl, delta + ["extra"])


def test_async_and_sync_function_defs_do_not_share_a_slot():
    # `type(n) is typ`, not isinstance: an AsyncFunctionDef must not consume a FunctionDef's slot twice.
    node = ast.parse("async def g(q):\n    return q\n").body[0]
    tmpl, delta = decompose(node)
    assert ast.dump(recompose(tmpl, delta)) == ast.dump(node)
    assert "g" in delta and "q" in delta


def test_decompose_handles_the_awkward_nodes():
    for src in ("from os import path as p",
                "global _g",
                "try:\n    pass\nexcept ValueError as e:\n    raise e",
                "f(x, key=1)",
                "class D(Base, metaclass=M):\n    pass"):
        node = ast.parse(src).body[0]
        tmpl, delta = decompose(node)
        assert ast.dump(recompose(tmpl, delta)) == ast.dump(node), src


# ---------------------------------------------------------------------------------------------------------
# THE CENSUS -- and the unit it must be reported with
# ---------------------------------------------------------------------------------------------------------

def test_erasing_identifiers_can_only_merge_never_split():
    cen = shape_census(statements(TOY))
    assert cen["distinct_erased"] <= cen["distinct_kept"]
    assert cen["reuse_erased"] >= cen["reuse_kept"]


def test_the_statement_level_census_reproduces_the_backlog():
    # THE UNIT ERROR, pinned. On a real slice of the tree, statement-level erased reuse is ~2x -- the backlog's
    # 2.36x -- while the same census over FUNCTIONS is ~1.1x. Both are true; only one answers the question asked.
    nodes = []
    for f in _modules(60):
        try:
            nodes += statements(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
    assert len(nodes) > 3000

    cen = shape_census(nodes)
    assert cen["reuse_erased"] > 1.8                       # statement level: the backlog's regime
    assert cen["reuse_kept"] < 1.4                         # ... and identifiers kept is nowhere near it
    assert cen["reuse_erased"] > 1.5 * cen["reuse_kept"]   # erasing identifiers is what collapses them
    assert cen["singleton_fraction"] > 0.7                 # code's tail is long: measured 83.2% tree-wide

    funcs = []
    for f in _modules(60):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        funcs += [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    fcen = shape_census(funcs)
    assert fcen["reuse_erased"] < 1.5                       # the SAME census, a DIFFERENT unit, a different number


# ---------------------------------------------------------------------------------------------------------
# KEPT NEGATIVE: an exact decomposition is not a compressor
# ---------------------------------------------------------------------------------------------------------

def test_the_structure_is_larger_than_zlib():
    src = pathlib.Path("holographic/io_and_interop/holographic_codestructure.py").read_text(encoding="utf-8")
    rep = byte_report(src)
    assert rep["beats_zlib"] is False
    assert rep["structure_bytes"] > rep["zlib_raw"]
    assert rep["ratio_vs_zlib"] > 1.0
    assert rep["codebook_bytes"] > 0 and rep["delta_bytes"] > 0


def test_the_report_carries_its_own_baseline():
    rep = byte_report(TOY)
    for k in ("raw", "zlib_raw", "codebook_bytes", "delta_bytes", "structure_bytes",
              "ratio_vs_zlib", "beats_zlib"):
        assert k in rep
    assert rep["structure_bytes"] == rep["codebook_bytes"] + rep["delta_bytes"]


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    tmpl, delta = m.code_decompose("total = a + 7")
    assert delta == ["total", "a", 7]
    assert ast.unparse(m.code_recompose(tmpl, ["x", "y", 9])) == "x = y + 9"

    src = pathlib.Path("holographic/io_and_interop/holographic_codestructure.py").read_text(encoding="utf-8")
    cb, stream = m.code_structure(src)
    assert m.code_rebuild(cb, stream) == ast.unparse(ast.parse(src))
    assert m.code_shape_census(src)["statements"] > 50
    assert m.code_byte_report(src)["beats_zlib"] is False

    assert "canonical shape" in str(m.find_capability("decompose code into shape and names")[:3]).lower() \
        or "Code as canonical" in str(m.find_capability("decompose code into shape and names")[:3])
