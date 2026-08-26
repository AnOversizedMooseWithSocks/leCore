"""A small structure-description language that lowers to the recipe IR (ISA-7 -- the top of the assembly tower).

SCOPE (the kept boundary): this is ONE domain -- *structure description*, not a general-purpose language. There
are no variables, no control flow, no user-defined functions: just a declarative surface for composing a
structure out of atoms, the base binds, and the ISA-6 templates. The whole point of ISA-7 is that the typed
unification (program = tree = scene = record = one StructureRecipe) IS the intermediate representation this
language targets; the language is just a readable surface that lowers to it. A general language is easy to
over-scope, so we deliberately do not build one.

Surface syntax is S-expressions:
    a                         -- a bare symbol is an atom
    (bind a b)                -- the base binds / bundle / permute lower to the matching recipe ops
    (bundle (pair a) b)       -- forms nest; sub-expressions compile to handles
    (record name moose)       -- a template call (ISA-6): the args fill the template's holes, inlined hygienically
    (permute a 1)             -- permute takes an integer shift

`compile_spec` lowers a parsed spec to a StructureRecipe (replayable, bit-exact). `realize_spec` parses,
compiles, and materialises the vector. The surface round-trips: parse(unparse(ast)) == ast.
"""

import numpy as np

from holographic.misc.holographic_recipe import StructureRecipe
from holographic.simulation_and_physics.holographic_template import STARTER_LIBRARY, _Builder

# The base forms the language understands directly; everything else is looked up in the template library.
_BASE_FORMS = ("bind", "bundle", "permute")


# ---- parser: text <-> AST (S-expressions) --------------------------------------------------------------------
def _tokenize(text):
    return text.replace("(", " ( ").replace(")", " ) ").split()


def _parse_tokens(toks, i):
    """Parse one node starting at toks[i]; return (node, next_index). A node is a str (atom) or a list (form)."""
    if i >= len(toks):
        raise ValueError("unexpected end of input")
    tok = toks[i]
    if tok == "(":
        i += 1
        node = []
        while i < len(toks) and toks[i] != ")":
            child, i = _parse_tokens(toks, i)
            node.append(child)
        if i >= len(toks):
            raise ValueError("missing closing )")
        return node, i + 1                                    # skip the ")"
    if tok == ")":
        raise ValueError("unexpected )")
    return tok, i + 1                                         # a bare symbol


def parse(text):
    """Parse a structure spec into an AST (nested lists; a bare symbol is a str)."""
    toks = _tokenize(text)
    node, i = _parse_tokens(toks, 0)
    if i != len(toks):
        raise ValueError(f"trailing tokens after a complete expression: {toks[i:]}")
    return node


def unparse(ast):
    """Render an AST back to surface text (for the round-trip)."""
    if isinstance(ast, str):
        return ast
    return "(" + " ".join(unparse(x) for x in ast) + ")"


# ---- compiler: AST -> StructureRecipe (lower to the IR) ------------------------------------------------------
def _compile_node(node, r):
    """Emit `node` into recipe `r`; return the result handle. Atoms and the base binds lower to recipe ops; a
    template call fills the template's holes with the compiled sub-expressions and inlines its build hygienically."""
    if isinstance(node, str):
        return r.atom(node)                                  # a bare symbol -> an atom
    if not node:
        raise ValueError("empty form ()")
    head = node[0]
    if not isinstance(head, str):
        raise ValueError(f"form head must be a symbol, got {head!r}")

    if head == "bind":
        if len(node) != 3:
            raise ValueError("bind takes exactly 2 arguments")
        return r.bind(_compile_node(node[1], r), _compile_node(node[2], r))
    if head == "bundle":
        if len(node) < 2:
            raise ValueError("bundle takes 1+ arguments")
        return r.bundle([_compile_node(x, r) for x in node[1:]])
    if head == "permute":
        if len(node) != 3:
            raise ValueError("permute takes an expression and an integer shift")
        return r.permute(_compile_node(node[1], r), int(node[2]))

    # otherwise it must be a template call (ISA-6). The macro layer becomes language constructs.
    if head in STARTER_LIBRARY:
        tmpl = STARTER_LIBRARY[head]
        argv = node[1:]
        if len(argv) != len(tmpl.params):
            raise ValueError(f"template {head!r} expects {tmpl.params}, got {len(argv)} args")
        holes = {p: _compile_node(arg, r) for p, arg in zip(tmpl.params, argv)}  # holes are compiled sub-exprs
        t = _Builder(r, tmpl.name, holes, hygienic=True)     # internal atoms namespaced -> no capture
        return tmpl._build(t)

    raise ValueError(f"unknown form {head!r}; base forms are {_BASE_FORMS} and templates {sorted(STARTER_LIBRARY)}")


def compile_spec(text_or_ast, dim, seed=0):
    """Lower a structure spec (text or AST) to a StructureRecipe with its output marked."""
    ast = parse(text_or_ast) if isinstance(text_or_ast, str) else text_or_ast
    r = StructureRecipe(dim, seed)
    out = _compile_node(ast, r)
    r.mark_output(out)
    return r


def realize_spec(text_or_ast, dim, seed=0):
    """Parse, compile, and materialise the structure vector (bit-exact for a given spec/dim/seed)."""
    r = compile_spec(text_or_ast, dim, seed)
    return r.get(r._outputs[-1])


def _selftest():
    from holographic.agents_and_reasoning.holographic_ai import bind, bundle, cosine, derived_atom, unbind
    dim, seed = 1024, 0

    # surface round-trip: parse(unparse(ast)) == ast
    for spec in ["a", "(bind a b)", "(bundle (pair a) (record name moose))", "(permute a 1)"]:
        assert unparse(parse(spec)) == spec.replace("  ", " "), spec

    # correctness: (bind a b) realizes EXACTLY to bind(atom a, atom b)
    v = realize_spec("(bind a b)", dim, seed)
    expect = bind(derived_atom(seed, "a", dim), derived_atom(seed, "b", dim))
    assert np.array_equal(v, expect), "bind form did not lower correctly"

    # bit-exact determinism: same spec -> same vector
    assert np.array_equal(realize_spec("(bundle a b c)", dim, seed),
                          realize_spec("(bundle a b c)", dim, seed))

    # a template form matches the ISA-6 template instantiated directly (the layers agree)
    lang_rec = realize_spec("(record name moose)", dim, seed)
    tmpl_rec = STARTER_LIBRARY["record"].build_vector(dim, seed, key="name", val="moose")
    assert np.array_equal(lang_rec, tmpl_rec), "language record != template record"

    # the structure is meaningful: a compiled pair recovers its value via the (namespaced) role
    pv = realize_spec("(pair alpha)", dim, seed)
    role = STARTER_LIBRARY["pair"].role_atom(dim, seed, "role")
    assert cosine(unbind(pv, role), derived_atom(seed, "alpha", dim)) > 0.99

    print("holographic_lang selftest: ok (structure language lowers to the recipe IR, round-trips, bit-exact)")


if __name__ == "__main__":
    _selftest()
