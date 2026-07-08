"""B7 keystone -- ONE typed holographic structure.

The integration review found that the engine's four "structure" types -- a build recipe, an
assembled PROGRAM, an EML/expression TREE, and a composed SCENE -- are not four different things.
They are one directed acyclic graph of the SAME handful of primitives (atom, bind, bundle,
permute, superpose) replayed to a vector. StructureRecipe (holographic_recipe.py) already IS that
graph. This module makes the unification CONCRETE and MEASURED: adapters that express a program, a
tree, and a nested scene AS a StructureRecipe, each reproducing its source module's vector
BIT-EXACTLY. After this there is one structure type, one save format, one realize() path -- the
single object the decode/denoise (B8) and manifold-aware (B9) work can target instead of four
siloed encoders.

WHAT IS UNIFIED, AND WHAT IS NOT (kept honestly):
  * UNIFIED: the forward ENCODING -- how a program / tree / scene becomes a vector. Every one
    reduces to atom/bind/bundle/permute/superpose over either name-addressable derived atoms
    (the program's opcodes, the tree's roles, the scene's group keys) or raw leaves (a SceneCoder
    sub-scene is drawn from an rng codebook, so it rides along as a `raw` payload). That mixed case
    is honest: the compositional STRUCTURE unifies into named atoms; rng leaves stay raw -- the same
    constructed-vs-measured split the recipe store already draws.
  * NOT UNIFIED -- the SEMANTICS on top: a program's EXECUTION (the HoloMachine interpreter, CALL,
    the function library) and an EML node's scalar evaluation are layers ABOVE the encoding. The
    assembled program's *vector* is a pure bind/bundle graph and reproduces exactly; its runtime
    stays in holographic_machine. We demo a CALL-free program for exactly this reason.
  * NOT UNIFIED -- the INVERSE: decoding a foreign vector back into a structure is the resonator's
    job (holographic_sbc.decompose_structure), bounded by crosstalk, and is what B8/B9 push on. A
    structure here is a GENERATOR, not a parser.

Pure NumPy + holostuff spirit; deterministic; reuses StructureRecipe (no new structure class -- that
is the whole point: one type, not five).
"""

import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, bundle, derived_atom
from holographic.misc.holographic_recipe import StructureRecipe


# ---------------------------------------------------------------------------------------------
# Adapter 1: a PROGRAM (HoloMachine.assemble) -> a StructureRecipe, bit-exact.
# HoloMachine assembles  program = bundle_i( bind(pos(i), bundle[bind(OP, op_i), bind(ARG, arg_i)]) )
# entirely from derived_atom(seed, name) leaves, so a recipe over the SAME names + seed reproduces it.
# ---------------------------------------------------------------------------------------------
def program_to_recipe(machine, program):
    """Express an assembled HoloMachine program as a StructureRecipe (atoms by name, bind, bundle).

    Reproduces machine.assemble(program) BIT-EXACTLY. `program` is a list of (opcode, operand). CALL is
    out of scope here (it pulls a body from the runtime library, not the program's own structure)."""
    r = StructureRecipe(machine.dim, machine.seed)
    cache = {}

    def atom(name, unitary=False):                     # reuse one handle per atom -> a lean recipe
        key = (name, unitary)
        if key not in cache:
            cache[key] = r.atom(name, unitary=unitary)
        return cache[key]

    instrs = []
    for i, (op, arg) in enumerate(program):
        if op in ("CALL", "ITERATE", "REPEAT"):
            raise ValueError(f"{op} is runtime control flow, not plain program structure -- out of scope")
        if op == "APPLY":
            arg_name = "fac:" + arg                                          # APPLY's operand is a faculty name
        elif arg in machine.data_names:
            arg_name = "dat:" + arg                                          # value / IFMATCH operand is a data atom
        else:
            arg_name = "op:HALT"                                             # HALT's operand is don't-care
        # build the instruction exactly as HoloMachine._instr does: bundle[bind(OP, op), bind(ARG, arg)]
        op_term = r.bind(atom("role:OP", True), atom("op:" + op))
        arg_term = r.bind(atom("role:ARG", True), atom(arg_name))
        instr = r.bundle([op_term, arg_term])
        instrs.append(r.bind(atom(f"pos:{i}", True), instr))   # bind the instruction to its position atom
    out = r.bundle(instrs)
    r.mark_output(out)
    return r


# ---------------------------------------------------------------------------------------------
# Adapter 2: an EXPRESSION TREE (the EML-tree's holographic encoding, generalised) <-> a recipe.
# Node encoding (standard VSA tree): a leaf is a symbol atom; an internal node binds its operator to
# an OP role and each child (recursively encoded) to a positional arg role, then bundles them.
# encode_tree is the direct kernel form; tree_to_recipe is the same graph as a StructureRecipe.
# ---------------------------------------------------------------------------------------------
def encode_tree(dim, seed, node):
    """Direct (kernel) holographic encoding of an expression tree. A leaf is a str symbol; an internal
    node is a tuple (op_name, child0, child1, ...). Binary eml(x, y) is the depth-2 special case."""
    if isinstance(node, str):
        return derived_atom(seed, "sym:" + node, dim)
    op, *children = node
    parts = [bind(derived_atom(seed, "role:OP", dim, unitary=True), derived_atom(seed, "op:" + op, dim))]
    for i, ch in enumerate(children):
        role = derived_atom(seed, f"role:arg{i}", dim, unitary=True)
        parts.append(bind(role, encode_tree(dim, seed, ch)))
    return bundle(parts)


def tree_to_recipe(dim, seed, node):
    """Express the same expression tree as a StructureRecipe. Reproduces encode_tree BIT-EXACTLY."""
    r = StructureRecipe(dim, seed)
    cache = {}

    def atom(name, unitary=False):
        key = (name, unitary)
        if key not in cache:
            cache[key] = r.atom(name, unitary=unitary)
        return cache[key]

    def handle(n):
        if isinstance(n, str):
            return atom("sym:" + n)
        op, *children = n
        parts = [r.bind(atom("role:OP", True), atom("op:" + op))]
        for i, ch in enumerate(children):
            parts.append(r.bind(atom(f"role:arg{i}", True), handle(ch)))
        return r.bundle(parts)

    r.mark_output(handle(node))
    return r


# ---------------------------------------------------------------------------------------------
# Adapter 3: a nested SCENE (UnifiedMind.compose_nested) -> a recipe, bit-exact.
# compose_nested does  sum_k bind(group_role(k), sub_scene_k).  Group roles are derived atoms
# (name-addressable); each sub_scene_k is a SceneCoder vector drawn from an rng codebook, so it
# rides as a `raw` leaf. The superposition is a raw np.sum -> the recipe's `superpose` op.
# ---------------------------------------------------------------------------------------------
def nested_scene_to_recipe(mind, groups):
    """Express mind.compose_nested(groups) as a StructureRecipe. Reproduces it BIT-EXACTLY. The group
    structure becomes named atoms + bind + superpose; the rng-drawn sub-scenes ride as raw leaves."""
    gr = mind._group_roles()
    sc = mind.scene()
    r = StructureRecipe(mind.dim, mind.seed)
    parts = []
    for k, tags in groups.items():
        role_vec = gr.get(str(k))                      # derived group-role vector (name-addressable)
        sub = sc.encode_scene(tags)                    # rng-codebook sub-scene -> a raw leaf
        parts.append(r.bind(r.raw(role_vec), r.raw(sub)))
    r.mark_output(r.superpose(parts))
    return r


# ---------------------------------------------------------------------------------------------
# A small equivalence harness: realize an adapter recipe and compare to the source module's vector.
# ---------------------------------------------------------------------------------------------
def max_abs_diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def op_kinds(recipe):
    """The set of distinct primitive op kinds a recipe uses -- the proof that a structure collapses to
    one small alphabet of operations regardless of which 'type' it came from."""
    return {op[0] for op in recipe._ops}
