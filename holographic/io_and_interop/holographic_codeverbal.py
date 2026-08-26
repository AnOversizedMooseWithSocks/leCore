"""holographic_codeverbal.py -- deterministic AST -> English verbalizer (backlog C1).

leCore has no LLM and never will in core, so "explain this code" here means something stricter and more useful
than prose vibes: LAYERED English where every sentence is derived from a distinct static analysis and is
CHECKABLE against the source. Same code in, same English out, every time -- the selftest asserts exact sentences,
because a verbalizer whose output drifts is a verbalizer whose output cannot be trusted in a test.

The layers, each honest about what it can know:
  SIGNATURE   -- name, parameters, annotations, return: read straight off the FunctionDef.
  DATA FLOW   -- per-variable narration from def-use analysis: what each assignment is computed from, how many
                 times it is read afterwards, and whether it feeds the return. Pure `ast`, no execution.
  CONTROL FLOW-- a census: loops (kind and count), branches, return sites, early exits. Structure, not meaning.
  IDIOM       -- the ONLY layer allowed to speak of PURPOSE, and only on a catalog match. Matching reuses
                 codestructure.shape_key on the blanked FunctionDef template, so an idiom is recognized by its
                 SHAPE -- iq's rounded box with different half-extents is still iq's rounded box (names and
                 constants are identity slots and are blanked before hashing). An unmatched function yields the
                 exact sentence "Purpose: not recognized (no registered idiom matches this shape)." -- a
                 first-class answer, never a guess. Guessed intent is the failure mode this module exists to
                 refuse.

KEPT NEGATIVE (scope): shape matching is EXACT on the blanked template. A semantically identical kernel written
with reordered independent statements is a DIFFERENT shape and will honestly not match -- normalization to a
canonical statement order is future work (and risky: reordering floats is not meaning-preserving, the whole
lesson of the zig ReleaseFast negative). Stated, not papered over.

KEPT NEGATIVE (scope): the data-flow layer narrates straight-line and simple-loop code well; it does not build a
full CFG, so a variable reassigned in one branch of an `if` is narrated per-assignment, not per-path. The census
layer tells the reader how many branches there are, which is the honest cue that per-path reasoning applies.
"""

import ast
import textwrap

from holographic.io_and_interop.holographic_codestructure import decompose, shape_key

#: The idiom catalog: shape_key -> {name, purpose, citation}. Seeded below (C6 grows it); register_idiom adds.
_IDIOMS = {}

#: COMPOSITION recognition (C6): a second catalog keyed on the shape of a single primitive's DISTANCE EXPRESSION
#: (the return subgraph with locals inlined and names/constants blanked), so a min()/max() of known primitives is
#: recognized as "box union sphere" instead of "not recognized". The key difference from _IDIOMS: that matches a
#: whole FunctionDef; this matches a SUB-EXPRESSION, which is what a composition is made of. Same shape_key engine.
_PRIMITIVE_SHAPES = {}                                    # operand-shape-key -> {name}


def register_idiom(name, example_src, purpose, citation=""):
    """Register an idiom by EXAMPLE: the example's blanked FunctionDef shape becomes the recognizer.

    Registration is additive and self-verifying: the example must parse to exactly one function, and the call
    returns the shape key so a test can assert recognition immediately."""
    node = ast.parse(textwrap.dedent(example_src)).body[0]
    if not isinstance(node, ast.FunctionDef):
        raise ValueError("an idiom example must be a single function definition")
    tmpl, _delta = decompose(node)
    key = shape_key(tmpl)
    _IDIOMS[key] = {"name": name, "purpose": purpose, "citation": citation}
    return key


def _param_sentence(node):
    ps = []
    for a in node.args.args:
        ann = getattr(a.annotation, "id", None)
        ps.append("%s: %s" % (a.arg, ann) if ann else a.arg)
    ret = getattr(node.returns, "id", None)
    core = "Function `%s` takes %d parameter%s (%s)" % (
        node.name, len(ps), "" if len(ps) == 1 else "s", ", ".join(ps)) if ps else \
        "Function `%s` takes no parameters" % node.name
    return core + (" and returns %s." % ret if ret else ".")


def _names_in(node):
    """Data-source names under `node` -- EXCLUDING call targets: in `max(abs(x) - hx, 0.0)` the sources are
    `x` and `hx`; `max` and `abs` are operations, and narrating them as inputs would be a category error."""
    called = {n.func.id for n in ast.walk(node)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name)} - called)


def _dataflow_sentences(node):
    """One sentence per assignment, in source order: computed-from, read-count, feeds-return. Plus unused params."""
    params = [a.arg for a in node.args.args]
    assigns = []                                          # (target, sources) in source order
    for st in ast.walk(node):
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            assigns.append((st.targets[0].id, _names_in(st.value), st))
    ret_names = set()
    for st in ast.walk(node):
        if isinstance(st, ast.Return) and st.value is not None:
            ret_names.update(_names_in(st.value))
    # read counts: every Name in a Load context, excluding the defining assignment's own target slot
    reads = {}
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            reads[n.id] = reads.get(n.id, 0) + 1
    out = []
    for tgt, srcs, _st in assigns:
        srcs = [s for s in srcs if s != tgt]
        r = reads.get(tgt, 0)
        bits = "`%s` is computed from %s" % (tgt, ", ".join("`%s`" % s for s in srcs)) if srcs else \
               "`%s` is set to a constant" % tgt
        bits += "; read %d time%s" % (r, "" if r == 1 else "s")
        if tgt in ret_names:
            bits += "; feeds the return value"
        out.append(bits + ".")
    unused = [p for p in params if reads.get(p, 0) == 0]
    if unused:
        out.append("Parameter%s %s %s never read." % ("" if len(unused) == 1 else "s",
                   ", ".join("`%s`" % p for p in unused), "is" if len(unused) == 1 else "are"))
    return out


def _controlflow_sentence(node):
    fors = sum(isinstance(n, ast.For) for n in ast.walk(node))
    whiles = sum(isinstance(n, ast.While) for n in ast.walk(node))
    ifs = sum(isinstance(n, ast.If) for n in ast.walk(node))
    rets = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
    parts = []
    if fors or whiles:
        lp = []
        if fors:
            lp.append("%d for" % fors)
        if whiles:
            lp.append("%d while" % whiles)
        parts.append("%s loop%s" % (" + ".join(lp), "" if fors + whiles == 1 else "s"))
    if ifs:
        parts.append("%d branch%s" % (ifs, "" if ifs == 1 else "es"))
    body = "straight-line (no loops or branches)" if not parts else ", ".join(parts)
    last_is_return = bool(node.body) and isinstance(node.body[-1], ast.Return)
    early = len(rets) - (1 if last_is_return else 0)
    rtxt = "%d return site%s" % (len(rets), "" if len(rets) == 1 else "s")
    if early > 0:
        rtxt += " (%d early)" % early
    return "Control flow: %s; %s." % (body, rtxt)


#: boolean SDF operators, recognized by their exact expression shape (iq's canonical forms).
#:   min(a, b)          -> union         max(a, b)          -> intersection
#:   max(a, -(b))       -> subtraction (a minus b)
_BOOL_OPS = {"min": "union", "max": "intersection"}


def register_composition_form(name, distance_expr_src):
    """Seed the composition catalog by EXAMPLE: `distance_expr_src` is a single primitive's distance expression
    (e.g. 'sqrt(px*px + py*py + pz*pz) - r'), whose blanked shape becomes the operand recognizer. Returns the
    key. This is what lets a min() of two such shapes be read as 'union of <name> and <name>'."""
    expr = ast.parse(distance_expr_src.strip(), mode="eval").body
    tmpl, _d = decompose(expr)
    key = shape_key(tmpl)
    _PRIMITIVE_SHAPES[key] = {"name": name}
    return key


def _inline_locals(node):
    """Return a dict local_name -> defining AST expression, so a return that names distance locals can be
    expanded into one self-contained expression tree for shape matching. Straight-line only (the kernel grammar);
    a reassigned local is the last write wins, which the census layer already flags as branchy if it matters."""
    defs = {}
    for st in ast.walk(node):
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            defs[st.targets[0].id] = st.value
    return defs


class _Inliner(ast.NodeTransformer):
    """Replace Name loads with their defining expression, recursively, so an operand subgraph is self-contained
    and matches a standalone primitive's shape. Guards against cycles (there are none in straight-line code, but
    a depth cap keeps a pathological input from looping)."""

    def __init__(self, defs, depth=0):
        self.defs, self.depth = defs, depth

    def visit_Name(self, n):
        if isinstance(n.ctx, ast.Load) and n.id in self.defs and self.depth < 64:
            sub = copy_ast(self.defs[n.id])
            return _Inliner(self.defs, self.depth + 1).visit(sub)
        return n


def copy_ast(n):
    import copy as _copy
    return _copy.deepcopy(n)


def _match_primitive(expr, defs):
    """Inline an operand expression and match its blanked shape against the primitive catalog. Returns the name
    or None -- None is honest, never a guess."""
    inlined = _Inliner(defs).visit(copy_ast(expr))
    ast.fix_missing_locations(inlined)
    tmpl, _d = decompose(inlined)
    hit = _PRIMITIVE_SHAPES.get(shape_key(tmpl))
    return hit["name"] if hit else None


def _recognize_composition(node):
    """If the function's return is a boolean composition (nested min/max, possibly with negation for subtraction)
    of recognized primitives, describe it. Returns a sentence or None. This is the C6 composition layer: it reads
    a min(box, sphere) as 'union of a box and a sphere' where the whole-function idiom layer sees nothing."""
    ret = None
    for st in node.body:
        if isinstance(st, ast.Return):
            ret = st.value
    if ret is None:
        return None
    defs = _inline_locals(node)

    names, ops = [], set()

    def walk(e):
        # a call to min/max is a boolean op; recurse into its args. Anything else is an operand to match.
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id in _BOOL_OPS:
            # subtraction: max(a, -(b)) -- a unary-negated second arg turns intersection into subtract
            if e.func.id == "max" and len(e.args) == 2 and isinstance(e.args[1], ast.UnaryOp)                     and isinstance(e.args[1].op, ast.USub):
                ops.add("subtraction")
                walk(e.args[0])
                walk(e.args[1].operand)
                return
            ops.add(_BOOL_OPS[e.func.id])
            for a in e.args:
                walk(a)
            return
        nm = _match_primitive(e, defs)
        names.append(nm)

    walk(ret)
    if len(names) < 2 or any(n is None for n in names):
        return None                                       # not a composition, or an operand we don't recognize
    op_txt = " and ".join(sorted(ops)) if ops else "composition"
    parts = ", ".join(n if n else "an unrecognized shape" for n in names)
    return "Purpose (composition recognized): a %s of %d primitives -- %s." % (op_txt, len(names), parts)


def _idiom_sentence(node):
    tmpl, _delta = decompose(node)
    hit = _IDIOMS.get(shape_key(tmpl))
    if hit is not None:
        cite = " [%s]" % hit["citation"] if hit["citation"] else ""
        return "Purpose (idiom match: %s): %s%s" % (hit["name"], hit["purpose"], cite)
    comp = _recognize_composition(node)                   # C6: a min/max of known primitives is a real answer
    if comp is not None:
        return comp
    return "Purpose: not recognized (no registered idiom matches this shape)."


def verbalize_function(node):
    """Verbalize one FunctionDef into the four labeled layers. Returns {signature, dataflow, controlflow,
    idiom, text} -- `text` is the layers joined, one paragraph per layer, deterministic."""
    sig = _param_sentence(node)
    df = _dataflow_sentences(node)
    cf = _controlflow_sentence(node)
    idm = _idiom_sentence(node)
    text = "\n".join([sig, " ".join(df) if df else "No local variables.", cf, idm])
    return {"signature": sig, "dataflow": df, "controlflow": cf, "idiom": idm, "text": text}


def verbalize(src):
    """Verbalize source text: every function gets its four layers, plus a one-line module summary.

    Returns {summary, functions: [per-function dicts], text}. Deterministic: the selftest asserts exact
    sentences, which is the property that makes this usable inside other tests."""
    tree = ast.parse(textwrap.dedent(src))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    other = len(tree.body) - len(fns)
    summary = "Module with %d function%s%s." % (len(fns), "" if len(fns) == 1 else "s",
              (" and %d other top-level statement%s" % (other, "" if other == 1 else "s")) if other else "")
    out = [verbalize_function(f) for f in fns]
    return {"summary": summary, "functions": out,
            "text": "\n\n".join([summary] + [o["text"] for o in out])}


# -- C6 seed: idioms the repo already owns, registered by their canonical example ------------------------------
register_idiom(
    "iq rounded-box SDF",
    """
def sdf_round_box(px: float, py: float, pz: float, bx: float, by: float, bz: float, r: float) -> float:
    qx = max(abs(px) - bx, 0.0)
    qy = max(abs(py) - by, 0.0)
    qz = max(abs(pz) - bz, 0.0)
    outside = sqrt(qx * qx + qy * qy + qz * qz)
    inside = min(max(abs(px) - bx, max(abs(py) - by, abs(pz) - bz)), 0.0)
    return outside + inside - r
""",
    "signed distance to an axis-aligned box with rounded edges; negative inside, zero on the surface",
    "Quilez, distfunctions")

register_idiom(
    "sphere SDF",
    """
def sdf_sphere(px: float, py: float, pz: float, r: float) -> float:
    d = sqrt(px * px + py * py + pz * pz)
    return d - r
""",
    "signed distance to a sphere centred at the origin",
    "Quilez, distfunctions")

register_idiom(
    "linear interpolation (lerp)",
    "def lerp(a: float, b: float, t: float) -> float:\n    return a + (b - a) * t\n",
    "blends between two values; t=0 gives the first, t=1 the second",
    "")


# -- C6 composition seed: the same primitive shapes codecompose emits, keyed on their distance EXPRESSION so a
#    min()/max() of them is recognized as a union/intersection/subtraction of named shapes ---------------------
# sphere: sqrt(sx*sx + sy*sy + sz*sz) - r, where sx=px-cx etc. -- inlined, the shape is a sqrt-of-sum-of-squares
register_composition_form("a sphere",
                          "sqrt((px - 0.0) * (px - 0.0) + (py - 0.0) * (py - 0.0) + (pz - 0.0) * (pz - 0.0)) - 1.0")
# rounded box: qo + qi - r, the full iq rounded-box distance expression with its locals inlined
register_composition_form(
    "a rounded box",
    "sqrt(max(abs(px - 0.0) - 0.5, 0.0) * max(abs(px - 0.0) - 0.5, 0.0)"
    " + max(abs(py - 0.0) - 0.5, 0.0) * max(abs(py - 0.0) - 0.5, 0.0)"
    " + max(abs(pz - 0.0) - 0.5, 0.0) * max(abs(pz - 0.0) - 0.5, 0.0))"
    " + min(max(abs(px - 0.0) - 0.5, max(abs(py - 0.0) - 0.5, abs(pz - 0.0) - 0.5)), 0.0) - 0.05")
# plane: py - h
register_composition_form("a plane", "py - 0.0")
register_composition_form("a plane", "py - -0.0")        # negative height emits as a UnaryOp -- a distinct shape


def _selftest():
    """Exact-sentence regression traps -- determinism IS the contract.

    - the seeded round-box idiom is recognized even with DIFFERENT parameter names and constants (shape identity
      blanks both), and the data-flow layer narrates computed-from / read-count / feeds-return exactly;
    - an unrecognized function yields the exact 'not recognized' sentence -- never a guess;
    - an unused parameter is called out (Zig would refuse it at compile time; the verbalizer names it in English);
    - control-flow census counts a while loop and an early return.
    """
    renamed = """
def box_distance(x: float, y: float, z: float, hx: float, hy: float, hz: float, round_r: float) -> float:
    ax = max(abs(x) - hx, 0.0)
    ay = max(abs(y) - hy, 0.0)
    az = max(abs(z) - hz, 0.0)
    far = sqrt(ax * ax + ay * ay + az * az)
    near = min(max(abs(x) - hx, max(abs(y) - hy, abs(z) - hz)), 0.0)
    return far + near - round_r
"""
    v = verbalize(renamed)["functions"][0]
    assert v["idiom"].startswith("Purpose (idiom match: iq rounded-box SDF):"), v["idiom"]
    assert v["signature"] == ("Function `box_distance` takes 7 parameters (x: float, y: float, z: float, "
                              "hx: float, hy: float, hz: float, round_r: float) and returns float."), v["signature"]
    assert v["dataflow"][0] == "`ax` is computed from `hx`, `x`; read 2 times.", v["dataflow"][0]
    assert v["dataflow"][3] == "`far` is computed from `ax`, `ay`, `az`; read 1 time; feeds the return value."
    assert v["controlflow"] == "Control flow: straight-line (no loops or branches); 1 return site."

    unknown = "def mystery(a: float) -> float:\n    b = a * 3.0 + 1.0\n    return b * b\n"
    u = verbalize(unknown)["functions"][0]
    assert u["idiom"] == "Purpose: not recognized (no registered idiom matches this shape).", u["idiom"]

    dead = "def f(a: float, ghost: float) -> float:\n    return a * 2.0\n"
    d = verbalize(dead)["functions"][0]
    assert "Parameter `ghost` is never read." in " ".join(d["dataflow"]), d["dataflow"]

    loopy = ("def march(t: float) -> float:\n"
             "    s = 0\n"
             "    while s < 10:\n"
             "        if t > 5.0:\n"
             "            return t\n"
             "        s = s + 1\n"
             "    return t * 2.0\n")
    c = verbalize(loopy)["functions"][0]
    assert c["controlflow"] == "Control flow: 1 while loop, 1 branch; 2 return sites (1 early).", c["controlflow"]

    # C6 composition recognition: a min()/max() of registered primitive shapes is read as a named union/
    # intersection/subtraction -- the loop C3 opened. A single primitive is NOT a composition (needs >=2), and an
    # operand we do not know makes the whole thing honestly "not recognized" -- never a partial guess.
    union2 = ("def s(px: float, py: float, pz: float) -> float:\n"
              "    sx = px - 0.0\n    sy = py - 0.0\n    sz = pz - 0.0\n"
              "    d0 = sqrt(sx * sx + sy * sy + sz * sz) - 0.5\n    d1 = py - -0.3\n    return min(d0, d1)\n")
    ui = verbalize(union2)["functions"][0]["idiom"]
    assert ui == "Purpose (composition recognized): a union of 2 primitives -- a sphere, a plane.", ui

    sub2 = ("def s(px: float, py: float, pz: float) -> float:\n"
            "    sx = px - 0.0\n    sy = py - 0.0\n    sz = pz - 0.0\n"
            "    d0 = py - 0.0\n    d1 = sqrt(sx * sx + sy * sy + sz * sz) - 0.4\n    return max(d0, -(d1))\n")
    si = verbalize(sub2)["functions"][0]["idiom"]
    assert si == "Purpose (composition recognized): a subtraction of 2 primitives -- a plane, a sphere.", si

    # an unknown operand -> honest "not recognized", NOT a partial composition claim
    mystery = ("def s(px: float, py: float, pz: float) -> float:\n"
               "    sx = px - 0.0\n    d0 = sqrt(sx * sx) - 0.5\n    d1 = px * py + 3.0\n    return min(d0, d1)\n")
    mi = verbalize(mystery)["functions"][0]["idiom"]
    assert mi == "Purpose: not recognized (no registered idiom matches this shape).", mi

    print("OK: holographic_codeverbal self-test passed (round-box idiom recognized across renamed params and "
          "changed constants -- shape identity, exact sentences asserted; unknown shape yields the literal "
          "'not recognized' answer, never a guess; unused parameter named; loop/branch/early-return census "
          "exact; C6 COMPOSITION recognition reads min/max of registered primitives as a named union/subtraction, "
          "and one unknown operand collapses the whole answer to 'not recognized', never a partial guess)")


if __name__ == "__main__":
    _selftest()
