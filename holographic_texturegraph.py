"""holographic_texturegraph.py -- CMP1: a COMPOSABLE texture map graph (readable object tree + compose-time schema).

A texture map is a TREE of nodes, and a node is one of two things:
  * a LEAF   -- a number, a color, or a field (a callable that returns a value at a point), or
  * a MAP    -- an operation ('mix', 'multiply', 'over', ...) over TYPED child inputs, each of which may ITSELF be a
               Map. That recursion is the whole point: a map's input can be another map, so you get shader-style
               graphs of arbitrary depth from one small piece.

Sampling walks the tree: to sample a Map at a UV coordinate you first sample each child, then apply the op. The
tree stays a plain, readable Python object -- you can print it, inspect it, and evaluate it directly. The VSA
encoding is OPTIONAL and only used where it earns its keep (cache an evaluated graph, search a library of graphs,
blend two graphs): encode() lowers the tree to a hypervector via holographic_typed.encode_tree. We deliberately do
NOT force a deep tree into one vector -- that would hit the HRR capacity cliff; the object tree is the source of
truth, the vector is a derived, cached form.

THE DISCIPLINE IS THE TYPE SCHEMA. Every op declares its input slots and, per slot, which of the four kinds
{map, color, field, number} it accepts. That schema is checked at COMPOSE time -- when you build the node -- so a
malformed graph (a color where a weight belongs, a missing input, an unknown op) is REFUSED up front with a clear
message, rather than rendered wrong and debugged later. That is the difference between "composable" and
"composable correctly".

Reuses (no new dependencies): holographic_texturehome.Texture for leaf field sources (fbm / voronoi / synth),
holographic_fieldhome.Field to wrap a callable as a field, holographic_typed.encode_tree for the optional vector
form. Everything here is plain NumPy + stdlib.
"""
import numpy as np

# The four KINDS a node can be, named once so the schema and the error messages agree.
NUMBER, COLOR, FIELD, MAP = "number", "color", "field", "map"

# Handy kind-sets for the op schema below.
_ANY = (MAP, COLOR, FIELD, NUMBER)          # any value at all
_SCALARISH = (MAP, FIELD, NUMBER)           # a value used as a WEIGHT or amount -- a color makes no sense here


# ------------------------------------------------------------------------------------------------------------
# NODES.  Three concrete node types: Const (a number or a color), FieldLeaf (a field callable), and Map (an op).
# Each knows its KIND (for schema checks) and how to SAMPLE itself at a UV/point.
# ------------------------------------------------------------------------------------------------------------

class Node:
    """Base class: a node knows its `kind` and can `sample(uv)` itself. Leaves override sample to return a value;
    a Map overrides it to evaluate its children and apply its op."""
    kind = None

    def sample(self, uv):
        raise NotImplementedError

    def describe(self, indent=0):
        return "  " * indent + repr(self)


class Const(Node):
    """A constant LEAF: a plain number (kind='number'), a color as a length-3/4 sequence (kind='color'), or a color
    NAME string ('red', 'blue', ...) which is resolved to its rgb (kind='color'), so the same colour vocabulary the
    scene system uses works here too. Sampling ignores the coordinate and returns the value; a constant is itself
    everywhere."""

    def __init__(self, value):
        if isinstance(value, str):                               # a colour NAME -> its rgb, so Const('red') just works
            rgb = _named_color(value)
            if rgb is None:
                raise TypeError("Const got the string %r, which isn't a known colour name. Pass a number (0.5), an "
                                "rgb list ([1,0,0]), or a known colour name (%s)." % (value, ", ".join(_color_names())))
            self.value = np.asarray(rgb, dtype=float)
            self.kind = COLOR
            return
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            self.kind = NUMBER
        elif arr.ndim == 1 and arr.shape[0] in (3, 4):
            self.kind = COLOR
        else:
            raise TypeError("Const takes a number, a length-3/4 rgb(a) colour, or a colour name; got shape %r"
                            % (arr.shape,))
        self.value = arr

    def sample(self, uv):
        return self.value

    def __repr__(self):
        return "Const(%s)" % ("color" if self.kind == COLOR else float(self.value))


def _named_color(name):
    """Resolve a colour NAME to its rgb via the scene system's colour vocabulary (so the names match everywhere), or
    None if it isn't a known colour. Imported lazily to keep this module light."""
    try:
        from holographic_semantic import COLORS
    except Exception:
        return None
    return COLORS.get(name.lower())


def _color_names():
    try:
        from holographic_semantic import COLORS
        return sorted(COLORS)
    except Exception:
        return []


class FieldLeaf(Node):
    """A field LEAF (kind='field'): wraps a field -- either a holographic_fieldhome.Field, or a raw callable
    `f(points (N,D)) -> values (N,) or (N,C)` such as the ones Texture.fbm()/voronoi() return. Sampling evaluates
    the field at the single query point and returns that one value (a scalar, or a colour if the field is C-valued)."""

    def __init__(self, field, name="field"):
        # accept a fieldhome.Field (has .sample) or a bare callable; store one uniform "sample N points" callable
        if hasattr(field, "sample"):
            self._eval = field.sample
        elif callable(field):
            self._eval = field
        else:
            raise TypeError("FieldLeaf needs a callable field or a Field object, got %r" % type(field))
        self.name = name
        self.kind = FIELD

    def sample(self, uv):
        pts = np.atleast_2d(np.asarray(uv, dtype=float))          # one point -> shape (1, D)
        out = np.asarray(self._eval(pts))
        return out[0]                                             # unwrap the single result

    def __repr__(self):
        return "FieldLeaf(%s)" % self.name


# ------------------------------------------------------------------------------------------------------------
# THE OP TABLE.  Each op has (1) a SCHEMA -- its slot names and the kinds each slot accepts (checked at compose
# time) -- and (2) an EVAL -- how to combine the already-sampled child values. Kept as two small, readable dicts
# so adding an op is: add one schema line and one eval line. Values may be scalars OR colours (NumPy broadcasts).
# ------------------------------------------------------------------------------------------------------------

def _s(x):
    """Coerce a sampled weight/amount to a plain float (a 1-element array from a field becomes its scalar)."""
    a = np.asarray(x, dtype=float)
    return float(a) if a.ndim == 0 else float(a.reshape(-1)[0])


OP_SCHEMA = {
    # op         slot -> allowed kinds
    "mix":      {"a": _ANY, "b": _ANY, "t": _SCALARISH},         # linear blend: (1-t)*a + t*b
    "add":      {"a": _ANY, "b": _ANY},                          # a + b
    "multiply": {"a": _ANY, "b": _ANY},                          # a * b (modulate)
    "scale":    {"x": _ANY, "k": _SCALARISH},                    # x * k
    "over":     {"a": _ANY, "b": _ANY, "alpha": _SCALARISH},     # alpha-composite a over b
    "remap":    {"x": _SCALARISH, "lo": _SCALARISH, "hi": _SCALARISH},   # lo + (hi-lo)*x  (x in 0..1)
    "min":      {"a": _ANY, "b": _ANY},
    "max":      {"a": _ANY, "b": _ANY},
    "clamp":    {"x": _ANY, "lo": _SCALARISH, "hi": _SCALARISH},     # keep x within [lo, hi]
    "saturate": {"x": _ANY},                                         # the common case: clamp to [0, 1]
}

OP_EVAL = {
    "mix":      lambda v: (1.0 - _s(v["t"])) * v["a"] + _s(v["t"]) * v["b"],
    "add":      lambda v: v["a"] + v["b"],
    "multiply": lambda v: v["a"] * v["b"],
    "scale":    lambda v: v["x"] * _s(v["k"]),
    "over":     lambda v: _s(v["alpha"]) * v["a"] + (1.0 - _s(v["alpha"])) * v["b"],
    "remap":    lambda v: _s(v["lo"]) + (_s(v["hi"]) - _s(v["lo"])) * _s(v["x"]),
    "min":      lambda v: np.minimum(v["a"], v["b"]),
    "max":      lambda v: np.maximum(v["a"], v["b"]),
    "clamp":    lambda v: np.clip(v["x"], _s(v["lo"]), _s(v["hi"])),
    "saturate": lambda v: np.clip(v["x"], 0.0, 1.0),
}


def _coerce(x):
    """Turn a raw input into a Node so the graph is uniform: a Node passes through; a Field or a callable becomes a
    FieldLeaf; a number, a color, or a color-name string becomes a Const. Anything else is a compose-time error."""
    if isinstance(x, Node):
        return x
    if hasattr(x, "sample") or callable(x):
        return FieldLeaf(x)
    return Const(x)                                              # number, rgb list, or colour name -- Const sorts it out


class Map(Node):
    """An operation over TYPED child inputs -- the composition node. The op's schema is checked HERE, at
    construction (compose time), so an ill-typed or ill-shaped graph is refused before you ever sample it.

    Example:
        red, blue = Const([1,0,0]), Const([0,0,1])
        noise     = FieldLeaf(Texture.fbm(n_dims=2))
        m = Map("mix", a=red, b=blue, t=noise)     # blend red<->blue by a noise field
        m.sample([0.3, 0.7])                        # -> an rgb value
    """
    kind = MAP

    def __init__(self, op, **inputs):
        if op not in OP_SCHEMA:
            raise ValueError("unknown texture op %r -- known ops: %s" % (op, ", ".join(sorted(OP_SCHEMA))))
        schema = OP_SCHEMA[op]
        want, got = set(schema), set(inputs)
        if want != got:                                          # missing or extra inputs -> a clear message
            missing, extra = want - got, got - want
            parts = []
            if missing:
                parts.append("missing %s" % ", ".join(sorted(missing)))
            if extra:
                parts.append("unexpected %s" % ", ".join(sorted(extra)))
            raise TypeError("op %r takes inputs {%s}; %s" % (op, ", ".join(sorted(want)), "; ".join(parts)))

        self.op = op
        self.inputs = {}
        for slot, allowed in schema.items():
            node = _coerce(inputs[slot])
            if node.kind not in allowed:                         # the type-hierarchy check, at compose time
                raise TypeError("op %r input %r accepts {%s}, but got a %s"
                                % (op, slot, ", ".join(allowed), node.kind))
            self.inputs[slot] = node

    def sample(self, uv):
        vals = {slot: node.sample(uv) for slot, node in self.inputs.items()}   # evaluate children, then the op
        return OP_EVAL[self.op](vals)

    def describe(self, indent=0):
        lines = ["  " * indent + "Map(%s)" % self.op]
        for slot, node in self.inputs.items():
            lines.append("  " * (indent + 1) + slot + ":")
            lines.append(node.describe(indent + 2))
        return "\n".join(lines)

    def __repr__(self):
        return "Map(%s, {%s})" % (self.op, ", ".join(self.inputs))


# ------------------------------------------------------------------------------------------------------------
# CONVENIENCE: pull a leaf field straight from the Texture library, and evaluate a whole graph over a UV grid.
# ------------------------------------------------------------------------------------------------------------

def field_leaf(source, **kw):
    """A FieldLeaf from a named Texture source, e.g. field_leaf('fbm', n_dims=2) or field_leaf('voronoi'). Wraps the
    callable Texture.<source>(**kw) returns. Keeps the leaf sources in ONE place (texturehome)."""
    from holographic_texturehome import Texture
    fn = getattr(Texture, source, None)
    if fn is None or not callable(fn):
        avail = sorted(n for n in dir(Texture) if not n.startswith("_") and callable(getattr(Texture, n)))
        raise ValueError("unknown Texture source %r -- available sources: %s" % (source, ", ".join(avail)))
    return FieldLeaf(fn(**kw), name=source)


def sample_grid(node, res=32, lo=0.0, hi=1.0):
    """Evaluate a graph over a res x res UV grid -> an array. Handy for baking (the Cache home) or a preview.
    Returns (res, res) for a scalar graph or (res, res, C) for a colour graph."""
    us = np.linspace(lo, hi, res)
    rows = [[node.sample([u, v]) for u in us] for v in us]
    return np.asarray(rows, dtype=float)


# ------------------------------------------------------------------------------------------------------------
# OPTIONAL VECTOR FORM.  Lower the readable tree to the (op, child, ...) / str-leaf shape holographic_typed wants,
# then encode it to a hypervector -- for CACHING a graph's identity or SEARCHING a library of graphs. This is the
# "encode where it earns its keep" path; the object tree above stays the source of truth.
# ------------------------------------------------------------------------------------------------------------

def to_expr(node):
    """The typed-tree form: a leaf becomes a string symbol, a Map becomes a tuple (op, child0, child1, ...). This
    encodes the graph's STRUCTURE and op/leaf identities (not a field's per-point values), which is what you match
    on when caching or searching."""
    if isinstance(node, Const):
        return "num:%.6g" % float(node.value) if node.kind == NUMBER else "color:%s" % ",".join("%.4g" % c for c in node.value)
    if isinstance(node, FieldLeaf):
        return "field:" + node.name
    if isinstance(node, Map):
        return tuple([node.op] + [to_expr(node.inputs[s]) for s in node.inputs])
    raise TypeError("cannot lower %r" % type(node))


def encode(node, dim, seed=0):
    """The graph as ONE hypervector (via holographic_typed.encode_tree): use it as a content-addressable key for a
    baked result, or to find similar graphs in a library. Structurally identical graphs encode identically."""
    from holographic_typed import encode_tree
    return encode_tree(dim, seed, to_expr(node))


def _selftest():
    from holographic_texturehome import Texture

    # a real composition: blend two colours by an fbm field, then modulate by a second child map
    red, blue, white = Const([1.0, 0.0, 0.0]), Const([0.0, 0.0, 1.0]), Const([1.0, 1.0, 1.0])
    noise = field_leaf("fbm", n_dims=2, seed=0)
    base = Map("mix", a=red, b=blue, t=noise)                    # child map
    top = Map("multiply", a=base, b=white)                       # a map whose input is another map
    val = top.sample([0.3, 0.7])
    assert val.shape == (3,), val                                # colour out

    # sampling is deterministic (same uv -> same value)
    assert np.allclose(top.sample([0.3, 0.7]), top.sample([0.3, 0.7]))

    # the SCHEMA refuses bad graphs at COMPOSE time -----------------------------------------------------------
    try:
        Map("mix", a=red, b=blue, t=red)                         # a colour as a weight -> refused
        raise AssertionError("schema should reject a color in the 't' (weight) slot")
    except TypeError as e:
        assert "accepts" in str(e)
    try:
        Map("mix", a=red, b=blue)                                # missing 't' -> refused
        raise AssertionError("schema should reject a missing input")
    except TypeError as e:
        assert "missing t" in str(e)
    try:
        Map("nope", a=red, b=blue)                               # unknown op -> refused
        raise AssertionError("unknown op should be refused")
    except ValueError:
        pass

    # grid bake + vector encode (the optional forms)
    g = sample_grid(base, res=8)
    assert g.shape == (8, 8, 3)
    v1 = encode(top, dim=1024, seed=0)
    v2 = encode(Map("multiply", a=Map("mix", a=Const([1.0, 0.0, 0.0]), b=Const([0.0, 0.0, 1.0]),
                                      t=field_leaf("fbm", n_dims=2, seed=0)), b=Const([1.0, 1.0, 1.0])),
                dim=1024, seed=0)
    assert float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) > 0.99   # same structure -> same code

    print("OK: holographic_texturegraph self-test passed (nested mix->multiply samples an rgb; schema refuses "
          "color-as-weight / missing-input / unknown-op at compose time; grid bake %s; structural encode matches)"
          % (g.shape,))


if __name__ == "__main__":
    _selftest()
