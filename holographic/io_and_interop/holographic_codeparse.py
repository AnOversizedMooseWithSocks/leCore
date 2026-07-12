"""holographic_codeparse.py -- reverse parsers: dialect source -> shared IR -> any dialect (backlog C2).

holographic_emit proves six languages share one IR for the kernel grammar (scalar, straight-line, float). C2 adds
the missing direction: parse c_f64 / c_f32 / wgsl / js / zig_f64 / zig_f32 back into that SAME IR -- a Python
FunctionDef -- after which everything already built applies: emit to any dialect, verbalize in English
(holographic_codeverbal, C4 for free), validate by execution (validate_c / validate_zig).

THE BAR IS ROUND-TRIP BYTE-IDENTITY, and it is executed by the selftest across every dialect pair:
    emit(parse(emit(k, d), d), d) == emit(k, d)              for all d          (self round-trip)
    emit(parse(emit(k, d1), d1), d2) == emit(k, d2)          for all d1 != d2   (translation commutes)
This is a strong bar and it is CHEAP to hold because the emitter's output is regular by construction: every
binary operation is fully parenthesized, every declaration is one statement, every literal is repr(float(v)).
The parser therefore does not need operator precedence at all for emitter output -- but it implements precedence
climbing anyway, because C2's value is parsing HAND-WRITTEN kernels in these dialects, not just our own echo.

ANTI-SILO: the per-dialect knowledge here is INVERTED from holographic_emit's own tables (INTRINSICS turned
inside out per dialect; Zig's template intrinsic recognized by its 'std.math.pow(' head with the type argument
dropped). A dialect added to the emit table gains a reverse mapping here automatically; only the signature /
declaration line shapes are per-dialect code, because those genuinely differ.

KEPT NEGATIVE (scope): the vector dialects (zigv_*) are NOT parsed -- their @as(V, @splat(...)) constant wrapping
and vector type alias make them a different surface grammar, and the scalar IR is the canonical form anyway
(the vector form is DERIVED from it by zigrun; parsing the derivative would be parsing our own exhaust).
KEPT NEGATIVE (scope): the grammar is the emit grammar -- one function, float scalars, straight-line, the
registered intrinsics. Anything else is refused BY NAME (unknown function, unsupported statement), same K10
discipline as the emitter: a parser that guesses produces plausible wrong IR, which is worse than no IR.
"""

import ast
import re

from holographic.io_and_interop.holographic_emit import DIALECTS, INTRINSICS, EmitError, emit_source

#: dialect -> {dialect_intrinsic_name: python_intrinsic_name}, inverted from the emit table so it cannot drift.
_REVERSE = {}
for _py, _per in INTRINSICS.items():
    for _d, _entry in _per.items():
        if "{args}" in _entry:                            # Zig's template pow: recognized by its call head
            _REVERSE.setdefault(_d, {})[_entry.split("(")[0]] = (_py, True)
        else:
            _REVERSE.setdefault(_d, {})[_entry] = (_py, False)

_SCALAR_DIALECTS = ("c_f64", "c_f32", "wgsl", "js", "zig_f64", "zig_f32")

#: signature line recognizers, one per dialect -- the ONLY genuinely per-dialect knowledge in this module.
_SIG = {
    "c_f64": re.compile(r"^\s*double\s+(\w+)\s*\(([^)]*)\)\s*\{\s*$"),
    "c_f32": re.compile(r"^\s*float\s+(\w+)\s*\(([^)]*)\)\s*\{\s*$"),
    "wgsl": re.compile(r"^\s*fn\s+(\w+)\s*\(([^)]*)\)\s*->\s*f32\s*\{\s*$"),
    "js": re.compile(r"^\s*function\s+(\w+)\s*\(([^)]*)\)\s*\{\s*$"),
    "zig_f64": re.compile(r"^\s*fn\s+(\w+)\s*\(([^)]*)\)\s*f64\s*\{\s*$"),
    "zig_f32": re.compile(r"^\s*fn\s+(\w+)\s*\(([^)]*)\)\s*f32\s*\{\s*$"),
}
_DECL = {
    "c_f64": re.compile(r"^\s*double\s+(\w+)\s*=\s*(.+);\s*$"),
    "c_f32": re.compile(r"^\s*float\s+(\w+)\s*=\s*(.+);\s*$"),
    "wgsl": re.compile(r"^\s*let\s+(\w+)\s*=\s*(.+);\s*$"),
    "js": re.compile(r"^\s*const\s+(\w+)\s*=\s*(.+);\s*$"),
    "zig_f64": re.compile(r"^\s*const\s+(\w+)\s*:\s*f64\s*=\s*(.+);\s*$"),
    "zig_f32": re.compile(r"^\s*const\s+(\w+)\s*:\s*f32\s*=\s*(.+);\s*$"),
}
_RETURN = re.compile(r"^\s*return\s+(.+);\s*$")


class ParseError(EmitError):
    """The parser refused. Same K10 discipline as the emitter: the message names the construct."""


# -- expression parsing: one tokenizer + precedence climber shared by every dialect ---------------------------

_TOKEN = re.compile(r"\s*(?:(\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)([fF]?)"
                    r"|(@?[A-Za-z_][\w.]*)|([()+\-*/,]))")


def _tokenize(s):
    out, i = [], 0
    while i < len(s):
        m = _TOKEN.match(s, i)
        if not m or m.end() == i:
            raise ParseError("cannot tokenize %r at position %d" % (s, i))
        num, _suf, name, op = m.groups()
        if num is not None:
            out.append(("num", num))                      # the f suffix is dialect noise; the value is the token
        elif name is not None:
            out.append(("name", name))
        else:
            out.append(("op", op))
        i = m.end()
    out.append(("end", ""))
    return out


class _Expr:
    """Precedence-climbing expression parser over the shared token stream, producing PYTHON source text.

    Producing text (not ast nodes) keeps the reconstruction honest: the text is handed to ast.parse, so the IR is
    built by the same parser Python itself uses, not by hand-assembled nodes that might disagree with it."""

    def __init__(self, tokens, dialect):
        self.t, self.i, self.d = tokens, 0, dialect

    def peek(self):
        return self.t[self.i]

    def take(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def parse(self):
        e = self.add()
        if self.peek()[0] != "end":
            raise ParseError("trailing tokens after expression: %r" % (self.t[self.i:],))
        return e

    def add(self):
        e = self.mul()
        while self.peek() == ("op", "+") or self.peek() == ("op", "-"):
            op = self.take()[1]
            e = "(%s %s %s)" % (e, op, self.mul())
        return e

    def mul(self):
        e = self.unary()
        while self.peek() == ("op", "*") or self.peek() == ("op", "/"):
            op = self.take()[1]
            e = "(%s %s %s)" % (e, op, self.unary())
        return e

    def unary(self):
        if self.peek() == ("op", "-"):
            self.take()
            return "(-%s)" % self.unary()
        return self.atom()

    def atom(self):
        kind, val = self.take()
        if kind == "num":
            return repr(float(val))                       # canonical literal: the digits the emitter would write
        if kind == "op" and val == "(":
            e = self.add()
            if self.take() != ("op", ")"):
                raise ParseError("unbalanced parenthesis")
            return e
        if kind == "name":
            if self.peek() == ("op", "("):                # a call: must be a registered intrinsic for this dialect
                self.take()
                rev = _REVERSE.get(self.d, {})
                if val not in rev:
                    raise ParseError("unknown function %r in dialect %r: not the inverse of any registered "
                                     "intrinsic. The parser refuses rather than guessing (K10)." % (val, self.d))
                py_name, is_template = rev[val]
                args = []
                if self.peek() != ("op", ")"):
                    args.append(self.add())
                    while self.peek() == ("op", ","):
                        self.take()
                        args.append(self.add())
                if self.take() != ("op", ")"):
                    raise ParseError("unbalanced call parenthesis in %r" % val)
                if is_template:
                    args = args[1:]                       # drop Zig's type argument (f64/f32) -- IR carries dtype
                return "%s(%s)" % (py_name, ", ".join(args))
            if "." in val:
                raise ParseError("dotted name %r is not a registered intrinsic call" % val)
            return val
        raise ParseError("unexpected token %r" % ((kind, val),))


def _parse_expr(s, dialect):
    return _Expr(_tokenize(s), dialect).parse()


def parse_kernel(src, dialect):
    """Parse one dialect kernel back into the shared IR, returned as PYTHON SOURCE (fully float-annotated, so it
    is immediately re-emittable by holographic_emit and explainable by holographic_codeverbal).

    Refuses, by name: an unknown dialect, a signature that does not match the dialect's shape, a statement that
    is neither a declaration nor a return, and any call that is not a registered intrinsic's inverse."""
    if dialect not in _SCALAR_DIALECTS:
        raise ParseError("unknown or unsupported dialect %r; parseable: %s (zigv_* is a declared negative -- "
                         "the vector form is derived exhaust, the scalar IR is canonical)"
                         % (dialect, list(_SCALAR_DIALECTS)))
    lines = [ln for ln in src.splitlines() if ln.strip()]
    if not lines:
        raise ParseError("empty source")
    m = _SIG[dialect].match(lines[0])
    if not m:
        raise ParseError("line 1 does not match the %r signature shape: %r" % (dialect, lines[0]))
    name, params_raw = m.group(1), m.group(2)
    params = []
    for p in [q.strip() for q in params_raw.split(",") if q.strip()]:
        pm = re.match(r"^(?:double|float)\s+(\w+)$", p) or re.match(r"^(\w+)\s*:\s*f(?:32|64)$", p) \
             or re.match(r"^(\w+)$", p)
        if not pm:
            raise ParseError("cannot parse parameter %r in dialect %r" % (p, dialect))
        params.append(pm.group(1))

    body = []
    saw_return = False
    for ln in lines[1:]:
        if ln.strip() == "}":
            break
        dm = _DECL[dialect].match(ln)
        if dm:
            if saw_return:
                raise ParseError("statement after return: %r" % ln)
            body.append("    %s = %s" % (dm.group(1), _parse_expr(dm.group(2), dialect)))
            continue
        rm = _RETURN.match(ln)
        if rm:
            body.append("    return %s" % _parse_expr(rm.group(1), dialect))
            saw_return = True
            continue
        raise ParseError("unsupported statement (not a declaration or return): %r. The grammar is the emit "
                         "grammar; the parser refuses rather than guessing (K10)." % ln)
    if not saw_return:
        raise ParseError("kernel %r never returns" % name)

    py = "def %s(%s) -> float:\n%s\n" % (name, ", ".join("%s: float" % p for p in params), "\n".join(body))
    ast.parse(py)                                         # the IR is built by Python's own parser, or not at all
    return py


def translate(src, from_dialect, to_dialect):
    """Translate a kernel between any two of: python + the six scalar dialects. Through the ONE shared IR --
    there are no pairwise paths to drift apart."""
    py = src if from_dialect == "python" else parse_kernel(src, from_dialect)
    if to_dialect == "python":
        return py
    return emit_source(py, to_dialect)


def _selftest():
    """The C2 bar, executed: round-trip byte-identity over every dialect pair, for kernels exercising every
    intrinsic class (plain, suffixed, dotted JS, Zig template pow), plus the refusals that keep the parser
    honest. 4 kernels x 6 self round-trips x 30 cross pairs -- all byte-compared, none sampled."""
    kernels = [
        "def sdf_sphere(px: float, py: float, pz: float, r: float) -> float:\n"
        "    d = sqrt(px * px + py * py + pz * pz)\n    return d - r\n",
        "def lerp(a: float, b: float, t: float) -> float:\n    return a + (b - a) * t\n",
        "def powk(a: float, b: float) -> float:\n    c = pow(a, b) + sin(a)\n    return c * 2.0\n",
        "def rbox(px: float, py: float, bx: float, r: float) -> float:\n"
        "    qx = max(abs(px) - bx, 0.0)\n    qy = max(abs(py) - bx, 0.0)\n"
        "    d = sqrt(qx * qx + qy * qy)\n    return d - r\n",
    ]
    n_self = n_cross = 0
    for k in kernels:
        for d1 in _SCALAR_DIALECTS:
            e1 = emit_source(k, d1)
            assert translate(e1, d1, d1) == e1, "self round-trip broke for %s" % d1
            n_self += 1
            for d2 in _SCALAR_DIALECTS:
                if d2 == d1:
                    continue
                assert translate(e1, d1, d2) == emit_source(k, d2), "translation %s->%s does not commute" % (d1, d2)
                n_cross += 1
        assert translate(emit_source(k, "c_f64"), "c_f64", "python") == parse_kernel(emit_source(k, "c_f64"), "c_f64")

    # hand-written (non-emitter) input: precedence must hold without the emitter's full parenthesization
    hand_c = "double f(double a, double b) {\n  double x = a + b * 2.0 - -a;\n  return x / (a + 1.0);\n}\n"
    py = parse_kernel(hand_c, "c_f64")
    import math                                            # noqa: F401 -- namespace for the check below
    ns = {"sqrt": math.sqrt}
    exec(compile(py, "<t>", "exec"), ns)                   # noqa: S102 -- our own reconstructed text
    assert abs(ns["f"](3.0, 4.0) - ((3.0 + 4.0 * 2.0 + 3.0) / 4.0)) < 1e-15, "precedence parse is wrong"

    # refusals, by name
    for bad, d, frag in [("int f(int a) { return a; }", "c_f64", "signature"),
                         ("double f(double a) {\n  for(;;){}\n  return a;\n}", "c_f64", "unsupported statement"),
                         ("double f(double a) {\n  return foo(a);\n}", "c_f64", "unknown function"),
                         ("fn f(a: f32) -> f32 { return a; }", "zigv_f32", "unsupported dialect")]:
        try:
            parse_kernel(bad, d)
            raise AssertionError("must refuse: %r" % bad)
        except ParseError as e:
            assert frag.split()[0] in str(e) or frag in str(e), (frag, str(e))

    print("OK: holographic_codeparse self-test passed (round-trip BYTE-IDENTITY held over %d self and %d cross "
          "dialect pairs -- emit(parse(emit(k,d1)),d2) == emit(k,d2) for all pairs, 4 kernels covering plain/"
          "suffixed/dotted/template intrinsics; hand-written C with real precedence parses correctly; refusals "
          "fire by name for a foreign signature, a loop, an unknown call, and the declared zigv negative)"
          % (n_self, n_cross))


if __name__ == "__main__":
    _selftest()
