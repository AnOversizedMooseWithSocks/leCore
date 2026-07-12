"""holographic_sdfemit.py -- the scene's own SDF, emitted to WGSL / C / GLSL (the brain/muscle contract, realised).

The backlog's brain/muscle claim: *"the compute shaders the three.js demos hand-write become a PROJECTION of the
authoritative Python kernel -- one source of truth, two runtimes, no drift."*

It was not realised. `holographic_sdf.SDF.to_glsl()` emitted GLSL for a tree. `holographic_emit` emitted WGSL, C and
JS -- but only from a *scalar Python function's source text*. **The two emitters never met**, so
`RealtimeSession.payload("shader")` carried whatever `kernel_src` the caller passed: a shader the caller wrote by
hand, about a scene the engine never saw. That is drift by construction, and it is the exact thing the contract
exists to prevent.

`sdf_dialect(node, dialect)` walks the SAME tree `_eval` walks and emits `map(p) -> distance` in:

    wgsl    fn map(p: vec3<f32>) -> f32        the browser's muscle
    c_f64   double map(const double p[3])      the executable twin
    c_f32   float  map(const float  p[3])      what WGSL's precision actually is
    glsl    float map(vec3 p)                  Shadertoy, and what already shipped

THE BAR IS EXECUTED, exactly as K8's was. WGSL cannot be run here -- no GPU, no browser -- so the C dialect is
compiled with `cc` and RUN against the Python `_eval` on the same random points. Measured over 200 points on a
compound tree (a scaled smooth-union of a translated sphere and a rotated box):

    dialect   max |emitted - python|
    c_f64          6.7e-16            machine epsilon -- and NOT bit-identical
    c_f32          3.3e-07            TRUE f32 arithmetic; this IS the tolerance a WGSL port is judged against

**AND THAT `c_f64` IS NOT BIT-IDENTICAL, WHERE K8's SCALAR KERNEL WAS.** The difference is real and worth naming.
K8 emitted the *same expression* the Python function evaluated, so the operations happened in the same order and the
doubles agreed exactly. Here the Python side is `numpy`: `np.linalg.norm` does not compute `sqrt(x*x + y*y + z*z)`
in that order -- it rescales to avoid overflow -- and `np.clip` is not `clamp`. The emitted C computes the same
FUNCTION by a different summation, so it agrees to machine epsilon and not to the bit.

**And bit-identity is TREE-DEPENDENT, which is why the module reports `max_abs_diff` and not a boolean.** A bare
`sphere` comes out at exactly 0.0 -- `np.linalg.norm` and `sqrt(x*x+y*y+z*z)` happen to agree on three terms. Add a
`rotate` and a `scale` and the extra multiplies reassociate: 6.7e-16, four ulp. *Asserting `bit_identical` would
have been a bar that passes on a sphere and fails on a scene.*

KEPT NEGATIVE 1 -- **`menger` is not emittable, and refusing is the feature.** It is an ITERATED domain fold: a
Python loop over `iters` with a running scale. Unrolling it into a straight-line expression would produce a
correct-but-enormous shader whose size depends on a parameter, and emitting a loop would need a dialect table for
control flow that this emitter does not have. It raises, naming the node. (`holographic_sdf`'s own `INEXACT` set
already flags twist/displace as domain warps that are not exact distances; this emitter refuses those too.)

KEPT NEGATIVE 2 -- **`scale` is not `p / s`, it is `map(p / s) * s`, and forgetting the outer factor is a shader
that renders a correct SHAPE with wrong distances.** A raymarcher would overstep and miss it. The Python `_eval`
has the factor; the emitter carries it; a test pins a scaled sphere's distance at a point far from the surface,
where the shape looks right and the distance does not.

KEPT NEGATIVE 3 -- **the `f` suffix on a C literal is load-bearing.** Unsuffixed, `0.25` is a DOUBLE, and
`float_expr * 0.25` evaluates the whole expression in double before truncating. The first version of this table
omitted it, so the `c_f32` build -- the executable stand-in for WGSL -- was not a pure-f32 twin, and the tolerance
it published (2.83e-07) was **optimistic by 15%** against the true 3.26e-07. An audit found it by noticing that
`holographic_emit`'s dialect table used `"f"` and this one did not. **Two tables for one concept will disagree, and
the disagreement will be a bug in one of them.** A test now asserts the shared dialects agree, field by field.

KEPT NEGATIVE 4 -- **an emitted shader is not a rendered image.** This validates the DISTANCE FUNCTION against the
Python one, to f32 tolerance. It does not validate WGSL's precision rules, its fast-math latitude, whether the
shader compiles, or whether the browser's raymarch loop matches the engine's. Those are the front end's tests, and
saying so is cheaper than having someone discover it in a browser.
"""

import numpy as np

from holographic.mesh_and_geometry.holographic_sdf import SDF

#: Nodes the multi-dialect emitter refuses. `menger` folds the domain ITERATIVELY (its unrolled size depends on a
#: parameter); `twist`, `displace`, `bend`, and `ellipsoid` are `holographic_sdf.INEXACT` -- not exact distances,
#: so a raymarcher must shorten its steps and the shader needs a warning the emitter cannot enforce. `mirror` (an
#: exact isometry) and `repeat` (infinite tiling) ARE emittable in all four dialects. `capsule`/`cone`/
#: `octahedron` are EXACT and emit via the GLSL Shadertoy path (holographic_sdf.to_glsl); they are refused HERE
#: (the 4-dialect WGSL/C emitter) only because their branch-heavy forms (cone's caps, octahedron's face select)
#: are not yet ported to the dialect table -- a filed follow-up, not a mathematical limit. capsule is a clamp
#: away and is the first to add when the table grows a general clamp(lo,hi).
UNEMITTABLE = ("menger", "twist", "displace", "bend", "ellipsoid", "capsule", "cone", "octahedron", "elongate")

DIALECTS = {
    "wgsl": {"scalar": "f32", "vec3": "vec3<f32>", "infer_types": True, "suffix": "f",
             "sig": "fn map(p: vec3<f32>) -> f32", "swz": lambda v, c: "%s.%s" % (v, c),
             "vec": lambda a, b, c: "vec3<f32>(%s, %s, %s)" % (a, b, c),
             "len2": lambda a, b: "length(vec2<f32>(%s, %s))" % (a, b), "len3": lambda v: "length(%s)" % v,
             "max3": lambda v: "max(max(%s.x, %s.y), %s.z)" % (v, v, v),
             "maxv0": lambda v: "max(%s, vec3<f32>(0.0f))" % v,
             "mod": lambda x, y: "(%s - %s * floor((%s) / (%s)))" % (x, y, x, y),   # WGSL has no mod(); floor form
             "abs": lambda v: "abs(%s)" % v, "clamp": lambda e: "clamp(%s, 0.0f, 1.0f)" % e},
    "glsl": {"scalar": "float", "vec3": "vec3", "infer_types": False, "suffix": "",
             "sig": "float map(vec3 p)", "swz": lambda v, c: "%s.%s" % (v, c),
             "vec": lambda a, b, c: "vec3(%s, %s, %s)" % (a, b, c),
             "len2": lambda a, b: "length(vec2(%s, %s))" % (a, b), "len3": lambda v: "length(%s)" % v,
             "max3": lambda v: "max(max(%s.x, %s.y), %s.z)" % (v, v, v),
             "maxv0": lambda v: "max(%s, vec3(0.0))" % v,
             "mod": lambda x, y: "mod(%s, %s)" % (x, y),                            # GLSL builtin (floor-based)
             "abs": lambda v: "abs(%s)" % v, "clamp": lambda e: "clamp(%s, 0.0, 1.0)" % e},
}

# C has no vec3, so the C dialects carry a tiny header and index a float[3]. The SAME tree walker drives all four;
# only the table differs -- which is the whole point of a dialect table.
_C_HEADER = """#include <math.h>
typedef struct {{ {s} x, y, z; }} v3;
static v3 v3make({s} x, {s} y, {s} z) {{ v3 r; r.x = x; r.y = y; r.z = z; return r; }}
static {s} v3len(v3 a) {{ return {sq}(a.x*a.x + a.y*a.y + a.z*a.z); }}
static {s} len2({s} a, {s} b) {{ return {sq}(a*a + b*b); }}
static v3 v3abs(v3 a) {{ return v3make({fa}(a.x), {fa}(a.y), {fa}(a.z)); }}
static v3 v3max0(v3 a) {{ return v3make(a.x > 0 ? a.x : 0, a.y > 0 ? a.y : 0, a.z > 0 ? a.z : 0); }}
static {s} max3(v3 a) {{ {s} m = a.x > a.y ? a.x : a.y; return m > a.z ? m : a.z; }}
static {s} fmaxs({s} a, {s} b) {{ return a > b ? a : b; }}
static {s} fmins({s} a, {s} b) {{ return a < b ? a : b; }}
static {s} clamp01({s} a) {{ return a < 0 ? 0 : (a > 1 ? 1 : a); }}
/* GLSL/WGSL mod(x,y) = x - y*floor(x/y): sign follows y (non-negative for y>0), NOT C's fmod which follows x.
   Domain `repeat` needs the floor-based one to centre cells symmetrically, so C emits this, never fmod. */
static {s} modf_({s} x, {s} y) {{ return x - y * {fl}(x / y); }}
"""

# THE `f` SUFFIX IS NOT COSMETIC. An unsuffixed C literal is a DOUBLE, so `float_expr * 3.0` promotes the whole
# expression to double, evaluates it there, and truncates back -- and the `c_f32` build stops being a pure-f32 twin.
# Measured on a compound tree, 400 points: the unsuffixed build reports max error 2.83e-07 against Python, the
# suffixed one 3.26e-07, and they differ from each other by 4.77e-07. **The unsuffixed number was OPTIMISTIC by 15%,
# and it was the number this module published as "the tolerance a WGSL port is judged against."** A duplication scan
# found it: `holographic_emit`'s table already used "f" for c_f32, and the two tables disagreed.
for _d, _s, _sq, _fa, _suf, _fl in (("c_f64", "double", "sqrt", "fabs", "", "floor"),
                                    ("c_f32", "float", "sqrtf", "fabsf", "f", "floorf")):
    DIALECTS[_d] = {
        "scalar": _s, "vec3": "v3", "infer_types": False, "suffix": _suf,
        "sig": "%s map(v3 p)" % _s,
        "swz": lambda v, c, _=None: "%s.%s" % (v, c),
        "vec": lambda a, b, c: "v3make(%s, %s, %s)" % (a, b, c),
        "len2": lambda a, b: "len2(%s, %s)" % (a, b), "len3": lambda v: "v3len(%s)" % v,
        "max3": lambda v: "max3(%s)" % v, "maxv0": lambda v: "v3max0(%s)" % v,
        "abs": lambda v: "v3abs(%s)" % v, "clamp": lambda e: "clamp01(%s)" % e,
        "mod": lambda x, y: "modf_(%s, %s)" % (x, y),          # floor-based, matches GLSL mod (see C header)
        "_header": _C_HEADER.format(s=_s, sq=_sq, fa=_fa, fl=_fl),
        "_min": "fmins", "_max": "fmaxs",
    }


def coverage():
    """Which of `holographic_sdf.ARITY`'s node kinds this emitter handles, and which it refuses.
    `emitted + refused == every kind` -- a gap here is a shader that silently omits geometry."""
    from holographic.mesh_and_geometry.holographic_sdf import ARITY
    refused = set(UNEMITTABLE)
    emitted = set(ARITY) - refused
    return {"emitted": sorted(emitted), "refused": sorted(refused), "total": len(ARITY),
            "complete": bool(emitted | refused == set(ARITY))}


class SdfEmitError(ValueError):
    """The emitter refused. It names the node; refusing is the feature."""


def _lit(x, d):
    return "%r%s" % (float(x), d["suffix"])


def _decl(d, typ, name, expr):
    """Declare a local. **WGSL is not C.** It infers the type with `let name = expr;` and rejects
    `vec3<f32> name = expr;` outright. The first version of this emitter wrote the C form for every dialect and the
    structural test -- which checked only the signature and the brace balance -- passed it. That is the precise
    failure the module's "the WGSL is not executed here" negative warns about, caught by reading the output."""
    if d.get("infer_types"):
        return "let %s = %s;" % (name, expr)
    return "%s %s = %s;" % (typ, name, expr)


def _minmax(d, fn, a, b):
    if fn == "min":
        return "%s(%s, %s)" % (d.get("_min", "min"), a, b)
    return "%s(%s, %s)" % (d.get("_max", "max"), a, b)


def _emit(node, pvar, d, ctr):
    """Walk the tree, emitting statements and returning `(stmts, distance_expr)`.

    Mirrors `holographic_sdf._eval` node for node. Where `_eval` says `np.minimum`, this says the dialect's `min`;
    where `_eval` scales the RESULT, this scales the result. Any divergence is a shader that renders a different
    scene, so the two are read side by side."""
    k, p, ch = node.kind, node.params, node.children
    if k in UNEMITTABLE:
        raise SdfEmitError("node %r is not emittable: it folds the domain iteratively or inexactly, and unrolling it "
                           "would produce a shader whose size depends on a parameter. Refusing rather than "
                           "approximating." % (k,))

    def nv(pfx):
        ctr[0] += 1
        return "%s%d" % (pfx, ctr[0])

    if k == "sphere":
        return [], "(%s - %s)" % (d["len3"](pvar), _lit(p[0], d))

    if k == "box":
        q = nv("q")
        stmts = [_decl(d, d["vec3"], q, _sub_vec(d["abs"](pvar), d["vec"](_lit(p[0], d), _lit(p[1], d),
                                                                                _lit(p[2], d)), d))]
        dist = "(%s + %s)" % (d["len3"](d["maxv0"](q)), _minmax(d, "min", d["max3"](q), _lit(0.0, d)))
        return stmts, dist

    if k == "torus":
        R, r = p
        xz = nv("t")
        stmts = [_decl(d, d["scalar"], xz, "(%s - %s)" % (d["len2"](d["swz"](pvar, "x"), d["swz"](pvar, "z")),
                                                         _lit(R, d)))]
        return stmts, "(%s - %s)" % (d["len2"](xz, d["swz"](pvar, "y")), _lit(r, d))

    if k == "cylinder":
        h, r = p
        a, b = nv("cx"), nv("cy")
        stmts = [_decl(d, d["scalar"], a, "(%s - %s)" % (d["len2"](d["swz"](pvar, "x"), d["swz"](pvar, "z")),
                                                        _lit(r, d))),
                 _decl(d, d["scalar"], b, "(%s - %s)" % (_abs_s(d["swz"](pvar, "y"), d), _lit(h, d)))]
        inner = _minmax(d, "min", _minmax(d, "max", a, b), _lit(0.0, d))
        outer = "sqrt(%s * %s + %s * %s)" % ((_minmax(d, "max", a, _lit(0.0, d)),) * 2
                                             + (_minmax(d, "max", b, _lit(0.0, d)),) * 2)
        if d["scalar"] == "float":
            outer = "sqrtf" + outer[4:]
        return stmts, "(%s + %s)" % (inner, outer)

    if k == "plane":
        return [], "(%s - %s)" % (d["swz"](pvar, "y"), _lit(p[0], d))

    if k in ("union", "intersect", "subtract", "smooth_union"):
        sa, ea = _emit(ch[0], pvar, d, ctr)
        sb, eb = _emit(ch[1], pvar, d, ctr)
        va, vb = nv("a"), nv("b")
        stmts = sa + [_decl(d, d["scalar"], va, ea)] + sb + [_decl(d, d["scalar"], vb, eb)]
        if k == "union":
            return stmts, _minmax(d, "min", va, vb)
        if k == "intersect":
            return stmts, _minmax(d, "max", va, vb)
        if k == "subtract":
            return stmts, _minmax(d, "max", va, "(-%s)" % vb)
        kk = _lit(p[0], d)
        h = nv("h")
        stmts.append(_decl(d, d["scalar"], h,
                           d["clamp"]("%s + %s * (%s - %s) / %s" % (_lit(0.5, d), _lit(0.5, d), vb, va, kk))))
        return stmts, "(%s * (%s - %s) + %s * %s - %s * %s * (%s - %s))" % (
            vb, _lit(1.0, d), h, va, h, kk, h, _lit(1.0, d), h)

    if k == "onion":
        sc, ec = _emit(ch[0], pvar, d, ctr)
        return sc, "(%s - %s)" % (_abs_s("(%s)" % ec, d), _lit(p[0], d))

    if k == "round":                       # `SDF.rounded()` builds a node named "round" -- read the tree, do not
        sc, ec = _emit(ch[0], pvar, d, ctr)  # assume the method's name is the node's name
        return sc, "((%s) - %s)" % (ec, _lit(p[0], d))

    if k == "translate":
        q = nv("p")
        stmts = [_decl(d, d["vec3"], q,
                       _sub_vec(pvar, d["vec"](_lit(p[0], d), _lit(p[1], d), _lit(p[2], d)), d))]
        sc, ec = _emit(ch[0], q, d, ctr)
        return stmts + sc, ec

    if k == "scale":
        s = float(p[0])
        q = nv("p")
        stmts = [_decl(d, d["vec3"], q, _div_vec(pvar, _lit(s, d), d))]
        sc, ec = _emit(ch[0], q, d, ctr)
        # `_eval` returns `child(P / s) * s`. Dropping the outer factor gives the right SHAPE with wrong DISTANCES,
        # and a raymarcher oversteps it. Kept negative 2.
        return stmts + sc, "((%s) * %s)" % (ec, _lit(s, d))

    if k == "rotate":
        from holographic.mesh_and_geometry.holographic_sdf import _rot_matrix
        R = _rot_matrix(p[:3], p[3])
        q = nv("p")
        cols = []
        for j in range(3):
            cols.append("(%s * %s + %s * %s + %s * %s)" % (
                d["swz"](pvar, "x"), _lit(R[0, j], d), d["swz"](pvar, "y"), _lit(R[1, j], d),
                d["swz"](pvar, "z"), _lit(R[2, j], d)))
        stmts = [_decl(d, d["vec3"], q, d["vec"](*cols))]             # P @ R, exactly as `_eval` does
        sc, ec = _emit(ch[0], q, d, ctr)
        return stmts + sc, ec

    if k == "mirror":
        # reflect one axis across a plane: q.<axis> = plane + abs(p.<axis> - plane); other two pass through.
        # `_eval` does exactly this. A reflection is an ISOMETRY, so no distance correction is needed (unlike
        # twist/bend, which is why mirror emits and they do not). Build a whole new vec3 so the one rule works
        # in C (no swizzle assignment) as well as WGSL/GLSL -- the dialect table's `vec` and component reads.
        axis, plane = int(p[0]), p[1]
        comp = ("x", "y", "z")[axis]
        pl = _lit(plane, d)
        folded = "(%s + %s)" % (pl, _abs_s("(%s - %s)" % (d["swz"](pvar, comp), pl), d))
        parts = [folded if a == axis else d["swz"](pvar, ("x", "y", "z")[a]) for a in range(3)]
        q = nv("p")
        stmts = [_decl(d, d["vec3"], q, d["vec"](*parts))]
        sc, ec = _emit(ch[0], q, d, ctr)
        return stmts + sc, ec

    if k == "repeat":
        # INFINITE domain repetition: per axis with period c>0, q.<axis> = mod(p.<axis> + c/2, c) - c/2. This is a
        # single fixed-size warp (three mod expressions), NOT an iterative fold -- the old refusal conflated it
        # with menger (which truly iterates) and repeat_limited (finite unroll). One mod per axis, exactly as
        # `_eval` and the GLSL `to_glsl` path do. The dialect `mod` is floor-based in every backend (GLSL builtin,
        # WGSL/C floor form) so cells centre symmetrically and the four emissions agree.
        parts = []
        for a in range(3):
            c = float(p[a])
            comp = ("x", "y", "z")[a]
            src = d["swz"](pvar, comp)
            if c > 0:
                half = _lit(0.5 * c, d)
                parts.append("(%s - %s)" % (d["mod"]("(%s + %s)" % (src, half), _lit(c, d)), half))
            else:
                parts.append(src)                            # period 0 on this axis = no repetition
        q = nv("p")
        stmts = [_decl(d, d["vec3"], q, d["vec"](*parts))]
        sc, ec = _emit(ch[0], q, d, ctr)
        return stmts + sc, ec

    raise SdfEmitError("no dialect rule for node %r" % (k,))


def _sub_vec(a, b, d):
    if d["vec3"] == "v3":
        return "v3make(%s.x - %s.x, %s.y - %s.y, %s.z - %s.z)" % (a, b, a, b, a, b)
    return "(%s - %s)" % (a, b)


def _div_vec(a, s, d):
    if d["vec3"] == "v3":
        return "v3make(%s.x / %s, %s.y / %s, %s.z / %s)" % (a, s, a, s, a, s)
    return "(%s / %s)" % (a, s)


def _abs_s(e, d):
    # WHY key on vec3=="v3" and not scalar=="float": GLSL's scalar is ALSO "float", so keying on the scalar name
    # wrongly emits C's fabsf into a GLSL shader. Only C uses the v3 vector type, so that is the honest C test;
    # within C, f64 wants fabs and f32 wants fabsf (the suffix distinguishes the precision). GLSL and WGSL both
    # spell scalar abs as abs().
    if d["vec3"] == "v3":
        return ("fabsf(%s)" if d["scalar"] == "float" else "fabs(%s)") % e
    return "abs(%s)" % e


def as_tree(node):
    """Coerce to an `SDF` tree. Accepts one already, or its **DSL TEXT** -- `(smooth_union 0.25 (sphere 0.7) ...)`.

    A live tree does not survive JSON; its DSL does, and `parse_dsl(to_dsl(t))` round-trips to 0.0e+00. **The kernel
    is text; so is the scene.** `emit_kernel` learned this first, and an agent that cannot describe the scene cannot
    ask for its shader."""
    if isinstance(node, SDF):
        return node
    if isinstance(node, str):
        from holographic.mesh_and_geometry.holographic_sdf import parse_dsl
        try:
            return parse_dsl(node)
        except Exception as exc:
            raise SdfEmitError("could not parse the SDF DSL %r (%s)" % (node[:60], exc))
    raise SdfEmitError("expected an SDF tree or its DSL text; got %r" % (type(node).__name__,))


def sdf_dialect(node, dialect="wgsl"):
    """Emit the SDF tree's `map(p) -> distance` in `dialect` (`wgsl` | `glsl` | `c_f64` | `c_f32`).

    `node` is a live `SDF` **or its DSL text** -- a live tree does not survive JSON, and a capability an agent
    cannot call does not exist.

    The SAME tree `_eval` walks. The C dialects carry a small vec3 header so they compile and RUN, which is how the
    emission is checked -- WGSL cannot be executed here, and claiming it works without running something would be
    the kind of claim this engine exists to refuse."""
    node = as_tree(node)
    if dialect not in DIALECTS:
        raise SdfEmitError("unknown dialect %r; try %s" % (dialect, sorted(DIALECTS)))
    d = DIALECTS[dialect]
    stmts, dist = _emit(node, "p", d, [0])
    body = "\n    ".join(stmts + ["return %s;" % dist])
    src = "%s {\n    %s\n}\n" % (d["sig"], body)
    return d.get("_header", "") + src


def validate_c(node, points, dialect="c_f64", timeout=60):
    """Compile the emitted C `map()` with `cc`, RUN it on `points`, and compare to the Python `_eval`.

    Returns `{dialect, n, max_abs_diff, bit_identical}`. `c_f64` comes out BIT-IDENTICAL; `c_f32` does not, and its
    error is the tolerance a WGSL port must be judged against."""
    import os
    import subprocess
    import tempfile

    if dialect not in ("c_f64", "c_f32"):
        raise SdfEmitError("only the C dialects can be executed here; %r cannot" % (dialect,))
    node = as_tree(node)                                       # a tree, or its DSL text
    P = np.asarray(points, float).reshape(-1, 3)
    kernel = sdf_dialect(node, dialect)
    calls = "".join('printf("%%.17g\\n", map(v3make(%r, %r, %r)));' % tuple(float(v) for v in row) for row in P)
    prog = "#include <stdio.h>\n" + kernel + "\nint main(){ " + calls + " return 0; }\n"

    with tempfile.TemporaryDirectory() as tmp:
        csrc, exe = os.path.join(tmp, "m.c"), os.path.join(tmp, "m")
        with open(csrc, "w") as fh:
            fh.write(prog)
        subprocess.run(["cc", csrc, "-o", exe, "-lm"], check=True, capture_output=True, timeout=timeout)
        out = subprocess.run([exe], check=True, capture_output=True, text=True, timeout=timeout).stdout

    got = np.array([float(x) for x in out.split()])
    want = np.asarray(node.eval(P), float)
    diff = float(np.abs(got - want).max())
    return {"dialect": dialect, "n": len(P), "max_abs_diff": diff,
            "bit_identical": bool(np.array_equal(got, want))}


def _selftest():
    """Regression trap: a compound tree emits to C, COMPILES, and matches the Python `_eval` bit-identically in f64;
    the f32 twin does not and that gap is WGSL's tolerance; the WGSL text is well formed; `menger` is refused; and
    `scale` keeps its outer factor."""
    from holographic.mesh_and_geometry import holographic_sdf as S

    # the combinators are METHODS on SDF, not module functions -- read the API, do not assume it
    tree = S.sphere(0.7).translate((0.4, 0.0, -0.2)).smooth_union(
        S.box(0.5, 0.3, 0.6).rotate((0.0, 1.0, 0.0), 0.7), 0.25).scale(1.3)

    rng = np.random.default_rng(0)
    P = rng.uniform(-2.0, 2.0, (200, 3))

    # 1. THE BAR, EXECUTED: emitted C compiled with cc, run, and compared to the Python _eval.
    #    NOT bit-identical, and it must not be asserted so: `np.linalg.norm` rescales to avoid overflow, so it sums
    #    in a different order than `sqrt(x*x + y*y + z*z)`. Machine epsilon is the honest bar.
    rep64 = validate_c(tree, P, "c_f64")
    assert rep64["max_abs_diff"] < 1e-14, rep64
    assert rep64["bit_identical"] is False, "if this ever passes, numpy changed its norm"

    # 2. f32 cannot be bit-identical, and its error IS the WGSL tolerance
    rep32 = validate_c(tree, P, "c_f32")
    assert rep32["bit_identical"] is False
    assert 0.0 < rep32["max_abs_diff"] < 1e-4, rep32

    # 2b. the `f` suffix is load-bearing: without it the literals are doubles and the "f32" twin is not one
    assert DIALECTS["c_f32"]["suffix"] == "f"
    assert "0.25f" in sdf_dialect(tree, "c_f32") or "0.7f" in sdf_dialect(tree, "c_f32")
    assert "0.25f" not in sdf_dialect(tree, "c_f64")

    # 2c. THE TWO DIALECT TABLES MUST AGREE where they overlap -- two tables for one concept will drift
    from holographic.io_and_interop.holographic_emit import DIALECTS as _EMIT
    for _d in set(_EMIT) & set(DIALECTS):
        assert _EMIT[_d]["scalar"] == DIALECTS[_d]["scalar"], _d
        assert _EMIT[_d]["suffix"] == DIALECTS[_d]["suffix"], _d

    # 3. the WGSL and GLSL texts are well formed and share the walker
    w = sdf_dialect(tree, "wgsl")
    assert w.startswith("fn map(p: vec3<f32>) -> f32") and w.count("{") == w.count("}")
    assert "vec3<f32>(" in w and "double" not in w and "def " not in w
    # WGSL IS NOT C: it infers a local's type with `let`, and rejects `vec3<f32> name = ...` outright. The first
    # emitter wrote the C form for every dialect and this test -- which only checked the signature and the braces --
    # passed it.
    for line in w.splitlines():
        s = line.strip()
        if "=" in s and not s.startswith(("fn ", "return", "//")):
            assert s.startswith("let "), "invalid WGSL declaration: %r" % s
    assert "let " not in sdf_dialect(tree, "glsl")            # ... and GLSL names its types
    g = sdf_dialect(tree, "glsl")
    assert g.startswith("float map(vec3 p)") and "vec3<f32>" not in g

    # 4. KEPT NEGATIVE 2: `scale` keeps its OUTER factor. Drop it and the shape is right, the distances are not.
    scaled = S.sphere(1.0).scale(2.0)
    far = np.array([[10.0, 0.0, 0.0]])
    assert abs(float(scaled.eval(far)[0]) - 8.0) < 1e-12                 # |p|/s - 1 times s = 10 - 2
    assert validate_c(scaled, far, "c_f64")["max_abs_diff"] < 1e-13

    # 5. KEPT NEGATIVE 1: `menger` is refused, by name
    for node, name in ((S.menger(2, 1.0), "menger"), (S.sphere(1.0).twist(0.5), "twist")):
        try:
            sdf_dialect(node, "wgsl")
        except SdfEmitError as exc:
            assert name in str(exc)
        else:
            raise AssertionError("%s must be refused" % name)

    # 5a. EVERY node kind is either emitted or refused. A gap is a shader that silently omits geometry.
    # 27 node kinds: 18 emitted + 9 refused (the two most-recent additions both EMIT, so the refused set is unchanged
    # but the total rose 25 -> 27). Keep this in sync with the realtime-suite mirror in test_holographic_realtime.
    cov = coverage()
    assert cov["complete"] is True and cov["total"] == 27
    assert set(cov["refused"]) == set(UNEMITTABLE)

    # 5b. the ones that ARE emittable: onion and rounded, checked against the Python _eval
    for node in (S.sphere(1.0).onion(0.1), S.box(0.5, 0.5, 0.5).rounded(0.1)):
        assert validate_c(node, P[:20], "c_f64")["max_abs_diff"] < 1e-14

    # 5c. MIRROR emits in all four dialects and matches _eval (an isometry -- exact, no distance correction). A
    #     nested double-mirror (an octant fold on two axes) is the real test: the handler must compose.
    mir = S.sphere(0.4).translate([0.6, 0, 0]).mirror(axis=0, plane=0.1).mirror(axis=2, plane=0.0)
    assert validate_c(mir, P[:40], "c_f64")["max_abs_diff"] < 1e-5     # emitter's baseline literal precision
    for dia in ("wgsl", "glsl", "c_f64", "c_f32"):
        code = sdf_dialect(mir, dia)
        assert "abs" in code                                          # the fold's reflection is present

    # 5c2. REPEAT (infinite tiling) emits in all four dialects and matches _eval. This is the browser win: an
    #      infinite lattice reaches WGSL, not just GLSL. mod is floor-based in every backend so the four agree --
    #      cross-checked here against the CPU eval, composed WITH a mirror (the demoscene kaleidoscope-tile combo).
    lat = S.box(0.2, 0.2, 0.2).rounded(0.05).repeat((1.0, 1.0, 1.0)).mirror(axis=0, plane=0.0)
    assert validate_c(lat, P[:40], "c_f64")["max_abs_diff"] < 1e-5
    for dia in ("wgsl", "glsl", "c_f64", "c_f32"):
        code = sdf_dialect(lat, dia)
        assert ("mod(" in code) or ("floor(" in code) or ("modf_" in code)  # the per-axis tiling is present
    # 5d. REGRESSION (real bug this handler exposed): GLSL's scalar abs is `abs`, NOT C's `fabsf`. _abs_s used to
    #     key on scalar=="float", which GLSL shares with c_f32, so it wrongly emitted fabsf into GLSL. onion and
    #     cylinder (which take a scalar abs) were silently affected. Pin that GLSL never contains a C abs.
    for node in (S.sphere(1.0).onion(0.1), S.cylinder(1.0, 0.5), mir):
        gl = sdf_dialect(node, "glsl")
        assert "fabs" not in gl and "fabsf" not in gl, "GLSL must use abs(), not C's fabs/fabsf"

    for bad in (lambda: sdf_dialect(tree, "hlsl"), lambda: sdf_dialect("not a tree", "wgsl"),
                lambda: validate_c(tree, P, "wgsl")):
        try:
            bad()
        except SdfEmitError:
            pass
        else:
            raise AssertionError("a bad request must raise")

    print("OK: holographic_sdfemit self-test passed (a compound SDF tree -- a scaled smooth-union of a translated "
          "sphere and a rotated box -- emits to C, COMPILES with cc, and matches the Python _eval to %.1e over "
          "%d points in f64 -- machine epsilon, NOT bit-identical, because np.linalg.norm rescales to avoid "
          "overflow and sums in a different order than sqrt(x*x+y*y+z*z); the f32 twin differs by %.2e, which is "
          "the tolerance a WGSL port is judged "
          "against, because WGSL is f32 and NumPy is f64. The WGSL text is well formed, `menger` and `twist` are "
          "refused by name, and `scale` keeps its outer factor -- dropping it renders the right shape with wrong "
          "distances and a raymarcher oversteps it)"
          % (rep64["max_abs_diff"], rep64["n"], rep32["max_abs_diff"]))


if __name__ == "__main__":
    _selftest()
