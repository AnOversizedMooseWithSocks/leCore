"""Holographic SDF / shader algebra (S1): a 3D signed-distance expression tree that evaluates, composes,
represents itself holographically, and reads/writes both a compact DSL and a Shadertoy-ready GLSL shader.

WHY THIS MODULE EXISTS
----------------------
holographic_field.py already carries the demoscene LINEAGE -- but its `Field` lives on the VSA
hypersphere (it unit-normalizes every point and measures geodesic distance), which is the right space
for "SDF = brain value = density" unification, and the WRONG space for actual geometry you want to
raymarch. This module is the CARTESIAN sibling: signed-distance fields over R^3, with the same family
of operators (union, smooth-union, domain warp/repeat), built so the result is:

  * EVALUABLE  -- node.eval(P) for P:(N,3) is a vectorized distance, so the engine's existing
                 mesh_from_sdf / marching renders any tree to a watertight mesh (brain = authoritative
                 SDF; the browser is the muscle that raymarches it -- the project's as-above-so-below).
  * REPRESENTABLE -- to_tree() is the (op, *children) form that typed.tree_to_recipe encodes as ONE
                 holographic recipe vector, so a shader IS a VSA structure you can store/compose/factor.
  * INPUT/OUTPUT -- to_dsl()/parse_dsl() round-trip a compact s-expression, and to_glsl() emits a
                 complete Shadertoy fragment shader (map() + raymarch + normals + lighting). The emitted
                 shader carries its own DSL in a header comment, so a shader round-trips back to a tree.

THE DEMOSCENE MOVE
------------------
A whole object is a few primitives under a few operators -- nothing stored that can be generated. The
operators are the canon (Quilez's seat): sphere/box/torus/cylinder/plane primitives; union / intersect /
subtract; the polynomial smooth-min that rounds a seam; rigid transforms; DOMAIN REPETITION (finite
kernel -> infinite field); rounding/onion shells; and cheap displacement/twist domain warps.

HONEST SCOPE (kept negatives)
-----------------------------
  * Exact SDFs for the rigid primitives and the exact CSG ops (union/intersect/subtract are exact;
    smooth-union is the standard bounded approximation). But TWIST and DISPLACE are domain warps that
    BREAK the unit-gradient property -- they are bounded (Lipschitz) fields, not true distances, so a
    raymarcher must shorten its steps near them (we mark them, we do not pretend they are exact).
  * Non-uniform scale is NOT provided (it does not preserve a distance field); uniform scale is, with
    the d*s correction.
  * GLSL is emitted (one direction, clean). The editable canonical form is the DSL / the tree; a shader
    is "read back" via the DSL the emitter embeds, NOT by parsing arbitrary GLSL.
"""

import numpy as np


# ---------------------------------------------------------------------------
# The node. One uniform type so eval / GLSL / DSL / holographic-tree all dispatch on `kind`.
# ---------------------------------------------------------------------------

# Per-kind arity: (number of scalar params, number of child SDFs). Drives DSL parsing and validation.
ARITY = {
    "sphere": (1, 0), "box": (3, 0), "torus": (2, 0), "cylinder": (2, 0), "plane": (1, 0),
    "union": (0, 2), "intersect": (0, 2), "subtract": (0, 2), "smooth_union": (1, 2),
    "translate": (3, 1), "scale": (1, 1), "rotate": (4, 1), "repeat": (3, 1),
    "round": (1, 1), "onion": (1, 1), "displace": (2, 1), "twist": (1, 1),
    "menger": (2, 0),
}
# Domain-warp kinds whose output is NOT an exact distance (a raymarcher must take shorter steps).
INEXACT = {"twist", "displace"}


def sdf_normal(sdf, P, eps=1e-3):
    """The surface normal at points P:(M,3) = the normalised gradient of the SDF, by central differences (6
    vectorised evals). WHY THIS LIVES HERE (backlog G1): the gradient is a property of the FIELD, not the
    renderer -- emission, collision, displacement, sculpting, field-effect falloff, and Walk-on-Spheres all need
    the SAME normal, so it is defined ONCE here and delegated everywhere (no drift, no six private copies). `sdf`
    is anything with an `.eval(P)`."""
    P = np.asarray(P, float)
    ex = np.array([eps, 0, 0]); ey = np.array([0, eps, 0]); ez = np.array([0, 0, eps])
    nx = sdf.eval(P + ex) - sdf.eval(P - ex)
    ny = sdf.eval(P + ey) - sdf.eval(P - ey)
    nz = sdf.eval(P + ez) - sdf.eval(P - ez)
    N = np.stack([nx, ny, nz], axis=1)
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)


class SDF:
    """A node in a signed-distance expression tree: `kind`, scalar `params`, and child SDFs."""

    def __init__(self, kind, params=(), children=()):
        if kind not in ARITY:
            raise ValueError(f"unknown SDF kind: {kind}")
        npar, nch = ARITY[kind]
        self.params = tuple(float(p) for p in params)
        self.children = list(children)
        if len(self.params) != npar or len(self.children) != nch:
            raise ValueError(f"{kind} needs {npar} params and {nch} children, "
                             f"got {len(self.params)} and {len(self.children)}")
        self.kind = kind

    # ----- evaluation: vectorized distance over P:(N,3) -----------------------------------------
    def eval(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        return _eval(self, P)

    # ----- combinators (operator sugar so trees read like math) ---------------------------------
    def union(self, other):           return SDF("union", (), [self, other])
    def intersect(self, other):       return SDF("intersect", (), [self, other])
    def subtract(self, other):        return SDF("subtract", (), [self, other])
    def smooth_union(self, other, k=0.3): return SDF("smooth_union", (k,), [self, other])
    def translate(self, t):           return SDF("translate", tuple(t), [self])
    def scale(self, s):               return SDF("scale", (s,), [self])
    def rotate(self, axis, angle):    return SDF("rotate", (axis[0], axis[1], axis[2], angle), [self])
    def repeat(self, period):         return SDF("repeat", tuple(period), [self])
    def rounded(self, r):             return SDF("round", (r,), [self])
    def onion(self, thickness):       return SDF("onion", (thickness,), [self])
    def displace(self, amount, freq): return SDF("displace", (amount, freq), [self])
    def twist(self, k):               return SDF("twist", (k,), [self])

    # ----- holographic representation: the (op, *children) tree typed.tree_to_recipe consumes ----
    def to_tree(self):
        """A nested tuple where the op name folds in the params (e.g. 'sphere(1.0)') so a leaf/op is a
        single symbol -- exactly the (op, child0, ...) shape encode_tree/tree_to_recipe expect."""
        tag = self.kind + "(" + ",".join(f"{p:.6g}" for p in self.params) + ")"
        if not self.children:
            return tag                                  # a primitive is a leaf symbol
        return tuple([tag] + [c.to_tree() for c in self.children])

    # ----- text I/O ------------------------------------------------------------------------------
    def to_dsl(self):
        """A compact s-expression: (kind p0 p1 ... child0 child1 ...). Round-trips via parse_dsl."""
        inner = " ".join([self.kind] + [f"{p:.6g}" for p in self.params]
                         + [c.to_dsl() for c in self.children])
        return "(" + inner + ")"

    def to_glsl(self, name="map"):
        """Emit a complete Shadertoy-ready fragment shader for this SDF (see _emit_shader)."""
        return _emit_shader(self, name=name)


# ---------------------------------------------------------------------------
# Constructors (the primitive leaves).
# ---------------------------------------------------------------------------

def sphere(r=1.0):
    """A sphere of radius `r`, centred at the origin. Returns an SDF you can transform (translate/rotate/scale) and
    combine with union/intersect/subtract. The simplest primitive leaf."""
    return SDF("sphere", (r,))


def box(bx=1.0, by=1.0, bz=1.0):
    """An axis-aligned box with half-extents (bx, by, bz) centred at the origin -- so the box spans [-bx, bx] on x,
    etc. Returns an SDF. Combine with union/intersect/subtract to build solids."""
    return SDF("box", (bx, by, bz))


def torus(R=1.0, r=0.3):
    """A torus in the XZ plane: `R` is the ring radius (centre to tube centre), `r` the tube radius. Returns an SDF."""
    return SDF("torus", (R, r))


def cylinder(h=1.0, r=0.5):
    """A capped cylinder of half-height `h` and radius `r`, axis along Y, centred at the origin. Returns an SDF."""
    return SDF("cylinder", (h, r))


def plane(h=0.0):
    """An infinite ground plane at height y = `h` (points above are outside). Returns an SDF -- handy as a floor."""
    return SDF("plane", (h,))


def menger(iterations=3, size=1.0):
    """The Menger sponge: the classic recursive fractal cube, carved `iterations` deep at the given `size`. Returns an
    SDF -- an example of rich geometry from a tiny deterministic rule."""
    return SDF("menger", (iterations, size))


# ---------------------------------------------------------------------------
# Evaluation handlers (vectorized). Primitives read P; ops recurse into children.
# ---------------------------------------------------------------------------

def _rot_matrix(axis, angle):
    axis = np.asarray(axis, float); axis = axis / (np.linalg.norm(axis) or 1.0)
    x, y, z = axis; c, s, t = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array([[t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
                     [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
                     [t*x*z - s*y, t*y*z + s*x, t*z*z + c]])


def _eval(node, P):
    k, p, ch = node.kind, node.params, node.children
    if k == "sphere":
        return np.linalg.norm(P, axis=1) - p[0]
    if k == "box":
        q = np.abs(P) - np.array(p)
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)
    if k == "torus":
        R, r = p
        xz = np.linalg.norm(P[:, [0, 2]], axis=1) - R
        return np.linalg.norm(np.stack([xz, P[:, 1]], axis=1), axis=1) - r
    if k == "cylinder":            # capped cylinder along y, half-height h, radius r
        h, r = p
        d_xz = np.linalg.norm(P[:, [0, 2]], axis=1) - r
        d_y = np.abs(P[:, 1]) - h
        dx = np.maximum(d_xz, 0.0); dy = np.maximum(d_y, 0.0)
        return np.minimum(np.maximum(d_xz, d_y), 0.0) + np.sqrt(dx * dx + dy * dy)
    if k == "plane":
        return P[:, 1] - p[0]
    if k == "menger":          # Inigo Quilez's recursive Menger sponge: a box minus crosses at every scale
        iters, size = int(p[0]), p[1]
        q = np.abs(P) - size
        d = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)
        s = 1.0
        for _ in range(iters):
            a = (P * s) % 2.0 - 1.0
            s *= 3.0
            r = np.abs(1.0 - 3.0 * np.abs(a))
            da = np.maximum(r[:, 0], r[:, 1]); db = np.maximum(r[:, 1], r[:, 2]); dc = np.maximum(r[:, 2], r[:, 0])
            cross = (np.minimum(da, np.minimum(db, dc)) - 1.0) / s
            d = np.maximum(d, cross)        # subtract the cross (carve the holes)
        return d
    if k == "union":
        return np.minimum(ch[0].eval(P), ch[1].eval(P))
    if k == "intersect":
        return np.maximum(ch[0].eval(P), ch[1].eval(P))
    if k == "subtract":
        return np.maximum(ch[0].eval(P), -ch[1].eval(P))
    if k == "smooth_union":
        a, b = ch[0].eval(P), ch[1].eval(P); kk = p[0]
        h = np.clip(0.5 + 0.5 * (b - a) / kk, 0.0, 1.0)
        return b * (1 - h) + a * h - kk * h * (1 - h)
    if k == "translate":
        return ch[0].eval(P - np.array(p))
    if k == "scale":
        s = p[0]
        return ch[0].eval(P / s) * s            # distance scales with the field
    if k == "rotate":
        Rm = _rot_matrix(p[:3], p[3])
        return ch[0].eval(P @ Rm)               # rotate the query point by R^-1 = R^T (R @ P columns)
    if k == "repeat":
        c = np.array(p)
        q = P.copy()
        for ax in range(3):
            if c[ax] > 0:
                q[:, ax] = (P[:, ax] + 0.5 * c[ax]) % c[ax] - 0.5 * c[ax]
        return ch[0].eval(q)
    if k == "round":
        return ch[0].eval(P) - p[0]
    if k == "onion":
        return np.abs(ch[0].eval(P)) - p[0]
    if k == "displace":
        amount, freq = p
        d = ch[0].eval(P)
        w = np.sin(freq * P[:, 0]) * np.sin(freq * P[:, 1]) * np.sin(freq * P[:, 2])
        return d + amount * w
    if k == "twist":
        kk = p[0]
        ang = kk * P[:, 1]
        c, s = np.cos(ang), np.sin(ang)
        q = P.copy()
        q[:, 0] = c * P[:, 0] - s * P[:, 2]
        q[:, 2] = s * P[:, 0] + c * P[:, 2]
        return ch[0].eval(q)
    raise ValueError(f"no eval for {k}")


def to_callable(node):
    """Wrap an SDF tree as a plain `sdf(P)->dist` callable for mesh_from_sdf / marching."""
    return lambda P: node.eval(P)


# ---------------------------------------------------------------------------
# DSL parsing (the inverse of to_dsl).
# ---------------------------------------------------------------------------

def _tokenize(s):
    return s.replace("(", " ( ").replace(")", " ) ").split()


def parse_dsl(text):
    """Parse a (kind p0 ... child0 ...) s-expression back into an SDF tree."""
    toks = _tokenize(text)
    pos = [0]

    def parse():
        if toks[pos[0]] != "(":
            raise ValueError(f"expected '(' at token {pos[0]}")
        pos[0] += 1                                   # consume '('
        kind = toks[pos[0]]; pos[0] += 1
        npar, nch = ARITY[kind]
        params = []
        for _ in range(npar):
            params.append(float(toks[pos[0]])); pos[0] += 1
        children = [parse() for _ in range(nch)]
        if toks[pos[0]] != ")":
            raise ValueError(f"expected ')' closing {kind}, got {toks[pos[0]]}")
        pos[0] += 1                                   # consume ')'
        return SDF(kind, params, children)

    return parse()


# ---------------------------------------------------------------------------
# GLSL emit: a complete Shadertoy fragment shader.
# ---------------------------------------------------------------------------

# GLSL source for each primitive's distance function (Inigo Quilez's canonical set).
_GLSL_PRIM = {
    "sphere":   "float sdSphere(vec3 p, float r){ return length(p)-r; }",
    "box":      "float sdBox(vec3 p, vec3 b){ vec3 q=abs(p)-b; return length(max(q,0.0))+min(max(q.x,max(q.y,q.z)),0.0); }",
    "torus":    "float sdTorus(vec3 p, float R, float r){ vec2 q=vec2(length(p.xz)-R,p.y); return length(q)-r; }",
    "cylinder": "float sdCyl(vec3 p, float h, float r){ vec2 d=vec2(length(p.xz)-r, abs(p.y)-h); return min(max(d.x,d.y),0.0)+length(max(d,0.0)); }",
    "plane":    "float sdPlane(vec3 p, float h){ return p.y-h; }",
    "smin":     "float opSmin(float a, float b, float k){ float h=clamp(0.5+0.5*(b-a)/k,0.0,1.0); return mix(b,a,h)-k*h*(1.0-h); }",
}


def _menger_glsl(iters, size):
    """Generate a GLSL helper for an `iters`-deep Menger sponge of half-size `size` (a real for-loop)."""
    return (f"float sdMenger{iters}(vec3 p){{\n"
            f"  vec3 q=abs(p)-vec3({size:.6g}); float d=length(max(q,0.0))+min(max(q.x,max(q.y,q.z)),0.0);\n"
            f"  float s=1.0;\n"
            f"  for(int m=0;m<{iters};m++){{\n"
            f"    vec3 a=mod(p*s,2.0)-1.0; s*=3.0; vec3 r=abs(1.0-3.0*abs(a));\n"
            f"    float da=max(r.x,r.y), db=max(r.y,r.z), dc=max(r.z,r.x);\n"
            f"    float c=(min(da,min(db,dc))-1.0)/s; d=max(d,c);\n"
            f"  }}\n  return d;\n}}")


def _emit_body(node, pvar, ctr, helpers):
    """Return (statements, distance_expr) for `node` at point variable `pvar`. `helpers` is a dict
    {fn_name: glsl_source} accumulating the helper functions this tree needs."""
    k, p, ch = node.kind, node.params, node.children
    stmts = []

    def newvar(prefix):
        ctr[0] += 1
        return f"{prefix}{ctr[0]}"

    if k in ("sphere", "box", "torus", "cylinder", "plane"):
        helpers[k] = _GLSL_PRIM[k]
        if k == "sphere":   return stmts, f"sdSphere({pvar},{p[0]:.6g})"
        if k == "box":      return stmts, f"sdBox({pvar},vec3({p[0]:.6g},{p[1]:.6g},{p[2]:.6g}))"
        if k == "torus":    return stmts, f"sdTorus({pvar},{p[0]:.6g},{p[1]:.6g})"
        if k == "cylinder": return stmts, f"sdCyl({pvar},{p[0]:.6g},{p[1]:.6g})"
        if k == "plane":    return stmts, f"sdPlane({pvar},{p[0]:.6g})"

    if k == "menger":
        iters = int(p[0])
        helpers[f"menger{iters}"] = _menger_glsl(iters, p[1])
        return stmts, f"sdMenger{iters}({pvar})"

    if k in ("union", "intersect", "subtract", "smooth_union"):
        sa, ea = _emit_body(ch[0], pvar, ctr, helpers)
        sb, eb = _emit_body(ch[1], pvar, ctr, helpers)
        stmts += sa + sb
        if k == "union":         return stmts, f"min({ea},{eb})"
        if k == "intersect":     return stmts, f"max({ea},{eb})"
        if k == "subtract":      return stmts, f"max({ea},-({eb}))"
        helpers["smin"] = _GLSL_PRIM["smin"]; return stmts, f"opSmin({ea},{eb},{p[0]:.6g})"

    # transforms / modifiers introduce a new point var or wrap the child's distance
    if k == "translate":
        q = newvar("q"); stmts.append(f"vec3 {q}={pvar}-vec3({p[0]:.6g},{p[1]:.6g},{p[2]:.6g});")
        sc, ec = _emit_body(ch[0], q, ctr, helpers); return stmts + sc, ec
    if k == "scale":
        q = newvar("q"); stmts.append(f"vec3 {q}={pvar}/{p[0]:.6g};")
        sc, ec = _emit_body(ch[0], q, ctr, helpers); return stmts + sc, f"(({ec})*{p[0]:.6g})"
    if k == "rotate":
        Rm = _rot_matrix(p[:3], p[3]).T            # GLSL multiplies p by R (we rotate the point)
        q = newvar("q")
        m = ",".join(f"{v:.6g}" for v in Rm.T.ravel())     # column-major for mat3
        stmts.append(f"vec3 {q}=mat3({m})*{pvar};")
        sc, ec = _emit_body(ch[0], q, ctr, helpers); return stmts + sc, ec
    if k == "repeat":
        q = newvar("q"); cx, cy, cz = p
        parts = []
        for ax, (cc, comp) in enumerate(zip((cx, cy, cz), ("x", "y", "z"))):
            if cc > 0:
                parts.append(f"{q}.{comp}=mod({pvar}.{comp}+{0.5*cc:.6g},{cc:.6g})-{0.5*cc:.6g};")
        stmts.append(f"vec3 {q}={pvar};")
        stmts += parts
        sc, ec = _emit_body(ch[0], q, ctr, helpers); return stmts + sc, ec
    if k == "round":
        sc, ec = _emit_body(ch[0], pvar, ctr, helpers); return stmts + sc, f"(({ec})-{p[0]:.6g})"
    if k == "onion":
        sc, ec = _emit_body(ch[0], pvar, ctr, helpers); return stmts + sc, f"(abs({ec})-{p[0]:.6g})"
    if k == "displace":
        sc, ec = _emit_body(ch[0], pvar, ctr, helpers)
        amount, freq = p
        w = f"(sin({freq:.6g}*{pvar}.x)*sin({freq:.6g}*{pvar}.y)*sin({freq:.6g}*{pvar}.z))"
        return stmts + sc, f"(({ec})+{amount:.6g}*{w})"
    if k == "twist":
        q = newvar("q"); kk = p[0]
        stmts.append(f"float a{ctr[0]}={kk:.6g}*{pvar}.y; "
                     f"vec3 {q}=vec3(cos(a{ctr[0]})*{pvar}.x-sin(a{ctr[0]})*{pvar}.z,{pvar}.y,"
                     f"sin(a{ctr[0]})*{pvar}.x+cos(a{ctr[0]})*{pvar}.z);")
        sc, ec = _emit_body(ch[0], q, ctr, helpers); return stmts + sc, ec
    raise ValueError(f"no GLSL for {k}")


def _emit_shader(node, name="map"):
    """A full Shadertoy fragment shader: helper fns, the map(), and a standard raymarch + normal + light."""
    helpers = {}
    stmts, dexpr = _emit_body(node, "p", [0], helpers)
    helper_src = "\n".join(helpers[h_name] for h_name in sorted(helpers))   # deterministic order
    body = "\n    ".join(stmts + [f"return {dexpr};"])
    warn = "// NOTE: contains a domain warp (twist/displace) -- not an exact SDF; shorten ray steps.\n" \
        if any(_k in node_kinds(node) for _k in INEXACT) else ""
    return f"""// Generated by holostuff holographic_sdf -- a demoscene SDF as code.
// DSL: {node.to_dsl()}
{warn}{helper_src}

float {name}(vec3 p){{
    {body}
}}

vec3 calcNormal(vec3 p){{
    vec2 e=vec2(0.001,0.0);
    return normalize(vec3({name}(p+e.xyy)-{name}(p-e.xyy),
                         {name}(p+e.yxy)-{name}(p-e.yxy),
                         {name}(p+e.yyx)-{name}(p-e.yyx)));
}}

void mainImage(out vec4 fragColor, in vec2 fragCoord){{
    vec2 uv=(fragCoord-0.5*iResolution.xy)/iResolution.y;
    vec3 ro=vec3(0.0,0.0,4.0), rd=normalize(vec3(uv,-1.5));
    float t=0.0; vec3 col=vec3(0.04);
    for(int i=0;i<96;i++){{
        vec3 p=ro+rd*t; float d={name}(p);
        if(d<0.001){{ vec3 n=calcNormal(p); float dif=clamp(dot(n,normalize(vec3(0.8,0.7,0.6))),0.0,1.0);
            col=vec3(0.2+0.8*dif); break; }}
        t+=d; if(t>20.0) break;
    }}
    fragColor=vec4(col,1.0);
}}
"""


def node_kinds(node):
    """The set of kinds used anywhere in the tree (for the inexact-warp warning and for tests)."""
    out = {node.kind}
    for c in node.children:
        out |= node_kinds(c)
    return out


# ---------------------------------------------------------------------------

def _selftest():
    # (1) PRIMITIVES are correct distances on known points.
    s = sphere(1.0)
    assert abs(s.eval([[2, 0, 0]])[0] - 1.0) < 1e-9            # outside by 1
    assert abs(s.eval([[0, 0, 0]])[0] + 1.0) < 1e-9            # inside by 1 (-1)
    b = box(1, 1, 1)
    assert abs(b.eval([[2, 0, 0]])[0] - 1.0) < 1e-9
    t = torus(1.0, 0.25)
    assert abs(t.eval([[1.0, 0.0, 0.0]])[0] + 0.25) < 1e-9     # on the ring centerline -> -r

    # (2) CSG ops: union is the min; subtract carves.
    a, c = sphere(1.0), sphere(1.0).translate([1.5, 0, 0])
    u = a.union(c)
    assert abs(u.eval([[0.75, 0, 0]])[0] - min(a.eval([[0.75, 0, 0]])[0], c.eval([[0.75, 0, 0]])[0])) < 1e-12

    # (3) SMOOTH_UNION is creaseless: less curvature along the seam than a hard union.
    hard = SDF("union", (), [a, c]); soft = a.smooth_union(c, 0.4)
    xs = np.linspace(0.0, 1.5, 60)[:, None]
    P = np.hstack([xs, np.zeros((60, 1)), np.zeros((60, 1))])
    kink_hard = float(np.max(np.abs(np.diff(hard.eval(P), 2))))
    kink_soft = float(np.max(np.abs(np.diff(soft.eval(P), 2))))
    assert kink_soft < kink_hard, f"smooth_union should be less creased: {kink_soft:.4f} !< {kink_hard:.4f}"

    # (4) DOMAIN REPETITION tiles: value at p equals value at p + period.
    rep = sphere(0.3).repeat([2.0, 0.0, 0.0])
    assert abs(rep.eval([[0.4, 0, 0]])[0] - rep.eval([[2.4, 0, 0]])[0]) < 1e-9

    # (5) renders to a watertight mesh through the existing bridge (a sphere -> closed surface).
    from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec
    vals, axes = sample_field(to_callable(sphere(0.6)), ((-1, -1, -1), (1, 1, 1)), 24)
    mesh = marching_tetrahedra_vec(vals, axes, 0.0)
    assert mesh.n_faces > 0 and mesh.is_manifold()

    # (6) DSL round-trips: parse(to_dsl(tree)) evaluates identically.
    tree = a.smooth_union(c, 0.4).translate([0, 0.2, 0]).rounded(0.05)
    back = parse_dsl(tree.to_dsl())
    Q = np.random.default_rng(0).uniform(-2, 2, (50, 3))
    assert np.allclose(tree.eval(Q), back.eval(Q), atol=1e-9), "DSL round-trip changed the field"

    # (7) holographic recipe: the tree encodes as a StructureRecipe whose op kinds match the tree.
    from holographic.misc.holographic_typed import tree_to_recipe, op_kinds
    rec = tree_to_recipe(512, 0, tree.to_tree())
    assert rec is not None and len(op_kinds(rec)) > 0

    # (8) GLSL emit is a complete, plausible shader carrying its own DSL and the right helpers.
    glsl = tree.to_glsl()
    assert "float map(vec3 p)" in glsl and "mainImage" in glsl and "opSmin" in glsl
    assert tree.to_dsl() in glsl                          # the shader round-trips via its embedded DSL

    # (9) MENGER fractal: the recursive sponge evals, carves holes (a point in a hole is OUTSIDE), and
    #     emits a GLSL loop helper.
    spng = menger(3, 1.0)
    assert spng.eval([[0.0, 0.0, 0.0]])[0] > 0            # the centre cross is carved out (outside)
    assert spng.eval([[0.95, 0.95, 0.95]])[0] < 0.2       # a corner pillar is solid/near-surface
    mglsl = spng.to_glsl()
    assert "sdMenger3(" in mglsl and "for(int m=0;m<3;m++)" in mglsl

    print("holographic_sdf selftest passed:",
          f"seam hard={kink_hard:.3f} soft={kink_soft:.3f} mesh_faces={mesh.n_faces} "
          f"glsl_chars={len(glsl)} menger_center={spng.eval([[0,0,0]])[0]:.3f} kinds={sorted(node_kinds(tree))}")


if __name__ == "__main__":
    _selftest()
