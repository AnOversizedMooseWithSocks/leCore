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
    "capsule": (2, 0), "cone": (2, 0), "ellipsoid": (3, 0), "octahedron": (1, 0),
    "union": (0, 2), "intersect": (0, 2), "subtract": (0, 2), "smooth_union": (1, 2),
    "translate": (3, 1), "scale": (1, 1), "rotate": (4, 1), "repeat": (3, 1),
    "round": (1, 1), "onion": (1, 1), "displace": (2, 1), "twist": (1, 1),
    "mirror": (2, 1), "bend": (2, 1),
    "elongate": (3, 1),
    "menger": (2, 0),
}
# Domain-warp kinds whose output is NOT an exact distance (a raymarcher must take shorter steps). mirror is an
# isometry (reflection) and stays exact; bend/twist/displace stretch space and do not. `ellipsoid` has no exact
# closed-form SDF -- iq's k1*(k1-1)/k2 is a tight BOUND (never oversteps), so it is INEXACT too: correct to
# raymarch, but the emitter refuses it (a shader consumer needs the shorter-step warning we cannot bake in).
INEXACT = {"twist", "displace", "bend", "ellipsoid"}


def as_eval(sdf):
    """Return a plain callable `P:(M,D) -> distances:(M,)` for ANY of the engine's three ways of naming an SDF:

      * a node object with `.eval(P)`   -- what `sphere()`/`box()`/`parse_dsl()` build
      * a bare callable                 -- what `collide`, `emitter` and every ad-hoc lambda pass around
      * a DSL STRING, e.g. "(sphere 1.0)" -- parsed here, which is what makes an SDF consumer agent-callable:
        a callable cannot cross a JSON boundary, but its s-expression can.

    WHY: the conventions grew independently and every consumer had to know which it was holding. The evidence was
    already in the tree -- `holographic_sdf_render` wraps a callable in a throwaway `_Obj()` class purely to give
    it an `.eval`, and `sdf_normal` below simply crashed on a lambda (`'function' object has no attribute
    'eval'`). One adapter at the boundary, instead of a private shim per call site."""
    if isinstance(sdf, str):
        return to_callable(parse_dsl(sdf))
    ev = getattr(sdf, "eval", None)
    return ev if callable(ev) else sdf


def sdf_normal(sdf, P, eps=1e-3):
    """The surface normal at points P:(M,3) = the normalised gradient of the SDF, by central differences (6
    vectorised evals). WHY THIS LIVES HERE (backlog G1): the gradient is a property of the FIELD, not the
    renderer -- emission, collision, displacement, sculpting, field-effect falloff, and Walk-on-Spheres all need
    the SAME normal, so it is defined ONCE here and delegated everywhere (no drift, no six private copies). `sdf`
    is anything with an `.eval(P)` OR a bare callable (see `as_eval`).

    A ZERO gradient is possible and is NOT an error: on an SDF's medial axis (the dead centre of a slab, the axis
    of a cylinder) the central differences cancel exactly and there is no normal to return. The `+ 1e-12` below
    makes that a zero vector rather than a NaN; callers that must MOVE a point (collision resolution) have to
    detect the zero and pick an escape direction themselves -- see holographic_collide.resolve_sdf_collision."""
    P = np.asarray(P, float)
    _ev = as_eval(sdf)
    ex = np.array([eps, 0, 0]); ey = np.array([0, eps, 0]); ez = np.array([0, 0, eps])
    nx = _ev(P + ex) - _ev(P - ex)
    ny = _ev(P + ey) - _ev(P - ey)
    nz = _ev(P + ez) - _ev(P - ez)
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
    def elongate(self, hx=0.0, hy=0.0, hz=0.0):
        """Stretch this shape by pulling it apart along the axes by half-extents (`hx`,`hy`,`hz`) -- iq's
        opElongate. A sphere becomes a capsule, a box a longer box, a torus an oval track. EXACT (it splits the
        shape and inserts a straight run, no distance distortion), so it raymarches cleanly and emits to GLSL.
        The clean way to make a family of shapes from one primitive."""
        return SDF("elongate", (float(hx), float(hy), float(hz)), [self])
    def mirror(self, axis=0, plane=0.0):
        """Fold space across a plane on one axis (kaleidoscopic symmetry from abs()). A DSL-tree node so it
        round-trips to GLSL; the same warp as holographic_domain.domain_mirror, here as an authorable modifier."""
        return SDF("mirror", (float(axis), float(plane)), [self])
    def fold(self, plane=0.0):
        """Mirror all three axes about `plane` -- map the world into one octant (an 8-fold kaleidoscope). Composes
        three `mirror` nodes so it emits to GLSL like any other warp. See holographic_domain.fold."""
        return self.mirror(0, plane).mirror(1, plane).mirror(2, plane)
    def bend(self, k, axis=0):
        """Bend space by `k` radians per unit along `axis` (iq's opCheapBend) -- curl a straight beam into an arc.
        A DSL node so it round-trips to GLSL. Cheap bend: warps distance slightly (fine for silhouettes)."""
        return SDF("bend", (float(k), float(axis)), [self])

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

    def cost(self):
        """Estimate the per-ray evaluation COST of this SDF tree (W2) -- a machine-model annotation for deciding
        if a scene is cheap enough to raymarch in real time. Returns a dict: `alu` (approximate arithmetic ops
        per map() call, the dominant term), `nodes` (tree size), `depth` (nesting), `iterative` (True if it
        contains a menger/repeat-style loop whose cost scales with a parameter), and `verdict` (a plain-language
        band). The ALU weights are RELATIVE (a sqrt/length is ~7 flops, a trig call ~8, a min/max ~1) -- honest
        as ratios, not absolute nanoseconds, because the real number depends on the GPU. iq's ask: know the price
        before you ship the scene."""
        # WHY these weights: a length()/sqrt is the expensive leaf op; trig (twist/bend) is worse; boolean ops are
        # nearly free. Grounded in the _eval / _GLSL_PRIM bodies -- e.g. a torus does two length()s (~14), a box
        # one length + a max (~8). Menger/repeat carry a LOOP whose body repeats `iterations` times.
        LEAF = {"sphere": 7, "box": 9, "torus": 14, "cylinder": 12, "plane": 1,
                "capsule": 8, "cone": 16, "octahedron": 12, "ellipsoid": 14}
        WARP = {"translate": 3, "scale": 2, "rotate": 12, "repeat": 6, "round": 1, "onion": 2,
                "displace": 10, "twist": 10, "mirror": 2, "bend": 12, "elongate": 6}
        COMBINE = {"union": 1, "intersect": 1, "subtract": 2, "smooth_union": 6}

        iterative = [False]

        def walk(node, depth):
            k = node.kind
            here = LEAF.get(k, WARP.get(k, COMBINE.get(k, 4)))
            if k == "menger":                                    # a real for-loop: body ~9 ALU x iterations
                iters = int(node.params[0])
                here = 9 * iters + 9
                iterative[0] = True
            if k == "repeat":
                iterative[0] = True                              # a mod per axis, cheap but domain-scaling
            sub = sum(walk(c, depth + 1) for c in node.children)
            return here + sub

        def count(node):
            return 1 + sum(count(c) for c in node.children)

        def treedepth(node):
            return 1 + (max((treedepth(c) for c in node.children), default=0))

        alu = walk(self, 0)
        n = count(self)
        d = treedepth(self)
        # verdict bands: rough, but useful. A 60fps 1080p budget is ~a few hundred ALU per map() at typical march
        # step counts; these bands assume ~64-128 steps per ray.
        if iterative[0] and alu > 120:
            verdict = "expensive (iterative/fractal) -- fine for a hero shot, budget carefully for realtime"
        elif alu <= 40:
            verdict = "cheap -- comfortable at realtime resolutions"
        elif alu <= 120:
            verdict = "moderate -- realtime at 1080p on a modern GPU, watch the march step count"
        else:
            verdict = "heavy -- likely offline or low-res realtime; consider baking or simplifying"
        return {"alu": alu, "nodes": n, "depth": d, "iterative": iterative[0], "verdict": verdict}

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


# W8 -- the primitive PACK. iq asked for the everyday SDF leaves a scene actually needs (his own articles give the
# exact closed forms). Each is an EXACT distance (not INEXACT), so they raymarch cleanly and emit to every dialect.
def capsule(h=1.0, r=0.3):
    """A capsule (a cylinder with hemispherical caps) along Y: segment from -h to +h on the Y axis, radius `r`.
    The exact distance to a line segment offset by r -- the primitive for limbs, pills, rounded rods. Returns an SDF."""
    return SDF("capsule", (h, r))


def cone(h=1.0, r=0.5):
    """A capped cone along Y: height `h` (apex at +h/2, base at -h/2), base radius `r`. iq's exact cone distance
    (a 2-D distance in the (radial, y) half-plane). Returns an SDF -- spikes, funnels, party hats."""
    return SDF("cone", (h, r))


def ellipsoid(ax=1.0, ay=0.7, az=0.5):
    """An ellipsoid with semi-axes (`ax`,`ay`,`az`). Uses iq's BOUNDED APPROXIMATION k1*(k1-1)/k2 -- the ellipsoid
    has no exact closed-form SDF, but this is a tight bound that raymarches correctly (never oversteps). Returns
    an SDF. Marked APPROX so a caller knows to step conservatively near it."""
    return SDF("ellipsoid", (ax, ay, az))


def octahedron(s=1.0):
    """A regular octahedron of 'radius' `s` (vertex distance along each axis). iq's exact octahedron distance.
    Returns an SDF -- crystals, gems, dice, the dual of the cube. Exact, emits to every dialect."""
    return SDF("octahedron", (s,))


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
    if k == "capsule":             # exact distance to a Y-axis segment [-h,h], inflated by r (iq's sdCapsule)
        h, r = p
        py = np.clip(P[:, 1], -h, h)                              # nearest point on the segment (only Y varies)
        d = P.copy(); d[:, 1] = P[:, 1] - py
        return np.linalg.norm(d, axis=1) - r
    if k == "cone":                # capped cone along Y, iq's exact 2-D form in the (radial, y) half-plane
        h, r = p
        qr = np.linalg.norm(P[:, [0, 2]], axis=1)                # radial distance from the Y axis
        # work in 2-D q=(qr, y). Cone from apex (0, h/2) to base rim (r, -h/2).
        y = P[:, 1]
        q2 = np.stack([qr, y], axis=1)
        # tip and base points of the slanted edge
        k1 = np.array([r, -h / 2.0])
        k2 = np.array([r, -h / 2.0]) - np.array([r, h])          # direction reference; use iq's sdCappedCone form
        # ca: distance to the caps; cb: distance to the side; combine with sign
        ca = np.stack([qr - np.minimum(qr, np.where(y < 0, r, 0.0)), np.abs(y) - h / 2.0], axis=1)
        e = k1 - np.array([0.0, h / 2.0])                         # slant edge vector (rim minus apex)
        t = np.clip(((q2 - np.array([0.0, h / 2.0])) @ e) / (e @ e), 0.0, 1.0)
        cb = (q2 - np.array([0.0, h / 2.0])) - t[:, None] * e
        s = np.where((cb[:, 0] < 0) & (ca[:, 1] < 0), -1.0, 1.0)
        return s * np.sqrt(np.minimum(np.sum(ca * ca, axis=1), np.sum(cb * cb, axis=1)))
    if k == "ellipsoid":           # iq's bounded ellipsoid approximation k1*(k1-1)/k2 (no exact SDF exists)
        rr = np.array(p)
        k1 = np.linalg.norm(P / rr, axis=1)
        k2 = np.linalg.norm(P / (rr * rr), axis=1)
        return k1 * (k1 - 1.0) / (k2 + 1e-12)
    if k == "octahedron":          # iq's exact regular octahedron
        s = p[0]
        pabs = np.abs(P)
        m = pabs[:, 0] + pabs[:, 1] + pabs[:, 2] - s
        out = np.empty(len(P))
        # iq's branch: pick the face region, else fall back to the plane distance
        for axis in range(3):
            pass
        # vectorised version of iq's sdOctahedron
        px, py, pz = pabs[:, 0], pabs[:, 1], pabs[:, 2]
        cond1 = 3.0 * px < m
        cond2 = 3.0 * py < m
        cond3 = 3.0 * pz < m
        q = np.where(cond1[:, None], np.stack([px, py, pz], axis=1),
             np.where(cond2[:, None], np.stack([py, pz, px], axis=1),
              np.where(cond3[:, None], np.stack([pz, px, py], axis=1), np.full((len(P), 3), np.nan))))
        kk = np.clip(0.5 * (q[:, 2] - q[:, 1] + s), 0.0, s)
        planar = m * 0.57735027                                   # 1/sqrt(3): distance when no face region matches
        edge = np.linalg.norm(np.stack([q[:, 0], q[:, 1] - s + kk, q[:, 2] - kk], axis=1), axis=1)
        return np.where(np.isnan(q[:, 0]), planar, edge)
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
    if k == "mirror":
        axis, plane = int(p[0]), p[1]
        q = P.copy()
        q[:, axis] = plane + np.abs(P[:, axis] - plane)     # reflect the far side onto the near side
        return ch[0].eval(q)
    if k == "elongate":            # iq's opElongate: split the shape, insert a straight run along each axis. EXACT.
        h = np.array(p)
        q = P - np.clip(P, -h, h)                            # subtract the clamped part -> a "hole" of size 2h
        inner = ch[0].eval(q)
        # the correction handles the interior of the stretched region (all three |q|==0 there)
        return inner + np.minimum(np.max(q, axis=1), 0.0)
    if k == "bend":
        kk, axis = p[0], int(p[1])
        # rotate the OTHER two axes by an angle proportional to position along `axis` (a bend, not a spin)
        a, b = (1, 2) if axis == 0 else (0, 2) if axis == 1 else (0, 1)
        ang = kk * P[:, axis]
        c, s = np.cos(ang), np.sin(ang)
        q = P.copy()
        q[:, a] = c * P[:, a] - s * P[:, b]
        q[:, b] = s * P[:, a] + c * P[:, b]
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
    "capsule":  "float sdCapsule(vec3 p, float h, float r){ p.y-=clamp(p.y,-h,h); return length(p)-r; }",
    "cone":     "float sdCone(vec3 p, float h, float r){ vec2 q=vec2(length(p.xz), p.y); vec2 tip=vec2(0.0,h*0.5); vec2 e=vec2(r,-h*0.5)-tip; vec2 ca=vec2(q.x-min(q.x,(q.y<0.0)?r:0.0), abs(q.y)-h*0.5); float t=clamp(dot(q-tip,e)/dot(e,e),0.0,1.0); vec2 cb=q-tip-e*t; float s=((cb.x<0.0)&&(ca.y<0.0))?-1.0:1.0; return s*sqrt(min(dot(ca,ca),dot(cb,cb))); }",
    "octahedron": "float sdOcta(vec3 p, float s){ p=abs(p); float m=p.x+p.y+p.z-s; vec3 q; if(3.0*p.x<m)q=p.xyz; else if(3.0*p.y<m)q=p.yzx; else if(3.0*p.z<m)q=p.zxy; else return m*0.57735027; float k=clamp(0.5*(q.z-q.y+s),0.0,s); return length(vec3(q.x,q.y-s+k,q.z-k)); }",
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

    if k in ("sphere", "box", "torus", "cylinder", "plane", "capsule", "cone", "octahedron"):
        helpers[k] = _GLSL_PRIM[k]
        if k == "sphere":   return stmts, f"sdSphere({pvar},{p[0]:.6g})"
        if k == "box":      return stmts, f"sdBox({pvar},vec3({p[0]:.6g},{p[1]:.6g},{p[2]:.6g}))"
        if k == "torus":    return stmts, f"sdTorus({pvar},{p[0]:.6g},{p[1]:.6g})"
        if k == "cylinder": return stmts, f"sdCyl({pvar},{p[0]:.6g},{p[1]:.6g})"
        if k == "plane":    return stmts, f"sdPlane({pvar},{p[0]:.6g})"
        if k == "capsule":  return stmts, f"sdCapsule({pvar},{p[0]:.6g},{p[1]:.6g})"
        if k == "cone":     return stmts, f"sdCone({pvar},{p[0]:.6g},{p[1]:.6g})"
        if k == "octahedron": return stmts, f"sdOcta({pvar},{p[0]:.6g})"
        if k == "capsule":  return stmts, f"sdCapsule({pvar},{p[0]:.6g},{p[1]:.6g})"
        if k == "cone":     return stmts, f"sdCone({pvar},{p[0]:.6g},{p[1]:.6g})"
        if k == "octahedron": return stmts, f"sdOcta({pvar},{p[0]:.6g})"

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
    if k == "mirror":
        # fold across a plane on one axis: q.<axis> = plane + abs(p.<axis> - plane)  (the kaleidoscope abs())
        axis, plane = int(p[0]), p[1]
        comp = ("x", "y", "z")[axis]
        q = newvar("q")
        stmts.append(f"vec3 {q}={pvar}; {q}.{comp}={plane:.6g}+abs({pvar}.{comp}-{plane:.6g});")
        sc, ec = _emit_body(ch[0], q, ctr, helpers); return stmts + sc, ec
    if k == "bend":
        # rotate the two axes other than `axis` by an angle proportional to position along `axis`
        kk, axis = p[0], int(p[1])
        a, b = ((1, 2) if axis == 0 else (0, 2) if axis == 1 else (0, 1))
        ca, cb, cc = ("x", "y", "z")[a], ("x", "y", "z")[b], ("x", "y", "z")[axis]
        q = newvar("q"); an = f"a{ctr[0]}"; ctr[0] += 1
        stmts.append(f"float {an}={kk:.6g}*{pvar}.{cc}; vec3 {q}={pvar}; "
                     f"{q}.{ca}=cos({an})*{pvar}.{ca}-sin({an})*{pvar}.{cb}; "
                     f"{q}.{cb}=sin({an})*{pvar}.{ca}+cos({an})*{pvar}.{cb};")
        sc, ec = _emit_body(ch[0], q, ctr, helpers); return stmts + sc, ec
    if k == "elongate":
        # iq's opElongate: q = p - clamp(p, -h, h); dist = child(q) + min(max(q.x,q.y,q.z), 0.0). EXACT stretch.
        hx, hy, hz = p
        q = newvar("q")
        stmts.append(f"vec3 {q}={pvar}-clamp({pvar},vec3(-{hx:.6g},-{hy:.6g},-{hz:.6g}),"
                     f"vec3({hx:.6g},{hy:.6g},{hz:.6g}));")
        sc, ec = _emit_body(ch[0], q, ctr, helpers)
        return stmts + sc, f"({ec}+min(max({q}.x,max({q}.y,{q}.z)),0.0))"
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

    # (1b) W8 PRIMITIVE PACK: capsule / cone / octahedron are EXACT (surface distance ~0, sign correct);
    #      ellipsoid is iq's bounded APPROX (0 on surface, right sign away from the centre degeneracy).
    cap = capsule(1.0, 0.3)
    assert abs(cap.eval([[0.3, 0.0, 0.0]])[0]) < 1e-9          # on the tube surface -> 0
    assert cap.eval([[0.0, 0.0, 0.0]])[0] < 0 < cap.eval([[0.0, 2.0, 0.0]])[0]   # inside/outside
    oct_ = octahedron(1.0)
    assert abs(oct_.eval([[1.0, 0.0, 0.0]])[0]) < 1e-9        # a vertex is on the surface
    assert oct_.eval([[0.0, 0.0, 0.0]])[0] < 0                # centre inside
    cn = cone(1.0, 0.5)
    assert cn.eval([[0.0, -0.3, 0.0]])[0] < 0 < cn.eval([[3.0, 3.0, 0.0]])[0]
    el = ellipsoid(1.0, 0.7, 0.5)
    assert abs(el.eval([[1.0, 0.0, 0.0]])[0]) < 1e-6          # on the surface along x -> 0
    assert el.eval([[0.4, 0.0, 0.0]])[0] < 0 < el.eval([[2.0, 0.0, 0.0]])[0]
    # the three exact ones EMIT to GLSL (the Shadertoy path); ellipsoid is INEXACT and refused there.
    for prim, fn in ((cap, "sdCapsule"), (cn, "sdCone"), (oct_, "sdOcta")):
        assert fn in prim.to_glsl()

    # (1c) W9 ELONGATE: stretching a sphere along an axis is EXACT -- the end cap and the side both sit on the
    #      surface, the interior of the run is inside, and it emits to GLSL (a clamp warp).
    el_s = sphere(0.5).elongate(1.0, 0.0, 0.0)
    assert abs(el_s.eval([[1.5, 0.0, 0.0]])[0]) < 1e-9        # end cap on the surface (0.5 past the +1 run)
    assert abs(el_s.eval([[0.0, 0.5, 0.0]])[0]) < 1e-9        # side on the surface
    assert el_s.eval([[0.5, 0.0, 0.0]])[0] < 0               # inside the straight run
    assert "clamp(" in el_s.to_glsl()

    # (1d) W2 scene.cost(): a bare sphere is cheap, a menger is iterative + pricier, and a compound scene costs
    #      MORE than any of its parts (the walk accumulates). The numbers are relative ALU, not nanoseconds.
    assert sphere(1.0).cost()["alu"] < menger(3, 1.0).cost()["alu"]      # a fractal costs more than a sphere
    assert menger(3, 1.0).cost()["iterative"] is True
    compound_cost = sphere(0.5).union(box(1, 1, 1)).union(torus(1, 0.3)).cost()
    assert compound_cost["alu"] > sphere(0.5).cost()["alu"]             # the whole exceeds a part
    assert compound_cost["nodes"] == 5                                  # 3 leaves + 2 unions

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

    # (4b) DOMAIN WARPS mirror/fold/bend (DEMO-1, iq): eval works AND emits GLSL (round-trips to Shadertoy).
    #      mirror is an isometry (a reflected query is exact); fold folds all axes into one octant; bend curls.
    m0 = sphere(0.3).translate([1.0, 0, 0]).mirror(axis=0, plane=0.0)
    # the mirrored copy: a point at x=-1 sees the sphere reflected from x=+1 (distance ~0 near the mirror image)
    assert abs(m0.eval([[-1.0, 0, 0]])[0] - m0.eval([[1.0, 0, 0]])[0]) < 1e-9   # symmetric about the plane
    fld = torus(0.5, 0.15).fold(0.0).repeat([1.3, 1.3, 1.3])
    bnt = box(0.3, 1.0, 0.3).bend(0.5, axis=1)
    for warped in (m0, fld, bnt):
        g = warped.to_glsl()
        assert "map(" in g and "no GLSL" not in g                              # the whole point: it emits
    assert "abs(" in fld.to_glsl()                                             # the fold's kaleidoscope abs()
    # DSL round-trips (one source of truth drives eval, GLSL, and parse)
    assert np.allclose(fld.eval([[0.5, 0.5, 0.5]]), parse_dsl(fld.to_dsl()).eval([[0.5, 0.5, 0.5]]), atol=1e-12)

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
