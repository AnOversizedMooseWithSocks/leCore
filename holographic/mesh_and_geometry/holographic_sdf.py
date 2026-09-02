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
    "fillet_union": (1, 2),
    "translate": (3, 1), "scale": (1, 1), "rotate": (4, 1), "repeat": (3, 1),
    "round": (1, 1), "onion": (1, 1), "displace": (2, 1), "twist": (1, 1),
    "mirror": (2, 1), "bend": (2, 1),
    "elongate": (3, 1),
    "menger": (2, 0),
    "fold_fractal": (4, 0),
    "mandelbulb": (3, 0),
}
# Domain-warp kinds whose output is NOT an exact distance (a raymarcher must take shorter steps). mirror is an
# isometry (reflection) and stays exact; bend/twist/displace stretch space and do not. `ellipsoid` has no exact
# closed-form SDF -- iq's k1*(k1-1)/k2 is a tight BOUND (never oversteps), so it is INEXACT too: correct to
# raymarch, but the emitter refuses it (a shader consumer needs the shorter-step warning we cannot bake in).
INEXACT = {"twist", "displace", "bend", "ellipsoid", "fold_fractal", "mandelbulb"}


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

    # An SDF IS a distance field, so let it be USED as one: field consumers that take a callable func(pts) ->
    # distances (mesh_from_sdf, sample_field, scatter layers) now accept the SDF object directly instead of
    # forcing callers to remember `.eval`. Additive: eval is unchanged and everything that dispatches on
    # isinstance(x, SDF) still sees an SDF (a class being callable does not change its type).
    def __call__(self, P):
        return self.eval(P)

    # ----- combinators (operator sugar so trees read like math) ---------------------------------
    def union(self, other):           return SDF("union", (), [self, other])
    def intersect(self, other):       return SDF("intersect", (), [self, other])
    def subtract(self, other):        return SDF("subtract", (), [self, other])
    def smooth_union(self, other, k=0.3): return SDF("smooth_union", (k,), [self, other])
    def fillet_union(self, other, r=0.1): return SDF("fillet_union", (r,), [self, other])
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
        COMBINE = {"union": 1, "intersect": 1, "subtract": 2, "smooth_union": 6, "fillet_union": 8}

        iterative = [False]

        def walk(node, depth):
            k = node.kind
            here = LEAF.get(k, WARP.get(k, COMBINE.get(k, 4)))
            if k == "menger":                                    # a real for-loop: body ~9 ALU x iterations
                iters = int(node.params[0])
                here = 9 * iters + 9
                iterative[0] = True
            if k == "fold_fractal":                              # box-fold + sphere-fold + scale, ~14 ALU x iters
                iters = int(node.params[0])
                here = 14 * iters + 6
                iterative[0] = True
            if k == "mandelbulb":                                # trig-heavy polar power map, ~30 ALU x iters
                iters = int(node.params[1])
                here = 30 * iters + 8
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

    # analytic-preservation contract (client S-3): an SDF node IS the analytic form. Every
    # combinator/transform method on this class returns another SDF, so the analytic description
    # survives by TYPE -- `result.preserves_analytic` is True on every node. Operations that
    # return a Mesh (mesh_from_sdf, decimators, remeshers) have crossed the boundary and the
    # analytic form is gone; a Mesh has no such attribute, and `getattr(x, "preserves_analytic",
    # False)` is the documented branch. The contract is the type; this flag makes it spellable.
    preserves_analytic = True

    def to_jit_expr(self):
        """Emit this tree as a SINGLE symbolic expression string in (x, y, z) -- the `jit_expr=`
        that unlocks render_sdf's compiled fast path (client S-5: the fast path existed but
        nothing produced its input). This is the THIRD dialect of the tree's emitter family
        (GLSL via to_glsl, WGSL/C/JS/Zig via the dialect emitters); sympy needs one expression
        rather than statements, so coordinate transforms substitute into the coordinate strings
        instead of binding temporaries.

        Supported: every exact primitive and boolean, smooth/fillet union, translate / scale /
        rotate / mirror / round / onion / elongate / repeat. REFUSED with the reason: the
        INEXACT set (twist, displace, bend, ellipsoid, fold_fractal, mandelbulb -- their fields
        are bounds, and the compiled marcher cannot be told to under-step) and menger (an
        iterative loop, not a closed form). Raises ValueError naming the offending kind.

        Verified the way the emitter family is always verified -- BOTH EXECUTED, not asserted:
        the selftest evaluates the emitted string numerically (Min/Max/Abs/sqrt/Mod mapped to
        numpy) against node.eval on random points to 1e-9."""
        return _emit_sympy(self, ("x", "y", "z"))

    def to_glsl(self, name="map", camera="fixed"):
        """Emit a complete Shadertoy-ready fragment shader for this SDF (see _emit_shader). camera="fixed" (default)
        is the classic head-on view, byte-identical to the historic output; camera="uniforms" emits an orbit camera
        driven by host-bound uAngle/uHeight/uDist uniforms (for a WebGL2 host that spins/zooms without re-emitting)."""
        return _emit_shader(self, name=name, camera=camera)


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


def fold_fractal(iterations=12, scale=2.0, min_radius=0.5, fold_limit=1.0):
    """The KALEIDOSCOPIC-IFS / MANDELBOX distance-estimator SDF -- the general 'fold engine' behind the fractal-forums
    3D fractals and the Yohei-Nishitsuji tweet-shader look. Iterate, `iterations` times: a BOX FOLD (conditional
    reflection -- reflect each coordinate outside +/-`fold_limit` back inward, `p = clamp(p,-L,L)*2 - p`), then a
    SPHERE FOLD (invert the point through nested spheres of radius `min_radius`), then a linear `scale`+translate.
    Track the running derivative so the readout is a true DISTANCE ESTIMATE the raymarcher can step on. `scale` is the
    Mandelbox constant (|scale|>1 gives the classic box; negative scales give the folded-inside-out variants);
    `min_radius` sets where the sphere fold bites; `fold_limit` is the box-fold half-extent.

    These are all CONFORMAL (angle-preserving) transforms -- which is exactly why the distance estimate stays usable
    and why the result raymarches and orbit-traps cleanly with the existing renderer. Returns an SDF that evals,
    marches to a mesh, and emits GLSL like any other. A tiny recipe (four floats) that regenerates megabytes of
    deterministic self-similar structure -- the 'determinism instead of storage' lever, as geometry.

    NOTE: like menger, this is an INEXACT distance (a distance ESTIMATE, standard for fractals) -- the raymarcher
    already steps conservatively for iterative SDFs, but a shader consumer must know to shorten steps near it."""
    return SDF("fold_fractal", (iterations, scale, min_radius, fold_limit))


def mandelbulb(power=8.0, iterations=8, bailout=2.0):
    """The MANDELBULB distance-estimator SDF (White & Nylander's polar-power fractal, the 3D Mandelbrot analogue).
    Iterate `z -> z^power + c` in SPHERICAL coordinates (raise the radius to `power`, multiply the two angles by
    `power`), where c is the query point, tracking the running derivative dr so the readout is the analytic distance
    estimate `0.5*log(r)*r/dr`. `power`=8 is the classic bulb; other powers give the 'bulb of order n'. `iterations`
    trades detail for cost; `bailout` is the escape radius. Unlike fold_fractal (a Mandelbox FOLD engine -- conditional
    reflections), this is the ESCAPE-TIME family in 3D: the same z->z^n+c that draws the Mandelbrot set, lifted to a
    triplex algebra. Returns an SDF that evals and raymarches/orbit-traps with the existing renderer.

    NOTE: INEXACT (a distance ESTIMATE, standard for escape-time fractals) -- the in-engine raymarcher steps
    conservatively; the GLSL emitter refuses it (a shader consumer must hand-tune the step size)."""
    return SDF("mandelbulb", (power, iterations, bailout))


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


def escape_time(width=256, height=256, center=(-0.5, 0.0), span=3.0, max_iter=100,
                power=2.0, julia_c=None, bounds_ratio=None, fast_square=False):
    """The 2D ESCAPE-TIME fractal FIELD -- Mandelbrot (`julia_c=None`) or Julia (`julia_c=(re,im)`), the classic
    z -> z^power + c iteration in the complex plane. Returns a (height, width) float array of SMOOTH (continuous)
    escape counts in [0, max_iter]: for each pixel, iterate until |z| exceeds 2, and record iter + the fractional
    'smooth iteration' term 1 - log2(log2(|z|)) so the bands are continuous (no staircase) -- the standard input to a
    palette. A point that never escapes gets max_iter (it is IN the set). The 2D sibling of the mandelbulb: same
    z^n+c recurrence, read as a field instead of a distance. Vectorised over the whole grid, deterministic.

    Mandelbrot: c = the pixel, z starts at 0. Julia: c = `julia_c` (fixed), z starts at the pixel. `center`/`span`
    frame the view (span = width of the window in complex units); `power`=2 is the classic set.

    `fast_square=True` replaces `np.power(z, 2.0)` with `z*z` when `power` is exactly 2 -- MEASURED 5.7x on the
    array op, and it is most of this function's cost, which is why the demo-scene sweep found this loop before it
    found anything else. IT IS OPT-IN AND IT MUST BE. The two are NOT bit-identical: np.power on a complex array
    goes through exp/log, and over 200,000 random complex128 values 57,908 of them differ, max |diff| 3.55e-15.
    This repo's rule is that a change bit-identical to 1e-12 has still flipped a creature's trajectory, so a
    silent swap is forbidden however small the delta -- the speed is real and it has to be ASKED for. Ignored
    unless power == 2; a non-integer power has no fast path."""
    cx, cy = center
    half = span * 0.5
    xs = np.linspace(cx - half, cx + half, width)
    ys = np.linspace(cy - half * height / width, cy + half * height / width, height)
    X, Y = np.meshgrid(xs, ys)
    C_grid = X + 1j * Y
    if julia_c is None:
        c = C_grid                                              # Mandelbrot: c varies per pixel
        z = np.zeros_like(C_grid)
    else:
        c = np.full_like(C_grid, complex(julia_c[0], julia_c[1]))   # Julia: c fixed, z = pixel
        z = C_grid.copy()
    out = np.full(C_grid.shape, float(max_iter))
    escaped = np.zeros(C_grid.shape, dtype=bool)
    # Resolved ONCE, outside the loop: an `if` per iteration on a value that cannot change mid-run is
    # exactly the kind of per-frame cost this sweep exists to remove.
    square_fast = bool(fast_square) and float(power) == 2.0
    for n in range(max_iter):
        live = ~escaped
        if square_fast:
            zl = z[live]
            z[live] = zl * zl + c[live]
        else:
            z[live] = np.power(z[live], power) + c[live]
        az = np.abs(z)
        now = live & (az > 2.0)
        if np.any(now):
            # smooth (continuous) escape count: n + 1 - log2(log2|z|) removes the integer-band staircase.
            out[now] = n + 1.0 - np.log2(np.maximum(np.log2(az[now]), 1e-12))
            escaped[now] = True
        if np.all(escaped):
            break
    return out


# ---------------------------------------------------------------------------
# Evaluation handlers (vectorized). Primitives read P; ops recurse into children.
# ---------------------------------------------------------------------------

def _rot_matrix(axis, angle):
    axis = np.asarray(axis, float); axis = axis / (np.linalg.norm(axis) or 1.0)
    x, y, z = axis; c, s, t = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array([[t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
                     [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
                     [t*x*z - s*y, t*y*z + s*x, t*z*z + c]])


#: The binary COMBINATOR kinds. Folding a list of N parts with any of these builds a LEFT-LEANING
#: chain of depth N, and a naive recursive eval then needs N stack frames -- which is why skinning a
#: 300-edge skeleton used to die with RecursionError at Python's default limit of 1000. The chain is
#: unrolled iteratively below, so depth is bounded by the C stack of the SHALLOW right children
#: instead of by the number of parts.
_BINARY_COMBINATORS = ("union", "intersect", "subtract", "smooth_union", "fillet_union")


def _combine(k, kk, a, b):
    """Apply one binary combinator to two already-evaluated distance arrays.

    ONE HOME for the four combinator formulas, so the iterative chain walk and the recursive dispatch
    below can never drift apart -- two copies of `smooth_union`'s polynomial that happened to match
    would be a silent-divergence bug of exactly the kind this engine keeps finding.
    """
    if k == "union":
        return np.minimum(a, b)
    if k == "intersect":
        return np.maximum(a, b)
    if k == "subtract":
        return np.maximum(a, -b)
    if k == "fillet_union":
        # EXACT constant-radius rolling-ball fillet (iq's opUnionRound), promoted from
        # holographic_fillet into the DSL so it can be COMPOSED inside a tree. The difference that
        # matters here is not the arc shape but the BOUND: ua/ub are clamped at 0, so once both
        # surfaces are further than r away the result is exactly min(a, b) -- the sharp union. A
        # smooth_union has no such cutoff and keeps depositing material at a distance, which is the
        # measured cause of webbing returning at large blend (backlog F-3).
        ua = np.maximum(kk - a, 0.0)
        ub = np.maximum(kk - b, 0.0)
        return np.maximum(kk, np.minimum(a, b)) - np.sqrt(ua * ua + ub * ub)
    h = np.clip(0.5 + 0.5 * (b - a) / kk, 0.0, 1.0)          # smooth_union (iq's polynomial smin)
    return b * (1 - h) + a * h - kk * h * (1 - h)


def _eval_chain(node, P):
    """Evaluate a left-leaning chain of binary combinators ITERATIVELY, bit-identically.

    WHY THIS IS BIT-IDENTICAL AND NOT AN APPROXIMATION. A fold builds ((a op b) op c) op d. The
    recursive evaluator computes a, then a op b, then (a op b) op c, then that op d -- left to right.
    This walks down the left spine collecting (kind, k, right_child), evaluates the base, then
    replays the ops in the SAME left-to-right order with the SAME formulas (see _combine). Identical
    operations on identical inputs in identical order, so identical floats. That matters more than it
    might seem: `smooth_union` is NOT associative (measured: rebalancing the tree moves the surface by
    ~3e-3, thousands of ULP), so any restructuring of the chain WOULD change every existing skinned
    mesh. Unrolling the evaluation changes nothing.

    Right children recurse normally -- they are shallow in a fold (a translated, rotated primitive),
    so the C stack is bounded by part complexity, not by part COUNT.
    """
    ops = []
    cur = node
    while cur.kind in _BINARY_COMBINATORS:
        ops.append((cur.kind, cur.params[0] if cur.params else 0.0, cur.children[1]))
        cur = cur.children[0]
    acc = _eval(cur, P)                                       # the base of the spine (shallow)
    for k, kk, right in reversed(ops):                        # replay outward: left-to-right order
        acc = _combine(k, kk, acc, right.eval(P))
    return acc


def _eval(node, P):
    k, p, ch = node.kind, node.params, node.children
    # A deep left spine is unrolled iteratively. Guarded by a depth probe so the ordinary shallow case
    # (two or three nested unions) takes the plain recursive path and pays nothing for this.
    if k in _BINARY_COMBINATORS:
        d, cur = 0, node
        while cur.kind in _BINARY_COMBINATORS and d < 64:
            d += 1
            cur = cur.children[0]
        if d >= 64:
            return _eval_chain(node, P)
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
    if k == "fold_fractal":    # Mandelbox / KIFS: iterate box-fold, sphere-fold, scale; track the derivative for a DE
        iters, scale, min_r, L = int(p[0]), float(p[1]), float(p[2]), float(p[3])
        min_r2 = min_r * min_r
        fixed_r2 = 1.0                                            # outer sphere-fold radius^2 (the classic Mandelbox)
        offset = P.copy()                                        # the Mandelbox adds the ORIGINAL point each step
        z = P.copy()
        dr = np.ones(P.shape[0])                                 # running derivative (scale factor of the map)
        for _ in range(iters):
            z = np.clip(z, -L, L) * 2.0 - z                     # BOX FOLD: reflect coords outside +/-L back inward
            r2 = np.sum(z * z, axis=1)                          # SPHERE FOLD: invert through nested spheres
            r2 = np.maximum(r2, 1e-12)                          # guard the inversion at the exact origin (r2 -> 0)
            m = np.ones_like(r2)
            inner = r2 < min_r2
            m = np.where(inner, fixed_r2 / min_r2, m)           # inside min radius -> linear magnification
            mid = (~inner) & (r2 < fixed_r2)
            m = np.where(mid, fixed_r2 / r2, m)                 # between -> spherical inversion
            z = z * m[:, None]
            dr = dr * m + 1.0
            z = z * scale + offset                              # linear part: scale + translate by the seed
            dr = dr * abs(scale)
        return np.linalg.norm(z, axis=1) / np.abs(dr)           # distance estimate = |z| / |dz/dp|
    if k == "mandelbulb":      # White-Nylander polar power fractal: z -> z^power + c in spherical coords, analytic DE
        power, iters, bailout = float(p[0]), int(p[1]), float(p[2])
        c = P.copy()
        z = P.copy()
        dr = np.ones(P.shape[0])                                 # running derivative for the analytic DE
        r = np.zeros(P.shape[0])
        active = np.ones(P.shape[0], dtype=bool)                 # rays still inside the bailout radius
        for _ in range(iters):
            r = np.linalg.norm(z, axis=1)
            live = active & (r < bailout) & (r > 1e-9)
            if not np.any(live):
                break
            rl = r[live]
            # spherical coords of z; raise radius to `power`, scale the two angles by `power` (the triplex power map).
            theta = np.arccos(np.clip(z[live, 2] / rl, -1.0, 1.0))
            phi = np.arctan2(z[live, 1], z[live, 0])
            dr[live] = np.power(rl, power - 1.0) * power * dr[live] + 1.0
            zr = np.power(rl, power)
            theta = theta * power
            phi = phi * power
            zn = zr[:, None] * np.stack([np.sin(theta) * np.cos(phi),
                                         np.sin(theta) * np.sin(phi),
                                         np.cos(theta)], axis=1)
            z[live] = zn + c[live]                               # + c: the escape-time recurrence
        r = np.linalg.norm(z, axis=1)
        r = np.maximum(r, 1e-9)
        return 0.5 * np.log(r) * r / np.abs(dr)                  # analytic distance estimate for z^n+c fractals
    if k in _BINARY_COMBINATORS:
        # Shallow case: recurse as before, but through the one _combine home so the formulas cannot
        # drift from the iterative path's copy.
        return _combine(k, p[0] if p else 0.0, ch[0].eval(P), ch[1].eval(P))
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


# =====================================================================================================
# ONE DOOR for shapes -- the step an agent hits before anything else in this module is usable
# =====================================================================================================
# WHY THIS EXISTS (measured, agent-authoring probe). Every primitive above shipped reachable only by
# importing this module. Asked "make a sphere", a mind returned a Lipschitz worst-view bound; asked "add a
# cube", the sky-observation capability; asked "union two shapes", a cosine palette. Ten stranger phrasings,
# ten unrelated fallbacks. The ONLY door was parse_dsl("(sphere 1.0)"), whose grammar lived in a module-level
# dict nothing surfaced -- so the one working path required already knowing the thing you were looking up.
#
# Same shape of answer as holographic_lights.make_light, deliberately: one factory keyed by a word a caller
# would type, not N sibling faculties. Two doors that do 80% of the same thing is a discoverability tax, and
# a caller who has learned one factory has learned both.

SHAPE_KINDS = {
    "sphere": (sphere, ("r",)), "ball": (sphere, ("r",)),
    "box": (box, ("bx", "by", "bz")), "cube": (box, ("bx", "by", "bz")),
    "plane": (plane, ("h",)), "floor": (plane, ("h",)), "ground": (plane, ("h",)),
    "cylinder": (cylinder, ("h", "r")), "tube": (cylinder, ("h", "r")),
    "cone": (cone, ("h", "r")),
    "capsule": (capsule, ("h", "r")), "pill": (capsule, ("h", "r")),
    "torus": (torus, ("R", "r")), "donut": (torus, ("R", "r")), "ring": (torus, ("R", "r")),
    "ellipsoid": (ellipsoid, ("ax", "ay", "az")), "egg": (ellipsoid, ("ax", "ay", "az")),
    "octahedron": (octahedron, ("s",)), "diamond": (octahedron, ("s",)),
    "menger": (menger, ("iterations", "scale")),
    "mandelbulb": (mandelbulb, ("power", "iterations", "bailout")),
}


def make_sdf_shape(kind="sphere", position=None, scale=None, rotate=None, **kw):
    """Build an SDF primitive by NAME, optionally placed -- the one door to the shapes above.

    NAMED make_sdf_shape, not `make_shape`: holographic_vision.make_shape already owns that name for a
    different job (drawing a 2-D shape image + mask). The name-collision budget MAY SHRINK AND MUST
    NEVER GROW, so the newer arrival takes the qualified name rather than spending budget on a homonym.

    `kind` is a word a caller would actually type: 'cube' and 'box' both give a box, 'floor' and 'ground'
    both give a plane, 'donut' gives a torus. Size parameters pass straight through (r, bx/by/bz, h, R, ...).

    PLACEMENT IS INCLUDED ON PURPOSE, in the order scale -> rotate -> translate. Every real use immediately
    wants "a sphere, over there", and doing it in the wrong order is a classic quiet bug: rotating after
    translating swings the object around the world origin instead of spinning it in place. Fixing the order
    here means a caller cannot get it wrong. `rotate` is (axis_x, axis_y, axis_z, radians).

    An unknown kind raises with the full sorted list rather than a bare KeyError -- guessing is what a caller
    does when it has never seen the API, and a wrong guess should teach the vocabulary."""
    key = str(kind).strip().lower()
    if key not in SHAPE_KINDS:
        raise KeyError("unknown shape kind %r -- pick one of: %s" % (kind, ", ".join(sorted(SHAPE_KINDS))))
    fn, params = SHAPE_KINDS[key]
    bad = [k for k in kw if k not in params]
    if bad:
        raise TypeError("shape %r takes %s, not %s" % (kind, list(params), bad))
    node = fn(**kw)
    if scale is not None and abs(float(scale) - 1.0) > 1e-12:
        node = node.scale(float(scale))
    if rotate is not None:
        ax, ay, az, ang = rotate
        node = node.rotate((ax, ay, az), float(ang))
    if position is not None and float(np.linalg.norm(np.asarray(position, float))) > 1e-12:
        node = node.translate(tuple(float(v) for v in position))
    return node


# One line per DSL node: what it is, and what its numeric parameters mean. The ARITY table above is the
# machine-readable half and was already correct; this is the half a caller needs to WRITE one, and without
# it the DSL is a cipher you can only read if you already know the answer.
DSL_HELP = {
    "sphere": "solid ball. params: radius",
    "box": "axis-aligned box. params: half-extent x, y, z (so it spans -bx..+bx)",
    "plane": "infinite ground plane. params: height y",
    "cylinder": "capped cylinder up the y axis. params: half-height, radius",
    "cone": "cone up the y axis. params: height, base radius",
    "capsule": "cylinder with rounded ends. params: half-height, radius",
    "torus": "donut in the xz plane. params: ring radius, tube radius",
    "ellipsoid": "stretched sphere. params: radius x, y, z",
    "octahedron": "eight-sided diamond. params: size",
    "menger": "menger sponge fractal. params: iterations, scale",
    "mandelbulb": "mandelbulb fractal. params: power, iterations, bailout",
    "fold_fractal": "kaleidoscopic folded fractal. params: iterations, scale, offset, min_radius",
    "union": "both shapes (nearest surface wins). 2 children, no params",
    "intersect": "only where both overlap. 2 children, no params",
    "subtract": "the first shape minus the second. 2 children, no params",
    "smooth_union": "union with a soft blend. params: blend radius k. 2 children",
    "fillet_union": "union with an EXACT radius-r fillet that is LOCAL -- beyond r it is the sharp union, so it cannot blend at a distance. params: fillet radius r. 2 children",
    "translate": "move a shape. params: dx, dy, dz. 1 child",
    "rotate": "spin a shape about an axis through the origin. params: axis x, y, z, radians. 1 child",
    "scale": "resize about the origin. params: factor. 1 child",
    "round": "inflate the surface, rounding every edge. params: radius. 1 child",
    "onion": "hollow shell of a solid. params: thickness. 1 child",
    "twist": "twist about the y axis. params: turns per unit. 1 child",
    "bend": "bend along an axis. params: amount, axis. 1 child",
    "mirror": "mirror across a plane. params: axis, offset. 1 child",
    "elongate": "stretch the middle without distorting the caps. params: dx, dy, dz. 1 child",
    "displace": "add a wobble to the surface. params: amplitude, frequency. 1 child",
    "repeat": "tile the shape infinitely on a grid. params: spacing x, y, z. 1 child",
}


def dsl_grammar():
    """The SDF DSL, described well enough to WRITE one -- node kinds, parameter meanings, and an example.

    parse_dsl has always been the compact way to state a whole shape tree in one string, and it was
    effectively secret: the node names and their parameter counts lived in the module-level ARITY dict, and
    nothing surfaced either. A grammar you can only use if you already know it is not a usable grammar.

    Returns {syntax, nodes: [{kind, params, children, does}], example}. Sorted primitives first, then
    modifiers, then combinators -- the order you build in."""
    rows = []
    for kind, (npar, nch) in ARITY.items():
        rows.append({"kind": kind, "params": int(npar), "children": int(nch),
                     "does": DSL_HELP.get(kind, "")})
    rows.sort(key=lambda r: (r["children"], r["kind"]))
    return {"syntax": "(kind param0 param1 ... child0 child1 ...) -- an s-expression; the inverse of node.to_dsl()",
            "nodes": rows,
            "example": "(smooth_union 0.3 (translate 0.0 0.6 0.0 (sphere 0.6)) (box 1.0 0.2 1.0))"}


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
    "ellipsoid": "float sdEllipsoid(vec3 p, vec3 r){ float k1=length(p/r); float k2=length(p/(r*r)); return k1*(k1-1.0)/(k2+1e-12); }",
    "smin":     "float opSmin(float a, float b, float k){ float h=clamp(0.5+0.5*(b-a)/k,0.0,1.0); return mix(b,a,h)-k*h*(1.0-h); }",
    "uround":   "float opUnionRound(float a, float b, float r){ float ua=max(r-a,0.0), ub=max(r-b,0.0); return max(r,min(a,b))-sqrt(ua*ua+ub*ub); }",
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


def _foldfractal_glsl(iters, scale, min_r, L):
    """GLSL helper for a `iters`-step Mandelbox/KIFS distance estimator -- the exact box-fold + sphere-fold + scale
    loop the NumPy _eval runs, translated 1:1. A distance ESTIMATE (INEXACT), so the emitted shader header warns to
    step conservatively; a Shadertoy raymarcher handles that fine."""
    fname = f"sdFold{iters}"
    return (f"float {fname}(vec3 p){{\n"
            f"  vec3 offset=p; vec3 z=p; float dr=1.0;\n"
            f"  for(int i=0;i<{iters};i++){{\n"
            f"    z=clamp(z,-{L:.6g},{L:.6g})*2.0-z;                 // box fold\n"
            f"    float r2=max(dot(z,z),1e-12);                      // sphere fold\n"
            f"    if(r2<{min_r*min_r:.6g}){{ float t={1.0/(min_r*min_r):.6g}; z*=t; dr*=t; }}\n"
            f"    else if(r2<1.0){{ float t=1.0/r2; z*=t; dr*=t; }}\n"
            f"    dr+=1.0;\n"
            f"    z=z*{scale:.6g}+offset; dr*=abs({scale:.6g});\n"
            f"  }}\n  return length(z)/abs(dr);\n}}")


def _mandelbulb_glsl(power, iters, bailout):
    """GLSL helper for a `iters`-step Mandelbulb (polar power z->z^n+c) with the analytic DE -- the 1:1 translation of
    the NumPy _eval. INEXACT (a distance estimate), so the shader header warns to step conservatively."""
    fname = f"sdBulb{iters}"
    return (f"float {fname}(vec3 pos){{\n"
            f"  vec3 z=pos; float dr=1.0; float r=0.0;\n"
            f"  for(int i=0;i<{iters};i++){{\n"
            f"    r=length(z); if(r>{bailout:.6g}) break;\n"
            f"    float theta=acos(clamp(z.z/r,-1.0,1.0)); float phi=atan(z.y,z.x);\n"
            f"    dr=pow(r,{power - 1.0:.6g})*{power:.6g}*dr+1.0;\n"
            f"    float zr=pow(r,{power:.6g}); theta*={power:.6g}; phi*={power:.6g};\n"
            f"    z=zr*vec3(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta))+pos;\n"
            f"  }}\n  return 0.5*log(max(r,1e-9))*max(r,1e-9)/abs(dr);\n}}")


def _emit_sympy(node, xyz):
    """The sympy-dialect tree walk behind SDF.to_jit_expr (see its docstring). `xyz` is the
    coordinate EXPRESSION triple at this node -- transforms recurse with substituted strings."""
    k, p, ch = node.kind, node.params, node.children
    x, y, z = xyz

    def wrap(s):
        return "(" + s + ")"
    if k == "sphere":
        return f"sqrt({x}**2 + {y}**2 + {z}**2) - {p[0]:.9g}"
    if k == "plane":
        return f"{y} - {p[0]:.9g}"
    if k == "box":
        qx, qy, qz = (f"(Abs({c}) - {e:.9g})" for c, e in zip(xyz, p))
        outside = f"sqrt(Max({qx},0)**2 + Max({qy},0)**2 + Max({qz},0)**2)"
        inside = f"Min(Max({qx}, Max({qy}, {qz})), 0)"
        return f"{outside} + {inside}"
    if k == "torus":
        return f"sqrt((sqrt({x}**2 + {z}**2) - {p[0]:.9g})**2 + {y}**2) - {p[1]:.9g}"
    if k == "cylinder":
        dr = f"(sqrt({x}**2 + {z}**2) - {p[1]:.9g})"
        dy = f"(Abs({y}) - {p[0]:.9g})"
        return (f"Min(Max({dr}, {dy}), 0) + sqrt(Max({dr},0)**2 + Max({dy},0)**2)")
    if k == "capsule":
        h, r = p
        cy = f"({y} - Min(Max({y}, {-h:.9g}), {h:.9g}))"
        return f"sqrt({x}**2 + {cy}**2 + {z}**2) - {r:.9g}"
    if k == "octahedron":
        # _eval uses iq's EXACT branchy octahedron; the tempting (|x|+|y|+|z|-s)*0.577 one-liner
        # is only the bound form and disagreed with _eval by 0.32 on the battery (measured) --
        # the emitter refuses rather than ship a shader that disagrees with the numpy field.
        raise ValueError("to_jit_expr: 'octahedron' is exact only in branchy form -- use the "
                         "numpy/GLSL paths (the single-expression form is a bound, refused)")
    if k == "cone":
        # iq's exact capped cone is long; the bound form (like ellipsoid) would be INEXACT, so
        # cone stays supported only through the numpy path -- refuse honestly here.
        raise ValueError("to_jit_expr: 'cone' has no single-expression exact form wired yet -- "
                         "use the numpy render path (or contribute the closed form)")
    if k in ("union", "intersect", "subtract"):
        a = _emit_sympy(ch[0], xyz); b = _emit_sympy(ch[1], xyz)
        if k == "union":
            return f"Min({a}, {b})"
        if k == "intersect":
            return f"Max({a}, {b})"
        return f"Max({a}, -({b}))"
    if k in ("smooth_union", "fillet_union"):
        a = _emit_sympy(ch[0], xyz); b = _emit_sympy(ch[1], xyz)
        kk = max(float(p[0]), 1e-9)
        # polynomial smin -- expressible in Min/Max/Abs, matching _eval's formula
        return (f"Min({a}, {b}) - Max({kk:.9g} - Abs(({a}) - ({b})), 0)**2 / {4*kk:.9g}")
    if k == "translate":
        nx = f"({x} - {p[0]:.9g})"; ny = f"({y} - {p[1]:.9g})"; nz = f"({z} - {p[2]:.9g})"
        return _emit_sympy(ch[0], (nx, ny, nz))
    if k == "scale":
        s = float(p[0])
        sub = _emit_sympy(ch[0], (f"({x}/{s:.9g})", f"({y}/{s:.9g})", f"({z}/{s:.9g})"))
        return f"({sub}) * {s:.9g}"
    if k == "rotate":
        # match _eval's semantics EXACTLY (probed, not assumed): params are (axis, angle) and the
        # evaluator does `ch[0].eval(P @ Rm)` -- a row vector times Rm, so new_i = sum_j c_j*Rm[j][i]
        Rm = _rot_matrix(p[:3], p[3])
        nxyz = tuple("(" + " + ".join(f"{Rm[j][i]:.12g}*{c}" for j, c in enumerate(xyz)) + ")"
                     for i in range(3))
        return _emit_sympy(ch[0], nxyz)
    if k == "mirror":
        # params are (axis, plane) -- reflect about coordinate == plane (matched to _eval)
        axis, pl = int(p[0]), float(p[1])
        nxyz = list(xyz)
        nxyz[axis] = f"(Abs({xyz[axis]} - {pl:.9g}) + {pl:.9g})"
        return _emit_sympy(ch[0], tuple(nxyz))
    if k == "round":
        return f"({_emit_sympy(ch[0], xyz)}) - {p[0]:.9g}"
    if k == "onion":
        return f"Abs({_emit_sympy(ch[0], xyz)}) - {p[0]:.9g}"
    if k == "elongate":
        # matched to _eval's EXACT opElongate: q = p - clamp(p, -h, h), child(q) + min(max(q), 0)
        q = tuple(f"({c} - Min(Max({c}, {-e:.9g}), {e:.9g}))" for c, e in zip(xyz, p))
        inner = _emit_sympy(ch[0], q)
        return f"({inner}) + Min(Max({q[0]}, Max({q[1]}, {q[2]})), 0)"
    if k == "repeat":
        nxyz = tuple(c if e <= 0 else f"(Mod({c} + {e/2:.9g}, {e:.9g}) - {e/2:.9g})"
                     for c, e in zip(xyz, p))
        return _emit_sympy(ch[0], nxyz)
    if k in INEXACT:
        raise ValueError("to_jit_expr refuses %r: its field is a BOUND, not an exact distance, "
                         "and the compiled marcher cannot be told to under-step (the INEXACT "
                         "contract) -- use the numpy render path" % k)
    raise ValueError("to_jit_expr: %r has no closed-form single expression (iterative kinds "
                     "stay on the numpy/GLSL paths)" % k)


def _emit_body(node, pvar, ctr, helpers):
    """Return (statements, distance_expr) for `node` at point variable `pvar`. `helpers` is a dict
    {fn_name: glsl_source} accumulating the helper functions this tree needs."""
    k, p, ch = node.kind, node.params, node.children
    stmts = []

    def newvar(prefix):
        ctr[0] += 1
        return f"{prefix}{ctr[0]}"

    if k in ("sphere", "box", "torus", "cylinder", "plane", "capsule", "cone", "octahedron", "ellipsoid"):
        helpers[k] = _GLSL_PRIM[k]
        if k == "sphere":   return stmts, f"sdSphere({pvar},{p[0]:.6g})"
        if k == "box":      return stmts, f"sdBox({pvar},vec3({p[0]:.6g},{p[1]:.6g},{p[2]:.6g}))"
        if k == "torus":    return stmts, f"sdTorus({pvar},{p[0]:.6g},{p[1]:.6g})"
        if k == "cylinder": return stmts, f"sdCyl({pvar},{p[0]:.6g},{p[1]:.6g})"
        if k == "plane":    return stmts, f"sdPlane({pvar},{p[0]:.6g})"
        if k == "capsule":  return stmts, f"sdCapsule({pvar},{p[0]:.6g},{p[1]:.6g})"
        if k == "cone":     return stmts, f"sdCone({pvar},{p[0]:.6g},{p[1]:.6g})"
        if k == "octahedron": return stmts, f"sdOcta({pvar},{p[0]:.6g})"
        if k == "ellipsoid": return stmts, f"sdEllipsoid({pvar},vec3({p[0]:.6g},{p[1]:.6g},{p[2]:.6g}))"
        if k == "capsule":  return stmts, f"sdCapsule({pvar},{p[0]:.6g},{p[1]:.6g})"
        if k == "cone":     return stmts, f"sdCone({pvar},{p[0]:.6g},{p[1]:.6g})"
        if k == "octahedron": return stmts, f"sdOcta({pvar},{p[0]:.6g})"

    if k == "menger":
        iters = int(p[0])
        helpers[f"menger{iters}"] = _menger_glsl(iters, p[1])
        return stmts, f"sdMenger{iters}({pvar})"

    if k == "fold_fractal":
        iters = int(p[0])
        helpers[f"fold{iters}"] = _foldfractal_glsl(iters, float(p[1]), float(p[2]), float(p[3]))
        return stmts, f"sdFold{iters}({pvar})"

    if k == "mandelbulb":
        iters = int(p[1])
        helpers[f"bulb{iters}"] = _mandelbulb_glsl(float(p[0]), iters, float(p[2]))
        return stmts, f"sdBulb{iters}({pvar})"

    if k in ("union", "intersect", "subtract", "smooth_union", "fillet_union"):
        sa, ea = _emit_body(ch[0], pvar, ctr, helpers)
        sb, eb = _emit_body(ch[1], pvar, ctr, helpers)
        stmts += sa + sb
        if k == "union":         return stmts, f"min({ea},{eb})"
        if k == "intersect":     return stmts, f"max({ea},{eb})"
        if k == "subtract":      return stmts, f"max({ea},-({eb}))"
        if k == "fillet_union":
            helpers["uround"] = _GLSL_PRIM["uround"]
            return stmts, f"opUnionRound({ea},{eb},{p[0]:.6g})"
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


def _emit_shader(node, name="map", camera="fixed"):
    """A full Shadertoy fragment shader: helper fns, the map(), and a standard raymarch + normal + light.

    `camera` picks how mainImage sets up the ray:
      * "fixed"    -- the classic head-on view (ro at z=+4 looking down -z). This is the historic output and stays
                      BYTE-IDENTICAL, so every existing caller/test is unchanged (the never-flip rule).
      * "uniforms" -- an ORBIT camera driven by three host-bound uniforms uAngle/uHeight/uDist, declared at the top
                      of the shader. A WebGL2 host (e.g. leStudio) can then spin/zoom the scene by setting uniforms
                      instead of string-splicing a new camera into the emitted source (which is exactly the regex
                      leStudio can now delete). iResolution/iTime remain Shadertoy built-ins; only the three orbit
                      controls are declared. Focal length (1.5) matches the fixed camera so the field of view is
                      unchanged between modes."""
    if camera not in ("fixed", "uniforms"):                # loud rejection beats a silently-ignored typo
        raise ValueError(f"_emit_shader: camera must be 'fixed' or 'uniforms', got {camera!r}")
    helpers = {}
    stmts, dexpr = _emit_body(node, "p", [0], helpers)
    helper_src = "\n".join(helpers[h_name] for h_name in sorted(helpers))   # deterministic order
    body = "\n    ".join(stmts + [f"return {dexpr};"])
    warn = "// NOTE: contains a domain warp (twist/displace) -- not an exact SDF; shorten ray steps.\n" \
        if any(_k in node_kinds(node) for _k in INEXACT) else ""
    # Camera setup, mode-dependent. cam_uniforms is prepended near the top (empty for fixed -> byte-identical);
    # cam_setup replaces the ro/rd line inside mainImage. Subsequent uniform-mode lines carry their own 4-space
    # indent so the emitted mainImage stays cleanly formatted.
    if camera == "uniforms":
        cam_uniforms = "uniform float uAngle;\nuniform float uHeight;\nuniform float uDist;\n\n"
        cam_setup = ("vec3 cw=normalize(-vec3(uDist*sin(uAngle),uHeight,uDist*cos(uAngle)));\n"
                     "    vec3 cu=normalize(cross(cw,vec3(0.0,1.0,0.0))); vec3 cv=cross(cu,cw);\n"
                     "    vec3 ro=vec3(uDist*sin(uAngle),uHeight,uDist*cos(uAngle));\n"
                     "    vec3 rd=normalize(uv.x*cu+uv.y*cv+1.5*cw);")
    else:
        cam_uniforms = ""
        cam_setup = "vec3 ro=vec3(0.0,0.0,4.0), rd=normalize(vec3(uv,-1.5));"
    return f"""// Generated by holostuff holographic_sdf -- a demoscene SDF as code.
// DSL: {node.to_dsl()}
{warn}{cam_uniforms}{helper_src}

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
    {cam_setup}
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

def _selftest_jit_expr():
    """S-5 pin, the emitter family's standing discipline (BOTH EXECUTED, not asserted): the
    sympy-dialect expression must agree numerically with _eval; the bound-only kinds must
    refuse rather than ship a shader that disagrees with the numpy field."""
    env = {"sqrt": np.sqrt, "Abs": np.abs, "Min": np.minimum, "Max": np.maximum, "Mod": np.mod}
    rng = np.random.default_rng(0)
    P = rng.uniform(-2, 2, (200, 3))
    nodes = [sphere(0.7), box(0.5, 0.3, 0.8),
             sphere(0.8).union(box(0.4, 0.4, 0.4)).subtract(cylinder(1.0, 0.2)),
             sphere(0.6).smooth_union(box(0.5, 0.2, 0.5), 0.25),
             sphere(0.5).translate((0.3, -0.2, 0.1)).scale(1.4).rotate((0, 1, 0), 0.7),
             sphere(0.25).repeat((1.2, 0.0, 1.2))]
    for node in nodes:
        expr = node.to_jit_expr()
        got = eval(expr, {"__builtins__": {}}, dict(env, x=P[:, 0], y=P[:, 1], z=P[:, 2]))
        err = float(np.max(np.abs(got - node.eval(P))))
        assert err < 1e-9, "to_jit_expr disagrees with _eval on %s (%.2e)" % (node.kind, err)
    for bad in (box(0.4, 0.4, 0.4).twist(1.0), menger(3)):
        try:
            bad.to_jit_expr()
            raise RuntimeError("%s must refuse" % bad.kind)
        except ValueError:
            pass
    assert sphere(1.0).preserves_analytic and sphere(1.0).translate((1, 0, 0)).preserves_analytic


def _selftest():
    _selftest_jit_expr()
    # (1) PRIMITIVES are correct distances on known points.
    s = sphere(1.0)
    assert abs(s.eval([[2, 0, 0]])[0] - 1.0) < 1e-9            # outside by 1
    assert abs(s.eval([[0, 0, 0]])[0] + 1.0) < 1e-9            # inside by 1 (-1)
    # an SDF is callable and identical to .eval, so field consumers (mesh_from_sdf, sample_field) take it direct
    assert np.array_equal(s([[2, 0, 0]]), s.eval([[2, 0, 0]])), "SDF.__call__ must equal .eval"
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

    # (8b) ITEM 8 -- camera modes. The DEFAULT ("fixed") is BYTE-IDENTICAL to the historic head-on camera (the
    #      never-flip rule: no existing shader output changes). camera="uniforms" instead declares AND uses the
    #      three orbit controls (uAngle/uHeight/uDist) and drops the fixed camera -- so a WebGL2 host can spin/zoom
    #      by binding uniforms rather than string-splicing a new camera into the source (the leStudio regex, now
    #      deletable). Unknown camera raises (no silent fall-through). Both modes keep map()/mainImage + the DSL.
    assert tree.to_glsl() == tree.to_glsl(camera="fixed")            # default IS fixed
    assert "vec3 ro=vec3(0.0,0.0,4.0), rd=normalize(vec3(uv,-1.5));" in glsl   # exact historic camera preserved
    assert "uAngle" not in glsl and "uniform float" not in glsl      # fixed declares no orbit uniforms
    uni = tree.to_glsl(camera="uniforms")
    for tok in ("uniform float uAngle;", "uniform float uHeight;", "uniform float uDist;",
                "uDist*sin(uAngle)", "1.5*cw"):
        assert tok in uni, tok                                       # declares + uses all three, orbit basis present
    assert "vec3 ro=vec3(0.0,0.0,4.0)" not in uni                    # the fixed camera is gone in orbit mode
    assert "float map(vec3 p)" in uni and "mainImage" in uni and tree.to_dsl() in uni
    try:
        tree.to_glsl(camera="orbit"); _raised = False
    except ValueError:
        _raised = True
    assert _raised, "unknown camera must raise ValueError, not silently emit the fixed camera"
    # KEPT NEGATIVE: 'uniforms' emits SELF-CONTAINED uniform declarations -- a host wrapper must NOT re-declare
    # uAngle/uHeight/uDist (GLSL ES 3.00 forbids redeclaring a uniform); leStudio drops both its regex AND its
    # own orbit-uniform decls when it adopts this mode.

    # (9) MENGER fractal: the recursive sponge evals, carves holes (a point in a hole is OUTSIDE), and
    #     emits a GLSL loop helper.
    spng = menger(3, 1.0)
    assert spng.eval([[0.0, 0.0, 0.0]])[0] > 0            # the centre cross is carved out (outside)
    assert spng.eval([[0.95, 0.95, 0.95]])[0] < 0.2       # a corner pillar is solid/near-surface
    mglsl = spng.to_glsl()
    assert "sdMenger3(" in mglsl and "for(int m=0;m<3;m++)" in mglsl

    # (10) FOLD_FRACTAL (Mandelbox / KIFS): the general fold engine. It must produce a usable DISTANCE ESTIMATE
    #      (never changes faster than the query point moves) and REAL spatial structure (not a constant field).
    ffr = fold_fractal(iterations=12, scale=2.0, min_radius=0.5, fold_limit=1.0)
    _rng = np.random.default_rng(0)
    _A = _rng.uniform(-4, 4, (3000, 3)); _B = _A + _rng.normal(0, 0.005, (3000, 3))
    _ratio = np.abs(ffr.eval(_A) - ffr.eval(_B)) / np.maximum(np.linalg.norm(_A - _B, axis=1), 1e-9)
    assert np.percentile(_ratio, 99) < 3.0, "fold_fractal must be a usable distance estimate (bounded Lipschitz)"
    _X, _Z = np.meshgrid(np.linspace(-3, 3, 60), np.linspace(-3, 3, 60))
    _grid = np.column_stack([_X.ravel(), np.zeros(_X.size), _Z.ravel()])
    _dg = ffr.eval(_grid)
    assert _dg.std() > 1e-3 and (np.percentile(_dg, 90) - np.percentile(_dg, 10)) > 1e-3, \
        "fold_fractal must have real spatial structure (a spread of distances), not a constant field"
    assert np.array_equal(ffr.eval(_grid), ffr.eval(_grid)), "fold_fractal must be deterministic"
    assert ffr.cost()["iterative"] is True, "the fold fractal is an iterative SDF (budget carefully for realtime)"

    # (11) MANDELBULB (escape-time z^n+c in 3D): analytic DE, real inside/outside structure.
    mbulb = mandelbulb(power=8.0, iterations=8, bailout=2.0)
    assert mbulb.eval([[0.0, 0.0, 0.0]])[0] <= 0.01, "the mandelbulb origin is inside the set"
    assert mbulb.eval([[3.0, 3.0, 3.0]])[0] > 2.0, "far points are far outside the bounded bulb"
    _Xb, _Zb = np.meshgrid(np.linspace(-1.3, 1.3, 50), np.linspace(-1.3, 1.3, 50))
    _db = mbulb.eval(np.column_stack([_Xb.ravel(), np.zeros(_Xb.size), _Zb.ravel()]))
    assert _db.min() < 0 < _db.max(), "the mandelbulb slab has both interior (d<0) and exterior (d>0)"

    # (12) ESCAPE_TIME (2D Mandelbrot / Julia field): smooth escape counts, real interior + exterior.
    _mset = escape_time(width=80, height=80, center=(-0.5, 0.0), span=3.0, max_iter=60)
    _in = np.mean(_mset >= 59.9)
    assert 0.05 < _in < 0.8, "the Mandelbrot set has both an interior (in-set) and an exterior"
    assert _mset[40, 40] >= 59.9, "the cardioid centre is in the set"
    _jset = escape_time(width=60, height=60, span=3.0, max_iter=60, julia_c=(-0.8, 0.156))
    assert _jset.std() > 1.0, "the Julia set has real structure"
    assert np.array_equal(escape_time(width=40, height=40), escape_time(width=40, height=40)), "escape_time det."

    # (13) FRACTAL -> SHADERTOY: fold_fractal + mandelbulb now EMIT a complete raymarch shader (the demoscene OUTPUT).
    #      They are distance ESTIMATES, so the header warns to step conservatively, but they emit (unlike ellipsoid).
    _fsh = _emit_shader(fold_fractal(iterations=8, scale=2.0), name="map")
    assert "sdFold8(" in _fsh and "for(int i=0;i<8;i++)" in _fsh, "fold_fractal emits its fold loop"
    _bsh = _emit_shader(mandelbulb(power=8.0, iterations=6), name="map")
    assert "sdBulb6(" in _bsh and "for(int i=0;i<6;i++)" in _bsh, "mandelbulb emits its polar-power loop"
    # ellipsoid now emits too (iq's bounded k1*(k1-1)/k2 form) -- needed by the humanoid muscle/breast morphs.
    _esh = _emit_shader(SDF("smooth_union", (0.1,), (sphere(0.3), ellipsoid(0.2, 0.3, 0.2))), name="map")
    assert "sdEllipsoid(" in _esh and "opSmin(" in _esh, "ellipsoid + smooth_union emit"

    # ---- make_shape + dsl_grammar (J-3D-13/14): the reach half, not the geometry half ----
    _s = make_sdf_shape("ball", r=0.5)
    assert abs(float(_s.eval(np.array([[0.0, 0.0, 0.5]]))[0])) < 1e-12, "alias 'ball' must build a sphere"
    # TRANSFORM ORDER IS THE ASSERTION THAT MATTERS. scale -> rotate -> translate. Rotating AFTER translating
    # swings the object around the world origin instead of spinning it in place -- a classic quiet bug that
    # looks like "my object jumped somewhere else" and is invisible in a single still frame.
    _bar = make_sdf_shape("box", bx=1.0, by=0.1, bz=0.1, position=(3.0, 0.0, 0.0), rotate=(0, 0, 1, np.pi / 2))
    assert float(_bar.eval(np.array([[3.0, 0.0, 0.0]]))[0]) < 0.0, "the bar must still be centred at (3,0,0)"
    assert float(_bar.eval(np.array([[3.0, 0.9, 0.0]]))[0]) < 0.0, "after a 90deg z-turn it must extend along y"
    assert float(_bar.eval(np.array([[3.9, 0.0, 0.0]]))[0]) > 0.0, "...and no longer along x"
    for _k, (_fn, _p) in SHAPE_KINDS.items():
        assert isinstance(make_sdf_shape(_k), SDF), "kind %r did not build" % _k
    try:
        make_sdf_shape("blob")                                   # a plausible-but-wrong guess
        raise AssertionError("an unknown kind must raise, not silently pick a default")
    except KeyError as _exc:
        assert "sphere" in str(_exc), "the error must TEACH the vocabulary, not just refuse"
    try:
        make_sdf_shape("sphere", bx=1.0)                         # right kind, wrong parameter name
        raise AssertionError("a wrong parameter must raise rather than be silently dropped")
    except TypeError as _exc:
        assert "'r'" in str(_exc) or "['r']" in str(_exc), "the error must name the parameters that DO apply"
    # the grammar must describe EVERY node the parser accepts, and its own example must round-trip -- a
    # grammar that documents a node set the parser does not implement is worse than none.
    _g = dsl_grammar()
    assert {r["kind"] for r in _g["nodes"]} == set(ARITY), "grammar and parser disagree on the node set"
    assert all(r["does"] for r in _g["nodes"]), "every node needs a plain-language line or the table is a cipher"
    assert parse_dsl(_g["example"]) is not None, "the grammar's own example must parse"

    print("holographic_sdf selftest passed:",
          f"seam hard={kink_hard:.3f} soft={kink_soft:.3f} mesh_faces={mesh.n_faces} "
          f"glsl_chars={len(glsl)} menger_center={spng.eval([[0,0,0]])[0]:.3f} "
          f"fold_fractal DE p99-Lipschitz={np.percentile(_ratio,99):.2f} struct_std={_dg.std():.4f} "
          f"mandelbulb slab d in [{_db.min():.2f},{_db.max():.2f}] mandelbrot_in_set={_in:.2f} "
          f"kinds={sorted(node_kinds(tree))}")


if __name__ == "__main__":
    _selftest()
