"""Procedural grammar (G5): L-systems and greebles as holographic recipes.

WHY THIS MODULE EXISTS
----------------------
This is the one genuinely NEW capability on the geometry list -- not reuse of a shipped faculty. The
substrate already existed (recipes, typed structures, scenegraph recursion); what was missing was the
GRAMMAR on top: a way to grow branching structures (plants) and recursive detail (greebles) from
rules.

THE VSA FRAMING
---------------
An L-system PRODUCTION is a rewrite rule, and a rewrite rule is a recipe. Each symbol is an atom; the
rule set is a holographic record, sum_s bind(symbol_s, expansion_s), so the grammar itself is one
composable vector a VSA program can carry, query (unbind a symbol -> its expansion), and compose. The
EXPANSION is classic parallel string rewriting; the INTERPRETATION is a 3D turtle that emits a
skeleton of segments; and the ASSEMBLY is a scenegraph -- each segment instanced through its own
transform, a recursive bundle that `scene_to_recipe` turns straight back into a holographic recipe.

So: rules are a record, the plant is a scenegraph, and both ends live in the same algebra as the
geometry they produce.

HONEST SCOPE (kept negatives)
-----------------------------
  * Deterministic, context-free L-systems (optionally seeded-stochastic productions). This is recursive
    COMPOSITION, not a biological growth simulation -- no phototropism, no competition, no resource
    flow. It makes structures with the right self-similarity, nothing more.
  * Greebles are recursive subdivision + extrusion -- mechanical-looking detail, not a learned style.
  * The branching is band-limited by the recursion depth you ask for; deeper = denser = more cost.
"""

import numpy as np

from holographic_mesh import Mesh, box
from holographic_scenegraph import SceneNode, flatten_scene


# ---------------------------------------------------------------------------
# L-system: parallel string rewriting.
# ---------------------------------------------------------------------------

class LSystem:
    """A context-free L-system: an axiom string and a dict of `productions` {symbol: replacement}.

    expand(n) applies all productions in PARALLEL n times (every symbol rewritten at once, the defining
    feature of an L-system vs a sequential grammar). Symbols with no production are copied through.
    Optionally `rng_seed` + `stochastic` {symbol: [(weight, replacement), ...]} chooses replacements
    randomly but reproducibly.
    """

    def __init__(self, axiom, productions, stochastic=None, rng_seed=0):
        self.axiom = axiom
        self.productions = dict(productions)
        self.stochastic = stochastic or {}
        self.rng_seed = rng_seed

    def expand(self, n):
        """Return the string after n parallel-rewrite steps."""
        rng = np.random.default_rng(self.rng_seed)
        s = self.axiom
        for _ in range(n):
            out = []
            for ch in s:
                if ch in self.stochastic:                  # weighted random replacement (reproducible)
                    opts = self.stochastic[ch]
                    weights = np.array([w for w, _ in opts], float)
                    pick = rng.choice(len(opts), p=weights / weights.sum())
                    out.append(opts[pick][1])
                elif ch in self.productions:
                    out.append(self.productions[ch])
                else:
                    out.append(ch)                         # constants copy through
            s = "".join(out)
        return s


def productions_record(productions, dim=1024, seed=0):
    """The rule set as ONE holographic record: sum_s bind(atom(s), atom(expansion_s)).

    Lets a VSA program carry the grammar as a vector and recover a rule by unbinding its symbol atom
    (returns the record plus the atom table so you can probe it). Atoms are deterministic per name.
    """
    import hashlib
    from holographic_ai import bind

    def atom(name):
        sd = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
        v = np.random.default_rng(sd).standard_normal(dim)
        return v / (np.linalg.norm(v) or 1.0)

    rec = np.zeros(dim)
    atoms = {}
    for sym, exp in productions.items():
        atoms[sym] = atom("sym:" + sym)
        atoms["exp:" + sym] = atom("exp:" + exp)
        rec = rec + bind(atoms[sym], atoms["exp:" + sym])
    return rec, atoms


# ---------------------------------------------------------------------------
# Turtle: interpret a string as a 3D skeleton of segments.
# ---------------------------------------------------------------------------

def _rodrigues(v, axis, angle):
    """Rotate vector v about a unit axis by angle (Rodrigues' formula)."""
    axis = axis / (np.linalg.norm(axis) or 1.0)
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def turtle_to_segments(symbols, angle_deg=25.0, step=1.0):
    """Interpret an L-system string with a 3D turtle -> a list of (start, end) segments (the skeleton).

    Commands: F draw forward; f move forward (no draw); +/- yaw; &/^ pitch; \\// roll; [ push state, ] pop.
    Heading starts along +z (up); the frame (heading, left, up) is rotated about its own axes so branches
    keep a consistent local orientation.
    """
    ang = np.radians(angle_deg)
    pos = np.zeros(3)
    H = np.array([0.0, 0, 1.0])      # heading (up the plant)
    L = np.array([1.0, 0, 0.0])      # left
    U = np.array([0.0, 1.0, 0.0])    # up (out of the branching plane)
    stack = []
    segs = []
    for ch in symbols:
        if ch == "F":
            nxt = pos + step * H
            segs.append((pos.copy(), nxt.copy()))
            pos = nxt
        elif ch == "f":
            pos = pos + step * H
        elif ch == "+":
            H, L = _rodrigues(H, U, ang), _rodrigues(L, U, ang)
        elif ch == "-":
            H, L = _rodrigues(H, U, -ang), _rodrigues(L, U, -ang)
        elif ch == "&":
            H, U = _rodrigues(H, L, ang), _rodrigues(U, L, ang)
        elif ch == "^":
            H, U = _rodrigues(H, L, -ang), _rodrigues(U, L, -ang)
        elif ch == "\\":
            L, U = _rodrigues(L, H, ang), _rodrigues(U, H, ang)
        elif ch == "/":
            L, U = _rodrigues(L, H, -ang), _rodrigues(U, H, -ang)
        elif ch == "[":
            stack.append((pos.copy(), H.copy(), L.copy(), U.copy()))
        elif ch == "]":
            pos, H, L, U = stack.pop()
    return segs


# ---------------------------------------------------------------------------
# Assembly: segments / greebles -> a scenegraph (recursive composition).
# ---------------------------------------------------------------------------

def _align_z_to(direction):
    """A 4x4 rotation mapping +z onto `direction` (shortest arc), for orienting a strut along a segment."""
    d = np.asarray(direction, float); d = d / (np.linalg.norm(d) or 1.0)
    z = np.array([0.0, 0, 1.0])
    v = np.cross(z, d); s = np.linalg.norm(v); c = float(np.dot(z, d))
    M = np.eye(4)
    if s < 1e-9:                                    # already aligned (or anti-aligned)
        if c < 0:
            M[:3, :3] = np.diag([1.0, -1.0, -1.0])  # 180-degree flip
        return M
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    M[:3, :3] = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return M


def segments_to_scene(segments, radius=0.04):
    """Instance a thin strut along every segment as a SceneNode child -> one scenegraph (a recursive bundle)."""
    children = []
    for (p0, p1) in segments:
        mid = 0.5 * (p0 + p1)
        length = float(np.linalg.norm(p1 - p0))
        if length < 1e-9:
            continue
        strut = box(width=2 * radius, height=2 * radius, depth=length)   # along +z by default
        T = np.eye(4); T[:3, 3] = mid                                    # translate to midpoint
        node = SceneNode(transform=T @ _align_z_to(p1 - p0), mesh=strut)
        children.append(node)
    return SceneNode(children=children, name="plant")


def grow_plant(lsystem, iterations, angle_deg=25.0, step=1.0, radius=0.04):
    """End-to-end: expand the L-system, interpret as a turtle skeleton, assemble a scenegraph, flatten to a Mesh."""
    symbols = lsystem.expand(iterations)
    segs = turtle_to_segments(symbols, angle_deg=angle_deg, step=step)
    scene = segments_to_scene(segs, radius=radius)
    return flatten_scene(scene), segs, scene


def greeble_panel(width, height, depth=0.1, seed=0, max_depth=3, _rng=None, _origin=(0.0, 0.0)):
    """Recursively subdivide a panel into sub-panels with random extrusion -> a scenegraph of boxes.

    Each level splits the panel along its longer side at a random ratio and recurses; leaves are
    extruded boxes at random depths. Deterministic given `seed`. Mechanical detail by recursion.
    """
    rng = np.random.default_rng(seed) if _rng is None else _rng
    ox, oy = _origin
    if max_depth <= 0 or (width < 0.15 and height < 0.15):
        h = depth * float(rng.uniform(0.4, 1.0))
        panel = box(width=width * 0.9, height=height * 0.9, depth=h,
                    center=(ox + width / 2, oy + height / 2, h / 2))
        return SceneNode(mesh=panel)
    children = []
    if width >= height:                              # split the longer side
        cut = width * float(rng.uniform(0.35, 0.65))
        children.append(greeble_panel(cut, height, depth, max_depth=max_depth - 1,
                                      _rng=rng, _origin=(ox, oy)))
        children.append(greeble_panel(width - cut, height, depth, max_depth=max_depth - 1,
                                      _rng=rng, _origin=(ox + cut, oy)))
    else:
        cut = height * float(rng.uniform(0.35, 0.65))
        children.append(greeble_panel(width, cut, depth, max_depth=max_depth - 1,
                                      _rng=rng, _origin=(ox, oy)))
        children.append(greeble_panel(width, height - cut, depth, max_depth=max_depth - 1,
                                      _rng=rng, _origin=(ox, oy + cut)))
    return SceneNode(children=children, name="greeble")


# ---------------------------------------------------------------------------

def _selftest():
    from holographic_ai import bind, unbind, cosine, involution
    from holographic_fractal import box_counting_dimension

    # (1) EXPANSION is deterministic and grows by the production rule. The classic algae system
    #     A->AB, B->A has Fibonacci-length generations -- an exact check, not a fuzzy one.
    algae = LSystem("A", {"A": "AB", "B": "A"})
    lengths = [len(algae.expand(n)) for n in range(7)]
    fib = [1, 2, 3, 5, 8, 13, 21]
    assert lengths == fib, f"algae lengths should be Fibonacci, got {lengths}"
    assert algae.expand(4) == algae.expand(4), "expansion must be deterministic"

    # (2) TURTLE -> skeleton: a branching plant system produces a connected set of segments, and deeper
    #     iterations produce strictly more segments (the structure grows).
    plant = LSystem("X", {"X": "F[+X][-X]FX", "F": "FF"})
    s2 = turtle_to_segments(plant.expand(2))
    s4 = turtle_to_segments(plant.expand(4))
    assert len(s4) > len(s2) > 0, f"deeper expansion should grow the skeleton: {len(s2)} -> {len(s4)}"

    # (3) BRANCHING fills more of the plane as it recurses: box-counting dimension of the skeleton points
    #     (projected to xz) grows (or holds) with depth -- the structure is space-filling, measurably.
    def skeleton_dim(symbols):
        segs = turtle_to_segments(symbols)
        pts = np.array([p for seg in segs for p in seg])[:, [0, 2]]     # project to the branching plane
        return box_counting_dimension(pts)
    d2, d4 = skeleton_dim(plant.expand(2)), skeleton_dim(plant.expand(4))
    assert d4 >= d2 - 0.05, f"deeper branching should not be less space-filling: {d2:.3f} -> {d4:.3f}"

    # (4) PRODUCTIONS as a holographic record: unbind a symbol -> its expansion direction (above chance).
    rec, atoms = productions_record(plant.productions)
    recovered = unbind(rec, atoms["X"])
    self_cos = cosine(recovered, atoms["exp:X"])
    other_cos = cosine(recovered, atoms["exp:F"])
    assert self_cos > 0.45 and self_cos > 3 * abs(other_cos), \
        f"rule did not recover from the record: self={self_cos:.3f} other={other_cos:.3f}"

    # (5) end-to-end grow_plant returns a real mesh; greeble subdivision is deterministic with leaves.
    mesh, segs, scene = grow_plant(plant, 3, angle_deg=25, step=0.5)
    assert mesh.n_vertices > 0 and len(mesh.faces) > 0, "grown plant should be a non-empty mesh"
    g1 = flatten_scene(greeble_panel(2.0, 1.0, seed=1, max_depth=3))
    g1b = flatten_scene(greeble_panel(2.0, 1.0, seed=1, max_depth=3))
    assert np.allclose(g1.vertices, g1b.vertices), "greeble must be deterministic for a fixed seed"
    assert g1.n_vertices > 0, "greeble should produce geometry"

    print("holographic_grammar selftest passed:",
          f"algae_lengths={lengths} skeleton_segs={len(s2)}->{len(s4)} "
          f"branch_dim={d2:.3f}->{d4:.3f} rule_recall={self_cos:.3f} plant_faces={len(mesh.faces)}")


if __name__ == "__main__":
    _selftest()
