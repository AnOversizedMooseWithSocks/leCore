"""Holographic vector-graphics (SVG) scenes -- the sharp, resolution-independent cousin of the splat archive.

A SCENE is a small list of typed primitives, each with a continuous position and size and a palette colour:

    (type_index, x, y, size, colour_index)     type in {rect, circle, triangle, ...}, x/y/size in [0,1].

The whole scene encodes into ONE hypervector the engine's usual way: each primitive is a bundle of role-bound
attributes (TYPE, X, Y, SIZE, COLOUR), and the scene is a bundle of those primitives each bound to a SLOT role,
so a whole picture is content-addressable. Two scenes MORPH by interpolating their VECTORS (vector arithmetic in
the holographic space) and decoding the blend -- measured to match a direct parameter lerp, so interpolating the
hypervectors really does interpolate the picture. Discrete attributes (type, colour) come back by cleanup;
continuous ones (position, size) by the ScalarEncoder's grid decode -- the continuous analogue of cleanup.

WHY SVG. The render is crisp BY CONSTRUCTION: an SVG <rect>/<circle>/<polygon> has analytically exact edges at
ANY zoom, because the rasteriser computes exact coverage from the maths. This is the very thing the splat work
kept fighting -- a Gaussian basis blurs edges and needs smaller splats or supersampling, while a vector primitive
is sharp and resolution-INDEPENDENT for free. SVG emission here is pure string formatting -- no new dependency
(NumPy and Flask only, the project rule); the engine writes the text, any browser or rasteriser draws it.

MEASURED (the _selftest is the earns-its-place gate): a 3-primitive scene round-trips with type and colour exact
and position/size within ~0.03 on [0,1]; a vector-space morph between two scenes matches a parameter lerp at the
midpoint to within ~0.03; and the composed-manifold diffusion generates distinct, valid novel scenes.

KEPT NEGATIVE / SCOPE: primitives are isotropic (one size, a palette colour). Anisotropic width/height, rotation,
gradients/strokes, and bezier paths are the honest next step, deliberately out of this baseline -- the same
boundary the anisotropic-splat work drew. And round-trip fidelity scales with dimension: a handful of primitives
is faithful at 2048+, but a crowded scene wants more dimension (the bundle's finite capacity, shown not hidden).
"""
import numpy as np

from holographic.agents_and_reasoning.holographic_ai import bind, involution, bundle, derived_atom
from holographic.io_and_interop.holographic_encoders import ScalarEncoder

DEFAULT_TYPES = ("rect", "circle", "triangle")
# a readable, well-separated palette (RGB in 0..1)
DEFAULT_PALETTE = [(0.93, 0.26, 0.21), (0.20, 0.60, 0.86), (0.18, 0.80, 0.44),
                   (0.95, 0.77, 0.06), (0.61, 0.35, 0.71), (0.90, 0.49, 0.13)]


class HolographicSVG:
    """Encode/decode/morph/generate vector-graphics scenes on the bind/bundle substrate, and render them as crisp,
    resolution-independent SVG. Owns its atoms and encoders, so everything is reproducible from the seed."""

    def __init__(self, dim=4096, seed=0, types=DEFAULT_TYPES, palette=None, max_slots=8):
        self.dim = int(dim)
        self.seed = int(seed)
        self.types = tuple(types)
        self.palette = list(palette) if palette is not None else list(DEFAULT_PALETTE)
        # roles for a primitive's attributes (unitary -> exact unbinding), and slot roles that separate primitives
        self.RT, self.RX, self.RY, self.RS, self.RC = (
            derived_atom(seed, f"svg:role:{n}", self.dim, unitary=True) for n in ("type", "x", "y", "size", "col"))
        self.type_atoms = np.stack([derived_atom(seed, f"svg:type:{t}", self.dim, unitary=True) for t in self.types])
        self.col_atoms = np.stack([derived_atom(seed, f"svg:col:{i}", self.dim, unitary=True)
                                   for i in range(len(self.palette))])
        self.slots = np.stack([derived_atom(seed, f"svg:slot:{k}", self.dim, unitary=True) for k in range(max_slots)])
        # continuous encoders: position on the unit square, size on a sensible primitive range
        self.encX = ScalarEncoder(self.dim, 0.0, 1.0, seed=seed + 101)
        self.encY = ScalarEncoder(self.dim, 0.0, 1.0, seed=seed + 102)
        self.encS = ScalarEncoder(self.dim, 0.03, 0.25, seed=seed + 103)
        self._gen_cache = {}                       # (k, grid) -> (parts, fillers, roles) for generation, built once

    # -- encode / decode ----------------------------------------------------------------------------------------
    def _prim_vec(self, prim):
        ti, x, y, s, ci = prim
        return bundle([bind(self.RT, self.type_atoms[ti]), bind(self.RX, self.encX.encode(x)),
                       bind(self.RY, self.encY.encode(y)), bind(self.RS, self.encS.encode(s)),
                       bind(self.RC, self.col_atoms[ci])])

    def encode(self, prims):
        """Encode a scene -- a list of (type_index, x, y, size, colour_index) -- into one unit hypervector."""
        if len(prims) > len(self.slots):
            raise ValueError(f"at most {len(self.slots)} primitives (max_slots); got {len(prims)}")
        return bundle([bind(self.slots[k], self._prim_vec(p)) for k, p in enumerate(prims)])

    def _decode_prim(self, pv):
        # discrete attributes by cleanup (argmax cosine); continuous by the ScalarEncoder's grid decode
        ti = int(np.argmax(self.type_atoms @ bind(pv, involution(self.RT))))
        ci = int(np.argmax(self.col_atoms @ bind(pv, involution(self.RC))))
        x = float(self.encX.decode(bind(pv, involution(self.RX))))
        y = float(self.encY.decode(bind(pv, involution(self.RY))))
        s = float(self.encS.decode(bind(pv, involution(self.RS))))
        return (ti, x, y, s, ci)

    def decode(self, vec, k):
        """Recover the k primitives from a scene hypervector (unbind each slot, then read its attributes back)."""
        return [self._decode_prim(bind(vec, involution(self.slots[i]))) for i in range(k)]

    # -- morph (vector arithmetic) ------------------------------------------------------------------------------
    def morph(self, prims_a, prims_b, steps=7):
        """Interpolate two scenes by BLENDING THEIR HYPERVECTORS and decoding -- the morph is vector arithmetic in
        the holographic space (measured to track a direct parameter lerp). Returns `steps` decoded scenes."""
        va, vb = self.encode(prims_a), self.encode(prims_b)
        k = min(len(prims_a), len(prims_b))
        out = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            v = (1.0 - t) * va + t * vb
            out.append(self.decode(v / (np.linalg.norm(v) + 1e-12), k))
        return out

    # -- generate (the composed-manifold diffusion over a discrete primitive codebook) --------------------------
    def _gen_codebook(self, k, grid):
        if (k, grid) not in self._gen_cache:
            cells = [(r, c) for r in range(grid) for c in range(grid)]
            parts = [(ti, rc, ci) for ti in range(len(self.types)) for rc in cells for ci in range(len(self.palette))]
            fillers = np.stack([derived_atom(self.seed, f"svg:gen:part:{i}", self.dim, unitary=True)
                                for i in range(len(parts))])
            roles = np.stack([derived_atom(self.seed, f"svg:gen:slot:{j}", self.dim, unitary=True) for j in range(k)])
            self._gen_cache[(k, grid)] = (parts, fillers, roles)
        return self._gen_cache[(k, grid)]

    def generate(self, k=4, grid=4, steps=16, seed=0, size=0.11):
        """Generate a NOVEL scene by running the engine's composed-manifold diffusion (holographic_hopfield's
        generate_structure) over a discrete primitive codebook (type x grid-cell x colour), then mapping the
        decoded parts to continuous primitives for rendering. Distinct seeds give distinct scenes. Returns the
        scene as a list of (type_index, x, y, size, colour_index)."""
        from holographic.agents_and_reasoning.holographic_hopfield import generate_structure
        parts, fillers, roles = self._gen_codebook(k, grid)
        g = generate_structure(roles, fillers, steps=steps, seed=seed, readout="sparsemax")
        scene = []
        for j in range(k):
            idx = int(np.argmax(fillers @ bind(g, involution(roles[j]))))
            ti, (r, c), ci = parts[idx]
            scene.append((ti, (c + 0.5) / grid, (r + 0.5) / grid, size, ci))
        return scene

    # -- render (crisp, resolution-independent SVG -- pure string, no dependency) --------------------------------
    def _fill(self, ci):
        c = self.palette[ci]
        return f"rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})"

    def to_svg(self, prims, size=240, bg="#11151c", rounded=True):
        """Render a scene as SVG text. Edges are analytically exact at any zoom -- no splat blur, no supersampling.
        `size` is the viewBox extent only; the SVG is resolution-independent, so it draws crisp at any pixel size."""
        el = [f'<rect width="{size}" height="{size}" fill="{bg}"/>']
        for (ti, x, y, s, ci) in prims:
            cx, cy, r = x * size, y * size, s * size
            t = self.types[ti]
            f = self._fill(ci)
            if t == "rect":
                rx = f' rx="{0.15 * r:.2f}"' if rounded else ""
                el.append(f'<rect x="{cx - r:.2f}" y="{cy - r:.2f}" width="{2 * r:.2f}" height="{2 * r:.2f}"{rx} '
                          f'fill="{f}"/>')
            elif t == "circle":
                el.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{f}"/>')
            elif t == "triangle":
                el.append(f'<polygon points="{cx:.2f},{cy - r:.2f} {cx - r:.2f},{cy + r:.2f} '
                          f'{cx + r:.2f},{cy + r:.2f}" fill="{f}"/>')
            else:                                  # unknown type -> a dot, so the render never silently drops a slot
                el.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{f}"/>')
        return f'<svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">' + "".join(el) + "</svg>"


def _selftest():
    svg = HolographicSVG(dim=4096, seed=0)

    # (1) ROUND-TRIP: a scene encodes to one vector and decodes back faithfully (type/colour exact, position close).
    scene = [(1, 0.30, 0.35, 0.12, 1), (0, 0.70, 0.62, 0.10, 0), (2, 0.52, 0.40, 0.09, 2)]
    dec = svg.decode(svg.encode(scene), len(scene))
    assert all(a[0] == b[0] for a, b in zip(scene, dec)), f"type not recovered: {dec}"
    assert all(a[4] == b[4] for a, b in zip(scene, dec)), f"colour not recovered: {dec}"
    perr = np.mean([abs(a[1] - b[1]) + abs(a[2] - b[2]) + abs(a[3] - b[3]) for a, b in zip(scene, dec)]) / 3
    assert perr < 0.05, f"position/size round-trip too lossy: {perr:.3f}"

    # (2) MORPH: interpolating the HYPERVECTORS matches a parameter lerp at the midpoint -- the picture interpolates.
    A = [(1, 0.25, 0.30, 0.13, 1), (0, 0.72, 0.68, 0.11, 3), (2, 0.50, 0.50, 0.10, 5)]
    B = [(1, 0.70, 0.30, 0.09, 2), (0, 0.28, 0.66, 0.13, 0), (2, 0.55, 0.42, 0.12, 4)]
    mid = svg.morph(A, B, steps=5)[2]
    lerp = [(a[0], 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2]), 0.5 * (a[3] + b[3]), a[4]) for a, b in zip(A, B)]
    merr = np.mean([abs(m[1] - l[1]) + abs(m[2] - l[2]) + abs(m[3] - l[3]) for m, l in zip(mid, lerp)]) / 3
    assert merr < 0.06, f"vector-space morph diverges from param lerp: {merr:.3f}"

    # (3) GENERATE: distinct valid novel scenes; the SVG is well-formed.
    scenes = [tuple(svg.generate(k=4, seed=s)) for s in range(6)]
    assert len(set(scenes)) >= 4, f"generation not diverse: {len(set(scenes))}/6"
    out = svg.to_svg(svg.generate(k=4, seed=0))
    assert out.startswith("<svg") and out.rstrip().endswith("</svg>") and "<rect" in out, "malformed SVG"

    print("holographic_svg selftest OK:")
    print(f"  round-trip: type/colour exact, position/size error {perr:.3f} on [0,1]")
    print(f"  morph: vector interpolation vs param lerp error {merr:.3f} (the picture interpolates by arithmetic)")
    print(f"  generate: {len(set(scenes))}/6 distinct novel scenes; SVG well-formed (crisp at any zoom)")


if __name__ == "__main__":
    _selftest()
