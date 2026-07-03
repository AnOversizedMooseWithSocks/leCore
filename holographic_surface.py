"""holographic_surface.py -- the FIRST-CLASS render material: every channel is a Param socket, resolved PER HIT.

WHY THIS EXISTS (the above/below audit's biggest gap, CORE_NOTES 2.1)
---------------------------------------------------------------------
The engine had all the parts but no object tying them together: `holographic_param.Param` (sockets: const / map /
field / source), `holographic_pattern` (procedural fields that plug into a socket), the `MATERIAL_RENDER` name table
in holographic_semantic (per-material channel constants), and the trace machinery. What was missing -- and what demo
backends kept re-implementing -- is a MATERIAL whose channels are sockets and a render call that resolves them at
each hit point. That object is `SurfaceMaterial`; that call is `render_surface`.

The tie-together this creates (each was a disconnected thread before):
    pattern (a field)  ->  Param(field=...)  ->  SurfaceMaterial channel  ->  resolved per hit  ->  the pixel
and `SurfaceMaterial.from_name('metal')` consumes the ONE canonical MATERIAL_RENDER table, so the name->channels
mapping has a single source instead of per-demo copies.

Channels (each accepts anything `resolve_param` accepts -- a constant, a Param, a callable field, or a map array):
    color (rgb), roughness [0..1], reflect [0..1], emission [0..1], opacity [0..1].
Because a channel can be a FIELD over world position, a pattern is a solid 3-D texture: it wraps a curved surface
with no UV unwrap (the field-native way).

HONEST SCOPE (kept loud): `render_surface` shades Lambert + Blinn specular + an ENVIRONMENT reflection (the reflected
ray samples the sky function, not other objects -- one-bounce object-object reflection lives in render_dispatch /
render_scene, use those when you need a mirror that shows the scene). Opacity is ONE transparency layer: a
continuation ray from the hit composites what is behind it (front*a + behind*(1-a)); stacked glass needs the full
renderer. NumPy only; deterministic.
"""
import numpy as np
from holographic_param import Param, resolve_param


def _rgb(vals, points, n):
    """Resolve a channel to an (M,3) rgb array whether it was given as a scalar, an (3,) colour, a per-point (M,)
    grey field, or an (M,3) colour field -- so grey patterns and colour maps both just work. resolve_param flattens
    per-point colour output to (3M,), so that case is detected by size and reshaped back."""
    v = resolve_param(vals, points=points, n=n) if not isinstance(vals, (tuple, list)) else np.asarray(vals, float)
    v = np.asarray(v, float)
    if v.ndim == 0:
        return np.full((n, 3), float(v))
    if v.ndim == 1:
        if v.size == n * 3:
            return v.reshape(n, 3)                                   # a flattened per-point colour field
        if v.shape[0] == 3 and v.size != n:
            return np.broadcast_to(v, (n, 3)).astype(float)          # one (3,) colour for all points
        return np.repeat(v[:, None], 3, axis=1)                      # per-point grey -> rgb
    return v                                                         # already (M,3)


class SurfaceMaterial:
    """A render material whose EVERY channel is a socket. Channels: color, roughness, reflect, emission, opacity --
    each a constant, a Param, a callable field f(points)->values, or a map array. Resolved per hit by render_surface,
    so a `holographic_pattern` field on any channel is a solid texture. Use `from_name` to build one from the
    canonical MATERIAL_RENDER table (single source for name->channels; no more per-demo copies)."""

    def __init__(self, color=(0.7, 0.7, 0.7), roughness=0.5, reflect=0.0, emission=0.0, opacity=1.0):
        self.color = color; self.roughness = roughness; self.reflect = reflect
        self.emission = emission; self.opacity = opacity

    @classmethod
    def from_name(cls, name, color=(0.7, 0.7, 0.7)):
        """Build from the ONE canonical material-name table (holographic_semantic.MATERIAL_RENDER). The name gives the
        channel constants (reflect etc.); `color` supplies the albedo. Channels stay sockets -- override any of them
        with a Param/field afterwards to texture a named material."""
        from holographic_semantic import MATERIAL_RENDER
        mr = MATERIAL_RENDER.get(name, MATERIAL_RENDER.get(None, {})) or {}
        m = cls(color=color, reflect=float(mr.get("reflect", 0.0)))
        pbr = mr.get("pbr")
        if pbr is not None:                                          # (metallic, roughness) in the table
            m.roughness = float(pbr[1])
        if mr.get("refract"):
            m.opacity = 1.0 - float(mr["refract"])                   # a refractive material is mostly transparent
        return m

    @classmethod
    def from_matlib(cls, name, color=None):
        """Build a render SurfaceMaterial from the PHYSICAL material library (holographic_matlib) -- so any of its
        ~130 glTF-PBR presets (metals, woods, stones, gems, biomes, planetary layers, ore deposits) drives
        render_surface / path_trace / RenderSession directly. This is the bridge from the fork's physical definitions
        into our first-class render pipeline: a data-driven, physically-plausible material, not a hand-set demo colour.

        The glTF metallic-roughness factors map onto our channels: base_color -> color (alpha -> opacity, so a
        dielectric like glass_clear becomes see-through), metallic -> reflect (metals read reflective, consistent with
        the path-tracer adapter), roughness -> roughness, emissive magnitude -> emission. Pass `color` to override the
        preset albedo; the channels stay sockets, so you can still drop a pattern field on any of them afterward."""
        import holographic_matlib as _ml
        m = _ml.material(name)                                       # -> glTF PBRMaterial (raises KeyError if unknown)
        rgb = color if color is not None else m.base_color[:3]
        emis = float(max(m.emissive)) if m.emissive else 0.0        # emissive strength (a scalar drives our channel)
        return cls(color=tuple(rgb), roughness=float(m.roughness), reflect=float(m.metallic),
                   emission=emis, opacity=float(m.base_color[3]))

    def resolve(self, points):
        """All channels at `points` (M,3) -> dict of arrays: color (M,3); roughness/reflect/emission/opacity (M,).
        This is THE per-hit resolution step demo backends used to hand-roll."""
        n = len(points)
        sc = lambda ch: np.asarray(resolve_param(ch, points=points, n=n), float).reshape(-1)[:n] \
            if np.ndim(resolve_param(ch, points=points, n=n)) else np.full(n, float(resolve_param(ch, points=points, n=n)))
        def scalar(ch):
            v = np.asarray(resolve_param(ch, points=points, n=n), float)
            return np.full(n, float(v)) if v.ndim == 0 else v.reshape(n)
        return {"color": _rgb(self.color, points, n), "roughness": scalar(self.roughness),
                "reflect": scalar(self.reflect), "emission": scalar(self.emission),
                "opacity": scalar(self.opacity)}


def render_surface(sdf, camera, width, height, materials, light_dir=(0.45, 0.75, 0.35),
                   sky=None, ambient=0.12, background=(0.55, 0.65, 0.82)):
    """Render an SDF scene with PER-HIT Param-channel resolution -- the render call the SurfaceMaterial exists for.
    `materials` maps object id -> SurfaceMaterial (or a single SurfaceMaterial for the whole SDF). Shading: Lambert
    diffuse * resolved color, Blinn specular sharpened by (1 - roughness), an environment reflection mixed in by
    `reflect` (the reflected ray samples `sky`), `emission` added, and `opacity` alpha-composited over ONE
    continuation ray. Returns (H, W, 3). Honest scope in the module docstring."""
    from holographic_raymarch import sphere_trace, sdf_normal
    L = np.asarray(light_dir, float); L = L / np.linalg.norm(L)
    if sky is None:
        sky = lambda d: np.clip(d @ L, 0, 1)[:, None] ** 3 * np.array([1.0, 0.97, 0.9]) + np.asarray(background) * 0.6

    eye, dirs = camera.ray_dirs(width, height)
    O = np.broadcast_to(eye, (width * height, 3)).astype(float); D = dirs.reshape(-1, 3)

    def shade_hits(Ph, Dh, ids):
        """Resolve each hit's material channels AT the hit point and shade -- grouped per material id so each
        material's sockets resolve once, vectorised (the dispatch_field pattern, inlined)."""
        N = sdf_normal(sdf, Ph)
        out = np.zeros((len(Ph), 3))
        alpha = np.ones(len(Ph))
        for mid in np.unique(ids):
            sel = ids == mid
            mat = materials[int(mid)] if not isinstance(materials, SurfaceMaterial) else materials
            ch = mat.resolve(Ph[sel])                                # <- the per-hit socket resolution
            n_sel = N[sel]; d_sel = Dh[sel]
            lam = np.clip(n_sel @ L, 0, 1)[:, None]
            H = L - d_sel; H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
            shin = 4.0 + (1.0 - ch["roughness"]) * 60.0             # roughness -> highlight width
            spec = np.clip((n_sel * H).sum(1), 0, 1) ** shin
            refl_dir = d_sel - 2.0 * (d_sel * n_sel).sum(1)[:, None] * n_sel
            env = np.clip(np.asarray(sky(refl_dir), float), 0, 1)
            base = ch["color"] * (ambient + (1 - ambient) * lam)     # Lambert
            r = ch["reflect"][:, None]
            out[sel] = np.clip(base * (1 - r) + env * r + spec[:, None] * (0.15 + r) + ch["emission"][:, None] * ch["color"], 0, 3)
            alpha[sel] = ch["opacity"]
        return out, alpha

    hit, t, P = sphere_trace(sdf, O, D)
    frame = np.zeros((width * height, 3))
    frame[~hit] = np.clip(np.asarray(sky(D[~hit]), float), 0, 1) * 0.5 + np.asarray(background) * 0.5
    if hit.any():
        Ph = P[hit]; Dh = D[hit]; ids = np.asarray(sdf.ids(Ph))
        front, alpha = shade_hits(Ph, Dh, ids)
        behind = np.clip(np.asarray(sky(Dh), float), 0, 1) * 0.5 + np.asarray(background) * 0.5
        transp = alpha < 0.999                                       # ONE continuation layer for transparency
        if transp.any():
            N = sdf_normal(sdf, Ph[transp])
            O2 = Ph[transp] - N * 6e-3 + Dh[transp] * 2e-2           # step through the surface, continue the ray
            h2, t2, P2 = sphere_trace(sdf, O2, Dh[transp])
            if h2.any():
                ids2 = np.asarray(sdf.ids(P2[h2]))
                b2, _ = shade_hits(P2[h2], Dh[transp][h2], ids2)     # what is behind, shaded with ITS material
                bt = behind[transp]; bt[h2] = b2; behind[transp] = bt
        frame[hit] = front * alpha[:, None] + behind * (1 - alpha[:, None])
    return np.clip(frame.reshape(height, width, 3), 0, 1)


def _selftest():
    """The tie-together works end to end: a pattern drives a channel through a Param, resolves per hit, and the
    rendered surface actually VARIES with the pattern; from_name consumes the canonical table; opacity composites."""
    from holographic_pattern import make_pattern, field_lerp
    from holographic_render import Camera

    class Balls:
        cs = np.array([[0.0, 0, 0], [1.9, 0, 0]])
        def eval(s, P): return np.min(np.stack([np.linalg.norm(P - c, axis=1) - 0.85 for c in s.cs]), axis=0)
        def ids(s, P): return np.argmin(np.stack([np.linalg.norm(P - c, axis=1) for c in s.cs]), axis=0)

    checker_col = field_lerp(make_pattern("checker", scale=2.5), (0.85, 0.2, 0.15), (0.95, 0.9, 0.85))
    m0 = SurfaceMaterial(color=Param(field=checker_col), roughness=0.35)          # textured via a Param socket
    m1 = SurfaceMaterial.from_name("metal", color=(0.8, 0.8, 0.85))               # from the ONE canonical table
    m1.opacity = 0.55                                                             # and a socket override on top
    cam = Camera(eye=(0.9, 1.0, 4.6), target=(0.9, 0, 0), fov_deg=52)
    img = render_surface(Balls(), cam, 72, 72, {0: m0, 1: m1})
    assert img.shape == (72, 72, 3) and np.isfinite(img).all()

    # the checker really is a solid texture: the textured ball's pixels are NOT one flat colour
    img_flat = render_surface(Balls(), cam, 72, 72, {0: SurfaceMaterial(color=(0.9, 0.5, 0.5)), 1: m1})
    assert img.std() > img_flat.std() * 1.02                                       # pattern adds variation
    # from_name pulled the canonical channels (metal reflect 0.45 in MATERIAL_RENDER)
    assert abs(float(np.mean(m1.resolve(np.zeros((4, 3)))["reflect"])) - 0.45) < 1e-9
    # opacity < 1 lets the background/behind contribute: transparent render differs from opaque
    m1o = SurfaceMaterial.from_name("metal", color=(0.8, 0.8, 0.85))
    img_op = render_surface(Balls(), cam, 72, 72, {0: m0, 1: m1o})
    assert not np.allclose(img, img_op)
    print("holographic_surface selftest OK: pattern->Param->channel->per-hit shade varies; from_name uses the one "
          "canonical table; opacity composites one layer")


if __name__ == "__main__":
    _selftest()
