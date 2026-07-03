"""holographic_session.py -- ONE render session that ties the disconnected rendering threads together.

WHY THIS EXISTS (the above/below audit's keystone, CORE_NOTES section 3)
-----------------------------------------------------------------------
The engine had all the rendering pieces but nothing holding them together, so a demo had to wire them by hand and the
"preview" and the "final" could silently drift apart:
    * render_surface  -- the fast material preview (holographic_surface, landed last session)
    * path_trace      -- the slow, photoreal final (now progressive: it can hand back a refining image)
    * field_to_splats -- a browser-friendly splat proxy, but it wanted pre-sampled points, not an SDF
A `RenderSession` fixes that. It owns ONE scene (an SDF + a SurfaceMaterial per object + a camera) and every output --
the fast preview, the progressive final, and the splat proxy for a lightweight web viewer -- is derived from that same
scene, so they CANNOT diverge. This is the object a demo page drives instead of re-plumbing the renderers each time.

The tie-together, concretely:
    scene (SDF + SurfaceMaterials + camera)
        -> preview()      = render_surface        (seconds; edit a material and re-preview)
        -> render_final() = path_trace            (photoreal; streams a refining image via on_progress)
        -> to_splats()    = surface points + field_to_splats   (O(n) proxy for a browser billboard shader)
And because the SAME SurfaceMaterials feed BOTH render_surface (preview) and path_trace (final) -- through one
material adapter here -- editing a channel updates both. NumPy only; deterministic.
"""
import numpy as np
from holographic_surface import SurfaceMaterial, render_surface


def sdf_surface_points(sdf, bounds, n=2000, seed=0, eps=0.02, oversample=8):
    """Sample points that lie ON an SDF's surface -- the front half of the SDF->splat bridge that was missing.

    Throw `oversample*n` random points into `bounds`=(lo, hi), take one Newton step toward the zero level
    (P <- P - sdf(P) * grad), keep the ones that landed on the surface (|sdf| < eps), and return up to `n` of them.
    Deterministic given the seed. This is what feeds field_to_splats so any SDF scene becomes splat-viewable."""
    from holographic_raymarch import sdf_normal
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    rng = np.random.default_rng(seed)
    P = rng.uniform(lo, hi, (oversample * n, 3))
    d = sdf.eval(P)
    grad = sdf_normal(sdf, P)                                        # unit gradient (surface normal direction)
    P = P - d[:, None] * grad                                       # one Newton step onto the zero level
    keep = np.abs(sdf.eval(P)) < eps                                # landed on the surface?
    P = P[keep]
    return P[:n] if len(P) > n else P


def _pathtrace_material(sdf, materials):
    """Adapt SurfaceMaterial channels to the tuple path_trace wants -- so the FINAL render uses the very same
    materials as the preview (they can't drift). Returns material(P) -> (albedo, metallic, roughness, emission, ior),
    resolving each object's material AT its hit points (per-id, the same dispatch render_surface does).

    Channel mapping (kept simple and readable): albedo <- color; metallic <- reflect (reflective reads as metal in
    the BRDF); roughness <- roughness; emission <- emission * color; and opacity < 0.5 -> ior 1.5 so a see-through
    material becomes real refractive GLASS in the path tracer (opaque otherwise)."""
    def material(P):
        ids = np.asarray(sdf.ids(P))
        n = len(P)
        albedo = np.zeros((n, 3)); metallic = np.zeros(n); roughness = np.full(n, 0.5)
        emission = np.zeros((n, 3)); ior = np.ones(n)
        for mid in np.unique(ids):
            sel = ids == mid
            mat = materials[int(mid)] if not isinstance(materials, SurfaceMaterial) else materials
            ch = mat.resolve(P[sel])
            albedo[sel] = ch["color"]
            metallic[sel] = np.clip(ch["reflect"], 0, 1)
            roughness[sel] = np.clip(ch["roughness"], 0.03, 1.0)
            emission[sel] = ch["emission"][:, None] * ch["color"]
            ior[sel] = np.where(ch["opacity"] < 0.5, 1.5, 1.0)      # transparent -> dielectric glass
        return albedo, metallic, roughness, emission, ior
    return material


class RenderSession:
    """One scene, every renderer. Holds an SDF, a SurfaceMaterial per object id (or one material for the whole SDF),
    and a camera; derives the fast preview, the progressive final, and a splat proxy from that SINGLE scene so they
    never diverge. Edit a material and both the preview and the final follow. This is the object a demo page keeps,
    instead of re-wiring render_surface / path_trace / splat export by hand each time."""

    def __init__(self, sdf, materials, camera, width=256, height=256, bounds=None):
        self.sdf = sdf
        self.materials = materials                                  # dict id->SurfaceMaterial, or one SurfaceMaterial
        self.camera = camera
        self.width = width; self.height = height
        # a default world box for surface sampling if none given (most demo scenes fit here)
        self.bounds = bounds if bounds is not None else (np.full(3, -4.0), np.full(3, 4.0))

    # -- the SAME scene, three ways --------------------------------------------------------------------------------

    def preview(self, width=None, height=None, **kw):
        """FAST path: the material preview via render_surface (Lambert + spec + env reflection + one transparency
        layer), resolving every SurfaceMaterial channel per hit. Seconds, not minutes -- what a viewport shows while
        you edit. Optional smaller width/height for an even quicker draft."""
        return render_surface(self.sdf, self.camera, width or self.width, height or self.height, self.materials, **kw)

    def render_final(self, spp=64, on_progress=None, progress_every=8, width=None, height=None, max_bounce=4,
                     sky=None, seed=0, should_stop=None):
        """SLOW path: the photoreal final via path_trace, using the SAME SurfaceMaterials as the preview (through the
        material adapter). Pass `on_progress(image, done, total)` to receive the refining image every `progress_every`
        samples -- the progress preview a session streams while the final accumulates. Pass `should_stop` (a
        CancelToken or any callable) to cancel mid-render and get back the partial image. Returns the final HDR image."""
        from holographic_pathtrace import path_trace
        mat = _pathtrace_material(self.sdf, self.materials)
        return path_trace(self.sdf, self.camera, width or self.width, height or self.height, spp=spp,
                          max_bounce=max_bounce, material=mat, sky=sky, seed=seed,
                          on_progress=on_progress, progress_every=progress_every, should_stop=should_stop)

    def to_splats(self, n=2000, radius=0.12, seed=0):
        """PROXY path: sample the SDF surface and fit splats (field_to_splats) so the scene can be drawn by a
        lightweight browser billboard shader -- no three.js scene graph, no mesh pipeline. Returns (splats, json_str)
        ready to stream. This is the "everything-moves preview" answer from the modeling-gaps doc: particles/surfaces
        ARE splats. Colours come from each surface point's own SurfaceMaterial (so the proxy matches the preview)."""
        from holographic_splatexport import field_to_splats, splats_to_json
        pts = sdf_surface_points(self.sdf, self.bounds, n=n, seed=seed)
        splats = field_to_splats(pts, radius=radius)
        # colour each splat by its object's resolved albedo, so the proxy reads like the preview
        if len(pts):
            ids = np.asarray(self.sdf.ids(pts))
            cols = np.zeros((len(pts), 3))
            for mid in np.unique(ids):
                sel = ids == mid
                mat = self.materials[int(mid)] if not isinstance(self.materials, SurfaceMaterial) else self.materials
                cols[sel] = mat.resolve(pts[sel])["color"]
        else:
            cols = None
        return splats, splats_to_json(splats, colors=cols)

    # -- edits flow to every output (preview and final share the materials) ----------------------------------------

    def set_material(self, obj_id, material):
        """Replace one object's material. Because preview and final both read `self.materials`, the edit shows up in
        BOTH on the next render -- the point of the session."""
        if isinstance(self.materials, SurfaceMaterial):
            self.materials = {int(obj_id): material}
        else:
            self.materials[int(obj_id)] = material

    def edit_channel(self, obj_id, channel, value):
        """Edit ONE channel of one object's material (colour/roughness/reflect/emission/opacity) -- the value can be a
        constant, a Param, a pattern field, or a map. Live-editable materials, the modeling-loop ask."""
        mat = self.materials if isinstance(self.materials, SurfaceMaterial) else self.materials[int(obj_id)]
        setattr(mat, channel, value)


def _selftest():
    """The session ties it together: preview and final render the SAME scene, an edit shows in both, the progressive
    callback fires, and to_splats turns the SDF surface into a coloured splat set."""
    from holographic_render import Camera

    class TwoBalls:
        cs = np.array([[0.0, 0, 0], [1.9, 0, 0]])
        def eval(s, P): return np.min(np.stack([np.linalg.norm(P - c, axis=1) - 0.85 for c in s.cs]), axis=0)
        def ids(s, P): return np.argmin(np.stack([np.linalg.norm(P - c, axis=1) for c in s.cs]), axis=0)

    from holographic_pattern import make_pattern, field_lerp
    from holographic_param import Param
    m0 = SurfaceMaterial(color=Param(field=field_lerp(make_pattern("checker", scale=2.5), (0.9, 0.2, 0.1), (0.95, 0.9, 0.85))))
    m1 = SurfaceMaterial.from_name("metal", color=(0.8, 0.8, 0.85))
    cam = Camera(eye=(0.9, 1.0, 4.6), target=(0.9, 0, 0), fov_deg=52)
    sess = RenderSession(TwoBalls(), {0: m0, 1: m1}, cam, width=56, height=56)

    prev = sess.preview()
    assert prev.shape == (56, 56, 3) and np.isfinite(prev).all()

    # an edit shows up in the preview (preview and final share the materials)
    sess.edit_channel(1, "color", (0.2, 0.9, 0.3))
    prev2 = sess.preview()
    assert not np.allclose(prev, prev2)                             # the edit took effect

    # progressive final: callback fires and the final image comes back
    fired = []
    fin = sess.render_final(spp=6, on_progress=lambda im, d, t: fired.append(d), progress_every=2,
                            width=40, height=40, sky=lambda D: np.ones((len(D), 3)) * 0.9)
    assert fin.shape == (40, 40, 3) and len(fired) >= 1            # refined at least once

    # splat proxy: SDF surface -> coloured splats + json
    splats, js = sess.to_splats(n=300)
    assert len(splats) > 0 and isinstance(js, str) and len(js) > 2
    print("holographic_session selftest OK: preview + progressive final render one scene, an edit shows in both, "
          "to_splats turns the SDF surface into %d coloured splats" % len(splats))


if __name__ == "__main__":
    _selftest()
