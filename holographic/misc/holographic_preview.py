"""holographic_preview.py -- SEE what you composed: a flat swatch for a texture graph, a shaded ball for a material.

The composability stack (CMP1-CMP5) builds things you sample with .sample(uv) -- texture graphs, multi/layered
materials. This module renders them to a small RGB image you can save and look at, which is the missing last step
between "I composed a material" and "does it look right?". Two previews, the two every DCC tool gives you:

  * texture_image(graph)   -- a flat SWATCH of a CMP1 texture graph (its colour over the UV square). Instant.
  * material_ball(material) -- the classic MATERIAL BALL: the material shaded on a sphere with a Cook-Torrance BRDF
                               and one light, so its roughness / metallic / albedo read the way they will on a
                               curved surface.

Both return a float image in [0,1], shape (res, res, 3) -- save it with any image writer (PIL, imageio) or hand it
straight to a viewer. Deterministic; plain NumPy.

Reuses: holographic_texturegraph.sample_grid (the swatch is a graph evaluated on a grid), holographic_brdf
(cook_torrance -- the same BRDF the real renderer uses, so a preview matches the render). The only per-pixel loop is
sampling the material over the sphere's visible pixels; everything else is vectorised.
"""
import numpy as np


def texture_image(graph, res=256, lo=0.0, hi=1.0):
    """A CMP1 texture graph as a flat RGB SWATCH: evaluate it over the UV square and return a (res,res,3) image in
    [0,1]. A colour graph shows its rgb; a scalar graph shows as greyscale. Values are CLAMPED to [0,1] for display
    (composition can legitimately push a value out of range -- see the CMP1 'saturate' op if you want that baked in)."""
    from holographic.materials_and_texture.holographic_texturegraph import sample_grid
    grid = np.asarray(sample_grid(graph, res=res, lo=lo, hi=hi), float)
    if grid.ndim == 2:                                      # scalar field -> greyscale
        grid = np.repeat(grid[:, :, None], 3, axis=2)
    grid = grid[:, :, :3]                                   # drop an alpha channel if the graph produced rgba
    return np.clip(grid, 0.0, 1.0)


def _channel_names(material):
    """The channels a material exposes, whether it's a plain Material (.channels dict) or a CMP2/CMP3 material
    (.channel_names())."""
    if hasattr(material, "channel_names"):
        return set(material.channel_names())
    return set(getattr(material, "channels", {}))


def _sample_scalar(material, name, uv, default):
    """Sample a scalar channel at uv, clamped to [0,1]; `default` if the material doesn't have that channel. Handles a
    channel that returns a colour by taking its mean (so a preview never crashes on an unexpected shape)."""
    if name not in _channel_names(material):
        return float(default)
    v = np.asarray(material.sample(name, uv), float)
    return float(np.clip(v if v.ndim == 0 else v.mean(), 0.0, 1.0))


def material_ball(material, res=192, base_color=(0.82, 0.80, 0.78), light_dir=(0.6, 0.7, 0.5),
                  background=0.14, ambient=0.06):
    """Render `material` on a preview SPHERE -- the standard 'material ball'. Works on a plain Material or a CMP2/CMP3
    layered/multi material (anything with .sample(channel, uv) + channels). Uses the material's `roughness` and
    `metallic` channels where present (else sensible defaults), and modulates `base_color` by an `albedo` channel if
    there is one. Shades with the same Cook-Torrance BRDF the real renderer uses, so the ball matches a render.
    Returns a (res, res, 3) float image in [0,1].

    Orthographic camera down -z onto a unit sphere at the origin, one directional light. The only loop is sampling the
    material at each visible pixel's UV; the shading is vectorised."""
    from holographic.rendering.holographic_brdf import cook_torrance

    base_color = np.asarray(base_color, float)
    L = np.asarray(light_dir, float)
    L = L / (np.linalg.norm(L) + 1e-12)
    V = np.array([0.0, 0.0, 1.0])                           # orthographic view direction (toward the camera)

    # image-plane coords in [-1.2, 1.2]; a pixel hits the sphere when x^2 + y^2 <= 1 (front hemisphere)
    xs = np.linspace(-1.2, 1.2, res)
    X, Y = np.meshgrid(xs, xs)
    Y = -Y                                                  # image row 0 at top
    r2 = X * X + Y * Y
    hit = r2 <= 1.0
    Z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))               # front-surface z on the unit sphere

    img = np.empty((res, res, 3), float)
    img[:] = _background(res, background)                   # neutral vertical gradient behind the ball

    ph, pw = np.where(hit)                                  # the pixels that land on the sphere
    P = np.stack([X[ph, pw], Y[ph, pw], Z[ph, pw]], axis=1)   # surface points (M,3)
    N = P                                                   # on a unit sphere the point IS the normal
    # spherical UVs: u around the equator, v from pole to pole -- a stable, seam-simple mapping for a preview
    u = np.arctan2(P[:, 0], P[:, 2]) / (2.0 * np.pi) + 0.5
    v = np.arccos(np.clip(P[:, 1], -1.0, 1.0)) / np.pi

    # sample the material per visible pixel (the one loop) -> albedo tint, roughness, metallic
    has_albedo = "albedo" in _channel_names(material)
    alb = np.empty((len(P), 3))
    rough = np.empty(len(P))
    metal = np.empty(len(P))
    for i in range(len(P)):
        uv = (float(u[i]), float(v[i]))
        rough[i] = _sample_scalar(material, "roughness", uv, 0.5)
        metal[i] = _sample_scalar(material, "metallic", uv, 0.0)
        # albedo channel (scalar) modulates the base tint; if the material has none, the tint is used as-is
        alb[i] = base_color * (_sample_scalar(material, "albedo", uv, 1.0) if has_albedo else 1.0)

    # shade all sphere pixels at once with Cook-Torrance + a touch of ambient, tone-mapped into [0,1]
    Nv = np.repeat(N[None, :, :], 1, axis=0)[0]            # (M,3) already
    shaded = cook_torrance(Nv, np.broadcast_to(V, N.shape), np.broadcast_to(L, N.shape), alb, metal, rough)
    shaded = shaded + ambient * alb                        # small ambient so shadowed side isn't pure black
    shaded = shaded / (1.0 + shaded)                       # Reinhard tone-map -> [0,1)
    img[ph, pw] = np.clip(shaded, 0.0, 1.0)
    return img


def _background(res, level):
    """A neutral vertical gradient behind the ball, so it reads as sitting in a soft studio rather than on flat grey."""
    col = np.linspace(level * 1.6, level * 0.7, res)[:, None]
    return np.repeat(np.repeat(col[:, :, None], res, axis=1), 3, axis=2)


# ---------------------------------------------------------------- the shader-ball PREVIEW SCENE (three slots)
#
# The material ball above is the fast, flat-lit thumbnail. This is the other preview every DCC tool ships: a full
# SCENE -- the shader ball (sphere on a collar ring on a pedestal) standing on a floor, path-traced with the real
# renderer, so reflections, soft shading and material-vs-material contrast read the way they will in a render.
# It REUSES the whole existing stack (Scene document + SDF primitives + matlib + render_scene_document); nothing
# here re-implements shading. Slot contract, from the user's mouth: one material fills every slot by default; name
# `trim=` / `base=` only when you want three materials displayed at once.

_PREVIEW_SLOT_NAMES = ("outer", "core", "trim", "base")         # the material slots
_PREVIEW_DEFAULT_MATERIAL = "matte_gray"                        # outer's neutral default diffuse (library name)


def _default_core():
    """The default CORE: a soothing dark grey diffuse -- the ball out of a 90s mouse. Built fresh per call so
    callers can't mutate a shared instance."""
    from holographic.materials_and_texture.holographic_materialio import PBRMaterial
    return PBRMaterial(name="mouse_ball_gray", base_color=(0.16, 0.16, 0.17), metallic=0.0, roughness=0.85)


def _default_trim():
    """The default for BOTH belts: the same soothing grey diffuse as the core and base (user direction -- one
    default family for every fixture slot; the hero material stands alone on the outer)."""
    return _default_core()


def _coerce_preview_material(material):
    """Accept the three shapes a caller reasonably has -- a matlib library NAME ('gold'), an already-built
    PBRMaterial-like object, or a plain dict of PBR numbers ({'roughness':0.2,'metallic':1.0,...}) -- and return
    something the Scene document / matlib.shade path understands. A dict becomes a real PBRMaterial so the preview
    and a later render agree on what those numbers mean (one definition, not two)."""
    if material is None:
        return _PREVIEW_DEFAULT_MATERIAL
    if isinstance(material, dict):
        from holographic.materials_and_texture.holographic_materialio import PBRMaterial
        kw = dict(material)
        bc = kw.pop("base_color", kw.pop("color", (0.8, 0.8, 0.8)))
        return PBRMaterial(name=kw.pop("name", "preview"), base_color=tuple(bc),
                           metallic=float(kw.pop("metallic", 0.0)), roughness=float(kw.pop("roughness", 0.8)),
                           emissive=tuple(kw.pop("emissive", (0.0, 0.0, 0.0))))
    return material                                             # a library name (str) or a material object


def preview_grid_albedo(P):
    """Graph-paper albedo for the preview floor, evaluated at world points (the albedo_socket contract:
    f(P (M,3)) -> (M,3) rgb). Light ground, fine lines every 0.30, heavier major lines every 1.50 -- the
    reference-sheet floor every DCC preview uses, because a patterned floor makes reflections, refraction and
    the contact shadow READ where a flat one hides them. Line widths are in world units, deliberately wide
    enough to survive minification at preview resolutions."""
    P = np.atleast_2d(np.asarray(P, float))

    def lines(cell, half):
        g = np.minimum(np.abs(((P[:, 0] / cell) % 1.0) - 0.5) * 2,
                       np.abs(((P[:, 2] / cell) % 1.0) - 0.5) * 2)
        return np.clip(g / half, 0.0, 1.0)             # 0 on a line, 1 mid-cell

    fine = lines(0.30, 0.14)
    major = lines(1.50, 0.05)
    ground = np.array([0.88, 0.88, 0.89]); line = np.array([0.46, 0.50, 0.56]); mline = np.array([0.28, 0.33, 0.40])
    col = np.repeat(ground[None, :], len(P), 0)
    col = col * fine[:, None] + line[None, :] * (1 - fine[:, None])
    col = col * major[:, None] + mline[None, :] * (1 - major[:, None])
    return col


def preview_scene_document(material=None, core=None, trim=None, base=None, floor="matte_white", floor_grid=True,
                           trim_top=None, trim_bottom=None):
    """Build the standard shader-ball preview SCENE as a Scene document and its framing camera -- geometry only,
    no pixels. The preview object is a COMPLEX solid, like Blender's ball or Substance's droid: a hollow outer
    SHELL with a camera-facing cutaway window, a THIN LENS dish (wall thinned to 0.012 -- the translucency/SSS
    test region, and a low-refraction view of the core), a CORE flush against the shell interior, TWO FLUSH
    INLAY BELTS ('trim_top' above the window and lens dish, 'trim_bottom' below the window -- cut INTO the
    ball as shell partitions, so the surface stays perfectly smooth with no outward bumps) and a wide thin
    puck ('base'), standing on a 'floor' -- proportions in the style of the classic DCC shader
    balls (sphere dominant, flush on a base of nearly its own radius). The core is the slot for the interacting cases -- an emissive core glows
    through a glass/translucent outer, and the cutaway keeps the core visible even under an opaque outer.
    Slot rule (revised): `material` dresses the OUTER; the other slots default to the mouse-ball grey diffuse
    on every fixture slot (core, base, both belts); trim= dresses BOTH belts, trim_top= / trim_bottom=
    override each individually, core= / base= the others.
    material=None -> the neutral default diffuse on the outer. floor_grid=True (default) rides a
    graph-paper albedo_socket on the floor (the existing per-point-albedo override the scene renderer already
    honours); floor_grid=False leaves the floor material's flat colour. Returns (scene, camera) -- hand them
    to render_scene_document yourself for turntables, or call preview_scene() for pixels. The floor is its own
    object with its own material, so the environment is styleable too."""
    import numpy as np
    from holographic.scene_and_pipeline.holographic_scene_doc import Scene
    from holographic.mesh_and_geometry.holographic_sdf import sphere, box, cylinder, plane
    from holographic.rendering.holographic_render import Camera

    main = _coerce_preview_material(material)
    # THE slot rule (revised on user direction): `material` dresses the OUTER; the other slots carry their own
    # sensible defaults -- mouse-ball grey core, base AND both belts -- each overridable with
    # core= / trim= / base=. One assignment styles the hero surface without repainting the fixtures.
    # trim= dresses BOTH belts; trim_top= / trim_bottom= override each individually (a glass top belt over a
    # chrome bottom belt is the user's canonical multi-material demo).
    _trim_both = _coerce_preview_material(trim) if trim is not None else None
    mats = {"outer": main,
            "core": _coerce_preview_material(core) if core is not None else _default_core(),
            "trim_top": (_coerce_preview_material(trim_top) if trim_top is not None
                         else (_trim_both if _trim_both is not None else _default_trim())),
            "trim_bottom": (_coerce_preview_material(trim_bottom) if trim_bottom is not None
                            else (_trim_both if _trim_both is not None else _default_trim())),
            "base": _coerce_preview_material(base) if base is not None else _default_core()}

    # Geometry in world space, proportions read off a reference lathe profile (3D-Coat-style ball: max radius per
    # height band of a reference mesh): the sphere DOMINATES and sits flush on a wide, THIN puck base of nearly
    # its own radius -- no tall pedestal. The trim slot is a tilted band around the sphere (half-embedded torus),
    # the classic detail that shows a material at every incidence angle in one image. The camera position
    # participates in the GEOMETRY on purpose: the cutaway window is placed off the view axis (rotated ~35
    # degrees about Y) so it reads as a feature AT AN ANGLE, the way the reference balls present their inset --
    # straight-on it read as a hole staring at the camera. Move the camera without moving the window and the
    # preview shows a hole pointing at nothing.
    center = np.array([0.0, 0.74, 0.0])                        # sphere centre: puck top (0.14) + radius, flush
    # Framing (user direction, measured): CLOSE view -- the base rim sits at the bottom frame edge (1 row of
    # 96 spare, so the rim never slices at higher res) with headroom above for displaced materials. The camera
    # PARTICIPATES in geometry (window and lens aim off the view direction), so moving the eye re-aims both.
    eye = np.array([1.18, 1.12, 1.82])                         # 3/4 view, close
    d = eye - center; d = d / np.linalg.norm(d)
    th = 0.62                                                  # window azimuth off the view axis (radians, ~35 deg)
    c_, s_ = np.cos(th), np.sin(th)
    wd = np.array([[c_, 0, s_], [0, 1, 0], [-s_, 0, c_]]) @ d  # view direction swung left about Y
    window = center + 0.60 * wd                                # cutaway centred where the swung ray meets the shell

    shell = sphere(0.60).translate(tuple(center)).subtract(sphere(0.52).translate(tuple(center)))  # hollow, 0.08 wall
    shell = shell.subtract(sphere(0.26).translate(tuple(window)))                                  # the window
    # THE THIN LENS: a shallow dish on the other side of the view axis that thins the wall from 0.08 to 0.012
    # over the core -- the translucency test region. Through 0.012 of glass the core is visible with almost no
    # refraction; through 0.012 of wax/skin/jade the SSS and Beer-Lambert terms read (thin = lighter), which an
    # 0.08 wall hides. Placement is a LIGHTING decision as much as a geometric one, measured twice: raised too
    # far the dish mirrors the bright sky and the reflection drowns the transmitted core (sky-facing negative);
    # swung too low it collides with the band. Near-camera-facing with a mild raise keeps the reflection on the
    # darker mid-gradient so the transmission shows.
    lens_u = np.array([[np.cos(-0.45), 0, np.sin(-0.45)], [0, 1, 0], [-np.sin(-0.45), 0, np.cos(-0.45)]]) @ d
    lens_u = lens_u + np.array([0.0, 0.22, 0.0]); lens_u = lens_u / np.linalg.norm(lens_u)
    shell = shell.subtract(sphere(0.34).translate(tuple(center + (0.532 + 0.34) * lens_u)))        # carve to r=0.532
    # TWO FLUSH INLAY BELTS (user direction, third iteration of this slot -- the design lesson each time was
    # about the PROFILE): the belts are cut INTO the ball, flush, no outward bumps. The construction that makes
    # flush TRIVIAL instead of a tolerance fight: PARTITION the shell by height -- each belt is shell-slice
    # (shell INTERSECT slab), the outer is shell MINUS both slabs. The union of the three objects is EXACTLY
    # the original ball surface, so flushness cannot drift, there is no groove/insert gap to keep above the
    # tracer's resolution floor, and the seams are material boundaries only, never geometry. Positions keep
    # clear room around both features: window spans ~y 0.585..1.084, lens dish reaches ~y 1.11 -- bottom belt
    # y0 0.46 (spans 0.419..0.501, ~0.08 below the hole), top belt y0 1.17 (spans 1.136..1.204, above both).
    # Widths are the previous belts minus 25% (user direction): half-heights 0.055->0.041, 0.045->0.034.
    slab_b = box(2.0, 0.041, 2.0).translate((0.0, 0.46, 0.0))
    slab_t = box(2.0, 0.034, 2.0).translate((0.0, 1.17, 0.0))
    belt_bottom = shell.intersect(slab_b)
    belt_top = shell.intersect(slab_t)
    unpartitioned_shell = shell                                # the partition's exact union -- see distance proxy
    shell = shell.subtract(slab_b).subtract(slab_t)

    sc = Scene(seed=0)
    fh = sc.add(name="floor", geometry=plane(0.0), material=_coerce_preview_material(floor))
    if floor_grid:
        # per-point albedo rides the EXISTING albedo_socket override the scene renderer already honours --
        # the floor material keeps its roughness/metallic; only the colour becomes graph-paper.
        sc.edit(fh, overrides={"albedo_socket": preview_grid_albedo})
    sc.add(name="outer", geometry=shell, material=mats["outer"])
    # The core FILLS the interior, flush against the shell -- like the reference balls' inset, the window
    # reveals a continuous inner surface, not a small floating ball. Radius 0.510 vs inner shell 0.52: a 0.010
    # gap, and the SIZE is load-bearing twice over (both measured, both reversed earlier attempts):
    #   * OVERLAP (0.53) merged shell and core into one solid and DELETED the core surface from the union --
    #     glass outers refracted through to nothing;
    #   * a 0.002 HAIR GAP kept the surface but sat BELOW THE TRACER'S GEOMETRIC RESOLUTION: the finite-
    #     difference normal (eps 1e-3) sampled both walls symmetrically and returned the ZERO VECTOR at the
    #     glass exit point, so refract_dir scattered every transmitted ray into grey mush -- the blue core
    #     was invisible through glass while the geometry was "correct".
    # The rule: an air gap must comfortably exceed max(2 x FD eps, ray re-offset 3e-3) ~= 0.006; 0.010 gives
    # margin. Still reads flush at preview scale.
    sc.add(name="core", geometry=sphere(0.510).translate(tuple(center)), material=mats["core"])
    sc.add(name="trim_top", geometry=belt_top, material=mats["trim_top"])
    sc.add(name="trim_bottom", geometry=belt_bottom, material=mats["trim_bottom"])
    sc.add(name="base", geometry=cylinder(0.07, 0.60).translate((0.0, 0.07, 0.0)), material=mats["base"])

    cam = Camera(eye=tuple(eye), target=(0.0, 0.63, 0.0), fov_deg=46.0, aspect=1.0)
    # THE DISTANCE PROXY, stashed on the scene for preview_scene to hand the renderer: the belts partition the
    # shell, so the union of outer + both belts is EXACTLY the unpartitioned shell (the v16 invariant) -- one
    # deep subtree evaluated once instead of three times. material_fn still uses the per-piece trees for
    # attribution; only the marching distance uses this. Measured on the preview render: see preview_scene.
    from holographic.mesh_and_geometry.holographic_sdf import sphere as _sp, cylinder as _cy, plane as _pl
    class _PreviewDistance:
        def __init__(self):
            self._parts = [unpartitioned_shell,
                           _sp(0.510).translate(tuple(center)),
                           _cy(0.07, 0.60).translate((0.0, 0.07, 0.0)),
                           _pl(0.0)]
        def eval(self, P):
            d = np.asarray(self._parts[0](np.atleast_2d(np.asarray(P, float))), float)
            for g in self._parts[1:]:
                d = np.minimum(d, np.asarray(g(np.atleast_2d(np.asarray(P, float))), float))
            return d
    sc.preview_distance_sdf = _PreviewDistance()
    return sc, cam


def preview_scene_lighting():
    """The studio RIG for the preview scene: (lights, sky). Three softboxes -- key (bright, high, camera-left),
    fill (broad, dim, camera-right, kills dead-black shadows), rim (behind, separates the ball from the backdrop)
    -- plus a soft grey gradient sky (bright toward the horizon, dark overhead), the neutral cyc-wall every
    product/material photo uses. Built with holographic_lights.make_light, the same one-door builder the rest of
    the engine uses. Returned separately from the document so turntables can re-light without rebuilding geometry."""
    import numpy as np
    from holographic.rendering.holographic_lights import make_light
    center = (0.0, 0.74, 0.0)
    # TWO boxes, not three, and measured: the rim box's separation job is already done by the gradient backdrop,
    # and each softbox pays per-sample area sampling (the cache that would amortise it is broken on mirrors --
    # see preview_scene). Dropping the rim: no visible loss on the copper/glass test frames, ~20% render saved.
    lights = [make_light("softbox", position=(2.4, 3.0, 1.6), target=center, width=2.2, height=2.2, intensity=55.0),
              make_light("softbox", position=(-2.6, 1.6, 2.2), target=center, width=2.6, height=2.0, intensity=14.0)]

    def studio_sky(D):
        # WHY a gradient and not a flat colour: metals and glass are mirrors of their environment -- a flat sky
        # renders them as flat discs. A vertical gradient gives every reflected ray a different value, which is
        # what makes 'shiny' legible. Bright low / dark high is the studio cyc-wall convention.
        D = np.atleast_2d(np.asarray(D, float))
        t = np.clip(D[..., 1] * 0.5 + 0.5, 0.0, 1.0)
        lo = np.array([0.82, 0.82, 0.84]); hi = np.array([0.26, 0.28, 0.32])
        col = lo[None, :] * (1 - t)[:, None] + hi[None, :] * t[:, None]
        # FLUORESCENT CEILING PANELS, for steep upward rays only (D.y > 0.40): a grid of soft-edged HDR-bright
        # rectangles on a virtual ceiling plane. The threshold is the trick -- reflections off the top of the
        # ball leave steeply and SEE the panels (the streaked highlights every product photo has), while the
        # near-horizontal background rays behind the object see only the clean gradient. Panels visible in the
        # direct view were measured as distracting stripes; this is the fix, kept.
        up = D[..., 1] > 0.40
        if np.any(up):
            px = D[up, 0] / D[up, 1]; pz = D[up, 2] / D[up, 1]      # project the ray onto the ceiling plane
            dx = np.abs(((px / 1.6) % 1.0) - 0.5); dz = np.abs(((pz / 0.9) % 1.0) - 0.5)
            s = np.clip((0.30 - dx) / 0.06, 0, 1) * np.clip((0.35 - dz) / 0.06, 0, 1)   # soft panel edges
            col[up] = col[up] + s[:, None] * np.array([2.6, 2.6, 2.5])[None, :]
        return col

    return lights, studio_sky


def preview_scene(material=None, core=None, trim=None, base=None, floor="matte_white",
                  res=192, quality="fast", seed=0, view="display", lighting="studio", floor_grid=True,
                  aa="fxaa", trim_top=None, trim_bottom=None):
    """Render the shader-ball PREVIEW SCENE: `material` on the classic complex preview object -- a hollow outer
    shell with an off-axis cutaway window, a THIN LENS dish (see the core with almost no refraction; wax/skin/
    jade show their translucency there), a CORE flush with the shell interior, two FLUSH INLAY belts (trim_top/trim_bottom, cut into the ball) and a puck base -- on a floor,
    path-traced by the real renderer under a STUDIO RIG (key/fill/rim softboxes, gradient backdrop, a graph-paper
    floor -- floor_grid=False for a plain floor -- and fluorescent ceiling panels visible in reflections; the flat-lit
    material_ball is the fast thumbnail). The core slot is for the interacting cases:
    preview_scene('glass_clear', core='neon_blue') shows the emissive core glowing THROUGH the glass shell;
    under an opaque outer the cutaway keeps the core visible. Slot rule: `material` dresses the OUTER; every fixture
    slot (core, base, both belts) defaults to the mouse-ball grey diffuse; trim= dresses both belts,
    trim_top= / trim_bottom= override each belt individually.
    Materials may be matlib names ('gold'), material objects, or plain PBR dicts. lighting='studio' (default) is
    the rig from preview_scene_lighting; lighting='plain' is the renderer's bare defaults (the pre-rig look, kept
    reachable). aa='fxaa' (default) cleans the stair-stepped edges at the
    same resolution for milliseconds; aa='ssaa2' renders at 2x and box-downsamples (true SSAA, ~4x time);
    aa='off' is the raw tracer output. Returns a (res, res, 3) float image in [0,1].
    COST, measured with the distance proxy: ~75 s at res=160 opaque (2.8x over the plain union -- the proxy
    evaluates the partition's shared shell subtree once; image diff below seed-to-seed noise); glass belts
    cost more; drop res to iterate. The soft-light cache is OFF because it paints
    false shadows on curved mirrors (see preview_scene_lighting); ssaa2 ~4x that. 'fast' is not a real quality
    preset (draft/medium/high/ultra); unknown names fall through to medium.
    See preview_scene_document for the geometry contract."""
    import numpy as np
    from holographic.rendering.holographic_scene_render import render_scene_document
    sc, cam = preview_scene_document(material=material, core=core, trim=trim, base=base, floor=floor,
                                     floor_grid=floor_grid, trim_top=trim_top, trim_bottom=trim_bottom)

    def _render_one(r):
        if lighting == "studio":
            lights, sky = preview_scene_lighting()
            # soft_light_cache stays OFF here, and this is a MEASURED reversal of v4: the cache's screen-space
            # interpolation assumes the shaded soft-light term varies smoothly across the image, but on a curved
            # MIRROR the softbox term varies with the REFLECTION vector, not screen position -- on the copper
            # ball it painted a large false dark crescent below the band plus milky streaks (A/B at 192px:
            # cache-on artifacted at 28 s, cache-off correct at 92 s with 3 boxes / ~73 s with 2). Correctness
            # wins; drop `res` for the fast loop. The cache remains right for the diffuse scenes it shipped on.
            # Translucency wiring: sss_dir points at the key softbox so the EXTERNAL subsurface term is live
            # (it is inert without a direction), and sss_interior=True turns on the interior-emission term --
            # an emissive core glows through thin translucent walls (wax/skin/jade), brightest at the lens.
            # emissive_mesh_lights TOGGLES BY THE OUTER'S CLASS (user direction): ON for translucent/SSS
            # outers (wax/skin/jade -- a glowing body inside a candle-like shell is treated as a light; its
            # measurable contribution is the beam through the open window, since NEE occlusion through the
            # wall itself stays binary -- documented scope), OFF for glass/refractive/transparent (the
            # refraction path already carries the emission; the earlier sealed-glass A/B measured the mesh
            # light as noise at 2x cost) and OFF for opaque outers.
            key_dir = np.array([2.4, 2.26, 1.6]); key_dir = key_dir / np.linalg.norm(key_dir)
            return render_scene_document(sc, cam, r, r, quality=quality, seed=seed, view=view,
                                         lights=lights, sky=sky, soft_light_cache=False,
                                         sss_dir=tuple(key_dir), sss_depth=0.30, sss_sigma=20.0,
                                         sss_interior=True,
                                         emissive_mesh_lights=_outer_is_translucent(material),
                                         distance_sdf=getattr(sc, "preview_distance_sdf", None))
        if lighting == "plain":
            return render_scene_document(sc, cam, r, r, quality=quality, seed=seed, view=view,
                                         distance_sdf=getattr(sc, "preview_distance_sdf", None))
        raise ValueError("lighting must be 'studio' or 'plain', got %r" % (lighting,))

    # ANTI-ALIASING fork, priced honestly: 'fxaa' (default) is the same-res edge-masked subpixel pass --
    # milliseconds, flat regions bit-identical; 'ssaa2' renders at 2x and box-averages down (postfx.supersample)
    # -- the quality answer at ~4x render time; 'off' is the raw tracer output.
    if aa == "fxaa":
        from holographic.rendering.holographic_postfx import fxaa as _fxaa
        return _fxaa(_render_one(int(res)))
    if aa == "ssaa2":
        from holographic.rendering.holographic_postfx import supersample as _ss
        return _ss(_render_one(int(res) * 2), factor=2)
    if aa in ("off", None):
        return _render_one(int(res))
    raise ValueError("aa must be 'fxaa', 'ssaa2', or 'off', got %r" % (aa,))


def _area_resize(img, out_h, out_w):
    """Anti-aliased resize to ANY size: integer box-average down as far as the sizes allow, then a bilinear
    step for the residual ratio. Exists so the demod-upscale carrier works at arbitrary thumbnail sizes -- the
    kept negative on file is POINT-sampling the carrier (thin belts aliased to charcoal); a box+bilinear chain
    keeps the average character at every ratio. Pure NumPy, deterministic."""
    a = np.asarray(img, float)
    h, w = a.shape[:2]
    fh, fw = h // int(out_h), w // int(out_w)
    f = max(1, min(fh, fw))
    if f > 1 and h % f == 0 and w % f == 0:
        a = a.reshape(h // f, f, w // f, f, -1).mean((1, 3))
        h, w = a.shape[:2]
    if (h, w) == (int(out_h), int(out_w)):
        return a if a.ndim == 3 else a[..., None]
    ys = (np.arange(out_h) + 0.5) * h / out_h - 0.5
    xs = (np.arange(out_w) + 0.5) * w / out_w - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1); y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1); x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (ys - y0)[:, None, None]; wx = (xs - x0)[None, :, None]
    a2 = (a[y0][:, x0] * (1 - wy) * (1 - wx) + a[y0][:, x1] * (1 - wy) * wx
          + a[y1][:, x0] * wy * (1 - wx) + a[y1][:, x1] * wy * wx)
    return a2


def _upscale_wants_native(material):
    """True when the demod upscale cannot match a native render for this material -- measured scope: the method
    restores ALBEDO-borne detail (grid floor, coloured/rough surfaces arrive sharp), but TRANSPORT-borne detail
    is exactly what it keeps low-res. Two material classes carry their detail in transport, both measured:
      * TRANSMISSIVE (transmission > 0): refraction of the scene behind -- glass rendered as speckle mush.
      * SMOOTH METAL (metallic > 0.5, roughness < 0.35): sharp reflections -- draft-undersampled lobes mottle,
        and buying them back with per-pixel tol/3 sampling measured 44.4 s at size=160, MORE than the ~23 s a
        native render costs. Upsampling smooth metals cannot beat native at equal quality; route native.
    Raw dicts are checked before coercion (PBRMaterial coercion drops `transmission`, measured)."""
    def _classify(trans, met, rough):
        return float(trans or 0.0) > 0.0 or (float(met or 0.0) > 0.5 and float(rough if rough is not None else 1.0) < 0.35)
    if isinstance(material, dict):
        return _classify(material.get("transmission", 0.0), material.get("metallic", 0.0),
                         material.get("roughness", 1.0))
    m = _coerce_preview_material(material)
    if isinstance(m, str):
        from holographic.materials_and_texture.holographic_matlib import material as _lib
        try:
            m = _lib(m)
        except Exception:
            return False
    return _classify(getattr(m, "transmission", 0.0), getattr(m, "metallic", 0.0), getattr(m, "roughness", None))


def _outer_is_translucent(material):
    """True when the OUTER carries subsurface/translucency (sss > 0) and is NOT a transmissive dielectric --
    the class the emissive core's MESH LIGHT is toggled ON for (user direction). A glowing body inside wax
    should be treated as a light; inside glass/refractive/transparent shells the refraction path already
    carries the emission and the mesh light stays off. Raw dicts checked before coercion (coercion drops
    fields, the standing instrument note)."""
    def _classify(sss, trans):
        return float(sss or 0.0) > 0.0 and float(trans or 0.0) <= 0.0
    if isinstance(material, dict):
        return _classify(material.get("sss", 0.0), material.get("transmission", 0.0))
    m = _coerce_preview_material(material)
    if isinstance(m, str):
        from holographic.materials_and_texture.holographic_matlib import material as _lib
        try:
            m = _lib(m)
        except Exception:
            return False
    return _classify(getattr(m, "sss", 0.0), getattr(m, "transmission", 0.0))


def _demod_upscale_display(linear_low, material, out_res):
    """Shared tail of the out_res path: cheap high-res G-buffer (primary rays only), demodulated upscale of the
    LINEAR low-res frame (irradiance upscales smoothly; the crisp detail is re-modulated in from the high-res
    albedo -- the grid floor and material colours arrive sharp), then display transform + FXAA at full size.
    Cost measured: G-buffer 1.7 s + upscale 0.2 s at 192 -- the lighting stays low-res priced."""
    from holographic.rendering.holographic_scene_render import scene_to_render, _view_transform
    from holographic.rendering.holographic_gbuffer import primary_gbuffer
    from holographic.misc.holographic_modulate import superres_demodulated
    from holographic.rendering.holographic_postfx import fxaa as _fxaa, supersample as _ss
    scd, cam = preview_scene_document(material)
    _, matfn = scene_to_render(scd)
    _, sky = preview_scene_lighting()
    # THE TAIL, tuned by measurement against the native render (v23; the user's two complaints were speckle
    # and residual aliasing, both addressed at the mechanism):
    #   1. DENOISE THE IRRADIANCE AT LOW RES (M4, levels=2): draft MC speckle upscales into blotches;
    #      filtering the demodulated irradiance kills it while the light touch keeps reflection structure
    #      (levels=4 measured over-smoothed; metered cost of 2: ~0.07 luma off the top belt's specular streak,
    #      everything else within 0.03 of native).
    #   2. ANTI-ALIASED CARRIER: demodulate by the BOX-DOWNSAMPLED 2x albedo, never the point-sampled low
    #      G-buffer albedo -- the module's own documented negative; point-sampling aliased the thin belts.
    #   3. GUIDED (joint-bilateral) UPSAMPLE of the irradiance, guided by the 2x depth AND albedo: geometry
    #      edges where the albedo barely varies (grey fixtures against grey floor) stayed stair-stepped under
    #      plain bilinear; the depth guide separates them. Albedo guide keeps belt boundaries crisp.
    #   4. Remodulate at 2x, box-average down in LINEAR (coverage AA), display transform, final FXAA.
    from holographic.misc.holographic_modulate import denoise_demodulated as _dn, demodulate as _dm, \
        remodulate as _rm
    from holographic.rendering.holographic_superres import guided_upsample as _gu
    R = int(linear_low.shape[0]); R2 = int(out_res) * 2                  # 2x carrier: any size works (v25)
    ln, lalb, lz = primary_gbuffer(scd.preview_distance_sdf, cam, R, R, matfn, sky=sky)
    hn, halb, hz = primary_gbuffer(scd.preview_distance_sdf, cam, R2, R2, matfn, sky=sky)
    halb = np.asarray(halb)
    # METAL-AWARE denoise blend (v24, tuned against the user's 'reflection quality is diminished'): the M4
    # denoise and the JBU both smooth irradiance, and on smooth METAL that irradiance IS the reflection --
    # measured: full denoise turned copper's streaks into blobs; no denoise left draft speckle. The blend
    # keeps the RAW irradiance on smooth-metal pixels (their glossy structure masks residual noise) and
    # applies the full filter on diffuse pixels (where the blotch actually lived). Weight from the primary
    # hits' own material: w = 1 on non-metal / miss; on metal, w ramps with roughness (rough metal blurs its
    # reflections anyway, so filtering costs nothing there).
    eye = np.array(cam.eye); tgt = np.array(cam.target)
    fwdv = tgt - eye; fwdv /= np.linalg.norm(fwdv)
    rightv = np.cross(fwdv, [0.0, 1.0, 0.0]); rightv /= np.linalg.norm(rightv)
    upv = np.cross(rightv, fwdv)
    halft = np.tan(np.radians(46.0 / 2))
    ys, xs = np.mgrid[0:R, 0:R]
    uu = (xs + 0.5) / R * 2 - 1; vv = 1 - (ys + 0.5) / R * 2
    Dv = fwdv[None, None, :] + uu[..., None] * halft * rightv[None, None, :] + vv[..., None] * halft * upv[None, None, :]
    Dv /= np.linalg.norm(Dv, axis=2, keepdims=True)
    zlow = np.asarray(lz); hitm = zlow < 1e8
    Phit = (eye[None, None, :] + Dv * zlow[..., None]).reshape(-1, 3)
    mout = matfn(Phit)
    met = np.asarray(mout[1]).reshape(R, R); rough = np.asarray(mout[2]).reshape(R, R)
    den = _dn(linear_low, np.asarray(ln), np.asarray(lalb), np.asarray(lz), levels=2)
    wgt = np.where(met > 0.5, np.clip((rough - 0.05) * 1.2, 0.05, 1.0), 1.0)
    wgt = np.where(hitm, wgt, 1.0)[..., None]
    den = linear_low * (1 - wgt) + den * wgt
    carrier_low = _area_resize(halb, R, R)                               # box+bilinear: never point-sampled
    irr = _dm(den, carrier_low)
    ih = _gu(irr, np.asarray(hn), guide_albedo=halb, guide_depth=np.asarray(hz), levels=3)
    hi = _ss(_rm(ih, halb), factor=2)
    return _fxaa(_view_transform(hi, "display"))


def preview_thumbnail(material=None, res=96, quality="draft", seed=0, fmt="png",
                      core=None, trim=None, trim_top=None, trim_bottom=None, base=None, out_res=None,
                      size=None, upsample=False):
    """THE one-call material thumbnail: feed a material (matlib name, material object, or plain PBR dict), get a
    small render of it on the shader ball back. Every fixture slot (core, base, both belts) stays the neutral
    grey diffuse unless overridden, so the thumbnail is ABOUT the material, nothing else. fmt='png' (default)
    returns PNG BYTES -- encoded by the engine's own holographic_render.png_bytes, stdlib-only, and the shape
    the HTTP /invoke door speaks natively (bytes travel as {'__bytes_b64__': ...}); fmt='array' returns the raw
    (res, res, 3) float image for in-process callers. Same studio rig, framing and AA as preview_scene -- this
    is a convenience DOOR, not a second renderer; it delegates entirely.
    RECOMMENDATION, from a measured three-way on copper at 192 (native exact 64.8 s / batch native warm
    33.1 s / upscale warm 15.8 s): for QUALITY thumbnails render NATIVE at the size you want -- the batch door
    at native res is the sweet spot (native quality at ~half the exact door's price). out_res upscaling saves
    a further ~2.1x but visibly softens LIGHTING detail (reflections, shadow noise) -- the user judged native
    superior at 192; use out_res for quick low-stakes grids.
    out_res=N (> res) returns an N-px image at res-px LIGHTING cost via demodulated upscale: the smooth
    irradiance upscales cleanly and the crisp detail is re-modulated in from a cheap high-res albedo G-buffer
    (grid floor and material colours arrive sharp), with the irradiance DENOISED at low res (M4, light touch),
    guided-upsampled (joint-bilateral on the 2x depth+albedo), remodulated at 2x and box-averaged down in
    linear -- speckle and stair-step aliasing both addressed at the mechanism (~2.5 s tail at 192).
    TRANSMISSIVE outers auto-route to a native
    out_res render instead -- their detail is transport-borne (refraction) and demod upscaling mushes it
    (measured; the honest price is paid, stated here).
    COST, measured with the distance proxy: ~36 s at res=96 quality='draft', ~24 s at res=64;
    size=N asks for ANY delivery size (square frame, so aspect is fixed by construction); upsample=False
    (default) renders NATIVE at N via the exact door; upsample=True takes the FAST path -- the batch
    machinery with its static cache, where each material routes by WHERE ITS DETAIL LIVES: diffuse/rough
    materials get the demod upscale (lighting at ~2N/3), transport-detail materials (transmissive, smooth
    metal) get a masked NATIVE render at N (measured: buying metal reflections back with samples on the
    upscale path cost MORE than native). Warm at size=160: wax ~21 s (upscale), chrome ~37 s (auto-native).
    res/out_res remain for direct control (box+bilinear carrier -- no divisibility constraint)."""
    if size is not None:
        # THE FRONT DOOR SPELLING (user direction): ask for the SIZE you want (any positive pixel count --
        # the frame is square, so aspect is preserved by construction) and CHOOSE whether it is upsampled.
        # upsample=False -> a native render at `size` (the higher-quality option, honest price);
        # upsample=True  -> lighting at ~2/3 size (the measured reflections sweet spot, floored at 64),
        #                   demod-upscaled to `size`. Overrides res/out_res when given.
        if upsample:
            # the fast path IS the batch machinery (static cache + transport-detail routing + masked
            # renders); a single material is a batch of one. The exact never-composited door remains the
            # upsample=False spelling.
            if any(x is not None for x in (core, trim, trim_top, trim_bottom, base)):
                raise ValueError("size+upsample supports only the outer material; render slot overrides natively")
            return preview_thumbnail_batch([material], quality=quality, seed=seed, fmt=fmt,
                                           size=size, upsample=True)[0]
        res = int(size)
        out_res = None
    if out_res is not None and int(out_res) > int(res) and not _upscale_wants_native(material):
        # render the LIGHTING at `res` (linear, no AA -- the upscale tail owns display+AA at full size), then
        # demod-upscale to `out_res`: a big thumbnail at small-render lighting cost. Slot overrides other than
        # the outer are not supported on this path (the G-buffer carrier is rebuilt from `material` alone).
        # Transmissive outers route to a NATIVE out_res render instead (see _upscale_wants_native).
        if any(x is not None for x in (core, trim, trim_top, trim_bottom, base)):
            raise ValueError("out_res upscaling supports only the outer material; render slot overrides at native res")
        lin = preview_scene(material, res=int(res), quality=quality, seed=seed, view=None, aa="off")
        img = _demod_upscale_display(lin, material, int(out_res))
    elif out_res is not None and int(out_res) > int(res):
        img = preview_scene(material, core=core, trim=trim, trim_top=trim_top, trim_bottom=trim_bottom,
                            base=base, res=int(out_res), quality=quality, seed=seed)
    else:
        img = preview_scene(material, core=core, trim=trim, trim_top=trim_top, trim_bottom=trim_bottom,
                            base=base, res=int(res), quality=quality, seed=seed)
    if fmt == "array":
        return img
    if fmt == "png":
        from holographic.rendering.holographic_render import png_bytes
        return png_bytes(img)
    raise ValueError("fmt must be 'png' or 'array', got %r" % (fmt,))


_THUMB_CACHE = {}


def _thumbnail_static(res, quality, seed):
    """The material-INDEPENDENT half of a thumbnail, cached for the process lifetime (the user's observation
    made mechanism: the camera and geometry are FIXED, so everything that depends only on them is the same for
    every thumbnail). Cached per (res, quality, seed): the LINEAR reference frame rendered with every slot at
    the neutral grey default, and the ACTIVE MASK -- pixels whose primary hit is the outer or the core, dilated
    6 px to cover silhouette edges and the contact region. Re-used by every subsequent thumbnail at the same
    settings; the first call pays for it once."""
    key = (int(res), str(quality), int(seed))
    if key in _THUMB_CACHE:
        return _THUMB_CACHE[key]
    from holographic.rendering.holographic_scene_render import render_scene_document, scene_to_render
    from holographic.rendering.holographic_gbuffer import primary_gbuffer
    R = int(res)
    scd, cam = preview_scene_document(None)
    lights, sky = preview_scene_lighting()
    key_dir = np.array([2.4, 2.26, 1.6]); key_dir = key_dir / np.linalg.norm(key_dir)
    common = dict(quality=quality, seed=seed, view=None, lights=lights, sky=sky, soft_light_cache=False,
                  sss_dir=tuple(key_dir), sss_depth=0.30, sss_sigma=20.0, sss_interior=True)
    ref = render_scene_document(scd, cam, R, R, distance_sdf=scd.preview_distance_sdf, **common)
    # ownership from primary hits: reconstruct P = eye + D * depth on the camera basis, argmin over object SDFs
    sdfP = scd.preview_distance_sdf
    _, matfn = scene_to_render(scd)
    _n, _a, depth = primary_gbuffer(sdfP, cam, R, R, matfn, sky=sky)
    eye = np.array(cam.eye); tgt = np.array(cam.target)
    fwd = tgt - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    half = np.tan(np.radians(46.0 / 2))
    ys, xs = np.mgrid[0:R, 0:R]
    u = (xs + 0.5) / R * 2 - 1; v = 1 - (ys + 0.5) / R * 2
    D = fwd[None, None, :] + u[..., None] * half * right[None, None, :] + v[..., None] * half * up[None, None, :]
    D /= np.linalg.norm(D, axis=2, keepdims=True)
    hit = np.asarray(depth) < 1e8
    P = (eye[None, None, :] + D * np.asarray(depth)[..., None]).reshape(-1, 3)
    objs = {o.name: o.geometry for o in scd.objects.values()}
    names = ("outer", "core", "trim_top", "trim_bottom", "base", "floor")
    dist = np.stack([np.abs(objs[n](P)) for n in names], 1)
    owner = np.argmin(dist, 1).reshape(R, R)
    ball = ((owner == 0) | (owner == 1)) & hit
    mask = ball.copy()
    for _ in range(6):                                          # dilation: silhouette AA edges + contact region
        mask = mask | np.roll(mask, 1, 0) | np.roll(mask, -1, 0) | np.roll(mask, 1, 1) | np.roll(mask, -1, 1)
    outer_px = (owner == 0) & hit                               # the OUTER's own pixels, undilated -- tol_scale site
    _THUMB_CACHE[key] = (ref, mask, outer_px)
    return _THUMB_CACHE[key]


def preview_thumbnail_batch(materials, res=96, quality="draft", seed=0, fmt="png", out_res=None,
                            size=None, upsample=False):
    """MANY thumbnails, fast: exploit the FIXED camera and geometry (the user's design) -- render the neutral
    reference ONCE per (res, quality, seed) and cache it for the process lifetime; then each material re-renders
    ONLY the pixels that can see the ball (the cached ACTIVE MASK, ~48% of the frame -- and the expensive half),
    composites the untouched fixtures from the reference in LINEAR light, applies the display transform once,
    and AA. `materials` is a list of matlib names / material objects / PBR dicts; returns a list of PNG bytes
    (fmt='png') or float arrays (fmt='array'), aligned with the input.
    MEASURED at res=96 draft: full thumbnail 36 s; masked per-material 26 s (1.37x -- the masked-off pixels were
    the cheap ones); reference amortised to zero from the second call on. QUALITY, gated by the honest standard:
    composite-vs-full diff mean 0.0134 / p99 0.134, both BELOW the renderer's own seed-to-seed noise floor
    (0.0167 / 0.176); the max is draft-sampler speckle, present between any two draft runs.
    out_res=N adds the denoised guided upscale per material (~2.5 s tail at 192; transmissive outers
    auto-route native). QUALITY ladder at 192, measured: batch-native 33 s (every specular streak) > res=128 upscale 24 s
    (structured reflections) > res=96 upscale 18 s (speed; coarser metal reflections).
    KEPT SCOPE: the copied fixture pixels carry the REFERENCE's indirect light -- a strongly coloured outer's
    bounce tint on far floor pixels is approximated by the grey reference's (measured within noise at draft;
    for a final-quality single frame use preview_thumbnail, which never composites)."""
    from holographic.rendering.holographic_scene_render import render_scene_document, _view_transform
    from holographic.rendering.holographic_postfx import fxaa as _fxaa
    if size is not None:
        # same front-door spelling as preview_thumbnail: size + optional upsampling (see there)
        _r = max(64, int(round(int(size) * 2 / 3)))
        if upsample and _r < int(size):
            res = _r
            out_res = int(size)
        else:
            res = int(size)
            out_res = None
    R = int(res)
    ref, mask, outer_px = _thumbnail_static(R, quality, seed)
    lights, sky = preview_scene_lighting()
    key_dir = np.array([2.4, 2.26, 1.6]); key_dir = key_dir / np.linalg.norm(key_dir)
    common = dict(quality=quality, seed=seed, view=None, lights=lights, sky=sky, soft_light_cache=False,
                  sss_dir=tuple(key_dir), sss_depth=0.30, sss_sigma=20.0, sss_interior=True)
    out = []
    for mat in materials:
        scd, cam = preview_scene_document(mat)
        # (v26: smooth metals now route NATIVE via _upscale_wants_native -- the per-pixel tol/3 experiment
        # measured 44.4 s vs ~23 s native at size=160, a kept negative; the tol_scale machinery itself stays
        # in the tracer as a general capability.)
        lin = render_scene_document(scd, cam, R, R, active=mask,
                                    distance_sdf=scd.preview_distance_sdf, **common)
        comp = ref.copy(); comp[mask] = lin[mask]
        if out_res is not None and int(out_res) > R and not _upscale_wants_native(mat):
            img = _demod_upscale_display(comp, mat, int(out_res))
        elif out_res is not None and int(out_res) > R:
            # transport-detail material (glass, smooth metal): render NATIVE at out_res -- but through THIS
            # door's own masked machinery at that size, not the exact door (v26 fix: the fallback was paying
            # the full 54 s exact price while a 160-class masked render costs ~23 s warm; the static cache at
            # out_res is built once and amortises exactly like the low-res one).
            refN, maskN, _oN = _thumbnail_static(int(out_res), quality, seed)
            linN = render_scene_document(scd, cam, int(out_res), int(out_res), active=maskN,
                                         distance_sdf=scd.preview_distance_sdf, **common)
            compN = refN.copy(); compN[maskN] = linN[maskN]
            img = _fxaa(_view_transform(compN, "display"))
        else:
            img = _fxaa(_view_transform(comp, "display"))
        if fmt == "array":
            out.append(img)
        elif fmt == "png":
            from holographic.rendering.holographic_render import png_bytes
            out.append(png_bytes(img))
        else:
            raise ValueError("fmt must be 'png' or 'array', got %r" % (fmt,))
    return out


def _selftest():
    from holographic.materials_and_texture.holographic_texturegraph import Map, Const, field_leaf
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
    from holographic.materials_and_texture.holographic_material import Material, texture_field

    # texture swatch: a colour graph -> a viewable rgb image in [0,1]
    g = Map("mix", a=Const("red"), b=Const("blue"), t=field_leaf("fbm", n_dims=2, seed=0))
    swatch = texture_image(g, res=64)
    assert swatch.shape == (64, 64, 3)
    assert swatch.min() >= 0.0 and swatch.max() <= 1.0
    # a scalar graph -> greyscale (all three channels equal)
    grey = texture_image(Map("scale", x=field_leaf("fbm", n_dims=2, seed=1), k=Const(1.0)), res=32)
    assert np.allclose(grey[:, :, 0], grey[:, :, 1]) and np.allclose(grey[:, :, 1], grey[:, :, 2])

    # material ball: a material with a roughness pattern -> a shaded sphere image
    enc = VectorFunctionEncoder(2, dim=512, bounds=[(0, 1), (0, 1)], kernel="rbf", bandwidth=3.0, seed=1)
    grid = [(a, b) for a in np.linspace(0.05, 0.95, 6) for b in np.linspace(0.05, 0.95, 6)]
    mat = Material(enc, {"roughness": texture_field(enc, grid, [a for (a, b) in grid]),
                         "metallic": texture_field(enc, grid, [0.0 for _ in grid])})
    ball = material_ball(mat, res=96)
    assert ball.shape == (96, 96, 3) and ball.min() >= 0.0 and ball.max() <= 1.0
    # the ball's centre (on the sphere) differs from a background corner (the sphere was actually shaded)
    assert not np.allclose(ball[48, 48], ball[0, 0])

    # a CMP2/CMP3 material also previews (has .sample(channel, uv) + channel_names)
    from holographic.materials_and_texture.holographic_layeredmaterial import Layer, LayeredMaterial
    stack = LayeredMaterial([Layer("base", mat), Layer("coat", mat, alpha=0.3)])
    ball2 = material_ball(stack, res=64)
    assert ball2.shape == (64, 64, 3)

    # --- the shader-ball preview SCENE. The DOCUMENT contract is asserted exactly (cheap); tiny renders prove
    # the pixels path.
    scd, cam = preview_scene_document("gold")
    by_name = {o.name: o for o in scd.objects.values()}
    assert set(by_name) == {"floor", "outer", "core", "trim_top", "trim_bottom", "base"}, sorted(by_name)
    # slot rule (revised): material dresses the OUTER only; other slots carry their own defaults
    assert by_name["outer"].material == "gold", "material= must dress the outer"
    assert getattr(by_name["core"].material, "name", "") == "mouse_ball_gray", "core default must be mouse-ball grey"
    assert getattr(by_name["trim_top"].material, "name", "") == "mouse_ball_gray", "top belt default must be the grey diffuse"
    assert getattr(by_name["trim_bottom"].material, "name", "") == "mouse_ball_gray", "bottom belt default must be the grey diffuse"
    assert getattr(by_name["base"].material, "name", "") == "mouse_ball_gray", "base default must be mouse-ball grey"
    # FLUSH INLAY BELT pins. The construction is a PARTITION of the shell, so the pins assert the partition:
    # (flush) the belt owns NO material beyond the ball radius -- a probe just outside r=0.60 at belt height is
    # outside the belt (an outward bump fails here); (body) mid-wall at belt height is inside the belt AND
    # outside the OUTER (the slab really was subtracted -- both own it and the seam is broken); (level) the
    # same mid-wall ring past the belt height flips owners (outer inside, belt outside); (clearance) the belt
    # does not reach the lens dish or the window band -- room around the features, as directed.
    _cc = np.array([0.0, 0.74, 0.0])
    _bb = by_name["trim_bottom"].geometry
    _bt = by_name["trim_top"].geometry
    _ou = by_name["outer"].geometry
    def _belt_pt(y, dist):
        rxy = np.sqrt(max(dist * dist - (y - 0.74) ** 2, 1e-9))
        return np.array([[rxy, y, 0.0]])
    for _g, _y0, _tag in ((_bb, 0.46, "bottom"), (_bt, 1.17, "top")):
        assert _g(_belt_pt(_y0, 0.605))[0] > 0.0, "%s belt bulges past the ball surface -- flush means flush" % _tag
        assert _g(_belt_pt(_y0, 0.56))[0] < 0.0, "%s belt lost its body (mid-wall probe outside)" % _tag
        assert _ou(_belt_pt(_y0, 0.56))[0] > 0.0, "outer still owns the %s belt region -- the partition broke" % _tag
    assert _bb(_belt_pt(0.56, 0.56))[0] > 0.0 and _ou(_belt_pt(0.56, 0.56))[0] < 0.0, \
        "the bottom belt leaked past its height (must be LEVEL; the outer owns the wall there)"
    # clearance: a point on the hole's lower rim region (y 0.55, window azimuth) and on the lens axis belong
    # to NEITHER belt -- 'the dip should be clear of the bands, with some room around the hole'
    _eyeB = np.array([1.18, 1.12, 1.82]); _dB = (_eyeB - _cc) / np.linalg.norm(_eyeB - _cc)
    _luB = np.array([[np.cos(-0.45), 0, np.sin(-0.45)], [0, 1, 0], [-np.sin(-0.45), 0, np.cos(-0.45)]]) @ _dB
    _luB = _luB + np.array([0.0, 0.22, 0.0]); _luB = _luB / np.linalg.norm(_luB)
    _dishP = np.array([_cc + 0.56 * _luB])
    assert _bt(_dishP)[0] > 0.0 and _bb(_dishP)[0] > 0.0, "a belt reached the lens dish -- the dip must stay clear"
    # explicit multi-material display: each slot carries ITS OWN material -- including per-belt overrides
    scd4, _ = preview_scene_document("copper", core="gold", trim_top="glass_clear", trim_bottom="chrome",
                                     base="matte_black")
    m4 = {o.name: o.material for o in scd4.objects.values()}
    assert (m4["outer"], m4["core"], m4["trim_top"], m4["trim_bottom"], m4["base"], m4["floor"]) \
        == ("copper", "gold", "glass_clear", "chrome", "matte_black", "matte_white")
    # trim= still dresses BOTH belts (backward-compatible spelling)
    scd5, _ = preview_scene_document("copper", trim="matte_black")
    m5 = {o.name: o.material for o in scd5.objects.values()}
    assert m5["trim_top"] == m5["trim_bottom"] == "matte_black", "trim= must dress both belts"
    # material=None -> the neutral default diffuse on the OUTER (a blank preview must still show something)
    scd0, _ = preview_scene_document()
    assert [o.material for o in scd0.objects.values() if o.name == "outer"] == [_PREVIEW_DEFAULT_MATERIAL]
    # a plain PBR dict is coerced to one real PBRMaterial on the OUTER slot
    scdd, _ = preview_scene_document({"base_color": (1.0, 0.2, 0.1), "roughness": 0.2, "metallic": 1.0})
    dm = [o.material for o in scdd.objects.values() if o.name == "outer"][0]
    assert float(dm.metallic) == 1.0 and dm.base_color[0] == 1.0
    # FLUSHNESS pinned geometrically, on the SDFs themselves: the core must reach past the shell's inner
    # surface (interpenetrate, no air gap) and stop inside the outer surface. Probed on the actual scene
    # geometry, not on remembered constants -- if someone shrinks the core back to a floating ball, this fires.
    core_geo = [o.geometry for o in scd.objects.values() if o.name == "core"][0]
    shell_geo = [o.geometry for o in scd.objects.values() if o.name == "outer"][0]
    _c = np.array([0.0, 0.74, 0.0])
    assert core_geo(np.array([_c + [0.505, 0, 0]]))[0] < 0.0, "core shrank back to a floating ball (visible gap)"
    # BOTH SDFs positive in the hair gap: a real surface pair exists there. This is the transmission contract --
    # if the core ever interpenetrates the shell again, the union swallows the core surface and a glass outer
    # refracts through to NOTHING (measured: the blue core vanished). Flush means hair-gap, never overlap.
    mid = np.array([_c + [0.515, 0, 0]])
    assert core_geo(mid)[0] > 0.0 and shell_geo(mid)[0] > 0.0,         "no gap between core and shell -- the core surface is buried and glass outers lose their refracted core"
    # THE ZERO-NORMAL REGRESSION TRAP: at the glass exit point (just inside the gap off the shell inner wall)
    # the union's finite-difference normal must be a real unit-ish vector. With a sub-resolution gap it came
    # back as the ZERO VECTOR (measured) and every refracted ray scattered -- the exact bug that hid the core.
    from holographic.rendering.holographic_scene_render import scene_to_render as _s2r
    from holographic.rendering.holographic_raymarch import sdf_normal as _nrm
    _sdf, _ = _s2r(scd)
    _n = _nrm(_sdf, np.array([_c + [0.5185, 0, 0]]))
    assert float(np.linalg.norm(_n[0])) > 0.9,         "zero/degenerate normal at the glass exit point -- the gap is below the tracer's FD resolution again"
    # THE THIN LENS pinned geometrically, recomputing the lens axis exactly as the builder does: the dish must
    # be carved (shell positive at r=0.55 along the axis) while a THIN wall survives beneath it (shell negative
    # at r=0.526) -- fail either way and the translucency test region is gone or the shell has a second hole.
    _eye = np.array([1.18, 1.12, 1.82]); _d = (_eye - _c) / np.linalg.norm(_eye - _c)
    _lu = np.array([[np.cos(-0.45), 0, np.sin(-0.45)], [0, 1, 0], [-np.sin(-0.45), 0, np.cos(-0.45)]]) @ _d
    _lu = _lu + np.array([0.0, 0.22, 0.0]); _lu = _lu / np.linalg.norm(_lu)
    assert shell_geo(np.array([_c + 0.55 * _lu]))[0] > 0.0, "the thin-lens dish is not carved"
    assert shell_geo(np.array([_c + 0.526 * _lu]))[0] < 0.0, "the lens went through the wall -- it must thin, not open"
    # pixels: tiny but real -- bounded display image whose object centre differs from the sky corner
    img = preview_scene("gold", res=40, quality="fast", seed=0)
    assert img.shape == (40, 40, 3) and float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert not np.allclose(img[20, 20], img[0, 0]), "the scene render shaded nothing"
    # THE CORE INTERACTS: an emissive core inside a glass shell must brighten the OBJECT vs a dark core --
    # the whole reason the core slot exists. Same geometry, same seed; only the core material differs.
    # INSTRUMENT NOTES (both earlier meters failed while the render was correct): (1) a full-frame mean is
    # mostly sky and floor, which do not care about the core -- measure the central crop, where the object is.
    # (2) under the studio rig the grey ambient dilutes the LUMINANCE margin below a robust gate; the core is
    # BLUE and the rig is grey, so the blue channel is the discriminating meter. Numbers on record at res=36
    # 'fast', studio rig: crop blue 0.598 lit vs 0.515 dark (+0.083); crop luminance +0.047.
    lit = preview_scene("glass_clear", core="neon_blue", res=36, quality="fast", seed=0)
    dark = preview_scene("glass_clear", core="matte_black", res=36, quality="fast", seed=0)
    c = slice(10, 26)
    assert float(lit[c, c, 2].mean()) > float(dark[c, c, 2].mean()) + 0.05, \
        "an emissive core did not glow through the glass shell (crop blue %.4f vs %.4f)" \
        % (lit[c, c, 2].mean(), dark[c, c, 2].mean())
    # the lighting flag is a real fork, and a typo must say so legibly rather than render something unasked-for
    plain = preview_scene("gold", res=24, quality="fast", seed=0, lighting="plain")
    studio = preview_scene("gold", res=24, quality="fast", seed=0, lighting="studio")
    assert plain.shape == studio.shape == (24, 24, 3)
    assert not np.allclose(plain, studio), "the studio rig changed nothing over 'plain' -- it is not lighting"
    try:
        preview_scene("gold", res=8, lighting="dramatic")
        raise AssertionError("an unknown lighting name must raise")
    except ValueError as exc:
        assert "studio" in str(exc)
    # the graph-paper floor: the socket rides the floor object by default, and OFF means off
    sg, _ = preview_scene_document("gold")
    fl = [o for o in sg.objects.values() if o.name == "floor"][0]
    assert fl.overrides.get("albedo_socket") is preview_grid_albedo, "the default floor lost its grid socket"
    sp, _ = preview_scene_document("gold", floor_grid=False)
    fl2 = [o for o in sp.objects.values() if o.name == "floor"][0]
    assert not (fl2.overrides or {}).get("albedo_socket"), "floor_grid=False must mean a plain floor"
    # the socket contract itself: (M,3) in -> (M,3) rgb in [0,1], darker ON a major line than mid-cell.
    # NOTE the line PHASE: lines sit at cell MIDPOINTS (the -0.5 centring), so (0.75, 0.75) is on the major
    # crossing and the ORIGIN is mid-cell -- the first draft of this assert had the two points backwards.
    Q = np.array([[0.75, 0.0, 0.75], [0.0, 0.0, 0.0]])          # on a major line crossing / mid-cell
    G = preview_grid_albedo(Q)
    assert G.shape == (2, 3) and G.min() >= 0.0 and G.max() <= 1.0
    assert G[0].mean() < G[1].mean() - 0.2, "grid lines must be clearly darker than the ground"
    # the ceiling panels: steep-up rays can exceed the gradient (HDR panels), near-horizontal rays cannot --
    # panels in the direct background view were the measured failure this threshold exists to prevent.
    # SAME PHASE LESSON as the grid, one function later: panels are centred at cell MIDPOINTS, so straight-up
    # (ceiling coords 0,0) is a GAP; (0.4, 0.8, 0.44) projects to (0.5, 0.55) -- mid-panel, measured 2.93.
    _, sky = preview_scene_lighting()
    steep = sky(np.array([[0.4, 0.8, 0.44], [0.0, 1.0, 0.0]]))
    horiz = sky(np.array([[0.995, 0.02, 0.1], [0.0, 0.02, 1.0]]))
    assert float(steep.max()) > 1.2, "no HDR ceiling panel found in the steep-up directions"
    assert float(horiz.max()) <= 1.0, "the background (near-horizontal) sky must stay panel-free"
    # and the grid must actually reach the pixels: same scene, grid on vs off, floors differ
    gon = preview_scene("gold", res=24, quality="fast", seed=0)
    goff = preview_scene("gold", res=24, quality="fast", seed=0, floor_grid=False)
    assert not np.allclose(gon, goff), "floor_grid changed nothing -- the socket is not reaching the renderer"
    # the AA fork: 'fxaa' (default) differs from 'off' (it did something), keeps shape/range, and equals
    # applying postfx.fxaa to the raw frame BY CONSTRUCTION -- pinned so the default cannot silently drift
    # from the postfx implementation it claims to be. 'ssaa2' honours the asked-for size. Typos raise legibly.
    from holographic.rendering.holographic_postfx import fxaa as _fx
    raw = preview_scene("gold", res=24, quality="fast", seed=0, aa="off")
    dflt = preview_scene("gold", res=24, quality="fast", seed=0)
    assert dflt.shape == raw.shape == (24, 24, 3) and float(dflt.min()) >= 0.0 and float(dflt.max()) <= 1.0
    assert not np.array_equal(dflt, raw), "aa='fxaa' changed nothing over 'off' -- it is not anti-aliasing"
    assert np.array_equal(dflt, _fx(raw)), "the default AA must BE postfx.fxaa on the raw frame, exactly"
    ss = preview_scene("gold", res=16, quality="fast", seed=0, aa="ssaa2")
    assert ss.shape == (16, 16, 3), "ssaa2 must return the size it was ASKED for, not the 2x internal frame"
    try:
        preview_scene("gold", res=8, aa="msaa")
        raise AssertionError("an unknown aa name must raise")
    except ValueError as exc:
        assert "fxaa" in str(exc)
    # INTERIOR-EMISSION TRANSLUCENCY gate, measured before pinning (0.0133 at this rig; gate at half): a wax
    # outer with an emissive core must glow through the BODY -- the left half of the object crop, away from
    # the open window -- versus a dark core. This is the end-to-end pin on the sss_interior wiring; when the
    # fixed-step version of the term silently skipped the 0.010 gap, this number was 0.0012 (noise).
    wl = preview_scene("wax", core="neon_blue", trim="matte_black", base="matte_black",
                       res=36, quality="draft", seed=0)
    wd = preview_scene("wax", core="matte_black", trim="matte_black", base="matte_black",
                       res=36, quality="draft", seed=0)
    _gl = float(wl[8:26, 5:18, 2].mean() - wd[8:26, 5:18, 2].mean())
    assert _gl > 0.0065, "emissive core no longer glows through the wax body (left-half blue delta %.4f)" % _gl
    # THE THUMBNAIL DOOR: fmt='png' returns real PNG bytes (magic header, decodable size), fmt='array' returns
    # the raw image, both delegate to preview_scene (same seed + settings -> the array IS the png's source),
    # and a typo'd fmt raises naming the options. This is the one-call contract leOS feeds materials through.
    tn = preview_thumbnail("gold", res=20, quality="draft", seed=0, fmt="png")
    assert isinstance(tn, bytes) and tn[:8] == b"\x89PNG\r\n\x1a\n", "fmt='png' must return real PNG bytes"
    ta = preview_thumbnail("gold", res=20, quality="draft", seed=0, fmt="array")
    assert ta.shape == (20, 20, 3) and float(ta.min()) >= 0.0 and float(ta.max()) <= 1.0
    from holographic.rendering.holographic_render import png_bytes as _pngb
    assert tn == _pngb(ta), "the png and the array must be the SAME render, not two renders"
    try:
        preview_thumbnail("gold", res=8, fmt="jpeg")
        raise AssertionError("an unknown fmt must raise")
    except ValueError as exc:
        assert "png" in str(exc)
    # THE BATCH DOOR: (a) PNG list aligned with the input; (b) deterministic on cache reuse (same call twice ->
    # byte-equal, the static cache really is static); (c) composite quality within the draft sampler's noise of
    # the never-composited door (measured at build: mean 0.0158 / p99 0.083 at 24px; gates at 2x those).
    bt = preview_thumbnail_batch(["copper", "gold"], res=24, quality="draft", seed=0)
    assert len(bt) == 2 and all(isinstance(x, bytes) and x[:8] == b"\x89PNG\r\n\x1a\n" for x in bt)
    ba = preview_thumbnail_batch(["copper"], res=24, quality="draft", seed=0, fmt="array")[0]
    ba2 = preview_thumbnail_batch(["copper"], res=24, quality="draft", seed=0, fmt="array")[0]
    assert np.array_equal(ba, ba2), "the batch door must be deterministic once the static cache is warm"
    bf = preview_thumbnail("copper", res=24, quality="draft", seed=0, fmt="array")
    _bd = np.abs(ba - bf)
    assert float(_bd.mean()) < 0.032 and float(np.percentile(_bd, 99)) < 0.17, \
        "batch composite drifted from the full render beyond draft noise (mean %.4f p99 %.4f)" % \
        (_bd.mean(), np.percentile(_bd, 99))
    # THE UPSCALE path: out_res returns the ASKED size at low lighting cost, bounded [0,1]; a transmissive
    # outer must route to a NATIVE render (the demod mush is the measured negative this predicate encodes).
    up = preview_thumbnail("gold", res=16, out_res=32, quality="draft", seed=0, fmt="array")
    assert up.shape == (32, 32, 3) and float(up.min()) >= 0.0 and float(up.max()) <= 1.0
    assert _upscale_wants_native("glass_clear") and _upscale_wants_native("copper"), \
        "the transport-detail predicate broke -- glass mushes and smooth metal mottles under demod upscale"
    assert not _upscale_wants_native("wax") and not _upscale_wants_native("matte_gray"), \
        "diffuse/rough materials are the upscale's home turf and must NOT route native"
    # MESH-LIGHT TOGGLE BY OUTER CLASS (user direction): translucent/SSS outers turn the emissive core's mesh
    # light ON; glass/refractive/transparent and opaque outers keep it OFF. Measured on the wax preview at
    # wiring: x1.21 cost for sub-noise-floor diff (NEE occlusion through the wall is binary) -- the toggle is
    # the CONTRACT; transmittance-aware shadow rays are the named follow-up that would make it pay.
    assert _outer_is_translucent("wax") and _outer_is_translucent("jade") and _outer_is_translucent("skin_light")
    assert not _outer_is_translucent("glass_clear") and not _outer_is_translucent("gold") \
        and not _outer_is_translucent("matte_gray")
    assert _outer_is_translucent({"base_color": (1, 1, 1), "sss": 0.5}), "raw-dict sss must be seen pre-coercion"
    assert not _outer_is_translucent({"base_color": (1, 1, 1), "sss": 0.5, "transmission": 1.0}), \
        "transmission must veto the toggle even when sss is set"
    # ANY-SIZE + OPTIONAL UPSAMPLING (v25): size= asks for an arbitrary delivery size; upsample chooses the
    # path. The awkward size (34: lighting floors at 64 ABOVE the target -- the sugar must then go native,
    # and 2*34=68 vs res 64 exercises the bilinear-residual carrier when upscaled sizes are odd-ratio).
    s50 = preview_thumbnail("gold", size=50, quality="draft", seed=0, fmt="array")
    assert s50.shape == (50, 50, 3), "size= must deliver exactly the asked native size"
    s77 = preview_thumbnail("wax", size=77, upsample=True, quality="draft", seed=0, fmt="array")
    assert s77.shape == (77, 77, 3) and float(s77.min()) >= 0.0 and float(s77.max()) <= 1.0, \
        "upsample=True must deliver the asked size at any ratio (box+bilinear carrier)"
    g40 = preview_thumbnail("glass_clear", size=40, upsample=True, quality="draft", seed=0, fmt="array")
    assert g40.shape == (40, 40, 3), "a transmissive outer with upsample=True must still route native"
    s45 = preview_thumbnail("gold", size=45, upsample=True, quality="draft", seed=0, fmt="array")
    assert s45.shape == (45, 45, 3), \
        "size below the lighting floor must fall back to a NATIVE render at the ASKED size (came back %s)" \
        % (s45.shape,)
    # THE DISTANCE PROXY contract: the proxy must be CONSERVATIVE (never exceed the plain per-object union
    # anywhere -- an overestimate would let the marcher overshoot geometry) and EXACT away from the partition
    # seams (identical beyond the near-surface band, so the speedup cannot smuggle in a different scene).
    # Measured at build: proxy < plain only within |d| < 0.05 of the surface (max 0.034, seam bands), byte-
    # equal elsewhere; render A/B 2.83x faster with image diff BELOW the seed-to-seed noise floor.
    from holographic.rendering.holographic_scene_render import scene_to_render as _s2rP
    _plainU, _ = _s2rP(scd)
    _prx = scd.preview_distance_sdf
    _rngP = np.random.default_rng(7)
    _PP = _rngP.normal(scale=0.9, size=(3000, 3)) + np.array([0.0, 0.7, 0.0])
    _dp = _plainU.eval(_PP); _dx = _prx.eval(_PP)
    assert (_dx <= _dp + 1e-9).all(), "the distance proxy OVERestimates somewhere -- marching can overshoot"
    _far = np.abs(_dp) >= 0.05
    assert float(np.abs(_dx - _dp)[_far].max()) < 1e-9, \
        "the proxy differs from the union away from the seams -- it is rendering a different scene"
    # FRAMING pin (user direction, measured at build: gap 1 row, 5 sky rows of headroom at 48px): the BASE
    # must reach the bottom frame edge (gap <= 2 rows -- 'base at the bottom of the image without a gap') but
    # not be sliced (its lowest row inside the frame), and headroom must survive for displaced materials
    # (>= 3 sky rows above the object). A black base makes the base-vs-floor boundary measurable.
    fr = preview_scene("matte_gray", base="matte_black", res=48, quality="draft", seed=0)
    _L = fr.mean(2)
    _dk = np.where((_L < 0.25).sum(1) > 2)[0]
    assert 47 - _dk.max() <= 2, "the base drifted off the bottom edge (gap %d rows)" % (47 - _dk.max())
    _skyish = (np.abs(_L - _L[0, 0]) < 0.06).all(1)
    assert int(np.argmax(~_skyish)) >= 3, "no headroom left above the object for displaced materials"

    print("OK: holographic_preview self-test passed (texture swatch is a %s rgb image in [0,1]; scalar graph -> "
          "greyscale; material ball shades a sphere with the real BRDF (centre != background); a CMP2 layered "
          "material previews too)" % (swatch.shape,))


if __name__ == "__main__":
    _selftest()
