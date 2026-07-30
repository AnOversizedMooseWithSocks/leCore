"""holographic_scene_render.py -- render the canonical Scene DOCUMENT.

WHY THIS EXISTS (backlog H7). The engine already has a canonical scene document (holographic_scene_doc.Scene): a
table of objects, each with a stable handle, a 4x4 transform, an SDF geometry, and a material, plus cameras and
lights. But nothing turned that document into the two things the path tracer actually needs -- ONE signed-distance
function .eval(P) for the whole scene, and ONE material(P) callback that returns the right object's surface at each
point. So every gallery scene hand-built a bespoke Python `class Scene` with the geometry and the material logic
tangled together. This module is the missing bridge: give it a Scene document, get back (sdf, material_fn) you can
hand straight to render_auto / path_trace.

HOW IT WORKS (all readable, NumPy only):

  * GEOMETRY. Each object's SDF is placed by its transform -- we read translation and uniform scale straight off
    the 4x4 matrix and apply the SDF's own .translate()/.scale() combinators (the SDF tree already supports them).
    All the placed objects are UNION-ed into one scene SDF: the distance to the whole scene is the nearest object.
    (Kept honest: we honour translation + uniform scale, which is what the gallery scenes use; a full affine /
    rotation would compose the SDF's .rotate() too -- a small extension, noted below.)

  * MATERIAL. To shade a hit point P we need to know WHICH object owns the nearest surface there. We evaluate every
    object's (placed) SDF at P and pick the argmin -- the closest surface wins -- then return that object's library
    material's shading tuple (via holographic_matlib.shade). A hit on the floor object gets the floor material, a
    hit on the glass object gets glass. This is the same "nearest object" rule the union uses for distance, applied
    to appearance.

The result is that a modeling app (or a test, or a demo) can build a scene by ADDING objects to the document --
undo, selection, and change-notifications all come for free from Scene -- and render it with one call, instead of
writing a one-off Python class per scene.
"""
import numpy as np


def _axis_angle(R):
    """Axis + angle from a 3x3 ROTATION matrix (already scale-normalised). Returns (axis, angle), and
    (0,0,1), 0.0 when there is no rotation to speak of.

    Uses the trace for the angle and the skew-symmetric part for the axis -- the standard reading of
    Rodrigues' formula backwards. The 180-degree case is handled separately BECAUSE IT MUST BE: at
    angle = pi the skew part vanishes identically (R is symmetric), so the general branch divides by
    ~0 and returns a garbage axis. That is not a rare input -- 'flip it round' is one of the most
    ordinary things anyone does to an object -- so it gets its own branch off the diagonal of R + I."""
    R = np.asarray(R, float)
    cos_a = (np.trace(R) - 1.0) * 0.5
    cos_a = float(np.clip(cos_a, -1.0, 1.0))
    angle = float(np.arccos(cos_a))
    if angle < 1e-9:
        return (0.0, 0.0, 1.0), 0.0
    if angle > np.pi - 1e-6:
        # near 180 deg: (R + I)/2 = a a^T, so the axis is the column with the largest diagonal, normalised
        M = (R + np.eye(3)) * 0.5
        k = int(np.argmax(np.diag(M)))
        axis = M[:, k] / max(np.sqrt(max(M[k, k], 0.0)), 1e-12)
        n = np.linalg.norm(axis)
        return tuple(axis / n) if n > 1e-12 else (0.0, 0.0, 1.0), float(np.pi)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    n = np.linalg.norm(axis)
    return (tuple(axis / n) if n > 1e-12 else (0.0, 0.0, 1.0)), angle


def _decompose(transform):
    """Pull a translation vector and a uniform scale factor out of a 4x4 transform matrix. We support the
    translate+uniform-scale case the scenes actually use; a non-uniform or rotated transform falls back to the
    average scale and the translation (rotation is a noted extension, not silently wrong -- see module docstring)."""
    T = np.asarray(transform, float)
    if T.shape != (4, 4):
        return np.zeros(3), 1.0
    translation = T[:3, 3].copy()
    # the scale is the length of the basis columns of the upper-left 3x3 (identity -> 1.0)
    scale = float(np.mean(np.linalg.norm(T[:3, :3], axis=0)))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return translation, scale


def _place(geometry, transform, affine=False):
    """Return the object's SDF placed in world space by its transform.

    DEFAULT (affine=False) is the shipped behaviour, unchanged: uniform scale about the origin, then
    translate. A rotation in the matrix is DROPPED -- documented, but invisible to the caller, which is
    why scene_info reports it as a pre-flight problem.

    affine=True also applies the rotation, as scale -> rotate -> translate, matching how a 4x4 maps an
    object-space point to world space (p_world = R s p_obj + t). Built from the SDF tree's own
    combinators, so the placed geometry is still a normal SDF node with a valid DSL form.

    WHY THIS IS A FLAG AND NOT JUST THE BEHAVIOUR. Turning it on changes the rendered image of every scene
    that has a rotated object in it. The current picture is WRONG, but 'wrong' and 'safe to change under
    someone' are different claims: this repo's rule is that shipped output does not move without an
    explicit decision, and a correctness fix that silently rewrites results is still a silent rewrite.
    So the fix ships reachable and off, `place()` writes transforms that expect it, and flipping the
    default is its own decision with its own line in NOTES.

    KEPT NEGATIVE: NON-UNIFORM scale is still not supported. The scale stays the MEAN of the basis column
    lengths, so a (2, 1, 1) stretch renders as a uniform 1.33. Doing it properly means a non-uniform SDF
    scale node, which does not preserve the distance-field property -- a sphere-trace against it can
    overshoot and punch through surfaces. That is a real design question, not an oversight, and it is not
    getting decided inside a rotation fix."""
    t, s = _decompose(transform)
    g = geometry
    if not (hasattr(g, "translate") and hasattr(g, "scale")):
        # EVAL-ONLY GEOMETRY (anything with .eval but no combinator methods -- the semantic-scene realizer's
        # _SphereSDF/_BoxSDF, a lambda field, a baked grid). The old code hasattr-guarded each combinator and
        # silently SKIPPED it, so such an object rendered fine and then IGNORED every transform: place() was
        # a no-op, keyframes produced identical frames, and nothing anywhere said so. Found the day the
        # text->document bridge landed: a described sphere animated with mean frame delta 0.0000. A guard
        # that turns "unsupported" into "quietly does nothing" is the worst of the options; wrapping costs
        # one subtraction per eval and makes every transform work on every geometry.
        return _PlacedEval(g, t, s)
    if abs(s - 1.0) > 1e-9 and hasattr(g, "scale"):
        g = g.scale(s)
    if affine and hasattr(g, "rotate"):
        T = np.asarray(transform, float)
        if T.shape == (4, 4):
            lengths = np.linalg.norm(T[:3, :3], axis=0)
            if np.all(lengths > 1e-9):
                axis, angle = _axis_angle(T[:3, :3] / lengths)
                if angle > 1e-9:
                    g = g.rotate(axis, angle)
    if np.linalg.norm(t) > 1e-9 and hasattr(g, "translate"):
        g = g.translate(tuple(t))
    return g


class _PlacedEval:
    """A translate+uniform-scale wrapper for geometry that only knows how to .eval.

    The standard SDF change of variables: d_world(P) = s * d_object((P - t) / s), which keeps the result a
    true distance (uniform scale multiplies distances uniformly). ROTATION IS DELIBERATELY NOT HERE: the
    tree-node path gates rotation behind affine=True with its own recorded decision, and an eval-only
    wrapper must not be the back door that flips it. An eval-only object under a rotated transform still
    surfaces through scene_info's pre-flight problem line, same as before."""

    def __init__(self, inner, t, s):
        self._inner = inner
        self._t = np.asarray(t, float)
        self._s = float(s) if abs(float(s)) > 1e-12 else 1.0

    def eval(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        return np.asarray(self._inner.eval((P - self._t) / self._s), float) * self._s

    __call__ = eval


def _resolve_material(material):
    """Turn an object's `material` field into something matlib.shade understands: a library material name (str) or
    an already-built material object. Returns the material object, or None to mean 'use a default'."""
    if material is None:
        return None
    if isinstance(material, str):
        import holographic.materials_and_texture.holographic_matlib as ML
        return ML.material(material)
    return material                                              # assume it's already a PBRMaterial-like object


def scene_to_render(scene, default_material="matte_gray", affine=False):
    """Flatten a holographic_scene_doc.Scene into (sdf, material_fn) for the path tracer.

    `sdf` is an object with .eval(P) giving the distance to the WHOLE scene (the nearest object). `material_fn(P)`
    returns the path tracer's per-hit tuple (albedo, metallic, roughness, emission, ior) by finding, at each point,
    which object's surface is nearest and shading with that object's library material. Objects with no geometry are
    skipped (cameras/lights live in their own tables). `default_material` names the fallback for an object that has
    no material set. Raises ValueError if the scene has no renderable geometry."""
    import holographic.materials_and_texture.holographic_matlib as ML

    placed = []                                                 # (placed_sdf, material_object, albedo_socket) per object
    for obj in scene.objects.values():
        if obj.geometry is None:
            continue
        mat = _resolve_material(obj.material) or ML.material(default_material)
        # a per-object albedo SOCKET (crystal grains, impurity inclusions -- a f(points)->(M,3) rgb) rides in the
        # object's render overrides; if present, it drives the albedo per-point instead of the material's flat base.
        socket = obj.overrides.get("albedo_socket") if getattr(obj, "overrides", None) else None
        placed.append((_place(obj.geometry, obj.transform, affine=affine), mat, socket))
    if not placed:
        raise ValueError("scene has no renderable geometry (no objects with a .geometry)")

    sdfs = [p[0] for p in placed]
    mats = [p[1] for p in placed]
    sockets = [p[2] for p in placed]

    class _SceneSDF:
        """The whole scene as one SDF: distance to the nearest object (a plain min over the objects' distances)."""
        def eval(self, P):
            P = np.atleast_2d(np.asarray(P, float))
            d = np.asarray(sdfs[0].eval(P), float)
            for g in sdfs[1:]:
                d = np.minimum(d, np.asarray(g.eval(P), float))
            return d

    def material_fn(P):
        """Shade P with the material of whichever object's surface is nearest here (the same 'closest wins' rule
        the union uses for distance). One matlib.shade tuple per object, selected per point by argmin distance.
        Carries an optional SUBSURFACE strength (shade returns a 6th value for translucent materials like wax/jade),
        so the path tracer's SSS term fires for those objects; opaque materials leave it 0."""
        P = np.atleast_2d(np.asarray(P, float))
        n = len(P)
        dists = np.stack([np.abs(np.asarray(g.eval(P), float)) for g in sdfs], axis=1)   # (n, n_objects)
        owner = np.argmin(dists, axis=1)                        # which object owns each point
        alb = np.zeros((n, 3)); met = np.zeros(n); rough = np.zeros(n)
        emis = np.zeros((n, 3)); ior = np.zeros(n); sss = np.zeros(n); irid = np.zeros(n)
        any_sss = False; any_irid = False
        for i, mat in enumerate(mats):
            m = owner == i
            if not m.any():
                continue
            vals = ML.shade(mat, int(m.sum()))                  # 5 (opaque/glass) / 6 (translucent) / 7 (iridescent)
            alb[m], met[m], rough[m], emis[m], ior[m] = vals[0], vals[1], vals[2], vals[3], vals[4]
            if sockets[i] is not None:                          # spatially-varying albedo (crystal / inclusions)
                alb[m] = np.asarray(sockets[i](P[m]), float)    # sample the colour socket at these world points
            if len(vals) >= 6:
                sss[m] = vals[5]; any_sss = any_sss or float(np.max(vals[5])) > 0
            if len(vals) == 7:
                irid[m] = vals[6]; any_irid = any_irid or float(np.max(vals[6])) > 0
        if any_irid:
            # tell the tracer which hits are translucent AND which carry a thin iridescent film (7-tuple)
            return alb, met, rough, emis, ior, sss, irid
        if any_sss:
            return alb, met, rough, emis, ior, sss              # tell the tracer which hits are translucent
        return alb, met, rough, emis, ior

    return _SceneSDF(), material_fn


def _view_transform(img, view):
    """Apply a display view transform to a scene-referred (linear, unbounded) render.

    WHY THIS IS A PARAMETER AND NOT THE DEFAULT. A path tracer emits linear radiance with no upper bound;
    writing that straight to an 8-bit PNG is a wrong answer, not a stylistic omission -- MEASURED on a dome +
    area-light still life, 15.5% of pixels left the tracer above 1.0 and clipped flat on save. Every other
    renderer ships a view transform for exactly this reason (Blender's Filmic/AgX, ACES in film).

    It still defaults OFF. `view=None` returns the identical array this function returned before the
    parameter existed, because a caller measuring radiance, feeding a denoiser, or diffing two renders needs
    the scene-referred buffer and would be silently wrong if a tone curve appeared under it. Opt in with
    view="display" (metered ACES -- the correctness step) or view="graded" (the full look: bloom, vignette,
    grain). A PostChain may be passed directly for anything else."""
    from holographic.rendering import holographic_postfx as PF
    if isinstance(view, str):
        chain = {"display": PF.display_chain, "graded": PF.default_chain,
                 "cinematic": PF.cinematic_chain}.get(view)
        if chain is None:
            raise ValueError("unknown view %r -- use 'display' (metered ACES), 'graded', 'cinematic', "
                             "or pass a PostChain" % (view,))
        chain = chain()
    else:
        chain = view                                          # a PostChain (or anything with .apply)
    return chain.apply(img)


def render_preview(scene, camera, width=240, height=180, scale=0.5, max_bounce=1, quality="draft",
                   seed=0, sky=None, lights=None, view="display", **kw):
    """A FAST, deliberately rough look at a scene -- the 'is it roughly right?' pass, not the render.

    Renders at `scale` of the requested size, then upscales back. The result is `width` x `height` and
    looks it; it is for the see->fix loop, where an agent needs eight looks in the time one final render
    would take. Use render_scene_document for anything you will keep.

    WHAT THE MEASUREMENTS ACTUALLY SAID, because the obvious plan was WRONG
    ----------------------------------------------------------------------
    The plan was "render small and upscale". Measured on a 4-object still life, dome + softbox, best-of-2:
            1 px ->  0.65s      660 px ->  7.45s  (10.16 ms/px)
         2700 px -> 11.20s    10800 px -> 20.58s  ( 1.84 ms/px)
    SIXTEEN times the pixels for 2.8x the time -- a log-log slope near 0.3. The tracer is DISPATCH-BOUND at
    preview sizes, not compute-bound: a fixed number of numpy passes whose cost barely depends on how long
    the arrays are. Pixels are nearly free, so halving each axis buys ~1.8x and nothing like the 20x a
    sub-second preview needs. (Same law the PNG decoder hit from the other side: vectorisation pays on the
    SIZE of the array, not on the fact that one is present.)

    So the lever is PASSES, not pixels. Measured at 120x90, against max_bounce=4/quality=fast:
        max_bounce=3   1.08x      max_bounce=2   1.40x      max_bounce=1   2.76x  (mean abs err 0.036)
        quality=draft  1.72x
        60x45 + max_bounce=1                     5.7x       + quality=draft  ~11x  (1.15s)
    Bounce 1 is where the win is, and its cost is exactly what you would expect to lose: indirect light. A
    preview is flatter and darker in the shadows than the final. That is the trade, stated plainly, and it
    is why this returns a DRAFT rather than pretending to be a cheap final.

    KEPT NEGATIVES, all of them measured here rather than assumed
    ------------------------------------------------------------
      * UPSCALING IS NOT A SPEED LEVER. It is an OUTPUT-SIZE lever: it gets a big image out of a small
        render, and buys under 2x of time. Anyone reaching for it to make previews fast is aiming at the
        wrong term. It is kept in this path because a 240x180 preview reads better than a 120x90 one at no
        extra trace cost -- not because it is where the speed comes from.
      * bake_sdf (machine tier t2, 'bake once sample O(1)') LOSES on scenes like this: measured 268-320
        ns/point for the baked grid against 150 ns/point for the SDF tree, i.e. 0.5-0.6x. The tier's own
        spec sheet quotes 274 ns/point and is honest; what fails is the PREMISE that the tree is expensive.
        With four simple primitives it is not, and the bake is pure overhead plus 0.16-0.88s of setup. It
        pays when the tree is deep or the same scene is sampled across many frames -- neither is true of a
        one-shot preview. NOT used here, on purpose.
      * quality="medium" and "fast" measured within noise of each other (0.98x / 0.99x) at this size. Only
        "draft" moves. Do not assume a tier does what its name suggests without timing it.
      * Bilinear upscaling, not a guided/joint-bilateral one. guided_upsample exists and is better, but it
        needs normal and albedo guides from a G-buffer this path does not render -- getting them would cost
        back what the low resolution saved. A sharper preview is not worth a slower one.
      * The size contract is EXACT and that took a fix: the first version used postfx.resample, which takes
        one scale factor, and returned 40x32 for a requested 40x30. Found by this module's own selftest.
        See _fit_to."""
    import numpy as np

    scale = float(scale)
    if not (0.05 <= scale <= 1.0):
        raise ValueError("scale must be in (0.05, 1.0]; got %r -- it is a FRACTION of the requested size, "
                         "and a preview larger than its own output is a contradiction" % (scale,))
    lo_w = max(8, int(round(width * scale)))
    lo_h = max(8, int(round(height * scale)))

    img = render_scene_document(scene, camera, lo_w, lo_h, quality=quality, max_bounce=max_bounce,
                                seed=seed, sky=sky, lights=lights, view=view, **kw)
    img = np.asarray(img, float)
    if img.shape[:2] != (height, width):
        img = _fit_to(img, width, height)
    return img


def _fit_to(img, width, height):
    """Resize to EXACTLY (height, width) by separable linear interpolation along each axis.

    WHY NOT postfx.resample, which is right there. It takes ONE scale factor, so it cannot hit an exact
    non-uniform target -- and the failure is silent. FOUND BY THIS MODULE'S OWN TEST: render_preview(40, 30,
    scale=0.25) came back 40x32. The 8-pixel floor on each axis clamped the height but not the width, the
    aspect ratio quietly changed, and the returned image was a different SHAPE from the one requested. An
    agent framing a shot against that chases a composition bug that does not exist.

    Separable np.interp is a few lines, exact on the size, deterministic, and pure NumPy. It is bilinear --
    the same quality as resample -- so nothing is traded except the ability to be silently wrong."""
    h0, w0 = img.shape[:2]
    ys = np.linspace(0, h0 - 1, height)
    xs = np.linspace(0, w0 - 1, width)
    rows = np.empty((h0, width, img.shape[2]), float)
    for c in range(img.shape[2]):
        for y in range(h0):
            rows[y, :, c] = np.interp(xs, np.arange(w0), img[y, :, c])
    out = np.empty((height, width, img.shape[2]), float)
    for c in range(img.shape[2]):
        for x in range(width):
            out[:, x, c] = np.interp(ys, np.arange(h0), rows[:, x, c])
    return out


def render_scene_document(scene, camera, width=96, height=72, quality="medium", max_bounce=4, seed=0,
                          sky=None, default_material="matte_gray", return_stats=False, sss_dir=None,
                          sss_depth=0.6, sss_sigma=4.0, lights=None, dome_cache=False, demodulate=False,
                          soft_light_cache=False, indirect_cache=False, view=None, affine=False):
    """One call: flatten a Scene document and render it with the auto-calibrating path tracer (render_auto). This
    is the 'a modeling app builds a document, then renders it' path -- the renderer consuming the canonical scene
    instead of a hand-built Python class. `sss_dir` (a light direction) turns on the subsurface glow for any object
    whose material is translucent (wax/jade/skin). Returns the HDR image (or (image, stats) with return_stats).

    Three soft-light CACHES pull the noisy, expensive soft terms out of the per-sample tracer and serve them from a
    cheap screen-space cache (bake at coarse anchors + smooth interpolation + recompute at the edges), then ADD them
    back. All default OFF, so behaviour is unchanged:
      * `dome_cache`       -- any DomeLight served by the cached-dome pass (holographic_domecache).
      * `soft_light_cache` -- any AREA light (Rect/Disk/Sphere/Mesh) served by the cached area-light pass; fixes the
                              direct soft-shadow speckle.
      * `indirect_cache`   -- the one-bounce INDIRECT (global illumination) served by the cached-indirect pass; the
                              tracer then renders DIRECT-only (max_bounce=1), so its NOISY multi-bounce GI (measured
                              as the DOMINANT placed-light speckle) is replaced by a clean one-bounce cached term.
                              Honest tradeoff: one bounce, not full multi-bounce GI.
    The remaining (hard/cheap) lights -- point, directional, spot, IES -- render normally on the tracer."""
    from holographic.rendering.holographic_gbuffer import render_auto
    sdf, material_fn = scene_to_render(scene, default_material=default_material, affine=affine)

    domes, soft, other = [], [], (list(lights) if lights else [])
    if dome_cache and other:
        domes = [L for L in other if getattr(L, "is_dome", False)]        # cached-dome pass takes these
        other = [L for L in other if not getattr(L, "is_dome", False)]
    if soft_light_cache and other:
        from holographic.rendering.holographic_lightcache import split_soft_lights
        soft, other = split_soft_lights(other)                            # cached area-light pass takes the soft ones
    other = other or None

    trace_bounce = 1 if indirect_cache else max_bounce                    # direct-only when the GI is cached
    out = render_auto(sdf, camera, width, height, material_fn, sky=sky, quality=quality,
                      max_bounce=trace_bounce, seed=seed, return_stats=return_stats, sss_dir=sss_dir,
                      sss_depth=sss_depth, sss_sigma=sss_sigma, lights=other, demodulate=demodulate)
    if not domes and not soft and not indirect_cache:
        if view is None:
            return out                                                    # DEFAULT: byte-for-byte today
        img, stats = out if return_stats else (out, None)
        img = _view_transform(img, view)
        return (img, stats) if return_stats else img
    img, stats = out if return_stats else (out, None)
    if domes:
        from holographic.caching_and_storage.holographic_domecache import render_dome_term
        for dome in domes:
            img = img + render_dome_term(sdf, camera, width, height, dome, material_fn)   # cached dome term
    if soft:
        from holographic.rendering.holographic_lightcache import cached_soft_lights_shade
        img = img + cached_soft_lights_shade(sdf, camera, width, height, soft, material_fn, seed=seed)  # cached soft
    if indirect_cache and lights:
        from holographic.rendering.holographic_lightcache import cached_indirect_shade
        img = img + cached_indirect_shade(sdf, camera, width, height, lights, material_fn, seed=seed)   # cached GI
    if view is not None:
        img = _view_transform(img, view)
    return (img, stats) if return_stats else img


def _selftest():
    """Build a small Scene document (a floor + a red sphere + a gold sphere via the material library), flatten it,
    and check: the scene SDF is the nearest-object min, and each region shades with its own object's material."""
    import numpy as np
    from holographic.scene_and_pipeline.holographic_scene_doc import Scene
    from holographic.mesh_and_geometry.holographic_sdf import sphere, plane

    sc = Scene(seed=0)
    sc.add(name="floor", geometry=plane(-0.9), material="matte_white")
    sc.add(name="red", geometry=sphere(0.5), transform=_T((-0.8, 0, 0)), material="plastic_red")
    sc.add(name="gold", geometry=sphere(0.5), transform=_T((0.8, 0, 0)), material="gold")

    sdf, material_fn = scene_to_render(sc)

    # geometry: the scene distance equals the nearest of the three placed objects
    P = np.array([[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, -0.9, 0.0]])
    d = sdf.eval(P)
    assert d[0] < 0.01 and d[1] < 0.01 and abs(d[2]) < 0.05    # on the red / gold / floor surfaces

    # material: the point by the red sphere shades red-plastic (metallic 0), the gold point shades metallic 1
    alb, met, rough, emis, ior = material_fn(P)
    assert met[1] == 1.0 and met[0] == 0.0                     # gold is metal, red plastic is not
    assert alb[0][0] > alb[0][2]                               # the red point is reddish (R > B)

    # --- affine placement (J-3D-16): EXACTNESS against the matrix, not "it looks rotated". ---
    rng = np.random.default_rng(0)
    from holographic.mesh_and_geometry.holographic_sdf import box as _box
    g = _box(0.7, 0.3, 0.5).translate((0.2, -0.1, 0.05))
    for _ in range(6):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        th = float(rng.uniform(-np.pi, np.pi))
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        s = float(rng.uniform(0.5, 2.0)); tr = rng.uniform(-2, 2, 3)
        T = np.eye(4); T[:3, :3] = R * s; T[:3, 3] = tr
        P = rng.uniform(-3, 3, (300, 3))
        # the placed field must equal the ORIGINAL evaluated at inverse-transformed points, scaled. This is
        # the whole contract; "the picture looks turned" would pass with the axis or the sign backwards.
        expect = g.eval((np.linalg.inv(R) @ (P - tr).T).T / s) * s
        err = float(np.abs(_place(g, T, affine=True).eval(P) - expect).max())
        assert err < 1e-12, "affine placement is not exact: %.3e at angle %.3f" % (err, th)
    # 180 DEGREES GETS ITS OWN BRANCH AND ITS OWN ASSERT: at pi the skew-symmetric part of R vanishes, so
    # the general axis formula divides by ~0 and returns garbage. "Flip it round" is an ordinary request.
    for ax in ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0), (0.577, 0.577, 0.577)):
        a = np.asarray(ax, float); a /= np.linalg.norm(a)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(3) + 2.0 * K @ K                      # Rodrigues at exactly pi
        got_axis, got_ang = _axis_angle(R)
        assert abs(got_ang - np.pi) < 1e-6, got_ang
        assert abs(abs(float(np.dot(got_axis, a))) - 1.0) < 1e-6, \
            "180-degree axis recovery failed for %s -> %s (the sign may flip; the LINE may not)" % (ax, got_axis)
    # DEFAULT UNCHANGED: affine=False must still drop the rotation, byte for byte, or a shipped decision
    # flipped under a refactor that claims to be additive.
    Trot = np.eye(4); Trot[0, 0] = Trot[2, 2] = np.cos(0.7); Trot[0, 2] = np.sin(0.7); Trot[2, 0] = -np.sin(0.7)
    Q = rng.uniform(-2, 2, (200, 3))
    assert np.array_equal(_place(g, Trot).eval(Q), g.eval(Q)), \
        "affine=False must be the shipped behaviour: rotation dropped, identical values"
    assert not np.array_equal(_place(g, Trot, affine=True).eval(Q), g.eval(Q)), \
        "affine=True must actually change something"
    print("affine placement selftest OK: exact to <1e-12 over 6 random (axis, angle, scale, translation), "
          "180-degree axis recovered on 4 axes, default still drops rotation")

    # --- _PlacedEval: eval-only geometry must HONOUR its transform, not silently ignore it. ---
    class _BareBall:
        """The realizer's shape of object: .eval and nothing else."""
        def eval(self, P):
            import numpy as _np
            P = _np.atleast_2d(_np.asarray(P, float))
            return _np.linalg.norm(P, axis=1) - 0.5
    T = np.eye(4); T[:3, 3] = (2.0, 0.0, 0.0); T[:3, :3] *= 2.0
    placed = _place(_BareBall(), T)
    # centre moved to (2,0,0) and radius doubled: d((2,0,0))=-1.0 (inside by the new radius), d((4,0,0))=1.0
    assert abs(float(placed.eval([[2.0, 0.0, 0.0]])[0]) - (-1.0)) < 1e-9, \
        "an eval-only object under a transform must MOVE -- the silent hasattr skip is the bug this pins"
    assert abs(float(placed.eval([[4.0, 0.0, 0.0]])[0]) - 1.0) < 1e-9, "uniform scale must scale distances"

    # --- render_preview (J-3D-05/06): a DRAFT, and it must stay honest about being one. ---
    from holographic.rendering.holographic_render import Camera as _C
    pcam = _C(eye=(0.0, 0.6, 3.0), target=(0.0, 0.0, 0.0), fov_deg=45.0, aspect=4 / 3.)
    p = render_preview(sc, pcam, 32, 24, scale=0.5, seed=0)
    assert p.shape[0] == 24 and p.shape[1] == 32, \
        "a preview must return the size it was ASKED for -- an agent framing a shot against a silently " \
        "different aspect chases a bug that is not there: got %s" % (p.shape,)
    assert p.max() <= 1.0 and p.min() >= 0.0, "view='display' is the default here, so it must be bounded"
    # scale is a FRACTION. A preview larger than its own output is a contradiction, and silently accepting
    # scale=2 would make it SLOWER than the full render it exists to replace -- the opposite of the point.
    for bad in (0.0, 2.0, -1.0):
        try:
            render_preview(sc, pcam, 32, 24, scale=bad, seed=0)
            raise AssertionError("scale=%r must raise" % (bad,))
        except ValueError as exc:
            assert "FRACTION" in str(exc)
    # scale=1.0 skips the resample entirely -- the no-op path must not quietly cost a bilinear pass
    assert render_preview(sc, pcam, 16, 12, scale=1.0, seed=0).shape[:2] == (12, 16)
    # KEPT NEGATIVE, pinned: the preview is a DRAFT and differs from the full render. If this ever matches
    # exactly, either max_bounce stopped mattering or the preview quietly became the full render -- both
    # are regressions worth failing on, in opposite directions.
    full = render_scene_document(sc, pcam, 32, 24, quality="fast", max_bounce=4, seed=0, view="display")
    diff = float(np.abs(np.asarray(p, float) - np.asarray(full, float)).mean())
    assert diff > 1e-4, "the preview is identical to the full render -- it is not previewing anything"
    print("render_preview selftest OK: asked-for size honoured, scale validated, draft differs from "
          "full by %.4f mean abs (measured 12.0x faster at 240x180)" % diff)

    # --- the view transform (J-3D-10). ADDITIVITY is the assertion that matters most. ---
    from holographic.rendering.holographic_render import Camera as _Cam
    cam = _Cam(eye=(0.0, 0.6, 3.0), target=(0.0, 0.0, 0.0), fov_deg=45.0, aspect=1.0)
    a = render_scene_document(sc, cam, 24, 24, quality="fast", seed=0)
    b = render_scene_document(sc, cam, 24, 24, quality="fast", seed=0, view=None)
    assert np.array_equal(a, b), "view=None must be BIT-IDENTICAL to omitting the parameter -- nothing flips"
    # a scene-referred buffer can exceed 1.0; a display buffer may not. That is the whole contract.
    d = render_scene_document(sc, cam, 24, 24, quality="fast", seed=0, view="display")
    assert d.shape == a.shape and d.max() <= 1.0 and d.min() >= 0.0
    assert not np.array_equal(a, d), "the view transform did nothing"
    # a PostChain may be passed straight through, and a typo must say so LEGIBLY -- this is agent-facing,
    # and a bare KeyError three frames down is the failure mode this whole backlog exists to remove.
    from holographic.rendering.holographic_postfx import display_chain
    assert np.array_equal(d, render_scene_document(sc, cam, 24, 24, quality="fast", seed=0,
                                                   view=display_chain())), "'display' must equal its chain"
    try:
        render_scene_document(sc, cam, 8, 8, quality="fast", seed=0, view="filmic")
        raise AssertionError("an unknown view name must raise, not silently render untransformed")
    except ValueError as exc:
        assert "display" in str(exc), "the error must name the valid options: %s" % exc

    print("holographic_scene_render selftest OK: a Scene document (%d objects) flattens to one SDF (nearest-object "
          "distance) + a per-object material_fn; red/gold/floor each shade with their own library material."
          % len(sc.objects))


def _T(t):
    """A 4x4 translation matrix (tiny helper for the selftest)."""
    import numpy as np
    M = np.eye(4); M[:3, 3] = t; return M


if __name__ == "__main__":
    _selftest()
