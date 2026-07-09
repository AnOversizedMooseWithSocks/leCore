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


def _place(geometry, transform):
    """Return the object's SDF placed in world space by its transform (uniform scale about the origin, then
    translate). Uses the SDF tree's own combinators, so the placed geometry is still a normal SDF node."""
    t, s = _decompose(transform)
    g = geometry
    if abs(s - 1.0) > 1e-9 and hasattr(g, "scale"):
        g = g.scale(s)
    if np.linalg.norm(t) > 1e-9 and hasattr(g, "translate"):
        g = g.translate(tuple(t))
    return g


def _resolve_material(material):
    """Turn an object's `material` field into something matlib.shade understands: a library material name (str) or
    an already-built material object. Returns the material object, or None to mean 'use a default'."""
    if material is None:
        return None
    if isinstance(material, str):
        import holographic.materials_and_texture.holographic_matlib as ML
        return ML.material(material)
    return material                                              # assume it's already a PBRMaterial-like object


def scene_to_render(scene, default_material="matte_gray"):
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
        placed.append((_place(obj.geometry, obj.transform), mat, socket))
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


def render_scene_document(scene, camera, width=96, height=72, quality="medium", max_bounce=4, seed=0,
                          sky=None, default_material="matte_gray", return_stats=False, sss_dir=None,
                          sss_depth=0.6, sss_sigma=4.0, lights=None, dome_cache=False, demodulate=False,
                          soft_light_cache=False, indirect_cache=False):
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
    sdf, material_fn = scene_to_render(scene, default_material=default_material)

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
        return out
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

    print("holographic_scene_render selftest OK: a Scene document (%d objects) flattens to one SDF (nearest-object "
          "distance) + a per-object material_fn; red/gold/floor each shade with their own library material."
          % len(sc.objects))


def _T(t):
    """A 4x4 translation matrix (tiny helper for the selftest)."""
    import numpy as np
    M = np.eye(4); M[:3, 3] = t; return M


if __name__ == "__main__":
    _selftest()
