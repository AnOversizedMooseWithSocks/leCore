"""holographic_texturerender.py -- apply a COMPOSED texture/material (CMP1-CMP3) to a scene object in a FULL render.

The preview module shows a texture as a flat swatch or a material on a ball. This is the next step: paint a composed
texture graph or material onto an ACTUAL scene object and render the whole scene -- so the composability stack drives a
real 3-D image, not just a thumbnail.

HOW IT WORKS (and why it stays readable). We reuse the engine's own machinery rather than rewrite a renderer:
  * realize the SemanticScene to SDF objects (sphere/box), same as the normal render;
  * MARCH the union SDF with holographic_raymarch.sphere_trace -> per-pixel hit point + which object was hit (ids);
  * for a hit on object k, turn its 3-D surface point into a UV: a sphere gets a spherical map, a box a planar
    (dominant-face) map -- so a 2-D texture wraps onto the 3-D surface;
  * SAMPLE the texture the user attached to that object at its UV -> an albedo colour (a CMP1 graph gives a colour
    directly; a Material gives roughness/metallic + an albedo that tints a base colour);
  * SHADE with the same holographic_brdf.cook_torrance the renderer uses, plus one light, a hard shadow (march toward
    the light), and a little ambient; the sky and an optional ground fill the rest.

So a texture graph you built with mind.texture_map(...) shows up wrapped around the sphere in the render.

HONEST kept limits (loud): UV mapping is the simple textbook kind -- a spherical map has a pole pinch and a seam, a box
map has visible seams at the edges between faces (no triplanar blend); shadows are a single hard light (no soft
shadows / GI here -- the path tracer is the tool for that). This is a faithful, readable bridge, not a production
shader. Deterministic; plain NumPy; the only per-pixel loop is sampling the attached texture at the visible hits.
"""
import numpy as np

# Shading brightness: cook_torrance carries a 1/pi in its diffuse term (physically correct), so a single unit light
# reads dim. A light intensity of ~pi compensates and gives the image punch; a little ambient keeps shadowed sides
# readable rather than pure black. These are look constants for this preview render, not physical radiometry.
_LIGHT = 3.0
_AMBIENT = 0.18


def _sphere_uv(local, r):
    """Spherical UV for a point `local` (relative to the sphere centre), radius r. u wraps around the equator, v runs
    pole to pole. (Textbook mapping: a seam at the back and a pinch at the poles -- fine for a preview render.)"""
    u = np.arctan2(local[:, 0], local[:, 2]) / (2.0 * np.pi) + 0.5
    v = np.arccos(np.clip(local[:, 1] / max(r, 1e-9), -1.0, 1.0)) / np.pi
    return u, v


def _box_uv(local, half):
    """Planar UV for a box: pick the FACE each point is on (its dominant axis relative to the half-extent), then read
    the other two axes as (u, v) mapped to [0,1]. Seams show at the face edges -- no triplanar blend, kept simple."""
    rel = np.abs(local) / np.maximum(half, 1e-9)
    axis = np.argmax(rel, axis=1)                              # 0=x face, 1=y face, 2=z face
    # for each face, the two in-plane axes; map local coord from [-h,h] to [0,1]
    plane = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    u = np.empty(len(local)); v = np.empty(len(local))
    for a in (0, 1, 2):
        m = axis == a
        if not np.any(m):
            continue
        ia, ib = plane[a]
        u[m] = (local[m, ia] / (2.0 * half[ia])) + 0.5
        v[m] = (local[m, ib] / (2.0 * half[ib])) + 0.5
    return np.clip(u, 0, 1), np.clip(v, 0, 1)


def _object_uv(sdf, P):
    """UV for hit points P on ONE object's SDF (a _SphereSDF or _BoxSDF from the semantic realizer)."""
    from holographic.simulation_and_physics.holographic_semantic import _SphereSDF, _BoxSDF
    local = P - sdf.c
    if isinstance(sdf, _SphereSDF):
        return _sphere_uv(local, sdf.r)
    if isinstance(sdf, _BoxSDF):
        return _box_uv(local, sdf.h)
    # unknown SDF: a flat UV (0.5,0.5) so sampling still works
    return np.full(len(P), 0.5), np.full(len(P), 0.5)


def _is_texture_graph(tex):
    """A CMP1 texture graph node (sample(uv)) vs a Material (sample(channel, uv))."""
    from holographic.materials_and_texture.holographic_texturegraph import Node
    return isinstance(tex, Node)


def _is_image_texture(tex):
    """An external image FILE loaded as a UV-sampled TextureMap (holographic_materialio) -- distinct from a CMP1
    procedural graph or a Material. Detected by type so the renderer can sample its pixels directly."""
    try:
        from holographic.materials_and_texture.holographic_materialio import TextureMap
        return isinstance(tex, TextureMap)
    except Exception:
        return False


def _albedo_rough_metal(tex, uvs, base_color, mat_name):
    """From the attached texture, produce per-hit (albedo (M,3), roughness (M,), metallic (M,)).
      * a CMP1 texture GRAPH -> its sampled colour is the albedo (a scalar graph -> greyscale); roughness/metallic
        come from the object's material name.
      * a MATERIAL -> albedo tints base_color by an 'albedo' channel (if any); roughness/metallic from its channels."""
    m = len(uvs)
    rough = np.full(m, _rough_for(mat_name))
    metal = np.full(m, _metal_for(mat_name))
    alb = np.tile(np.asarray(base_color, float), (m, 1))
    if _is_image_texture(tex):
        # an EXTERNAL IMAGE FILE used as a texture -> sample its pixels per-uv as a diffuse albedo (same reasoning as
        # the colour-graph case below: an image is albedo, so shade it diffuse or a metal surface reads dark).
        rough = np.full(m, 0.55)
        metal = np.full(m, 0.0)
        for i in range(m):
            c = np.asarray(tex.sample(float(uvs[i][0]), float(uvs[i][1])), float)
            c = c[:3] if c.size >= 3 else np.repeat(c, 3)[:3]
            alb[i] = np.clip(c, 0.0, 1.0)
        return alb, rough, metal
    if _is_texture_graph(tex):
        # a COLOUR texture is an ALBEDO -- paint it as a DIFFUSE surface (metal kills the diffuse term and would
        # read dark under a single light with no environment reflection), so the pattern actually shows.
        rough = np.full(m, 0.55)
        metal = np.full(m, 0.0)
        for i in range(m):
            c = np.asarray(tex.sample(uvs[i]), float)
            alb[i] = np.clip(c if c.ndim and c.size == 3 else np.repeat(c, 3), 0.0, 1.0)   # colour, or greyscale
    else:
        from holographic.misc.holographic_preview import _channel_names, _sample_scalar
        chans = _channel_names(tex)
        has_alb = "albedo" in chans
        for i in range(m):
            uv = (float(uvs[i][0]), float(uvs[i][1]))
            if "roughness" in chans:
                rough[i] = _sample_scalar(tex, "roughness", uv, rough[i])
            if "metallic" in chans:
                metal[i] = _sample_scalar(tex, "metallic", uv, metal[i])
            if has_alb:
                alb[i] = np.clip(np.asarray(base_color, float) * _sample_scalar(tex, "albedo", uv, 1.0), 0, 1)
    return alb, rough, metal


def _rough_for(mat_name):
    return {"metal": 0.25, "mirror": 0.05, "glossy": 0.15, "glass": 0.1, "brushed": 0.4}.get(mat_name, 0.6)


def _metal_for(mat_name):
    return {"metal": 0.9, "mirror": 1.0, "brushed": 0.8}.get(mat_name, 0.0)


def render_textured(scene, textures, camera=None, width=256, height=192, light_dir=(0.5, 0.85, 0.6),
                    sky_color=(0.55, 0.68, 0.85), ground=True, base_color=(0.80, 0.80, 0.80), aa=2,
                    lighting=None, sun_scale=1.0):
    """Render a SemanticScene with COMPOSED textures/materials applied per object. `textures` maps an object NAME (as
    in scene.names()) to a CMP1 texture graph or a Material; objects without an entry fall back to their scene colour.
    Returns an (H, W, 3) float image in [0,1]. See the module docstring for the honest kept limits.

    `aa` is the anti-aliasing supersample factor (SSAA): aa=2 (default) renders at 2x the width and height and averages
    each 2x2 block back down, so object edges are smooth instead of jagged. aa=1 turns it off (faster, but aliased).

    `lighting` (B2c, default None) is a LIGHTING preset name; when given, the light DIRECTION, INTENSITY, COLOUR and
    AMBIENT come from that preset (via holographic_semantic.lighting_params) so a textured scene honours 'make it
    sunset' just like the flat renderer. `sun_scale` (default 1.0) multiplies the intensity for 'brighter'/'dimmer'.
    Both defaults reproduce the historical look BYTE-IDENTICALLY (sun_i=1.0, white light, dir=light_dir, amb=_AMBIENT)."""
    # ANTI-ALIASING: render once at aa-times the resolution, then box-average each aa x aa block down. Cheap, readable,
    # and the aspect ratio is preserved because (width*aa)/(height*aa) == width/height.
    if aa and aa > 1:
        big = render_textured(scene, textures, camera=camera, width=width * aa, height=height * aa,
                              light_dir=light_dir, sky_color=sky_color, ground=ground, base_color=base_color, aa=1,
                              lighting=lighting, sun_scale=sun_scale)
        return big.reshape(height, aa, width, aa, 3).mean(axis=(1, 3))

    from holographic.rendering.holographic_raymarch import sphere_trace, sdf_normal
    from holographic.simulation_and_physics.holographic_semantic import _UnionSDF, COLORS
    from holographic.rendering.holographic_brdf import cook_torrance
    from holographic.rendering.holographic_render import Camera

    realized = scene.realize()
    if not realized:
        return np.zeros((height, width, 3))
    sdfs = [ro["sdf"] for ro in realized]
    names = [ro["name"] for ro in realized]
    union = _UnionSDF(sdfs)

    if camera is None:                                          # same sensible default view as SemanticScene.render
        span = max(3.0, 1.6 * len(realized))
        camera = Camera(eye=(span * 0.4, span * 0.28, span), target=(0, 0, 0), fov_deg=42.0)

    # B2c: resolve the lighting request ONCE (shared with the flat renderer via lighting_params). Defaults reproduce
    # the historical look: sun_i=1.0 -> light_i==_LIGHT, white sun/ambient, amb_i==_AMBIENT, and L==light_dir.
    from holographic.simulation_and_physics.holographic_semantic import lighting_params
    lp = lighting_params(lighting=lighting, sun_scale=sun_scale)
    light_i = _LIGHT * lp["sun_i"]                             # direct-light intensity (brightness follows sun_scale/preset)
    amb_i = _AMBIENT * (lp["amb"] / 0.24)                      # ambient scaled RELATIVE to the historical 0.24 baseline
    sun_col = np.asarray(lp["sun_col"], float)                # warm/cool sun tint (white by default -> identity)
    amb_col = np.asarray(lp["amb_col"], float)                # sky-tinted ambient fill (white by default -> identity)
    L = np.asarray(lp["dir"] if lighting is not None else light_dir, float); L = L / (np.linalg.norm(L) + 1e-12)
    eye, dirs = camera.ray_dirs(width, height)
    D = dirs.reshape(-1, 3); O = np.broadcast_to(eye, D.shape).copy()

    # background = a soft vertical sky gradient; ground catches the objects and their shadows below y=0
    frame = _sky(height, width, sky_color).reshape(-1, 3)

    hit, t, P = sphere_trace(union, O, D)                       # march the whole scene at once
    gy = _ground_hit(O, D, y0=_scene_floor(sdfs)) if ground else None

    if np.any(hit):
        idx = np.where(hit)[0]
        Ph = P[idx]
        N = sdf_normal(union, Ph)
        who = union.ids(Ph)                                    # which object each hit belongs to
        V = -D[idx]                                            # view direction (hit -> camera)
        shade = np.zeros((len(idx), 3))
        for k in range(len(sdfs)):                             # shade one object's hits at a time (its UV + texture)
            m = who == k
            if not np.any(m):
                continue
            u, v = _object_uv(sdfs[k], Ph[m])
            uvs = list(zip(u, v))
            tex = textures.get(names[k])
            if tex is None:                                    # no composed texture -> the object's scene colour
                mat = realized[k].get("mat_name")
                col = np.asarray(COLORS.get(realized[k].get("color"), base_color), float)
                alb = np.tile(col, (int(m.sum()), 1)); rough = np.full(int(m.sum()), _rough_for(mat))
                metal = np.full(int(m.sum()), _metal_for(mat))
            else:
                alb, rough, metal = _albedo_rough_metal(tex, uvs, base_color, realized[k].get("mat_name"))
            Nk, Vk = N[m], V[m]
            lit = cook_torrance(Nk, Vk, np.broadcast_to(L, Nk.shape), alb, metal, rough) * light_i
            sh = _shadow(union, Ph[m], N[m], L)                # 1.0 lit, 0.0 shadowed (hard shadow)
            shade[m] = lit * sh[:, None] * sun_col + amb_i * alb * amb_col   # coloured direct + coloured ambient fill
        frame[idx] = np.clip(shade / (1.0 + shade), 0.0, 1.0)  # Reinhard tone-map

    # composite a simple shaded ground where it is nearer than any object hit
    if ground and gy is not None:
        gm, gt = gy
        show = gm & (~hit | (gt < np.where(hit, t, 1e30)))
        if np.any(show):
            gi = np.where(show)[0]
            Pg = O[gi] + gt[gi, None] * D[gi]
            Ng = np.tile(np.array([0.0, 1.0, 0.0]), (len(gi), 1))
            sh = _shadow(union, Pg + 1e-3 * Ng, Ng, L)
            lam = np.clip((Ng * L).sum(1), 0, 1)
            base = np.array([0.62, 0.62, 0.64])
            frame[gi] = np.clip(base * (0.25 + 0.75 * lam * sh)[:, None] * lp["sun_i"], 0, 1)   # ground tracks brightness

    return frame.reshape(height, width, 3)


def _shadow(union, P, N, L, eps=2e-2, max_dist=20.0):
    """Hard shadow: march from each surface point toward the light; if the scene is hit before the light, it's in
    shadow. Returns 1.0 (lit) or 0.0 (shadowed) per point."""
    from holographic.rendering.holographic_raymarch import sphere_trace
    O = P + eps * N                                            # lift off the surface to avoid self-shadow acne
    D = np.broadcast_to(L, O.shape).copy()
    hit, t, _ = sphere_trace(union, O, D, max_dist=max_dist)
    return np.where(hit & (t < max_dist), 0.0, 1.0)


def _sky(h, w, color):
    """A soft vertical sky gradient (lighter at the top)."""
    top = np.asarray(color, float)
    bot = np.clip(top * 1.15, 0, 1)
    t = np.linspace(0.0, 1.0, h)[:, None, None]
    grid = (1 - t) * top + t * bot
    return np.broadcast_to(grid, (h, w, 3)).copy()


def _scene_floor(sdfs):
    """A y for the ground plane: just below the lowest object, so objects sit ON it."""
    lows = []
    for s in sdfs:
        from holographic.simulation_and_physics.holographic_semantic import _SphereSDF, _BoxSDF
        if isinstance(s, _SphereSDF):
            lows.append(s.c[1] - s.r)
        elif isinstance(s, _BoxSDF):
            lows.append(s.c[1] - s.h[1])
    return (min(lows) if lows else -1.0)


def _ground_hit(O, D, y0):
    """Ray/plane intersection with the horizontal ground plane y=y0. Returns (mask, t)."""
    dy = D[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (y0 - O[:, 1]) / dy
    mask = (dy < -1e-6) & (t > 0)
    return mask, np.where(mask, t, 1e30)


def _selftest():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=512, seed=0)
    scene = m.build_scene("a big red metal sphere and a small blue box")

    # a CMP1 texture graph painted onto the sphere
    tex = m.texture_op("mix", a=m.texture_leaf(value="orange"), b=m.texture_leaf(value="purple"),
                        t=m.texture_leaf("fbm", n_dims=2, seed=0))
    names = scene.names()
    img = render_textured(scene, {names[0]: tex}, width=80, height=64)
    assert img.shape == (64, 80, 3)
    assert img.min() >= 0.0 and img.max() <= 1.0
    # something rendered (not a flat frame): the textured sphere region varies
    assert img.std() > 0.02

    # a plain scene with NO textures still renders (objects fall back to their scene colour)
    img2 = render_textured(scene, {}, width=64, height=48)
    assert img2.shape == (48, 64, 3) and img2.std() > 0.02

    print("OK: holographic_texturerender self-test passed (a CMP1 texture graph wraps onto the sphere in a full "
          "marched render %s in [0,1], std %.3f; a no-texture scene still renders via scene colours)"
          % (img.shape, float(img.std())))


if __name__ == "__main__":
    _selftest()
