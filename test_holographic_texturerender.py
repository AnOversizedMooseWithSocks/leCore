"""Tests for holographic_texturerender.py -- a composed texture/material painted onto a scene object in a full render."""
import numpy as np
from holographic_unified import UnifiedMind
from holographic_texturerender import render_textured, _sphere_uv, _box_uv


def _scene_and_tex():
    m = UnifiedMind(dim=512, seed=0)
    scene = m.build_scene("a big red metal sphere and a small blue box")
    tex = m.texture_op("mix", a=m.texture_leaf(value="red"), b=m.texture_leaf(value="cyan"),
                        t=m.texture_leaf("fbm", n_dims=2, seed=1))
    return m, scene, tex


def test_renders_shape_and_range():
    _, scene, tex = _scene_and_tex()
    img = render_textured(scene, {scene.names()[0]: tex}, width=80, height=64)
    assert img.shape == (64, 80, 3)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_texture_actually_wraps_the_sphere():
    """The painted sphere's colour must VARY across its surface (UV mapping), not be a flat tint."""
    m = UnifiedMind(dim=512, seed=0)
    scene = m.build_scene("a big red metal sphere")
    tex = m.texture_op("mix", a=m.texture_leaf(value="red"), b=m.texture_leaf(value="cyan"),
                        t=m.texture_leaf("fbm", n_dims=2, seed=1, octaves=5))
    from holographic_semantic import _UnionSDF
    from holographic_raymarch import sphere_trace
    from holographic_render import Camera
    W = H = 96
    img = render_textured(scene, {scene.names()[0]: tex}, width=W, height=H)
    union = _UnionSDF([r["sdf"] for r in scene.realize()])
    span = max(3.0, 1.6)
    cam = Camera(eye=(span * 0.4, span * 0.28, span), target=(0, 0, 0), fov_deg=42.0)
    eye, dirs = cam.ray_dirs(W, H)
    D = dirs.reshape(-1, 3); O = np.broadcast_to(eye, D.shape).copy()
    hit, _, _ = sphere_trace(union, O, D)
    sph = img.reshape(-1, 3)[hit]
    assert len(sph) > 100
    assert sph[:, 0].std() > 0.02 and sph[:, 2].std() > 0.02      # R and B vary across the surface


def test_no_texture_falls_back_to_scene_colours():
    _, scene, _ = _scene_and_tex()
    img = render_textured(scene, {}, width=64, height=48)         # no textures at all
    assert img.shape == (48, 64, 3) and img.std() > 0.02          # still a real render


def test_material_can_be_painted_too():
    from holographic_fpe import VectorFunctionEncoder
    from holographic_material import Material, texture_field
    m = UnifiedMind(dim=512, seed=0)
    scene = m.build_scene("a big red metal sphere")
    enc = VectorFunctionEncoder(2, dim=512, bounds=[(0, 1), (0, 1)], kernel="rbf", bandwidth=3.0, seed=1)
    grid = [(a, b) for a in np.linspace(0.05, 0.95, 6) for b in np.linspace(0.05, 0.95, 6)]
    mat = Material(enc, {"roughness": texture_field(enc, grid, [a for (a, b) in grid])})
    img = render_textured(scene, {scene.names()[0]: mat}, width=64, height=64)
    assert img.shape == (64, 64, 3)


def test_uv_helpers():
    # a point on the +z axis of a unit sphere maps near u=0.5 (front), v=0.5 (equator)
    local = np.array([[0.0, 0.0, 1.0]])
    u, v = _sphere_uv(local, 1.0)
    assert abs(u[0] - 0.5) < 1e-6 and abs(v[0] - 0.5) < 1e-6
    # box face UV stays in [0,1]
    u2, v2 = _box_uv(np.array([[0.3, 0.5, -0.2]]), np.array([0.5, 0.5, 0.5]))
    assert 0 <= u2[0] <= 1 and 0 <= v2[0] <= 1


def _sphere_scene():
    from holographic_unified import UnifiedMind
    m = UnifiedMind(dim=512, seed=0)
    tex = m.texture_op("mix", a=m.texture_leaf(value="red"), b=m.texture_leaf(value="cyan"),
                       t=m.texture_leaf("fbm", n_dims=2, seed=1))
    sc = m.build_scene("a big sphere")
    sc.paint(sc.names()[0], tex)
    return sc, tex


def test_aspect_ratio_sphere_is_round_at_non_square_resolution():
    """A sphere rendered into a 4:3 frame must be ROUND, not stretched -- the aspect-correction bug fix. We render the
    same sphere at a wide frame and a square frame with the SAME height; a correct camera gives the sphere the SAME
    pixel width in both (its size follows the vertical FOV, not the frame width)."""
    import numpy as np
    from holographic_texturerender import render_textured
    sc, tex = _sphere_scene()

    def sphere_width(img):
        red = (img[..., 0] > 0.25) & (img[..., 0] - img[..., 2] > 0.05)   # the reddish sphere vs the sky
        xs = np.where(red.any(axis=0))[0]
        return int(xs.max() - xs.min()) if len(xs) else 0

    wide = np.asarray(render_textured(sc, {sc.names()[0]: tex}, width=240, height=180, aa=1))
    square = np.asarray(render_textured(sc, {sc.names()[0]: tex}, width=180, height=180, aa=1))
    w_wide, w_square = sphere_width(wide), sphere_width(square)
    # if the aspect were wrong (stretched by width), the wide frame's sphere would be ~1.33x wider; correct -> ~equal
    assert abs(w_wide - w_square) <= 3, (w_wide, w_square)


def test_anti_aliasing_smooths_the_silhouette():
    """aa=2 (default) must produce more partial-blend edge pixels along the silhouette than aa=1 (off)."""
    import numpy as np
    from holographic_texturerender import render_textured
    sc, tex = _sphere_scene()

    def soft_edges(img):
        g = img.mean(-1); gx = np.abs(np.diff(g, axis=1))
        return int(((gx > 0.03) & (gx < 0.25)).sum())

    off = np.asarray(render_textured(sc, {sc.names()[0]: tex}, width=120, height=90, aa=1))
    on = np.asarray(render_textured(sc, {sc.names()[0]: tex}, width=120, height=90, aa=2))
    assert soft_edges(on) > soft_edges(off)


def test_render_textured_returns_requested_resolution():
    """The saved image must match the requested resolution exactly, at any aa."""
    import numpy as np
    from holographic_texturerender import render_textured
    sc, tex = _sphere_scene()
    for aa in (1, 2, 3):
        img = np.asarray(render_textured(sc, {sc.names()[0]: tex}, width=96, height=72, aa=aa))
        assert img.shape == (72, 96, 3)
