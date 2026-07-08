"""Tests for holographic_rayindex: bidirectional ray<->object lookup and bounded, bit-exact delta re-shading."""
import numpy as np
from holographic.simulation_and_physics.holographic_semantic import parse_description, render_scene, _scene_setup, _shade_rays
from holographic.rendering.holographic_rayindex import build_ray_index, delta_reshade
from holographic.rendering.holographic_render import Camera


def _through_glass_scene():
    objs = parse_description("a glass ball beside a red ball")["objects"]
    cam = Camera(eye=(5.6, 0.7, 0.0), target=(0, 0.0, 0), fov_deg=40.0)   # beyond the glass, looking through it
    ctx = _scene_setup(objs, True, "clear", "bright", (0.75, 0.9, 0.85))
    return objs, cam, ctx


def test_index_catches_through_glass_pixels():
    """The index flags pixels that see the red ball THROUGH the glass -- where the glass is the primary hit."""
    objs, cam, ctx = _through_glass_scene()
    index = build_ray_index(ctx, cam, 96, 96)
    indirect = index.indirect_pixels(1)                          # red ball = id 1, seen through glass
    assert indirect.sum() > 0                                    # there ARE through-glass pixels
    assert np.all(index.primary[indirect] != 1)                 # and the primary hit there is NOT the ball (it's glass)


def test_delta_reshade_is_bit_exact_and_bounded():
    """Re-shading only the touched pixels equals a full re-render exactly, and touches a fraction of the frame."""
    objs, cam, ctx = _through_glass_scene()
    W = H = 96
    base = render_scene(objs, cam, width=W, height=H, ss=1, dither=0.0)
    index = build_ray_index(ctx, cam, W, H)
    objs2 = [dict(o) for o in objs]; objs2[1] = dict(objs2[1]); objs2[1]["color"] = "blue"
    ctx2 = _scene_setup(objs2, True, "clear", "bright", (0.75, 0.9, 0.85))
    updated, mask = delta_reshade(ctx2, index, [1], base, cam)
    full = render_scene(objs2, cam, width=W, height=H, ss=1, dither=0.0)
    assert np.abs(updated - full).max() < 1e-9                   # bit-exact vs full re-render
    assert 0.0 < mask.mean() < 0.9                              # bounded: not the whole frame


def test_primary_only_would_miss_indirect():
    """A primary-id-only update (the old incremental path) misses the through-glass pixels the index catches."""
    objs, cam, ctx = _through_glass_scene()
    W = H = 96
    base = render_scene(objs, cam, width=W, height=H, ss=1, dither=0.0)
    index = build_ray_index(ctx, cam, W, H)
    objs2 = [dict(o) for o in objs]; objs2[1] = dict(objs2[1]); objs2[1]["color"] = "blue"
    full = render_scene(objs2, cam, width=W, height=H, ss=1, dither=0.0)
    truth_changed = np.abs(full - base).reshape(-1, 3).max(1) > 1e-6
    primary_only = (index.primary == 1)                          # what the old incremental renderer would refresh
    missed = truth_changed & ~primary_only
    assert missed.sum() > 0                                      # the old path misses real changes
    assert index.pixels_touching(1)[missed].all()               # the index covers every missed pixel


def test_pixels_touching_empty_for_unused_id():
    objs, cam, ctx = _through_glass_scene()
    index = build_ray_index(ctx, cam, 48, 48)
    assert not index.pixels_touching(999).any()                 # an object no ray touched -> no pixels


def test_index_catches_mirror_reflected_object():
    """A mirror reflects other objects (real shader), and the index records the reflected hit so an edit to the
    reflected object updates the mirror pixels -- bit-exact, bounded."""
    import numpy as np
    objs = parse_description("a big mirror box beside a red ball")["objects"]
    cam = Camera(eye=(-2.2, 1.1, 4.2), target=(1.2, 0.2, -0.3), fov_deg=48.0)
    W = H = 96
    base = render_scene(objs, cam, width=W, height=H, ss=1, dither=0.0)
    ctx = _scene_setup(objs, True, "clear", "bright", (0.75, 0.9, 0.85))
    index = build_ray_index(ctx, cam, W, H)
    assert index.indirect_pixels(1).sum() > 0                   # the ball IS reflected in the mirror
    objs2 = [dict(o) for o in objs]; objs2[1] = dict(objs2[1]); objs2[1]["color"] = "green"
    ctx2 = _scene_setup(objs2, True, "clear", "bright", (0.75, 0.9, 0.85))
    updated, mask = delta_reshade(ctx2, index, [1], base, cam)
    full = render_scene(objs2, cam, width=W, height=H, ss=1, dither=0.0)
    assert np.abs(updated - full).max() < 1e-9                  # reflection update is bit-exact
    # the reflected pixels would be missed by a primary-id-only path
    truth = np.abs(full - base).reshape(-1, 3).max(1) > 1e-6
    assert (index.indirect_pixels(1) & truth).sum() > 0


def test_brick_index_move_is_bit_exact_and_covers_changes():
    """A MOVE re-shades only rays through the vacated/occupied bricks, bit-exact, covering every changed pixel."""
    import numpy as np
    from holographic.rendering.holographic_rayindex import build_brick_index, delta_reshade_move
    objs = parse_description("a red ball beside a blue box beside a green ball")["objects"]
    cam = Camera(eye=(0.3, 1.6, 5.4), target=(0, 0.2, 0), fov_deg=48.0)
    W = H = 96
    ctx = _scene_setup(objs, True, "clear", "bright", (0.75, 0.9, 0.85))

    def flat(c):
        eye, dirs = cam.ray_dirs(W, H); D = dirs.reshape(-1, 3)
        O = np.broadcast_to(eye, D.shape).astype(float).copy()
        return _shade_rays(c, O, D)[0].reshape(H, W, 3)
    base = flat(ctx)
    bidx = build_brick_index(ctx, cam, W, H, grid=12, samples=14)
    updated, mask, ctxn = delta_reshade_move(ctx, 0, (0.0, 0.9, -0.6), bidx, base, cam)
    full = flat(ctxn)
    assert np.abs(updated - full).max() < 1e-9                  # bit-exact
    changed = np.abs(full - base).reshape(-1, 3).max(1) > 1e-6
    assert mask.reshape(-1)[changed].all()                     # every changed pixel is covered
    assert 0.0 < mask.mean() < 0.9                             # and it is bounded


def test_brick_move_updates_through_glass_object():
    """Moving an object seen THROUGH glass updates the through-glass pixels (the secondary-ray move case), bit-exact,
    and is strictly necessary: without the secondary test those pixels are missed."""
    import numpy as np
    from holographic.rendering.holographic_rayindex import build_brick_index, delta_reshade_move, _object_aabb
    objs = parse_description("a glass ball beside a red ball")["objects"]
    cam = Camera(eye=(5.6, 0.7, 0.0), target=(0, 0.0, 0), fov_deg=40.0)
    W = H = 110
    ctx = _scene_setup(objs, True, "clear", "bright", (0.75, 0.9, 0.85))

    def flat(c):
        eye, dirs = cam.ray_dirs(W, H); D = dirs.reshape(-1, 3)
        O = np.broadcast_to(eye, D.shape).astype(float).copy()
        return _shade_rays(c, O, D)[0].reshape(H, W, 3)
    base = flat(ctx)
    bidx = build_brick_index(ctx, cam, W, H, grid=14)
    upd, mask, ctxn = delta_reshade_move(ctx, 1, (0.0, 0.5, 0.0), bidx, base, cam)
    full = flat(ctxn)
    changed = np.abs(full - base).reshape(-1, 3).max(1) > 1e-6
    assert np.abs(upd - full).max() < 1e-9                      # bit-exact
    assert mask.reshape(-1)[changed].all()                     # covers every changed pixel
    ab = [_object_aabb(ctx["sdfs"][1], 0.6), _object_aabb(ctxn["sdfs"][1], 0.6)]
    missed_without_secondary = changed & ~bidx.pixels_through_region(ab, secondary=False)
    assert missed_without_secondary.sum() > 0                   # the secondary test is REQUIRED here


def test_sss_material_edit_bit_exact():
    """A material edit to a subsurface (wax) object is a bit-exact bounded delta -- the SSS term re-shades with it."""
    import numpy as np
    objs = parse_description("a wax ball beside a red ball")["objects"]
    cam = Camera(eye=(0.3, 1.0, 4.4), target=(0, 0.1, 0), fov_deg=44.0)
    W = H = 90
    ctx = _scene_setup(objs, True, "clear", "bright", (0.75, 0.9, 0.85))
    assert ctx["is_sss"][0]                                     # the wax ball is flagged SSS
    base = render_scene(objs, cam, width=W, height=H, ss=1, dither=0.0)
    idx = build_ray_index(ctx, cam, W, H)
    o2 = [dict(o) for o in objs]; o2[0] = dict(o2[0]); o2[0]["color"] = "green"
    ctx2 = _scene_setup(o2, True, "clear", "bright", (0.75, 0.9, 0.85))
    upd, mask = delta_reshade(ctx2, idx, [0], base, cam)
    full = render_scene(o2, cam, width=W, height=H, ss=1, dither=0.0)
    assert np.abs(upd - full).max() < 1e-9


def test_translucent_object_behind_is_indexed():
    """An object seen THROUGH a translucent (frosted) object gets the see-through secondary in the index, so editing
    it updates the translucent pixels -- bit-exact."""
    import numpy as np
    objs = parse_description("a translucent ball beside a red ball")["objects"]
    cam = Camera(eye=(5.6, 0.7, 0.0), target=(0, 0.0, 0), fov_deg=40.0)
    W = H = 90
    ctx = _scene_setup(objs, True, "clear", "bright", (0.75, 0.9, 0.85))
    assert ctx["is_translucent"][0]
    base = render_scene(objs, cam, width=W, height=H, ss=1, dither=0.0)
    idx = build_ray_index(ctx, cam, W, H)
    assert idx.indirect_pixels(1).sum() > 0                     # the red ball shows through the frosted ball
    o2 = [dict(o) for o in objs]; o2[1] = dict(o2[1]); o2[1]["color"] = "blue"
    ctx2 = _scene_setup(o2, True, "clear", "bright", (0.75, 0.9, 0.85))
    upd, mask = delta_reshade(ctx2, idx, [1], base, cam)
    full = render_scene(o2, cam, width=W, height=H, ss=1, dither=0.0)
    assert np.abs(upd - full).max() < 1e-9


def test_incremental_renderer_unchanged_is_free_and_edit_is_delta():
    """The session serves an unchanged re-render for free (empty mask) and an edit as a bit-exact bounded delta."""
    import numpy as np
    from holographic.rendering.holographic_rayindex import IncrementalRenderer
    from holographic.simulation_and_physics.holographic_semantic import render_scene
    objs = parse_description("a red ball beside a blue box")["objects"]
    cam = Camera(eye=(0.4, 1.5, 5.0), target=(0, 0.1, 0), fov_deg=46.0)
    W = H = 96
    r = IncrementalRenderer(cam, W, H, ss=1)
    f0, m0 = r.render(objs)
    assert m0.all()                                            # first render: whole frame is new
    f1, m1 = r.render(objs)                                    # SAME scene
    assert not m1.any()                                        # nothing changed -> free, empty delta
    assert f1 is f0                                            # and the cached frame is returned as-is
    f2, m2 = r.edit(0, "color", "yellow")                     # colour edit -> bounded delta
    assert 0 < m2.sum() < W * H
    full = render_scene([dict(objs[0], color="yellow"), objs[1]], cam, width=W, height=H, ss=1, dither=0.0)
    assert np.abs(f2 - full).max() < 1e-9                      # bit-exact vs a full re-render
    ys, xs, rgb = r.stream_delta(m2)
    assert len(ys) == int(m2.sum()) and rgb.shape == (len(ys), 3)   # stream carries only the changed pixels


def test_reproject_camera_move_reuses_most_pixels():
    """A camera move reprojects the cached frame's world hits and re-shades only holes + view-dependent pixels --
    faster than a full re-render, close to it in quality, and exact on the re-shaded pixels."""
    import numpy as np, math
    from holographic.rendering.holographic_rayindex import IncrementalRenderer
    from holographic.simulation_and_physics.holographic_semantic import render_scene
    objs = parse_description("a big red ball beside a big blue box")["objects"]
    base = Camera(eye=(0.2, 0.6, 3.2), target=(0, 0.1, 0), fov_deg=52.0)
    W = H = 96
    s = IncrementalRenderer(base, W, H, ss=1)
    s.render(objs)
    a = math.radians(5.0); ex = 0.2 * math.cos(a) + 3.2 * math.sin(a); ez = -0.2 * math.sin(a) + 3.2 * math.cos(a)
    newcam = Camera(eye=(ex, 0.6, ez), target=(0, 0.1, 0), fov_deg=52.0)
    fr, rmask = s.reproject(newcam)
    full = render_scene(objs, newcam, width=W, height=H, ss=1, dither=0.0)
    assert rmask.mean() < 0.6                                   # most of the frame was REUSED, not re-shaded
    assert np.abs(fr[rmask] - full.reshape(H, W, 3)[rmask]).max() < 1e-9   # re-shaded pixels are exact
    mse = np.mean((fr - full) ** 2)
    assert mse < 5e-3                                           # reprojected frame is close to a full re-render
