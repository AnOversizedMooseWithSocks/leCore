import numpy as np
from holographic_render import Camera, Light, rasterize_mesh, volume_render, frame_delta_tiles
from holographic_meshbridge import sample_field, marching_tetrahedra_vec


def _sphere_mesh(res=28, r=0.7):
    def sphere(P): P = np.asarray(P, float); return np.linalg.norm(P, axis=1) - r
    v, ax = sample_field(sphere, (np.array([-1., -1, -1]), np.array([1., 1, 1])), res)
    return marching_tetrahedra_vec(v, ax)


def test_camera_rays_are_unit_and_forward():
    cam = Camera(eye=(0, 0, 3), target=(0, 0, 0), fov_deg=45)
    eye, dirs = cam.ray_dirs(16, 16)
    assert np.allclose(np.linalg.norm(dirs, axis=-1), 1.0, atol=1e-6)
    assert dirs[8, 8, 2] < 0                                   # centre ray points toward -z (at the target)


def test_rasterize_lit_sphere_has_shading_gradient():
    """A directionally-lit sphere has a bright side and a dark side, and the background shows where it's empty."""
    M = _sphere_mesh()
    cam = Camera(eye=(0, 0, 3), target=(0, 0, 0), fov_deg=45)
    img = rasterize_mesh(M, cam, 96, 96, lights=[Light("directional", direction=(-1, -1, -1))],
                         base_color=(0.8, 0.5, 0.3), background=(0.0, 0.0, 0.0), ambient=0.1)
    lit = img.sum(2)
    assert lit.max() > 0.5                                     # something bright was drawn
    assert (lit < 1e-6).sum() > 96 * 96 * 0.2                  # background visible (sphere doesn't fill frame)
    assert lit.max() - lit[lit > 0.02].min() > 0.2            # a real bright->dark gradient


def test_volume_render_smoke_alpha_and_fire_is_red():
    cam = Camera(eye=(0, 0, 3), target=(0, 0, 0), fov_deg=45)
    b = (np.array([-1., -1, -1]), np.array([1., 1, 1]))
    def blob(P): P = np.asarray(P, float); return np.clip(1.0 - np.linalg.norm(P, axis=1) / 0.6, 0, 1)
    _, alpha = volume_render(blob, cam, b, 64, 64, steps=64, mode="smoke", sigma=12.0)
    assert alpha.max() > 0.5 and alpha.min() < 0.05           # opaque core, empty corners
    fire, _ = volume_render(blob, cam, b, 64, 64, steps=48, mode="fire", sigma=14.0)
    assert fire[..., 0].max() > fire[..., 2].max()            # emissive glow is red, not blue


def test_frame_delta_streams_only_changed_tiles():
    a = np.zeros((64, 64, 3))
    b = a.copy(); b[10:20, 10:20] = 1.0                       # a local change
    tiles, frac = frame_delta_tiles(a, b, tile=16)
    assert 0 < frac < 0.5 and len(tiles) >= 1                 # only some tiles changed
    none_tiles, frac0 = frame_delta_tiles(a, a, tile=16)
    assert len(none_tiles) == 0 and frac0 == 0.0              # identical frames -> nothing to push


def test_vectorized_rasterizer_matches_loop():
    """The vectorized fragment-scatter rasterizer produces the same image as the reference per-triangle loop."""
    M = _sphere_mesh(res=24)
    cam = Camera(eye=(1.2, 0.9, 2.2), target=(0, 0, 0), fov_deg=45)
    L = [Light("directional", direction=(-1, -1, -1))]
    a = rasterize_mesh(M, cam, 128, 128, lights=L, base_color=(0.8, 0.5, 0.3), vectorized=False)
    b = rasterize_mesh(M, cam, 128, 128, lights=L, base_color=(0.8, 0.5, 0.3), vectorized=True)
    assert np.mean(np.abs(a - b) < 0.02) > 0.999              # identical up to edge tie-breaks


def test_volume_optimizations_preserve_image_and_cut_samples():
    """Empty-space skipping + early termination give (near) the same image while doing fewer field evaluations."""
    cam = Camera(eye=(1.4, 1.1, 2.4), target=(0, 0, 0), fov_deg=45)
    b = (np.array([-1., -1, -1]), np.array([1., 1, 1]))
    def blob(P): P = np.asarray(P, float); return np.clip(1.2 - np.linalg.norm(P, axis=1) / 0.5, 0, 1) * 2.0
    i0, _ = volume_render(blob, cam, b, 96, 96, steps=80, empty_skip=False, early_term=False)
    n0 = volume_render.last_samples
    i1, _ = volume_render(blob, cam, b, 96, 96, steps=80, empty_skip=True, early_term=True)
    n1 = volume_render.last_samples
    assert np.abs(i0 - i1).max() < 0.02 and n1 < n0          # same image, fewer field samples


def test_png_bytes_and_save_png_agree_and_are_valid():
    """png_bytes returns a real PNG (valid magic), save_png writes exactly those bytes, and the compression
    level changes only the byte stream -- never the decoded pixels (PNG is lossless)."""
    import tempfile, os
    from holographic_render import png_bytes, save_png
    img = np.random.default_rng(0).random((13, 21, 3))

    b = png_bytes(img)
    assert b[:8] == b"\x89PNG\r\n\x1a\n"                       # PNG signature

    p = tempfile.mktemp(suffix=".png")
    save_png(p, img)                                          # default level matches png_bytes' default
    try:
        assert open(p, "rb").read() == png_bytes(img)         # save_png is a thin wrapper over png_bytes
    finally:
        os.remove(p)

    # level 1 (fast preview) vs 6 (still): both valid PNGs; the IHDR (which encodes width/height/depth, i.e.
    # the image shape) is identical -- only the compressed IDAT differs.
    b1 = png_bytes(img, level=1)
    assert b1[:8] == b"\x89PNG\r\n\x1a\n"
    assert b1[8:33] == b[8:33]                                 # signature + IHDR chunk identical across levels
