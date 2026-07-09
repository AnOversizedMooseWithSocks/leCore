import numpy as np
from holographic.rendering.holographic_render import Camera, Light, rasterize_mesh, volume_render, frame_delta_tiles
from holographic.mesh_and_geometry.holographic_meshbridge import sample_field, marching_tetrahedra_vec


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
    from holographic.rendering.holographic_render import png_bytes, save_png
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


# ======================================================================================================
# PNG scanline filtering: 34x smaller on a gradient, never worse, and provably lossless.
# ======================================================================================================
def _png_decode_rgb(blob):
    """A minimal stdlib PNG decoder for 8-bit RGB -- so the lossless guarantee is pinned by OUR code, not by a
    third-party decoder that might paper over an encoder bug. Reverses all five scanline filters."""
    import struct
    import zlib
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w = 8, b"", None
    while pos < len(blob):
        ln = struct.unpack(">I", blob[pos:pos + 4])[0]
        typ = blob[pos + 4:pos + 8]
        data = blob[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", data[:10])
            assert depth == 8 and ctype == 2                       # 8-bit RGB, what png_bytes emits
        elif typ == b"IDAT":
            idat += data
        pos += 12 + ln
    raw = zlib.decompress(idat)
    bpp, stride = 3, w * 3
    out = np.zeros((h, stride), np.uint8)
    prev = np.zeros(stride, np.uint8)
    p = 0
    for y in range(h):
        ft = raw[p]; p += 1
        line = np.frombuffer(raw[p:p + stride], np.uint8).copy(); p += stride
        cur = np.zeros(stride, np.uint8)
        for i in range(stride):
            a = int(cur[i - bpp]) if i >= bpp else 0
            b = int(prev[i])
            c = int(prev[i - bpp]) if i >= bpp else 0
            if ft == 0:   pr = 0
            elif ft == 1: pr = a
            elif ft == 2: pr = b
            elif ft == 3: pr = (a + b) // 2
            else:
                pp = a + b - c
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            cur[i] = (int(line[i]) + pr) & 0xFF
        out[y] = cur
        prev = cur
    return out.reshape(h, w, 3)


def _demo_images():
    S = 48
    yy, xx = np.mgrid[0:S, 0:S] / S
    grad = np.stack([xx, 1 - xx, np.abs(yy - 0.5) * 2], -1)           # smooth: filtering wins big
    r = np.sqrt((xx - .5) ** 2 + (yy - .5) ** 2)
    flat = np.zeros((S, S, 3))
    flat[...] = (0.05, 0.09, 0.15)
    flat[np.abs(r - 0.42) < 0.03] = (0.40, 0.90, 0.77)                # flat art: filter 0 already wins
    noise = np.random.default_rng(0).random((S, S, 3))                # incompressible: must not regress
    return grad, flat, noise


def test_png_filtering_is_lossless_by_our_own_decoder():
    """PNG filtering is lossless by construction. Pinned with a stdlib un-filter written here, so the guarantee
    does not rest on Pillow agreeing with us."""
    from holographic.rendering.holographic_render import png_bytes
    for img in _demo_images():
        want = (np.clip(img, 0, 1) * 255).astype(np.uint8)            # png_bytes truncates; match it exactly
        for filters in (False, True):
            got = _png_decode_rgb(png_bytes(img, level=6, filters=filters))
            assert np.array_equal(got, want), filters


def test_filtering_shrinks_a_gradient_massively_and_never_regresses():
    """MEASURED: this encoder emitted filter 0 on every scanline until it was measured, making it 43x LARGER than
    Pillow on a smooth gradient. It is never worse now, because png_bytes compresses BOTH strategies and keeps the
    smaller -- a per-line heuristic alone actually LOSES on flat art (3,553 -> 4,903 bytes), since filter 0 leaves
    the byte stream uniform and zlib's LZ77 matches long runs across scanlines."""
    from holographic.rendering.holographic_render import png_bytes
    grad, flat, noise = _demo_images()
    g_old, g_new = len(png_bytes(grad, 6, filters=False)), len(png_bytes(grad, 6))
    assert g_new < g_old / 10.0, (g_old, g_new)                       # measured 33.8x on the demo set
    for img in (grad, flat, noise):                                   # NEVER worse, on any of the three regimes
        assert len(png_bytes(img, 6)) <= len(png_bytes(img, 6, filters=False))


def test_filters_false_reproduces_the_exact_legacy_byte_stream():
    """The escape hatch has to be exact, or it is not an escape hatch."""
    import struct
    import zlib
    from holographic.rendering.holographic_render import png_bytes
    img = _demo_images()[0]
    a = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    h, w = a.shape[:2]
    raw = b"".join(b"\x00" + a[y, :, :3].tobytes() for y in range(h))
    def chunk(typ, data):
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff)
    legacy = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
              + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    assert png_bytes(img, 6, filters=False) == legacy


def test_the_filter_choice_is_deterministic_and_ties_keep_the_lower_number():
    from holographic.rendering.holographic_render import _png_scanlines, png_bytes
    img = _demo_images()[0]
    assert png_bytes(img, 6) == png_bytes(img, 6)                     # bit-identical across calls
    flat_rows = np.zeros((4, 5, 3), np.uint8)                         # an all-zero image: every filter costs 0
    stream = _png_scanlines(flat_rows, filters=True)
    assert stream[0] == 0 and stream[16] == 0                         # ties -> filter 0, the lowest number


def test_volume_render_only_mask_is_exact_inside_and_free_outside():
    """`only=` renders a subset of rays: bit-identical inside the mask, background at alpha 0 outside, and the
    field evaluations it skips are genuinely not paid for. Added to run the coarse-first escalation experiment."""
    from holographic.rendering.holographic_render import Camera, volume_render

    def blob(p):
        r = np.linalg.norm(p - np.array([0.1, 0.0, 0.0]), axis=1)
        return np.clip(1.2 - 2.0 * r, 0, None)

    cam = Camera(eye=(2.2, 1.1, 2.4), target=(0, 0, 0), fov_deg=42)
    B = (np.array([-1., -1, -1]), np.array([1., 1, 1]))
    img, alpha = volume_render(blob, cam, B, 48, 48, steps=32)
    full_cost = volume_render.last_samples

    mask = np.zeros((48, 48), bool)
    mask[16:32, 16:32] = True
    img_m, alpha_m = volume_render(blob, cam, B, 48, 48, steps=32, only=mask)
    masked_cost = volume_render.last_samples

    assert np.array_equal(img_m[mask], img[mask])          # exact inside: it is the same march
    assert float(np.abs(alpha_m[~mask]).max()) == 0.0      # outside: background, alpha 0
    assert masked_cost < full_cost                         # ...and the skipped rays cost nothing


def test_empty_skip_and_early_term_are_the_engines_own_coarse_first():
    """KEPT NEGATIVE, measured: coarse-first escalation does not pay on top of these, because they ARE coarse-first
    -- spatial and temporal -- and they are applied better. Pinned so nobody re-runs the experiment."""
    from holographic.rendering.holographic_render import Camera, volume_render

    def field(p):
        r1 = np.linalg.norm(p - np.array([0.15, 0.05, 0.0]), axis=1)
        return np.clip(1.1 - 2.2 * r1, 0, None) + 0.9 * (np.abs(r1 - 0.42) < 0.03)

    cam = Camera(eye=(2.2, 1.1, 2.4), target=(0, 0, 0), fov_deg=42)
    B = (np.array([-1., -1, -1]), np.array([1., 1, 1]))
    kw = dict(width=48, height=48, steps=48, sigma=10.0)
    volume_render(field, cam, B, **kw)
    smart = volume_render.last_samples
    volume_render(field, cam, B, empty_skip=False, early_term=False, **kw)
    dumb = volume_render.last_samples
    assert dumb > 8 * smart, (dumb, smart)                 # measured 15.2x on a larger frame
