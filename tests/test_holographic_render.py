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


# ---------------------------------------------------------------------------
# PNG READ-BACK -- the direction the engine never had (J-3D-03/04)
# ---------------------------------------------------------------------------

def test_png_round_trips_to_one_eight_bit_step():
    """save_png -> load_png must return the image, to the tolerance the FILE FORMAT allows and no further.

    Asserting equality here would be asserting something false about PNG: save_png quantises to 8 bits. A
    test that demanded exactness would fail for the wrong reason and teach the next reader the wrong thing."""
    import tempfile
    import numpy as np
    from holographic.rendering.holographic_render import save_png, load_png
    rng = np.random.default_rng(0)
    x = rng.random((17, 23, 3))
    with tempfile.TemporaryDirectory() as d:
        p = d + "/rt.png"
        save_png(p, x)
        y = load_png(p)
        assert y.shape == x.shape
        assert np.abs(x - y).max() <= 1.0 / 255.0 + 1e-9


def test_decoder_handles_the_adaptive_filters_the_encoder_actually_picks():
    """MEASURED on a 192x144 path-traced frame: 114 of 144 rows chose Paeth, 27 Up, 3 Sub. A decoder that
    only did None/Up would pass a random-noise test and fail on every real render, so the fixture here is a
    SMOOTH gradient -- the case that makes the encoder reach for the expensive filters."""
    import tempfile
    import numpy as np
    from holographic.rendering.holographic_render import save_png, load_png
    grad = np.clip(np.mgrid[0:48, 0:48][0][..., None] / 47.0 * np.ones(3), 0, 1)
    with tempfile.TemporaryDirectory() as d:
        p = d + "/grad.png"
        save_png(p, grad)
        assert np.abs(grad - load_png(p)).max() <= 1.0 / 255.0 + 1e-9
        save_png(p, grad, filters=False)                  # and the legacy unfiltered stream
        assert np.abs(grad - load_png(p)).max() <= 1.0 / 255.0 + 1e-9


def test_unsupported_png_refuses_instead_of_returning_garbage():
    """A quietly wrong decode is worse than a loud refusal: nothing downstream can tell a scrambled image
    from a real one, so the failure would surface as a mysterious render diff hours later."""
    from holographic.rendering.holographic_render import png_decode
    for bad, expect in ((b"not a png at all", "signature"), (b"\x89PNG\r\n\x1a\n", "IHDR")):
        try:
            png_decode(bad)
            raise AssertionError("expected a refusal for %r" % bad[:12])
        except ValueError as exc:
            assert expect in str(exc)


def test_compare_image_files_needs_no_pillow():
    """The core promises NumPy/Flask/stdlib/hashlib. This faculty -- the one whose own docstring calls it the
    check an agent runs after a render -- used to hard-import Pillow and raise ImportError on a clean install.
    Pinned by BLOCKING the import, because 'it works on a machine that happens to have PIL' is not the claim."""
    import builtins
    import lecore
    import numpy as np
    import tempfile
    real_import = builtins.__import__

    def no_pil(name, *a, **kw):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("PIL blocked by the test -- core must not need it")
        return real_import(name, *a, **kw)

    m = lecore.UnifiedMind(dim=128, seed=0)
    with tempfile.TemporaryDirectory() as d:
        a, b = d + "/a.png", d + "/b.png"
        # NOT a flat image on purpose -- see test_edge_agreement_is_degenerate_on_constant_gradients.
        tex = np.clip(np.mgrid[0:16, 0:16][0][..., None] / 15.0 * np.ones(3), 0, 1)
        tex[4:9, 4:9] = 0.9                              # a block, so the gradient map is not constant
        m.save_render(a, tex)
        m.save_render(b, tex)
        builtins.__import__ = no_pil
        try:
            r = m.compare_image_files(a, b)
        finally:
            builtins.__import__ = real_import
    assert r["similarity"] > 0.999, "two identical images must score ~1.0"


def test_see_then_fix_loop_closes_through_the_mind():
    """The whole point of the item: render -> save -> LOOK -> compare, with nothing imported past lecore.

    Before this, 'look' had nowhere to start and the loop could not be written at all."""
    import lecore
    import numpy as np
    import tempfile
    m = lecore.UnifiedMind(dim=128, seed=0)
    with tempfile.TemporaryDirectory() as d:
        p = d + "/frame.png"
        frame = np.clip(np.mgrid[0:24, 0:24][1][..., None] / 23.0 * np.ones(3), 0, 1)
        m.save_render(p, frame)
        back = m.load_image(p)                            # the step that did not exist
        assert back.shape == frame.shape
        assert np.abs(frame - back).max() <= 1.0 / 255.0 + 1e-9
        assert "Read a render back" in str(m.find_capability("look at my own render")[0])


def test_edge_agreement_is_degenerate_on_constant_gradients():
    """KNOWN DEFECT, pinned rather than fixed -- found by a test fixture that used a flat image.

    `edge_agreement` correlates the two gradient-MAGNITUDE maps after subtracting their means. An image whose
    gradient magnitude is CONSTANT -- a flat colour, or a linear ramp -- leaves both vectors identically zero,
    so the correlation is 0/0, falls through the 1e-12 guard, and returns 0.5. An image compared WITH ITSELF
    then scores 0.9 instead of 1.0 at the default weights (w_edge=0.2).

    MEASURED: constant at 0.5 across 8x8 through 128x128, so it is not a small-image artefact. Real renders
    are unaffected (a 192x144 path-traced frame self-scores 0.99999) because their gradient maps vary.

    NOT FIXED HERE, deliberately. perceptual_distance is what the analysis-by-synthesis loop MINIMISES, so
    changing this changes an optimiser's landscape and every score it has ever recorded -- that is exactly
    the kind of existing decision this repo does not flip inside an unrelated item. Filed as J-3D-23. When it
    IS fixed (identity must be 1.0: if both gradient maps are constant, they agree perfectly), this test
    should fail -- update it, do not relax it."""
    import numpy as np
    from holographic.io_and_interop import holographic_imagecompare as IC
    ramp = np.clip(np.mgrid[0:32, 0:32][1][..., None] / 31.0 * np.ones(3), 0, 1)
    assert abs(IC.ms_ssim(ramp, ramp) - 1.0) < 1e-9, "structure term is fine"
    assert abs(IC.color_agreement(ramp, ramp) - 1.0) < 1e-9, "colour term is fine"
    assert abs(IC.edge_agreement(ramp, ramp) - 0.5) < 1e-9, "the edge term is the degenerate one"
    assert abs(IC.perceptual_similarity(ramp, ramp) - 0.9) < 1e-9, "identity scores 0.9, not 1.0"
    # and the case that matters in practice is NOT affected
    varied = ramp.copy(); varied[8:16, 8:16] = 0.9
    assert IC.perceptual_similarity(varied, varied) > 0.999, "a non-degenerate image self-scores ~1.0"


# ---------------------------------------------------------------------------
# J-3D-19: Radiance .hdr (RGBE) -- the missing piece of image-based lighting.
# ---------------------------------------------------------------------------

def _rgbe(rgb):
    import numpy as np
    m = rgb.max(axis=-1)
    e = np.where(m <= 1e-32, 0, np.floor(np.log2(np.maximum(m, 1e-32))) + 129).astype(np.int32)
    s = np.where(e == 0, 0.0, np.ldexp(1.0, -(e - 128 - 8)))
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.clip(rgb * s[..., None], 0, 255).astype(np.uint8)
    out[..., 3] = e.astype(np.uint8)
    return out


def _write_hdr(path, rgb, rle=False):
    import numpy as np
    h, w = rgb.shape[:2]
    px = _rgbe(rgb)
    head = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y %d +X %d\n" % (h, w)
    if not rle:
        body = px.tobytes()
    else:
        body = b""
        for y in range(h):
            body += bytes([2, 2, (w >> 8) & 255, w & 255])
            for c in range(4):
                row = px[y, :, c].tobytes()
                i = 0
                while i < len(row):
                    n = min(128, len(row) - i)
                    body += bytes([n]) + row[i:i + n]
                    i += n
    with open(path, "wb") as f:
        f.write(head + body)


def test_hdr_preserves_dynamic_range(tmp_path):
    """THE assertion. A reader that got the pixels right and lost the RANGE would be worse than none: it
    would look like it worked while every render it lit was quietly lit by a flat sky. An 8-bit path would
    collapse this planted 2000x sun/sky ratio to about 2x, which is the whole reason load_png is not enough."""
    import numpy as np
    from holographic.rendering.holographic_render import load_hdr

    img = np.zeros((8, 16, 3))
    img[:, :8] = (0.4, 0.5, 0.7)
    img[2:4, 10:12] = (900.0, 850.0, 700.0)
    p = tmp_path / "sun.hdr"
    _write_hdr(str(p), img)
    a = load_hdr(str(p))
    assert a.shape == (8, 16, 3) and a.dtype.name == "float32"
    assert a.max() > 100.0, "the sun was clipped -- a bounded HDR reader defeats the purpose"
    assert a[2, 10, 0] / a[0, 0, 0] > 1000.0
    assert a[0, 12, 0] == 0.0, "exponent 0 must be exactly black, not a denormal smear"


def test_rle_and_flat_scanlines_agree(tmp_path):
    """Same pixels, two containers. Every HDRI a person downloads is adaptive-RLE, so a decoder that only
    handled flat scanlines would work on every file this repo writes and none that anyone actually uses."""
    import numpy as np
    from holographic.rendering.holographic_render import load_hdr

    rng = np.random.default_rng(0)
    img = rng.uniform(0, 6, (6, 24, 3))
    _write_hdr(str(tmp_path / "a.hdr"), img, rle=False)
    _write_hdr(str(tmp_path / "b.hdr"), img, rle=True)
    assert np.array_equal(load_hdr(str(tmp_path / "a.hdr")), load_hdr(str(tmp_path / "b.hdr")))


def test_wrong_formats_raise_rather_than_decode_wrongly(tmp_path):
    """KEPT NEGATIVE, asserted. XYZE files carry CIE XYZ primaries; returning them as RGB would silently
    shift every colour in the render -- the kind of wrong that looks plausible. Refusing is the correct
    answer, and so is refusing a PNG rather than reading its bytes as radiance."""
    import numpy as np
    import pytest
    from holographic.rendering.holographic_render import load_hdr, save_png

    save_png(str(tmp_path / "x.hdr"), np.zeros((4, 4, 3)))
    with pytest.raises(ValueError, match="RADIANCE"):
        load_hdr(str(tmp_path / "x.hdr"))
    with open(tmp_path / "xyze.hdr", "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_xyze\n\n-Y 2 +X 2\n" + b"\0" * 16)
    with pytest.raises(ValueError, match="XYZ"):
        load_hdr(str(tmp_path / "xyze.hdr"))


def test_an_hdri_actually_lights_a_scene_directionally(tmp_path):
    """Cross-faculty, and the point of the whole item: load_hdr -> sky_dome -> DomeLight -> a render.

    MEASURED and pinned as the reason this is a capability rather than a docs fix: a flat dome and a smooth
    procedural sky field differ by only 0.0054 mean abs, while the SAME env mirrored left/right differs by
    0.0336. Gradients do not pay; directional structure does. So the test mirrors an env and requires the
    image to change -- if it does not, the mapping is not oriented and the HDRI is just a tint."""
    import numpy as np
    import lecore

    m = lecore.UnifiedMind(dim=128, seed=0)
    env_img = np.zeros((16, 32, 3))
    env_img[:, :16] = (1.2, 1.1, 0.9)
    env_img[:, 16:] = (0.03, 0.03, 0.05)
    p = tmp_path / "half.hdr"
    _write_hdr(str(p), env_img)
    env = m.load_hdr(str(p))
    assert env.shape == (16, 32, 3)

    sc = m.new_scene()
    sc.add(name="ball", geometry=m.sdf_parse("(sphere 0.6)"), material="matte_gray")
    sc.add(name="floor", geometry=m.sdf_parse("(plane -0.7)"), material="matte_gray")
    cam = m.camera(eye=(0.0, 0.8, 2.6), target=(0.0, 0.0, 0.0), fov_deg=40.0, aspect=4 / 3.)
    dark = lambda d: np.broadcast_to(np.array([0.01, 0.01, 0.01]), (len(d), 3))

    def shot(e):
        L = [m.scene_light("dome", color=lambda d: m.sky_dome(d, env=e), intensity=1.0)]
        return np.asarray(m.render_scene_document(sc, cam, 32, 24, quality="fast", max_bounce=1,
                                                  seed=0, lights=L, sky=dark), float)

    left, right = shot(env), shot(env[:, ::-1].copy())
    assert np.abs(left - right).mean() > 1e-3, \
        "mirroring the environment changed nothing -- the map is a tint, not a light"
    assert "HDRI environment" in str(m.find_capability("image based lighting")[0])
