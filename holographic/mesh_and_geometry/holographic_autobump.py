"""holographic_autobump.py -- AUTO-BUMP: image -> height -> normal map -> material channel (inverse-rendering IR1).

When a material has an albedo texture but no bump/normal map, derive a PLAUSIBLE tangent-space normal map from the
image alone, so an "auto bump" toggle fills the channel that was already waiting. Pure classical arithmetic on the
`vision` front-end -- no learned prior:

  1. image -> HEIGHT: grayscale (vision.to_gray), then a HIGH-PASS (the image minus a Gaussian blur) so a slow
     brightness ramp across the photo -- baked lighting, large-scale albedo -- does NOT become a giant fake slope.
     Only the fine surface detail survives; brighter detail reads as raised.
  2. HEIGHT -> NORMAL: the surface is (x, y, h(x,y)); its tangent-space normal is
     normalize(-strength*dh/dx, -strength*dh/dy, 1) -- the standard grayscale-to-normal every DCC tool ships.
  3. feed the MATERIAL: the height (a scalar UV texture) drops straight into a Material `height` channel (which
     `displace` consumes for real relief, IR5); the normal map is the (H,W,3) artifact the renderer's normal
     channel consumes, and `octnormal` quantizes it compactly on its manifold.

HONEST (kept negative, LOUD): luminance-as-height is a HEURISTIC, not a measurement. Baked directional light
becomes fake grooves (a cast shadow reads as a crevice); albedo that isn't height (a painted stripe) becomes fake
relief (a ridge) -- a fundamental ambiguity no arithmetic resolves without more information (that is what IR2's
light-aware pass and IR8's photometric stereo are for, both still bounded). The `bump_confidence` gate measures
whether there is ENOUGH fine detail to derive a bump AT ALL (abstain -> flat when there isn't); it does NOT, and
cannot, verify the detail is relief rather than albedo. This is a plausible perceptual bump, not a depth map, and
`strength` is a user knob, not an inferred physical scale. NumPy + stdlib only; deterministic.
"""
import numpy as np

from holographic.misc.holographic_vision import to_gray
import importlib


def _gaussian_kernel1d(sigma):
    """A normalized 1-D Gaussian kernel out to ~3 sigma."""
    r = max(1, int(round(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(img, sigma=4.0):
    """A small, readable SEPARABLE Gaussian blur (reflect-padded at the borders) -- the low-pass we subtract to
    get the high-pass. Separable = two 1-D passes, O(N*r) not O(N*r^2)."""
    k = _gaussian_kernel1d(sigma)
    r = len(k) // 2

    def blur_axis(a, axis):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (r, r)
        ap = np.pad(a, pad, mode="reflect")
        out = np.zeros_like(a)
        for i, w in enumerate(k):                            # slide the kernel: sum w_i * shifted(a)
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(i, i + a.shape[axis])
            out += w * ap[tuple(sl)]
        return out

    a = np.asarray(img, float)
    return blur_axis(blur_axis(a, 0), 1)


def image_to_height(rgb, sigma=4.0):
    """Grayscale -> HIGH-PASS -> the height field (fine detail only, brighter = raised, centred near 0). The
    high-pass (gray - blur) removes the slow lighting/albedo component, which is exactly the part that would
    otherwise become a giant fake slope across the whole image."""
    a = np.asarray(rgb, float)
    gray = to_gray(a) if a.ndim == 3 else a
    return gray - gaussian_blur(gray, sigma)                 # the high-passed luminance IS the height


def normal_from_height(h, strength=2.0):
    """A tangent-space normal map (H, W, 3), unit-length, from a height field. The surface is (x, y, h); its
    normal is normalize(-strength*dh/dx, -strength*dh/dy, 1). z=1 keeps it a valid unit normal pointing out of
    the surface; `strength` tilts it toward the gradient (deeper-looking relief)."""
    h = np.asarray(h, float)
    gy, gx = np.gradient(h)                                  # np.gradient returns [d/drow (y), d/dcol (x)]
    nx = -strength * gx
    ny = -strength * gy
    nz = np.ones_like(h)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)         # normalize to unit length
    return np.stack([nx * inv, ny * inv, nz * inv], axis=-1)


def flat_normal_map(shape):
    """A flat tangent-space normal map (all +z) -- the abstain fallback (no relief)."""
    H, W = shape[0], shape[1]
    out = np.zeros((H, W, 3))
    out[..., 2] = 1.0
    return out


def pack_normal_rgb(nmap):
    """Pack a [-1, 1] normal map into an [0, 1] RGB image (n*0.5 + 0.5) -- the standard normal-map encoding."""
    return np.asarray(nmap, float) * 0.5 + 0.5


def bump_confidence(h):
    """How much fine RELIEF SIGNAL is present: the standard deviation of the high-passed height, measured over the
    INTERIOR (a 10% border is cropped, because the high-pass has reflect-padding edge artifacts there that would
    otherwise fake a signal on an otherwise-flat image). Low std -> essentially no fine detail to turn into a bump
    -> abstain to flat. IMPORTANT (kept negative): this measures DETAIL PRESENCE, not relief-vs-albedo -- a busy
    printed texture has high detail and will read as relief; the gate stops us inventing relief from a near-
    featureless image, it does not resolve the albedo/relief ambiguity."""
    h = np.asarray(h, float)
    m = max(1, int(0.1 * min(h.shape[0], h.shape[1])))
    core = h[m:-m, m:-m] if (h.shape[0] > 2 * m and h.shape[1] > 2 * m) else h
    return float(np.std(core))


def quantize_normals(nmap, bits=8):
    """Compactly store the normal map on its manifold via octahedral encoding (octnormal) -- (H, W, 2) quantized
    uv that round-trips back to unit normals. Reuses the shipped manifold-correct quantizer."""
    from holographic.mesh_and_geometry.holographic_octnormal import oct_quantize
    flat = np.asarray(nmap, float).reshape(-1, 3)
    q = oct_quantize(flat, bits=bits)
    return q.reshape(nmap.shape[0], nmap.shape[1], -1)


def auto_bump(rgb, strength=2.0, sigma=4.0, abstain_below=0.005):
    """The full auto-bump. image -> height -> normal map, with an honest CONFIDENCE GATE: if there is too little
    fine detail (confidence < abstain_below), ABSTAIN and return a flat normal map (no invented relief). Returns a
    dict: normal (H,W,3), height (H,W), confidence (float), abstained (bool)."""
    a = np.asarray(rgb, float)
    h = image_to_height(a, sigma=sigma)
    conf = bump_confidence(h)
    if conf < abstain_below:
        return {"normal": flat_normal_map(a.shape), "height": np.zeros_like(h),
                "confidence": conf, "abstained": True}
    return {"normal": normal_from_height(h, strength=strength), "height": h,
            "confidence": conf, "abstained": False}


def add_height_channel(material, encoder, grid, rgb, sigma=4.0):
    """Wire the derived HEIGHT into a Material as a scalar UV `height` channel (which `displace` consumes for real
    relief). Samples the high-passed height at the encoder's UV grid and encodes it as a texture field. Returns
    the material (mutated). The NORMAL map is returned separately by auto_bump for the renderer's normal channel."""
    from holographic.materials_and_texture.holographic_material import texture_field
    a = np.asarray(rgb, float)
    h = image_to_height(a, sigma=sigma)
    H, W = h.shape[0], h.shape[1]
    # sample the height at each UV grid point (uv in [0,1]^2 -> pixel)
    vals = []
    for (u, v) in grid:
        px = min(W - 1, max(0, int(round(u * (W - 1)))))
        py = min(H - 1, max(0, int(round(v * (H - 1)))))
        vals.append(float(h[py, px]))
    material.add("height", texture_field(encoder, grid, vals))
    return material


def _selftest():
    """A flat image gives flat normals and abstains; a slow RAMP is removed by the high-pass (no fake slope); a
    bumpy pattern gives varied unit normals with high confidence; octnormal round-trips; the height wires into a
    Material and samples back; deterministic."""
    N = 48

    # (1) a slow left->right RAMP must NOT become a slope -- the high-pass removes it (in the interior; the border
    #     carries small reflect-padding artifacts, which is exactly why the confidence gate crops the border)
    ramp = np.tile(np.linspace(0.2, 0.8, N), (N, 1))         # smooth gradient across x
    ramp_rgb = np.stack([ramp, ramp, ramp], axis=-1)
    hr = image_to_height(ramp_rgb, sigma=4.0)
    m = int(0.1 * N)
    assert np.std(hr[m:-m, m:-m]) < 0.01                     # interior: the ramp is gone -> ~no height signal
    res_ramp = auto_bump(ramp_rgb)
    assert res_ramp["abstained"]                             # nothing to bump -> abstain to flat

    # (2) a BUMPY pattern (a sine grid) is real relief -> varied unit normals, high confidence
    u = np.linspace(0, 6 * np.pi, N)
    bump = 0.5 + 0.4 * np.outer(np.sin(u), np.cos(u))
    bump_rgb = np.stack([bump, bump, bump], axis=-1)
    res = auto_bump(bump_rgb, strength=2.0)
    assert not res["abstained"] and res["confidence"] > 0.01
    nmap = res["normal"]
    assert nmap.shape == (N, N, 3)
    lens = np.linalg.norm(nmap, axis=-1)
    assert np.allclose(lens, 1.0, atol=1e-6)                 # every normal is unit length
    assert np.all(nmap[..., 2] > 0)                          # all point OUT of the surface (+z)
    assert np.std(nmap[..., 0]) > 0.05                       # the normals actually vary (real relief)

    # (3) a flat height -> flat normals
    flat = normal_from_height(np.zeros((N, N)))
    assert np.allclose(flat[..., 2], 1.0) and np.allclose(flat[..., :2], 0.0)

    # (4) octnormal quantization round-trips (unit normals back out)
    q = quantize_normals(nmap, bits=8)
    assert q.shape[:2] == (N, N)
    from holographic.mesh_and_geometry.holographic_octnormal import oct_decode
    back = oct_decode(q.reshape(-1, q.shape[-1]))
    assert np.allclose(np.linalg.norm(back, axis=-1), 1.0, atol=1e-6)

    # (5) wire the height into a Material and sample it back
    from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder
    from holographic.materials_and_texture.holographic_material import Material
    enc = VectorFunctionEncoder(2, dim=1024, bounds=[(0, 1), (0, 1)], bandwidth=3.0, seed=1)
    grid = [(uu, vv) for uu in np.linspace(0.05, 0.95, 9) for vv in np.linspace(0.05, 0.95, 9)]
    mat = Material(enc, {"albedo": importlib.import_module("holographic.materials_and_texture.holographic_material").texture_field(enc, grid, [0.5] * len(grid))})
    add_height_channel(mat, enc, grid, bump_rgb)
    assert "height" in mat.channels                          # the channel is populated
    assert np.isfinite(mat.sample("height", [0.5, 0.5]))     # and samples a finite value

    # (6) deterministic
    assert np.array_equal(image_to_height(bump_rgb), image_to_height(bump_rgb))

    print("holographic_autobump selftest OK: a slow ramp is removed by the high-pass (std %.4f -> abstain, no "
          "fake slope); a bumpy pattern gives unit normals that vary (relief), confidence %.3f; a flat height "
          "gives flat normals; octnormal round-trips to unit normals; the height wires into a Material and samples "
          "back; deterministic" % (float(np.std(hr)), float(res["confidence"])))


if __name__ == "__main__":
    _selftest()
