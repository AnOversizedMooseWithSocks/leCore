"""holographic_checkerboard.py -- CHECKERBOARD / SPARSE RENDERING (inverse-rendering IR13).

Shade only ~50% of the pixels (a 2x2 checkerboard, alternated between frames) and RECONSTRUCT the rest -- roughly
halving the shading cost for a near-full-resolution result. The "larger resolution without it taking forever" trick,
done as a sampling PATTERN rather than a naive lower-resolution render.

The holographic reading (Ozcan's seat): the unshaded pixels are "DAMAGE", and reconstruction is RECOVERY FROM A
PARTIAL/MASKED MEASUREMENT -- the archive's literal job, here in the pixel domain. The gem of the 2x2 checkerboard
is that every unshaded pixel's four cross-neighbours (up/down/left/right) are ALL shaded, so the recovery is a clean
cross-neighbour average -- no iteration, no learned prior. Flipping the parity each frame fills a different half, so
over two frames every pixel is eventually shaded (the temporal variant, which reuses temporal reprojection).

render_checkerboard genuinely traces ONLY the masked rays (the cost saving is real, not simulated), reusing the
shipped sphere_trace / sdf_normal / sky_dome, then reconstructs the rest.

KEPT NEGATIVES (loud): reconstruction is a TRADE (better accuracy per unit render cost), not free -- it costs more
than simply rendering at a lower resolution, but reconstructs more accurately at matched cost (the documented
checkerboard advantage over plain upscaling). Under MOTION it can shimmer in fine detail unless the temporal
reprojection rejects disoccluded pixels by depth/motion (out of scope for this single-frame v1). Colour/visibility
exist only at the render-target resolution, so it is a RECONSTRUCTION, not true supersampling. NumPy + stdlib only;
deterministic.
"""
import numpy as np

from holographic.rendering.holographic_raymarch import sphere_trace, sdf_normal, sky_dome


def checkerboard_mask(height, width, parity=0):
    """The 2x2 checkerboard shading mask: True where a pixel is SHADED. `parity` flips the pattern each frame so the
    gaps fill over time (the temporal variant)."""
    yy, xx = np.mgrid[0:height, 0:width]
    return ((xx + yy + parity) % 2 == 0)


def _shift(a, dy, dx):
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)


def reconstruct_checkerboard(image, mask):
    """Fill the unshaded (mask False) pixels from their four cross-neighbours, which in a 2x2 checkerboard are all
    SHADED. A cross-neighbour average weighted by how many neighbours are known -- the classic, exact-in-the-interior
    checkerboard reconstruction. Shaded pixels are kept exactly."""
    img = np.asarray(image, float)
    m = mask.astype(float)
    known = m[..., None] if img.ndim == 3 else m
    vals = img * known
    ssum = _shift(vals, 1, 0) + _shift(vals, -1, 0) + _shift(vals, 0, 1) + _shift(vals, 0, -1)
    scnt = _shift(known, 1, 0) + _shift(known, -1, 0) + _shift(known, 0, 1) + _shift(known, 0, -1)
    filled = ssum / np.clip(scnt, 1e-8, None)
    fill_where = (~mask)[..., None] if img.ndim == 3 else (~mask)
    return np.where(fill_where, filled, img)


def _shade_rays(sdf, D, eye, light_dir, base_color, ambient):
    """Shade a set of ray directions with Lambert-plus-sky (the same model render_checkerboard and its full-shade
    reference share, so a comparison measures RECONSTRUCTION, not a shader mismatch)."""
    Om = np.broadcast_to(eye, D.shape)
    L = np.asarray(light_dir, float)
    L = L / (np.linalg.norm(L) + 1e-12)
    col = sky_dome(D)                                             # background = sky
    hit, t, P = sphere_trace(sdf, Om, D)
    if hit.any():
        N = sdf_normal(sdf, P[hit])
        ndl = np.clip(N @ L, 0, None)
        col[hit] = np.asarray(base_color, float) * (ambient + (1.0 - ambient) * ndl)[:, None]
    return col


def _shade_all(sdf, camera, width, height, light_dir=(-0.4, 0.7, -0.3), base_color=(0.85, 0.5, 0.35), ambient=0.25):
    """Shade EVERY pixel with the shared shader -- the full-resolution reference for the reconstruction test."""
    eye, dirs = camera.ray_dirs(width, height)
    col = _shade_rays(sdf, dirs.reshape(-1, 3), eye, light_dir, base_color, ambient)
    return np.clip(col.reshape(height, width, 3), 0.0, 1.0)


def render_checkerboard(sdf, camera, width, height, parity=0, light_dir=(-0.4, 0.7, -0.3),
                        base_color=(0.85, 0.5, 0.35), ambient=0.25):
    """Shade ONLY the checkerboard-masked pixels (~half the rays traced), then reconstruct the rest. Returns
    (image (H,W,3) in [0,1], mask). The shaded pixels use the shared Lambert-plus-sky shader, so a checkerboard
    frame + reconstruction can be compared apples-to-apples against a full shade of the same scene."""
    mask = checkerboard_mask(height, width, parity)
    eye, dirs = camera.ray_dirs(width, height)
    D = dirs.reshape(-1, 3)
    flat_mask = mask.ravel()

    img = np.zeros((height * width, 3))
    img[flat_mask] = _shade_rays(sdf, D[flat_mask], eye, light_dir, base_color, ambient)   # ONLY the masked rays
    img = img.reshape(height, width, 3)
    return np.clip(reconstruct_checkerboard(img, mask), 0.0, 1.0), mask


def _psnr(a, b):
    mse = float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))
    return 99.0 if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))


def _row_halved(full):
    """A matched-cost (50%-shaded) baseline: keep the EVEN rows (shade half), fill odd rows from the rows above/below
    -- 'reduce the resolution in one direction then upscale'. Checkerboard spreads its 50% in 2D and should beat it."""
    out = full.copy()
    odd = np.zeros(full.shape[0], bool); odd[1::2] = True
    up = _shift(full, 1, 0); dn = _shift(full, -1, 0)
    out[odd] = 0.5 * (up[odd] + dn[odd])
    return out


def _selftest():
    """The mask shades ~50% and flips with parity; reconstruction from the checkerboard matches full-res at high
    PSNR and beats both a no-fill baseline and a matched-cost row-halved render; render_checkerboard traces only the
    masked half and matches a full render after reconstruction; deterministic."""
    from holographic.rendering.holographic_render import Camera
    from holographic.mesh_and_geometry.holographic_sdf import box

    # (1) the mask is ~half and parity flips it
    m0 = checkerboard_mask(40, 40, 0)
    m1 = checkerboard_mask(40, 40, 1)
    assert abs(m0.mean() - 0.5) < 0.02 and np.all(m0 == ~m1)

    # (2) reconstruction from a checkerboard of a full render matches it, and beats no-fill + row-halving
    sdf = box(1.0, 0.7, 0.5)
    cam = Camera(eye=(2.5, 1.8, 2.5), target=(0, 0, 0), fov_deg=50.0)
    full = _shade_all(sdf, cam, 80, 80)                          # full shade with the SAME shader (the reference)
    mask = checkerboard_mask(80, 80, 0)
    recon = reconstruct_checkerboard(full, mask)                  # uses full's values only at the masked pixels
    nofill = full * mask[..., None]                              # the naive 'leave gaps' baseline
    rowh = _row_halved(full)
    assert _psnr(recon, full) > 30.0                            # near-full-resolution from half the shaded pixels
    assert _psnr(recon, full) > _psnr(nofill, full)             # reconstruction clearly beats leaving gaps
    assert _psnr(recon, full) > _psnr(rowh, full)               # and beats a matched-cost row-halved render

    # (3) render_checkerboard traces only ~half and, after reconstruction, matches the full render
    ck, ck_mask = render_checkerboard(sdf, cam, 80, 80)
    assert abs(ck_mask.mean() - 0.5) < 0.02                     # ~half the rays traced
    assert _psnr(ck, full) > 30.0                               # reconstructed frame ~ the full render

    # (4) parity flips fill the OTHER half (temporal variant shades everything over two frames)
    ck1, m1b = render_checkerboard(sdf, cam, 80, 80, parity=1)
    assert np.all(m1b == ~ck_mask)

    # (5) deterministic
    assert np.array_equal(render_checkerboard(sdf, cam, 40, 40)[0], render_checkerboard(sdf, cam, 40, 40)[0])

    print("holographic_checkerboard selftest OK: the 2x2 mask shades %.0f%% and parity flips it; reconstructing "
          "from the checkerboard matches the full render at %.1f dB (>30), beating no-fill (%.1f) and a matched-cost "
          "row-halved render (%.1f); render_checkerboard traces only the masked half and reconstructs to %.1f dB; "
          "deterministic" % (100 * mask.mean(), _psnr(recon, full), _psnr(nofill, full), _psnr(rowh, full),
                             _psnr(ck, full)))


if __name__ == "__main__":
    _selftest()
