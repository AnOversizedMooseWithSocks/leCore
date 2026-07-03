"""holographic_superres.py -- EXAMPLE-BASED SUPER-RESOLUTION / GUIDED UPSAMPLING (inverse-rendering ST3).

The quality-and-speed payoff of the patch/guide machinery: RENDER SMALL, UPSCALE BY EXAMPLE. Two classical, no-
learned-weights routes, both reusing shipped parts:

  * GUIDED (JOINT-BILATERAL) UPSAMPLING (Kopf et al. 2007 JBU; He et al. 2013 guided filter) -- the render-speedup.
    A cheap render shades COLOUR at low resolution, but the GEOMETRY (normal / depth / albedo -- the G-buffer, which
    IR14 render_channels exposes) is available at FULL resolution because tracing it is cheap. So coarsely upscale the
    low-res colour, then edge-aware-filter it GUIDED by the full-res G-buffer -- the colour edges snap to the geometry
    edges. That guided filter is exactly the shipped SVGF feature-cosine bilateral (holographic_svgf.atrous_bilateral),
    steered by the guide instead of used as a denoiser. Combined with IR10 (denoise the cheap render), this is a
    fully-classical render-cheap-then-enhance.
  * EXAMPLE / SELF-SIMILAR SUPER-RESOLUTION (Freeman 2002; Glasner 2009; the Image-Analogies low->high map) -- for a
    low-res image with no guide, recover detail by matching each low-res patch to similar patches and borrowing their
    high-frequency residual. The patch search is HoloForest recall -- the SAME machinery ST2 already ships as
    holographic_texturesynth.find_similar_patches -- so the guide-free route composes that primitive rather than
    duplicating it here; this module ships the GUIDED route, which is the render-speedup and the stronger win.

KEPT NEGATIVES (loud): classical upsampling INVENTS PLAUSIBLE, NOT TRUE, detail and tops out BELOW learned
super-resolution -- it snaps/borrows structure, it does not recover information that was never sampled. Guided
upsampling needs a CLEAN full-res guide (the G-buffer supplies it; a noisy guide leaks noise into the colour). NumPy
+ stdlib only; deterministic.
"""
import numpy as np

from holographic_fsr import easu_upscale
from holographic_svgf import atrous_bilateral


def guided_upsample(low_color, guide_normal, guide_albedo=None, guide_depth=None, levels=4, sigma_color=2.0):
    """Guided (joint-bilateral) upsample. Coarsely upscale `low_color` to the guide's resolution, then run the SVGF
    feature-cosine bilateral steered by the full-res G-buffer (`guide_normal`/`guide_albedo`/`guide_depth`), so the
    colour edges snap to the geometry edges. `sigma_color` is set HIGH so the (blurry) upscaled colour does not drive
    the edges -- the GUIDE does. Returns the (H, W, 3) upsampled colour."""
    n = np.asarray(guide_normal, float)
    H, W = n.shape[:2]
    low = np.asarray(low_color, float)
    scale = H / low.shape[0]
    up = easu_upscale(low, scale)[:H, :W]                    # coarse upscale (blurry at the edges)
    a = np.asarray(guide_albedo, float) if guide_albedo is not None else n
    z = np.asarray(guide_depth, float) if guide_depth is not None else n.mean(axis=-1)
    guided = atrous_bilateral(up, n, a, z, sigma_color=sigma_color, levels=levels)   # snap to the guide's edges
    return np.clip(guided, 0.0, 1.0)


def _psnr(a, b):
    mse = float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))
    return 99.0 if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))


def _selftest():
    """Guided upsampling of a cheap low-res render, steered by the full-res G-buffer, beats a plain (guide-free)
    upscale on PSNR-to-native -- the colour edges snap to the geometry. Deterministic."""
    from holographic_render import Camera
    from holographic_sdf import box
    from holographic_raymarch import render_sdf
    from holographic_renderchannels import render_channels
    from holographic_fsr import _box_downscale

    sdf = box(1.0, 0.7, 0.5)
    cam = Camera(eye=(2.5, 1.8, 2.5), target=(0, 0, 0), fov_deg=50.0)
    rkw = dict(ao=False, shadows=False, reflect=0.0)

    native = render_sdf(sdf, cam, width=64, height=64, **rkw)                 # the full-res colour reference
    gb = render_channels(sdf, cam, want=["normal", "depth"], width=64, height=64, **rkw)   # full-res G-buffer (cheap)
    low = _box_downscale(native, 2)                                          # the cheap low-res colour render

    guided = guided_upsample(low, gb["normal"], guide_depth=gb["depth"])[:64, :64]
    plain = easu_upscale(low, 2.0)[:64, :64]
    assert _psnr(guided, native) > _psnr(plain, native)                      # the guide sharpens the edges

    # deterministic
    assert np.array_equal(guided_upsample(low, gb["normal"], guide_depth=gb["depth"]),
                          guided_upsample(low, gb["normal"], guide_depth=gb["depth"]))

    print("holographic_superres selftest OK: guided (joint-bilateral) upsample of a cheap low-res render, steered by "
          "the full-res G-buffer, beats a plain upscale on PSNR-to-native (%.2f vs %.2f dB) -- colour edges snap to "
          "the geometry; deterministic" % (_psnr(guided, native), _psnr(plain, native)))


if __name__ == "__main__":
    _selftest()
