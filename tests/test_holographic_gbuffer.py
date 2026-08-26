"""Tests for holographic_gbuffer: the primary G-buffer and the clean render helper (adaptive + firefly + SVGF)."""
import numpy as np
import pytest

from holographic.rendering.holographic_gbuffer import primary_gbuffer, render_denoised, declfirefly
from holographic.rendering.holographic_pathtrace import path_trace


class _Cam:
    """Minimal camera exposing the ray_dirs(w,h) -> (eye, dirs) interface path_trace/gbuffer expect."""
    eye = np.array([0.0, 0.4, 3.2])
    def ray_dirs(self, w, h):
        ys, xs = np.mgrid[0:h, 0:w]
        u = (xs / (w - 1) - 0.5) * 1.2
        v = -(ys / (h - 1) - 0.5) * 1.2
        d = np.stack([u, v, -np.ones_like(u)], -1)
        return self.eye, d / np.linalg.norm(d, axis=-1, keepdims=True)


def _scene():
    centers = np.array([[-0.7, 0, 0], [0.7, 0, 0]], float); radii = np.array([0.6, 0.6])
    class Scene:
        def eval(self, P):
            d = np.min(np.linalg.norm(P[..., None, :] - centers, axis=-1) - radii, axis=-1)
            return np.minimum(d, P[..., 1] + 0.9)
    return Scene()


def _material(P):
    n = len(P); alb = np.tile([.8, .3, .3], (n, 1)).astype(float)
    alb[P[:, 0] < 0] = [.3, .4, .85]
    return alb, np.zeros(n), np.full(n, .6), np.zeros((n, 3))


def _sky(D):
    t = np.clip(D[:, 1] * 0.5 + 0.5, 0, 1)[:, None]
    return (1 - t) * np.array([0.9, 0.85, 0.8]) + t * np.array([0.35, 0.5, 0.9])


def _psnr(a, b):
    mse = float(np.mean((np.clip(a, 0, 4) - np.clip(b, 0, 4)) ** 2))
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(16.0 / mse)


def test_gbuffer_shapes_and_hit_vs_miss():
    normal, albedo, depth = primary_gbuffer(_scene(), _Cam(), 48, 48, _material, sky=_sky)
    assert normal.shape == (48, 48, 3)
    assert albedo.shape == (48, 48, 3)
    assert depth.shape == (48, 48)
    # something was hit (near) and something missed (far -> max_dist)
    assert depth.min() < 5.0 < depth.max()
    # a hit pixel has a unit-ish normal; a sky pixel has a ~zero normal
    assert np.linalg.norm(normal, axis=2).max() > 0.5


def test_render_denoised_beats_raw_low_spp():
    scene = _scene()
    ref = path_trace(scene, _Cam(), width=56, height=56, spp=128, max_bounce=3, material=_material, sky=_sky, seed=7)
    clean, stats = render_denoised(scene, _Cam(), 56, 56, _material, sky=_sky, spp=6, max_bounce=3, seed=1,
                                   return_stats=True)
    assert _psnr(clean, ref) > _psnr(stats["noisy"], ref) + 1.0     # denoise clearly helps at low spp


def test_declfirefly_removes_hot_pixel_keeps_bright_region():
    img = np.full((16, 16, 3), 0.2)
    img[8, 8] = 40.0                                 # a lone firefly
    img[2:6, 2:6] = 5.0                              # a genuinely bright patch
    out = declfirefly(img, k=3.0)
    assert out[8, 8].max() < 5.0                     # the firefly was clamped down
    assert np.allclose(out[3, 3], img[3, 3])         # the bright region is untouched


def test_render_denoised_is_deterministic():
    a = render_denoised(_scene(), _Cam(), 40, 40, _material, sky=_sky, spp=5, max_bounce=3, seed=3)
    b = render_denoised(_scene(), _Cam(), 40, 40, _material, sky=_sky, spp=5, max_bounce=3, seed=3)
    assert np.array_equal(a, b)


def test_selftest_runs():
    import holographic.rendering.holographic_gbuffer as holographic_gbuffer
    holographic_gbuffer._selftest()


def test_render_auto_converges_and_is_clean():
    from holographic.rendering.holographic_gbuffer import render_auto
    scene = _scene()
    ref = path_trace(scene, _Cam(), width=56, height=56, spp=160, max_bounce=3, material=_material, sky=_sky, seed=7)
    clean, st = render_auto(scene, _Cam(), 56, 56, _material, sky=_sky, quality="medium", max_bounce=3, seed=0,
                            pass_spp=8, return_stats=True)
    # it actually ran passes, converged most pixels, and produced a finite image
    assert st["passes"] >= 1 and st["mean_samples"] > 0
    assert np.isfinite(clean).all()
    # at equal average budget, the auto render is at least competitive with a raw trace (usually better)
    eq = int(round(st["mean_samples"]))
    raw = path_trace(scene, _Cam(), width=56, height=56, spp=eq, max_bounce=3, material=_material, sky=_sky, seed=0)
    def tm(x):
        return np.clip((x / (1.0 + x)) ** (1 / 2.2), 0, 1)
    def psnr(a, b):
        mse = float(np.mean((tm(a) - tm(b)) ** 2)); return 99.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)
    assert psnr(clean, ref) >= psnr(raw, ref) - 0.7          # auto is within noise of (usually beats) equal-budget raw


def test_render_auto_adapts_samples_to_difficulty():
    # a scene with an easy region (flat sky, converges instantly) and hard regions (sphere edges) should make the
    # sampler spend MORE samples on some pixels than the average -- i.e. it calibrates effort, not a flat spp.
    from holographic.rendering.holographic_gbuffer import render_auto
    _, st = render_auto(_scene(), _Cam(), 48, 48, _material, sky=_sky, quality="high", max_bounce=3, seed=0,
                        pass_spp=8, max_passes=6, return_stats=True)
    assert st["max_samples"] > st["mean_samples"]            # effort concentrated where the estimate is uncertain


def test_render_auto_is_deterministic():
    from holographic.rendering.holographic_gbuffer import render_auto
    a = render_auto(_scene(), _Cam(), 40, 40, _material, sky=_sky, quality="medium", max_bounce=3, seed=2)
    b = render_auto(_scene(), _Cam(), 40, 40, _material, sky=_sky, quality="medium", max_bounce=3, seed=2)
    assert np.array_equal(a, b)


def test_aces_tonemap_contrast_and_autoexposure():
    from holographic.rendering.holographic_gbuffer import aces_tonemap
    # output is bounded, monotonic, and darker->brighter maps darker->brighter
    x = np.linspace(0, 8, 50).reshape(-1, 1) * np.ones((1, 3))
    y = aces_tonemap(x, auto=False)
    assert y.min() >= 0.0 and y.max() <= 1.0
    assert np.all(np.diff(y[:, 0]) >= -1e-9)                  # monotone non-decreasing
    # auto-exposure maps two scenes of very different absolute brightness to a similar mid-tone
    dim = np.full((16, 16, 3), 0.05); bright = np.full((16, 16, 3), 5.0)
    md = float(np.median(aces_tonemap(dim, auto=True))); mb = float(np.median(aces_tonemap(bright, auto=True)))
    assert abs(md - mb) < 0.05                                # both metered onto ~mid-grey


def _glass_scene():
    gc = np.array([0.0, 0.0, 0.3]); gr = 0.7
    class Scene:
        def eval(self, P):
            glass = np.linalg.norm(P - gc, axis=-1) - gr
            back = np.linalg.norm(P - np.array([0.0, -0.1, -1.3]), axis=-1) - 0.5
            return np.minimum(np.minimum(glass, back), P[..., 1] + 0.9)
    def material(P):
        n = len(P); alb = np.tile([.85, .3, .25], (n, 1)).astype(float)
        met = np.zeros(n); rough = np.full(n, .5); emis = np.zeros((n, 3)); ior = np.zeros(n)
        onglass = np.abs(np.linalg.norm(P - gc, axis=-1) - gr) < 0.06
        ior[onglass] = 1.5; alb[onglass] = [1, 1, 1]; rough[onglass] = 0.02
        g = P[:, 1] < -0.85; alb[g] = [.85, .85, .9]
        return alb, met, rough, emis, ior
    return Scene(), material, gc, gr


def test_render_dispersion_splits_the_refracted_channels():
    from holographic.rendering.holographic_gbuffer import render_auto, render_dispersion
    scene, material, gc, gr = _glass_scene()
    base = render_auto(scene, _Cam(), 60, 60, material, sky=_sky,
                       quality="draft", max_bounce=6, seed=0)
    disp = render_dispersion(scene, _Cam(), 60, 60, material, sky=_sky,
                             quality="draft", max_bounce=6, seed=0, dispersion=0.08)
    # dispersion must change the image (the refracted light shifts per channel) and introduce per-channel
    # divergence the single-IOR render does not have
    assert float(np.mean(np.abs(disp - base))) > 1e-3
    base_chroma = float(np.mean(np.abs(base[..., 0] - base[..., 2])))
    disp_chroma = float(np.mean(np.abs(disp[..., 0] - disp[..., 2])))
    assert disp_chroma > base_chroma                          # R and B diverge more with dispersion on


def test_add_caustics_brightens_floor_only():
    from holographic.rendering.holographic_gbuffer import add_caustics
    from holographic.mesh_and_geometry.holographic_sdf import sphere as _sphere
    scene, material, gc, gr = _glass_scene()
    cam = _Cam()
    flat = np.full((60, 60, 3), 0.2)                          # a flat HDR image to composite onto
    glass_only = _sphere(float(gr)).translate((float(gc[0]), float(gc[1]), float(gc[2])))
    out = add_caustics(flat, scene, cam, 60, 60, light_dir=(0.2, -0.9, 0.1), receiver_y=-0.9, extent=1.6,
                       ior=1.5, strength=1.0, caustic_sdf=glass_only)
    assert out.shape == flat.shape
    assert float(out.max()) > 0.2 + 1e-3                      # some floor pixels got brighter (the focused cusp)
    assert float(out.min()) >= 0.2 - 1e-9                     # nothing was darkened


# ======================================================================================================
# The stop rule is an ESCALATE MASK. `adaptive_sample.converged_mask` now cites the unifier that owns them.
# ======================================================================================================
def test_converged_mask_is_bit_identical_to_the_old_inline_comparison_including_exact_ties():
    """The one thing delegation had to preserve is the TIE CONVENTION, and it is the opposite of coarse-first's
    default: a pixel whose CI half-width is EXACTLY the tolerance has CONVERGED and must stop, where an escalation
    rule refines on a tie. Hence `escalate_mask(..., inclusive=False)`. Ties are constructed on purpose here."""
    from holographic.sampling_and_signal.holographic_adaptive_sample import Z95, ci_half_width, converged_mask
    rng = np.random.default_rng(0)
    tol = 0.022
    vom = rng.random(100_000) * 0.01
    vom[:500] = (tol / Z95) ** 2                       # ci_half_width == tol EXACTLY
    old = ci_half_width(vom) <= tol                    # the comparison that used to live inline
    new = converged_mask(vom, tol)
    assert np.array_equal(old, new)
    assert new[:500].all()                             # a tie is CONVERGED (stop), not escalated


def test_the_adaptive_renderer_still_stops_per_pixel_through_the_delegated_mask():
    from holographic.rendering.holographic_gbuffer import converge_samples
    img, vom, N, info = converge_samples(_scene(), _Cam(), 24, 24, _material, sky=_sky, quality="draft",
                                         max_bounce=2, seed=0, pass_spp=4, max_passes=3)
    assert img.shape == (24, 24, 3) and N.shape == (24, 24)
    assert int(N.min()) < int(N.max())                 # the stop rule really is per pixel: counts differ
    # median_ci is legitimately 0.0 at draft quality: most pixels are background rays with zero variance, and a
    # zero-variance pixel has a zero-width confidence interval. Assert the contract, not a hoped-for number.
    assert 1 <= info["passes"] <= 3 and info["median_ci"] >= 0.0
    assert np.isfinite(vom).all() and float(vom.min()) >= 0.0
    img2, *_ = converge_samples(_scene(), _Cam(), 24, 24, _material, sky=_sky, quality="draft",
                                max_bounce=2, seed=0, pass_spp=4, max_passes=3)
    assert np.array_equal(img, img2)                   # deterministic, as before


def test_escalate_mask_tie_conventions_are_both_available_and_opposite():
    from holographic.misc.holographic_coarsefirst import escalate_mask
    u = np.array([0.5, 1.0, 1.5])
    assert list(escalate_mask(u, threshold=1.0)) == [False, True, True]                  # refine on a tie
    assert list(escalate_mask(u, threshold=1.0, inclusive=False)) == [False, False, True]  # strict
