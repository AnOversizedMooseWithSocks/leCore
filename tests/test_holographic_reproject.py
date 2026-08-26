"""F7 -- frame-to-frame motion by ONE UNBIND, and the premise that did not survive measurement.

The backlog: *"TAA's analytic reprojection velocity is our `est_dx`. Recovering per-pixel motion between frames is
edit recovery applied frame-to-frame -- one unbind per tile instead of motion vectors from geometry (F7; owes a
measurement on real frames)."* It owed a measurement. Here it is, and the estimator is excellent while the premise
is not supported.

THE DATA IS REAL. Every image here is either a rendered frame from `RenderSession` or a structured, band-limited
signal. White noise would flatter the estimator's peak-finding at integer shifts and punish it at sub-pixel, and a
test pins that it is in fact the WORST case -- so it is used only to demonstrate that, never as the baseline.
"""

import numpy as np
import pytest

from holographic.rendering.holographic_reproject import (
    est_dx, est_dx_tiles, flow_uniformity, psnr, reproject, reproject_report, warp)


def _smooth(n=96, seed=0):
    """A structured, band-limited signal: the correlation peak has CURVATURE, which is what the parabola needs."""
    x = np.linspace(0, 6, n)
    base = np.outer(np.sin(x), np.cos(1.7 * x)) + 0.3 * np.outer(x, x[::-1])
    return (base - base.min()) / (base.max() - base.min() + 1e-12)


def _frames(width=192, dx=0.0, dz=0.0):
    """REAL rendered frames from the shipped session: two spheres at different depths, so parallax is present."""
    from holographic.mesh_and_geometry.holographic_surface import SurfaceMaterial
    from holographic.rendering.holographic_render import Camera
    from holographic.scene_and_pipeline.holographic_session import RenderSession

    class Two:
        cs = np.array([[0.0, 0, 0], [1.9, 0, 0]])

        def eval(self, P):
            return np.min(np.stack([np.linalg.norm(P - c, axis=1) - 0.85 for c in self.cs]), axis=0)

        def ids(self, P):
            return np.argmin(np.stack([np.linalg.norm(P - c, axis=1) for c in self.cs]), axis=0)

    mats = {0: SurfaceMaterial.from_name("plastic"), 1: SurfaceMaterial.from_name("metal")}
    cam = Camera(eye=(0.9 + dx, 1.0, 4.6 - dz), target=(0.9 + dx, 0, 0), fov_deg=52)
    return RenderSession(Two(), mats, cam, width=width, height=width).preview().mean(axis=2)


def test_selftest_runs():
    from holographic.rendering import holographic_reproject as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("truth", [(3.0, -5.0), (12.0, 7.0), (0.0, 0.0)])
def test_integer_shifts_are_recovered_exactly(truth):
    a = _smooth()
    got = est_dx(a, warp(a, *truth, wrap=True))
    assert np.abs(got - np.array(truth)).max() < 1e-6


@pytest.mark.parametrize("truth", [(0.5, 0.25), (-1.75, 2.3), (0.33, -0.67)])
def test_subpixel_shifts_land_within_the_stated_tolerance(truth):
    a = _smooth()
    err = float(np.linalg.norm(est_dx(a, warp(a, *truth, wrap=True)) - np.array(truth)))
    assert err < 0.15                                       # measured 0.084 / 0.121 / 0.120 px on this fixture


def test_subpixel_off_returns_the_integer_peak():
    a = _smooth()
    got = est_dx(a, warp(a, 2.6, -3.4, wrap=True), subpixel=False)
    assert np.array_equal(got, np.round(got))               # integers, no interpolation


def test_est_dx_on_a_real_rendered_frame_with_a_known_warp():
    a = _frames(width=128)
    errs = [float(np.linalg.norm(est_dx(a, warp(a, *t, wrap=True)) - np.array(t)))
            for t in ((0.5, 0.25), (-1.75, 2.3), (3.0, -5.0))]
    assert max(errs) < 0.3                                  # measured 0.0705 mean / 0.1087 worst on 128^2


# ---------------------------------------------------------------------------------------------------------
# THE KEPT NEGATIVES
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_phase_normalization_is_worse_at_subpixel():
    # Textbook phase correlation sharpens the peak toward a delta. A parabola needs CURVATURE.
    a = _smooth()
    truths = ((0.5, 0.25), (0.33, -0.67))
    plain = np.mean([np.linalg.norm(est_dx(a, warp(a, *t, wrap=True)) - np.array(t)) for t in truths])
    norm = np.mean([np.linalg.norm(est_dx(a, warp(a, *t, wrap=True), normalize=True) - np.array(t))
                    for t in truths])
    assert plain < norm
    assert norm / plain > 1.5                               # measured 2.2x on this fixture, 2.3x on a real frame


def test_kept_negative_white_noise_is_the_worst_case_for_the_same_reason():
    # Its autocorrelation IS a delta. Using white noise as a test signal would understate the estimator on real
    # frames at integer shifts and overstate the sub-pixel error. Pinned so nobody "simplifies" the fixtures.
    truth = (1.4, -2.6)
    noise = np.random.default_rng(0).normal(size=(64, 64))
    e_noise = float(np.linalg.norm(est_dx(noise, warp(noise, *truth, wrap=True)) - np.array(truth)))
    e_smooth = float(np.linalg.norm(est_dx(_smooth(64), warp(_smooth(64), *truth, wrap=True)) - np.array(truth)))
    assert e_noise > 1.8 * e_smooth                         # measured 0.1995 px vs 0.0963 px


def test_kept_negative_a_hann_window_makes_it_worse():
    a = _smooth()
    truth = (0.5, 0.25)
    hann = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    b = warp(a, *truth, wrap=True)
    plain = np.linalg.norm(est_dx(a, b) - np.array(truth))
    windowed = np.linalg.norm(est_dx((a - a.mean()) * hann, (b - a.mean()) * hann) - np.array(truth))
    assert windowed > 3.0 * plain                           # measured 0.334 px vs 0.084 px, even after mean removal


def test_the_edge_model_is_the_estimators_one_structural_assumption():
    # FFT correlation assumes wrap-around. A clamped warp -- a real frame -- carries a bias, and the bias is driven
    # by the BORDER DISCONTINUITY the clamp creates. I asserted "bounded under 1 px" and it came back 5.76: this
    # fixture has a strong linear ramp, so clamping it manufactures a huge edge. On a real rendered frame, whose
    # borders are near-uniform background, the same shift biases by 0.40 px. The bias is signal-dependent, and
    # saying "bounded" without saying by what would have been a false comfort.
    a = _smooth()
    truth = (3.0, -5.0)
    circ = float(np.linalg.norm(est_dx(a, warp(a, *truth, wrap=True)) - np.array(truth)))
    clamp = float(np.linalg.norm(est_dx(a, warp(a, *truth, wrap=False)) - np.array(truth)))
    assert circ < 1e-6 < clamp                              # exact vs biased

    frame = _frames(width=128)
    clamp_frame = float(np.linalg.norm(est_dx(frame, warp(frame, *truth, wrap=False)) - np.array(truth)))
    assert clamp_frame < clamp                              # a gentler border, a smaller bias


def _slide(centres):
    """Render a scene of unit spheres at `centres` from a FIXED camera. Moving the SCENE, not the camera, is the
    only way to isolate depth effects: a far-away camera stops the image changing at all, and warping an unchanged
    frame perfectly proves nothing. (My first version of this test did exactly that.)"""
    from holographic.mesh_and_geometry.holographic_surface import SurfaceMaterial
    from holographic.rendering.holographic_render import Camera
    from holographic.scene_and_pipeline.holographic_session import RenderSession

    C = np.array(centres, float)

    class S:
        cs = C

        def eval(self, P):
            return np.min(np.stack([np.linalg.norm(P - c, axis=1) - 0.85 for c in self.cs]), axis=0)

        def ids(self, P):
            return np.argmin(np.stack([np.linalg.norm(P - c, axis=1) for c in self.cs]), axis=0)

    mats = {0: SurfaceMaterial.from_name("plastic"), 1: SurfaceMaterial.from_name("plastic")}
    cam = Camera(eye=(0.0, 1.0, 4.6), target=(0.0, 0, 0), fov_deg=52)
    return RenderSession(S(), mats, cam, width=128, height=128).preview().mean(axis=2)


def test_the_far_camera_control_is_vacuous_and_must_not_be_used():
    # THE ERROR I SHIPPED, pinned so it cannot come back. "Pull the camera far back until parallax vanishes and the
    # same code reaches 99 dB" is TRUE and MEANINGLESS: the two frames are identical, so there is nothing to warp.
    from holographic.mesh_and_geometry.holographic_surface import SurfaceMaterial
    from holographic.rendering.holographic_render import Camera
    from holographic.scene_and_pipeline.holographic_session import RenderSession

    class Two:
        cs = np.array([[0.0, 0, 0], [1.9, 0, 0]])

        def eval(self, P):
            return np.min(np.stack([np.linalg.norm(P - c, axis=1) - 0.85 for c in self.cs]), axis=0)

        def ids(self, P):
            return np.argmin(np.stack([np.linalg.norm(P - c, axis=1) for c in self.cs]), axis=0)

    mats = {0: SurfaceMaterial.from_name("plastic"), 1: SurfaceMaterial.from_name("metal")}

    def far(dx):
        cam = Camera(eye=(0.9 + dx, 1.0, 40.0), target=(0.9 + dx, 0, 0), fov_deg=8)
        return RenderSession(Two(), mats, cam, width=128, height=128).preview().mean(axis=2)

    fa, fb = far(0.0), far(0.03)
    assert psnr(fa, fb) > 90.0                              # the frames are IDENTICAL before any warping
    assert np.abs(est_dx(fa, fb)).max() == 0.0              # ... and the estimator says so


def test_kept_negative_parallax_halves_what_one_translation_can_explain():
    # THE VALID CONTROL: camera fixed, scene moves, both frames provably different. Same motion, different depths.
    same = (_slide([[-1.0, 0, 0], [1.0, 0, 0]]), _slide([[-0.88, 0, 0], [1.12, 0, 0]]))
    diff = (_slide([[-1.0, 0, 0], [1.0, 0, -4.0]]), _slide([[-0.88, 0, 0], [1.12, 0, -4.0]]))

    for a, b in (same, diff):
        assert psnr(a, b) < 40.0                            # the frames really moved

    gain_same = psnr(reproject(*same), same[1]) - psnr(*same)
    gain_diff = psnr(reproject(*diff), diff[1]) - psnr(*diff)
    assert gain_same > 9.0                                  # measured 11.65 dB
    assert gain_diff < 0.7 * gain_same                      # measured 6.06 dB: parallax halves it


def test_a_depth_slide_is_a_scale_change_and_a_translation_cannot_model_it():
    a, b = _slide([[0.0, 0, 0]]), _slide([[0.0, 0, -0.35]])
    assert psnr(a, b) < 40.0
    gain_depth = psnr(reproject(a, b), b) - psnr(a, b)

    la, lb = _slide([[0.0, 0, 0]]), _slide([[0.12, 0, 0]])
    gain_lateral = psnr(reproject(la, lb), lb) - psnr(la, lb)
    assert gain_lateral > 1.8 * gain_depth                   # 11.63 dB vs 5.48 dB


# ---------------------------------------------------------------------------------------------------------
# THE PREMISE: one unbind per TILE
# ---------------------------------------------------------------------------------------------------------

def test_a_global_warp_beats_doing_nothing():
    a, b = _frames(width=128, dx=0.0), _frames(width=128, dx=0.05)
    rep = reproject_report(a, b, tiles=(32,))
    assert rep["global"] > rep["no_warp"] + 3.0             # measured 28.5 -> 36.5 dB


def test_kept_negative_tiling_loses_on_uniform_motion():
    # A pure image translation is the most uniform field there is.
    a = _smooth()
    b = warp(a, 1.4, -2.6, wrap=True)
    rep = reproject_report(a, b, tiles=(32, 48))
    assert rep["best"] == "global"
    for t, v in rep["tiled"].items():
        assert rep["global"] > v, t


def test_tiling_wins_on_a_non_uniform_field():
    # A dolly gives a RADIAL flow: no single translation explains it.
    a, b = _frames(width=192, dz=0.0), _frames(width=192, dz=0.06)
    rep = reproject_report(a, b, tiles=(48,))
    assert rep["tiled"][48] > rep["global"]                  # measured 37.43 vs 34.82 dB
    assert rep["best"] == "tile:48"


def test_kept_negative_the_shift_spread_is_not_a_regime_gate():
    # THE RETRACTION. I built `flow_uniformity` to be the free diagnostic and measured it instead: a PURE
    # TRANSLATION has the largest spread of the three regimes, and the global warp still wins by ~22 dB. The spread
    # is dominated by border tiles, whose correlation peaks are noise.
    a = _smooth()
    trans = warp(a, 1.4, -2.6, wrap=True)
    spread_uniform = flow_uniformity(est_dx_tiles(a, trans, tile=32)).max()
    rep = reproject_report(a, trans, tiles=(32,))
    assert rep["best"] == "global"
    assert spread_uniform > 0.2                             # large spread, and tiling still loses

    # ... so the spread cannot be used to choose. `best` needs the true next frame -- which reprojection exists to
    # avoid rendering. At runtime, the camera knows how it moved. That is F7's honest conclusion.
    assert "uniformity" in rep and 32 in rep["uniformity"]


def test_flat_tiles_are_skipped_rather_than_assigned_a_hallucinated_shift():
    flat = np.full((64, 64), 0.5)
    assert est_dx_tiles(flat, flat, tile=32) == []
    out = reproject(flat, flat, tile=32)
    assert np.array_equal(out, flat)                        # copied through, not warped by noise


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    a = _smooth(64)
    b = warp(a, 1.4, -2.6, wrap=True)
    assert float(np.linalg.norm(m.est_dx(a, b) - np.array([1.4, -2.6]))) < 0.12
    assert psnr(m.reproject(a, b), b) > psnr(a, b)

    rep = m.reproject_report(a, b, tiles=(32,))
    assert rep["best"] == "global" and rep["global"] > rep["no_warp"]

    assert "one unbind" in str(m.find_capability("estimate the shift between two images")[:3])
