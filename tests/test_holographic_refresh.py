"""W4 -- information-rate rendering: shade the news, reproject the rest.

THE BAR, reproduced: on a parallax-free procedural scene (a pure function of world coords), 12 frames, 20% budget,
with a KNOWN camera shift -- **57.5 dB mean, 55.9 dB worst, tail slope +0.22 dB**. The backlog reported 55.6 / 54.1
at 20.6%. Five times fewer shader evaluations at visually-indistinguishable quality, and it does not decay.

THREE KEPT NEGATIVES, each with a test:
  1. recovering the shift from pixels costs 10.5 dB and turns the tail slope to -9.52 (decay) -- the loop warps its
     own output, so `est_dx`'s 0.07 px error compounds. The renderer knows how the camera moved.
  2. integer `np.roll` decays: 40.7 dB against 57.5 for the same budget.
  3. THE FAKE-PERFECT BUG: a threshold selection takes ALL 16,384 pixels when ages are tied (frame 0), reporting
     PSNR 99 dB by shading everything. `exact_k_oldest` takes exactly k, ties broken on the flat index.

AND THE SCOPE: 57.5 dB belongs to a scene with no parallax and no view-dependent shading. On a real 3-D scene the
reprojection ceiling is itself ~38-41 dB (see `holographic_reproject`), so refresh cannot beat it.
"""

import numpy as np
import pytest

from holographic.rendering.holographic_refresh import (
    RefreshRenderer, disocclusion_border, exact_k_oldest, refresh_report, threshold_oldest)
from holographic.rendering.holographic_reproject import psnr


H = W = 128
STEP = 1.7


def world(ox):
    """A procedural scene: a PURE FUNCTION OF WORLD COORDS. Panning translates the sample grid, so reprojection is
    an exact translation. This is W4's own scene class -- no parallax, no view-dependent shading, and saying so is
    the difference between a result and an overclaim."""
    yy, xx = np.meshgrid(np.arange(H), np.arange(W) + ox, indexing="ij")
    v = (np.sin(xx * 0.11) * np.cos(yy * 0.09) + 0.4 * np.sin((xx + yy) * 0.05)
         + 0.3 * np.sin(xx * 0.31) * np.sin(yy * 0.27))
    return (v - v.min()) / (v.max() - v.min() + 1e-12)


def test_selftest_runs():
    from holographic.rendering import holographic_refresh as mod
    mod._selftest()


# ---------------------------------------------------------------------------------------------------------
# THE FAKE-PERFECT BUG
# ---------------------------------------------------------------------------------------------------------

def test_the_threshold_rule_selects_everything_on_tied_ages():
    # 100% shaded, and a perfect PSNR. A perfect score achieved by doing all the work is the most dangerous kind
    # of bug: it passes every quality gate.
    tied = np.zeros((64, 64))
    k = int(0.2 * tied.size)
    assert threshold_oldest(tied, k).sum() == tied.size
    assert exact_k_oldest(tied, k).sum() == k


def test_exact_k_breaks_ties_deterministically_on_the_flat_index():
    tied = np.zeros((8, 8))
    picked = np.flatnonzero(exact_k_oldest(tied, 5).ravel())
    assert list(picked) == [0, 1, 2, 3, 4]
    # ... and the same call twice gives the same pixels
    assert np.array_equal(exact_k_oldest(tied, 5), exact_k_oldest(tied, 5))


def test_exact_k_takes_the_oldest_when_ages_differ():
    age = np.array([[0.0, 5.0], [3.0, 1.0]])
    m = exact_k_oldest(age, 2)
    assert m[0, 1] and m[1, 0]                              # ages 5 and 3
    assert not m[0, 0] and not m[1, 1]


def test_exact_k_clamps_its_budget():
    age = np.zeros((4, 4))
    assert exact_k_oldest(age, 0).sum() == 0
    assert exact_k_oldest(age, 99).sum() == 16
    assert exact_k_oldest(age, -5).sum() == 0


# ---------------------------------------------------------------------------------------------------------
# the border
# ---------------------------------------------------------------------------------------------------------

def test_the_disocclusion_border_is_on_the_side_the_camera_revealed():
    left = disocclusion_border((8, 8), 0.0, 2.0)            # content moved right -> new pixels on the left
    assert left[:, 0].all() and not left[:, -1].any()
    right = disocclusion_border((8, 8), 0.0, -2.0)
    assert right[:, -1].all() and not right[:, 0].any()
    assert disocclusion_border((8, 8), 0.0, 0.0).sum() == 0  # no motion, no news


def test_a_bigger_shift_reveals_a_wider_strip():
    small = disocclusion_border((16, 16), 0.0, -1.0).sum()
    big = disocclusion_border((16, 16), 0.0, -5.0).sum()
    assert big > small


# ---------------------------------------------------------------------------------------------------------
# THE BAR
# ---------------------------------------------------------------------------------------------------------

def test_the_bar_five_times_fewer_shader_evaluations_at_indistinguishable_quality():
    rep = refresh_report(lambda i: world(i * STEP), n_frames=12, budget=0.20, known_shift=(0.0, -STEP))
    assert abs(rep["shaded_fraction"] - 0.20) < 0.01        # the budget is honoured EXACTLY, border included
    assert rep["psnr_mean"] > 55.0                          # measured 57.5
    assert rep["psnr_worst"] > 54.0                         # measured 55.9


def test_the_reconstruction_is_stable_not_decaying():
    rep = refresh_report(lambda i: world(i * STEP), n_frames=12, budget=0.20, known_shift=(0.0, -STEP))
    assert abs(rep["tail_slope"]) < 1.0                     # measured +0.22 dB


def test_stability_is_the_tail_slope_not_first_minus_last():
    # Frame 1 warps a PERFECT frame 0 and scores well above the mean for free. Measuring "decay" from there reports
    # a collapse on a run that is in fact flat. I measured the wrong thing first; the free lunch is why.
    # (Measured at this configuration: first 62.8 dB, mean 57.5, last 56.2 -> 6.6 dB of apparent decay, tail +0.22.)
    rep = refresh_report(lambda i: world(i * STEP), n_frames=12, budget=0.20, known_shift=(0.0, -STEP))
    assert rep["psnr_first"] > rep["psnr_mean"] + 4.0       # the free lunch is real
    assert rep["psnr_first"] - rep["psnr_last"] > 4.0       # ... so first-minus-last looks alarming
    assert abs(rep["tail_slope"]) < 1.0                     # ... while the tail is flat


def test_a_full_budget_reproduces_the_reference_exactly():
    full = refresh_report(lambda i: world(i * STEP), n_frames=4, budget=1.0, known_shift=(0.0, -STEP))
    assert full["shaded_fraction"] == 1.0 and full["psnr_mean"] > 90.0


def test_a_smaller_budget_costs_quality_monotonically():
    scores = [refresh_report(lambda i: world(i * STEP), n_frames=8, budget=b,
                             known_shift=(0.0, -STEP))["psnr_mean"] for b in (0.05, 0.10, 0.20)]
    assert scores == sorted(scores)


# ---------------------------------------------------------------------------------------------------------
# THE KEPT NEGATIVES
# ---------------------------------------------------------------------------------------------------------

def test_kept_negative_recovering_the_shift_from_pixels_costs_ten_dB_and_decays():
    known = refresh_report(lambda i: world(i * STEP), n_frames=12, budget=0.20, known_shift=(0.0, -STEP))
    est = refresh_report(lambda i: world(i * STEP), n_frames=12, budget=0.20)
    assert est["psnr_mean"] < known["psnr_mean"] - 5.0      # measured 47.1 vs 57.5
    assert est["tail_slope"] < -2.0                         # measured -9.52: it slides
    assert known["tail_slope"] > -1.0                       # ... while the known shift does not


def test_kept_negative_an_integer_roll_decays_where_a_bilinear_warp_does_not():
    # Sub-pixel drift accumulates. Simulate the integer-roll variant by rounding the known shift each frame.
    r_int = RefreshRenderer(world(0.0), budget=0.20)
    r_sub = RefreshRenderer(world(0.0), budget=0.20)
    p_int, p_sub = [], []
    for i in range(1, 12):
        ref = world(i * STEP)
        oi = r_int.step(lambda _m, _r=ref: _r, known_shift=(0.0, -round(STEP)))
        os_ = r_sub.step(lambda _m, _r=ref: _r, known_shift=(0.0, -STEP))
        p_int.append(psnr(oi, ref))
        p_sub.append(psnr(os_, ref))
    assert np.mean(p_sub) > np.mean(p_int) + 5.0            # measured 57.5 vs 40.7


def test_the_loop_reports_what_it_actually_shaded():
    r = RefreshRenderer(world(0.0), budget=0.20)
    r.step(lambda _m: world(STEP), known_shift=(0.0, -STEP))
    st = r.last_stats
    assert st["shaded"] == int(round(0.20 * H * W))
    assert st["border"] > 0                                  # the border is real work, counted inside the budget
    assert st["shift"] == (0.0, -STEP)


def test_every_pixel_is_eventually_refreshed():
    # The age budget's whole purpose: nothing goes stale forever.
    r = RefreshRenderer(world(0.0), budget=0.20)
    seen = np.zeros((H, W), bool)
    for i in range(1, 12):
        ref = world(i * STEP)
        before = r.age.copy()
        r.step(lambda _m, _r=ref: _r, known_shift=(0.0, -STEP))
        seen |= (r.age == 0.0) & (before > 0.0)
    assert seen.mean() > 0.9                                 # after 11 frames at 20%, nearly everything refreshed


# ---------------------------------------------------------------------------------------------------------
# THE SCOPE -- and it is the whole caveat
# ---------------------------------------------------------------------------------------------------------

def test_the_scene_class_is_parallax_free_and_that_is_why_it_scores_57_dB():
    # A pure function of world coords: translating the grid IS the camera motion, exactly. Stating this is what
    # separates the result from an overclaim -- a real 3-D scene reprojects to ~38-41 dB at best.
    a, b = world(0.0), world(STEP)
    from holographic.rendering.holographic_reproject import warp
    assert psnr(warp(a, 0.0, -STEP), b) > 45.0               # the warp alone is nearly exact here
    assert psnr(a, b) < 30.0                                 # ... and the frames really did move


# ---------------------------------------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------------------------------------

def test_wired_to_the_mind_and_discoverable():
    import lecore
    m = lecore.UnifiedMind(dim=256, seed=0)

    tied = np.zeros((8, 8))
    assert m.exact_k_oldest(tied, 5).sum() == 5

    r = m.refresh_renderer(world(0.0), budget=0.20)
    out = r.step(lambda _m: world(STEP), known_shift=(0.0, -STEP))
    assert out.shape == (H, W) and r.last_stats["fraction"] == pytest.approx(0.20, abs=0.01)

    rep = m.refresh_report(lambda i: world(i * STEP), n_frames=6, known_shift=(0.0, -STEP))
    assert rep["psnr_mean"] > 50.0

    assert "Information-rate" in str(m.find_capability("shade fewer pixels")[:3])
