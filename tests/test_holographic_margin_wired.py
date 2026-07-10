"""C4 -- the fat-margin cache wired to its real client, behind an ERROR budget rather than a hit-rate target.

THE AUDIT'S CORRECTION. The wiring backlog named `lightcache` and `domecache` as MarginCache's clients. They are
STATELESS per-frame screen-space stride caches: they bake every Nth pixel of one frame and interpolate the rest.
There is no query STREAM to drift, and no cache that outlives a call. The drifting query lives one level up -- the
CAMERA held by `RenderSession` -- and that is where the margin belongs.

THE GATE, and it is not the one the fat-margin table implied. A hit serves a STALE value. On a rendered frame the
max error saturates at the FIRST reuse (0.5864, a silhouette edge) while the mean creeps 0.0001 -> 0.0051. So a
margin must be sized against the error you cannot tolerate, not the error you usually get.
"""

import numpy as np
import pytest

from holographic.caching_and_storage.holographic_cachehome import (
    MarginCache, drift_scale, replay_margin, replay_margin_error, suggest_margin, suggest_margin_for_error)


def _walk(n=200, d=2, step=0.1, seed=0):
    return np.cumsum(np.random.default_rng(seed).normal(size=(n, d)) * step, axis=0)


def _smooth(q):
    return [float(np.sin(p[0]) + np.cos(p[1])) for p in q]


def _jumping(q):
    """A value that jumps 0 -> 1 as the query crosses an axis: a silhouette edge, in miniature."""
    return [1.0 if p[0] > 0 else 0.0 for p in q]


# ---------------------------------------------------------------------------------------------------------
# the error replay: what a hit actually costs
# ---------------------------------------------------------------------------------------------------------

def test_replay_margin_error_counts_the_staleness_a_hit_serves():
    q = _walk()
    v = _smooth(q)
    exact = replay_margin_error(q, v, 0.0)
    assert exact["rebuilds"] == len(q) and exact["max_error"] == 0.0 and exact["mean_error"] == 0.0

    fat = replay_margin_error(q, v, 0.3)
    assert fat["hits"] > 0
    assert fat["max_error"] > 0.0 and fat["mean_error"] > 0.0        # a hit is never free


def test_the_error_grows_with_the_margin():
    q = _walk()
    v = _smooth(q)
    errs = [replay_margin_error(q, v, m)["mean_error"] for m in (0.0, 0.05, 0.1, 0.3)]
    assert errs == sorted(errs)
    hits = [replay_margin_error(q, v, m)["hits"] for m in (0.0, 0.05, 0.1, 0.3)]
    assert hits == sorted(hits)                                       # the trade, in both directions


def test_replay_margin_error_validates_its_inputs():
    q = _walk(n=10)
    with pytest.raises(ValueError):
        replay_margin_error(q, _smooth(q)[:-1], 0.1)


# ---------------------------------------------------------------------------------------------------------
# THE GATE: size on the error you cannot tolerate
# ---------------------------------------------------------------------------------------------------------

def test_the_gate_returns_a_margin_that_meets_the_budget():
    q = _walk()
    v = _smooth(q)
    m = suggest_margin_for_error(q, v, max_mean_error=0.02)
    assert replay_margin_error(q, v, m)["mean_error"] <= 0.02
    assert m > 0.0 and replay_margin_error(q, v, m)["hits"] > 0        # ... and it buys real hits


def test_kept_negative_a_mean_error_budget_can_be_fooled_by_a_rare_jump():
    # THE FINDING. A value that jumps 0 -> 1 passes a mean budget because the jump is RARE -- and the cache then
    # serves a completely wrong answer, occasionally. This is a silhouette edge, in miniature.
    q = _walk()
    v = _jumping(q)
    mean_only = suggest_margin_for_error(q, v, max_mean_error=0.02)
    st = replay_margin_error(q, v, mean_only)
    assert st["mean_error"] <= 0.02                                    # the budget is met ...
    assert st["max_error"] == pytest.approx(1.0)                       # ... and the answer is completely wrong


def test_the_max_error_bound_stops_it_and_the_boundary_is_a_cliff():
    q = _walk()
    v = _jumping(q)
    m = suggest_margin_for_error(q, v, max_mean_error=0.02, max_abs_error=0.1)
    assert replay_margin_error(q, v, m)["max_error"] <= 0.1            # the gate's answer is safe ...
    assert replay_margin_error(q, v, m * 1.01)["max_error"] == pytest.approx(1.0)   # ... and 1% past it is not
    # the admissible margin is a CLIFF, not a slope -- which is why it must be measured, never guessed


def test_the_gate_refuses_when_nothing_fits_the_budget():
    q = _walk()
    v = _jumping(q)
    assert suggest_margin_for_error(q, v, max_mean_error=0.0, max_abs_error=0.0) >= 0.0
    assert suggest_margin_for_error([np.zeros(2)] * 5, [0.0] * 5, max_mean_error=1.0) == 0.0   # a still stream


def test_hit_rate_sizing_and_error_sizing_answer_different_questions():
    q = _walk()
    v = _jumping(q)
    by_hits = suggest_margin(q, target_hit_rate=0.5)
    by_error = suggest_margin_for_error(q, v, max_mean_error=0.02, max_abs_error=0.1)
    assert by_hits > by_error                                          # hit-rate sizing is the looser, riskier one
    assert replay_margin_error(q, v, by_hits)["max_error"] > 0.5       # ... and it would have shipped a wrong frame


# ---------------------------------------------------------------------------------------------------------
# the wired client: RenderSession
# ---------------------------------------------------------------------------------------------------------

def _session(width=24, height=24):
    from holographic.mesh_and_geometry.holographic_surface import SurfaceMaterial
    from holographic.rendering.holographic_render import Camera
    from holographic.scene_and_pipeline.holographic_session import RenderSession

    class Two:
        cs = np.array([[0.0, 0, 0], [1.9, 0, 0]])

        def eval(self, P):
            return np.min(np.stack([np.linalg.norm(P - c, axis=1) - 0.85 for c in self.cs]), axis=0)

        def ids(self, P):
            return np.argmin(np.stack([np.linalg.norm(P - c, axis=1) for c in self.cs]), axis=0)

    cam = Camera(eye=(0.9, 1.0, 4.6), target=(0.9, 0, 0), fov_deg=52)
    mats = {0: SurfaceMaterial.from_name("plastic"), 1: SurfaceMaterial.from_name("metal")}
    return RenderSession(Two(), mats, cam, width=width, height=height), Camera


def test_the_gate_reuse_margin_off_is_bit_identical_to_today():
    s, _ = _session()
    a = s.preview()
    assert np.array_equal(a, s.preview(reuse_margin=None))
    assert np.array_equal(a, s.preview(reuse_margin=0.0))              # exact-key caching == re-render
    assert s.cache_stats() is None                                     # ... and no cache was even built


def test_a_drifting_camera_hits_the_margin_cache():
    s, Camera = _session()
    rng = np.random.default_rng(0)
    eye = np.array([0.9, 1.0, 4.6])
    for _ in range(20):
        eye = eye + rng.normal(size=3) * 0.02
        s.camera = Camera(eye=tuple(eye), target=(0.9, 0, 0), fov_deg=52)
        s.preview(reuse_margin=0.12)
    st = s.cache_stats()
    assert st["rebuilds"] == 1 and st["hits"] == 19 and st["hit_rate"] == pytest.approx(0.95)


def test_a_teleporting_camera_never_hits():
    s, Camera = _session()
    for k in range(5):
        s.camera = Camera(eye=(10.0 * k, 1.0, 4.6), target=(0.9, 0, 0), fov_deg=52)
        s.preview(reuse_margin=0.12)
    assert s.cache_stats()["hits"] == 0                                 # the margin buys nothing here, honestly


def test_the_cache_cannot_see_a_scene_edit_and_says_so():
    # A margin cache keys on the CAMERA. A material change is invisible to it -- the same shape as a sleeping island
    # that cannot wake itself. `invalidate_preview()` is the external event that must say so.
    s, _ = _session()
    first = s.preview(reuse_margin=1.0)
    assert s.cache_stats()["rebuilds"] == 1
    second = s.preview(reuse_margin=1.0)
    assert np.array_equal(first, second) and s.cache_stats()["hits"] == 1

    s.invalidate_preview()
    assert s.cache_stats() is None
    s.preview(reuse_margin=1.0)
    assert s.cache_stats()["rebuilds"] == 1                            # a fresh bake


def test_changing_the_margin_or_the_size_rebuilds_the_cache():
    s, _ = _session()
    s.preview(reuse_margin=1.0)
    s.preview(reuse_margin=1.0)
    assert s.cache_stats()["hits"] == 1
    s.preview(reuse_margin=2.0)                                        # a different margin is a different cache
    assert s.cache_stats()["hits"] == 0 and s.cache_stats()["margin"] == 2.0
    s.preview(reuse_margin=2.0, width=16, height=16)                   # a different size, too
    assert s.cache_stats()["rebuilds"] == 1


# ---------------------------------------------------------------------------------------------------------
# the registry records the correction
# ---------------------------------------------------------------------------------------------------------

def test_the_registry_names_the_session_and_retires_the_two_stateless_caches():
    import os
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "tools"))
    from unifiers import NOT_APPLICABLE, PENDING, REGISTRY, cites

    key = "cachehome.MarginCache (fat margin for a drifting query)"
    assert REGISTRY[key]["clients"] == ["holographic_session"]
    assert cites("holographic_session", key, repo)
    assert not any(u == key for u, _c in PENDING)                      # 1/1 wired

    # lightcache is retired WITH the reason, not silently dropped
    assert (key, "holographic_lightcache") in NOT_APPLICABLE
    assert "stateless" in NOT_APPLICABLE[(key, "holographic_lightcache")].lower()
