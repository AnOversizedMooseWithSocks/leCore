"""Tests for the keyframe Timeline and the tiered delta FrameCache (ANIM-2)."""

import numpy as np
from holographic_anim import Timeline, FrameCache, bake_deformation


def test_timeline_lerps_and_clamps():
    tl = Timeline().key("p", 0.0, [0.0, 0.0, 0.0]).key("p", 2.0, [2.0, 4.0, 6.0])
    assert np.allclose(tl.sample("p", 1.0), [1.0, 2.0, 3.0])  # midpoint lerp
    assert np.allclose(tl.sample("p", -5.0), [0.0, 0.0, 0.0]) # clamp before first key
    assert np.allclose(tl.sample("p", 99.0), [2.0, 4.0, 6.0]) # clamp after last key


def test_timeline_samples_a_vector_of_times():
    tl = Timeline().key("s", 0.0, 0.0).key("s", 1.0, 10.0)
    out = tl.sample("s", np.array([0.0, 0.25, 0.5, 1.0]))
    assert np.allclose(out, [0.0, 2.5, 5.0, 10.0])           # vectorised over times


def test_framecache_reconstructs_exactly():
    base = np.zeros((50, 3))
    def fr(b, f):
        s = b.copy(); s[f:f + 4, 2] = 1.0; return s
    cache = bake_deformation(base, 15, fr)
    for f in range(15):
        assert np.allclose(cache.get(f), fr(base, f))         # bit-exact reconstruction from base + delta


def test_framecache_is_smaller_for_local_change_and_hot_tier_works():
    base = np.zeros((200, 3))
    def fr(b, f):
        s = b.copy(); s[f:f + 3, 2] = 1.0; return s           # a tiny local change per frame
    cache = FrameCache(base, hot=4)
    for f in range(40):
        cache.put(f, fr(base, f))
    assert cache.full_bytes() > cache.memory_bytes()          # delta storage beats store-every-frame-full
    assert cache.get(39).shape == (200, 3)                    # recent frame served from the hot tier
    assert np.allclose(cache.get(0), fr(base, 0))             # cold frame reconstructed from its delta
