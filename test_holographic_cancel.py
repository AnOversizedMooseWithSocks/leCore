"""Modeling-app backlog item F: cooperative cancellation."""
import numpy as np
from holographic_cancel import CancelToken, run_cancellable


def test_token_flag_and_callable():
    t = CancelToken()
    assert not t.cancelled and not t.should_stop() and not t() and not bool(t)
    t.cancel(); assert t.cancelled and t.should_stop() and t() and bool(t)
    t.reset(); assert not t.cancelled


def test_run_cancellable_stops_early():
    t = CancelToken(); seen = []
    def step(i, item):
        seen.append(item)
        if item == 2: t.cancel()
    assert list(run_cancellable(range(100), t, on_step=step)) == [0, 1, 2]


def test_pathtrace_cancels_with_partial_image():
    from holographic_pathtrace import path_trace
    from holographic_sdf import sphere
    from holographic_render import Camera
    cam = Camera(eye=(0, 0, 3), target=(0, 0, 0), up=(0, 1, 0), fov_deg=45, aspect=1.0)
    t = CancelToken(); calls = []
    def prog(im, done, total):
        calls.append(done)
        if len(calls) >= 2: t.cancel()
    img = path_trace(sphere(1.0), cam, width=24, height=24, spp=12, progress_every=1,
                     on_progress=prog, should_stop=t, seed=0)
    assert len(calls) <= 3 and np.isfinite(img).all()          # stopped early, valid partial image


def test_pathtrace_backward_compatible():
    from holographic_pathtrace import path_trace
    from holographic_sdf import sphere
    from holographic_render import Camera
    cam = Camera(eye=(0, 0, 3), target=(0, 0, 0), up=(0, 1, 0), fov_deg=45, aspect=1.0)
    a = path_trace(sphere(1.0), cam, width=20, height=20, spp=4, seed=0)
    b = path_trace(sphere(1.0), cam, width=20, height=20, spp=4, seed=0, should_stop=None)
    assert np.array_equal(a, b)                                # should_stop=None == omitting it
