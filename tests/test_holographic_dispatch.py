"""Composability of calculation methods: per-element operator dispatch, on-the-fly switching, method resolution."""
import numpy as np
from holographic.scene_and_pipeline.holographic_dispatch import dispatch_field, resolve_methods


def test_per_element_dispatch():
    x = np.arange(8.0)
    tags = np.array(["A", "B"] * 4)
    out = dispatch_field(x, tags, {"A": lambda v: v * 10, "B": lambda v: v + 100})
    assert np.allclose(out[tags == "A"], x[tags == "A"] * 10)
    assert np.allclose(out[tags == "B"], x[tags == "B"] + 100)


def test_on_the_fly_switch():
    # a 'mirror' method that reflects then dispatches the rest to 'collapse' -- a method switch mid-computation
    def mirror(sub):
        return dispatch_field(-sub, np.array(["collapse"] * len(sub)), {"collapse": lambda v: v * 0.5})
    y = dispatch_field(np.array([2.0, 4.0, 6.0, 8.0]), np.array(["mirror", "collapse", "mirror", "collapse"]),
                       {"mirror": mirror, "collapse": lambda v: v * 0.5})
    assert np.allclose(y, [-1.0, 2.0, -3.0, 4.0])


def test_dispatch_preserves_trailing_shape():
    x = np.arange(12.0).reshape(6, 2)                             # vector-valued elements
    tags = np.array(["A", "B", "A", "B", "A", "B"])
    out = dispatch_field(x, tags, {"A": lambda v: v * 2, "B": lambda v: v + 1})
    assert out.shape == (6, 2)
    assert np.allclose(out[0], x[0] * 2) and np.allclose(out[1], x[1] + 1)


def test_resolve_methods_table_and_region_override():
    ids = np.array([0, 1, 2, 0])
    tags = resolve_methods(ids, {0: "collapse", 1: "mirror", 2: "glossy"})
    assert list(tags) == ["collapse", "mirror", "glossy", "collapse"]
    class RF:
        def method_at(self, points, default=None):
            return np.array(["mirror", None, None, None], dtype=object)   # override only the first
    tags2 = resolve_methods(ids, {0: "collapse", 1: "mirror", 2: "glossy"},
                            points=np.zeros((4, 3)), region_field=RF())
    assert tags2[0] == "mirror" and tags2[2] == "glossy"         # region overrides object 0 only


def test_missing_op_raises():
    try:
        dispatch_field(np.arange(4.0), np.array(["A", "A", "B", "B"]), {"A": lambda v: v})
        assert False
    except KeyError:
        pass


def test_bake_scene_first_render_is_a_relight():
    """bake_scene precomputes visibility + transfer so render_baked is a pure relight -- the first frame equals what
    render_dispatch produces, and relighting a second light changes only the shading (no re-trace)."""
    import numpy as np
    from holographic.scene_and_pipeline.holographic_dispatch import bake_scene, render_baked, render_dispatch
    from holographic.rendering.holographic_render import Camera
    class S:
        cs = np.array([[0, 0, 0], [-1.6, 0, 0]]); cols = np.array([[0.7, 0.7, 0.7], [0.8, 0.3, 0.3]])
        def eval(s, P): return np.min(np.stack([np.linalg.norm(P - c, axis=1) - 0.8 for c in s.cs]), axis=0)
        def ids(s, P): return np.argmin(np.stack([np.linalg.norm(P - c, axis=1) for c in s.cs]), axis=0)
    cam = Camera(eye=(0, 1, 5), target=(0, 0, 0), fov_deg=52)
    warm = lambda w: np.clip(w @ np.array([0.4, 0.7, 0.3]), 0, 1)[:, None] * np.ones(3) + 0.05
    cool = lambda w: np.clip(w @ np.array([-0.5, 0.4, 0.2]), 0, 1)[:, None] * np.ones(3) + 0.05
    baked = bake_scene(S(), cam, 50, 50, {0: "trace", 1: "collapse"}, S.cols)   # precompute BEFORE any render
    first = render_baked(baked, warm)                                            # first frame is already a relight
    disp, _, _ = render_dispatch(S(), cam, 50, 50, {0: "trace", 1: "collapse"}, S.cols, warm)
    assert np.allclose(first, disp)                                              # baked path == dispatch path
    assert not np.allclose(first, render_baked(baked, cool))                     # relight from the same bake, no re-trace
    assert baked.info["collapse"] > 0 and baked.info["trace"] > 0
