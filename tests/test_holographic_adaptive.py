"""Adaptive render orchestrator: the decision layer adapts to scene + workload; render_adaptive executes it."""
import numpy as np
from holographic.misc.holographic_adaptive import plan_render, render_adaptive, _reflect_of


def test_bake_decision_tracks_break_even():
    small = [{"material": "matte"}] * 3
    big = [{"material": "matte"}] * 30
    assert plan_render(small)["bake"] is None                    # tiny still -> no bake (bake would lose)
    assert plan_render(big)["bake"] is not None                  # complex -> bake amortises
    assert plan_render(small, frames=8)["bake"] is not None      # animation -> bake amortises even for few objects


def test_relax_stays_off_by_default():
    assert plan_render([{"material": "matte"}])["relax"] == 1.0   # measurement set over-relaxation off by default


def test_method_derived_from_material():
    mixed = [{"material": "matte"}, {"material": "mirror"}, {"material": "plastic"}, {"material": "glossy"}]
    p = plan_render(mixed, relight=True)
    assert p["path"] == "dispatch"
    assert p["methods"][0] == "collapse"                         # matte -> collapse
    assert p["methods"][1] == "trace"                            # mirror -> trace
    assert p["methods"][2] == "collapse"                         # plastic (0.12) -> collapse
    assert p["methods"][3] == "trace"                            # glossy (0.35) -> trace


def test_single_frame_uses_trace_path():
    p = plan_render([{"material": "matte"}, {"material": "mirror"}])
    assert p["path"] == "trace" and p["methods"] is None         # single frame -> render_scene's own material dispatch


def test_plan_reasons_present():
    p = plan_render([{"material": "matte"}] * 20, relight=True)
    assert "bake" in p["reasons"] and "shade" in p["reasons"] and "relax" in p["reasons"]   # legible automation


def test_render_adaptive_executes_both_paths():
    from holographic.simulation_and_physics.holographic_semantic import parse_description
    from holographic.rendering.holographic_render import Camera
    cam = Camera(eye=(0, 1, 5), target=(0.6, 0, 0), fov_deg=55)
    objs = parse_description("a mirror ball beside a red ball")["objects"]
    # single frame
    f1, r1, p1 = render_adaptive(objs, cam, 48, 48)
    assert f1.shape == (48, 48, 3) and r1 is None and p1["path"] == "trace"
    # relight -> dispatch path with a working relight handle
    warm = lambda w: np.clip(w @ np.array([0.4, 0.7, 0.3]), 0, 1)[:, None] * np.ones(3) + 0.05
    cool = lambda w: np.clip(w @ np.array([-0.5, 0.4, 0.2]), 0, 1)[:, None] * np.ones(3) + 0.05
    f2, r2, p2 = render_adaptive(objs, cam, 48, 48, relight=True, light=warm)
    assert f2.shape == (48, 48, 3) and p2["path"] == "dispatch" and callable(r2)
    assert not np.allclose(f2, r2(cool))                         # relight through the adaptive handle
