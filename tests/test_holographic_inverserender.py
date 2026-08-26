"""Inverse-rendering IR4 (part 2): the analysis-by-synthesis loop -- gradient-free camera + sun recovery."""
import numpy as np
from holographic.mesh_and_geometry.holographic_sdf import box, sphere
from holographic.rendering.holographic_inverserender import render_params, image_objective, compass_search, recover_scene, calibrate_accept_threshold, params_to_camera_light

RKW = dict(width=32, height=32, fov_deg=50.0)
TRUTH = np.array([0.6, 0.4, 4.0, -0.6, 0.5])
INIT = TRUTH + np.array([0.3, -0.25, 0.7, 0.35, -0.3])


def _sdf():
    return box(1.0, 0.7, 0.5)


def test_self_recovery_camera_and_light():
    sdf = _sdf()
    target = render_params(sdf, TRUTH, **RKW)
    res = recover_scene(sdf, target, INIT, **RKW, max_evals=500)
    assert res["distance"] < 0.05                           # good absolute match
    assert res["distance"] < 0.25 * res["init_distance"]    # big improvement
    err = np.abs(res["params"] - TRUTH)
    assert err[0] < 0.15 and err[1] < 0.15 and err[2] < 0.6   # camera az/el/radius recovered
    assert err[3] < 0.4 and err[4] < 0.4                    # sun az/el recovered (coarser)


def test_compass_search_reduces_objective():
    sdf = _sdf()
    target = render_params(sdf, TRUTH, **RKW)
    obj = lambda p: image_objective(sdf, p, target, **RKW)
    best, fbest, evals = compass_search(obj, INIT, max_evals=300)
    assert fbest < obj(INIT) and evals > 0


def test_gate_accepts_good_match():
    sdf = _sdf()
    target = render_params(sdf, TRUTH, **RKW)
    thr = calibrate_accept_threshold(sdf, TRUTH, **RKW)
    res = recover_scene(sdf, target, INIT, accept_threshold=thr, **RKW, max_evals=500)
    assert res["accepted"] and not res["abstained"]


def test_gate_abstains_on_unmatchable():
    sdf = _sdf()
    thr = calibrate_accept_threshold(sdf, TRUTH, **RKW)
    sphere_target = render_params(sphere(1.0), TRUTH, **RKW)   # box can't reproduce a sphere
    res = recover_scene(sdf, sphere_target, INIT, accept_threshold=thr, **RKW, max_evals=300)
    assert res["abstained"]


def test_params_to_camera_light():
    cam, ld = params_to_camera_light([0.0, 0.0, 3.0, 0.0, 0.5])
    assert np.allclose(np.linalg.norm(cam.eye - np.array([0, 0, 0])), 3.0)   # radius honored
    assert abs(np.linalg.norm(ld) - 1.0) < 1e-9             # unit light direction


def test_deterministic():
    sdf = _sdf()
    target = render_params(sdf, TRUTH, **RKW)
    a = recover_scene(sdf, target, INIT, **RKW, max_evals=300)
    b = recover_scene(sdf, target, INIT, **RKW, max_evals=300)
    assert np.array_equal(a["params"], b["params"])
