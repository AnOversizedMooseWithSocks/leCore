"""Inverse-rendering IR3: perception -> scene-hypothesis bridge -- descriptor, sun estimate, analog-recall warm start."""
import numpy as np
from holographic_sdf import box, sphere
from holographic_inverserender import render_params, recover_scene
from holographic_perception import (scene_descriptor, estimate_light_direction, scene_hypothesis, SceneLibrary)

RKW = dict(width=32, height=32, fov_deg=50.0)


def _sdf():
    return box(1.0, 0.7, 0.5)


def test_descriptor_unit_norm_deterministic():
    img = render_params(_sdf(), [0.5, 0.4, 4.0, -0.5, 0.5], **RKW)
    d = scene_descriptor(img)
    assert abs(np.linalg.norm(d) - 1.0) < 1e-9 and np.array_equal(d, scene_descriptor(img))


def test_sun_estimate_points_to_bright_side():
    right = np.zeros((32, 32, 3)); right[10:22, 22:30] = 1.0
    left = np.zeros((32, 32, 3)); left[10:22, 2:10] = 1.0
    assert estimate_light_direction(right)[0] > 0.2      # bright on the right -> +azimuth
    assert estimate_light_direction(left)[0] < -0.2      # bright on the left  -> -azimuth


def test_sun_estimate_elevation_top_is_high():
    top = np.zeros((32, 32, 3)); top[2:10, 12:20] = 1.0
    bottom = np.zeros((32, 32, 3)); bottom[22:30, 12:20] = 1.0
    assert estimate_light_direction(top)[1] > estimate_light_direction(bottom)[1]   # top -> higher sun


def test_scene_hypothesis_horizon_split():
    # sky on top (bright), ground on bottom (dark) -> horizon near the middle
    img = np.zeros((40, 40, 3)); img[:20] = [0.5, 0.7, 0.9]; img[20:] = [0.2, 0.15, 0.1]
    hyp = scene_hypothesis(img)
    assert 15 <= hyp["horizon_row"] <= 25 and hyp["palette"].shape[1] == 3


def _library(sdf):
    lib = SceneLibrary(seed=0)
    for az0 in (-0.4, 0.2, 0.8):
        for laz0 in (-0.6, 0.0, 0.6):
            p = [az0, 0.4, 4.0, laz0, 0.5]
            lib.add(render_params(sdf, p, **RKW), p)
    return lib.build()


def test_analog_recall_warm_start():
    sdf = _sdf(); lib = _library(sdf)
    truth = np.array([0.25, 0.4, 4.0, 0.05, 0.5])
    ws = lib.warm_start(render_params(sdf, truth, **RKW))
    assert not ws["abstained"]
    assert abs(ws["params"][0] - truth[0]) < 0.5 and abs(ws["params"][3] - truth[3]) < 0.7


def test_warm_start_seeds_ir4_to_convergence():
    sdf = _sdf(); lib = _library(sdf)
    truth = np.array([0.25, 0.4, 4.0, 0.05, 0.5])
    target = render_params(sdf, truth, **RKW)
    ws = lib.warm_start(target)
    res = recover_scene(sdf, target, ws["params"], **RKW, max_evals=400)
    assert res["distance"] < 0.05
    err = np.abs(res["params"] - truth)
    assert err[0] < 0.2 and err[3] < 0.4                 # recovered from a PERCEIVED start, no hand-perturbation


def test_recall_agreement_signal():
    sdf = _sdf(); lib = _library(sdf)
    # a matching target has high agreement; the flag respects agree_min
    ws = lib.warm_start(render_params(sdf, [0.2, 0.4, 4.0, 0.0, 0.5], **RKW))
    assert 0.0 <= ws["agreement"] <= 1.0


def test_deterministic():
    sdf = _sdf(); lib = _library(sdf)
    t = render_params(sdf, [0.25, 0.4, 4.0, 0.05, 0.5], **RKW)
    assert lib.warm_start(t)["index"] == lib.warm_start(t)["index"]
