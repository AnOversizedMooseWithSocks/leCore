"""holographic_inverserender.py -- ANALYSIS-BY-SYNTHESIS: render -> compare -> adjust (inverse-rendering IR4, pt 2).

The headline of the inverse-rendering backlog: the auto-calibration loop. Given a TARGET image, recover the scene
parameters that reproduce it -- here the CAMERA (a turntable orbit) and the SUN direction -- by rendering a
hypothesis, comparing it to the target with the perceptual metric (IR4 part 1, NOT pixel MSE), and ADJUSTING the
parameters to reduce the difference. Autodiff is banned by the constitution -- the exact reason differentiable
renderers (Mitsuba 3, nvdiffrast) can't be borrowed -- so the search is GRADIENT-FREE: a compass / pattern search
(try each parameter +/- a step; move to the best neighbour; shrink the step when stuck). Cheap because the shipped
SDF renderer at low resolution is ~4 ms a frame, so hundreds of candidate renders cost well under a second.

The measurable milestone (the honest way to validate inverse rendering without a real photo) is the SELF-RECOVERY
test: render a KNOWN scene, hand the pixels back, and recover the camera + sun direction within tolerance -- because
the ground truth is known going in, the error is exact. A conformal-style ACCEPT/ABSTAIN gate then reports whether
the final match is confidently good, or whether the loop should abstain (the scene it cannot match).

KEPT NEGATIVES (loud): (1) gradient-free is COARSER and SLOWER than differentiable inverse rendering, and it leans
on a decent warm start (IR3's perception seed / analog recall) to land in the right basin -- from a wild init it can
stall in a local minimum. (2) The objective's ceiling is the perceptual metric's: SSIM-structural, not a learned
perceptual loss. (3) It matches the VISIBLE frame; occluded geometry and absolute metric depth are not recoverable
(see IR6) -- it recovers the viewpoint and light, not the unseen back of the object. NumPy + stdlib only;
deterministic (fixed sampling, seeded).
"""
import numpy as np

from holographic.rendering.holographic_render import Camera
from holographic.rendering.holographic_raymarch import render_sdf
from holographic.io_and_interop.holographic_imagecompare import perceptual_distance


# Parameter vector: [cam_azimuth, cam_elevation, cam_radius, light_azimuth, light_elevation].
# The camera orbits a fixed target on a turntable; the sun is a direction on the sphere. Five scalars -- enough to
# pin the viewpoint and the key light, few enough for a gradient-free search to handle.
PARAM_NAMES = ("cam_az", "cam_el", "cam_radius", "light_az", "light_el")
_DEFAULT_SCALE = np.array([0.30, 0.30, 0.40, 0.30, 0.30])   # per-parameter step magnitudes (radians / world units)


def _sph_to_vec(az, el):
    """A unit direction from azimuth/elevation (elevation measured from the horizon, azimuth around +y)."""
    ce = np.cos(el)
    return np.array([ce * np.sin(az), np.sin(el), ce * np.cos(az)])


def params_to_camera_light(params, target=(0.0, 0.0, 0.0), fov_deg=50.0):
    """Turn a parameter vector into a Camera (orbiting `target`) and a sun light direction."""
    az, el, radius, laz, lel = params
    target = np.asarray(target, float)
    eye = target + max(0.5, float(radius)) * _sph_to_vec(az, el)     # keep the camera outside the object
    cam = Camera(eye=eye, target=target, up=(0, 1, 0), fov_deg=fov_deg)
    return cam, _sph_to_vec(laz, lel)


def render_params(sdf, params, width=32, height=32, target=(0.0, 0.0, 0.0), fov_deg=50.0,
                  base_color=(0.85, 0.5, 0.35)):
    """Render the scene for a parameter vector -- the 'synthesis' half. Effects off (ao/shadows/reflect) so it is
    fast and the shading is clean Lambert (which is what carries the light-direction signal)."""
    cam, light_dir = params_to_camera_light(params, target=target, fov_deg=fov_deg)
    return render_sdf(sdf, cam, width=width, height=height, light_dir=tuple(light_dir),
                      base_color=base_color, ao=False, shadows=False, reflect=0.0)


def image_objective(sdf, params, target_img, **render_kw):
    """The objective the loop MINIMIZES: the perceptual distance between the render for `params` and the target."""
    return perceptual_distance(render_params(sdf, params, **render_kw), target_img)


def compass_search(objective, x0, scale=_DEFAULT_SCALE, step0=1.0, shrink=0.5, min_step=0.06, max_evals=500):
    """A readable GRADIENT-FREE optimizer (compass / pattern search). Try each coordinate at +step and -step
    (scaled per-parameter); greedily move to the first neighbour that improves; when no neighbour improves, shrink
    the step and refine. Stops at a small step or an evaluation budget. Returns (best_x, best_f, n_evals)."""
    x = np.array(x0, float)
    fx = objective(x)
    evals = 1
    step = step0
    scale = np.asarray(scale, float)
    while step >= min_step and evals < max_evals:
        improved = False
        for i in range(len(x)):
            for s in (step, -step):
                cand = x.copy()
                cand[i] += s * scale[i]
                fc = objective(cand)
                evals += 1
                if fc < fx - 1e-9:                           # a real improvement -> take it and move on
                    x, fx = cand, fc
                    improved = True
                    break
                if evals >= max_evals:
                    break
            if evals >= max_evals:
                break
        if not improved:                                     # stuck at this scale -> refine the step
            step *= shrink
    return x, fx, evals


def calibrate_accept_threshold(sdf, truth_params, alpha=0.1, n=12, jitter=0.05, seed=0, **render_kw):
    """A conformal-style ACCEPT threshold: render the true scene, then render a handful of TINY perturbations of it
    and measure their perceptual distances to the target -- these are the distances a 'confidently good' match
    produces. The (1-alpha) quantile of those is the accept threshold. A recovered distance at or below it is a
    confident match; above it, the loop should ABSTAIN. (Split-conformal in spirit: calibrate 'what good looks
    like' on known-good samples, then gate.)"""
    rng = np.random.default_rng(seed)
    target = render_params(sdf, truth_params, **render_kw)
    dists = []
    for _ in range(n):
        p = np.array(truth_params, float) + jitter * rng.standard_normal(len(truth_params))
        dists.append(perceptual_distance(render_params(sdf, p, **render_kw), target))
    dists = np.sort(dists)
    # the conformal (1-alpha) quantile with the finite-sample correction
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(dists[k])


def recover_scene(sdf, target_img, init_params, accept_threshold=None, seed=0, **kw):
    """The full analysis-by-synthesis loop: from an initial guess (a warm start), gradient-free-search the camera +
    sun parameters that MINIMIZE the perceptual distance to the target, then gate. Returns a dict with the recovered
    params, the final distance, the evaluation count, and accepted/abstained (if an accept_threshold is given)."""
    render_kw = {k2: v for k2, v in kw.items() if k2 in ("width", "height", "target", "fov_deg", "base_color")}
    obj = lambda p: image_objective(sdf, p, target_img, **render_kw)
    best, dist, evals = compass_search(obj, init_params, **{k2: v for k2, v in kw.items()
                                                            if k2 in ("scale", "step0", "shrink", "min_step",
                                                                      "max_evals")})
    out = {"params": best, "distance": float(dist), "evals": evals, "init_distance": float(obj(np.asarray(init_params, float)))}
    if accept_threshold is not None:
        out["accepted"] = bool(dist <= accept_threshold)
        out["abstained"] = not out["accepted"]
        out["accept_threshold"] = float(accept_threshold)
    return out


def _selftest():
    """SELF-RECOVERY: render a known box scene, then recover the camera + sun direction from a perturbed warm start
    to within tolerance -- the perceptual distance collapses and the parameters land near the truth. The conformal
    gate ACCEPTS the good match, and ABSTAINS on a target that cannot be matched (a different scene). Deterministic."""
    from holographic.mesh_and_geometry.holographic_sdf import box
    sdf = box(1.0, 0.7, 0.5)
    rkw = dict(width=32, height=32, fov_deg=50.0)

    truth = np.array([0.6, 0.4, 4.0, -0.6, 0.5])            # cam az/el/radius, light az/el
    target = render_params(sdf, truth, **rkw)

    # a warm start: perturbed from the truth (this is what IR3's perception seed / analog recall would provide)
    init = truth + np.array([0.3, -0.25, 0.7, 0.35, -0.3])
    res = recover_scene(sdf, target, init, **rkw, max_evals=500)

    assert res["distance"] < 0.25 * res["init_distance"]    # the match improved dramatically
    assert res["distance"] < 0.05                           # and is a good match in absolute terms
    # camera viewpoint recovered within tolerance (the identifiable part); light within a looser tolerance
    err = np.abs(res["params"] - truth)
    assert err[0] < 0.15 and err[1] < 0.15 and err[2] < 0.6  # cam az / el / radius
    assert err[3] < 0.4 and err[4] < 0.4                    # light az / el (coarser -- shading is a softer cue)

    # the conformal gate: ACCEPT the recovered match, ABSTAIN on a target no camera/light can reproduce
    thr = calibrate_accept_threshold(sdf, truth, **rkw)
    good = recover_scene(sdf, target, init, accept_threshold=thr, **rkw, max_evals=500)
    assert good["accepted"]
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    sphere_target = render_params(sphere(1.0), truth, **rkw)   # a SPHERE -- the box hypothesis cannot become it
    hard = recover_scene(sdf, sphere_target, init, accept_threshold=thr, **rkw, max_evals=300)
    assert hard["abstained"]                                # best match still above threshold -> honestly abstain

    # deterministic
    r2 = recover_scene(sdf, target, init, **rkw, max_evals=500)
    assert np.array_equal(res["params"], r2["params"])

    print("holographic_inverserender selftest OK: SELF-RECOVERY -- from a perturbed warm start the loop drove the "
          "perceptual distance %.3f -> %.3f (%.0f%% down) and recovered the camera az/el/radius to |err| "
          "(%.2f, %.2f, %.2f) and the sun to (%.2f, %.2f); the conformal gate ACCEPTED the good match (threshold "
          "%.3f); gradient-free, deterministic"
          % (res["init_distance"], res["distance"], 100 * (1 - res["distance"] / res["init_distance"]),
             err[0], err[1], err[2], err[3], err[4], thr))


if __name__ == "__main__":
    _selftest()
