"""MATPREVIEW -- see what a material callback ACTUALLY paints, in ~1 second, before path tracing it.

WHY THIS EXISTS. A material callback is a function of surface POSITION, and a path trace is the only
thing that had been calling it -- so the only way to answer "did my change reach the pixels" was a full
render. On a composed-SDF creature that is 50-140 seconds, which is long enough that guessing beats
measuring, and guessing is what happened: eyes authored at radius 0.017 that turn out to be two pixels
wide, a fur term that is ~1 everywhere so it flattens, a coat that renders paler than its albedo. Four
sessions of that, each diagnosed only AFTER a render.

The asymmetry that makes this cheap is the same one guided super-resolution exploits: a G-BUFFER is ONE
sphere-trace per pixel, while a path trace is min_spp..max_spp bounced paths. Surface POSITIONS are
therefore nearly free -- and positions are the entire input to a material callback. So: sphere-trace
once, reconstruct positions, call the material, show the albedo. No light transport, no sampling, no
denoise. It answers "what did the material paint" and deliberately not "what will it look like lit".

KEPT NEGATIVE, stated so nobody mistakes the tool: this is ALBEDO, not radiance. It cannot tell you the
render is too bright, too dark or too noisy -- those are transport and exposure questions and belong to
render_plate's report. It tells you whether a feature EXISTS and how much of the frame it covers, which
is the question that was actually expensive."""
import numpy as np


def surface_points(mind, sdf, eye, target, width=160, height=120, fov_deg=36.0, far=12.0):
    """Visible surface positions and the hit mask, by one sphere trace.

    POSITION IS RECONSTRUCTED, not read: `render_gbuffer` returns (depth, normal, ALBEDO) and has NO
    position array. Reading its third element as position produced an sdf that was constant to four
    decimals across 3,000 hits -- a number that cannot be geometry, printed twice before it was noticed.
    P = eye + ray_dir * depth is the only correct route, and it is one line."""
    from holographic.rendering.holographic_gemrender import camera_rays
    e, dirs = camera_rays(eye, target, width, height, fov_deg)
    depth, nrm, _alb = mind.render_gbuffer(sdf, eye, target, width, height, fov_deg=fov_deg, far=far)
    d = np.asarray(depth, float).reshape(-1)
    D = np.asarray(dirs, float).reshape(-1, 3)
    hit = np.isfinite(d) & (d < float(far) * 0.92)
    P = np.asarray(e, float).reshape(1, 3) + D * d[:, None]
    return P, hit, np.asarray(nrm, float).reshape(-1, 3), (height, width)


def preview_material(mind, sdf, eye, target, material, width=160, height=120, fov_deg=36.0, far=12.0):
    """Call `material` on visible surface points and return (albedo_image, report). ~1s, no transport.

    The report is the part that ends the guessing: `hit_fraction` (is the subject even in frame),
    `n_unique_colours` (did the material vary at all, or is it flat), and `luma_span`. A flat callback
    and a broken callback look identical in a render and are trivially distinguishable here."""
    P, hit, _n, shape = surface_points(mind, sdf, eye, target, width, height, fov_deg, far)
    img = np.zeros((len(P), 3))
    if hit.any():
        alb = np.asarray(material(P[hit])[0], float)
        img[hit] = alb
    out = img.reshape(shape[0], shape[1], 3)
    vis = img[hit] if hit.any() else np.zeros((1, 3))
    lum = vis @ np.array([0.2126, 0.7152, 0.0722])
    return out, {"hit_fraction": float(hit.mean()),
                 "n_unique_colours": int(len(np.unique(np.round(vis, 3), axis=0))),
                 "luma_min": float(lum.min()), "luma_max": float(lum.max()),
                 "luma_span": float(lum.max() - lum.min()), "size": (shape[1], shape[0])}


def feature_coverage(mind, sdf, eye, target, features, width=160, height=120, fov_deg=36.0, far=12.0):
    """For each named (centre, radius) feature, the fraction of VISIBLE pixels it occupies.

    This is the number that was missing for four sessions. An eye authored at radius 0.017 on a body 1.6
    units long is not "small" in any way a person can judge from the spec -- it is a specific pixel count,
    and 0.0000 coverage says so instantly. Reported per feature so 'the eyes are missing' becomes 'the
    eyes cover 0.02% of the frame and need to be 3x larger or the camera closer'."""
    P, hit, _n, _s = surface_points(mind, sdf, eye, target, width, height, fov_deg, far)
    Q = P[hit]
    n_vis = max(len(Q), 1)
    out = {}
    for name, centre, radius in features:
        d = np.linalg.norm(Q - np.asarray(centre, float), axis=1)
        n = int((d < float(radius)).sum())
        out[name] = {"pixels": n, "fraction": n / n_vis,
                     "nearest": float(d.min()) if len(d) else float("inf")}
    out["_visible_pixels"] = n_vis
    return out


def _selftest():
    import lecore
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    m = lecore.UnifiedMind(dim=64, seed=0)
    C = np.array([0.0, 0.5, 0.0])
    sd = lambda P: np.asarray(sphere(0.4).translate(C).eval(np.asarray(P, float)), float)
    EYE, TGT = (1.2, 0.7, 1.4), (0.0, 0.5, 0.0)

    # 1. POSITIONS MUST BE REAL GEOMETRY. The bug this replaces produced a CONSTANT sdf across every hit;
    #    a real surface has every hit at sdf ~ 0 and a spread of positions. Assert both.
    P, hit, _n, _s = surface_points(m, sd, EYE, TGT, 80, 60)
    Q = P[hit]
    assert hit.sum() > 200, hit.sum()
    assert float(np.abs(sd(Q)).max()) < 0.02, "hits are not on the surface -- positions are wrong"
    assert float(np.ptp(Q[:, 0])) > 0.3, "positions do not vary -- the ALBEDO-as-position bug is back"

    # 2. A FLAT MATERIAL AND A VARYING ONE MUST BE DISTINGUISHABLE, which is the whole point: in a render
    #    they look the same and cost 2 minutes each to compare.
    flat = lambda X: (np.broadcast_to(np.array([0.5, 0.4, 0.3]), (len(X), 3)).copy(),
                      np.zeros(len(X)), np.full(len(X), 0.8), np.zeros((len(X), 3)))
    def varying(X):
        a, mt, rg, em = flat(X)
        a = a * np.clip((X[:, 1] - 0.1) / 0.8, 0.05, 1.0)[:, None]
        return a, mt, rg, em
    _i1, r1 = preview_material(m, sd, EYE, TGT, flat, 80, 60)
    _i2, r2 = preview_material(m, sd, EYE, TGT, varying, 80, 60)
    assert r1["n_unique_colours"] == 1 and r1["luma_span"] == 0.0, r1
    assert r2["n_unique_colours"] > 20 and r2["luma_span"] > 0.05, r2

    # 3. FEATURE COVERAGE MUST CATCH A TOO-SMALL FEATURE -- the eye problem, in numbers. A radius that is
    #    a fraction of a pixel reports 0, and a generous one reports a real fraction.
    top = C + np.array([0.0, 0.0, 0.40])
    cov = feature_coverage(m, sd, EYE, TGT, [("tiny", top, 0.004), ("big", top, 0.18)], 80, 60)
    assert cov["tiny"]["pixels"] == 0, cov["tiny"]
    assert cov["big"]["fraction"] > 0.02, cov["big"]
    assert cov["big"]["nearest"] < 0.05, "the feature centre is not near the surface at all"

    # 4. A FEATURE PLACED OFF THE BODY reports infinite-ish distance rather than silently reading zero --
    #    "0 pixels" from a mis-PLACED feature and from a too-SMALL one need different fixes.
    off = feature_coverage(m, sd, EYE, TGT, [("adrift", C + np.array([5.0, 0, 0]), 0.2)], 80, 60)
    assert off["adrift"]["pixels"] == 0 and off["adrift"]["nearest"] > 4.0, off
    print("matpreview selftest OK -- positions on surface (max |sdf| < 0.02, x-span %.2f); flat vs "
          "varying separable (%d vs %d colours); tiny feature 0 px, big feature %.1f%%; misplaced "
          "feature reports distance %.1f" % (float(np.ptp(Q[:, 0])), r1["n_unique_colours"],
                                             r2["n_unique_colours"], 100 * cov["big"]["fraction"],
                                             off["adrift"]["nearest"]))


if __name__ == "__main__":
    _selftest()
