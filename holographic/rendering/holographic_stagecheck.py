"""STAGECHECK -- assert each render stage's contract in one pass, because end-to-end testing missed four.

WHY, with the receipts. Four defects shipped in this render arc, every one of them a STAGE that had never
been measured in isolation, and every one found only by a human looking at a picture:

  1. `tonemap_fixed` had a white point and NO EXPOSURE. The formula looked plausible and did nothing --
     doubling white moved the plate mean 0.444 -> 0.424.
  2. `render_plate` had no `budget_s`. First real use ran past 16 minutes with no output.
  3. `render_plan` was handed the OUTPUT size while upsample traces at 1/N, over-estimating by N^2 and
     shrinking 560x420 to 162x122 for a render that then took 50s.
  4. `gbuffer(albedo_fn=None)` gave SVGF a FLAT albedo guide, so the denoiser smoothed across every
     texture the material painted -- four sessions of "fur does not work".

All four passed every end-to-end test, because an end-to-end test only asks "did an image come out".
Each check below asserts the EFFECT of one stage, never its implementation -- the standing lesson from
those four, which is that reading the code back is exactly how they survived.

KEPT NEGATIVE: this verifies MECHANISM, not beauty. Every check can pass on a render that looks wrong,
because "does it look right" is a question for an eye and `material_preview` at 0.044s, not for an
assertion."""
import numpy as np


def verify_render_stages(mind, sdf, eye, target, material, sky, width=96, height=72, budget_s=60):
    """Run each stage's contract and return a per-stage verdict. Small by default -- a verifier that
    costs a full render is one nobody runs, which is the same trap the instrument-cost sheet documents.

    Returns {checks: [{stage, ok, detail}], all_ok, n_failed}."""
    from holographic.rendering.holographic_plate import tonemap_fixed, highlight_fraction, suggest_white
    import time

    checks = []

    def add(stage, ok, detail):
        checks.append({"stage": stage, "ok": bool(ok), "detail": detail})

    # ---- 1. TONEMAP HAS AN EXPOSURE, and it must bite HARDER than the white point at normal levels.
    # Defect 1 shipped a tonemap whose only knob did nothing; asserting the EFFECT catches that, while
    # asserting "the function exists" would have passed.
    lin = np.abs(np.random.default_rng(0).standard_normal((16, 16, 3))) * 0.6
    d_exp = abs(float(tonemap_fixed(lin, white=2.0, exposure=2.0).mean())
                - float(tonemap_fixed(lin, white=2.0, exposure=0.25).mean()))
    d_wht = abs(float(tonemap_fixed(lin, white=0.5).mean()) - float(tonemap_fixed(lin, white=4.0).mean()))
    add("tonemap_exposure", d_exp > d_wht,
        "exposure moves %.3f vs white %.3f -- exposure must dominate" % (d_exp, d_wht))

    # ---- 2. THE BUDGET IS HONOURED. Defect 2 was a missing cap; the contract is wall-clock, so measure
    # wall-clock rather than checking the parameter is accepted.
    t0 = time.perf_counter()
    img, rep = mind.render_plate(sdf, eye, target, material, sky, width=width, height=height,
                                 tol=0.04, min_spp=4, max_spp=16, budget_s=budget_s)
    elapsed = time.perf_counter() - t0
    add("budget_honoured", elapsed <= budget_s * 2.5,
        "%.1fs against a %.0fs budget (2.5x slack for probe overhead)" % (elapsed, budget_s))

    # ---- 3. THE PLAN COSTS THE TRACE, NOT THE DELIVERABLE. Defect 3 lost 4x resolution because the
    # planner was shown the output size while upsample traces at 1/N.
    _iu, rep_u = mind.render_plate(sdf, eye, target, material, sky, width=width * 2, height=height * 2,
                                   tol=0.04, min_spp=4, max_spp=16, budget_s=budget_s, upsample=2)
    tr = rep_u.get("traced_at")
    ok3 = tr is not None and rep_u["width"] == tr[0] * 2 and rep_u["height"] == tr[1] * 2
    add("plan_costs_trace", ok3, "output %sx%s from a %s trace" % (rep_u["width"], rep_u["height"], tr))

    # ---- 4. THE DENOISER'S GUIDE CARRIES THE MATERIAL. Defect 4: a FLAT albedo guide tells a joint
    # bilateral "this is all one surface", so it smooths across every texture. Check the guide VARIES.
    from holographic.rendering.holographic_gemrender import gbuffer, camera_rays
    e_, d_ = camera_rays(eye, target, width, height, 36.0)
    _dep, _nrm, alb = gbuffer(sdf, e_, d_, far=12.0,
                              albedo_fn=lambda P: np.asarray(material(P)[0], float))
    a = np.asarray(alb, float).reshape(-1, 3)
    spread = float(np.ptp(a @ np.array([0.2126, 0.7152, 0.0722])))
    add("denoise_guide_varies", spread > 1e-6,
        "albedo guide luminance spread %.4f (flat guide erases texture)" % spread)

    # ---- 5. THE REPORT MEASURES, rather than asserting. A pipeline that cannot say whether it blew out
    # is one where "is it blown out" becomes an opinion, which cost three sessions.
    add("report_measures_clipping", "highlight_fraction" in rep,
        "highlight_fraction %.3f" % rep.get("highlight_fraction", -1))

    n_bad = sum(1 for c in checks if not c["ok"])
    return {"checks": checks, "all_ok": n_bad == 0, "n_failed": n_bad}


def _selftest():
    import lecore
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    m = lecore.UnifiedMind(dim=64, seed=0)
    sd = lambda P: np.asarray(sphere(0.4).translate([0, 0.5, 0]).eval(np.asarray(P, float)), float)

    def mat(P):
        P = np.asarray(P, float); n = len(P)
        t = np.clip((P[:, 1] - 0.1) / 0.8, 0, 1)[:, None]      # VARYING, so check 4 is meaningful
        return (np.array([0.2, 0.15, 0.1]) + np.array([0.5, 0.4, 0.25]) * t,
                np.zeros(n), np.full(n, 0.85), np.zeros((n, 3)))

    sky = m.studio_sky("soft", backdrop=(0.02,) * 3)
    rep = verify_render_stages(m, sd, (1.2, 0.7, 1.4), (0.0, 0.5, 0.0), mat, sky,
                               width=64, height=48, budget_s=60)
    assert len(rep["checks"]) == 5
    for c in rep["checks"]:
        assert c["detail"], c
    assert rep["all_ok"], [c for c in rep["checks"] if not c["ok"]]

    # THE VERIFIER MUST BE ABLE TO FAIL. A check that cannot fail is decoration -- the exact lesson from
    # the dead `assert True` conditional in the metered seam. Feed it a FLAT material and check 4 must
    # go red while the others stay green: a stage verifier that fails everything is equally useless.
    flat = lambda P: (np.broadcast_to(np.array([0.4, 0.4, 0.4]), (len(P), 3)).copy(),
                      np.zeros(len(P)), np.full(len(P), 0.85), np.zeros((len(P), 3)))
    bad = verify_render_stages(m, sd, (1.2, 0.7, 1.4), (0.0, 0.5, 0.0), flat, sky,
                               width=64, height=48, budget_s=60)
    guide = [c for c in bad["checks"] if c["stage"] == "denoise_guide_varies"][0]
    assert guide["ok"] is False, "a FLAT albedo guide must fail the guide check -- defect 4 by hand"
    assert bad["n_failed"] == 1, ("only the guide check should fail on a flat material",
                                  [c["stage"] for c in bad["checks"] if not c["ok"]])
    print("stagecheck selftest OK -- 5 stage contracts pass on a healthy pipeline; a FLAT albedo guide "
          "fails exactly one check (defect 4 reproduced by hand), so the verifier can fail")


if __name__ == "__main__":
    _selftest()
