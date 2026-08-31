"""INSTRUMENTCOST -- what does each verification instrument COST, measured, so a session can pick one.

WHY THIS EXISTS, and it is the most expensive lesson of the render arc. Four sessions were spent guessing
instead of measuring, and the cause was not a missing instrument -- `material_preview` existed the whole
time at 8 seconds while the render it replaced took 50-140. **Instrument LATENCY, not instrument absence,
was the bug.** A verification step that costs two minutes is one a person under time pressure will skip,
and skipping it is exactly how "is the eye there" became four renders instead of one call.

So: measure the instruments the way `machine_spec_sheet` measures the compute units, and publish the
number. A session that knows `material_preview` is 8s and `render_plate` is 120s picks correctly without
having to have learned it the hard way.

WHAT THIS IS NOT: a benchmark of QUALITY. A cheap instrument that answers a different question is not a
substitute for an expensive one -- `material_preview` answers "what did the material paint", never "is
the render too bright". The `answers` field states the question each one settles, because a cost table
without it invites picking the cheap tool for the wrong question.

KEPT NEGATIVE: these are wall-clock on THIS box at a stated size, not complexity classes. They move with
resolution, catalog size and hardware, which is why the sheet RE-MEASURES rather than shipping constants
-- the same reason the machine model re-measures its tiers."""
import time


def _timed(fn, *a, **kw):
    """One warm call then one timed call -- the warm pass pays import and cache costs that would
    otherwise be attributed to the instrument."""
    try:
        fn(*a, **kw)
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, str(e)[:60])
    t0 = time.perf_counter()
    try:
        fn(*a, **kw)
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, str(e)[:60])
    return time.perf_counter() - t0, None


def instrument_costs(mind, quick=True):
    """Measure the verification instruments and report seconds + the question each one answers.

    `quick=True` skips anything expected over ~20s, so the cost sheet itself is cheap enough to run --
    an instrument catalogue that takes two minutes to produce would have the problem it documents."""
    import numpy as np
    from holographic.mesh_and_geometry.holographic_sdf import sphere

    sd = lambda P: np.asarray(sphere(0.4).translate([0, 0.5, 0]).eval(np.asarray(P, float)), float)
    mat = lambda P: (np.broadcast_to(np.array([0.7, 0.5, 0.3]), (len(P), 3)).copy(),
                     np.zeros(len(P)), np.full(len(P), 0.8), np.zeros((len(P), 3)))
    EYE, TGT = (1.2, 0.7, 1.4), (0.0, 0.5, 0.0)

    probes = [
        ("signature_of", "what ARITY does this take (nothing executes)",
         lambda: mind.signature_of(mind.quadruped_spec)),
        ("shape_of", "what does this actually RETURN",
         lambda: mind.shape_of(mind.rig_ratios)),
        ("find_capability", "does this already exist",
         lambda: mind.find_capability("smooth two shapes together")),
        ("feature_coverage", "is this feature VISIBLE, and how many pixels",
         lambda: mind.feature_coverage(sd, EYE, TGT, [("top", (0, 0.9, 0), 0.15)], 80, 60)),
        ("material_preview", "what did the material PAINT (albedo, no transport)",
         lambda: mind.material_preview(sd, EYE, TGT, mat, 120, 90)),
        ("render_gbuffer", "where are the surfaces (depth/normal/albedo)",
         lambda: mind.render_gbuffer(sd, EYE, TGT, 120, 90)),
    ]
    if not quick:
        probes.append(("render_plate", "what does it LOOK like, lit and denoised",
                       lambda: mind.render_plate(sd, EYE, TGT, mat,
                                                 mind.studio_sky("soft", backdrop=(0.02,) * 3),
                                                 width=120, height=90, tol=0.03,
                                                 min_spp=8, max_spp=24, budget_s=60)))
    rows = []
    for name, answers, fn in probes:
        secs, err = _timed(fn)
        rows.append({"instrument": name, "answers": answers,
                     "seconds": None if secs is None else round(secs, 4), "error": err})
    ok = [r for r in rows if r["seconds"] is not None]
    ok.sort(key=lambda r: r["seconds"])
    return {"rows": ok + [r for r in rows if r["seconds"] is None],
            "fastest": ok[0]["instrument"] if ok else None,
            "slowest": ok[-1]["instrument"] if ok else None,
            "spread": round(ok[-1]["seconds"] / max(ok[0]["seconds"], 1e-9), 1) if ok else None,
            "quick": bool(quick)}


def _selftest():
    import lecore
    m = lecore.UnifiedMind(dim=64, seed=0)
    rep = instrument_costs(m, quick=True)
    rows = rep["rows"]

    # 1. EVERY instrument reports a cost or an ERROR -- never a silent gap. A cost table with holes is
    #    the thing it replaces: a session guessing which tool to reach for.
    assert len(rows) >= 5
    for r in rows:
        assert (r["seconds"] is not None) or r["error"], r

    # 2. EVERY instrument states the QUESTION IT ANSWERS. A cost table without this invites picking the
    #    cheap tool for the wrong question -- material_preview cannot tell you a render is too bright.
    assert all(r["answers"] for r in rows)

    # 3. SORTED CHEAPEST FIRST, so the default read is the fast one. This is the whole ergonomic point.
    got = [r["seconds"] for r in rows if r["seconds"] is not None]
    assert got == sorted(got), got

    # 4. THE SPREAD IS REAL, and it is why the sheet exists: pure introspection versus a sphere trace
    #    differ by orders of magnitude, so "just check it" has no single meaning.
    # A RATIO IS THE WRONG HEADLINE HERE: signature_of is sub-microsecond, so the spread reads as
    # 41,000,000x -- a true number that tells a reader nothing actionable. What a session needs is the
    # ABSOLUTE seconds of each option, so the report leads with those and `spread` is a footnote.
    assert rep["spread"] is not None and rep["spread"] > 5.0, rep["spread"]
    assert rows[0]["seconds"] < 0.01, "the cheapest instrument should be effectively free"
    assert rep["fastest"] in ("signature_of", "shape_of", "find_capability"), rep["fastest"]
    print("instrumentcost selftest OK -- %d instruments, %.0fx spread, cheapest %s, "
          "every row carries the question it answers" % (len(got), rep["spread"], rep["fastest"]))


if __name__ == "__main__":
    _selftest()
