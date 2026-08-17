"""holographic_proccodec.py -- C-5: procedural storage (store the PROGRAM, verify, or refuse).

THE GAP (Rule-0 on record): procedural_compression MEASURES the DSL-vs-mesh ratio and stops --
no round trip; "compress by storing the program not the data" returned fallbacks plus the
ingredients (fit_deterministic, bank_or_formula's economy, the sentinel's philosophy). This
module is the round trip: fit a generator, VERIFY the regeneration against the original at a
STATED tolerance, and only then commit -- or refuse with the reason and a route hint. The
sentinel's discipline for non-streams: noise is never fake-compressed, and neither is a signal
whose fit misses the declared bar.

TWO TIERS, cheapest first (Quilez: don't pay for a climb the flat rung already covers):

  TIER 'generator'  fit_deterministic's bank (sine/chirp/gauss/sawtooth/harmonic/AM...) plus a
                    least-squares amplitude+offset (the bank fits SHAPE; scale is two floats).
                    ~100 bytes, CONSTANT IN n -- the whole point: a 100k-sample tone costs the
                    same blob as a 1k-sample one. Regeneration at ANY length; past 2x the
                    fitted window it carries valid=False (extend_generator's reprojection-ghost
                    negative, inherited verbatim -- a formula fit on t in [0,1] evaluated at
                    t=100 is confident nonsense).
  TIER 'recipes'    decompose_piecewise's per-segment Formula recipes (C-2's model head,
                    reused byte-for-byte -- no second fitter). ~300-600 bytes. Regeneration at
                    the ORIGINAL length only: each recipe lives on its segment's normalized
                    axis, so extension is undefined and REFUSED rather than extrapolated.

VERIFY-THEN-COMMIT (the load-bearing property, per tier): regenerate at full length, measure
max |err| pointwise against tol * amplitude(y). A tier that misses the bar is not stored --
the next tier runs, and when both miss, store_procedural REFUSES with mode='refused', the
measured errors, and the route: exactness wants residual_encode; ranked choices want
codec_place. fit_deterministic's own band-limited verification is NOT reused as the commit
gate, deliberately: band-limited correlation certifies the FAMILY at the snap grain, while a
storage contract is pointwise -- two different claims, and conflating them would ship blobs
that verify at a grain the caller never stated.

KEPT NEGATIVES:
  * the generator tier's pointwise bar is hard to meet for real-world signals -- the bank
    fits canonical shapes, and a few-percent shape mismatch fails a 1% tol; that is the
    DESIGN (a loose tol is the caller's declaration, not the codec's assumption);
  * tier 'recipes' cannot extend -- regenerate(n != original) raises; play-the-future
    belongs to tier 'generator' and to the HRNN's horizon discipline;
  * amplitude scaling is least-squares against the fitted shape, so a DC-heavy signal with
    a poor shape fit can pass a sloppy tol on offset alone -- the report carries both the
    error AND the tier so the caller can see what actually verified.
"""

import json
import struct
import zlib

import numpy as np

from holographic.agents_and_reasoning.holographic_fitgen import FAMILIES
from holographic.agents_and_reasoning.holographic_symbolic import Formula

_MAGIC = b"LPC1"
_TIER_GEN, _TIER_RECIPES = 1, 2


def _amp(y):
    a = float(np.abs(y - y.mean()).max())
    return a if a > 0 else 1.0


def store_procedural(y, tol=0.02, mind=None):
    """Store a 1-D signal as its PROGRAM: try the generator bank (constant-size blob,
    extendable), then piecewise recipes (small blob, original length only); each tier is
    VERIFIED pointwise at tol*amplitude before commit, and when both miss the codec REFUSES
    with the measured errors and a route hint. Returns {blob|None, report:{mode:'generator'|
    'recipes'|'refused', bytes, raw_bytes, zlib_bytes, ratio_vs_zlib, max_abs_error, tol_abs,
    family|n_segments, why|route}}. Regenerate with regen_procedural(blob[, n])."""
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
    n = len(y)
    raw = y.tobytes()
    zbase = zlib.compress(raw, 6)
    tol_abs = float(tol) * _amp(y)
    if mind is None:
        import lecore
        mind = lecore.UnifiedMind(dim=256, seed=0)

    errors = {}

    # ---- TIER 1: generator bank + LS scale/offset --------------------------------------
    # FIT ON A PREFIX, VERIFY ON THE WHOLE: the bank's snap is band-limited, so a long
    # window pushes a tone's cycle count past what the coarse band can see (MEASURED: the
    # same tone fit at n=4000 was REFUSED outright at n=16000, correlation 0.012). The
    # generator therefore lives on the prefix's [0,1] axis (timebase L, shipped in the
    # blob) and is verified pointwise against the FULL signal -- verification against real
    # data outranks any extrapolation heuristic.
    L = min(n, 4096)
    fit = mind.fit_deterministic(y[:L])
    if fit.get("family") is not None:
        tgrid = np.arange(n) / max(1, L - 1)
        # TRY EVERY TIE, keep the best verified: the snap's tie-break optimises the snap's
        # own criterion, not the storage contract (MEASURED: on a 16k tone the tie-break
        # chose 'am' with mod depth 0.5 -- 2.5 max error, a basin GN cannot leave -- while
        # the tied 'sine' polishes to 1e-3). Equifinality at the snap grain is real; the
        # pointwise verify is the arbiter here.
        candidates = [fit["family"]] + [f for f in fit.get("ties", []) if f != fit["family"]]

        # POLISH per candidate family: fit_deterministic's params are snapped to its
        # refine grid (measured: a 12.012-cycle tone came back as 12.0000 -- 0.073 pointwise
        # error on a 0.05 budget, a grid artifact, not a family error). A damped Gauss-Newton
        # on the RMS residual (numeric Jacobian; alpha/beta re-solved by LS inside each step)
        # closes the gap -- coordinate-wise golden section was tried first and CRAWLED (freq
        # and phase are strongly coupled; 3 rounds moved 0.073 -> 0.069, kept as the
        # negative). The GN step is NEGATIVE of the normal-equation solve because J is the
        # RESIDUAL's Jacobian (the first attempt used +step: every candidate was worse and
        # lambda inflated to the ceiling -- a silent no-op polish; the sign is load-bearing).
        def _polish(family):
            fn, _ = FAMILIES[family]
            # Candidate families start from the WINNER's param vector: for the periodic
            # bank families the leading slot is frequency-like, which is the coupled/hard
            # coordinate -- GN recovers phase-like slots from a rough start but not a
            # frequency off by whole cycles. A tie family whose param layout genuinely
            # differs just polishes badly and loses the min() below; the pointwise verify
            # is the arbiter, never the starting point.
            params = np.array([float(p) for p in fit["params"]])

            def _resid(p):
                shape = np.asarray(fn(tgrid, *p), dtype=float)
                A = np.column_stack([shape, np.ones(n)])
                (al, be), *_ = np.linalg.lstsq(A, y, rcond=None)
                return y - (al * shape + be), al, be, shape

            lam = 1e-3
            r0, al, be, sh = _resid(params)
            for _ in range(20):
                J = np.empty((n, len(params)))
                for i in range(len(params)):
                    h = 1e-6 * max(1.0, abs(params[i]))
                    pp = params.copy(); pp[i] += h
                    J[:, i] = (_resid(pp)[0] - r0) / h
                step = -np.linalg.solve(J.T @ J + lam * np.eye(len(params)), J.T @ r0)
                cand = params + step
                r1, a1, b1, s1 = _resid(cand)
                if (r1 ** 2).sum() < (r0 ** 2).sum():
                    params, r0, al, be, sh = cand, r1, a1, b1, s1
                    lam = max(lam * 0.5, 1e-9)
                else:
                    lam *= 4.0
                    if lam > 1e6:
                        break
            return float(np.abs(r0).max()), [float(p) for p in params], float(al), float(be)

        best = min((( _polish(fam), fam) for fam in candidates), key=lambda x: x[0][0])
        (err, params, alpha, beta), best_family = best
        errors["generator"] = err
        if err <= tol_abs:
            payload = json.dumps(dict(family=best_family,
                                      params=params,
                                      alpha=alpha, beta=beta,
                                      n=n, timebase=L), sort_keys=True).encode()
            blob = _MAGIC + struct.pack("<B", _TIER_GEN) + zlib.compress(payload, 9)
            return dict(blob=blob, report=dict(
                mode="generator", bytes=len(blob), raw_bytes=len(raw),
                zlib_bytes=len(zbase), ratio_vs_zlib=len(zbase) / len(blob),
                max_abs_error=err, tol_abs=tol_abs, family=best_family))

    # ---- TIER 2: piecewise recipes (C-2's model head, reused) --------------------------
    d = mind.decompose_piecewise(y, min_seg=64)
    recipes = [dict(segment=list(p["segment"]), recipe=p["formula"].to_recipe())
               for p in d["pieces"]]
    regen = np.zeros(n)
    for r in recipes:
        a, b = r["segment"]
        regen[a:b] = Formula.from_recipe(r["recipe"]).generate(np.linspace(0.0, 1.0, b - a))
    err = float(np.abs(y - regen).max())
    errors["recipes"] = err
    if err <= tol_abs:
        payload = json.dumps(dict(recipes=recipes, n=n), sort_keys=True).encode()
        blob = _MAGIC + struct.pack("<B", _TIER_RECIPES) + zlib.compress(payload, 9)
        return dict(blob=blob, report=dict(
            mode="recipes", bytes=len(blob), raw_bytes=len(raw),
            zlib_bytes=len(zbase), ratio_vs_zlib=len(zbase) / len(blob),
            max_abs_error=err, tol_abs=tol_abs, n_segments=len(recipes)))

    # ---- REFUSE, with the evidence and the route ---------------------------------------
    return dict(blob=None, report=dict(
        mode="refused", bytes=0, raw_bytes=len(raw), zlib_bytes=len(zbase),
        ratio_vs_zlib=0.0, max_abs_error=min(errors.values()), tol_abs=tol_abs,
        why="no tier verified within tol_abs=%.3g (generator %.3g, recipes %.3g)"
            % (tol_abs, errors.get("generator", float("inf")), errors["recipes"]),
        route="exactness -> mind.residual_encode; ranked lossless/lossy -> mind.codec_place"))


def regen_procedural(blob, n=None):
    """Regenerate a signal from its program blob. Tier 'generator' regenerates at ANY n
    (returns {samples, valid}; valid=False past 2x the fitted window -- the reprojection-
    ghost negative, inherited). Tier 'recipes' regenerates the ORIGINAL length only and
    RAISES for any other n (each recipe lives on its segment's normalized axis; extension
    is undefined, so it is refused, not extrapolated)."""
    if blob[:4] != _MAGIC:
        raise ValueError("not a procedural blob (bad magic)")
    tier, = struct.unpack("<B", blob[4:5])
    h = json.loads(zlib.decompress(blob[5:]).decode())
    n_orig = h["n"]
    if tier == _TIER_GEN:
        n_out = n_orig if n is None else int(n)
        fn, _ = FAMILIES[h["family"]]
        # the generator lives on its TIMEBASE prefix's [0,1] axis; the VERIFIED window is
        # n_orig samples, and validity extends to 2x that (the reprojection-ghost bound is
        # anchored to what was verified, not to the fit prefix).
        L = h.get("timebase", n_orig)
        t = np.arange(n_out) / max(1, L - 1)
        samples = h["alpha"] * np.asarray(fn(t, *h["params"]), dtype=float) + h["beta"]
        return dict(samples=samples, valid=bool(n_out <= 2 * n_orig))
    if n is not None and int(n) != n_orig:
        raise ValueError("tier 'recipes' regenerates the original length only "
                         "(%d); extension is undefined on per-segment axes" % n_orig)
    out = np.zeros(n_orig)
    for r in h["recipes"]:
        a, b = r["segment"]
        out[a:b] = Formula.from_recipe(r["recipe"]).generate(np.linspace(0.0, 1.0, b - a))
    return dict(samples=out, valid=True)


def _selftest():
    import lecore
    rng = np.random.default_rng(0)
    mind = lecore.UnifiedMind(dim=256, seed=0)

    # 1) TIER GENERATOR: a scaled/offset tone verifies, blob is tiny, regeneration matches.
    t = np.arange(4000.)
    tone = 2.5 * np.sin(2 * np.pi * t / 333) + 7.0
    r = store_procedural(tone, tol=0.02, mind=mind)
    assert r["report"]["mode"] == "generator", r["report"]
    assert r["report"]["max_abs_error"] <= r["report"]["tol_abs"]
    assert r["report"]["ratio_vs_zlib"] > 20, r["report"]
    g = regen_procedural(r["blob"])
    assert g["valid"] and np.abs(g["samples"] - tone).max() <= r["report"]["tol_abs"]

    # 2) CONSTANT-SIZE claim: 4x the samples, the SAME blob bytes (that is the whole point).
    tone_big = 2.5 * np.sin(2 * np.pi * np.arange(16000.) / 333) + 7.0
    r_big = store_procedural(tone_big, tol=0.02, mind=mind)
    assert r_big["report"]["mode"] == "generator"
    assert abs(r_big["report"]["bytes"] - r["report"]["bytes"]) <= 8, \
        (r["report"]["bytes"], r_big["report"]["bytes"])
    assert r_big["report"]["ratio_vs_zlib"] > 3.5 * r["report"]["ratio_vs_zlib"]

    # 3) EXTENSION with the validity flag: within 2x valid, past 2x flagged.
    e_ok = regen_procedural(r["blob"], n=6000)
    e_far = regen_procedural(r["blob"], n=20000)
    assert e_ok["valid"] and not e_far["valid"]
    truth = 2.5 * np.sin(2 * np.pi * np.arange(6000.) / 333) + 7.0
    assert np.abs(e_ok["samples"] - truth).max() <= 2 * r["report"]["tol_abs"], \
        "the formula must actually play the future it claims"

    # 4) TIER RECIPES: a 3-regime signal misses the single-generator bar, verifies on recipes.
    y3 = np.concatenate([np.sin(2 * np.pi * t[:400] / 23), 0.002 * t[400:800] - 0.3,
                         0.5 * np.cos(2 * np.pi * t[:400] / 41)])
    r3 = store_procedural(y3, tol=0.02, mind=mind)
    assert r3["report"]["mode"] == "recipes", r3["report"]
    g3 = regen_procedural(r3["blob"])
    assert np.abs(g3["samples"] - y3).max() <= r3["report"]["tol_abs"]
    try:
        regen_procedural(r3["blob"], n=999)
        assert False, "recipes tier must refuse extension"
    except ValueError:
        pass

    # 5) REFUSAL: white noise fails both tiers; the report carries errors and the route.
    rn = store_procedural(rng.standard_normal(1200), tol=0.02, mind=mind)
    assert rn["blob"] is None and rn["report"]["mode"] == "refused"
    assert "residual_encode" in rn["report"]["route"]

    # 6) Determinism.
    assert store_procedural(tone, tol=0.02, mind=mind)["blob"] == r["blob"]

    print("proccodec selftest OK -- generator %.0fx (n=4k) / %.0fx (n=16k, same blob), "
          "recipes %.1fx" % (r["report"]["ratio_vs_zlib"], r_big["report"]["ratio_vs_zlib"],
                             r3["report"]["ratio_vs_zlib"]))


if __name__ == "__main__":
    _selftest()
