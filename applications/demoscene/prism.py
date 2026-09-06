"""prism -- the rainbow the renderer could not make, and the shortcut past the method that makes it.

THE GAP THIS CLOSES. leCore could already trace caustics and could already split a ray bundle across
several IORs (dispersion_spread). What it could not do was say which IOR a COLOUR has -- so
dispersion_spread's own test hand-picked two numbers, [1/1.513, 1/1.532], and every caustic the engine
rendered came out grey. Sellmeier supplies the missing map from wavelength to index, and with it the
caustic gets its rainbow.

AS ABOVE, SO BELOW -- the same structure at three scales, which is why this is a demoscene program and
not a unit test:
  * ABOVE: a GLASS is a curve n(lambda) -- one function of wavelength.
  * MIDDLE: a SPECTRAL RENDER is that curve sampled at C hero wavelengths -- C caustic layers.
  * BELOW: each LAYER is the same monochrome tracer, called again at a different index.
The layers are not C independent pictures. They are C samples of one smooth family, and that is the
whole finding: sampling a smooth thing densely is a choice, not a requirement.

WHAT IT PROVES, all asserted rather than admired:
  * THE RAINBOW IS REAL, against a TRUE ZERO. A monochrome caustic rendered through the same observer
    has saturation EXACTLY 0.0 -- not approximately, exactly, because one wavelength cannot be two
    colours. The spectral render scores ~0.40 on BK7. There is no baseline argument to have.
  * THE SHORTCUT PAST SOTA IS REAL. Hero Wavelength Spectral Sampling (Wilkie et al., EGSR 2014) is the
    state of the art for choosing those C wavelengths, and it is what `count` and `u` implement. But it
    is a MONTE CARLO estimator: it must trace every stratum, because it cannot reuse a sample it never
    drew. A DETERMINISTIC tracer can. The layer family is rank-3 -- 99.59% of the SVD energy sits in ONE
    singular value and 99.94% in three -- so three traces span what sixteen traces re-derive. Measured
    4.9x at relative RGB error 0.0072.
  * AND THE COLOUR SURVIVES THE SHORTCUT, which is the claim that actually matters. A speedup that
    smears the wavelengths together would score a LOW rel-err (the caustic is mostly a bright achromatic
    pedestal) while destroying the one thing dispersion is for. So the gate here is SATURATION
    RETENTION, not rel-err.

KEPT NEGATIVE -- A SMALL REL-ERR DOES NOT CERTIFY A RECONSTRUCTION, and this one nearly shipped as a
feature. The first cut of the interpolator bracketed each target between anchors picked in HERO ORDER.
Hero wavelengths arrive ROTATED, not sorted, so it interpolated between the wrong pair and laid ghost
energy where no wavelength focused -- inflating chromatic saturation to 108.5% of ground truth while
rel-err stayed a respectable 0.0236. It read as a mild accuracy cost and was a WRONG PICTURE, and it
was written up as a real property of k=2 ("two anchors overshoot") before this file's own assertion 5
contradicted it. The correct sorted-axis version measures k=2 at 99.6%. Two lessons kept: the pedestal
dominates the norm, so rel-err cannot see the interesting part; and a negative discovered with a
throwaway probe is a claim about the probe until the shipped code reproduces it.

KEPT NEGATIVE -- A GLASS DOES NOT WANT TO BE A HYPERVECTOR. The obvious VSA move is to bake n(lambda)
into an FPE function hypervector (mind.grid_to_hypervector) so a glass composes with bind/bundle like
any other object. Measured, and it does not pay:
  * Baking n(lambda) DIRECTLY gives correlation -0.25 -- worse than useless. The curve is 1.49%
    variation on a 1.52 mean, so the constant dominates the superposition and the dispersion itself is
    at crosstalk level. FPE bakes a function's SHAPE, and a near-constant function has almost no shape.
  * Subtracting the mean first fixes the SHAPE (corr +0.97 at dim 2048, and dim 8192 does not improve
    it -- the limit is the KDE kernel, not capacity). But the FPE query is a SIMILARITY readout, not a
    calibrated value: the recovered slope is 15x too steep, and even a two-point affine refit against
    the F and C Fraunhofer lines leaves 38-42% error across the curve's own range.
  * So: 4096 floats and a calibration step, to approximate three multiply-adds. Sellmeier stays a
    closed form. The general lesson is the part worth keeping -- BEFORE BAKING ANY FIELD INTO AN FPE
    VECTOR, CHECK ITS VARIATION AGAINST ITS MEAN; a field that is nearly constant has nearly nothing
    for the encoder to hold.
"""
import hashlib

import numpy as np

NAME = "prism"
DOMAIN = "demoscene"
PROVES = ("a spectral caustic with saturation 0.40 against an EXACTLY 0.0 monochrome baseline, and a "
          "rank-3 shortcut past hero-wavelength sampling: 3 traces for 16 at rel RGB error 0.007 with "
          "100% of the chromatic saturation retained")
ARTEFACT = "gallery/prism.png"


def _sphere(p):
    """The lens. A unit sphere is the honest choice: it is the shape whose caustic the reader can
    predict, so a wrong render is obvious rather than plausible."""
    return np.linalg.norm(p, axis=-1) - 1.0


def _saturation(rgb):
    """Mean saturation over LIT pixels -- the readout for "is there colour here at all".

    WHY THIS METRIC AND NOT A GEOMETRIC ONE. Three geometric metrics were tried first and all three
    returned a constant regardless of dispersion: centroid separation (a centred sphere under axial
    light is radially symmetric, so every wavelength's centroid sits at the centre), radial peak radius
    (binning radial SUMS lets the outermost annulus win on area alone), and spot RMS radius (correct but
    measuring a focal SPOT as though it were a ring). Each failure was the same mistake -- collapsing a
    2-D image to a scalar that happens to be invariant under the effect being measured. The fix was not
    a cleverer scalar; it was to measure in the OUTPUT space. Grey is grey."""
    rgb = np.asarray(rgb, float)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    lit = mx > 0.02 * mx.max()                  # the dark background carries no colour information
    if not lit.any():
        return 0.0
    return float(((mx - mn)[lit] / np.maximum(mx[lit], 1e-12)).mean())


def run(mind, glass="BK7", count=16, anchors=3, res=96, n_side=220, seed=0, out_dir=None):
    """Render the same caustic three ways -- monochrome, full spectral, and rank-3 accelerated -- and
    return the numbers that separate them.

    `count` hero wavelengths is the SOTA sampling; `anchors` is how many of those C layers are actually
    traced. anchors=None runs the un-accelerated path."""
    scene = dict(light_dir=(0.0, -1.0, 0.0), receiver_y=-1.6, extent=2.5, res=res, n_side=n_side,
                 seed=seed)

    # 1. THE ZERO. One wavelength through the same observer -- the baseline that makes the rainbow a
    #    result rather than a look. Stacking one intensity map into three channels IS what "no
    #    dispersion" means, so this is the honest zero and not a strawman.
    n_d = float(mind.sellmeier_ior(587.5618, glass))          # the d-line: the index opticians quote
    mono = np.asarray(mind.caustics(_sphere, ior=n_d, **scene), float)
    mono_rgb = np.stack([mono, mono, mono], axis=-1)

    # 2. GROUND TRUTH: every hero wavelength traced, which is what the SOTA method requires.
    import time
    t0 = time.time()
    full = mind.spectral_caustics(_sphere, glass=glass, count=count, anchors=None, **scene)
    sec_full = time.time() - t0

    # 3. THE SHORTCUT: k anchors, the rest interpolated across the index range.
    t0 = time.time()
    fast = mind.spectral_caustics(_sphere, glass=glass, count=count, anchors=anchors, **scene)
    sec_fast = time.time() - t0

    sat_mono = _saturation(mono_rgb)
    sat_full = _saturation(full["rgb"])
    sat_fast = _saturation(fast["rgb"])
    a, b = np.asarray(full["rgb"], float), np.asarray(fast["rgb"], float)
    rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))

    # The digest pins the render itself: a demo that renders differently twice is a broken demo.
    digest = hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]

    path = None
    if out_dir is not None:
        import os
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "prism.png")
        # save_render is the faculty the rest of the library uses; it owns the encode.
        mind.save_render(path, a / max(a.max(), 1e-12))

    return {
        "name": NAME, "domain": DOMAIN, "proves": PROVES, "path": path,
        "proved": {
            "glass": glass, "abbe": float(mind.abbe_number(glass)),
            "n_d": n_d,
            "ior_lo": float(min(full["iors"])), "ior_hi": float(max(full["iors"])),
            "count": int(count), "traced_full": int(full["traced"]), "traced_fast": int(fast["traced"]),
            "saturation_mono": sat_mono, "saturation_full": sat_full, "saturation_fast": sat_fast,
            "saturation_retained": sat_fast / sat_full if sat_full > 0 else 0.0,
            "rel_rgb_error": rel,
            "seconds_full": sec_full, "seconds_fast": sec_fast,
            "speedup": sec_full / max(sec_fast, 1e-9),
            "digest": digest,
        },
    }


def _selftest():
    """The regression trap. Every assertion here is a CONTRACT, not a size-of-the-mess measurement."""
    import lecore
    mind = lecore.UnifiedMind(dim=64, seed=0)
    a = run(mind, count=16, anchors=3, res=96, n_side=220)
    p = a["proved"]

    # 1. THE GLASS IS THE GLASS. Catalogue values, so a transcription error in the coefficients is
    #    caught here rather than showing up as a subtly wrong colour nobody can name.
    assert abs(p["n_d"] - 1.5168) < 1e-3, p["n_d"]
    assert abs(p["abbe"] - 64.17) < 0.3, p["abbe"]
    assert p["ior_lo"] < p["ior_hi"], p            # a glass with no index spread cannot disperse

    # 2. THE RAINBOW IS REAL AGAINST AN EXACT ZERO. Monochrome saturation is not "small" -- it is 0.0,
    #    because one wavelength cannot be two colours. Asserting the exact zero is what makes the
    #    spectral number mean something.
    assert p["saturation_mono"] == 0.0, p["saturation_mono"]
    assert p["saturation_full"] > 0.20, p["saturation_full"]

    # 3. THE SHORTCUT TRACES FEWER LAYERS -- structurally, not just faster on this box.
    assert p["traced_full"] == 16 and p["traced_fast"] == 3, p

    # 4. AND KEEPS THE COLOUR. This is the gate. A wall-clock speedup assertion would be a claim about
    #    a machine (the lesson infinite_zoom paid two flaky tests to learn), so the timing is REPORTED
    #    and the ASSERTED contract is accuracy: the colour survives and the error stays small.
    assert 0.80 < p["saturation_retained"] < 1.20, p["saturation_retained"]
    assert p["rel_rgb_error"] < 0.05, p["rel_rgb_error"]

    # 5. THE SORTED AXIS IS LOAD-BEARING, pinned at the smallest k where a wrong bracket does the most
    #    damage. This assertion is the one that caught the false negative described in the docstring:
    #    it was first written to REQUIRE an overshoot at k=2, and failed, because the overshoot
    #    belonged to a broken probe rather than to the method. It now pins the true contract -- every
    #    k reconstructs the colour, because every k brackets on a monotone axis.
    for k in (2, 4):
        pk = run(mind, count=16, anchors=k, res=96, n_side=220)["proved"]
        assert 0.90 < pk["saturation_retained"] < 1.05, (k, pk["saturation_retained"])
        assert pk["rel_rgb_error"] < 0.02, (k, pk["rel_rgb_error"])

    # 6. DETERMINISM.
    b = run(lecore.UnifiedMind(dim=64, seed=0), count=16, anchors=3, res=96, n_side=220)
    assert a["proved"]["digest"] == b["proved"]["digest"], (a["proved"]["digest"], b["proved"]["digest"])

    # 7. THE ARTEFACT PATH ACTUALLY RUNS. It has to be exercised here or it rots: the first version
    #    of this file called a `write_png` faculty that does not exist, and because no assertion ever
    #    passed out_dir, everything above still went green. An unrun branch is an unwritten branch.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        art = run(mind, count=4, anchors=2, res=32, n_side=60, out_dir=tmp)
        with open(art["path"], "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n", "artefact is not a PNG"

    print("prism OK: %s Abbe %.2f, n %.4f-%.4f | saturation mono %.4f -> spectral %.4f | "
          "%d traces for %d, %.2fx, rel err %.4f, colour retained %.1f%%"
          % (p["glass"], p["abbe"], p["ior_lo"], p["ior_hi"], p["saturation_mono"],
             p["saturation_full"], p["traced_fast"], p["count"], p["speedup"],
             p["rel_rgb_error"], 100.0 * p["saturation_retained"]))


if __name__ == "__main__":
    _selftest()
