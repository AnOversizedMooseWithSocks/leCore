"""STUDIORIG -- a three-point studio as an EMISSIVE ENVIRONMENT, not as three small lamps.

WHY THIS SHAPE AND NOT scene_light() x3, which is the obvious thing and is wrong here. path_trace's own
kept negative says it plainly: "no next-event estimation, so light is gathered only when a bounce hits
the emissive environment -- great for a big sky, VERY NOISY for small emitters (NEE/MIS is the next
step)". A studio softbox IS a small emitter, so building the rig from point/rect lights fights the
renderer's single documented weakness and buys noise for nothing.

The fix is not a better sampler, it is a better description of the same physics. A studio softbox is
already a LARGE source subtending a wide solid angle -- that is what makes its shadows soft and what
"studio" means optically. Expressed as sky(D) -> radiance with broad cosine lobes, the same rig lands in
exactly the regime path_trace converges in. Play the negative, do not fight it.

THE RIG (the standard three-point, and the ratios are the convention, not invented here):
  KEY   -- high, off to one side, brightest; sets the form and the shadow direction.
  FILL  -- opposite, low, and WEAK. Its job is to lift the shadow side without erasing it; a fill that
           matches the key is flat light, which is the failure mode this ratio exists to avoid.
  RIM   -- behind and high, aimed back at camera; separates the subject from the backdrop.
  plus a small ambient floor and a dimmer downward BOUNCE term, which a real sweep/cyc gives you.

KEPT NEGATIVE: this is an ENVIRONMENT, so it lights the subject but casts no light onto a floor that is
not in the SDF -- there is no backdrop unless the scene contains one. A "studio" render off a bare
creature field is a subject floating in graded space, which is a lighting result and NOT a set.
"""
import numpy as np

#: name -> (key_intensity, fill_intensity, rim_intensity, key_tightness). Ratios, not absolutes.
PRESETS = {
    # ~4:1 key:fill -- the textbook portrait ratio: modelled form, shadow side still readable.
    "classic": (7.0, 1.6, 5.0, 0.55),
    # ~1.8:1 -- beauty/e-comm: soft, low contrast, big sources.
    "soft": (5.5, 3.0, 3.5, 0.85),
    # ~10:1 -- dramatic/low key: deep shadow side, strong separation.
    "dramatic": (9.0, 0.9, 7.0, 0.35),
}


def studio_sky(preset="classic", key_dir=(0.62, 0.66, 0.42), fill_dir=(-0.70, 0.30, 0.55),
               rim_dir=(-0.35, 0.55, -0.76), warmth=0.06, ambient=0.22, bounce=0.30, gain=1.0,
               backdrop=None, backdrop_falloff=1.6):
    """Build a `sky(D) -> (n,3)` radiance function for path_trace: a three-point studio as environment.

    `preset` picks the key:fill:rim ratios (see PRESETS) -- 'classic' ~4:1, 'soft' ~1.8:1,
    'dramatic' ~10:1. The three `*_dir` vectors aim the lobes in world space (they are normalised for
    you). `warmth` tints the key warm and the fill cool by that fraction, which is what a gelled key over
    a daylight fill does and is most of why studio light reads as studio. `ambient` is the floor radiance,
    `bounce` the dimmer downward term a sweep gives back, `gain` scales the whole rig.

    Returns a closure taking directions of ANY leading shape (n,3) or (H,W,3) and returning matching
    radiance -- path_trace calls it on escaped rays, so it must broadcast, not assume a flat list.

    `backdrop` FIXES THE GREY FOG, and the fog was a real bug in the first version, not a taste problem.
    A sky serves TWO jobs at once: it lights the subject (via bounce directions) AND it is what the camera
    sees on a miss. The first version returned the same radiance for both, so `ambient=0.22` painted the
    entire frame flat mid-grey and every render came out looking hazed. A real studio does not work that
    way -- the softboxes are OUT OF FRAME and the camera sees a dark, gently graded sweep behind the
    subject. Passing `backdrop=(r,g,b)` (or a scalar) makes CAMERA-FACING directions return that graded
    sweep while the lighting lobes keep illuminating exactly as before: the rig is unchanged, only what
    the lens sees is. `backdrop_falloff` sets how fast the sweep darkens downward; None restores the old
    all-lobes-visible behaviour.

    KEPT NEGATIVE: still lights the SUBJECT only. `backdrop` paints what the camera sees on a miss; it is
    not geometry, so nothing catches a shadow. A cast shadow needs a real floor in the SDF."""
    if preset not in PRESETS:
        raise ValueError("unknown preset %r -- have %s" % (preset, sorted(PRESETS)))
    k_i, f_i, r_i, tight = PRESETS[preset]
    w = float(warmth)
    lobes = []
    for d, inten, tint, sharp in ((key_dir, k_i, (1.0 + w, 1.0, 1.0 - w), tight),
                                  (fill_dir, f_i, (1.0 - w, 1.0, 1.0 + w), min(1.0, tight * 1.8)),
                                  (rim_dir, r_i, (1.0, 1.0, 1.0 + w * 0.5), tight * 1.2)):
        v = np.asarray(d, float)
        v = v / (np.linalg.norm(v) + 1e-12)
        # exponent from tightness: a WIDE lobe (tight->1) is a big softbox, a narrow one is a snoot.
        lobes.append((v, float(gain) * inten * np.asarray(tint, float), 6.0 / max(float(sharp), 1e-3)))
    amb = np.asarray([ambient] * 3, float) * float(gain)
    bnc = np.asarray([bounce] * 3, float) * float(gain)

    bd = None if backdrop is None else np.broadcast_to(
        np.asarray(backdrop, float) if np.ndim(backdrop) else np.asarray([backdrop] * 3, float), (3,)).copy()
    fall = float(backdrop_falloff)

    def sky(D):
        D = np.asarray(D, float)
        n = np.linalg.norm(D, axis=-1, keepdims=True)
        U = D / (n + 1e-12)                       # NORMALISE: path_trace hands us unit dirs, but a caller
        # THE BASE IS THE ONLY THING THE BACKDROP REPLACES. `sky(D)` is called for BOTH camera rays that
        # miss and bounce rays that escape -- path_trace cannot tell them apart, and neither can this. So
        # the fix is not to swap the whole function on a flag; it is to make the FLAT part dark and graded
        # while the LOBES keep their full intensity. A flat `ambient` everywhere was the fog; the lobes
        # were never the problem. (First version replaced `out` outright and took the key light with it:
        # the frame went clean and the subject went black at lum 0.136. Fixing a bug by deleting the
        # feature next to it is not a fix.)
        if bd is None:
            base = np.broadcast_to(amb, U.shape).copy()
        else:
            up = np.clip(U[..., 1] * 0.5 + 0.5, 0.0, 1.0)[..., None]
            base = bd * (up ** fall) + bd * 0.12
        out = base
        for v, rgb, expo in lobes:
            c = np.clip(U @ v, 0.0, 1.0)
            out = out + rgb * (c ** expo)[..., None]
        # THE BOUNCE FLOOR MUST TRACK THE BACKDROP. A dark sweep bounces dark light; leaving the default
        # bounce (0.30) under a 0.045 backdrop made straight-DOWN brighter than straight-UP, i.e. a floor
        # glowing more than the sky above it. Caught by the cyclorama assertion, which is why it asserts
        # the gradient direction rather than an absolute level.
        b_use = bnc if bd is None else bd * 0.35
        down = np.clip(-U[..., 1], 0.0, 1.0)[..., None]
        out = out * (1.0 - down) + b_use * down
        return out
    return sky


def rig_ratios(preset="classic"):
    """The measured key:fill and key:rim ratios of a preset, as plain data -- so a caller can state the
    lighting ratio it used rather than describing it as 'nice'."""
    k, f, r, tight = PRESETS[preset]
    return {"preset": preset, "key": k, "fill": f, "rim": r, "key_tightness": tight,
            "key_fill_ratio": round(k / f, 2), "key_rim_ratio": round(k / r, 2)}


def _selftest():
    # 1. SHAPE CONTRACT: path_trace calls sky() on escaped-ray directions, so it must broadcast over any
    #    leading shape and return matching (..., 3). Getting this wrong is a crash mid-render.
    sky = studio_sky("classic")
    for shp in ((7, 3), (4, 5, 3)):
        D = np.random.default_rng(0).standard_normal(shp)
        out = sky(D)
        assert out.shape == shp, (shp, out.shape)
        assert np.all(np.isfinite(out)) and np.all(out >= 0.0)

    # 2. THE KEY MUST DOMINATE, and the fill must NOT erase the shadow side. A rig whose fill matches its
    #    key is flat light -- the exact failure the 4:1 convention exists to prevent, asserted not assumed.
    k = np.array([[0.62, 0.66, 0.42]]); k /= np.linalg.norm(k)
    f = np.array([[-0.70, 0.30, 0.55]]); f /= np.linalg.norm(f)
    lk, lf = float(sky(k).mean()), float(sky(f).mean())
    assert lk > lf * 2.0, "key does not dominate fill (%.3f vs %.3f) -- this is flat light" % (lk, lf)

    # 3. THE PRESETS MUST ACTUALLY DIFFER IN CONTRAST, or they are three names for one rig.
    ratios = [rig_ratios(p)["key_fill_ratio"] for p in ("soft", "classic", "dramatic")]
    assert ratios == sorted(ratios), ratios
    assert ratios[-1] > ratios[0] * 3, ratios
    c_soft = float(studio_sky("soft")(k).mean() / studio_sky("soft")(f).mean())
    c_dram = float(studio_sky("dramatic")(k).mean() / studio_sky("dramatic")(f).mean())
    assert c_dram > c_soft, (c_soft, c_dram)

    # 4. WARMTH is real: the key lobe must be warmer (R>B) and the fill cooler (B>R). This is most of why
    #    studio light reads as studio, so a warmth that does nothing is a silently broken knob.
    ck, cf = sky(k)[0], sky(f)[0]
    assert ck[0] > ck[2], ck
    assert cf[2] > cf[0], cf

    # 5. DOWNWARD directions get the dim bounce, not the key -- a floor term that outshone the key would
    #    invert the whole rig.
    dn = float(sky(np.array([[0.0, -1.0, 0.0]])).mean())
    assert dn < lk, (dn, lk)

    # 6. BACKDROP SEPARATES WHAT THE LENS SEES FROM WHAT LIGHTS THE SUBJECT. The first version returned
    #    one radiance for both, so ambient painted the whole frame flat grey and every render looked
    #    hazed -- a real bug, found by inspecting an image, not a taste call. With a backdrop the camera
    #    sees a dark graded sweep while the key lobe keeps its full intensity.
    lit = studio_sky("classic")
    bg = studio_sky("classic", backdrop=(0.05, 0.05, 0.055))
    kk = np.array([[0.62, 0.66, 0.42]]); kk /= np.linalg.norm(kk)
    horiz = np.array([[1.0, 0.0, 0.0]])
    # THE FRAME GOES DARK where no lobe points -- that is the fog gone.
    assert float(bg(horiz).mean()) < 0.25, "backdrop is still fogging the frame"
    assert float(lit(horiz).mean()) > float(bg(horiz).mean()) * 1.5, "backdrop changed nothing"
    # ...AND THE KEY MUST SURVIVE IT. The first version replaced the whole radiance and took the key with
    # it (subject rendered at lum 0.136). Pinning this stops that regression returning.
    assert float(bg(kk).mean()) > float(bg(horiz).mean()) * 5.0, \
        "the backdrop swallowed the KEY LIGHT -- it must replace only the flat ambient base"
    assert float(bg(kk).mean()) > 0.5 * float(lit(kk).mean()), "key intensity dropped with a backdrop on"
    assert float(bg(np.array([[0.0, 1.0, 0.0]])).mean()) > float(bg(np.array([[0.0, -1.0, 0.0]])).mean()), \
        "the sweep must fall off DOWNWARD, like a cyclorama"

    # 7. gain scales linearly; determinism (no rng anywhere in the closure).
    assert np.allclose(studio_sky("classic", gain=2.0)(k), 2.0 * sky(k))
    assert np.allclose(sky(k), studio_sky("classic")(k))
    print("studiorig selftest OK -- broadcasts (n,3) and (H,W,3); key:fill %.1f:1 dominates; presets "
          "differ in contrast (%.1f -> %.1f); key warm / fill cool; bounce below key; gain linear"
          % (rig_ratios()["key_fill_ratio"], c_soft, c_dram))


if __name__ == "__main__":
    _selftest()
