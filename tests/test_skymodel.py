"""Regression traps for the parametric sky -- the contracts that make it a SKY and not a texture."""
import numpy as np
import pytest

from holographic.rendering.holographic_skymodel import sky_model, sun_direction


def _hemisphere(n=3000, seed=7):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d[:, 1] = np.abs(d[:, 1])
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def test_time_moves_both_the_palette_and_the_sun():
    """The two halves of 'time of day': midnight is much darker than noon, and the sun's DIRECTION moves --
    the same probe direction is bright at 9h and not at 15h. A sky where only brightness changed would be
    a dimmer switch, not a day."""
    d = _hemisphere(500)
    assert sky_model(12.0)(d).mean() > 8 * sky_model(0.0)(d).mean()
    probe = sun_direction(9.0)[None, :]
    assert sky_model(9.0)(probe)[0].max() > 5.0
    assert sky_model(15.0)(probe)[0].max() < 5.0


def test_stars_are_deterministic_night_only_and_occluded():
    """Three properties in one because they share machinery: same seed = same sky FOREVER (the determinism
    rule for 'random' content); noon stars are invisible (daylight fade); and a nimbostratus blanket hides
    them (celestial light obeys the cloud transmittance like everything else)."""
    d = _hemisphere()
    s1, s2 = sky_model(0.0, stars_seed=42)(d), sky_model(0.0, stars_seed=42)(d)
    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, sky_model(0.0, stars_seed=43)(d))
    assert (sky_model(12.0, stars_seed=42)(d) - sky_model(12.0)(d)).max() < 1e-6
    blanket = [("nimbostratus", 1.0)]
    leak = (sky_model(0.0, stars_seed=42, clouds=blanket)(d) - sky_model(0.0, clouds=blanket)(d)).max()
    assert leak < 0.05, "stars leak %.3f through a full blanket" % leak


def test_cloud_kinds_order_by_opacity_and_altostratus_keeps_the_disk():
    """The vocabulary's load-bearing distinction, with per-kind extinction as the mechanism: the same
    coverage of different KINDS must produce clear > milky (disk visible) > buried. This is the assertion
    that caught the shared-extinction bug -- one coefficient could not serve both ends."""
    at_sun = sun_direction(9.0)[None, :]
    clear = sky_model(9.0)(at_sun)[0].max()
    milky = sky_model(9.0, clouds=[("altostratus", 0.7)])(at_sun)[0].max()
    buried = sky_model(9.0, clouds=[("nimbostratus", 1.0)])(at_sun)[0].max()
    assert clear > milky > buried
    assert milky > 0.25 * clear, "the milky sun must stay a visible disk"
    assert buried < 0.15 * clear, "a rain blanket must effectively hide it"


def test_low_clouds_are_refused_toward_the_right_tool():
    """Cumulus has depth and self-shadowing; pretending a dome texture is a cumulus would be a worse cloud
    than the volumetric stack already ships. The refusal must NAME cloud_scene so the caller lands there."""
    with pytest.raises(ValueError, match="cloud_scene"):
        sky_model(12.0, clouds=[("cumulus", 0.5)])


def test_it_plugs_into_the_render_pipeline_and_autoexposure_caveat_is_real():
    """Cross-faculty: the sky drives a dome light and the tracer's sky= in a real preview. And the caveat
    from the measurement session is pinned: view='display' AUTO-EXPOSES, so midnight and noon come out at
    similar display means -- the honest comparison is view=None, where the ratio must be large."""
    import lecore
    m = lecore.UnifiedMind(dim=128, seed=0)
    sc = m.new_scene()
    sc.add(name="floor", geometry=m.shape("plane"), material="matte_gray")
    cam = m.camera(eye=(0.0, 1.0, 3.0), target=(0.0, 0.6, -1.0), fov_deg=50.0, aspect=4 / 3.)
    means = {}
    for tag, hour in (("noon", 12.0), ("night", 0.5)):
        sky = m.sky_model(hour=hour, stars_seed=42 if hour < 6 else None)
        L = [m.scene_light("dome", color=sky, intensity=1.0)]
        lin = np.asarray(m.render_preview(sc, cam, 32, 24, lights=L, sky=sky, view=None), float)
        means[tag] = lin.mean()
    assert means["noon"] > 8 * means["night"], \
        "linear noon/night ratio collapsed: %.4f vs %.4f" % (means["noon"], means["night"])
    assert "Parametric sky" in str(m.find_capability("night sky with stars")[0])


def test_the_sky_is_a_sphere_not_a_plane():
    """MOOSE'S REVIEW, pinned. The first version projected high clouds onto the plane y=+1 and faded them
    at the horizon -- so an 'overcast' render showed clear sky exactly where a real deck visually thickens,
    and nothing extended past the horizontal. The shell fixes both, and both are contracts now:
    (a) a full deck covers the horizon AND continues below the geometric horizontal (the shell curves over
        the earth: you see its underside beyond the horizon dip);
    (b) at partial coverage the horizon is optically heavier than the zenith (grazing path through the
        layer -- the 'depth' in a sky that is a sphere)."""
    import numpy as np
    from holographic.rendering.holographic_skymodel import sky_model

    deck = sky_model(12.0, clouds=[("nimbostratus", 0.9)])
    clear = sky_model(12.0)
    at_h = deck(np.array([[1.0, 0.0, 0.0]]))[0]
    below = deck(np.array([[0.999, -0.03, 0.0]]) / np.linalg.norm([0.999, -0.03, 0.0]))[0]
    assert np.abs(at_h - clear(np.array([[1.0, 0.0, 0.0]]))[0]).max() > 0.05, "deck missing AT the horizon"
    assert abs(float(at_h.mean() - below.mean())) < 0.05, "deck must continue BELOW the horizontal"

    part = sky_model(12.0, clouds=[("altostratus", 0.5)])
    zen = np.array([[0.0, 1.0, 0.0]])
    hor = np.array([[1.0, 0.02, 0.0]]) / np.linalg.norm([1.0, 0.02, 0.0])
    zen_dev = np.abs(part(zen) - clear(zen)).mean()
    hor_dev = np.abs(part(hor) - clear(hor)).mean()
    assert hor_dev > zen_dev, "slant thickening missing: zen %.3f hor %.3f" % (zen_dev, hor_dev)


def test_stars_have_renderable_extent():
    """MOOSE'S REVIEW, second half: the delivered night render contained no visible stars, because the
    lattice made each one ~0.06 deg -- a fraction of a pixel at any sane size. Correct radiance, invisible
    image. A star must now light MORE THAN ONE nearby direction (a core with falloff), so it survives the
    sampler instead of losing a lottery against it."""
    import numpy as np
    from holographic.rendering.holographic_skymodel import sky_model, _star_cells, _STAR_LATTICE

    # find one star cell, then probe a small angular neighbourhood around its direction
    rng = np.random.default_rng(3)
    d = rng.normal(size=(20000, 3)); d[:, 1] = np.abs(d[:, 1])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    hv, core = _star_cells(d, seed=42)
    stars = np.where((hv > 0.9985) & (core > 0.5))[0]
    assert len(stars) > 0, "no star found in 20k samples -- density machinery broken"
    # probe around the CELL CENTRE, which is where the core's falloff is anchored -- the first version
    # probed around an arbitrary in-cell sample, and when that sample sat near a cell edge the +/-eps
    # offsets crossed into neighbour (non-star) cells and read 0. That was a fragile probe failing, not
    # extent failing; extent IS falloff about the centre, so the centre is what the test must orbit.
    qc = np.floor(d[stars[0]] * _STAR_LATTICE)
    centre = (qc + 0.5) / _STAR_LATTICE
    centre /= np.linalg.norm(centre)
    eps = 0.3 / _STAR_LATTICE                                   # well inside one cell's angular width
    ring = np.stack([centre, centre + [eps, 0, 0], centre - [0, 0, eps]], axis=0)
    ring /= np.linalg.norm(ring, axis=1, keepdims=True)
    sky = sky_model(0.0, stars_seed=42)
    base = sky_model(0.0)
    lit = (sky(ring) - base(ring)).max(axis=1)
    assert (lit > 0.05).sum() >= 2, "a star must light a neighbourhood, not a single lattice point: %s" % lit


def test_new_cloud_kinds_have_structure_not_solid_color():
    """MOOSE'S FOLLOW-UP, pinned: 'add cloud types that are not going to just result in a solid color'.
    Structure means two measurable things at moderate coverage: VARIANCE across the sky (elements exist)
    and GAPS (near-clear directions between them -- a cellular sky is mostly the space between its
    elements). Bands are wide because texture tuning may drift; ZERO gaps or near-zero variance is the
    regression this exists to catch. The sheet kinds are exempt from the gap test on purpose: a veil
    legitimately has none."""
    import numpy as np
    from holographic.rendering.holographic_skymodel import sky_model

    rng = np.random.default_rng(5)
    d = rng.normal(size=(6000, 3))
    d[:, 1] = np.abs(d[:, 1]) * 0.8 + 0.2                      # mid-sky band, away from horizon slant
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    clear = sky_model(12.0)(d)

    broken = {"cirrocumulus": (0.55, 0.35, 0.90), "altocumulus": (0.55, 0.30, 0.90),
              "stratocumulus": (0.60, 0.20, 0.85), "cirrus": (0.55, 0.30, 0.90)}
    for kind, (cov, gap_lo, gap_hi) in broken.items():
        s = sky_model(12.0, clouds=[(kind, cov)])(d)
        dev = np.abs(s - clear).mean(axis=1)
        gap = float((dev < 0.02).mean())
        assert gap_lo < gap < gap_hi, "%s gap fraction %.2f outside (%.2f, %.2f) -- solid or vanished" % (
            kind, gap, gap_lo, gap_hi)
        assert dev.std() > 0.02, "%s has no structure (dev std %.4f)" % (kind, dev.std())

    for kind in ("cirrostratus", "altostratus"):               # sheets: no gaps, but the veil must be REAL
        s = sky_model(12.0, clouds=[(kind, 0.5)])(d)
        dev = np.abs(s - clear).mean(axis=1)
        assert (dev < 0.02).mean() < 0.05, "%s is a sheet; it must cover, not vanish" % kind

    # the full rain deck may be near-solid in transmittance, but base shading must keep TEXTURE in it --
    # the exact complaint that started this: an overcast render that was one flat grey.
    deck = sky_model(12.0, clouds=[("nimbostratus", 1.0)])(d)
    assert deck.mean(axis=1).std() > 0.02, "a full deck rendered as one flat colour again"


def test_warp_and_erosion_change_the_field_and_stay_deterministic():
    """Look-dev is taste, but two things are contract: the warp/erosion machinery must actually be IN the
    field (the same kind with the mechanisms present differs from a hypothetical straight-threshold
    version -- proxied here by variance rising with erosion present), and it must stay deterministic
    (same seed = same sky, still). What it LOOKS like belongs to the reviewer; that it exists and repeats
    belongs to the suite."""
    import numpy as np
    from holographic.rendering.holographic_skymodel import sky_model

    rng = np.random.default_rng(9)
    d = rng.normal(size=(4000, 3))
    d[:, 1] = np.abs(d[:, 1]) * 0.7 + 0.3
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    a = sky_model(15.0, clouds=[("altocumulus", 0.6)], cloud_seed=0)(d)
    b = sky_model(15.0, clouds=[("altocumulus", 0.6)], cloud_seed=0)(d)
    assert np.array_equal(a, b), "warp/erosion broke determinism"
    c = sky_model(15.0, clouds=[("altocumulus", 0.6)], cloud_seed=1)(d)
    assert not np.array_equal(a, c), "cloud_seed must change the layout"
    # edges torn, not die-cut: the transition band (partial cloud) must be a real fraction of covered
    # directions, because erosion by construction manufactures intermediate densities at element edges
    clear = sky_model(15.0)(d)
    dev = np.abs(a - clear).mean(axis=1)
    covered = dev > 0.02
    partial = (dev > 0.02) & (dev < 0.35 * dev.max())
    assert covered.any() and partial.sum() / max(covered.sum(), 1) > 0.15, \
        "no transition band -- edges are die-cut again (erosion inert?)"


def test_clouds_move_and_evolve_with_time_deterministically():
    """MOOSE'S REVIEW, pinned: 'clouds should be changing shape and moving naturally' in a timelapse.
    Three contracts: time changes the field at all (motion exists); EVOLUTION alone -- wind zeroed --
    still changes it (shapes morph via the slide through the solid noise's third axis, not mere
    translation); and the same time twice is bit-identical (a timelapse must replay). The look of the
    motion is the reviewer's; that it exists, morphs, and repeats is the suite's."""
    import numpy as np
    from holographic.rendering.holographic_skymodel import sky_model

    rng = np.random.default_rng(2)
    d = rng.normal(size=(3000, 3))
    d[:, 1] = np.abs(d[:, 1]) * 0.6 + 0.4
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    kw = dict(clouds=[("altocumulus", 0.6)])

    a = sky_model(12.0, time_s=0.0, **kw)(d)
    b = sky_model(12.0, time_s=120.0, **kw)(d)
    assert np.abs(a - b).mean() > 1e-3, "time did not move the clouds"
    assert np.array_equal(b, sky_model(12.0, time_s=120.0, **kw)(d)), "cloud motion broke determinism"
    e0 = sky_model(12.0, time_s=0.0, wind=(0, 0), **kw)(d)
    e1 = sky_model(12.0, time_s=120.0, wind=(0, 0), **kw)(d)
    assert np.abs(e0 - e1).mean() > 1e-3, \
        "no shape evolution without wind -- the solid-noise slide went inert; motion is translation only"


def test_sun_light_syncs_to_the_sky_and_casts_cloud_shadows():
    """MOOSE'S REQUEST, pinned in its three parts. (1) SYNC: scene_light('sun', sky=) reads direction,
    colour, and day-scaling from the sky closure -- one source of truth, so the disk overhead and the
    light on the ground cannot disagree; at midnight the synced sun contributes nothing. (2) CLOUD
    SHADOWS: with cloud_shadows=True the intensity is a FIELD gated by the sky's own transmittance toward
    the sun -- the SAME shell and layer densities the sky paints (verified by the machinery being one
    closure, not a copy). (3) CUSTOM lighting stays untouched: 'sun' without sky= behaves as before, and
    sky= on a non-sun kind refuses.

    Probe discipline (fourth fragile-probe lesson of this arc, applied in advance): the shadow assertion
    uses a 400-point grid and a BROKEN deck, never a handful of points that can all land in gaps."""
    import numpy as np
    import pytest
    import lecore

    m = lecore.UnifiedMind(dim=128, seed=0)
    sky = m.sky_model(hour=9.0, clouds=[("stratocumulus", 0.6)])
    L = m.scene_light("sun", sky=sky, intensity=4.0, cloud_shadows=True)
    assert np.allclose(L.direction, sky.sun_direction), "sun light and sky disk point different ways"

    g = np.stack(np.meshgrid(np.linspace(-8, 8, 20), np.linspace(-8, 8, 20)), axis=-1).reshape(-1, 2)
    P = np.stack([g[:, 0], np.zeros(len(g)), g[:, 1]], axis=1)
    _, _, rad = L.sample(P, None)
    assert rad[:, 0].std() > 0.1, "cloud_shadows produced a uniform field -- the gate is inert"
    assert rad[:, 0].max() <= 4.0 + 1e-9, "transmittance must only DIM the sun, never brighten it"

    _, _, rn = m.scene_light("sun", sky=m.sky_model(hour=0.0), intensity=4.0).sample(P[:1], None)
    assert rn.max() < 1e-6, "a below-horizon synced sun must contribute nothing"

    plain = m.scene_light("sun", direction=(0.3, -1.0, 0.2), intensity=2.0)
    assert float(plain.intensity) == 2.0, "custom directional lighting must be untouched by the sync path"
    with pytest.raises(ValueError, match="SUN"):
        m.scene_light("spot", sky=sky)
