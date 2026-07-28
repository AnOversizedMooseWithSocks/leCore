"""holographic_skymodel.py -- a PARAMETRIC sky: time of day, sun, moon, stars, and HIGH cloud layers, as
one deterministic radiance field f(directions) -> rgb.

WHERE THIS SITS (the audit, so nobody rebuilds the neighbours). sky_dome() already does a zenith->horizon
gradient + sun disk + ground, and samples a real HDRI. cloud_scene() already does the LOW, volumetric layer
properly -- cumulus/wispy/storm presets with self-shadowing, marched density, measured quality tiers. What
did not exist is everything BETWEEN those two: the sky as a function of TIME (13/13 audit phrasings missed
'time of day sky gradient', 'night sky with stars', 'render the moon', 'cirrus or stratus layer'). This
module is that middle: the CELESTIAL and HIGH-ALTITUDE part of the sky, which is thin enough to be a 2-D
radiance field over direction rather than a marched volume.

THE LAYERING DECISION, stated because it is the design: high clouds (cirrus, altostratus, nimbostratus)
are kilometres up and optically THIN-to-sheetlike -- from the ground they read as a textured TRANSMITTANCE
painted on the dome, not as parallax volumes. So they live HERE, as density fields over direction that
attenuate the sun/moon/sky behind them and pick up forward-scatter glow near the sun. LOW clouds (cumulus
and friends) have real depth and self-shadowing and belong to the existing volumetric stack -- this module
deliberately does not duplicate it. Use both together: sky_model as the `sky=`/dome radiance, cloud_scene
for the puffy foreground.

Everything is a function of direction and PARAMETERS only -- no state, no RNG object. The starfield is a
hash of direction (same seed = same sky, forever), which is how a "random" sky obeys the determinism rule.
"""
import numpy as np

# time-of-day anchor palette. Columns: zenith rgb, horizon rgb. Rows are KEY TIMES; between rows we lerp.
# WHY A TABLE, not a scattering model: a real Rayleigh/Mie solve (Preetham/Hosek-Wilkie) needs fitted
# constants this repo cannot verify from first principles -- the AgX tonemapper was already declined once
# for exactly that reason (a kept negative on record). A keyed palette is honest about being artistic,
# fully inspectable, and each anchor is editable in place.
_PALETTE = [
    #  hour   zenith                horizon
    (0.0,  (0.010, 0.012, 0.030), (0.020, 0.022, 0.045)),   # deep night
    (4.5,  (0.015, 0.018, 0.045), (0.060, 0.045, 0.070)),   # astronomical dawn
    (6.0,  (0.100, 0.140, 0.320), (0.950, 0.550, 0.300)),   # sunrise: warm horizon, cool zenith
    (8.0,  (0.220, 0.420, 0.850), (0.700, 0.800, 0.950)),   # morning
    (12.0, (0.250, 0.480, 0.950), (0.750, 0.850, 0.980)),   # noon
    (17.0, (0.230, 0.430, 0.870), (0.820, 0.780, 0.750)),   # late afternoon
    (19.0, (0.120, 0.150, 0.350), (0.980, 0.450, 0.220)),   # sunset
    (20.5, (0.030, 0.035, 0.090), (0.180, 0.090, 0.130)),   # dusk
    (24.0, (0.010, 0.012, 0.030), (0.020, 0.022, 0.045)),   # wraps to deep night
]

# high-cloud vocabulary: (texture name, (sx, sy) anisotropy, sheet_floor, density_gain)
#   * cirrus: fbm stretched hard along one axis -> the wind-combed streaks; mostly transparent
#   * altostratus: gentle large-scale sheet, the "milky sun" layer -- the disk stays visible through it
#   * nimbostratus: near-opaque grey blanket; the sun becomes a bright smear, not a disk
# Per-kind fields: (texture, texture_params, (sx, sy) anisotropy, sheet_floor, density_gain, extinction,
# threshold_sharpness). EXTINCTION is per kind because one coefficient cannot serve both ends of the
# vocabulary (a star leaked 7% through a "full" nimbostratus at 3.0, but raising it globally would have
# killed the altostratus milky-sun contract). THRESHOLD_SHARPNESS is per kind for the same shape of
# reason, found in review: the first vocabulary was three fbm SHEETS, and a full deck rendered as ONE FLAT
# GREY -- correct transmittance, no structure ("not going to just result in a solid color"). Broken and
# cellular skies need a remap that leaves GAPS (near-clear directions between elements), and gaps come
# from thresholding a field steeply, not from more octaves of the same sheet.
#
# The vocabulary, by altitude band and by what each one does to the light:
#   cirrus        wind-combed streaks (fbm stretched hard); mostly transparent
#   cirrostratus  a thin milky VEIL; the disk survives almost untouched -- the halo-weather sky
#   cirrocumulus  fine cellular ripples (voronoi f2f1, small cells): the "mackerel sky"
#   altocumulus   larger cellular clumps with real gaps between them
#   altostratus   the translucent sheet; the milky sun
#   stratocumulus broken lumpy deck -- steep threshold on fbm, wide gaps, high contrast
#   nimbostratus  the rain blanket; near-opaque, and its own texture now mottles the base
# ...plus per-kind WARP (domain-warp strength) and ERODE (detail-erosion strength), added after look
# review ("the clouds don't look very good"): a single octave through a hard threshold gives flat blobs
# with cutout edges -- an unfinished look, not a recorded decision. WARP bends the sample coordinates with
# warped_noise (the engine's own dFBM -- its docstring says "weather fronts"; delegated, not hand-rolled),
# which turns straight cutout borders into fronts and wisps. ERODE subtracts a finer fbm octave scaled by
# (1 - v), eating the EDGES of every element while leaving cores solid -- the standard two-scale cloud
# trick, and the reason real cloud edges look torn rather than die-cut.
_CLOUD_KINDS = {
    #                 texture    tex_params        (sx, sy)   floor gain  ext  sharp warp erode
    "cirrus":        ("fbm",     {},               (6.0, 1.2), 0.00, 0.9, 2.5, 1.0, 0.9, 0.55),
    "cirrostratus":  ("fbm",     {},               (2.5, 2.0), 0.20, 0.25, 1.2, 1.0, 0.4, 0.20),
    "cirrocumulus":  ("voronoi", {"kind": "f2f1"}, (9.0, 7.0), 0.00, 1.9, 2.2, 1.6, 0.3, 0.45),
    "altocumulus":   ("voronoi", {"kind": "f2f1"}, (4.0, 3.2), 0.00, 1.9, 3.0, 1.5, 0.5, 0.50),
    "altostratus":   ("fbm",     {},               (2.0, 1.6), 0.35, 0.6, 3.0, 1.0, 0.5, 0.25),
    "stratocumulus": ("fbm",     {"octaves": 5},   (2.6, 2.2), 0.00, 1.35, 4.5, 2.6, 0.6, 0.55),
    "nimbostratus":  ("fbm",     {},               (1.5, 1.3), 0.65, 0.9, 7.0, 1.0, 0.5, 0.35),
}


_PALETTE_TL = None      # built once; a module-level cache is fine because the palette table is a constant


def _lerp_palette(hour):
    """Sample the anchor palette at `hour` -- DELEGATED to the keyframe Timeline (holographic_anim), which
    is the engine's own keyed-interpolation machine. The first version hand-rolled the segment walk + lerp
    in nine lines; the audit ('utilize what exists instead of hand rolling') found it was the Timeline in a
    different costume -- a palette keyed by hour IS keyframes keyed by time. Delegating buys the shared
    code path AND the option of per-anchor easing later ('smooth' at dawn/dusk) for one keyword."""
    global _PALETTE_TL
    if _PALETTE_TL is None:
        from holographic.misc.holographic_anim import Timeline
        _PALETTE_TL = Timeline()
        for h, z, r in _PALETTE:
            _PALETTE_TL.key("zenith", h, np.asarray(z, float))
            _PALETTE_TL.key("horizon", h, np.asarray(r, float))
    h = float(hour) % 24.0
    return np.asarray(_PALETTE_TL.sample("zenith", h)), np.asarray(_PALETTE_TL.sample("horizon", h))


def sun_direction(hour, axis_tilt=0.35):
    """Where the sun is at `hour` (0-24): a simple arc rising in +x, peaking overhead-ish, setting in -x.
    Elevation drives everything downstream (star fade, palette is keyed separately), so the arc being an
    idealised circle rather than an ephemeris is fine -- and HONEST: this is a look model, not an almanac.
    Below the horizon at night, which is what makes the stars' fade term work without a special case."""
    a = (float(hour) - 6.0) / 12.0 * np.pi                    # 6h -> rising (angle 0), 18h -> setting (pi)
    el = np.sin(a)                                             # elevation in [-1, 1]
    az = np.cos(a)
    d = np.array([az, el, -axis_tilt], float)
    return d / np.linalg.norm(d)


_RE, _LAYER_H = 6371.0, 6.0    # km: earth radius + high-cloud shell altitude. MODULE scope on purpose --
                               # the sky's radiance and the sun light's cloud-shadow transmittance must
                               # agree on the shell; a NameError at first wiring proved they were one
                               # scope away from silently diverging.

_STAR_LATTICE = 120.0        # cells of ~0.5 deg. THE FIRST VERSION used 997 (~0.06 deg): every star was a
                             # fraction of a pixel at any sane render size, and the delivered night render
                             # showed NO stars at all -- correct radiance, invisible image. Moose caught it
                             # from the artifact. A star needs angular EXTENT to survive sampling; ~0.5 deg
                             # is artistic licence (real stars are points), traded knowingly for existing.


def _star_cells(d, seed):
    """Deterministic starfield with EXTENT: quantise to a coarse lattice, hash the CELL (existence +
    magnitude), then shade each direction by its alignment with the cell centre -- a bright core with a
    soft falloff, so a star covers a pixel or two instead of losing the lottery against the sampler.
    Integer mix (splitmix-style), not hashlib: this runs per sample direction per frame; determinism needs
    stability, not security."""
    d = np.asarray(d, float)
    q = np.floor(d * _STAR_LATTICE).astype(np.int64)
    # DELEGATED to hash_unit (holographic_determinism) -- the engine's stateless coordinate-keyed
    # randomness, whose docstring is literally this use case ("a pure FUNCTION of where and which: same
    # inputs, same value, on any node, in any order"). The first version hand-rolled a splitmix-style
    # integer mix; the audit replaced it with the audited primitive. Star LAYOUTS from a given seed
    # change with this swap -- allowed, the seed is an aesthetic knob and no test pins positions; the
    # contracts (determinism, night-only, extent, count band, occlusion) are what is pinned, and hold.
    from holographic.misc.holographic_determinism import hash_unit
    rnd = np.asarray(hash_unit(q[:, 0], q[:, 1], q[:, 2], int(seed)), float)
    centre = (q + 0.5) / _STAR_LATTICE
    centre = centre / np.maximum(np.linalg.norm(centre, axis=1, keepdims=True), 1e-12)
    align = np.clip((d * centre).sum(axis=1), 0.0, 1.0)
    core = np.exp(-(1.0 - align) * (_STAR_LATTICE * _STAR_LATTICE * 0.55))   # falloff ~ the cell's own width
    return rnd, core


def sky_model(hour=12.0, clouds=(), stars_seed=None, star_density=0.9985, moon=None,
              sun_intensity=18.0, cloud_seed=0, time_s=0.0, wind=(0.05, 0.02), evolve=0.10):
    """Build the sky: returns a callable f(directions (M,3)) -> rgb (M,3), pluggable anywhere sky_dome is
    (the tracer's sky=, a DomeLight's color=, sky_dome's own env slot conceptually replaced).

    hour        0-24. Drives the gradient palette, the sun's arc, star visibility, moon default position.
    clouds      sequence of (kind, coverage) with kind in 'cirrus'|'altostratus'|'nimbostratus' and
                coverage 0..1. Layers COMPOSE: each contributes a transmittance and a scattered term, so
                'partially cloudy with the sun casting through' is cirrus at 0.4 -- the disk survives,
                dimmed, with silver-lining glow where the layer thins near the sun.
    stars_seed  int -> deterministic starfield (same seed, same sky, forever); None -> no stars. Stars
                fade with sun elevation, so a noon starfield correctly shows nothing.
    moon        None, True (auto-place opposite the sun), or a dict {dir:(x,y,z), size, brightness}.
    time_s / wind / evolve: CLOUD MOTION, review-driven ("they should be changing shape and moving
                naturally"). Two motions, both pure functions of time (a timelapse must replay
                bit-identically): WIND drifts the sample plane by wind*time_s -- the whole layer slides
                downwind; EVOLVE slides the sample slice through the SOLID 3-D texture along the axis the
                static model held at zero -- the boring-axis move again: the third dimension of a solid
                noise is a free evolution parameter, so shapes genuinely morph (elements grow, split,
                dissolve) rather than merely translate, and no new noise machinery was built to get it.
    Deterministic throughout; the only 'randomness' is a hash of direction and seed."""
    z_col, h_col = _lerp_palette(hour)
    sdir = sun_direction(hour)
    day = float(np.clip(sdir[1] * 4.0, 0.0, 1.0))              # 0 at night, 1 once the sun is decently up

    if moon is True or (moon is None and stars_seed is not None):
        mdir = -sdir                                           # full-moon geometry: opposite the sun
        moon = {"dir": mdir / np.linalg.norm(mdir), "size": 0.9995, "brightness": 1.2}
    elif isinstance(moon, dict):
        md = np.asarray(moon.get("dir", -sdir), float)
        moon = {"dir": md / np.linalg.norm(md), "size": float(moon.get("size", 0.9995)),
                "brightness": float(moon.get("brightness", 1.2))}

    layers = []
    if clouds:
        from holographic.materials_and_texture.holographic_proctex import proc_texture
        for kind, coverage in clouds:
            if kind not in _CLOUD_KINDS:
                raise ValueError("unknown high-cloud kind %r -- cirrus, altostratus, nimbostratus (LOW "
                                 "puffy clouds are the volumetric stack: use cloud_scene)" % (kind,))
            tex, tex_kw, (sx, sy), floor_, gain, extinction, sharp, warp, erode = _CLOUD_KINDS[kind]
            field = proc_texture(tex, scale=1.0, seed=cloud_seed, **tex_kw)
            # the warp field displaces sample coordinates; the detail field erodes edges. Both are the
            # SAME seeded machinery every other texture uses -- deterministic by the same argument.
            warp_f = proc_texture("fbm", scale=2.3, seed=cloud_seed + 101) if warp > 0 else None
            warp_g = proc_texture("fbm", scale=2.3, seed=cloud_seed + 202) if warp > 0 else None
            detail = proc_texture("fbm", scale=4.7, seed=cloud_seed + 303, octaves=4) if erode > 0 else None
            layers.append((field, sx, sy, floor_, gain, float(np.clip(coverage, 0.0, 1.0)),
                           extinction, sharp, warp, warp_f, warp_g, erode, detail))

    def radiance(dirs):
        d = np.atleast_2d(np.asarray(dirs, float))
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        n = len(d)
        up = np.clip(d[:, 1], -1.0, 1.0)

        # 1. GRADIENT: horizon -> zenith by a curve that keeps the horizon band wide (as skies look).
        t = np.clip(up, 0.0, 1.0) ** 0.45
        rgb = (1 - t)[:, None] * h_col[None, :] + t[:, None] * z_col[None, :]

        # 2. SUN: disk + glow, both scaled by daylight so the disk vanishes below the horizon.
        cos_s = d @ sdir
        sun = np.clip(cos_s, 0.0, 1.0)
        disk = (cos_s > 0.9997).astype(float) * sun_intensity
        glow = sun ** 220 * 1.6 + sun ** 12 * 0.22
        sun_rgb = np.array([1.0, 0.88, 0.72])
        # low sun is redder: tilt the sun colour toward the horizon palette as elevation drops
        warm = float(np.clip(1.0 - sdir[1] * 2.2, 0.0, 1.0))
        sun_col = (1 - warm) * sun_rgb + warm * np.array([1.0, 0.55, 0.30])
        sun_term = (disk + glow)[:, None] * sun_col[None, :] * day

        # 3. HIGH CLOUD LAYERS on a SPHERICAL SHELL, because the sky is not a plane over your head. THE
        #    FIRST VERSION projected onto the plane y=+1 and faded the layer toward the horizon -- exactly
        #    backwards: a real deck VISUALLY THICKENS toward the horizon (grazing rays take a longer path
        #    through the layer) and extends past the geometric horizontal, because the shell curves over
        #    the earth and you see its underside beyond the horizon dip. Moose caught it from the render
        #    ("the sky is a sphere that extends beyond the horizon with depth"; "the overcast example looks
        #    incorrect"). So: intersect each ray with a shell of radius Re+H about the earth centre
        #    (0, -Re, 0). The hit distance gives BOTH the texture sample point (which compresses into
        #    perspective bands near the horizon -- the depth cue) and the slant factor (path length /
        #    vertical thickness), the secant thickening that makes grazing rays optically heavier. Still
        #    one closed-form intersection per ray; no marching.
        trans = np.ones(n)
        scatter = np.zeros((n, 3))
        if layers:
            Rc = _RE + _LAYER_H
            b = d[:, 1] * _RE                                   # d . (origin - centre), centre = (0, -Re, 0)
            t_hit = -b + np.sqrt(np.maximum(b * b + (Rc * Rc - _RE * _RE), 0.0))   # shell encloses us: real root
            hit = d * t_hit[:, None]
            # slant thickening: vertical thickness / cos(local zenith angle at the hit point), where the
            # local zenith is the shell normal (hit - centre)/Rc. Clamped so the horizon is heavy, not infinite.
            cos_local = np.clip(((hit[:, 0] * d[:, 0]) + ((hit[:, 1] + _RE) * d[:, 1])
                                 + (hit[:, 2] * d[:, 2])) / Rc, 0.06, 1.0)
            slant = 1.0 / cos_local
            for (field, sx, sy, floor_, gain, coverage, extinction, sharp,
                 warp, warp_f, warp_g, erode, detail) in layers:
                P = np.stack([hit[:, 0] / 40.0 * sx - wind[0] * time_s,
                              np.full(n, evolve * time_s),
                              hit[:, 2] / 40.0 * sy - wind[1] * time_s], axis=1)
                if warp_f is not None:
                    # domain warp: sample WHERE another field says, not on the straight grid -- edges
                    # become fronts instead of cutouts (iq's dFBM shape, via the seeded texture stack)
                    wx = np.asarray(warp_f(P), float)
                    wz = np.asarray(warp_g(P), float)
                    P = P + np.stack([wx - 0.5, np.zeros(n), wz - 0.5], axis=1) * warp
                v = np.asarray(field(P), float)
                if v.ndim > 1:
                    v = v.mean(axis=1)
                if detail is not None:
                    # detail erosion: the fine octave eats where the element is already THIN (weight 1-v),
                    # tearing the edges while leaving cores untouched
                    dv = np.asarray(detail(P), float)
                    if dv.ndim > 1:
                        dv = dv.mean(axis=1)
                    v = np.clip(v - erode * dv * (1.0 - np.clip(v, 0.0, 1.0)), 0.0, 1.0)
                # coverage-threshold remap, with per-kind STEEPNESS. sharp=1 is the old soft ramp (sheets);
                # sharp>1 pushes the remap toward a cutout, which is where GAPS come from -- a cellular or
                # broken sky is mostly the space between its elements, and a soft ramp fills that space
                # with haze until the whole dome averages to one grey.
                # gain-before-threshold for the cellular kinds: voronoi f2f1 lives mostly in the low
                # half of [0,1], so thresholding it raw at 0.55 coverage left 94-98% of the sky EMPTY --
                # measured, a mackerel sky with almost no mackerel. gain (in the kind table) lifts the
                # field into the threshold's working range; sharp then cuts the gaps between elements.
                v = np.clip((v * max(gain, 1.0) - (1.0 - coverage)) / max(coverage, 1e-6), 0.0, 1.0) ** sharp
                dens = np.clip(floor_ * coverage + v * min(gain, 1.0) * coverage, 0.0, 1.0)
                tau = extinction * dens * slant                # Beer-Lambert with the geometric path length
                layer_T = np.exp(-tau)
                # forward scatter: cloud lights up near the sun; saturates with tau so a thick horizon band
                # reads as SOLID CLOUD, never as missing sky. BASE SHADING: thick cores are darker than thin
                # edges (bases in shadow of their own tops) -- the term that keeps even a full deck from
                # rendering as one flat grey, because the texture survives INTO the lit colour.
                fwd = np.clip(cos_s, 0.0, 1.0) ** 8
                base_dark = 1.0 - 0.38 * dens
                lit = (0.75 + 0.6 * fwd * day) * base_dark
                cloud_col = np.array([0.9, 0.9, 0.93]) * (0.35 + 0.65 * day)
                amount = 1.0 - np.exp(-0.8 * tau)
                scatter += (amount * lit)[:, None] * cloud_col[None, :] * trans[:, None]
                trans = trans * layer_T

        # 4. STARS: a hashed sparkle field, faded by daylight AND by cloud transmittance -- stars behind
        #    a nimbostratus blanket correctly disappear.
        star_term = np.zeros((n, 3))
        if stars_seed is not None:
            hv, core = _star_cells(d, stars_seed)
            mask = (hv > star_density) & (up > 0.0)
            mag = ((hv - star_density) / max(1e-9, 1.0 - star_density))
            star_term[mask] = ((0.8 + 3.0 * mag[mask]) * core[mask])[:, None] * np.array([0.95, 0.97, 1.0])
            star_term *= (1.0 - day) * trans[:, None]

        # 5. MOON: disk + soft glow, faded by clouds like everything celestial.
        moon_term = np.zeros((n, 3))
        if isinstance(moon, dict):
            cos_m = d @ moon["dir"]
            mdisk = (cos_m > moon["size"]).astype(float) * moon["brightness"]
            mglow = np.clip(cos_m, 0.0, 1.0) ** 400 * 0.25
            moon_term = (mdisk + mglow)[:, None] * np.array([0.92, 0.94, 1.0]) * (1.0 - 0.85 * day)

        return rgb * trans[:, None] + (sun_term + star_term + moon_term) * trans[:, None] + scatter

    def sun_transmittance(P, shadow_scale=60.0):
        """Cloud transmittance ALONG THE SUN DIRECTION from world points P (M,3) -- the field a sun light
        multiplies its intensity by to cast CLOUD SHADOWS on the ground.

        Geometry: from each point, march the sun ray to the SAME shell the sky paints its clouds on and
        evaluate the SAME per-kind layer densities there -- one machinery, two consumers, so the shadow on
        the ground and the cloud overhead can never disagree about where the cloud is.

        `shadow_scale` is declared ARTISTIC LICENCE, same class as the star extent: scene units are metres
        while shell cloud features are kilometres, so a physically-projected shadow pattern across a
        10-unit scene is one constant value -- technically right, visually nothing. shadow_scale
        multiplies the points' world XZ before projection so cloud features sweep the scene at a visible
        size. Set it to 1.0 for the physical answer."""
        P = np.atleast_2d(np.asarray(P, float))
        if not layers or sdir[1] <= 0.0:
            return np.ones(len(P))                             # no clouds, or the sun is down: no cloud gate
        b = sdir[1] * _RE
        t_hit = -b + np.sqrt(max(b * b + ((_RE + _LAYER_H) ** 2 - _RE * _RE), 0.0))
        hitp = P * shadow_scale + sdir[None, :] * t_hit        # entry point on the shell, per ground point
        cos_local = max(float(sdir[1]), 0.06)
        T = np.ones(len(P))
        for (field, sx, sy, floor_, gain, coverage, extinction, sharp,
             warp, warp_f, warp_g, erode, detail) in layers:
            Q = np.stack([hitp[:, 0] / 40.0 * sx - wind[0] * time_s,
                          np.full(len(P), evolve * time_s),
                          hitp[:, 2] / 40.0 * sy - wind[1] * time_s], axis=1)
            if warp_f is not None:
                Q = Q + np.stack([np.asarray(warp_f(Q), float) - 0.5, np.zeros(len(P)),
                                  np.asarray(warp_g(Q), float) - 0.5], axis=1) * warp
            v = np.asarray(field(Q), float)
            if v.ndim > 1:
                v = v.mean(axis=1)
            if detail is not None:
                dv = np.asarray(detail(Q), float)
                if dv.ndim > 1:
                    dv = dv.mean(axis=1)
                v = np.clip(v - erode * dv * (1.0 - np.clip(v, 0.0, 1.0)), 0.0, 1.0)
            v = np.clip((v * max(gain, 1.0) - (1.0 - coverage)) / max(coverage, 1e-6), 0.0, 1.0) ** sharp
            dens = np.clip(floor_ * coverage + v * min(gain, 1.0) * coverage, 0.0, 1.0)
            T = T * np.exp(-extinction * dens / cos_local)
        return T

    # SUN STATE on the closure -- the metadata a synced sun light needs. Attributes on the returned
    # callable, so the sky remains one object that crosses the service boundary as one ref and a light
    # can be built FROM it without re-stating (and drifting from) hour/clouds/wind.
    warm = float(np.clip(1.0 - sdir[1] * 2.2, 0.0, 1.0))
    radiance.sun_direction = sdir.copy()
    radiance.sun_color = tuple(((1 - warm) * np.array([1.0, 0.88, 0.72])
                                + warm * np.array([1.0, 0.55, 0.30])).tolist())
    radiance.day = day
    radiance.sun_transmittance = sun_transmittance
    return radiance


def _selftest():
    """The contracts that make this a sky and not a texture: time moves the palette AND the sun; stars are
    deterministic, night-only, and occluded by cloud; layers dim the sun without deleting it (except the
    blanket, which must); everything is a pure function of (direction, parameters)."""
    dirs = np.array([[0, 1, 0], [0.2, 0.1, 0.0], [0.7, 0.7, 0.0]], float)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    noon = sky_model(12.0)(dirs)
    night = sky_model(0.0)(dirs)
    assert noon.mean() > 8 * night.mean(), "noon must be much brighter than midnight: %.4f vs %.4f" % (
        noon.mean(), night.mean())
    dawn = sky_model(6.0)(np.array([[0.99, 0.05, 0.0]]) / np.linalg.norm([0.99, 0.05, 0.0]))
    assert dawn[0, 0] > dawn[0, 2], "a sunrise horizon must be warmer (R>B); got %s" % (dawn[0],)

    # SUN ARC: the sun term must move with the hour -- same direction, different hours, different answer.
    probe = sun_direction(9.0)[None, :]
    assert sky_model(9.0)(probe)[0].max() > 5.0, "looking AT the 9h sun must be bright"
    assert sky_model(15.0)(probe)[0].max() < 5.0, "by 15h the sun has moved off that direction"

    # STARS: deterministic, night-only, seed-controlled.
    rng = np.random.default_rng(7)
    many = rng.normal(size=(4000, 3)); many[:, 1] = np.abs(many[:, 1])
    many /= np.linalg.norm(many, axis=1, keepdims=True)
    s1 = sky_model(0.0, stars_seed=42)(many)
    s2 = sky_model(0.0, stars_seed=42)(many)
    s3 = sky_model(0.0, stars_seed=43)(many)
    assert np.array_equal(s1, s2), "same seed must be the same sky, forever"
    assert not np.array_equal(s1, s3), "a different seed must be a different starfield"
    base = sky_model(0.0)(many)
    n_bright = int(((s1 - base).max(axis=1) > 0.2).sum())
    assert 3 < n_bright < 400, "star count out of range: %d (density knob broken?)" % n_bright
    assert (sky_model(12.0, stars_seed=42)(many) - sky_model(12.0)(many)).max() < 1e-6, \
        "stars at NOON must be invisible"

    # HIGH CLOUDS: altostratus dims the sun but the disk survives; nimbostratus buries it.
    sun9 = sun_direction(9.0)[None, :]
    clear = sky_model(9.0)(sun9)[0].max()
    milky = sky_model(9.0, clouds=[("altostratus", 0.7)])(sun9)[0].max()
    buried = sky_model(9.0, clouds=[("nimbostratus", 1.0)])(sun9)[0].max()
    assert clear > milky > buried, "layer opacity ordering broken: %.2f, %.2f, %.2f" % (clear, milky, buried)
    assert milky > 0.25 * clear, "altostratus must be the MILKY-SUN layer -- the disk visible through it"
    assert buried < 0.15 * clear, "a full nimbostratus blanket must effectively hide the disk"
    # ...and stars behind the blanket vanish too
    s_blanket = sky_model(0.0, stars_seed=42, clouds=[("nimbostratus", 1.0)])(many)
    assert (s_blanket - sky_model(0.0, clouds=[("nimbostratus", 1.0)])(many)).max() < 0.05, \
        "stars must not shine through an opaque cloud blanket"

    # MOON: present at night opposite the sun, dimmed by day.
    mn = sky_model(0.0, moon=True)
    md = -sun_direction(0.0)
    assert mn(md[None, :])[0].max() > 0.5, "the auto-placed moon must be visible looking straight at it"

    # THE SPHERE, not a plane over your head (Moose's review of the first renders). Two contracts:
    # (a) a full overcast must cover the sky TO AND PAST the geometric horizon -- the first version faded
    #     the layer out exactly there, which read as clear sky at the horizon under a solid deck;
    # (b) at PARTIAL coverage the horizon must be optically HEAVIER than the zenith (grazing rays take the
    #     long way through the shell), measured as lower transmitted sky, i.e. more cloud signal.
    deck = sky_model(12.0, clouds=[("nimbostratus", 0.9)])
    at_horizon = deck(np.array([[1.0, 0.0, 0.0]]))[0]
    below = deck(np.array([[0.999, -0.03, 0.0]]) / np.linalg.norm([0.999, -0.03, 0.0]))[0]
    clear_horizon = sky_model(12.0)(np.array([[1.0, 0.0, 0.0]]))[0]
    assert abs(float(at_horizon.mean()) - float(below.mean())) < 0.05 and \
        np.abs(at_horizon - clear_horizon).max() > 0.05, \
        "a full deck must cover the horizon and extend past it, not fade to clear sky there"
    # altostratus for this probe, NOT cirrus: cirrus at low coverage is genuinely empty over most of the
    # dome (floor 0), so both probes can land in a gap and read 0/0 -- which the first version of this very
    # assertion did. The sheet layer has a nonzero floor everywhere, so the slant term must show.
    part = sky_model(12.0, clouds=[("altostratus", 0.5)])
    zen_dev = np.abs(part(np.array([[0.0, 1.0, 0.0]]))[0] - sky_model(12.0)(np.array([[0.0, 1.0, 0.0]]))[0]).mean()
    hor_dev = np.abs(part(np.array([[1.0, 0.02, 0.0]]) / np.linalg.norm([1.0, 0.02, 0.0]))[0]
                     - sky_model(12.0)(np.array([[1.0, 0.02, 0.0]]) / np.linalg.norm([1.0, 0.02, 0.0]))[0]).mean()
    assert hor_dev > zen_dev, "partial cover must read HEAVIER at the horizon (slant path): zen %.3f hor %.3f" % (
        zen_dev, hor_dev)

    try:
        sky_model(12.0, clouds=[("cumulus", 0.5)])
        raise AssertionError("cumulus must be REFUSED here -- it is the volumetric stack's job")
    except ValueError as e:
        assert "cloud_scene" in str(e), "the refusal must point at the right tool"

    print("skymodel selftest OK -- palette+sun move with the hour, stars deterministic/night-only/occluded,"
          " altostratus is the milky-sun layer, nimbostratus buries disk AND stars, moon auto-placed,"
          " low clouds correctly refused toward cloud_scene")


if __name__ == "__main__":
    _selftest()
