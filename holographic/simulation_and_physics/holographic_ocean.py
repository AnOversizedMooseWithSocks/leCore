"""Gerstner (trochoidal) ocean surface -- the one-call WATER preset.

WHY THIS EXISTS. The engine could EVOLVE water (spectral_ocean advances an existing height array under the
deep-water dispersion; free_surface overturns), but nothing GENERATED an ocean surface from nothing: no wave
spectrum, no directional swell, no chop. "Make water for my scene" had no answer. This module is that answer:
a sum of Gerstner waves -- the technique every real-time ocean (including the famous Shadertoy seascapes) is
built on -- producing a heightfield + exact analytic normals in one call, animatable via `t`.

THE METHOD (published; implemented from the papers, not from any shader source):
  * Fournier & Reeves, "A Simple Model of Ocean Waves", SIGGRAPH 1986 -- the trochoidal (Gerstner) wave:
    surface points move in circles, which sharpens crests and flattens troughs (a pure height-sum sine looks
    like jelly; the HORIZONTAL displacement is what reads as water).
  * Tessendorf, "Simulating Ocean Water", SIGGRAPH course 2001 -- deep-water dispersion omega = sqrt(g*k)
    (long waves travel faster; this de-synchronises the sum over time, which is what kills the visible tiling),
    and the steepness bound sum(Q_i * k_i * A_i) < 1 that prevents crest self-intersection loops.
  * GPU Gems 1 ch. 1 (Finch 2004) -- the practical parameterisation: per-wave steepness Q, direction spread
    around a wind heading, amplitude falling with frequency.

DESIGN
  * `gerstner_waves(seed, n_waves, ...)` builds a deterministic wave BANK (directions, wavelengths, amplitudes,
    phases) from a seed -- same seed, same ocean, any machine.
  * `water_surface(bank, X, Z, t)` evaluates the bank at grid points and time t: returns displaced positions
    (the trochoidal x/z motion included) AND the exact analytic normal (the partial derivatives of the Gerstner
    sum are closed-form -- no finite differences, no epsilon).
  * `make_water(...)` is the preset: one call -> {"height", "positions", "normals", "bank", ...}; `shaded=True`
    adds a simple sun-shaded preview image so the result is VISIBLE without a render pipeline.

Deterministic (seeded bank, hash-free), NumPy-only, vectorised over the whole grid.

KEPT NEGATIVE: Gerstner waves are a KINEMATIC model -- they do not conserve mass, refract at shores, or break.
For dynamics, seed spectral_ocean with make_water's height (the intended handoff: generate here, evolve there).
KEPT NEGATIVE: the steepness bound is ENFORCED by scaling, not by raising -- an artist sliding "choppiness" to 11
gets the steepest legal ocean, not a self-intersecting loop; the scale factor is reported in the bank.
"""

import numpy as np

G = 9.81                                   # deep-water gravity; omega = sqrt(G * k) is the dispersion relation


def gerstner_waves(seed=0, n_waves=24, wind_heading=0.35, spread=0.9,
                   wavelength_range=(3.0, 40.0), amplitude_scale=1.0, choppiness=0.8):
    """Build a deterministic Gerstner wave BANK. Returns a dict of per-wave arrays (all shape (n_waves,)):
    dir_x/dir_z (unit direction), k (wavenumber 2*pi/wavelength), amp, phase, omega (deep-water dispersion),
    q (per-wave steepness), plus 'steepness_scale' -- the factor applied to keep sum(q*k*amp) < 1 (Tessendorf's
    no-self-intersection bound; 1.0 means the request was already legal).

    wind_heading: mean travel direction in radians. spread: direction jitter around it (radians, uniform).
    wavelength_range: (min, max) metres, sampled log-uniformly so short chop and long swell both appear.
    amplitude_scale: overall height multiplier. Amplitude per wave falls as wavelength does (a/lambda fixed),
    which is what real spectra do -- big waves are tall, ripples are short. choppiness in [0,1+]: 0 = pure
    height sines (jelly), ~0.8 = natural, >1 gets clamped by the bound."""
    rng = np.random.default_rng(seed)
    lam_lo, lam_hi = float(wavelength_range[0]), float(wavelength_range[1])
    if not (0 < lam_lo < lam_hi):
        raise ValueError("wavelength_range must be 0 < min < max, got %r" % (wavelength_range,))
    # log-uniform wavelengths: equal representation per OCTAVE, so chop does not drown the swell
    lam = np.exp(rng.uniform(np.log(lam_lo), np.log(lam_hi), n_waves))
    k = 2.0 * np.pi / lam
    heading = wind_heading + rng.uniform(-spread, spread, n_waves)
    dir_x, dir_z = np.cos(heading), np.sin(heading)
    # amplitude proportional to wavelength (constant steepness per wave before the global bound), jittered
    amp = amplitude_scale * lam * (1.0 / 60.0) * rng.uniform(0.6, 1.4, n_waves)
    phase = rng.uniform(0.0, 2.0 * np.pi, n_waves)
    omega = np.sqrt(G * k)                                   # Tessendorf: long waves travel faster
    q = np.full(n_waves, float(choppiness)) / n_waves        # per-wave steepness share
    # ENFORCE the no-loop bound: sum(q * k * amp) < 1. Scale down, never raise. (KEPT NEGATIVE above.)
    s = float(np.sum(q * k * amp))
    steepness_scale = 1.0 if s < 0.95 else 0.95 / s
    q = q * steepness_scale
    return {"dir_x": dir_x, "dir_z": dir_z, "k": k, "amp": amp, "phase": phase, "omega": omega,
            "q": q, "steepness_scale": steepness_scale, "seed": int(seed), "n_waves": int(n_waves)}


def water_surface(bank, X, Z, t=0.0):
    """Evaluate the Gerstner bank at grid points (X, Z) and time t.

    Returns (positions, normals): positions is (..., 3) with the TROCHOIDAL displacement -- points move in
    circles, so x/z shift toward crests (this asymmetry is what makes it read as water rather than jelly) --
    and normals is the EXACT analytic unit normal from the closed-form partial derivatives (Tessendorf eq. 12
    form), not a finite difference.

    X, Z: broadcastable arrays of sample coordinates (metres). t: seconds."""
    X = np.asarray(X, float)
    Z = np.asarray(Z, float)
    shape = np.broadcast(X, Z).shape
    x = np.broadcast_to(X, shape)[..., None]                 # (..., 1) against the (W,) wave axis
    z = np.broadcast_to(Z, shape)[..., None]
    d_x, d_z = bank["dir_x"], bank["dir_z"]
    k, amp, ph, om, q = bank["k"], bank["amp"], bank["phase"], bank["omega"], bank["q"]

    theta = k * (d_x * x + d_z * z) - om * t + ph            # (..., W) phase of every wave at every point
    c, s = np.cos(theta), np.sin(theta)

    # Gerstner displacement: horizontal toward the crest (q term), vertical the height sum
    px = x[..., 0] + np.sum(q * amp * d_x * c, axis=-1)
    pz = z[..., 0] + np.sum(q * amp * d_z * c, axis=-1)
    py = np.sum(amp * s, axis=-1)

    # exact analytic normal (derivatives of the displaced surface, Tessendorf/GPU-Gems form)
    wa = k * amp
    nx = -np.sum(d_x * wa * c, axis=-1)
    nz = -np.sum(d_z * wa * c, axis=-1)
    ny = 1.0 - np.sum(q * wa * s, axis=-1)
    n = np.stack([nx, ny, nz], axis=-1)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-300)
    return np.stack([px, py, pz], axis=-1), n


def make_water(res=128, extent=40.0, t=0.0, seed=0, preset="ocean", shaded=False, sun_dir=(0.4, 0.8, 0.3),
               **bank_overrides):
    """ONE CALL -> a water surface. Returns a dict:
      height     (res, res) float -- the y displacement on the regular grid (feed spectral_ocean to EVOLVE it)
      positions  (res, res, 3)    -- trochoidally displaced points (feed depth/height mesh builders directly)
      normals    (res, res, 3)    -- exact analytic unit normals (no finite differences)
      bank                        -- the deterministic wave bank (reuse for other t: animation is just t=...)
      image      (res, res, 3)    -- ONLY when shaded=True: a simple sun-shaded preview in [0,1]

    preset picks the character, each just a parameter set over gerstner_waves:
      "ocean" -- mixed swell + chop (the default open sea)
      "calm"  -- long low swell, little chop (lake on a quiet day)
      "storm" -- steep, short, tall (near the legal steepness bound)
    Any gerstner_waves keyword may be overridden explicitly (e.g. wind_heading=1.2, n_waves=48).

    Animation: call again with a different `t` and the SAME seed -- the bank is deterministic, so frames are
    coherent. The deep-water dispersion makes long waves outrun short ones, which kills visible looping."""
    presets = {
        "ocean": dict(n_waves=24, wavelength_range=(3.0, 40.0), amplitude_scale=1.0, choppiness=0.8, spread=0.9),
        "calm":  dict(n_waves=16, wavelength_range=(8.0, 60.0), amplitude_scale=0.45, choppiness=0.35, spread=0.5),
        "storm": dict(n_waves=32, wavelength_range=(2.0, 24.0), amplitude_scale=1.9, choppiness=1.1, spread=1.2),
    }
    if preset not in presets:
        raise ValueError("preset must be one of %s, got %r" % (sorted(presets), preset))
    kw = dict(presets[preset]); kw.update(bank_overrides)
    bank = gerstner_waves(seed=seed, **kw)

    xs = np.linspace(-extent / 2.0, extent / 2.0, res)
    X, Z = np.meshgrid(xs, xs)
    pos, nrm = water_surface(bank, X, Z, t=t)
    out = {"height": pos[..., 1].copy(), "positions": pos, "normals": nrm, "bank": bank,
           "extent": float(extent), "t": float(t), "preset": preset}
    if shaded:
        # a deliberately SIMPLE preview: lambert + a fresnel-ish rim toward a sky tint. It exists so the result
        # is visible without the render pipeline, not to compete with it -- render properly via the mesh path.
        L = np.asarray(sun_dir, float); L /= (np.linalg.norm(L) + 1e-12)
        lam = np.clip(nrm @ L, 0.0, 1.0)
        up = np.clip(nrm[..., 1], 0.0, 1.0)
        fres = (1.0 - up) ** 3                                # grazing normals catch the sky
        deep = np.array([0.04, 0.16, 0.28]); sky = np.array([0.62, 0.78, 0.88]); sun = np.array([1.0, 0.95, 0.85])
        img = deep[None, None, :] * (0.35 + 0.65 * lam[..., None]) + sky[None, None, :] * fres[..., None] * 0.6
        img += sun[None, None, :] * (np.clip(lam, 0.0, 1.0) ** 24)[..., None] * 0.25   # cheap sun glint
        out["image"] = np.clip(img, 0.0, 1.0)
    return out


def _selftest():
    rng_checks = []

    # 1. determinism: same seed -> bit-identical bank and surface; different seed -> different ocean
    b1 = gerstner_waves(seed=3); b2 = gerstner_waves(seed=3); b3 = gerstner_waves(seed=4)
    assert all(np.array_equal(b1[k], b2[k]) for k in ("dir_x", "k", "amp", "phase")), "same seed must reproduce"
    assert not np.array_equal(b1["amp"], b3["amp"]), "a different seed must differ"

    X, Z = np.meshgrid(np.linspace(-10, 10, 48), np.linspace(-10, 10, 48))
    p1, n1 = water_surface(b1, X, Z, t=1.25)
    p2, n2 = water_surface(b2, X, Z, t=1.25)
    assert np.array_equal(p1, p2) and np.array_equal(n1, n2), "evaluation must be deterministic"

    # 2. the analytic normal matches a central finite difference of the HEIGHT sum (loose tol: the FD is the
    #    approximation here; the analytic form is exact). Checked on the height-only field (q=0) where the
    #    surface is a graph y(x,z) and the normal has the closed comparison form.
    flat = gerstner_waves(seed=3, choppiness=0.0)
    eps = 1e-4
    _, n_a = water_surface(flat, X, Z, t=0.7)
    hL = water_surface(flat, X - eps, Z, t=0.7)[0][..., 1]
    hR = water_surface(flat, X + eps, Z, t=0.7)[0][..., 1]
    hD = water_surface(flat, X, Z - eps, t=0.7)[0][..., 1]
    hU = water_surface(flat, X, Z + eps, t=0.7)[0][..., 1]
    n_fd = np.stack([-(hR - hL) / (2 * eps), np.ones_like(hL), -(hU - hD) / (2 * eps)], axis=-1)
    n_fd /= np.linalg.norm(n_fd, axis=-1, keepdims=True)
    err = float(np.abs(n_a - n_fd).max())
    assert err < 1e-5, ("analytic normal vs finite difference", err)
    rng_checks.append(("normal vs FD", err))

    # 3. the steepness bound holds: even an absurd choppiness request yields sum(q*k*amp) < 1 (scaled, not raised)
    wild = gerstner_waves(seed=0, choppiness=25.0, amplitude_scale=3.0)
    s = float(np.sum(wild["q"] * wild["k"] * wild["amp"]))
    assert s < 1.0 and wild["steepness_scale"] < 1.0, ("the no-loop bound must be enforced by scaling", s)

    # 4. trochoidal asymmetry is REAL -- but note WHERE it lives: q displaces points HORIZONTALLY toward crests;
    #    the height value on the regular grid (py = sum amp*sin) is INDEPENDENT of q (first draft asserted skew
    #    of the raw height grid and measured identical values -- the wrong quantity; kept as the lesson).
    #    Assert the actual mechanism twice:
    #    (a) with chop the planar points MOVE (|dx,dz| > 0) and move MOST near crests, zero with q=0;
    #    (b) the DISPLACED surface has sharper crests: sample y against the displaced x and the skewness of the
    #        arc-length-resampled profile rises vs the q=0 sine sum.
    chop = make_water(res=96, extent=60.0, seed=1, preset="ocean")
    jelly = make_water(res=96, extent=60.0, seed=1, preset="ocean", choppiness=0.0)
    xs = np.linspace(-30.0, 30.0, 96)
    X96, Z96 = np.meshgrid(xs, xs)
    disp_chop = np.hypot(chop["positions"][..., 0] - X96, chop["positions"][..., 2] - Z96)
    disp_jelly = np.hypot(jelly["positions"][..., 0] - X96, jelly["positions"][..., 2] - Z96)
    assert disp_jelly.max() < 1e-12, "q=0 must not displace horizontally"
    assert disp_chop.mean() > 0.02, ("chop must displace points toward crests", disp_chop.mean())
    # (b) the GERSTNER SIGNATURE, asserted robustly: horizontal point spacing COMPRESSES at crests (points ride
    #     circles, converging on top). corr(displaced-x spacing, height) measured -0.73..-0.87 across seeds --
    #     strong and stable, unlike per-row profile skew which is +0.004-0.006 (real but too small to pin without
    #     flaking; measured, kept as the reason this asserts the correlation instead).
    x_d_all, y_all = chop["positions"][:, :, 0], chop["positions"][:, :, 1]
    spacing = np.diff(x_d_all, axis=1)
    height_mid = 0.5 * (y_all[:, 1:] + y_all[:, :-1])
    r = float(np.corrcoef(spacing.ravel(), height_mid.ravel())[0, 1])
    assert r < -0.5, ("points must compress at crests (the trochoid), corr", r)
    # and with q=0 there is no compression to correlate: spacing is exactly the grid constant
    j_spacing = np.diff(jelly["positions"][:, :, 0], axis=1)
    assert float(j_spacing.std()) < 1e-12, "q=0 spacing must be uniform (pure height sines)"
    def skew(h):
        h = h - h.mean()
        return float(np.mean(h ** 3) / (np.mean(h ** 2) ** 1.5 + 1e-300))

    # 5. dispersion de-synchronises: the surface at t=0 and t=5 must differ everywhere meaningfully (animation
    #    is real), while the bank is unchanged (animation costs no rebuild)
    w0 = make_water(res=48, t=0.0, seed=2); w5 = make_water(res=48, t=5.0, seed=2)
    assert np.abs(w0["height"] - w5["height"]).mean() > 0.01, "time must move the surface"
    assert all(np.array_equal(w0["bank"][k], w5["bank"][k]) for k in ("k", "amp")), "the bank must not rebuild"

    # 6. presets are distinct and the preset gate refuses an unknown name
    calm = make_water(res=48, preset="calm", seed=0)["height"]
    storm = make_water(res=48, preset="storm", seed=0)["height"]
    # measured contrast at seed 0: storm std 1.54 vs calm 0.85 (1.8x); assert 1.5x -- a real margin under the
    # measurement, not a guessed round number (first draft guessed 2.0x and flaked).
    assert storm.std() > 1.5 * calm.std(), ("storm must be rougher than calm", storm.std(), calm.std())
    try:
        make_water(preset="tsunami")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown preset must raise")

    # 7. the shaded preview is a valid image and only appears when asked
    sw = make_water(res=48, shaded=True, seed=0)
    assert sw["image"].shape == (48, 48, 3) and 0.0 <= sw["image"].min() and sw["image"].max() <= 1.0
    assert "image" not in make_water(res=32, seed=0), "no image unless shaded=True"

    # 8. water_body -- the container-first assembler
    #    open water: mesh + a LIT fast render (the whole point is that lighting comes out right first time)
    ob = water_body(extent=50.0, seed=1, res=96)
    img = ob.render("fast", width=160, height=120)
    assert img.shape == (120, 160, 3)
    lum = img.mean(axis=2)
    assert float(np.percentile(lum, 50)) > 0.25, ("the bundled render must come out LIT", np.percentile(lum, 50))
    assert float(lum.std()) > 0.05, "and with real contrast, not a flat wash"
    #    contained: the water region exists inside the vessel, is absent outside, ripples animate, and the
    #    material callback assigns the library IOR to the water region
    gb = water_body(container="glass", level=0.7, ripple=0.4, seed=1)
    inside = np.array([[0.0, 0.35, 0.0]])                      # mid-water inside the glass
    above = np.array([[0.0, 1.6, 0.0]])                        # over the rim
    assert gb.water_sdf(inside)[0] < 0 < gb.water_sdf(above)[0], "water fills to level, not above"
    gb.at_time(1.5); da = gb.water_sdf(np.array([[0.1, 0.85, 0.05]]))[0]
    gb.at_time(0.0); db = gb.water_sdf(np.array([[0.1, 0.85, 0.05]]))[0]
    assert abs(da - db) > 1e-9, "ripples must animate with t"
    # the material callback runs at SURFACE HIT points (that is where the tracer calls it); nearest-subfield
    # classification is only meaningful there, so probe ON the water's top surface, not deep interior
    # (first draft probed mid-water and read IOR 0.0 -- the classifier legitimately picked another region).
    level_y = 0.0
    for yy in np.linspace(1.2, 0.0, 400):                      # walk down to the free surface at x=z=0
        if gb.water_sdf(np.array([[0.0, yy, 0.0]]))[0] <= 0:
            level_y = yy; break
    mats = gb.material_fn(np.array([[0.0, level_y + 1e-4, 0.0]]))
    assert abs(mats[4][0] - 1.33) < 0.01, ("the water surface must carry the library IOR", mats[4][0])
    #    the library is the source of truth: oil refracts at oil's index, not water's (the silent-fallback
    #    bug this pins: physical_properties lacks 'oil', and the first draft delivered 1.333 confidently)
    assert abs(water_body(container="glass", material="oil").ior - 1.47) < 1e-9
    #    refusals are loud
    for bad in (dict(container="bathtub"), dict(material="granite2")):
        try:
            water_body(**bad); raise AssertionError("should refuse %r" % bad)
        except ValueError:
            pass

    # KEPT NEGATIVE (asserted nowhere because it is a scope statement, recorded here loudly): this is KINEMATIC
    # water -- no mass conservation, shoaling, or breaking. Evolve via spectral_ocean; overturn via free_surface.
    print("OK: holographic_ocean self-test passed (deterministic bank + surface; analytic normal matches FD to "
          "%.1e; steepness bound enforced by scaling; the trochoid is real -- points compress at crests, "
          "corr(spacing, height) = %.2f vs exactly-uniform spacing at q=0; time animates without rebuilding the "
          "bank; storm rougher than calm; shaded preview opt-in)"
          % (rng_checks[0][1], r))




# ======================================================================================================
# WATER BODY -- the container-first assembler. make_water gives the SURFACE; this gives the SCENE.
# ======================================================================================================
#
# WHY: assembling good water was measured (in-session) at ~40 lines of expert glue -- grid meshing, a
# hand-tuned shading recipe, SDF container construction, per-region IOR assignment, camera placement, and
# three iterations of lighting. water_body bundles exactly that glue: containers by name or SDF, waves at
# any scale (open-sea swell down to ripples in a glass), materials pulled from the material library (colors
# from matlib, IOR 1.333 from physical_properties -- not hardcoded), and a render() with the lighting
# already balanced. The two quality tiers are the two renderers the engine actually has: "fast" rasterises
# a shaded mesh (seconds), "final" path-traces with true refraction (minutes) -- named, so the speed/quality
# trade is one word instead of a pipeline choice.

def _grid_mesh(positions):
    """(R,R,3) Gerstner positions -> a triangle Mesh (two triangles per quad). The session-proven mesher."""
    from holographic.mesh_and_geometry.holographic_mesh import Mesh
    R = positions.shape[0]
    V = positions.reshape(-1, 3)
    idx = np.arange(R * R).reshape(R, R)
    a = idx[:-1, :-1].ravel(); b = idx[:-1, 1:].ravel()
    c = idx[1:, :-1].ravel(); d = idx[1:, 1:].ravel()
    F = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
    return Mesh(V, F)


def shade_water_vertices(w, eye, sun_dir=(0.45, 0.55, -0.55), deep=None, shallow=None,
                         sky=(0.66, 0.80, 0.92), sun=(1.0, 0.96, 0.85),
                         glint_power=180.0, glint_gain=1.6):
    """Full per-vertex water radiance from the surface's own EXACT normals: depth-graded albedo x sun
    lambert + Schlick-fresnel sky reflection + a Blinn sun glint. Bake this into vertex colours and
    rasterise with ambient=1.0 / lights=[] so it passes through -- the relit recipe that fixed the
    double-darkening (vertex colours are multiplied by the rasteriser's lambert term; shading fully here
    and passing through avoids lighting the surface twice). deep/shallow default to the material library's
    water_deep / water colours."""
    from holographic.materials_and_texture.holographic_matlib import RENDER_MATERIALS
    if deep is None:
        deep = np.asarray(RENDER_MATERIALS["water_deep"][1], float) * 2.2    # matlib colours are dim albedos;
    if shallow is None:                                                       # lift for direct radiance use
        shallow = np.asarray(RENDER_MATERIALS["water"][1], float) * 3.5
    P = w["positions"].reshape(-1, 3); N = w["normals"].reshape(-1, 3)
    L = np.asarray(sun_dir, float); L /= (np.linalg.norm(L) + 1e-12)
    Vv = np.asarray(eye, float)[None, :] - P
    Vv /= (np.linalg.norm(Vv, axis=1, keepdims=True) + 1e-12)
    Hh = Vv + L[None, :]; Hh /= (np.linalg.norm(Hh, axis=1, keepdims=True) + 1e-12)
    lam = np.clip(N @ L, 0.0, 1.0)
    fres = 0.04 + 0.96 * (1.0 - np.clip(np.sum(N * Vv, axis=1), 0.0, 1.0)) ** 5     # Schlick
    hgt = w["height"].ravel()
    tt = (hgt - hgt.min()) / (np.ptp(hgt) + 1e-12)                                   # crests lighter
    alb = np.asarray(deep, float)[None, :] * (1 - tt[:, None]) + np.asarray(shallow, float)[None, :] * tt[:, None]
    col = alb * (0.35 + 0.75 * lam[:, None])
    col += np.asarray(sky, float)[None, :] * fres[:, None] * 0.85
    spec = np.clip(np.sum(N * Hh, axis=1), 0.0, 1.0) ** glint_power
    col += np.asarray(sun, float)[None, :] * spec[:, None] * glint_gain
    return np.clip(col, 0.0, 1.0)


# stock containers: name -> (sdf builder, inner radius/half-width, inner floor y, rim y).
# Each is a THIN SHELL opened at the top; the water fits the reported inner cavity.
def _stock_container(name, size=1.0, wall=0.03):
    from holographic.mesh_and_geometry.holographic_sdf import cylinder, box, sphere
    s, wl = float(size), float(wall)
    if name == "glass":
        h, r = 1.15 * s, 0.42 * s
        shell = cylinder(h=h, r=r).onion(wl)
        opener = box(bx=3 * s, by=0.6 * s, bz=3 * s).translate((0, h / 2 + 0.6 * s - wl * 0.5, 0))
        sdf = shell.subtract(opener).translate((0, h / 2, 0))
        return sdf, {"kind": "cyl", "r": r - 1.6 * wl, "floor": wl, "rim": h - wl}
    if name == "pool":
        hx, hy = 1.6 * s, 0.45 * s
        shell = box(bx=hx, by=hy, bz=hx).onion(wl)
        opener = box(bx=3 * s, by=0.5 * s, bz=3 * s).translate((0, hy + 0.5 * s - wl * 0.5, 0))
        sdf = shell.subtract(opener).translate((0, hy, 0))
        return sdf, {"kind": "box", "r": hx - 1.6 * wl, "floor": wl, "rim": 2 * hy - wl}
    if name == "bowl":
        r = 0.75 * s
        shell = sphere(r=r).onion(wl)
        opener = box(bx=3 * s, by=r, bz=3 * s).translate((0, r * 1.15, 0))
        sdf = shell.subtract(opener).translate((0, r * 0.55, 0))
        return sdf, {"kind": "cyl", "r": r * 0.80, "floor": r * 0.55 - r + 1.5 * wl, "rim": r * 0.55 + r * 0.6}
    raise ValueError("unknown container %r; stock containers are 'glass', 'pool', 'bowl' "
                     "(or pass your own SDF as the cavity)" % (name,))


class _ContainedWaterSDF:
    """The water region: the container's cavity, capped by a RIPPLED top surface. The cap is
    max(cavity, y - (level + h(x,z,t))) with h a small Gerstner height sum -- an approximate SDF (the
    rippled cap is not a true distance for large amplitudes), which is exactly the sphere-tracing-safe
    regime the small ripple amplitudes keep us in. This is what puts real, adjustable WAVES on water
    inside a glass."""

    def __init__(self, cavity, level_y, bank, ripple_amp):
        self.cavity, self.level_y, self.bank, self.amp = cavity, float(level_y), bank, float(ripple_amp)
        self.t = 0.0

    def __call__(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        d_cav = np.asarray(self.cavity(P)).reshape(len(P))
        if self.amp <= 0.0:
            h = 0.0
        else:
            th = (self.bank["k"] * (self.bank["dir_x"] * P[:, 0:1] + self.bank["dir_z"] * P[:, 2:3])
                  - self.bank["omega"] * self.t + self.bank["phase"])
            h = self.amp * np.sum(self.bank["amp"] * np.sin(th), axis=1)
        return np.maximum(d_cav, P[:, 1] - (self.level_y + h))


def water_body(container=None, level=0.72, preset="ocean", size=1.0, extent=40.0, res=192,
               t=0.0, seed=0, ripple=0.35, material="water", **wave_overrides):
    """The CONTAINER-FIRST water assembler: everything between 'I want water' and pixels, in one object.

    container:
      * None       -- OPEN water (an ocean or pond): a Gerstner surface over `extent` metres.
      * 'glass' / 'pool' / 'bowl' -- a stock vessel at `size` scale, filled to `level` (0..1 of its depth),
        with real Gerstner RIPPLES on the water's top (amplitude scaled by `ripple`, wavelengths scaled to
        the vessel, animated by `t` like open water).
      * an SDF     -- treated as the CAVITY the water fills (pass the interior region, not the shell).

    preset / wave_overrides tune the waves at any scale ('ocean'/'calm'/'storm' + any gerstner_waves
    keyword: choppiness, wind_heading, wavelength_range, n_waves ...). `material` picks the liquid from the
    material library ('water', 'water_deep', 'oil', 'honey' -- colour from matlib, IOR from
    physical_properties; an opaque liquid like 'milk' renders without transmission).

    Returns a WaterBody with:
      .surface / .mesh          the Gerstner surface dict + triangle mesh (open water)
      .scene_sdf / .material_fn the path-trace-ready SDF + per-point material callback (contained water)
      .camera()                 a sensible default camera for this body
      .render(quality='fast')   'fast': shaded-mesh raster (open, ~seconds) or low-spp trace (contained);
                                'final': high-res raster / 64-spp refractive path trace. The lighting is
                                the session-balanced recipe -- bright fresnel sky, sun glints, mid-tone
                                backdrop -- so the first render comes out LIT.
    Deterministic in (seed, t). See make_water for the wave model and its kept negatives (kinematic)."""
    return WaterBody(container, level, preset, size, extent, res, t, seed, ripple, material, wave_overrides)


class WaterBody:
    """The assembled water scene. Built by water_body(); see its docstring for the contract."""

    def __init__(self, container, level, preset, size, extent, res, t, seed, ripple, material, wave_overrides):
        from holographic.materials_and_texture.holographic_matlib import RENDER_MATERIALS
        from holographic.materials_and_texture.holographic_materialindex import physical_properties
        if material not in RENDER_MATERIALS:
            raise ValueError("unknown material %r; liquids include water, water_deep, oil, honey, milk" % (material,))
        self.material_name = material
        self.mat_color = np.asarray(RENDER_MATERIALS[material][1], float)
        # IOR lookup order: matlib's render IOR table (complete for the liquids: oil 1.47, honey 1.50 ...),
        # then the physical-properties table, then water's 1.333. The first draft used physical_properties
        # alone and oil silently fell back to 1.333 -- a wrong number delivered confidently; caught by
        # exercising every liquid, kept as the reason for this order.
        from holographic.materials_and_texture.holographic_matlib import _IOR, _CLASS_IOR
        if material in _IOR:
            self.ior = float(_IOR[material])
        else:
            try:
                self.ior = float(physical_properties(material).get("refractive",
                                 _CLASS_IOR.get(RENDER_MATERIALS[material][0], 1.333)))
            except Exception:
                self.ior = float(_CLASS_IOR.get(RENDER_MATERIALS[material][0], 1.333))
        self.t, self.seed, self.preset = float(t), int(seed), preset
        self.kind = "open" if container is None else "contained"

        if self.kind == "open":
            self.surface = make_water(res=res, extent=extent, t=t, seed=seed, preset=preset, **wave_overrides)
            self.mesh = _grid_mesh(self.surface["positions"])
            self.extent = float(extent)
            self.scene_sdf = None
        else:
            from holographic.mesh_and_geometry.holographic_sdf import plane
            if isinstance(container, str):
                self.container_sdf, inner = _stock_container(container, size=size)
                self.container_name = container
            else:
                self.container_sdf, inner = container, None      # a user SDF IS the cavity
                self.container_name = "custom"
            # ripple bank: wavelengths scaled to the vessel so a glass gets millimetre chop, a pool gets
            # centimetre wavelets -- the SAME wave model at a different octave.
            scale = (inner["r"] if inner else size)
            kw = dict(n_waves=12, wavelength_range=(0.25 * scale, 1.6 * scale),
                      amplitude_scale=0.05 * scale, choppiness=0.0, spread=3.14)
            kw.update(wave_overrides)
            self.bank = gerstner_waves(seed=seed, **kw)
            if inner is not None:
                level_y = inner["floor"] + float(level) * (inner["rim"] - inner["floor"])
                from holographic.mesh_and_geometry.holographic_sdf import cylinder, box
                if inner["kind"] == "cyl":
                    cavity = cylinder(h=inner["rim"] - inner["floor"], r=inner["r"]) \
                        .translate((0, (inner["rim"] + inner["floor"]) / 2, 0))
                else:
                    cavity = box(bx=inner["r"], by=(inner["rim"] - inner["floor"]) / 2, bz=inner["r"]) \
                        .translate((0, (inner["rim"] + inner["floor"]) / 2, 0))
                self._rim = inner["rim"]
            else:
                cavity, level_y = self.container_sdf, float(level)
                self._rim = level_y
            ripple_amp = float(ripple)
            self.water_sdf = _ContainedWaterSDF(cavity, level_y, self.bank, ripple_amp)
            self.water_sdf.t = float(t)
            self.ground_sdf = plane(h=0.0)
            def _scene(P):
                P = np.atleast_2d(np.asarray(P, float))
                return np.minimum(np.minimum(np.asarray(self.container_sdf(P)).reshape(len(P)),
                                             self.water_sdf(P)),
                                  np.asarray(self.ground_sdf(P)).reshape(len(P)))

            class _SceneSDF:                    # the tracer's dielectric traversal calls .eval(); a bare
                eval = staticmethod(_scene)     # closure has none, so carry both call forms
                __call__ = staticmethod(_scene)
            self.scene_sdf = _SceneSDF()
            glass_ior = 1.5
            mat_color, ior, self_c = self.mat_color, self.ior, self
            def material_fn(P):
                n = len(P)
                dg = np.abs(np.asarray(self_c.container_sdf(P)).reshape(n))
                dw = np.abs(self_c.water_sdf(P))
                df = np.abs(np.asarray(self_c.ground_sdf(P)).reshape(n))
                which = np.argmin(np.stack([dg, dw, df]), axis=0)
                albedo = np.empty((n, 3)); iors = np.zeros(n)
                rough = np.full(n, 0.04); metal = np.zeros(n)
                g, w_, f = which == 0, which == 1, which == 2
                albedo[g] = (0.97, 0.98, 0.99); iors[g] = glass_ior
                tint = np.clip(0.55 + mat_color * 4.0, 0.0, 0.99)        # matlib albedo -> transmission tint
                albedo[w_] = tint; iors[w_] = ior
                albedo[f] = (0.55, 0.50, 0.44); rough[f] = 0.75
                return (albedo, metal, rough, np.zeros((n, 3)), iors)
            self.material_fn = material_fn

    # -- the bundled knowledge: cameras and lighting that come out right first time -----------------
    def camera(self, aspect=4.0 / 3.0):
        from holographic.rendering.holographic_render import Camera
        if self.kind == "open":
            e = self.extent
            return Camera(eye=(0.0, 0.058 * e, 0.48 * e), target=(0.0, 0.008 * e, -0.08 * e),
                          fov_deg=55.0, aspect=aspect)
        r = self._rim
        return Camera(eye=(1.17 * r, 0.87 * r, 1.91 * r), target=(0.0, 0.48 * r, 0.0),
                      fov_deg=38.0, aspect=aspect)

    def render(self, quality="fast", width=None, height=None, camera=None, seed=0, sky=None):
        """Render this body. OPEN water: 'fast' = 560x420 shaded-mesh raster (~2 s), 'final' = 1024x768
        (~8 s). CONTAINED water: 'fast' = 300x225 @ 20 spp refractive path trace, 'final' = 420x315 @ 64
        spp (both true Fresnel/Snell/TIR dielectric; the fast tier is noisier, not wronger). Lighting and
        tone (plain gamma; Reinhard measured to wash out these mid-range scenes) are pre-balanced.

        `sky` (open water only; default None = the flat sky colour, unchanged): an (H,W,3) image -- e.g. a
        cloud_scene render -- composited as the BACKGROUND behind the water, nearest-resized to fit. The
        water+clouds one-liner: wb.render(sky=m.cloud_scene('sunset', 'fast')). Implementation: the
        rasteriser writes the exact background value to uncovered pixels, so rendering against a NEGATIVE
        sentinel (shading is clipped >= 0) yields an exact coverage mask -- no alpha channel needed."""
        if quality not in ("fast", "final"):
            raise ValueError("quality must be 'fast' or 'final', got %r" % (quality,))
        cam = camera or self.camera()
        if self.kind == "open":
            from holographic.rendering.holographic_render import rasterize_mesh
            W, H = (width or (560 if quality == "fast" else 1024)), (height or (420 if quality == "fast" else 768))
            eye = tuple(cam.eye)
            bg = (0.66, 0.80, 0.92) if sky is None else (-1.0, -1.0, -1.0)   # sentinel: impossible shading output
            img = rasterize_mesh(self.mesh, cam, width=W, height=H,
                              vertex_colors=shade_water_vertices(self.surface, eye),
                              lights=[], ambient=1.0, smooth=True, background=bg)
            img = np.asarray(img, float)
            if sky is not None:
                mask = np.all(img == -1.0, axis=-1)                          # exact: untouched background only
                s = np.asarray(sky, float)
                if s.ndim == 2:
                    s = np.stack([s] * 3, axis=-1)
                ys = np.clip((np.arange(H) * (s.shape[0] - 1) / max(H - 1, 1)).round().astype(int), 0, s.shape[0] - 1)
                xs = np.clip((np.arange(W) * (s.shape[1] - 1) / max(W - 1, 1)).round().astype(int), 0, s.shape[1] - 1)
                img[mask] = s[np.ix_(ys, xs)][mask]
            return np.clip(img, 0.0, 1.0)
        from holographic.rendering.holographic_pathtrace import path_trace
        from holographic.rendering.holographic_raymarch import sky_dome
        def bright_sky(D):
            return 1.2 * sky_dome(D, sun_dir=(0.5, 0.65, 0.35), sun_color=(1.0, 0.96, 0.88),
                                  sky_color=(0.46, 0.64, 0.95), horizon=(0.88, 0.90, 0.92),
                                  ground=(0.42, 0.40, 0.37), sun_size=0.05)
        W, H, spp = ((width or 300), (height or 225), 20) if quality == "fast" else ((width or 420), (height or 315), 64)
        img = path_trace(self.scene_sdf, cam, width=W, height=H, spp=spp, max_bounce=8,
                         material=self.material_fn, sky=bright_sky, seed=seed)
        return np.clip(np.asarray(img, float), 0.0, 1.0) ** (1.0 / 2.2)

    def at_time(self, t):
        """The same body at a different time -- coherent animation (same seed/bank). Returns a NEW WaterBody
        for open water; for contained water it retunes the ripple phase in place and returns self."""
        if self.kind == "open":
            return water_body(None, 0.72, self.preset, 1.0, self.extent,
                              self.surface["height"].shape[0], t, self.seed, 0.35, self.material_name)
        self.water_sdf.t = float(t)
        self.t = float(t)
        return self

if __name__ == "__main__":
    _selftest()
