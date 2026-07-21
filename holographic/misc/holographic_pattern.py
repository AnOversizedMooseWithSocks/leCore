"""holographic_pattern.py -- deterministic PROCEDURAL PATTERN FIELDS for material/parameter maps.

WHY THIS EXISTS
---------------
`holographic_param` gave every parameter a SOCKET: a channel can be a constant, a baked `map`, a wired `source`,
or a `field` -- a callable f(points (M,D)) -> (M,) values. `noise_field` gave ONE band-limited noise field. What
was missing is the small library of NAMED procedural patterns every material system exposes (Substance, Blender's
texture nodes, RenderMan patterns): checker, stripes, gradient, dots, value-noise, fbm. This module supplies them
as plain callables so they drop straight into a `Param(field=...)` and get resolved per point by `resolve_param` --
which means ANY faculty that reads a Param (the region field's material/reflect/roughness, an emitter's rate, a
material channel) can be driven by a procedural texture, not just a number.

DESIGN (kept honest and on-thesis)
----------------------------------
  * DETERMINISTIC. The value-noise uses an INTEGER lattice hash (pure arithmetic, PYTHONHASHSEED-independent) -- not
    Python's `hash()` and not `np.random` inside the field -- so the same point always yields the same value, run to
    run, exactly like the planet's fBm. Seedable via an integer that perturbs the hash.
  * Every generator returns a callable that maps world points -> a scalar field in [0, 1]. Compose them (a channel
    LERPs between lo..hi by the pattern) with `field_lerp`. NumPy / stdlib only; no learned weights.
  * These are FIELDS over space (3-D world position), so a pattern on a curved surface is a genuine solid texture --
    it wraps around the object without a UV unwrap, the field-native way.
"""
import numpy as np


# ---------------------------------------------------------------------------------------------------------------
# Deterministic value noise: an integer-lattice hash (arithmetic, seed-perturbed) trilinearly interpolated. Same
# family as the planet's fBm -- reproducible to the bit, independent of PYTHONHASHSEED.
# ---------------------------------------------------------------------------------------------------------------
def _hash01(ix, iy, iz, seed):
    """A deterministic hash of an integer lattice cell -> a float in [0,1). Pure int64 arithmetic (wraps), so it is
    reproducible run-to-run and does NOT use Python's salted hash()."""
    h = (ix.astype(np.int64) * np.int64(73856093)) ^ (iy.astype(np.int64) * np.int64(19349663)) \
        ^ (iz.astype(np.int64) * np.int64(83492791)) ^ (np.int64(seed) * np.int64(2654435761))
    h = (h ^ (h >> np.int64(13))) * np.int64(1274126177)
    h = h ^ (h >> np.int64(16))
    return (h & np.int64(0xFFFFFF)).astype(np.float64) / float(0x1000000)     # 24 bits -> [0,1)


def _smooth(t):
    """Quintic smoothstep (Perlin's) -- C2 interpolation weight, so the noise has no visible lattice creases."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def value_noise(scale=3.0, seed=0):
    """A callable f(P (M,3)) -> [0,1] value noise: hash the 8 lattice corners around each point and trilinearly blend
    with a quintic weight. `scale` sets the feature frequency (cells per unit)."""
    def field(P):
        P = np.atleast_2d(np.asarray(P, float)) * float(scale)
        i = np.floor(P).astype(np.int64); f = P - i
        w = _smooth(f)
        ix, iy, iz = i[:, 0], i[:, 1], i[:, 2]
        def corner(dx, dy, dz):
            return _hash01(ix + dx, iy + dy, iz + dz, seed)
        c000, c100 = corner(0, 0, 0), corner(1, 0, 0)
        c010, c110 = corner(0, 1, 0), corner(1, 1, 0)
        c001, c101 = corner(0, 0, 1), corner(1, 0, 1)
        c011, c111 = corner(0, 1, 1), corner(1, 1, 1)
        x00 = c000 * (1 - w[:, 0]) + c100 * w[:, 0]
        x10 = c010 * (1 - w[:, 0]) + c110 * w[:, 0]
        x01 = c001 * (1 - w[:, 0]) + c101 * w[:, 0]
        x11 = c011 * (1 - w[:, 0]) + c111 * w[:, 0]
        y0 = x00 * (1 - w[:, 1]) + x10 * w[:, 1]
        y1 = x01 * (1 - w[:, 1]) + x11 * w[:, 1]
        return y0 * (1 - w[:, 2]) + y1 * w[:, 2]
    return field


def fbm(scale=2.0, octaves=4, seed=0, gain=0.5, lacunarity=2.0):
    """Fractal (fBm) noise: sum octaves of value_noise at rising frequency and falling amplitude -- clouds, marble,
    weathering. Returns f(P) -> [0,1]."""
    bands = [value_noise(scale * lacunarity ** k, seed + 17 * k) for k in range(octaves)]
    amps = [gain ** k for k in range(octaves)]
    norm = sum(amps)
    def field(P):
        acc = np.zeros(len(np.atleast_2d(P)))
        for b, a in zip(bands, amps):
            acc = acc + a * b(P)
        return acc / norm
    return field


def domain_warped_fbm(scale=2.0, octaves=4, seed=0, warp=0.4, warp_scale=1.0, gain=0.5, lacunarity=2.0):
    """DOMAIN-WARPED fBm (iq's "warped noise" / dFBM): sample fbm at a point that has itself been DISPLACED by a
    vector of other fbm fields. f(P) = fbm(P + warp * [fbm_x(P), fbm_y(P), fbm_z(P)]). The displacement swirls the
    iso-contours into the flowing, turbulent, marbled look plain fbm cannot make -- smoke, magma, wood grain,
    weather fronts. `warp` is the displacement strength; `warp_scale` the frequency of the warp field. Returns
    f(P) -> [0,1]. This is the single most demoscene-recognisable noise (iq's "clouds" and "warping" articles).

    WHY three warp fields: each output axis needs its OWN noise or the displacement is diagonal (all axes move
    together) and the swirl collapses to a shear. Independent seeds give a true 3-D flow."""
    base = fbm(scale, octaves, seed, gain, lacunarity)
    # three INDEPENDENT warp fields (different seeds) so the displacement is a real 3-D vector, not a shear.
    wx = fbm(scale * warp_scale, max(1, octaves - 1), seed + 101, gain, lacunarity)
    wy = fbm(scale * warp_scale, max(1, octaves - 1), seed + 211, gain, lacunarity)
    wz = fbm(scale * warp_scale, max(1, octaves - 1), seed + 331, gain, lacunarity)

    def field(P):
        P = np.atleast_2d(np.asarray(P, float))
        # centre the warp fields to ~[-0.5, 0.5] so the displacement pushes both ways, not just positive.
        disp = np.stack([wx(P) - 0.5, wy(P) - 0.5, wz(P) - 0.5], axis=1)
        return base(P + warp * disp)
    return field


# ---------------------------------------------------------------------------------------------------------------
# GPU-REPRODUCIBLE noise (backlog C2): value_noise/fbm above use the int64 _hash01, which GLSL ES 3.00 cannot
# reproduce (32-bit ints) -- that is why pattern_to_glsl REFUSED them. value_noise32/fbm32 are the 32-bit twins: same
# quintic-trilinear structure, but the corner hash is holographic_determinism.hash32_unit (PCG, 32-bit), so the SAME
# value is recomputable in a GLSL `uint`. The int64 noise stays the DEFAULT (its output is unchanged); noise32/fbm32
# are the new, additive, EMITTABLE kinds.
def _hash01_32(ix, iy, iz, seed):
    """One lattice cell -> [0,1) via the 32-bit PCG fold (reuses hash32_unit; NO second hash convention)."""
    from holographic.misc.holographic_determinism import hash32_unit
    return hash32_unit(ix, iy, iz, seed=seed)


def value_noise32(scale=3.0, seed=0):
    """GPU-REPRODUCIBLE value noise: identical to value_noise (quintic-weighted trilinear blend of 8 corner hashes)
    but the corner hash is the 32-bit PCG (hash32_unit), so pattern_to_glsl('noise32') emits a GLSL twin that matches
    this per-point (to float32 precision). Returns f(P (M,3)) -> [0,1]."""
    def field(P):
        P = np.atleast_2d(np.asarray(P, float)) * float(scale)
        i = np.floor(P).astype(np.int64); f = P - i
        w = _smooth(f)
        ix, iy, iz = i[:, 0], i[:, 1], i[:, 2]
        def corner(dx, dy, dz):
            return _hash01_32(ix + dx, iy + dy, iz + dz, seed)
        c000, c100 = corner(0, 0, 0), corner(1, 0, 0)
        c010, c110 = corner(0, 1, 0), corner(1, 1, 0)
        c001, c101 = corner(0, 0, 1), corner(1, 0, 1)
        c011, c111 = corner(0, 1, 1), corner(1, 1, 1)
        x00 = c000 * (1 - w[:, 0]) + c100 * w[:, 0]
        x10 = c010 * (1 - w[:, 0]) + c110 * w[:, 0]
        x01 = c001 * (1 - w[:, 0]) + c101 * w[:, 0]
        x11 = c011 * (1 - w[:, 0]) + c111 * w[:, 0]
        y0 = x00 * (1 - w[:, 1]) + x10 * w[:, 1]
        y1 = x01 * (1 - w[:, 1]) + x11 * w[:, 1]
        return y0 * (1 - w[:, 2]) + y1 * w[:, 2]
    return field


def fbm32(scale=2.0, octaves=4, seed=0, gain=0.5, lacunarity=2.0):
    """fBm on the GPU-reproducible value_noise32 -- octave-summed, same shape as fbm, and EMITTABLE. f(P) -> [0,1]."""
    bands = [value_noise32(scale * lacunarity ** k, seed + 17 * k) for k in range(octaves)]
    amps = [gain ** k for k in range(octaves)]
    norm = sum(amps)
    def field(P):
        acc = np.zeros(len(np.atleast_2d(P)))
        for b, a in zip(bands, amps):
            acc = acc + a * b(P)
        return acc / norm
    return field


# ---------------------------------------------------------------------------------------------------------------
# Exact (non-noise) patterns: checker, stripes, gradient, dots. Deterministic by construction.
# ---------------------------------------------------------------------------------------------------------------
def checker(scale=2.0):
    """3-D checkerboard: parity of the summed floored coordinates -> {0,1}. A solid texture (wraps any surface)."""
    def field(P):
        P = np.atleast_2d(np.asarray(P, float)) * float(scale)
        s = np.floor(P[:, 0]) + np.floor(P[:, 1]) + np.floor(P[:, 2])
        return (np.mod(s, 2.0) < 1.0).astype(float)
    return field


def stripes(scale=3.0, axis=1, sharp=False):
    """Sinusoidal (or hard) stripes along an axis, in [0,1]. `sharp=True` -> square-wave bands."""
    def field(P):
        P = np.atleast_2d(np.asarray(P, float))
        v = 0.5 + 0.5 * np.sin(P[:, int(axis)] * float(scale) * 2.0 * np.pi)
        return (v > 0.5).astype(float) if sharp else v
    return field


def gradient(axis=1, lo=-1.0, hi=1.0):
    """A linear ramp along an axis, normalised to [0,1] over [lo,hi] -- a height/position gradient."""
    def field(P):
        P = np.atleast_2d(np.asarray(P, float))
        return np.clip((P[:, int(axis)] - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return field


def dots(scale=3.0, radius=0.3):
    """Round dots on a lattice: 1 near each cell centre, 0 between -> polka / rivets. In [0,1]."""
    def field(P):
        P = np.atleast_2d(np.asarray(P, float)) * float(scale)
        d = np.abs(P - np.round(P))
        r = np.linalg.norm(d, axis=1)
        return np.clip(1.0 - r / max(radius, 1e-3), 0.0, 1.0)
    return field


# --------------------------------------------------------------------------------------------------------------
# GLSL emission (leStudio backlog 10): the CLOSED-FORM patterns compile to a `float pattern(vec3 p)` GLSL function,
# so a procedural background renders client-side and composes with the postfx / SDF emitters into GPU-resident looks.
#
# WHY ONLY THE CLOSED-FORM KINDS. checker/stripes/gradient/dots are pure arithmetic and transcribe to GLSL EXACTLY
# (match the numpy field per-point to float precision on a probe grid). value_noise/fbm are NOT emittable per-point:
# their determinism comes from an INTEGER-LATTICE hash done in int64 with wraparound (multipliers like 73856093,
# 1274126177 and >>13/>>16 shifts), and GLSL ES 3.00 has only 32-bit ints -- the 64-bit wrap cannot be reproduced,
# so a GPU noise would look similar but would NOT match on a probe grid (the stated acceptance). They raise with that
# reason rather than emit a shader that disagrees with the numpy field.
_GLSL_CLOSED_FORM = ("checker", "stripes", "gradient", "dots")


def _gf(x):
    """Format a float as a GLSL literal. Delegates to the shared holographic_emit.glsl_float (sweep-consolidated)."""
    from holographic.io_and_interop.holographic_emit import glsl_float
    return glsl_float(x)


def pattern_to_glsl(name, fn_name="pattern", **params):
    """Compile a CLOSED-FORM pattern to a GLSL `float <fn_name>(vec3 p)` function -- matches the numpy pattern field
    per-point to float precision (checker/stripes/gradient/dots). value_noise/fbm raise: their int64-lattice hash
    cannot be reproduced in GLSL ES 3.00's 32-bit ints (see the module note). For a 2-D background call the emitted
    function as `<fn_name>(vec3(uv, 0.0))`."""
    _AXIS = ("x", "y", "z")
    if name == "checker":
        s = _gf(params.get("scale", 2.0))
        body = ("vec3 q = p * %s;\n    float s = floor(q.x) + floor(q.y) + floor(q.z);\n"
                "    return mod(s, 2.0) < 1.0 ? 1.0 : 0.0;" % s)
    elif name == "stripes":
        s = _gf(params.get("scale", 3.0)); ax = _AXIS[int(params.get("axis", 1))]
        sharp = bool(params.get("sharp", False))
        line = "float v = 0.5 + 0.5 * sin(p.%s * %s * 6.28318530717959);" % (ax, s)
        body = line + ("\n    return v > 0.5 ? 1.0 : 0.0;" if sharp else "\n    return v;")
    elif name == "gradient":
        ax = _AXIS[int(params.get("axis", 1))]
        lo = _gf(params.get("lo", -1.0)); hi = _gf(params.get("hi", 1.0))
        body = "return clamp((p.%s - %s) / (%s - %s + 1e-9), 0.0, 1.0);" % (ax, lo, hi, lo)
    elif name == "dots":
        s = _gf(params.get("scale", 3.0)); rad = _gf(params.get("radius", 0.3))
        # numpy: d = abs(P - round(P)); r = |d|; clip(1 - r/max(radius,1e-3)). round-half handled by floor(x+0.5)
        # (differs from numpy's banker's rounding ONLY at exact .5, avoided on a generic probe grid).
        body = ("vec3 q = p * %s;\n    vec3 d = abs(q - floor(q + 0.5));\n    float r = length(d);\n"
                "    return clamp(1.0 - r / max(%s, 1e-3), 0.0, 1.0);" % (s, rad))
    elif name in ("noise32", "fbm32"):
        # GPU-REPRODUCIBLE noise (C2): emit the PCG helper + a seed-parameterised value-noise, so the shader matches
        # value_noise32/fbm32 per-point (to float32 precision). This is what value_noise/fbm could NOT do (int64 hash).
        prelude = _glsl_noise_prelude()
        if name == "noise32":
            s = _gf(params.get("scale", 3.0)); seed = int(params.get("seed", 0))
            body = "return _vnoise32(p * %s, %du);" % (s, seed & 0xFFFFFFFF)
        else:
            scale = float(params.get("scale", 2.0)); octaves = int(params.get("octaves", 4))
            seed = int(params.get("seed", 0)); gain = float(params.get("gain", 0.5))
            lac = float(params.get("lacunarity", 2.0))
            amps = [gain ** k for k in range(octaves)]; norm = sum(amps)
            lines = ["float f = 0.0;"]
            for k in range(octaves):
                lines.append("f += %s * _vnoise32(p * %s, %du);"
                             % (_gf(amps[k]), _gf(scale * lac ** k), (seed + 17 * k) & 0xFFFFFFFF))
            lines.append("return f / %s;" % _gf(norm))
            body = "\n    ".join(lines)
        return "%s\n\n// pattern '%s' as GLSL (matches holographic_pattern.%s per-point to float32)\nfloat %s(vec3 p){\n    %s\n}\n" \
            % (prelude, name, name, fn_name, body)
    elif name in ("noise", "fbm"):
        raise ValueError("pattern_to_glsl: %r uses the int64 lattice hash, not GLSL-emittable in ES 3.00 32-bit ints; "
                         "use the GPU-reproducible twin %r instead (same look, PCG 32-bit hash). Closed-form kinds: %s"
                         % (name, name + "32", ", ".join(_GLSL_CLOSED_FORM)))
    else:
        raise ValueError("pattern_to_glsl: unknown pattern %r; GLSL-emittable kinds: %s (+ noise32/fbm32)"
                         % (name, ", ".join(_GLSL_CLOSED_FORM)))
    return "// pattern '%s' as GLSL (matches holographic_pattern.%s to float precision)\nfloat %s(vec3 p){\n    %s\n}\n" \
        % (name, name, fn_name, body)


def pattern_to_wgsl(name, fn_name="pattern", **params):
    """Compile a CLOSED-FORM pattern to a WGSL `fn <fn_name>(p: vec3<f32>) -> f32` (backlog C5, WebGPU) -- matches the
    numpy pattern field per-point to f32 precision (checker/stripes/gradient/dots). WGSL is NOT GLSL with renamed
    types: no `mod` (spelled out as x - k*floor(x/k)), `select(false, true, cond)` for the ternary, `vec3<f32>` and
    `let`. So this EMITS from the same math, it does not machine-translate the GLSL string.

    DEFERRED (documented scope, not built): noise32/fbm32 (WGSL u32 bit-ops differ enough to warrant their own
    verified pass), palette, and the texture-sampling shaders (postfx/sdf) -- those need WGSL's binding/entry-point
    model (@group/@binding, textureSample), a much larger surface reconciled against the WebGPU plan. This ships the
    closed-form beachhead, per-point verified."""
    _AXIS = ("x", "y", "z")
    if name == "checker":
        s = _gf(params.get("scale", 2.0))
        body = ("let q = p * %s;\n    let s = floor(q.x) + floor(q.y) + floor(q.z);\n"
                "    return select(0.0, 1.0, (s - 2.0 * floor(s / 2.0)) < 1.0);" % s)
    elif name == "stripes":
        s = _gf(params.get("scale", 3.0)); ax = _AXIS[int(params.get("axis", 1))]
        sharp = bool(params.get("sharp", False))
        line = "let v = 0.5 + 0.5 * sin(p.%s * %s * 6.28318530717959);" % (ax, s)
        body = line + ("\n    return select(0.0, 1.0, v > 0.5);" if sharp else "\n    return v;")
    elif name == "gradient":
        ax = _AXIS[int(params.get("axis", 1))]
        lo = _gf(params.get("lo", -1.0)); hi = _gf(params.get("hi", 1.0))
        body = "return clamp((p.%s - %s) / (%s - %s + 1e-9), 0.0, 1.0);" % (ax, lo, hi, lo)
    elif name == "dots":
        s = _gf(params.get("scale", 3.0)); rad = _gf(params.get("radius", 0.3))
        body = ("let q = p * %s;\n    let d = abs(q - floor(q + 0.5));\n    let r = length(d);\n"
                "    return clamp(1.0 - r / max(%s, 1e-3), 0.0, 1.0);" % (s, rad))
    elif name in ("noise", "fbm", "noise32", "fbm32"):
        raise ValueError("pattern_to_wgsl: %r not yet emitted to WGSL (deferred scope -- closed-form kinds only: %s). "
                         "The noise WGSL path needs its own verified u32 pass." % (name, ", ".join(_GLSL_CLOSED_FORM)))
    else:
        raise ValueError("pattern_to_wgsl: unknown pattern %r; WGSL-emittable kinds: %s"
                         % (name, ", ".join(_GLSL_CLOSED_FORM)))
    return "// pattern '%s' as WGSL (matches holographic_pattern.%s to f32 precision)\nfn %s(p: vec3<f32>) -> f32 {\n    %s\n}\n" \
        % (name, name, fn_name, body)


def _glsl_noise_prelude():
    """The GLSL helpers for value_noise32/fbm32: the PCG hash (from holographic_determinism) + a seed-parameterised
    lattice hash `_h01_32` (matching hash32_unit's fold) + `_vnoise32` (quintic-trilinear). Shared by noise32/fbm32."""
    from holographic.misc.holographic_determinism import hash32_pcg_glsl
    pcg = hash32_pcg_glsl("_pcg32")
    h01 = ("float _h01_32(ivec3 c, uint seed){\n"
           "    uint acc = (seed * 0x9E3779B1u) ^ (uint(c.x) * 0x9E3779B1u)\n"
           "             ^ (uint(c.y) * 0x85EBCA77u) ^ (uint(c.z) * 0xC2B2AE3Du);\n"
           "    return float(_pcg32(acc)) / 4294967296.0;\n"
           "}")
    vn = ("float _vnoise32(vec3 q, uint seed){\n"
          "    ivec3 i = ivec3(floor(q));\n"
          "    vec3 f = q - vec3(i);\n"
          "    vec3 w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);\n"
          "    float c000 = _h01_32(i + ivec3(0,0,0), seed); float c100 = _h01_32(i + ivec3(1,0,0), seed);\n"
          "    float c010 = _h01_32(i + ivec3(0,1,0), seed); float c110 = _h01_32(i + ivec3(1,1,0), seed);\n"
          "    float c001 = _h01_32(i + ivec3(0,0,1), seed); float c101 = _h01_32(i + ivec3(1,0,1), seed);\n"
          "    float c011 = _h01_32(i + ivec3(0,1,1), seed); float c111 = _h01_32(i + ivec3(1,1,1), seed);\n"
          "    return mix(mix(mix(c000, c100, w.x), mix(c010, c110, w.x), w.y),\n"
          "               mix(mix(c001, c101, w.x), mix(c011, c111, w.x), w.y), w.z);\n"
          "}")
    return pcg + "\n\n" + h01 + "\n\n" + vn


# named registry so a UI / serialized spec can ask for a pattern by name
PATTERNS = {
    "noise": value_noise, "fbm": fbm, "checker": checker,
    "stripes": stripes, "gradient": gradient, "dots": dots,
    "noise32": value_noise32, "fbm32": fbm32,               # GPU-reproducible twins (backlog C2), emittable to GLSL
}


# Kwargs that MEAN something for the stochastic kinds (value_noise/fbm take a seed) but are INERT for the exact,
# deterministic kinds (checker/stripes/gradient/dots have nothing to randomise). WHY this list exists: a UI that
# exposes one "kind" dropdown next to a "seed" slider sends the union of parameters for every kind; without this,
# make_pattern("checker", seed=3) raised TypeError on 4 of the 6 advertised kinds and crashed the UI. We DROP only
# these documented-inert keys when the target builder can't take them -- any OTHER unexpected kwarg still raises a
# TypeError, so a genuine typo (scl= for scale=) is NOT silently swallowed (same honesty rule as color_transfer's
# mode validation: absorb the known no-op, surface the real mistake).
_INERT_FOR_DETERMINISTIC = ("seed",)


def make_pattern(name, **params):
    """Build a pattern FIELD by name (for a serialized map spec). Unknown name -> a constant 0.5 field.

    Uniform signature across kinds: every advertised kind accepts `seed` -- it drives the noise for 'noise'/'fbm'
    and is a documented NO-OP for the deterministic kinds ('checker'/'stripes'/'gradient'/'dots'), so a kind-agnostic
    caller (a UI dropdown + seed slider) never has to special-case which kinds are stochastic. Any kwarg that is
    neither accepted by the builder nor a known no-op still raises TypeError, so typos are not swallowed."""
    import inspect
    fn = PATTERNS.get(name)
    if fn is None:
        return lambda P: np.full(len(np.atleast_2d(P)), 0.5)
    accepted = set(inspect.signature(fn).parameters)
    # drop ONLY the documented-inert keys the chosen builder does not accept; leave everything else to raise.
    params = {k: v for k, v in params.items()
              if k in accepted or k not in _INERT_FOR_DETERMINISTIC}
    return fn(**params)


def field_lerp(pattern, lo, hi):
    """Wrap a [0,1] pattern into a channel field that LERPs between `lo` and `hi` (scalars OR (3,) colours). This is
    how a pattern DRIVES a material channel: roughness lo..hi, or colour A..B, by the pattern value."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    def field(P):
        w = np.asarray(pattern(P), float)
        if lo.ndim == 0:
            return lo + (hi - lo) * w
        return lo[None, :] + (hi - lo)[None, :] * w[:, None]        # colour / vector channel
    return field


def _selftest():
    """Determinism + range: every pattern is in [0,1], and the noise is identical across two evaluations and two
    process-independent calls (integer hash, not salted hash())."""
    P = np.random.default_rng(0).uniform(-2, 2, (2000, 3))
    for name in PATTERNS:
        f = make_pattern(name)
        v = np.asarray(f(P), float)
        assert v.min() >= -1e-9 and v.max() <= 1.0 + 1e-9, (name, v.min(), v.max())
    n = value_noise(3.0, seed=7)
    assert np.allclose(n(P), n(P))                                   # deterministic
    # a channel driven lo..hi by a pattern stays within [lo,hi]
    g = field_lerp(make_pattern("checker", scale=2.0), 0.1, 0.9)
    gv = g(P); assert gv.min() >= 0.1 - 1e-9 and gv.max() <= 0.9 + 1e-9

    # ITEM 2: uniform signature -- make_pattern(kind, seed=n) succeeds for EVERY advertised kind (it used to raise
    # TypeError on checker/stripes/gradient/dots). And seed is a genuine NO-OP for the deterministic kinds: the
    # field is bit-identical with and without it (so a UI slider can't accidentally change a checker pattern).
    for name in PATTERNS:
        f = make_pattern(name, seed=3)                                  # no crash on any kind
        assert np.asarray(f(P)).shape[0] == len(P)
    for name in ("checker", "stripes", "gradient", "dots"):             # exact kinds: seed changes nothing
        a = np.asarray(make_pattern(name)(P))
        b = np.asarray(make_pattern(name, seed=999)(P))
        assert np.array_equal(a, b), name
    # KEPT NEGATIVE: only the documented-inert `seed` is absorbed; a genuine typo still raises, so the fix does NOT
    # reintroduce silent-parameter-drop (the exact failure item 3 exists to kill).
    try:
        make_pattern("checker", not_a_real_param=1); raised = False
    except TypeError:
        raised = True
    assert raised, "make_pattern must still reject unknown (non-inert) kwargs"

    # ITEM 10: the closed-form patterns compile to a GLSL `float pattern(vec3 p)` that matches the numpy field
    # per-point to float precision; noise/fbm raise (int64 hash not GLSL-emittable). Verified with an INDEPENDENT
    # transcription of the emitted GLSL math on a probe grid (offset off exact halves for dots' rounding).
    Pg = np.random.default_rng(1).uniform(-2.3, 2.3, (400, 3)) + 0.017
    def _glsl_ref(nm, P, **kw):
        if nm == "checker":
            q = P * kw.get("scale", 2.0); s = np.floor(q[:, 0]) + np.floor(q[:, 1]) + np.floor(q[:, 2])
            return np.where(np.mod(s, 2.0) < 1.0, 1.0, 0.0)
        if nm == "stripes":
            v = 0.5 + 0.5 * np.sin(P[:, kw.get("axis", 1)] * kw.get("scale", 3.0) * 6.28318530717959)
            return np.where(v > 0.5, 1.0, 0.0) if kw.get("sharp", False) else v
        if nm == "gradient":
            return np.clip((P[:, kw.get("axis", 1)] - kw.get("lo", -1.0)) / (kw.get("hi", 1.0) - kw.get("lo", -1.0) + 1e-9), 0, 1)
        if nm == "dots":
            q = P * kw.get("scale", 3.0); d = np.abs(q - np.floor(q + 0.5))
            return np.clip(1 - np.linalg.norm(d, axis=1) / max(kw.get("radius", 0.3), 1e-3), 0, 1)
    for nm, kw in [("checker", {"scale": 2.0}), ("stripes", {"scale": 3.0, "axis": 0}),
                   ("gradient", {"axis": 2, "lo": -1.5, "hi": 1.5}), ("dots", {"scale": 3.0, "radius": 0.35})]:
        g = pattern_to_glsl(nm, **kw)
        assert "float pattern(vec3 p)" in g
        assert np.abs(np.asarray(make_pattern(nm, **kw)(Pg), float) - _glsl_ref(nm, Pg, **kw)).max() < 1e-9, nm
    for bad in ("noise", "fbm"):
        try:
            pattern_to_glsl(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("%s must raise -- int64 lattice hash is not GLSL-emittable" % bad)
    # KEPT NEGATIVE (int64 noise/fbm): the DEFAULT value_noise/fbm are still NOT emitted -- their int64-wrap hash
    # cannot be reproduced in GLSL ES 3.00's 32-bit ints. That negative stands. C2 does NOT change them; it adds the
    # 32-bit TWINS below.

    # C2: value_noise32/fbm32 ARE GPU-reproducible -- built on the 32-bit PCG hash (hash32_unit), so pattern_to_glsl
    # emits a GLSL twin that matches them PER-POINT to float32 precision (the int64 noise could not do this). Verified
    # by an independent float32 transcription of the emitted _vnoise32 (same PCG both sides).
    from holographic.misc.holographic_determinism import hash32_pcg as _h32
    def _glsl_vnoise32(P, scale, seed):
        with np.errstate(over="ignore"):
            q = np.asarray(P, np.float32) * np.float32(scale); i = np.floor(q).astype(np.int64)
            f = (q - i.astype(np.float32)).astype(np.float32)
            w = (f * f * f * (f * (f * np.float32(6) - np.float32(15)) + np.float32(10))).astype(np.float32)
            def h01(c):
                acc = (np.uint32(seed) * np.uint32(0x9E3779B1)) ^ (c[:, 0].astype(np.uint32) * np.uint32(0x9E3779B1)) \
                      ^ (c[:, 1].astype(np.uint32) * np.uint32(0x85EBCA77)) ^ (c[:, 2].astype(np.uint32) * np.uint32(0xC2B2AE3D))
                return _h32(acc).astype(np.float32) * np.float32(1.0 / 4294967296.0)
            def cor(dx, dy, dz): return h01(i + np.array([dx, dy, dz]))
            c000, c100 = cor(0, 0, 0), cor(1, 0, 0); c010, c110 = cor(0, 1, 0), cor(1, 1, 0)
            c001, c101 = cor(0, 0, 1), cor(1, 0, 1); c011, c111 = cor(0, 1, 1), cor(1, 1, 1)
            def mix(a, b, t): return (a * (np.float32(1) - t) + b * t).astype(np.float32)
            wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
            return mix(mix(mix(c000, c100, wx), mix(c010, c110, wx), wy),
                       mix(mix(c001, c101, wx), mix(c011, c111, wx), wy), wz).astype(float)
    Pn = np.random.default_rng(2).uniform(-3, 3, (400, 3)) + 0.013
    for sc, sd in [(3.0, 0), (2.5, 7)]:
        gg = pattern_to_glsl("noise32", scale=sc, seed=sd)
        assert "uint _pcg32(uint v)" in gg and "_vnoise32" in gg and "float pattern(vec3 p)" in gg
        err = np.abs(np.asarray(make_pattern("noise32", scale=sc, seed=sd)(Pn), float) - _glsl_vnoise32(Pn, sc, sd)).max()
        assert err < 3e-6, ("noise32", sc, sd, err)
    # fbm32 = octave sum of value_noise32, also emittable and matching
    gf = pattern_to_glsl("fbm32", scale=2.0, octaves=3, seed=1)
    assert "_vnoise32" in gf and gf.count("_vnoise32(p") == 3                     # one call per octave
    # and noise32/fbm32 differ from their int64 twins (different hash) -- they are genuinely new kinds, not aliases
    assert not np.array_equal(np.asarray(make_pattern("noise", scale=3.0, seed=0)(Pn)),
                              np.asarray(make_pattern("noise32", scale=3.0, seed=0)(Pn)))

    # C5: closed-form patterns also emit to WGSL (WebGPU), matching the numpy field per-point to f32. WGSL is not GLSL
    # renamed -- no `mod`, `select` for the ternary, vec3<f32>/let -- so it is emitted from the same math.
    def _wgsl_ref(nm, P, **kw):
        P = P.astype(np.float32)
        if nm == "checker":
            q = P * np.float32(kw.get("scale", 2.0)); s = np.floor(q[:, 0]) + np.floor(q[:, 1]) + np.floor(q[:, 2])
            return np.where((s - np.float32(2.0) * np.floor(s / np.float32(2.0))) < np.float32(1.0), 1.0, 0.0)
        if nm == "gradient":
            ax = kw.get("axis", 1); lo = np.float32(kw.get("lo", -1.0)); hi = np.float32(kw.get("hi", 1.0))
            return np.clip((P[:, ax] - lo) / (hi - lo + np.float32(1e-9)), 0, 1)
        if nm == "dots":
            q = P * np.float32(kw.get("scale", 3.0)); d = np.abs(q - np.floor(q + np.float32(0.5)))
            return np.clip(np.float32(1.0) - np.sqrt((d * d).sum(1)) / max(kw.get("radius", 0.3), 1e-3), 0, 1)
    Pw = np.random.default_rng(4).uniform(-2.3, 2.3, (300, 3)).astype(np.float32) + np.float32(0.017)
    for nm, kw in [("checker", {"scale": 2.0}), ("gradient", {"axis": 2, "lo": -1.5, "hi": 1.5}),
                   ("dots", {"scale": 3.0, "radius": 0.35})]:
        w = pattern_to_wgsl(nm, **kw)
        assert "fn pattern(p: vec3<f32>) -> f32" in w and "float" not in w and "mod(" not in w   # real WGSL, no GLSL-isms
        assert np.abs(np.asarray(make_pattern(nm, **kw)(Pw), float) - np.asarray(_wgsl_ref(nm, Pw, **kw), float)).max() < 3e-6, nm
    for bad in ("noise", "noise32", "fbm32"):                          # DEFERRED kinds raise (documented scope)
        try:
            pattern_to_wgsl(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("pattern_to_wgsl(%r) must raise -- WGSL noise is deferred scope" % bad)

    # W11 dFBM: domain-warped fbm stays in [0,1], DIFFERS from plain fbm (the warp swirled it), reduces EXACTLY to
    # fbm at warp=0 (clean degradation), and is deterministic.
    plain = fbm(scale=2.0, seed=0)(P)
    warped = domain_warped_fbm(scale=2.0, seed=0, warp=0.5)(P)
    assert warped.min() >= -1e-9 and warped.max() <= 1.0 + 1e-9
    assert np.abs(warped - plain).mean() > 0.01                       # the warp actually changed the field
    assert np.allclose(domain_warped_fbm(scale=2.0, seed=0, warp=0.0)(P), plain, atol=1e-9)   # warp=0 == fbm
    assert np.allclose(domain_warped_fbm(scale=2.0, seed=0)(P), domain_warped_fbm(scale=2.0, seed=0)(P))
    print("holographic_pattern selftest OK (dFBM in [0,1], swirls vs fbm, warp=0 reduces to fbm, deterministic)")


if __name__ == "__main__":
    _selftest()
