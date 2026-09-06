"""Glass as a BAKED SPECTRAL-REFRACTIVE TRANSFER: trace the refraction once, relight by dot product (RENDER = QUERY).

WHY THIS MODULE EXISTS -- the number it attacks
------------------------------------------------
Sweeps 138-148 rendered glass by Monte-Carlo path tracing: per sample, reflect OR refract chosen at random by the
Fresnel term, march through the interior, refract out, bounce on, repeat for spp samples x 3-5 wavelength layers.
The corrected (solid) ladybird cost 30-100 minutes a frame; the 1-fold Mandelbox took ~5 minutes for 192 spp.
Almost all of that cost is VARIANCE: the estimator draws a coin per sample to decide reflect-vs-refract and a
wavelength per layer, then averages the coins away.

The engine's own lineage already says what to do instead. PRT ("collapse, don't trace"): the expensive part of
shading depends only on GEOMETRY, so precompute it once as a TRANSFER and make shading a dot product with the
light. bake_scene/render_baked: trace visibility once, every frame after is a relight. This module is that idea
applied to a dielectric:

  * The geometry of a glass pixel is DETERMINISTIC. Entry point, entry normal, the refracted interior ray, the exit
    point, the exit normal, the exit direction, the Fresnel split -- none of it is random. Only the path tracer's
    ESTIMATOR was random. So evaluate BOTH Fresnel branches, weighted by R and (1-R), ONCE. No coin, no variance,
    no spp. That is the collapse.
  * Wavelength enters only through n(lambda) in two Snell steps, and the exit direction is smooth in lambda -- so a
    handful of wavelengths (default 5, stratified) is not an approximation of a sampling loop, it is the sampling
    loop's exact limit for a smooth function. Each wavelength's exit direction is stored; the colour-matching
    weights fold in at READ time (the same move holographic_holocaustic makes with its query vectors).
  * The ENVIRONMENT is a field over directions. A softbox, a sky, a sun -- each is a set of (direction, radiance)
    samples, bundled into ONE FPE hypervector L over the unit sphere. Radiance in direction D is <L, encode(D)>.
    So a pixel's colour is

        pixel_c = sum_k w_c(lambda_k) [ (1-R) <L, encode(D_out_k)> + R <L, encode(D_refl)> ]

    -- a DOT PRODUCT between the pixel's baked exit directions and the light field. Change the light: rebuild L
    (one bundle), re-read every pixel (one matmul). Move the light: L changes, the bake does not. That is PRT
    generalised from diffuse spherical harmonics to spectral refractive transfer -- the "next rung" the render
    audit named.

WHAT THE BAKE STORES per pixel (compact -- this is what makes it a bake and not a cache):
    hit mask; K exit directions (K x 3); reflection direction (3); Fresnel R; Beer-Lambert path length;
    floor hit point + normal for pixels that miss the glass. Nothing per-sample, nothing per-light.

MEASURED (selftest + sweep 150 notes):
    bake of a 320x200 frame of a 1-fold Mandelbox at 5 wavelengths: seconds, one pass. Relight: one matmul.
    The MC path tracer needed 192 spp x 3 wavelengths x ~5 min for the same object. See NOTES for the numbers.

KEPT HONEST (loud):
  * Internal total-internal-reflection bounces ARE followed, deterministically, up to `max_internal` (default 4):
    a TIR ray reflects and marches to the next face -- no coin, no variance. Rays still trapped after the cap are
    the remaining loss. MEASURED on the 1-fold Mandelbox at 320x200: first-order (no internal bounces) dropped
    65.5% of glass-wavelength pairs and left the facets black; with 4 internal bounces the loss falls -- the
    figure is printed by the demo and pinned by the selftest. Each bounce is one more interior march per
    still-trapped ray, so the bake cost grows sub-linearly with the cap (most rays escape early).
  * The environment field is an FPE kernel-density estimate over S^2: a sharp softbox comes back as a soft one.
    Bandwidth is the knob; the default is derived from the dimension exactly as holocaustic's is. A light seen
    THROUGH the glass is therefore blurrier than a path tracer would draw it; a smooth sky is exact.
  * Shadows on the floor are direct-only (one shadow ray per light toward the floor hit) -- no caustic from the
    glass onto the floor here; composite holographic_caustic / caustic_pass for that, as before.
  * Empty pixels (sky) read the env field directly; a floor is Lambert against the same field.

Basis: Sloan/Kautz/Snyder PRT (2002) for the transfer idea; the engine's holographic_prt, holographic_dispatch
(bake_scene/render_baked) and holographic_holocaustic for the shape. NumPy/stdlib only, deterministic.
"""
import numpy as np

from holographic.rendering.holographic_raymarch import sphere_trace, sdf_normal, refract_dir
from holographic.rendering.holographic_pathtrace import _march_through
from holographic.rendering.holographic_brdf import fresnel_dielectric
from holographic.sampling_and_signal.holographic_fpe import VectorFunctionEncoder


# ------------------------------------------------------------------------------------------ the light as a field
class EnvField:
    """The environment's radiance over the unit sphere as ONE FPE hypervector per colour channel.

    Directions are encoded by their (x, y, z) components on S^2 -- a 3-axis encoder over [-1, 1]^3 -- so a light
    is deposited as encode(D_i) weighted by its radiance and read back as <L_c, encode(D)>. Deposit a sky as many
    samples, a softbox as a disc of samples, a sun as one bright sample. Bundling is superposition: add a light by
    adding its samples; two lights are two bundles summed. Reading many directions at once is one matmul."""

    def __init__(self, dim=512, bandwidth=None, seed=0):
        # dim 512 by default, not the caustic's 2048-16384: a LIGHT FIELD is smooth (softboxes, skies), so ~sqrt(512)
        # = 22 cells per axis on [-1,1]^3 is ~8 degrees of angular resolution -- softer than a softbox edge, sharper
        # than any diffuse shading needs. The relight cost is O(pixels x K x dim) direction encodes, so dim is the
        # knob that trades a sharper light for a slower read; measured 48^2 x 5 wavelengths: 1.08s at 1024.
        self.dim = int(dim)
        bw = 0.6 * np.sqrt(self.dim) if bandwidth is None else float(bandwidth)   # the capacity rule, as holocaustic
        self.enc = VectorFunctionEncoder(3, dim=self.dim, bounds=[(-1.0, 1.0)] * 3, bandwidth=[bw] * 3, seed=seed)
        self.Theta = np.stack([ax.scale * ax.phases for ax in self.enc.axes], axis=0)      # (3, dim)
        self.L_spec = np.zeros((3, self.dim), dtype=complex)                                # one field per channel
        self._cal = None

    def deposit(self, dirs, rgb, chunk=4096):
        """Bundle directional radiance samples in: L_c += sum_i rgb_i,c * encode(D_i)."""
        D = np.atleast_2d(np.asarray(dirs, float)); D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        C = np.atleast_2d(np.asarray(rgb, float))
        if C.shape[0] == 1 and D.shape[0] > 1:
            C = np.repeat(C, D.shape[0], axis=0)
        for s in range(0, len(D), chunk):
            e = min(len(D), s + chunk)
            E = np.exp(1j * (D[s:e] @ self.Theta))                                          # (c, dim)
            self.L_spec += C[s:e].T @ E                                                     # (3, dim)
        self._cal = None
        return self

    def add_softbox(self, position, target, width, height, intensity, color=(1.0, 1.0, 1.0), n=256, seed=0):
        """A rectangular area light: `n` stratified sample points on the rectangle, each seen from the scene centre
        (`target`) -- its direction as a light, its radiance intensity*color/n. Inverse-square is the caller's
        (holographic_preview's make_light convention: intensity is already at the light's distance)."""
        p = np.asarray(position, float); t = np.asarray(target, float)
        fwd = t - p; fwd /= (np.linalg.norm(fwd) + 1e-12)
        up = np.array([0.0, 1.0, 0.0]) if abs(fwd[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(fwd, up); u /= (np.linalg.norm(u) + 1e-12); v = np.cross(u, fwd)
        rng = np.random.default_rng(seed)
        g = int(np.ceil(np.sqrt(n)))
        a, b = np.meshgrid((np.arange(g) + rng.random(g)) / g - 0.5, (np.arange(g) + rng.random(g)) / g - 0.5)
        pts = p[None, :] + a.ravel()[:, None] * width * u[None, :] + b.ravel()[:, None] * height * v[None, :]
        dirs = pts - t[None, :]                                                             # from scene toward light
        col = np.asarray(color, float) * float(intensity) / len(pts)
        return self.deposit(dirs, np.repeat(col[None, :], len(pts), axis=0))

    def add_sky(self, fn, n=2048, seed=0):
        """A sky given as a callable D -> (n,3): sample it on a Fibonacci sphere and bundle. Any sky the path tracer
        took is a valid `fn` here."""
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
        D = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], axis=1)
        return self.deposit(D, np.asarray(fn(D), float) * (4 * np.pi / n))

    def radiance(self, dirs, chunk=8192):
        """Read radiance for many directions at once: (n,3). One matmul per chunk. Clipped at 0 (crosstalk lobes)."""
        D = np.atleast_2d(np.asarray(dirs, float)); D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        out = np.empty((len(D), 3))
        for s in range(0, len(D), chunk):
            e = min(len(D), s + chunk)
            E = np.exp(-1j * (D[s:e] @ self.Theta))                                         # (c, dim)
            out[s:e] = np.real(E @ self.L_spec.T)                                           # (c, 3)
        # the raw read is in FFT units; normalise by the kernel's own peak so a unit sample reads ~1 at its centre
        return np.clip(out / self._peak(), 0.0, None)

    def _peak(self):
        if self._cal is None:
            E = np.exp(1j * (np.array([[0.0, 1.0, 0.0]]) @ self.Theta))
            self._cal = float(np.real(np.exp(-1j * (np.array([[0.0, 1.0, 0.0]]) @ self.Theta)) @ E.T)[0, 0])
        return self._cal


class AnalyticEnv:
    """An environment given as a plain callable D -> (n,3) -- studio_sky(preset), a sky lambda, an HDRI sampler --
    read DIRECTLY, no FPE. Zero crosstalk by construction. WHY IT EXISTS: the FPE EnvField earns its place when you
    want superposition and algebra on the LIGHT (add a lamp by adding samples, move it by a bind). When the light is
    already a smooth analytic function, bundling it into a finite-dim field only adds a crosstalk texture that
    adjacent facets, exiting in slightly different directions, read as speckle. Measured on the 1-fold Mandelbox:
    the FPE env at dim 512 vs 4096 made no difference to that speckle (0.59 vs 0.62 roughness) -- because the env
    field was never the dominant source -- but a sharp analytic light is exactly right, and free."""
    def __init__(self, fn):
        self.fn = fn
    def radiance(self, dirs):
        D = np.atleast_2d(np.asarray(dirs, float)); D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        return np.clip(np.asarray(self.fn(D), float).reshape(len(D), 3), 0.0, None)


def _as_env(env):
    return env if hasattr(env, "radiance") else AnalyticEnv(env)


# ------------------------------------------------------------------------------------------ the bake
class GlassBake:
    """Everything about a frame that does not depend on the light: per-pixel refraction geometry for K wavelengths.
    Build once with bake_glass(); read with .relight(env, cmf)."""

    def __init__(self, width, height, lams):
        self.width, self.height = int(width), int(height)
        self.lams = np.asarray(lams, float)
        n = self.width * self.height; K = len(self.lams)
        self.glass = np.zeros(n, bool)                  # pixel hits the glass
        self.floor = np.zeros(n, bool)                  # pixel hits the floor
        self.exit_dirs = np.zeros((K, n, 3))            # transmitted exit direction per wavelength
        self.exit_pos = np.zeros((K, n, 3))             # where it leaves the body (so it can go on to hit the floor)
        self.entry_P = np.zeros((n, 3))                 # where the primary ray hit the glass (reflection origin)
        self.floor_y = None
        self.exit_ok = np.zeros((K, n), bool)           # False where the exit went TIR (first-order: treated as lost)
        self.refl_dir = np.zeros((n, 3))                # surface reflection direction
        self.R = np.zeros(n)                            # Fresnel reflectance at the entry (at the middle wavelength)
        self.path_len = np.zeros(n)                     # interior path length (for Beer-Lambert tint)
        self.floor_P = np.zeros((n, 3)); self.floor_N = np.zeros((n, 3))
        self.sky_dirs = np.zeros((n, 3))                # primary direction (for pixels that hit nothing)
        # OPAQUE body (default absent): a second SDF shaded Lambert like the floor -- the rock rind of a geode, the
        # matrix a crystal grows from. Primary hits, plus what exit and reflection rays see, are stored per ray.
        self.opaque = np.zeros(n, bool); self.opaque_P = np.zeros((n, 3)); self.opaque_N = np.zeros((n, 3))
        self.exit_opq = np.zeros((K, n), bool); self.exit_opq_t = np.full((K, n), np.inf)
        self.exit_opq_P = np.zeros((K, n, 3)); self.exit_opq_N = np.zeros((K, n, 3))
        self.refl_opq = np.zeros(n, bool); self.refl_opq_t = np.full(n, np.inf)
        self.refl_opq_P = np.zeros((n, 3)); self.refl_opq_N = np.zeros((n, 3))
        self.opaque_sdf = None
        # INTERIOR SEGMENTS for the middle wavelength: [(start (m,3), end (m,3), pixel idx (m,))] per bounce --
        # what a flaw volume (holographic_gemvolume) integrates in closed form. Light-independent, so it is bake data.
        self.segs = []

    def relight(self, env, cmf, floor_albedo=(0.30, 0.30, 0.30), glass_tint=(1.0, 1.0, 1.0), absorb=0.0,
                lights_for_shadow=None, sdf=None, sun_dir=None, shadow_transmittance=0.85, see_floor=True,
                opaque_albedo=(0.42, 0.40, 0.38), ambient_samples=48, background=None,
                flaw_volume=None, flaw_sigma=1.0, flaw_albedo=(1.0, 1.0, 1.0),
                absorb_volume=None, absorb_volume_sigma=(0.0, 0.0, 0.0), opaque_bounce=0.0,
                opaque_bounce_where=None, footprint_filter=False,
                sun_solid_angle=1.0):
        """The dot product. `env` is an EnvField or any callable D->rgb; `cmf(lams) -> (K,3)` the colour-matching
        weights. Returns (H,W,3) linear HDR.

        THREE PIECES OF PHYSICS that separate 'a glass-shaped thing' from glass, all added after Moose said the
        frames did not read as photoreal -- and all of them were the RELIGHT ignoring geometry it already had:
          * A ray that leaves the glass heading DOWN hits the FLOOR, and the floor is what you see through a glass
            object standing on one. The first relight read only the environment for exit rays, so every downward
            exit saw a dark sky: black facets. Now exit and reflection rays are intersected with the floor plane and
            take its shaded colour (`see_floor`). This is the single biggest realism fix.
          * A glass object does NOT cast a black shadow. Its shadow is the light it did not transmit -- (R + absorb),
            a few percent -- and the focused part the caustic pass adds back. `shadow_transmittance` is the fraction
            of sun that gets through an occluding glass body (default 0.85); the old behaviour (opaque) is 0.0.
          * The floor's ambient is the light field read along the normal; with a dim sky that is nearly black, so
            the sky/env you pass should be as bright as a studio wall actually is.
        No tracing here beyond the shadow rays toward `sun_dir` (one per floor point, including floor points seen
        THROUGH the glass)."""
        n = self.width * self.height; K = len(self.lams)
        env = _as_env(env)
        W = np.asarray(cmf(self.lams), float)                                               # (K, 3)
        W = W / (W.sum(axis=0, keepdims=True) + 1e-12)                                      # each channel sums to 1
        # ALBEDOS MAY BE FIELDS: floor_albedo / opaque_albedo accept a callable P (m,3) -> (m,3) as well as one rgb,
        # so a checkerboard floor or a noise-mottled rock is a texture lookup at the shaded point, nothing more.
        def _albedo(a):
            if callable(a):
                return lambda Pw: np.asarray(a(Pw), float).reshape(len(Pw), 3)
            v = np.asarray(a, float).reshape(1, 3)
            return lambda Pw: np.broadcast_to(v, (len(Pw), 3))
        floor_alb = _albedo(floor_albedo)
        S = None
        if sun_dir is not None:
            S = np.asarray(sun_dir, float); S = S / (np.linalg.norm(S) + 1e-12)
            sun_rad = env.radiance(S[None, :])

        def _alb_footprint(P, fp):
            """Floor albedo averaged over a disc of radius fp (m,) around P -- the ray's FOOTPRINT on the floor. A
            5-tap box (centre + 4 axis offsets); fp=0 is the plain lookup. This is the spectral ray differential:
            where adjacent wavelengths exit a dispersive body along different directions, the pixel's footprint on
            the floor is not a point but a smear of width (angular spread x distance), and reading a checker at a
            point there is aliasing -- the confetti in the diamond's TIR-heavy regions at 39 spectral samples."""
            if fp is None:
                return floor_alb(P)
            fp = np.asarray(fp, float).reshape(-1, 1)
            acc = floor_alb(P).copy()
            for ox, oz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                acc = acc + floor_alb(P + np.column_stack([ox * fp[:, 0], np.zeros(len(P)), oz * fp[:, 0]]))
            return acc / 5.0

        def shade_floor(P, fp=None):
            """Lambert floor at world points P (m,3): ambient from the field along +y, plus sun with a TRANSMISSIVE
            shadow ray (a glass occluder passes `shadow_transmittance` of the light, not zero).
            With an HdriEnv the floor uses the map's exact cosine-weighted irradiance instead, and the shadow ray
            (toward the map's dominant lobe) darkens only that lobe's share of it -- a soft window casts almost
            no shadow, a bare sun a hard one, and the map decides which."""
            N = np.tile([[0.0, 1.0, 0.0]], (len(P), 1))
            E = getattr(env, "floor_irradiance", None)
            if E is not None:
                Sd = np.asarray(getattr(env, "dominant_dir", (0.0, 1.0, 0.0)), float)
                share = float(getattr(env, "sun_share", 0.0))
                vis = np.ones((len(P), 1))
                if share > 0.0:
                    vis = _occluded(P, N, Sd)                 # glass: tinted transmittance over the chord; rock: blocks
                return _alb_footprint(P, fp) * (E[None, :] * (1.0 - share) + E[None, :] * share * vis) / np.pi   # Lambert
            shade = env.radiance(N)
            if S is not None:
                vis = _occluded(P, N, S)                      # glass: tinted transmittance over the chord; rock: blocks
                # IRRADIANCE from a light is its radiance x the SOLID ANGLE it subtends. `sun_rad` is the env read
                # along the key direction -- a RADIANCE. For a softbox of angular sigma ~0.16 rad that is ~0.16 sr,
                # a factor of six; treating radiance as irradiance blew the floor out white in the studio renders
                # while the glass (which reads radiance correctly) was fine. Default 1.0 keeps the old numbers.
                shade = shade + float(np.clip(N[0] @ S, 0.0, None)) * vis * sun_rad * float(sun_solid_angle)
            return _alb_footprint(P, fp) * shade

        def _occluded(P, N, Dl):
            """Visibility toward light direction Dl from points P (m,3): glass passes `shadow_transmittance` TINTED
            by Beer-Lambert over the chord it crosses (the material's per-RGB absorption x the marched interior
            length -- an amethyst's shadow is purple, a ruby's red; Moose: light through a coloured gem carries its
            colour), the opaque body blocks. Returns (m,3) in [0,1]; (m,1) when no absorption is set."""
            vis = np.ones((len(P), 1))
            Dl = np.broadcast_to(Dl, (len(P), 3)).copy()
            if sdf is not None:
                hit, th, Ph = sphere_trace(sdf, P + N * 2e-3, Dl, max_steps=64, max_dist=12.0)
                vis = np.where(hit[:, None], float(shadow_transmittance), 1.0)
                sig_s = np.asarray(absorb, float).reshape(-1)
                if hit.any() and sig_s.size and float(np.max(sig_s)) > 0.0:
                    sig_s = sig_s if sig_s.size == 3 else np.repeat(sig_s[:1], 3)
                    hh = np.flatnonzero(hit)
                    exitP = _march_through(sdf, Ph[hh] + Dl[hh] * 3e-3, Dl[hh], require_inside=True)
                    chord = np.linalg.norm(exitP - Ph[hh], axis=1)[:, None]
                    tintv = np.ones((len(P), 3)); tintv[hh] = np.exp(-sig_s[None, :] * chord)
                    vis = vis * tintv
            if self.opaque_sdf is not None:
                hit, _, _ = sphere_trace(self.opaque_sdf, P + N * 2e-3, Dl, max_steps=64, max_dist=12.0)
                vis = np.where(hit[:, None], 0.0, vis)
            return vis

        oalb = _albedo(opaque_albedo)

        bounce_E = {"v": None}

        def shade_opaque(P, N, _direct_only=False):
            """Lambert on an arbitrary normal. Ambient = the MAP read over a cosine-weighted hemisphere about N
            (`ambient_samples` deterministic directions per point; the map only -- HdriEnv.map_radiance -- so the
            analytic lobes are not double counted), plus each analytic lobe as L x solid angle x cos x visibility,
            plus the map's own dominant lobe scaled from the floor's measured share (its energy sits in a few pixels
            the hemisphere samples would miss). The same accounting shade_floor does, without the +y shortcut."""
            mP = len(P)
            k = int(ambient_samples)
            i = np.arange(k) + 0.5
            u1 = i / k; u2 = (i * 0.618033988749895) % 1.0                                # Fibonacci lattice, seeded by index
            r = np.sqrt(u1); ph = 2 * np.pi * u2                                             # cosine-weighted disc -> hemisphere
            loc = np.stack([r * np.cos(ph), r * np.sin(ph), np.sqrt(np.clip(1 - u1, 0, 1))], 1)   # (k,3), z = up
            up = np.where(np.abs(N[:, 1:2]) < 0.9, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]])
            T = np.cross(up, N); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12; B = np.cross(N, T)
            dirs = (loc[None, :, 0:1] * T[:, None, :] + loc[None, :, 1:2] * B[:, None, :] + loc[None, :, 2:3] * N[:, None, :])
            read = getattr(env, "map_radiance", env.radiance)
            amb = read(dirs.reshape(-1, 3)).reshape(mP, k, 3).mean(axis=1) * np.pi        # E = pi * mean(L) for cos-weighted
            E = amb
            for d, L, sg in getattr(env, "lobes", ()):
                cosv = np.clip(N @ d, 0.0, None)[:, None]
                E = E + L[None, :] * (2.0 * np.pi * sg * sg) * cosv * _occluded(P, N, d)
            Ef = getattr(env, "floor_irradiance", None)
            Sd = np.asarray(getattr(env, "map_dominant_dir", getattr(env, "dominant_dir", (0.0, 1.0, 0.0))), float)
            share_map = float(getattr(env, "map_sun_share", getattr(env, "sun_share", 0.0)))
            if Ef is not None and share_map > 0.0 and Sd[1] > 1e-3:
                Efloor_map = np.asarray(getattr(env, "map_floor_irradiance", Ef), float)
                cosv = np.clip(N @ Sd, 0.0, None)[:, None] / float(Sd[1])                   # E_floor / cos(elev) x N.S
                E = E + Efloor_map[None, :] * share_map * cosv * _occluded(P, N, Sd)
            if _direct_only:
                return E
            if opaque_bounce > 0.0:
                # ONE-NUMBER INDIRECT for a cavity (the radiosity closure): every point of a concave rock bowl sees the
                # other walls, so its irradiance is direct + rho_bar * <direct> / (1 - rho_bar). With no indirect the
                # geode's white cavity wall rendered black behind bright crystals (measured: wall E ~3 vs floor 14.6
                # under the key lobe). `opaque_bounce` = rho_bar/(1-rho_bar); 0 = the old behaviour. <direct> is the
                # mean direct irradiance over the bake's primary opaque points, computed once per relight.
                if bounce_E["v"] is None:
                    o = np.flatnonzero(self.opaque)
                    bounce_E["v"] = (shade_opaque(self.opaque_P[o], self.opaque_N[o], _direct_only=True).mean(axis=0)
                                     if o.size else np.zeros(3))
                # `opaque_bounce_where(P) -> (m,)` in [0,1] confines the bounce to the CONCAVE part (a cavity wall sees
                # other walls; the convex outside of the same rock sees only sky) -- without it the rind went grey.
                w = np.ones((len(P), 1)) if opaque_bounce_where is None else \
                    np.clip(np.asarray(opaque_bounce_where(P), float).reshape(-1, 1), 0.0, 1.0)
                E = E + float(opaque_bounce) * w * bounce_E["v"][None, :]
            return oalb(P) * E / np.pi

        def radiance_along(P, D, opq=None, spread=None):
            """What a ray from P in direction D sees: the opaque body or the floor -- whichever is nearer -- else
            the environment. `opq` = (hit, t, P, N) for these rays, from the bake."""
            out = env.radiance(D, blur=spread) if (spread is not None and "blur" in getattr(env.radiance, "__code__", env.radiance).co_varnames) \
                else env.radiance(D)
            tf = np.full(len(P), np.inf)
            if see_floor and self.floor_y is not None:
                dy = D[:, 1]
                t = (self.floor_y - P[:, 1]) / np.where(np.abs(dy) < 1e-9, -1e-9, dy)
                tf = np.where(t > 1e-6, t, np.inf)
            to = np.full(len(P), np.inf)
            if opq is not None and opq[0].any():
                to = np.where(opq[0], opq[1], np.inf)
            hitf = np.isfinite(tf) & (tf <= to)
            hito = np.isfinite(to) & (to < tf)
            if hitf.any():
                fp = None if spread is None else np.asarray(spread, float)[hitf] * tf[hitf]     # angle x distance
                out[hitf] = shade_floor(P[hitf] + D[hitf] * tf[hitf, None], fp)
            if hito.any():
                out[hito] = shade_opaque(opq[2][hito], opq[3][hito])
            return out

        img = np.zeros((n, 3))
        sky = ~(self.glass | self.floor | self.opaque)
        if sky.any():
            # `background`: a flat colour for the pixels that see NOTHING -- the map still lights, reflects and
            # refracts (every other read goes through env.radiance), it just is not the backdrop. Default: the map.
            img[sky] = env.radiance(self.sky_dirs[sky]) if background is None else np.asarray(background, float)[None, :]
        g = np.flatnonzero(self.glass)
        if g.size:
            trans = np.zeros((g.size, 3))
            # spectral footprint: half the angle between this pixel's neighbouring-wavelength exit directions
            # (0 where they agree -- a non-dispersive or single-transit ray -- so the plain lookup is unchanged)
            spread_g = None
            if footprint_filter and K > 1:
                ang = np.zeros(g.size)
                for k in range(K - 1):
                    both = self.exit_ok[k, g] & self.exit_ok[k + 1, g]
                    c = np.clip(np.sum(self.exit_dirs[k, g] * self.exit_dirs[k + 1, g], axis=1), -1.0, 1.0)
                    ang = np.maximum(ang, np.where(both, np.arccos(c), 0.0))
                spread_g = 0.5 * ang
            for k in range(K):
                ok = self.exit_ok[k, g]
                Lk = np.zeros((g.size, 3))
                if ok.any():
                    gi = g[ok]
                    opq = (self.exit_opq[k, gi], self.exit_opq_t[k, gi], self.exit_opq_P[k, gi], self.exit_opq_N[k, gi]) \
                        if self.opaque_sdf is not None else None
                    Lk[ok] = radiance_along(self.exit_pos[k, gi], self.exit_dirs[k, gi], opq,
                                            None if spread_g is None else spread_g[ok])
                trans += Lk * W[k][None, :]
            sig = np.asarray(absorb, float).reshape(-1)
            sig = sig if sig.size == 3 else np.repeat(sig[:1], 3)          # scalar or per-RGB Beer-Lambert sigma
            tint = np.exp(-sig[None, :] * self.path_len[g][:, None])       # ruby: green/blue die with depth, red survives
            if absorb_volume is not None and self.segs:
                # COLOUR ZONING: a second density -- where the chromophore is (amethyst's iron at the tips, a
                # citrine core) -- integrated along the same segments and applied as per-RGB Beer-Lambert.
                # An amethyst's purple lives in its last few millimetres of growth; a uniform absorb cannot say so.
                tau_c = absorb_volume.segments_optical_depth(self.segs, n)[g][:, None]
                tint = tint * np.exp(-np.asarray(absorb_volume_sigma, float).reshape(1, 3) * tau_c)
            if flaw_volume is not None and self.segs:
                # IMPERFECTIONS: the flaw density's integral along each pixel's interior path, in closed form
                # (holographic_gemvolume): extinction + single-scatter glow lit by the ambient the gem sits in.
                from holographic.rendering.holographic_gemvolume import flaw_shading
                tau_all = flaw_volume.segments_optical_depth(self.segs, n)
                E_amb = getattr(env, "floor_irradiance", None)
                if E_amb is None:
                    E_amb = np.pi * env.radiance(np.array([[0.0, 1.0, 0.0]]))[0]
                E_amb = np.asarray(E_amb, float) * np.ones(3)
                # DIRECTIONAL in-scatter: the haze is lit by the scene's lights, not by a constant. The dominant lobe's
                # share of E is applied per pixel with a shadow ray from the entry point (glass passes
                # shadow_transmittance, rock blocks) times a wrap-around cosine on the entry normal -- so a frosted body
                # has a lit side and a dark side. Measured need: with a constant E_amb the milky quartz read as flat
                # plastic fog (Moose: "looks terrible"); the ambient (1 - share) part stays isotropic.
                share = float(getattr(env, "sun_share", 0.0))
                if share > 0.0 and S is not None:
                    Ng = sdf_normal(sdf, self.entry_P[g]) if sdf is not None else -self.sky_dirs[g]
                    wrap = np.clip(0.5 + 0.5 * (Ng @ S), 0.0, 1.0)[:, None]                 # wrap lighting for a volume
                    vis = _occluded(self.entry_P[g], Ng, S)
                    E_px = E_amb[None, :] * (1.0 - share) + E_amb[None, :] * share * 2.0 * wrap * vis
                else:
                    E_px = np.broadcast_to(E_amb[None, :], (g.size, 3))
                T = np.exp(-float(flaw_sigma) * tau_all[g])[:, None]
                trans = trans * T + np.asarray(flaw_albedo, float).reshape(1, 3) * E_px / np.pi * (1.0 - T)
            opq = (self.refl_opq[g], self.refl_opq_t[g], self.refl_opq_P[g], self.refl_opq_N[g]) \
                if self.opaque_sdf is not None else None
            refl = radiance_along(self.entry_P[g], self.refl_dir[g], opq)
            Rg = self.R[g][:, None]
            img[g] = (1.0 - Rg) * trans * tint + Rg * refl
        f = np.flatnonzero(self.floor)
        if f.size:
            img[f] = shade_floor(self.floor_P[f])
        o = np.flatnonzero(self.opaque)
        if o.size:
            img[o] = shade_opaque(self.opaque_P[o], self.opaque_N[o])
        return img.reshape(self.height, self.width, 3)


def bake_glass(sdf, camera, width, height, ior_fn, lams, floor_y=None, max_steps=96, max_dist=20.0, max_internal=4,
               samples=1, seed=0, opaque=None, lam_jitter=False):
    """ANTI-ALIASED bake: `samples` jittered sub-pixel ray sets, exactly the offsets path_trace's antialias=True uses
    (Sampling.low_discrepancy), each baked independently and AVERAGED at relight (relight is linear, so the mean of
    the relit sub-bakes is the box-filtered pixel). samples=1 is the historical single-ray bake. WHY: a one-ray-per-
    pixel bake of a faceted body is a stair-stepped silhouette and pixel-speckled fringes -- every facet edge lands
    on a pixel boundary somewhere -- and no post-process undoes that. The bake is cheap (1-2s), so 4-8 sub-samples
    cost seconds; the relight cost scales with `samples`. Returns a GlassBake, or a MultiBake (same .relight)."""
    if int(samples) > 1:
        from holographic.sampling_and_signal.holographic_samplinghome import Sampling
        offs = Sampling.low_discrepancy(int(samples), d=2, seed=seed) - 0.5
        lams = np.asarray(lams, float); n_s = int(samples)
        subs = []
        for i, o in enumerate(offs):
            # SPECTRAL STRATIFICATION (`lam_jitter`): each sub-bake shifts the wavelength grid by i/samples of one
            # spacing, so the union of sub-bakes samples the spectrum samples-times more finely at no extra cost per
            # sub-bake. WHY: K hero wavelengths through a strongly dispersive body (diamond) land on DIFFERENT checker
            # squares, and the per-pixel sum of 11 discrete colours read as confetti -- aliasing in lambda, the same
            # thing an MC tracer avoids by drawing lambda per sample. Default off: byte-identical to before.
            lk = lams if not lam_jitter or len(lams) < 2 else \
                np.clip(lams + (i / n_s) * (lams[1] - lams[0]), lams[0], lams[-1] + (lams[1] - lams[0]) * (n_s - 1) / n_s)
            subs.append(_bake_glass_one(sdf, camera, width, height, ior_fn, lk, floor_y, max_steps, max_dist, max_internal,
                                        jitter=(float(o[0]), float(o[1])), opaque=opaque))
        return MultiBake(subs)
    return _bake_glass_one(sdf, camera, width, height, ior_fn, lams, floor_y, max_steps, max_dist, max_internal,
                           opaque=opaque)


class MultiBake:
    """Several jittered sub-pixel GlassBakes; relight each and average. The linearity of the read is what makes the
    average exact: mean over sub-bakes of <L, encode(D)> is <L, mean encode(D)> -- the pixel's box filter."""
    def __init__(self, subs):
        self.subs = list(subs); self.width, self.height = subs[0].width, subs[0].height; self.lams = subs[0].lams
        self.glass = np.any([b.glass for b in subs], axis=0); self.floor = np.any([b.floor for b in subs], axis=0)
        self.opaque = np.any([b.opaque for b in subs], axis=0)
        # loss statistic across all sub-samples: masks differ per jitter, so gather per sub-bake, then pool
        self.exit_ok = np.concatenate([b.exit_ok[:, b.glass] for b in subs], axis=1)
        self.glass_pooled = np.ones(self.exit_ok.shape[1], bool)
    def relight(self, env, cmf, **kw):
        return np.mean([b.relight(env, cmf, **kw) for b in self.subs], axis=0)


def _bake_glass_one(sdf, camera, width, height, ior_fn, lams, floor_y, max_steps, max_dist, max_internal, jitter=None,
                    opaque=None):
    """Trace the refraction geometry ONCE for K wavelengths. `ior_fn(lam) -> n` (e.g. from holographic_dispersion's
    cauchy_n); `lams` the K wavelengths. Everything returned is light-independent. `floor_y` adds an infinite
    floor plane (pixels that miss the glass but hit it are Lambert-shaded at relight). `opaque` (default None) is a
    second SDF -- rock, matrix, a geode's rind -- that is Lambert-shaded like the floor: it takes the pixel where it is
    nearer than the glass, and exit and reflection rays that strike it are recorded (position, normal, distance) so
    the relight can shade what is seen THROUGH the crystals. One extra sphere trace per ray set; nothing per light."""
    lams = np.asarray(lams, float); K = len(lams)
    eye, dirs = camera.ray_dirs(width, height, jitter=jitter) if jitter is not None else camera.ray_dirs(width, height)
    D = np.asarray(dirs, float).reshape(-1, 3); n = len(D)
    O = np.broadcast_to(np.asarray(eye, float), (n, 3)).copy()
    bake = GlassBake(width, height, lams)
    bake.sky_dirs[:] = D
    hit, t, P = sphere_trace(sdf, O, D, max_steps=max_steps, max_dist=max_dist)
    # floor: analytic plane intersection, kept only where it is nearer than the glass hit (or the glass missed)
    if floor_y is not None:
        tf = (float(floor_y) - O[:, 1]) / np.where(np.abs(D[:, 1]) < 1e-9, -1e-9, D[:, 1])
        fl = (tf > 0) & (~hit | (tf < t))
        bake.floor[fl] = True
        bake.floor_P[fl] = O[fl] + D[fl] * tf[fl, None]
        bake.floor_N[fl] = np.array([0.0, 1.0, 0.0])
        hit = hit & ~fl
    if opaque is not None:
        bake.opaque_sdf = opaque
        ho, to, Po = sphere_trace(opaque, O, D, max_steps=max_steps, max_dist=max_dist)
        tglass = np.where(hit, t, np.inf)
        tfloor = np.full(n, np.inf)
        if floor_y is not None:
            tfloor = np.where(bake.floor, tf, np.inf)
        op = ho & (to < tglass) & (to < tfloor)                                   # the opaque body is what the pixel sees
        bake.opaque[op] = True; bake.opaque_P[op] = Po[op]; bake.opaque_N[op] = sdf_normal(opaque, Po[op])
        hit = hit & ~op; bake.floor[op] = False
    bake.glass[hit] = True
    g = np.flatnonzero(hit)
    if g.size:
        Ph, Dh = P[g], D[g]
        Nh = sdf_normal(sdf, Ph)
        cosi = np.abs(np.sum(Nh * (-Dh), axis=1))
        n_mid = float(ior_fn(float(lams[K // 2])))
        bake.R[g] = fresnel_dielectric(cosi, n_mid)
        bake.refl_dir[g] = Dh - 2.0 * np.sum(Dh * Nh, axis=1)[:, None] * Nh
        bake.entry_P[g] = Ph
        bake.floor_y = None if floor_y is None else float(floor_y)
        for k in range(K):
            nk = float(ior_fn(float(lams[k])))
            r_in = refract_dir(Dh, Nh, nk)
            # INTERNAL BOUNCES, DETERMINISTICALLY. The first draft stopped at one transit and marked a TIR exit as
            # lost; on the 1-fold Mandelbox that dropped 65.5% of glass-wavelength pairs -- a cube-ish body sends
            # most rays that enter one face into total internal reflection at the next. The path tracer followed
            # those bounces with a coin; here there is no coin to flip: a TIR ray simply reflects and marches
            # to the next face, up to `max_internal` times. Still one pass, still no variance. Rays that are
            # still trapped after the cap are the honest remaining loss, reported by the selftest.
            cur_P = Ph + r_in * 3e-3; cur_D = r_in
            out_dir = np.zeros_like(Dh); out_pos = np.zeros_like(Dh); done = np.zeros(g.size, bool); plen = np.zeros(g.size)
            for _bounce in range(max_internal + 1):
                live = ~done
                if not live.any():
                    break
                exitP = _march_through(sdf, cur_P[live], cur_D[live], require_inside=True)
                if k == K // 2:
                    bake.segs.append((cur_P[live].copy(), exitP.copy(), g[live]))
                Nx = sdf_normal(sdf, exitP)
                r_out = refract_dir(cur_D[live], Nx, nk)
                leaving = np.sum(r_out * Nx, axis=1) > 0.0                  # transmitted, or TIR (reflected back in)
                plen[live] += np.linalg.norm(exitP - cur_P[live], axis=1)
                idx = np.flatnonzero(live)
                out_dir[idx[leaving]] = r_out[leaving]
                out_pos[idx[leaving]] = exitP[leaving]
                done[idx[leaving]] = True
                cur_P[idx[~leaving]] = exitP[~leaving] + r_out[~leaving] * 3e-3   # reflect inside, keep marching
                cur_D[idx[~leaving]] = r_out[~leaving]
            bake.exit_dirs[k, g] = out_dir
            bake.exit_pos[k, g] = out_pos
            bake.exit_ok[k, g] = done
            if k == K // 2:
                bake.path_len[g] = plen
            if opaque is not None and done.any():
                gd = g[done]
                ho, to, Po = sphere_trace(opaque, out_pos[done] + out_dir[done] * 2e-3, out_dir[done],
                                          max_steps=max_steps, max_dist=max_dist)
                bake.exit_opq[k, gd] = ho; bake.exit_opq_t[k, gd] = np.where(ho, to, np.inf)
                bake.exit_opq_P[k, gd] = Po
                if ho.any():
                    bake.exit_opq_N[k, gd[ho]] = sdf_normal(opaque, Po[ho])
        if opaque is not None:
            ho, to, Po = sphere_trace(opaque, Ph + bake.refl_dir[g] * 2e-3, bake.refl_dir[g],
                                      max_steps=max_steps, max_dist=max_dist)
            bake.refl_opq[g] = ho; bake.refl_opq_t[g] = np.where(ho, to, np.inf); bake.refl_opq_P[g] = Po
            if ho.any():
                bake.refl_opq_N[g[ho]] = sdf_normal(opaque, Po[ho])
    return bake


def _selftest():
    """Pin: (1) the bake is light-independent and the relight is the only thing that changes with the light;
    (2) relighting is fast relative to the bake -- it is a read, not a trace; (3) dispersion is in the bake: the
    exit directions differ across wavelengths where the glass is hit, and do not where it is not; (4) the env
    field reads a deposited direction back as its own colour and an unlit direction as ~0."""
    import time
    from holographic.mesh_and_geometry.holographic_sdf import sphere
    from holographic.rendering.holographic_render import Camera
    from holographic.rendering.holographic_dispersion import cauchy_n

    # (4) the light field
    env = EnvField()
    env.deposit([[0.0, 1.0, 0.0]], [[1.0, 0.2, 0.2]])
    up = env.radiance([[0.0, 1.0, 0.0]])[0]; down = env.radiance([[0.0, -1.0, 0.0]])[0]
    assert up[0] >= 4 * up[1] and up[0] > 0.5, "a red sample straight up must read back red and bright: %r" % up
    assert down.max() < 0.15 * up[0], "the opposite direction must read ~dark (crosstalk floor only): %r" % down

    cam = Camera(eye=(0.0, 1.0, 4.0), target=(0.0, 0.0, 0.0), fov_deg=35.0, aspect=1.0)
    lams = np.linspace(430.0, 670.0, 5)
    ior = lambda lam: float(cauchy_n(lam, 1.55, 25.68))
    t0 = time.time()
    bake = bake_glass(sphere(0.8), cam, 48, 48, ior, lams, floor_y=-0.9)
    t_bake = time.time() - t0
    assert bake.glass.sum() > 200 and bake.floor.sum() > 200, "the fixture must show glass AND floor"

    # (3) dispersion lives in the bake
    g = np.flatnonzero(bake.glass)
    spread = np.linalg.norm(bake.exit_dirs[0, g] - bake.exit_dirs[-1, g], axis=1)
    assert np.median(spread[bake.exit_ok[0, g] & bake.exit_ok[-1, g]]) > 1e-3, \
        "blue and red exit directions must differ through glass: median %.2e" % np.median(spread)
    assert np.all(bake.exit_dirs[:, ~bake.glass] == 0.0), "no exit direction where there is no glass"

    # (1)+(2) relight: two lights, two different frames, from ONE bake, each far cheaper than the bake
    env_a = EnvField().add_softbox((3, 4, 2), (0, 0, 0), 2.0, 2.0, 40.0, color=(1.0, 0.9, 0.8))
    env_b = EnvField().add_softbox((-3, 4, 2), (0, 0, 0), 2.0, 2.0, 40.0, color=(0.7, 0.8, 1.0))
    cmf = lambda l: np.stack([np.exp(-0.5 * ((l - c) / 45.0) ** 2) for c in (610.0, 545.0, 465.0)], axis=1)
    t0 = time.time(); fa = bake.relight(env_a, cmf); t_relight = time.time() - t0
    fb = bake.relight(env_b, cmf)
    assert fa.shape == (48, 48, 3) and np.isfinite(fa).all()
    assert not np.allclose(fa, fb), "a different light must give a different frame from the same bake"
    assert fa[..., 0].mean() > fa[..., 2].mean() and fb[..., 2].mean() > fb[..., 0].mean(), \
        "a warm light gives a warm frame, a cool light a cool one -- the light is what changed"
    # HONEST about the relight cost: it is O(pixels x K x dim) direction encodes, not free. On an analytic sphere
    # the bake is trivially cheap (0.02s at 48^2), so "relight < bake" is the wrong claim here -- the claim that
    # matters is that bake AND relight are both far below any Monte-Carlo estimate of the same frame (minutes),
    # and that relight needs no tracing at all. Pinned as an absolute bound on this fixture.
    assert t_relight < 2.0, "relight must stay a cheap read on a 48^2 frame: %.3fs" % t_relight
    tir = 1.0 - bake.exit_ok[:, g].mean()
    print("OK: glass bake -- %d glass px, %d floor px; bake %.2fs, relight %.3fs (%.0fx); blue-red exit spread "
          "median %.3e; first-order TIR loss %.1f%% of glass-wavelength pairs; env field reads back its samples"
          % (bake.glass.sum(), bake.floor.sum(), t_bake, t_relight, t_bake / max(t_relight, 1e-9),
             np.median(spread), 100 * tir))


if __name__ == "__main__":
    _selftest()


# ------------------------------------------------------------------------------------------ real light: an HDRI
class HdriEnv:
    """A measured environment map as the light. Wraps an equirectangular (H,W,3) linear radiance image (load_hdr /
    load_exr) behind the same `.radiance(dirs)` contract the relight reads, via the engine's own sky_dome sampler.

    What an HDRI changes about the FLOOR, and why this class also carries the floor's irradiance: with a hand-placed
    key light the floor was 'ambient + one sun with a shadow ray'. Under a real map the correct diffuse answer is
    the cosine-weighted integral of the whole hemisphere,  E = integral L(w) cos(theta) dw,  and the floor is a plane,
    so that integral is ONE rgb per map -- computed once here by Fibonacci sampling (`floor_irradiance`). The shadow
    then applies only to the DOMINANT lobe's share of E (`sun_share`, the fraction of irradiance arriving within
    `sun_cone` of the brightest direction), because a soft studio window or a cloudy sky casts almost no shadow
    while a bare sun casts a hard one -- the map itself says which. `dominant_dir` is that brightest direction,
    found as the irradiance-weighted mean of the top 0.5% of samples, and is what the caustic pass should aim
    along. All of it is a READ of the map; nothing is hand-set."""

    def __init__(self, env_img, exposure=1.0, n_samples=8192, sun_cone=0.20, seed=0):
        from holographic.rendering.holographic_raymarch import sky_dome
        self.img = np.asarray(env_img, np.float32) * float(exposure)
        self._sky = sky_dome
        # Integrate over the MAP'S OWN PIXELS with sky_dome's exact convention (u = atan2(x,-z)/2pi + 0.5,
        # v = 0.5 - asin(y)/pi), not a Fibonacci sample of it: a sun through a window is a handful of pixels
        # carrying most of the energy -- measured, the lounge map peaks at 41,984 and 8,192 direction samples
        # gave its lobe a 2% share because they missed it. Each pixel is a solid angle (2pi/W)(pi/H) cos(lat).
        H, W = self.img.shape[:2]
        step = max(1, int(round(np.sqrt(H * W / 2.0e6))))                           # cap at ~2M pixels
        sub = np.asarray(self.img[::step, ::step], float); h, w = sub.shape[:2]
        v = (np.arange(h) + 0.5) / h; u = (np.arange(w) + 0.5) / w
        lat = (0.5 - v) * np.pi                                                     # +pi/2 at row 0 (= +y)
        lon = (u - 0.5) * 2 * np.pi
        y = np.sin(lat)[:, None] * np.ones((1, w))
        x = (np.cos(lat)[:, None] * np.sin(lon)[None, :])
        z = -(np.cos(lat)[:, None] * np.cos(lon)[None, :])
        dOm = (2 * np.pi / w) * (np.pi / h) * np.cos(lat)[:, None] * np.ones((1, w))
        up = y > 0
        Ecos = sub * (y * dOm * up)[..., None]                                      # L cos(theta) dOmega, hemisphere
        self.floor_irradiance = Ecos.sum(axis=(0, 1))
        # the dominant lobe: brightest 0.1% of pixels by ENERGY (L x dOmega), their irradiance-weighted direction
        energy = sub.mean(axis=-1) * dOm * up
        # the SINGLE brightest pixel seeds the lobe, then the energy-weighted mean within `sun_cone` of it refines
        # it. A percentile-mean over all bright pixels lands BETWEEN two windows, in the dark (measured: it pointed
        # at radiance 1 in a map whose sun reads 41,984).
        k = np.unravel_index(np.argmax(energy), energy.shape)
        seed = np.array([x[k], y[k], z[k]])
        near = (x * seed[0] + y * seed[1] + z * seed[2] > np.cos(sun_cone)) & up
        d = (np.stack([x[near], y[near], z[near]], axis=1) * energy[near][:, None]).sum(axis=0)
        self.dominant_dir = d / (np.linalg.norm(d) + 1e-12)
        cosang = x * self.dominant_dir[0] + y * self.dominant_dir[1] + z * self.dominant_dir[2]
        cone = (cosang > np.cos(sun_cone)) & up
        E_sun = (Ecos * cone[..., None]).sum(axis=(0, 1))
        self.sun_share = float(np.clip(E_sun.mean() / max(self.floor_irradiance.mean(), 1e-12), 0.0, 1.0))

        self.lobes = []                                                             # (dir, rgb radiance, sigma)
        self.map_floor_irradiance = self.floor_irradiance.copy()                    # the MAP's own, before any lobe
        self.map_sun_share = self.sun_share
        self.map_dominant_dir = self.dominant_dir.copy()

    def map_radiance(self, dirs):
        """The photograph alone -- no analytic lobes. shade_opaque samples this for ambient and adds lobes analytically."""
        D = np.atleast_2d(np.asarray(dirs, float)); D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        return np.clip(np.asarray(self._sky(D, env=self.img), float).reshape(len(D), 3), 0.0, None)

    def add_light(self, direction, radiance=None, sigma=0.06, irradiance=None):
        """Layer an ANALYTIC Gaussian lobe on top of the measured map: an extra key / rim light for showing off
        dispersion and caustics without repainting the HDRI. `radiance` is the lobe's peak radiance (rgb or scalar),
        `sigma` its angular width in radians. Returns self for chaining.

        WHY IT UPDATES THE INTEGRALS TOO. `floor_irradiance`, `dominant_dir` and `sun_share` are what the relight
        and caustic passes read, so a lobe that only changed `radiance()` would light the gem and leave the floor and
        the caustic aimed at the map's own window -- the extra light would cast no caustic at all. A narrow Gaussian
        of peak L and width sigma carries solid angle ~ 2 pi sigma^2, so it adds L 2 pi sigma^2 cos(theta) to the
        floor irradiance and, when it is the strongest lobe, becomes the dominant direction (its irradiance share
        replaces the map's `sun_share`). Kept simple on purpose: one lobe wins the caustic; the map still lights
        everything else.

        Sizing without blowing the scene out: pass `irradiance` instead of `radiance` to size the lobe by how much
        it adds to the FLOOR (same units as `floor_irradiance`, e.g. `irradiance=env.floor_irradiance.mean()` makes
        a key as strong as the whole map). Measured on the lounge map: peak 3000 at sigma 0.05 took the lobe to a
        0.96 share -- the HDRI had stopped being the base light. Its own sun is only ~0.67 of irradiance."""
        d = np.asarray(direction, float); d = d / (np.linalg.norm(d) + 1e-12)
        omega = 2.0 * np.pi * float(sigma) ** 2                                     # solid angle of the lobe
        if radiance is None:
            if irradiance is None:
                raise ValueError("add_light: give radiance (peak) or irradiance (floor contribution)")
            radiance = np.asarray(irradiance, float) / (omega * max(float(d[1]), 1e-3))
        L = np.asarray(radiance, float) * np.ones(3)
        E_lobe = L * omega * max(float(d[1]), 0.0)                                  # its cos-weighted floor irradiance
        E_dom_old = self.floor_irradiance.mean() * self.sun_share                   # the current dominant lobe's share
        self.floor_irradiance = self.floor_irradiance + E_lobe
        self.lobes.append((d, L, float(sigma)))
        if E_lobe.mean() > E_dom_old:                                               # the new lobe is now the shadow-caster
            self.dominant_dir = d
            self.sun_share = float(np.clip(E_lobe.mean() / max(self.floor_irradiance.mean(), 1e-12), 0.0, 1.0))
        else:
            self.sun_share = float(np.clip(E_dom_old / max(self.floor_irradiance.mean(), 1e-12), 0.0, 1.0))
        return self

    def radiance(self, dirs, blur=None):
        """Radiance along `dirs`; `blur` (n,) is an angular footprint in radians: each analytic lobe is read with
        sigma_eff = sqrt(sigma^2 + blur^2) at conserved energy (L sigma^2 = L_eff sigma_eff^2) -- a ray whose
        wavelengths fan out over `blur` sees the lobe averaged over that fan, not a point sample of its peak. WHY: a
        single hero wavelength that happened to exit a diamond straight at the key read the full peak and painted a
        saturated speck; the fan-averaged read is what thousands of MC wavelengths converge to. The map itself is
        read unblurred (its own texels are the finer scale)."""
        D = np.atleast_2d(np.asarray(dirs, float)); D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        out = np.clip(np.asarray(self._sky(D, env=self.img), float).reshape(len(D), 3), 0.0, None)
        b2 = None if blur is None else np.asarray(blur, float).reshape(-1) ** 2
        for d, L, sg in getattr(self, "lobes", ()):                                 # analytic lobes ride on the map
            ang2 = 2.0 * np.clip(1.0 - D @ d, 0.0, None)                            # |D-d|^2 = 2(1-cos) ~ angle^2
            s2 = sg * sg if b2 is None else sg * sg + b2
            out = out + (L[None, :] * (sg * sg / s2)[:, None] if b2 is not None else L[None, :]) * \
                np.exp(-ang2 / (2.0 * s2))[:, None]
        return out
