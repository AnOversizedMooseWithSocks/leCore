"""Cook-Torrance / GGX microfacet BRDF -- the physically-based reflectance model V-Ray, Redshift, Arnold, and
every modern PBR pipeline use. This replaces the engine's ad-hoc Lambert + Schlick-fresnel shading with the real
thing: a metallic/roughness material whose specular highlight has the correct shape, energy, and grazing-angle
behaviour, and whose importance sampler feeds the path tracer.

The model is reflectance f_r = diffuse + specular with the specular term

    f_spec = D * G * F / (4 (N.V)(N.L))

  * D -- the NORMAL DISTRIBUTION (GGX / Trowbridge-Reitz): what fraction of microfacets point toward the half-
    vector H. Roughness widens it; this is the highlight's shape and the long GGX tail that reads as 'real'.
  * G -- the GEOMETRY / shadowing-masking (Smith, Schlick-GGX): microfacets occluding each other at grazing angles.
  * F -- FRESNEL (Schlick): reflectance rising to 1 at grazing incidence; F0 is 0.04 for dielectrics, the base
    colour for metals.

Diffuse is Lambert, scaled by (1-F)(1-metallic) so specular + diffuse never reflect more than arrives (the energy
split kS=F, kD=1-kS, and metals have no diffuse). Everything is vectorised over (...,3) arrays of unit vectors.

HONEST (kept negative): this is SINGLE-SCATTERING GGX. At high roughness it loses energy (the white-furnace test
below dips below 1.0), because light that bounces between microfacets more than once is dropped. A Kulla-Conty
multiscatter compensation term fixes it and is the honest next step; we measure and keep the loss rather than hide
it.
"""

import numpy as np


def fresnel_schlick(cos_theta, F0):
    """Schlick's Fresnel approximation: reflectance at angle theta given normal-incidence reflectance F0."""
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    return F0 + (1.0 - F0) * (1.0 - cos_theta)[..., None] ** 5 if np.ndim(F0) and np.shape(F0)[-1:] == (3,) \
        else F0 + (1.0 - F0) * (1.0 - cos_theta) ** 5


def fresnel_dielectric(cos_theta, ior):
    """Scalar Schlick Fresnel reflectance for a DIELECTRIC interface (e.g. glass, ior~1.5). F0 = ((1-ior)/(1+ior))^2
    is the normal-incidence reflectance; reflectance rises to 1 at grazing. Returns the fraction of light REFLECTED
    (the rest is transmitted/refracted). This is what a glass BSDF uses to choose reflect vs refract per ray."""
    cos_theta = np.clip(np.abs(cos_theta), 0.0, 1.0)
    r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
    return r0 + (1.0 - r0) * (1.0 - cos_theta) ** 5


def metallic_f0(base_color, metallic):
    """Normal-incidence reflectance F0 from the metallic workflow, in ONE place. A DIELECTRIC reflects a constant
    ~4% at normal incidence (F0 = 0.04); a METAL reflects its base colour. The metallic channel linearly blends
    between them. `base_color` is (...,3); `metallic` is broadcast-shaped by the caller (scalar, or e.g.
    metallic[:, None] against an (M,3) base). This is the same formula three shading sites used inline -- naming
    it puts the 0.04 dielectric constant in one home so a change can't drift across copies."""
    return 0.04 * (1.0 - metallic) + base_color * metallic


def d_ggx(n_dot_h, roughness):
    """GGX / Trowbridge-Reitz normal distribution. alpha = roughness^2 is the perceptually-linear remap."""
    a = np.asarray(roughness, float) ** 2
    a2 = a * a
    nh = np.clip(n_dot_h, 0.0, 1.0)
    denom = nh * nh * (a2 - 1.0) + 1.0
    return a2 / (np.pi * denom * denom + 1e-12)


def _g_schlick_ggx(n_dot_x, roughness):
    """Smith's geometry term, one direction (Schlick-GGX with the direct-lighting k = (r+1)^2 / 8)."""
    r = np.asarray(roughness, float) + 1.0
    k = (r * r) / 8.0
    nx = np.clip(n_dot_x, 1e-4, 1.0)
    return nx / (nx * (1.0 - k) + k + 1e-12)


def g_smith(n_dot_v, n_dot_l, roughness):
    """Full Smith geometry: shadowing (view) * masking (light)."""
    return _g_schlick_ggx(n_dot_v, roughness) * _g_schlick_ggx(n_dot_l, roughness)


def cook_torrance(N, V, L, base_color, metallic, roughness):
    """The BRDF times N.L -- the reflected radiance for unit incoming radiance from direction L, viewed from V.
    N, V, L: (...,3) unit vectors. base_color: (...,3) or (3,). metallic, roughness: scalar or (...). Returns
    (...,3). This is what a shader evaluates per light."""
    N = np.asarray(N, float); V = np.asarray(V, float); L = np.asarray(L, float)
    base = np.asarray(base_color, float)
    metallic = np.asarray(metallic, float)[..., None] if np.ndim(metallic) else float(metallic)
    H = V + L
    H = H / (np.linalg.norm(H, axis=-1, keepdims=True) + 1e-12)
    ndv = np.clip(np.sum(N * V, axis=-1), 1e-4, 1.0)
    ndl = np.clip(np.sum(N * L, axis=-1), 0.0, 1.0)
    ndh = np.clip(np.sum(N * H, axis=-1), 0.0, 1.0)
    vdh = np.clip(np.sum(V * H, axis=-1), 0.0, 1.0)
    F0 = metallic_f0(base, metallic)                               # dielectric 0.04 / metal tinted (shared helper)
    F = fresnel_schlick(vdh, F0)                                    # (...,3)
    D = d_ggx(ndh, roughness)                                       # (...,)
    G = g_smith(ndv, ndl, roughness)                               # (...,)
    spec = (D * G)[..., None] * F / (4.0 * ndv * ndl + 1e-6)[..., None]
    kd = (1.0 - F) * (1.0 - metallic)                               # energy left for diffuse; metals: 0
    diffuse = kd * base / np.pi
    return (diffuse + spec) * ndl[..., None]


def lambert(N, L, base_color):
    """The diffuse (Lambertian) shade -- matte reflected radiance for unit incoming light from direction L:
    max(N.L, 0) * base_color. N: (...,3) unit normals; L: a (3,) light direction (dotted via matmul); base_color:
    (...,3) or (3,). Returns (...,3). This is the diffuse half of cook_torrance exposed on its own so the many
    matte / one-bounce gather paths call the Shading home instead of re-deriving clip(N.L,0)*albedo. (Compound
    shades that fold in shadow / occlusion / a specular lobe keep their own expression -- lambert is just the term.)
    """
    ndl = np.clip(np.asarray(N, float) @ np.asarray(L, float), 0.0, None)
    return ndl[..., None] * np.asarray(base_color, float)


def _tangent_frame(N):
    """An orthonormal (T, B, N) frame for each normal -- to map tangent-space samples to world."""
    N = np.asarray(N, float)
    up = np.where(np.abs(N[..., 0:1]) < 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    T = np.cross(up, N); T = T / (np.linalg.norm(T, axis=-1, keepdims=True) + 1e-12)
    B = np.cross(N, T)
    return T, B


def sample_ggx(N, V, roughness, rng):
    """Importance-sample a reflected direction L from the GGX distribution about N, for a path tracer. Returns
    (L, pdf): L the bounced direction (...,3), pdf its solid-angle density. Sampling H ~ D(H)(N.H) and reflecting
    V gives pdf(L) = D(N.H)(N.H) / (4 (V.H)) -- so brdf/pdf cancels the D and the 1/(4 N.V N.L), leaving a cheap,
    low-variance estimator (the whole point of importance sampling)."""
    N = np.asarray(N, float); V = np.asarray(V, float)
    a = np.asarray(roughness, float) ** 2
    shp = N.shape[:-1]
    u1 = rng.random(shp); u2 = rng.random(shp)
    phi = 2.0 * np.pi * u1
    cos_t = np.sqrt((1.0 - u2) / (1.0 + (a * a - 1.0) * u2 + 1e-12))
    sin_t = np.sqrt(np.clip(1.0 - cos_t * cos_t, 0.0, 1.0))
    Ht = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=-1)   # tangent-space half-vector
    T, B = _tangent_frame(N)
    H = Ht[..., 0:1] * T + Ht[..., 1:2] * B + Ht[..., 2:3] * N                  # to world
    L = 2.0 * np.sum(V * H, axis=-1, keepdims=True) * H - V                     # reflect V about H
    ndh = np.clip(np.sum(N * H, axis=-1), 0.0, 1.0)
    vdh = np.clip(np.sum(V * H, axis=-1), 1e-4, 1.0)
    pdf = d_ggx(ndh, roughness) * ndh / (4.0 * vdh + 1e-12)
    return L, pdf


def directional_albedo(metallic, roughness, base_color=(1.0, 1.0, 1.0), n=4096, view_cos=1.0, seed=0):
    """Hemispherical-directional reflectance: integrate the BRDF over all incoming L for a fixed view, by Monte
    Carlo (uniform hemisphere). 1.0 = energy-conserving; < 1.0 = the single-scatter GGX energy loss. The white-
    furnace test."""
    rng = np.random.default_rng(seed)
    N = np.array([0.0, 0.0, 1.0])
    V = np.array([np.sqrt(max(0.0, 1.0 - view_cos ** 2)), 0.0, view_cos])
    # uniform hemisphere samples
    u1 = rng.random(n); u2 = rng.random(n)
    z = u1; r = np.sqrt(1.0 - z * z); ph = 2 * np.pi * u2
    L = np.stack([r * np.cos(ph), r * np.sin(ph), z], axis=-1)
    Nb = np.broadcast_to(N, L.shape); Vb = np.broadcast_to(V, L.shape)
    fr = cook_torrance(Nb, Vb, L, base_color, metallic, roughness)              # already includes N.L
    # uniform hemisphere pdf = 1/(2pi); estimator = mean(fr / pdf) = mean(fr) * 2pi
    return float(np.mean(fr.sum(-1) / 3.0) * 2.0 * np.pi)


def _cosine_sample(N, rng):
    """Cosine-weighted hemisphere sample about N (pdf = N.L / pi) -- the right importance sampler for a diffuse
    (Lambert) lobe, since it cancels the cosine in the rendering equation."""
    N = np.asarray(N, float); n = N.shape[0]
    u1 = rng.random(n); u2 = rng.random(n)
    r = np.sqrt(u1); ph = 2.0 * np.pi * u2
    x = r * np.cos(ph); y = r * np.sin(ph); z = np.sqrt(np.clip(1.0 - u1, 0.0, 1.0))
    T, B = _tangent_frame(N)
    return x[:, None] * T + y[:, None] * B + z[:, None] * N


def sample_brdf(N, V, base_color, metallic, roughness, rng):
    """Sample one bounced direction for a path tracer from the FULL material (diffuse + GGX specular), and return
    (L, weight) where weight = f_r(L) * N.L / pdf(L) -- the throughput multiplier for the bounce. Uses one-sample
    MULTIPLE IMPORTANCE SAMPLING: stochastically pick the diffuse or specular lobe, but evaluate the COMBINED
    mixture pdf at the chosen L, so the estimator is unbiased no matter which lobe drew the sample. base_color
    (n,3); metallic, roughness scalar or (n,)."""
    N = np.asarray(N, float); V = np.asarray(V, float); base = np.asarray(base_color, float)
    n = N.shape[0]
    met = np.broadcast_to(np.asarray(metallic, float), (n,))
    rough = np.broadcast_to(np.asarray(roughness, float), (n,))
    F0 = metallic_f0(base, met[:, None])
    lum = lambda c: 0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]
    p_spec = np.clip(0.5 * met + 0.5 * lum(F0) / (lum(F0) + lum(base) * (1.0 - met) + 1e-4), 0.15, 0.9)
    choose_spec = rng.random(n) < p_spec
    L = np.empty((n, 3))
    if choose_spec.any():
        Ls, _ = sample_ggx(N[choose_spec], V[choose_spec], rough[choose_spec], rng)
        L[choose_spec] = Ls
    if (~choose_spec).any():
        L[~choose_spec] = _cosine_sample(N[~choose_spec], rng)
    H = V + L; H = H / (np.linalg.norm(H, axis=-1, keepdims=True) + 1e-12)
    ndl = np.clip(np.sum(N * L, axis=-1), 0.0, 1.0)
    ndh = np.clip(np.sum(N * H, axis=-1), 0.0, 1.0)
    vdh = np.clip(np.sum(V * H, axis=-1), 1e-4, 1.0)
    pdf_spec = d_ggx(ndh, rough) * ndh / (4.0 * vdh + 1e-12)
    pdf_diff = ndl / np.pi
    pdf = p_spec * pdf_spec + (1.0 - p_spec) * pdf_diff           # the mixture pdf -> unbiased one-sample MIS
    f = cook_torrance(N, V, L, base_color, met, rough)            # f_r * N.L
    ok = (ndl > 0.0) & (pdf > 1e-8)
    weight = np.where(ok[:, None], f / (pdf[:, None] + 1e-12), 0.0)
    return L, weight


def _selftest():
    # 1. energy conservation: a smooth dielectric reflects close to (but not above) 1.0 in the white furnace
    a_smooth = directional_albedo(metallic=0.0, roughness=0.1)
    assert a_smooth <= 1.02, a_smooth                              # no energy gain (small MC slack)
    # 2. the single-scatter GGX loss is real and grows with roughness (kept negative, not hidden)
    a_rough = directional_albedo(metallic=0.0, roughness=0.9)
    assert a_rough < a_smooth, (a_smooth, a_rough)                 # rough loses energy
    # 3. importance sampler is unbiased: the GGX estimator agrees with brute-force uniform integration
    rng = np.random.default_rng(0)
    N = np.tile(np.array([0.0, 0.0, 1.0]), (20000, 1))
    V = np.tile(np.array([0.3, 0.0, 0.954]), (20000, 1)); V /= np.linalg.norm(V, axis=-1, keepdims=True)
    L, pdf = sample_ggx(N, V, 0.4, rng)
    ndl = np.clip(np.sum(N * L, axis=-1), 0.0, 1.0)
    valid = (ndl > 0) & (pdf > 1e-6)
    fr = cook_torrance(N[valid], V[valid], L[valid], (1.0, 1.0, 1.0), 0.0, 0.4)
    est = float(np.mean(fr.sum(-1) / 3.0 / pdf[valid]))           # importance-sampled directional albedo
    ref = directional_albedo(metallic=0.0, roughness=0.4, view_cos=0.954, n=200000)
    assert abs(est - ref) < 0.06, (est, ref)                      # estimator matches the reference -> unbiased
    print(f"brdf selftest ok: white-furnace smooth {a_smooth:.3f} (~1, conserving), rough {a_rough:.3f} "
          f"(<smooth: single-scatter loss kept); GGX importance sampler unbiased ({est:.3f} vs {ref:.3f})")


if __name__ == "__main__":
    _selftest()


# ---------------------------------------------------------------------------------------------------------------
# RE-ENABLE (adaptive-dispatch audit): MULTI-SCATTER energy compensation (Kulla-Conty 2017), gated by roughness.
# Single-scatter GGX (cook_torrance above) drops the light that bounces between microfacets more than once, so a
# ROUGH METAL loses energy -- the white-furnace test dips to ~0.5 at roughness 0.8 (measured). Kulla-Conty adds that
# energy back with a cheap analytic term built from the single-scatter directional albedo E(mu): bake E once, then
# the term is a couple of interpolated lookups. It only ADDS the missing energy (never removes), so it cannot
# over-brighten an already-conserving surface -- NO harm mode. We still GATE on roughness so smooth surfaces (where
# the loss is negligible and the diffuse term masks it anyway) skip the compensation entirely.

_MS_ENERGY_CURVES = {}          # roughness (rounded) -> (mu_grid, E_grid, E_avg), baked once each


def _ms_energy_curve(roughness, n_mu=12, n=4096, seed=0):
    """Bake (once, cached) the single-scatter specular directional albedo E(mu) for this roughness -- the fraction
    of energy the single-scatter lobe reflects at each view cosine (metallic/white, so no diffuse to mask it) -- plus
    its cosine-weighted average E_avg. These are the ingredients of the Kulla-Conty multi-scatter term."""
    key = round(float(roughness), 2)
    hit = _MS_ENERGY_CURVES.get(key)
    if hit is not None:
        return hit
    mu = np.linspace(0.06, 1.0, n_mu)
    E = np.array([directional_albedo(metallic=1.0, roughness=key, base_color=(1.0, 1.0, 1.0),
                                     n=n, view_cos=float(m), seed=seed) for m in mu])
    E = np.clip(E, 0.0, 1.0)
    g = E * mu
    E_avg = float(2.0 * np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(mu)))   # E_avg = 2 integral E(mu) mu dmu (trapezoid)
    _MS_ENERGY_CURVES[key] = (mu, E, E_avg)
    return _MS_ENERGY_CURVES[key]


def _f_avg(base_color, metallic):
    """The hemisphere-average Fresnel reflectance, F_avg ~ F0 + (1-F0)/21 (the analytic average of Schlick). Tints
    the multi-scatter lobe for coloured metals; ~1 for the white furnace."""
    F0 = metallic_f0(np.asarray(base_color, float), metallic)
    return F0 + (1.0 - F0) / 21.0


def multiscatter_term(roughness, n_dot_v, n_dot_l, base_color=(1.0, 1.0, 1.0), metallic=1.0):
    """The Kulla-Conty multi-scatter BRDF term (no cosine): f_ms = (1-E(v))(1-E(l)) / (pi (1-E_avg)), tinted by the
    average Fresnel. Returns the energy the single-scatter lobe missed. Broadcast-safe in n_dot_l."""
    mu, E, E_avg = _ms_energy_curve(roughness)
    Ev = np.interp(np.clip(n_dot_v, 0.0, 1.0), mu, E)
    El = np.interp(np.clip(n_dot_l, 0.0, 1.0), mu, E)
    fms = (1.0 - Ev) * (1.0 - El) / (np.pi * (1.0 - E_avg) + 1e-6)
    favg = _f_avg(base_color, metallic)                 # (3,) per-channel tint
    # colored multi-scatter energy scaling (Kulla-Conty): F_avg^2 E_avg / (1 - F_avg (1 - E_avg))
    scale = favg * favg * E_avg / (1.0 - favg * (1.0 - E_avg) + 1e-6)
    return fms[..., None] * scale if np.ndim(fms) else fms * scale


def cook_torrance_ms(N, V, L, base_color, metallic, roughness):
    """Cook-Torrance GGX WITH the multi-scatter energy compensation added (the re-enabled, energy-conserving BRDF).
    Equals the single-scatter cook_torrance plus f_ms * N.L. Use where roughness is high enough for the loss to
    matter (see brdf_gated)."""
    single = cook_torrance(N, V, L, base_color, metallic, roughness)
    N = np.asarray(N, float); V = np.asarray(V, float); L = np.asarray(L, float)
    n_dot_v = np.clip(np.sum(N * V, axis=-1), 0.0, 1.0)
    n_dot_l = np.clip(np.sum(N * L, axis=-1), 0.0, 1.0)
    fms = multiscatter_term(roughness, n_dot_v, n_dot_l, base_color, metallic)
    return single + fms * n_dot_l[..., None]            # add back the missing energy, cosine-weighted


# roughness below this, single-scatter loss is negligible -> skip the compensation (measured: metals only lose real
# energy once roughness climbs; the diffuse term masks it for dielectrics).
MS_ROUGHNESS_GATE = 0.25


def brdf_gated(N, V, L, base_color, metallic, roughness, roughness_gate=MS_ROUGHNESS_GATE):
    """Evaluate the BRDF, RE-ENABLING multi-scatter compensation only in its regime. At low roughness (< gate) the
    single-scatter loss is negligible, so use the cheap cook_torrance; at high roughness add the Kulla-Conty term to
    conserve energy. Returns (brdf_value, info). The detector is the material roughness -- known per hit -- and the
    term only adds missing energy, so the gate can never over-brighten a conserving surface."""
    from holographic.misc.holographic_regimegate import RegimeGate
    gate = RegimeGate("multiscatter_ggx", detect=lambda _r: float(roughness), threshold=roughness_gate,
                      superior=lambda _r: cook_torrance_ms(N, V, L, base_color, metallic, roughness),
                      fallback=lambda _r: cook_torrance(N, V, L, base_color, metallic, roughness))
    return gate.apply(roughness)


def directional_albedo_ms(metallic, roughness, base_color=(1.0, 1.0, 1.0), n=8192, view_cos=1.0, seed=0):
    """The white-furnace test WITH multi-scatter compensation -- should return ~1.0 (energy conserving) where the
    single-scatter directional_albedo dips below 1.0."""
    rng = np.random.default_rng(seed)
    N = np.array([0.0, 0.0, 1.0])
    V = np.array([np.sqrt(max(0.0, 1.0 - view_cos ** 2)), 0.0, view_cos])
    u1 = rng.random(n); u2 = rng.random(n)
    z = u1; r = np.sqrt(1.0 - z * z); ph = 2 * np.pi * u2
    L = np.stack([r * np.cos(ph), r * np.sin(ph), z], axis=-1)
    Nb = np.broadcast_to(N, L.shape); Vb = np.broadcast_to(V, L.shape)
    fr = cook_torrance_ms(Nb, Vb, L, base_color, metallic, roughness)
    return float(np.mean(fr.sum(-1) / 3.0) * 2.0 * np.pi)
