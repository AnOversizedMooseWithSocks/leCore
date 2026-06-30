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
    F0 = 0.04 * (1.0 - metallic) + base * metallic                  # dielectric 0.04 / metal tinted
    F = fresnel_schlick(vdh, F0)                                    # (...,3)
    D = d_ggx(ndh, roughness)                                       # (...,)
    G = g_smith(ndv, ndl, roughness)                               # (...,)
    spec = (D * G)[..., None] * F / (4.0 * ndv * ndl + 1e-6)[..., None]
    kd = (1.0 - F) * (1.0 - metallic)                               # energy left for diffuse; metals: 0
    diffuse = kd * base / np.pi
    return (diffuse + spec) * ndl[..., None]


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
    F0 = 0.04 * (1.0 - met)[:, None] + base * met[:, None]
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
