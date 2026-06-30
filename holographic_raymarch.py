"""Field-native lighting on signed-distance fields (LIGHT-1): a CPU sphere-tracer and the shading effects that
fall out of an SDF -- ambient occlusion, soft shadows, an HDRI sky dome, refraction, and subsurface translucency.

WHY THIS MODULE EXISTS, AND WHAT IS / ISN'T "VSA-NATIVE"
-------------------------------------------------------
These are LIGHT-TRANSPORT effects, not hypervector algebra, and this file does not pretend otherwise. What is
genuinely true -- and the reason they belong in this engine -- is that holostuff is SDF / FIELD-native, and on a
field these effects are cheap and composable because the field answers the only questions they ask:
  * "how far to the nearest surface from here?" -> the SDF value itself (sphere tracing, ambient occlusion).
  * "is the path to the light blocked, and by how much?" -> march the field toward the light (soft shadows).
  * "which way does the surface face?" -> the SDF gradient (normals, reflection, refraction).
  * "how much solid does light cross inside the object?" -> the field's interior, integrated (subsurface).
Every per-ray quantity is computed VECTORISED over all rays at once (the loops are over march STEPS, ~tens, not
over pixels). Where a step is genuinely VSA-shaped it is flagged (an accumulation is a scatter = a bundle); where
it is just optics (Snell's law) it is called optics, not dressed up.

The expensive global terms (global illumination, caustics) live in holographic_globalillum.py and lean on the
engine's real contributions: the adaptive-anchor IRRADIANCE CACHE and the forward light-splat (a scatter/bundle).
"""

import numpy as np


def sdf_normal(sdf, P, eps=1e-3):
    """The surface normal at points P:(M,3) = the normalised gradient of the SDF, by central differences
    (6 vectorised evals). The field tells the geometry which way it faces."""
    P = np.asarray(P, float)
    ex = np.array([eps, 0, 0]); ey = np.array([0, eps, 0]); ez = np.array([0, 0, eps])
    nx = sdf.eval(P + ex) - sdf.eval(P - ex)
    ny = sdf.eval(P + ey) - sdf.eval(P - ey)
    nz = sdf.eval(P + ez) - sdf.eval(P - ez)
    N = np.stack([nx, ny, nz], axis=1)
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)


def sphere_trace(sdf, O, D, max_steps=96, max_dist=20.0, surf_eps=1e-3):
    """Sphere-trace rays (O, D both (M,3), D unit) through an SDF: at each step the SDF value is a SAFE distance to
    advance (the largest step that cannot overshoot a surface). Vectorised over all rays; the loop is over steps.
    Returns (hit mask (M,), t (M,), pos (M,3))."""
    O = np.asarray(O, float); D = np.asarray(D, float)
    M = len(D)
    t = np.zeros(M); hit = np.zeros(M, bool); active = np.ones(M, bool)
    for _ in range(max_steps):
        P = O + t[:, None] * D
        d = sdf.eval(P)
        newhit = active & (d < surf_eps)
        hit |= newhit
        active &= ~newhit
        active &= (t < max_dist)
        if not active.any():
            break
        t = t + np.where(active, np.clip(d, 0.0, None), 0.0)  # advance only the still-marching rays
    return hit, t, O + t[:, None] * D


def ambient_occlusion(sdf, P, N, samples=6, step=0.06, k=1.6):
    """SDF ambient occlusion (Quilez): march a short way along the normal; if the SDF says a surface is closer
    than the distance marched, geometry is nearby and the point is occluded. The field-native AO -- no rays, no
    hemisphere sampling, just the distance function read a few times. Vectorised; loop is over `samples` taps."""
    P = np.asarray(P, float)
    occ = np.zeros(len(P)); sca = 1.0
    for i in range(1, samples + 1):
        h = step * i
        d = sdf.eval(P + N * h)                               # nearest-surface distance h along the normal
        occ += (h - d) * sca                                 # if d<h something is filling the gap -> occlusion
        sca *= 0.85
    return np.clip(1.0 - k * occ, 0.0, 1.0)


def soft_shadow(sdf, P, Ldir, k=12.0, mint=0.02, maxt=12.0, steps=48):
    """SDF soft shadow (Quilez): march from P toward the light; the closest the ray passes to any surface,
    scaled by distance, is the penumbra. A hard occluder -> 0, clear path -> 1, a grazing miss -> a soft edge.
    Field-native (one SDF read per step), vectorised over points."""
    P = np.asarray(P, float); Ldir = np.asarray(Ldir, float)
    res = np.ones(len(P)); t = np.full(len(P), mint); alive = np.ones(len(P), bool)
    for _ in range(steps):
        h = sdf.eval(P + Ldir * t[:, None])
        res = np.where(alive, np.minimum(res, k * np.clip(h, 0, None) / np.maximum(t, 1e-6)), res)
        t = t + np.clip(h, 0.01, 0.25)
        alive &= (h > 1e-3) & (t < maxt)                     # stop on a hit (shadowed) or past the light
        if not alive.any():
            break
    return np.clip(res, 0.0, 1.0)


def sky_dome(D, sun_dir=(-0.4, 0.7, -0.3), sun_color=(1.0, 0.95, 0.85), sky_color=(0.35, 0.55, 0.95),
             horizon=(0.8, 0.85, 0.9), ground=(0.25, 0.22, 0.2), sun_size=0.04, env=None):
    """HDRI sky dome: the environment radiance arriving from direction D:(M,3) (unit). With `env` (an
    equirectangular (H,W,3) image) it is sampled by longitude/latitude -- real HDRI support. Otherwise a
    procedural physical-ish sky: a zenith->horizon gradient, a warm sun disk + glow, and a ground hemisphere.
    The incoming light field is a SUPERPOSITION of these directional sources -- a bundle of radiance. Vectorised."""
    D = np.asarray(D, float)
    if env is not None:
        env = np.asarray(env, float); H, W = env.shape[:2]
        u = (np.arctan2(D[:, 0], -D[:, 2]) / (2 * np.pi) + 0.5) % 1.0          # longitude -> [0,1)
        v = np.clip(0.5 - np.arcsin(np.clip(D[:, 1], -1, 1)) / np.pi, 0, 1)    # latitude  -> [0,1]
        xi = np.clip((u * W).astype(int), 0, W - 1); yi = np.clip((v * H).astype(int), 0, H - 1)
        return env[yi, xi]
    up = np.clip(D[:, 1], -1, 1)
    sky = np.asarray(horizon) * (1 - np.clip(up, 0, 1)[:, None]) + np.asarray(sky_color) * np.clip(up, 0, 1)[:, None]
    grd = np.broadcast_to(np.asarray(ground), D.shape)
    col = np.where((up < 0)[:, None], grd, sky).astype(float)
    s = np.asarray(sun_dir, float); s = s / (np.linalg.norm(s) + 1e-12)
    cs = D @ s                                                # alignment with the sun
    disk = np.clip((cs - (1 - sun_size)) / sun_size, 0, 1)    # the sun's bright core
    glow = np.clip(cs, 0, 1) ** 8 * 0.4                       # a soft glow around it
    return np.clip(col + (disk + glow)[:, None] * np.asarray(sun_color), 0, None)


def refract_dir(D, N, ior=1.5):
    """Snell's law refraction of incident unit ray D at a surface with unit normal N (entering a medium of index
    `ior`). Returns the refracted direction, or the REFLECTED direction on total internal reflection. This is
    optics -- plain vector math, not a hypervector trick, and labelled as such. Vectorised."""
    D = np.asarray(D, float); N = np.asarray(N, float)
    cosi = -np.clip(np.sum(D * N, axis=1), -1, 1)
    eta = np.where(cosi < 0, ior, 1.0 / ior)[:, None]        # entering vs exiting
    n = np.where((cosi < 0)[:, None], -N, N)
    cosi = np.abs(cosi)[:, None]
    k = 1 - eta ** 2 * (1 - cosi ** 2)
    refr = eta * D + (eta * cosi - np.sqrt(np.clip(k, 0, None))) * n
    refl = D - 2 * np.sum(D * n, axis=1)[:, None] * n
    return np.where(k < 0, refl, refr)                        # TIR -> reflect


def subsurface(sdf, P, N, Ldir, depth=0.6, steps=10, sigma=4.0):
    """A field-native subsurface / translucency term: from just under the surface, march toward the light and
    measure how much SOLID the light must cross to reach P (the SDF is negative inside). Thin regions transmit
    more, so they GLOW -- the wax/skin/leaf look. An approximation of true diffusion SSS, computed from the
    field's interior. Vectorised; loop over march steps."""
    P = np.asarray(P, float)
    start = P - N * 1e-2                                      # step just inside the surface
    inside = np.zeros(len(P))
    dl = depth / steps
    for i in range(steps):
        Q = start + Ldir * (i * dl)
        inside += (sdf.eval(Q) < 0.0) * dl                    # accumulate interior path length toward the light
    return np.exp(-sigma * inside)                            # Beer-Lambert transmission: thin -> bright


def render_sdf(sdf, camera, width=256, height=256, light_dir=(-0.4, 0.7, -0.3), base_color=(0.85, 0.5, 0.35),
               sky=None, ao=True, shadows=True, reflect=0.25, refract=0.0, ior=1.5, sss=0.0,
               sss_color=(1.0, 0.4, 0.3), ambient=0.25):
    """Compose the field-native effects into one image. Primary rays are sphere-traced; hits get Lambert direct
    light gated by a SOFT SHADOW, ambient gated by AMBIENT OCCLUSION, an environment REFLECTION sampled from the
    HDRI sky, optional REFRACTION (the sky seen bent through the surface), and optional SUBSURFACE glow; misses
    show the sky dome. Returns (H,W,3) in [0,1]. `sky` may be an equirectangular HDRI array. Vectorised over all
    pixels."""
    eye, dirs = camera.ray_dirs(width, height)
    D = dirs.reshape(-1, 3); O = np.broadcast_to(eye, D.shape)
    L = np.asarray(light_dir, float); L = L / (np.linalg.norm(L) + 1e-12)
    skyfn = (lambda d: sky_dome(d, env=sky)) if sky is not None else (lambda d: sky_dome(d))

    hit, t, P = sphere_trace(sdf, O, D)
    col = skyfn(D)                                            # background = the sky for every ray
    if hit.any():
        Ph = P[hit]; Dh = D[hit]
        Nh = sdf_normal(sdf, Ph)
        ndl = np.clip(Nh @ L, 0, None)
        sh = soft_shadow(sdf, Ph + Nh * 2e-3, L) if shadows else 1.0
        occ = ambient_occlusion(sdf, Ph, Nh) if ao else 1.0
        base = np.asarray(base_color, float)
        shade = (ambient * occ)[:, None] * base + (ndl * sh)[:, None] * base   # ambient(AO) + direct(shadow)
        if reflect > 0:                                      # environment reflection off the surface
            R = Dh - 2 * np.sum(Dh * Nh, axis=1)[:, None] * Nh
            fres = reflect * (0.04 + 0.96 * (1 - np.clip(-np.sum(Dh * Nh, axis=1), 0, 1)) ** 5)  # Schlick fresnel
            shade = (1 - fres)[:, None] * shade + fres[:, None] * skyfn(R)
        if refract > 0:                                      # the sky seen bent through the surface (optics)
            Rt = refract_dir(Dh, Nh, ior)
            shade = (1 - refract) * shade + refract * skyfn(Rt)
        if sss > 0:                                          # subsurface glow added where the object is thin
            trans = subsurface(sdf, Ph, Nh, L)
            shade = shade + sss * trans[:, None] * np.asarray(sss_color)
        col[hit] = np.clip(shade, 0, 1)
    return np.clip(col.reshape(height, width, 3), 0, 1)


def _selftest():
    from holographic_sdf import sphere, plane
    from holographic_render import Camera
    scene = sphere(0.8).union(plane(-0.8))
    cam = Camera(eye=(1.6, 1.0, 2.4), target=(0, 0, -0.2), fov_deg=45)
    img = render_sdf(scene, cam, 64, 64, ao=True, shadows=True, reflect=0.2)
    assert img.shape == (64, 64, 3) and 0.0 <= img.min() and img.max() <= 1.0
    # AO must DARKEN the crease where the sphere meets the plane vs an unoccluded patch of plane
    P_open = np.array([[3.0, -0.8, 0.0]])                     # open floor, far from the sphere
    P_crease = np.array([[0.0, -0.78, 0.78]])                # near the sphere/plane contact
    N = np.array([[0.0, 1.0, 0.0]])
    ao_open = ambient_occlusion(scene, P_open, N)[0]
    ao_crease = ambient_occlusion(scene, P_crease, sdf_normal(scene, P_crease))[0]
    assert ao_crease < ao_open                                # the crease is more occluded
    # soft shadow: a point under the sphere is shadowed from an overhead light
    lit = soft_shadow(scene, np.array([[3.0, -0.79, 0.0]]), np.array([0., 1, 0]))[0]
    shad = soft_shadow(scene, np.array([[0.0, -0.79, 0.0]]), np.array([0., 1, 0]))[0]
    assert shad < lit
    print(f"raymarch selftest ok: render {img.shape}, AO crease {ao_crease:.2f} < open {ao_open:.2f}, "
          f"shadow under-sphere {shad:.2f} < open {lit:.2f}")


if __name__ == "__main__":
    _selftest()
