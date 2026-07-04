#!/usr/bin/env python3
"""
make_gallery.py -- render the "fresh" showcase visuals for GALLERY.md, straight from the engine.

Plain NumPy + matplotlib, no other deps. Every visual is wrapped in try/except so one failure doesn't lose the
rest. Run from the repo root:   python make_gallery.py   ->  writes PNGs into ./gallery/

Two kinds of visual:
  * 3-D renders  : the from-scratch path tracer on signed-distance geometry (spheres, glass, a fractal sponge).
  * data charts  : measured behaviour of the core algebra -- op cost, compression vs SQL, memory capacity,
                   and graceful degradation. These are the "non-3-D" story: how the thing actually behaves.
(The rest of the images in GALLERY.md come from the committed test/benchmark harness, in ./figures/.)
"""
import os, io, gzip, csv, time, sqlite3, tempfile, traceback
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "gallery"; os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------- small shared helpers
def _tonemap(hdr, exposure=1.0):
    """HDR -> displayable sRGB via the ACES FILMIC curve with AUTO-EXPOSURE (Narkowicz 2015 + log-average metering)
    -- filmic contrast, a graceful highlight roll-off, and each scene self-exposed to mid-grey, instead of the old
    flat Reinhard that greyed everything out. See holographic_gbuffer.aces_tonemap."""
    from holographic_gbuffer import aces_tonemap
    return aces_tonemap(hdr, exposure=exposure, auto=True)

class _Cam:
    """A tiny pinhole camera: an eye point and a grid of ray directions. Enough for the path tracer."""
    def __init__(self, eye=(0.0, 0.6, 4.2), tilt=-0.12, fov=1.3):
        self.eye = np.array(eye, float); self.tilt = tilt; self.fov = fov
    def ray_dirs(self, w, h, jitter=None):
        ys, xs = np.mgrid[0:h, 0:w]
        jx, jy = (0.0, 0.0) if jitter is None else (jitter[0], jitter[1])   # sub-pixel offset for anti-aliasing
        u = ((xs + jx) / (w - 1) - 0.5) * self.fov
        v = -((ys + jy) / (h - 1) - 0.5) * self.fov
        d = np.stack([u, v + self.tilt, -np.ones_like(u)], -1)
        return self.eye, d / np.linalg.norm(d, axis=-1, keepdims=True)

# One KEY-LIGHT direction, shared by the sky (so the sun disk sits here) and the caustics (so the focused cusp
# lands consistently). A bright HDR sun (values >> 1) gives metal/glass a strong, high-contrast world to reflect
# and refract -- the flat low-contrast gradient the old gallery used was the washed-out culprit.
SUN = (-0.35, 0.72, -0.30)

def _sky(D):
    """A punchy HDRI-style environment (holographic_raymarch.sky_dome): graduated blue sky, a warm BRIGHT sun disk
    + glow along SUN, and a darker ground -- high dynamic range, so reflections and refractions actually pop and
    the ACES tonemap has real highlights to roll off."""
    from holographic_raymarch import sky_dome
    return sky_dome(D, sun_dir=SUN, sun_color=(7.0, 6.2, 5.2), sky_color=(0.25, 0.42, 0.85),
                    horizon=(0.85, 0.88, 0.95), ground=(0.18, 0.16, 0.15), sun_size=0.018)

def _sky2(D):
    """A brighter STUDIO variant for the metal/glass/water hero shots: a big soft key sun for smooth highlights,
    a bright sky, and a lighter ground for fill. Same sky_dome machinery, softer and higher-key than _sky."""
    from holographic_raymarch import sky_dome
    return sky_dome(D, sun_dir=SUN, sun_color=(6.0, 5.7, 5.2), sky_color=(0.55, 0.70, 1.0),
                    horizon=(1.05, 1.05, 1.1), ground=(0.42, 0.40, 0.37), sun_size=0.05)


# =========================================================================== 3-D RENDERS
# Every 3-D scene goes through render_auto (holographic_gbuffer): an AUTO-CALIBRATING render. It samples in
# passes and, after each pass, asks the calibrated stop rule (holographic_adaptive_sample.converged_mask) which
# pixels have reached the quality target -- those stop, the rest keep sampling -- then denoises with a
# VARIANCE-GUIDED SVGF filter whose per-pixel strength comes from the noise the sampler measured. So there is
# NO per-scene spp or denoise tuning: one `QUALITY` knob renders every scene to the same bar, spending samples
# and blur exactly where each scene needs them (which is why the old fixed-spp gallery was grainy in the hard
# spots and wasteful in the easy ones).
WIDTH, HEIGHT = 240, 180        # render size for the 3-D showcase
QUALITY = "high"                # the ONE render knob (a target confidence interval); see render_auto
REF_SPP = 128                   # a high-sample reference, used ONLY by the benchmark print

def _save_render(name, scene, cam, material, max_bounce=4, bench=False, sky=None, dispersion=0.0, caustics=None):
    """Render `scene` with the auto-calibrating pipeline, tonemap (ACES filmic), save gallery/<name>.png. With
    bench=True also render a RAW path trace at the same mean sample count the auto-render chose, plus a high-spp
    reference, and PRINT the honest tonemap-space PSNR. `dispersion`>0 uses the 3-pass RGB spectral trace (glass
    splits light into a rainbow fringe); `caustics` (a kwargs dict) composites a focused caustic onto the floor
    BEFORE tonemapping. `max_bounce` is a physical property of the scene (glass/water need more), not a quality
    knob, so it stays per-scene."""
    from holographic_gbuffer import render_auto, render_dispersion, add_caustics
    from holographic_pathtrace import path_trace
    skyfn = sky if sky is not None else _sky
    if dispersion > 0:
        clean, stats = render_dispersion(scene, cam, WIDTH, HEIGHT, material, sky=skyfn, quality=QUALITY,
                                         max_bounce=max_bounce, seed=0, dispersion=dispersion, return_stats=True)
    else:
        clean, stats = render_auto(scene, cam, WIDTH, HEIGHT, material, sky=skyfn, quality=QUALITY,
                                   max_bounce=max_bounce, seed=0, return_stats=True)
    if caustics is not None:                                  # add the focused light on the floor (HDR, pre-tonemap)
        clean = add_caustics(clean, scene, cam, WIDTH, HEIGHT, **caustics)
    plt.imsave(f"{OUT}/{name}.png", _tonemap(clean))
    if bench:
        eq = int(round(stats["mean_samples"]))               # raw baseline at the SAME average sample budget
        raw = path_trace(scene, cam, width=WIDTH, height=HEIGHT, spp=eq, max_bounce=max_bounce,
                         material=material, sky=skyfn, seed=0, antialias=True)   # same AA as render_auto (fair)
        ref = path_trace(scene, cam, width=WIDTH, height=HEIGHT, spp=REF_SPP, max_bounce=max_bounce,
                         material=material, sky=skyfn, seed=99, antialias=True)  # AA reference (matches the renders)
        refT, rawT, cleanT = _tonemap(ref), _tonemap(raw), _tonemap(clean)
        def psnr(a, b):
            mse = float(np.mean((a - b) ** 2))
            return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)
        print(f"  [BENCH {name}] auto quality='{QUALITY}' = {psnr(cleanT, refT):.1f} dB vs {REF_SPP}spp-ref; "
              f"raw at the same {eq} spp = {psnr(rawT, refT):.1f} dB  (adaptive spent {stats['mean_samples']:.0f} "
              f"mean / {stats['max_samples']:.0f} max spp over {stats['passes']} passes, {stats['seconds']:.1f}s)")
    print(f"  {name}.png")


def render_spheres():
    """Three spheres (red plastic, gold metal, blue plastic) on a checker floor -- the basic material showcase, and
    the BENCHMARK scene: it prints the raw-vs-pipeline PSNR so the quality win is a measured number. The materials
    are pulled from the library (holographic_matlib) so the renderer reads their metallic/roughness/colour rather
    than hand-typed numbers."""
    import holographic_matlib as ML
    centers = np.array([[-1.3, 0, 0], [0, 0, 0], [1.3, 0, 0]], float); radii = np.array([0.7, 0.9, 0.6])
    m_red, m_gold, m_blue = ML.material("plastic_red"), ML.material("gold"), ML.material("plastic_blue")
    m_pale, m_dark = ML.material("matte_white"), ML.material("matte_black")
    class Scene:
        def eval(self, P):
            d = np.min(np.linalg.norm(P[..., None, :] - centers, axis=-1) - radii, axis=-1)
            return np.minimum(d, P[..., 1] + 0.9)
    def _put(mask, mat, alb, met, rough):                       # write one library material's physics into the arrays
        alb[mask] = mat.base_color[:3]; met[mask] = mat.metallic; rough[mask] = mat.roughness
    def material(P):
        n = len(P); alb = np.tile([.8, .8, .8], (n, 1)).astype(float)
        met = np.zeros(n); rough = np.full(n, .6); emis = np.zeros((n, 3))
        g = P[:, 1] < -0.85; chk = ((np.floor(P[:, 0] * 1.5) + np.floor(P[:, 2] * 1.5)).astype(int) % 2 == 0)
        _put(g & chk, m_pale, alb, met, rough); _put(g & ~chk, m_dark, alb, met, rough)          # checker floor
        _put((P[:, 0] < -0.7) & ~g, m_red, alb, met, rough)                                      # red plastic
        _put(((P[:, 0] >= -0.7) & (P[:, 0] <= 0.7)) & ~g, m_gold, alb, met, rough)               # gold metal
        _put((P[:, 0] > 0.7) & ~g, m_blue, alb, met, rough)                                      # blue plastic
        return alb, met, rough, emis
    _save_render("render_spheres", Scene(), _Cam(), material, max_bounce=4, bench=True)


def render_glass():
    """A clear GLASS sphere in front of two coloured spheres. The material returns IOR>1 so the tracer bends rays
    through it; DISPERSION splits white light into a coloured fringe (R/G/B traced with different IOR); and a
    real CAUSTIC -- forward-traced light focused through the sphere -- brightens the floor beneath it. Glass and the two coloured spheres are physical materials from the
    library (holographic_matlib)."""
    import holographic_matlib as _ML
    back = np.array([[-0.8, -0.1, -1.4], [0.9, -0.1, -1.6]], float); br = np.array([0.55, 0.6])
    gc = np.array([0.0, 0.0, 0.4]); gr = 0.75
    class Scene:
        def eval(self, P):
            d = np.min(np.linalg.norm(P[..., None, :] - back, axis=-1) - br, axis=-1)
            glass = np.linalg.norm(P - gc, axis=-1) - gr
            return np.minimum(np.minimum(d, glass), P[..., 1] + 0.9)
    def material(P):
        n = len(P); alb = np.tile([.8, .8, .8], (n, 1)).astype(float)
        met = np.zeros(n); rough = np.full(n, .5); emis = np.zeros((n, 3)); ior = np.zeros(n)
        g = P[:, 1] < -0.85; chk = ((np.floor(P[:, 0] * 1.5) + np.floor(P[:, 2] * 1.5)).astype(int) % 2 == 0)
        alb[g] = np.where(chk[g, None], [.85, .85, .9], [.1, .1, .15])
        onglass = np.abs(np.linalg.norm(P - gc, axis=-1) - gr) < 0.05
        _gl = _ML.material("glass_clear")                                       # physical glass from the library
        ior[onglass] = _gl.ior; alb[onglass] = _gl.attenuation_color; rough[onglass] = _gl.roughness
        left = (P[:, 0] < -0.3) & ~onglass & ~g; alb[left] = _ML.material("plastic_red").base_color[:3]
        right = (P[:, 0] > 0.3) & ~onglass & ~g; alb[right] = _ML.material("plastic_blue").base_color[:3]
        return alb, met, rough, emis, ior
    cam = _Cam(eye=(0.0, 0.55, 3.6), tilt=-0.14, fov=1.25)                       # angled down a bit to see the caustic
    from holographic_sdf import sphere as _sphere
    glass_only = _sphere(gr).translate((float(gc[0]), float(gc[1]), float(gc[2])))  # refractor for the caustic map
    _save_render("render_glass", Scene(), cam, material, max_bounce=6, dispersion=0.06,
                 caustics=dict(light_dir=tuple(-np.array(SUN)), receiver_y=-0.9, extent=1.6, ior=1.5,
                               strength=1.1, tint=(1.0, 0.97, 0.9), caustic_sdf=glass_only))


def render_fractal():
    """A Menger-sponge fractal (the SDF is ~12 bytes; the geometry is generated, not stored)."""
    from holographic_sdf import menger
    sponge = menger(3, 1.4)
    class Scene:
        def eval(self, P):
            return np.minimum(np.asarray(sponge.eval(P)), P[..., 1] + 1.0)
    def material(P):
        n = len(P); alb = np.tile([.75, .55, .35], (n, 1)).astype(float)
        met = np.zeros(n); rough = np.full(n, .55); emis = np.zeros((n, 3))
        g = P[:, 1] < -0.95; alb[g] = [.2, .22, .28]
        return alb, met, rough, emis
    cam = _Cam(eye=(2.2, 1.6, 2.4), tilt=-0.18, fov=1.2)
    _save_render("render_fractal", Scene(), cam, material, max_bounce=4)


def render_identities():
    """ONE surface, THREE identities. The SAME rounded shape is rendered three times side by side as matte clay,
    polished copper, and clear glass -- and now those aren't hand-typed numbers: each is a PHYSICAL material pulled
    from the library (holographic_matlib), so the renderer reads the metallic/roughness/IOR straight off the
    material. 'One surface, a different material read of it' -- the mesh<->SDF<->splat 'three costumes' thesis."""
    import holographic_matlib as ML
    from holographic_sdf import sphere, box
    shape = box(0.5, 0.5, 0.5).rounded(0.12).smooth_union(sphere(0.62), k=0.28)      # rounded cube fused to a sphere
    xs = [-1.6, 0.0, 1.6]
    inst = [shape.translate((x, 0.0, 0.0)) for x in xs]                              # same shape, three places
    m_clay, m_copper, m_glass = ML.material("clay"), ML.material("copper"), ML.material("glass_clear")
    m_pale, m_dark = ML.material("checker_white") if "checker_white" in ML.names() else ML.material("matte_white"), \
                     ML.material("matte_black")
    class Scene:
        def eval(self, P):
            d = np.minimum(np.minimum(np.asarray(inst[0].eval(P)), np.asarray(inst[1].eval(P))),
                           np.asarray(inst[2].eval(P)))
            return np.minimum(d, P[..., 1] + 0.75)                                    # + a floor
    def _put(mask, mat, alb, met, rough, ior):                                       # write one material's physics
        if mat.transmission > 0:
            alb[mask] = mat.attenuation_color; ior[mask] = mat.ior; rough[mask] = mat.roughness
        else:
            alb[mask] = mat.base_color[:3]; met[mask] = mat.metallic; rough[mask] = mat.roughness
    def material(P):
        n = len(P); alb = np.tile([.8, .8, .8], (n, 1)).astype(float)
        met = np.zeros(n); rough = np.full(n, .5); emis = np.zeros((n, 3)); ior = np.zeros(n)
        g = P[:, 1] < -0.72; chk = ((np.floor(P[:, 0] * 1.4) + np.floor(P[:, 2] * 1.4)).astype(int) % 2 == 0)
        _put(g & chk, m_pale, alb, met, rough, ior); _put(g & ~chk, m_dark, alb, met, rough, ior)
        _put((P[:, 0] < -0.8) & ~g, m_clay, alb, met, rough, ior)                     # matte clay
        _put((P[:, 0] >= -0.8) & (P[:, 0] <= 0.8) & ~g, m_copper, alb, met, rough, ior)   # polished copper
        _put((P[:, 0] > 0.8) & ~g, m_glass, alb, met, rough, ior)                     # clear glass
        return alb, met, rough, emis, ior
    cam = _Cam(eye=(0.0, 0.85, 4.3), tilt=-0.12, fov=1.5)
    _save_render("render_identities", Scene(), cam, material, max_bounce=6, sky=_sky2)


def render_light_types():
    """THE LIGHT RIG -- the full set of placed lights, each shaping the beam differently (holographic_lights). Three
    pillars and a sphere on a floor, lit by: a SPOT with a GOBO (a striped light cookie projected across its cone,
    left, warm), a soft RECT area light (a softbox, middle), and an IES-style DOWNLIGHT (a real luminaire's measured
    beam shape, right, cool). A DOME light provides soft, shadowed sky fill (ambient occlusion falls out for free),
    so the shadows are gently lit rather than crushed to black. Every light samples directly via next-event
    estimation and the area lights are multi-sampled, so shadows are clean and correctly shaped."""
    from holographic_scene_doc import Scene
    from holographic_scene_render import render_scene_document
    from holographic_sdf import box, sphere
    from holographic_render import Camera
    from holographic_lights import SpotLight, RectLight, IESLight, DomeLight

    def _T(t):
        M = np.eye(4); M[:3, 3] = t; return M
    doc = Scene(seed=0)
    doc.add(name="floor", geometry=box(5.0, 0.1, 3.0).translate((0, -0.7, 0)), material="matte_white")
    doc.add(name="backdrop", geometry=box(5.0, 3.0, 0.15).translate((0, 0.8, -1.4)), material="matte_white")
    doc.add(name="p_left",  geometry=box(0.3, 1.0, 0.3).rounded(0.04), transform=_T((-1.5, -0.15, 0.3)), material="matte_white")
    doc.add(name="p_mid",   geometry=sphere(0.5),                        transform=_T((0.0, -0.15, 0.3)), material="matte_white")
    doc.add(name="p_right", geometry=box(0.3, 1.0, 0.3).rounded(0.04), transform=_T((1.5, -0.15, 0.3)), material="matte_white")

    cam = Camera(eye=(0.0, 1.0, 4.2), target=(0.0, -0.2, -0.3), fov_deg=46, aspect=WIDTH / HEIGHT)
    dark = lambda D: np.tile([0.01, 0.01, 0.015], (len(D), 1))         # dark sky: the dome does the ambient fill

    # a striped GOBO for the spot -- a projected light cookie (a callable over the cone's [-1,1]^2 cross-section)
    def stripes(uv):
        return (np.sin(uv[:, 1] * 5.0) > 0).astype(float)             # bold bars across the beam

    ies_profile = np.cos(np.linspace(0.0, np.pi / 2, 64)) ** 5        # a finer table -> smoother beam falloff

    lights = [
        # a cool sky DOME with a warmer ground -> soft shadowed fill, lifts the blacks (ambient occlusion for free)
        DomeLight(color=(0.30, 0.38, 0.55), ground_color=(0.14, 0.12, 0.10), intensity=1.0),
        # a warm SPOT with a wide penumbra (inner 20 / outer 46 -> a soft-edged pool) carrying a striped gobo
        SpotLight(position=(-1.5, 1.5, 1.3), direction=(0, -0.35, -1.0), inner_deg=20, outer_deg=46,
                  color=(1.0, 0.85, 0.6), intensity=48.0, gobo=stripes),
        RectLight(position=(0.0, 2.4, 1.4), u_vec=(0.7, 0, 0), v_vec=(0, 0.5, 0.3),                   # soft softbox
                  color=(1.0, 1.0, 1.0), intensity=38.0),
        IESLight(position=(1.5, 2.6, 0.4), direction=(0, -1, -0.1), profile=ies_profile,             # IES downlight
                 profile_max_deg=90.0, color=(0.8, 0.88, 1.0), intensity=80.0),
    ]
    # ALL THREE soft-light caches on: dome_cache (no dome here, so a no-op), soft_light_cache (the Rect area light's
    # penumbra, baked noise-free instead of sampled), and indirect_cache (the one-bounce GI, the dominant speckle,
    # cached so the tracer runs DIRECT-only). The scene that was ~323s brute-forcing the dome / ~117s with just the
    # dome cache now renders far faster AND with the placed-light speckle gone.
    hdr = render_scene_document(doc, cam, width=WIDTH, height=HEIGHT, quality=QUALITY, max_bounce=3, seed=0,
                                sky=dark, lights=lights, dome_cache=True, soft_light_cache=True, indirect_cache=True)
    plt.imsave(f"{OUT}/render_light_types.png", _tonemap(np.clip(hdr, 0, None)))
    print("  render_light_types.png")


def render_lit_scene():
    """PLACED LIGHTS + NEXT-EVENT ESTIMATION -- a scene lit by real lamps you put in the world, with correct
    shadows, instead of only a big sky. Before this the path tracer only gathered light when a bounce ray happened
    to escape and hit the emissive environment; a small bright lamp was almost never hit, so it was hopeless noise.
    NEE looks STRAIGHT at each light with a shadow ray (holographic_lights) and adds its contribution directly, so
    lamps converge instantly. Two lamps here: a warm SPHERE light (its AREA gives SOFT shadows -- a penumbra) and a
    cooler point light from the other side, over three objects on a floor in a dark room."""
    from holographic_scene_doc import Scene
    from holographic_scene_render import render_scene_document
    from holographic_sdf import sphere, box
    from holographic_render import Camera
    from holographic_lights import SphereLight, PointLight

    def _T(t):
        M = np.eye(4); M[:3, 3] = t; return M
    doc = Scene(seed=0)
    doc.add(name="floor", geometry=box(4.0, 0.1, 3.0).translate((0, -0.6, 0)), material="matte_white")
    doc.add(name="red",   geometry=sphere(0.5), transform=_T((-0.95, -0.1, 0.0)), material="plastic_red")
    doc.add(name="steel", geometry=sphere(0.5), transform=_T((0.35, -0.1, -0.2)), material="steel")
    doc.add(name="gold",  geometry=box(0.4, 0.5, 0.4).rounded(0.06), transform=_T((1.3, -0.1, 0.1)), material="gold")

    cam = Camera(eye=(0.0, 0.7, 3.8), target=(0.0, -0.15, 0.0), fov_deg=46, aspect=WIDTH / HEIGHT)
    dark = lambda D: np.tile([0.02, 0.025, 0.035], (len(D), 1))          # a dark room: the lamps do the lighting
    lights = [
        SphereLight(position=(-2.0, 2.4, 1.6), radius=0.6, color=(1.0, 0.85, 0.6), intensity=26.0),   # warm key, soft
        PointLight(position=(2.4, 1.8, 1.2), color=(0.6, 0.7, 1.0), intensity=9.0),                   # cool rim, crisp
    ]
    hdr = render_scene_document(doc, cam, width=WIDTH, height=HEIGHT, quality=QUALITY, max_bounce=3, seed=0,
                                sky=dark, lights=lights)
    plt.imsave(f"{OUT}/render_lit_scene.png", _tonemap(np.clip(hdr, 0, None)))
    print("  render_lit_scene.png")


def render_iridescence():
    """THIN-FILM IRIDESCENCE -- the rainbow sheen of a soap bubble / oil slick, from real interference physics. A
    thin transparent film on the surface makes light reflected off its top and bottom INTERFERE; whether a given
    colour reinforces or cancels depends on the film thickness and the VIEW ANGLE, so the hue sweeps across a
    curved surface and shifts as you'd tilt it (holographic_thinfilm -> the path tracer tints the reflection by
    angle). Left: a soap bubble (thin ~300 nm film). Right: an oil-slick sphere (thicker ~420 nm) with a
    position-varying film so the colours swirl. The look comes from the MATERIAL's film thickness, not paint."""
    from holographic_scene_doc import Scene
    from holographic_scene_render import render_scene_document
    from holographic_sdf import sphere, box
    from holographic_render import Camera
    import holographic_matlib as ML

    def _T(t):
        M = np.eye(4); M[:3, 3] = t; return M
    doc = Scene(seed=0)
    doc.add(name="floor", geometry=box(4.0, 0.1, 3.0).translate((0, -1.0, 0)), material="matte_black")
    # a soap bubble: dark base so the SHEEN dominates; the thin film gives the rainbow
    doc.add(name="bubble", geometry=sphere(0.7), transform=_T((-1.0, -0.25, 0.0)), material="soap_bubble")
    # an oil-slick sphere: a thicker film, slightly bumpy so the swirl of colours reads across the surface
    _oil = ML.iridesce("oil_slick", 440.0)
    doc.add(name="oil", geometry=sphere(0.72).displace(0.05, 4.0), transform=_T((1.0, -0.23, -0.1)), material=_oil)

    cam = Camera(eye=(0.0, 0.5, 3.8), target=(0.0, -0.2, 0.0), fov_deg=44, aspect=WIDTH / HEIGHT)
    # a COLOURFUL sky is essential: iridescence tints REFLECTED light, so the environment needs colour to tint.
    from holographic_raymarch import sky_dome
    sky = lambda D: sky_dome(D, sun_dir=(0.4, 0.6, 0.5), sun_color=(7.0, 6.5, 6.0), sky_color=(0.25, 0.45, 0.85),
                             horizon=(0.95, 0.75, 0.55), ground=(0.15, 0.14, 0.13), sun_size=0.04)
    hdr = render_scene_document(doc, cam, width=WIDTH, height=HEIGHT, quality=QUALITY, max_bounce=4, seed=0, sky=sky)
    plt.imsave(f"{OUT}/render_iridescence.png", _tonemap(np.clip(hdr, 0, None)))
    print("  render_iridescence.png")


def render_crystal():
    """PHYSICAL-STRUCTURE MATERIALS -- the material's colour comes from its internal STRUCTURE, sampled per point,
    not a flat swatch. Left: a POLYCRYSTALLINE gem -- a Voronoi grain partition where each facet is a slightly
    different colour, darkened along the grain boundaries (crystal_material / holographic_cellular). Right: an ORE
    boulder -- a base rock with impurity INCLUSIONS (metallic pockets at a calibrated coverage, the planet's
    ore-deposit pattern scoped to a material -- material_inclusions / holographic_inclusions). Both are albedo
    SOCKETS f(points)->rgb that the renderer samples at each hit, carried on the scene object -- physical internal
    structure driving appearance."""
    from holographic_scene_doc import Scene
    from holographic_scene_render import render_scene_document
    from holographic_sdf import sphere, box
    from holographic_render import Camera
    from holographic_cellular import VoronoiCells, cell_albedo
    from holographic_inclusions import with_inclusions

    def _T(t):
        M = np.eye(4); M[:3, 3] = t; return M
    # the polycrystalline grain socket (a cool blue-violet gem), and the ore socket (grey rock + gold pockets)
    _cells = VoronoiCells(n_seeds=40, bounds=((-1.3, -1.3, -1.3), (1.3, 1.3, 1.3)), seed=3, jitter=1.0)
    _crystal = cell_albedo(_cells, base=(0.35, 0.42, 0.72), spread=0.30, crack=(0.03, 0.03, 0.05),
                           crack_width=0.04, seed=3)
    _ore = with_inclusions("rock", [("gold", 0.16, 4.0), ("iron_ore", 0.10, 6.0)], seed=1)

    doc = Scene(seed=0)
    doc.add(name="floor", geometry=box(4.0, 0.1, 3.0).translate((0, -1.0, 0)), material="matte_black")
    # a faceted gem (an octahedron-ish shape reads as "crystal"): intersect two boxes / use a rounded box tilted
    doc.add(name="gem", geometry=box(0.62, 0.62, 0.62).rotate((0.4, 1.0, 0.2), 0.9),
            transform=_T((-1.05, -0.35, 0.0)), material="matte_white", overrides={"albedo_socket": _crystal})
    doc.add(name="ore", geometry=sphere(0.72).displace(0.06, 5.0), transform=_T((1.05, -0.28, -0.1)),
            material="matte_white", overrides={"albedo_socket": _ore})

    cam = Camera(eye=(0.0, 0.6, 4.0), target=(0.0, -0.25, 0.0), fov_deg=44, aspect=WIDTH / HEIGHT)
    from holographic_raymarch import sky_dome
    sky = lambda D: sky_dome(D, sun_dir=SUN, sun_color=(6.5, 6.0, 5.4), sky_color=(0.30, 0.44, 0.82),
                             horizon=(0.80, 0.84, 0.92), ground=(0.16, 0.15, 0.14), sun_size=0.03)
    hdr = render_scene_document(doc, cam, width=WIDTH, height=HEIGHT, quality=QUALITY, max_bounce=3, seed=0, sky=sky)
    plt.imsave(f"{OUT}/render_crystal.png", _tonemap(np.clip(hdr, 0, None)))
    print("  render_crystal.png")


def _all_mat_names():
    import holographic_matlib as ML
    return ML.names()


def render_hot_metal():
    """THERMAL EMISSION -- a material glows because it is HOT, with the colour set by its temperature (Planck's
    law / blackbody): dull red near ~800K, orange by ~1400K, yellow-white past ~2500K. Left to right, the SAME
    iron bars are heated to increasing temperatures, so the blackbody ramp is visible as a row; a hotter poker
    lights the dark scene by its own glow. The emission is DERIVED from the material's temperature (matlib.heat +
    holographic_blackbody), not a hand-picked colour -- a physical property driving the render."""
    from holographic_sdf import box, cylinder, sphere
    from holographic_render import Camera
    import holographic_matlib as ML

    temps = [700, 1000, 1400, 2000, 2800]                        # Kelvin, cold-ish to white-hot
    bars = []
    for i, T in enumerate(temps):
        x = -2.0 + i * 1.0
        g = box(0.28, 0.9, 0.28).rounded(0.05).translate((x, -0.1, 0.0))
        bars.append((g, ML.heat("iron" if "iron" in ML.names() else "gold", T)))
    floor = (box(4.0, 0.1, 3.0).translate((0, -1.0, 0)), ML.material("matte_black"))
    objs = bars + [floor]

    class Scene:
        def eval(self, P):
            d = np.asarray(objs[0][0].eval(P), float)
            for g, _ in objs[1:]:
                d = np.minimum(d, np.asarray(g.eval(P), float))
            return d
    def material(P):
        P = np.atleast_2d(np.asarray(P, float)); n = len(P)
        dists = np.stack([np.abs(np.asarray(g.eval(P), float)) for g, _ in objs], axis=1)
        owner = np.argmin(dists, axis=1)
        alb = np.zeros((n, 3)); met = np.zeros(n); rough = np.full(n, 0.4); emis = np.zeros((n, 3))
        for i, (_, mat) in enumerate(objs):
            m = owner == i
            if not m.any():
                continue
            a_i, met_i, r_i, e_i, _ = ML.shade(mat, int(m.sum()))
            alb[m] = a_i; met[m] = met_i; rough[m] = r_i; emis[m] = e_i    # emis carries the thermal glow
        return alb, met, rough, emis

    cam = Camera(eye=(0.0, 0.7, 4.6), target=(0.0, -0.15, 0.0), fov_deg=46, aspect=WIDTH / HEIGHT)
    from holographic_raymarch import sky_dome
    sky = lambda D: sky_dome(D, sun_dir=SUN, sun_color=(0.6, 0.65, 0.8), sky_color=(0.05, 0.06, 0.09),
                             horizon=(0.06, 0.07, 0.10), ground=(0.03, 0.03, 0.04), sun_size=0.02)   # dark room
    _save_render("render_hot_metal", Scene(), cam, material, max_bounce=3)


def render_subsurface():
    """SUBSURFACE SCATTERING -- a translucent material glows where it is THIN, lit from behind. One big wax blob
    fills the frame: its surface is DISPLACED (bumpy), so thickness varies bump-to-valley across the whole shape,
    and the thin ridges light up while the thick body stays dark. The path tracer measures how much solid the
    light crosses inside the object toward the sun (holographic_raymarch.subsurface -- Beer-Lambert on the SDF
    interior) and adds that as a coloured glow. Built on the canonical scene DOCUMENT: the wax material carries a
    subsurface strength that drives the effect -- no per-scene shading code."""
    from holographic_scene_doc import Scene
    from holographic_scene_render import render_scene_document
    from holographic_sdf import sphere, box
    from holographic_render import Camera
    import holographic_matlib as ML

    doc = Scene(seed=0)
    doc.add(name="floor", geometry=box(4.0, 0.1, 3.0).translate((0, -1.15, 0)), material="matte_black")
    # ONE big ORANGE blob (honey -- an amber translucent), surface displaced so thickness varies all over. Orange
    # makes the effect legible: the thick body reads dark brown-orange, the thin bumps blaze bright orange.
    _honey = ML.material("honey")
    _honey.sss = 1.0                                             # full subsurface strength for the demo
    doc.add(name="blob", geometry=sphere(1.0).displace(0.14, 5.0), material=_honey)

    cam = Camera(eye=(0.0, 0.15, 3.15), target=(0.0, 0.0, 0.0), fov_deg=42, aspect=WIDTH / HEIGHT)
    # a SIDE-BACK light: behind the blob but well off the camera axis, so the camera is NOT staring into the sun
    # (a dead-on backlight blows the background white and the blob reads as a silhouette -- measured, kept). The
    # SSS march points from each surface point toward that light; thin bumps on the lit side glow orange, the
    # thick middle absorbs toward black, and the background stays dark.
    back = (0.72, 0.30, -0.62)                                   # from the surface toward the light (well off-frame)
    from holographic_raymarch import sky_dome
    sky = lambda D: sky_dome(D, sun_dir=(0.72, 0.30, -0.62), sun_color=(0.03, 0.028, 0.026), sky_color=(0.002, 0.0025, 0.005),
                             horizon=(0.003, 0.003, 0.006), ground=(0.0015, 0.0015, 0.002), sun_size=0.03)
    hdr = render_scene_document(doc, cam, width=WIDTH, height=HEIGHT, quality=QUALITY, max_bounce=3, seed=0,
                                sky=sky, sss_dir=back, sss_depth=1.3, sss_sigma=2.8)
    # march deep enough to tell thick from thin (1.3 of a ~2-unit diameter), absorb softly enough that the glow
    # PENETRATES: thin bumps transmit ~0.5 (bright orange), the thick middle ~0.03 (near black).
    # FIXED exposure, not the gallery's auto-metering: auto-exposure lifts a deliberately dark room to mid-grey,
    # washing out exactly the dark-body-vs-glowing-thin contrast this demo exists to show. Keep the room black.
    from holographic_gbuffer import aces_tonemap
    plt.imsave(f"{OUT}/render_subsurface.png", aces_tonemap(np.clip(hdr, 0, None), exposure=2.6, auto=False))
    print("  render_subsurface.png")


def render_smoke_fire():
    """SMOKE & FIRE -- the Stable-Fluids solver composed with the volumetric renderer, side by side. Both pieces
    existed (the FFT smoke solver and the volume ray-marcher) but nothing ever pointed one at the other for a
    picture (backlog I1). Here: run a small 3-D smoke sim (a rising heated plume), then RENDER its density field
    twice -- once as grey SMOKE (absorption), once as FIRE (blackbody emission from the same density) -- through
    holographic_render.volume_render. The bridge is a trilinear SAMPLER that turns the sim's voxel grid into the
    callable density field the renderer marches."""
    from holographic_fluid import StableFluid
    from holographic_render import volume_render, Camera

    # --- simulate: a small 3-D plume rising from a hot patch at the bottom ---
    Nx, Ny, Nz = 40, 56, 40                                    # (up = axis 1); modest so the demo stays quick
    fluid = StableFluid((Nx, Ny, Nz), dt=0.1, buoyancy_alpha=0.20, vorticity=3.0, dissipation=0.02)
    src = (slice(Nx // 2 - 4, Nx // 2 + 4), slice(2, 7), slice(Nz // 2 - 4, Nz // 2 + 4))   # the emitter box
    for step in range(46):
        if step < 30:                                          # keep feeding smoke + heat for the first frames
            fluid.add_source(region=src, density=1.0, temperature=1.0)
        fluid.step()
    grid = np.asarray(fluid.to_numpy("density"))               # (Nx,Ny,Nz) voxel density
    grid *= 1.6 / (grid.max() + 1e-6)                          # normalise so sigma reads consistently
    temp = np.asarray(fluid.to_numpy("temperature"))           # heat field: hot at the base, cooling as it rises
    # the FIRE field is where it is BOTH dense and HOT (the burning core), so the blackbody ramp glows at the
    # base and fades to smoke above -- density alone would try to "burn" the cold smoke at the top.
    hot = grid * (temp / (temp.max() + 1e-6))
    hot *= 0.95 / (np.percentile(hot[hot > 0], 96) + 1e-6)     # core sits ~0.95 -> the ramp's orange/yellow glow

    # --- bridge: a trilinear sampler exposing a voxel grid as volume_render's callable density field ---
    lo = np.array([-1.0, -1.0, -1.0]); hi = np.array([1.0, 1.4, 1.0])   # world box the grid fills (taller in Y)
    def _sampler(vox):
        def field(P):
            """value at world points P (M,3): map world->grid coords and trilinearly interpolate `vox`."""
            P = np.asarray(P, float)
            g = (P - lo) / (hi - lo) * (np.array(vox.shape) - 1)       # continuous grid coordinates
            g = np.clip(g, 0, np.array(vox.shape) - 1.001)
            i = np.floor(g).astype(int); f = g - i
            out = np.zeros(len(P))
            for dx in (0, 1):                                          # 8-corner trilinear blend
                for dy in (0, 1):
                    for dz in (0, 1):
                        w = (np.where(dx, f[:, 0], 1 - f[:, 0]) * np.where(dy, f[:, 1], 1 - f[:, 1]) *
                             np.where(dz, f[:, 2], 1 - f[:, 2]))
                        out += w * vox[np.clip(i[:, 0] + dx, 0, vox.shape[0] - 1),
                                       np.clip(i[:, 1] + dy, 0, vox.shape[1] - 1),
                                       np.clip(i[:, 2] + dz, 0, vox.shape[2] - 1)]
            return out
        return field

    cam = Camera(eye=(1.7, 0.6, 2.6), target=(0.0, 0.15, 0.0), fov_deg=45.0, aspect=WIDTH / HEIGHT)
    bounds = (lo, hi)
    smoke, _ = volume_render(_sampler(grid), cam, bounds, WIDTH, HEIGHT, steps=110, mode="smoke",
                             sigma=14.0, background=(0.10, 0.11, 0.14))
    fire, _ = volume_render(_sampler(hot), cam, bounds, WIDTH, HEIGHT, steps=110, mode="fire",
                            sigma=22.0, emission_color=(1.0, 0.55, 0.2), background=(0.02, 0.02, 0.04))
    combo = np.concatenate([smoke, fire], axis=1)              # smoke | fire, one image
    plt.imsave(f"{OUT}/render_smoke_fire.png", _tonemap(np.clip(combo, 0, None)))
    print("  render_smoke_fire.png")


def render_sparks_over_scene():
    """PARTICLES AS A PIPELINE STAGE (backlog H6) -- a cloud of glowing ember sparks composited over a solid scene
    by the render PIPELINE. The particle system simulates points; this makes those points a rendered LAYER. Here a
    swarm of warm sparks (positions from a little upward-drift sim) floats around two spheres on a dark floor; the
    scene is handed to the pipeline as a RenderSpec whose `particles` field is the point cloud, and the pipeline's
    particle stage projects and splats them, over-compositing onto the surface render. Nearer sparks correctly
    cover farther ones, and a depth fade dims the ones drifting away."""
    from holographic_render import Camera
    from holographic_pipeline import build_pipeline, PipelineConfig, RenderSpec
    from holographic_integrate import ParticleSim
    import holographic_matlib as ML

    # --- simulate a drifting ember swarm: buoyant upward pull + a little swirl, a few steps ---
    rng = np.random.default_rng(3)
    n = 90                                                                  # fewer sparks so they read as distinct dots
    pos = rng.uniform([-1.4, -0.4, -0.6], [1.4, 0.3, 0.7], (n, 3))          # spread wide so they don't clump on screen
    vel = np.zeros((n, 3))

    def embers(pos, vel):
        # gentle rise (embers float up) + a small position-dependent swirl so they don't move as a rigid block
        acc = np.zeros_like(pos)
        acc[:, 1] += 0.6                                                    # buoyant lift (world +Y is up)
        acc[:, 0] += -0.4 * pos[:, 2]                                       # swirl: x pushed by z
        acc[:, 2] += 0.4 * pos[:, 0]                                        # swirl: z pushed by x
        return acc
    sim = ParticleSim(pos, vel, embers, integrator="symplectic")
    for _ in range(26):                                                    # a few more steps: the swarm rises above the spheres
        sim.advance(0.06)
    pts = sim.pos
    # warm ember colours: a spread from deep orange to yellow, per particle
    heat = rng.uniform(0.0, 1.0, n)
    cols = np.stack([0.9 + 0.1 * heat, 0.35 + 0.5 * heat, 0.05 + 0.2 * heat], axis=1)

    # --- the solid scene: two spheres on a dark floor ---
    _metal, _red = ML.material("steel"), ML.material("plastic_red")
    centers = np.array([[-0.6, -0.2, 0.1], [0.6, -0.2, -0.1]], float); radii = np.array([0.5, 0.55])
    class Scene:
        def eval(self, P):
            d = np.min(np.linalg.norm(P[..., None, :] - centers, axis=-1) - radii, axis=-1)
            return np.minimum(d, P[..., 1] + 0.75)
    def material(P):
        n = len(P); alb = np.tile([.2, .2, .22], (n, 1)).astype(float)
        met = np.zeros(n); rough = np.full(n, .5); emis = np.zeros((n, 3))
        left = (P[:, 0] < 0) & (P[:, 1] > -0.7)
        right = (P[:, 0] >= 0) & (P[:, 1] > -0.7)
        alb[left] = _red.base_color[:3]; rough[left] = _red.roughness
        alb[right] = _metal.base_color[:3]; met[right] = _metal.metallic; rough[right] = _metal.roughness
        return alb, met, rough, emis

    cam = Camera(eye=(0.0, 0.4, 3.4), target=(0.0, 0.0, 0.0), fov_deg=46, aspect=WIDTH / HEIGHT)
    # a dim dusk sky so the sparks are the bright thing in the frame
    from holographic_raymarch import sky_dome
    sky = lambda D: sky_dome(D, sun_dir=SUN, sun_color=(0.5, 0.5, 0.6), sky_color=(0.05, 0.06, 0.10),
                             horizon=(0.08, 0.07, 0.09), ground=(0.03, 0.03, 0.04), sun_size=0.02)
    spec = RenderSpec(scene=Scene(), camera=cam, material=material, sky=sky, width=WIDTH, height=HEIGHT,
                      quality=QUALITY, max_bounce=3,
                      particles={"points": pts, "colors": cols, "radius_px": 1.6, "intensity": 1.0,
                                 "depth_fade": (2.6, 4.6)})                 # dim sparks drifting to the back
    cfg = PipelineConfig(denoise="svgf", dirty_only=False, adaptive_samples=False, particles=True)
    hdr = build_pipeline(cfg).run(scene=spec, seed=0).image
    plt.imsave(f"{OUT}/render_sparks_over_scene.png", _tonemap(np.clip(hdr, 0, None)))
    print("  render_sparks_over_scene.png")


def render_smoke_over_scene():
    """VOLUME AS A PIPELINE STAGE (backlog H5) -- a smoke plume composited over a SOLID scene, done by the render
    PIPELINE, not hand-composited in the demo. A little 3-D smoke sim rises behind two spheres on a floor; the
    scene is handed to the pipeline as a RenderSpec whose `volume` field is the sim, and the pipeline's volume
    stage renders the smoke and over-composites it onto the surface render. This is the difference between 'the
    volume renderer exists' and 'a scene with a volume actually renders as one frame'."""
    from holographic_fluid import StableFluid
    from holographic_render import Camera
    from holographic_pipeline import build_pipeline, PipelineConfig, RenderSpec
    import holographic_matlib as ML

    # --- a rising smoke plume, using the same recipe as render_smoke_fire (which produces a clean column). The
    # solver's UP axis is axis 0 by default; we keep that and map the sim's up-axis (0) to WORLD Y in the sampler.
    Nx, Ny, Nz = 40, 56, 40
    fluid = StableFluid((Nx, Ny, Nz), dt=0.1, buoyancy_alpha=0.20, vorticity=3.0, dissipation=0.02)
    src = (slice(Nx // 2 - 4, Nx // 2 + 4), slice(2, 7), slice(Nz // 2 - 4, Nz // 2 + 4))   # emitter at the "bottom"
    for step in range(46):
        if step < 30:
            fluid.add_source(region=src, density=1.0, temperature=1.0)
        fluid.step()
    grid = np.asarray(fluid.to_numpy("density"))
    # sim axes are (up, a, b); reorder to (a, up, b) so grid axis 1 = world up, matching the world box below.
    grid = np.transpose(grid, (1, 0, 2)); grid *= 2.4 / (grid.max() + 1e-6)
    lo = np.array([-0.6, -0.55, -0.55]); hi = np.array([0.6, 1.75, 0.55])   # the column, rising behind the spheres

    def smoke_field(P):
        P = np.asarray(P, float)
        g = (P - lo) / (hi - lo) * (np.array(grid.shape) - 1)
        g = np.clip(g, 0, np.array(grid.shape) - 1.001)
        i = np.floor(g).astype(int); f = g - i
        out = np.zeros(len(P))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (np.where(dx, f[:, 0], 1 - f[:, 0]) * np.where(dy, f[:, 1], 1 - f[:, 1]) *
                         np.where(dz, f[:, 2], 1 - f[:, 2]))
                    out += w * grid[np.clip(i[:, 0] + dx, 0, grid.shape[0] - 1),
                                    np.clip(i[:, 1] + dy, 0, grid.shape[1] - 1),
                                    np.clip(i[:, 2] + dz, 0, grid.shape[2] - 1)]
        return out

    # --- the solid scene: two spheres on a floor, library materials ---
    _red, _metal = ML.material("plastic_red"), ML.material("steel")
    _pale, _dark = ML.material("matte_white"), ML.material("matte_black")
    centers = np.array([[-0.65, -0.2, 0.2], [0.6, -0.15, -0.1]], float); radii = np.array([0.55, 0.6])
    class Scene:
        def eval(self, P):
            d = np.min(np.linalg.norm(P[..., None, :] - centers, axis=-1) - radii, axis=-1)
            return np.minimum(d, P[..., 1] + 0.8)
    def _put(mask, mat, alb, met, rough):
        alb[mask] = mat.base_color[:3]; met[mask] = mat.metallic; rough[mask] = mat.roughness
    def material(P):
        n = len(P); alb = np.tile([.8, .8, .8], (n, 1)).astype(float)
        met = np.zeros(n); rough = np.full(n, .6); emis = np.zeros((n, 3))
        g = P[:, 1] < -0.75; chk = ((np.floor(P[:, 0] * 1.4) + np.floor(P[:, 2] * 1.4)).astype(int) % 2 == 0)
        _put(g & chk, _pale, alb, met, rough); _put(g & ~chk, _dark, alb, met, rough)
        _put((P[:, 0] < 0) & ~g, _red, alb, met, rough); _put((P[:, 0] >= 0) & ~g, _metal, alb, met, rough)
        return alb, met, rough, emis

    cam = Camera(eye=(0.0, 0.5, 3.6), target=(0.0, 0.1, 0.0), fov_deg=45, aspect=WIDTH / HEIGHT)
    sky = lambda D: _sky(D)                                                  # the shared sun-sky HDRI
    spec = RenderSpec(scene=Scene(), camera=cam, material=material, sky=sky, width=WIDTH, height=HEIGHT,
                      quality=QUALITY, max_bounce=3,
                      volume={"field": smoke_field, "bounds": (lo, hi), "mode": "smoke", "sigma": 11.0,
                              "steps": 110, "background": (0.0, 0.0, 0.0)})
    # run it through the PIPELINE with the volume stage on -- the smoke is composited by the stage, not by us
    cfg = PipelineConfig(denoise="svgf", dirty_only=False, adaptive_samples=False, volume=True)
    hdr = build_pipeline(cfg).run(scene=spec, seed=0).image
    plt.imsave(f"{OUT}/render_smoke_over_scene.png", _tonemap(np.clip(hdr, 0, None)))
    print("  render_smoke_over_scene.png")


def render_fur_over_scene():
    """HAIR AS A PIPELINE STAGE (backlog H4) -- fur composited over a PATH-TRACED body by the render pipeline, not
    by hair's own standalone renderer. A groomed coat is grown on a creature body; the body is path-traced with a
    library skin material, then the pipeline's hair stage renders the strands WITH a coverage alpha and
    over-composites them onto that render. So the fur is lit and sits on a properly shaded, shadowed body -- the
    difference between 'the hair renderer exists' and 'hair is a layer in the frame'."""
    from holographic_render import Camera
    from holographic_pipeline import build_pipeline, PipelineConfig, RenderSpec
    from holographic_groom import groom
    from holographic_sdf import sphere
    import holographic_matlib as ML

    # --- a simple creature body (a big sphere + a head), path-traced with a warm skin material ---
    body = sphere(0.85).smooth_union(sphere(0.5).translate((0.0, 0.9, 0.1)), k=0.2)
    bnds = ((-1.4, -1.4, -1.4), (1.4, 1.9, 1.4))
    coat = groom(body.eval, 9000, bnds, length=0.42, n_pts=7, curl=0.25, seed=0, length_jitter=0.22)
    under = groom(body.eval, 4500, bnds, length=0.22, n_pts=5, curl=0.12, seed=1, length_jitter=0.18)
    strands = coat + under

    _skin = ML.material("skin_light" if "skin_light" in ML.names() else "matte_white")
    def material(P):
        n = len(P); return (np.tile(_skin.base_color[:3], (n, 1)).astype(float), np.zeros(n),
                            np.full(n, _skin.roughness), np.zeros((n, 3)))

    cam = Camera(eye=(0.0, 0.6, 3.6), target=(0.0, 0.3, 0.0), fov_deg=46, aspect=WIDTH / HEIGHT)
    sky = lambda D: _sky(D)
    # physical fiber look from a fur material (Marschner beta_r / alpha_r) -- the look comes from the material
    _fur = ML.material("fur_ginger" if "fur_ginger" in ML.names() else "matte_white")
    fp = ML.fiber_params(_fur) if hasattr(ML, "fiber_params") else {}
    spec = RenderSpec(scene=body, camera=cam, material=material, sky=sky, width=WIDTH, height=HEIGHT,
                      quality=QUALITY, max_bounce=3,
                      hair={"strands": strands, "shader": "marschner", "hair_color": (0.62, 0.34, 0.14),
                            "light_dir": (0.35, 0.7, 0.55), "roughness": fp.get("roughness"),
                            "tilt_deg": fp.get("tilt_deg"), "smooth_levels": 2})
    cfg = PipelineConfig(denoise="svgf", dirty_only=False, adaptive_samples=False, hair=True)
    hdr = build_pipeline(cfg).run(scene=spec, seed=0).image
    plt.imsave(f"{OUT}/render_fur_over_scene.png", _tonemap(np.clip(hdr, 0, None)))
    print("  render_fur_over_scene.png")


def render_fur():
    """FUR shaded by a physical FIBER material (holographic_matlib 'fur_ginger' -> a Marschner strand BSDF), and
    properly GROOMED. Three fixes over the standing-on-end version:
      * COMB -- groom grows each strand straight out along the surface normal, so raw fur stands on end. We bend
        each strand from its normal toward a world FLOW direction (down and back), so it lies along the body and
        flows, like a brushed coat, instead of sticking straight out.
      * ANTI-ALIAS -- the strand rasteriser draws 1-px lines, which alias badly. We render at 2x and box-downsample
        (supersampling), so the coat reads as smooth fur, not pixel noise.
      * LIGHTING -- a KEY light from the front-upper reveals the groomed form; a softer warm RIM from behind makes
        the translucent fur edges glow. (The old single back-light gave the blotchy orange-and-blown-white look.)"""
    import holographic_matlib as ML
    from holographic_groom import groom, Strand
    from holographic_hairshade import render_hair
    from holographic_render import Camera
    from holographic_sdf import sphere

    def _nrm(v):
        v = np.asarray(v, float); return v / (np.linalg.norm(v) + 1e-12)

    fur = ML.material("fur_ginger"); fp = ML.fiber_params(fur)                    # physical fiber material
    body = sphere(0.95).smooth_union(sphere(0.60).translate((0.0, 0.98, 0.10)), k=0.22)   # body + head
    bnds = ([-1.7, -1.7, -1.7], [1.7, 2.0, 1.7])
    # A DENSE coat so the body doesn't show through: a long top coat + a short undercoat that fills the base.
    # (More strands is the honest fix for coverage -- the render cost is the strand count, so this is the lever.)
    coat = groom(body.eval, 16000, bnds, length=0.55, n_pts=8, curl=0.28, seed=0, length_jitter=0.25)
    under = groom(body.eval, 8000, bnds, length=0.30, n_pts=6, curl=0.15, seed=1, length_jitter=0.20)
    strands = coat + under

    def _comb(strands, flow=(0.10, -1.0, -0.30), lift=0.16, bend=1.45, droop=0.32):
        """Reshape each strand so it curves from its outward normal (at the root) toward `flow` (at the tip),
        projected onto the surface -- i.e. comb the fur to lie down and flow. `lift` keeps a little loft so it
        doesn't clip into the body; `bend` is how far the tip lays over; `droop` adds a gravity SAG that grows
        toward the tip (t^2), so the coat RELAXES and brushes down instead of standing off the surface."""
        f = _nrm(flow); down = np.array([0.0, -1.0, 0.0]); out = []
        for s in strands:
            n = s.root_normal if s.root_normal is not None else _nrm(s.points[1] - s.points[0])
            ft = f - np.dot(f, n) * n                                             # flow in the surface tangent plane
            ft = ft / np.linalg.norm(ft) if np.linalg.norm(ft) > 1e-6 else _nrm(np.cross(n, [0.0, 1.0, 0.0]) + 1e-6)
            tip = _nrm(ft + lift * n)                                             # tip dir: mostly tangential, a little loft
            seglen = float(np.linalg.norm(s.points[1] - s.points[0]))            # uniform segment length from the groom
            pts = [s.points[0].copy()]
            for i in range(1, len(s.points)):
                t = i / (len(s.points) - 1)
                w = min(bend * t, 1.0)                                            # lay over more toward the tip (capped)
                d = _nrm(_nrm((1.0 - w) * n + w * tip) + droop * (t * t) * down)  # + gravity sag that grows to the tip
                pts.append(pts[-1] + d * seglen)
            out.append(Strand(np.array(pts), root_normal=n, width=s.width, attrs=s.attrs))
        return out
    strands = _comb(strands)

    SS = 2                                                                        # supersample factor (2x -> downsample)
    w, h = WIDTH * SS, HEIGHT * SS
    cam = Camera(eye=(0.0, 0.5, 3.4), target=(0.0, 0.30, 0.0), fov_deg=46.0, aspect=WIDTH / HEIGHT)
    hc = fp["hair_color"]
    # KEY (front-upper 3/4) reveals the groomed form; RIM (behind) gives the warm translucent edge glow.
    key = render_hair(strands, cam, light_dir=(0.5, 0.7, 0.4), width=w, height=h, shader="marschner",
                      hair_color=hc, roughness=fp["roughness"], tilt_deg=fp["tilt_deg"], reflect=0.06,
                      background=(0.0, 0.0, 0.0), smooth_levels=2)
    rim = render_hair(strands, cam, light_dir=(-0.35, 0.45, -0.9), width=w, height=h, shader="marschner",
                      hair_color=hc, roughness=fp["roughness"], tilt_deg=fp["tilt_deg"], reflect=0.06,
                      background=(0.0, 0.0, 0.0), smooth_levels=2)
    lit = key + 0.6 * rim                                                         # combine the two lights
    ys = np.linspace(0, 1, h)[:, None, None]                                      # soft background gradient
    bg = (0.78 + 0.16 * ys) * np.array([0.93, 0.94, 0.98])
    mask = lit.sum(2) > 0.008                                                     # where a strand actually drew
    hi = np.where(mask[..., None], lit, bg)
    img = hi.reshape(HEIGHT, SS, WIDTH, SS, 3).mean(axis=(1, 3))                  # box downsample = anti-aliasing
    plt.imsave(f"{OUT}/render_fur.png", _tonemap(np.clip(img, 0.0, None)))
    print("  render_fur.png")

def render_ocean():
    """OCEAN IN A BOX with BUOYANCY -- rendered with a dedicated, readable WATER shader. (The path tracer's glass
    is a CLOSED-object model; an open water surface sitting over a floor doesn't fit it, which is why the old
    version went black.) The physics is the standard water model:
      * where a camera ray meets the rippled surface, FRESNEL splits it into a reflection of the sky and a
        refraction down INTO the water (Snell's law);
      * the refracted ray is marched to the sandy floor (or the cube's submerged side), and the colour it brings
        back is faded by BEER-LAMBERT absorption over the underwater distance -- water absorbs red first, so deep
        water reads blue and dark. THAT is the volumetric depth / 'fog';
      * POOL CAUSTICS (light focused through the wavy surface) brighten the sand;
      * a wooden cube floats at its Archimedes waterline (submerged fraction = density ratio ~0.6);
      * the sun glints off the ripples.
    The camera looks DOWN at the water, so the sky is only a thin strip -- no blown-out sky."""
    from holographic_raymarch import sky_dome
    from holographic_globalillum import caustics as _caustics
    W, H = WIDTH, HEIGHT
    water_y, floor_y, box_half = 0.0, -0.85, 2.6
    cube_h, wood_density = 0.40, 0.6
    cube_c = np.array([0.35, water_y + cube_h - 2 * cube_h * wood_density, -0.10])   # centre at the Archimedes line
    sun = np.array(SUN, float); sun /= np.linalg.norm(sun)

    def sky(D):                                                  # a calm sky (reflections + the thin strip up top)
        return sky_dome(D, sun_dir=SUN, sun_color=(4.0, 3.7, 3.2), sky_color=(0.30, 0.48, 0.78),
                        horizon=(0.72, 0.80, 0.88), ground=(0.25, 0.30, 0.32), sun_size=0.02)

    # rippled surface: a few summed sinusoids, and its ANALYTIC normal (the derivative of the height field)
    def wave(x, z):
        return 0.045 * np.sin(2.2 * x + 0.6 * z) + 0.030 * np.sin(3.1 * z - 1.2 * x) + 0.020 * np.sin(5.0 * x + 3.0 * z + 1.0)
    def wave_normal(x, z):
        dhdx = 0.045 * 2.2 * np.cos(2.2 * x + 0.6 * z) - 0.030 * 1.2 * np.cos(3.1 * z - 1.2 * x) + 0.020 * 5.0 * np.cos(5.0 * x + 3.0 * z + 1.0)
        dhdz = 0.045 * 0.6 * np.cos(2.2 * x + 0.6 * z) + 0.030 * 3.1 * np.cos(3.1 * z - 1.2 * x) + 0.020 * 3.0 * np.cos(5.0 * x + 3.0 * z + 1.0)
        N = np.stack([-dhdx, np.ones_like(dhdx), -dhdz], -1)
        return N / np.linalg.norm(N, axis=-1, keepdims=True)

    def sand(x, z):                                             # a sandy checker floor, seen through the water
        chk = ((np.floor(x * 1.3) + np.floor(z * 1.3)).astype(int) % 2 == 0)
        return np.where(chk[..., None], np.array([0.82, 0.73, 0.52]), np.array([0.60, 0.50, 0.34]))

    # a look-AT camera angled down at the water (so the frame is mostly water, not sky)
    eye = np.array([0.0, 1.55, 2.95])
    fwd = np.array([0.0, -0.46, -1.0]); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0])); right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)
    ys, xs = np.mgrid[0:H, 0:W]; fov = 1.15
    px = (xs / (W - 1) - 0.5) * fov; py = -(ys / (H - 1) - 0.5) * fov * (H / W)
    D = fwd + px[..., None] * right + py[..., None] * upv
    D = (D / np.linalg.norm(D, axis=-1, keepdims=True)).reshape(-1, 3)
    O = np.broadcast_to(eye, D.shape).astype(float)
    npix = D.shape[0]

    # pool caustics: focus sunlight through the wavy surface onto the sand (holographic_globalillum.caustics)
    class _WaterSDF:
        def eval(self, P):
            return (P[..., 1] - (water_y + wave(P[..., 0], P[..., 2]))) * 0.6
    cmap = _caustics(_WaterSDF(), light_dir=tuple(-sun), receiver_y=floor_y, extent=box_half, res=220, ior=1.33, n_side=380)
    def caustic_at(x, z):
        xi = np.clip(((x + box_half) / (2 * box_half) * (cmap.shape[0] - 1)).astype(int), 0, cmap.shape[0] - 1)
        zi = np.clip(((z + box_half) / (2 * box_half) * (cmap.shape[0] - 1)).astype(int), 0, cmap.shape[0] - 1)
        return np.clip(cmap[zi, xi] - 1.0, 0.0, None)

    def box_hit(O, D):                                          # ray vs the wooden cube (slab method) -> hit, t, normal
        inv = 1.0 / np.where(np.abs(D) < 1e-9, 1e-9, D)
        lo = (cube_c - cube_h - O) * inv; hi = (cube_c + cube_h - O) * inv
        tmin = np.max(np.minimum(lo, hi), axis=-1); tmax = np.min(np.maximum(lo, hi), axis=-1)
        t = np.where(tmin > 0, tmin, tmax)
        P = O + D * t[..., None]
        d = np.abs(P - cube_c); ax = np.argmax(d, axis=-1)
        N = np.zeros_like(P)
        for a in range(3):
            m = ax == a; N[m, a] = np.sign(P[m, a] - cube_c[a])
        return (tmax >= np.maximum(tmin, 0)) & (t > 0), t, N

    # primary hits: the water plane (mean y=water_y, ripples via the normal), and the cube
    tw = (water_y - O[:, 1]) / np.where(np.abs(D[:, 1]) < 1e-9, -1e-9, D[:, 1])
    Pw = O + D * tw[:, None]
    in_tank = (np.abs(Pw[:, 0]) < box_half) & (np.abs(Pw[:, 2]) < box_half) & (tw > 0) & (D[:, 1] < 0)
    chit, tc, cN = box_hit(O, D); Pc = O + D * tc[:, None]
    cube_above = chit & (Pc[:, 1] > water_y + 0.002)           # the cube's part standing above the surface

    col = sky(D).copy()                                        # sky is the default background

    # direct hit on the cube ABOVE water (simple sun-lit wood)
    is_cube = cube_above & ((~in_tank) | (tc < tw))
    if is_cube.any():
        diff = np.clip(cN[is_cube] @ sun, 0, 1)
        col[is_cube] = np.array([0.55, 0.36, 0.20]) * (0.30 + 0.70 * diff)[:, None]

    # WATER where the surface is the nearest thing
    is_water = in_tank & ~is_cube & (tw < np.where(chit, tc, np.inf))
    if is_water.any():
        Ow = Pw[is_water]; Dw = D[is_water]
        N = wave_normal(Ow[:, 0], Ow[:, 2])
        cosi = np.clip(-np.sum(Dw * N, axis=-1), 0.0, 1.0)
        F = 0.02 + 0.98 * (1.0 - cosi) ** 5                    # Schlick Fresnel (water F0 ~ 0.02)
        # reflection of the sky
        R = Dw - 2.0 * np.sum(Dw * N, axis=-1)[:, None] * N
        refl = sky(R)
        # refraction into the water (Snell, air->water eta=1/1.33)
        eta = 1.0 / 1.33
        k = 1.0 - eta * eta * (1.0 - cosi * cosi)
        T = eta * Dw + (eta * cosi - np.sqrt(np.clip(k, 0.0, None)))[:, None] * N
        T = T / np.linalg.norm(T, axis=-1, keepdims=True)
        # march the refracted ray to the floor plane, or the cube's submerged side if that's nearer
        tf = (floor_y - Ow[:, 1]) / np.where(np.abs(T[:, 1]) < 1e-9, -1e-9, T[:, 1])
        Pf = Ow + T * tf[:, None]
        chit2, tc2, _ = box_hit(Ow, T)
        floor_first = (~chit2) | (tf < tc2)
        hitP = np.where(floor_first[:, None], Pf, Ow + T * tc2[:, None])
        dist = np.linalg.norm(hitP - Ow, axis=-1)              # how far the light travelled underwater
        base = np.where(floor_first[:, None],
                        sand(hitP[:, 0], hitP[:, 2]) + 0.9 * caustic_at(hitP[:, 0], hitP[:, 2])[:, None] * np.array([1.0, 0.98, 0.9]),
                        np.array([0.55, 0.36, 0.20]))           # sand (+caustics) or the cube underside (wood)
        sigma = np.array([0.90, 0.30, 0.16]) * 2.4             # Beer-Lambert extinction: red absorbed most
        Tab = np.exp(-sigma * dist[:, None])                   # fraction of light surviving the water depth
        deep = np.array([0.015, 0.10, 0.15])                   # colour the water tends toward with depth (the 'fog')
        refr = base * Tab + deep * (1.0 - Tab)
        glint = np.clip(np.sum(R * sun, axis=-1), 0.0, 1.0) ** 120 * 2.5   # sharp sun sparkle on the ripples
        col[is_water] = F[:, None] * refl + (1.0 - F)[:, None] * refr + glint[:, None] * np.array([1.0, 0.97, 0.85])

    img = col.reshape(H, W, 3)
    plt.imsave(f"{OUT}/render_ocean.png", _tonemap(np.clip(img, 0.0, None)))
    print("  render_ocean.png")

def render_patterns():
    """Procedural pattern fields (fBm, value noise, checker, dots) -- solid 3-D textures, no UV unwrap."""
    import holographic_pattern as P
    res = 220; xs = np.linspace(-2, 2, res); X, Y = np.meshgrid(xs, xs)
    grid = np.stack([X, Y, np.zeros_like(X)], -1).reshape(-1, 3)          # a z=0 slice of 3-D world space
    def sample(f):
        v = np.asarray(f(grid)); return v.reshape(res, res) if v.ndim == 1 else v.reshape(res, res, -1)[..., 0]
    items = [("fBm noise", P.fbm(scale=2.5, octaves=5, seed=1)), ("value noise", P.value_noise(scale=5.0, seed=2)),
             ("checker", P.checker(scale=3.0)), ("dots", P.dots(scale=4.0, radius=0.35))]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    for a, (name, f) in zip(ax, items):
        a.imshow(sample(f), cmap="magma"); a.set_title(name); a.axis("off")
    fig.suptitle("Procedural pattern fields — solid 3-D textures, no UV unwrap", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/patterns.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print("  patterns.png")

def render_reaction_diffusion():
    """A vector-valued reaction-diffusion cellular automaton -- Turing patterns living in hypervector space."""
    from holographic_automaton import HyperCA
    ca = HyperCA(size=140, dim=32, seed=3); [ca.step() for _ in range(24)]
    g = ca.grid                                                          # (size, size, dim)
    B = np.random.default_rng(0).standard_normal((g.shape[-1], 3))       # project the 32-D state down to RGB
    rgb = g @ B; rgb = (rgb - rgb.min()) / (np.ptp(rgb) + 1e-9)
    plt.imsave(f"{OUT}/reaction_diffusion.png", rgb); print("  reaction_diffusion.png")


# =========================================================================== DATA-DRIVEN CHARTS
def chart_core_ops():
    """Cost of the two core operations vs hypervector dimension -- the algebra is cheap and scales gently."""
    from holographic_ai import bind, bundle
    dims = [512, 1024, 2048, 4096, 8192, 16384]; tb = []; tu = []
    for D in dims:
        a = np.random.default_rng(0).standard_normal(D); b = np.random.default_rng(1).standard_normal(D)
        vs = [np.random.default_rng(i).standard_normal(D) for i in range(16)]
        t = time.time(); [bind(a, b) for _ in range(200)]; tb.append((time.time() - t) / 200 * 1e6)
        t = time.time(); [bundle(vs) for _ in range(200)]; tu.append((time.time() - t) / 200 * 1e6)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(dims, tb, "o-", label="bind (FFT circular convolution)")
    ax.plot(dims, tu, "s-", label="bundle (16-way superposition)")
    ax.set_xscale("log", base=2); ax.set_xlabel("hypervector dimension"); ax.set_ylabel("microseconds / op")
    ax.set_title("Core op cost vs dimension (NumPy, single thread)"); ax.legend(); ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/perf_core_ops.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print("  perf_core_ops.png")

def chart_compression():
    """MEASURED: bytes/record for the engine's low-rank code vs SQLite, as the table grows.
    The VSA store's shared basis amortises, so per-record cost FALLS with N and crosses under SQLite."""
    from holographic_query import from_rows
    from holographic_ratedistortion import geometry_preserving_code, pack_code
    rng = np.random.default_rng(0)
    def rows(n, distinct):
        cats = [f"cat_{i}" for i in range(distinct)]; regs = [f"reg_{i}" for i in range(max(2, distinct // 2))]
        st = ["open", "closed", "pending", "void"]; ti = ["gold", "silver", "bronze"]
        return ([{"region": rng.choice(regs), "category": rng.choice(cats),
                  "status": rng.choice(st), "tier": rng.choice(ti)} for _ in range(n)],
                ["region", "category", "status", "tier"])
    def sqlite_bpr(rw, co):
        fd, p = tempfile.mkstemp(suffix=".db"); os.close(fd)
        c = sqlite3.connect(p); cur = c.cursor()
        cur.execute(f"CREATE TABLE t ({', '.join(x + ' TEXT' for x in co)})")
        cur.executemany(f"INSERT INTO t VALUES ({','.join('?' * len(co))})", [tuple(r[x] for x in co) for r in rw])
        c.commit(); cur.execute("VACUUM"); c.commit(); c.close()
        s = os.path.getsize(p); os.remove(p); return s / len(rw)
    Ns = [500, 1000, 2000, 5000, 10000, 25000, 50000]; vsa = []; sql = []
    for n in Ns:
        rw, co = rows(n, 6)                                    # low-cardinality (structured) categorical data
        X = from_rows(rw, co, dim=1024, seed=0).records
        vsa.append(len(pack_code(geometry_preserving_code(X, target_cos=0.9999))) / n)
        sql.append(sqlite_bpr(rw, co))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(Ns, vsa, "o-", label="VSA store (low-rank rate-distortion code)")
    ax.plot(Ns, sql, "s--", label="SQLite (same data)")
    ax.set_xscale("log"); ax.set_xlabel("rows in the table"); ax.set_ylabel("bytes / record")
    ax.set_title("Compression vs SQL: per-record cost falls with N (basis amortises)")
    ax.legend(); ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/compression_vs_sql.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"  compression_vs_sql.png  (VSA {vsa[-1]:.1f} vs SQLite {sql[-1]:.1f} B/rec @ {Ns[-1]})")

def _kv_recall(D, K, corrupt=0.0, seed=0):
    """A key->value associative memory test: bind K key/value pairs, bundle them into ONE vector, then read
    each value back by unbinding its key and cleaning up to the nearest codebook atom. Returns the fraction
    recovered correctly. `corrupt` zeroes that fraction of the bundle's dimensions before readout."""
    from holographic_ai import bind, unbind, bundle
    rng = np.random.default_rng(seed)
    def atoms(m): 
        A = rng.standard_normal((m, D)); return A / np.linalg.norm(A, axis=1, keepdims=True)
    keys, vals = atoms(K), atoms(K)                            # K random unit keys and values
    mem = bundle([bind(keys[i], vals[i]) for i in range(K)])   # one superposed memory vector
    if corrupt > 0:                                            # knock out a fraction of the dimensions
        mask = rng.random(D) < corrupt; mem = mem.copy(); mem[mask] = 0.0
    ok = 0
    for i in range(K):
        rec = unbind(mem, keys[i])                             # noisy estimate of vals[i]
        guess = int(np.argmax(vals @ rec))                    # cleanup = nearest value atom (cosine)
        ok += (guess == i)
    return ok / K

def chart_capacity():
    """Recall accuracy vs how many pairs are stored, at three dimensions -- the honest capacity 'cliff',
    and how it moves right as you add dimensions."""
    loads = [5, 10, 20, 40, 60, 80, 120, 160, 220]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for D in (512, 1024, 2048):
        acc = [np.mean([_kv_recall(D, K, seed=s) for s in range(3)]) for K in loads]
        ax.plot(loads, acc, "o-", label=f"dim = {D}")
    ax.axhline(0.9, ls=":", color="grey"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("pairs stored in one vector"); ax.set_ylabel("recall accuracy")
    ax.set_title("Memory capacity: the cliff, and 'add dimensions to move it right'")
    ax.legend(); ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/capacity_curve.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print("  capacity_curve.png")

def chart_degradation():
    """Recall accuracy as the memory vector is progressively corrupted -- graceful decline, not a hard crash."""
    fracs = np.linspace(0, 0.9, 10)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for K in (20, 40, 80):
        acc = [np.mean([_kv_recall(1024, K, corrupt=c, seed=s) for s in range(3)]) for c in fracs]
        ax.plot(fracs * 100, acc, "o-", label=f"{K} pairs stored")
    ax.set_ylim(0, 1.02); ax.set_xlabel("% of the memory vector zeroed"); ax.set_ylabel("recall accuracy")
    ax.set_title("Graceful degradation: recall vs damage (dim = 1024)")
    ax.legend(); ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/graceful_degradation.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print("  graceful_degradation.png")


if __name__ == "__main__":
    visuals = [("spheres", render_spheres), ("glass", render_glass), ("fractal", render_fractal),
               ("identities", render_identities), ("fur", render_fur), ("ocean", render_ocean),
               ("smoke_fire", render_smoke_fire), ("smoke_over_scene", render_smoke_over_scene),
               ("sparks_over_scene", render_sparks_over_scene),
               ("fur_over_scene", render_fur_over_scene),
               ("subsurface", render_subsurface),
               ("hot_metal", render_hot_metal), ("crystal", render_crystal),
               ("iridescence", render_iridescence), ("lit_scene", render_lit_scene),
               ("light_types", render_light_types),
               ("patterns", render_patterns), ("reaction_diffusion", render_reaction_diffusion),
               ("core_ops", chart_core_ops), ("compression", chart_compression),
               ("capacity", chart_capacity), ("degradation", chart_degradation)]
    for name, fn in visuals:
        try:
            fn()
        except Exception as e:
            print(f"  [skip {name}] {e}"); traceback.print_exc()
    print("done -> ./gallery/")
