"""Gem/glass MATERIAL test scene, after Moose's references: one smooth object, soft studio key on an HDRI base,
checker floor, grey backdrop. What must read: refraction of the checker, dispersion fringes at edges, a caustic
pool with rainbow rim, broad specular highlights, and (variant 'fog') a foggy/jade-like interior via the
holographic flaw volume. Subject is deliberately SIMPLE so the optics are judged, not the geometry.
   python3 gem_test.py <material> <fog 0|1> <W> <H> <samples>"""
import os, sys, time
os.environ.setdefault("PYTHONHASHSEED", "0"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import lecore
def mind():
    m = lecore.UnifiedMind(dim=512, seed=0); m.set_file_root('.'); return m
import holographic.rendering.holographic_postfx as pf
from holographic.mesh_and_geometry.holographic_sdf import sphere, box, SDF
from holographic.rendering.holographic_holocaustic import spectral_landings, TiledHolographicCaustic, caustic_concentration
from holographic.materials_and_texture.holographic_matlib import glass_optics
m = mind()
mat, FOG, W, H, S = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
go = glass_optics(mat)

# --- subject: a smooth blob -- sphere fused with a tilted rounded box and a small sphere (like a glass "toy") -------
def T(s, t): return SDF("translate", tuple(t), [s])
body = sphere(0.62).smooth_union(T(sphere(0.34), (0.55, -0.25, 0.30)), k=0.25)
body = body.smooth_union(T(SDF("rotate", (0.0, 1.0, 0.0, 0.6), [box(0.42, 0.22, 0.30).rounded(0.10)]), (-0.15, -0.55, 0.15)), k=0.18)
class S_:
    def __call__(self, P): return np.asarray(body.eval(np.atleast_2d(np.asarray(P, float))), float)
    eval = __call__
s = S_()
gx = np.linspace(-1.6, 1.6, 97); G = np.stack(np.meshgrid(gx, gx, gx, indexing='ij'), -1).reshape(-1, 3)
d = s(G); ins = G[d < 0]; lo, hi = ins.min(0), ins.max(0); MID = (lo + hi) / 2; EXT = float((hi - lo).max()); FLOOR = float(lo[1]) - 0.005
print("bbox", np.round(lo, 2), np.round(hi, 2), flush=True)

# --- light: HDRI base + a BIG soft key (broad highlights, like a softbox) + a small hard kicker for sparkle ----------
env = m.hdri_env(np.load('hdri_plains.npy')); E0, sh = float(env.floor_irradiance.mean()), env.sun_share
KEY = np.array([-0.55, 0.75, 0.45]); KEY /= np.linalg.norm(KEY)
m.env_add_light(env, KEY, sigma=0.28, irradiance=1.2 * E0 * max(sh, 0.15))          # softbox: sigma 0.28 rad ~ 16 deg
m.env_add_light(env, (0.4, 0.5, -0.8), sigma=0.05, irradiance=0.25 * E0 * max(sh, 0.15))   # rim kicker
AIM = env.dominant_dir
MAXI = 24 if go["n_d"] > 2.0 else 10
EYE = MID + np.array([0.9, 0.55, 1.3]) / np.linalg.norm([0.9, 0.55, 1.3]) * EXT * 2.2
cam = m.camera(eye=tuple(EYE), target=tuple(MID - [0, 0.05 * EXT, 0]), fov_deg=34.0, aspect=W / H)

t0 = time.time()
bake = m.bake_glass(s, cam, W, H, material=mat, n_lams=13, floor_y=FLOOR, max_internal=MAXI, samples=S, lam_jitter=True); tb = time.time() - t0
FL = {}
if FOG:
    # a foggy interior: denser toward the core (jade-like), greenish scattering albedo, mild extinction
    fog = lambda P: np.clip(1.0 - np.linalg.norm(np.atleast_2d(P) - MID, axis=1) / 0.9, 0, 1) ** 0.7
    fb = tuple((float(lo[k] - 0.05), float(hi[k] + 0.05)) for k in range(3))
    FL = dict(flaw_volume=m.gem_flaw_volume(field=fog, bounds=fb, res=16, dim=2048), flaw_sigma=1.6, flaw_albedo=(0.55, 0.85, 0.72))
CHK = 0.45
def checker(P):
    P = np.atleast_2d(np.asarray(P, float)); q = (np.floor(P[:, 0] / CHK) + np.floor(P[:, 2] / CHK)).astype(int) % 2
    a = np.where(q[:, None] == 0, [[0.70, 0.68, 0.65]], [[0.16, 0.155, 0.15]])
    far = np.clip((np.linalg.norm(P - EYE[None, :], axis=1) - 3.0 * EXT) / (6.0 * EXT), 0, 1)[:, None]
    return a * (1 - far) + np.array([[0.43, 0.42, 0.40]]) * far
BG = (float(0.5 * (env.floor_irradiance @ [0.2126, 0.7152, 0.0722]) / np.pi),) * 3
t1 = time.time()
beauty = m.relight_glass(bake, env, floor_albedo=checker, sdf=s, sun_dir=tuple(AIM), material=mat, shadow_transmittance=0.25, background=BG, footprint_filter=True, **FL); tr = time.time() - t1

t2 = time.time()
xz, lam = spectral_landings(s, go["n_d"], go["abbe"], light_dir=tuple(-AIM), receiver_y=FLOOR, extent=EXT * 0.6, n_side=640, dispersion_scale=1.0, seed=0, aim=tuple(MID))
lo2, hi2 = np.percentile(xz, 1, axis=0), np.percentile(xz, 99, axis=0); Cc = tuple((lo2 + hi2) / 2); CWIN = float(max(hi2 - lo2) / 2 * 1.4)
tc = TiledHolographicCaustic(((Cc[0] - CWIN, Cc[0] + CWIN), (Cc[1] - CWIN, Cc[1] + CWIN)), grid=12, dim=2048, seed=0).deposit(xz, lam)
xs = np.linspace(Cc[0] - CWIN, Cc[0] + CWIN, 512); zs = np.linspace(Cc[1] - CWIN, Cc[1] + CWIN, 512)
crgb = tc.read_rgb(xs, zs, m.wavelength_cmf, normalise="mean")
px = m.caustic_pass(crgb, cam, W, H, plane_y=FLOOR, window=CWIN, center=Cc, occluder_sdf=lambda P: s(P))
conc = caustic_concentration(xz, EXT * 0.6, 640, tuple(-AIM))
E_sun = float(env.floor_irradiance.mean() * env.sun_share) / np.pi * 0.43 * 0.9 * conc
if FL:   # a foggy body scatters the light it would have focused: scale the caustic by the median interior transmittance
    _b = bake.subs[0] if hasattr(bake, "subs") else bake
    _tau = FL["flaw_volume"].segments_optical_depth(_b.segs, _b.width * _b.height)
    T_med = float(np.exp(-FL["flaw_sigma"] * np.median(_tau[_b.glass]))); E_sun *= T_med; print("   fog: caustic x %.2f" % T_med, flush=True)
final = m.composite_caustic(beauty, px, strength=E_sun, receiver_mask=bake.floor.reshape(H, W)); tcs = time.time() - t2
lum = final @ np.array([0.2126, 0.7152, 0.0722]); EV = float(np.clip(np.log2(0.22 / np.median(lum[lum > 1e-6])), -6, 6))
ch = pf.PostChain([("exposure", {"ev": EV}), ("bloom", {"threshold": 0.9, "sigma": 2.5, "intensity": 0.08}), ("aces", {}), ("gamma", {})])
out = np.clip(np.asarray(m.post_process(final, chain=ch), float), 0, 1).reshape(H // 2, 2, W // 2, 2, 3).mean(axis=(1, 3))
name = "gemtest_%s%s.png" % (mat, "_fog" if FOG else ""); m.save_render(name, out)
print("%s fog=%d: bake %.1fs relight %.1fs caustic %.1fs (conc %.1fx) | EV %+.2f | %s" % (mat, FOG, tb, tr, tcs, conc, EV, name), flush=True)
