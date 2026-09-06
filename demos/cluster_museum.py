"""Clear quartz cluster on a small rock, lit like a museum piece. Reference: a Brazilian clear-quartz cluster on
matrix (Australian Museum shop listing: "multiple clear quartz points growing in harmony from a common matrix",
"well-formed terminations that radiate from a central core", neutral background). Everything physical: quartz habit
on the real unit cell (m^r 141 deg 47'), competitive growth (each point sized by its Voronoi room), crystals exist
only outside the rock, material = the library's quartz (n 1.55, Abbe 70, faint absorption), a faint milky base as a
holographic flaw volume, full-transit spectrally tinted caustics, footprint-filtered dispersion.
   python3 cluster_museum.py <W> <H> <samples>"""
import os, sys, time
os.environ.setdefault("PYTHONHASHSEED", "0"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import lecore
def mind():
    m = lecore.UnifiedMind(dim=512, seed=0); m.set_file_root('.'); return m
import holographic.rendering.holographic_postfx as pf
from holographic.rendering.holographic_holocaustic import spectral_landings, caustic_concentration
from holographic.materials_and_texture.holographic_matlib import glass_optics
from holographic.mesh_and_geometry.holographic_sdfbake import GridSDF
m = mind()
W, H, S = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
MAT = "quartz"; go = glass_optics(MAT)

# --- the rock: a small irregular nodule (ellipsoid + baked holographic fBm), rough, dark grey -------------------------
fbm = np.load('rock_fbm48.npy'); FBM = GridSDF((fbm - fbm.mean()) / (fbm.std() + 1e-9), (-1.1,) * 3, (1.1,) * 3)
fine = np.load('rock_fbm96.npy'); FINE = GridSDF((fine - fine.mean()) / (fine.std() + 1e-9), (-1.1,) * 3, (1.1,) * 3)
RX, RY, RZ = 0.42, 0.22, 0.36
TOP = 0.14
class Rock(object):
    """A nodule with a roughly PLANAR top: crystals on a matrix piece grew on a wall, so they are roughly parallel --
    seeding on a rounded crest gave a crown of outward-leaning prisms instead."""
    def __call__(self, P):
        P = np.atleast_2d(np.asarray(P, float)); q = P / np.array([RX, RY, RZ])
        d = (np.linalg.norm(q, axis=1) - 1.0) * min(RX, RY, RZ)                 # ellipsoid distance bound
        d = d + 0.04 * FBM.eval(P * 0.9) + 0.015 * FINE.eval(P)                 # rough sides
        return 0.6 * np.maximum(d, P[:, 1] - TOP + 0.012 * FINE.eval(P * 1.3))  # near-flat top (slight grain): parallel growth
    eval = __call__
rock = Rock()

def upper(P):                                                                   # seeds only on the upper face
    P = np.atleast_2d(np.asarray(P, float)); return np.clip((P[:, 1] - (TOP - 0.06)) / 0.04, 0.0, 1.0)   # the flat top only

# --- the crystals: 16 points, competitive growth, one dominant seed favoured by size cap ------------------------------
# ART-DIRECTED SEEDS, physical everything else: a dominant central point, an inner ring of 5 and an outer ring of 6
# smaller ones on the flat top, c-axes leaning ~12-20 deg outward from the centre (competitive growth on a wall
# leaves the survivors roughly parallel, fanning slightly). Sizes: centre largest, rings smaller; pack still caps
# each by the room to its neighbours so points meet along faces rather than passing through each other.
rng = np.random.default_rng(4)
seedsP = [np.array([0.02, TOP, -0.02])]; seedsN = [np.array([0.0, 1.0, 0.0])]; sizes = [0.62]
for ring, n_r, lean, sz in ((0.14, 5, 0.22, 0.42), (0.27, 6, 0.36, 0.30)):
    for k in range(n_r):
        a = 2 * np.pi * (k + 0.5 * (ring > 0.2)) / n_r + rng.normal(0, 0.12)
        p = np.array([ring * np.cos(a), TOP, ring * np.sin(a)]) * np.array([1.0, 1.0, RZ / RX])
        n = np.array([lean * np.cos(a), 1.0, lean * np.sin(a)]) + rng.normal(0, 0.05, 3)
        seedsP.append(p); seedsN.append(n / np.linalg.norm(n)); sizes.append(sz * rng.uniform(0.85, 1.15))
lining = m.crystal_grow_on(rock, ((-0.8, -0.4, -0.7), (0.8, 1.0, 0.7)), habit="quartz_long", size=sizes, pack=2.6, size_jitter=0.0,
                           tilt=0.0, seed=11, substrate=False, cull=True, real_cell=True, clip_to_substrate=True,
                           seeds=(np.array(seedsP), np.array(seedsN)))
class Lin(object):
    def __call__(self, P): return np.asarray(lining.eval(np.atleast_2d(np.asarray(P, float))), float)
    eval = __call__
lin = Lin()
gx = np.linspace(-1.2, 1.2, 97); G = np.stack(np.meshgrid(gx, gx, gx, indexing='ij'), -1).reshape(-1, 3)
d = np.minimum(lin(G), rock(G)); ins = G[d < 0]; lo, hi = ins.min(0), ins.max(0); MID = (lo + hi) / 2; EXT = float((hi - lo).max())
FLOOR = float(lo[1]) - 0.005
print("bbox", np.round(lo, 2), np.round(hi, 2), "ext %.2f" % EXT, flush=True)
# below-surface check: crystal volume inside the rock must be zero
probe = G[(rock(G) < -0.01)]; print("   crystal volume inside the rock: %d of %d probe points" % (int((lin(probe) < 0).sum()), len(probe)), flush=True)

# --- museum lighting: the dim interior HDR as base, softbox key, fill, rim ------------------------------------------
env = m.hdri_env(np.load('hdri_voortrekker.npy')); E0 = float(env.floor_irradiance.mean())
KEY = np.array([-0.55, 0.75, 0.55]); KEY /= np.linalg.norm(KEY)
m.env_add_light(env, KEY, sigma=0.26, irradiance=4.5 * E0)                     # big softbox, upper left front
m.env_add_light(env, (0.8, 0.35, 0.45), sigma=0.40, irradiance=1.0 * E0)       # fill, right
m.env_add_light(env, (0.1, 0.45, -0.9), sigma=0.12, irradiance=2.4 * E0)       # rim behind: fires the terminations
AIM = env.dominant_dir
VIEW = np.array([0.75, 0.5, 1.1]); EYE = MID + VIEW / np.linalg.norm(VIEW) * EXT * 1.9
cam = m.camera(eye=tuple(EYE), target=tuple(MID + [0, 0.02 * EXT, 0]), fov_deg=34.0, aspect=W / H)

t0 = time.time()
key = __import__("hashlib").sha1(np.round(lin(G), 5).tobytes() + b"|256").hexdigest()[:12]; cache = "sdfgrid_museum_%s.npz" % key
if os.path.exists(cache):
    z = np.load(cache); TRACE = GridSDF(z["dist"], z["lo"], z["hi"])
else:
    TRACE = m.bake_sdf(lin, tuple(lo - 0.06), tuple(hi + 0.06), 208)   # 208^3: fits the 7 GB box; 256^3 was killed; np.savez(cache, dist=TRACE.dist, lo=TRACE.lo, hi=TRACE.hi)
print("   grid %.1fs" % (time.time() - t0), flush=True)
bake = m.bake_glass(TRACE, cam, W, H, material=MAT, n_lams=9, floor_y=FLOOR, max_internal=10, samples=S, opaque=rock, lam_jitter=True); tb = time.time() - t0
# faint milky base (real clear quartz clouds toward its root), holographic flaw volume
def milky(P):
    P = np.atleast_2d(np.asarray(P, float)); r = np.clip(1.0 - (P[:, 1] - TOP) / 0.22, 0.0, 1.0); return 0.04 + 0.6 * r ** 2
fb = tuple((float(lo[k] - 0.05), float(hi[k] + 0.05)) for k in range(3))
fv = m.gem_flaw_volume(field=milky, bounds=fb, res=22, dim=2048)
def rock_albedo(P):
    P = np.atleast_2d(np.asarray(P, float)); t = np.clip(0.5 + 0.3 * FBM.eval(P * 1.7) + 0.2 * FINE.eval(P), 0, 1)[:, None]
    return np.array([[0.16, 0.15, 0.14]]) * (1 - t) + np.array([[0.34, 0.31, 0.28]]) * t
floor_alb = lambda P: np.tile([[0.42, 0.42, 0.43]], (len(np.atleast_2d(P)), 1))
BG = (float(0.55 * (env.floor_irradiance @ [0.2126, 0.7152, 0.0722]) / np.pi),) * 3
t1 = time.time()
beauty = m.relight_glass(bake, env, floor_albedo=floor_alb, sdf=TRACE, sun_dir=tuple(AIM), material=MAT, shadow_transmittance=0.6,
                         opaque_albedo=rock_albedo, background=BG, footprint_filter=True,
                         flaw_volume=fv, flaw_sigma=4.0, flaw_albedo=(0.97, 0.96, 0.96)); tr = time.time() - t1
t2 = time.time()
xz, lam, wts = spectral_landings(TRACE, go["n_d"], go["abbe"], light_dir=tuple(-AIM), receiver_y=FLOOR, extent=EXT * 0.6, n_side=1100,
                                 seed=0, aim=tuple(MID), occluder=rock, through=True, absorb_rgb=go["absorb"], cmf=m.wavelength_cmf, max_internal=10)
final = beauty
if len(xz) > 2000:
    lo2, hi2 = np.percentile(xz, 1, axis=0), np.percentile(xz, 99, axis=0); Cc = tuple((lo2 + hi2) / 2); CWIN = float(max(hi2 - lo2) / 2 * 1.4)
    crgb = m.caustic_histogram_rgb(xz, lam, ((Cc[0] - CWIN, Cc[0] + CWIN), (Cc[1] - CWIN, Cc[1] + CWIN)), res=512, blur_px=1.2, weights=wts)
    px = m.caustic_pass(crgb, cam, W, H, plane_y=FLOOR, window=CWIN, center=Cc, occluder_sdf=lambda P: np.minimum(np.asarray(TRACE.eval(np.atleast_2d(P)), float), rock(P)))
    conc = caustic_concentration(xz, EXT * 0.6, 1100, tuple(-AIM))
    E_sun = float(env.floor_irradiance.mean() * env.sun_share) / np.pi * 0.42 * 0.9 * conc * float(np.mean(wts))
    final = m.composite_caustic(beauty, px, strength=E_sun, receiver_mask=bake.floor.reshape(H, W))
    print("   caustic: %d landings, conc %.1fx, strength %.3f" % (len(xz), conc, E_sun), flush=True)
tc = time.time() - t2
lum = final @ np.array([0.2126, 0.7152, 0.0722]); EV = float(np.clip(np.log2(0.20 / np.median(lum[lum > 1e-6])), -6, 6))
ch = pf.PostChain([("exposure", {"ev": EV}), ("bloom", {"threshold": 0.9, "sigma": 2.5, "intensity": 0.07}), ("aces", {}), ("gamma", {})])
out = np.clip(np.asarray(m.post_process(final, chain=ch), float), 0, 1).reshape(H // 2, 2, W // 2, 2, 3).mean(axis=(1, 3))
m.save_render("cluster_museum.png", out)
print("museum cluster: bake %.1fs relight %.1fs caustic %.1fs TOTAL %.1fs | EV %+.2f" % (tb, tr, tc, time.time() - t0, EV), flush=True)
