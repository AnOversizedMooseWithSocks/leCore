"""Run from the repo root:  python3 demos/crystal_gem_render.py geode amethyst plains 640 400 2 1.0
(needs hdri_<name>.npy in the cwd: np.save of m.load_exr(path)). Subjects: single | octa | cluster | geode | geodeball.

Crystal formations (single / cluster / cut geode) as GEM MATERIALS under a real HDRI plus additional lights.

The rig the brief asks for: the HDRI is the BASE light (floor irradiance, reflections, the soft ambient that makes
glass read as glass), and analytic lobes are layered on top ONLY to show off dispersion and caustics -- a key lobe
sized by the floor irradiance it adds (a fraction of what the map already delivers, so the map stays the base),
and a rim lobe behind the subject for internal sparkle. Brightness is then set by AUTO-EXPOSURE from the floor's
median luminance, never by pumping the lights: a lobe that clips the tonemapper is exactly the 'blown out' failure.
"""
import os, sys, time
os.environ.setdefault("PYTHONHASHSEED", "0"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import lecore
def mind():
    m = lecore.UnifiedMind(dim=512, seed=0); m.set_file_root('.'); return m
import holographic.rendering.holographic_postfx as pf
from holographic.rendering.holographic_holocaustic import spectral_landings, TiledHolographicCaustic, caustic_concentration
from holographic.materials_and_texture.holographic_matlib import glass_optics
m = mind()
which, mat, hd = sys.argv[1], sys.argv[2], sys.argv[3]; W, H = int(sys.argv[4]), int(sys.argv[5]); S = int(sys.argv[6])
KEYF = float(sys.argv[7]) if len(sys.argv) > 7 else 0.8        # key lobe irradiance as a fraction of the map's dominant lobe

# --- subject: all three come from the crystal-formation pipeline, unchanged --------------------------------------
OPAQUE = None
if which == "single":
    s = m.crystal_single("quartz", 0.95, real_cell=True); TILT = (1.25, 0.0, 0.35)      # c-axis is +z: tip it up and lean it
elif which == "octa":
    s = m.crystal_single("octahedron", 0.9, real_cell=True); TILT = (0.6, 0.0, 0.4)     # the diamond habit, tipped onto a face
elif which == "cluster":
    # AN AMETHYST PLATE, as in the reference: a rock slab (opaque) paved with short fat "druse" points growing UP from
    # its top face, shoulder to shoulder -- not a ball of interpenetrating prisms. grow_on's `where` gates the seeds
    # to the upward-facing part of the slab; the slab itself is the opaque body.
    from holographic.mesh_and_geometry.holographic_sdf import box
    slab_sdf = box(0.62, 0.11, 0.42).rounded(0.07)
    class _Slab(object):
        def __call__(self, P): return np.asarray(slab_sdf.eval(np.atleast_2d(np.asarray(P, float))), float)
        eval = __call__
    slab = _Slab()
    def top_only(P):                                  # weight 1 on the top face, 0 on the sides/bottom
        P = np.atleast_2d(np.asarray(P, float)); return np.clip((P[:, 1] - 0.06) / 0.05, 0.0, 1.0)
    # Physics (Wikipedia/Amethyst; GIA): geode/plate amethyst grows as close-packed six-sided PYRAMIDS with little
    # prism, c-axes near-perpendicular to the substrate (competitive growth), similar sizes, colour at the tips.
    # quartz_point at size 0.22: radius 0.20, half-length 0.20 (measured proportions), ~half projecting above the root
    # COMPETITIVE GROWTH: 40 seeds on the top face, each crystal sized by the room to its nearest neighbour (pack),
    # so points meet along their prism faces instead of interpenetrating into shards. "quartz" habit (radius 0.58,
    # half-length 1.5 x size) at pack 0.85 -> neighbours touch; height above the root ~1.15 x size ~ its width, as in
    # a real plate. Real unit cell. Nothing here is a draw of independent sizes.
    s = m.crystal_grow_on(slab, ((-0.8, -0.3, -0.6), (0.8, 0.9, 0.6)), count=70, habit="quartz", size=0.5, pack=1.0,
                          size_jitter=0.08, tilt=0.12, where=top_only, seed=3, substrate=False, cull=True, real_cell=True); TILT = None
    OPAQUE = slab
elif which == "geode":
    # rock and crystals apart (parts=True): the rind is OPAQUE, only the lining is glass. clip_to_skin stops the
    # crystals' back halves running out through the rind (39% of the lining volume was outside the nodule).
    # DENSE lining (reference photos: the cavity wall is paved shoulder-to-shoulder with small points, not dotted
    # with a few big ones): 340 crystals of size ~0.11 on a 0.9 nodule. mixed habits keep each its own lattice.
    GR, GT = 0.9, 0.18
    # PAVED lining: 900 short fat "druse" points of size 0.16 cover 99% of the cavity wall (measured); the old
    # 560 needle-thin "quartz" covered 32% and the geode looked bald.
    GR, GT = 0.9, 0.18
    # PAVED lining of pyramidal terminations (geode amethyst lacks prism faces), c-axes near-radial (tilt 0.08),
    # similar sizes: measured 99% wall coverage at 900 points; here 700 at size 0.19.
    GR, GT = 0.9, 0.18
    # amethyst_pyramid at size 0.11: radius 0.115, half-length 0.083; root sunk 0.04 -> tips ~0.045 past the wall,
    # 420 of them on a wall of area 6.5 (footprint 0.042 each -> 2.7x cover, shoulder to shoulder)
    # amethyst_pyramid was too squat here (radius 1.05 x half 0.75 -> only 0.4 x size projects past the root, the wall
    # buried most of it: measured as sparse patches on a white wall). quartz_point projects 0.55 x size: at 0.13 that is
    # 0.07 tall by 0.117 wide, 600 of them = 4x cover.
    # COMPETITIVE GROWTH on the cavity wall: 320 seeds, each point sized by its Voronoi room (pack=1.0), "quartz"
    # habit on the real cell -- neighbours meet along prism faces, tips project ~1.15 x size (~0.09 here).
    rind, lining = m.crystal_geode(radius=GR, shell=GT, count=320, habit="quartz", size=0.3, pack=1.0, size_jitter=0.08,
                                   tilt=0.10, seed=1, cull=True, parts=True, clip_to_skin=True, skin_margin=0.06, real_cell=True)
    nrm = np.array([0.0, 0.80, 0.60]); nrm /= np.linalg.norm(nrm)                    # cut tilted UP and toward the camera: ~47 deg off the view axis, a bowl not a disc
    s = m.crystal_cut(lining, normal=tuple(nrm), point=tuple(nrm * 0.30)); TILT = None
    # ROUGH ROCK: the outside of a geode is weathered stone, not a polished ball. Displace the rind by a holographic
    # fBm (procedural_noise, 4 octaves, baked once on a 48^3 lattice with sample_grid_fast -- 4 s -- and read
    # trilinearly), then cut: the saw face stays flat, which is what a cut geode looks like. The displaced field is
    # no longer Lipschitz-1, so it is scaled by 0.7 to keep the sphere tracer conservative.
    from holographic.mesh_and_geometry.holographic_sdfbake import GridSDF
    fbm_grid = np.load('rock_fbm48.npy') if os.path.exists('rock_fbm48.npy') else None
    if fbm_grid is None:
        fbm_grid = m.procedural_noise(n_dims=3, dim=1024, bounds=((-1.1, 1.1),) * 3, octaves=4, base_bandwidth=3.0,
                                      gain=0.55, seed=7).sample_grid_fast(48); np.save('rock_fbm48.npy', fbm_grid)
    FBM = GridSDF((fbm_grid - fbm_grid.mean()) / (fbm_grid.std() + 1e-9), (-1.1, -1.1, -1.1), (1.1, 1.1, 1.1))
    fine = np.load('rock_fbm96.npy') if os.path.exists('rock_fbm96.npy') else None
    if fine is None:                                      # a second, finer octave set: grain, not just lumps
        fine = m.procedural_noise(n_dims=3, dim=1024, bounds=((-1.1, 1.1),) * 3, octaves=3, base_bandwidth=9.0,
                                  gain=0.6, seed=11).sample_grid_fast(96); np.save('rock_fbm96.npy', fine)
    FINE = GridSDF((fine - fine.mean()) / (fine.std() + 1e-9), (-1.1, -1.1, -1.1), (1.1, 1.1, 1.1))
    ROUGH_AMP, FINE_AMP = 0.02, 0.016
    class _Rough(object):                                 # rind: rough OUTSIDE, smooth cavity wall
        def __call__(self, P):
            P = np.atleast_2d(np.asarray(P, float)); r = np.linalg.norm(P, axis=1)
            # displace ONLY the outer skin. Displacing the whole shell bulged the cavity wall inward by up to 0.07 and
            # buried the lining (crystals span r 0.60-0.72): measured as a "glossy white ring" with purple only at
            # the far wall. The cavity wall is where the chalcedony was deposited -- smooth by formation.
            outer = r - GR + ROUGH_AMP * FBM.eval(P) + FINE_AMP * FINE.eval(P)
            inner = (GR - GT) - r
            return 0.6 * np.maximum(outer, inner)
        eval = __call__
    rind_cut = m.crystal_cut(_Rough(), normal=tuple(nrm), point=tuple(nrm * 0.30))
    def rock_albedo(P):
        """Weathered brown stone outside; toward the cavity the AGATE BANDS every cut geode shows -- white and grey
        chalcedony layers deposited before the crystals -- keyed on radius, so the cut face reads them as rings."""
        P = np.atleast_2d(np.asarray(P, float)); t = np.clip(0.5 + 0.3 * FBM.eval(P * 1.7) + 0.2 * FINE.eval(P), 0.0, 1.0)[:, None]
        r = np.linalg.norm(P, axis=1)[:, None]; cav = GR - GT
        depth = np.clip((r - cav) / GT, 0.0, 1.0)                                  # 0 at the cavity wall, 1 at the skin
        band = 0.5 + 0.5 * np.cos(depth * 9.0 * np.pi + 0.8 * FBM.eval(P * 1.1)[:, None])  # rings, slightly wavy
        chalcedony = np.array([[0.62, 0.62, 0.64]]) * (0.72 + 0.28 * band)          # white-grey layers
        blend = np.clip((depth - 0.45) / 0.15, 0.0, 1.0)                            # inner 45% of the shell is banding
        # pits are darker: the displacement value itself is a cheap ambient-occlusion proxy (negative = a crevice)
        pit = np.clip(0.5 + 0.5 * (ROUGH_AMP * FBM.eval(P) + FINE_AMP * FINE.eval(P)) / (ROUGH_AMP + FINE_AMP), 0.0, 1.0)[:, None]
        rock = (np.array([[0.30, 0.27, 0.24]]) * (1 - t) + np.array([[0.50, 0.47, 0.43]]) * t) * (0.45 + 0.55 * pit)
        return chalcedony * (1 - blend) + rock * blend
    class _R(object):
        def __call__(self, P): return np.asarray(rind_cut.eval(np.atleast_2d(np.asarray(P, float))), float)
        eval = __call__
    OPAQUE = _R()
elif which == "geodeball":
    # THE GEOMETRY MOOSE PICKED ("the most correct result so far"): the sweep-156 diagnostic -- the long "amethyst_druse"
    # drums (700 at size 0.19) radiating from the cavity wall to a coarse crystal mass in the middle, in a smooth
    # rind with a flat cut face. Same seeds, same cut, same camera as the diagnostic; only the materials are new.
    GR, GT = 0.9, 0.18
    rind, lining = m.crystal_geode(radius=GR, shell=GT, count=700, habit="amethyst_druse", size=0.19, size_jitter=0.2, real_cell=True,
                                   tilt=0.08, seed=1, cull=True, parts=True, clip_to_skin=True, skin_margin=0.06)
    nrm = np.array([0.0, 0.80, 0.60]); nrm /= np.linalg.norm(nrm)
    s = m.crystal_cut(lining, normal=tuple(nrm), point=tuple(nrm * 0.30)); TILT = None
    rind_cut = m.crystal_cut(rind, normal=tuple(nrm), point=tuple(nrm * 0.30))     # smooth rind, as in the diagnostic
    class _R(object):
        def __call__(self, P): return np.asarray(rind_cut.eval(np.atleast_2d(np.asarray(P, float))), float)
        eval = __call__
    OPAQUE = _R()
    def rock_albedo(P):
        """Pale grey stone with faint agate rings toward the cavity -- the diagnostic's look, kept."""
        P = np.atleast_2d(np.asarray(P, float)); r = np.linalg.norm(P, axis=1)[:, None]
        depth = np.clip((r - (GR - GT)) / GT, 0.0, 1.0)
        band = 0.5 + 0.5 * np.cos(depth * 7.0 * np.pi)
        return np.array([[0.60, 0.60, 0.61]]) * (0.88 + 0.12 * band)
else:
    raise SystemExit("single | octa | cluster | geode | geodeball")
if which == "cluster":
    def rock_albedo(P):
        """The plate's matrix: white quartz crust where the points grow, brown host rock beneath."""
        P = np.atleast_2d(np.asarray(P, float)); t = np.clip((P[:, 1] + 0.02) / 0.10, 0.0, 1.0)[:, None]
        return np.array([[0.36, 0.31, 0.27]]) * (1 - t) + np.array([[0.80, 0.78, 0.76]]) * t
elif OPAQUE is None:
    rock_albedo = (0.40, 0.36, 0.33)

def sdf(P):
    P = np.atleast_2d(np.asarray(P, float))
    if TILT is not None:                                      # lay the single crystal on its side, rotated: a standing prism is a dull view
        ax, ay, az = TILT
        Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
        P = P @ (Rz @ Rx)
    return np.asarray(s.eval(P), float)
class _S(object):
    def __call__(self, P): return sdf(P)
    def eval(self, P): return sdf(P)
S_ = _S()

# --- measured extents: never hand-set the floor, find the lowest point of the body -----------------------------
gx = np.linspace(-1.8, 1.8, 121); G = np.stack(np.meshgrid(gx, gx, gx, indexing='ij'), -1).reshape(-1, 3)
d = sdf(G)
if OPAQUE is not None:
    d = np.minimum(d, OPAQUE(G))                                                   # the rock is part of the body's extent
ins = G[d < 0]
lo, hi = ins.min(0), ins.max(0); CTR = (lo + hi) / 2; CTR[1] = lo[1]                # centre the composition on the base
FLOOR = float(lo[1]) - 0.01; EXT = float((hi - lo).max())
print("%s: bbox %s..%s ext %.2f floor %.2f" % (which, np.round(lo, 2), np.round(hi, 2), EXT, FLOOR), flush=True)

# --- lever 1 for the many-crystal bodies: bake the union onto a 256^3 grid ONCE and trace the grid --------------------
# MEASURED (cluster, 160x100): bake_glass 5.2s exact -> 1.3s grid (4x), relit image mean abs diff 1.2%, 95th pct 0.02%,
# side-by-side indistinguishable -- a trilinear grid reproduces PLANAR facets exactly, only edges blend over one cell
# (h = 0.0075). The HybridSDF (grid far, exact near) was measured 0.9x: the exact field's cost is per CALL, not per
# point, and almost every march step has some ray near a surface -- kept as a negative. The grid is a deterministic
# function of the seed (lever 3), so it is cached on disk by a content hash and regenerated when absent.
import hashlib
if which != "single":
    pad = 0.06
    # keyed by the FIELD's values on the probe grid, not by its parameters: a stale grid was reused once when the
    # bbox stayed the same while the lining changed (skin_margin) -- the render silently showed the old geometry.
    key = hashlib.sha1(np.round(sdf(G), 5).tobytes() + b"|256").hexdigest()[:12]
    cache = "sdfgrid_%s_%s.npz" % (which, key)
    from holographic.mesh_and_geometry.holographic_sdfbake import GridSDF
    if os.path.exists(cache):
        z = np.load(cache); TRACE = GridSDF(z["dist"], z["lo"], z["hi"]); print("   grid cache hit %s" % cache, flush=True)
    else:
        tg = time.time(); TRACE = m.bake_sdf(S_, tuple(lo - pad), tuple(hi + pad), 256)
        np.savez(cache, dist=TRACE.dist, lo=TRACE.lo, hi=TRACE.hi); print("   grid bake 256^3 %.1fs -> %s" % (time.time() - tg, cache), flush=True)
else:
    TRACE = S_

# --- light: HDRI base + key + rim -------------------------------------------------------------------------------
go = glass_optics(mat)
# a regular octahedron of diamond (n 2.42) traps light: 6 internal bounces lost 23.8% of glass-wavelength pairs, 24 lose 5.9%
MAXI = 24 if go["n_d"] > 2.0 else 8
env = m.hdri_env(np.load('hdri_%s.npy' % hd))
E_map, share_map, dom_map = env.floor_irradiance.mean(), env.sun_share, env.dominant_dir.copy()
KEY = np.array([-0.45, 0.80, 0.40]) if which != "geode" else np.array([0.25, 0.85, 0.45])   # a bowl is lit from its open side
KEY = KEY / np.linalg.norm(KEY)
RIM = np.array([0.30, 0.55, -1.00]); RIM /= np.linalg.norm(RIM)                     # behind the subject: fires the facets
m.env_add_light(env, KEY, sigma=0.22, irradiance=KEYF * E_map * max(share_map, 0.15))   # a SOFTBOX: broad highlights, as in the references
m.env_add_light(env, RIM, sigma=0.06, irradiance=0.35 * KEYF * E_map * max(share_map, 0.15))
print("   map E %.3f share %.2f -> with lobes E %.3f share %.2f dominant %s" % (E_map, share_map, env.floor_irradiance.mean(), env.sun_share, np.round(env.dominant_dir, 2)), flush=True)
AIM = env.dominant_dir                                                                # the caustic follows the strongest lobe

MID = (lo + hi) / 2                                                                  # frame the WHOLE body: look at its centre from ~1.7 extents away
CAMD = {"cluster": 1.45, "geode": 2.0, "geodeball": 2.0}.get(which, 2.3)
VIEW = np.array([0.8, 0.6, 1.1]) if which == "cluster" else np.array([0.95, 0.55, 1.25])   # a plate is photographed from above
EYE = MID + VIEW / np.linalg.norm(VIEW) * EXT * CAMD
cam = m.camera(eye=tuple(EYE), target=tuple(MID - [0, 0.04 * EXT, 0]), fov_deg=36.0, aspect=W / H)
t0 = time.time()
bake = m.bake_glass(TRACE, cam, W, H, material=mat, n_lams=9, floor_y=FLOOR, max_internal=MAXI, samples=S, opaque=OPAQUE, lam_jitter=True); tb = time.time() - t0
t1 = time.time()
# IMPERFECTIONS (sweep 155): milky cloudiness dense at the crystals' BASES fading to clear tips, and the amethyst
# chromophore concentrated at the TIPS -- both as holographic flaw volumes read in closed form along each pixel's
# interior path. For the geode the "base" is the cavity wall (radius cav); for the cluster it is the rocky core.
FLAWS = {}
if which in ("geode", "cluster", "geodeball"):
    cloud = m.crystal_cloudiness(strength=1.0, freq=7.0, seed=5, threshold=0.3, sharp=4.0)
    if which == "geodeball":
        wall = GR - GT
        base_w = lambda P: np.clip(1.0 - (wall - np.linalg.norm(P, axis=1)) / 0.20, 0.0, 1.0)      # bases at the wall, tips toward the centre
    elif which == "geode":
        wall = GR - GT
        # the lining's crystals span r in [0.60, 0.72] (tips inward): "base" must mean the last ~0.04 at the wall, or
        # the whole lining reads as milky white (measured: it did -- a white glossy ring, purple only at the far wall)
        base_w = lambda P: np.clip(1.0 - (wall - np.linalg.norm(P, axis=1)) / 0.06, 0.0, 1.0)     # tips project ~0.09
    else:
        base_w = lambda P: np.clip(1.0 - (P[:, 1] - 0.11) / 0.16, 0.0, 1.0)                          # 1 at the plate, 0 above 0.27
    def milky(P):
        P = np.atleast_2d(np.asarray(P, float)); b = base_w(P)
        # a smooth milky base plus a faint uniform haze: real amethyst is translucent, not window glass. The noise
        # field was dropped here -- at dim 2048 its speckle read as frost, not as cloudiness.
        return 0.28 + 0.72 * b ** 2.0                   # translucent throughout: real amethyst hides its interior
    def chromo(P):
        P = np.atleast_2d(np.asarray(P, float)); return (1.0 - base_w(P)) ** (2.0 if which == "cluster" else 1.5 if which == "geodeball" else 1.0)   # iron colour toward the tips
    fb = tuple((float(lo[k] - 0.05), float(hi[k] + 0.05)) for k in range(3))
    tf = time.time()
    # sigma scales with 1/crystal size: the density is [0,1] and a 0.11 druse point must go milky over ~0.1 of path
    # (measured: median interior path 0.065 in the geode -> sigma 14 gives T~0.4 at the base), the 0.42 cluster over ~0.4
    # Every optical constant comes from the material library (glass_optics): n_d, Abbe, and the per-RGB absorption.
    # Colour ZONING is that same absorption redistributed along the crystal -- a pale base (0.25x the library value
    # everywhere) and the rest concentrated where the chromophore density is (the terminations) -- so the hue is the
    # library's amethyst, only its placement follows the physics of growth.
    LIB = np.asarray(go["absorb"], float)
    ZONE_GAIN = {"geode": 6.0, "geodeball": 3.0}.get(which, 4.0)      # tips: how much denser than the mean
    FLAWS = dict(flaw_volume=m.gem_flaw_volume(field=milky, bounds=fb, res=22, dim=2048),
                 flaw_sigma=(9.0 if which == "geode" else 4.0 if which == "geodeball" else 5.0), flaw_albedo=(0.96, 0.94, 0.95),
                 absorb_volume=m.gem_flaw_volume(field=chromo, bounds=fb, res=18, dim=2048),
                 absorb_volume_sigma=tuple(LIB * ZONE_GAIN), absorb=tuple(LIB * 0.25))
    print("   flaw + zoning volumes %.1fs" % (time.time() - tf), flush=True)

def flat_floor(P):
    return np.tile([[0.46, 0.46, 0.47]], (len(np.atleast_2d(P)), 1))
def checker(P):
    """A checkerboard floor: 0.25-unit squares in x,z; light and dark neutral greys (albedo, so it stays a floor
    under whatever the map does)."""
    P = np.atleast_2d(np.asarray(P, float)); q = (np.floor(P[:, 0] / CHK) + np.floor(P[:, 2] / CHK)).astype(int) % 2
    a = np.where(q[:, None] == 0, [[0.66, 0.64, 0.61]], [[0.20, 0.19, 0.18]])
    # fade to the mean with distance from the eye: at 2 AA samples a checker at the horizon is moire, not floor
    far = np.clip((np.linalg.norm(P - np.asarray(EYE)[None, :], axis=1) - 3.0 * EXT) / (6.0 * EXT), 0.0, 1.0)[:, None]
    return a * (1 - far) + np.array([[0.43, 0.415, 0.395]]) * far
CHK = 0.55 * EXT / 1.5
BG = (float(0.5 * (env.floor_irradiance @ [0.2126, 0.7152, 0.0722]) / np.pi),) * 3     # a NEUTRAL grey card lit like the floor
beauty = m.relight_glass(bake, env, floor_albedo=(flat_floor if which == "geodeball" else checker), sdf=TRACE, sun_dir=tuple(AIM), material=(None if FLAWS else mat), shadow_transmittance=0.55,   # Fresnel + scatter losses; colour now comes from the chord   # light the glass FOCUSES into the caustic leaves the shadow: 0.8 double-counted it
                        
                         opaque_albedo=rock_albedo, background=BG, opaque_bounce=(1.2 if which == "geode" else 0.8 if which == "geodeball" else 0.0), footprint_filter=True,
                         opaque_bounce_where=(lambda P: np.clip((GR - GT + 0.5 * GT - np.linalg.norm(np.atleast_2d(P), axis=1)) / (0.3 * GT), 0, 1)) if which in ("geode", "geodeball") else None, **FLAWS); tr = time.time() - t1

# --- caustic: one trace with per-ray wavelength, written into one hypervector, read as an outer product -----------
t2 = time.time()
# FULL-TRANSIT caustic: entry AND exit refraction (a prism's exit is what sets where the light lands), interior path
# length per ray, and each landing weighted by the material library's Beer-Lambert transmission over that path --
# a coloured gem throws a coloured caustic and a tinted shadow (Moose's review). Rock occludes.
xz, lam, wts = spectral_landings(TRACE, go["n_d"], go["abbe"], light_dir=tuple(-AIM), receiver_y=FLOOR, extent=EXT * 0.60, n_side=1100,
                                 dispersion_scale=1.0, seed=0, aim=tuple((lo + hi) / 2), occluder=OPAQUE,
                                 through=True, absorb_rgb=go["absorb"], cmf=m.wavelength_cmf, max_internal=MAXI)
if len(xz) > 2000:   # a paved druse spills light, it does not focus; skip
    lo2, hi2 = np.percentile(xz, 1, axis=0), np.percentile(xz, 99, axis=0); Cc = tuple((lo2 + hi2) / 2); CWIN = float(max(hi2 - lo2) / 2 * 1.5)
    # HISTOGRAM caustic (the measured baseline): the tiled holographic read smeared the plate's filaments into
    # blotches and showed tile seams at this landing count -- see histogram_caustic_rgb. More emitter rays for a
    # smooth field: n_side 1100 = 1.2M rays through the grid-baked lining.
    crgb = m.caustic_histogram_rgb(xz, lam, ((Cc[0] - CWIN, Cc[0] + CWIN), (Cc[1] - CWIN, Cc[1] + CWIN)), res=512, blur_px=1.2, weights=wts)
    px = m.caustic_pass(crgb, cam, W, H, plane_y=FLOOR, window=CWIN, center=Cc, occluder_sdf=lambda P: (lambda Q: np.asarray(TRACE.eval(Q), float) if OPAQUE is None else np.minimum(np.asarray(TRACE.eval(Q), float), OPAQUE(Q)))(np.atleast_2d(np.asarray(P, float))))
    conc = caustic_concentration(xz, EXT * 0.60, 1100, tuple(-AIM))                  # measured hotspot concentration, 3-10x typical
    E_sun = float(env.floor_irradiance.mean() * env.sun_share) / np.pi * 0.48 * 0.9 * conc * float(np.mean(wts))   # absorbed light is not on the floor   # Lambert x albedo x transmittance x concentration
    final = m.composite_caustic(beauty, px, strength=E_sun, receiver_mask=bake.floor.reshape(H, W))   # only where the FLOOR is seen
    print("   caustic: %d landings, window %.2f at %s, concentration %.1fx, strength %.3f" % (len(xz), CWIN, np.round(Cc, 2), conc, E_sun), flush=True)
else:
    final = beauty; print("   caustic EMPTY (%d landings)" % len(xz), flush=True)
tc = time.time() - t2
np.save("crystal_%s_%s_%s.npy" % (which, mat, hd), final)

# --- auto-exposure from the floor, then ACES ------------------------------------------------------------------------
lum = final @ np.array([0.2126, 0.7152, 0.0722])
med = float(np.median(lum[lum > 1e-6])); EV = float(np.log2(0.22 / max(med, 1e-9)))
EV = float(np.clip(EV, -6, 6))
ch = pf.PostChain([("exposure", {"ev": EV}), ("bloom", {"threshold": 0.95, "sigma": 2.0, "intensity": 0.06}), ("aces", {}), ("gamma", {})])
out = np.clip(np.asarray(m.post_process(final, chain=ch), float), 0, 1).reshape(H // 2, 2, W // 2, 2, 3).mean(axis=(1, 3))
m.save_render("crystal_%s_%s_%s.png" % (which, mat, hd), out)
clip = float((np.asarray(m.post_process(final, chain=pf.PostChain([("exposure", {"ev": EV})])), float).max(-1) > 16.0).mean())
print("%s %s @%s: bake %.1fs relight %.1fs caustic %.1fs TOTAL %.1fs | auto EV %+.2f (median %.4f) | >16x pixels %.2f%%" %
      (which, mat, hd, tb, tr, tc, time.time() - t0, EV, med, 100 * clip), flush=True)
