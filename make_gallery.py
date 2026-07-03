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
def _tonemap(hdr):
    """HDR -> displayable sRGB-ish: Reinhard tone map + gamma. Keeps highlights from clipping to white."""
    return np.clip((hdr / (1.0 + hdr)) ** (1 / 2.2), 0, 1)

class _Cam:
    """A tiny pinhole camera: an eye point and a grid of ray directions. Enough for the path tracer."""
    def __init__(self, eye=(0.0, 0.6, 4.2), tilt=-0.12, fov=1.3):
        self.eye = np.array(eye, float); self.tilt = tilt; self.fov = fov
    def ray_dirs(self, w, h):
        ys, xs = np.mgrid[0:h, 0:w]
        u = (xs / (w - 1) - 0.5) * self.fov
        v = -(ys / (h - 1) - 0.5) * self.fov
        d = np.stack([u, v + self.tilt, -np.ones_like(u)], -1)
        return self.eye, d / np.linalg.norm(d, axis=-1, keepdims=True)

def _sky(D):
    """A soft two-tone sky gradient used to light every render (warm near the horizon, blue up high)."""
    t = np.clip(D[:, 1] * 0.5 + 0.5, 0, 1)[:, None]
    return (1 - t) * np.array([0.9, 0.85, 0.8]) + t * np.array([0.35, 0.5, 0.9])


# =========================================================================== 3-D RENDERS
def render_spheres():
    """Three spheres (diffuse red, gold metal, blue) on a checker floor -- the basic material showcase."""
    from holographic_pathtrace import path_trace
    centers = np.array([[-1.3, 0, 0], [0, 0, 0], [1.3, 0, 0]], float); radii = np.array([0.7, 0.9, 0.6])
    class Scene:
        def eval(self, P):
            d = np.min(np.linalg.norm(P[..., None, :] - centers, axis=-1) - radii, axis=-1)   # union of 3 spheres
            return np.minimum(d, P[..., 1] + 0.9)                                              # + ground at y=-0.9
    def material(P):
        n = len(P); alb = np.tile([.8, .8, .8], (n, 1)).astype(float)
        met = np.zeros(n); rough = np.full(n, .6); emis = np.zeros((n, 3))
        g = P[:, 1] < -0.85; chk = ((np.floor(P[:, 0] * 1.5) + np.floor(P[:, 2] * 1.5)).astype(int) % 2 == 0)
        alb[g] = np.where(chk[g, None], [.9, .9, .9], [.15, .15, .18])
        l = P[:, 0] < -0.7; alb[l] = [.85, .2, .2]; rough[l] = .7
        m = (P[:, 0] >= -0.7) & (P[:, 0] <= 0.7); met[m] = 1.; rough[m] = .15; alb[m] = [.95, .85, .55]
        r = P[:, 0] > 0.7; alb[r] = [.2, .4, .85]; rough[r] = .35
        return alb, met, rough, emis
    img = path_trace(Scene(), _Cam(), width=240, height=200, spp=48, max_bounce=4, material=material, sky=_sky, seed=0)
    plt.imsave(f"{OUT}/render_spheres.png", _tonemap(img)); print("  render_spheres.png")

def render_glass():
    """A clear GLASS sphere in front of two coloured spheres -- shows refraction (the material returns IOR>1)."""
    from holographic_pathtrace import path_trace
    back = np.array([[-0.8, -0.1, -1.4], [0.9, -0.1, -1.6]], float); br = np.array([0.55, 0.6])
    gc = np.array([0.0, 0.0, 0.4]); gr = 0.75
    class Scene:
        def eval(self, P):
            d = np.min(np.linalg.norm(P[..., None, :] - back, axis=-1) - br, axis=-1)          # two solid spheres
            glass = np.linalg.norm(P - gc, axis=-1) - gr                                        # the glass sphere
            return np.minimum(np.minimum(d, glass), P[..., 1] + 0.9)                            # + floor
    def material(P):
        n = len(P); alb = np.tile([.8, .8, .8], (n, 1)).astype(float)
        met = np.zeros(n); rough = np.full(n, .5); emis = np.zeros((n, 3)); ior = np.zeros(n)   # ior=0 -> opaque
        g = P[:, 1] < -0.85; chk = ((np.floor(P[:, 0] * 1.5) + np.floor(P[:, 2] * 1.5)).astype(int) % 2 == 0)
        alb[g] = np.where(chk[g, None], [.85, .85, .9], [.1, .1, .15])
        onglass = np.abs(np.linalg.norm(P - gc, axis=-1) - gr) < 0.05
        ior[onglass] = 1.5; alb[onglass] = [1, 1, 1]; rough[onglass] = 0.02                     # ior=1.5 -> glass
        left = (P[:, 0] < -0.3) & ~onglass & ~g; alb[left] = [.9, .3, .2]
        right = (P[:, 0] > 0.3) & ~onglass & ~g; alb[right] = [.2, .5, .9]
        return alb, met, rough, emis, ior                                                       # 5th value = IOR
    cam = _Cam(eye=(0.0, 0.4, 3.6), tilt=-0.06, fov=1.25)
    img = path_trace(Scene(), cam, width=240, height=200, spp=64, max_bounce=6, material=material, sky=_sky, seed=0)
    plt.imsave(f"{OUT}/render_glass.png", _tonemap(img)); print("  render_glass.png")

def render_fractal():
    """A Menger-sponge fractal (the SDF is ~12 bytes; the geometry is generated, not stored)."""
    from holographic_pathtrace import path_trace
    from holographic_sdf import menger
    sponge = menger(3, 1.4)                                     # 3 recursion levels
    class Scene:
        def eval(self, P):
            return np.minimum(np.asarray(sponge.eval(P)), P[..., 1] + 1.0)     # sponge + a floor
    def material(P):
        n = len(P); alb = np.tile([.75, .55, .35], (n, 1)).astype(float)       # warm stone
        met = np.zeros(n); rough = np.full(n, .55); emis = np.zeros((n, 3))
        g = P[:, 1] < -0.95; alb[g] = [.2, .22, .28]
        return alb, met, rough, emis
    cam = _Cam(eye=(2.2, 1.6, 2.4), tilt=-0.18, fov=1.2)
    img = path_trace(Scene(), cam, width=220, height=200, spp=40, max_bounce=4, material=material, sky=_sky, seed=0)
    plt.imsave(f"{OUT}/render_fractal.png", _tonemap(img)); print("  render_fractal.png")


# =========================================================================== PROCEDURAL / GENERATIVE
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
               ("patterns", render_patterns), ("reaction_diffusion", render_reaction_diffusion),
               ("core_ops", chart_core_ops), ("compression", chart_compression),
               ("capacity", chart_capacity), ("degradation", chart_degradation)]
    for name, fn in visuals:
        try:
            fn()
        except Exception as e:
            print(f"  [skip {name}] {e}"); traceback.print_exc()
    print("done -> ./gallery/")
