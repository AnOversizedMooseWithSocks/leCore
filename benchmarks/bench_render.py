"""Benchmark: the RENDER path's throughput and time-to-quality -- the speed-regression guard for pipeline work.

WHY THIS EXISTS (backlog J1). The bench suite covered compression/recall only; the sole render measurement in
the repo was an ad-hoc print inside make_gallery.py. That means rewiring the render pipeline (backlog H) had no
safety net: a change could halve the tracer's throughput and nothing would notice. This bench measures the five
workhorses so a before/after diff is one command:

  * path_trace       -- pixel-samples/sec (the tracer's raw throughput), one bounce budget held fixed
  * render_auto      -- time-to-quality: seconds to reach the 'medium' confidence target, and the PSNR it buys
                        vs a raw equal-budget trace (the pipeline must at least not LOSE to raw)
  * render_hair      -- strand-segments/sec through the strand rasteriser (the known Python-loop hot spot, F3)
  * StableFluid.step -- sim steps/sec at 64x64 (smoke), the volume-sim workhorse
  * volume_render    -- field-samples/sec marching a smoke blob (the volume RENDER workhorse)

Timings are indicative and machine-dependent (single-thread NumPy in a sandbox); the QUALITY numbers (PSNR
deltas) are deterministic. Run:  python benchmarks/bench_render.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


# ---------------------------------------------------------------------- shared tiny scene (spheres + floor)
class _Cam:
    """Minimal pinhole camera exposing ray_dirs(w,h) -> (eye, dirs), the interface the tracer expects."""
    eye = np.array([0.0, 0.4, 3.2])
    def ray_dirs(self, w, h):
        ys, xs = np.mgrid[0:h, 0:w]
        u = (xs / (w - 1) - 0.5) * 1.2
        v = -(ys / (h - 1) - 0.5) * 1.2
        d = np.stack([u, v, -np.ones_like(u)], -1)
        return self.eye, d / np.linalg.norm(d, axis=-1, keepdims=True)


_centers = np.array([[-0.7, 0, 0], [0.7, 0, 0]], float)
_radii = np.array([0.6, 0.6])


class _Scene:
    def eval(self, P):
        d = np.min(np.linalg.norm(P[..., None, :] - _centers, axis=-1) - _radii, axis=-1)
        return np.minimum(d, P[..., 1] + 0.9)


def _material(P):
    n = len(P); alb = np.tile([.8, .3, .3], (n, 1)).astype(float)
    alb[P[:, 0] < 0] = [.3, .4, .85]
    return alb, np.zeros(n), np.full(n, .6), np.zeros((n, 3))


def _sky(D):
    t = np.clip(D[:, 1] * 0.5 + 0.5, 0, 1)[:, None]
    return (1 - t) * np.array([0.9, 0.85, 0.8]) + t * np.array([0.35, 0.5, 0.9])


def _tm(x):
    """Tonemap for the PSNR comparisons (what the viewer sees)."""
    return np.clip((np.asarray(x, float) / (1.0 + np.asarray(x, float))) ** (1 / 2.2), 0, 1)


def _psnr(a, b):
    mse = float(np.mean((_tm(a) - _tm(b)) ** 2))
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


# ---------------------------------------------------------------------- the five benches
def bench_path_trace(W=96, H=72, spp=8, max_bounce=3):
    """Raw tracer throughput: pixel-samples per second at a fixed bounce budget."""
    from holographic_pathtrace import path_trace
    t0 = time.time()
    path_trace(_Scene(), _Cam(), width=W, height=H, spp=spp, max_bounce=max_bounce,
               material=_material, sky=_sky, seed=0)
    dt = time.time() - t0
    ps = W * H * spp / dt
    print(f"path_trace      : {ps/1e3:8.1f}k pixel-samples/sec   ({W}x{H} @ {spp}spp, {max_bounce} bounces, {dt:.2f}s)")
    return ps


def bench_render_auto(W=96, H=72):
    """Time-to-quality: seconds for render_auto to hit its 'medium' target, and the honest PSNR comparison
    against a raw trace at the SAME average sample budget (the pipeline must not lose to raw)."""
    from holographic_gbuffer import render_auto
    from holographic_pathtrace import path_trace
    ref = path_trace(_Scene(), _Cam(), width=W, height=H, spp=128, max_bounce=3,
                     material=_material, sky=_sky, seed=99)
    t0 = time.time()
    clean, st = render_auto(_Scene(), _Cam(), W, H, _material, sky=_sky, quality="medium",
                            max_bounce=3, seed=0, return_stats=True)
    dt = time.time() - t0
    eq = int(round(st["mean_samples"]))
    raw = path_trace(_Scene(), _Cam(), width=W, height=H, spp=eq, max_bounce=3,
                     material=_material, sky=_sky, seed=0)
    p_auto, p_raw = _psnr(clean, ref), _psnr(raw, ref)
    print(f"render_auto     : {dt:8.2f}s to 'medium'          (mean {st['mean_samples']:.0f}spp over "
          f"{st['passes']} passes; {p_auto:.1f} dB vs raw@{eq}spp {p_raw:.1f} dB -> {p_auto-p_raw:+.1f} dB)")
    return dt, p_auto - p_raw


def bench_render_hair(n_strands=800, W=160, H=120):
    """Strand rasteriser throughput: shaded strand-SEGMENTS per second (the F3 hot spot to watch)."""
    from holographic_groom import groom
    from holographic_hairshade import render_hair
    from holographic_render import Camera
    from holographic_sdf import sphere
    strands = groom(sphere(1.0).eval, n_strands, ([-1.6] * 3, [1.6] * 3), length=0.5, n_pts=6, curl=0.5, seed=0)
    cam = Camera(eye=(0.0, 0.0, 3.2), target=(0.0, 0.0, 0.0), fov_deg=45.0, aspect=W / H)
    n_segments = sum(len(s.points) - 1 for s in strands)
    t0 = time.time()
    render_hair(strands, cam, width=W, height=H, shader="marschner", smooth_levels=1)
    dt = time.time() - t0
    print(f"render_hair     : {n_segments/dt/1e3:8.1f}k segments/sec         ({n_strands} strands, {W}x{H}, {dt:.2f}s)")
    return n_segments / dt


def bench_fluid(size=64, steps=30):
    """Smoke-sim throughput: StableFluid steps per second on a size^2 grid."""
    from holographic_fluid import StableFluid
    f = StableFluid((size, size), dt=0.05)
    f.add_source(region=(slice(size // 2 - 4, size // 2 + 4), slice(4, 12)), density=1.0, temperature=1.0)
    t0 = time.time()
    for _ in range(steps):
        f.step()
    dt = time.time() - t0
    print(f"StableFluid.step: {steps/dt:8.1f} steps/sec              ({size}x{size} smoke, {dt:.2f}s for {steps})")
    return steps / dt


def bench_volume_render(W=128, H=128, steps=64):
    """Volume-render throughput: field samples per second marching a smoke blob."""
    from holographic_render import volume_render, Camera
    def blob(P):
        P = np.asarray(P, float)
        return np.clip(1.0 - np.linalg.norm(P, axis=1) / 0.6, 0, 1)
    cam = Camera(eye=(1.4, 1.1, 2.4), target=(0, 0, 0), fov_deg=45.0)
    bounds = (np.array([-1.0, -1, -1]), np.array([1.0, 1, 1]))
    t0 = time.time()
    volume_render(blob, cam, bounds, W, H, steps=steps, mode="smoke", sigma=10.0)
    dt = time.time() - t0
    samples = W * H * steps
    print(f"volume_render   : {samples/dt/1e6:8.2f}M field-samples/sec     ({W}x{H} x {steps} steps, {dt:.2f}s)")
    return samples / dt


if __name__ == "__main__":
    print("render benches (single-thread NumPy; timings indicative, quality deltas deterministic)")
    print("-" * 100)
    bench_path_trace()
    bench_render_auto()
    bench_render_hair()
    bench_fluid()
    bench_volume_render()
