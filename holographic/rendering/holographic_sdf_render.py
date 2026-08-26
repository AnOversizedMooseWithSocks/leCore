"""A fully-JIT'd renderer for ANALYTIC (symbolic) SDFs -- the end-to-end payoff of the SymPy -> Numba SDF.

The story so far: holostuff's renderer marches numpy rays calling sdf.eval(P) (a Python closure). Numba could never
accelerate that march because njit cannot cross into a Python closure. With sdf_numba_fn the SDF and its gradient are
njit functions, so the WHOLE march -- primary ray, exact normal, ambient occlusion, soft shadow -- compiles into one
njit kernel per pixel. Measured ~15x faster than the numpy render_sdf for a sphere at 200x200 with AO + soft shadows,
producing the same hit geometry. The Quilez field-native effects (AO = march along the normal; soft shadow = march
toward the light) port directly to scalar njit loops.

This is OPT-IN and SCOPED honestly. It covers the common, field-native shading -- Lambert direct light gated by a
soft shadow, ambient gated by AO, sky for misses. It does NOT implement the advanced render_sdf features
(physically-based Cook-Torrance, environment reflection/refraction, subsurface); render_sdf keeps the numpy path for
those and only routes here when the requested features are the basic set. The compiled renderer is cached per SDF
(the sympy lambdify + the Numba JIT are each paid once), and recompiles only when the SDF expression changes.

Needs sympy + numba (requirements-accel.txt). Without them, callers fall back to the numpy render_sdf.
"""

import numpy as np


def build_sdf_renderer(sdf_numba):
    """Given the dict from holographic_codegen.sdf_numba_fn (njit scalar value + grad), build and return a njit
    `render(O, D, L, base, ambient, do_ao, do_shadow)` that marches every ray and shades the hits. Closing over the
    njit SDF functions makes the SDF call INLINE in the compiled kernel (faster than passing it as an argument).
    Returns (img (M,3), hit (M,) bool)."""
    from numba import njit

    fv = sdf_numba["scalar_value"]
    snormal = sdf_numba["scalar_normal"]

    @njit
    def _march(ox, oy, oz, dx, dy, dz):
        t = 0.0
        for _ in range(96):
            dd = fv(ox + t * dx, oy + t * dy, oz + t * dz)
            if dd < 1e-3:
                return t, True
            t += dd
            if t > 20.0:
                break
        return t, False

    @njit
    def _ao(px, py, pz, nx, ny, nz):                         # Quilez AO: march along the normal a few taps
        occ = 0.0; sca = 1.0
        for i in range(1, 7):
            h = 0.06 * i
            dd = fv(px + nx * h, py + ny * h, pz + nz * h)
            occ += (h - dd) * sca
            sca *= 0.85
        v = 1.0 - 1.6 * occ
        return min(max(v, 0.0), 1.0)

    @njit
    def _shadow(px, py, pz, lx, ly, lz):                     # Quilez soft shadow: march toward the light
        res = 1.0; t = 0.02
        for _ in range(48):
            h = fv(px + lx * t, py + ly * t, pz + lz * t)
            res = min(res, 12.0 * max(h, 0.0) / max(t, 1e-6))
            t += min(max(h, 0.01), 0.25)
            if h < 1e-3 or t > 12.0:
                break
        return min(max(res, 0.0), 1.0)

    @njit
    def render(O, D, L, base, ambient, do_ao, do_shadow):
        M = O.shape[0]
        img = np.zeros((M, 3))
        hit = np.zeros(M, np.bool_)
        depth = np.full(M, 1e30)                             # ray-distance t at the hit; 1e30 = miss (for DOF)
        lx, ly, lz = L[0], L[1], L[2]
        for i in range(M):
            ox, oy, oz = O[i, 0], O[i, 1], O[i, 2]
            dx, dy, dz = D[i, 0], D[i, 1], D[i, 2]
            t, h = _march(ox, oy, oz, dx, dy, dz)
            if not h:
                continue
            hit[i] = True
            depth[i] = t
            px, py, pz = ox + t * dx, oy + t * dy, oz + t * dz
            nx, ny, nz = snormal(px, py, pz)
            lam = nx * lx + ny * ly + nz * lz
            if lam < 0.0:
                lam = 0.0
            sh = _shadow(px + nx * 2e-3, py + ny * 2e-3, pz + nz * 2e-3, lx, ly, lz) if do_shadow else 1.0
            ao = _ao(px, py, pz, nx, ny, nz) if do_ao else 1.0
            shade = ambient * ao + lam * sh
            img[i, 0] = base[0] * shade
            img[i, 1] = base[1] * shade
            img[i, 2] = base[2] * shade
        return img, hit, depth

    return render


def compiled_sdf_renderer(expr, variables=("x", "y", "z"), cache=None):
    """Compile (and cache) a njit renderer specialized to the symbolic SDF `expr`. Reuses the compiled renderer for
    the same SDF; recompiles when it changes. Built on the cached sdf_numba_fn."""
    from holographic.scene_and_pipeline.holographic_compile import compiled_sdf_numba, compiled

    def _compile(src):
        e, vs = src
        return build_sdf_renderer(compiled_sdf_numba(e, vs, cache=cache))

    return compiled((expr, tuple(variables)), _compile, tag="sdf_renderer", cache=cache)


def render_analytic(expr, camera, width=256, height=256, light_dir=(-0.4, 0.7, -0.3),
                    post=None, return_depth=False,
                    base_color=(0.85, 0.5, 0.35), ao=True, shadows=True, ambient=0.25, sky=None,
                    variables=("x", "y", "z")):
    """Drop-in fast renderer for an analytic SDF given as a symbolic expression. Marches + shades the hits in a njit
    kernel (Lambert + soft shadow + AO), composites the numpy sky dome for misses, returns (H, W, 3) in [0, 1].
    ~15x the numpy render_sdf on the field-native shading path. Needs sympy + numba."""
    from holographic.rendering.holographic_raymarch import sky_dome

    render = compiled_sdf_renderer(expr, variables)
    eye, dirs = camera.ray_dirs(width, height)
    D = np.ascontiguousarray(dirs.reshape(-1, 3))
    O = np.ascontiguousarray(np.broadcast_to(eye, D.shape))
    L = np.asarray(light_dir, float); L = L / (np.linalg.norm(L) + 1e-12)
    base = np.asarray(base_color, float)

    img, hit, depth = render(O, D, L, base, float(ambient), bool(ao), bool(shadows))
    bg = sky_dome(D, env=sky) if sky is not None else sky_dome(D)   # numpy sky for the misses (cheap, vectorised)
    out = np.where(hit[:, None], img, bg)
    frame = np.clip(out.reshape(height, width, 3), 0.0, 1.0)
    depth_img = depth.reshape(height, width)
    if post is not None:                                     # compose the post-processing PROGRAM onto the frame
        frame = post.apply(frame, depth=depth_img)
    if return_depth:
        return frame, depth_img
    return frame


def _selftest():
    try:
        import sympy  # noqa: F401
        from holographic.misc.holographic_jit import HAS_NUMBA
        if not HAS_NUMBA:
            raise ImportError
    except Exception:
        print("sdf_render selftest skipped (needs sympy + numba)")
        return
    import time
    from holographic.rendering.holographic_render import Camera
    from holographic.rendering.holographic_raymarch import sphere_trace

    class _Obj:
        def eval(self, P):
            return np.linalg.norm(np.asarray(P, float), axis=1) - 1.0

    cam = Camera(eye=(0, 0, 3.0), target=(0, 0, 0), fov_deg=50.0)
    W = H = 160
    img = render_analytic("sqrt(x**2+y**2+z**2) - 1.0", cam, width=W, height=H)
    assert img.shape == (H, W, 3) and img.min() >= 0.0 and img.max() <= 1.0

    # the njit hit geometry agrees with the numpy sphere_trace hit mask
    eye, dirs = cam.ray_dirs(W, H)
    D = dirs.reshape(-1, 3); O = np.broadcast_to(eye, D.shape)
    hit_np, _, _ = sphere_trace(_Obj(), O, D)
    from holographic.rendering.holographic_sdf_render import compiled_sdf_renderer as _r
    render = _r("sqrt(x**2+y**2+z**2) - 1.0")
    L = np.array([-0.4, 0.7, -0.3]); L = L / np.linalg.norm(L)
    _, hit_jit, _ = render(np.ascontiguousarray(O), np.ascontiguousarray(D), L,
                        np.array([0.85, 0.5, 0.35]), 0.25, True, True)
    agree = float(np.mean(hit_jit == hit_np))
    assert agree > 0.99, agree                                # hit geometry matches numpy to <1% of pixels

    # speed: time the njit renderer vs render_sdf on the same scene
    from holographic.rendering.holographic_raymarch import render_sdf
    t = time.perf_counter(); render_sdf(_Obj(), cam, width=W, height=H, ao=True, shadows=True, reflect=0.0); t_np = time.perf_counter() - t
    render_analytic("sqrt(x**2+y**2+z**2) - 1.0", cam, width=8, height=8)   # warm
    t = time.perf_counter(); render_analytic("sqrt(x**2+y**2+z**2) - 1.0", cam, width=W, height=H); t_jit = time.perf_counter() - t
    print(f"sdf_render selftest ok: njit renderer matches numpy hit geometry ({agree*100:.1f}%); "
          f"{W}x{H} sphere render numpy {t_np*1000:.0f} ms -> njit {t_jit*1000:.1f} ms = {t_np/t_jit:.0f}x")


if __name__ == "__main__":
    _selftest()
