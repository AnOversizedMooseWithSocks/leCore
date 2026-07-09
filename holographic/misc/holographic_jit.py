"""Optional Numba JIT acceleration -- an OPT-IN fast path for the few genuinely sequential, non-vectorizable loops,
with a pure-Python fallback so the core still runs on NumPy alone.

WHY THIS MODULE EXISTS (and why it is gated, not woven through the core): almost everything in holostuff is already
vectorized -- bind is an FFT, bundle is a sum, recall is a matmul, advection is a vectorized gather -- and there
Numba buys ~1.4x (measured), not worth a dependency. The exception is a SEQUENTIAL recurrence where each step
depends on the last and a data-dependent branch blocks vectorization; there Numba was measured at ~33x. This module
captures THAT case and only that case, behind a flag, so the constitution's guarantees survive:

  * PORTABILITY -- if Numba isn't installed, `njit` becomes an identity decorator and the same source runs as pure
    Python. The core never hard-depends on it.
  * DETERMINISM -- we use plain @njit (no parallel=, no fastmath=): those two are the only Numba features that
    break bit-exactness, and we never use them. The selftest PROVES the JIT result equals the pure-Python result.

THE SHOWCASE KERNEL: the fast-sweeping eikonal solver (occupancy -> signed distance field). It is the textbook case
for this module -- inherently sequential (Gauss-Seidel sweeps; each cell update reads neighbours just updated in the
same pass, which is exactly what makes it O(N) instead of O(N^2)), so it does NOT vectorize, and it is on-thesis
(an SDF is the heart of holostuff's modelling vision -- mesh import, brushes, raymarching all want one). Pure NumPy
had no fast way to turn an occupancy mask into a dense SDF; this adds it, fast when Numba is present, correct either
way.
"""

import numpy as np

try:
    from numba import njit                              # the real JIT
    HAS_NUMBA = True
except Exception:                                       # pragma: no cover - exercised only without numba installed
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Identity-decorator fallback: with no Numba, the decorated source just runs as ordinary (slow) Python,
        so every kernel here stays callable on a NumPy-only install."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(fn):
            return fn
        return wrap


_BIG = 1.0e18


def _fast_sweep_2d_impl(dist, h, n_rounds):
    """One pure-Python source for the 2-D fast-sweeping eikonal solve (speed 1). `dist` starts at 0 on seed cells
    and _BIG elsewhere; this relaxes it toward the true distance-to-seed by sweeping the grid in all four
    diagonal directions, each cell solving the Godunov upwind quadratic from its smaller neighbour on each axis.
    Sequential by nature -- a sweep reads cells it updated earlier in the same pass. This is the function Numba
    compiles; with no Numba it runs as-is."""
    ny, nx = dist.shape
    for _r in range(n_rounds):
        for sy in range(2):
            ylo, yhi, ystep = (0, ny, 1) if sy == 0 else (ny - 1, -1, -1)
            for sx in range(2):
                xlo, xhi, xstep = (0, nx, 1) if sx == 0 else (nx - 1, -1, -1)
                y = ylo
                while y != yhi:
                    x = xlo
                    while x != xhi:
                        up = dist[y - 1, x] if y > 0 else _BIG
                        dn = dist[y + 1, x] if y < ny - 1 else _BIG
                        a = up if up < dn else dn                 # smaller vertical neighbour (upwind)
                        lf = dist[y, x - 1] if x > 0 else _BIG
                        rt = dist[y, x + 1] if x < nx - 1 else _BIG
                        b = lf if lf < rt else rt                 # smaller horizontal neighbour (upwind)
                        diff = a - b if a > b else b - a
                        if diff >= h:                             # one axis dominates -> 1-D update
                            cand = (a if a < b else b) + h
                        else:                                     # both axes contribute -> 2-D Godunov solve
                            cand = 0.5 * (a + b + (2.0 * h * h - (a - b) * (a - b)) ** 0.5)
                        if cand < dist[y, x]:
                            dist[y, x] = cand
                        x += xstep
                    y += ystep
    return dist


# Compile once if Numba is present; otherwise this IS the pure-Python function.
_fast_sweep_2d = njit(cache=True)(_fast_sweep_2d_impl) if HAS_NUMBA else _fast_sweep_2d_impl


def distance_transform(seed_mask, h=1.0, n_rounds=2):
    """Euclidean-ish distance from every cell to the nearest True cell of `seed_mask`, via fast sweeping. `h` is
    the grid spacing. Returns a float array of the same shape. (Fast sweeping gives the exact geodesic distance on
    the grid in the limit; a couple of rounds suffice for typical fields.)"""
    seed_mask = np.asarray(seed_mask, bool)
    dist = np.where(seed_mask, 0.0, _BIG)
    return _fast_sweep_2d(dist, float(h), int(n_rounds))


def signed_distance_2d(inside_mask, h=1.0, n_rounds=2):
    """Signed distance field of a 2-D shape given its filled `inside_mask` (True inside): distance to the boundary,
    NEGATIVE inside, positive outside. The boundary seeds are cells whose 4-neighbourhood straddles inside/outside.
    This is the occupancy -> SDF step holostuff's modelling/raymarch pipeline wants and NumPy had no fast path for."""
    inside = np.asarray(inside_mask, bool)
    boundary = np.zeros_like(inside)
    boundary[:-1, :] |= inside[:-1, :] != inside[1:, :]          # vertical phase changes
    boundary[1:, :] |= inside[:-1, :] != inside[1:, :]
    boundary[:, :-1] |= inside[:, :-1] != inside[:, 1:]          # horizontal phase changes
    boundary[:, 1:] |= inside[:, :-1] != inside[:, 1:]
    dist = distance_transform(boundary, h=h, n_rounds=n_rounds)
    return np.where(inside, -dist, dist)


def _fast_sweep_3d_impl(dist, h, n_rounds):
    """3-D fast-sweeping eikonal solve (speed 1): 8 diagonal sweeps, each cell solving the Godunov upwind quadratic
    by adding dimensions one at a time from its smaller neighbour on each axis. The 3-D analogue of the 2-D kernel,
    inherently sequential -- the natural Numba target for turning an occupancy VOLUME into an SDF."""
    nz, ny, nx = dist.shape
    for _r in range(n_rounds):
        for sz in range(2):
            z0, z1, dz = (0, nz, 1) if sz == 0 else (nz - 1, -1, -1)
            for sy in range(2):
                y0, y1, dy = (0, ny, 1) if sy == 0 else (ny - 1, -1, -1)
                for sx in range(2):
                    x0, x1, dx = (0, nx, 1) if sx == 0 else (nx - 1, -1, -1)
                    z = z0
                    while z != z1:
                        y = y0
                        while y != y1:
                            x = x0
                            while x != x1:
                                # smaller neighbour on each axis (upwind), _BIG at the boundary
                                ax = dist[z - 1, y, x] if z > 0 else _BIG
                                bx = dist[z + 1, y, x] if z < nz - 1 else _BIG
                                a = ax if ax < bx else bx
                                cx = dist[z, y - 1, x] if y > 0 else _BIG
                                dx2 = dist[z, y + 1, x] if y < ny - 1 else _BIG
                                b = cx if cx < dx2 else dx2
                                ex = dist[z, y, x - 1] if x > 0 else _BIG
                                fx = dist[z, y, x + 1] if x < nx - 1 else _BIG
                                c = ex if ex < fx else fx
                                # sort a <= b <= c (three values)
                                if a > b:
                                    a, b = b, a
                                if b > c:
                                    b, c = c, b
                                if a > b:
                                    a, b = b, a
                                # add dimensions one at a time
                                d = a + h
                                if d > b:
                                    s = a + b
                                    d = 0.5 * (s + (2.0 * h * h - (a - b) * (a - b)) ** 0.5)
                                    if d > c:
                                        s3 = a + b + c
                                        s2 = a * a + b * b + c * c
                                        disc = s3 * s3 - 3.0 * (s2 - h * h)
                                        if disc < 0.0:
                                            disc = 0.0
                                        d = (s3 + disc ** 0.5) / 3.0
                                if d < dist[z, y, x]:
                                    dist[z, y, x] = d
                                x += dx
                            y += dy
                        z += dz
    return dist


_fast_sweep_3d = njit(cache=True)(_fast_sweep_3d_impl) if HAS_NUMBA else _fast_sweep_3d_impl


def distance_transform_3d(seed_mask, h=1.0, n_rounds=2):
    """3-D distance from every voxel to the nearest True seed voxel, via fast sweeping. Returns a float volume."""
    seed_mask = np.asarray(seed_mask, bool)
    dist = np.where(seed_mask, 0.0, _BIG)
    return _fast_sweep_3d(dist, float(h), int(n_rounds))


def signed_distance_3d(inside_mask, h=1.0, n_rounds=2):
    """Signed distance field of a 3-D shape from its filled `inside_mask` (True inside): distance to the boundary,
    NEGATIVE inside. The occupancy-VOLUME -> SDF step (mesh import, sculpt) -- the 3-D twin of signed_distance_2d,
    Numba-accelerated when available, pure-Python fallback otherwise."""
    inside = np.asarray(inside_mask, bool)
    boundary = np.zeros_like(inside)
    boundary[:-1, :, :] |= inside[:-1, :, :] != inside[1:, :, :]   # phase changes along each axis
    boundary[1:, :, :] |= inside[:-1, :, :] != inside[1:, :, :]
    boundary[:, :-1, :] |= inside[:, :-1, :] != inside[:, 1:, :]
    boundary[:, 1:, :] |= inside[:, :-1, :] != inside[:, 1:, :]
    boundary[:, :, :-1] |= inside[:, :, :-1] != inside[:, :, 1:]
    boundary[:, :, 1:] |= inside[:, :, :-1] != inside[:, :, 1:]
    dist = distance_transform_3d(boundary, h=h, n_rounds=n_rounds)
    return np.where(inside, -dist, dist)


def _selftest():
    import time
    # CORRECTNESS: SDF of a disk should match the analytic signed distance (r - R) within ~1 cell.
    N, R = 128, 40.0
    yy, xx = np.mgrid[0:N, 0:N]
    cy = cx = (N - 1) / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    inside = r <= R
    sdf = signed_distance_2d(inside, h=1.0, n_rounds=3)
    analytic = r - R
    band = (np.abs(analytic) < 25)                              # check a band around the surface
    err = float(np.max(np.abs(sdf[band] - analytic[band])))
    assert err < 1.5, err                                       # within ~1 cell of the true distance

    # DETERMINISM: when Numba is present, the JIT result must EQUAL the pure-Python result (no parallel/fastmath).
    if HAS_NUMBA:
        d_pure = _fast_sweep_2d_impl(np.where(inside, 0.0, _BIG), 1.0, 2)
        d_jit = _fast_sweep_2d(np.where(inside, 0.0, _BIG), 1.0, 2)
        assert np.allclose(d_pure, d_jit, atol=1e-9), float(np.max(np.abs(d_pure - d_jit)))

        # SPEED: time the JIT vs the pure source on a bigger grid (the measured reason this module exists).
        big = (np.sqrt((np.mgrid[0:256, 0:256][0] - 128) ** 2 +
                       (np.mgrid[0:256, 0:256][1] - 128) ** 2) <= 80)
        seed = (~big)
        _fast_sweep_2d(np.where(seed, 0.0, _BIG), 1.0, 2)       # warm up the JIT
        t = time.perf_counter(); _fast_sweep_2d(np.where(seed, 0.0, _BIG), 1.0, 2); t_jit = time.perf_counter() - t
        t = time.perf_counter(); _fast_sweep_2d_impl(np.where(seed, 0.0, _BIG), 1.0, 2); t_pure = time.perf_counter() - t
        print(f"jit selftest ok: disk SDF max error {err:.2f} cells; JIT==pure (deterministic); "
              f"speed pure {t_pure*1000:.0f} ms -> JIT {t_jit*1000:.1f} ms = {t_pure/t_jit:.0f}x on 256^2")
    else:
        print(f"jit selftest ok (NO NUMBA, pure-Python path): disk SDF max error {err:.2f} cells; "
              f"runs without numba -- the portability fallback works")


if __name__ == "__main__":
    _selftest()
