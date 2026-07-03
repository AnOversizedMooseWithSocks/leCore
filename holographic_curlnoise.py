"""holographic_curlnoise.py -- #1 from the SIGGRAPH list: CURL NOISE. Turbulent-looking flow that is
divergence-free BY CONSTRUCTION, plus the boundary improvement (flow goes AROUND obstacles).

WHY THIS EXISTS (SIGGRAPH scouting list #1 -- the cheapest real win)
-------------------------------------------------------------------
Procedural turbulence you can add anywhere -- wind, smoke wisps, drifting detail -- normally has a problem: if
you just make up a velocity field from noise, it has DIVERGENCE (sources and sinks), so smoke bunches up and
thins out unphysically. Curl noise fixes this for free: take the CURL of a noise POTENTIAL, and the result is
divergence-free by a vector-calculus identity (div of a curl is exactly zero) -- incompressible turbulence with
NO fluid solve. This reuses the fBm noise we already have (`holographic_noise.FractalNoise`) and the field
operators we already have; it is a few lines, and it improves everything with a flow field (the fluid solver's
detail, the wind for the acoustics work, animated procedural fields).

THE METHOD (readable, first-principles)
---------------------------------------
In 2-D, pick a scalar STREAMFUNCTION psi(x,y) (here: fBm noise). The velocity
    u =  d(psi)/dy,   v = -d(psi)/dx
is divergence-free because div(u,v) = d2psi/dxdy - d2psi/dydx = 0 (mixed partials commute -- and they commute
DISCRETELY too for matching central differences, so our grid field is divergence-free to machine precision).

THE BOUNDARY IMPROVEMENT (Bridson-style, what the 2025 paper refines): to make the flow go AROUND an obstacle
instead of through it, RAMP the streamfunction to a constant (0) at the obstacle surface. A surface of constant
streamfunction IS a streamline, so no flow crosses it -- the velocity there is purely tangential. We ramp psi by
a smoothstep of the distance to the obstacle, so far away the noise is untouched and at the wall it is pinned.

HONEST SCOPE (kept negative): 2-D streamfunction curl noise (the 3-D version takes the curl of a VECTOR
potential -- a follow-up); the obstacle handling pins the streamfunction (no-penetration) but does not enforce
no-SLIP (tangential flow is unconstrained), which is the standard curl-noise trade-off. Deterministic; reuses
FractalNoise; NumPy + stdlib.
"""
import numpy as np


def _smoothstep(t):
    """The classic 3t^2 - 2t^3 smoothstep on [0,1] -- a soft 0->1 ramp with zero slope at both ends (so the
    obstacle ramp blends in without a crease)."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def streamfunction(res, bounds=((0.0, 8.0), (0.0, 8.0)), octaves=4, seed=0):
    """A scalar fBm streamfunction psi on a res x res grid, from the engine's FractalNoise -- the potential whose
    curl becomes the flow. `bounds` sets the spatial frequency (smaller box = larger swirls)."""
    from holographic_noise import FractalNoise
    fb = FractalNoise(2, dim=512, bounds=[tuple(bounds[0]), tuple(bounds[1])], octaves=octaves, seed=seed)
    return fb.sample_grid(int(res))


def curl_of_streamfunction(psi, dx=1.0):
    """The divergence-free velocity (u, v) = (d psi/dy, -d psi/dx) of a streamfunction grid. Divergence-free to
    machine precision because the discrete mixed partials commute."""
    dpsi_dy, dpsi_dx = np.gradient(psi, dx)                        # np.gradient -> [d/d(row)=d/dy, d/d(col)=d/dx]
    return dpsi_dy, -dpsi_dx


def divergence(u, v, dx=1.0):
    """The divergence d u/dx + d v/dy of a velocity field -- should be ~0 for a curl-noise field."""
    _, du_dx = np.gradient(u, dx)
    dv_dy, _ = np.gradient(v, dx)
    return du_dx + dv_dy


def curl_noise(res, bounds=((0.0, 8.0), (0.0, 8.0)), octaves=4, seed=0, obstacle_sdf=None, ramp=1.0, dx=1.0):
    """A divergence-free turbulence field (u, v) on a res x res grid. If `obstacle_sdf` (a callable point->signed
    distance, negative inside) is given, the streamfunction is ramped to 0 near the surface so the flow goes
    AROUND the obstacle (no penetration). `ramp` is the width of that blend in world units. Deterministic."""
    psi = streamfunction(res, bounds=bounds, octaves=octaves, seed=seed)
    if obstacle_sdf is not None:
        # build the distance field on the grid and ramp psi -> 0 within `ramp` of the surface (and inside it)
        (xlo, xhi), (ylo, yhi) = bounds
        xs = np.linspace(xlo, xhi, res); ys = np.linspace(ylo, yhi, res)
        gx, gy = np.meshgrid(xs, ys)                              # (res,res) world coords, [row=y, col=x]
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        d = np.asarray(obstacle_sdf(pts), float).reshape(res, res)
        mask = _smoothstep(d / max(ramp, 1e-9))                   # 0 at/inside the surface, 1 far outside
        psi = psi * mask                                          # pin the streamfunction on the obstacle -> a streamline
    u, v = curl_of_streamfunction(psi, dx=dx)
    return u, v


def curl_noise_3d(res, bounds=((0, 8), (0, 8), (0, 8)), octaves=3, seed=0, dx=1.0):
    """3-D curl noise: velocity = curl of a VECTOR potential (three fBm components). Divergence-free in 3-D.
    Returns (u, v, w) on a res^3 grid. (The 2-D streamfunction version is the common case; this is the 3-D
    companion.)"""
    from holographic_noise import FractalNoise
    b = [tuple(bounds[0]), tuple(bounds[1]), tuple(bounds[2])]
    P = [FractalNoise(3, dim=512, bounds=b, octaves=octaves, seed=seed + k).sample_grid(int(res)) for k in range(3)]
    Px, Py, Pz = P
    # curl(P) = (dPz/dy - dPy/dz, dPx/dz - dPz/dx, dPy/dx - dPx/dy)
    dPx = np.gradient(Px, dx); dPy = np.gradient(Py, dx); dPz = np.gradient(Pz, dx)  # each: [d/dz? order axis0,1,2]
    # axes order for a (res,res,res) grid from meshgrid default is (y,x,z)?  keep it simple: axis0,1,2 = a,b,c
    dPz_da, dPz_db, dPz_dc = dPz
    dPy_da, dPy_db, dPy_dc = dPy
    dPx_da, dPx_db, dPx_dc = dPx
    u = dPz_db - dPy_dc
    v = dPx_dc - dPz_da
    w = dPy_da - dPx_db
    return u, v, w


def _selftest():
    """Curl noise is divergence-free to machine precision, carries real turbulent structure, flows AROUND an
    obstacle when one is given, and is deterministic."""
    # (1) divergence-free BY CONSTRUCTION: max |divergence| is machine-tiny vs the velocity magnitude
    u, v = curl_noise(64, octaves=4, seed=0)
    speed = np.sqrt(u ** 2 + v ** 2)
    div = divergence(u, v)
    assert np.abs(div).max() < 1e-9 * max(speed.max(), 1e-9) + 1e-9, np.abs(div).max()
    assert speed.mean() > 0.0                                     # it actually flows (nonzero field)

    # (2) it has STRUCTURE (turbulent), not a constant: the velocity varies across the grid
    assert speed.std() > 0.1 * speed.mean()

    # (3) an OBSTACLE: a disk in the middle -> the flow does not penetrate it (normal velocity ~0 at the surface)
    cx = cy = 4.0; R = 1.5
    disk = lambda p: np.sqrt((p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2) - R      # SDF: negative inside the disk
    u2, v2 = curl_noise(96, octaves=4, seed=0, obstacle_sdf=disk, ramp=1.0)
    xs = np.linspace(0, 8, 96); ys = np.linspace(0, 8, 96)
    gx, gy = np.meshgrid(xs, ys)
    dist = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
    inside = dist < R * 0.8
    sp_in = np.sqrt(u2 ** 2 + v2 ** 2)[inside]
    sp_out = np.sqrt(u2 ** 2 + v2 ** 2)[dist > R * 1.5]
    assert sp_in.mean() < 0.2 * sp_out.mean()                     # flow is killed inside the obstacle (goes around)
    assert np.abs(divergence(u2, v2)).max() < 1e-6                # still divergence-free

    # (4) 3-D curl noise is divergence-free too
    u3, v3, w3 = curl_noise_3d(24, octaves=2, seed=1)
    da_u = np.gradient(u3)[0]; db_v = np.gradient(v3)[1]; dc_w = np.gradient(w3)[2]
    div3 = da_u + db_v + dc_w
    assert np.abs(div3).max() < 1e-9 * np.sqrt(u3 ** 2 + v3 ** 2 + w3 ** 2).max() + 1e-8

    # (5) deterministic
    a, b = curl_noise(32, seed=2); c, d = curl_noise(32, seed=2)
    assert np.array_equal(a, c) and np.array_equal(b, d)
    print("holographic_curlnoise selftest OK: curl of an fBm streamfunction is divergence-free to machine "
          "precision (max|div|=%.1e), carries turbulence, flows AROUND an obstacle (inside speed %.0f%% of "
          "outside), 3-D too; deterministic" % (np.abs(div).max(), 100 * sp_in.mean() / sp_out.mean()))


if __name__ == "__main__":
    _selftest()
