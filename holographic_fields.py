"""Grid fields + particle simulation, exposed to VSA -- the fluid/smoke/particle layer.

WHY THIS IS VSA-NATIVE (the Stam match). The engine's bind is circular convolution on a periodic domain,
done with the FFT. That is EXACTLY the machinery Jos Stam's "Stable Fluids" uses to diffuse and project a
velocity field on a torus. So the fluid steps here are not a bolt-on numerical solver -- the diffusion step
IS a bind with a Gaussian kernel, and the pressure projection is an FFT Helmholtz solve, the same transform
the algebra already runs. Fields (density / temperature / velocity / pressure) live on a periodic grid;
particles read those fields and feel forces. Everything is NumPy/FFT, deterministic.

WHAT IT COVERS (of the requested list):
  * density / temperature fields  -> scalar grids, diffused and advected.
  * velocity fields               -> vector grids (vx, vy), diffused, made divergence-free, self-advected.
  * pressure / divergence fields  -> divergence() and the FFT pressure-projection that removes it.
  * forces / attractors           -> point/field forces particles integrate against.
  * particles                     -> a ParticleSystem advected by a velocity field or driven by forces.
  * smoke / fluid                 -> fluid_step() (the Stable-Fluids loop) + density advection.

KEPT NEGATIVES (honest):
  * The grid is PERIODIC (a torus) -- there are no solid walls/boundaries; flow wraps around (this is exactly
    why the FFT applies, and the same periodicity binding assumes).
  * Semi-Lagrangian advection is numerically DIFFUSIVE -- a sharp blob smears over time (the classic
    Stable-Fluids trade for unconditional stability); it is not a high-order conservative scheme.
  * FFT diffusion is an exact Gaussian on the torus, so it cannot represent anisotropic/heterogeneous
    viscosity -- one global amount per step.
"""

import numpy as np

from holographic_transfer import scatter as _scatter, gather as _gather   # the shared bundle/readout primitive
import itertools


# ---------------------------------------------------------------------------
# Spectral helpers (the FFT-on-a-torus the bind operator already uses).
# ---------------------------------------------------------------------------

def _wavenumbers(shape):
    """Derivative-safe 2*pi*frequency grids (kx, ky) for the real FFT of an (H, W) periodic grid, plus a
    consistent k^2 (DC set to 1 to avoid divide-by-zero). We use rfft2/irfft2 so the inverse is exactly real.

    The Nyquist mode (frequency W/2 or H/2 on an EVEN grid) is zeroed here: a FIRST derivative of the Nyquist
    cosine is not representable on the grid (it has no matching sine), so i*k is ill-defined there. Leaving it
    in leaks divergence the pressure projection can't remove (measured: it was the entire residual). k^2 is
    built from these zeroed wavenumbers so the projection's k.v_new cancels to machine zero. (Diffusion is an
    EVEN-order operator -- the Nyquist IS representable for it -- so diffuse() uses the full k^2 instead.)"""
    H, W = shape
    ky = 2.0 * np.pi * np.fft.fftfreq(H)[:, None]       # (H, 1)
    kx = 2.0 * np.pi * np.fft.rfftfreq(W)[None, :]      # (1, W//2+1)
    if H % 2 == 0:
        ky[H // 2, 0] = 0.0                             # zero the Nyquist row (first-derivative undefined)
    if W % 2 == 0:
        kx[0, -1] = 0.0                                 # zero the Nyquist column
    k2 = kx ** 2 + ky ** 2
    k2[k2 == 0.0] = 1.0                                 # protect DC and the double-Nyquist corner from /0
    return kx, ky, k2


def diffuse(field, amount):
    """Diffuse a scalar (or component) field by the heat kernel, via the FFT -- multiply by exp(-amount*k^2)
    in Fourier space. This IS a bind with a Gaussian kernel on the torus (the engine's own operator): the
    same circular convolution, here used to spread heat/viscosity. The DC term (k=0) is multiplied by 1, so
    the total mass/mean is conserved exactly. Uses the FULL k^2 (Nyquist intact) -- diffusion is even-order,
    so the Nyquist mode is representable and should be damped, not dropped."""
    field = np.asarray(field, float)
    H, W = field.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(H)[:, None]
    kx = 2.0 * np.pi * np.fft.rfftfreq(W)[None, :]
    k2 = kx ** 2 + ky ** 2                              # DC is 0 here -> exp(0)=1 -> mass conserved
    return np.fft.irfft2(np.fft.rfft2(field) * np.exp(-amount * k2), s=field.shape)


def divergence(vx, vy):
    """The divergence field d = dvx/dx + dvy/dy, computed spectrally (i*k multiply). Nonzero divergence is
    compression/expansion; an incompressible (fluid) velocity field has divergence ~0 everywhere."""
    vx = np.asarray(vx, float); vy = np.asarray(vy, float)
    kx, ky, _ = _wavenumbers(vx.shape)
    return np.fft.irfft2(1j * kx * np.fft.rfft2(vx) + 1j * ky * np.fft.rfft2(vy), s=vx.shape)


def curl(vx, vy):
    """The scalar vorticity w = dvy/dx - dvx/dy (the rotational part of the flow), computed spectrally."""
    vx = np.asarray(vx, float); vy = np.asarray(vy, float)
    kx, ky, _ = _wavenumbers(vx.shape)
    return np.fft.irfft2(1j * kx * np.fft.rfft2(vy) - 1j * ky * np.fft.rfft2(vx), s=vx.shape)


def project_divergence_free(vx, vy):
    """Make a velocity field incompressible -- the PRESSURE PROJECTION of a fluid solver. Helmholtz says any
    field splits into a curl-free (gradient-of-pressure) part and a divergence-free part; this removes the
    gradient part in Fourier space (v_hat -= k (k . v_hat) / |k|^2), leaving divergence ~0. Done with the real
    FFT so the inverse is exactly real -- no symmetry-breaking truncation. Returns the projected (vx, vy)."""
    vx = np.asarray(vx, float); vy = np.asarray(vy, float)
    kx, ky, k2 = _wavenumbers(vx.shape)
    VX = np.fft.rfft2(vx); VY = np.fft.rfft2(vy)
    dot = (kx * VX + ky * VY) / k2                      # the offending gradient component, per frequency
    VX = VX - kx * dot; VY = VY - ky * dot
    return np.fft.irfft2(VX, s=vx.shape), np.fft.irfft2(VY, s=vx.shape)


# ---------------------------------------------------------------------------
# Advection (semi-Lagrangian backtrace -- stable, diffusive).
# ---------------------------------------------------------------------------

def _bilinear_periodic(field, x, y):
    """Sample `field` (H, W) at continuous coords (x=column, y=row) with periodic wrap and bilinear weights."""
    H, W = field.shape
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    fx = x - x0; fy = y - y0
    x0m = x0 % W; x1m = (x0 + 1) % W; y0m = y0 % H; y1m = (y0 + 1) % H
    return ((field[y0m, x0m] * (1 - fx) + field[y0m, x1m] * fx) * (1 - fy)
            + (field[y1m, x0m] * (1 - fx) + field[y1m, x1m] * fx) * fy)


def advect(field, vx, vy, dt):
    """Move a scalar field along a velocity field by semi-Lagrangian backtrace: each cell pulls its new value
    from where the flow came FROM (x - v*dt). Unconditionally stable; numerically diffusive (kept negative)."""
    H, W = field.shape
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return _bilinear_periodic(field, X - vx * dt, Y - vy * dt)


def fluid_step(vx, vy, density, dt=0.1, viscosity=0.0, fx=None, fy=None, source=None, solid=None):
    """One Stable-Fluids step on the torus: add forces -> diffuse velocity -> project divergence-free ->
    self-advect velocity -> project again -> advect density. Returns (vx, vy, density). Forces fx/fy and a
    density `source` are optional (H, W) fields. If `solid` (a 0/1 mask) is given, the flow is forced to go
    AROUND that obstacle and density cannot enter it. Built on the engine's FFT."""
    if fx is not None: vx = vx + dt * fx
    if fy is not None: vy = vy + dt * fy
    if viscosity > 0:
        vx = diffuse(vx, viscosity * dt); vy = diffuse(vy, viscosity * dt)
    vx, vy = project_divergence_free(vx, vy)
    vx, vy = advect(vx, vx, vy, dt), advect(vy, vx, vy, dt)
    vx, vy = project_divergence_free(vx, vy)
    if solid is not None:
        vx, vy = enforce_solid(vx, vy, solid)              # the flow diverts around the obstacle
    if source is not None: density = density + dt * source
    density = advect(density, vx, vy, dt)
    if solid is not None: density = density * (1.0 - solid)  # smoke cannot occupy the solid
    return vx, vy, density


# ---------------------------------------------------------------------------
# Particles, forces, attractors -- exposed to VSA (they read the grid fields / VSA-encoded forces).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Smoke: temperature-driven buoyancy + vorticity confinement (Fedkiw et al. 2001).
# ---------------------------------------------------------------------------

def buoyancy_force(temperature, density=None, alpha=0.0, beta=1.0, ambient=0.0):
    """Boussinesq buoyancy: hotter fluid RISES (a +y force proportional to temperature above ambient), heavier
    smoke SINKS (a -y force proportional to density). Returns (fx, fy) with fx=0 -- buoyancy is vertical. This
    is what turns a temperature field into motion, the heart of a smoke/convection sim. ('up' is +y = +row.)"""
    fy = beta * (np.asarray(temperature, float) - ambient)
    if density is not None:
        fy = fy - alpha * np.asarray(density, float)
    return np.zeros_like(fy), fy


def vorticity_confinement(vx, vy, epsilon=0.5):
    """Vorticity confinement (Fedkiw, Stam, Jensen 2001): semi-Lagrangian advection numerically DAMPS small
    vortices, so smoke goes mushy. This adds a force that pushes velocity back toward the existing vortex
    centres -- f = epsilon * (N x w), N = grad|w| / |grad|w|| pointing toward higher vorticity -- restoring the
    curl the advection lost. Returns (fx, fy). Larger epsilon = curlier smoke."""
    w = curl(vx, vy)                                        # scalar z-vorticity
    gy, gx = np.gradient(np.abs(w))                         # grad of |w| (axis0=y, axis1=x)
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-12
    nx = gx / mag; ny = gy / mag                            # unit vector toward higher vorticity
    return epsilon * (ny * w), epsilon * (-nx * w)          # N x w in 2-D


def smoke_step(vx, vy, density, temperature, dt=0.1, viscosity=0.0, ambient=0.0,
               buoyancy=1.0, gravity=0.0, confinement=0.0, dens_source=None, temp_source=None, solid=None):
    """One smoke step: inject sources, apply buoyancy (from temperature) + vorticity confinement to the
    velocity, then the usual diffuse -> project -> advect, carrying BOTH density and temperature along. If
    `solid` (a 0/1 mask) is given, smoke rises and curls AROUND the obstacle. Returns (vx, vy, density,
    temperature). A hot source at the bottom gives rising, curling smoke -- all on the FFT fluid solver the
    bind operator already provides."""
    if dens_source is not None:
        density = density + dt * np.asarray(dens_source, float)
    if temp_source is not None:
        temperature = temperature + dt * np.asarray(temp_source, float)
    fx, fy = buoyancy_force(temperature, density, alpha=gravity, beta=buoyancy, ambient=ambient)
    if confinement > 0:
        cfx, cfy = vorticity_confinement(vx, vy, epsilon=confinement)
        fx = fx + cfx; fy = fy + cfy
    vx = vx + dt * fx; vy = vy + dt * fy
    if viscosity > 0:
        vx = diffuse(vx, viscosity * dt); vy = diffuse(vy, viscosity * dt)
    vx, vy = project_divergence_free(vx, vy)
    vx, vy = advect(vx, vx, vy, dt), advect(vy, vx, vy, dt)
    vx, vy = project_divergence_free(vx, vy)
    if solid is not None:
        vx, vy = enforce_solid(vx, vy, solid)
    density = advect(density, vx, vy, dt)
    temperature = advect(temperature, vx, vy, dt)
    if solid is not None:
        density = density * (1.0 - solid); temperature = temperature * (1.0 - solid)
    return vx, vy, density, temperature


# ---------------------------------------------------------------------------
# Immersed boundary: solid OBSTACLES the flow goes around (not just momentum sources).
# ---------------------------------------------------------------------------

def disc_mask(shape, center, radius):
    """A circular solid mask (1.0 inside the disc, 0.0 outside) on an (H, W) grid -- a round obstacle."""
    H, W = shape
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return ((X - center[0]) ** 2 + (Y - center[1]) ** 2 <= radius ** 2).astype(float)


def enforce_solid(vx, vy, solid_mask, solid_vx=0.0, solid_vy=0.0, iters=2):
    """Make the flow respect a SOLID obstacle: inside the mask, force the fluid velocity to the solid's own
    velocity (0 for a static obstacle), then re-project to divergence-free so the displaced flow goes AROUND
    the solid -- repeating a couple of times, since each projection slightly re-leaks velocity into the solid
    (this alternation is the immersed-boundary idea; on a periodic FFT grid it is an approximate, not exact,
    no-slip -- kept negative). Returns (vx, vy)."""
    keep = 1.0 - solid_mask
    for _ in range(iters):
        vx = vx * keep + solid_vx * solid_mask
        vy = vy * keep + solid_vy * solid_mask
        vx, vy = project_divergence_free(vx, vy)
    return vx, vy


class ParticleSystem:
    """N particles with positions and velocities on the same periodic grid as the fields. They can be pushed
    by forces (gravity, attractors, or any (N,2) force array a VSA program supplies) and advected by a
    velocity field they SAMPLE -- so smoke particles ride the fluid solved above."""

    def __init__(self, positions, velocities=None):
        self.pos = np.asarray(positions, float)               # (N, 2): columns are (x, y)
        self.vel = (np.zeros_like(self.pos) if velocities is None else np.asarray(velocities, float))

    def step(self, force=None, dt=0.1, damping=0.0, wrap_to=None, collider=None, collide_radius=0.0):
        """Integrate one step (semi-implicit Euler). `force` is an (N, 2) acceleration array. `collider` (a callable
        P->signed distance) is an obstacle the particles cannot enter: any particle inside it is pushed back out to
        `collide_radius` (the same SDF-collision resolve the softbody uses -- so 2-D particles avoid scene geometry)."""
        if force is not None:
            self.vel = self.vel + dt * np.asarray(force, float)
        if damping:
            self.vel *= (1.0 - damping)
        self.pos = self.pos + dt * self.vel
        if collider is not None:                                  # environment collision: keep particles outside it
            from holographic_collide import resolve_sdf_collision
            self.pos = resolve_sdf_collision(self.pos, collider, radius=collide_radius)
        if wrap_to is not None:
            self.pos = np.mod(self.pos, np.asarray(wrap_to, float))
        return self

    def advect_by(self, vx, vy, dt=0.1):
        """Move the particles by the velocity field they sit in (bilinear sample at each particle)."""
        v = sample_field(vx, self.pos); w = sample_field(vy, self.pos)
        self.pos = self.pos + dt * np.stack([v, w], axis=1)
        self.pos[:, 0] %= vx.shape[1]; self.pos[:, 1] %= vx.shape[0]
        return self


def attractor_force(positions, center, strength=1.0, softening=1.0):
    """A force pulling every particle toward `center` (an inverse-distance well, softened so it stays finite
    at the center). Negative strength repels. Returns an (N, 2) force array."""
    d = np.asarray(center, float)[None, :] - np.asarray(positions, float)
    r2 = (d ** 2).sum(axis=1, keepdims=True) + softening ** 2
    return strength * d / r2


def sample_field(field, positions):
    """Read a scalar grid `field` (H, W) at continuous particle positions (N, 2) = (x, y), bilinear+periodic.
    This is how particles feel a VSA-encoded or solved field (velocity component, density, temperature).
    This is a GATHER = the readout of a bundle; it delegates to the shared holographic_transfer.gather (fields
    uses (x=col, y=row) -> grid[y, x], so the coords are swapped to (row, col))."""
    p = np.asarray(positions, float)
    return _gather(field, p[:, ::-1], kernel="bilinear", periodic=True)


def scatter_to_field(shape, positions, values):
    """The ADJOINT of sample_field: accumulate per-particle `values` (N,) onto a grid of `shape` (H, W) at
    continuous positions (N, 2)=(x, y), spreading each value bilinearly over its four nearest cells (periodic).
    This is how particles IMPRINT onto a field -- e.g. a moving cloth depositing its momentum into the fluid
    velocity grid (the cloth->fluid half of two-way coupling).
    This is a SCATTER = a bundle (superposition of kernel-weighted contributions); it delegates to the shared
    holographic_transfer.scatter (coords swapped (x,y)->(row,col) to match grid[y, x])."""
    p = np.asarray(positions, float)
    return _scatter(p[:, ::-1], np.asarray(values, float), shape, kernel="bilinear", periodic=True)


def drag_force(positions, velocities, vx, vy, k=1.0):
    """Drag force on particles from a fluid: F = k * (v_fluid_at_particle - v_particle), with the fluid
    velocity sampled at each particle. This is the fluid->cloth half of two-way coupling -- the flow pushes
    the body. Returns an (N, 2) force array."""
    p = np.asarray(positions, float); v = np.asarray(velocities, float)
    vf = np.stack([sample_field(vx, p), sample_field(vy, p)], axis=1)
    return k * (vf - v)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3-D fluid + smoke. The exact same operators, generalised to a 3-D periodic grid via the n-D real FFT --
# the bind operator's circular convolution is dimension-agnostic, so the fluid solver is too. Grid shape is
# (Nx, Ny, Nz); velocity components (vx, vy, vz) map to axes (0, 1, 2); 'up' for buoyancy is +y (axis 1).
# ---------------------------------------------------------------------------

def _wavenumbers_3d(shape):
    """Derivative-safe 2*pi*frequency grids (kx, ky, kz) for the 3-D real FFT, plus a consistent k^2. Same
    Nyquist care as 2-D: a first derivative of the Nyquist mode is undefined on an even grid, so it is zeroed,
    and k^2 is built from the zeroed wavenumbers so the pressure projection cancels to machine zero."""
    Nx, Ny, Nz = shape
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx)[:, None, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny)[None, :, None]
    kz = 2.0 * np.pi * np.fft.rfftfreq(Nz)[None, None, :]
    if Nx % 2 == 0: kx[Nx // 2, 0, 0] = 0.0
    if Ny % 2 == 0: ky[0, Ny // 2, 0] = 0.0
    if Nz % 2 == 0: kz[0, 0, -1] = 0.0
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[k2 == 0.0] = 1.0
    return kx, ky, kz, k2


def _trilinear_periodic(field, x, y, z):
    """Sample a 3-D `field` (Nx, Ny, Nz) at continuous coords (x, y, z) with periodic wrap and trilinear
    weights (the 3-D analogue of the bilinear sampler)."""
    Nx, Ny, Nz = field.shape
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); z0 = np.floor(z).astype(int)
    fx = x - x0; fy = y - y0; fz = z - z0
    x0 %= Nx; y0 %= Ny; z0 %= Nz
    x1 = (x0 + 1) % Nx; y1 = (y0 + 1) % Ny; z1 = (z0 + 1) % Nz
    out = 0.0
    for xi, wx in ((x0, 1 - fx), (x1, fx)):
        for yi, wy in ((y0, 1 - fy), (y1, fy)):
            for zi, wz in ((z0, 1 - fz), (z1, fz)):
                out = out + field[xi, yi, zi] * wx * wy * wz
    return out


def diffuse_3d(field, amount):
    """3-D heat-kernel diffusion via the FFT -- a Gaussian bind on the 3-D torus. Mass conserved (DC gain 1).
    Uses the full k^2 (Nyquist intact) since diffusion is even-order."""
    field = np.asarray(field, float)
    Nx, Ny, Nz = field.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx)[:, None, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny)[None, :, None]
    kz = 2.0 * np.pi * np.fft.rfftfreq(Nz)[None, None, :]
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    return np.fft.irfftn(np.fft.rfftn(field) * np.exp(-amount * k2), s=field.shape, axes=(0, 1, 2))


def divergence_3d(vx, vy, vz):
    """3-D divergence dvx/dx + dvy/dy + dvz/dz, computed spectrally."""
    kx, ky, kz, _ = _wavenumbers_3d(vx.shape)
    return np.fft.irfftn(1j * kx * np.fft.rfftn(vx) + 1j * ky * np.fft.rfftn(vy)
                         + 1j * kz * np.fft.rfftn(vz), s=vx.shape, axes=(0, 1, 2))


def curl_3d(vx, vy, vz):
    """3-D vorticity vector omega = curl(v) = (dvz/dy - dvy/dz, dvx/dz - dvz/dx, dvy/dx - dvx/dy)."""
    kx, ky, kz, _ = _wavenumbers_3d(vx.shape)
    VX = np.fft.rfftn(vx); VY = np.fft.rfftn(vy); VZ = np.fft.rfftn(vz)
    wx = np.fft.irfftn(1j * ky * VZ - 1j * kz * VY, s=vx.shape, axes=(0, 1, 2))
    wy = np.fft.irfftn(1j * kz * VX - 1j * kx * VZ, s=vx.shape, axes=(0, 1, 2))
    wz = np.fft.irfftn(1j * kx * VY - 1j * ky * VX, s=vx.shape, axes=(0, 1, 2))
    return wx, wy, wz


def project_divergence_free_3d(vx, vy, vz):
    """3-D pressure projection (Helmholtz): remove the gradient part so divergence ~ 0, via the real FFT."""
    kx, ky, kz, k2 = _wavenumbers_3d(vx.shape)
    VX = np.fft.rfftn(vx); VY = np.fft.rfftn(vy); VZ = np.fft.rfftn(vz)
    dot = (kx * VX + ky * VY + kz * VZ) / k2
    VX = VX - kx * dot; VY = VY - ky * dot; VZ = VZ - kz * dot
    return (np.fft.irfftn(VX, s=vx.shape, axes=(0, 1, 2)), np.fft.irfftn(VY, s=vx.shape, axes=(0, 1, 2)), np.fft.irfftn(VZ, s=vx.shape, axes=(0, 1, 2)))


def advect_3d(field, vx, vy, vz, dt):
    """Semi-Lagrangian advection of a 3-D scalar field along a 3-D velocity field (trilinear, periodic)."""
    Nx, Ny, Nz = field.shape
    X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
    return _trilinear_periodic(field, X - vx * dt, Y - vy * dt, Z - vz * dt)


def fluid_step_3d(vx, vy, vz, density, dt=0.1, viscosity=0.0, fx=None, fy=None, fz=None, source=None,
                  solid=None):
    """One 3-D Stable-Fluids step: add forces -> diffuse -> project -> self-advect -> project -> advect
    density. If `solid` (a 0/1 (Nx,Ny,Nz) mask, e.g. from sphere_mask) is given, the flow is forced AROUND
    the obstacle and density cannot enter it (the immersed boundary lifted into 3-D). Returns (vx, vy, vz,
    density)."""
    if fx is not None: vx = vx + dt * fx
    if fy is not None: vy = vy + dt * fy
    if fz is not None: vz = vz + dt * fz
    if viscosity > 0:
        vx = diffuse_3d(vx, viscosity * dt); vy = diffuse_3d(vy, viscosity * dt); vz = diffuse_3d(vz, viscosity * dt)
    vx, vy, vz = project_divergence_free_3d(vx, vy, vz)
    vx, vy, vz = (advect_3d(vx, vx, vy, vz, dt), advect_3d(vy, vx, vy, vz, dt), advect_3d(vz, vx, vy, vz, dt))
    vx, vy, vz = project_divergence_free_3d(vx, vy, vz)
    if solid is not None:
        vx, vy, vz = enforce_solid_3d(vx, vy, vz, solid)       # the flow diverts around the 3-D obstacle
    if source is not None: density = density + dt * source
    density = advect_3d(density, vx, vy, vz, dt)
    if solid is not None: density = density * (1.0 - solid)     # smoke cannot occupy the solid
    return vx, vy, vz, density


def smoke_step_3d(vx, vy, vz, density, temperature, dt=0.1, viscosity=0.0, ambient=0.0,
                  buoyancy=1.0, gravity=0.0, confinement=0.0, dens_source=None, temp_source=None,
                  solid=None):
    """One 3-D smoke step: temperature drives velocity by buoyancy along +y (hot rises), optional 3-D vorticity
    confinement keeps it curly, and density + temperature advect with the flow. Returns
    (vx, vy, vz, density, temperature)."""
    if dens_source is not None: density = density + dt * np.asarray(dens_source, float)
    if temp_source is not None: temperature = temperature + dt * np.asarray(temp_source, float)
    fy = buoyancy * (temperature - ambient) - gravity * density        # buoyancy is along +y (axis 1)
    vy = vy + dt * fy
    if confinement > 0:
        wx, wy, wz = curl_3d(vx, vy, vz)
        mag = np.sqrt(wx ** 2 + wy ** 2 + wz ** 2)
        gx, gy, gz = np.gradient(mag)
        gm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-12
        nx, ny, nz = gx / gm, gy / gm, gz / gm
        vx = vx + dt * confinement * (ny * wz - nz * wy)               # N x omega
        vy = vy + dt * confinement * (nz * wx - nx * wz)
        vz = vz + dt * confinement * (nx * wy - ny * wx)
    if viscosity > 0:
        vx = diffuse_3d(vx, viscosity * dt); vy = diffuse_3d(vy, viscosity * dt); vz = diffuse_3d(vz, viscosity * dt)
    vx, vy, vz = project_divergence_free_3d(vx, vy, vz)
    vx, vy, vz = (advect_3d(vx, vx, vy, vz, dt), advect_3d(vy, vx, vy, vz, dt), advect_3d(vz, vx, vy, vz, dt))
    vx, vy, vz = project_divergence_free_3d(vx, vy, vz)
    if solid is not None:
        vx, vy, vz = enforce_solid_3d(vx, vy, vz, solid)       # smoke rises and curls AROUND the 3-D obstacle
    density = advect_3d(density, vx, vy, vz, dt)
    temperature = advect_3d(temperature, vx, vy, vz, dt)
    if solid is not None:
        density = density * (1.0 - solid); temperature = temperature * (1.0 - solid)
    return vx, vy, vz, density, temperature


def _expand_ranges(lo, hi):
    """Flatten a set of ragged integer ranges [lo_i, hi_i) into one index array, vectorised: for ranges of
    lengths cnt = hi - lo, return lo repeated with an intra-range 0..cnt-1 counter added. (The standard numpy
    ragged-range trick -- no Python per-range loop.)"""
    cnt = hi - lo
    total = int(cnt.sum())
    if total == 0:
        return np.empty(0, np.int64)
    starts = np.repeat(lo, cnt)                                 # each candidate's range start, repeated
    inner = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)   # 0,1,..,cnt-1 within each range
    return starts + inner


def spatial_hash_pairs(positions, radius):
    """Find every index pair (i<j) whose points lie within `radius` -- the 'cull, don't batch' primitive,
    VECTORISED (no Python per-pair or per-point loop).

    A uniform grid of cell size `radius` means two points within `radius` are at most one cell apart per axis,
    so every close pair lives in one of the 3^D cell offsets. We linear-index each point's cell, SORT by that
    key, and for each of the (3^D+1)/2 canonical offsets use searchsorted to find, for every point at once, the
    contiguous block of points sitting in the target neighbour cell -- then expand those ragged blocks into
    candidate pairs and filter by true distance, all in array ops. The only Python loop left is over the 3^D
    offsets (a tiny constant, not per-point) -- the per-point/per-pair work that was the bottleneck is gone.
    Cost O(N log N + pairs). Works in ANY dimension. Returns an int array (P, 2) of close pairs, sorted by
    (i, j) for determinism.

    Reusable wherever local neighbours are needed -- softbody self-collision, particle interaction (the
    short-range n-body force) -- the same lesson the mesh-distance and sculpting work kept re-learning: a
    spatial index that CULLS work, kept in array-land, beats batching a dense O(N^2) reduction."""
    pts = np.asarray(positions, float)
    N, D = pts.shape
    if N < 2 or radius <= 0:
        return np.empty((0, 2), int)
    cell = np.floor(pts / radius).astype(np.int64)
    cell -= cell.min(axis=0) - 1                                # shift so min cell coord is 1 (pad below)
    dims = cell.max(axis=0) + 2                                 # pad above -> cell+/-1 stays in [0, dims), no alias
    strides = np.ones(D, np.int64)
    for d in range(1, D):
        strides[d] = strides[d - 1] * dims[d - 1]
    key = (cell * strides).sum(axis=1)                         # (N,) unique linear cell index
    order = np.argsort(key, kind="stable")
    key_s = key[order]                                         # sorted cell keys
    r2 = float(radius) ** 2
    out_i, out_j = [], []
    for off in itertools.product((-1, 0, 1), repeat=D):        # tiny fixed loop over the 3^D neighbour block
        off = np.array(off, np.int64)
        nz = np.nonzero(off)[0]
        if nz.size and off[nz[0]] < 0:                        # keep only the canonical half (its mirror covers it)
            continue
        target = key + int((off * strides).sum())             # (N,) the neighbour cell key for each source i
        lo = np.searchsorted(key_s, target, side="left")
        hi = np.searchsorted(key_s, target, side="right")
        cnt = hi - lo
        if cnt.sum() == 0:
            continue
        src = np.repeat(np.arange(N), cnt)                     # source point i (repeated per candidate)
        dst = order[_expand_ranges(lo, hi)]                   # candidate point j (original index)
        keep = (src < dst) if not nz.size else (src != dst)   # off==0: each in-cell pair once; else all
        i_idx, j_idx = src[keep], dst[keep]
        d2 = ((pts[i_idx] - pts[j_idx]) ** 2).sum(axis=1)     # true distance filter (cells overlap the radius)
        m = d2 <= r2
        out_i.append(i_idx[m]); out_j.append(j_idx[m])
    if not out_i:
        return np.empty((0, 2), int)
    I = np.concatenate(out_i); J = np.concatenate(out_j)
    lo_ij = np.minimum(I, J); hi_ij = np.maximum(I, J)         # canonical (i<j); each pair appears exactly once
    out = np.stack([lo_ij, hi_ij], axis=1)
    order2 = np.lexsort((out[:, 1], out[:, 0]))                # deterministic (i, j) order
    return out[order2]


def pairwise_repulsion(positions, radius, strength=1.0):
    """Short-range repulsion between nearby particles -- the n-body short-range force, CULLED by the spatial
    hash and accumulated as a SCATTER (no Python per-pair loop). For every pair within `radius`, a soft-sphere
    push that falls linearly to zero at the radius: F = strength * (1 - d/radius) * (p_i - p_j)/d, summed per
    particle via np.add.at (the same adjoint/scatter the field coupling uses). Returns (N, D), to be passed to
    ParticleSystem.step(force=...) like attractor_force / drag_force. Only close pairs contribute, and the
    whole computation stays in array-land: O(N log N + pairs), no boundary crossing. Any dimension.
    (granular piles, collision avoidance, flocking separation)."""
    pts = np.asarray(positions, float)
    N, D = pts.shape
    F = np.zeros((N, D))
    if radius <= 0:
        return F
    pairs = spatial_hash_pairs(pts, radius)
    if pairs.shape[0] == 0:
        return F
    i, j = pairs[:, 0], pairs[:, 1]
    d = pts[i] - pts[j]                                         # (P, D) all pair separations at once
    dist = np.sqrt((d ** 2).sum(axis=1))                       # (P,)
    good = dist > 1e-12                                        # guard coincident points
    f = np.zeros_like(d)
    f[good] = (strength * (1.0 - dist[good] / radius))[:, None] * d[good] / dist[good][:, None]
    np.add.at(F, i, f)                                          # scatter the push onto each particle ...
    np.add.at(F, j, -f)                                        # ... and its equal-and-opposite onto the other
    return F


def sphere_mask(shape, center, radius):
    """A spherical solid mask (1.0 inside the ball, 0.0 outside) on an (Nx, Ny, Nz) grid -- the 3-D lift of
    disc_mask. `center` is (cx, cy, cz) in grid coords (axes 0,1,2)."""
    Nx, Ny, Nz = shape
    X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
    return ((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2 <= radius ** 2).astype(float)


def enforce_solid_3d(vx, vy, vz, solid_mask, solid_vx=0.0, solid_vy=0.0, solid_vz=0.0, iters=2):
    """The 3-D immersed boundary (disc_mask/enforce_solid lifted to the ball): inside the mask, force the fluid
    velocity to the solid's own velocity, then re-project divergence-free so the displaced flow goes AROUND the
    solid -- repeated a couple of times since each projection slightly re-leaks into the solid. On the periodic
    FFT grid this is an approximate, not exact, no-slip (kept negative). Returns (vx, vy, vz)."""
    keep = 1.0 - solid_mask
    for _ in range(iters):
        vx = vx * keep + solid_vx * solid_mask
        vy = vy * keep + solid_vy * solid_mask
        vz = vz * keep + solid_vz * solid_mask
        vx, vy, vz = project_divergence_free_3d(vx, vy, vz)
    return vx, vy, vz


def sample_field_3d(field, positions):
    """Read a 3-D scalar grid `field` (Nx,Ny,Nz) at continuous particle positions (N,3)=(x,y,z), trilinear +
    periodic -- the 3-D lift of sample_field. How particles/softbody nodes feel a 3-D solved or VSA-encoded
    field (a velocity component, density, temperature)."""
    p = np.asarray(positions, float)
    return _trilinear_periodic(field, p[:, 0], p[:, 1], p[:, 2])


def scatter_to_field_3d(shape, positions, values):
    """The ADJOINT of sample_field_3d (the 3-D lift of scatter_to_field): accumulate per-particle `values` (N,)
    onto a grid of `shape` (Nx,Ny,Nz) at continuous positions (N,3), spreading each value trilinearly over its
    eight nearest cells (periodic). How a moving 3-D body IMPRINTS onto a field (the body->fluid half of two-way
    coupling)."""
    Nx, Ny, Nz = shape
    out = np.zeros((Nx, Ny, Nz))
    p = np.asarray(positions, float); vals = np.asarray(values, float)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); z0 = np.floor(z).astype(int)
    fx = x - x0; fy = y - y0; fz = z - z0
    for xi, wx in ((x0 % Nx, 1 - fx), ((x0 + 1) % Nx, fx)):     # the eight trilinear weights, summed in
        for yi, wy in ((y0 % Ny, 1 - fy), ((y0 + 1) % Ny, fy)):
            for zi, wz in ((z0 % Nz, 1 - fz), ((z0 + 1) % Nz, fz)):
                np.add.at(out, (xi, yi, zi), vals * wx * wy * wz)
    return out


def drag_force_3d(positions, velocities, vx, vy, vz, k=1.0):
    """Drag on particles/nodes from a 3-D fluid: F = k*(v_fluid_at_node - v_node), the fluid sampled trilinearly
    at each node -- the 3-D lift of drag_force (the fluid->body half of two-way coupling). Returns (N,3). A
    softbody couples to fluid_step_3d exactly as it does to the 2-D solver: pass this as external_force."""
    p = np.asarray(positions, float); v = np.asarray(velocities, float)
    vf = np.stack([sample_field_3d(vx, p), sample_field_3d(vy, p), sample_field_3d(vz, p)], axis=1)
    return k * (vf - v)


def spectral_field(shape, beta=2.0, seed=0):
    """Synthesise a SEAMLESS FRACTAL field directly in the Fourier domain -- the demoscene 'procedural volume
    from a tiny seed' move, done on the engine's own FFT. Each frequency gets amplitude |k|^(-beta/2) (a 1/f^beta
    power spectrum -> fractal/fBm) and a random phase; the inverse real FFT is a zero-mean, unit-std field that
    is PERIODIC by construction, so it TILES with no seam (the torus is its native domain). Works in any
    dimension (2-D or 3-D from `shape`). beta controls roughness: ~0 is white/rough, larger is smoother (more
    low-frequency). The whole volume is reproducible from just (shape, beta, seed) -- that compression IS the
    point: a few numbers, a rich seamless volume, composable with tiling / the fluid solver / the archive."""
    shape = tuple(int(s) for s in shape)
    axes = tuple(range(len(shape)))
    ks = []
    for ax, n in enumerate(shape):
        f = np.fft.rfftfreq(n) if ax == len(shape) - 1 else np.fft.fftfreq(n)
        sh = [1] * len(shape); sh[ax] = f.shape[0]
        ks.append(f.reshape(sh))
    kmag = np.sqrt(sum(k ** 2 for k in ks))
    amp = np.zeros_like(kmag)
    nz = kmag > 0
    amp[nz] = kmag[nz] ** (-beta / 2.0)                    # 1/f^beta power -> fractal
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=amp.shape)
    spectrum = amp * (np.cos(phase) + 1j * np.sin(phase))
    field = np.fft.irfftn(spectrum, s=shape, axes=axes)
    field = field - field.mean()
    sd = field.std()
    return field / sd if sd > 0 else field


def seam_continuity(field, axis=0):
    """How seamlessly a field TILES along `axis`: the mean jump across the periodic wrap (last cell -> first
    cell) divided by the mean jump between adjacent interior cells. ~1.0 means the wrap looks like any other
    step -> seamless; >> 1 means a visible seam. A torus-synthesised field (spectral_field, or any FFT solve)
    is ~1; a non-periodic field is not."""
    f = np.moveaxis(np.asarray(field, float), axis, 0)
    wrap = np.abs(f[0] - f[-1]).mean()
    interior = np.abs(np.diff(f, axis=0)).mean()
    return float(wrap / (interior + 1e-12))


def _selftest():
    rng = np.random.default_rng(0)
    H = W = 48

    def blob(cx, cy, s=4.0):
        Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        return np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s ** 2)))

    # DIFFUSE: mean (mass) conserved exactly; high-frequency energy drops.
    f = blob(24, 24)
    d = diffuse(f, amount=4.0)
    assert abs(f.mean() - d.mean()) < 1e-9, "diffusion must conserve mass (DC)"
    assert d.var() < f.var(), "diffusion must smooth (variance drops)"

    # PROJECTION: a random velocity field has large divergence; after projection it is ~0.
    vx = rng.normal(size=(H, W)); vy = rng.normal(size=(H, W))
    before = np.abs(divergence(vx, vy)).max()
    px, py = project_divergence_free(vx, vy)
    after = np.abs(divergence(px, py)).max()
    assert after < before * 1e-6, f"projection must remove divergence: {before:.3f} -> {after:.2e}"

    # ADVECT: a blob moves by ~v*dt under a uniform flow.
    dens = blob(12, 24)
    U = np.full((H, W), 3.0); V = np.zeros((H, W))
    moved = advect(dens, U, V, dt=2.0)                    # expect a shift of +6 in x
    cx_before = (np.arange(W)[None, :] * dens).sum() / dens.sum()
    cx_after = (np.arange(W)[None, :] * moved).sum() / moved.sum()
    assert 5.0 < (cx_after - cx_before) < 7.0, f"advection should move the blob ~6 cells, got {cx_after-cx_before:.1f}"

    # FLUID STEP: an injected force builds a divergence-free flow that transports density.
    vx = np.zeros((H, W)); vy = np.zeros((H, W)); density = blob(24, 24)
    force_x = blob(10, 24) * 5.0
    for _ in range(6):
        vx, vy, density = fluid_step(vx, vy, density, dt=0.2, viscosity=0.05, fx=force_x)
    assert np.abs(divergence(vx, vy)).max() < 1e-3, "fluid velocity should stay ~incompressible"
    assert density.sum() > 0

    # PARTICLES: an attractor pulls particles inward; a velocity field carries them.
    pos = rng.uniform(8, 40, size=(200, 2))
    ps = ParticleSystem(pos)
    d0 = np.linalg.norm(ps.pos - np.array([24, 24]), axis=1).mean()
    for _ in range(40):
        ps.step(force=attractor_force(ps.pos, (24, 24), strength=8.0), dt=0.1, damping=0.05)
    d1 = np.linalg.norm(ps.pos - np.array([24, 24]), axis=1).mean()
    assert d1 < d0, f"attractor should pull particles inward: {d0:.1f} -> {d1:.1f}"

    ps2 = ParticleSystem(np.array([[10.0, 24.0]]))
    ps2.advect_by(np.full((H, W), 4.0), np.zeros((H, W)), dt=1.0)
    assert ps2.pos[0, 0] > 13.0, "particles should ride the velocity field"

    print(f"holographic_fields selftest: ok (DIFFUSE conserves mass, var {f.var():.3f}->{d.var():.3f}; "
          f"PROJECTION divergence {before:.2f}->{after:.1e}; ADVECT moved blob ~{cx_after-cx_before:.1f} cells; "
          f"FLUID stays incompressible (max|div| {np.abs(divergence(vx,vy)).max():.1e}); "
          f"ATTRACTOR pulls particles {d0:.1f}->{d1:.1f})")


if __name__ == "__main__":
    _selftest()
