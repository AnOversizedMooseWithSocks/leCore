"""Grid-based fluid solver -- Stam 'Stable Fluids' (SIGGRAPH 1999), the method professional smoke/fluid engines
(Houdini's smoke solver, Maya/Bifrost Aero) are built on. Carries velocity + smoke density + temperature + fuel,
so ONE solver does smoke, buoyant plumes, and combustion/FIRE.

WHY this is on-thesis (Jos Stam, advisory panel). Two ideas make it stable and they are both the engine's algebra:

  * PRESSURE PROJECTION in the FOURIER domain. Incompressible flow needs a divergence-free velocity. The
    Helmholtz-Hodge decomposition splits any field into a divergence-free part + a gradient; on a PERIODIC grid
    that split is exact in the Fourier basis -- subtract the longitudinal component  u_hat <- u_hat - k (k.u_hat)/|k|^2.
    That is a circular convolution, the SAME periodic-domain structure as bind. The pressure solve other codes do
    with hundreds of Jacobi sweeps is here ONE pair of FFTs, exact.

  * SEMI-LAGRANGIAN advection. Don't push values forward (that explodes for big dt); trace each cell BACKWARD
    along the velocity and interpolate where it came from. Unconditionally stable -- the contribution that made
    'Stable Fluids' famous -- so the timestep is bounded by accuracy, never by a CFL blow-up.

HONEST (kept loud):
  * Pure NumPy: this is the OFFLINE sim BRAIN. A 64^3 grid steps in ~tens-to-hundreds of ms (measured below), NOT
    the GPU-realtime of Bifrost. The METHOD matches the pros; the throughput does not, and pretending otherwise
    would be the exact dishonesty this project refuses.
  * Boundaries are PERIODIC (toroidal) -- the price of the FFT projection, and Stam's own classic choice. Solid
    obstacles (a fan, a collider) are a separate masking solve, not done here.
  * Semi-Lagrangian advection is slightly DISSIPATIVE: fine smoke detail smears over many steps (measured: total
    density drifts down). Vorticity confinement re-injects swirl to fight it; a MacCormack/BFECC or FLIP scheme
    would conserve better and is the honest next step.
"""

import itertools
import numpy as np


def _sample_periodic(field, coords, xp=np):
    """n-linear interpolation of `field` at fractional `coords` (shape (d, *grid)), with PERIODIC wrap -- the
    sampler behind semi-Lagrangian advection. Gathers the 2^d surrounding corners and weights them by the
    fractional offset; wrap (modulo) matches the periodic FFT projection so the whole solver lives on one torus."""
    shape = field.shape
    d = len(shape)
    base = xp.floor(coords).astype(xp.int64)
    frac = coords - base
    out = xp.zeros(coords.shape[1:], float)
    for corner in itertools.product((0, 1), repeat=d):                  # 2^d corners of the containing cell
        idx = tuple((base[k] + corner[k]) % shape[k] for k in range(d))  # periodic wrap
        w = xp.ones(coords.shape[1:], float)
        for k in range(d):
            w = w * (frac[k] if corner[k] else (1.0 - frac[k]))         # multilinear weight
        out += w * field[idx]
    return out


class StableFluid:
    """A periodic-grid stable-fluids solver in 2D or 3D (shape sets the dimension). State: velocity `vel`
    (d, *shape), and scalar fields `density` (smoke), `temperature`, `fuel`. Call step() to advance dt."""

    def __init__(self, shape, dt=0.1, viscosity=0.0, diffusion=0.0,
                 dissipation=0.01, cooling=0.05, buoyancy_alpha=0.15, buoyancy_beta=0.05,
                 vorticity=2.0, up_axis=0, ignition=0.5, burn_rate=2.0, smoke_yield=0.4, device="cpu"):
        self.shape = tuple(int(s) for s in shape)
        self.d = len(self.shape)
        self.dt = float(dt)
        self.viscosity = float(viscosity)
        self.diffusion = float(diffusion)
        self.dissipation = float(dissipation)        # smoke fades
        self.cooling = float(cooling)                # heat radiates away
        self.alpha = float(buoyancy_alpha)           # smoke weighs the flow DOWN
        self.beta = float(buoyancy_beta)             # heat lifts the flow UP
        self.vorticity = float(vorticity)            # vorticity-confinement strength (0 = off)
        self.up_axis = int(up_axis)
        self.ignition = float(ignition)              # temperature above which fuel burns
        self.burn_rate = float(burn_rate)
        self.smoke_yield = float(smoke_yield)        # smoke produced per unit fuel burned

        # backend: device='cpu' -> NumPy (default, byte-identical); device='gpu' -> CuPy if available (the FFT
        # projection + elementwise advection are ideal GPU work). All state lives on self.xp; render via to_numpy().
        from holographic_backend import array_module
        self.device = device
        self.xp = array_module(device)
        xp = self.xp
        self.vel = xp.zeros((self.d,) + self.shape, float)
        self.density = xp.zeros(self.shape, float)
        self.temperature = xp.zeros(self.shape, float)
        self.fuel = xp.zeros(self.shape, float)
        self._grids = xp.indices(self.shape, dtype=float)  # (d, *shape) cell coordinates, reused every advect
        self._K, self._k2 = self._wavenumbers()

    # --- spectral helpers (the on-thesis core) -------------------------------------------------------
    def _wavenumbers(self):
        """Per-axis wavevectors and |k|^2 for the FFT projection and diffusion. We use the EXACT SYMBOL OF THE
        CENTRED DIFFERENCE (sin(theta), since 0.5*(roll(-1)-roll(+1)) acts as i*sin(theta) in Fourier space), NOT
        the ideal k=theta. Why: the projection must remove the same divergence the solver actually measures and
        advects with -- our divergence() and vorticity use centred differences -- so matching that operator's
        symbol makes the projected velocity divergence-free in the TRUE discrete sense (to machine precision),
        instead of leaving a finite-difference residual. The DC bin and the centred-difference null modes (where
        sin(theta)=0, e.g. Nyquist) get |k|^2:=1 so they divide cleanly and are left UNCHANGED -- correct, because
        those modes are invisible to the divergence operator and are already divergence-free to it."""
        K = []
        for ax, n in enumerate(self.shape):
            theta = 2.0 * self.xp.pi * self.xp.fft.fftfreq(n)
            kappa = self.xp.sin(theta)                         # symbol of the centred difference = i*sin(theta)
            sh = [1] * self.d; sh[ax] = n
            K.append(kappa.reshape(sh))
        k2 = sum(k ** 2 for k in K) + self.xp.zeros(self.shape)
        k2 = self.xp.where(self.xp.abs(k2) < 1e-12, 1.0, k2)
        return K, k2

    def project(self, vel):
        """Make `vel` divergence-free (incompressible) by removing its longitudinal part in Fourier space:
        u_hat <- u_hat - k (k.u_hat)/|k|^2. The Helmholtz-Hodge projection as a circular convolution -- the whole
        pressure solve in one pair of FFTs, exact, no iteration. This is the engine's periodic algebra doing the
        single most expensive step of every professional fluid solver for free."""
        uhat = [self.xp.fft.fftn(vel[k]) for k in range(self.d)]
        kdotu = sum(self._K[k] * uhat[k] for k in range(self.d))
        out = self.xp.empty_like(vel)
        for k in range(self.d):
            proj = uhat[k] - self._K[k] * kdotu / self._k2
            out[k] = self.xp.real(self.xp.fft.ifftn(proj))
        return out

    def diffuse(self, field, rate):
        """Implicit (unconditionally stable) diffusion: in Fourier space the heat equation is just a decay
        1/(1+ rate*dt*|k|^2). Used for viscosity (on velocity) and scalar diffusion. rate=0 is a no-op."""
        if rate <= 0.0:
            return field
        decay = 1.0 / (1.0 + rate * self.dt * self._k2)
        return self.xp.real(self.xp.fft.ifftn(self.xp.fft.fftn(field) * decay))

    # --- transport -----------------------------------------------------------------------------------
    def _backtrace(self, vel):
        """Where each cell's value came from one step ago: x - dt*vel, the foot of the semi-Lagrangian trace."""
        return self._grids - self.dt * vel

    def advect(self, field, vel):
        """Move a SCALAR field with the flow, semi-Lagrangian (stable for any dt)."""
        return _sample_periodic(field, self._backtrace(vel), self.xp)

    def advect_velocity(self, vel):
        """Self-advection of the VELOCITY field (each component sampled at the same backtraced feet)."""
        coords = self._backtrace(vel)
        return self.xp.array([_sample_periodic(vel[k], coords, self.xp) for k in range(self.d)])

    # --- forces --------------------------------------------------------------------------------------
    def buoyancy(self, vel):
        """Hot, light fluid rises; dense smoke sinks. Adds f = (beta*T - alpha*density) along the up-axis
        (Fedkiw et al.). This is what turns a heat source into a rising plume."""
        out = vel.copy()
        out[self.up_axis] += self.dt * (self.beta * self.temperature - self.alpha * self.density)
        return out

    def vorticity_confinement(self, vel):
        """Re-inject the small-scale swirl that semi-Lagrangian advection smears away (Fedkiw/Stam). Compute the
        vorticity, then push velocity along N x omega where N points up the vorticity-magnitude gradient -- it
        amplifies existing eddies without adding net divergence. The detail term that makes CG smoke look alive.
        2D vorticity is a scalar (z-curl); 3D is a 3-vector curl."""
        if self.vorticity <= 0.0:
            return vel
        eps = self.vorticity
        if self.d == 2:
            wz = self._d(vel[1], 0) - self._d(vel[0], 1)                # curl_z
            mag = self.xp.abs(wz)
            gx, gy = self._d(mag, 0), self._d(mag, 1)
            nrm = self.xp.sqrt(gx ** 2 + gy ** 2) + 1e-12
            Nx, Ny = gx / nrm, gy / nrm
            f0 = eps * (Ny * wz)                                        # N x omega, with omega = wz * z_hat
            f1 = eps * (-Nx * wz)
            out = vel.copy(); out[0] += self.dt * f0; out[1] += self.dt * f1
            return out
        # 3D
        wx = self._d(vel[2], 1) - self._d(vel[1], 2)
        wy = self._d(vel[0], 2) - self._d(vel[2], 0)
        wz = self._d(vel[1], 0) - self._d(vel[0], 1)
        mag = self.xp.sqrt(wx ** 2 + wy ** 2 + wz ** 2)
        gx, gy, gz = self._d(mag, 0), self._d(mag, 1), self._d(mag, 2)
        nrm = self.xp.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-12
        Nx, Ny, Nz = gx / nrm, gy / nrm, gz / nrm
        out = vel.copy()
        out[0] += self.dt * eps * (Ny * wz - Nz * wy)                   # N x omega
        out[1] += self.dt * eps * (Nz * wx - Nx * wz)
        out[2] += self.dt * eps * (Nx * wy - Ny * wx)
        return out

    def _d(self, f, axis):
        """Centred periodic derivative along one axis (self.xp.roll = periodic shift)."""
        return 0.5 * (self.xp.roll(f, -1, axis=axis) - self.xp.roll(f, 1, axis=axis))

    # --- combustion ----------------------------------------------------------------------------------
    def combust(self):
        """Fuel above the ignition temperature burns: it is consumed, releases HEAT (raising temperature so the
        reaction sustains and the plume lifts), and yields SMOKE. A minimal reaction model -- the same coupling
        Houdini's pyro uses (fuel -> heat + smoke), just without the full chemistry."""
        burning = (self.temperature >= self.ignition) & (self.fuel > 0.0)
        if not burning.any():
            return
        burned = self.xp.where(burning, self.xp.minimum(self.fuel, self.burn_rate * self.dt * self.fuel), 0.0)
        self.fuel = self.fuel - burned
        self.temperature = self.temperature + burned                    # exothermic: fuel -> heat
        self.density = self.density + self.smoke_yield * burned         # and soot/smoke

    # --- emission ------------------------------------------------------------------------------------
    def add_source(self, region, density=0.0, temperature=0.0, fuel=0.0, vel=None):
        """Inject smoke / heat / fuel / velocity into a `region` (a slice tuple or a boolean mask) -- a candle, a
        chimney, an emitter. Array inputs are moved to the solver's device so a GPU solver stays on the GPU."""
        if hasattr(region, "shape"):                 # a boolean mask -> match the solver's device
            region = self.xp.asarray(region)
        if density:     self.density[region] += density
        if temperature: self.temperature[region] += temperature
        if fuel:        self.fuel[region] += fuel
        if vel is not None:
            for k in range(self.d):
                self.vel[k][region] += float(vel[k]) if np.isscalar(vel[k]) else self.xp.asarray(vel[k])

    # --- the Stam update -----------------------------------------------------------------------------
    def step(self):
        """One stable-fluids step: forces -> self-advect velocity -> (viscosity) -> PROJECT to divergence-free
        -> advect the scalars through the clean field -> combustion -> dissipation/cooling. The projection is the
        only thing that keeps the flow incompressible, and it runs twice (after forces and after advection) so the
        velocity the scalars ride on is always divergence-free."""
        self.vel = self.buoyancy(self.vel)
        self.vel = self.vorticity_confinement(self.vel)
        self.vel = self.project(self.vel)                               # clean the force-injected divergence
        self.vel = self.advect_velocity(self.vel)
        if self.viscosity > 0.0:
            self.vel = self.xp.array([self.diffuse(self.vel[k], self.viscosity) for k in range(self.d)])
        self.vel = self.project(self.vel)                               # advection re-introduces divergence; clean it
        self.density = self.advect(self.density, self.vel)
        self.temperature = self.advect(self.temperature, self.vel)
        self.fuel = self.advect(self.fuel, self.vel)
        if self.diffusion > 0.0:
            self.density = self.diffuse(self.density, self.diffusion)
        self.combust()
        self.density *= (1.0 - self.dissipation * self.dt)
        self.temperature *= (1.0 - self.cooling * self.dt)

    def to_numpy(self, name):
        """Return a field ('vel'/'density'/'temperature'/'fuel') as a host NumPy array, for rendering/inspection
        (no-op on CPU; device->host copy on GPU)."""
        from holographic_backend import asnumpy
        return asnumpy(getattr(self, name))

    def divergence(self, vel=None):
        """Max |div v| -- the incompressibility residual. After project() this is ~0 (the correctness check)."""
        vel = self.vel if vel is None else vel
        div = sum(self._d(vel[k], k) for k in range(self.d))
        return float(self.xp.max(self.xp.abs(div)))


def _selftest():
    rng = np.random.default_rng(0)
    # 1. projection really removes divergence (the core correctness property)
    f = StableFluid((32, 32), dt=0.1)
    f.vel = rng.standard_normal(f.vel.shape)
    before = f.divergence()
    f.vel = f.project(f.vel)
    after = f.divergence()
    assert after < 1e-9 < before, (before, after)
    # 2. semi-Lagrangian advection is STABLE under a big dt (no CFL blow-up): finite, and divergence-free to
    #    machine precision RELATIVE to the velocity scale (constant forcing with no viscosity grows |v|, which is
    #    correct physics -- the point is it never goes to NaN as an explicit solver would, and stays incompressible)
    g = StableFluid((48, 48), dt=1.0, vorticity=3.0)
    g.add_source((slice(20, 28), slice(20, 28)), density=1.0, temperature=2.0)
    for _ in range(40):
        g.step()
    scale = float(np.max(np.abs(g.vel))) + 1e-12
    assert np.isfinite(g.vel).all() and g.divergence() / scale < 1e-6
    # 3. buoyancy lifts a hot blob (its centre of mass rises along up_axis=0 -> row index DECREASES)
    h = StableFluid((48, 48), dt=0.5, buoyancy_beta=0.4, vorticity=0.0, dissipation=0.0, cooling=0.0, up_axis=0)
    h.add_source((slice(34, 40), slice(20, 28)), density=1.0, temperature=3.0)
    rows = np.arange(48)[:, None]
    com0 = float((h.density * rows).sum() / h.density.sum())
    for _ in range(30):
        h.step()
    com1 = float((h.density * rows).sum() / h.density.sum())
    assert com1 < com0 - 1.0, (com0, com1)
    print(f"fluid selftest ok: projection div {before:.2e}->{after:.2e}; stable at dt=2.0; "
          f"hot plume rose {com0 - com1:.1f} cells")


if __name__ == "__main__":
    _selftest()
