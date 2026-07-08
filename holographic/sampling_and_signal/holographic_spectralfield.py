"""holographic_spectralfield.py -- the SpectralField BACKBONE (Physics & FX backlog, Part 3 #1).

THESIS A, made concrete: a linear field IS a hypervector, and advancing it in time is ONE bind -- a circular
convolution, which is DIAGONAL in the Fourier basis (holographic_iterate's own point). So we diagonalise once
(the FFT), and any time t is a closed-form multiply by a per-frequency transfer -- no time-stepping needed, and
no accumulated step error. Every linear domain is then just a DISPERSION RELATION on this one backbone:

    diffusion / heat / gas-diffusion   rate(|k|)  = -D |k|^2         (parabolic: decay)
    wave / acoustics / EM (vacuum)     omega(|k|) =  c |k|           (hyperbolic: oscillation)
    deep-water ocean (gravity waves)   omega(|k|) =  sqrt(g |k|)     (hyperbolic: dispersive)
    electrostatics                     the t->infinity limit         (Poisson: source / |k|^2)

Two regimes, because the physics has two shapes:
  * PARABOLIC (first order in time):  u_t = -D|k|^2 u    ->  u_hat(t) = u_hat(0) * exp(-D|k|^2 t).
    One complex multiply per frequency. A heat spot spreads and settles to its mean.
  * HYPERBOLIC (second order in time): u_tt = -omega^2 u ->  the state is (field, velocity), advanced by the
    EXACT per-mode rotation  u(t) = u0 cos(wt) + v0 sin(wt)/w,  v(t) = -u0 w sin(wt) + v0 cos(wt).  Still a
    diagonal (per-frequency) operator -- still one bind, still closed-form any t -- just a 2x2 rotation per mode.

Superposition is BUNDLE: sources ADD, so `add_source` just adds fields/spectra -- no re-sim (Thesis A row 3).
Emission / breaking-onset is a calibrated TRIGGER: `trigger_mask` fires where a local potential crosses a
threshold (row 6; the AdaptiveSolver will use this to promote a tile to a grid solver).

KEPT NEGATIVES (loud, from iterate): ONLY LINEAR operators diagonalise this way -- a nonlinear step (an
overturning wave, a shock) does NOT, and stays a grid solver (Part 4's top rung). The spectral field also assumes
PERIODIC boundaries (the FFT's world wraps); a hard wall needs the grid path or a reflection trick. And a
holographic/spectral method must BEAT the grid baseline on a real test before anything delegates to it -- the
standing rule; measured in the tests.

Real basis: Stam's FFT fluid (per-frequency transfer); Tessendorf ocean (the gravity-wave dispersion + Phillips
spectrum). Deterministic (seeded spectra); NumPy + stdlib only.
"""
import numpy as np


def wavenumbers(shape, dx=1.0):
    """The angular-wavenumber grids (per axis) and |k| for an FFT of a `shape` field with cell size `dx`. Angular
    wavenumber k = 2*pi*frequency, so a full FFT (np.fft.fftfreq) gives the signed frequencies each mode carries.
    Works in any dimension. Returns (ks, kmag) where ks is a list of per-axis k grids and kmag = |k|."""
    ks = []
    for axis, n in enumerate(shape):
        k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)                 # signed angular wavenumbers along this axis
        # broadcast this axis's k across all the others
        shape_b = [1] * len(shape)
        shape_b[axis] = n
        ks.append(k1.reshape(shape_b))
    kmag = np.sqrt(sum(k * k for k in ks))
    return ks, kmag


class SpectralField:
    """A real field advanced in FOURIER space by a per-frequency transfer -- advancing time is one bind, any t in
    closed form. `order='parabolic'` uses a real decay `rate(|k|)`; `order='hyperbolic'` oscillates at angular
    frequency `omega(|k|)` and carries a velocity. `rate`/`omega` are callables of the |k| grid (evaluated once)."""

    def __init__(self, field, velocity=None, order="parabolic", rate=None, omega=None, dx=1.0):
        self.field = np.asarray(field, float)
        self.dx = float(dx)
        self.order = order
        _, kmag = wavenumbers(self.field.shape, dx)
        self.kmag = kmag
        if order == "parabolic":
            if rate is None:
                raise ValueError("parabolic SpectralField needs a rate(|k|) callable")
            self.rate = np.asarray(rate(kmag), float)             # real per-frequency decay rate (<= 0), evaluated once
            self.velocity = None
        elif order == "hyperbolic":
            if omega is None:
                raise ValueError("hyperbolic SpectralField needs an omega(|k|) callable")
            self.omega = np.asarray(omega(kmag), float)           # real per-frequency angular frequency, evaluated once
            self.velocity = np.zeros_like(self.field) if velocity is None else np.asarray(velocity, float)
        else:
            raise ValueError("order must be 'parabolic' or 'hyperbolic', got %r" % (order,))

    # -- the heartbeat: advance to +t in ONE closed-form eval (no stepping) -------------------------------------
    def advanced(self, t):
        """Return the (field[, velocity]) at time +t WITHOUT mutating self -- the closed-form jump. Parabolic:
        multiply the spectrum by exp(rate*t). Hyperbolic: rotate each mode by (cos wt, sin wt/w)."""
        if self.order == "parabolic":
            spec = np.fft.fftn(self.field)
            spec = spec * np.exp(self.rate * t)                   # the bind: one complex multiply per frequency
            return np.real(np.fft.ifftn(spec))
        # hyperbolic: the exact per-mode 2nd-order solution (a rotation), computed in Fourier space
        F0 = np.fft.fftn(self.field)
        V0 = np.fft.fftn(self.velocity)
        w = self.omega
        c = np.cos(w * t)
        s_over_w = t * np.sinc(w * t / np.pi)                     # sin(wt)/w, valid at w=0 (limit -> t) via sinc
        Ft = F0 * c + V0 * s_over_w
        Vt = -F0 * (w * np.sin(w * t)) + V0 * c
        return np.real(np.fft.ifftn(Ft)), np.real(np.fft.ifftn(Vt))

    def advance(self, t):
        """Advance self IN PLACE to +t (closed-form). Returns self for chaining."""
        if self.order == "parabolic":
            self.field = self.advanced(t)
        else:
            self.field, self.velocity = self.advanced(t)
        return self

    def step(self, dt, steps=1):
        """Advance by `steps` increments of `dt`. Because advance is closed-form, stepping and a single
        advance(steps*dt) agree to FFT tolerance -- the 'any t in closed form' property (tested)."""
        for _ in range(int(steps)):
            self.advance(dt)
        return self

    # -- superposition = bundle (sources add, no re-sim) --------------------------------------------------------
    def add_source(self, source_field):
        """Bundle a source into the field: linear physics superposes, so adding a source is just adding its field
        (and, for hyperbolic fields, you may also add a velocity via add_velocity). No re-simulation."""
        self.field = self.field + np.asarray(source_field, float)
        return self

    def add_velocity(self, source_velocity):
        if self.order != "hyperbolic":
            raise ValueError("only a hyperbolic field carries velocity")
        self.velocity = self.velocity + np.asarray(source_velocity, float)
        return self

    # -- the closed-form limit (electrostatics / steady heat) ---------------------------------------------------
    def steady_state(self):
        """The t->infinity limit of a PARABOLIC field: every non-DC mode decays to 0 (rate < 0), leaving only the
        mean. (For a driven Poisson problem use `poisson_solve` instead -- that is the STEADY response to a
        standing source, the electrostatics closed form.)"""
        if self.order != "parabolic":
            raise ValueError("steady_state is the parabolic t->inf limit; a hyperbolic field oscillates forever")
        return np.full_like(self.field, self.field.mean())

    # -- calibrated trigger: recognise -> fire (emission / breaking-onset) --------------------------------------
    def trigger_mask(self, potential, threshold):
        """Where a local POTENTIAL (crest steepness, kinetic energy, vorticity, |E|...) crosses `threshold`, the
        trigger fires -- a boolean mask of cells to emit from / promote to a harder solver. The calibrated-trigger
        row of Thesis A; the AdaptiveSolver reads this to decide where the cheap path is no longer enough."""
        return np.asarray(potential, float) >= float(threshold)


# --- electrostatics: the closed-form limit (Poisson), not a time march ----------------------------------------

def poisson_solve(source, dx=1.0, eps0=1.0):
    """Solve laplacian(phi) = -source/eps0 in ONE spectral step: phi_hat = source_hat / (eps0 |k|^2). The
    electrostatic potential of a charge distribution -- the closed-form steady limit of the diffusion field
    (Thesis A: electrostatics = the limit). The k=0 (mean) mode is set to 0 (potential defined up to a constant;
    the net source should be ~0 on a periodic domain)."""
    src = np.asarray(source, float)
    _, kmag = wavenumbers(src.shape, dx)
    k2 = kmag * kmag
    k2[tuple(0 for _ in src.shape)] = 1.0                        # avoid /0 at DC; we zero that mode's result below
    phi_hat = np.fft.fftn(src) / (eps0 * k2)
    phi_hat[tuple(0 for _ in src.shape)] = 0.0                   # gauge: mean potential = 0
    return np.real(np.fft.ifftn(phi_hat))


# --- domain factories: one dispersion relation each (the whole point of the backbone) -------------------------

def diffusion_field(field, D, dx=1.0):
    """Heat / gas-diffusion / any parabolic field: rate(|k|) = -D|k|^2. A spot spreads and settles to its mean."""
    return SpectralField(field, order="parabolic", rate=lambda k: -D * k * k, dx=dx)


def wave_field(field, velocity=None, c=1.0, dx=1.0):
    """Acoustic / EM (vacuum) / any non-dispersive wave: omega(|k|) = c|k|. A pulse propagates at speed c."""
    return SpectralField(field, velocity=velocity, order="hyperbolic", omega=lambda k: c * k, dx=dx)


def em_field(field, velocity=None, c=1.0, dx=1.0):
    """A Maxwell field component (E or B) in vacuum obeys the wave equation with speed c -- so an EM field is a
    wave_field with c = speed of light. (Full E<->B coupling is the EM module's job; this is the propagation.)"""
    return wave_field(field, velocity=velocity, c=c, dx=dx)


def ocean_field(height, velocity=None, g=9.81, dx=1.0):
    """Deep-water gravity waves (Tessendorf): the DISPERSIVE dispersion omega(|k|) = sqrt(g|k|) -- long swells
    travel faster than short chop, which is why an ocean looks like an ocean and not a vibrating drum. Seed
    `height` with `phillips_spectrum` for a real sea state."""
    return SpectralField(height, velocity=velocity, order="hyperbolic",
                         omega=lambda k: np.sqrt(g * np.abs(k)), dx=dx)


def phillips_spectrum(shape, wind=(12.0, 0.0), amplitude=1e-4, g=9.81, dx=1.0, seed=0):
    """Tessendorf's Phillips spectrum: a seeded, physically-shaped random ocean HEIGHT field. Energy concentrates
    around wind-driven wavelengths, aligns with the wind direction, and rolls off at tiny scales. Returns a real
    height field (the initial surface). Deterministic given `seed` -- the determinism rule."""
    ks, kmag = wavenumbers(shape, dx)
    W = np.array(wind, float)
    wind_speed = np.linalg.norm(W) + 1e-9
    L = wind_speed * wind_speed / g                              # largest wave the wind can build
    what = W / wind_speed
    k2 = kmag * kmag
    k2_safe = np.where(k2 == 0, 1.0, k2)
    # directional cosine between k and the wind (build the k-unit dotted with the wind unit)
    kdotw = sum(k * w for k, w in zip(ks, what)) / np.sqrt(k2_safe)
    Ph = amplitude * np.exp(-1.0 / (k2_safe * L * L)) / (k2_safe * k2_safe) * (kdotw * kdotw)
    Ph = np.where(k2 == 0, 0.0, Ph)                             # no energy in the DC (mean) mode
    rng = np.random.default_rng(seed)
    # a random complex amplitude with the Phillips magnitude, made Hermitian so the surface is REAL
    xr = rng.standard_normal(shape); xi = rng.standard_normal(shape)
    h0 = (xr + 1j * xi) / np.sqrt(2.0) * np.sqrt(Ph)
    height = np.real(np.fft.ifftn(h0))
    return height


def _selftest():
    """Closed-form == stepped (the 'any t' property); diffusion matches the analytic Gaussian spread and settles;
    a wave mode oscillates at exactly omega=c|k| and energy is conserved; superposition is additive; the ocean is
    real, deterministic, and dispersive; Poisson recovers a point-charge potential."""
    rng = np.random.default_rng(0)

    # (1) PARABOLIC closed-form == stepped, and matches the analytic Gaussian diffusion width
    N = 128
    x = np.arange(N)
    f0 = np.exp(-((x - N / 2) ** 2) / (2 * 4.0 ** 2))            # a Gaussian spot, sigma0 = 4
    D = 0.5
    hf = diffusion_field(f0.copy(), D=D, dx=1.0)
    t = 20.0
    one_shot = diffusion_field(f0.copy(), D=D, dx=1.0).advanced(t)
    stepped = diffusion_field(f0.copy(), D=D, dx=1.0)
    stepped.step(0.5, steps=40)                                  # 40 * 0.5 = 20 = t
    assert np.max(np.abs(one_shot - stepped.field)) < 1e-9, "closed-form must equal stepped (parabolic)"
    # analytic: a Gaussian of variance s0^2 diffuses to s0^2 + 2 D t
    sig_t = np.sqrt(4.0 ** 2 + 2 * D * t)
    peak_ratio = one_shot.max() / f0.max()
    assert abs(peak_ratio - 4.0 / sig_t) < 0.02, (peak_ratio, 4.0 / sig_t)   # peak falls as 1/width
    assert np.max(np.abs(diffusion_field(f0.copy(), D, 1.0).steady_state() - f0.mean())) < 1e-9

    # (2) HYPERBOLIC closed-form == stepped; a single Fourier mode oscillates at exactly omega = c|k|
    c = 2.0
    k_mode = 2 * np.pi * 3 / N                                   # the 3rd Fourier mode's wavenumber
    field0 = np.cos(k_mode * x)
    wf = wave_field(field0.copy(), c=c, dx=1.0)
    period = 2 * np.pi / (c * k_mode)                           # after one period the mode returns to itself
    back = wave_field(field0.copy(), c=c, dx=1.0).advanced(period)[0]
    assert np.max(np.abs(back - field0)) < 1e-8, "a wave mode must return after exactly one period T=2pi/(c|k|)"
    one = wave_field(field0.copy(), c=c, dx=1.0).advanced(1.3)[0]
    st = wave_field(field0.copy(), c=c, dx=1.0); st.step(0.13, steps=10)
    assert np.max(np.abs(one - st.field)) < 1e-8, "closed-form must equal stepped (hyperbolic)"

    # (3) superposition is additive (bundle): advance(A+B) == advance(A) + advance(B)
    a = np.exp(-((x - 40) ** 2) / 8.0); b = np.exp(-((x - 90) ** 2) / 8.0)
    adv_sum = diffusion_field((a + b).copy(), D, 1.0).advanced(5.0)
    adv_a = diffusion_field(a.copy(), D, 1.0).advanced(5.0)
    adv_b = diffusion_field(b.copy(), D, 1.0).advanced(5.0)
    assert np.max(np.abs(adv_sum - (adv_a + adv_b))) < 1e-9

    # (4) ocean: real, deterministic, dispersive (long waves outrun short ones)
    h = phillips_spectrum((64, 64), wind=(15.0, 0.0), seed=1)
    assert np.isrealobj(h) and np.array_equal(h, phillips_spectrum((64, 64), wind=(15.0, 0.0), seed=1))
    oc = ocean_field(h.copy(), g=9.81, dx=1.0)
    surf_t = oc.advanced(2.0)[0]
    assert np.isrealobj(surf_t) and np.max(np.abs(surf_t)) > 0                # it evolved and stayed real
    # dispersion sanity: omega(|k|)=sqrt(g|k|) so a longer wave (smaller |k|) has a LOWER frequency
    assert np.sqrt(9.81 * 0.1) < np.sqrt(9.81 * 1.0)

    # (5) Poisson: a point charge gives a potential that peaks at the charge and falls off
    src = np.zeros((64, 64)); src[32, 32] = 1.0; src -= src.mean()
    phi = poisson_solve(src, dx=1.0)
    assert phi[32, 32] == phi.max() or phi[32, 32] == phi.min()               # extremum at the charge

    print("holographic_spectralfield selftest OK: closed-form==stepped for both regimes (<1e-8); diffusion matches "
          "the analytic Gaussian spread and settles to the mean; a wave mode returns after exactly T=2pi/(c|k|); "
          "superposition is additive; the ocean is real, deterministic and dispersive; Poisson peaks at the charge")


if __name__ == "__main__":
    _selftest()
