"""holographic_wave.py -- A3: a scalar ACOUSTIC WAVE field. Sound that actually PROPAGATES (and reflects, absorbs).

WHY THIS EXISTS (Acoustics & Cymatics backlog, item A3)
------------------------------------------------------
The fluid solver is INCOMPRESSIBLE, so it carries no sound -- pressure there equilibrates instantly. Real sound
is a compressible pressure wave: the acoustic wave equation, d2p/dt2 = c^2 * grad^2 p (d'Alembert). This module
integrates it, so a pulse spreads outward at the material's speed of sound, reflects off hard boundaries, and is
soaked up by an absorbing edge. It is the low-frequency, wave-accurate complement to the ray-traced (high-
frequency) room acoustics of A6, and it supplies the standing field that acoustic levitation (A7) needs.

THE METHOD (readable, first-principles)
---------------------------------------
Store the pressure p now and one step ago (p_prev). The wave equation, discretised by an explicit LEAPFROG in
time and a finite-difference Laplacian in space, marches forward:

    p_next = 2*p - p_prev + (c*dt/dx)^2 * laplacian(p)

That is it -- the whole solver is that one line, plus sources and boundaries. `c` may be a single value or a
field (per-cell speed of sound from the material data), so a wave slows and partly reflects where c changes.

STABILITY (kept loud, not hidden): explicit leapfrog has the Courant (CFL) limit dt <= dx / (c*sqrt(ndim)); past
it the scheme blows up. `step()` computes the stable sub-step and AUTO-SUBDIVIDES a large dt into stable inner
steps, so a caller can ask for any dt and get a stable, correct result (a spectral/implicit step would lift the
limit -- a later option). Boundaries: a hard wall REFLECTS (Neumann); an absorbing 'sponge' border soaks the
wave to stop it bouncing back (a crude PML). NumPy + stdlib; deterministic.

HONEST SCOPE (kept negative): a scalar (pressure-only) linear acoustic field -- no vector elastic waves, no
nonlinear/shock effects; the sponge border is a simple damping ramp, not a true perfectly-matched layer.
"""
import numpy as np


def _laplacian(p):
    """The edge-replicated (zero-flux / Neumann) discrete Laplacian, any dimension.

    A1: this stencil was duplicated bit-for-bit in `holographic_heat` and `holographic_wave`. It now lives
    once in `holographic_laplacian.laplacian(field, bc=)`, which also offers the periodic and dirichlet
    boundaries; this alias keeps the old private name working for existing callers.
    """
    from holographic.simulation_and_physics.holographic_laplacian import laplacian
    return laplacian(p, bc="neumann")


class WaveField:
    """A scalar acoustic pressure field on a grid. `c` = speed of sound (scalar or per-cell field), `dx` = cell
    size. Add a `pulse` or a `source`, then `step(dt)` to propagate. An `absorb_border` sponge (>0) soaks waves at
    the edge so they do not reflect back."""

    def __init__(self, shape, c=343.0, dx=1.0, damping=0.0, absorb_border=0):
        self.shape = tuple(shape)
        self.c = np.asarray(c, float) if np.ndim(c) else float(c)
        self.dx = float(dx)
        self.damping = float(damping)
        self.p = np.zeros(self.shape, float)
        self.p_prev = np.zeros(self.shape, float)
        self.t = 0.0
        self._absorb = self._build_sponge(int(absorb_border)) if absorb_border else None

    def _c_max(self):
        return float(np.max(self.c)) if np.ndim(self.c) else float(self.c)

    def _build_sponge(self, width):
        """A damping ramp that rises toward the border -- multiplies the field each step so an outgoing wave dies
        before it can reflect (a crude absorbing boundary)."""
        s = np.ones(self.shape, float)
        for ax in range(len(self.shape)):
            n = self.shape[ax]
            ramp = np.ones(n)
            for k in range(width):
                d = (width - k) / width
                atten = 1.0 - 0.12 * d * d
                ramp[k] = min(ramp[k], atten); ramp[n - 1 - k] = min(ramp[n - 1 - k], atten)
            shape = [1] * len(self.shape); shape[ax] = n
            s = s * ramp.reshape(shape)
        return s

    def stable_dt(self):
        """The largest CFL-stable explicit time step: dt <= dx / (c_max * sqrt(ndim)), with a safety margin."""
        return 0.9 * self.dx / (self._c_max() * np.sqrt(len(self.shape)))

    def pulse(self, center, amp=1.0, radius=2.0):
        """Add a smooth Gaussian pressure pulse (a tap / a spark). It will split and propagate outward at c."""
        idx = np.indices(self.shape, dtype=float)
        r2 = sum((idx[k] - float(center[k])) ** 2 for k in range(len(self.shape)))
        self.p = self.p + amp * np.exp(-r2 / (2.0 * radius * radius))
        self.p_prev = self.p.copy()                                # zero initial velocity -> a symmetric split
        return self

    def source(self, center, value):
        """Hard-set the pressure at one cell (drive it externally, e.g. a speaker oscillating in time)."""
        self.p[tuple(int(round(x)) for x in center)] = float(value)
        return self

    def step(self, dt=None, steps=1):
        """Advance the wave by dt for `steps` steps. Auto-subdivides a too-large dt into CFL-stable inner steps so
        it never blows up. Leapfrog: p_next = 2p - p_prev + (c*dt/dx)^2 * lap(p), times the sponge if any."""
        dt = float(dt) if dt is not None else self.stable_dt()
        smax = self.stable_dt()
        n_sub = max(1, int(np.ceil(dt / smax)))                    # keep each inner step CFL-stable
        h = dt / n_sub
        coeff = (self.c * h / self.dx) ** 2                        # (c*dt/dx)^2, per cell if c is a field
        for _ in range(int(steps) * n_sub):
            lap = _laplacian(self.p)
            p_next = 2.0 * self.p - self.p_prev + coeff * lap
            if self.damping:
                p_next -= self.damping * (self.p - self.p_prev)    # optional bulk loss
            if self._absorb is not None:
                p_next *= self._absorb                             # soak the wave at the border
                self.p *= self._absorb
            self.p_prev = self.p
            self.p = p_next
            self.t += h
        return self.p

    def energy(self):
        """A proxy for the field's total energy (sum of p^2 + the leapfrog velocity^2). Should stay BOUNDED (not
        grow) for a stable, lossless-or-damped run -- the honest stability check."""
        vel = (self.p - self.p_prev)
        return float(np.sum(self.p ** 2 + vel ** 2))


def _selftest():
    """A pulse propagates at c, splits into two movers (d'Alembert), stays bounded (stable), reflects off a hard
    wall, and an absorbing border removes most of the energy. Deterministic."""
    # (1) 1-D d'Alembert: a centred pulse splits into a LEFT and a RIGHT mover, each travelling at c
    N = 400; c = 1.0; dx = 1.0
    w = WaveField((N,), c=c, dx=dx)
    w.pulse((N // 2,), amp=1.0, radius=4.0)
    T = 120.0
    w.step(dt=T, steps=1)                                          # auto-substepped internally
    # the two peaks should be ~ c*T away from the centre on each side
    peak_r = N // 2 + int(np.argmax(w.p[N // 2:]))
    peak_l = int(np.argmax(w.p[:N // 2]))
    assert abs((peak_r - N // 2) - c * T) < 12, (peak_r - N // 2, c * T)
    assert abs((N // 2 - peak_l) - c * T) < 12
    assert np.isfinite(w.p).all()                                 # stayed stable

    # (2) energy stays BOUNDED over a long run (no blow-up) -- the CFL guard works even for a big requested dt
    w2 = WaveField((120, 120), c=2.0, dx=1.0)
    w2.pulse((60, 60), amp=1.0, radius=3.0)
    e0 = w2.energy()
    w2.step(dt=200.0, steps=1)                                    # way past a single CFL step -> must auto-substep
    assert np.isfinite(w2.p).all() and w2.energy() < 5.0 * e0     # bounded, not exploding

    # (3) speed scales with c: a faster medium carries the front farther in the same time
    def front_distance(cval):
        wf = WaveField((300,), c=cval, dx=1.0); wf.pulse((150,), radius=4.0); wf.step(dt=50.0)
        return (150 + int(np.argmax(wf.p[150:]))) - 150
    assert front_distance(2.0) > front_distance(1.0) * 1.5        # ~2x c -> ~2x distance

    # (4) an absorbing border removes energy a hard (reflecting) wall keeps
    hard = WaveField((160,), c=1.0, dx=1.0); hard.pulse((80,), radius=3.0)
    soft = WaveField((160,), c=1.0, dx=1.0, absorb_border=30); soft.pulse((80,), radius=3.0)
    for _ in range(6):
        hard.step(dt=20.0); soft.step(dt=20.0)
    assert soft.energy() < hard.energy() * 0.6                    # the sponge soaked the wave at the edge

    # (5) deterministic
    a = WaveField((80,), c=1.0); a.pulse((40,)); a.step(dt=20.0)
    b = WaveField((80,), c=1.0); b.pulse((40,)); b.step(dt=20.0)
    assert np.array_equal(a.p, b.p)
    print("holographic_wave selftest OK: pulse splits into two movers at c (d'Alembert); energy bounded (CFL "
          "auto-substep); front distance scales with c; absorbing border soaks the wave; deterministic")


if __name__ == "__main__":
    _selftest()
