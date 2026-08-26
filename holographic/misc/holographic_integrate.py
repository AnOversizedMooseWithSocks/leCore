"""holographic_integrate.py -- the simulation TIME-STEP, in one place, plus the uniform SimStep interface the
pipeline's sim stage calls.

WHY THIS EXISTS (Render/Sim Pipeline backlog, Phase 0 / G2)
----------------------------------------------------------
Two problems the backlog names: (1) time integration is inlined by hand in `emitter`/`fields`, and (2) every
solver has a DIFFERENT step signature -- `fluid.step()`, `softbody.step(dt, gravity, ...)`, `physics.step(S_x,
S_v)` -- so the pipeline can't advance "the sim" uniformly. This module fixes both:
  * the INTEGRATORS (semi-implicit/symplectic Euler, and position Verlet) are written once, readably;
  * `SimStep` is the ONE interface -- `advance(dt, ctx) -> ctx` -- that every solver is WRAPPED behind. Each
    solver keeps its own internals and math; only the *interface* is unified, so the pipeline calls one method.

Semi-implicit (symplectic) Euler is exactly what `emitter`/`fields` already do by hand -- update velocity from
the force FIRST, then move the position with the new velocity. It is stabler than explicit Euler (it does not
pump energy into an orbit), which the selftest measures. Verlet is the drop-in second-order upgrade.

HONEST SCOPE (kept loud): SimStep WRAPS, it does not replace -- an adapter around an existing solver calls that
solver's own `.step(...)`, so nothing's math changes (the backlog's rule). The integrators here are the shared
building block for NEW sims (like a particle system or a field effect's forces); existing solvers keep theirs.
Deterministic; NumPy + stdlib.
"""
import numpy as np


# --- the integrators (written once, readably) ------------------------------------------------------------------

def semi_implicit_euler(pos, vel, force, dt):
    """One symplectic Euler step. Velocity is updated from the force FIRST, then the position moves with the NEW
    velocity -- the small reordering that makes it energy-stable (it does not spiral out of an orbit the way
    explicit Euler does). `force` may be an acceleration field (force / mass already applied). Returns (pos, vel)."""
    vel = np.asarray(vel, float) + dt * np.asarray(force, float)      # velocity first (the symplectic bit)
    pos = np.asarray(pos, float) + dt * vel
    return pos, vel


def explicit_euler(pos, vel, force, dt):
    """One EXPLICIT Euler step (position from the OLD velocity), kept only as the honest baseline the symplectic
    integrator is compared against -- it pumps energy and is not what a sim should use."""
    new_pos = np.asarray(pos, float) + dt * np.asarray(vel, float)
    new_vel = np.asarray(vel, float) + dt * np.asarray(force, float)
    return new_pos, new_vel


def verlet(pos, prev_pos, force, dt):
    """One position-Verlet step: next = 2*pos - prev + force*dt^2. Second-order and time-reversible; the drop-in
    upgrade where you keep the previous position instead of an explicit velocity. Returns (new_pos, pos) so the
    caller threads (new_pos becomes pos, pos becomes prev_pos next step)."""
    pos = np.asarray(pos, float); prev_pos = np.asarray(prev_pos, float)
    new_pos = 2.0 * pos - prev_pos + np.asarray(force, float) * dt * dt
    return new_pos, pos


# --- the uniform interface -------------------------------------------------------------------------------------

class SimStep:
    """The ONE interface the pipeline's sim stage calls: `advance(dt, ctx)` steps the sim forward by `dt`,
    reading/writing its fields on the shared context `ctx`, and returns `ctx`. Concrete sims subclass it; existing
    solvers are WRAPPED by SolverAdapter. This is the interface unification -- not a new solver."""

    def advance(self, dt, ctx):
        raise NotImplementedError


class SolverAdapter(SimStep):
    """Wrap an EXISTING solver (fluid/softbody/physics -- anything with its own step) behind the uniform
    `advance`. `step_call(solver, dt, ctx)` is a tiny closure you supply that calls that solver's real step with
    its real arguments and writes the result back into ctx. The solver's math is untouched; only the interface is
    unified (the backlog's WRAP-don't-replace rule)."""

    def __init__(self, solver, step_call):
        self.solver = solver
        self._step_call = step_call

    def advance(self, dt, ctx):
        return self._step_call(self.solver, dt, ctx)


class ParticleSim(SimStep):
    """A concrete SimStep built ON the shared integrator -- a simple point-mass system advanced by a force
    function. Demonstrates the integrator + interface, and is exactly what a FieldEffect's summed forces drive.
    `force_fn(pos, vel) -> accelerations (N,3)`. `integrator` is 'symplectic' (default) or 'explicit' (baseline)."""

    def __init__(self, pos, vel, force_fn, integrator="symplectic"):
        self.pos = np.asarray(pos, float)
        self.vel = np.asarray(vel, float)
        self.force_fn = force_fn
        self.integrator = integrator

    def advance(self, dt, ctx=None):
        f = self.force_fn(self.pos, self.vel)
        if self.integrator == "explicit":
            self.pos, self.vel = explicit_euler(self.pos, self.vel, f, dt)
        else:
            self.pos, self.vel = semi_implicit_euler(self.pos, self.vel, f, dt)
        if ctx is not None and hasattr(ctx, "buffers"):
            ctx.buffers["particles"] = self.pos
        return ctx


def _selftest():
    """Symplectic Euler keeps a circular orbit's energy roughly constant where explicit Euler pumps it up (the
    reason we default to symplectic); Verlet round-trips a free-fall; SolverAdapter routes to a wrapped solver's
    own step; deterministic."""
    # (1) orbit: a unit mass in a 1/r-ish central spring force f = -k*pos. Symplectic should not blow up.
    def central(pos, vel):
        return -1.0 * pos                                            # spring toward origin -> circular orbit

    def energy(p, v):
        return 0.5 * float(np.sum(v * v)) + 0.5 * float(np.sum(p * p))

    p0 = np.array([[1.0, 0.0, 0.0]]); v0 = np.array([[0.0, 1.0, 0.0]])
    dt = 0.05; steps = 2000
    sym = ParticleSim(p0.copy(), v0.copy(), central, integrator="symplectic")
    exp = ParticleSim(p0.copy(), v0.copy(), central, integrator="explicit")
    e_start = energy(sym.pos, sym.vel)
    for _ in range(steps):
        sym.advance(dt); exp.advance(dt)
    drift_sym = abs(energy(sym.pos, sym.vel) - e_start) / e_start
    drift_exp = abs(energy(exp.pos, exp.vel) - e_start) / e_start
    assert drift_sym < drift_exp, (drift_sym, drift_exp)            # symplectic is far stabler
    assert drift_sym < 0.05                                         # and stays bounded

    # (2) Verlet free-fall matches the analytic drop y = 0.5 g t^2 closely
    g = np.array([0.0, -9.8, 0.0])
    pos = np.array([0.0, 0.0, 0.0]); prev = pos.copy()             # start at rest (prev == pos)
    dt = 0.001
    for _ in range(1000):                                           # 1 second
        pos, prev = verlet(pos, prev, g, dt)
    analytic = 0.5 * (-9.8) * (1.0 ** 2)
    assert abs(pos[1] - analytic) < 0.05, (pos[1], analytic)

    # (3) SolverAdapter wraps an arbitrary solver's own step (interface unified, math untouched)
    class _FakeSolver:
        def __init__(self): self.t = 0.0
        def my_weird_step(self, amount): self.t += amount; return self.t
    fake = _FakeSolver()
    adapter = SolverAdapter(fake, lambda solver, dt, ctx: solver.my_weird_step(dt) or ctx)
    adapter.advance(0.5, None); adapter.advance(0.5, None)
    assert abs(fake.t - 1.0) < 1e-9                                # advance() drove the wrapped solver

    # (4) deterministic
    a = ParticleSim(p0.copy(), v0.copy(), central); b = ParticleSim(p0.copy(), v0.copy(), central)
    for _ in range(50):
        a.advance(dt); b.advance(dt)
    assert np.array_equal(a.pos, b.pos)
    print("holographic_integrate selftest OK: symplectic Euler energy drift %.4f vs explicit %.4f (stabler); "
          "Verlet free-fall matches analytic; SolverAdapter unifies a wrapped solver's step; deterministic"
          % (drift_sym, drift_exp))


if __name__ == "__main__":
    _selftest()
