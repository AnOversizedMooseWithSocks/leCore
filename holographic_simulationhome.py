"""holographic_simulationhome.py -- the SIMULATION scaffold (consolidation backlog R9): ONE step loop over any
solver. This is a scaffold, NOT a merge.

WHY THIS EXISTS
---------------
The engine has several time-stepped solvers and they are legitimately DIFFERENT algorithms -- Stable Fluids
(advect + project), combustion, smoke presets, position-based softbody, Cosserat rods / grooming, MPM snow, SDF
collision, reaction-diffusion automata. Each has its own step, with its own signature (fluid.step() takes no args,
softbody.step(dt, gravity, iterations, ...), Fire.step(dt, ambient_K, ...), HyperCA.evolve(steps), ...). What was
missing is a shared STEP LOOP so a caller (or the render Pipeline) can advance ANY of them the same way and draw the
result -- WITHOUT flattening the solvers into one (that would destroy the very differences that make each correct).

`Simulation` is that scaffold. The solver's math is untouched; the Simulation only:
  * gives it ONE interface -- step(dt) / run(steps, dt) -- via holographic_integrate.SolverAdapter (the SimStep the
    Pipeline's sim stage already calls), and
  * exposes its field as a Field (R2), so volume_render / the Pipeline (R1) can draw it.

    Simulation(solver, step_fn, field_fn, lo, hi, name)   # wrap ANY solver in three closures
      step_fn(solver, dt) -- call the solver's REAL step with its real args
      field_fn(solver)    -- return the solver's current density/scalar grid (what gets rendered)
    Simulation.step(dt) / run(steps, dt)   # the shared loop
    Simulation.field() -> Field (R2)       # sample-able, ready for the renderer (a 2D grid is lifted to a slab)
    Simulation.render(camera, ...)         # convenience: volume_render the field

Strategy factories adapt a specific solver in a few lines, keeping its construction where it belongs:
    Simulation.for_fluid(StableFluid)      Simulation.for_automaton(HyperCA)
Add another (softbody / mpm / combustion / ...) the same way: its step closure + its field extractor.
"""
import numpy as np


class Simulation:
    """A shared step loop wrapping ONE solver behind a uniform step(dt), with its field exposed for rendering."""

    def __init__(self, solver, step_fn, field_fn, lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0), name="sim"):
        self.solver = solver
        self._step_fn = step_fn                                  # (solver, dt) -> advance the solver's real step
        self._field_fn = field_fn                               # (solver) -> the current grid to render
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.name = str(name)
        self.t = 0.0
        self.steps_run = 0
        # the uniform SimStep the render Pipeline's sim stage calls: advance(dt, ctx) steps the solver.
        from holographic_integrate import SolverAdapter
        self.adapter = SolverAdapter(solver, lambda s, dt, ctx: self._step_fn(s, dt))

    def step(self, dt=1.0 / 60.0):
        """Advance the wrapped solver ONE step through the shared loop -- its own math, one uniform interface."""
        self._step_fn(self.solver, dt)
        self.t += float(dt)
        self.steps_run += 1
        return self

    def run(self, steps, dt=1.0 / 60.0):
        """Run the shared step loop `steps` times. The same loop drives every solver."""
        for _ in range(int(steps)):
            self.step(dt)
        return self

    def grid(self):
        """The solver's current density/scalar grid (raw), as its field_fn extracts it."""
        return np.asarray(self._field_fn(self.solver), float)

    def field(self):
        """The solver's field as a Field (R2), sample-able at points -- the bridge to volume_render / the Pipeline.
        A 2D grid is lifted to a thin 3D slab so the volumetric marcher has depth to march through."""
        from holographic_fieldhome import Field
        g = self.grid()
        if g.ndim == 2:
            g = np.repeat(g[:, :, None], 6, axis=2)             # lift a 2D field to a slab for the 3D renderer
        return Field.grid(g, self.lo, self.hi)

    def render(self, camera, width=48, height=48, steps=48, mode="smoke", sigma=10.0):
        """Convenience: draw the solver's field with the volumetric renderer (the Pipeline's field-render path).
        Returns (image RGB (H,W,3), alpha (H,W))."""
        from holographic_render import volume_render
        fld = self.field()
        return volume_render(lambda p: fld.sample(p), camera, (self.lo, self.hi), width, height, steps=steps,
                             mode=mode, sigma=sigma)

    # --- strategy factories: adapt a specific solver without touching its math ---
    @staticmethod
    def for_fluid(fluid, lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0)):
        """Wrap a holographic_fluid.StableFluid: its step takes no args; its render field is the density grid."""
        return Simulation(fluid, lambda s, dt: s.step(), lambda s: np.asarray(s.density, float), lo, hi, "fluid")

    @staticmethod
    def for_automaton(ca, lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0)):
        """Wrap a holographic_automaton.HyperCA (reaction-diffusion -- a DIFFERENT algorithm from the fluid solver):
        step = evolve(1); render field = per-cell activity (the hypervector magnitude), a 2D map lifted to a slab."""
        return Simulation(ca, lambda s, dt: s.evolve(1),
                          lambda s: np.linalg.norm(np.asarray(s.grid, float), axis=2), lo, hi, "automaton")


def known_solver_strategies():
    """The solver strategies with a ready-made factory (others plug in the same way)."""
    return ("for_fluid", "for_automaton")


def _selftest():
    from holographic_fluid import StableFluid
    from holographic_automaton import HyperCA
    from holographic_render import Camera

    cam = Camera(eye=(0.5, 0.5, 3.0), target=(0.5, 0.5, 0.5), fov_deg=45)

    # SOLVER 1: Stable Fluids (advect + project) -- a 3D smoke density
    fluid = StableFluid((16, 16, 16), dt=0.1)
    fluid.density[6:10, 2:6, 6:10] = 1.0
    fluid.vel[1, :, :5, :] = 1.0                                 # a little upward flow
    sim1 = Simulation.for_fluid(fluid)

    # SOLVER 2: reaction-diffusion automaton -- a genuinely different algorithm
    ca = HyperCA(size=20, dim=16, seed=0)
    sim2 = Simulation.for_automaton(ca)

    # BOTH step through the SAME loop
    sim1.run(4)
    sim2.run(4)
    assert sim1.steps_run == 4 and sim2.steps_run == 4
    assert sim1.name == "fluid" and sim2.name == "automaton"

    # the shared SimStep interface (what the Pipeline's sim stage calls) is the same type for both
    from holographic_integrate import SimStep
    assert isinstance(sim1.adapter, SimStep) and isinstance(sim2.adapter, SimStep)

    # the Pipeline renders BOTH fields
    img1, a1 = sim1.render(cam, width=24, height=24, steps=24, sigma=12.0)
    img2, a2 = sim2.render(cam, width=24, height=24, steps=24, sigma=8.0)
    assert np.isfinite(img1).all() and float(a1.max()) > 0.1     # smoke drew something
    assert np.isfinite(img2).all() and float(a2.max()) > 0.1     # the automaton field drew something

    # the solvers stayed SEPARATE -- each kept its own step signature, only the wrapper is shared
    assert fluid.density.shape == (16, 16, 16)                   # fluid still a StableFluid
    assert ca.grid.shape[:2] == (20, 20)                        # automaton still a HyperCA
    print("OK: holographic_simulationhome self-test passed (two distinct solvers -- Stable Fluids + reaction-"
          "diffusion -- stepped through the SAME loop; the Pipeline rendered both fields (alpha %.2f / %.2f); the "
          "solvers stayed separate; strategies %s)" % (float(a1.max()), float(a2.max()),
                                                       ", ".join(known_solver_strategies())))


if __name__ == "__main__":
    _selftest()
