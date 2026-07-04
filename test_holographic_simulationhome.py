"""Tests for holographic_simulationhome -- the Simulation scaffold (R9: one step loop, solvers kept separate)."""
import numpy as np
from holographic_simulationhome import Simulation, known_solver_strategies


def _fluid_sim():
    from holographic_fluid import StableFluid
    f = StableFluid((16, 16, 16), dt=0.1)
    f.density[6:10, 2:6, 6:10] = 1.0
    f.vel[1, :, :5, :] = 1.0
    return Simulation.for_fluid(f), f


def _automaton_sim():
    from holographic_automaton import HyperCA
    ca = HyperCA(size=20, dim=16, seed=0)
    return Simulation.for_automaton(ca), ca


def test_two_solvers_step_through_same_loop():
    s1, _ = _fluid_sim(); s2, _ = _automaton_sim()
    s1.run(5); s2.run(5)
    assert s1.steps_run == 5 and s2.steps_run == 5
    # both wrap the SAME SimStep interface the pipeline calls
    from holographic_integrate import SimStep
    assert isinstance(s1.adapter, SimStep) and isinstance(s2.adapter, SimStep)


def test_pipeline_renders_both_fields():
    from holographic_render import Camera
    cam = Camera(eye=(0.5, 0.5, 3.0), target=(0.5, 0.5, 0.5), fov_deg=45)
    s1, _ = _fluid_sim(); s2, _ = _automaton_sim()
    s1.run(4); s2.run(4)
    img1, a1 = s1.render(cam, width=24, height=24, steps=24, sigma=12.0)
    img2, a2 = s2.render(cam, width=24, height=24, steps=24, sigma=8.0)
    assert np.isfinite(img1).all() and a1.max() > 0.1
    assert np.isfinite(img2).all() and a2.max() > 0.1


def test_solvers_stay_separate():
    s1, fluid = _fluid_sim(); s2, ca = _automaton_sim()
    s1.step(); s2.step()
    assert fluid.density.shape == (16, 16, 16)          # still a StableFluid, its own math
    assert ca.grid.shape[:2] == (20, 20)                # still a HyperCA, its own math


def test_field_is_a_field_r2():
    from holographic_fieldhome import Field
    s1, _ = _fluid_sim(); s1.run(2)
    fld = s1.field()
    assert isinstance(fld, Field)
    v = fld.sample(np.array([[0.5, 0.5, 0.5]]))
    assert v.shape[0] == 1 and np.isfinite(v).all()


def test_grid_advances_over_steps():
    s1, _ = _fluid_sim()
    g0 = s1.grid().copy()
    s1.run(3)
    assert not np.array_equal(g0, s1.grid())            # the field actually evolved


def test_strategies_listed():
    assert set(known_solver_strategies()) == {"for_fluid", "for_automaton"}
