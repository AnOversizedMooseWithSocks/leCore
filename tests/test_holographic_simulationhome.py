"""Tests for holographic_simulationhome -- the Simulation scaffold (R9: one step loop, solvers kept separate)."""
import numpy as np
import pytest
from holographic.misc.holographic_simulationhome import Simulation, known_solver_strategies


def _fluid_sim():
    from holographic.simulation_and_physics.holographic_fluid import StableFluid
    f = StableFluid((16, 16, 16), dt=0.1)
    f.density[6:10, 2:6, 6:10] = 1.0
    f.vel[1, :, :5, :] = 1.0
    return Simulation.for_fluid(f), f


def _automaton_sim():
    from holographic.misc.holographic_automaton import HyperCA
    ca = HyperCA(size=20, dim=16, seed=0)
    return Simulation.for_automaton(ca), ca


def test_two_solvers_step_through_same_loop():
    s1, _ = _fluid_sim(); s2, _ = _automaton_sim()
    s1.run(5); s2.run(5)
    assert s1.steps_run == 5 and s2.steps_run == 5
    # both wrap the SAME SimStep interface the pipeline calls
    from holographic.misc.holographic_integrate import SimStep
    assert isinstance(s1.adapter, SimStep) and isinstance(s2.adapter, SimStep)


def test_pipeline_renders_both_fields():
    from holographic.rendering.holographic_render import Camera
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
    from holographic.misc.holographic_fieldhome import Field
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


# ======================================================================================================
# The scaffold as a mind faculty: in-process wrapper + stateless twin for /invoke.
# ======================================================================================================
def test_simulation_faculty_wraps_a_solver_in_process():
    from holographic.misc.holographic_unified import UnifiedMind
    from holographic.simulation_and_physics.holographic_fluid import StableFluid
    m = UnifiedMind(dim=64, seed=0)
    fluid = StableFluid((16, 16, 16), dt=0.1)
    fluid.density[6:10, 2:6, 6:10] = 1.0
    sim = m.simulation(fluid, lambda s, dt: s.step(), lambda s: np.asarray(s.density, float))
    sim.run(4)
    assert sim.steps_run == 4
    assert sim.grid().shape == (16, 16, 16)
    assert fluid.density.shape == (16, 16, 16)                   # the solver stayed itself


def test_run_simulation_is_the_stateless_twin_and_both_kinds_run():
    from holographic.misc.holographic_unified import UnifiedMind
    m = UnifiedMind(dim=64, seed=0)
    gf = m.run_simulation("fluid", 6, grid=16)
    ga = m.run_simulation("automaton", 6, grid=16)
    assert gf.shape == (16, 16, 16) and np.isfinite(gf).all() and float(gf.sum()) > 0.0
    assert ga.shape == (16, 16) and np.isfinite(ga).all() and float(ga.max()) > 0.0
    # the two are DIFFERENT algorithms behind one loop -- not the same output
    with pytest.raises(ValueError):
        m.run_simulation("nope", 1)


def test_run_simulation_survives_a_real_http_invoke():
    """The live Simulation holds a solver + step adapter that do not survive JSON. run_simulation is the twin --
    plain arguments in, a plain field grid out. Proven over a real socket, like gather_samples."""
    import json
    import threading
    import urllib.request
    from http.server import HTTPServer

    import holographic_service as svc_mod
    from holographic.misc.holographic_unified import UnifiedMind

    svc = svc_mod.Service(mind=UnifiedMind(dim=64, seed=0))
    httpd = HTTPServer(("127.0.0.1", 0), svc_mod.make_handler(svc))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % httpd.server_address[1]
    try:
        tools = json.loads(urllib.request.urlopen(base + "/tools", timeout=30).read())
        assert "run_simulation" in {t["name"] for t in tools["tools"]}
        body = json.dumps({"name": "run_simulation", "args": {"kind": "automaton", "steps": 6, "grid": 16}})
        req = urllib.request.Request(base + "/invoke", data=body.encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=60).read())
        assert r["ok"] and isinstance(r["result"], list) and len(r["result"]) == 16
    finally:
        httpd.shutdown()
        httpd.server_close()
